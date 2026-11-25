#!/usr/bin/env python3
"""Run the PLAN.md Section 5 workload matrix with `vllm bench serve`.

Works the same on Trainium2 and H100: it only needs an OpenAI-compatible vLLM
server on a loopback address and the `vllm` CLI of the client venv.

For every (workload, concurrency) row it
  1. runs a small warmup invocation (different seed, shorter output, not saved),
  2. runs the measured invocation (random dataset, --ignore-eos, fixed per-row
     seed, temperature 1.0, top_p 0.95) and saves vLLM's raw JSON,
and then writes matrix.csv next to the raw files. Re-running skips finished rows.

Example:
    python bench/run_matrix.py --platform trn2 --port 8000 \
        --model nemotron --tokenizer $MODEL_DIR --max-model-len 40960
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# --------------------------------------------------------------------------
# The matrix (PLAN.md Section 5). `prompts` = measured prompts per concurrency.
# --------------------------------------------------------------------------
WORKLOADS = [
    dict(id="agent_8k", label="8,192 / 512 (agent / RAG turn)", priority="must",
         input_len=8192, output_len=512, prefix_len=0,
         concurrency=[1, 32, 128], prompts={1: 8, 32: 96, 128: 256}),
    dict(id="reasoning_1k_8k", label="1,024 / 8,192 (reasoning, long output)", priority="must",
         input_len=1024, output_len=8192, prefix_len=0,
         concurrency=[1, 32], prompts={1: 4, 32: 64}),
    dict(id="ref_1k_1k", label="1,024 / 1,024 (reference row)", priority="must",
         input_len=1024, output_len=1024, prefix_len=0,
         concurrency=[1, 32], prompts={1: 8, 32: 128}),
    dict(id="long_32k", label="32,768 / 512 (long agent session)", priority="must",
         input_len=32768, output_len=512, prefix_len=0,
         concurrency=[1, 16], prompts={1: 6, 16: 32}),
    dict(id="long_128k", label="131,072 / 512 (long document)", priority="stretch",
         input_len=131072, output_len=512, prefix_len=0,
         concurrency=[1], prompts={1: 3}),
    # input_len here is the random part; total prompt = prefix_len + input_len = 8,192.
    dict(id="prefix_8k", label="8,192 / 512, shared 6,144-token prefix (prefix caching on)",
         priority="stretch", input_len=2048, output_len=512, prefix_len=6144,
         concurrency=[32], prompts={32: 96}),
]

# Tiny version of the same matrix for harness smoke tests (small models, 2K context).
SMOKE = {
    "agent_8k": dict(input_len=192, output_len=48, concurrency=[1, 4], prompts={1: 4, 4: 8}),
    "reasoning_1k_8k": dict(input_len=64, output_len=96, concurrency=[1, 4], prompts={1: 3, 4: 8}),
    "ref_1k_1k": dict(input_len=128, output_len=64, concurrency=[1, 4], prompts={1: 4, 4: 8}),
    "long_32k": dict(input_len=1024, output_len=32, concurrency=[1, 2], prompts={1: 3, 2: 4}),
    "long_128k": dict(input_len=1536, output_len=16, concurrency=[1], prompts={1: 2}),
    "prefix_8k": dict(input_len=128, output_len=32, prefix_len=256, concurrency=[2], prompts={2: 6}),
}

CSV_FIELDS = [
    "workload", "label", "priority", "input_len", "output_len", "prefix_len", "concurrency",
    "num_prompts", "seed", "completed", "failed", "duration_s", "output_tok_s", "total_tok_s",
    "req_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms", "itl_p50_ms",
    "itl_p99_ms", "e2el_p50_ms", "e2el_p99_ms", "status", "raw_json", "platform", "timestamp",
]


def build_rows(args) -> list[dict]:
    rows = []
    only = set(args.rows.split(",")) if args.rows else None
    only_c = {int(x) for x in args.conc.split(",")} if args.conc else None
    for w in WORKLOADS:
        if only and w["id"] not in only:
            continue
        if w["priority"] == "stretch" and not (args.stretch or (only and w["id"] in only)):
            continue
        w = dict(w)
        if args.smoke:
            w.update(SMOKE[w["id"]])
            w.setdefault("prefix_len", 0)
        for c in w["concurrency"]:
            if only_c and c not in only_c:
                continue
            n = w["prompts"].get(c) or max(2 * c, 4)
            n = max(1, int(round(n * args.prompts_scale)))
            seed = args.seed + zlib.crc32(f"{w['id']}:{c}".encode()) % 100_000
            label = w["label"]
            if args.smoke:
                label = f"SMOKE {w['input_len'] + w.get('prefix_len', 0):,} / {w['output_len']:,} [{w['id']}]"
            rows.append(dict(
                workload=w["id"], label=label, priority=w["priority"],
                input_len=w["input_len"], output_len=w["output_len"],
                prefix_len=w.get("prefix_len", 0), concurrency=c, num_prompts=n, seed=seed,
            ))
    return rows


def row_name(r: dict) -> str:
    return f"{r['workload']}_c{r['concurrency']}"


def vllm_cmd(args, model: str, r: dict, *, prompts: int, conc: int, seed: int,
             out_len: int, save: Path | None) -> list[str]:
    cmd = [
        args.vllm_bin, "bench", "serve",
        "--backend", "openai", "--endpoint", "/v1/completions",
        "--host", args.host, "--port", str(args.port),
        "--model", model, "--tokenizer", args.tokenizer or model,
        "--dataset-name", "random",
        "--random-input-len", str(r["input_len"]),
        "--random-output-len", str(out_len),
        "--num-prompts", str(prompts),
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--seed", str(seed),
        "--ignore-eos",
        "--temperature", str(common.BENCH_SAMPLING["temperature"]),
        "--top-p", str(common.BENCH_SAMPLING["top_p"]),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,99",
        "--disable-tqdm",
    ]
    if r["prefix_len"]:
        cmd += ["--random-prefix-len", str(r["prefix_len"])]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if save is not None:
        cmd += ["--save-result", "--result-dir", str(save.parent), "--result-filename", save.name]
    cmd += args.extra_bench_args
    return cmd


def pick(res: dict, *keys, default=None):
    for k in keys:
        if k in res and res[k] is not None:
            return res[k]
    return default


def summarize(res: dict) -> dict:
    def ms(metric, pct):
        return pick(res, f"p{pct}_{metric}_ms", f"median_{metric}_ms" if pct == 50 else "_none_")
    completed = int(pick(res, "completed", default=0))
    failed = int(pick(res, "failed", default=0))
    return dict(
        completed=completed, failed=failed,
        duration_s=pick(res, "duration"),
        output_tok_s=pick(res, "output_throughput"),
        total_tok_s=pick(res, "total_token_throughput"),
        req_s=pick(res, "request_throughput"),
        ttft_p50_ms=ms("ttft", 50), ttft_p99_ms=ms("ttft", 99),
        tpot_p50_ms=ms("tpot", 50), tpot_p99_ms=ms("tpot", 99),
        itl_p50_ms=ms("itl", 50), itl_p99_ms=ms("itl", 99),
        e2el_p50_ms=ms("e2el", 50), e2el_p99_ms=ms("e2el", 99),
    )


def write_csv(path: Path, records: list[dict]) -> None:
    order = {w["id"]: i for i, w in enumerate(WORKLOADS)}
    records = sorted(records, key=lambda r: (order.get(r["workload"], 99), int(r["concurrency"])))
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS})
    tmp.replace(path)


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def run_row(args, model: str, r: dict, raw_dir: Path) -> dict:
    name = row_name(r)
    raw = raw_dir / f"{name}.json"
    logf = raw_dir / f"{name}.log"
    rec = dict(r, platform=args.platform, raw_json=str(raw.relative_to(args.out_dir)),
               timestamp=common.utc_now(), status="failed")

    total = r["input_len"] + r["prefix_len"] + r["output_len"]
    if args.max_model_len and total > args.max_model_len:
        rec["status"] = "skipped_ctx"
        common.log(f"{name}: skipped, {total} tokens > --max-model-len {args.max_model_len}")
        return rec

    env = dict(os.environ)
    if not args.no_warmup:
        w = args.warmup_prompts or max(2, min(r["concurrency"], 8))
        wcmd = vllm_cmd(args, model, r, prompts=w, conc=min(r["concurrency"], w),
                        seed=r["seed"] + 7919, out_len=min(r["output_len"], args.warmup_output_len),
                        save=None)
        common.log(f"{name}: warmup {w} prompts")
        with logf.open("w") as lf:
            lf.write("# warmup (not measured)\n# " + " ".join(wcmd) + "\n")
            lf.flush()
            try:
                subprocess.run(wcmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                               timeout=args.row_timeout, check=False)
            except subprocess.TimeoutExpired:
                lf.write("\n# warmup timed out\n")
    cmd = vllm_cmd(args, model, r, prompts=r["num_prompts"], conc=r["concurrency"],
                   seed=r["seed"], out_len=r["output_len"], save=raw)
    common.log(f"{name}: measuring {r['num_prompts']} prompts at concurrency {r['concurrency']}")
    if raw.exists():
        raw.unlink()
    t0 = time.time()
    with logf.open("a") as lf:
        lf.write("\n# measured run\n# " + " ".join(cmd) + "\n")
        lf.flush()
        try:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                               timeout=args.row_timeout, check=False)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -9
            lf.write(f"\n# timed out after {args.row_timeout}s\n")
    if rc != 0 or not raw.exists():
        common.log(f"{name}: FAILED rc={rc} (see {logf})")
        return rec
    res = json.loads(raw.read_text())
    s = summarize(res)
    rec.update(s)
    rec["status"] = "ok" if s["failed"] == 0 and s["completed"] > 0 else "partial"
    common.log(f"{name}: {rec['status']} out_tok/s={s['output_tok_s']} "
               f"ttft_p50={s['ttft_p50_ms']} tpot_p50={s['tpot_p50_ms']} ({time.time() - t0:.0f}s)")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_server_args(ap)
    ap.add_argument("--out-dir", default=None,
                    help="default results/<platform>/baseline; raw JSON goes to <out-dir>/raw/")
    ap.add_argument("--variant", default=None, help="free-text label stored in meta.json")
    ap.add_argument("--rows", default=None, help="comma list of workload ids (see --list)")
    ap.add_argument("--conc", default=None, help="comma list of concurrencies to keep")
    ap.add_argument("--stretch", action="store_true", help="also run the Stretch rows")
    ap.add_argument("--smoke", action="store_true", help="tiny lengths and concurrencies for harness tests")
    ap.add_argument("--list", action="store_true", help="print the rows and exit")
    ap.add_argument("--seed", type=int, default=1234, help="base seed; each row adds a fixed offset")
    ap.add_argument("--prompts-scale", type=float, default=1.0, help="multiply measured prompts per row")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="server context length; rows that cannot fit are marked skipped_ctx")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--warmup-prompts", type=int, default=0, help="default max(2, min(concurrency, 8))")
    ap.add_argument("--warmup-output-len", type=int, default=128)
    ap.add_argument("--row-timeout", type=int, default=4 * 3600, help="seconds per vllm bench invocation")
    ap.add_argument("--force", action="store_true", help="re-run rows that already have ok results")
    ap.add_argument("--vllm-bin", default=shutil.which("vllm") or "vllm")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--note", default=None, help="free text stored in meta.json (serve flags, etc.)")
    ap.add_argument("--serve-cmd-file", default=None, help="file with the serve command, copied to meta.json")
    ap.add_argument("--extra-bench-args", default="",
                    help="quoted string of extra flags passed to vllm bench serve")
    args = ap.parse_args()
    args.extra_bench_args = shlex.split(args.extra_bench_args)

    rows = build_rows(args)
    if args.list:
        for r in rows:
            print(f"{row_name(r):28s} in={r['input_len']}+{r['prefix_len']} out={r['output_len']} "
                  f"prompts={r['num_prompts']} seed={r['seed']} [{r['priority']}]")
        return 0
    common.require_platform(args)
    base = common.base_url(args)
    model = common.resolve_model(args, base)
    args.out_dir = Path(args.out_dir or common.REPO / "results" / args.platform / "baseline")
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "matrix.csv"

    versions = {}
    for pkg in ("vllm", "vllm-neuron", "torch", "transformers"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    meta = dict(
        platform=args.platform, variant=args.variant, started=common.utc_now(), model=model,
        tokenizer=args.tokenizer or model, server=common.server_info(base),
        client_versions=versions, git_rev=common.git_rev(), seed_base=args.seed,
        sampling=common.BENCH_SAMPLING, smoke=args.smoke, note=args.note,
        max_model_len=args.max_model_len, argv=sys.argv[1:],
        bench_flags="--dataset-name random --ignore-eos --request-rate inf --max-concurrency <c>",
    )
    if args.serve_cmd_file and Path(args.serve_cmd_file).exists():
        meta["serve_cmd"] = Path(args.serve_cmd_file).read_text().strip()
    common.write_json(args.out_dir / "meta.json", meta)

    records = {(r["workload"], str(r["concurrency"])): r for r in load_existing(csv_path)}
    rc = 0
    for r in rows:
        key = (r["workload"], str(r["concurrency"]))
        old = records.get(key)
        if old and old.get("status") == "ok" and not args.force \
                and (args.out_dir / old["raw_json"]).exists():
            common.log(f"{row_name(r)}: already done, skipping (use --force to redo)")
            continue
        rec = run_row(args, model, r, raw_dir)
        records[key] = rec
        write_csv(csv_path, list(records.values()))
        if rec["status"] not in ("ok", "skipped_ctx"):
            rc = 1
    write_csv(csv_path, list(records.values()))
    common.log(f"wrote {csv_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
