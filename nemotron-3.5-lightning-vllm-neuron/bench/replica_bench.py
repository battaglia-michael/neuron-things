#!/usr/bin/env python3
"""T11: per-instance throughput with one vLLM server per Trainium2 device.

One `vllm bench serve` client per replica, all started together (static client-side
round-robin: client i only talks to replica i, so every replica gets the same load, with
`--conc-per` requests in flight each and no load balancer in between). Every client uses the
same arguments as bench/run_matrix.py for the row (same lengths, sampling, --ignore-eos), with
its own seed. The aggregate is computed from the per-request data the clients save:

* output tok/s = total output tokens of all completed requests / the longest client duration
  (so the tail where some replicas have finished counts against the result);
* TTFT / TPOT percentiles over all requests of all replicas (TPOT per request =
  (latency - TTFT) / (output tokens - 1), as vLLM defines it).

Usage: replica_bench.py --row agent_8k --devices 0,4,5,... --out-dir results/trn2/stretch/replicas
All hosts are 127.0.0.1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bench"))
import common  # noqa: E402
import run_matrix  # noqa: E402


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def client_cmd(a, w, port, *, prompts, conc, seed, out_len, save):
    cmd = [
        a.vllm_bin, "bench", "serve", "--backend", "openai", "--endpoint", "/v1/completions",
        "--host", "127.0.0.1", "--port", str(port), "--model", "nemotron", "--tokenizer", a.tokenizer,
        "--dataset-name", "random", "--random-input-len", str(w["input_len"]),
        "--random-output-len", str(out_len), "--num-prompts", str(prompts),
        "--max-concurrency", str(conc), "--request-rate", "inf", "--seed", str(seed), "--ignore-eos",
        "--temperature", str(common.BENCH_SAMPLING["temperature"]),
        "--top-p", str(common.BENCH_SAMPLING["top_p"]),
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,99", "--disable-tqdm",
    ]
    if w.get("prefix_len"):
        cmd += ["--random-prefix-len", str(w["prefix_len"])]
    if save is not None:
        cmd += ["--save-result", "--save-detailed", "--result-dir", str(save.parent),
                "--result-filename", save.name]
    return cmd


def run_all(cmds, logs, timeout):
    procs = []
    for cmd, lg in zip(cmds, logs):
        f = open(lg, "w")
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f))
    t0 = time.time()
    rcs = []
    for p, f in procs:
        try:
            rcs.append(p.wait(timeout=max(1, timeout - (time.time() - t0))))
        except subprocess.TimeoutExpired:
            p.kill()
            rcs.append(-9)
        f.close()
    return rcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, help="workload id from run_matrix.WORKLOADS")
    ap.add_argument("--devices", required=True, help="comma list; port = --port-base + device")
    ap.add_argument("--port-base", type=int, default=8500, help="replica on device d listens on port-base + d (default 8500)")
    ap.add_argument("--conc-per", type=int, default=32)
    ap.add_argument("--prompts-per", type=int, default=None, help="default: the row's c32 prompt count")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--vllm-bin", default="vllm")
    ap.add_argument("--timeout", type=int, default=3 * 3600)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    import os
    a.tokenizer = a.tokenizer or os.environ["MODEL_DIR"]
    devs = [int(x) for x in a.devices.split(",")]
    w = next(x for x in run_matrix.WORKLOADS if x["id"] == a.row)
    prompts = a.prompts_per or w["prompts"].get(a.conc_per) or 3 * a.conc_per
    base_seed = a.seed + zlib.crc32(f"{w['id']}:{a.conc_per}".encode()) % 100_000
    out = Path(a.out_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    name = f"{a.row}_{len(devs)}x{a.conc_per}"

    common.log(f"{name}: warmup on {len(devs)} replicas")
    wc = [client_cmd(a, w, a.port_base + d, prompts=8, conc=8, seed=base_seed + 7919 + 10007 * i,
                     out_len=min(w["output_len"], 128), save=None) for i, d in enumerate(devs)]
    run_all(wc, [raw / f"{name}_warmup_dev{d}.log" for d in devs], a.timeout)

    common.log(f"{name}: measuring {prompts} prompts x {len(devs)} replicas, {a.conc_per} in flight each "
               f"({a.conc_per * len(devs)} total)")
    t_start = common.utc_now()
    t0 = time.time()
    files = [raw / f"{name}_dev{d}.json" for d in devs]
    for f in files:
        f.unlink(missing_ok=True)
    mc = [client_cmd(a, w, a.port_base + d, prompts=prompts, conc=a.conc_per, seed=base_seed + 10007 * i,
                     out_len=w["output_len"], save=f) for i, (d, f) in enumerate(zip(devs, files))]
    rcs = run_all(mc, [raw / f"{name}_dev{d}.log" for d in devs], a.timeout)
    wall = time.time() - t0

    per, ttft, tpot, itl, e2el = [], [], [], [], []
    tot_out = tot_in = completed = failed = 0
    dur_max = 0.0
    for d, f, rc in zip(devs, files, rcs):
        if rc != 0 or not f.exists():
            per.append(dict(device=d, status="failed", rc=rc))
            continue
        r = json.loads(f.read_text())
        completed += int(r["completed"])
        failed += int(r["failed"])
        tot_out += int(r["total_output_tokens"])
        tot_in += int(r["total_input_tokens"])
        dur_max = max(dur_max, float(r["duration"]))
        for tt, its, ol, err in zip(r["ttfts"], r["itls"], r["output_lens"], r.get("errors") or [""] * len(r["ttfts"])):
            if err or not its or ol < 2:
                continue
            ttft.append(tt * 1000)
            lat = tt + sum(its)
            e2el.append(lat * 1000)
            tpot.append((lat - tt) / (ol - 1) * 1000)
            itl.extend(x * 1000 for x in its)
        per.append(dict(device=d, port=a.port_base + d, status="ok" if r["failed"] == 0 else "partial",
                        completed=r["completed"], failed=r["failed"], duration_s=r["duration"],
                        output_tok_s=r["output_throughput"], ttft_p50_ms=r["median_ttft_ms"],
                        tpot_p50_ms=r["median_tpot_ms"], raw_json=str(f.relative_to(out))))
    res = dict(
        row=a.row, label=w["label"], replicas=len(devs), devices=devs, conc_per_replica=a.conc_per,
        total_concurrency=a.conc_per * len(devs), prompts_per_replica=prompts,
        completed=completed, failed=failed, clients_failed=sum(1 for p in per if p["status"] == "failed"),
        started=t_start, wall_s=wall, longest_client_duration_s=dur_max,
        total_input_tokens=tot_in, total_output_tokens=tot_out,
        output_tok_s=tot_out / dur_max if dur_max else None,
        sum_of_replica_output_tok_s=sum(p.get("output_tok_s", 0) for p in per),
        ttft_p50_ms=pct(ttft, 50), ttft_p99_ms=pct(ttft, 99),
        tpot_p50_ms=pct(tpot, 50), tpot_p99_ms=pct(tpot, 99),
        itl_p50_ms=pct(itl, 50), itl_p99_ms=pct(itl, 99),
        e2el_p50_ms=pct(e2el, 50), e2el_p99_ms=pct(e2el, 99),
        method="one vllm bench serve client per replica, started together; static client-side round-robin "
               "(client i -> replica i), no load balancer; tok/s = total output tokens / longest client duration",
        note=a.note, per_replica=per, seed_base=base_seed,
    )
    (out / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")
    common.log(f"{name}: completed={completed} failed={failed} output_tok_s={res['output_tok_s']} "
               f"ttft_p50={res['ttft_p50_ms']} tpot_p50={res['tpot_p50_ms']}")


if __name__ == "__main__":
    main()
