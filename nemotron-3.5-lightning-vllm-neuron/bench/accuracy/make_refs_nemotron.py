#!/usr/bin/env python3
"""A03: HF teacher-forced reference labels for Nemotron 3.5 Lightning on CPU.

Three stages (each resumable, one file per sequence under <out>/parts/):

  build   Build the fixed set of chat-formatted token sequences -> <out>/sequences.jsonl.
          Text source: GSM8K test split (openai/grade-school-math, MIT; vendored in
          bench/data/gsm8k_test.jsonl), question/solution pairs joined into one multi-turn
          chat, rendered with the model's chat template (enable_thinking=False), cut to the
          target token length. Sequence i uses random.Random(SEED * 1000 + i).
  label   One forward pass per sequence on CPU in --dtype fp32|bf16. Stores argmax at
          every position (fp32 also: top-5 ids / logprobs).
  merge   Joins parts into <out>/fp32.jsonl (the file teacher_forced.py reads, with
          bf16_argmax_ids when available) and writes <out>/summary.json: HF BF16 vs fp32
          match rate by position bucket.

    python bench/accuracy/make_refs_nemotron.py build  --model $MODEL_DIR --out results/references
    python bench/accuracy/make_refs_nemotron.py label  --model $MODEL_DIR --out results/references --dtype fp32 --threads 48
    python bench/accuracy/make_refs_nemotron.py label  --model $MODEL_DIR --out results/references --dtype bf16 --threads 48
    python bench/accuracy/make_refs_nemotron.py merge  --out results/references
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SEED = 20261112
LENGTHS = ([256, 384, 512, 640, 768, 896, 960, 1024]
           + [1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096]
           + [4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6656, 7168, 7680, 8192])
SOURCE = "openai/grade-school-math test split (bench/data/gsm8k_test.jsonl), multi-turn chat, thinking off"
BUCKETS = [(0, 1024, "0-1K"), (1024, 4096, "1K-4K"), (4096, 8192, "4K-8K")]


def bucket_of(pos: int) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= pos < hi:
            return name
    return BUCKETS[-1][2]


def conversation(rng, qa, tok, target):
    msgs = []
    while True:
        r = qa[rng.randrange(len(qa))]
        sol = re.sub(r"<<[^>]*>>", "", r["answer"]).replace("####", "The answer is")
        msgs += [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": sol}]
        text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) >= target:
            return ids[:target], len(msgs) // 2


def stage_build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    qa = common.read_jsonl(common.DATA_DIR / "gsm8k_test.jsonl")
    rows = []
    for i, L in enumerate(LENGTHS):
        rng = random.Random(SEED * 1000 + i)
        ids, turns = conversation(rng, qa, tok, L)
        rows.append(dict(id=f"seq-{i:02d}-{L}", length=len(ids), turns=turns, seed=SEED * 1000 + i,
                         source=SOURCE, token_ids=ids))
    common.write_jsonl(Path(a.out) / "sequences.jsonl", rows)
    common.log(f"wrote {len(rows)} sequences, {sum(r['length'] for r in rows)} tokens")


def stage_label(a):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    seqs = common.read_jsonl(out / "sequences.jsonl")
    pdir = out / "parts" / a.dtype
    pdir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seqs if not (pdir / f"{s['id']}.json").exists()]
    if a.only:
        todo = [s for s in todo if s["id"] in a.only.split(",")]
    common.log(f"{a.dtype}: {len(todo)} of {len(seqs)} sequences to label, {a.threads} threads")
    if not todo:
        return
    dt = dict(fp32=torch.float32, bf16=torch.bfloat16)[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).eval()
    if a.dt_limit == "none":
        # T03b: stock HF NemotronHMamba2Mixer hard-codes time_step_limit = (time_step_min, inf)
        # for its chunked scan. NVIDIA's remote code, Megatron, mamba_ssm and vLLM GPU use
        # (0, inf). "none" makes the HF reference match those.
        n = 0
        for m in model.modules():
            if hasattr(m, "time_step_limit"):
                m.time_step_limit = (0.0, float("inf"))
                n += 1
        common.log(f"dt limit none: patched time_step_limit on {n} mixers")
    head = model.get_output_embeddings()
    # shortest first: early failures are cheap, and partial results span the length range
    for s in sorted(todo, key=lambda s: s["length"]):
        t0 = time.time()
        with torch.no_grad():
            h = model.base_model(input_ids=torch.tensor([s["token_ids"]])).last_hidden_state[0]
            arg, t5i, t5l = [], [], []
            for c in range(0, h.shape[0], 1024):
                lg = head(h[c:c + 1024]).float()
                arg += lg.argmax(-1).tolist()
                if a.dtype == "fp32":
                    v, i = torch.log_softmax(lg, -1).topk(5, -1)
                    t5i += i.tolist()
                    t5l += [[round(x, 5) for x in row] for row in v.tolist()]
        rec = dict(id=s["id"], dtype=a.dtype, argmax_ids=arg, seconds=round(time.time() - t0, 1),
                   torch=torch.__version__)
        if a.dtype == "fp32":
            rec.update(top5_ids=t5i, top5_logprobs=t5l)
        tmp = pdir / f"{s['id']}.json.tmp"
        tmp.write_text(json.dumps(rec))
        tmp.rename(pdir / f"{s['id']}.json")
        common.log(f"{s['id']}: {s['length']} tokens, {rec['seconds']}s ({a.dtype})")


def stage_merge(a):
    out = Path(a.out)
    seqs = common.read_jsonl(out / "sequences.jsonl")
    rows, per_seq = [], []
    agg = {b[2]: [0, 0] for b in BUCKETS}
    # also by margin: how many mismatches sit on near-ties of the fp32 top-2
    tie = {"n_pos": 0, "n_mismatch": 0, "mismatch_margin_lt_0.1": 0, "mismatch_margin_lt_0.5": 0}
    for s in seqs:
        f32p = out / "parts" / "fp32" / f"{s['id']}.json"
        if not f32p.exists():
            continue
        f32 = json.loads(f32p.read_text())
        row = dict(id=s["id"], token_ids=s["token_ids"], argmax_ids=f32["argmax_ids"],
                   top5_ids=f32["top5_ids"], top5_logprobs=f32["top5_logprobs"],
                   source=f"{SOURCE}; seed={s['seed']}", dtype="fp32", model_dtype_ref="bf16 checkpoint upcast")
        b16p = out / "parts" / "bf16" / f"{s['id']}.json"
        if b16p.exists():
            b16 = json.loads(b16p.read_text())
            row["bf16_argmax_ids"] = b16["argmax_ids"]
            m = {b[2]: [0, 0] for b in BUCKETS}
            for i, (x, y) in enumerate(zip(f32["argmax_ids"], b16["argmax_ids"])):
                bk = bucket_of(i)
                ok = x == y
                m[bk][0] += 1
                m[bk][1] += ok
                agg[bk][0] += 1
                agg[bk][1] += ok
                tie["n_pos"] += 1
                if not ok:
                    tie["n_mismatch"] += 1
                    lp = f32["top5_logprobs"][i]
                    margin = lp[0] - lp[1]
                    tie["mismatch_margin_lt_0.1"] += margin < 0.1
                    tie["mismatch_margin_lt_0.5"] += margin < 0.5
            per_seq.append(dict(id=s["id"], length=s["length"], match={k: (v[1] / v[0] if v[0] else None) for k, v in m.items()},
                                overall=sum(v[1] for v in m.values()) / max(1, sum(v[0] for v in m.values()))))
        rows.append(row)
    common.write_jsonl(out / "fp32.jsonl", rows)
    n_all = sum(v[0] for v in agg.values())
    m_all = sum(v[1] for v in agg.values())
    summary = dict(
        task="A03", model=a.model if hasattr(a, "model") and a.model else "Nemotron-3.5-Lightning-30B-A3B-BF16",
        source=SOURCE, seed=SEED, n_sequences_fp32=len(rows), n_sequences_bf16_compared=len(per_seq),
        total_tokens=sum(len(r["token_ids"]) for r in rows),
        lengths=[s["length"] for s in seqs],
        hf_bf16_vs_fp32_match={**{k: dict(n=v[0], matched=v[1], rate=(v[1] / v[0] if v[0] else None)) for k, v in agg.items()},
                               "all": dict(n=n_all, matched=m_all, rate=(m_all / n_all if n_all else None))},
        mismatch_margins=tie, per_sequence=per_seq,
        note="match = HF BF16 argmax equals HF fp32 argmax at the same position under teacher forcing (same input tokens).")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    common.log(f"merged {len(rows)} fp32 rows, {len(per_seq)} with bf16; summary -> {out / 'summary.json'}")
    for k, v in summary["hf_bf16_vs_fp32_match"].items():
        if v["rate"] is not None:
            common.log(f"  {k}: {v['rate']:.4f} ({v['matched']}/{v['n']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "label", "merge"])
    ap.add_argument("--model")
    ap.add_argument("--out", default="results/references")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--only", default="")
    ap.add_argument("--dt-limit", choices=["hf", "none"], default="hf",
                    help="label stage: 'hf' = stock HF (dt clamped to time_step_min in the scan); "
                         "'none' = no clamp (NVIDIA remote code, Megatron, vLLM GPU)")
    a = ap.parse_args()
    if a.threads > 96:
        sys.exit("cap torch threads at 96 on the shared host")
    {"build": stage_build, "label": stage_label, "merge": stage_merge}[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
