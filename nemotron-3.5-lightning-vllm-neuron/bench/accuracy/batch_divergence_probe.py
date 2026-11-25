#!/usr/bin/env python3
"""Explain early batch-vs-solo divergences found by batch_invariance.py.

For each prompt id given, using exactly the prompts batch_invariance.py builds (same seed):
  1. solo:        the prompt alone, greedy, with top-5 logprobs at every output token;
  2. mixed:       all 32 prompts at once (15 ms stagger), as in batch_invariance.py;
  3. homogeneous: 32 copies of the same prompt at once (same stagger), so every other
                  request in the batch has identical content.
At the first solo-vs-mixed divergence it reports the solo top-1/top-2 logprob margin and the
rank of the mixed token in the solo top-5. If the homogeneous copy at the prompt's own slot
gives the mixed-batch tokens, the divergence does not depend on what the other requests
contain (no cross-request state leakage); it comes from the batch shape (decode bucket).

    python bench/accuracy/batch_divergence_probe.py --port 8100 --ids p00,p08,p22,p26
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import batch_invariance as bi  # noqa: E402


async def ask(client, base, model, text, max_tokens, top=5):
    payload = common.chat_payload(model, [{"role": "user", "content": text}], max_tokens=max_tokens,
                                  temperature=0.0, extra={"return_token_ids": True, "logprobs": True,
                                                          "top_logprobs": top})
    res = await common.post_json(client, f"{base}/v1/chat/completions", payload)
    ch = res["choices"][0]
    content = (ch.get("logprobs") or {}).get("content") or []
    return dict(tokens=list(ch.get("token_ids") or []),
                top=[[(t["token"], t["logprob"]) for t in c["top_logprobs"]] for c in content],
                tok_str=[c["token"] for c in content], finish=ch.get("finish_reason"))


async def batch(client, base, model, texts, max_tokens, stagger_ms):
    async def one(i):
        await asyncio.sleep(i * stagger_ms / 1000)
        return await ask(client, base, model, texts[i], max_tokens)
    return await common.run_bounded(one, len(texts), len(texts))


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


async def main_async(args):
    base = f"http://127.0.0.1:{args.port}"
    model = args.model
    prompts = bi.build_prompts(args.n, args.long_questions, args.seed)
    idx = {p["id"]: i for i, p in enumerate(prompts)}
    ids = args.ids.split(",")
    out = dict(params=vars(args), rows=[])
    async with common.make_client(args.n) as client:
        solo = {pid: await ask(client, base, model, prompts[idx[pid]]["text"], args.max_tokens) for pid in ids}
        common.log("mixed batch")
        mixed = await batch(client, base, model, [p["text"] for p in prompts], args.max_tokens, args.stagger_ms)
        homo = {}
        for pid in ids:
            common.log(f"homogeneous batch of {pid}")
            homo[pid] = await batch(client, base, model, [prompts[idx[pid]]["text"]] * args.n,
                                    args.max_tokens, args.stagger_ms)
    for pid in ids:
        i = idx[pid]
        s, m, h = solo[pid], mixed[i], homo[pid][i]
        d = first_div(s["tokens"], m["tokens"])
        row = dict(id=pid, kind=prompts[i]["kind"], first_div_solo_vs_mixed=d,
                   first_div_mixed_vs_homogeneous_same_slot=first_div(m["tokens"], h["tokens"]),
                   first_div_solo_vs_homogeneous_same_slot=first_div(s["tokens"], h["tokens"]),
                   homogeneous_copies_equal_to_solo=sum(first_div(s["tokens"], x["tokens"]) is None for x in homo[pid]),
                   homogeneous_copies_equal_to_mixed=sum(first_div(m["tokens"], x["tokens"]) is None for x in homo[pid]),
                   n_solo=len(s["tokens"]), n_mixed=len(m["tokens"]))
        if d is not None and d < len(s["top"]):
            tops = s["top"][d]
            row["solo_top5_at_div"] = tops
            row["solo_top1_top2_margin"] = tops[0][1] - tops[1][1] if len(tops) > 1 else None
            mt = m["tok_str"][d] if d < len(m["tok_str"]) else None
            row["mixed_token_at_div"] = mt
            row["mixed_token_rank_in_solo_top5"] = next((k + 1 for k, (t, _) in enumerate(tops) if t == mt), None)
            if d < len(m["top"]):
                mtops = m["top"][d]
                row["mixed_top5_at_div"] = mtops
                row["mixed_top1_top2_margin"] = mtops[0][1] - mtops[1][1] if len(mtops) > 1 else None
            row["solo_text_after"] = "".join(s["tok_str"][d:d + 20])
            row["mixed_text_after"] = "".join(m["tok_str"][d:d + 20])
        out["rows"].append(row)
    # Margins over the whole solo outputs, to show how common near-ties are.
    margins = [t[0][1] - t[1][1] for s in solo.values() for t in s["top"] if len(t) > 1]
    out["solo_margin_lt_0.1_frac"] = sum(x < 0.1 for x in margins) / max(len(margins), 1)
    out["solo_margin_lt_0.5_frac"] = sum(x < 0.5 for x in margins) / max(len(margins), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--model", default="nemotron")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--long-questions", type=int, default=40)
    ap.add_argument("--stagger-ms", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(common.REPO / "results/accuracy/batch_divergence_probe-trn2.json"))
    args = ap.parse_args()
    out = asyncio.run(main_async(args))
    common.write_json(args.out, out)
    for r in out["rows"]:
        print(json.dumps({k: v for k, v in r.items() if k not in ("solo_top5_at_div", "mixed_top5_at_div")}))
    print("solo margin<0.1 frac", out["solo_margin_lt_0.1_frac"])


if __name__ == "__main__":
    main()
