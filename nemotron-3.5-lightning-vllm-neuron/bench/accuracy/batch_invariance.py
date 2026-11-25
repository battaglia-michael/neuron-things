#!/usr/bin/env python3
"""Batch invariance: the same prompts alone and mixed into a batch of 32 must give the same
greedy tokens, apart from the small BF16 drift that different batch shapes cause.

This is the check for requests reading each other's recurrent (Mamba) state. That bug shows
up as gross divergence in the first tokens, not as late drift.

Phases
  solo    each prompt alone, one at a time (run twice, to measure the server's own noise floor)
  batch   all prompts at once (staggered by --stagger-ms so prefills overlap other requests'
          decodes), --repeats times

Prompts: 32 chat prompts built from GSM8K questions in three length classes (short, medium,
long) so a batch holds different prompt lengths. Fixed seed.

Pass rule: in every batch run, at most --max-gross-frac of the prompts (default 10%) diverge
before --min-prefix tokens ("gross divergence") and no output is empty. Late drift is
reported, not failed. Cross-request state leakage corrupts most prompts within the first few
tokens, so it is far above the 10% line; a handful of early flips on near-tied logits (more
likely with MoE routing and tiny models) is not. The solo-vs-solo and batch-vs-batch rates
in the result show how deterministic the server is on its own.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def build_prompts(n: int, long_questions: int, seed: int) -> list[dict]:
    qs = [r["question"] for r in common.read_jsonl(common.DATA_DIR / "gsm8k_test.jsonl")]
    rng = random.Random(seed)
    rng.shuffle(qs)
    n_short, n_med = n // 2, n // 4
    out, k = [], 0

    def take(m):
        nonlocal k
        chunk = qs[k:k + m]
        k += m
        return chunk

    for i in range(n):
        if i < n_short:
            kind, text = "short", take(1)[0]
        elif i < n_short + n_med:
            kind = "medium"
            parts = take(4)
            text = "Here are several problems. Solve only the last one.\n\n" + "\n\n".join(parts)
        else:
            kind = "long"
            parts = take(long_questions)
            text = ("Here are many problems. Solve only the last one and ignore the others.\n\n"
                    + "\n\n".join(parts))
        out.append(dict(id=f"p{i:02d}", kind=kind, text=text))
    return out


async def ask(client, base, model, text, max_tokens):
    payload = common.chat_payload(model, [{"role": "user", "content": text}], max_tokens=max_tokens,
                                  temperature=0.0, extra={"return_token_ids": True})
    res = await common.post_json(client, f"{base}/v1/chat/completions", payload)
    ch = res["choices"][0]
    toks = ch.get("token_ids")
    return dict(tokens=list(toks) if toks else None, text=common.message_text(ch)
                + common.message_reasoning(ch), finish=ch.get("finish_reason"),
                prompt_tokens=res.get("usage", {}).get("prompt_tokens"))


def first_divergence(a: dict, b: dict) -> int | None:
    """Index of the first differing output token (None if identical)."""
    if a["tokens"] is not None and b["tokens"] is not None:
        x, y = a["tokens"], b["tokens"]
    else:  # fall back to characters when the server does not return token ids
        x, y = a["text"], b["text"]
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


async def main_async(args, base, model):
    prompts = build_prompts(args.n, args.long_questions, args.seed)
    async with common.make_client(args.n) as client:
        async def solo():
            out = []
            for i, p in enumerate(prompts):
                out.append(await ask(client, base, model, p["text"], args.max_tokens))
            return out

        common.log("solo pass 1")
        s1 = await solo()
        s2 = None
        if not args.no_solo_repeat:
            common.log("solo pass 2 (noise floor)")
            s2 = await solo()
        batches = []
        for rep in range(args.repeats):
            common.log(f"batch pass {rep + 1}/{args.repeats}")

            async def staggered(i, rep=rep):
                await asyncio.sleep(i * args.stagger_ms / 1000)
                return await ask(client, base, model, prompts[i]["text"], args.max_tokens)

            batches.append(await common.run_bounded(staggered, len(prompts), len(prompts)))
    return prompts, s1, s2, batches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_server_args(ap)
    common.add_output_args(ap)
    ap.add_argument("--n", type=int, default=32, help="prompts, and batch size")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--long-questions", type=int, default=40, help="GSM8K questions in a 'long' prompt (~60 tokens each)")
    ap.add_argument("--stagger-ms", type=float, default=15.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--min-prefix", type=int, default=16, help="divergence before this token index is 'gross'")
    ap.add_argument("--max-gross-frac", type=float, default=0.10, help="allowed fraction of grossly divergent prompts")
    ap.add_argument("--no-solo-repeat", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    common.require_platform(args)
    base = common.base_url(args)
    model = common.resolve_model(args, base)
    prompts, s1, s2, batches = asyncio.run(main_async(args, base, model))

    def compare(ref, other):
        rows = []
        for p, a, b in zip(prompts, ref, other):
            d = first_divergence(a, b)
            rows.append(dict(id=p["id"], kind=p["kind"], first_divergence=d,
                             n_ref=len(a["tokens"]) if a["tokens"] is not None else len(a["text"]),
                             empty=(not b["text"].strip() and not b["tokens"])))
        return rows

    def summarize(rows):
        ident = sum(r["first_divergence"] is None for r in rows)
        gross = sum(r["first_divergence"] is not None and r["first_divergence"] < args.min_prefix for r in rows)
        empty = sum(r["empty"] for r in rows)
        divs = [r["first_divergence"] for r in rows if r["first_divergence"] is not None]
        by_kind = {}
        for k in ("short", "medium", "long"):
            kr = [r for r in rows if r["kind"] == k]
            by_kind[k] = dict(n=len(kr), identical=sum(r["first_divergence"] is None for r in kr),
                              gross=sum(r["first_divergence"] is not None and r["first_divergence"] < args.min_prefix
                                        for r in kr))
        return dict(n=len(rows), identical=ident, identical_frac=ident / max(len(rows), 1),
                    gross_divergence=gross, empty_outputs=empty,
                    median_first_divergence=statistics.median(divs) if divs else None, by_kind=by_kind)

    noise = summarize(compare(s1, s2)) if s2 else None
    batch_rows, batch_sum = [], []
    for i, b in enumerate(batches):
        rows = compare(s1, b)
        batch_rows += [dict(r, batch_run=i) for r in rows]
        batch_sum.append(summarize(rows))
    gross_total = sum(s["gross_divergence"] + s["empty_outputs"] for s in batch_sum)
    passed = all(s["gross_divergence"] <= args.max_gross_frac * s["n"] and s["empty_outputs"] == 0 for s in batch_sum)
    bvb = summarize(compare(batches[0], batches[1])) if len(batches) > 1 else None
    finishes = {}
    for b in batches[0]:
        finishes[b["finish"]] = finishes.get(b["finish"], 0) + 1
    ident = [s["identical_frac"] for s in batch_sum]
    head = (f"batch of {args.n} vs solo: identical outputs "
            f"{', '.join(f'{100 * x:.0f}%' for x in ident)}; gross divergences {gross_total}")
    if noise:
        head += f"; solo-vs-solo identical {100 * noise['identical_frac']:.0f}%"
    if bvb:
        head += f"; batch-vs-batch identical {100 * bvb['identical_frac']:.0f}%"
    notes = []
    if any(r["first_divergence"] is not None and r["n_ref"] and
           isinstance(r["first_divergence"], int) for r in batch_rows):
        notes.append("late divergence (>= min-prefix) is BF16 batch-shape drift and is not a failure")
    if s1[0]["tokens"] is None:
        notes.append("server did not return token_ids; compared output text by characters")
    common.write_accuracy_result(
        args, "batch_invariance", headline=head, score=(sum(ident) / len(ident)) if ident else None,
        passed=passed,
        criterion=f"<= {args.max_gross_frac:.0%} of prompts diverge before token {args.min_prefix}, no empty output, in every batch run",
        metrics=dict(batch_runs=batch_sum, solo_vs_solo=noise, batch_vs_batch=bvb, finish_reasons_batch0=finishes),
        params=dict(n=args.n, max_tokens=args.max_tokens, long_questions=args.long_questions,
                    stagger_ms=args.stagger_ms, repeats=args.repeats, min_prefix=args.min_prefix),
        server=common.server_info(base), details=batch_rows, notes=notes,
    )
    print(head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
