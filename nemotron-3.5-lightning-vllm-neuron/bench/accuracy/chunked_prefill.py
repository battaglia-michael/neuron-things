#!/usr/bin/env python3
"""Chunked-prefill check: prompts longer than --max-num-batched-tokens must still match HF.

The server must be started with the chunk size given here as --max-num-batched-tokens.
Each reference sequence longer than one chunk is prefilled alone (concurrency 1 by default,
so the chunk boundaries fall at multiples of the chunk size) with prompt_logprobs, and the
server's top-1 token is compared with the HF fp32 label at every position. Results are
grouped by chunk index. If recurrent state or KV is lost or corrupted at a chunk boundary,
the match rate drops from chunk 1 on.

Modes: `logprobs` (one prompt_logprobs prefill per sequence) or `window` (no prompt_logprobs
needed: for sampled positions p, prefill tokens[:p] and compare the single next token with
the label; a prompt cut at position p in chunk k has crossed k chunk boundaries). `auto`
tries logprobs and falls back to window. vllm-neuron 0.29 does not return full
prompt_logprobs, so expect window mode there.

Pass rule: for each chunk k >= 1 with enough samples, the match rate is not significantly
below the floor (99% Wilson upper bound under it). The floor is
  - HF BF16's own rate in that chunk minus --margin-pp, if the references carry BF16 labels;
  - otherwise the chunk-0 rate minus --fallback-drop-pp (default 5 pp).
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import _tf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_server_args(ap)
    common.add_output_args(ap)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--max-num-batched-tokens", type=int, required=True,
                    help="the server's chunk size; sequences must be longer than this")
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--margin-pp", type=float, default=1.0)
    ap.add_argument("--fallback-drop-pp", type=float, default=5.0)
    ap.add_argument("--min-samples", type=int, default=200, help="positions a chunk needs to be judged (window mode: 20)")
    ap.add_argument("--mode", choices=("auto", "logprobs", "window"), default="auto")
    ap.add_argument("--samples-per-chunk", type=int, default=30, help="window mode: sampled positions per chunk per sequence")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    common.require_platform(args)
    base = common.base_url(args)
    model = common.resolve_model(args, base)
    chunk = args.max_num_batched_tokens
    refs = [r for r in _tf.load_references(args.refs) if len(r["token_ids"]) > chunk]
    if args.max_seqs:
        refs = refs[: args.max_seqs]
    if not refs:
        sys.exit(f"no reference sequence is longer than {chunk} tokens; lower "
                 "--max-num-batched-tokens on the server and here, or add longer references")
    common.log(f"{len(refs)} sequences longer than {chunk} tokens")

    def bucket_fn(pos: int) -> str:
        return f"chunk {pos // chunk}"

    async def run_logprobs():
        pairs, details, errors, used = [], [], [], {}
        async with common.make_client(args.concurrency) as client:
            async def one(i):
                r = refs[i]
                n = min(len(r["token_ids"]), args.max_seq_len or 10**9)
                try:
                    return n, await _tf.server_argmax_prefill(client, base, model, r["token_ids"][:n]), None
                except Exception as e:  # noqa: BLE001
                    return n, None, str(e)
            res = await common.run_bounded(one, len(refs), args.concurrency, progress_every=2)
        for r, (n, srv, err) in zip(refs, res):
            if err:
                errors.append(dict(id=r["id"], error=err))
                continue
            used[r["id"]] = n
            m = 0
            for i in range(1, n):
                if srv[i] is None:
                    continue
                ok = srv[i] == r["argmax_ids"][i - 1]
                pairs.append((i, ok))
                m += ok
            details.append(dict(id=r["id"], n_tokens=n, n_chunks=-(-n // chunk), rate=m / max(n - 1, 1)))
        return pairs, details, errors, used

    async def run_window():
        rng = random.Random(args.seed)
        jobs = []
        for r in refs:
            n = min(len(r["token_ids"]), args.max_seq_len or 10**9)
            for k in range(-(-n // chunk)):
                lo, hi = max(8, k * chunk), min((k + 1) * chunk, n - 1)
                if hi <= lo:
                    continue
                jobs += [(r, rng.randrange(lo, hi)) for _ in range(args.samples_per_chunk)]
        pairs, details, errors = [], [], []
        async with common.make_client(args.concurrency) as client:
            async def one(j):
                r, p = jobs[j]
                try:
                    return (await _tf.server_greedy_tokens(client, base, model, r["token_ids"][:p], 1))[0], None
                except Exception as e:  # noqa: BLE001
                    return None, str(e)
            res = await common.run_bounded(one, len(jobs), args.concurrency, progress_every=10)
        for (r, p), (tok, err) in zip(jobs, res):
            if err:
                errors.append(dict(id=r["id"], start=p, error=err))
                continue
            ok = tok == r["argmax_ids"][p - 1]
            pairs.append((p, ok))
            details.append(dict(id=r["id"], position=p, chunk=p // chunk, match=bool(ok)))
        return pairs, details, errors, None

    mode = args.mode
    if mode in ("auto", "logprobs"):
        pairs, details, errors, used = asyncio.run(run_logprobs())
        if mode == "auto" and not pairs and errors and "prompt_logprobs" in errors[0]["error"].lower():
            common.log(f"prompt_logprobs failed ({errors[0]['error'][:120]}); falling back to window mode")
            mode = "window"
        else:
            mode = "logprobs"
    if mode == "window":
        pairs, details, errors, used = asyncio.run(run_window())
        args.min_samples = min(args.min_samples, 20)
    stats = _tf.tally(pairs, bucket_fn)
    paired = None
    if mode == "window":
        bf_pairs = [(d["position"], int(r["bf16_argmax_ids"][d["position"] - 1]) == r["argmax_ids"][d["position"] - 1])
                    for d in details for r in [next(x for x in refs if x["id"] == d["id"])] if r.get("bf16_argmax_ids")]
        paired = _tf.tally(bf_pairs, bucket_fn) if bf_pairs else None
    line = _tf.reference_line(refs, bucket_fn, use_len=used) if mode == "logprobs" else _tf.reference_line(refs, bucket_fn)
    chunks = sorted((b for b in stats if b != "all"), key=lambda s: int(s.split()[1]))
    verdict, why = None, []
    judged = [b for b in chunks if b != "chunk 0" and stats[b]["n"] >= args.min_samples]
    if judged:
        verdict = True
        for b in judged:
            if line and b in line and line[b]["rate"] is not None:
                floor = line[b]["rate"] - args.margin_pp / 100
                src = "HF BF16"
            elif "chunk 0" in stats:
                floor = stats["chunk 0"]["rate"] - args.fallback_drop_pp / 100
                src = "chunk 0"
            else:
                continue
            ok = not _tf.below_floor(stats[b], floor)
            verdict &= ok
            why.append(f"{b}: {100 * stats[b]['rate']:.1f}% vs floor {100 * floor:.1f}% ({src}) -> {'ok' if ok else 'FAIL'}")
    if errors and not pairs:
        verdict = False
    head = "; ".join(f"{b}: {100 * stats[b]['rate']:.1f}%" for b in chunks) or "no data"
    common.write_accuracy_result(
        args, "chunked_prefill", headline=head, score=stats["all"]["rate"], passed=verdict,
        criterion=f"chunk>=1 match >= HF BF16 - {args.margin_pp} pp (or chunk 0 - {args.fallback_drop_pp} pp "
                  f"without BF16 labels); chunks need >= {args.min_samples} positions",
        metrics=dict(mode=mode, chunk_size=chunk, chunks=_tf.json_safe(stats), hf_bf16_line=_tf.json_safe(line),
                     checks=why, n_sequences=len(refs), n_errors=len(errors),
                     hf_bf16_same_positions=_tf.json_safe(paired)),
        params=dict(refs=str(args.refs), max_num_batched_tokens=chunk, concurrency=args.concurrency,
                    mode=mode, samples_per_chunk=args.samples_per_chunk, seed=args.seed, max_seq_len=args.max_seq_len),
        server=common.server_info(base), details=details + errors,
        notes=([f"{len(errors)} request(s) failed; first: {errors[0]['error'][:300]}"] if errors else []),
    )
    print(head)
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())
