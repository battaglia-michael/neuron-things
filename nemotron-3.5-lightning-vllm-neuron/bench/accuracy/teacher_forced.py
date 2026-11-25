#!/usr/bin/env python3
"""Teacher-forced greedy token match against HF fp32 references, by position bucket.

Reference labels come from task A03 (format: see _tf.py / bench/README.md).

Modes
  logprobs  One prefill per sequence. The whole reference sequence is the prompt and the
            server returns its own top-1 token at every position (prompt_logprobs=1).
            That is true teacher forcing: every position sees the reference prefix.
  window    Needs no prompt_logprobs. For sampled start positions p the server gets
            tokens[:p] and greedily decodes WINDOW tokens. The first generated token is a
            prefill-path teacher-forced sample and is what the rate counts. Later tokens are
            reported separately: they are comparable only while the server's tokens equal the
            sequence's own tokens (biased toward easy tokens).
  auto      logprobs; if the server rejects prompt_logprobs, fall back to window.

Pass rule (T05): match rate in every bucket with enough samples is at least the HF BF16
rate minus --margin-pp (default 1.0); a bucket fails only if it is significantly below that
floor (99% Wilson upper bound under it), so tiny samples do not fail on one flipped token. Without a BF16 line the verdict is null unless
--min-match is given.
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


async def run_logprobs(args, base, model, refs):
    pairs, details, errors = [], [], []
    used_len: dict[str, int] = {}
    async with common.make_client(args.concurrency) as client:
        async def one(i):
            r = refs[i]
            n = min(len(r["token_ids"]), args.max_seq_len or 10**9)
            toks = r["token_ids"][:n]
            try:
                srv = await _tf.server_argmax_prefill(client, base, model, toks)
            except Exception as e:  # noqa: BLE001
                return r["id"], n, None, str(e)
            return r["id"], n, srv, None

        res = await common.run_bounded(one, len(refs), args.concurrency, progress_every=5)
    for (rid, n, srv, err), r in zip(res, refs):
        if err:
            errors.append(dict(id=rid, error=err))
            continue
        used_len[rid] = n
        m = 0
        for i in range(1, n):
            if srv[i] is None:
                continue
            ok = srv[i] == r["argmax_ids"][i - 1]
            pairs.append((i, ok))
            m += ok
        details.append(dict(id=rid, n_tokens=n, matched=m, total=n - 1, rate=m / max(n - 1, 1)))
    return pairs, details, errors, used_len


async def run_window(args, base, model, refs):
    rng = random.Random(args.seed)
    jobs = []
    if args.samples_per_bucket:
        # A fixed number of start positions per bucket, spread round-robin over the
        # sequences that reach into the bucket (long buckets have fewer sequences).
        for lo, hi, name in _tf.BUCKETS:
            if name in (args.skip_buckets or "").split(","):
                continue
            elig = []
            for r in refs:
                n = min(len(r["token_ids"]), args.max_seq_len or 10**9)
                hi_eff, lo_eff = min(hi, n - args.window), max(lo, 8)
                if hi_eff > lo_eff:
                    elig.append((r, lo_eff, hi_eff))
            for k in range(args.samples_per_bucket if elig else 0):
                r, lo_eff, hi_eff = elig[k % len(elig)]
                jobs.append((r, rng.randrange(lo_eff, hi_eff)))
    for r in refs if not args.samples_per_bucket else []:
        n = min(len(r["token_ids"]), args.max_seq_len or 10**9)
        for lo, hi, name in _tf.BUCKETS:
            hi_eff = min(hi, n - args.window)
            lo_eff = max(lo, 8)
            if hi_eff <= lo_eff:
                continue
            for _ in range(args.windows_per_bucket):
                jobs.append((r, rng.randrange(lo_eff, hi_eff)))
    pairs, details, errors = [], [], []
    async with common.make_client(args.concurrency) as client:
        async def one(j):
            r, p = jobs[j]
            try:
                gen = await _tf.server_greedy_tokens(client, base, model, r["token_ids"][:p], args.window)
            except Exception as e:  # noqa: BLE001
                return None, str(e)
            return gen, None

        res = await common.run_bounded(one, len(jobs), args.concurrency, progress_every=10)
    cond_n = cond_ok = 0
    # Paired HF BF16 lines: HF BF16's own match at exactly the sampled positions, for the
    # first-token samples and for the decode-path samples.
    paired_first, paired_dec, dec_pairs = [], [], []
    for (r, p), (gen, err) in zip(jobs, res):
        if err:
            errors.append(dict(id=r["id"], start=p, error=err))
            continue
        bf = r.get("bf16_argmax_ids")
        if bf:
            paired_first.append((p, int(bf[p - 1]) == r["argmax_ids"][p - 1]))
            for j in range(1, len(gen)):
                if p + j >= len(r["token_ids"]) or gen[j - 1] != r["token_ids"][p + j - 1]:
                    break
                dec_pairs.append((p + j, gen[j] == r["argmax_ids"][p + j - 1]))
                paired_dec.append((p + j, int(bf[p + j - 1]) == r["argmax_ids"][p + j - 1]))
        # Generated token j predicts sequence index p+j; its label is argmax_ids[p+j-1].
        # Token 0 sees exactly the reference prefix, so it is a valid teacher-forced sample.
        # Token j>0 is only comparable while every earlier generated token equals the
        # sequence's own token (then the server context is still the reference context).
        first_ok, cn, co = False, 0, 0
        for j, t in enumerate(gen):
            if p + j >= len(r["token_ids"]):
                break
            if j > 0 and gen[j - 1] != r["token_ids"][p + j - 1]:
                break
            ok = t == r["argmax_ids"][p + j - 1]
            if j == 0:
                pairs.append((p, ok))
                first_ok = ok
            else:
                cn += 1
                co += ok
        cond_n += cn
        cond_ok += co
        details.append(dict(id=r["id"], start=p, window=len(gen), first_token_match=bool(first_ok),
                            decode_comparable=cn, decode_matched=co))
    run_window.cond = (cond_n, cond_ok)
    run_window.paired = dict(
        first_token_hf_bf16_same_positions=_tf.tally(paired_first) if paired_first else None,
        decode_path_server=_tf.tally(dec_pairs) if dec_pairs else None,
        decode_path_hf_bf16_same_positions=_tf.tally(paired_dec) if paired_dec else None)
    return pairs, details, errors, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_server_args(ap)
    common.add_output_args(ap)
    ap.add_argument("--refs", required=True, help="reference .jsonl file or directory (task A03)")
    ap.add_argument("--mode", choices=("auto", "logprobs", "window"), default="auto")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="truncate sequences to this many tokens (set to server max-model-len - 8)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--window", type=int, default=32, help="window mode: tokens decoded per sample")
    ap.add_argument("--windows-per-bucket", type=int, default=3)
    ap.add_argument("--samples-per-bucket", type=int, default=None,
                    help="window mode: this many start positions per bucket, round-robin over the "
                         "sequences that reach the bucket (overrides --windows-per-bucket)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-buckets", default=None, help="with --samples-per-bucket: comma list of buckets to skip, e.g. 0-1K")
    ap.add_argument("--margin-pp", type=float, default=1.0, help="allowed gap below HF BF16, percentage points")
    ap.add_argument("--min-samples", type=int, default=200, help="a bucket needs this many positions to be judged")
    ap.add_argument("--min-match", type=float, default=None, help="absolute floor (0-1) when no BF16 line exists")
    args = ap.parse_args()

    common.require_platform(args)
    base = common.base_url(args)
    model = common.resolve_model(args, base)
    refs = _tf.load_references(args.refs, args.max_seqs)
    common.log(f"{len(refs)} reference sequences, lengths {min(len(r['token_ids']) for r in refs)}"
               f"-{max(len(r['token_ids']) for r in refs)}")

    mode = args.mode
    errors: list = []
    if mode in ("auto", "logprobs"):
        pairs, details, errors, used_len = asyncio.run(run_logprobs(args, base, model, refs))
        if mode == "auto" and not pairs and errors and "prompt_logprobs" in errors[0]["error"].lower():
            common.log(f"prompt_logprobs failed ({errors[0]['error'][:120]}); falling back to window mode")
            mode = "window"
        else:
            mode = "logprobs"
    if mode == "window":
        pairs, details, errors, used_len = asyncio.run(run_window(args, base, model, refs))

    stats = _tf.tally(pairs)
    if mode == "window":
        min_samples = max(5, args.min_samples // 20)
    else:
        min_samples = args.min_samples
    line = _tf.reference_line(refs, use_len=used_len) if mode == "logprobs" else _tf.reference_line(refs)
    verdict, gaps = None, {}
    judged = [b for b, v in stats.items() if b != "all" and v["n"] >= min_samples]
    if line:
        ok = True
        for b in judged:
            if b in line and line[b]["rate"] is not None:
                gap = (stats[b]["rate"] - line[b]["rate"]) * 100
                gaps[b] = round(gap, 3)
                ok &= not _tf.below_floor(stats[b], line[b]["rate"] - args.margin_pp / 100)
        verdict = ok if gaps else None
        criterion = f"each bucket >= HF BF16 rate - {args.margin_pp} pp (buckets with >= {min_samples} samples)"
    elif args.min_match is not None and judged:
        verdict = not any(_tf.below_floor(stats[b], args.min_match) for b in judged)
        criterion = f"each bucket >= {args.min_match:.2f} (no BF16 line available)"
    else:
        criterion = "no BF16 line in references and no --min-match: verdict not computed"
    if errors and not pairs:
        verdict = False

    head = f"{_tf.fmt_rate(stats['all'])} overall"
    for lo, hi, name in _tf.BUCKETS:
        if name in stats:
            head += f"; {name}: {100 * stats[name]['rate']:.1f}%"
    notes = []
    if errors:
        notes.append(f"{len(errors)} request(s) failed; first: {errors[0]['error'][:300]}")
    if mode == "window":
        cn, co = run_window.cond
        notes.append("window mode: the rate counts only the first generated token of each window (exact "
                     "reference context). Later tokens are comparable only while the server's tokens equal "
                     f"the sequence's own; those decode-path samples matched {co}/{cn}"
                     + (f" ({100 * co / cn:.1f}%)" if cn else "") + " and are biased toward easy tokens.")
    metrics = dict(decode_path_conditional=(dict(n=run_window.cond[0], matched=run_window.cond[1])
                                            if mode == "window" else None), mode=mode, buckets=_tf.json_safe(stats), hf_bf16_line=_tf.json_safe(line),
                   gap_vs_bf16_pp=gaps, n_sequences=len(refs), n_errors=len(errors),
                   paired=(_tf.json_safe(run_window.paired) if mode == "window" else None))
    common.write_accuracy_result(
        args, "teacher_forced", headline=head, metrics=metrics, score=stats["all"]["rate"], passed=verdict,
        criterion=criterion, server=common.server_info(base), details=details + errors, notes=notes,
        params=dict(refs=str(args.refs), mode=mode, window=args.window, max_seq_len=args.max_seq_len,
                    margin_pp=args.margin_pp, seed=args.seed, samples_per_bucket=args.samples_per_bucket, skip_buckets=args.skip_buckets,
                    windows_per_bucket=args.windows_per_bucket, concurrency=args.concurrency),
    )
    print(head)
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())
