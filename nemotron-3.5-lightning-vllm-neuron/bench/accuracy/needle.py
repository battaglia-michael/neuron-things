#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval at 1K / 16K / 32K prompt tokens (128K with --include-128k).

A synthetic, non-repeating filler document (seeded) of the target token length hides one
sentence, "The special magic number for <city> is <7 digits>.", at a chosen depth. The model
is asked for the number, greedy. The chat template (thinking off) is rendered locally so the
prompt length is exact, and the text goes to /v1/completions. A trial passes
when the exact digit string is in the answer. Several depths are tested per length.

Prompt lengths are measured with the model's tokenizer (chat template included), so the
target is met to within a few tokens; the server's reported prompt_tokens are stored too.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

ADJ = "quiet old bright narrow heavy distant green hollow simple crowded patient golden rough early lonely wide".split()
NOUN = ("river market lantern harbor orchard bridge teacher engine garden window ledger meadow "
        "station village mirror carpenter valley harvest signal archive").split()
VERB = "crossed watched repaired followed measured described carried painted counted visited sorted rebuilt".split()
PLACE = "the north gate|the old mill|a small chapel|the east road|a stone quay|the market square|the river bend".split("|")
CITIES = ("Valdoria Kestrelburg Ombrelle Tarnwick Lysander Quillon Marrowdale Ostrava-Nine Pellucida "
          "Brannoch Corvane Haldersby Ismere Juniper-Falls Norvenna").split()


def make_sentences(n: int, rng: random.Random) -> list[str]:
    out = []
    for _ in range(n):
        a, b = rng.sample(ADJ, 2)
        s = (f"The {a} {rng.choice(NOUN)} {rng.choice(VERB)} the {b} {rng.choice(NOUN)} near "
             f"{rng.choice(PLACE)}, and {rng.randint(2, 97)} {rng.choice(NOUN)}s {rng.choice(VERB)} "
             f"the {rng.choice(ADJ)} {rng.choice(NOUN)} before day {rng.randint(100, 999)}.")
        out.append(s)
    return out


def build_prompt(tok, target: int, depth: float, city: str, number: str, pool, lens, rng) -> tuple[str, int]:
    needle = f"The special magic number for {city} is {number}."
    question = f"What is the special magic number for {city}? Answer with only the number."

    def render(doc: str) -> str:
        msg = [{"role": "user", "content": "Read the document below and answer the question at the end.\n\n"
                                             f"<document>\n{doc}\n</document>\n\nQuestion: {question}"}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    overhead = len(tok(render(""), add_special_tokens=False).input_ids) + \
        len(tok(needle + "\n", add_special_tokens=False).input_ids)
    budget = target - overhead
    if budget < 32:
        raise ValueError(f"target {target} too small for the template ({overhead} tokens of overhead)")
    start = rng.randrange(len(pool))
    sel, total, i = [], 0, start
    while total < budget:
        sel.append(pool[i % len(pool)])
        total += lens[i % len(pool)]
        i += 1
    k = int(round(depth * len(sel)))
    sel.insert(k, needle)
    text = render("\n".join(sel))
    return text, len(tok(text, add_special_tokens=False).input_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_server_args(ap, need_tokenizer=True)
    common.add_output_args(ap)
    ap.add_argument("--lengths", default="1024,16384,32768", help="comma list of prompt token lengths")
    ap.add_argument("--include-128k", action="store_true", help="also test 131072 tokens (Stretch)")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--max-model-len", type=int, default=None, help="skip lengths that do not fit")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--min-acc", type=float, default=0.8, help="pass floor per length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.tokenizer:
        sys.exit("--tokenizer is required (model path or HF id)")
    common.require_platform(args)
    base = common.base_url(args)
    model = common.resolve_model(args, base)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    lengths = [int(x) for x in args.lengths.split(",") if x]
    if args.include_128k and 131072 not in lengths:
        lengths.append(131072)
    depths = [float(x) for x in args.depths.split(",")]
    rng = random.Random(args.seed)
    pool = make_sentences(max(4000, max(lengths) // 10), rng)
    lens = [len(x) for x in tok([s + "\n" for s in pool], add_special_tokens=False).input_ids]

    trials, skipped = [], []
    for L in lengths:
        if args.max_model_len and L + args.max_tokens > args.max_model_len:
            skipped.append(L)
            common.log(f"length {L}: skipped, does not fit --max-model-len {args.max_model_len}")
            continue
        for d in depths:
            city = rng.choice(CITIES)
            number = str(rng.randint(1_000_000, 9_999_999))
            text, n_tok = build_prompt(tok, L, d, city, number, pool, lens, rng)
            trials.append(dict(length=L, depth=d, city=city, number=number, prompt=text, local_tokens=n_tok))
    common.log(f"{len(trials)} trials over lengths {[x for x in lengths if x not in skipped]}")

    async def run():
        async with common.make_client(args.concurrency, timeout_s=7200) as client:
            async def one(i):
                t = trials[i]
                payload = {"model": model, "prompt": t["prompt"], "max_tokens": args.max_tokens,
                           "temperature": 0.0, "add_special_tokens": False}
                t0 = time.time()
                try:
                    res = await common.post_json(client, f"{base}/v1/completions", payload, retries=0)
                except Exception as e:  # noqa: BLE001
                    return dict(error=str(e)[:300])
                return dict(text=res["choices"][0].get("text", ""), elapsed_s=time.time() - t0,
                            prompt_tokens=res.get("usage", {}).get("prompt_tokens"))
            return await common.run_bounded(one, len(trials), args.concurrency, progress_every=1)

    outs = asyncio.run(run()) if trials else []
    per_len: dict[int, dict] = {}
    details = []
    for t, o in zip(trials, outs):
        row = per_len.setdefault(t["length"], dict(n=0, correct=0, errors=0, prompt_tokens=[], elapsed_s=[]))
        row["n"] += 1
        if "error" in o:
            row["errors"] += 1
            details.append(dict(length=t["length"], depth=t["depth"], error=o["error"]))
            continue
        ok = t["number"] in o["text"].replace(",", "").replace(" ", "")
        row["correct"] += ok
        row["prompt_tokens"].append(o["prompt_tokens"])
        row["elapsed_s"].append(round(o["elapsed_s"], 2))
        details.append(dict(length=t["length"], depth=t["depth"], correct=bool(ok), expected=t["number"],
                            answer=o["text"][:80], local_tokens=t["local_tokens"],
                            server_prompt_tokens=o["prompt_tokens"], elapsed_s=round(o["elapsed_s"], 2)))
    by_len = {}
    for L, r in per_len.items():
        pts = [p for p in r["prompt_tokens"] if p]
        by_len[str(L)] = dict(n=r["n"], correct=r["correct"], accuracy=r["correct"] / r["n"], errors=r["errors"],
                              mean_prompt_tokens=(sum(pts) / len(pts) if pts else None),
                              mean_elapsed_s=(sum(r["elapsed_s"]) / len(r["elapsed_s"]) if r["elapsed_s"] else None))
    tot_n = sum(r["n"] for r in per_len.values())
    tot_ok = sum(r["correct"] for r in per_len.values())
    passed = (all(v["accuracy"] >= args.min_acc for v in by_len.values()) and bool(by_len)) if by_len else None
    def tag(L):
        return f"{int(L) // 1024}K" if int(L) % 1024 == 0 else f"{int(L)} tok"
    head = "; ".join(f"{tag(L)}: {v['correct']}/{v['n']}" for L, v in by_len.items()) or "no trials ran"
    notes = []
    if skipped:
        notes.append(f"skipped lengths (do not fit max-model-len): {skipped}")
    common.write_accuracy_result(
        args, "needle", headline=head, score=(tot_ok / tot_n if tot_n else None), passed=passed,
        criterion=f"accuracy >= {args.min_acc:.0%} at every tested length",
        metrics=dict(by_length=by_len, skipped_lengths=skipped, depths=depths),
        params=dict(lengths=lengths, depths=depths, seed=args.seed, max_tokens=args.max_tokens),
        server=common.server_info(base), details=details, notes=notes,
    )
    print(head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
