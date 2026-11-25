#!/usr/bin/env python3
"""GSM8K (all 1,319 test questions), zero-shot chain of thought, thinking off, greedy.

Talks to /v1/chat/completions with chat_template_kwargs {"enable_thinking": false},
temperature 0, 64 requests in flight. The test split is vendored in bench/data/
(openai/grade-school-math, MIT) so the script needs no dataset download.

The score is exact numeric match of the extracted final answer. It is a parity check
between platforms and against HF BF16, not a measure of the model's use case.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

PROMPT = (
    "Solve the following math problem step by step. When you are done, write the final "
    "numeric answer on its own last line in exactly this form: Final answer: <number>\n\n"
    "Problem: {q}"
)
NUM = r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+"


def to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract(text: str) -> tuple[float | None, bool]:
    """Returns (answer, followed_format)."""
    m = re.findall(r"final answer\s*(?:is)?\s*[:\-]?\s*[\*\$\\\(\[\{ ]*(" + NUM + ")", text, flags=re.I)
    if m:
        return to_float(m[-1]), True
    m = re.findall(r"\\boxed\{\s*\$?\s*(" + NUM + ")", text)
    if m:
        return to_float(m[-1]), False
    nums = re.findall(NUM, text)
    return (to_float(nums[-1]) if nums else None), False


def gold(answer: str) -> float:
    return float(answer.split("####")[-1].strip().replace(",", ""))


async def run(args, base, model, items):
    async with common.make_client(args.concurrency) as client:
        async def one(i):
            payload = common.chat_payload(
                model, [{"role": "user", "content": PROMPT.format(q=items[i]["question"])}],
                max_tokens=args.max_tokens, thinking=False, temperature=0.0)
            try:
                res = await common.post_json(client, f"{base}/v1/chat/completions", payload)
            except Exception as e:  # noqa: BLE001
                return dict(error=str(e)[:300])
            ch = res["choices"][0]
            return dict(text=common.message_text(ch), finish=ch.get("finish_reason"),
                        completion_tokens=res.get("usage", {}).get("completion_tokens"))
        return await common.run_bounded(one, len(items), args.concurrency, progress_every=100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_server_args(ap)
    common.add_output_args(ap)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (smoke tests)")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--hf-bf16-acc", type=float, default=None,
                    help="HF BF16 accuracy (0-1) to compare with; pass = within --margin-pp")
    ap.add_argument("--margin-pp", type=float, default=2.0)
    args = ap.parse_args()

    common.require_platform(args)
    base = common.base_url(args)
    model = common.resolve_model(args, base)
    items = common.read_jsonl(common.DATA_DIR / "gsm8k_test.jsonl")
    if args.limit:
        items = items[: args.limit]
    common.log(f"GSM8K: {len(items)} questions, concurrency {args.concurrency}")
    outs = asyncio.run(run(args, base, model, items))

    correct = errors = truncated = formatted = 0
    toks, details = [], []
    for it, o in zip(items, outs):
        g = gold(it["answer"])
        if "error" in o:
            errors += 1
            details.append(dict(gold=g, error=o["error"]))
            continue
        pred, fmt = extract(o["text"])
        ok = pred is not None and abs(pred - g) < 1e-4
        correct += ok
        truncated += o["finish"] == "length"
        formatted += fmt
        if o["completion_tokens"]:
            toks.append(o["completion_tokens"])
        details.append(dict(gold=g, pred=pred, correct=bool(ok), finish=o["finish"],
                            completion_tokens=o["completion_tokens"], tail=o["text"][-160:]))
    n = len(items)
    acc = correct / n
    ci = 1.96 * math.sqrt(acc * (1 - acc) / n)
    passed = None
    crit = "no HF BF16 accuracy given (--hf-bf16-acc): verdict not computed"
    if args.hf_bf16_acc is not None:
        passed = acc >= args.hf_bf16_acc - args.margin_pp / 100
        crit = f"accuracy >= HF BF16 ({100 * args.hf_bf16_acc:.2f}%) - {args.margin_pp} pp"
    head = f"{100 * acc:.2f}% ({correct}/{n}), 95% CI ±{100 * ci:.1f} pp"
    notes = []
    if errors:
        notes.append(f"{errors} request(s) failed and are counted as wrong")
    if args.limit:
        notes.append(f"limited to the first {args.limit} questions: not the full 1,319")
    common.write_accuracy_result(
        args, "gsm8k", headline=head, score=acc, passed=passed, criterion=crit,
        metrics=dict(n=n, correct=correct, accuracy=acc, ci95_pp=100 * ci, errors=errors,
                     truncated=truncated, followed_format=formatted,
                     mean_completion_tokens=(sum(toks) / len(toks) if toks else None),
                     hf_bf16_acc=args.hf_bf16_acc),
        params=dict(concurrency=args.concurrency, max_tokens=args.max_tokens, temperature=0,
                    enable_thinking=False, limit=args.limit, prompt=PROMPT),
        server=common.server_info(base), details=details, notes=notes,
    )
    print(head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
