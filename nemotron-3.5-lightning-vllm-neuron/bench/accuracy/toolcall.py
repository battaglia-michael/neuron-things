#!/usr/bin/env python3
"""Tool calling on BFCL v3 'simple' and 'multiple' (200 items), thinking off, greedy.

The server must run with tool parsing, for Nemotron 3.5 Lightning:
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3

Each item is sent to /v1/chat/completions with `tools` and tool_choice "auto". The first
returned tool call is scored on function name and arguments with a BFCL-style AST check:
  - the name must equal the ground-truth function (dots in names are sent as '_' because
    OpenAI-style names cannot contain '.', and compared the same way);
  - every argument must be one of the allowed values for that parameter (strings compare
    case-insensitively with ' ,./-_*^' removed, numbers by value, lists/dicts recursively);
  - a parameter whose allowed values include "" may be omitted; others must be present;
  - arguments that are not in the ground truth fail the item.
It follows the leaderboard's AST checker closely but is not the official scorer, so scores
are for comparing platforms and HF BF16 with each other, not with the public leaderboard.
Items come from bench/data/bfcl_subset.jsonl (made by prepare_toolcall_data.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "integer": "integer",
            "string": "string", "array": "array", "boolean": "boolean", "number": "number",
            "object": "object"}


def to_json_schema(o):
    """BFCL parameter schemas use python-ish type names; map them to JSON Schema."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k == "type" and isinstance(v, str):
                if v == "any":
                    continue
                out[k] = TYPE_MAP.get(v, v)
            else:
                out[k] = to_json_schema(v)
        return out
    if isinstance(o, list):
        return [to_json_schema(v) for v in o]
    return o


def wire_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def build_tools(functions: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": wire_name(f["name"]), "description": f.get("description", ""),
        "parameters": to_json_schema(f.get("parameters", {"type": "dict", "properties": {}}))}}
        for f in functions]


# ---- scoring ---------------------------------------------------------------
def norm(s: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def match_one(p, a) -> bool:
    if isinstance(a, dict):
        if not isinstance(p, dict) or set(p) - set(a):
            return False
        for k, opts in a.items():
            if k not in p:
                if not (isinstance(opts, list) and "" in opts):
                    return False
                continue
            if not match_any(p[k], opts if isinstance(opts, list) else [opts]):
                return False
        return True
    if isinstance(a, list):
        if isinstance(p, list):
            if len(p) == len(a) and all(match_one(pi, ai) for pi, ai in zip(p, a)):
                return True
        return any(match_one(p, ai) for ai in a)
    if isinstance(a, bool):
        return p is a
    if is_num(a):
        return is_num(p) and abs(p - a) <= 1e-6 * max(1.0, abs(a))
    if isinstance(a, str):
        return isinstance(p, str) and norm(p) == norm(a)
    return p == a


def match_any(p, options: list) -> bool:
    return any(match_one(p, o) for o in options if o != "") or (p == "" and "" in options)


def score_args(args: dict, gt: dict) -> str | None:
    """None if correct, else a short reason."""
    for k in args:
        if k not in gt:
            return f"unexpected parameter {k!r}"
    for k, opts in gt.items():
        if k not in args:
            if "" in opts:
                continue
            return f"missing parameter {k!r}"
        if not match_any(args[k], opts):
            return f"parameter {k!r}: got {json.dumps(args[k])[:80]}, allowed {json.dumps(opts)[:80]}"
    return None


def score_item(item: dict, message: dict) -> tuple[str, str]:
    """Returns (outcome, detail). Outcome: ok | no_tool_call | multiple_calls | wrong_name |
    bad_json | wrong_args."""
    calls = message.get("tool_calls") or []
    if not calls:
        return "no_tool_call", (message.get("content") or "")[:160]
    gt_name, gt_args = next(iter(item["ground_truth"].items()))
    fn = calls[0]["function"]
    if wire_name(gt_name) != fn["name"]:
        return "wrong_name", f"got {fn['name']!r}, want {wire_name(gt_name)!r}"
    try:
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
        if not isinstance(args, dict):
            raise ValueError("arguments are not an object")
    except Exception as e:  # noqa: BLE001
        return "bad_json", f"{e}: {str(fn['arguments'])[:120]}"
    why = score_args(args, gt_args)
    if why:
        return "wrong_args", why
    if len(calls) > 1:
        return "multiple_calls", f"{len(calls)} calls returned"
    return "ok", ""


async def run(args, base, model, items):
    async with common.make_client(args.concurrency) as client:
        async def one(i):
            it = items[i]
            payload = common.chat_payload(model, it["messages"], max_tokens=args.max_tokens,
                                          thinking=False, temperature=0.0,
                                          extra={"tools": build_tools(it["functions"]), "tool_choice": "auto"})
            try:
                res = await common.post_json(client, f"{base}/v1/chat/completions", payload)
            except Exception as e:  # noqa: BLE001
                return dict(error=str(e)[:300])
            return dict(message=res["choices"][0]["message"], finish=res["choices"][0].get("finish_reason"))
        return await common.run_bounded(one, len(items), args.concurrency, progress_every=50)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_server_args(ap)
    common.add_output_args(ap)
    ap.add_argument("--data", default=str(common.DATA_DIR / "bfcl_subset.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="first N items per category")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--min-score", type=float, default=None, help="optional pass floor (0-1)")
    args = ap.parse_args()

    common.require_platform(args)
    base = common.base_url(args)
    model = common.resolve_model(args, base)
    items = common.read_jsonl(args.data)
    if args.limit:
        keep, seen = [], {}
        for it in items:
            seen[it["category"]] = seen.get(it["category"], 0) + 1
            if seen[it["category"]] <= args.limit:
                keep.append(it)
        items = keep
    common.log(f"tool calling: {len(items)} items, concurrency {args.concurrency}")
    outs = asyncio.run(run(args, base, model, items))

    tally: dict[str, dict] = {}
    details, errors = [], 0
    for it, o in zip(items, outs):
        cat = tally.setdefault(it["category"], dict(n=0, ok=0, name_ok=0, called=0))
        cat["n"] += 1
        if "error" in o:
            errors += 1
            details.append(dict(id=it["id"], outcome="request_error", detail=o["error"]))
            continue
        outcome, detail = score_item(it, o["message"])
        cat["ok"] += outcome == "ok"
        cat["called"] += outcome != "no_tool_call"
        cat["name_ok"] += outcome in ("ok", "wrong_args", "bad_json", "multiple_calls")
        details.append(dict(id=it["id"], category=it["category"], outcome=outcome, detail=detail))
    n = len(items)
    ok = sum(c["ok"] for c in tally.values())
    outcomes: dict[str, int] = {}
    for d in details:
        outcomes[d["outcome"]] = outcomes.get(d["outcome"], 0) + 1
    score = ok / n
    head = f"{100 * score:.1f}% ({ok}/{n}); " + ", ".join(
        f"{k}: {100 * v['ok'] / v['n']:.1f}%" for k, v in tally.items())
    passed = None if args.min_score is None else score >= args.min_score
    common.write_accuracy_result(
        args, "toolcall", headline=head, score=score, passed=passed,
        criterion=(f"score >= {args.min_score}" if args.min_score is not None else "no floor given: verdict not computed"),
        metrics=dict(n=n, correct=ok, by_category=tally, outcomes=outcomes, request_errors=errors,
                     name_accuracy=sum(c["name_ok"] for c in tally.values()) / n,
                     tool_call_rate=sum(c["called"] for c in tally.values()) / n),
        params=dict(data=str(args.data), concurrency=args.concurrency, max_tokens=args.max_tokens,
                    enable_thinking=False, temperature=0, scorer="BFCL-style AST match (simplified)"),
        server=common.server_info(base), details=details,
        notes=[f"{errors} request(s) failed and are counted as wrong"] if errors else [],
    )
    print(head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
