#!/usr/bin/env python3
"""Prefix-cache correctness gate: a cache hit must give the same output as the uncached run.

For each long reference conversation (results/references/fp32.jsonl, multi-turn GSM8K chats with
HF fp32 next-token labels), on a server started with --enable-prefix-caching:

  1. ``miss``: prompt P = tokens[:L - cut], greedy, sent for the first time (nothing cached).
  2. ``hit``:  the same prompt again. Its whole-block prefix is served from the cache (KV and,
     in Mamba align mode, the recurrent state at the block boundary).
  3. ``next``: the longer prompt tokens[:L] (the next turn), which hits the same cached blocks.

Requests are sent one at a time, so batch composition cannot differ between miss and hit.
The cache hit is verified per request from the server's prefix-cache counters (/metrics).

Pass rule: every ``hit`` output is token-for-token identical to its ``miss`` output, every
``hit`` really hit the cache (cached tokens > 0), and no request failed. Also reported, not
gated: the first generated token of miss / hit / next vs the HF fp32 argmax label, and, with
--ref-port, the miss output of a prefix-caching-off server.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests


def metrics(base: str) -> dict:
    """Prefix-cache counters (vLLM v1 names; summed over label sets)."""
    out = {}
    text = requests.get(f"{base}/metrics", timeout=30).text
    for name in ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits"):
        tot = 0.0
        for m in re.finditer(rf"^{re.escape(name)}(?:_total)?(?:{{[^}}]*}})?\s+([0-9.eE+-]+)$", text, re.M):
            tot += float(m.group(1))
        out[name.split(":")[1]] = tot
    return out


def complete(base: str, model: str, prompt: list[int], max_tokens: int) -> dict:
    t0 = time.time()
    r = requests.post(
        f"{base}/v1/completions",
        json=dict(model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.0,
                  ignore_eos=True, return_token_ids=True, logprobs=1),
        timeout=1200,
    )
    r.raise_for_status()
    ch = r.json()["choices"][0]
    ids = ch.get("token_ids")
    if ids is None:  # older servers: no token ids in the response
        ids = ch.get("logprobs", {}).get("tokens")
    return dict(text=ch["text"], token_ids=ids, latency_s=time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--ref-port", type=int, default=None, help="optional prefix-caching-off server")
    ap.add_argument("--model", default="nemotron")
    ap.add_argument("--refs", default="results/references/fp32.jsonl")
    ap.add_argument("--ids", default="seq-23-5120,seq-25-5632,seq-27-6144,seq-28-6656,seq-29-7168,seq-30-7680,seq-31-8192")
    ap.add_argument("--cut", type=int, default=700, help="P = tokens[:L - cut]")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    refs = {}
    want = args.ids.split(",")
    for line in open(args.refs):
        d = json.loads(line)
        if d["id"] in want:
            refs[d["id"]] = d
    rows, failed = [], 0
    for sid in want:
        d = refs[sid]
        toks, labels = d["token_ids"], d["argmax_ids"]
        L = len(toks)
        p_len = L - args.cut
        row = dict(id=sid, prompt_len=p_len, next_len=L)
        for kind, plen in (("miss", p_len), ("hit", p_len), ("next", L)):
            m0 = metrics(base)
            try:
                res = complete(base, args.model, toks[:plen], args.max_tokens)
            except Exception as e:  # noqa: BLE001
                failed += 1
                row[kind] = dict(error=repr(e))
                continue
            m1 = metrics(base)
            res["cached_tokens"] = m1["prefix_cache_hits"] - m0["prefix_cache_hits"]
            res["queried_tokens"] = m1["prefix_cache_queries"] - m0["prefix_cache_queries"]
            res["first_token_matches_fp32_label"] = bool(res["token_ids"] and res["token_ids"][0] == labels[plen - 1])
            row[kind] = res
        if args.ref_port:
            try:
                ref = complete(f"http://127.0.0.1:{args.ref_port}", args.model, toks[:p_len], args.max_tokens)
                row["ref_nocache"] = ref
            except Exception as e:  # noqa: BLE001
                row["ref_nocache"] = dict(error=repr(e))
        miss, hit = row.get("miss", {}), row.get("hit", {})
        a, b = miss.get("token_ids"), hit.get("token_ids")
        row["hit_equals_miss"] = bool(a is not None and a == b)
        row["hit_common_prefix"] = next((i for i, (x, y) in enumerate(zip(a or [], b or [])) if x != y), len(a or []))
        row["hit_used_cache"] = hit.get("cached_tokens", 0) > 0
        if "ref_nocache" in row and row["ref_nocache"].get("token_ids") is not None and a is not None:
            r = row["ref_nocache"]["token_ids"]
            row["miss_common_prefix_vs_nocache_server"] = next(
                (i for i, (x, y) in enumerate(zip(a, r)) if x != y), min(len(a), len(r)))
        print(json.dumps({k: v for k, v in row.items() if k not in ("miss", "hit", "next", "ref_nocache")}
                         | {k: {kk: vv for kk, vv in row[k].items() if kk in ("cached_tokens", "queried_tokens", "first_token_matches_fp32_label", "latency_s", "error")}
                            for k in ("miss", "hit", "next") if k in row}), flush=True)
        rows.append(row)
    passed = failed == 0 and all(r["hit_equals_miss"] and r["hit_used_cache"] for r in rows)
    summary = dict(
        passed=passed,
        rule="every hit output == its miss output token for token, every hit used the cache, 0 failed requests",
        n=len(rows), failed_requests=failed,
        hit_equals_miss=sum(r["hit_equals_miss"] for r in rows),
        hit_used_cache=sum(r["hit_used_cache"] for r in rows),
        first_token_vs_fp32_label={k: sum(bool(r.get(k, {}).get("first_token_matches_fp32_label")) for r in rows)
                                   for k in ("miss", "hit", "next")},
        cached_tokens={r["id"]: {k: r.get(k, {}).get("cached_tokens") for k in ("miss", "hit", "next")} for r in rows},
        args=vars(args),
    )
    if args.ref_port:
        summary["miss_vs_nocache_server_identical"] = sum(
            r.get("miss_common_prefix_vs_nocache_server") == args.max_tokens for r in rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=summary, rows=rows), open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
