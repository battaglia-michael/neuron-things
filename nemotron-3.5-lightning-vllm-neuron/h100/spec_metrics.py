#!/usr/bin/env python3
"""Speculative-decoding acceptance from a vLLM server's Prometheus /metrics (stdlib only, 127.0.0.1 only).

  spec_metrics.py snapshot --port 8000 --out before.json
  spec_metrics.py delta before.json after.json --out row.json [--label agent_8k_c32]

vLLM 0.27.1 exposes cumulative counters (names end in _total on /metrics):
  vllm:spec_decode_num_drafts                     draft proposals (one per request per decoding step)
  vllm:spec_decode_num_draft_tokens               draft tokens proposed
  vllm:spec_decode_num_accepted_tokens            draft tokens accepted by the verifier
  vllm:spec_decode_num_accepted_tokens_per_pos    accepted, by draft position (label position="0".."k-1")
The delta file holds, for the interval between two snapshots:
  acceptance_rate      = accepted / draft_tokens
  mean_acceptance_len  = 1 + accepted / drafts   (tokens emitted per verification step, bonus token included)
  per_position_rate[i] = accepted_at_position_i / drafts
Exit status of `delta`: 0 if the interval had draft tokens, 3 if it had none (for example a skipped row).
"""
import argparse
import json
import re
import sys
import time
import urllib.request

LINE = re.compile(r'^(vllm:spec_decode_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$')


def scrape(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=20) as r:
        text = r.read().decode()
    out = {"time": time.time(), "totals": {}, "per_pos": {}}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        if name.endswith("_per_pos"):
            p = re.search(r'position="(\d+)"', labels)
            if p:
                out["per_pos"][p.group(1)] = out["per_pos"].get(p.group(1), 0.0) + val
        else:
            out["totals"][name] = out["totals"].get(name, 0.0) + val
    return out


def pick(totals: dict, suffix: str) -> float:
    return sum(v for k, v in totals.items() if k.endswith(suffix))


def delta(a: dict, b: dict) -> dict:
    d = {k: pick(b["totals"], k) - pick(a["totals"], k)
         for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")}
    pos = {k: b["per_pos"].get(k, 0.0) - a["per_pos"].get(k, 0.0) for k in sorted(b["per_pos"], key=int)}
    res = dict(drafts=d["num_drafts"], draft_tokens=d["num_draft_tokens"], accepted_tokens=d["num_accepted_tokens"],
               interval_s=b["time"] - a["time"], accepted_by_position=pos)
    if d["num_draft_tokens"] > 0:
        res["acceptance_rate"] = d["num_accepted_tokens"] / d["num_draft_tokens"]
    if d["num_drafts"] > 0:
        res["mean_acceptance_len"] = 1 + d["num_accepted_tokens"] / d["num_drafts"]
        res["per_position_rate"] = {k: v / d["num_drafts"] for k, v in pos.items()}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--out", required=True)
    d = sub.add_parser("delta")
    d.add_argument("before")
    d.add_argument("after")
    d.add_argument("--out", required=True)
    d.add_argument("--label", default="")
    a = ap.parse_args()
    if a.cmd == "snapshot":
        snap = scrape(a.port)
        if not snap["totals"] and not snap["per_pos"]:
            print("no vllm:spec_decode_* series on /metrics (speculative decoding off?)", file=sys.stderr)
            return 1
        json.dump(snap, open(a.out, "w"))
        return 0
    res = delta(json.load(open(a.before)), json.load(open(a.after)))
    res["label"] = a.label
    if res["draft_tokens"] <= 0:
        print(f"{a.label}: no draft tokens in the interval; not written", file=sys.stderr)
        return 3
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"{a.label}: acceptance {res['acceptance_rate']:.3f}, mean accepted length {res['mean_acceptance_len']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
