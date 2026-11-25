#!/usr/bin/env python3
"""Build bench/data/bfcl_subset.jsonl from gorilla-llm/Berkeley-Function-Calling-Leaderboard.

Takes BFCL v3 'simple' and 'multiple' (single turn, exactly one expected call), joins each
question with its possible_answer ground truth, and keeps a seeded sample of N per category
(default 100 + 100 = 200 items). The result is checked into the repo so benchmark machines
need no download. Needs huggingface_hub (present in the vLLM venvs).
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

REPO_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def main():
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(common.DATA_DIR / "bfcl_subset.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for cat in ("simple", "multiple"):
        q = common.read_jsonl(hf_hub_download(REPO_ID, f"BFCL_v3_{cat}.json", repo_type="dataset"))
        a = {r["id"]: r["ground_truth"] for r in common.read_jsonl(
            hf_hub_download(REPO_ID, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset"))}
        q = [r for r in q if r["id"] in a and len(a[r["id"]]) == 1 and len(r["question"]) == 1]
        for r in rng.sample(q, min(args.per_category, len(q))):
            rows.append(dict(id=r["id"], category=cat, messages=r["question"][0],
                             functions=r["function"], ground_truth=a[r["id"]][0]))
    common.write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} items to {args.out}")


if __name__ == "__main__":
    main()
