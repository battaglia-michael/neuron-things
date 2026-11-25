"""Reference loading and server-side next-token probing shared by
teacher_forced.py and chunked_prefill.py.

Reference file format (JSON Lines, one sequence per line):

    {"id": "seq-007",
     "token_ids":   [int, ...],   # full input sequence x_0 .. x_{n-1}, already chat-templated
     "argmax_ids":  [int, ...],   # len n. argmax_ids[i] = HF fp32 argmax of the logits at
                                  # position i, i.e. the label for token x_{i+1}
     "bf16_argmax_ids": [int, ...],   # optional: same, from HF run in BF16 (the reference line)
     "top5_ids": [[int]*5, ...], "top5_logprobs": [[float]*5, ...]}   # optional

`load_references` also accepts the aliases listed in ALIASES. A directory is
read as the union of its *.jsonl files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

BUCKETS = [(0, 1024, "0-1K"), (1024, 4096, "1K-4K"), (4096, 8192, "4K-8K"), (8192, 1 << 30, "8K+")]

ALIASES = {
    "token_ids": ("token_ids", "input_ids", "tokens", "ids"),
    "argmax_ids": ("argmax_ids", "fp32_argmax_ids", "labels", "argmax", "pred_ids"),
    "bf16_argmax_ids": ("bf16_argmax_ids", "bf16_labels", "bf16_argmax"),
}


def _get(row: dict, name: str):
    for k in ALIASES[name]:
        if k in row and row[k] is not None:
            return row[k]
    return None


def load_references(path: str | Path, max_seqs: int | None = None) -> list[dict]:
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        sys.exit(f"no *.jsonl reference files in {p}")
    refs = []
    for f in files:
        for i, row in enumerate(common.read_jsonl(f)):
            toks, lab = _get(row, "token_ids"), _get(row, "argmax_ids")
            if toks is None or lab is None:
                continue
            if len(toks) != len(lab):
                sys.exit(f"{f}:{i + 1}: token_ids ({len(toks)}) and argmax_ids ({len(lab)}) differ in length")
            refs.append(dict(id=str(row.get("id", f"{f.stem}-{i}")), token_ids=[int(t) for t in toks],
                             argmax_ids=[int(t) for t in lab],
                             bf16_argmax_ids=_get(row, "bf16_argmax_ids")))
    if not refs:
        sys.exit(f"{p}: no rows with token ids and argmax labels (see bench/accuracy/_tf.py for the format)")
    if max_seqs:
        refs = refs[:max_seqs]
    return refs


def bucket_of(pos: int, buckets=BUCKETS) -> str:
    for lo, hi, name in buckets:
        if lo <= pos < hi:
            return name
    return buckets[-1][2]


def tally(pairs, bucket_fn=bucket_of) -> dict:
    """pairs: iterable of (position, matched_bool). Returns {bucket: {n, matched, rate}} + total."""
    out: dict[str, dict] = {}
    n_all = m_all = 0
    for pos, ok in pairs:
        b = out.setdefault(bucket_fn(pos), dict(n=0, matched=0))
        b["n"] += 1
        b["matched"] += int(ok)
        n_all += 1
        m_all += int(ok)
    for b in out.values():
        b["rate"] = b["matched"] / b["n"] if b["n"] else None
    out["all"] = dict(n=n_all, matched=m_all, rate=(m_all / n_all if n_all else None))
    return out


def reference_line(refs: list[dict], bucket_fn=bucket_of, use_len: dict | None = None) -> dict | None:
    """HF BF16-vs-fp32 match rate per bucket, if the references carry bf16 labels."""
    pairs = []
    for r in refs:
        bf = r.get("bf16_argmax_ids")
        if not bf:
            continue
        n = use_len.get(r["id"], len(r["token_ids"])) if use_len else len(r["token_ids"])
        for i in range(n - 1):
            pairs.append((i + 1, int(bf[i]) == r["argmax_ids"][i]))
    return tally(pairs, bucket_fn) if pairs else None


# --------------------------------------------------------------------------
# server probing
# --------------------------------------------------------------------------
def _rank1(entry: dict) -> int | None:
    """Argmax token id from one prompt_logprobs entry {token_id: {logprob, rank, ...}}."""
    best, best_lp = None, -1e30
    for k, v in entry.items():
        if v.get("rank") == 1:
            return int(k)
        lp = v.get("logprob", -1e30)
        if lp > best_lp:
            best, best_lp = int(k), lp
    return best


async def server_argmax_prefill(client, base: str, model: str, token_ids: list[int]) -> list[int | None]:
    """One prefill pass; returns, for each position i>=1, the server's argmax for token i
    (index 0 is None). Uses prompt_logprobs=1 on /v1/completions."""
    payload = {"model": model, "prompt": token_ids, "max_tokens": 1, "temperature": 0.0,
               "prompt_logprobs": 1}
    res = await common.post_json(client, f"{base}/v1/completions", payload)
    pl = res["choices"][0].get("prompt_logprobs")
    if pl is None:
        raise RuntimeError("server returned no prompt_logprobs (feature unsupported?)")
    if len(pl) != len(token_ids):
        raise RuntimeError(f"prompt_logprobs has {len(pl)} entries for {len(token_ids)} prompt tokens")
    return [None if e is None else _rank1(e) for e in pl]


async def server_greedy_tokens(client, base: str, model: str, prompt_ids: list[int], n: int) -> list[int]:
    """Greedy continuation of `n` tokens (EOS ignored) as token ids."""
    payload = {"model": model, "prompt": prompt_ids, "max_tokens": n, "temperature": 0.0,
               "ignore_eos": True, "return_token_ids": True}
    res = await common.post_json(client, f"{base}/v1/completions", payload)
    ch = res["choices"][0]
    if ch.get("token_ids"):
        return [int(t) for t in ch["token_ids"]]
    # fallback: ask for per-token logprobs rendered as "token_id:N"
    payload.pop("return_token_ids")
    payload.update(logprobs=1, return_tokens_as_token_ids=True)
    res = await common.post_json(client, f"{base}/v1/completions", payload)
    toks = res["choices"][0].get("logprobs", {}).get("tokens") or []
    try:
        return [int(t.split(":")[1]) for t in toks]
    except (IndexError, ValueError):
        raise RuntimeError("server returned neither token_ids nor token_id:N logprob tokens")


def wilson_upper(k: int, n: int, z: float = 2.58) -> float:
    """Upper end of the Wilson score interval for k successes in n trials (z=2.58: 99%)."""
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return min(1.0, (c + w) / d)


def below_floor(stat: dict, floor: float) -> bool:
    """True when a bucket's match rate is significantly below `floor`: the rate is under it
    and even the upper 99% confidence bound does not reach it. Small samples therefore do
    not fail on a single flipped token."""
    return stat["rate"] < floor and wilson_upper(stat["matched"], stat["n"]) < floor


def fmt_rate(d: dict | None) -> str:
    if not d or d.get("rate") is None:
        return "n/a"
    return f"{100 * d['rate']:.2f}% ({d['matched']}/{d['n']})"


def json_safe(o):
    return json.loads(json.dumps(o, default=str))
