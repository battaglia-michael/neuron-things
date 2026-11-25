"""Shared helpers for the benchmark and accuracy scripts.

Everything here talks to an OpenAI-compatible server (vLLM on Trainium2 or on
an H100) over HTTP. The scripts only ever connect to a loopback address.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import ipaddress
import json
import os
import platform as _platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"

PLATFORMS = ("trn2", "h100")

# Sampling used for throughput runs (model card) and for accuracy runs (greedy).
BENCH_SAMPLING = {"temperature": 1.0, "top_p": 0.95}


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")
    tmp.replace(p)
    return p


def write_jsonl(path: str | Path, rows) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def git_rev() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def log(msg: str) -> None:
    print(f"[{_dt.datetime.now(_dt.timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# server arguments / connection
# --------------------------------------------------------------------------
def check_loopback(host: str) -> None:
    """Refuse to talk to anything that is not on this machine."""
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    sys.exit(f"refusing non-loopback host {host!r}: servers must bind 127.0.0.1 only")


def add_server_args(p: argparse.ArgumentParser, *, need_tokenizer: bool = False) -> None:
    g = p.add_argument_group("server")
    g.add_argument("--host", default="127.0.0.1", help="loopback address only (default 127.0.0.1)")
    g.add_argument("--port", type=int, default=int(os.environ.get("BENCH_PORT", 8000)))
    g.add_argument("--model", default=None,
                   help="served model name (default: first entry of /v1/models)")
    g.add_argument("--tokenizer", default=os.environ.get("BENCH_TOKENIZER"),
                   help="tokenizer path or HF id" + (" (required)" if need_tokenizer else
                                                      " (default: served model name)"))
    g.add_argument("--platform", choices=PLATFORMS, default=os.environ.get("BENCH_PLATFORM"),
                   help="trn2 or h100 (or set BENCH_PLATFORM)")
    g.add_argument("--wait", type=int, default=0, metavar="SEC",
                   help="wait up to SEC seconds for the server to answer /v1/models")


def base_url(args) -> str:
    check_loopback(args.host)
    return f"http://{args.host}:{args.port}"


def wait_ready(base: str, timeout_s: int) -> None:
    deadline = time.time() + max(timeout_s, 0)
    last = None
    while True:
        try:
            r = httpx.get(f"{base}/v1/models", timeout=5)
            if r.status_code == 200 and r.json().get("data"):
                return
            last = f"HTTP {r.status_code}"
        except Exception as e:  # server not up yet
            last = repr(e)
        if time.time() >= deadline:
            sys.exit(f"server at {base} not ready: {last}")
        time.sleep(3)


def resolve_model(args, base: str) -> str:
    wait_ready(base, args.wait)
    if args.model:
        return args.model
    r = httpx.get(f"{base}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def server_info(base: str) -> dict:
    info: dict[str, Any] = {"base_url": base}
    try:
        r = httpx.get(f"{base}/version", timeout=5)
        if r.status_code == 200:
            info["version"] = r.json()
    except Exception:
        pass
    try:
        r = httpx.get(f"{base}/v1/models", timeout=5)
        if r.status_code == 200:
            m = r.json()["data"][0]
            info["model"] = m.get("id")
            info["max_model_len"] = m.get("max_model_len")
    except Exception:
        pass
    return info


def host_info() -> dict:
    return {
        "machine": _platform.machine(),
        "python": sys.version.split()[0],
        "git_rev": git_rev(),
    }


def require_platform(args) -> str:
    if not args.platform:
        sys.exit("--platform trn2|h100 is required (or set BENCH_PLATFORM)")
    return args.platform


# --------------------------------------------------------------------------
# async HTTP
# --------------------------------------------------------------------------
def make_client(concurrency: int = 64, timeout_s: float = 3600) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=max(concurrency, 8) + 4,
                          max_keepalive_connections=max(concurrency, 8) + 4)
    return httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=30), limits=limits)


async def post_json(client: httpx.AsyncClient, url: str, payload: dict, retries: int = 2) -> dict:
    """POST and return the parsed JSON. Raises RuntimeError with the server message on failure."""
    last = ""
    for attempt in range(retries + 1):
        try:
            r = await client.post(url, json=payload)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code < 500:  # client error, retry will not help
                break
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            last = f"{type(e).__name__}: {e}"
        await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(last)


async def run_bounded(coros_factory, n: int, concurrency: int, progress_every: int = 0):
    """Run n jobs (coros_factory(i) -> coroutine) with at most `concurrency` in flight."""
    sem = asyncio.Semaphore(concurrency)
    results: list[Any] = [None] * n
    done = 0
    t0 = time.time()

    async def one(i):
        nonlocal done
        async with sem:
            results[i] = await coros_factory(i)
        done += 1
        if progress_every and (done % progress_every == 0 or done == n):
            log(f"  {done}/{n} done ({time.time() - t0:.0f}s)")

    await asyncio.gather(*(one(i) for i in range(n)))
    return results


def chat_payload(model: str, messages: list[dict], *, max_tokens: int, thinking: bool = False,
                 temperature: float = 0.0, extra: dict | None = None) -> dict:
    p = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }
    if temperature > 0:
        p["top_p"] = BENCH_SAMPLING["top_p"]
    if extra:
        p.update(extra)
    return p


def message_text(choice: dict) -> str:
    """Final answer text of a chat choice (content only, never the reasoning field)."""
    msg = choice.get("message", {})
    return msg.get("content") or ""


def message_reasoning(choice: dict) -> str:
    msg = choice.get("message", {})
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


# --------------------------------------------------------------------------
# accuracy result files (one schema for every accuracy script)
# --------------------------------------------------------------------------
def default_accuracy_path(test: str, platform: str) -> Path:
    return REPO / "results" / "accuracy" / f"{test}-{platform}.json"


def add_output_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("output")
    g.add_argument("--out", default=None,
                   help="result JSON path (default results/accuracy/<test>-<platform>.json)")
    g.add_argument("--variant", default=None,
                   help="free-text label of the server config, stored in the JSON")


def write_accuracy_result(
    args, test: str, *, headline: str, metrics: dict, score: float | None,
    passed: bool | None, criterion: str, params: dict, server: dict,
    details: list[dict] | None = None, notes: list[str] | None = None,
) -> Path:
    platform = require_platform(args)
    out = Path(args.out) if args.out else default_accuracy_path(test, platform)
    result = {
        "test": test,
        "platform": platform,
        "variant": getattr(args, "variant", None),
        "timestamp": utc_now(),
        "headline": headline,
        "score": score,
        "passed": passed,
        "criterion": criterion,
        "metrics": metrics,
        "params": params,
        "server": server,
        "host": host_info(),
        "notes": notes or [],
    }
    if details is not None:
        det = out.with_suffix(".details.jsonl")
        write_jsonl(det, details)
        result["details_file"] = str(det)
    write_json(out, result)
    log(f"wrote {out}")
    return out
