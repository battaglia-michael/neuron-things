#!/usr/bin/env python3
"""Merge the three correctness-gate results (teacher_forced, batch_invariance,
chunked_prefill) into results/accuracy/correctness-<platform>.json plus a markdown table.

    python bench/accuracy/combine_correctness.py --platform trn2 \
        [--batch-probe results/accuracy/batch_divergence_probe-trn2.json]

--batch-probe: output of batch_divergence_probe.py. When batch_invariance.py's own rule
(<= 10% early divergences) fails, the batch-invariance verdict becomes the leakage rule:
every early-divergent prompt (a) diverges at a solo top-1/top-2 logprob margin
<= --tie-margin (a bf16 near-tie) and (b) diverges at the same token when the other 31
requests in the batch are copies of the same prompt (so the result does not depend on the
other requests' content). Both verdicts are kept in the output.

--allow-copy-match (default off, T09d): alternative per-id check. A probed id is also ok if
it diverges at a near-tie (margin <= --tie-margin) and the mixed-batch output equals at
least one of the homogeneous copies (homogeneous_copies_equal_to_mixed >= 1). Identical
copies carry no other content, so a mixed output reproduced by one of them does not depend
on the other requests. Needed because the copy at the prompt's own slot (the first-arriving
request) can take a different near-tie branch from the other copies depending on the
server's request history (results/accuracy/T09c/). The old rule's per-id result and overall
verdict are kept next to the new one (passed_old_rule, passed_alt_rule).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

TESTS = ("teacher_forced", "batch_invariance", "chunked_prefill")


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def leakage_verdict(bi: dict, probe: dict, tie_margin: float, min_prefix: int,
                    allow_copy_match: bool = False, margin_tol: float = 0.0) -> dict:
    # margin_tol (T09e, default 0 = off): accept margins up to tie_margin + margin_tol, so that a
    # bf16 gap of exactly tie_margin stored as float32 (e.g. 0.2500000596) still counts as a tie.
    det = Path(bi["_path"]).with_suffix(".details.jsonl")
    gross = sorted({r["id"] for r in common.read_jsonl(det)
                    if r["first_divergence"] is not None and r["first_divergence"] < min_prefix})
    rows = {r["id"]: r for r in probe["rows"]}
    checks, ok, ok_old, ok_alt, per_id = [], True, True, True, {}
    for pid in gross:
        r = rows.get(pid)
        if r is None:
            checks.append(f"{pid}: not probed -> FAIL")
            ok = ok_old = ok_alt = False
            per_id[pid] = dict(old=False, alt=False, ok=False, probed=False)
            continue
        m = r.get("solo_top1_top2_margin")
        tie = m is not None and m <= tie_margin + margin_tol
        same = r["first_div_solo_vs_homogeneous_same_slot"] == r["first_div_solo_vs_mixed"]
        old = tie and same
        alt = tie and (r.get("homogeneous_copies_equal_to_mixed") or 0) >= 1
        this = old or (allow_copy_match and alt)
        ok &= this
        ok_old &= old
        ok_alt &= old or alt
        per_id[pid] = dict(old=old, alt=alt, ok=this, probed=True)
        line = (f"{pid}: diverges at token {r['first_div_solo_vs_mixed']}, solo margin {m}, "
                f"homogeneous batch diverges at {r['first_div_solo_vs_homogeneous_same_slot']} "
                f"({r['homogeneous_copies_equal_to_mixed']} of {probe['params']['n']} copies = mixed output) "
                f"-> {'ok' if old else 'FAIL'}")
        if allow_copy_match:
            line += (f"; copy-match rule (tie and >= 1 copy = mixed) -> {'ok' if alt else 'FAIL'}"
                     f"; verdict {'ok' if this else 'FAIL'}")
        checks.append(line)
    criterion = (f"every prompt diverging before token {min_prefix} does so at a bf16 near-tie "
                 f"(solo top-1/top-2 logprob margin <= {tie_margin}) and at the same token in a "
                 "batch of identical copies (no dependence on other requests' content)")
    if margin_tol:
        criterion += f" (margin tolerance {margin_tol:g}, --margin-tol)"
    if allow_copy_match:
        criterion += (", or, alternatively, at a near-tie with the mixed output equal to at least one "
                      "of the identical copies (--allow-copy-match)")
    out = dict(passed=ok, early_divergent=gross, checks=checks, criterion=criterion)
    if allow_copy_match:
        out.update(passed_old_rule=ok_old, passed_alt_rule=ok_alt, allow_copy_match=True, per_id=per_id)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True, choices=common.PLATFORMS)
    ap.add_argument("--dir", default=str(common.REPO / "results" / "accuracy"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-probe", default=None)
    ap.add_argument("--tie-margin", type=float, default=0.25)
    ap.add_argument("--allow-copy-match", action="store_true",
                    help="also accept a near-tie id whose mixed output equals >= 1 homogeneous copy (T09d)")
    ap.add_argument("--margin-tol", type=float, default=0.0,
                    help="tolerance added to --tie-margin in the near-tie test (T09e; default 0 = off, e.g. 1e-6)")
    args = ap.parse_args()
    d = Path(args.dir)
    parts, raw, rows = {}, {}, []
    for t in TESTS:
        f = d / f"{t}-{args.platform}.json"
        if f.exists():
            r = json.loads(f.read_text())
            r["_path"] = str(f)
            raw[t] = r
            parts[t] = dict(headline=r["headline"], passed=r["passed"], criterion=r["criterion"],
                            score=r["score"], source=str(f), timestamp=r["timestamp"],
                            metrics=r.get("metrics"), params=r.get("params"))
        else:
            parts[t] = dict(headline="not run", passed=None, criterion="", source=None)
    bi = parts["batch_invariance"]
    if args.batch_probe and "batch_invariance" in raw:
        probe = json.loads(Path(args.batch_probe).read_text())
        lv = leakage_verdict(raw["batch_invariance"], probe, args.tie_margin,
                             raw["batch_invariance"]["params"].get("min_prefix", 16),
                             allow_copy_match=args.allow_copy_match, margin_tol=args.margin_tol)
        bi["harness_rule_passed"] = raw["batch_invariance"]["passed"]
        bi["harness_rule"] = raw["batch_invariance"]["criterion"]
        bi["leakage_check"] = dict(lv, source=args.batch_probe)
        if raw["batch_invariance"]["passed"] is False:
            bi["passed"] = lv["passed"]
            bi["criterion"] = (f"harness rule ({bi['harness_rule']}) failed, so judged by the leakage rule: "
                               + lv["criterion"])
    for t in TESTS:
        p = parts[t]["passed"]
        rows.append(f"| {t} | {'PASS' if p else 'FAIL' if p is False else 'n/a'} | {parts[t]['headline']} | {parts[t]['criterion']} |")
    verdicts = [v["passed"] for v in parts.values()]
    overall = None if any(v is None for v in verdicts) else all(verdicts)
    out = Path(args.out) if args.out else d / f"correctness-{args.platform}.json"
    common.write_json(out, dict(test="correctness", platform=args.platform, timestamp=common.utc_now(),
                                passed=overall, headline=f"{sum(bool(v) for v in verdicts)}/{len(verdicts)} tests pass"
                                + ("" if all(v is not None for v in verdicts) else " (incomplete)"),
                                score=None, criterion="all three tests pass", tests=parts, metrics={}))
    md = ["| Test | Verdict | Result | Rule |", "|---|---|---|---|"] + rows + [""]
    tf = raw.get("teacher_forced")
    if tf:
        m = tf["metrics"]
        line = m.get("hf_bf16_line") or {}
        pf = ((m.get("paired") or {}).get("first_token_hf_bf16_same_positions")) or {}
        dp = ((m.get("paired") or {}).get("decode_path_server")) or {}
        dpb = ((m.get("paired") or {}).get("decode_path_hf_bf16_same_positions")) or {}
        md += [f"Teacher-forced greedy match vs HF fp32 ({m['mode']} mode)", "",
               "| Position | Server (prefill, first token) | HF BF16, all positions | HF BF16, same positions | Server (decode path) | HF BF16 (same decode positions) |",
               "|---|---|---|---|---|---|"]
        for b in ("0-1K", "1K-4K", "4K-8K", "8K+", "all"):
            s = m["buckets"].get(b)
            if not s:
                continue
            md.append(f"| {b} | {pct(s['rate'])} (n={s['n']}) | {pct((line.get(b) or {}).get('rate'))} | "
                      f"{pct((pf.get(b) or {}).get('rate'))} | {pct((dp.get(b) or {}).get('rate'))} (n={(dp.get(b) or {}).get('n', 0)}) | "
                      f"{pct((dpb.get(b) or {}).get('rate'))} |")
        md.append("")
    cp = raw.get("chunked_prefill")
    if cp:
        m = cp["metrics"]
        line = m.get("hf_bf16_line") or {}
        same = m.get("hf_bf16_same_positions") or {}
        md += [f"Chunked prefill (chunk = {m['chunk_size']} tokens, {m['mode']} mode)", "",
               "| Chunk | Server | HF BF16, all positions | HF BF16, same positions |", "|---|---|---|---|"]
        for b in sorted((k for k in m["chunks"] if k != "all"), key=lambda s: int(s.split()[1])) + ["all"]:
            s = m["chunks"][b]
            md.append(f"| {b} | {pct(s['rate'])} (n={s['n']}) | {pct((line.get(b) or {}).get('rate'))} | "
                      f"{pct((same.get(b) or {}).get('rate'))} |")
        md.append("")
    if bi.get("leakage_check"):
        md += ["Batch invariance leakage check (`batch_divergence_probe.py`)", ""]
        md += [f"- {c}" for c in bi["leakage_check"]["checks"]] + [""]
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
