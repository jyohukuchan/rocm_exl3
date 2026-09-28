#!/usr/bin/env python3
"""Compare two saved teacher-forced top-1 artifacts (Phase 0-2 accuracy gate).

Pure stdlib: runs on the host, no torch, no GPU. This is where the accuracy
criterion lives: agreement of the teacher-forced argmax token at every manifest
position -- NOT exact generated-token sequences and NOT numeric allclose of
weights. A low rate is only evidence about kernels if both sides ran the same
frozen manifest; arbitrary weight differences (different checkpoints or
different quantizations) show up here as top-1 differences too, so the report
names both model roots, their config fingerprints and their execution modes
up front -- read the identity lines before reading the rate.

The comparison refuses (exit 1) rather than reporting a misleading number when:

  - either artifact is structurally invalid, flagged incomplete, carries
    error records, or has non-finite logit rows flagged at any position;
  - the two artifacts are tied to different token manifests (digest), or
    different per-case literal-input hashes / position lists;
  - vocab sizes differ, cases are missing on either side, or top-1 lists are
    shorter than their position lists or hold IDs outside the valid vocab;
  - an artifact covers only a subset of the manifest: collect_top1 keeps
    total_positions at the FULL manifest count, so identically truncated
    (e.g. both --limit-cases) artifacts are refused, never a vacuous 1.0;
  - (optional, --manifest) a provided manifest disagrees with either artifact
    (including manifest cases missing from the artifacts).

Exit codes: 0 = comparison valid and (if --min-agreement given) at or above
threshold; 1 = rejected (inputs not comparable / incomplete); 2 = agreement
below the given --min-agreement threshold.

Usage:
    /opt/venv/bin/python rocm_tools/rdna2/compare_top1.py \
        --reference /work/phase0/top1_ref_transformers_fp16.json \
        --candidate /work/phase0/top1_cand_exl3_bulk.json \
        --min-agreement 0.995

    # same EXL3 checkpoint, conservative vs optimized execution (env switches
    # differ at collection time; identical manifest means differences ARE
    # kernel-path differences):
    /opt/venv/bin/python rocm_tools/rdna2/compare_top1.py \
        --reference /work/phase0/top1_exl3_ref.json \
        --candidate /work/phase0/top1_exl3_opt.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rocm_tools.rdna2.common import (
    TOP1_FORMAT,
    read_json,
    validate_manifest,
    validate_top1_result,
)

# Keys that must agree for two artifacts to be comparable at all.
# (model identity is deliberately NOT rejected here -- comparing runs on
# different model roots is exactly what the reference-vs-candidate gate does.)
_HARD_KEYS = (
    "manifest_sha256",
    "valid_vocab_size",
    "position_semantics",
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog = "compare_top1.py",
        description = "Teacher-forced top-1 agreement between two collection artifacts.",
    )
    ap.add_argument("--reference", required = True, help = "top-1 result JSON (reference run)")
    ap.add_argument("--candidate", required = True, help = "top-1 result JSON (candidate run)")
    ap.add_argument("--manifest", default = None,
                    help = "optional: the token manifest both artifacts should derive from; "
                           "if given, cross-validates each artifact against it")
    ap.add_argument("--min-agreement", type = float, default = None,
                    help = "if set, exit 2 when overall agreement < this value (0..1)")
    ap.add_argument("--json-out", default = None, help = "optional: write machine-readable summary JSON")
    ap.add_argument("-v", "--verbose", action = "store_true",
                    help = "list first differing positions per case")
    return ap


def _identity(art: dict) -> str:
    ex = art.get("execution") or {}
    fp = art.get("model_fingerprint") or {}
    return (f"backend={art.get('backend')} dir={art.get('model_dir')} "
            f"config_sha={str(fp.get('config.json_sha256'))[:16]} "
            f"mode={ex.get('mode')} chunk={ex.get('chunk_size')} "
            f"exec={ex.get('attn_mode')} dtype={ex.get('dtype_label', ex.get('dtype_effective'))} "
            f"git={str(art.get('repo_git_commit'))[:12]} "
            f"envswitches={sorted((art.get('env') or {}).get('exl3_rocm_switches', {}).items())}")


def _reject(reasons: list[str]) -> int:
    print("REJECT: artifacts are not comparable")
    for r in reasons:
        print(f"  - {r}")
    return 1


def compare_files(ref_path: str | Path, cand_path: str | Path,
                  manifest_path: str | Path | None = None,
                  verbose: bool = False) -> tuple[int, dict]:
    """
    Returns (exit_code, summary). exit_code 0 = valid comparison; 1 = rejected.
    (Threshold escalation to 2 happens in main(), not here, so the summary is
    always populated for the JSON output.)
    """
    summary: dict = {
        "reference": str(Path(ref_path).resolve()),
        "candidate": str(Path(cand_path).resolve()),
        "rejected": True,
        "reject_reasons": [],
    }
    reasons: list[str] = []

    try:
        ref = read_json(ref_path)
    except Exception as e:
        reasons.append(f"reference unreadable: {e!r}")
        return 1, {**summary, "reject_reasons": reasons}
    try:
        cand = read_json(cand_path)
    except Exception as e:
        reasons.append(f"candidate unreadable: {e!r}")
        return 1, {**summary, "reject_reasons": reasons}

    for name, art in (("reference", ref), ("candidate", cand)):
        if art.get("format") != TOP1_FORMAT:
            reasons.append(f"{name}: format {art.get('format')!r} != {TOP1_FORMAT!r}")
        errs = validate_top1_result(art)
        for e in errs:
            reasons.append(f"{name}: {e}")

    # Structural cross-consistency: same frozen inputs, positions and vocab.
    for key in _HARD_KEYS:
        a, b = ref.get(key), cand.get(key)
        if a != b:
            reasons.append(f"manifest linkage differs: {key}: "
                           f"reference={str(a)[:24]!r} candidate={str(b)[:24]!r}")

    if manifest_path:
        try:
            m = read_json(manifest_path)
        except Exception as e:
            reasons.append(f"manifest unreadable: {e!r}")
            m = None
        if m is not None:
            merrs = validate_manifest(m)
            for e in merrs:
                reasons.append(f"manifest: {e}")
            md = m.get("manifest_sha256")
            if not merrs and md != ref.get("manifest_sha256"):
                reasons.append("manifest digest does not match the one embedded in the artifacts")
            if not merrs:
                for name, art in (("reference", ref), ("candidate", cand)):
                    for case in m.get("cases", []):
                        cid = case["case_id"]
                        got = art.get("cases", {}).get(cid)
                        if got is None:
                            reasons.append(f"{name}: case {cid} from the manifest is "
                                           f"missing from the artifact")
                            continue
                        if got.get("positions") != case.get("positions"):
                            reasons.append(f"{name}: case {cid} positions differ from manifest")
                        if got.get("ids_sha256") != case.get("ids_sha256"):
                            reasons.append(f"{name}: case {cid} literal input differs from manifest")

    # Reject with ALL accumulated reasons (self-validation, linkage, manifest
    # coverage) instead of stopping at the first, so one run diagnoses the pair.
    if reasons:
        summary["reject_reasons"] = reasons
        return 1, summary

    ref_cases = ref.get("cases", {})
    cand_cases = cand.get("cases", {})
    only_ref = sorted(set(ref_cases) - set(cand_cases))
    only_cand = sorted(set(cand_cases) - set(ref_cases))
    if only_ref:
        reasons.append(f"cases missing in candidate: {only_ref}")
    if only_cand:
        reasons.append(f"cases missing in reference: {only_cand}")
    if reasons:
        summary["reject_reasons"] = reasons
        return 1, summary

    vocab = ref["valid_vocab_size"]
    per_case = []
    total_n = 0
    total_hits = 0
    for cid in sorted(ref_cases):
        rc, cc = ref_cases[cid], cand_cases[cid]
        if rc.get("ids_sha256") != cc.get("ids_sha256"):
            reasons.append(f"case {cid}: literal input hashes differ "
                           f"({str(rc.get('ids_sha256'))[:12]} vs {str(cc.get('ids_sha256'))[:12]})")
            continue
        if rc.get("positions") != cc.get("positions"):
            reasons.append(f"case {cid}: position lists differ")
            continue
        n = len(rc.get("positions") or [])
        if n == 0:
            reasons.append(f"case {cid}: empty position list")
            continue
        if len(cc.get("top1") or []) != n or len(rc.get("top1") or []) != n:
            reasons.append(f"case {cid}: top-1 list length != position count")
            continue
        if any(t is None or not (0 <= t < vocab) for t in rc["top1"] + cc["top1"]):
            reasons.append(f"case {cid}: top-1 IDs outside [0, vocab)")
            continue
        diffs = [p for p, a, b in zip(rc["positions"], rc["top1"], cc["top1"]) if a != b]
        hits = n - len(diffs)
        total_n += n
        total_hits += hits
        per_case.append({
            "case_id": cid,
            "n_positions": n,
            "agreement": hits,
            "rate": (hits / n) if n else 0.0,
            "first_diff_positions": diffs[:12] if verbose else diffs[:5],
            "n_diffs": len(diffs),
        })

    if reasons:
        summary["reject_reasons"] = reasons
        return 1, summary
    if total_n == 0:
        summary["reject_reasons"] = ["no positions were comparable"]
        return 1, summary

    rate = total_hits / total_n
    summary.update({
        "rejected": False,
        "reject_reasons": [],
        "total_positions": total_n,
        "top1_agreements": total_hits,
        "top1_agreement_rate": rate,
        "per_case": per_case,
        "reference_identity": _identity(ref),
        "candidate_identity": _identity(cand),
    })
    return 0, summary


def print_report(summary: dict) -> None:
    print(f" -- reference : {summary['reference']}")
    print(f"    identity   : {summary['reference_identity']}")
    print(f" -- candidate : {summary['candidate']}")
    print(f"    identity   : {summary['candidate_identity']}")
    print("    (Different checkpoints include quantization/weight differences. "
          "For a fixed checkpoint, compare the recorded execution settings; "
          "directory/config fingerprints are not full weight-content hashes.)")
    print(f"\n  {'case':22} {'N':>6} {'agree':>6} {'rate':>8}")
    for c in summary["per_case"]:
        print(f"  {c['case_id']:22} {c['n_positions']:6} {c['agreement']:6} {c['rate']:8.4f}"
              + (f"  diffs={c['n_diffs']} at {c['first_diff_positions']}" if c["n_diffs"] else ""))
    print(f"\n  TOP1 AGREEMENT {summary['top1_agreements']}/{summary['total_positions']} "
          f"= {summary['top1_agreement_rate']:.6f}")


def main(argv = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code, summary = compare_files(args.reference, args.candidate, args.manifest, args.verbose)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f" !! comparison failed: {e}", file = sys.stderr)
        return 1

    if code != 0:
        print("REJECT: artifacts are not comparable")
        for r in summary.get("reject_reasons", []):
            print(f"  - {r}")
    else:
        print_report(summary)
        if args.min_agreement is not None:
            rate = summary["top1_agreement_rate"]
            if rate < args.min_agreement:
                print(f"\n  FAIL: agreement {rate:.6f} < --min-agreement {args.min_agreement:.6f}")
                code = 2
            else:
                print(f"\n  ok: agreement {rate:.6f} >= --min-agreement {args.min_agreement:.6f}")

    if args.json_out:
        from rocm_tools.rdna2.common import write_json
        try:
            write_json(args.json_out, summary)
            print(f" -- summary written: {args.json_out}")
        except Exception as e:
            print(f" !! could not write summary: {e}", file = sys.stderr)
            code = code or 1
    return code


if __name__ == "__main__":
    sys.exit(main())
