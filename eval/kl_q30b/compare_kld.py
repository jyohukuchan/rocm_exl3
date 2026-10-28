#!/usr/bin/env python3
"""Macro case-mean KLD of candidate .f32 dumps vs the llama.cpp BF16 reference.

Metric: per-position KL(ref || cand) at temperature 1 (log-softmax both
sides, float64 accumulation). Case score = mean over positions; macro KLD =
mean over cases. Top-1 agreement likewise (macro + position-weighted).
Only the shared leading vocab columns are compared.

Usage: compare_kld.py --refdir REF --canddir CAND [--out JSON]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_f32(p, rows, cols):
    a = np.fromfile(p, dtype="<f4")
    assert a.size == rows * cols, f"{p}: size {a.size} != {rows}x{cols}"
    return a.reshape(rows, cols).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refdir", required=True)
    ap.add_argument("--canddir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", default=None,
                    help="comma-separated case ids to compare (default: all)")
    args = ap.parse_args()
    only = set(args.cases.split(",")) if args.cases else None

    refdir, canddir = Path(args.refdir), Path(args.canddir)
    refman = json.loads((refdir / "manifest.json").read_text())
    candman = json.loads((canddir / "manifest.json").read_text())
    candcases = {c["id"]: c for c in candman["cases"]}

    rows = []
    tot_pos = tot_agree = 0
    for rc in refman["cases"]:
        cid = rc["id"]
        if only is not None and cid not in only:
            continue
        assert cid in candcases, f"candidate missing case {cid}"
        cc = candcases[cid]
        assert cc["input_token_ids_sha256"] == rc["input_token_ids_sha256"], \
            f"{cid}: input ids differ"
        rr, rv = rc["shape"]
        cr, cv = cc["shape"]
        assert rr == cr, f"{cid}: row count {cr} != ref {rr}"
        v = min(rv, cv)
        r = load_f32(refdir / rc["logits_file"], rr, rv)[:, :v]
        c = load_f32(canddir / cc["logits_file"], cr, cv)[:, :v]
        assert int((~np.isfinite(r)).sum()) == 0, f"{cid}: ref nonfinite"
        cnf = int((~np.isfinite(c)).sum())
        assert cnf == 0, f"{cid}: candidate nonfinite={cnf}"
        r -= r.max(axis=1, keepdims=True)
        c -= c.max(axis=1, keepdims=True)
        lr = r - np.log(np.exp(r).sum(axis=1, keepdims=True))
        lc = c - np.log(np.exp(c).sum(axis=1, keepdims=True))
        pr = np.exp(lr)
        kld = (pr * (lr - lc)).sum(axis=1)
        agree = (r.argmax(axis=1) == c.argmax(axis=1))
        rows.append({"id": cid, "positions": rr, "kld": float(kld.mean()),
                     "top1": float(agree.mean())})
        tot_pos += rr
        tot_agree += int(agree.sum())

    macro_kld = float(np.mean([x["kld"] for x in rows]))
    macro_top1 = float(np.mean([x["top1"] for x in rows]))
    wt_top1 = tot_agree / tot_pos
    print(f"cases={len(rows)} positions={tot_pos}")
    print(f"macro KLD={macro_kld:.4f}  macro top1={macro_top1:.4f}  weighted top1={wt_top1:.4f}")
    for x in rows:
        print(f"  {x['id']:18s} n={x['positions']:5d} kld={x['kld']:.4f} top1={x['top1']:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"macro_kld": macro_kld, "macro_top1": macro_top1,
             "weighted_top1": wt_top1, "cases": rows}, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
