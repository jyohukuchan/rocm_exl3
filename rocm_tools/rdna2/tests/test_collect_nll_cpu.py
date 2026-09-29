#!/usr/bin/env python3
"""CPU-only tests for collect_nll.py -- full-corpus NLL labeling logic. No GPU, no
model, no exllamav3: dense manifest derivation, flat cursor alignment and the
fail-closed gates are driven against hand-built fixtures (and small host-torch
rows for analytically known NLLs).

Run from the repo root:
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2 import collect_nll, collect_top1
from rocm_tools.rdna2.common import (
    MANIFEST_FORMAT, TOP1_FORMAT, case_ids_sha256, manifest_digest, select_positions,
    sha256_text, validate_manifest)

VOCAB = 4  # tiny fixture vocab; every id stays inside [0, VOCAB)
CASES = [("alpha", [1, 2, 3, 0]), ("beta", [0, 3, 1])]  # labels: 3 + 2 = 5


def make_src(cases):
    recs = []
    for cid, ids in cases:
        recs.append({
            "case_id": cid, "language": "en", "kind": "fixture", "provenance": "test",
            "text_sha256": sha256_text(cid), "add_bos": False,
            "len_ids": len(ids), "ids": list(ids), "ids_sha256": case_ids_sha256(ids),
            "positions": select_positions(len(ids), min(3, len(ids))),  # sampled, as the real one
        })
    m = {"format": MANIFEST_FORMAT, "valid_vocab_size": VOCAB, "config_vocab_size": VOCAB,
         "total_positions": sum(len(r["positions"]) for r in recs), "cases": recs}
    m["manifest_sha256"] = manifest_digest(m)
    return m


def build(cases=CASES):
    return collect_nll.build_dense_manifest(make_src(cases), "fixtures/src.json")


def make_art(dense, plan_len, err_case=None):
    """Top-1 sidecar as the collector would write it for a fully dense run."""
    cases = {}
    for c in dense["cases"]:
        ok = c["case_id"] != err_case
        cases[c["case_id"]] = {
            "status": "ok" if ok else "error", "case_id": c["case_id"],
            "ids_sha256": c["ids_sha256"], "len_ids": c["len_ids"],
            "positions": list(c["positions"]), "n_positions": len(c["positions"]),
            "top1": [0 if ok else None] * len(c["positions"]), "nonfinite_positions": []}
    return {"format": TOP1_FORMAT, "manifest_sha256": dense["manifest_sha256"],
            "total_positions": plan_len, "collected_positions": plan_len,
            "complete": err_case is None, "errors": [], "cases": cases}


def drive(rows):
    """Feed rows through the wrapped original _harvest_row; returns (cursor, returns)."""
    dense, plan, errs = build()
    assert not errs
    cur = collect_nll.NllCursor(plan)
    wrapped = cur.wrap(collect_top1._harvest_row)
    import torch
    return cur, [wrapped(torch, torch.tensor(r, dtype=torch.float32), VOCAB) for r in rows]


class DensePlanTests(unittest.TestCase):
    def test_next_label_shift_and_final_token_excluded(self):
        dense, plan, errs = build()
        self.assertEqual(errs, [])
        self.assertEqual(plan, [("alpha", 0, 2), ("alpha", 1, 3), ("alpha", 2, 0),
                                 ("beta", 0, 3), ("beta", 1, 1)])  # target = ids[p+1]
        for cid, p, t in plan:
            self.assertLess(p, len(dict(CASES)[cid]) - 1)  # final unlabeled token never planned
        self.assertEqual(len(plan), sum(len(ids) - 1 for _, ids in CASES))

    def test_case_boundary_order(self):
        _, plan, _ = build()
        self.assertEqual([c for c, _, _ in plan], ["alpha"] * 3 + ["beta"] * 2)
        for cid in ("alpha", "beta"):
            self.assertEqual([p for c, p, _ in plan if c == cid],
                             list(range(len(dict(CASES)[cid]) - 1)))

    def test_single_token_and_len_mismatch_refused(self):
        _, plan, errs = build([("x", [2])])
        self.assertEqual(plan, [])
        self.assertTrue(any("only 1 token" in e for e in errs))
        src = make_src(CASES)
        src["cases"][0]["len_ids"] += 1
        _, _, errs = collect_nll.build_dense_manifest(src, "f")
        self.assertTrue(any("len_ids" in e for e in errs))


class DenseManifestTests(unittest.TestCase):
    def test_validates_with_matching_digest_and_provenance(self):
        src = make_src(CASES)
        dense, plan, errs = collect_nll.build_dense_manifest(src, "fixtures/src.json")
        self.assertEqual(errs, [])
        self.assertEqual(validate_manifest(dense), [])
        self.assertEqual(dense["manifest_sha256"], manifest_digest(dense))
        self.assertEqual(dense["total_positions"], len(plan))
        self.assertEqual(dense["source_manifest"]["sha256"], src["manifest_sha256"])
        for c in dense["cases"]:
            self.assertEqual(c["positions"], list(range(c["len_ids"] - 1)))
            self.assertEqual(c["ids"], dict(CASES)[c["case_id"]])  # literal IDs untouched

    def test_tampered_dense_manifest_rejected(self):
        dense, _, _ = build()
        dense["cases"][0]["ids"][0] ^= 1
        self.assertTrue(validate_manifest(dense))


class CursorTests(unittest.TestCase):
    def test_analytic_small_vocab_nll_via_wrapper(self):
        e = math.e
        rows = [[0.0] * VOCAB, [0.0, 0.0, 0.0, 2.0], [0.0] * VOCAB,
                [0.0, 0.0, 0.0, -1.0], [0.5] * VOCAB]
        want = [math.log(VOCAB), math.log(3 + e ** 2) - 2, math.log(VOCAB),
                math.log(3 + e ** -1) + 1, math.log(VOCAB)]  # logsumexp over {0,0,0,x} - x
        cur, tops = drive(rows)
        self.assertEqual(len(cur.losses), 5)
        for got, exp in zip(cur.losses, want):
            self.assertAlmostEqual(got["nll"], exp, places=9)
        self.assertEqual(tops[1], (3, True))  # original top-1 sidecar behavior preserved
        agg = collect_nll.aggregate(cur.losses)
        mean = sum(want) / 5
        self.assertAlmostEqual(agg["mean_nll"], mean, places=12)
        self.assertAlmostEqual(agg["perplexity"], math.exp(mean), places=9)
        self.assertEqual([c["case_id"] for c in agg["per_case"]], ["alpha", "beta"])
        self.assertEqual([c["n_labels"] for c in agg["per_case"]], [3, 2])
        dense, plan, _ = build()
        self.assertEqual(collect_nll.cross_check(cur, dense, make_art(dense, len(plan)), 0), [])

    def test_nonfinite_row_breaks_cursor_and_stops_labeling(self):
        cur, _ = drive([[0.0] * VOCAB, [float("nan")] * VOCAB, [0.0] * VOCAB])
        self.assertTrue(cur.broken)
        self.assertEqual(len(cur.losses), 1)  # rows after the anomaly are never labeled
        self.assertIn("non-finite logit row", cur.problems[0])
        dense, plan, _ = build()
        errs = collect_nll.cross_check(cur, dense, make_art(dense, len(plan)), 1)
        self.assertTrue(any("partial corpus" in x for x in errs))

    def test_extra_harvest_call_beyond_plan_rejected(self):
        cur, _ = drive([[0.0] * VOCAB] * 6)
        self.assertTrue(any("beyond the dense plan" in p for p in cur.problems))
        self.assertEqual(len(cur.losses), 5)

    def test_collector_error_case_rejected_even_with_matching_counts(self):
        dense, plan, _ = build()
        cur = collect_nll.NllCursor(plan)
        import torch
        wrapped = cur.wrap(collect_top1._harvest_row)
        for _ in plan:
            wrapped(torch, torch.zeros(VOCAB, dtype=torch.float32), VOCAB)
        errs = collect_nll.cross_check(cur, dense, make_art(dense, len(plan), err_case="beta"), 1)
        self.assertTrue(any("beta" in x for x in errs) and any("rc=" in x for x in errs))
        art2 = make_art(dense, len(plan))
        art2["complete"] = False
        self.assertTrue(collect_nll.cross_check(cur, dense, art2, 0))

    def test_missing_sidecar_rejected(self):
        dense, plan, _ = build()
        cur = collect_nll.NllCursor(plan)  # nothing harvested
        errs = collect_nll.cross_check(cur, dense, None, 1)
        self.assertTrue(errs and cur.losses == [])  # no coverage -> no plausible PPL
        self.assertTrue(any("rc=1" in x for x in errs))


class FailureReportingTests(unittest.TestCase):
    def test_harvest_exception_prevents_later_labeling(self):
        _, plan, _ = build()
        cur = collect_nll.NllCursor(plan)
        wrapped = cur.wrap(lambda *args: (0, True))
        with patch.object(collect_nll, "nll_row", side_effect=RuntimeError("harvest failure")):
            with self.assertRaises(RuntimeError):
                wrapped(None, None, VOCAB)
        self.assertTrue(cur.broken)
        wrapped(None, None, VOCAB)
        self.assertEqual(cur.losses, [])

    def test_collector_crash_writes_incomplete_and_restores_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp)/"source.json", Path(tmp)/"nll.json"
            src.write_text(json.dumps(make_src(CASES)))
            args = collect_nll.build_parser().parse_args([
                "--manifest", str(src), "--backend", "exl3", "-m", tmp,
                "--nll-out", str(out)])
            original = collect_top1._harvest_row
            with patch.object(collect_top1, "run", side_effect=RuntimeError("load failure")):
                self.assertEqual(collect_nll.run(args), 1)
            self.assertIs(collect_top1._harvest_row, original)
            report = json.loads(out.read_text())
            self.assertFalse(report["complete"])
            self.assertIsNone(report["nll"])


if __name__ == "__main__":
    unittest.main()
