#!/usr/bin/env python3
"""CPU-only tests for the rdna2 harness (no GPU, no torch required).

Covers the parts whose correctness matters before anything touches a V620:
manifest self-validation, deterministic position selection, and -- centrally --
compare_top1's accept/reject behaviour and exact agreement arithmetic, driven
through the real code paths with hand-checked expected numbers (fixtures are
input data, not implementation mirrors).

Run from the repo root (or anywhere):
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2 import common
from rocm_tools.rdna2 import compare_top1

VOCAB = 1000

def make_manifest(n_positions=(8, 4)) -> dict:
    cases = []
    for i, n in enumerate(n_positions):
        ids = list(range(10, 10 + 20))          # arbitrary in-vocab literals
        positions = list(range(n))
        cases.append({
            "case_id": f"case_{i}",
            "language": "en",
            "kind": "natural",
            "provenance": "test",
            "text_sha256": "x",
            "len_ids": len(ids),
            "ids": ids,
            "ids_sha256": common.case_ids_sha256(ids),
            "positions": positions,
        })
    m = {
        "format": common.MANIFEST_FORMAT,
        "valid_vocab_size": VOCAB,
        "config_vocab_size": VOCAB,
        "total_positions": sum(n_positions),
        "position_semantics": common.POSITION_SEMANTICS,
        "cases": cases,
    }
    m["manifest_sha256"] = common.manifest_digest(m)
    return m


def make_result(manifest: dict, top1_by_case: dict[str, list[int]],
                **overrides) -> dict:
    r = {
        "format": common.TOP1_FORMAT,
        "backend": "stub",
        "model_dir": "/models/stub",
        "repo_git_commit": "0" * 40,
        "manifest_sha256": manifest["manifest_sha256"],
        "position_semantics": common.POSITION_SEMANTICS,
        "valid_vocab_size": manifest["valid_vocab_size"],
        "cases": {},
        "total_positions": manifest["total_positions"],
        "complete": True,
        "errors": [],
        "execution": {},
        "env": {},
        "model_fingerprint": {},
    }
    for case in manifest["cases"]:
        cid = case["case_id"]
        if cid not in top1_by_case:
            continue          # simulate a subset run (--limit-cases style)
        r["cases"][cid] = {
            "status": "ok",
            "case_id": cid,
            "ids_sha256": case["ids_sha256"],
            "positions": case["positions"],
            "top1": list(top1_by_case[cid]),
            "nonfinite_positions": [],
        }
    r.update(overrides)
    return r


def write(tmpdir: Path, name: str, obj) -> str:
    p = tmpdir / name
    with open(p, "w", encoding = "utf-8") as f:
        json.dump(obj, f)
    return str(p)


class ManifestTests(unittest.TestCase):
    def test_valid_manifest_passes_validation(self):
        self.assertEqual(common.validate_manifest(make_manifest()), [])

    def test_digest_catches_edited_ids(self):
        m = make_manifest()
        m["cases"][0]["ids"][3] += 1          # edited without re-hashing anything
        errors = common.validate_manifest(m)
        self.assertTrue(any("manifest_sha256 does not match" in e for e in errors))

    def test_position_out_of_ids_range_rejected(self):
        m = make_manifest()
        m["cases"][0]["positions"] = [0, 1, 2, 3, 4, 5, 6, 19]  # ids len = 20 -> 19 legal
        m["manifest_sha256"] = common.manifest_digest(m)
        self.assertEqual(common.validate_manifest(m), [])
        m2 = make_manifest()
        m2["cases"][0]["positions"] = [0, 1, 2, 3, 4, 5, 6, 20]  # 20 == len -> invalid
        m2["manifest_sha256"] = common.manifest_digest(m2)
        errors = common.validate_manifest(m2)
        self.assertTrue(any("outside [0, len(ids)-1]" in e for e in errors), errors)

    def test_unsorted_positions_rejected(self):
        m = make_manifest()
        m["cases"][0]["positions"] = [1, 0, 2, 3, 4, 5, 6, 7]
        m["manifest_sha256"] = common.manifest_digest(m)
        errors = common.validate_manifest(m)
        self.assertTrue(any("sorted" in e for e in errors), errors)

    def test_token_id_outside_vocab_rejected(self):
        m = make_manifest()
        m["cases"][0]["ids"][0] = VOCAB  # one past valid vocab
        m["manifest_sha256"] = common.manifest_digest(m)
        errors = common.validate_manifest(m)
        self.assertTrue(any("outside [0, valid_vocab_size)" in e for e in errors), errors)


class PositionSelectionTests(unittest.TestCase):
    def test_allocate_sums_and_bounds(self):
        counts = common.allocate_positions([20, 10, 170], 64)
        self.assertEqual(sum(counts), 64)
        for c, L in zip(counts, [20, 10, 170]):
            self.assertGreaterEqual(c, 1)
            self.assertLessEqual(c, L)
        # proportional: the 170-long case gets the lion's share
        self.assertGreaterEqual(counts[2], 40)

    def test_allocate_deterministic(self):
        a = common.allocate_positions([20, 10, 170], 64)
        b = common.allocate_positions([20, 10, 170], 64)
        self.assertEqual(a, b)

    def test_allocate_rejects_over_capacity(self):
        with self.assertRaises(ValueError):
            common.allocate_positions([5, 5], 11)

    def test_select_positions_strictly_increasing_in_range(self):
        pos = common.select_positions(20, 8)
        self.assertEqual(len(pos), 8)
        self.assertTrue(all(0 <= p < 20 for p in pos))
        self.assertEqual(pos, sorted(set(pos)))

    def test_select_positions_full_density_edge(self):
        pos = common.select_positions(4, 4)
        self.assertEqual(len(pos), 4)
        self.assertEqual(sorted(pos), pos)
        self.assertTrue(max(pos) <= 3)

    def test_round_up_page(self):
        self.assertEqual(common.round_up_page(256), 256)
        self.assertEqual(common.round_up_page(257), 512)
        self.assertEqual(common.round_up_page(4000), 4096)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.manifest = make_manifest((8, 4))

    def tearDown(self):
        self.tmp.cleanup()

    def _files(self, ref_result, cand_result):
        return (write(self.dir, "ref.json", ref_result),
                write(self.dir, "cand.json", cand_result))

    def test_exact_agreement_arithmetic(self):
        # Hand-computed expectations: case_0 agrees on 6/8 (indices 2 and 6
        # flipped), case_1 on 4/4 => overall 10/12 = 0.833333...
        ref0 = [10, 11, 12, 13, 14, 15, 16, 17]
        cand0 = [10, 11, 999, 13, 14, 15, 777, 17]
        ref1 = [5, 6, 7, 8]
        cand1 = [5, 6, 7, 8]
        ref = make_result(self.manifest, {"case_0": ref0, "case_1": ref1})
        cand = make_result(self.manifest, {"case_0": cand0, "case_1": cand1})
        a, b = self._files(ref, cand)
        code, summary = compare_top1.compare_files(a, b)
        self.assertEqual(code, 0, summary.get("reject_reasons"))
        self.assertFalse(summary["rejected"])
        self.assertEqual(summary["total_positions"], 12)
        self.assertEqual(summary["top1_agreements"], 10)
        self.assertAlmostEqual(summary["top1_agreement_rate"], 10 / 12, places = 12)
        per = {c["case_id"]: c for c in summary["per_case"]}
        self.assertEqual(per["case_0"]["agreement"], 6)
        self.assertEqual(per["case_0"]["n_diffs"], 2)
        self.assertEqual(per["case_1"]["agreement"], 4)
        self.assertEqual(per["case_0"]["first_diff_positions"], [2, 6])

    def test_rejects_different_manifest_digest(self):
        other = make_manifest((8, 4))
        other["cases"][1]["positions"] = [0, 1, 2, 4]     # genuinely different inputs...
        other["manifest_sha256"] = common.manifest_digest(other)  # ...so genuinely different digest
        ref = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand = make_result(other, {"case_0": [1] * 8, "case_1": [2] * 4})
        a, b = self._files(ref, cand)
        code, summary = compare_top1.compare_files(a, b)
        self.assertEqual(code, 1)
        self.assertTrue(summary["rejected"])
        self.assertTrue(any("manifest_sha256" in r for r in summary["reject_reasons"]))

    def test_rejects_different_positions_same_manifest_field(self):
        ref = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        # positions altered (8 distinct in-range entries, different set); the
        # stored manifest field is left identical on purpose, so the reject can
        # only come from the per-case position comparison:
        cand["cases"]["case_0"]["positions"] = [0, 2, 3, 4, 5, 6, 7, 8]
        a, b = self._files(ref, cand)
        code, summary = compare_top1.compare_files(a, b)
        self.assertEqual(code, 1)
        self.assertTrue(any("position lists differ" in r for r in summary["reject_reasons"]),
                        summary["reject_reasons"])

    def test_rejects_different_vocab(self):
        ref = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand["valid_vocab_size"] = VOCAB + 10
        cand["manifest_sha256"] = self.manifest["manifest_sha256"]
        a, b = self._files(ref, cand)
        code, summary = compare_top1.compare_files(a, b)
        self.assertEqual(code, 1)
        self.assertTrue(any("valid_vocab_size" in r for r in summary["reject_reasons"]))

    def test_rejects_incomplete_result(self):
        ref = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4},
                           complete = False, errors = ["boom"])
        cand["cases"]["case_1"] = {"status": "error", "case_id": "case_1",
                                   "error": "RuntimeError('boom')"}
        a, b = self._files(ref, cand)
        code, summary = compare_top1.compare_files(a, b)
        self.assertEqual(code, 1)
        reasons = summary["reject_reasons"]
        self.assertTrue(any("incomplete" in r for r in reasons), reasons)

    def test_rejects_nonfinite_flag(self):
        ref = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand["cases"]["case_0"]["nonfinite_positions"] = [3]
        a, b = self._files(ref, cand)
        code, summary = compare_top1.compare_files(a, b)
        self.assertEqual(code, 1)
        self.assertTrue(any("non-finite" in r for r in summary["reject_reasons"]),
                        summary["reject_reasons"])

    def test_rejects_top1_outside_vocab(self):
        ref = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand["cases"]["case_0"]["top1"][2] = VOCAB  # one past valid vocab
        a, b = self._files(ref, cand)
        code, summary = compare_top1.compare_files(a, b)
        self.assertEqual(code, 1)
        self.assertTrue(any("outside [0, valid_vocab_size)" in r
                            for r in summary["reject_reasons"]), summary["reject_reasons"])

    def test_rejects_missing_case_on_one_side(self):
        ref = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        del cand["cases"]["case_1"]
        cand["total_positions"] = 8
        cand["complete"] = False
        a, b = self._files(ref, cand)
        code, summary = compare_top1.compare_files(a, b)
        self.assertEqual(code, 1)
        self.assertTrue(any("missing" in r or "incomplete" in r
                            for r in summary["reject_reasons"]), summary["reject_reasons"])

    def test_cross_validation_against_manifest_file(self):
        ref = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        cand = make_result(self.manifest, {"case_0": [1] * 8, "case_1": [2] * 4})
        a, b = self._files(ref, cand)
        mpath = write(self.dir, "m.json", self.manifest)
        code, summary = compare_top1.compare_files(a, b, mpath)
        self.assertEqual(code, 0, summary.get("reject_reasons"))
        # now corrupt the manifest copy on disk (positions edited, digest recomputed):
        m2 = make_manifest((8, 4))
        m2["cases"][0]["positions"] = [1, 2, 3, 4, 5, 6, 7, 8]
        m2["manifest_sha256"] = common.manifest_digest(m2)
        m2path = write(self.dir, "m2.json", m2)
        code, summary = compare_top1.compare_files(a, b, m2path)
        self.assertEqual(code, 1)
        self.assertTrue(any("digest does not match" in r or "differ from manifest" in r
                            for r in summary["reject_reasons"]),
                        summary["reject_reasons"])
        # a case from the manifest missing from the artifacts is rejected even
        # when both artifacts agree with each other (e.g. both --limit-cases runs):
        m3 = make_manifest((8, 4, 4))
        ref_part = make_result(m3, {"case_0": [1] * 8, "case_1": [2] * 4, "case_2": [3] * 4})
        cand_part = make_result(m3, {"case_0": [1] * 8, "case_1": [2] * 4, "case_2": [3] * 4})
        del ref_part["cases"]["case_2"]
        del cand_part["cases"]["case_2"]
        ref_part["total_positions"] = cand_part["total_positions"] = 12
        a3 = write(self.dir, "a3.json", ref_part)
        b3 = write(self.dir, "b3.json", cand_part)
        m3path = write(self.dir, "m3.json", m3)
        code, summary = compare_top1.compare_files(a3, b3, m3path)
        self.assertEqual(code, 1)
        self.assertTrue(any("missing from the artifact" in r
                            for r in summary["reject_reasons"]),
                        summary["reject_reasons"])



class ThresholdCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.manifest = make_manifest((8, 4))

    def tearDown(self):
        self.tmp.cleanup()

    def _write_pair(self):
        ref = make_result(self.manifest, {"case_0": [10, 11, 12, 13, 14, 15, 16, 17],
                                          "case_1": [5, 6, 7, 8]})
        cand = make_result(self.manifest, {"case_0": [10, 11, 999, 13, 14, 15, 777, 17],
                                           "case_1": [5, 6, 7, 8]})
        a = write(self.dir, "ref.json", ref)
        b = write(self.dir, "cand.json", cand)
        return a, b  # agreement 10/12 ~= 0.8333

    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = compare_top1.main(argv)
        return rc, buf.getvalue()

    def test_threshold_pass_and_fail_exit_codes(self):
        a, b = self._write_pair()
        rc, out = self._run(["--reference", a, "--candidate", b, "--min-agreement", "0.8"])
        self.assertEqual(rc, 0, out)
        self.assertIn("10/12", out)
        rc, out = self._run(["--reference", a, "--candidate", b, "--min-agreement", "0.995"])
        self.assertEqual(rc, 2, out)
        self.assertIn("FAIL", out)
        rc, out = self._run(["--reference", a, "--candidate", b])
        self.assertEqual(rc, 0, out)


class TruncationRegressionTests(unittest.TestCase):
    """
    Review point: a --limit-cases subset must never become "complete" just
    because collected and total were reduced together. total_positions is
    relative to the FULL manifest, so a truncated artifact self-flags and
    comparison refuses even two identically truncated artifacts.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, obj):
        p = self.dir / name
        with open(p, "w", encoding = "utf-8") as f:
            json.dump(obj, f)
        return str(p)

    def test_identically_truncated_pair_is_rejected(self):
        full = make_manifest((8, 4, 4))            # 16 positions, 3 cases
        subset = {"case_0": [10, 11, 12, 13, 14, 15, 16, 17],
                  "case_1": [5, 6, 7, 8]}
        # both sides ran the SAME subset with IDENTICAL top-1s: agreement would
        # be a vacuous 1.0 if truncation were tolerated.
        ref = make_result(full, subset)            # total stays 16, cases cover 12
        cand = make_result(full, subset)
        a = self._write("ref.json", ref)
        b = self._write("cand.json", cand)
        code, summary = compare_top1.compare_files(a, b)
        self.assertEqual(code, 1, summary)
        reasons = summary["reject_reasons"]
        # self-validation flags collected(12) < full-manifest total(16)
        self.assertTrue(any("12 positions collected out of 16" in r for r in reasons), reasons)
        # ...on BOTH sides, not just one
        self.assertTrue(any(r.startswith("reference:") for r in reasons), reasons)
        self.assertTrue(any(r.startswith("candidate:") for r in reasons), reasons)

    def test_truncated_pair_rejected_with_manifest_too(self):
        full = make_manifest((8, 4, 4))
        subset = {"case_0": [10, 11, 12, 13, 14, 15, 16, 17],
                  "case_1": [5, 6, 7, 8]}
        ref = make_result(full, subset)
        cand = make_result(full, subset)
        a = self._write("ref.json", ref)
        b = self._write("cand.json", cand)
        mpath = self._write("m.json", full)
        code, summary = compare_top1.compare_files(a, b, mpath)
        self.assertEqual(code, 1, summary)
        self.assertTrue(any("missing from the artifact" in r
                            for r in summary["reject_reasons"]), summary["reject_reasons"])

    def test_subset_collected_matches_subset_total_still_flagged(self):
        # The buggy shape the review targets: an artifact that re-based
        # total_positions down to the subset AND claimed complete.
        full = make_manifest((8, 4, 4))
        ref = make_result(full, {"case_0": [1] * 8, "case_1": [2] * 4},
                          total_positions = 12)   # re-based: WRONG, must not be
        cand = make_result(full, {"case_0": [1] * 8, "case_1": [2] * 4},
                           total_positions = 12)  # trusted when a manifest exists
        a = self._write("ref.json", ref)
        b = self._write("cand.json", cand)
        mpath = self._write("m.json", full)
        code, summary = compare_top1.compare_files(a, b, mpath)
        self.assertEqual(code, 1, summary)
        self.assertTrue(any("case case_2 from the manifest is missing" in r
                            for r in summary["reject_reasons"]), summary["reject_reasons"])


class DtypeChoiceTests(unittest.TestCase):
    """collect_top1's transformers --dtype auto logic (pure, CPU-testable)."""

    def test_auto_keeps_checkpoint_native_dtype(self):
        choice, note = common.resolve_dtype_choice("auto", "bfloat16")
        self.assertEqual(choice, "bf16")
        self.assertIn("resolved from checkpoint", note)

    def test_auto_defaults_when_no_declaration(self):
        choice, note = common.resolve_dtype_choice("auto", None)
        self.assertEqual(choice, "fp16")
        self.assertIn("fell back", note)

    def test_explicit_request_wins_over_declaration(self):
        # an fp16 reference is a real option (gfx1030 runs BF16 fine, but fp16
        # may be wanted for a cheaper pass); "auto" never overrides an explicit
        # choice in either direction.
        self.assertEqual(common.resolve_dtype_choice("fp16", "bfloat16")[0], "fp16")
        self.assertEqual(common.resolve_dtype_choice("bf16", "float32")[0], "bf16")

    def test_auto_handles_torch_prefixed_and_float16_declarations(self):
        self.assertEqual(common.resolve_dtype_choice("auto", "torch.bfloat16")[0], "bf16")
        self.assertEqual(common.resolve_dtype_choice("auto", "float32")[0], "fp32")
        self.assertEqual(common.resolve_dtype_choice("auto", "Half")[0], "fp16")


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_transfer_flags_are_recorded_without_unrelated_environment(self):
        flags = {"HSA_ENABLE_SDMA": "0", "HSA_ENABLE_PEER_SDMA": "0",
                 "EXLLAMA_NO_P2P_COPY": "1", "EXL3_ROCM_GQA_TUNE": "1"}
        with patch.dict(os.environ, {**flags, "UNRELATED_SECRET": "not-recorded"}, clear=True):
            self.assertEqual(common.rocm_patch_env(), flags)


if __name__ == "__main__":
    unittest.main()
