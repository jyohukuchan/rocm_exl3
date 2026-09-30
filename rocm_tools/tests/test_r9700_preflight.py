#!/usr/bin/env python3
"""CPU-only tests for rocm_tools/r9700_preflight.py.

No GPU, no native extension, no network: analysis runs over offline fact
dicts / snapshot files, and the "live" path runs against a fake torch module
injected into sys.modules. unittest only (stdlib).
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import r9700_preflight as pf  # noqa: E402


def make_device(index=0, gcn="gfx1201:sramecc+:xnack-", smem=65536,
                vram=34359738368, name="AMD Radeon AI PRO R9700"):
    return {"index": index, "name": name, "gcn_arch_name": gcn,
            "total_memory": vram, "shared_memory_per_block": smem}


def make_facts(devices=None, torch_obj=None, rocm="default", env=None,
               **dev_kw):
    if devices is None:
        devices = [make_device(**dev_kw)]
    if torch_obj is None:
        torch_obj = {"importable": True, "version": "2.12.0+rocm7.2",
                     "hip_version": "7.2.53211-12345678",
                     "cuda_is_available": True,
                     "device_count": max(len(devices), 1)}
    facts = {
        "kind": pf.SNAPSHOT_KIND,
        "schema_version": pf.SNAPSHOT_SCHEMA_VERSION,
        "torch": torch_obj,
        "devices": devices,
    }
    if rocm == "default":
        facts["rocm"] = {"version": "7.2.4", "parsed": [7, 2, 4],
                         "version_file": "/opt/rocm/.info/version"}
    else:
        facts["rocm"] = rocm  # explicit null / bogus shapes pass through
    if env is not None:
        facts["env"] = env
    return facts


def run_cli(args):
    out, err = io.StringIO(), io.StringIO()
    code = pf.main(args, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def diag_codes(report):
    return [d["code"] for d in report["diagnostics"]]


class EnvOnSemantics(unittest.TestCase):
    """_env_on equality: these are the exact rocm_py semantics to match."""

    CASES = [
        (None, False), ("1", True), ("0", False), ("", False),
        ("false", False), ("False", False), ("true", True), ("TRUE", True),
        ("no", True),      # not in the off-list: ON -- matches runtime exactly
        ("FALSE", True),   # case-sensitive off-list: ON
        ("  ", False), (" 0 ", False), (" x ", True),
    ]

    def test_env_on_table(self):
        for value, expected in self.CASES:
            env = {} if value is None else {"V": value}
            self.assertEqual(pf.env_on(env, "V", False), expected,
                             f"value={value!r}")

    def test_env_on_default_true(self):
        self.assertTrue(pf.env_on({}, "V", True))
        self.assertFalse(pf.env_on({"V": "0"}, "V", True))


class NormalizeAndBuildTargetParsing(unittest.TestCase):
    def test_feature_suffix_stripped(self):
        self.assertEqual(pf.normalize_arch("gfx1201:sramecc+:xnack-"),
                         ("gfx1201", "sramecc+:xnack-"))
        self.assertEqual(pf.normalize_arch("gfx1201"), ("gfx1201", ""))

    def test_comma_separated_targets(self):
        archs, src = pf.resolve_build_targets(
            {"PYTORCH_ROCM_ARCH": "gfx1201,gfx1151"})
        self.assertEqual((archs, src), (["gfx1201", "gfx1151"],
                                        "PYTORCH_ROCM_ARCH"))

    def test_space_separated_and_padded(self):
        archs, _ = pf.resolve_build_targets({"GPU_ARCHS": "  gfx1151   gfx1201 "})
        self.assertEqual(archs, ["gfx1151", "gfx1201"])

    def test_targets_may_carry_feature_suffix(self):
        archs, _ = pf.resolve_build_targets(
            {"PYTORCH_ROCM_ARCH": "gfx1201:sramecc+:xnack-"})
        self.assertEqual(archs, ["gfx1201"])

    def test_empty_primary_falls_through_to_gpu_archs(self):
        archs, src = pf.resolve_build_targets(
            {"PYTORCH_ROCM_ARCH": "", "GPU_ARCHS": "gfx1201"})
        self.assertEqual((archs, src), (["gfx1201"], "GPU_ARCHS"))

    def test_primary_wins_when_both_set(self):
        archs, src = pf.resolve_build_targets(
            {"PYTORCH_ROCM_ARCH": "gfx1201", "GPU_ARCHS": "gfx1030"})
        self.assertEqual((archs, src), (["gfx1201"], "PYTORCH_ROCM_ARCH"))

    def test_unset_means_auto_detect(self):
        self.assertEqual(pf.resolve_build_targets({}), ([], None))


class Gfx1201HappyPath(unittest.TestCase):
    def setUp(self):
        self.report = pf.analyze(make_facts(), 0, {}, "test")

    def test_status_is_workarounds_not_ready(self):
        self.assertEqual(self.report["status"],
                         "comparison_workarounds_required")
        self.assertNotIn(self.report["status"],
                         ("ready", "general_supported"))
        self.assertEqual(self.report["exit_code"], 0)
        self.assertTrue(self.report["ok"])

    def test_arch_and_suffix_reported(self):
        dev = self.report["device"]
        self.assertEqual(dev["arch"], "gfx1201")
        self.assertEqual(dev["arch_features"], "sramecc+:xnack-")
        self.assertEqual(dev["gcn_arch_name"], "gfx1201:sramecc+:xnack-")

    def test_64kib_lds_expectation_met(self):
        lds = self.report["lds_expectation"]
        self.assertEqual(lds["expected_shared_memory_per_block"], 65536)
        self.assertEqual(lds["measured_shared_memory_per_block"], 65536)
        self.assertTrue(lds["match"])
        self.assertNotIn("lds_inadequate", diag_codes(self.report))
        self.assertEqual(self.report["device"]["total_memory_bytes"],
                         34359738368)

    def test_binary_target_stays_unknown(self):
        self.assertEqual(self.report["binary_target_verification"]["state"],
                         "unknown")

    def test_env_is_not_claimed_as_binary_evidence(self):
        rep = pf.analyze(make_facts(), 0, {"PYTORCH_ROCM_ARCH": "gfx1201"},
                         "test")
        self.assertTrue(rep["build_env"]["device_arch_in_targets"])
        self.assertIn("NOT evidence", rep["build_env"]["note"])
        self.assertEqual(rep["binary_target_verification"]["state"], "unknown")
        self.assertEqual(rep["exit_code"], 0)

    def test_next_steps_are_actionable_and_honest(self):
        joined = " ".join(self.report["next_steps"])
        for needle in ("PYTORCH_ROCM_ARCH", "65536", "hipcc_probe",
                       "EXL3_ROCM_RDNA4_FUSED_MOE", "r9700_vs_v620.md"):
            self.assertIn(needle, joined)

    def test_exit_zero_note(self):
        self.assertIn("does NOT", self.report["exit_note"])
        self.assertIn("report was collected", self.report["exit_note"])


class LdsMismatchDiagnosis(unittest.TestCase):
    def test_inadequate_lds_diagnosed(self):
        rep = pf.analyze(make_facts(smem=32768), 0, {}, "test")
        self.assertIn("lds_inadequate", diag_codes(rep))
        self.assertEqual(rep["lds_expectation"]["verdict"], "inadequate")
        self.assertFalse(rep["lds_expectation"]["match"])

    def test_unexpected_90k_lds_diagnosed(self):
        rep = pf.analyze(make_facts(smem=92160), 0, {}, "test")
        self.assertIn("lds_unexpected", diag_codes(rep))
        text = json.dumps(rep)
        self.assertIn("coop_autotune", text)  # points at the real failure

    def test_unreported_lds_diagnosed(self):
        rep = pf.analyze(make_facts(smem=None), 0, {}, "test")
        self.assertIn("lds_unreported", diag_codes(rep))
        self.assertIsNone(rep["lds_expectation"]["match"])


class FusedMoeForcing(unittest.TestCase):
    def _env(self, value):
        return {"EXL3_ROCM_RDNA4_FUSED_MOE": value} if value is not None else {}

    def test_forcing_on_gfx1201_blocks_with_finite_nonzero_exit(self):
        for value in ("1", "true", "TRUE", "no", "FALSE"):
            rep = pf.analyze(make_facts(), 0, self._env(value), "test")
            self.assertEqual(rep["status"], "blocked_fused_moe_forcing",
                             f"value={value!r}")
            self.assertIn(rep["exit_code"], (1, 2, 3))
            self.assertNotEqual(rep["exit_code"], 0)
            self.assertIn("unsafe_fused_moe_forcing", diag_codes(rep))
            self.assertIn("Unblock", " ".join(rep["next_steps"]))

    def test_off_values_do_not_block(self):
        for value in (None, "0", "false", "False", ""):
            rep = pf.analyze(make_facts(), 0, self._env(value), "test")
            self.assertEqual(rep["status"],
                             "comparison_workarounds_required",
                             f"value={value!r}")
            self.assertEqual(rep["exit_code"], 0)

    def test_gfx1200_also_blocks(self):
        rep = pf.analyze(
            make_facts(gcn="gfx1200:sramecc+:xnack-", smem=92160,
                       name="RDNA4 desktop"),
            0, {"EXL3_ROCM_RDNA4_FUSED_MOE": "1"}, "test")
        self.assertEqual(rep["status"], "blocked_fused_moe_forcing")
        self.assertNotEqual(rep["exit_code"], 0)

    def test_inert_on_non_gfx120x(self):
        rep = pf.analyze(
            make_facts(gcn="gfx1151:sramecc+:xnack-", smem=65536,
                       name="Strix Halo"),
            0, {"EXL3_ROCM_RDNA4_FUSED_MOE": "1"}, "test")
        self.assertEqual(rep["exit_code"], 0)
        self.assertIn("fused_moe_flag_inert", diag_codes(rep))
        self.assertEqual(rep["status"], "not_r9700_supported_part")


class DeviceAndSoftwareStatuses(unittest.TestCase):
    def test_non_r9700_supported_part(self):
        rep = pf.analyze(make_facts(gcn="gfx1030", smem=65536,
                                    name="V620"),
                         0, {}, "test")
        self.assertEqual(rep["status"], "not_r9700_supported_part")
        self.assertEqual(rep["exit_code"], 0)
        self.assertNotIn("EXL3_ROCM_RDNA4_FUSED_MOE",
                         " ".join(rep["next_steps"]))

    def test_gfx1200_unvalidated(self):
        rep = pf.analyze(make_facts(gcn="gfx1200", smem=92160), 0, {}, "test")
        self.assertEqual(rep["status"], "rdna4_unvalidated")

    def test_outside_supported_matrix(self):
        rep = pf.analyze(make_facts(gcn="gfx90a", smem=65536, name="MI210"),
                         0, {}, "test")
        self.assertEqual(rep["status"], "outside_supported_matrix")
        self.assertIn("outside_supported_matrix", diag_codes(rep))

    def test_hip_missing_no_devices(self):
        facts = make_facts(torch_obj={
            "importable": True, "version": "2.12.0+rocm7.2",
            "hip_version": "7.2.53211", "cuda_is_available": False,
            "device_count": 0}, devices=[])
        rep = pf.analyze(facts, 0, {}, "test")
        self.assertEqual(rep["status"], "no_hip_device_visible")
        self.assertEqual(rep["exit_code"], 2)

    def test_not_rocm_torch(self):
        facts = make_facts(torch_obj={
            "importable": True, "version": "2.8.0+cu126",
            "hip_version": None, "cuda_is_available": True,
            "device_count": 1})
        rep = pf.analyze(facts, 0, {}, "test")
        self.assertEqual(rep["status"], "not_rocm_torch")
        self.assertEqual(rep["exit_code"], 2)

    def test_torch_not_importable(self):
        facts = make_facts(torch_obj={
            "importable": False, "version": None, "hip_version": None,
            "cuda_is_available": False, "device_count": 0,
            "import_error": "ModuleNotFoundError: No module named 'torch'"},
            devices=[])
        rep = pf.analyze(facts, 0, {}, "test")
        self.assertEqual(rep["status"], "torch_not_importable")
        self.assertEqual(rep["exit_code"], 2)

    def test_rocm_below_minimum_warns(self):
        facts = make_facts(rocm={"version": "7.1.0", "parsed": [7, 1, 0],
                                 "version_file": "/opt/rocm/.info/version"})
        rep = pf.analyze(facts, 0, {}, "test")
        self.assertIn("rocm_below_minimum", diag_codes(rep))
        rep = pf.analyze(facts, 0, {"EXL3_SKIP_ROCM_VERSION_CHECK": "1"},
                         "test")
        self.assertNotIn("rocm_below_minimum", diag_codes(rep))


class SnapshotRoundTripAndValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, obj, name="snap.json", raw=None):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf8") as fp:
            fp.write(raw if raw is not None else json.dumps(obj))
        return path

    def test_snapshot_env_is_authoritative(self):
        path = self._write(make_facts(env={
            "EXL3_ROCM_RDNA4_FUSED_MOE": "1"}))
        with mock.patch.dict(os.environ, {}, clear=True):
            code, out, _ = run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(out)["status"],
                         "blocked_fused_moe_forcing")

    def test_snapshot_without_env_treated_as_unset(self):
        path = self._write(make_facts())  # no 'env' key
        with mock.patch.dict(os.environ,
                              {"EXL3_ROCM_RDNA4_FUSED_MOE": "1"}):
            code, out, _ = run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 0)
        rep = json.loads(out)
        self.assertEqual(rep["status"], "comparison_workarounds_required")
        self.assertEqual(rep["recorded_env"], {})

    def test_missing_file(self):
        code, _, err = run_cli(["--snapshot",
                                os.path.join(self.tmp.name, "nope.json")])
        self.assertEqual(code, 1)
        self.assertIn("cannot read snapshot", err)

    def test_malformed_json(self):
        path = self._write(None, raw="{not json")
        code, out, err = run_cli(["--snapshot", path])
        self.assertEqual(code, 1)
        self.assertIn("not valid JSON", err)

    def test_json_output_still_parsable_on_errors(self):
        path = self._write(None, raw="{not json")
        code, out, _ = run_cli(["--snapshot", path, "--json"])
        rep = json.loads(out)
        self.assertEqual(rep["status"], "invalid_snapshot")
        self.assertEqual(rep["exit_code"], 1)
        self.assertFalse(rep["ok"])

    def test_wrong_kind_and_schema(self):
        facts = make_facts()
        facts["kind"] = "other_tool"
        facts["schema_version"] = 99
        path = self._write(facts)
        code, out, err = run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 1)
        problems = " ".join(d["message"]
                            for d in json.loads(out)["diagnostics"])
        self.assertIn("'kind'", problems)
        self.assertIn("schema_version", problems)

    def test_missing_device_fields_reported_with_paths(self):
        facts = make_facts()
        del facts["devices"][0]["gcn_arch_name"]
        del facts["devices"][0]["shared_memory_per_block"]
        path = self._write(facts)
        code, out, _ = run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 1)
        problems = " ".join(d["message"]
                            for d in json.loads(out)["diagnostics"])
        self.assertIn("devices[0]", problems)
        self.assertIn("gcn_arch_name", problems)
        self.assertIn("shared_memory_per_block", problems)

    def test_bad_field_types_rejected(self):
        facts = make_facts()
        facts["devices"][0]["total_memory"] = "34GB"
        facts["devices"][0]["index"] = -2
        facts["env"] = {"PYTORCH_ROCM_ARCH": 1201}
        path = self._write(facts)
        code, out, _ = run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 1)
        problems = " ".join(d["message"]
                            for d in json.loads(out)["diagnostics"])
        for needle in ("total_memory", "index", "env"):
            self.assertIn(needle, problems)

    def test_non_object_root(self):
        path = self._write(None, raw="[1, 2, 3]")
        code, out, _ = run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 1)
        self.assertIn("JSON object", json.loads(out)["diagnostics"][0]["message"])

    def test_invalid_device_index_diagnostic(self):
        devices = [make_device(index=0),
                   make_device(index=2, gcn="gfx1151", smem=65536,
                               name="Strix Halo")]
        path = self._write(make_facts(devices=devices,
                                      torch_obj={"importable": True,
                                                 "version": "2.12.0+rocm7.2",
                                                 "hip_version": "7.2",
                                                 "cuda_is_available": True,
                                                 "device_count": 2}))
        code, out, _ = run_cli(["--snapshot", path, "--device", "1",
                                "--json"])
        self.assertEqual(code, 1)
        rep = json.loads(out)
        self.assertEqual(rep["status"], "invalid_device_selection")
        self.assertIn("[0, 2]", rep["diagnostics"][0]["message"])
        # a valid index works
        code, out, _ = run_cli(["--snapshot", path, "--device", "2",
                                "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["device"]["arch"], "gfx1151")


class CliTextAndJson(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "snap.json")
        with open(self.path, "w", encoding="utf8") as fp:
            json.dump(make_facts(env={"PYTORCH_ROCM_ARCH": "gfx1201"}), fp)

    def test_help_runs(self):
        with self.assertRaises(SystemExit) as cm:
            pf.main(["--help"], out=io.StringIO(), err=io.StringIO())
        self.assertEqual(cm.exception.code, 0)

    def test_text_output_is_human_readable_exit0(self):
        code, out, _ = run_cli(["--snapshot", self.path])
        self.assertEqual(code, 0)
        self.assertIn("status: comparison_workarounds_required", out)
        self.assertIn("gfx1201", out)
        self.assertIn("65536", out)
        self.assertIn("[next steps]", out)
        self.assertIn("does NOT", out)

    def test_json_output_matches_exit_code(self):
        code, out, _ = run_cli(["--snapshot", self.path, "--json"])
        rep = json.loads(out)
        self.assertEqual(code, rep["exit_code"])
        self.assertEqual(rep["tool"], "r9700_preflight")
        self.assertEqual(rep["device"]["arch"], "gfx1201")

    def test_blocked_exit_text_mode(self):
        bad = os.path.join(self.tmp.name, "bad.json")
        with open(bad, "w", encoding="utf8") as fp:
            json.dump(make_facts(
                env={"EXL3_ROCM_RDNA4_FUSED_MOE": "1"}), fp)
        code, out, err = run_cli(["--snapshot", bad])
        self.assertEqual(code, 3)
        self.assertIn("blocked_fused_moe_forcing", out)
        self.assertTrue(err)

    def test_hip_missing_snapshot_exit2(self):
        none = os.path.join(self.tmp.name, "none.json")
        with open(none, "w", encoding="utf8") as fp:
            json.dump(make_facts(devices=[], torch_obj={
                "importable": True, "version": "2.12.0+rocm7.2",
                "hip_version": "7.2", "cuda_is_available": False,
                "device_count": 0}), fp)
        code, _, _ = run_cli(["--snapshot", none])
        self.assertEqual(code, 2)


class ReviewRegressionF1Encoding(unittest.TestCase):
    """Bytes that are not UTF-8 are a diagnostic, not a traceback."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_binary_snapshot_finite_nonzero_json_diagnostic(self):
        path = os.path.join(self.tmp.name, "utf16.json")
        with open(path, "wb") as fp:
            fp.write(b"\xff\xfe")
        code, out, err = run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 1)
        rep = json.loads(out)
        self.assertEqual(rep["status"], "invalid_snapshot")
        self.assertIn("UTF-8", rep["diagnostics"][0]["message"])
        code, out, err = run_cli(["--snapshot", path])
        self.assertEqual(code, 1)
        self.assertIn("UTF-8", err)

    def test_non_utf8_rocm_version_file_is_unavailable_fact(self):
        # read_rocm_version must degrade to None, never leak UnicodeDecodeError
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, ".info"))
            with open(os.path.join(d, ".info", "version"), "wb") as fp:
                fp.write(b"\xff\xfe\x00\x01")
            parsed, raw, path = pf.read_rocm_version({"ROCM_PATH": d})
        self.assertIsNone(parsed)
        self.assertIsNone(raw)
        facts = make_facts(rocm={"version": raw, "parsed": parsed,
                                 "version_file": path})
        rep = pf.analyze(facts, 0, {}, "test")
        self.assertEqual(rep["exit_code"], 0)
        self.assertIn("rocm_version_unreadable", diag_codes(rep))


class ReviewRegressionF2Types(unittest.TestCase):
    """Documented types are enforced; legitimate null stays accepted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _reject(self, facts, *needles):
        path = os.path.join(self.tmp.name, "s.json")
        with open(path, "w", encoding="utf8") as fp:
            json.dump(facts, fp)
        code, out, _ = run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 1)
        problems = " ".join(d["message"]
                            for d in json.loads(out)["diagnostics"])
        for needle in needles:
            self.assertIn(needle, problems)

    def test_hip_version_bogus_types_rejected(self):
        for bad in (False, 1, {}, [], "", "  "):
            facts = make_facts(torch_obj={
                "importable": True, "version": "2.12.0+rocm7.2",
                "hip_version": bad, "cuda_is_available": True,
                "device_count": 1})
            with self.subTest(bad=bad):
                self._reject(facts, "hip_version")

    def test_torch_version_bogus_type_rejected(self):
        facts = make_facts(torch_obj={
            "importable": True, "version": ["2.12"],
            "hip_version": "7.2", "cuda_is_available": True,
            "device_count": 1})
        self._reject(facts, "torch.version")

    def test_null_versions_are_legitimate(self):
        # null hip_version is the *documented* spelling of "no HIP"; it must
        # analyze to the HIP-missing diagnostic (exit 2), not be schema-rejected
        facts = make_facts(torch_obj={
            "importable": True, "version": "2.8.0+cu126",
            "hip_version": None, "cuda_is_available": True,
            "device_count": 1})
        rep = pf.analyze(pf.validate_facts(facts, "test"), 0, {}, "test")
        self.assertEqual(rep["status"], "not_rocm_torch")
        self.assertEqual(rep["exit_code"], 2)

    def test_rocm_bogus_shapes_rejected_not_raised(self):
        for bad in ("7.2.4", [], 42, True):
            with self.subTest(bad=bad):
                self._reject(make_facts(rocm=bad), "rocm")

    def test_rocm_parsed_must_be_three_nonneg_ints(self):
        self._reject(make_facts(rocm={"version": "7.14.0",
                                      "parsed": ["7", "14", "0"],
                                      "version_file": "/x"}),
                     "rocm.parsed")
        self._reject(make_facts(rocm={"version": "7.2.4",
                                      "parsed": [7, 2],
                                      "version_file": "/x"}),
                     "rocm.parsed")
        self._reject(make_facts(rocm={"version": "7.2.4",
                                      "parsed": [7, 2, 4, 1],
                                      "version_file": "/x"}),
                     "rocm.parsed")
        self._reject(make_facts(rocm={"version": "7.2.4",
                                      "parsed": [7, 2, True],
                                      "version_file": "/x"}),
                     "rocm.parsed")
        # negative components rejected
        self._reject(make_facts(rocm={"version": "?", "parsed": [-1, 0, 0],
                                      "version_file": "/x"}),
                     "rocm.parsed")

    def test_rocm_version_empty_string_rejected(self):
        self._reject(make_facts(rocm={"version": "", "parsed": None,
                                      "version_file": ""}),
                     "rocm.version")

    def test_rocm_null_accepted(self):
        rep = pf.analyze(pf.validate_facts(make_facts(rocm=None), "t"),
                         0, {}, "test")
        self.assertEqual(rep["exit_code"], 0)

    def test_schema_version_bool_rejected(self):
        # True == 1 in Python, but schema_version must be an INTEGER
        facts = make_facts()
        facts["schema_version"] = True
        self._reject(facts, "schema_version")


class ReviewRegressionF3F4ArchAdvice(unittest.TestCase):

    def _gfx1200(self):
        return make_facts(gcn="gfx1200:xnack-", smem=92160,
                          name="RX 9070-class RDNA4")

    def _steps(self, report):
        return " ".join(report["next_steps"])

    def test_gfx1200_unforced_advice_uses_gfx1200(self):
        rep = pf.analyze(self._gfx1200(), 0, {}, "test")
        steps = self._steps(rep)
        self.assertIn("PYTORCH_ROCM_ARCH=gfx1200", steps)
        self.assertNotIn("PYTORCH_ROCM_ARCH=gfx1201", steps)
        self.assertNotIn("GPU_ARCH=gfx1201", steps)

    def test_gfx1200_forced_advice_uses_gfx1200(self):
        rep = pf.analyze(self._gfx1200(), 0,
                         {"EXL3_ROCM_RDNA4_FUSED_MOE": "1"}, "test")
        self.assertEqual(rep["status"], "blocked_fused_moe_forcing")
        steps = self._steps(rep)
        self.assertIn("PYTORCH_ROCM_ARCH=gfx1200", steps)
        self.assertNotIn("PYTORCH_ROCM_ARCH=gfx1201", steps)
        self.assertNotIn("GPU_ARCH=gfx1201", steps)

    def test_table_expectations_stay_distinct(self):
        g1200 = pf.analyze(self._gfx1200(), 0, {}, "test")
        self.assertEqual(
            g1200["lds_expectation"]["expected_shared_memory_per_block"],
            92160)  # gfx1200: table assumption, not the R9700 measurement
        steps = self._steps(g1200)
        self.assertIn("table-level assumption", steps)
        self.assertIn("92160", steps)
        self.assertIn("64 KiB figure blindly", steps)
        g1201 = pf.analyze(make_facts(), 0, {}, "test")
        self.assertEqual(
            g1201["lds_expectation"]["expected_shared_memory_per_block"],
            65536)
        self.assertIn("65536", self._steps(g1201))
        self.assertIn("measured", self._steps(g1201))

    def test_no_invented_quoted_claim(self):
        for arch_facts in (self._gfx1200(), make_facts()):
            rep = pf.analyze(arch_facts, 0, {}, "test")
            text = json.dumps(rep)
            self.assertNotIn("no RDNA4 hardware has ever run this port", text)
        rep = pf.analyze(self._gfx1200(), 0, {}, "test")
        self.assertEqual(rep["status"], "rdna4_unvalidated")
        self.assertIn("unvalidated",
                      " ".join(d["message"] for d in rep["diagnostics"]))

    def test_probe_is_source_compile_not_binary_inspection(self):
        steps = self._steps(pf.analyze(make_facts(), 0, {}, "test"))
        self.assertNotIn("compiled artifacts", steps)
        self.assertIn("SOURCE-compilation", steps)
        self.assertIn("fresh exllamav3_ext translation units", steps)
        self.assertIn("does NOT inspect", steps)
        self.assertIn("already-installed", steps)
        self.assertIn("outside this preflight", steps)
        rep = pf.analyze(make_facts(), 0, {}, "test")
        self.assertEqual(rep["binary_target_verification"]["state"], "unknown")


class ReviewRegressionF5ReadOnly(unittest.TestCase):
    def test_save_snapshot_flag_is_gone(self):
        # the read-only CLI must not offer a file-writing mode any more
        with self.assertRaises(SystemExit) as cm:
            pf.main(["--save-snapshot", "/tmp/whatever.json"],
                    out=io.StringIO(), err=io.StringIO())
        self.assertEqual(cm.exception.code, 2)
        self.assertNotIn("save-snapshot",
                         pf.__doc__ + str(pf.main.__doc__))

    def test_help_lists_only_read_only_modes(self):
        buf = io.StringIO()
        real_stdout, sys.stdout = sys.stdout, buf
        try:
            with self.assertRaises(SystemExit):
                pf.main(["--help"])
        finally:
            sys.stdout = real_stdout
        help_text = buf.getvalue()
        self.assertIn("--snapshot FILE", help_text)
        self.assertNotIn("--save-snapshot", help_text)


class ReviewRegressionF6SkipGateTruthiness(unittest.TestCase):
    """setup.py tests raw truthiness of EXL3_SKIP_ROCM_VERSION_CHECK, not
    _env_on: ANY non-empty string (even "0") skips the gate; "" does not."""

    def _facts(self):
        return make_facts(rocm={"version": "7.1.0", "parsed": [7, 1, 0],
                                "version_file": "/opt/rocm/.info/version"})

    def _warns(self, env):
        rep = pf.analyze(self._facts(), 0, env, "test")
        return "rocm_below_minimum" in diag_codes(rep)

    def test_unset_and_empty_warn(self):
        self.assertTrue(self._warns({}))
        self.assertTrue(self._warns({"EXL3_SKIP_ROCM_VERSION_CHECK": ""}))

    def test_any_nonempty_string_skips_including_zero(self):
        for value in ("0", "1", "false", "False", " ", "anything"):
            with self.subTest(value=value):
                self.assertFalse(
                    self._warns({"EXL3_SKIP_ROCM_VERSION_CHECK": value}))

    def test_moe_env_facts_follow_real_defaults(self):
        rep = pf.analyze(make_facts(), 0, {
            "EXL3_ROCM_MOE_MGEMM_ROUTE": "0",   # default True in rocm_py
            "EXL3_ROCM_MOE_BSZN": "1",          # default False
            "EXL3_ROCM_MOE_MGEMM_MAX_ROWS": "20"}, "test")
        moe = rep["fused_moe_env"]
        self.assertFalse(moe["EXL3_ROCM_MOE_MGEMM_ROUTE"]["env_on"])
        self.assertTrue(moe["EXL3_ROCM_MOE_BSZN"]["env_on"])
        # integer setting: recorded as a value, no boolean reading guessed
        self.assertNotIn("env_on", moe["EXL3_ROCM_MOE_MGEMM_MAX_ROWS"])
        self.assertEqual(moe["EXL3_ROCM_MOE_MGEMM_MAX_ROWS"]["value"], "20")


class LiveQueryWithFakeTorch(unittest.TestCase):
    """The default path must work by lazily importing torch and reading only
    device properties -- here with a stub module so it runs CPU-only."""

    def _fake_torch(self, props=None):
        torch = types.ModuleType("torch")
        torch.__version__ = "2.12.0+rocm7.2"
        torch.version = types.SimpleNamespace(hip="7.2.53211-12345678")
        devices = props or [types.SimpleNamespace(
            name="AMD Radeon AI PRO R9700",
            gcnArchName="gfx1201:sramecc+:xnack-",
            total_memory=34359738368,
            shared_memory_per_block=65536)]
        cuda = types.ModuleType("torch.cuda")
        cuda.is_available = lambda: True
        cuda.device_count = lambda: len(devices)
        cuda.get_device_properties = lambda i: devices[i]
        torch.cuda = cuda
        return torch

    def test_live_query_records_only_relevant_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            rocm_fake = pathlib.Path(tmp) / "rocm"
            (rocm_fake / ".info").mkdir(parents=True)
            (rocm_fake / ".info" / "version").write_text("7.2.4-12345\n")
            env = {"ROCM_PATH": str(rocm_fake),
                   "PYTORCH_ROCM_ARCH": "gfx1201",
                   "SUPER_SECRET_TOKEN": "hunter2"}
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}), \
                 mock.patch.dict(os.environ, env, clear=True):
                code, out, _ = run_cli(["--json"])
            rep = json.loads(out)
            self.assertEqual(code, 0)
            self.assertEqual(rep["status"],
                             "comparison_workarounds_required")
            self.assertEqual(rep["device"]["arch"], "gfx1201")
            self.assertEqual(rep["software"]["rocm_path_version"],
                             "7.2.4-12345")
            self.assertEqual(rep["software"]["min_rocm_expected"], "7.2.4")
            self.assertEqual(rep["build_env"]["target_archs"], ["gfx1201"])
        # allowlist: unrelated env vars (here: a secret-shaped one) are never
        # collected, recorded, or printed anywhere
        env = {k: v for k, v in os.environ.items()}
        env["SUPER_SECRET_TOKEN"] = "hunter2"
        with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
            with mock.patch.dict(os.environ, env, clear=True):
                facts = pf.collect_live_facts(dict(os.environ))
        self.assertNotIn("SUPER_SECRET_TOKEN", facts["env"])
        self.assertNotIn("SUPER_SECRET_TOKEN", json.dumps(rep))
        self.assertNotIn("hunter2", json.dumps(rep) + out)

    def test_live_path_tolerates_missing_torch(self):
        with mock.patch.dict(sys.modules, {"torch": None}), \
             mock.patch.dict(os.environ, {}, clear=True):
            code, _, _ = run_cli([])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
