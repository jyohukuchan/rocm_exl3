#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r9700_preflight.py のオフライン unittest。

GPU もネイティブ拡張も使わない: torch は sys.modules に偽モジュールを
注入して差し替え、実機事実は JSON スナップショットまたは偽 torch の
デバイスプロパティで模倣する。

実行: python3 -m unittest discover -s rocm_tools/tests -p test_r9700_preflight.py -v
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest

_ROCM_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROCM_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _ROCM_TOOLS_DIR)

import r9700_preflight as preflight

R9700_LDS = 65536


def make_snapshot(
    arch="gfx1201:sramecc+:xnack-",
    smem=R9700_LDS,
    index=0,
    name="Radeon 9070",
    total_memory=16 * 1024 * 1024 * 1024,
    hip="6.4.43481-151a884a",
):
    """検証を通過する gfx1201 スナップショット辞書を返す。"""
    return {
        "schema": preflight.SNAPSHOT_SCHEMA,
        "torch_version": "2.7.1+rocm6.4",
        "hip_version": hip,
        "device": {
            "index": index,
            "name": name,
            "gcn_arch_name": arch,
            "total_memory_bytes": total_memory,
            "shared_memory_per_block": smem,
        },
    }


def make_fake_torch(devices, hip="6.4.43481-151a884a", available=True,
                    version="2.7.1+rocm6.4"):
    """torch.cuda プロパティ照会だけを模倣した偽 torch モジュールを作る。"""
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = version
    torch_mod.version = types.SimpleNamespace(hip=hip)
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: len(devices),
        get_device_properties=lambda i: devices[i],
    )
    return torch_mod


def make_props(name="Radeon 9070", gcn="gfx1201:sramecc+:xnack-",
               total_memory=16 * 1024 * 1024 * 1024, smem=R9700_LDS):
    return types.SimpleNamespace(
        name=name,
        gcnArchName=gcn,
        total_memory=total_memory,
        shared_memory_per_block=smem,
    )


class PreflightTestCase(unittest.TestCase):
    """os.environ を隔離し、torch の注入を後始末する基底クラス。"""

    def setUp(self):
        self._saved_environ = dict(os.environ)
        os.environ.clear()
        self._saved_torch = sys.modules.get("torch")

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_environ)
        if self._saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._saved_torch

    def install_fake_torch(self, devices, **kwargs):
        sys.modules["torch"] = make_fake_torch(devices, **kwargs)

    def run_cli(self, argv):
        """main(argv) を実行し (exit_code, stdout, stderr) を返す。"""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = preflight.main(argv)
        return code, out.getvalue(), err.getvalue()

    def write_snapshot(self, obj):
        """スナップショット辞書を一時 JSON に書き、パスを返す (後始末付き)。"""
        fd, path = tempfile.mkstemp(suffix=".json", text=True)
        self.addCleanup(os.unlink, path)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)
        return path


class TestEnvOnSemantics(PreflightTestCase):
    """_env_on は exllamav3/rocm_py の意味論 (strip 後 ""/"0"/"false"/"False"
    以外なら ON) と一致しなければならない。"""

    def check(self, raw, expected):
        os.environ[preflight.FUSED_MOE_ENV] = raw
        self.assertEqual(preflight._env_on(preflight.FUSED_MOE_ENV, False), expected,
                         msg="raw=%r" % raw)

    def test_off_values(self):
        for raw in ("", "0", "false", "False", "  false  ", " 0 "):
            self.check(raw, False)

    def test_on_values_including_confusable_strings(self):
        # "off" / "no" / "FALSE" は ON になる (exllamav3 と同じ罠)。
        for raw in ("1", "true", "TRUE", "off", "no", "FALSE", " yes "):
            self.check(raw, True)

    def test_unset_uses_default(self):
        self.assertFalse(preflight._env_on(preflight.FUSED_MOE_ENV, False))
        self.assertTrue(preflight._env_on(preflight.FUSED_MOE_ENV, True))


class TestNormalizeAndParseArch(PreflightTestCase):
    def test_feature_suffix_stripped(self):
        self.assertEqual(preflight.normalize_arch("gfx1201:sramecc+:xnack-"), "gfx1201")
        self.assertEqual(preflight.normalize_arch("gfx1201"), "gfx1201")
        self.assertEqual(preflight.normalize_arch(" GFX1201:xnack+ "), "gfx1201")
        self.assertEqual(preflight.normalize_arch(None), "")

    def test_build_env_comma_and_space_formats(self):
        self.assertEqual(
            preflight.parse_build_archs("gfx1201,gfx1100"), ["gfx1201", "gfx1100"])
        self.assertEqual(
            preflight.parse_build_archs("gfx1201 gfx1100"), ["gfx1201", "gfx1100"])
        self.assertEqual(
            preflight.parse_build_archs(" gfx1201:sramecc+:xnack- , gfx1151 "),
            ["gfx1201", "gfx1151"])
        self.assertEqual(preflight.parse_build_archs(""), [])
        self.assertEqual(preflight.parse_build_archs(None), [])


class TestCollectEnv(PreflightTestCase):
    def test_only_relevant_vars_recorded(self):
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1201"
        os.environ[preflight.FUSED_MOE_ENV] = "off"  # exllamav3 では ON になる罠
        os.environ["MY_DB_PASSWORD"] = "hunter2"
        env = preflight.collect_env()
        self.assertEqual(env["PYTORCH_ROCM_ARCH"], "gfx1201")
        self.assertIsNone(env["GPU_ARCHS"])
        self.assertEqual(env[preflight.FUSED_MOE_ENV], {"raw": "off", "on": True})
        # 無関係な環境変数 / 認証情報は決して記録しない。
        self.assertNotIn("MY_DB_PASSWORD", env)
        self.assertNotIn("hunter2", json.dumps(env, ensure_ascii=False))


class TestAnalyzeR9700(PreflightTestCase):
    def facts(self, **kwargs):
        return preflight.validate_snapshot(make_snapshot(**kwargs))

    def test_valid_r9700_comparison_workarounds_required(self):
        report = preflight.analyze(self.facts(), "snapshot:test", preflight.collect_env())
        self.assertEqual(report["normalized_arch"], "gfx1201")
        self.assertEqual(report["status"], "comparison_workarounds_required")
        self.assertEqual(report["exit_code"], preflight.EXIT_OK)
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("r9700_detected", codes)
        self.assertIn("lds_ok_r9700", codes)
        # 環境変数はビルド意図にすぎない: バイナリターゲットは常に unknown。
        self.assertEqual(report["binary_target"]["status"], "unknown")

    def test_next_steps_grounding(self):
        report = preflight.analyze(self.facts(), "snapshot:test", preflight.collect_env())
        joined = "\n".join(report["next_steps"])
        self.assertIn("PYTORCH_ROCM_ARCH=gfx1201", joined)
        self.assertIn("65536", joined)
        self.assertIn("doc/r9700_vs_v620.md", joined)
        # 存在しない主張をしてはいけない: WMMA 有効化も V620 適用も否定する。
        self.assertIn("設定しないでください", joined)
        self.assertIn("適用しないでください", joined)

    def test_lds_mismatch_blocks(self):
        report = preflight.analyze(
            self.facts(smem=92160), "snapshot:test", preflight.collect_env())
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["exit_code"], preflight.EXIT_BLOCKED)
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("lds_mismatch_r9700", codes)

    def test_lds_unknown_is_warning_not_blocker(self):
        report = preflight.analyze(
            self.facts(smem=None), "snapshot:test", preflight.collect_env())
        self.assertEqual(report["status"], "comparison_workarounds_required")
        self.assertEqual(report["exit_code"], preflight.EXIT_OK)
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("lds_unknown", codes)

    def test_unsafe_fused_moe_forcing_blocks(self):
        # "off" は exllamav3 の _env_on では ON → 安全側の方向付け無効化 → blocked。
        os.environ[preflight.FUSED_MOE_ENV] = "off"
        report = preflight.analyze(self.facts(), "snapshot:test", preflight.collect_env())
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["exit_code"], preflight.EXIT_BLOCKED)
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("fused_moe_forced", codes)

    def test_fused_moe_false_string_blocks(self):
        os.environ[preflight.FUSED_MOE_ENV] = "FALSE"
        report = preflight.analyze(self.facts(), "snapshot:test", preflight.collect_env())
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["exit_code"], preflight.EXIT_BLOCKED)

    def test_fused_moe_off_semantic_value_ok(self):
        os.environ[preflight.FUSED_MOE_ENV] = "false"
        report = preflight.analyze(self.facts(), "snapshot:test", preflight.collect_env())
        self.assertEqual(report["status"], "comparison_workarounds_required")
        self.assertEqual(report["exit_code"], preflight.EXIT_OK)

    def test_fused_moe_forcing_on_non_rdna4_is_warning(self):
        os.environ[preflight.FUSED_MOE_ENV] = "1"
        facts = preflight.validate_snapshot(make_snapshot(arch="gfx1100"))
        report = preflight.analyze(facts, "snapshot:test", preflight.collect_env())
        self.assertEqual(report["status"], "not_r9700")
        self.assertEqual(report["exit_code"], preflight.EXIT_OK)
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("fused_moe_forced_non_rdna4", codes)
        self.assertNotIn("fused_moe_forced", codes)

    def test_build_env_mismatch_warns(self):
        os.environ["GPU_ARCHS"] = "gfx1100,gfx1151"
        report = preflight.analyze(self.facts(), "snapshot:test", preflight.collect_env())
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("build_env_mismatch", codes)
        self.assertEqual(report["status"], "comparison_workarounds_required")
        self.assertEqual(report["build_targets"]["GPU_ARCHS"]["archs"],
                         ["gfx1100", "gfx1151"])

    def test_build_env_match_no_warning(self):
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1201:sramecc+:xnack- gfx1100"
        report = preflight.analyze(self.facts(), "snapshot:test", preflight.collect_env())
        codes = [f["code"] for f in report["findings"]]
        self.assertNotIn("build_env_mismatch", codes)
        self.assertEqual(report["build_targets"]["PYTORCH_ROCM_ARCH"]["archs"],
                         ["gfx1201", "gfx1100"])


class TestAnalyzeOtherArchs(PreflightTestCase):
    def test_non_r9700_supported_device(self):
        facts = preflight.validate_snapshot(
            make_snapshot(arch="gfx1100", smem=92160, name="Radeon RX 7900 XTX"))
        report = preflight.analyze(facts, "snapshot:test", preflight.collect_env())
        self.assertEqual(report["status"], "not_r9700")
        self.assertEqual(report["exit_code"], preflight.EXIT_OK)

    def test_non_r9700_lds_mismatch_is_warning_only(self):
        facts = preflight.validate_snapshot(
            make_snapshot(arch="gfx1100", smem=65536))
        report = preflight.analyze(facts, "snapshot:test", preflight.collect_env())
        self.assertEqual(report["status"], "not_r9700")
        self.assertEqual(report["exit_code"], preflight.EXIT_OK)
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("lds_unexpected", codes)

    def test_unsupported_arch_blocks(self):
        facts = preflight.validate_snapshot(
            make_snapshot(arch="gfx900", smem=65536, name="Radeon VII"))
        report = preflight.analyze(facts, "snapshot:test", preflight.collect_env())
        self.assertEqual(report["status"], "unsupported_arch")
        self.assertEqual(report["exit_code"], preflight.EXIT_BLOCKED)


class TestSnapshotValidation(PreflightTestCase):
    def test_valid_snapshot_roundtrip(self):
        facts = preflight.validate_snapshot(make_snapshot())
        self.assertEqual(facts["device"]["index"], 0)
        self.assertEqual(facts["device"]["shared_memory_per_block"], R9700_LDS)

    def test_extra_keys_allowed(self):
        obj = make_snapshot()
        obj["extra_future_field"] = {"whatever": 1}
        facts = preflight.validate_snapshot(obj)
        self.assertEqual(facts["schema"], preflight.SNAPSHOT_SCHEMA)

    def test_wrong_schema_rejected(self):
        obj = make_snapshot()
        obj["schema"] = "some-other-schema/v9"
        with self.assertRaises(preflight.PreflightError):
            preflight.validate_snapshot(obj)

    def test_missing_required_keys_rejected(self):
        for key in ("torch_version", "device"):
            obj = make_snapshot()
            del obj[key]
            with self.assertRaises(preflight.PreflightError):
                preflight.validate_snapshot(obj)

    def test_bad_field_types_rejected(self):
        cases = [
            lambda o: o.update(torch_version=""),
            lambda o: o.update(hip_version="   "),
            lambda o: o["device"].update(index=-1),
            lambda o: o["device"].update(index=True),
            lambda o: o["device"].update(name="  "),
            lambda o: o["device"].update(gcn_arch_name=""),
            lambda o: o["device"].update(total_memory_bytes=0),
            lambda o: o["device"].update(shared_memory_per_block=-5),
        ]
        for mutate in cases:
            obj = make_snapshot()
            mutate(obj)
            with self.assertRaises(preflight.PreflightError, msg=str(cases.index(mutate))):
                preflight.validate_snapshot(obj)

    def test_load_snapshot_bad_json(self):
        fd, path = tempfile.mkstemp(suffix=".json", text=True)
        self.addCleanup(os.unlink, path)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        with self.assertRaises(preflight.PreflightError):
            preflight.load_snapshot(path)

    def test_load_snapshot_missing_file(self):
        with self.assertRaises(preflight.PreflightError):
            preflight.load_snapshot("/nonexistent/r9700-snapshot.json")


class TestLiveDeviceFacts(PreflightTestCase):
    def test_collect_from_fake_torch(self):
        self.install_fake_torch([make_props()])
        facts = preflight.collect_device_facts(0)
        self.assertEqual(facts["schema"], preflight.SNAPSHOT_SCHEMA)
        self.assertEqual(facts["torch_version"], "2.7.1+rocm6.4")
        self.assertEqual(facts["device"]["gcn_arch_name"], "gfx1201:sramecc+:xnack-")
        self.assertEqual(facts["device"]["shared_memory_per_block"], R9700_LDS)

    def test_missing_torch_import(self):
        def boom():
            raise ImportError("No module named 'torch'")
        orig = preflight.load_torch
        preflight.load_torch = boom
        self.addCleanup(setattr, preflight, "load_torch", orig)
        with self.assertRaises(preflight.PreflightError) as ctx:
            preflight.collect_device_facts(0)
        self.assertIn("torch", str(ctx.exception))

    def test_missing_hip_version(self):
        self.install_fake_torch([make_props()], hip=None)
        with self.assertRaises(preflight.PreflightError) as ctx:
            preflight.collect_device_facts(0)
        self.assertIn("torch.version.hip", str(ctx.exception))

    def test_cuda_unavailable(self):
        self.install_fake_torch([make_props()], available=False)
        with self.assertRaises(preflight.PreflightError):
            preflight.collect_device_facts(0)

    def test_invalid_device_index(self):
        self.install_fake_torch([make_props(), make_props(name="Second")])
        with self.assertRaises(preflight.PreflightError) as ctx:
            preflight.collect_device_facts(5)
        self.assertIn("範囲外", str(ctx.exception))


class TestCliExitSemantics(PreflightTestCase):
    def test_text_mode_valid_snapshot_exit_0(self):
        path = self.write_snapshot(make_snapshot())
        code, out, err = self.run_cli(["--snapshot", path])
        self.assertEqual(code, 0)
        self.assertIn("comparison_workarounds_required", out)
        self.assertIn("gfx1201", out)
        self.assertIn("exit 0 は事実の収集", out)

    def test_json_mode_valid_snapshot(self):
        path = self.write_snapshot(make_snapshot())
        code, out, err = self.run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["schema"], preflight.REPORT_SCHEMA)
        self.assertEqual(report["status"], "comparison_workarounds_required")
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["normalized_arch"], "gfx1201")
        self.assertEqual(report["binary_target"]["status"], "unknown")
        self.assertEqual(report["source"], "snapshot:%s" % path)

    def test_json_mode_blocked_exit_2(self):
        os.environ[preflight.FUSED_MOE_ENV] = "off"
        path = self.write_snapshot(make_snapshot())
        code, out, err = self.run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 2)
        report = json.loads(out)
        self.assertEqual(report["status"], "blocked")
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("fused_moe_forced", codes)

    def test_text_mode_blocked_exit_2(self):
        path = self.write_snapshot(make_snapshot(smem=92160))
        code, out, err = self.run_cli(["--snapshot", path])
        self.assertEqual(code, 2)
        self.assertIn("blocked", out)

    def test_invalid_snapshot_exit_1_text(self):
        path = self.write_snapshot({"schema": "wrong", "torch_version": "x"})
        code, out, err = self.run_cli(["--snapshot", path])
        self.assertEqual(code, 1)
        self.assertIn("エラー", err)

    def test_invalid_snapshot_exit_1_json(self):
        path = self.write_snapshot({"schema": "wrong", "torch_version": "x"})
        code, out, err = self.run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 1)
        report = json.loads(out)
        self.assertEqual(report["status"], "error")
        self.assertIn("schema", report["message"])

    def test_device_snapshot_mismatch_exit_1(self):
        path = self.write_snapshot(make_snapshot(index=0))
        code, out, err = self.run_cli(["--snapshot", path, "--device", "1"])
        self.assertEqual(code, 1)
        self.assertIn("デバイス", err)

    def test_device_flag_matches_snapshot_index(self):
        path = self.write_snapshot(make_snapshot(index=2))
        code, out, err = self.run_cli(["--snapshot", path, "--device", "2", "--json"])
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["device"]["index"], 2)

    def test_negative_device_without_snapshot_exit_1(self):
        self.install_fake_torch([make_props()])
        code, out, err = self.run_cli(["--device", "-1"])
        self.assertEqual(code, 1)

    def test_live_device_json_report(self):
        self.install_fake_torch([make_props()])
        code, out, err = self.run_cli(["--json"])
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["source"], "device:0")
        self.assertEqual(report["status"], "comparison_workarounds_required")

    def test_missing_hip_cli_exit_1(self):
        self.install_fake_torch([make_props()], hip=None)
        code, out, err = self.run_cli([])
        self.assertEqual(code, 1)
        self.assertIn("torch.version.hip", err)

    def test_invalid_device_index_cli_exit_1(self):
        self.install_fake_torch([make_props()])
        code, out, err = self.run_cli(["--device", "3"])
        self.assertEqual(code, 1)

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_report_excludes_unrelated_env(self):
        os.environ["AWS_SECRET_ACCESS_KEY"] = "leak-me"
        path = self.write_snapshot(make_snapshot())
        code, out, err = self.run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, 0)
        self.assertNotIn("leak-me", out)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", out)


class TestReviewRegressions(PreflightTestCase):
    def test_snapshot_missing_hip_is_nonzero(self):
        path = self.write_snapshot(make_snapshot(hip=None))
        code, out, _ = self.run_cli(["--snapshot", path, "--json"])
        self.assertNotEqual(code, 0)
        self.assertIn("hip_missing", [x["code"] for x in json.loads(out)["findings"]])

    def test_effective_build_target_precedence_and_fallback(self):
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx1201"
        os.environ["GPU_ARCHS"] = "gfx1030"
        facts = preflight.validate_snapshot(make_snapshot())
        report = preflight.analyze(facts, "test", preflight.collect_env())
        self.assertNotIn("build_env_mismatch", [x["code"] for x in report["findings"]])
        self.assertFalse(report["build_targets"]["GPU_ARCHS"]["effective"])
        os.environ["PYTORCH_ROCM_ARCH"] = ""
        report = preflight.analyze(facts, "test", preflight.collect_env())
        self.assertIn("build_env_mismatch", [x["code"] for x in report["findings"]])

    def test_rdna4_other_device_keeps_inference_caveat(self):
        facts = preflight.validate_snapshot(make_snapshot(arch="gfx1200", smem=92160))
        report = preflight.analyze(facts, "test", preflight.collect_env())
        self.assertIn("rdna4_wmma_caveats", [x["code"] for x in report["findings"]])
        self.assertTrue(any("未検証" in x for x in report["next_steps"]))

    def test_non_utf8_snapshot_is_useful_json_error(self):
        fd, path = tempfile.mkstemp()
        self.addCleanup(os.unlink, path)
        with os.fdopen(fd, "wb") as f:
            f.write(b"\xff\xfe")
        code, out, _ = self.run_cli(["--snapshot", path, "--json"])
        self.assertEqual(code, preflight.EXIT_INVALID_INPUT)
        self.assertIn("UTF-8", json.loads(out)["message"])

    def test_runtime_query_failure_is_useful_json_error(self):
        for operation in ("is_available", "device_count", "get_device_properties"):
            self.install_fake_torch([make_props()])
            def fail(*args):
                raise RuntimeError("query failed")
            setattr(sys.modules["torch"].cuda, operation, fail)
            code, out, _ = self.run_cli(["--json"])
            self.assertEqual(code, preflight.EXIT_INVALID_INPUT)
            self.assertIn("query failed", json.loads(out)["message"])


if __name__ == "__main__":
    unittest.main()