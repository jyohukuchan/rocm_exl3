#!/usr/bin/env python3
"""CPU-only tests for rocm_tools/rdna2/qwen_mtp_run.py.

No torch, no exllamav3, no GPU: they prove frozen-prompt hashing (little-endian
int64), duplicate/cache-hit rejection, truncation detection, batch coverage and
timed-flag homogeneity, the multi-token delivery-rate formula, CLI defaults and
argument validation, and that the module keeps native imports lazy. The
--mtp-model (standalone independently quantized MTP head) path is covered at
three levels: safetensors-header/strong-fingerprint inspection of real temp-dir
files, mocked config-compatibility checks (fail clearly before GPU weight load;
quant bits differences must NOT fail), and fake-stack runs proving the default
path is unchanged, a custom directory feeds only the draft component, and an
incompatible directory aborts before any model construction or load.

Run from the repo root:
    python3 -m unittest rocm_tools.rdna2.tests.test_qwen_mtp_run_cpu
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2 import qwen_mtp_run as qmr


def entry(ids, sha=None, **meta):
    p = {"ids": [ids], "timed": True,
         "sha": sha if sha is not None else qmr.prompt_sha256(ids)}
    p.update(meta)
    return p


def safetensors_blob(keys, metadata=None, row_bytes=4):
    """Minimal well-formed .safetensors bytes: u64 header length + JSON header + data."""
    header, off = {}, 0
    for k in keys:
        header[k] = {"dtype": "F16", "shape": [2], "data_offsets": [off, off + row_bytes]}
        off += row_bytes
    if metadata:
        header["__metadata__"] = dict(metadata)
    blob = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(blob)) + blob + bytes(off)


class PromptValidationTests(unittest.TestCase):
    def test_accepts_sweep_shape_and_preserves_metadata(self):
        ids = [5, 1, 151643, 2**40]
        payload = {"prompts": [entry(ids, language="ja", repeat=3)]}
        out = qmr.validate_prompts(payload, 1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["ids"], ids)
        self.assertEqual(out[0]["language"], "ja")
        self.assertEqual(out[0]["repeat"], 3)
        self.assertTrue(out[0]["timed"])
        self.assertEqual(out[0]["sha"], qmr.prompt_sha256(ids))

    def test_sha_matches_tensor_little_endian_bytes(self):
        # Independent reference: exactly what torch.tensor([ids], long).numpy().tobytes() hashes.
        ids = [0, 7, 123456789]
        self.assertEqual(qmr.prompt_sha256(ids),
                         hashlib.sha256(b"".join(struct.pack("<q", v) for v in ids)).hexdigest())

    def test_truncation_rejected(self):
        ids = list(range(32))
        good = entry(ids)
        bad = dict(good, ids=[ids[:-1]])  # truncated row, stale sha from the artifact
        with self.assertRaisesRegex(ValueError, "sha mismatch"):
            qmr.validate_prompts({"prompts": [bad]}, 1)

    def test_duplicate_rejected_as_cache_hit(self):
        ids = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "duplicate|cache hit"):
            qmr.validate_prompts({"prompts": [entry(ids), entry(ids)]}, 1)

    def test_malformed_entries_rejected(self):
        cases = {
            "empty row": {"ids": [[]], "sha": "x"},
            "float id": {"ids": [[1.5]], "sha": "x"},
            "bool id": {"ids": [[True, False]], "sha": "x"},
            "string id": {"ids": [["7"]], "sha": "x"},
            "int64 overflow": {"ids": [[2**63]], "sha": "x"},
            "missing sha": {"ids": [[1, 2]]},
        }
        for label, p in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                qmr.validate_prompts({"prompts": [p]}, 1)

    def test_negative_ids_and_nonboolean_timed_are_rejected(self):
        for p in [entry([-1]), entry([1], timed="false")]:
            with self.assertRaises(ValueError):
                qmr.validate_prompts({"prompts": [p]}, 1)

    def test_ids_must_be_single_row(self):
        two_rows = {"prompts": [{"ids": [[1, 2], [3, 4]], "timed": True, "sha": "x"}]}
        with self.assertRaisesRegex(ValueError, "one row"):
            qmr.validate_prompts(two_rows, 1)

    def test_batch_coverage_rejected(self):
        prompts = [entry([i]) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "batch-size"):
            qmr.validate_prompts({"prompts": prompts}, 2)
        qmr.validate_prompts({"prompts": prompts + [entry([99])]}, 2)  # 4 into 2 groups: fine

    def test_timed_flags_homogeneous_per_group(self):
        warm = entry([1], timed=False)
        cold = entry([2], timed=True)
        mixed = {"prompts": [warm, cold, entry([3], timed=False), entry([4], timed=False)]}
        with self.assertRaisesRegex(ValueError, "mixed timed"):
            qmr.validate_prompts(mixed, 2)
        # Homogeneous groups, even when timed alternates ACROSS groups: accepted.
        qmr.validate_prompts({"prompts": [warm, entry([5], timed=False), cold, entry([6], timed=True)]}, 2)

    def test_empty_or_wrong_container_rejected(self):
        for bad in ({}, {"prompts": []}, {"prompts": "nope"}):
            with self.assertRaises(ValueError):
                qmr.validate_prompts(bad, 1)


class DeliveryRateTests(unittest.TestCase):
    def test_multitoken_first_iterate_not_amortized(self):
        # MTP burst: first iterate delivers 3 cumulative tokens, last delivers 256.
        events = [[0.05, 3], [0.4, 40], [1.05, 256]]
        rate = qmr.delivery_rate(events)
        self.assertAlmostEqual(rate, (256 - 3) / (1.05 - 0.05))
        self.assertNotAlmostEqual(rate, (256 - 1) / 1.05)  # NOT the (N-1)/duration formula

    def test_degenerate_inputs_none(self):
        self.assertIsNone(qmr.delivery_rate([]))
        self.assertIsNone(qmr.delivery_rate([[0.1, 4]]))
        self.assertIsNone(qmr.delivery_rate([[2.0, 5], [2.0, 9]]))  # zero time span

    def test_plain_stream_matches_cumulative_delta(self):
        self.assertAlmostEqual(qmr.delivery_rate([[1.0, 10], [3.0, 30]]), 10.0)


class CliTests(unittest.TestCase):
    REQUIRED = ["-m", "/models/m", "--prompts-json", "/tmp/p.json", "--mode", "mtp",
                "--power-socket", "/tmp/sock", "--output", "/tmp/out.json"]

    def test_defaults(self):
        a = qmr.build_parser().parse_args(self.REQUIRED)
        self.assertEqual((a.draft_tokens, a.batch_size, a.cache_tokens, a.max_chunk_size,
                          a.new_tokens), (4, 1, 8704, 2048, 256))
        self.assertEqual(a.use_per_device, [28, 28])
        self.assertFalse(a.dynamic_draft)
        self.assertAlmostEqual(a.draft_confidence, 0.4)
        self.assertFalse(a.validate_finite)

    def test_positive_ceiling_and_ar_dynamic_rejected(self):
        base = qmr.build_parser().parse_args(self.REQUIRED)
        for kwargs in ({"draft_tokens": 0}, {"batch_size": 0}, {"new_tokens": -5},
                       {"use_per_device": [28, 0]}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                qmr.validate_args(self._patched(base, **kwargs))
        ar = qmr.build_parser().parse_args(["-m", "/models/m", "--prompts-json", "/tmp/p.json",
                                            "--mode", "ar", "--dynamic-draft",
                                            "--power-socket", "/tmp/sock", "--output", "/tmp/out.json"])
        with self.assertRaisesRegex(ValueError, "dynamic-draft"):
            qmr.validate_args(ar)
        qmr.validate_args(qmr.build_parser().parse_args(self.REQUIRED))  # sane defaults pass

    @staticmethod
    def _patched(args, **kwargs):
        return argparse.Namespace(**(vars(args) | kwargs))

    def test_mtp_model_defaults_to_none_and_parses(self):
        a = qmr.build_parser().parse_args(self.REQUIRED)
        self.assertIsNone(a.mtp_model)
        qmr.validate_args(a)  # old behavior stays valid
        b = qmr.build_parser().parse_args(self.REQUIRED + ["--mtp-model", "/models/mtp_q3"])
        self.assertEqual(b.mtp_model, "/models/mtp_q3")
        qmr.validate_args(b)

    def test_mtp_model_blank_rejected(self):
        b = qmr.build_parser().parse_args(self.REQUIRED + ["--mtp-model", "  "])
        with self.assertRaisesRegex(ValueError, "mtp-model"):
            qmr.validate_args(b)

    def test_lazy_native_imports(self):
        # Module-level source must not import torch/exllamav3 (CPU import safety).
        tree = ast.parse(Path(qmr.__file__).read_text(encoding="utf-8"))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(n.name.split(".")[0] for n in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                top.add(node.module.split(".")[0])
        self.assertFalse(top & {"torch", "exllamav3", "exllamav3_ext"},
                         f"module-level native imports: {top}")


class MtpDirectoryTests(unittest.TestCase):
    """inspect_mtp_directory / read_safetensors_header: stdlib-only, real temp-dir files."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(self._td.name)

    def _mtp_dir(self, name="mtp_q3", with_config=True):
        d = self.root / name
        d.mkdir()
        if with_config:
            (d / "config.json").write_text(
                '{"architectures": ["Qwen4ExpForConditionalGeneration"]}', encoding="utf-8")
        return d

    def test_hashes_contents_and_collects_source_metadata(self):
        d = self._mtp_dir()
        blob = safetensors_blob(["mtp.fc_hidden.q_weight", "mtp.layers.0.mlp.gate.q_weight",
                                 "model.embed_tokens.weight"],
                                metadata={"converted_by": "util/convert_mtp.py", "mtp_bits": "3"})
        (d / "mtp.safetensors").write_bytes(blob)
        insp = qmr.inspect_mtp_directory(d)
        self.assertEqual(insp["directory"], str(d.resolve()))
        self.assertEqual(insp["mtp_tensor_count"], 2)
        self.assertEqual(insp["non_mtp_tensor_count"], 1)
        self.assertEqual(insp["total_bytes"], len(blob))
        self.assertTrue(insp["content_hashed"])
        self.assertEqual(insp["files"],
                         [{"name": "mtp.safetensors", "bytes": len(blob),
                           "sha256": hashlib.sha256(blob).hexdigest()}])
        self.assertEqual(insp["source_metadata"],
                         {"mtp.safetensors": {"converted_by": "util/convert_mtp.py", "mtp_bits": "3"}})
        self.assertEqual(insp["combined_sha256"], hashlib.sha256(
            f"mtp.safetensors|{len(blob)}|{insp['files'][0]['sha256']}".encode("utf-8")).hexdigest())
        self.assertIn("SHA256 over the full bytes", insp["coverage"])

    def test_files_sorted_by_name(self):
        d = self._mtp_dir("mtp_sorted")
        (d / "b.safetensors").write_bytes(safetensors_blob(["mtp.fc_hidden.weight"]))
        (d / "a.safetensors").write_bytes(safetensors_blob(["mtp.layers.0.mlp.gate.weight"]))
        insp = qmr.inspect_mtp_directory(d)
        self.assertEqual([f["name"] for f in insp["files"]], ["a.safetensors", "b.safetensors"])

    def test_missing_directory_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "directory not found"):
            qmr.inspect_mtp_directory(self.root / "absent")

    def test_no_safetensors_points_at_the_converter(self):
        d = self._mtp_dir("mtp_empty")
        with self.assertRaisesRegex(ValueError, r"no \*\.safetensors.*util/convert_mtp\.py"):
            qmr.inspect_mtp_directory(d)

    def test_directory_without_mtp_tensors_rejected(self):
        d = self._mtp_dir("not_mtp")
        (d / "model-00001.safetensors").write_bytes(
            safetensors_blob(["model.layers.0.mlp.gate_proj.weight", "lm_head.weight"]))
        with self.assertRaisesRegex(ValueError, "not.*exported.*MTP head"):
            qmr.inspect_mtp_directory(d)

    def test_malformed_safetensors_header_rejected(self):
        for label, raw in (("garbage", b"junkjunkjunkjunk"), ("truncated header", b"\x10\x00" + b"{")):
            d = self._mtp_dir(f"bad_{label.split()[0]}")
            (d / "mtp.safetensors").write_bytes(raw)
            with self.subTest(label), self.assertRaisesRegex(ValueError, r"mtp\.safetensors"):
                qmr.inspect_mtp_directory(d)

    def test_over_cap_degrades_to_listing_only(self):
        from unittest.mock import patch
        d = self._mtp_dir("mtp_big")
        blob = safetensors_blob(["mtp.fc_hidden.weight"])
        (d / "mtp.safetensors").write_bytes(blob)
        with patch.object(qmr, "MTP_STRONG_HASH_MAX_BYTES", len(blob) - 1):
            insp = qmr.inspect_mtp_directory(d)
        self.assertFalse(insp["content_hashed"])
        self.assertIsNone(insp["files"][0]["sha256"])
        self.assertEqual(insp["combined_sha256"], hashlib.sha256(
            f"mtp.safetensors|{len(blob)}|-".encode("utf-8")).hexdigest())
        self.assertIn("NOT rehashed", insp["coverage"])


def _compat_cfg(**over):
    base = dict(architecture="Qwen4ExpForConditionalGeneration", hidden_size=4096,
                vocab_size=248077, hc_mult=4, mtp_num_hidden_layers=1,
                model_classes={"text": object(), "mtp": object()})
    base.update(over)
    return types.SimpleNamespace(**base)


class MtpCompatibilityTests(unittest.TestCase):
    """check_mtp_config_compatibility: mocked configs, no exllamav3, no GPU."""

    def test_matching_passes_and_reports_compared_values(self):
        vals = qmr.check_mtp_config_compatibility(_compat_cfg(), _compat_cfg())
        self.assertEqual(vals["hidden_size"], {"target": 4096, "mtp": 4096})
        self.assertEqual(set(vals), set(qmr.MTP_COMPAT_FIELDS))

    def test_attachment_field_mismatches_fail_clearly(self):
        cases = {"architecture": "Qwen3_5MoeForConditionalGeneration",
                 "hidden_size": 2048,
                 "vocab_size": 151936,
                 "hc_mult": 2,
                 "mtp_num_hidden_layers": 2}
        for field, bad in cases.items():
            with self.subTest(field=field):
                with self.assertRaises(ValueError) as cm:
                    qmr.check_mtp_config_compatibility(_compat_cfg(), _compat_cfg(**{field: bad}))
                msg = str(cm.exception)
                self.assertIn("--mtp-model is not attachable", msg)
                self.assertIn(f"{field}: target=", msg)

    def test_zero_mtp_layers_and_unregistered_component_fail(self):
        with self.assertRaisesRegex(ValueError, "mtp_num_hidden_layers"):
            qmr.check_mtp_config_compatibility(_compat_cfg(), _compat_cfg(mtp_num_hidden_layers=0))
        with self.assertRaisesRegex(ValueError, "model_classes"):
            qmr.check_mtp_config_compatibility(
                _compat_cfg(), _compat_cfg(model_classes={"text": object()}))

    def test_missing_field_reported_as_none(self):
        mtp = _compat_cfg()
        del mtp.hc_mult
        with self.assertRaisesRegex(ValueError, "hc_mult"):
            qmr.check_mtp_config_compatibility(_compat_cfg(), mtp)

    def test_quantization_differences_are_NOT_compared(self):
        t = _compat_cfg(bits=3.05, head_bits=5, mtp_bits=3, qmap={"block": 3.05})
        m = _compat_cfg(bits=None, head_bits=None, mtp_bits=2, qmap={"block.mtp.fc": 2})
        self.assertEqual(qmr.check_mtp_config_compatibility(t, m)["mtp_num_hidden_layers"],
                         {"target": 1, "mtp": 1})


class _FakeTensor:
    def __init__(self, values): self._v = list(values)
    def reshape(self, *a): return self
    def tolist(self): return list(self._v)


class _FakeGenerator:
    HOLD_FIRST_BURST = False
    """Emulates the engine: prefill iterate, then decode iters where MTP drafting
    runs first and the engine streams ONE event per accepted token (bursty)."""
    BURST, N = 3, 9   # cumulative events 3/6/9 from 9 streaming items

    def __init__(self, **kw):
        self.model, self.tokenizer = kw["model"], kw["tokenizer"]
        self.draft_model, self.draft_cache = kw.get("draft_model"), kw.get("draft_cache")
        self.num_draft_tokens = kw.get("num_draft_tokens", 0)
        self.mtp_draft = self.draft_model is not None
        self.ngram_match_min = kw.get("ngram_match_min", 0)
        self.record_draft_stats = kw.get("record_draft_stats", False)
        self.dynamic_draft = kw.get("dynamic_draft_tokens", False) and self.num_draft_tokens > 0
        self.active_jobs, self._queued, self._prefilled = [], [], False

    def enqueue(self, job): self._queued.append(job)
    def num_remaining_jobs(self): return len(self._queued) + len(self.active_jobs)
    def iterate_draftmodel_mtp_gen(self, results): return results
    def iterate_gen(self, results): return results
    def on_queue_drained(self): return None

    def iterate_start_jobs(self, results):
        while self._queued: self.active_jobs.append(self._queued.pop(0))

    def iterate(self):
        self.iterate_start_jobs([])
        if not self._prefilled:
            self._prefilled = True
            for j in self.active_jobs: j.prefilled = True
            return [{"job": j, "stage": "prefill"} for j in self.active_jobs]
        (self.iterate_draftmodel_mtp_gen if self.draft_model else self.iterate_gen)([])
        items = getattr(self, "_held_items", [])
        self._held_items = []
        for j in list(self.active_jobs):
            for _ in range(min(self.BURST, j.max_new_tokens - j.new_tokens)):
                j.new_tokens += 1                     # one streaming event per token
                items.append({"job": j, "stage": "streaming", "text": "x",
                              "token_ids": _FakeTensor([7])})
            if j.new_tokens >= j.max_new_tokens:
                items[-1].update(eos=True, time_prefill=0.2, time_generate=2.0,
                                 new_tokens=j.max_new_tokens, prompt_tokens=j.prompt_len,
                                 cached_tokens=0, eos_reason="max-length",
                                 accepted_draft_tokens=6, rejected_draft_tokens=3)
                self.active_jobs.remove(j)
        if self.HOLD_FIRST_BURST and self.active_jobs and all(j.new_tokens == self.BURST for j in self.active_jobs):
            self._held_items = items
            return []
        if not self.num_remaining_jobs(): self.on_queue_drained()
        return items


class _FakeJob:
    def __init__(self, input_ids=None, max_new_tokens=0, min_new_tokens=0, sampler=None):
        self.new_tokens, self.max_new_tokens = 0, max_new_tokens
        self.min_new_tokens, self.prefilled = min_new_tokens, False
        self.prompt_len = len(input_ids._v)
        self.draft_stats = [{"accepted": 6, "rejected": 3}]
    def is_prefill_done(self): return self.prefilled


def _install_fake_stack(log):
    """sys.modules fakes for torch/exllamav3/exllamav3_ext/multi_gpu; power_policy stays REAL."""
    import types
    torch = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        device_count=lambda: 2,
        get_device_properties=lambda d: types.SimpleNamespace(gcnArchName=f"gfx1030:x{d}"),
        memory_allocated=lambda d: 111 * (d + 1), max_memory_allocated=lambda d: 222,
        is_available=lambda: True, synchronize=lambda d: log.append(("sync", d)))
    torch.cuda, torch.long, torch.Tensor = cuda, int, _FakeTensor
    torch.tensor = lambda data, dtype=None: _FakeTensor(data[0])
    torch.isfinite = lambda t: types.SimpleNamespace(all=lambda: types.SimpleNamespace(item=lambda: True))

    ext = types.ModuleType("exllamav3_ext")
    ext.__qmr_fake__ = True
    handle, ext.__file__ = tempfile.mkstemp(suffix=".so", prefix="qmr_fake_native_")
    with os.fdopen(handle, "w") as f:
        f.write("fake-native")

    exl = types.ModuleType("exllamav3"); smp = types.ModuleType("exllamav3.generator.sampler")
    genmod = types.ModuleType("exllamav3.generator")

    class FakeModel:
        def __init__(self, component="text"):
            self.component = component
            self.forward = lambda *a, **k: _FakeTensor([0.0])
        @classmethod
        def from_config(cls, cfg, component="text"):
            log.append(("from_config", component, cfg.infer_params.ngram_stream_from_disk, cfg.path))
            return cls(component)
        def load(self, **kw): log.append(("load", self.component, dict(kw)))
        def unload(self): log.append(("unload", self.component))
    class FakeConfig:
        # Test hook: str(directory) -> config fields to replace (mocks an incompatible
        # standalone MTP config.json without touching real exllamav3). Fresh dict per install.
        OVERRIDES = {}
        @staticmethod
        def from_directory(path):
            fields = dict(architecture="Qwen4ExpForConditionalGeneration", hidden_size=4096,
                          vocab_size=248077, hc_mult=4, mtp_num_hidden_layers=1,
                          model_classes={"text": FakeModel, "mtp": FakeModel})
            fields.update(FakeConfig.OVERRIDES.get(str(path), {}))
            return types.SimpleNamespace(path=str(path),
                                         infer_params=types.SimpleNamespace(ngram_stream_from_disk=True),
                                         **fields)
    class FakeCache:
        def __init__(self, model, max_num_tokens=0, max_batch_size=1, max_history=0):
            self.model, self.max_num_tokens = model, max_num_tokens
            self.num_slots, self.max_history = max_batch_size, max_history
    smp.ArgmaxSampler = lambda: types.SimpleNamespace(kind="argmax")
    exl.Model, exl.Config = FakeModel, FakeConfig
    exl.Cache, exl.Tokenizer = FakeCache, types.SimpleNamespace(from_config=lambda cfg: types.SimpleNamespace(actual_vocab_size=248077))
    exl.Generator, exl.Job = _FakeGenerator, _FakeJob
    genmod.sampler = smp; exl.generator = genmod

    mg = types.ModuleType("rocm_tools.rdna2.multi_gpu")
    mg.audit_placement = lambda model, idx: {"ok": True, "expected": list(idx)}
    mg.collect_ngram_state = lambda model, require_ram: {"ok": True, "require_ram": require_ram}

    import rocm_tools.rdna2 as pkg
    mods = {"torch": torch, "exllamav3": exl, "exllamav3.generator": genmod,
            "exllamav3.generator.sampler": smp, "exllamav3_ext": ext,
            "rocm_tools.rdna2.multi_gpu": mg}
    saved = {k: sys.modules.get(k) for k in mods}
    saved_attr = getattr(pkg, "multi_gpu", None)
    sys.modules.update(mods)
    pkg.multi_gpu = mg
    return saved, saved_attr, pkg


def _uninstall(saved, saved_attr, pkg):
    ext = sys.modules.get("exllamav3_ext")
    for k, v in saved.items():
        if v is None: sys.modules.pop(k, None)
        else: sys.modules[k] = v
    if getattr(ext, "__qmr_fake__", False):
        os.unlink(ext.__file__)
    if saved_attr is None: del pkg.multi_gpu
    else: pkg.multi_gpu = saved_attr


class FakeHelper(threading.Thread):
    """Protocol-compatible stand-in for the privileged power_switch_server.py."""
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = str(path)
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path); self.srv.listen(1)
    def run(self):
        while True:
            try: conn, _ = self.srv.accept()
            except OSError: return
            with conn, conn.makefile("rwb", buffering=0) as f:
                while True:
                    line = f.readline()
                    if not line: break
                    req = json.loads(line)
                    f.write((json.dumps({"mode": req["mode"], "label": req["label"],
                                         "after": [req["mode"], req["mode"]]}) + "\n").encode())
    def stop(self):
        try: self.srv.shutdown(socket.SHUT_RDWR)
        except OSError: pass
        self.srv.close(); self.join(timeout=1); os.unlink(self.path)


class RunIntegrationTests(unittest.TestCase):
    ARGS = ["-m", "/models/qwen38", "--cache-tokens", "1024", "--new-tokens", "9"]

    def _case(self, mode, sock, out, extra_args=(), mtp_config_overrides=None):
        log = []
        saved, saved_attr, pkg = _install_fake_stack(log)
        try:
            if mtp_config_overrides:
                sys.modules["exllamav3"].Config.OVERRIDES.update(mtp_config_overrides)
            args = qmr.build_parser().parse_args(
                self.ARGS + ["--mode", mode, "--power-socket", sock, "--output", out,
                             "--prompts-json", "/unused"] + list(extra_args))
            prompts = qmr.validate_prompts(
                {"prompts": [entry(list(range(1, 6)), language="code", repeat=1)]}, 1)
            return qmr.run(args, prompts), json.loads(Path(out).read_text()), log
        finally:
            _uninstall(saved, saved_attr, pkg)

    def _mtp_shard_dir(self, td, name="mtp_q3"):
        d = Path(td) / name
        d.mkdir()
        (d / "config.json").write_text('{"architectures": ["Qwen4ExpForConditionalGeneration"]}',
                                       encoding="utf-8")
        blob = safetensors_blob(["mtp.fc_hidden.q_weight", "mtp.layers.0.mlp.gate.q_weight"],
                                metadata={"converted_by": "util/convert_mtp.py", "mtp_bits": "3"})
        (d / "mtp.safetensors").write_bytes(blob)
        return d, blob

    def test_mtp_and_ar_happy_paths(self):
        with tempfile.TemporaryDirectory() as td:
            helper = FakeHelper(Path(td) / "power.sock"); helper.start()
            try:
                for mode, drafts in (("mtp", 4), ("ar", 0)):
                    with self.subTest(mode=mode):
                        out = str(Path(td) / f"{mode}.json")
                        code, rep, log = self._case(mode, str(Path(td) / "power.sock"), out)
                        self.assertEqual(code, 0, rep.get("error"))
                        self.assertTrue(rep["complete"])
                        # main built first, MTP second, ngram-from-disk off BEFORE either;
                        # MTP head resident on cuda:1 loaded before the target in BOTH modes.
                        # Default (--mtp-model omitted): both components build from the --model dir.
                        self.assertEqual([e for e in log if e[0] == "from_config"],
                                         [("from_config", "text", False, "/models/qwen38"),
                                          ("from_config", "mtp", False, "/models/qwen38")])
                        self.assertEqual(rep["mtp_model"],
                                         {"source": "target-checkpoint", "directory": "/models/qwen38"})
                        self.assertEqual([e for e in log if e[0] == "load"],
                                         [("load", "mtp", {"device": "cuda:1", "max_chunk_size": 2048,
                                                           "progressbar": False}),
                                          ("load", "text", {"use_per_device": [28, 28],
                                                             "max_chunk_size": 2048, "progressbar": False})])
                        self.assertEqual(len([e for e in log if e[0] == "sync"]), 6)  # 2 devices x 3 switches
                        self.assertEqual(rep["cache_capacity"]["main"]["max_history"], 4)
                        self.assertEqual(rep["cache_capacity"]["draft"]["max_num_tokens"],
                                         rep["cache_capacity"]["main"]["max_num_tokens"])
                        self.assertTrue(rep["placement"]["ok"] and rep["ngram"]["ok"])
                        row = rep["runs"][0]
                        self.assertEqual(row["mode"], mode)
                        # ONE delivery sample per progressing iterate, not per streaming event:
                        # 9 cumulative single-token events -> 3 samples 3/6/9.
                        self.assertEqual([t for _, t in row["delivery_events"]], [3, 6, 9])
                        self.assertEqual(row["tokens"], [7] * 9)
                        self.assertEqual(row["language"], "code")
                        self.assertEqual(row["ids_sha256"], qmr.prompt_sha256(list(range(1, 6))))
                        self.assertEqual((row["time_prefill"], row["prompt_tokens"],
                                          row["cached_tokens"]), (0.2, 5, 0))
                        self.assertAlmostEqual(row["legacy_engine_decode_tps"], 8 / 2.0)
                        ev = row["delivery_events"]
                        self.assertAlmostEqual(row["observed_decode_tps"],
                                               (ev[-1][1] - ev[0][1]) / (ev[-1][0] - ev[0][0]))
                        self.assertEqual(rep["generator"],
                                         {"mode": mode, "mtp_draft": mode == "mtp",
                                          "num_draft_tokens": drafts, "dynamic_draft": False,
                                          "ngram_match_min": 0, "record_draft_stats": True})
                        if mode == "mtp":
                            self.assertEqual((row["accepted_draft_tokens"],
                                              row["rejected_draft_tokens"], row["draft_stats"]),
                                             (6, 3, [{"accepted": 6, "rejected": 3}]))
                        modes = [m for m, _, _, _ in rep["power_policy"]["transitions"]]
                        self.assertEqual(modes, ["auto", "profile_peak", "auto"])  # prefill/decode/drain
                        self.assertEqual(rep["groups"][0]["total_new_tokens"], 9)
            finally:
                helper.stop()

    def test_custom_mtp_directory_feeds_draft_only_and_is_fingerprinted(self):
        with tempfile.TemporaryDirectory() as td:
            helper = FakeHelper(Path(td) / "power.sock"); helper.start()
            mtp, blob = self._mtp_shard_dir(td)
            out = str(Path(td) / "custom.json")
            try:
                code, rep, log = self._case("mtp", str(Path(td) / "power.sock"), out,
                                            extra_args=["--mtp-model", str(mtp)])
                self.assertEqual(code, 0, rep.get("error"))
                self.assertTrue(rep["complete"])
                # Only the draft component is built from the standalone directory; the target
                # (and hence embed/lm_head/tokenizer/Engram) still come from --model.
                self.assertEqual([e for e in log if e[0] == "from_config"],
                                 [("from_config", "text", False, "/models/qwen38"),
                                  ("from_config", "mtp", False, str(mtp.resolve()))])
                self.assertEqual([e for e in log if e[0] == "load"],
                                 [("load", "mtp", {"device": "cuda:1", "max_chunk_size": 2048,
                                                   "progressbar": False}),
                                  ("load", "text", {"use_per_device": [28, 28],
                                                    "max_chunk_size": 2048, "progressbar": False})])
                blk = rep["mtp_model"]
                self.assertEqual(blk["source"], "custom-directory")
                self.assertEqual(blk["directory"], str(mtp.resolve()))
                self.assertEqual(blk["model_fingerprint"]["name"], "mtp_q3")
                self.assertTrue(blk["model_fingerprint"]["exists"])
                wf = blk["weights_fingerprint"]
                self.assertTrue(wf["content_hashed"])
                self.assertEqual(wf["mtp_tensor_count"], 2)
                self.assertEqual(wf["files"], [{"name": "mtp.safetensors", "bytes": len(blob),
                                                "sha256": hashlib.sha256(blob).hexdigest()}])
                self.assertEqual(blk["source_metadata"],
                                 {"mtp.safetensors": {"converted_by": "util/convert_mtp.py",
                                                      "mtp_bits": "3"}})
                self.assertEqual(blk["config_compatibility"]["hidden_size"],
                                 {"target": 4096, "mtp": 4096})
                # The target fingerprint keeps its original meaning: --model and nothing else.
                self.assertEqual(rep["model_fingerprint"]["dir"], "/models/qwen38")
                self.assertEqual(rep["cli"]["mtp_model"], str(mtp))
            finally:
                helper.stop()

    def test_incompatible_mtp_config_fails_before_construction_or_load(self):
        with tempfile.TemporaryDirectory() as td:
            mtp, _ = self._mtp_shard_dir(td)
            out = str(Path(td) / "bad.json")
            code, rep, log = self._case(
                "mtp", str(Path(td) / "power.sock"), out, extra_args=["--mtp-model", str(mtp)],
                mtp_config_overrides={str(mtp.resolve()): {"hidden_size": 2048}})
            self.assertEqual(code, 1)
            self.assertFalse(rep["complete"])
            self.assertIn("--mtp-model is not attachable", rep["error"])
            self.assertIn("hidden_size: target=4096 mtp-directory=2048", rep["error"])
            # Failed before any model construction and long before any GPU weight load.
            self.assertEqual([e for e in log if e[0] in ("from_config", "load")], [])
            self.assertEqual(rep["mtp_model"], {"source": "custom-directory",
                                                "directory": str(mtp.resolve())})

    def test_mtp_model_directory_without_shards_fails_clearly(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "no_shards"; d.mkdir()
            (d / "config.json").write_text("{}", encoding="utf-8")
            out = str(Path(td) / "noshard.json")
            code, rep, log = self._case("mtp", str(Path(td) / "power.sock"), out,
                                        extra_args=["--mtp-model", str(d)])
            self.assertEqual(code, 1)
            self.assertIn("util/convert_mtp.py", rep["error"])
            self.assertEqual([e for e in log if e[0] in ("from_config", "load")], [])

    def test_progress_recorded_when_text_emission_is_delayed(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            helper = FakeHelper(Path(td) / "power.sock"); helper.start()
            try:
                with patch.object(_FakeGenerator, "HOLD_FIRST_BURST", True):
                    code, rep, _ = self._case("mtp", str(Path(td) / "power.sock"), str(Path(td) / "out.json"))
                self.assertEqual(code, 0, rep.get("error"))
                self.assertEqual([n for _, n in rep["runs"][0]["delivery_events"]], [3, 6, 9])
                self.assertEqual(len(rep["runs"][0]["tokens"]), 9)
            finally:
                helper.stop()

    def test_missing_power_socket_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "out.json")
            code, rep, log = self._case("mtp", str(Path(td) / "absent.sock"), out)
            self.assertEqual(code, 1)
            self.assertFalse(rep["complete"])
            self.assertIn("power helper socket", rep["error"])
            self.assertIn("absent.sock", rep["error"])
            self.assertIn(("unload", "mtp"), log)      # both models still unloaded
            self.assertIn(("unload", "text"), log)
            self.assertEqual(rep["cleanup_errors"], [])
            self.assertTrue(Path(out).exists())       # JSON written even on failure


if __name__ == "__main__":
    unittest.main()
