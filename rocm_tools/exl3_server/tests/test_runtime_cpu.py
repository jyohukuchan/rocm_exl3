#!/usr/bin/env python3
"""CPU-only tests for the exl3_server runtime API (rocm_tools/exl3_server/runtime.py).

No torch, no exllamav3, no GPU, no os._exit, no power helper: pure-data
validation, the /props runtime_report over fake audits, the real power_policy
client against a fake Unix-socket helper, and the verified TP2+MTP load
sequence (draft unsharded on cuda:1 BEFORE the RCCL TP target, identical K5/V4
cache kwargs, max_history = draft window, Engram single locked RAM owner)
against the sys.modules fake engine stack. Fail-closed paths additionally
prove models are unloaded before RuntimeStartupError escapes.

Gate:
    python3 -m pytest -q -p no:cacheprovider rocm_tools/exl3_server/tests/test_runtime_cpu.py
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from rocm_tools.exl3_server import runtime  # noqa: E402
from rocm_tools.rdna2 import tp_run  # real CPU-safe harness (aggregation stays honest)  # noqa: E402


def args(**overrides):
    base = dict(
        model_dir="/models/example", tensor_parallel=False, mtp=False,
        draft_model_dir=None, num_draft_tokens=4, ngram_match_min=0,
        dynamic_draft=False, draft_confidence=0.4, ngram_ram=False,
        cache_quant="5,4", cache_compand_a=0.0, cache_size=8192,
        gpu_split=None, chunk_size=2048, autosplit_max_batch_size=1,
        tp_backend="native", tp_moe_tensor_split=False, context_limit=None,
        max_output_tokens=256, power_socket=None, layer_map=None,
        swa_full=False, load_verbose=False, override=None,
        moe_cpu_offload=0, moe_cpu_split=0, draft_moe_cpu_layers=0,
        tp_max_parallelism_attn=None, tp_max_parallelism_mlp=None,
        tp_max_parallelism_moe=None, tp_max_parallelism_linear=None,
        tp_max_parallelism_linear_attn=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


VERIFIED = dict(tensor_parallel=True, mtp=True, ngram_ram=True, gpu_split="28,28",
                cache_size=34816, cache_quant="5,4", num_draft_tokens=None)


def verified_args(**over):
    return args(**{**VERIFIED, **over})


# ---------------------------------------------------------------------------
# Import purity / no side effects / never autoeditOS or os._exit
# ---------------------------------------------------------------------------

def test_import_and_verified_settings_are_cpu_safe():
    st = runtime.settings_for(args(tensor_parallel=True, mtp=True, gpu_split="28,28"))
    assert st["path"] == "verified_tp_mtp"
    assert st["cache_quant"] == (5, 4)
    assert st["tp_backend_effective"] == "nccl"
    assert st["tp_output_device"] == "cuda:1"
    assert st["draft_device"] == "cuda:1"


def test_module_level_imports_stay_engine_free():
    src = Path(runtime.__file__).read_text(encoding="utf-8")
    top = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= {"__future__", "os", "resource", "types"}, top
    # Hard rules from the task: library code never exits the interpreter and
    # never mutates OS limits (validation reports requirements instead). The
    # docstrings MENTION those rules; only real calls are forbidden.
    import re
    assert not re.search(r"os\._exit\s*\(", src)
    assert not re.search(r"\.setrlimit\s*\(", src)


# ---------------------------------------------------------------------------
# parse / validate
# ---------------------------------------------------------------------------

def test_cache_and_context_validation_is_fail_closed():
    assert runtime.parse_cache_quant("5,4") == (5, 4)
    assert runtime.parse_cache_quant("8") == (8, 8)
    assert runtime.parse_cache_quant(None) is None
    with pytest.raises(ValueError):
        runtime.parse_cache_quant("1,4")
    with pytest.raises(ValueError):
        runtime.parse_cache_quant("4,4,4")
    assert runtime.aligned_context(8704, 8192) == 8192
    assert runtime.aligned_context(8704, 8500) == 8448      # page-rounded down
    assert runtime.aligned_context(34816, None) == 34816
    problems, _ = runtime.validate_runtime_args(args(context_limit=9000))
    assert any("exceeds -cs" in p for p in problems)


def test_validated_happy_path_and_generic_stay_clean(monkeypatch):
    monkeypatch.setenv(runtime.NGRAM_MLOCK_ENV, "1")
    problems, requirements = runtime.validate_runtime_args(verified_args())
    assert problems == [], problems
    assert any("RLIMIT_MEMLOCK" in r for r in requirements)
    assert any("power" in r.lower() for r in requirements)
    # Generic (non-TP) models are NOT forced into the verified KV/budget rules.
    problems, _ = runtime.validate_runtime_args(args(cache_quant=None, num_draft_tokens=None))
    assert problems == [], problems
    # Generic -tp (no -mtp) goes through model_init: KV stays the user's choice.
    problems, _ = runtime.validate_runtime_args(
        args(tensor_parallel=True, gpu_split="12,12", cache_quant=None, num_draft_tokens=None))
    assert problems == [], problems


def test_validation_problems_are_clear_and_never_edit_the_os(monkeypatch):
    monkeypatch.delenv(runtime.NGRAM_MLOCK_ENV, raising=False)
    problems, _ = runtime.validate_runtime_args(verified_args())
    assert any(runtime.NGRAM_MLOCK_ENV in p for p in problems)          # mlock env demanded
    problems, _ = runtime.validate_runtime_args(verified_args(gpu_split="28"))
    assert any("two" in p for p in problems)                            # two-device budget
    problems, _ = runtime.validate_runtime_args(verified_args(gpu_split="28,x"))
    assert any("-gs" in p for p in problems)                            # unparseable budget
    problems, _ = runtime.validate_runtime_args(verified_args(moe_cpu_split=2048))
    assert any("-mcs" in p for p in problems)                           # layer-split-only flag
    problems, _ = runtime.validate_runtime_args(verified_args(mtp=False, tensor_parallel=True,
                                                              ngram_ram=False, cache_quant=None,
                                                              gpu_split="28,28"))
    assert all("verified TP2+MTP requires" not in p for p in problems)  # KV enforced on verified only
    problems, _ = runtime.validate_runtime_args(verified_args(cache_quant="4,4"))
    assert any("requires -cq 5,4" in p for p in problems)
    problems, _ = runtime.validate_runtime_args(verified_args(ngram_match_min=2))
    assert any("ngram" in p for p in problems)
    problems, _ = runtime.validate_runtime_args(verified_args(draft_model_dir="/models/draft"))
    assert any("draft_model_dir" in p for p in problems)
    problems, _ = runtime.validate_runtime_args(verified_args(cache_size=8000))
    assert any("multiple of" in p for p in problems)
    problems, _ = runtime.validate_runtime_args(verified_args(override="spec.yaml"))
    assert any("override" in p for p in problems)
    # Validation must not touch resource.setrlimit.
    import resource as real_resource
    def _boom(*a, **k):
        raise AssertionError("validate_runtime_args must never change OS limits")
    monkeypatch.setattr(real_resource, "setrlimit", _boom, raising=False)
    runtime.validate_runtime_args(verified_args())


def test_power_socket_missing_is_a_requirement_not_a_problem(monkeypatch, tmp_path):
    monkeypatch.setenv(runtime.NGRAM_MLOCK_ENV, "1")
    missing = str(tmp_path / "helper.sock")
    problems, requirements = runtime.validate_runtime_args(verified_args(power_socket=missing))
    assert problems == []
    assert any("power-socket" in r and "does not exist" in r for r in requirements)


# ---------------------------------------------------------------------------
# runtime_report (pure CPU)
# ---------------------------------------------------------------------------

def test_runtime_report_exposes_actual_audit_without_engine_imports():
    audit = {
        "ok": True, "path": "verified_tp_mtp", "execution_actual": "tp2",
        "loaded_tp": True, "requested_backend": "nccl",
        "backend_actual": "TPBackendRCCL", "expected_devices": [0, 1],
        "output_device": 1, "ranks": [{"device": 0, "pid": 10, "gcnArchName": "gfx1030"}],
        "cache": {"requested": {"k_bits": 5, "v_bits": 4},
                  "observed": {"target": {"layers_total": 12}}, "notes": []},
        "mtp": {"unsharded": True, "device": "cuda:1"},
        "ngram_expected_keys": ["ple.ngram"],
        "ngram_ram_owners": {"ple.ngram": [{"device": 1, "pid": 11, "bytes": 10,
                                               "residency": {"all_resident": True}}]},
        "ngram_mlock": {"env_requested": True, "locked_total_bytes": 10,
                        "tables": [{"key": "ple.ngram", "device": 1, "pid": 11,
                                    "is_locked": True, "locked_bytes": 10}]},
        "problems": [], "notes": [],
    }
    report = runtime.runtime_report(audit)
    assert report["available"] is True
    assert report["ok"] is True
    assert report["execution"]["backend"]["actual"] == "TPBackendRCCL"
    assert report["cache_policy"]["observed"]["target"]["layers_total"] == 12
    assert report["ngram"]["single_ram_owner"]["ple.ngram"]["mlocked"] is True


def test_runtime_report_degrades_honestly():
    assert runtime.runtime_report({})["available"] is False
    assert runtime.runtime_report(None)["available"] is False
    minimal = runtime.runtime_report({"execution_actual": "single_device", "loaded_tp": False})
    assert minimal["available"] is True and minimal["ok"] is False
    assert minimal["devices"]["ranks"] == []
    assert minimal["ngram"]["single_ram_owner"] == {}
    assert minimal["cache_policy"]["observed"] == {}
    # A tableless (dense/MoE D/M) audit reports owners as empty, never fabricated.
    tl = runtime.runtime_report({"execution_actual": "tp2", "ok": True,
                                 "ngram_expected_keys": [], "ngram_ram_owners": {}})
    assert tl["ngram"]["expected_keys"] == [] and tl["ngram"]["single_ram_owner"] == {}


# ---------------------------------------------------------------------------
# mlock rank helper
# ---------------------------------------------------------------------------

def test_tp_mlock_rank_reports_lock_state(monkeypatch):
    monkeypatch.setenv(runtime.NGRAM_MLOCK_ENV, "1")
    lock = SimpleNamespace(is_locked=True, locked_bytes=4096, locked_ranges=[(1000, 4096)])
    ng = SimpleNamespace(key="ngram", mode="trellis_ram", _ram_lock=lock)
    plain = SimpleNamespace(key="other", mode="trellis_ram", _ram_lock=None)
    broken = SimpleNamespace(key="broken", mode="fp16_ram",
                             _ram_lock=SimpleNamespace(
                                 is_locked=True, locked_bytes=None, locked_ranges=[]))
    rec = runtime.tp_mlock_rank({"device": 0, "modules": [ng, plain, broken]})
    assert rec["device"] == 0 and rec["env_mlock"] == "1"
    by = {t["key"]: t for t in rec["tables"]}
    assert by["ngram"]["is_locked"] is True and by["ngram"]["locked_bytes"] == 4096
    assert by["other"]["is_locked"] is False and by["other"]["has_lock_object"] is False
    assert "lock_error" in by["broken"]                      # unreadable lock is recorded, not hidden


def test_parent_mlock_records_shape(monkeypatch):
    monkeypatch.setenv(runtime.NGRAM_MLOCK_ENV, "0")
    ng = SimpleNamespace(key="ngram", mode="fp16_ram",
                         _ram_lock=SimpleNamespace(is_locked=False, locked_bytes=0, locked_ranges=[]))
    rec = runtime.parent_mlock_records(SimpleNamespace(modules=[ng]))
    assert rec["device"] is None and rec["env_mlock"] == "0"
    assert rec["tables"] == [{"key": "ngram", "has_lock_object": True, "is_locked": False,
                              "locked_bytes": 0, "locked_ranges": 0}]


# ---------------------------------------------------------------------------
# TP power-sync facade
# ---------------------------------------------------------------------------

def test_tp_sync_facade_dispatches_only_owned_rank():
    calls = []

    class Model:
        active_devices = [0, 1]

        def tp_worker_dispatch_single(self, idx, fn, args):
            calls.append((idx, fn, args))
            return {"device": idx}

    facade = runtime.TPRankSyncTorch(Model())
    assert facade.cuda.synchronize(1) == {"device": 1}
    # the dispatch target is the REAL validated in-rank sync function, not a copy
    assert calls[0][0] == 1 and calls[0][1] is tp_run.tp_sync_rank and calls[0][2] == ()
    assert facade.syncs[-1]["worker"] == {"device": 1}
    with pytest.raises(runtime.PowerContextError):
        facade.cuda.synchronize(2)
    with pytest.raises(runtime.PowerContextError):
        facade.cuda.synchronize("cpu")
    # 'cuda:0' strings route like ints (power_policy passes device objects).
    facade.cuda.synchronize("cuda:0")
    assert calls[-1][0] == 0


def test_helper_flags_are_owned_by_runtime_module():
    import argparse
    parser = argparse.ArgumentParser()
    runtime.add_helper_flags(parser)
    parsed = parser.parse_args([])
    assert parsed.power_socket is None
    assert parsed.context_limit is None
    assert parsed.max_output_tokens is None


# ---------------------------------------------------------------------------
# PowerContext against the REAL power_policy client + fake helper socket
# ---------------------------------------------------------------------------

class FakeHelper(threading.Thread):
    """Protocol-compatible stand-in for the privileged power_switch_server.py."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = str(path)
        self.requests = []
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(1)

    def run(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            with conn, conn.makefile("rwb", buffering=0) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    req = json.loads(line)
                    self.requests.append((req["mode"], req["label"]))
                    f.write((json.dumps({"after": [req["mode"], req["mode"]]}) + "\n").encode())

    def modes(self):
        return [m for m, _ in self.requests]

    def stop(self):
        try:
            self.srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.srv.close()
        self.join(timeout=1)
        os.unlink(self.path)


class FakeSyncGen:
    """Duck-typed sync Generator exposing the methods power_policy hooks."""

    def __init__(self):
        self.draft_model = "mtp-head"
        self.mtp_draft = True
        self.dflash_draft = False
        self.ngram_match_min = 0
        self.active_jobs = []

    def iterate_start_jobs(self, *a, **k):
        return "started"

    def iterate_draftmodel_mtp_gen(self, *a, **k):
        return "mtp"

    def on_queue_drained(self, *a, **k):
        return "drained"


def _fake_torch(syncs):
    return SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d: syncs.append(d)))


def test_power_context_batch1_auto_and_hook_restore(tmp_path):
    helper = FakeHelper(tmp_path / "p.sock")
    helper.start()
    try:
        syncs = []
        gen = FakeSyncGen()
        before = dict(gen.__dict__)
        agen = SimpleNamespace(generator=gen)
        ctx = runtime.PowerContext(agen, batch_size=1, devices=[0, 1],
                                   socket_path=helper.path, torch_module=_fake_torch(syncs))
        assert ctx.target is gen                       # attaches to AsyncGenerator.generator
        with ctx:
            assert helper.modes() == ["auto"]          # attach-batch1: prefill policy
            assert "iterate_start_jobs" in gen.__dict__  # MTP draft+verify compute hook live
            assert "iterate_draftmodel_mtp_gen" in gen.__dict__
        assert helper.modes() == ["auto"]              # nothing switched while idle
        assert gen.__dict__ == before                  # hooks fully restored
        assert ctx.summary()["policy"] == "phase-switched"
        assert syncs == [0, 1]                         # every transition drained BOTH devices
    finally:
        helper.stop()


def test_power_context_batch_gt1_holds_peak_then_auto(tmp_path):
    helper = FakeHelper(tmp_path / "p.sock")
    helper.start()
    try:
        syncs = []
        gen = FakeSyncGen()
        agen = SimpleNamespace(generator=gen)
        ctx = runtime.PowerContext(agen, batch_size=2, devices=[0, 1],
                                   socket_path=helper.path, torch_module=_fake_torch(syncs))
        with ctx:
            assert helper.modes() == ["profile_peak"]
            assert "iterate_start_jobs" not in gen.__dict__   # no per-job toggles for batch>1
        assert helper.modes() == ["profile_peak", "auto"]     # exitauto
        assert ctx.summary()["policy"] == "held-profile-peak"
        assert syncs == [0, 1, 0, 1]
    finally:
        helper.stop()


def test_power_context_fails_closed_on_wrong_wrapper_or_bad_args(tmp_path):
    with pytest.raises(runtime.PowerContextError):
        runtime.PowerContext(SimpleNamespace(), batch_size=1, devices=[0, 1],
                             socket_path=str(tmp_path / "x.sock"))
    with pytest.raises(runtime.PowerContextError):
        runtime.PowerContext(SimpleNamespace(generator=FakeSyncGen()), batch_size=0,
                             devices=[0, 1], socket_path=str(tmp_path / "x.sock"))
    with pytest.raises(runtime.PowerContextError):
        runtime.PowerContext(SimpleNamespace(generator=FakeSyncGen()), batch_size=1,
                             devices=[1, 1], socket_path=str(tmp_path / "x.sock"))


def test_power_context_disabled_without_socket_is_recorded_noop():
    gen = FakeSyncGen()
    ctx = runtime.PowerContext(SimpleNamespace(generator=gen), batch_size=1,
                               devices=[0, 1], socket_path=None)
    assert ctx.enabled is False
    with ctx:
        pass
    assert ctx.summary()["policy"] == "disabled"
    assert "policy-controlled" in ctx.summary()["note"]


# ---------------------------------------------------------------------------
# Fake engine stack for load_runtime tests
# ---------------------------------------------------------------------------

class FakeQuantLayer:
    """Named CacheLayer_quant so tp_run's requested-vs-actual audit sees K5/V4."""

    def __init__(self, dev_index):
        self.device = f"cuda:{dev_index}"
        self.k_bits, self.v_bits = 5, 4
        t = SimpleNamespace(device=self.device, numel=lambda: 250,
                            element_size=lambda: 1, shape=[250])
        self.qk = self.qv = self.sk = self.sv = t

    @property
    def __class_name(self):
        return type(self).__name__


FakeQuantLayer.__name__ = "CacheLayer_quant"  # class NAME is the audit key


class FakeFp16Layer:
    def __init__(self, dev_index):
        self.device = f"cuda:{dev_index}"
        self.k_bits = self.v_bits = None
        t = SimpleNamespace(device=self.device, numel=lambda: 100,
                            element_size=lambda: 2, shape=[100])
        self.k = self.v = t


FakeFp16Layer.__name__ = "CacheLayer_fp16"

NG_REC = {"key": "ngram", "mode": "trellis_ram", "device": "cpu",
          "num_table_tensors": 1, "num_disk_handles": 0,
          "table_residence": ["cpu"], "table_ram_bytes": 4096,
          "residency": {"probe": "mincore", "pages_total": 4, "pages_resident": 4,
                        "pages_nonresident": 0, "all_resident": True, "error": None,
                        "tensors": [{"index": 0, "bytes": 4096, "all_resident": True,
                                     "error": None}]}}


def _rank_rec(dev, pid, ng_mode=None):
    # Each TP rank owns half of the target's twelve KV layers; the parent
    # aggregator combines rank geometry into the twelve-layer total.
    geom = {"layers_total": 6, "bytes_total": 6000,
            "cls_counts": {"CacheLayer_quant": 6},
            "bits_variants": {"CacheLayer_quant": [[5, 4]]},
            "sample": [{"device": f"cuda:{dev}", "device_index": dev,
                        "cls": "CacheLayer_quant", "k_bits": 5, "v_bits": 4,
                        "n_tensors": 4,
                        "tensors": {n: {"device": f"cuda:{dev}", "shape": [250],
                                        "dtype": "uint8", "bytes": 250}
                                    for n in ("qk", "qv", "sk", "sv")}}]}
    mod = {"key": "layers.0", "type": "Qwen4ExpDecoderLayer", "device": f"cuda:{dev}",
           "device_index": dev, "caps": {"kv_cache": True}, "transformer": True,
           "kv_cache_modules": 1, "recurrent_cache_modules": 0,
           "cache_layers": geom,
           "recurrent_layers": {"layers_total": 0, "bytes_total": None,
                                 "cls_counts": {}, "bits_variants": {}, "sample": []},
           "facts": {"stub": False, "num_q_heads": 16, "linear_count": 4,
                     "linear": [{"name": "q", "type": "LinearExl3", "in_features": 4096,
                                 "out_features": 2048, "qtype": "EXL3",
                                 "weight_present": True, "weight_attr": "weight"}]}}
    ngram = {"modules": []}
    if dev == 0:
        rec = dict(NG_REC)
        if ng_mode:
            rec = {**rec, "mode": ng_mode}
        ngram = {"modules": [rec]}
    return {"device": dev, "rank": dev, "world_size": 2, "active_devices": [0, 1],
            "output_device": 1, "pid": pid,
            "gcnArchName": "gfx1030:10300:512:sramecc+:xnack-", "device_name": "Radeon V620",
            "backend_class": "RCCLProcessGroup",
            "torch_allocated_bytes": 1000, "torch_peak_bytes": 2000, "torch_reserved_bytes": 3000,
            "proc_mem": {"vm_rss_kb": 100, "pss_kb": 90, "vm_swap_kb": 0,
                          "minflt": 1, "majflt": 0},
            "runtime_libraries": {}, "modules": [mod], "ngram": ngram}


def _mlock_rec(dev, pid, locked=True):
    tables = []
    if dev == 0:
        tables = [{"key": "ngram", "has_lock_object": True,
                   "is_locked": locked, "locked_bytes": 4096 if locked else 0,
                   "locked_ranges": 1 if locked else 0}]
    return {"device": dev, "pid": pid, "env_mlock": "1", "tables": tables}


def make_fake_stack(events, fail=None):
    """sys.modules fakes for torch/exllamav3/exllamav3(.cache)(.model_init).
    Child rank pid is fabricated (4242); the pseudo output rank must be OUR pid,
    which is what the real parent/child layout looks like to the aggregator."""
    child_pid = 4242
    my_pid = os.getpid()

    torch_mod = __import__("types").ModuleType("torch")
    torch_mod.cuda = SimpleNamespace(
        device_count=lambda: 2,
        get_device_properties=lambda d: SimpleNamespace(
            gcnArchName="gfx1030:10300:512:sramecc+:xnack-", name="Radeon V620"),
        memory_allocated=lambda d=0: 0, max_memory_allocated=lambda d=0: 0,
        memory_reserved=lambda d=0: 0, synchronize=lambda d: None,
        is_available=lambda: True)

    class FakeModelFactory:
        @staticmethod
        def from_config(cfg, swa_full=False, component="text"):
            events.append(("from_config", component, cfg.path, swa_full))
            return FakeTarget() if component == "text" else FakeDraft()

    class FakeNGramModule:
        key, mode = "ngram", "trellis_ram"
        modules = []

    class FakeTarget:
        caps = {"supports_tp": True}

        def __init__(self):
            self.modules = [FakeNGramModule()]     # pre-load tree exposes the Engram key
            self.loaded_tp = False
            self.active_devices = []
            self.output_device = None
            self.tp_backend = None
            self.mp_children = None

        def load(self, **kw):
            events.append(("target-load", dict(kw)))
            self.modules = []                      # the engine exports CPU-loaded modules
            if fail != "no_tp":                    # to the ranks; the parent keeps shells
                self.loaded_tp = True
                self.active_devices = [0, 1]
                self.tp_backend = "nccl"
                self.output_device = "cuda:0" if fail == "out0" else "cuda:1"

        def tp_worker_dispatch_wait_multi(self, devs, fn, args_):
            events.append(("dispatch", tuple(devs), getattr(fn, "__name__", "?")))
            if fn is runtime.tp_mlock_rank:
                return [_mlock_rec(d, child_pid if d == 0 else my_pid,
                                   locked=(fail != "unlocked")) for d in devs]
            if getattr(fn, "__name__", "") == "tp_audit_rank":
                return [_rank_rec(d, child_pid if d == 0 else my_pid,
                                  ng_mode="trellis_disk" if fail == "disk" else None)
                        for d in devs]
            raise RuntimeError("no such dispatch in the fake (cpu helper absent)")

        def unload(self):
            events.append("target-unload")

    class FakeDraft:
        caps = {"default_draft_size": 4, "mtp_draft": True}

        def __init__(self):
            layer = FakeQuantLayer(1)
            child = SimpleNamespace(key="mtp.layers.0", cache_layers=[layer],
                                    recurrent_layers=[], modules=[])
            self.modules = [SimpleNamespace(key="mtp.head", device="cuda:1", caps={},
                                            modules=[child])]

        def load(self, **kw):
            events.append(("draft-load", dict(kw)))

        def unload(self):
            events.append("draft-unload")

    class FakeConfig:
        @staticmethod
        def from_directory(path, layer_map=None):
            events.append(("config", str(path), layer_map))
            return SimpleNamespace(path=str(path),
                                   architecture="Qwen4ExpForConditionalGeneration",
                                   model_classes={"text": FakeModelFactory, "mtp": FakeModelFactory},
                                   infer_params=SimpleNamespace(ngram_stream_from_disk=True),
                                   stc=None)

    class FakeCache:
        def __init__(self, model, max_num_tokens=None, max_batch_size=1, max_history=None,
                     layer_type=None, k_bits=None, v_bits=None, compand_a=0.0):
            self.model, self.max_num_tokens = model, max_num_tokens
            self.layer_type, self.k_bits, self.v_bits = layer_type, k_bits, v_bits
            self.max_history, self.max_batch_size = max_history, max_batch_size
            self.num_slots = max_batch_size
            events.append(("cache", getattr(layer_type, "__name__", "?"), k_bits, v_bits,
                           max_history, max_num_tokens, max_batch_size))

    cache_mod = __import__("types").ModuleType("exllamav3.cache")
    cache_mod.CacheLayer_quant = FakeQuantLayer
    cache_mod.CacheLayer_fp16 = FakeFp16Layer

    exl = __import__("types").ModuleType("exllamav3")
    exl.Model, exl.Config, exl.Cache = FakeModelFactory, FakeConfig, FakeCache
    exl.Tokenizer = SimpleNamespace(from_config=lambda cfg: SimpleNamespace(name="tok"))
    exl.cache = cache_mod

    model_init_mod = __import__("types").ModuleType("exllamav3.model_init")

    def _init(a, **kw):
        events.append(("model_init.init", a.model_dir, a.tensor_parallel, a.mtp))
        model = FakeTarget()
        model.modules = [SimpleNamespace(key="embed", device="cuda:0", caps={},
                                         modules=[SimpleNamespace(key="layers.0",
                                                                   cache_layers=[FakeFp16Layer(0)],
                                                                   recurrent_layers=[], modules=[])])]
        cache = FakeCache(model, max_num_tokens=8192, layer_type=FakeFp16Layer)
        return model, SimpleNamespace(path=a.model_dir, eos_token_id_list=[],
                                      stc=None, architecture="x",
                                      model_classes={"text": object}), cache, exl.Tokenizer.from_config(
            None), None, None, None

    model_init_mod.init = _init
    exl.model_init = model_init_mod

    return {"torch": torch_mod, "exllamav3": exl, "exllamav3.cache": cache_mod,
            "exllamav3.model_init": model_init_mod}


@contextlib.contextmanager
def fake_engine_stack(events, fail=None):
    mods = make_fake_stack(events, fail=fail)
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ---------------------------------------------------------------------------
# Verified TP2 + MTP load (fake engine, real audits)
# ---------------------------------------------------------------------------

def test_verified_load_wires_the_benchmark_sequence(monkeypatch):
    monkeypatch.setenv(runtime.NGRAM_MLOCK_ENV, "1")
    events = []
    with fake_engine_stack(events):
        rt = runtime.load_runtime(verified_args())
    # config -> target shell -> (pre-load Engram scan) -> draft -> BOTH CACHES
    # (same quant kwargs) -> draft load -> target load -> rank dispatches.
    kinds = [e for e in events if e[0] != "dispatch"]
    assert [k[0] for k in kinds] == ["config", "from_config", "from_config",
                                      "cache", "cache", "draft-load", "target-load"]
    assert kinds[1][1:3] == ("text", "/models/example") and kinds[2][1:3] == ("mtp", "/models/example")
    draft_kw = kinds[5][1]
    assert draft_kw["device"] == "cuda:1" and draft_kw["max_chunk_size"] == 2048
    target_kw = kinds[6][1]
    assert target_kw["tensor_p"] is True and target_kw["tp_backend"] == "nccl"
    assert target_kw["tp_output_device"] == "cuda:1"
    assert target_kw["use_per_device"] == [28.0, 28.0]
    # SAME quant cache kwargs target/draft; max_history = draft window 4 (target side only).
    assert kinds[3][1:7] == ("CacheLayer_quant", 5, 4, 4, 34816, 1)
    assert kinds[4][1:7] == ("CacheLayer_quant", 5, 4, None, 34816, 1)
    # every rank audit + mlock dispatch ran through the real tp_run helpers
    assert ("dispatch", (0, 1), "tp_audit_rank") in events
    assert ("dispatch", (0, 1), "tp_mlock_rank") in events
    assert ("dispatch", (-1,), "tp_cpu_helper_meta") in events
    # ngram_stream_from_disk False BEFORE any construction consumed the config.
    assert rt.config.infer_params.ngram_stream_from_disk is False
    assert rt.settings["draft_window"] == 4 and rt.settings["batch_size"] == 1
    assert rt.settings["power_devices"] == [0, 1]
    assert rt.draft_model is not None and rt.draft_cache is not None
    report = rt.report()
    assert report["ok"] is True and report["problems"] == []
    assert report["execution"]["actual"] == "tp2"
    assert report["execution"]["loaded_tp"] is True
    assert report["execution"]["backend"]["requested"] == "nccl"
    assert report["execution"]["backend"]["rank_classes"] == {0: "RCCLProcessGroup",
                                                              1: "RCCLProcessGroup"}
    assert report["devices"]["expected"] == [0, 1] and report["devices"]["output"] == 1
    assert report["cache_policy"]["requested"]["k_bits"] == 5
    assert report["cache_policy"]["observed"]["target"]["cls_counts"] == {"CacheLayer_quant": 12}
    assert report["cache_policy"]["observed"]["draft"]["bits_variants"] == \
        {"CacheLayer_quant": [[5, 4]]}
    owner = report["ngram"]["single_ram_owner"]["ngram"]
    assert owner["device"] == 0 and owner["pid"] == 4242 and owner["bytes"] == 4096
    assert owner["mlocked"] is True and owner["all_pages_resident"] is True
    assert report["mtp_placement"] == {
        "component_present": True, "device": "cuda:1", "unsharded": True,
        "loaded_before_target": True, "same_cache_kwargs_as_target": True,
        "draft_depth": 4, "draft_window": 4,
        "target_max_history": 4, "num_draft_tokens_arg": None}
    assert rt.generator_kwargs["num_draft_tokens"] == 4
    # -ndt wins for the Generator draft length; max_history stays >= depth.
    events2 = []
    with fake_engine_stack(events2):
        rt2 = runtime.load_runtime(verified_args(num_draft_tokens=3))
    assert rt2.settings["draft_depth"] == 3 and rt2.settings["draft_window"] == 4
    assert rt2.generator_kwargs["num_draft_tokens"] == 3
    cache_ev2 = [e for e in events2 if e[0] == "cache"][0]
    assert cache_ev2[4] == 4                       # target max_history stays 4


def test_verified_load_fail_closed_cases_tear_down(monkeypatch):
    monkeypatch.setenv(runtime.NGRAM_MLOCK_ENV, "1")
    cases = {
        "no_tp": "loaded_tp",
        "out0": "output_device",
        "disk": "disk-offloaded",
        "unlocked": "no ACTIVE EXL3_NGRAM_MLOCK lock",
    }
    for fail, needle in cases.items():
        events = []
        with fake_engine_stack(events, fail=fail):
            with pytest.raises(runtime.RuntimeStartupError) as ei:
                runtime.load_runtime(verified_args())
        assert needle in str(ei.value), (fail, str(ei.value))
        # models were unloaded BEFORE the error escaped; no os._exit anywhere.
        assert "draft-unload" in events and "target-unload" in events, fail


def test_verified_load_rejects_missing_mlock_env_before_any_load():
    with pytest.MonkeyPatch.context() as mp:   # monkeypatch fixture not needed: env only
        mp.delenv(runtime.NGRAM_MLOCK_ENV, raising=False)
        events = []
        with fake_engine_stack(events):
            with pytest.raises(runtime.RuntimeStartupError) as ei:
                runtime.load_runtime(verified_args())
    assert runtime.NGRAM_MLOCK_ENV in str(ei.value)
    assert events == []                        # nothing was constructed


def test_verified_load_requires_ngr_when_checkpoint_exposes_engram(monkeypatch):
    monkeypatch.setenv(runtime.NGRAM_MLOCK_ENV, "1")
    events = []
    with fake_engine_stack(events):
        with pytest.raises(runtime.RuntimeStartupError) as ei:
            runtime.load_runtime(verified_args(ngram_ram=False))
    assert "Engram" in str(ei.value)
    # the rejection happens before the draft is even constructed; the target
    # shell that was built must still be torn down.
    assert "target-unload" in events and "draft-unload" not in events


# ---------------------------------------------------------------------------
# Generic path: model_init delegation
# ---------------------------------------------------------------------------

def test_generic_path_delegates_to_model_init(monkeypatch):
    events = []
    with fake_engine_stack(events):
        rt = runtime.load_runtime(args(cache_quant=None, num_draft_tokens=None,
                                       ngram_ram=False))
    assert ("model_init.init", "/models/example", False, False) in events
    assert rt.settings["path"] == "generic_model_init"
    assert rt.draft_model is None and rt.draft_cache is None
    report = rt.report()
    assert report["available"] is True and report["ok"] is True
    assert report["execution"]["actual"] == "single_device"
    assert report["cache_policy"]["requested"]["policy"] == "fp16"
    assert report["cache_policy"]["observed"]["target"]["cls_counts"] == {"CacheLayer_fp16": 1}
    assert report["mtp_placement"]["note"].startswith("generic model_init")
    assert rt.context_length == 8192
    assert rt.power_context(SimpleNamespace(generator=FakeSyncGen())).enabled is False


# ---------------------------------------------------------------------------
# Runtime lifecycle
# ---------------------------------------------------------------------------

class FakeModel:
    def __init__(self, events, loaded_tp=True):
        self.events, self.loaded_tp = events, loaded_tp
        self.mp_children = None

    def unload(self):
        self.events.append("model-unload")


class FakeDraft:
    def __init__(self, events):
        self.events = events

    def unload(self):
        self.events.append("draft-unload")


class FakeAsyncGenerator:
    def __init__(self, events):
        self.events = events

    async def close(self):
        self.events.append("generator-close")


class FakePower:
    def __init__(self, events):
        self.events = events

    def __exit__(self, *_):
        self.events.append("power-restore")


def test_shutdown_order_closes_generator_restores_power_then_unloads():
    events = []
    rt = runtime.Runtime(
        model=FakeModel(events), config=None, cache=None, tokenizer=None,
        draft_model=FakeDraft(events), draft_config=None, draft_cache=None,
        settings={"path": "generic_model_init", "context_limit": None,
                  "power_devices": [0], "power_socket": None, "batch_size": 1},
        audit={"execution_actual": "single_device", "ok": True})
    agen = FakeAsyncGenerator(events)
    rt.bind(agen)
    rt._power = FakePower(events)
    assert asyncio.run(rt.shutdown()) == []
    assert events == ["generator-close", "power-restore", "draft-unload", "model-unload"]
    # idempotent: a second teardown never double-unloads
    asyncio.run(rt.shutdown())
    assert events.count("model-unload") == 1


def test_runtime_exposes_generator_kwargs_for_server_wiring():
    rt = runtime.Runtime(
        model="target", config=None, cache="cache", tokenizer="tokenizer",
        draft_model="draft", draft_config=None, draft_cache="draft-cache",
        settings={"path": "verified_tp_mtp", "batch_size": 1, "chunk_size": 2048,
                  "num_draft_tokens": 4, "ngram_match_min": 0,
                  "dynamic_draft": True, "draft_confidence": 0.6,
                  "cpu_cache_size": 1.0, "recurrent_cache_size": 4.0},
        audit={"execution_actual": "tp2", "ok": True})
    kwargs = rt.generator_kwargs
    assert kwargs["draft_model"] == "draft"
    assert kwargs["num_draft_tokens"] == 4
    assert kwargs["dynamic_draft_tokens"] is True
    assert kwargs["record_draft_stats"] is True
    assert kwargs["cpu_cache_size"] == 1024**3


def test_runtime_bind_records_peak_active_jobs_and_restores_iterate_wrapper():
    class Sync:
        max_batch_size = 4

        def __init__(self):
            self.active_jobs = []

        def iterate(self):
            self.active_jobs[:] = [object(), object(), object(), object()]
            return []

    class Agen:
        def __init__(self, sync):
            self.generator = sync

        async def close(self):
            pass

    sync = Sync()
    original = sync.iterate
    agen = Agen(sync)
    rt = runtime.Runtime(
        model=FakeModel([]), config=None, cache=None, tokenizer=None,
        draft_model=None, draft_config=None, draft_cache=None,
        settings={"path": "generic_model_init", "context_limit": None,
                  "power_devices": [0], "power_socket": None, "batch_size": 4},
        audit={"execution_actual": "single_device", "ok": True})
    rt.bind(agen)
    sync.iterate()
    report = rt.report()
    assert report["generator_runtime"]["active_jobs"] == 4
    assert report["generator_runtime"]["peak_active_jobs"] == 4
    asyncio.run(rt.shutdown())
    assert sync.iterate == original


def test_power_context_wires_rank_facade_for_tp_runtime():
    events = []
    rt = runtime.Runtime(
        model=SimpleNamespace(active_devices=[0, 1], loaded_tp=True,
                              tp_worker_dispatch_single=lambda *a: {}),
        config=None, cache=None, tokenizer=None, draft_model=None, draft_config=None,
        draft_cache=None,
        settings={"path": "verified_tp_mtp", "batch_size": 1,
                  "power_devices": [0, 1], "power_socket": "/tmp/runtime-test.sock"},
        audit={"execution_actual": "tp2", "ok": True})
    ctx = rt.power_context(SimpleNamespace(generator=FakeSyncGen()))
    assert ctx.enabled is True
    assert isinstance(ctx.torch, runtime.TPRankSyncTorch)
    assert ctx.devices == [0, 1]
    assert ctx.batch == 1
