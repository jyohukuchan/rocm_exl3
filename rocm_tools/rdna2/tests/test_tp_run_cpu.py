#!/usr/bin/env python3
"""CPU-only tests for rocm_tools/rdna2/tp_run.py (the TP2-vs-LS evaluation CLI).

No torch, no exllamav3, no GPU. Frozen-prompt validation, id hashing and
burst-aware delivery math are SHARED objects with the reviewed qwen_mtp_run
helpers (asserted here, not re-duplicated); this file adds the TP-specific
surface: parser/validation, plan-summary and cache-capacity math, the worker
audit function driven against fake local_context dicts (picklable output,
actual device/plan/expert/ngram/geometry capture), the parent aggregator
(stub-exclusion, wrong-device, duplicate-RAM-table and disk-offload
rejection, tableless no-op, per-PID memory handled once, pseudo output rank
== parent PID), and TP-vs-LS load dispatch plus no-MTP model behavior on the
fake engine stack built by test_qwen_mtp_run_cpu (load kwargs, dispatch
sequence, finite-hook install/collect ordering, fail-closed audits, partial
init worker drain, cache-hit/short-output rejection, power-context restore).

Gate:
    python3 -m pytest -q -p no:cacheprovider rocm_tools/rdna2/tests/test_tp_run_cpu.py
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2 import multi_gpu as real_multi_gpu
from rocm_tools.rdna2 import qwen_mtp_run as qmr
from rocm_tools.rdna2 import tp_run as tr
from rocm_tools.rdna2.tests import test_qwen_mtp_run_cpu as tqmr


def entry(ids, sha=None, **meta):
    p = {"ids": [ids], "timed": True,
         "sha": sha if sha is not None else tr.prompt_sha256(ids)}
    p.update(meta)
    return p


# ---------------------------------------------------------------------------
# Fake tensors/modules standing in for what the worker audit functions touch.
# ---------------------------------------------------------------------------

class FT:
    """Fake tensor: metadata-only, plus a REAL page-backed anonymous mapping
    for CPU tensors so mincore residency can be measured for true (the touched
    page is resident because THIS fixture wrote it, never to fake a table:
    helpers under test must not fault pages in themselves)."""
    def __init__(self, values=(1, 2), device="cuda:0", dtype="torch.float16", esize=2):
        self._v = list(values)
        self.device = device
        self.shape = (len(self._v),)
        self.dtype = dtype
        self._esize = esize
        self._mm = self._buf = None
        self._addr = None
        if device == "cpu":
            import ctypes as _ct
            import mmap as _mmap
            self._mm = _mmap.mmap(-1, _mmap.PAGESIZE)
            self._mm.seek(0)
            self._mm.write(b"\1" * 8)               # fault the fixture page only
            self._buf = _ct.c_char.from_buffer(self._mm)
            self._addr = _ct.addressof(self._buf)

    def data_ptr(self):
        if self._addr is None:
            raise RuntimeError("no storage for a non-CPU fake tensor")
        return self._addr

    def numel(self):
        return len(self._v)

    def element_size(self):
        return self._esize

    def reshape(self, *a):
        return self

    def tolist(self):
        return list(self._v)


class FModule:
    def __init__(self, key, device, caps=None, children=(), cache_layers=(),
                 recurrent_layers=(), forward=None, **attrs):
        self.key = key
        self.device = device
        self.caps = caps or {}
        self.modules = list(children)
        self.cache_layers = list(cache_layers)
        self.recurrent_layers = list(recurrent_layers)
        if forward is not None:
            self.forward = forward
        for name, value in attrs.items():
            setattr(self, name, value)


class FCacheLayer:
    def __init__(self, k, v, device):
        self.k, self.v, self.device = k, v, device


class FState:
    def __init__(self, conv_state=None, recurrent_state=None, id_state=None, device=None):
        if conv_state is not None:
            self.conv_state = conv_state
        if recurrent_state is not None:
            self.recurrent_state = recurrent_state
        if id_state is not None:
            self.id_state = id_state
        self.device = device


class FNgram:
    """Duck-types NGramEmbedding for multi_gpu.describe/find."""
    def __init__(self, key, mode, tables=(), handles=(), device="cuda:1"):
        self.key, self.mode = key, mode
        self.tables, self.handles = list(tables), list(handles)
        self.device = device
        self.num_rows, self.rows_per_shard = 65536, 32768


@contextlib.contextmanager
def fake_worker_torch():
    """sys.modules['torch'] stand-in good enough for tp_audit_rank/hooks/sync."""
    torch = types.ModuleType("torch")

    class Props:
        gcnArchName = "gfx1030:sramecc+xnack-"
        name = "AMD Radeon V620"

    syncs = []
    torch.cuda = types.SimpleNamespace(
        get_device_properties=lambda d: Props(),
        memory_allocated=lambda d: 1000 * (d + 1),
        max_memory_allocated=lambda d: 2000 * (d + 1),
        memory_reserved=lambda d: 3000 * (d + 1),
        synchronize=lambda d: syncs.append(d))
    torch.syncs = syncs
    torch.Tensor = FT
    torch.isfinite = lambda t: types.SimpleNamespace(
        all=lambda: types.SimpleNamespace(
            item=lambda: all(math.isfinite(float(x)) for x in t._v)))
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch
    try:
        yield torch
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def audit_ctx(device, modules, plan=None, out=1):
    return {"device": device, "rank": device, "world_size": 2,
            "active_devices": [0, 1], "output_device": out, "modules": modules,
            "plan": plan if plan is not None else [{}, {}]}


PLAN0 = {"model.embed_tokens": (0, 4096, "linear"),
         "model.layers.0.attn.q_proj": (0, 2048, "attn"),
         "model.layers.0.moe": (0, 64, "experts"),
         "model.layers.0.moe.experts.11.w": (0, 512, "moe"),
         "model.layers.0.moe.experts.3.w": (512, 1024, "moe"),
         "model.layers.0.attn.idle": (7, 7, "attn")}
PLAN1 = {"model.layers.1.attn.q_proj": (2048, 4096, "attn"),
         "lm_head": (0, 100, "linear")}


def good_modules(device):
    ok_dev = f"cuda:{device}"
    # A TP-imported linear: real in/out features + a present weight tensor.
    qkv = FModule(f"model.layers.{device}.attn.q_proj", device, caps={"tp_col": True},
                  in_features=4096, out_features=1024, qtype="q4_0", weight=FT(device=ok_dev))
    attn = FModule(f"model.layers.{device}", device, caps={"kv_cache": True},
                   num_q_heads=28, num_kv_heads=4, children=[qkv],
                   cache_layers=[FCacheLayer(FT(device=ok_dev), FT(device=ok_dev), device=ok_dev)])
    state = FState(conv_state=FT(device=ok_dev), id_state=FT(device="cpu"), device=ok_dev)
    ple = FModule(f"model.ple.{device}", device, caps={"recurrent_cache": True},
                  recurrent_layers=[state],
                  children=[FNgram("model.ple.ngram", "fp16_ram",
                                   tables=[FT(device="cpu")] if device == 1 else [],
                                   device=device)])
    moe = FModule(f"model.layers.{device}.moe", device,
                  num_local_experts=16, routing_first=device * 16, routing_last=(device + 1) * 16)
    gather = FModule("output_gather", device, caps={}, stub=True, children=[])
    return [attn, ple, moe, gather]


# ---------------------------------------------------------------------------
# Parser / shared helper reuse / pure math
# ---------------------------------------------------------------------------

class ReuseAndParserTests(unittest.TestCase):
    def test_frozen_prompt_and_metric_helpers_are_shared_not_duplicated(self):
        self.assertIs(tr.validate_prompts, qmr.validate_prompts)
        self.assertIs(tr.prompt_sha256, qmr.prompt_sha256)
        self.assertIs(tr.delivery_rate, qmr.delivery_rate)
        self.assertIs(tr._raise_nofile, qmr._raise_nofile)
        self.assertEqual(tr.RESULT_KEYS, qmr.RESULT_KEYS)
        self.assertEqual(tr.MODES, qmr.MODES)

    def test_delivery_rate_burst_aware_via_reused_helper(self):
        ev = [[0.05, 3], [0.4, 40], [1.05, 256]]
        self.assertAlmostEqual(tr.delivery_rate(ev), (256 - 3) / (1.05 - 0.05))
        self.assertIsNone(tr.delivery_rate([[1.0, 5]]))

    REQUIRED = ["-m", "/models/m", "--prompts-json", "/tmp/p.json", "--execution", "tp",
                "--mode", "mtp", "--power-socket", "/tmp/s", "--output", "/tmp/o.json"]

    def test_defaults(self):
        a = tr.build_parser().parse_args(self.REQUIRED)
        self.assertEqual((a.execution, a.mode), ("tp", "mtp"))
        self.assertEqual((a.draft_tokens, a.batch_size, a.cache_tokens, a.max_chunk_size,
                          a.new_tokens), (4, 1, 8704, 2048, 256))
        self.assertEqual(a.use_per_device, [28, 28])
        self.assertFalse(a.dynamic_draft)
        self.assertFalse(a.validate_finite)
        self.assertEqual(a.draft_confidence, 0.4)
        tr.validate_args(a)

    def test_execution_and_mode_required(self):
        for drop in (["--execution", "tp"], ["--mode", "mtp"]):
            argv = [x for x in self.REQUIRED if x not in drop and x != drop[0]]
            with self.subTest(drop=drop[0]), self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    tr.build_parser().parse_args(argv)

    def test_validate_args_rejects(self):
        base = tr.build_parser().parse_args(self.REQUIRED)
        bad = {"zero budget excludes a device": {"use_per_device": [28, 0]},
               "negative budget": {"use_per_device": [-1, 28]},
               "nonpositive draft": {"draft_tokens": 0},
               "nonpositive new tokens": {"new_tokens": -2},
               "cache not page multiple": {"cache_tokens": 100},
               "chunk not page multiple": {"max_chunk_size": 2047},
               "confidence out of range": {"draft_confidence": 1.5},
               "nan confidence": {"draft_confidence": float("nan")}}
        for label, over in bad.items():
            patched = argparse.Namespace(**(vars(base) | over))
            with self.subTest(label), self.assertRaises(ValueError):
                tr.validate_args(patched)
        ar_dyn = argparse.Namespace(**(vars(base) | {"mode": "ar", "dynamic_draft": True}))
        with self.assertRaisesRegex(ValueError, "dynamic-draft"):
            tr.validate_args(ar_dyn)

    def test_required_cache_tokens_math(self):
        one = [{"ids": list(range(200))}]
        self.assertEqual(tr.required_cache_tokens(one, 256, 4), 512)  # 200+256+4=460 -> page-rounded
        two = [{"ids": list(range(200))}, {"ids": list(range(53))}]
        self.assertEqual(tr.required_cache_tokens(two, 52, 4), 256 + 256)  # 256 exactly fits, 109 rounds
        self.assertEqual(tr.required_cache_tokens([{"ids": [1] * 256}], 1, 0), 512)  # 257 -> 512

    def test_frozen_prompts_flow_through_shared_validator(self):
        ids = [3, 1, 4, 1, 5]
        ok = tr.validate_prompts({"prompts": [entry(ids, language="ja", repeat=1)]}, 1)
        self.assertEqual(ok[0]["ids"], ids)
        with self.assertRaisesRegex(ValueError, "sha mismatch"):
            tr.validate_prompts({"prompts": [entry(ids, sha="0" * 64)]}, 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            tr.validate_prompts({"prompts": [entry(ids), entry(ids)]}, 1)

    def test_lazy_native_imports(self):
        tree = ast.parse(Path(tr.__file__).read_text(encoding="utf-8"))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(n.name.split(".")[0] for n in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                top.add(node.module.split(".")[0])
        self.assertFalse(top & {"torch", "exllamav3", "exllamav3_ext"},
                         f"module-level native imports: {top}")


# ---------------------------------------------------------------------------
# Worker-side audit function, driven with fake local_contexts
# ---------------------------------------------------------------------------

class WorkerAuditTests(unittest.TestCase):
    def test_records_actual_worker_state_and_is_picklable(self):
        with fake_worker_torch():
            rec = tr.tp_audit_rank(audit_ctx(1, good_modules(1), plan=[PLAN0, PLAN1]))
        self.assertEqual(rec["device"], 1)
        self.assertEqual(rec["pid"], os.getpid())          # pseudo rank runs in-parent
        self.assertIsNone(rec["backend_class"])            # no backend in this fake ctx
        self.assertEqual(rec["gcnArchName"].split(":")[0], "gfx1030")
        json.dumps(rec)                                    # crosses a pipe only if this works
        self.assertEqual(rec["torch_allocated_bytes"], 2000)
        self.assertEqual(rec["torch_peak_bytes"], 4000)
        self.assertIn("proc_mem", rec)
        plan = rec["plan"]
        self.assertEqual(plan["nonempty_slices"], 2)       # attn row + lm_head
        self.assertEqual(plan["width_by_unit"], {"attn": 2048, "linear": 100})
        ng = {m["key"]: m for m in rec["ngram"]["modules"]}
        self.assertTrue(ng["model.ple.ngram"]["ram_backed"])
        self.assertEqual(ng["model.ple.ngram"]["table_residence"], ["cpu"])
        self.assertEqual(ng["model.ple.ngram"]["table_ram_bytes"],
                         sum(t.numel() * t.element_size() for t in [FT(device="cpu")]))
        resid = ng["model.ple.ngram"]["residency"]          # measured live inside the rank
        self.assertEqual(resid["probe"], "mincore")
        self.assertTrue(resid["all_resident"], resid)
        self.assertGreaterEqual(resid["pages_total"], 1)
        self.assertEqual(resid["pages_nonresident"], 0)
        attn = [m for m in rec["modules"] if m["key"] == "model.layers.1"][0]
        self.assertEqual(attn["cache_layers"]["layers_total"], 1)
        self.assertEqual(attn["cache_layers"]["sample"][0]["tensors"]["k"]["device"], "cuda:1")
        self.assertEqual(attn["cache_layers"]["sample"][0]["tensors"]["k"]["bytes"], 4)
        self.assertEqual(attn["cache_layers"]["bytes_total"], 8)
        self.assertEqual(attn["facts"]["num_q_heads"], 28)
        self.assertEqual(attn["facts"]["num_kv_heads"], 4)
        self.assertTrue(attn["facts"]["linear"][0]["weight_present"])
        self.assertEqual(attn["facts"]["linear"][0]["qtype"], "q4_0")
        self.assertEqual(attn["facts"]["linear"][0]["in_features"], 4096)
        ple = [m for m in rec["modules"] if m["key"] == "model.ple.1"][0]
        self.assertEqual(ple["recurrent_layers"]["sample"][0]["tensors"]["id_state"]["device"], "cpu")
        moe = [m for m in rec["modules"] if m["key"].endswith(".moe")][0]
        self.assertEqual(moe["facts"]["num_local_experts"], 16)
        self.assertEqual((moe["facts"]["routing_first"], moe["facts"]["routing_last"]), (16, 32))
        gather = [m for m in rec["modules"] if m["key"] == "output_gather"][0]
        self.assertTrue(gather["facts"]["stub"])

    def test_plan_expert_units_and_zero_width_rows(self):
        with fake_worker_torch():
            rec = tr.tp_audit_rank(audit_ctx(0, good_modules(0), plan=[PLAN0, PLAN1]))
        plan = rec["plan"]
        self.assertEqual(plan["plan_entries"], 6)
        self.assertEqual(plan["nonempty_slices"], 5)       # zero-width idle row excluded
        self.assertEqual(plan["width_by_unit"],
                         {"attn": 2048, "linear": 4096, "moe": 1024, "experts": 64})
        self.assertEqual(plan["expert_ids_owned"], [3, 11])
        self.assertEqual(plan["expert_unit_owned"], {"model.layers.0.moe": [0, 64]})
        self.assertEqual(plan["expert_unit_count"], 1)

    def test_wrong_device_module_is_recorded(self):
        mods = [FModule("model.layers.9", 5, caps={"kv_cache": True})]   # claims cuda:5 in a cuda:0 rank
        with fake_worker_torch():
            rec = tr.tp_audit_rank(audit_ctx(0, mods, plan=[PLAN0, PLAN1]))
        self.assertEqual(rec["modules"][0]["device_index"], 5)
        audit = tr.aggregate_tp_audit([rec], [0, 1], 1, os.getpid())
        self.assertIn("model.layers.9", " ".join(audit["problems"]))

    def test_finite_hook_installer_checks_only_returned_tensors(self):
        class Counter:
            n = 0

        def ok_fwd(*a, **k):
            Counter.n += 1
            return FT((1.0, 2.0))

        def bad_fwd(*a, **k):
            return FT((1.0, float("inf")))

        def none_fwd(*a, **k):
            Counter.n += 1
            return {"hidden": "not observed as a tensor"}

        mods = [FModule("layers.0", 0, forward=ok_fwd), FModule("layers.1", 0, forward=bad_fwd)]
        ctx = audit_ctx(0, mods)
        with fake_worker_torch():
            info = tr.tp_install_finite_hook(ctx)
            self.assertEqual(info["wrapped"], 2)
            mods[0].forward([1])                                   # finite pass-through counts
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                mods[1].forward([1])                               # raises INSIDE the rank
            stats = tr.tp_finite_hook_stats(ctx)
        self.assertEqual(stats["checks"], 1)
        self.assertEqual(Counter.n, 1)
        with fake_worker_torch():
            ctx2 = audit_ctx(0, [FModule("layers.2", 0, forward=none_fwd)])
            tr.tp_install_finite_hook(ctx2)
            self.assertEqual(ctx2["modules"][0].forward([1]), {"hidden": "not observed as a tensor"})
            self.assertEqual(tr.tp_finite_hook_stats(ctx2)["checks"], 0)


# ---------------------------------------------------------------------------
# TP power-phase sync routing (fix: parent synchronize() cannot drain a child)
# ---------------------------------------------------------------------------

class PowerSyncFacadeTests(unittest.TestCase):
    class FakeTPModel:
        active_devices = [0, 1]

        def __init__(self):
            self.calls = []

        def tp_worker_dispatch_single(self, device, fn, args):
            self.calls.append((fn.__name__, device))
            return {"device": device, "pid": 4001 if device == 0 else os.getpid()}

    def test_tp_sync_rank_synchronizes_own_device_in_worker(self):
        with fake_worker_torch() as torch:
            res = tr.tp_sync_rank(audit_ctx(1, []))
        self.assertEqual(torch.syncs, [1])
        self.assertEqual(res["device"], 1)
        self.assertEqual(res["pid"], os.getpid())

    def test_facade_routes_each_device_to_its_own_rank(self):
        model = self.FakeTPModel()
        facade = tr._TPPowerTorch(model)
        facade.cuda.synchronize(0)
        facade.cuda.synchronize(1)
        self.assertEqual(model.calls, [("tp_sync_rank", 0), ("tp_sync_rank", 1)])
        self.assertEqual([s["worker"]["pid"] for s in facade.syncs], [4001, os.getpid()])

    def test_facade_refuses_a_device_that_is_not_a_tp_rank(self):
        facade = tr._TPPowerTorch(self.FakeTPModel())
        with self.assertRaisesRegex(RuntimeError, "cannot route power sync"):
            facade.cuda.synchronize(5)


# ---------------------------------------------------------------------------
# mincore(2) table-page residency: real kernel queries, no faulting, no mlock
# ---------------------------------------------------------------------------

class MincoreResidencyTests(unittest.TestCase):
    def _ns_tensor(self, addr, nbytes, device="cpu", contiguous=True):
        return types.SimpleNamespace(device=device, data_ptr=lambda: addr,
                                     numel=lambda: nbytes, element_size=lambda: 1,
                                     is_contiguous=lambda: contiguous)

    def test_page_span_accounts_unaligned_head_and_tail(self):
        p = 4096
        self.assertEqual(tr._page_span(0, 8192, p), (0, 2))               # exact pages
        self.assertEqual(tr._page_span(p, p, p), (p, 1))
        self.assertEqual(tr._page_span(100, 3900, p), (0, 1))             # inside one page
        self.assertEqual(tr._page_span(100, 4000, p), (0, 2))             # unaligned head straddle
        self.assertEqual(tr._page_span(4095, 2, p), (0, 2))               # straddles boundary
        self.assertEqual(tr._page_span(4096 + 1, 4096, p), (4096, 2))     # head+tail waste
        aligned, total = tr._page_span(12345, 32640156672, p)             # the real Q size
        self.assertEqual((aligned, total), (12288, 7968789))
        self.assertEqual(aligned % p, 0)                                  # mincore needs page addr
        self.assertTrue(total * p >= 32640156672 + (12345 - aligned))     # span covers the range

    def test_real_touched_anonymous_mapping_reports_resident(self):
        import ctypes
        import mmap
        ps = __import__("resource").getpagesize()
        mm = mmap.mmap(-1, ps * 2)
        buf = ctypes.c_char.from_buffer(mm)
        try:
            addr = ctypes.addressof(buf)
            mm.seek(0); mm.write(b"x" * 4)
            mm.seek(ps); mm.write(b"y" * 4)
            aligned = (addr // ps) * ps
            res = tr._mincore_range(aligned, ps * 2)
            if res["error"]:
                self.skipTest(f"mincore unsupported here: {res['error']}")
            self.assertEqual(res["pages_total"], 2)
            self.assertEqual(res["pages_resident"], 2)
            self.assertEqual(res["pages_nonresident"], 0)
        finally:
            del buf
            mm.close()

    def test_untouched_page_can_be_proven_nonresident(self):
        import ctypes
        import mmap
        ps = __import__("resource").getpagesize()
        mm = mmap.mmap(-1, ps * 2)
        buf = ctypes.c_char.from_buffer(mm)
        try:
            aligned = (ctypes.addressof(buf) // ps) * ps
            mm.seek(0); mm.write(b"x" * 4)                                # only page 0
            res = tr._mincore_range(aligned, ps * 2)
            if res["error"] or res["pages_resident"] == 2:
                self.skipTest("kernel reports both pages resident")
            self.assertEqual(res["pages_resident"], 1)
            self.assertEqual(res["pages_nonresident"], 1)
            # and the same range via a fake tensor: partial != proven resident
            t = self._ns_tensor(aligned, ps * 2)
            r = tr._table_residency(t)
            self.assertFalse(r["all_resident"])
        finally:
            del buf
            mm.close()

    def test_unmapped_range_errors_and_is_not_resident(self):
        import ctypes
        import mmap
        ps = __import__("resource").getpagesize()
        mm = mmap.mmap(-1, ps)
        buf = ctypes.c_char.from_buffer(mm)
        addr = (ctypes.addressof(buf) // ps) * ps
        del buf
        mm.close()                                                        # munmap
        res = tr._mincore_range(addr, ps)
        self.assertIsNotNone(res["error"])                                # ENOMEM, fail closed
        self.assertNotEqual(res.get("pages_resident"), res.get("pages_total"))

    def test_non_cpu_or_non_contiguous_tensor_rejected_without_probing(self):
        r = tr._table_residency(self._ns_tensor(0x1000, 4096, device="cuda:0"))
        self.assertIn("not a CPU tensor", r["error"])
        r = tr._table_residency(self._ns_tensor(0x1000, 4096, contiguous=False))
        self.assertIn("non-contiguous", r["error"])

    def test_measure_ngram_residency_over_cpu_fake_tensors(self):
        res = tr.measure_ngram_residency([FT(device="cpu")])
        if res["error"]:
            self.skipTest(f"mincore unsupported here: {res['error']}")
        self.assertTrue(res["all_resident"])
        self.assertEqual(res["probe"], "mincore")
        self.assertEqual(res["pages_total"], 1)
        res = tr.measure_ngram_residency([FT(device="cuda:1")])
        self.assertFalse(res["all_resident"])
        self.assertIsNotNone(res["error"])

    def test_proc_mem_exposes_fault_counters(self):
        pm = tr._proc_mem()
        self.assertEqual(set(pm), {"vm_rss_kb", "pss_kb", "vm_swap_kb", "minflt", "majflt"})
        self.assertIsInstance(pm["minflt"], int)
        self.assertGreaterEqual(pm["majflt"], 0)

    def test_attach_ngram_residency_matches_live_modules(self):
        t = FT(device="cpu")
        recs = [{"key": "model.ple.ngram", "num_table_tensors": 1},
                {"key": "other.replica", "num_table_tensors": 0}]
        tr.attach_ngram_residency(recs, [FNgram("model.ple.ngram", "fp16_ram", tables=[t])])
        if recs[0]["residency"]["error"]:
            self.skipTest(f"mincore unsupported here: {recs[0]['residency']['error']}")
        self.assertTrue(recs[0]["residency"]["all_resident"])
        self.assertIsNone(recs[1]["residency"])                    # absent = explicit None
        orphan = [{"key": "orphan", "num_table_tensors": 1}]
        tr.attach_ngram_residency(orphan, [])
        self.assertFalse(orphan[0]["residency"]["all_resident"])   # unmatched live module


# ---------------------------------------------------------------------------
# Parent-side aggregation policy (pure data)
# ---------------------------------------------------------------------------

def mod_rec(key, device, *, stub=False, weight=True, heads=0, experts=0,
            cache_layers=None, recurrent_layers=None):
    facts = {"stub": stub, "linear_count": 0 if stub or not weight else 1,
             "linear": ([] if stub or not weight else
                        [{"name": key + ".q_proj", "type": "TPColLinear",
                          "in_features": 4096, "out_features": 1024, "qtype": "q4_0",
                          "weight_present": True}])}
    if heads:
        facts["num_q_heads"] = heads
    if experts:
        facts["num_local_experts"] = experts
    empty_geom = {"layers_total": 0, "bytes_total": None, "sample": []}
    return {"key": key, "type": "X", "device": f"cuda:{device}", "device_index": device,
            "caps": {}, "transformer": not stub, "kv_cache_modules": 0,
            "recurrent_cache_modules": 0,
            "cache_layers": cache_layers or dict(empty_geom),
            "recurrent_layers": recurrent_layers or dict(empty_geom),
            "facts": facts}


def geom(device, cl_device="cuda:0", tensors=None):
    return {"layers_total": 1, "bytes_total": 8,
            "sample": [{"device": cl_device, "device_index": _dev(cl_device),
                        "n_tensors": 2,
                        "tensors": tensors or {"k": {"device": cl_device},
                                               "v": {"device": cl_device}}}]}


def _dev(s):
    tail = str(s).rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None


def ng_rec(key, mode, tables, ram_bytes=None, handles=0, residence=None, residency="auto"):
    rec = {"key": key, "mode": mode, "ram_backed": bool(tables) and str(mode).endswith("_ram"),
           "num_table_tensors": tables, "table_ram_bytes": ram_bytes,
           "num_disk_handles": handles,
           "table_residence": residence if residence is not None
           else (["cpu"] if tables else [])}
    if residency == "auto":
        residency = table_residency_ok() if (tables and str(mode).endswith("_ram")) else None
    rec["residency"] = residency
    return rec


def table_residency_ok(pages=7968789):
    """Synthetic all-resident mincore evidence. The real Q table is
    32,640,156,672 B -> 7,968,789 span pages of 4 KiB (~8 MB kernel vector)."""
    return {"probe": "mincore",
            "tensors": [{"index": 0, "page_size": 4096, "bytes": 32640156672,
                         "pages_total": pages, "pages_resident": pages,
                         "pages_nonresident": 0, "error": None}],
            "pages_total": pages, "pages_resident": pages, "pages_nonresident": 0,
            "all_resident": True, "error": None}


def table_residency_partial(pages=7968789, resident=7968788):
    r = table_residency_ok(pages)
    r.update(pages_resident=resident, pages_nonresident=pages - resident, all_resident=False,
             tensors=[dict(r["tensors"][0], pages_resident=resident,
                           pages_nonresident=pages - resident)])
    return r


def table_residency_error(msg="OSError: mincore rc=-1: Cannot allocate memory"):
    return {"probe": "mincore", "tensors": [{"index": 0, "error": msg}], "pages_total": None,
            "pages_resident": None, "pages_nonresident": None, "all_resident": False,
            "error": msg}


OWNER_NG = [ng_rec("model.ple.ngram", "fp16_ram", 1, ram_bytes=32 << 30)]


def rank_rec(device, pid, *, arch="gfx1030:sdma", slices=40, modules=None, ngram=(),
             rss=1000, pss=900, swap=0):
    return {"device": device, "rank": device, "world_size": 2, "active_devices": [0, 1],
            "output_device": 1, "pid": pid, "gcnArchName": arch,
            "torch_allocated_bytes": 10 ** 9, "torch_peak_bytes": 2 * 10 ** 9,
            "torch_reserved_bytes": 3 * 10 ** 9,
            "proc_mem": {"vm_rss_kb": rss, "pss_kb": pss, "vm_swap_kb": swap},
            "plan": {"plan_entries": slices, "nonempty_slices": slices,
                     "width_by_unit": {}, "nonempty_sample": {},
                     "expert_ids_owned": [], "expert_count_owned": 0,
                     "expert_unit_owned": {}, "expert_unit_count": 0},
            "modules": modules if modules is not None else [mod_rec(f"layers.{device}.0", device)],
            "ngram": {"modules": list(ngram)}}


class AggregateTests(unittest.TestCase):
    def test_happy_path_single_ram_owner_on_parent_pid(self):
        r0 = rank_rec(0, 4001)                                    # no ngram exposure on non-owner
        r1 = rank_rec(1, os.getpid(), ngram=OWNER_NG)
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=["model.ple.ngram"])
        self.assertTrue(audit["ok"], audit["problems"])
        self.assertEqual(list(audit["ngram_ram_owners"]), ["model.ple.ngram"])
        self.assertEqual(audit["ngram_ram_owners"]["model.ple.ngram"][0]["device"], 1)
        self.assertEqual(audit["memory"]["totals"]["vm_rss_kb"], 2000)   # two unique PIDs
        self.assertEqual(len(audit["memory"]["unique_pids"]), 2)

    def test_missing_unexpected_and_duplicate_rank_records(self):
        r0 = rank_rec(0, 4001)
        audit = tr.aggregate_tp_audit([r0], [0, 1], 1, os.getpid())
        self.assertFalse(audit["ok"])
        self.assertIn("no audit record from device 1", " ".join(audit["problems"]))
        audit = tr.aggregate_tp_audit([r0, r0], [0, 1], 1, os.getpid())
        self.assertIn("duplicate audit records", " ".join(audit["problems"]))
        audit = tr.aggregate_tp_audit([r0, rank_rec(1, 4002), rank_rec(5, 4003)],
                                      [0, 1], 1, 4003)
        self.assertIn("unexpected device 5", " ".join(audit["problems"]))

    def test_one_gpu_can_never_be_labelled_tp2(self):
        # Aggregate level: a lone rank cannot satisfy the requested {0,1}. The
        # run() level check against model.active_devices is covered in
        # RunDispatchTests.test_single_active_device_cannot_be_labelled_tp2.
        audit = tr.aggregate_tp_audit([rank_rec(0, 4001)], [0, 1], 1, 4001)
        self.assertIn("no audit record from device 1", " ".join(audit["problems"]))

    def test_stub_rank_rejected_even_with_nonempty_plan(self):
        gather_only = [mod_rec("output_gather", 1, stub=True, weight=False)]
        r0 = rank_rec(0, 4001)
        r1 = rank_rec(1, os.getpid(), modules=gather_only)
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid())
        self.assertFalse(audit["ok"])
        self.assertIn("NO provable computation", " ".join(audit["problems"]))

    def test_transformer_flag_on_stub_is_not_evidence(self):
        stub_with_plan = [mod_rec("model.layers.9", 1, stub=True, weight=False)]
        r0, r1 = rank_rec(0, 4001), rank_rec(1, os.getpid(), modules=stub_with_plan, slices=99)
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid())
        self.assertIn("NO provable computation", " ".join(audit["problems"]))

    def test_module_and_tensor_foreign_device_rejected(self):
        r = rank_rec(0, 4001, modules=[mod_rec("model.layers.9", 0,
                                                cache_layers=geom(0, cl_device="cuda:1"))])
        audit = tr.aggregate_tp_audit([r, rank_rec(1, os.getpid())], [0, 1], 1, os.getpid())
        self.assertIn("live on cuda:1", " ".join(audit["problems"]))

    def test_wrong_cuda_cache_tensor_with_correct_owner_label_rejected(self):
        # cl.device label says cuda:0 (correct rank) but the actual k tensor sits on cuda:1
        bad = geom(0, cl_device="cuda:0",
                   tensors={"k": {"device": "cuda:1"}, "v": {"device": "cuda:0"}})
        r = rank_rec(0, 4001, modules=[mod_rec("layers.0.a", 0, cache_layers=bad)])
        audit = tr.aggregate_tp_audit([r, rank_rec(1, os.getpid())], [0, 1], 1, os.getpid())
        self.assertIn("state tensor 'k' lives on cuda:1", " ".join(audit["problems"]))

    def test_id_state_must_stay_host_side(self):
        g = geom(0, cl_device="cuda:0",
                 tensors={"conv_state": {"device": "cuda:0"}, "id_state": {"device": "cuda:0"}})
        r = rank_rec(0, 4001, modules=[mod_rec("ple.0", 0, recurrent_layers=g)])
        audit = tr.aggregate_tp_audit([r, rank_rec(1, os.getpid())], [0, 1], 1, os.getpid())
        self.assertIn("id_state is on cuda:0", " ".join(audit["problems"]))

    def test_duplicate_ram_tables_rejected(self):
        r0 = rank_rec(0, 4001, ngram=OWNER_NG)
        r1 = rank_rec(1, os.getpid(), ngram=OWNER_NG)
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=["model.ple.ngram"])
        self.assertFalse(audit["ok"])
        self.assertIn("RAM owners", " ".join(audit["problems"]))

    def test_disk_offload_rejected(self):
        r0 = rank_rec(0, 4001)
        r1 = rank_rec(1, os.getpid(),
                      ngram=[ng_rec("model.ple.ngram", "trellis_disk", 0, handles=2)])
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=["model.ple.ngram"])
        self.assertFalse(audit["ok"])
        self.assertIn("disk-offloaded", " ".join(audit["problems"]))

    def test_expected_engram_absent_fails_closed(self):
        # THE former false pass: Qwen3.8 expects a table, every rank exposes nothing.
        r0, r1 = rank_rec(0, 4001), rank_rec(1, os.getpid())
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=["model.ple.ngram"])
        self.assertFalse(audit["ok"])
        self.assertIn("exposed by NO rank", " ".join(audit["problems"]))

    def test_ram_table_on_cuda_rejected_despite_ram_mode_label(self):
        bad = [ng_rec("model.ple.ngram", "fp16_ram", 1, ram_bytes=32 << 30,
                      residence=["cuda:1"])]
        r0, r1 = rank_rec(0, 4001), rank_rec(1, os.getpid(), ngram=bad)
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=["model.ple.ngram"])
        self.assertIn("not host-resident", " ".join(audit["problems"]))

    def test_ram_mode_without_tensors_and_bad_byte_counts_rejected(self):
        cases = {"mode None": [ng_rec("model.ple.ngram", None, 0)],
                 "ram no tensors": [ng_rec("model.ple.ngram", "fp16_ram", 0)],
                 "zero bytes": [ng_rec("model.ple.ngram", "fp16_ram", 1, ram_bytes=0)],
                 "unknown bytes": [ng_rec("model.ple.ngram", "fp16_ram", 1, ram_bytes=None)]}
        for label, ng in cases.items():
            r0, r1 = rank_rec(0, 4001), rank_rec(1, os.getpid(), ngram=ng)
            with self.subTest(label):
                audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                              expected_ngram_keys=["model.ple.ngram"])
                self.assertFalse(audit["ok"])

    def test_unplanned_owner_fails_even_for_tableless_expectation(self):
        r1 = rank_rec(1, os.getpid(), ngram=OWNER_NG)
        audit = tr.aggregate_tp_audit([rank_rec(0, 4001), r1], [0, 1], 1, os.getpid())
        self.assertIn("unexpected n-gram table", " ".join(audit["problems"]))

    def test_owner_swap_with_proven_table_residency_is_noted_not_fatal(self):
        # The measured correction: process-wide VmSwap (host reclaim + a second
        # user's pages) is NOT evidence the TABLE is swapped. With all table
        # pages mincore-resident the run passes, and the swap stays visible.
        r0 = rank_rec(0, 4001)
        r1 = rank_rec(1, os.getpid(), ngram=OWNER_NG, swap=30328)
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=["model.ple.ngram"])
        self.assertTrue(audit["ok"], audit["problems"])
        self.assertIn("VmSwap 30328 KB", " ".join(audit["notes"]))
        self.assertIn("diagnostic", " ".join(audit["notes"]))

    def test_process_swap_alone_can_never_pass(self):
        # Identical owner metadata but NO residency probe result: fail closed.
        r0 = rank_rec(0, 4001)
        r1 = rank_rec(1, os.getpid(),
                      ngram=[ng_rec("model.ple.ngram", "fp16_ram", 1,
                                    ram_bytes=32 << 30, residency=None)], swap=30328)
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=["model.ple.ngram"])
        self.assertFalse(audit["ok"])
        self.assertIn("NO mincore page-residency evidence", " ".join(audit["problems"]))
        self.assertEqual(audit["ngram_ram_owners"], {})

    def test_partially_nonresident_table_pages_rejected(self):
        r0 = rank_rec(0, 4001)
        r1 = rank_rec(1, os.getpid(),
                      ngram=[ng_rec("model.ple.ngram", "fp16_ram", 1, ram_bytes=32 << 30,
                                    residency=table_residency_partial())])
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=["model.ple.ngram"])
        self.assertIn("NOT fully resident", " ".join(audit["problems"]))

    def test_residency_probe_error_rejected(self):
        r0 = rank_rec(0, 4001)
        r1 = rank_rec(1, os.getpid(),
                      ngram=[ng_rec("model.ple.ngram", "fp16_ram", 1, ram_bytes=32 << 30,
                                    residency=table_residency_error())])
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=["model.ple.ngram"])
        self.assertIn("NO mincore page-residency evidence", " ".join(audit["problems"]))
        self.assertIn("probe failed", " ".join(audit["problems"]))

    def test_tableless_checkpoint_is_recorded_noop(self):
        r0, r1 = rank_rec(0, 4001), rank_rec(1, os.getpid())
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid(),
                                      expected_ngram_keys=[])
        self.assertTrue(audit["ok"], audit["problems"])
        self.assertIn("no-op", " ".join(audit["notes"]))
        self.assertEqual(audit["ngram_ram_owners"], {})

    def test_per_pid_memory_counted_once_and_parent_identity_enforced(self):
        r0 = rank_rec(0, 777, rss=1500)
        r1 = rank_rec(1, 777, rss=1500)                    # impossible in-engine; still dedup-safe
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, 777)
        self.assertEqual(audit["memory"]["totals"]["vm_rss_kb"], 1500)
        self.assertIn("span", " ".join(audit["notes"]))
        # pseudo output rank must be the parent PID itself
        audit = tr.aggregate_tp_audit([rank_rec(0, 4001), rank_rec(1, 9999)],
                                      [0, 1], 1, os.getpid())
        self.assertIn("in-process pseudo rank IS the parent", " ".join(audit["problems"]))

    def test_arch_suffix_normalized_but_mismatch_fails(self):
        r0, r1 = rank_rec(0, 4001, arch="gfx1030:sramecc+"), rank_rec(1, os.getpid())
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid())
        self.assertTrue(audit["ok"], audit["problems"])
        self.assertEqual(audit["ranks"][0]["gcnArchName"], "gfx1030:sramecc+")  # full string kept
        r0 = rank_rec(0, 4001, arch="gfx900:xnack-")
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid())
        self.assertIn("gcnArchName", " ".join(audit["problems"]))

    def test_nonzero_swap_on_non_owner_recorded_as_note(self):
        r0 = rank_rec(0, 4001)
        r1 = rank_rec(1, os.getpid(), swap=4200)
        audit = tr.aggregate_tp_audit([r0, r1], [0, 1], 1, os.getpid())
        self.assertTrue(audit["ok"], audit["problems"])
        self.assertIn("swapped out", " ".join(audit["notes"]))

    def test_cpu_helper_is_optional_metadata(self):
        meta = {"device": -1, "pid": 4001, "backend": "TPBackendNative", "proc_mem": {}}
        audit = tr.aggregate_tp_audit([rank_rec(0, 4001), rank_rec(1, os.getpid())],
                                      [0, 1], 1, os.getpid(), cpu_meta=meta)
        self.assertEqual(audit["cpu_helper"], meta)
        self.assertTrue(audit["ok"])


# ---------------------------------------------------------------------------
# Fake engine stack for run() dispatch tests: build on the qwen test fakes and
# extend the Model fake with the TP mixin surface the CLI interrogates.
# ---------------------------------------------------------------------------

class FChild:
    """Real multiprocessing.Process stand-in for the manual-drain path."""
    def __init__(self, pid):
        self.pid, self.terminated, self.joined = pid, False, []
        self._alive = True

    def join(self, timeout=None):
        self.joined.append(timeout)

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False


class PseudoChild:
    """The engine's in-process pseudo rank: must NEVER be terminated/joined by us."""
    terminated = False


class FConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class PseudoParentConn(FConn):
    pass


NGRAM_SHELL_KEY = "model.ple.ngram"


def install_tp_stack(log, *, with_mtp=True, supports_tp=True, dispatch=None,
                     load_exc_cls=None, gen_cls=None, ngram_shell=True, active=(0, 1),
                     children=None, destroy_exc=None, mp_conns=None, ngram_tables=False):
    saved, saved_attr, pkg = tqmr._install_fake_stack(log)
    exl = sys.modules["exllamav3"]
    torch = sys.modules["torch"]
    idx = lambda d: int(d) if isinstance(d, int) else int(str(d).rsplit(":", 1)[-1])
    torch.device = lambda s: s
    torch.cuda.memory_allocated = lambda d: 111 * (idx(d) + 1)
    torch.cuda.max_memory_allocated = lambda d: 222 * (idx(d) + 1)
    torch.cuda.memory_reserved = lambda d: 555 * (idx(d) + 1)
    base_model = exl.Model

    class TPModel(base_model):
        DISPATCH = dict(dispatch or {})
        LOAD_EXC = load_exc_cls
        ACTIVE = list(active)
        CHILDREN = children
        DESTROY_EXC = destroy_exc
        MP_CONNS = mp_conns

        def __init__(self, component="text"):
            super().__init__(component)
            self.caps = {"supports_tp": supports_tp}
            self.loaded_tp = False
            self.mp_children = []
            self.mp_parent_conn = []
            self.active_devices = []
            self.tp_output_device = None
            self.tp_backend = None
            self.output_device = None
            self.plan = None
            # PRE-load module tree: an (unloaded) Engram shell for Qwen3.8 so the
            # CLI captures expected n-gram keys. LS tests additionally give it a
            # real CPU table tensor; TP tests keep empty shells so the stray-
            # parent-table scan stays clean while the ranks carry the owner.
            self.modules = ([FModule("model.ple", 1, children=[
                                FNgram(NGRAM_SHELL_KEY, "fp16_ram",
                                       tables=[FT(device="cpu")] if ngram_tables else [])])]
                            if ngram_shell and component == "text" else [])

        def load(self, **kw):
            super().load(**kw)
            if self.component == "mtp":
                return
            if kw.get("tensor_p"):
                self.active_devices = list(TPModel.ACTIVE)
                self.mp_children = list(TPModel.CHILDREN or ["w0", PseudoChild(), "cpu"])
                self.mp_parent_conn = list(TPModel.MP_CONNS or ["c0", PseudoParentConn(), "c-1"])
                if TPModel.LOAD_EXC is not None:
                    exc = TPModel.LOAD_EXC
                    TPModel.LOAD_EXC = None
                    raise exc("injected failure after worker spawn, before loaded_tp")
                self.loaded_tp = True
                self.tp_output_device = 1
                self.tp_backend = kw.get("tp_backend")
                self.output_device = kw.get("tp_output_device")
                self.plan = [dict(PLAN0), dict(PLAN1)]
            else:
                self.output_device = "cuda:1"

        def tp_worker_dispatch_wait_multi(self, devices, fn, args):
            log.append(("dispatch", fn.__name__, tuple(devices)))
            return list(TPModel.DISPATCH.get(fn.__name__, [None] * len(devices)))

        def tp_worker_dispatch_single(self, device, fn, args):
            log.append(("dispatch_single", fn.__name__, device))
            return {"device": device, "pid": os.getpid() if device == 1 else 4001}

        def destroy_tp_context(self):
            log.append(("destroy_tp",))
            if TPModel.DESTROY_EXC is not None:
                raise TPModel.DESTROY_EXC("engine teardown failed")
            self.mp_children = []

    exl.Model = TPModel
    if gen_cls is not None:
        exl.Generator = gen_cls
    if not with_mtp:
        exl.Config.OVERRIDES["/models/qwen38"] = {"model_classes": {"text": base_model}}
    mg = sys.modules["rocm_tools.rdna2.multi_gpu"]
    # Real geometry/ngram walkers (CPU-safe); placement audit stays the canned
    # tqmr stub -- the real auditor against LS fake models is qwen's suite's job.
    for name in ("device_memory_snapshot", "find_ngram_modules", "describe_ngram_module"):
        setattr(mg, name, getattr(real_multi_gpu, name))

    def ls_collect_ngram(model, require_ram):
        # Pure pass-through to the REAL describe/find: run() then attaches live
        # mincore residency to these records, so the LS owner path is proven
        # exactly like production (no synthetic owner claims).
        mods = [real_multi_gpu.describe_ngram_module(m)
                for m in real_multi_gpu.find_ngram_modules(model)]
        return {"requested": require_ram, "tables_found": len(mods), "modules": mods,
                "problems": [], "notes": [], "ok": True}

    mg.collect_ngram_state = ls_collect_ngram
    return saved, saved_attr, pkg


def uninstall_tp_stack(saved, saved_attr, pkg):
    tqmr._uninstall(saved, saved_attr, pkg)


def happy_dispatch(ngram_owner_on_out=True):
    r0 = rank_rec(0, 4001)                                   # no ngram exposure on non-owner
    r1 = rank_rec(1, os.getpid(), ngram=OWNER_NG if ngram_owner_on_out else ())
    return {"tp_audit_rank": [r0, r1],
            "tp_cpu_helper_meta": [{"device": -1, "pid": 4001, "backend": "TPBackendRCCL"}],
            "tp_install_finite_hook": [{"device": 0, "wrapped": 30}, {"device": 1, "wrapped": 31}],
            "tp_finite_hook_stats": [{"device": 0, "checks": 82}, {"device": 1, "checks": 82}]}


class RunDispatchTests(unittest.TestCase):
    BASE = ["-m", "/models/qwen38", "--cache-tokens", "1024", "--new-tokens", "9"]

    def _case(self, execution, mode, sock, out, *, prompts=None, extra=(), **stack_kw):
        log = []
        saved, saved_attr, pkg = install_tp_stack(log, **stack_kw)
        try:
            args = tr.build_parser().parse_args(
                self.BASE + ["--execution", execution, "--mode", mode,
                             "--power-socket", sock, "--output", out,
                             "--prompts-json", "/unused"] + list(extra))
            if prompts is None:
                prompts = tr.validate_prompts(
                    {"prompts": [entry(list(range(1, 6)), language="code", repeat=1)]},
                    args.batch_size)
            return tr.run(args, prompts), json.loads(Path(out).read_text()), log
        finally:
            uninstall_tp_stack(saved, saved_attr, pkg)

    def test_tp_execution_loads_tensor_parallel_and_audits_workers(self):
        with tempfile.TemporaryDirectory() as td:
            helper = tqmr.FakeHelper(Path(td) / "power.sock")
            helper.start()
            try:
                out = str(Path(td) / "tp.json")
                code, rep, log = self._case("tp", "mtp", str(Path(td) / "power.sock"), out,
                                            dispatch=happy_dispatch())
                self.assertEqual(code, 0, rep.get("error"))
                self.assertTrue(rep["complete"])
                self.assertEqual([e for e in log if e[0] == "load"],
                                 [("load", "mtp", {"device": "cuda:1", "max_chunk_size": 2048,
                                                   "progressbar": False}),
                                  ("load", "text", {"use_per_device": [28, 28],
                                                    "max_chunk_size": 2048, "progressbar": False,
                                                    "tensor_p": True, "tp_backend": "nccl",
                                                    "tp_output_device": "cuda:1"})])
                self.assertEqual([e[1] for e in log if e[0] == "dispatch"],
                                 ["tp_audit_rank", "tp_cpu_helper_meta", "tp_audit_rank"])
                self.assertEqual([e[2] for e in log if e[0] == "dispatch"],
                                 [(0, 1), (-1,), (0, 1)])
                # phase boundaries went THROUGH the ranks, not a parent-only sync
                singles = [e for e in log if e[0] == "dispatch_single"]
                self.assertEqual([e[1] for e in singles], ["tp_sync_rank"] * 6)
                self.assertEqual([e[2] for e in singles], [0, 1] * 3)
                self.assertEqual(len(rep["power_rank_syncs"]), 6)
                self.assertEqual([s["worker"]["pid"] for s in rep["power_rank_syncs"]],
                                 [4001, os.getpid()] * 3)
                self.assertEqual(rep["execution"],
                                 {"requested": "tp", "tp_backend": "nccl",
                                  "tp_output_device": "cuda:1", "loaded_tp": True,
                                  "actual": "tp2", "actual_backend": "nccl",
                                  "output_device": 1})
                self.assertEqual(rep["ngram_expected_keys"], [NGRAM_SHELL_KEY])
                self.assertTrue(rep["tp_audit"]["ok"])
                self.assertEqual(list(rep["tp_audit"]["ngram_ram_owners"]), [NGRAM_SHELL_KEY])
                self.assertEqual(rep["tp_audit"]["cpu_helper"]["backend"], "TPBackendRCCL")
                self.assertEqual(rep["generator"]["mode"], "mtp")
                self.assertTrue(rep["generator"]["mtp_draft"])
                # actual child peaks were queried post-inference BEFORE unload...
                self.assertEqual([r["device"] for r in rep["tp_final_audit"]["ranks"]], [0, 1])
                self.assertEqual(rep["peak_allocated_bytes"]["scope"], "parent_process_only")
                self.assertEqual(rep["peak_allocated_bytes"]["tp_rank_peak_bytes"],
                                 {"cuda:0": 2 * 10 ** 9, "cuda:1": 2 * 10 ** 9})
                self.assertEqual(rep["peak_allocated_bytes"]["parent_devices"],
                                 {"cuda:0": 222, "cuda:1": 444})
                row = rep["runs"][0]
                self.assertEqual(row["execution"], "tp")
                self.assertEqual([t for _, t in row["delivery_events"]], [3, 6, 9])
                modes = [m for m, _, _, _ in rep["power_policy"]["transitions"]]
                self.assertEqual(modes, ["auto", "profile_peak", "auto"])
                self.assertEqual(rep["cleanup_errors"], [])
                self.assertIn(("unload", "mtp"), log)
                self.assertIn(("unload", "text"), log)
            finally:
                helper.stop()

    def test_ls_execution_uses_plain_split_load_with_same_policy(self):
        with tempfile.TemporaryDirectory() as td:
            helper = tqmr.FakeHelper(Path(td) / "power.sock")
            helper.start()
            try:
                out = str(Path(td) / "ls.json")
                code, rep, log = self._case("ls", "mtp", str(Path(td) / "power.sock"), out,
                                            ngram_tables=True)
                self.assertEqual(code, 0, rep.get("error"))
                self.assertTrue(rep["complete"])
                self.assertEqual([e for e in log if e[0] == "load"],
                                 [("load", "mtp", {"device": "cuda:1", "max_chunk_size": 2048,
                                                   "progressbar": False}),
                                  ("load", "text", {"use_per_device": [28, 28],
                                                    "max_chunk_size": 2048, "progressbar": False})])
                self.assertEqual([e for e in log if e[0] in ("dispatch", "dispatch_single")], [])
                self.assertEqual(len([e for e in log if e[0] == "sync"]), 6)  # plain parent torch
                self.assertNotIn("power_rank_syncs", rep)
                self.assertEqual(rep["execution"]["actual"], "layer_split")
                self.assertTrue(rep["placement"]["ok"])
                self.assertEqual(rep["ngram"]["expected_keys"], [NGRAM_SHELL_KEY])
                self.assertTrue(rep["ngram"]["ok"])
                owner = rep["ngram"]["ram_owners"][NGRAM_SHELL_KEY][0]
                self.assertTrue(owner["residency"]["all_resident"])       # measured pre-inference
                self.assertEqual(owner["residency"]["probe"], "mincore")
                post = rep["ngram"]["residency_post_inference"]            # re-probed post-inference
                self.assertEqual([p["key"] for p in post], [NGRAM_SHELL_KEY])
                self.assertTrue(post[0]["all_resident"])
                self.assertNotIn("tp_audit", rep)
                self.assertNotIn("scope", rep["peak_allocated_bytes"])  # plain dict as before
            finally:
                helper.stop()

    def test_validate_finite_counts_collected_after_inference_and_must_be_positive(self):
        with tempfile.TemporaryDirectory() as td:
            helper = tqmr.FakeHelper(Path(td) / "power.sock")
            helper.start()
            try:
                out = str(Path(td) / "vf.json")
                code, rep, log = self._case("tp", "mtp", str(Path(td) / "power.sock"), out,
                                            extra=["--validate-finite"], dispatch=happy_dispatch())
                self.assertEqual(code, 0, rep.get("error"))
                self.assertTrue(rep["validation_only"])
                self.assertEqual([e[1] for e in log if e[0] == "dispatch"],
                                 ["tp_install_finite_hook", "tp_audit_rank", "tp_cpu_helper_meta",
                                  "tp_audit_rank", "tp_finite_hook_stats"])
                self.assertEqual([r["checks"] for r in
                                  rep["tp_final_audit"]["finite_hook_stats"]], [82, 82])
                self.assertEqual(rep["finite_forward_counts"], {"target": 0, "draft": 0})
                # zero counters must fail closed, never pass as a validated run
                zero = happy_dispatch()
                zero["tp_finite_hook_stats"] = [{"device": 0, "checks": 0},
                                                {"device": 1, "checks": 82}]
                out2 = str(Path(td) / "vf0.json")
                code, rep, _ = self._case("tp", "mtp", str(Path(td) / "power.sock"), out2,
                                          extra=["--validate-finite"], dispatch=zero)
                self.assertEqual(code, 1)
                self.assertIn("must be positive", rep["error"])
                self.assertFalse(rep["complete"])
            finally:
                helper.stop()

    def test_single_active_device_cannot_be_labelled_tp2(self):
        with tempfile.TemporaryDirectory() as td:
            helper = tqmr.FakeHelper(Path(td) / "power.sock")
            helper.start()
            try:
                out = str(Path(td) / "one.json")
                code, rep, log = self._case("tp", "mtp", str(Path(td) / "power.sock"), out,
                                            dispatch=happy_dispatch(), active=(0,))
                self.assertEqual(code, 1)
                self.assertIn("expected exactly [0, 1]", rep["error"])
                self.assertIn("must never be labelled TP2", rep["error"])
                self.assertEqual([e for e in log if e[0] == "dispatch"], [])
                self.assertIn(("unload", "text"), log)
            finally:
                helper.stop()

    def test_no_mtp_model_ar_runs_without_constructing_a_draft(self):
        with tempfile.TemporaryDirectory() as td:
            helper = tqmr.FakeHelper(Path(td) / "power.sock")
            helper.start()
            try:
                out = str(Path(td) / "nomtp.json")
                code, rep, log = self._case("tp", "ar", str(Path(td) / "power.sock"), out,
                                            with_mtp=False, ngram_shell=False,
                                            dispatch=happy_dispatch(ngram_owner_on_out=False))
                self.assertEqual(code, 0, rep.get("error"))
                self.assertFalse(rep["mtp_residency"]["component_present"])
                self.assertFalse(rep["mtp_residency"]["resident"])
                self.assertEqual(rep["ngram_expected_keys"], [])
                self.assertIn("no-op", " ".join(rep["tp_audit"]["notes"]))  # tableless D/M no-op
                self.assertEqual([e for e in log if e[0] == "from_config" and e[1] == "mtp"], [])
                self.assertEqual([e[1] for e in log if e[0] == "load"], ["text"])
                self.assertNotIn("draft", rep["cache_capacity"])
                self.assertFalse(rep["generator"]["mtp_draft"])
                self.assertEqual(rep["generator"]["num_draft_tokens"], 0)
            finally:
                helper.stop()

    def test_no_mtp_model_mtp_mode_fails_before_any_construction(self):
        with tempfile.TemporaryDirectory() as td:
            code, rep, log = self._case("tp", "mtp", str(Path(td) / "absent.sock"),
                                        str(Path(td) / "nomtp_mtp.json"), with_mtp=False)
            self.assertEqual(code, 1)
            self.assertIn("Qwen3-30B-A3B", rep["error"])
            self.assertEqual([e for e in log if e[0] in ("from_config", "load", "dispatch")], [])

    def test_tp_execution_reports_supports_tp_dependency(self):
        with tempfile.TemporaryDirectory() as td:
            code, rep, log = self._case("tp", "mtp", str(Path(td) / "absent.sock"),
                                        str(Path(td) / "supports.json"), supports_tp=False)
            self.assertEqual(code, 1)
            self.assertIn("supports_tp", rep["error"])
            self.assertIn("execution ls", rep["error"])      # names the working alternative
            self.assertEqual([e for e in log if e[0] == "load"], [])

    def test_vocab_and_capacity_fail_before_model_construction(self):
        vocab_prompts = tr.validate_prompts({"prompts": [entry([1, 2, 300000])]}, 1)
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "v.json")
            code, rep, log = self._case("tp", "mtp", str(Path(td) / "absent.sock"), out,
                                        prompts=vocab_prompts)
            self.assertEqual(code, 1)
            self.assertIn("vocabulary", rep["error"])
            self.assertEqual([e for e in log if e[0] in ("from_config", "load")], [])
        long_prompt = list(range(1, 501))
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "c.json")
            code, rep, log = self._case("tp", "mtp", str(Path(td) / "absent.sock"), out,
                                        prompts=tr.validate_prompts(
                                            {"prompts": [entry(long_prompt)]}, 1),
                                        extra=["--cache-tokens", "512"])
            self.assertEqual(code, 1)
            self.assertIn("cache tokens", rep["error"])
            self.assertEqual([e for e in log if e[0] in ("from_config", "load")], [])

    def test_audit_problems_fail_closed_but_cleanup_still_runs(self):
        bad = happy_dispatch()
        bad["tp_audit_rank"] = [rank_rec(0, 4001, ngram=OWNER_NG),
                                rank_rec(1, os.getpid(), ngram=OWNER_NG)]
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "auditfail.json")
            code, rep, log = self._case("tp", "mtp", str(Path(td) / "absent.sock"), out, dispatch=bad)
            self.assertEqual(code, 1)
            self.assertIn("TP worker audit failed", rep["error"])
            self.assertIn("RAM owners", rep["error"])
            self.assertEqual(rep["ngram_expected_keys"], [NGRAM_SHELL_KEY])
            self.assertFalse(rep["complete"])
            self.assertIn(("unload", "text"), log)
            self.assertEqual(rep["cleanup_errors"], [])
            self.assertTrue(Path(out).exists())

    def test_expected_engram_silently_absent_fails_closed_on_q(self):
        # THE former false pass at run level: a Qwen3.8 checkpoint (Engram shell
        # present pre-load) whose loaded ranks expose no table anywhere.
        with tempfile.TemporaryDirectory() as td:
            code, rep, log = self._case("tp", "mtp", str(Path(td) / "absent.sock"),
                                        str(Path(td) / "absent_ng.json"),
                                        dispatch=happy_dispatch(ngram_owner_on_out=False))
            self.assertEqual(code, 1)
            self.assertIn("TP worker audit failed", rep["error"])
            self.assertIn("exposed by NO rank", rep["error"])
            self.assertEqual(rep["ngram_expected_keys"], [NGRAM_SHELL_KEY])
            self.assertIn(("unload", "text"), log)

    def test_partial_tp_init_drains_spawned_workers_then_unloads(self):
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "partial.json")
            code, rep, log = self._case("tp", "mtp", str(Path(td) / "absent.sock"), out,
                                        load_exc_cls=RuntimeError, dispatch=happy_dispatch())
            self.assertEqual(code, 1)
            self.assertIn("injected failure", rep["error"])
            # loaded_tp was never set, so Model.unload() alone would leak the spawned
            # ranks: the CLI must destroy the half-built TP context explicitly.
            self.assertEqual([e for e in log if e[0] in ("dispatch", "dispatch_single")], [])
            self.assertIn(("destroy_tp",), log)
            self.assertLess(log.index(("destroy_tp",)), log.index(("unload", "text")))
            self.assertIn(("unload", "mtp"), log)
            self.assertEqual(rep["cleanup_errors"], [])
            self.assertFalse(rep["complete"])

    def test_failed_engine_teardown_still_joins_and_terminates_only_own_children(self):
        kids = [FChild(4001), PseudoChild(), FChild(4002)]     # dev0 rank, pseudo dev1, cpu slot
        conns = [FConn(), PseudoParentConn(), FConn()]
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "teardown.json")
            code, rep, log = self._case("tp", "mtp", str(Path(td) / "absent.sock"), out,
                                        load_exc_cls=RuntimeError, dispatch=happy_dispatch(),
                                        children=kids, mp_conns=conns, destroy_exc=RuntimeError)
            self.assertEqual(code, 1)
            self.assertIn(("destroy_tp",), log)
            # the failed engine destroy is surfaced, not swallowed...
            self.assertIn("tp worker drain: destroy_tp_context", " ".join(rep["cleanup_errors"]))
            # ...and the bounded manual fallback drained ONLY the real children
            self.assertTrue(kids[0].terminated and kids[0].joined == [2, 2])
            self.assertFalse(getattr(kids[1], "terminated", False))      # pseudo never killed
            self.assertTrue(kids[2].terminated)
            self.assertTrue(conns[0].closed and conns[2].closed)
            self.assertFalse(conns[1].closed)                            # pseudo conn untouched
            self.assertFalse(rep["complete"])

    def test_cache_hit_or_short_output_fails_closed(self):
        class TamperGen(tqmr._FakeGenerator):
            TAMPER = None

            def iterate(self):
                items = super().iterate()
                for it in items:
                    if it.get("eos") and TamperGen.TAMPER == "cached":
                        it["cached_tokens"] = 128
                    if it.get("eos") and TamperGen.TAMPER == "short":
                        it["new_tokens"] = it["new_tokens"] - 1
                return items

        with tempfile.TemporaryDirectory() as td:
            helper = tqmr.FakeHelper(Path(td) / "power.sock")
            helper.start()
            try:
                for tamper, needle in (("cached", "cache-miss"), ("short", "violate exact")):
                    with self.subTest(tamper):
                        TamperGen.TAMPER = tamper
                        out = str(Path(td) / f"tamp_{tamper}.json")
                        code, rep, log = self._case("ls", "mtp", str(Path(td) / "power.sock"),
                                                    out, gen_cls=TamperGen, ngram_tables=True)
                        self.assertEqual(code, 1)
                        self.assertIn(needle, rep["error"])
                        self.assertFalse(rep["complete"])
                        self.assertIn(("unload", "text"), log)
            finally:
                TamperGen.TAMPER = None
                helper.stop()


if __name__ == "__main__":
    unittest.main()
