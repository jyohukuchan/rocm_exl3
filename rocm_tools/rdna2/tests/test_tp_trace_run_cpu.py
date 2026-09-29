#!/usr/bin/env python3
"""CPU-only tests for the bounded TP trace wrapper's parent-side logic
(rocm_tools/rdna2/tp_trace_run.py): argv split, timed-group window
selection, the Generator.iterate hook (start/stop exactly once, group-end
stop, stop-on-error, never re-armed), the dispatch callbacks and the
rank-side forward tagging/restore.

The profiler starts with CPU activity only where exercised: the CUDA/HIP
activities are validated on hardware by the GPU run itself; here the worker
functions' ERROR path (profiler cannot start -> rank left exactly as found,
no orphaned state) is what pins the fail-safe contract.

Run: python3 -m pytest -q rocm_tools/rdna2/tests/test_tp_trace_run_cpu.py
"""
from __future__ import annotations

import contextlib
import json
import os
import pickle
import subprocess
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from rocm_tools.rdna2 import tp_trace_run as tr


def _mod(name):
    return types.ModuleType(name)


class FakeModel:
    def __init__(self, loaded_tp=True, fail=False):
        self.loaded_tp = loaded_tp
        self.active_devices = [0, 1]
        self.dispatches = []
        self.fail = fail

    def tp_worker_dispatch_wait_multi(self, devices, fn, args):
        self.dispatches.append((list(devices), fn.__name__, args))
        if self.fail:
            raise RuntimeError("worker dead")
        return [{"device": d, "pid": 100 + d} for d in devices]


class FakeGen:
    """num_remaining_jobs() pops a scripted sequence: the hook reads it once
    before and once after every wrapped call, so rems alternates
    (before, after) per iterate."""

    def __init__(self, rems, model=None):
        self.rems = list(rems)
        self.model = model if model is not None else FakeModel()

    def num_remaining_jobs(self):
        return self.rems.pop(0) if self.rems else 0


def fake_orig(store):
    def orig(self, *a, **kw):
        store.append("ran")
        return "items"
    return orig


class SplitAndGroupTests(unittest.TestCase):
    def test_split_argv(self):
        w, h = tr.split_argv(["--trace-dir", "d", "--", "-m", "M", "--mode", "ar"])
        self.assertEqual(w, ["--trace-dir", "d"])
        self.assertEqual(h, ["-m", "M", "--mode", "ar"])
        with self.assertRaises(ValueError):
            tr.split_argv(["--trace-dir", "d"])

    def test_first_timed_group(self):
        p = [{"timed": False}, {"timed": False}, {"timed": True}, {"timed": True}]
        self.assertEqual(tr.first_timed_group(p, 2), 1)      # bs2: group 1 is first timed
        self.assertEqual(tr.first_timed_group(p, 1), 2)      # bs1: index 2
        with self.assertRaises(ValueError):
            tr.first_timed_group([{"timed": False}], 1)


class WindowStateTests(unittest.TestCase):
    def test_only_window_group_arms(self):
        st = tr.WindowState(window_group=1, max_iterates=2)
        st.pre_iterate(3)                                    # group 0 (warmup) starts
        self.assertFalse(st.want_start)
        st.post_iterate(2)
        st.pre_iterate(2)
        st.post_iterate(0)                                   # warmup group drained
        st.pre_iterate(0)                                    # gap between groups
        self.assertFalse(st.want_start)                      # no jobs: no new group
        st.post_iterate(0)
        st.pre_iterate(5)                                    # 0 -> 5: group 1 begins
        self.assertTrue(st.want_start)

    def test_stop_on_count_and_group_end(self):
        st = tr.WindowState(1, 2)
        st.active = True
        st.post_iterate(4)
        self.assertFalse(st.want_stop)
        st.post_iterate(3)
        self.assertTrue(st.want_stop)                        # 2 window iterates
        st2 = tr.WindowState(1, 8)
        st2.active = True
        st2.post_iterate(0)
        self.assertTrue(st2.want_stop)                       # group ended early


class FakeJob:
    def __init__(self, new_tokens=0):
        self.new_tokens = new_tokens


class FakeDecodeGen:
    """num_remaining_jobs() pops alternately (before, after) like FakeGen;
    active_jobs reflects what decode-window read_tokens() must see."""

    def __init__(self, rems, model=None, jobs=()):
        self.rems = list(rems)
        self.model = model if model is not None else FakeModel()
        self.active_jobs = list(jobs)

    def num_remaining_jobs(self):
        return self.rems.pop(0) if self.rems else 0


def decode_orig(seq):
    """state.orig: every call applies the next scripted token state. None
    drains the job list (finished/removed); a fresh job activates on first
    non-None value (mirrors prefill completing and producing token 1)."""
    def orig(self, *a, **kw):
        t = seq.pop(0)
        if t is None:
            self.active_jobs = []
        elif not self.active_jobs:
            self.active_jobs = [FakeJob(t)]
        else:
            self.active_jobs[0].new_tokens = t
        return "items"
    return orig


class DecodeWindowStateTests(unittest.TestCase):
    def test_ar_exact_bounds_prefill_excluded(self):
        # warmup group (never armed), gap, then timed group: tokens step 1
        # (AR); start fires when PRE-iterate count already >= 3.
        rems = [1,0, 0,0, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,0]
        seq = [10, None, 1, 2, 3, 4, 5, 6]
        gen = FakeDecodeGen(rems)
        st = tr.DecodeWindowState(1, 3, 3)
        st.orig = decode_orig(seq)
        starts, stops = [], []
        def on_start():
            starts.append(1); st.active = True; return True
        def on_stop():
            stops.append(1)
        hook = tr.make_iterate_hook(st, on_start, on_stop)
        def no_sync(idx):
            raise AssertionError("no per-iterate/per-token sync is allowed here")
        with mock.patch.object(tr, "_sync_device", no_sync):
            for _ in range(8):
                hook(gen)
        self.assertEqual((starts, stops), ([1], [1]))
        self.assertEqual(st.window_calls, [6, 7, 8])   # calls 1-5 are warmup/gap/prefill decodes
        self.assertEqual(st.start_actual, 3)
        self.assertEqual(st.end_actual, 6)
        self.assertEqual(st.window_iterates, 3)
        self.assertFalse(st.truncated)
        self.assertEqual(st.errors, [])

    def test_mtp_burst_overshoot_records_actual(self):
        # window_group 0; bursts: 1->2, 2->5 (start ACTUAL 5 > requested 3),
        # 5->7, 7->10: additional 5 >= window 4, end ACTUAL 10 (overshoot 1)
        rems = [1,1, 1,1, 1,1, 1,1, 1,1, 1,0]
        seq = [2, 5, 7, 10]
        gen = FakeDecodeGen(rems)
        st = tr.DecodeWindowState(0, 3, 4)
        st.orig = decode_orig(seq)
        starts, stops = [], []
        def on_start():
            starts.append(1); return True
        def on_stop():
            stops.append(1)
        hook = tr.make_iterate_hook(st, on_start, on_stop)
        for _ in range(4):
            hook(gen)
        self.assertEqual((starts, stops), ([1], [1]))
        self.assertEqual(st.start_actual, 5)           # recorded actual, not requested
        self.assertEqual(st.end_actual, 10)
        self.assertEqual(st.window_iterates, 2)        # iterates 3 and 4
        self.assertFalse(st.truncated)

    def test_legacy_trace_iterations_do_not_stop_decode_window(self):
        st = tr.DecodeWindowState(0, 1, 10)
        self.assertFalse(hasattr(st, "max_iterates"))
        st.armed = True
        st.active = True
        st.start_actual = 1
        for t in range(2, 11):                         # 9 iterates, additional 9 < 10
            st.post_iterate(1, t)
            self.assertFalse(st.want_stop)
        st.post_iterate(1, 11)                         # additional 10: stop
        self.assertTrue(st.want_stop)

    def test_group_end_truncation_flagged(self):
        st = tr.DecodeWindowState(0, 3, 4)
        st.armed = True
        st.active = True
        st.start_actual = 3
        st.post_iterate(0, 5)                          # drained at additional 2 < 4
        self.assertTrue(st.truncated)
        self.assertTrue(st.want_stop)

    def test_ambiguous_job_selection_refused(self):
        gen = FakeDecodeGen([1,1], jobs=[FakeJob(5), FakeJob(7)])
        st = tr.DecodeWindowState(0, 1, 4)
        st.orig = decode_orig([])
        self.assertIsNone(st.read_tokens(gen))
        self.assertTrue(st.done)
        self.assertTrue(any("ambiguous" in e for e in st.errors))
        st.pre_iterate(1, None)
        self.assertFalse(st.want_start)                # refuses, never guesses

    def test_warmup_group_high_tokens_never_arms(self):
        st = tr.DecodeWindowState(1, 3, 3)
        st.gen = FakeDecodeGen([])
        st.pre_iterate(1, None)                        # group 0 (warmup) begins
        st.post_iterate(0, 100)                        # warmup drains far past start
        st.pre_iterate(0, None)
        self.assertFalse(st.want_start)
        self.assertEqual(st.group, 0)


class HookTests(unittest.TestCase):
    def _hook(self, state, starts, stops, start_ok=True):
        def on_start():
            starts.append(1)
            return True if start_ok else False
        def on_stop():
            stops.append(1)
        return tr.make_iterate_hook(state, on_start, on_stop)

    def test_start_stop_once_around_window(self):
        # groups: warmup(2 jobs), window(3 jobs, max_iterates=2), third group
        rems = [2,2, 2,1, 1,0,  3,2, 2,1, 1,0,  1,0]
        gen = FakeGen(rems)
        st = tr.WindowState(1, 2)
        st.orig = fake_orig([])
        starts, stops = [], []
        hook = self._hook(st, starts, stops)
        for _ in range(5):
            hook(gen)
        self.assertEqual(starts, [1])                        # started once
        self.assertEqual(stops, [1])                         # stopped once
        self.assertEqual(st.window_calls, [4, 5])            # window = iterates 4-5
        self.assertFalse(st.active)
        self.assertEqual(hook(gen), "items")                 # group 2 stays untraced
        self.assertEqual((starts, stops), ([1], [1]))

    def test_group_end_stops_when_shorter_than_max(self):
        gen = FakeGen([3,2, 2,1, 1,0, 0,0])
        st = tr.WindowState(0, 8)                            # window == first group
        st.orig = fake_orig([])
        starts, stops = [], []
        hook = self._hook(st, starts, stops)
        for _ in range(3):
            hook(gen)
        self.assertEqual((starts, stops), ([1], [1]))
        self.assertEqual(st.window_calls, [1, 2, 3])

    def test_error_inside_window_stops_and_reraises(self):
        gen = FakeGen([3, 0])                                # only the pre-read happens
        st = tr.WindowState(0, 8)

        def boom(self, *a, **kw):
            raise RuntimeError("generation died")
        st.orig = boom
        starts, stops = [], []
        hook = self._hook(st, starts, stops)
        with self.assertRaises(RuntimeError):
            hook(gen)
        self.assertEqual((starts, stops), ([1], [1]))        # stopped before propagating
        self.assertFalse(st.active)

    def test_failed_start_never_stops_or_tracks(self):
        gen = FakeGen([3,3, 3,3])
        st = tr.WindowState(0, 8)
        st.orig = fake_orig([])
        starts, stops = [], []
        hook = self._hook(st, starts, stops, start_ok=False)
        hook(gen)
        hook(gen)
        self.assertEqual(stops, [])                          # nothing was started
        self.assertFalse(st.active)
        self.assertTrue(st.done)


class DispatchCallbackTests(unittest.TestCase):
    def test_start_cb_dispatches_named_worker_and_records(self):
        st = tr.WindowState(0, 4)
        st.spec = {"trace_dir": "/tmp/x"}
        st.gen = FakeGen([])
        self.assertTrue(tr._start_cb(st))
        devs, fname, args = st.gen.model.dispatches[0]
        self.assertEqual((devs, fname), ([0, 1], "tp_trace_start_worker"))
        self.assertEqual(args, ({"trace_dir": "/tmp/x"},))
        self.assertEqual(st.records["start"], [{"device": 0, "pid": 100},
                                               {"device": 1, "pid": 101}])
        self.assertIsNone(tr._stop_cb(st))
        self.assertEqual(st.gen.model.dispatches[1][1], "tp_trace_stop_worker")

    def test_start_cb_refuses_non_tp_model(self):
        st = tr.WindowState(0, 4)
        st.gen = FakeGen([], model=FakeModel(loaded_tp=False))
        self.assertIs(tr._start_cb(st), False)
        self.assertTrue(any("loaded_tp" in e for e in st.errors))

    def test_start_cb_survives_dead_workers(self):
        st = tr.WindowState(0, 4)
        st.spec = {"trace_dir": "/tmp/x"}
        st.gen = FakeGen([], model=FakeModel(fail=True))
        self.assertIs(tr._start_cb(st), False)
        self.assertTrue(any("dispatch failed" in e for e in st.errors))


class TaggingTests(unittest.TestCase):
    class Mod:
        def __init__(self, key):
            self.key = key
        def forward(self, x, params=None):
            return 42

    def test_tag_call_untag(self):
        m = self.Mod("model.layers.0.linear_attn")
        try:
            handles = []
            tr._tag_module_forwards({"modules": [m]}, handles)
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                self.assertEqual(m.forward(torch.zeros(1, 7, 4)), 42)
                self.assertEqual(m.forward(torch.zeros(1, 1, 4)), 42)
            names = [e.name() if callable(e.name) else e.name for e in prof.events()]
            joined = " ".join(names)
            self.assertIn("model.layers.0.linear_attn seq=7", joined)
            self.assertIn("model.layers.0.linear_attn seq=1", joined)
            self.assertIn("forward", m.__dict__)             # shadowed while tagged
            tr._restore_module_forwards(handles)
            self.assertNotIn("forward", m.__dict__)          # instance shadow gone
            self.assertEqual(m.forward(torch.zeros(1, 7, 4)), 42)
        finally:
            tr._ACTIVE.clear()

    def test_pre_existing_instance_forward_restored(self):
        m = self.Mod("k")
        prev = lambda *a, **kw: "previous"
        m.forward = prev
        handles = []
        tr._tag_module_forwards({"modules": [m]}, handles)
        tr._restore_module_forwards(handles)
        self.assertIs(m.forward, prev)

    def test_start_worker_error_path_leaves_rank_clean(self):
        if torch.cuda.is_available():
            self.skipTest("CPU-only host check: on CUDA boxes the profiler starts")
        m = self.Mod("k")
        # CUDA activity cannot start here: the worker must report the error and
        # leave modules untouched, with no orphaned _ACTIVE state
        out = tr.tp_trace_start_worker({"device": None, "modules": [m],
                                        "output_device": 1}, {"trace_dir": "/tmp"})
        self.assertIn("error", out)
        self.assertNotIn("forward", m.__dict__)
        self.assertEqual(tr._ACTIVE, {})
        stop = tr.tp_trace_stop_worker({"device": None})
        self.assertIn("error", stop)                         # no orphaned profiler


class FakeRoctx:
    def __init__(self, fail=None):
        self.ops = []
        self.fail = set(fail or ())        # mutable: tests add/remove entries

    def _do(self, op):
        if op.split(":", 1)[0] in self.fail:
            raise RuntimeError(f"boom {op}")
        self.ops.append(op)

    def pause(self): self._do("pause")
    def resume(self): self._do("resume")
    def push(self, label): self._do(f"push:{label}")
    def pop(self): self._do("pop")


@contextlib.contextmanager
def _rank_state(controls):
    """Swap the per-process globals the roctx workers use, restore after."""
    saved = (dict(tr._ROCTX), dict(tr._ACTIVE))
    tr._ROCTX.clear(); tr._ROCTX.update({"controls": controls, "status": "paused"})
    tr._ACTIVE.clear()
    try:
        with mock.patch.object(tr, "_sync_device", lambda idx: None):
            yield
    finally:
        tr._ROCTX.clear(); tr._ROCTX.update(saved[0])
        tr._ACTIVE.clear(); tr._ACTIVE.update(saved[1])


class RoctxWorkerTests(unittest.TestCase):
    class Mod:
        def __init__(self, key):
            self.key = key
        def forward(self, x, params=None):
            return 42

    def test_start_then_stop_operation_order(self):
        rx = FakeRoctx()
        m = self.Mod("model.layers.0.mlp")
        ctx = {"device": 0, "modules": [m], "output_device": 1}
        spec = {"range": "W", "trace_dir": "/tmp"}
        with _rank_state(rx):
            start = tr.tp_trace_roctx_start_worker(ctx, spec)
            self.assertEqual(start["tagged_modules"], 1)
            self.assertNotIn("error", start)
            self.assertEqual(rx.ops, ["resume", "push:W rank0 pid%d" % os.getpid()])
            self.assertEqual(tr._ROCTX["status"], "paused")   # gate reported, not re-done
            # nested module markers land INSIDE the open window range
            m.forward(torch.zeros(1, 5, 2))
            self.assertEqual(rx.ops[2:], ["push:model.layers.0.mlp seq=5", "pop"])
            stop = tr.tp_trace_roctx_stop_worker(ctx)
            self.assertEqual(rx.ops[-2:], ["pop", "pause"])   # sync->pop->pause order
            self.assertEqual(stop["window_s"] >= 0, True)
            self.assertNotIn("forward", m.__dict__)           # wrappers restored
            self.assertEqual(tr._ACTIVE, {})

    def test_start_failure_regates_and_restores(self):
        rx = FakeRoctx(fail={"push"})
        m = self.Mod("k")
        with _rank_state(rx):
            out = tr.tp_trace_roctx_start_worker({"device": 0, "modules": [m],
                                                  "output_device": 0}, {"range": "W"})
            self.assertIn("error", out)
            # resume happened, push died: pop is skipped (no open range) but the
            # rank is RE-GATED so a half-open window never leaks into the trace
            self.assertEqual(rx.ops, ["resume", "pause"])
            self.assertNotIn("forward", m.__dict__)
            self.assertEqual(tr._ACTIVE, {})
            stop = tr.tp_trace_roctx_stop_worker({"device": 0})
            self.assertIn("error", stop)                      # ...and stop stays inert
            self.assertEqual(rx.ops, ["resume", "pause"])     # nothing further toggled

    def test_stop_survives_errors_and_still_restores(self):
        rx = FakeRoctx()
        m = self.Mod("k")
        with _rank_state(rx):
            tr.tp_trace_roctx_start_worker({"device": 0, "modules": [m],
                                            "output_device": 0}, {"range": "W"})
            rx.fail.add("pop")
            stop = tr.tp_trace_roctx_stop_worker({"device": 0})
            self.assertIn("pop_error", stop)
            self.assertEqual(stop.get("paused_after_window"), True)  # pause still ran
            self.assertNotIn("forward", m.__dict__)                  # restored regardless


class RoctxNestedTaggingTests(unittest.TestCase):
    """Milestone-2 nested labels: bounded class traversal, dedup against the
    top-level wrappers, method-level tags (hc/ple/ngram), thread-local ranges
    and exact restoration. FakeRoctx proves pairing, not the ctypes shim."""

    class Block:                          # stand-in TransformerBlock
        def __init__(self, key, *subs):
            self.key = key
            self.modules = list(subs)
        def forward(self, x, params):
            return 7

    class Attention:
        qsa_indexer = None
        def __init__(self, key):
            self.key = key
            self.modules = []
        def forward(self, x, params):
            return 8

    class Linear:                         # by NAME: traversal must not tag
        def __init__(self, key):          # or descend into it
            self.key = key
            self.norm = RoctxNestedTaggingTests.RMSNorm("inner.norm")
            self.modules = [self.norm]
        def forward(self, x, params):
            return 9

    class RMSNorm:
        def __init__(self, key):
            self.key = key
            self.modules = []
        def forward(self, x, params):
            return 10

    class MLP:
        def __init__(self, key):
            self.key = key
            self.modules = []
        def forward(self, x, params):
            return 11

    class BlockSparseMLP(MLP):
        def __init__(self, key, shared):
            super().__init__(key)
            self.shared_experts = shared

    class PLELayer:
        def __init__(self, key, ngram):
            self.key = key
            self.modules = [ngram]
        def forward(self, x, params):
            return 12
        def prefetch(self, ids, params):
            return 13

    class NGramEmbedding:
        def __init__(self, key):
            self.key = key
            self.modules = []
        def forward(self, x, params):
            return 14
        def _stage(self, history, pin):
            return self._gather_rows(torch.zeros(3), None)
        def _gather_rows(self, uids, out):
            return 15

    class GatedResidual:
        def __init__(self, key):
            self.key = key
            self.modules = []
        def mix(self, streams, params):
            return "mixed"
        def apply_(self, x, y, post, comb, params):
            return "applied"
        def forward(self, x, params):
            return 16

    def _tags(self, modules, rx, prefix=""):
        handles = []
        n = tr._tag_nested_roctx(modules, rx, handles, prefix=prefix)
        return handles, n

    def _labels(self, rx, start=0):
        return [op for op in rx.ops[start:] if op.startswith("push:")]

    def test_class_labels_and_subclass_and_qsa(self):
        rx = FakeRoctx()
        attn = self.Attention("model.layers.0.self_attn")
        qsa = type("QwenAttn", (self.Attention,), {})(                # subclass
            "model.layers.1.self_attn")
        qsa.qsa_indexer = object()
        norm = self.RMSNorm("model.layers.0.input_layernorm")
        lin = self.Linear("model.layers.0.self_attn.q_proj")
        blk = self.Block("model.layers.0", attn, qsa, norm, lin)
        handles, n = self._tags([blk], rx)
        self.assertEqual(n, 3)                                        # 2 attn + 1 norm; Linear skipped
        blk.forward(torch.zeros(1, 4, 2), {})
        self.assertEqual(self._labels(rx), [])                        # block itself untagged
        self.assertEqual(attn.forward(torch.zeros(1, 4, 2), {}), 8)
        self.assertEqual(qsa.forward(torch.zeros(1, 4, 2), {}), 8)
        self.assertEqual(norm.forward(torch.zeros(1, 4, 2), {}), 10)
        self.assertEqual(lin.norm.forward(torch.zeros(1, 4, 2), {}), 10)   # inside Linear: NOT wrapped
        self.assertEqual(self._labels(rx), [
            "push:attn model.layers.0.self_attn seq=4",
            "push:attn.qsa model.layers.1.self_attn seq=4",
            "push:norm model.layers.0.input_layernorm seq=4"])
        tr._restore_roctx_handles(handles)
        for m in (attn, qsa, norm, lin.norm):                         # shadows gone, class intact
            self.assertNotIn("forward", m.__dict__)
        self.assertEqual(attn.forward(torch.zeros(1, 4, 2), {}), 8)

    def test_moe_shared_expert_when_exposed(self):
        rx = FakeRoctx()
        shared = self.MLP("model.layers.0.mlp.shared_experts")
        moe = self.BlockSparseMLP("model.layers.0.mlp", shared)
        handles, n = self._tags([moe], rx)
        self.assertEqual(n, 2)                                        # moe.forward + moe.shared.forward
        shared.forward(torch.zeros(1, 2, 2), {})
        self.assertEqual(self._labels(rx),
                         ["push:moe.shared model.layers.0.mlp.shared_experts seq=2"])
        self.assertEqual(moe.forward(torch.zeros(1, 2, 2), {}), 11)
        self.assertEqual(self._labels(rx)[1:],
                         ["push:moe model.layers.0.mlp seq=2"])
        tr._restore_roctx_handles(handles)
        self.assertNotIn("forward", shared.__dict__)
        self.assertNotIn("forward", moe.__dict__)
        nos = self.BlockSparseMLP("k", None)
        h2, n2 = self._tags([nos], rx)
        self.assertEqual((n2, len(h2)), (1, 1))                       # shared None: only moe

    def test_hc_site_methods_and_ple_ngram_nesting(self):
        rx = FakeRoctx()
        ng = self.NGramEmbedding("model.layers.0.ple.ple_embedding")
        ple = self.PLELayer("model.layers.0.ple", ng)
        hc = self.GatedResidual("model.layers.0.hc_attn")
        handles, n = self._tags([ple, hc], rx)
        self.assertEqual(n, 5 + 3)          # ple fwd+prefetch, ngram fwd/_stage/_gather_rows, hc fwd/mix/apply_
        self.assertEqual(ple.prefetch(torch.zeros(1, 3), {}), 13)
        self.assertEqual(hc.mix(torch.zeros(1, 3, 2, 4), {}), "mixed")
        self.assertEqual(hc.apply_(torch.zeros(1, 3, 2, 4), None, None, None, {}), "applied")
        ng._stage(torch.zeros(1, 9), None)              # gather nests INSIDE stage
        self.assertEqual(self._labels(rx), [
            "push:ple.prefetch model.layers.0.ple seq=3",
            "push:hc.mix model.layers.0.hc_attn seq=3",
            "push:hc.apply_ model.layers.0.hc_attn seq=3",
            "push:ngram._stage model.layers.0.ple.ple_embedding seq=9",
            "push:ngram._gather_rows model.layers.0.ple.ple_embedding seq=3"])
        self.assertEqual(rx.ops.count("pop"), 5)
        # _stage on a foreign thread: the thread-local push/pop still pairs
        t = threading.Thread(target=lambda: ng._stage(torch.zeros(1, 9), None))
        t.start(); t.join()
        self.assertEqual(rx.ops.count("push:ngram._stage "
                                      "model.layers.0.ple.ple_embedding seq=9"), 2)
        tr._restore_roctx_handles(handles)
        for m, attrs in ((ple, ("forward", "prefetch")),
                         (ng, ("forward", "_stage", "_gather_rows")),
                         (hc, ("forward", "mix", "apply_"))):
            for a in attrs:
                self.assertNotIn(a, m.__dict__)

    def test_dedup_against_toplevel_wrappers(self):
        rx = FakeRoctx()
        ng = self.NGramEmbedding("embed.ple")
        ple = self.PLELayer("model.ple", ng)            # PLELayer itself top-level
        handles = []
        tr._tag_module_forwards_roctx({"modules": [ple]}, rx, handles)
        n = tr._tag_nested_roctx([ple], rx, handles)
        self.assertEqual(n, 4)      # ple.forward deduped; prefetch + 3 ngram attrs remain
        self.assertEqual(ple.forward(torch.zeros(1, 5, 2), {}), 12)
        self.assertEqual(self._labels(rx), ["push:model.ple seq=5"])   # ONE range, top-level label
        tr._restore_roctx_handles(handles)
        self.assertNotIn("forward", ple.__dict__)
        self.assertNotIn("prefetch", ple.__dict__)
        self.assertNotIn("_stage", ng.__dict__)

    def test_prefix_for_draft_labels(self):
        rx = FakeRoctx()
        attn = self.Attention("trunk.layers.0.attn")
        handles, n = self._tags([attn], rx, prefix="draft.")
        attn.forward(torch.zeros(1, 2, 2), {})
        self.assertEqual(self._labels(rx), ["push:draft.attn trunk.layers.0.attn seq=2"])

    def test_callable_passthrough_args_returns_errors(self):
        rx = FakeRoctx()
        ng = self.NGramEmbedding("k")
        sentinel = object()
        def fwd(x, params, y=sentinel):                 # kw + multi-return
            return (x, params, y)
        ng.forward = fwd                                # PRE-EXISTING instance shadow
        handles, _ = self._tags([ng], rx)               # wraps shadow, prev=fwd
        self.assertEqual(ng.forward(1, 2), (1, 2, sentinel))   # args/kw/return pass through
        self.assertEqual(self._labels(rx), ["push:ngram k seq=?"])    # fwd -> plain base; non-tensor -> "?"
        tr._restore_roctx_handles(handles)
        self.assertIs(ng.forward, fwd)                  # exact previous restored
        def boom(*a, **kw):
            raise RuntimeError("stage died")
        ng._stage = boom
        h2, _ = self._tags([ng], rx)
        with self.assertRaises(RuntimeError):
            ng._stage(torch.zeros(1, 4), None)          # exception propagates...
        self.assertEqual(rx.ops.count("pop"), 2)        # ...both ranges closed
        self.assertEqual(rx.ops[-2:], ["push:ngram._stage k seq=4", "pop"])
        tr._restore_roctx_handles(h2)
        self.assertIs(ng._stage, boom)


class RoctxCommTaggingTests(unittest.TestCase):
    """tp.<op> comm labels: metadata only (numel x element_size, never
    .item()), behavior pass-through, exact restoration."""

    class T:
        def __init__(self, numel, esz=2):
            self._n, self._e = numel, esz
        def numel(self):
            return self._n
        def element_size(self):
            return self._e

    class Backend:
        def fwd_barrier(self):
            return "bar"
        def broadcast(self, tensor, src_device):
            return ("bcast", tensor, src_device)
        def all_reduce(self, tensor, contribution=True):
            if contribution:
                raise RuntimeError("reduce wire died")
            return ("red", tensor)
        def gather(self, tensor, out, devices, dev, ldims):
            return ("gather", out)
        def gather_small(self, tensor, out, devices, dev, ldims):
            return ("small", out)

    def test_ops_labels_passthrough_restore(self):
        rx = FakeRoctx()
        b = self.Backend()
        handles = []
        n = tr._tag_backend_roctx(b, rx, handles)
        self.assertEqual(n, 5)
        t, o = self.T(4096), self.T(9000)
        self.assertEqual(b.fwd_barrier(), "bar")
        self.assertEqual(b.broadcast(t, 1), ("bcast", t, 1))
        self.assertEqual(b.gather(t, o, None, 0, [1]), ("gather", o))
        with self.assertRaises(RuntimeError):
            b.all_reduce(t)                             # exception propagates...
        self.assertEqual(rx.ops, [                      # labels + balanced ranges
            "push:tp.fwd_barrier", "pop",
            "push:tp.broadcast 4096x2B=8192B", "pop",
            "push:tp.gather 4096x2B=8192B+9000x2B=18000B", "pop",
            "push:tp.all_reduce 4096x2B=8192B", "pop"])  # ...closed by the raise
        tr._restore_roctx_handles(handles)
        self.assertEqual(b.__dict__, {})                # instance shadows removed
        self.assertEqual(b.all_reduce(o, False), ("red", o))   # class methods intact

    def test_preexisting_instance_attr_restored_and_no_tensor_meta(self):
        rx = FakeRoctx()
        b = self.Backend()
        prev = lambda: "instance-shadow"
        b.fwd_barrier = prev
        handles = []
        tr._tag_backend_roctx(b, rx, handles)
        self.assertEqual(b.gather_small(self.T(3), None, None, 0, None),
                         ("small", None))               # None out: no second meta
        self.assertEqual(rx.ops[-2], "push:tp.gather_small 3x2B=6B")
        tr._restore_roctx_handles(handles)
        self.assertIs(b.fwd_barrier, prev)

    def test_start_worker_tags_nested_and_comm(self):
        rx = FakeRoctx()
        attn = RoctxNestedTaggingTests.Attention("model.layers.0.self_attn")
        blk = RoctxNestedTaggingTests.Block("model.layers.0", attn)
        b = RoctxCommTaggingTests.Backend()
        ctx = {"device": 0, "modules": [blk], "backend": b, "output_device": 0}
        with _rank_state(rx):
            start = tr.tp_trace_roctx_start_worker(ctx, {"range": "W"})
            self.assertEqual((start["tagged_modules"], start["tagged_nested"],
                              start["tagged_comm"]), (1, 1, 5))
            b.fwd_barrier()
            attn.forward(torch.zeros(1, 1, 2), {})
            self.assertEqual(rx.ops[2:],
                             ["push:tp.fwd_barrier", "pop",
                              "push:attn model.layers.0.self_attn seq=1", "pop"])
            stop = tr.tp_trace_roctx_stop_worker(ctx)
            self.assertNotIn("error", stop)
            for m in (blk, attn):
                self.assertNotIn("forward", m.__dict__)
            self.assertEqual(b.__dict__, {})


class WorkerPhaseTests(unittest.TestCase):
    """Milestone-2 worker phase attribution: the three module-level wrappers
    are top-level (PICKLABLE by reference), call the ORIGINAL canonical
    model_tp_fn functions, label with the rank's own _ACTIVE rx and restore
    every replaced alias exactly. exllamav3 is faked via sys.modules; the
    canonical module is never written to (getattr-only contract)."""

    def _engine(self):
        calls = []

        def canon(name, ret=None, boom=False):
            def fn(lc, *a, **kw):
                calls.append((name, lc, a, kw))
                if boom:
                    raise RuntimeError("%s died" % name)
                return ret
            return fn

        fnmod = _mod("exllamav3.model.model_tp_fn")
        tpmod = _mod("exllamav3.model.model_tp")
        originals = {}
        for alias in ("mp_model_forward", "mp_model_forward_embedding",
                      "mp_model_forward_lm_head_argmax"):
            originals[alias] = canon(alias, ret=("R", alias))
            setattr(fnmod, alias, originals[alias])
            setattr(tpmod, alias, originals[alias])      # star-import alias
        pkg = _mod("exllamav3.model")
        pkg.model_tp, pkg.model_tp_fn = tpmod, fnmod
        exl = _mod("exllamav3")
        exl.model = pkg
        mods = {"exllamav3": exl, "exllamav3.model": pkg,
                "exllamav3.model.model_tp": tpmod,
                "exllamav3.model.model_tp_fn": fnmod}
        return mock.patch.dict(sys.modules, mods), calls, originals, tpmod

    def test_wrappers_top_level_picklable(self):
        for _alias, fn in tr._ROCTX_WORKER_ALIASES:
            self.assertEqual(fn.__module__.rsplit(".", 1)[-1], "tp_trace_run")
            self.assertIs(getattr(tr, fn.__qualname__), fn)      # resolvable
            self.assertIs(pickle.loads(pickle.dumps(fn)), fn)    # by reference

    def test_wrapper_labels_passthrough_and_no_rx(self):
        patcher, calls, originals, _tp = self._engine()
        rx = FakeRoctx()
        with patcher, mock.patch.dict(tr._ACTIVE, {"rx": rx}, clear=True):
            for fn, label, alias in (
                    (tr.mp_model_forward_target, "phase=target.forward",
                     "mp_model_forward"),
                    (tr.mp_model_forward_mtp_embedding,
                     "phase=mtp.borrowed_embedding", "mp_model_forward_embedding"),
                    (tr.mp_model_forward_mtp_head, "phase=mtp.borrowed_head",
                     "mp_model_forward_lm_head_argmax")):
                lc = {"device": 0}
                self.assertEqual(fn(lc, "x", {"p": 1}, 7), ("R", alias))
                self.assertEqual(calls[-1], (alias, lc, ("x", {"p": 1}, 7), {}))
                self.assertEqual(rx.ops[-2:], ["push:" + label, "pop"])
        with patcher:                                      # window NOT open here:
            rx2 = FakeRoctx()                              # plain passthrough,
            with mock.patch.dict(tr._ACTIVE, {}, clear=True):
                self.assertEqual(tr.mp_model_forward_target({"d": 1}, "x"),
                                 ("R", "mp_model_forward"))
            self.assertEqual(rx2.ops, [])
        boom = _mod("exllamav3.model.model_tp_fn")         # worker-side exception:

        def raiser(lc, *a, **kw):
            raise RuntimeError("forward wedged")
        boom.mp_model_forward = raiser
        pkg = _mod("exllamav3.model")
        pkg.model_tp_fn = boom
        pkg.model_tp = _mod("x")
        exl = _mod("exllamav3")
        exl.model = pkg
        with mock.patch.dict(sys.modules, {"exllamav3": exl,
                                           "exllamav3.model": pkg,
                                           "exllamav3.model.model_tp_fn": boom}), \
                mock.patch.dict(tr._ACTIVE, {"rx": FakeRoctx()}, clear=True):
            rxr = tr._ACTIVE["rx"]
            with self.assertRaises(RuntimeError):
                tr.mp_model_forward_target({}, "x")        # propagates...
            self.assertEqual(rxr.ops, ["push:phase=target.forward", "pop"])

    def test_install_parent_roctx_labels_draft_gen_and_aliases(self):
        class Blk(RoctxNestedTaggingTests.Block):
            pass
        class Attn(RoctxNestedTaggingTests.Attention):
            pass
        patcher, _calls, originals, tpmod = self._engine()
        blk = Blk("trunk.layers.0", Attn("trunk.layers.0.attn"))
        draft = _mod("draft"); draft.modules = [blk]

        class Gen:
            def __init__(self):
                self.draft_model = draft
            def iterate_gen(self, results, draft_tokens=None):
                return "verified"
            def iterate_draftmodel_mtp_gen(self, results):
                return "draft-tokens"
        gen = Gen()
        st = tr.WindowState(0, 4)
        st.gen, st.spec = gen, {"backend": "roctx"}
        rx = FakeRoctx()
        try:
            with patcher, mock.patch.dict(tr._ACTIVE,
                                          {"rx": rx, "handles": []}, clear=True):
                self.assertIsNone(tr._install_parent_roctx(st))
                self.assertIs(tpmod.mp_model_forward, tr.mp_model_forward_target)
                self.assertIs(tpmod.mp_model_forward_embedding,
                              tr.mp_model_forward_mtp_embedding)
                self.assertIs(tpmod.mp_model_forward_lm_head_argmax,
                              tr.mp_model_forward_mtp_head)
                self.assertEqual(gen.iterate_draftmodel_mtp_gen([]), "draft-tokens")
                self.assertEqual(gen.iterate_gen([]), "verified")
                self.assertEqual(blk.forward(torch.zeros(1, 1, 2), {}), 7)
                self.assertEqual(blk.modules[0].forward(torch.zeros(1, 1, 2), {}), 8)
                self.assertEqual(rx.ops, [
                    "push:phase=mtp.draft", "pop",
                    "push:phase=target.verify_or_ar", "pop",
                    "push:draft.trunk.layers.0 seq=1", "pop",
                    "push:draft.attn trunk.layers.0.attn seq=1", "pop"])
                tr._restore_roctx_handles(tr._ACTIVE["handles"])   # = pseudo stop worker
                for alias in originals:
                    self.assertIs(tpmod.__dict__[alias], originals[alias])
                for meth in ("iterate_draftmodel_mtp_gen", "iterate_gen"):
                    self.assertNotIn(meth, gen.__dict__)           # shadows deleted
                self.assertNotIn("forward", blk.__dict__)
                self.assertNotIn("forward", blk.modules[0].__dict__)
        finally:
            tr._ACTIVE.clear()

    def test_install_skipped_without_parent_window(self):
        st = tr.WindowState(0, 4)
        st.gen, st.spec = FakeGen([]), {"backend": "roctx"}
        with mock.patch.dict(tr._ACTIVE, {}, clear=True):
            self.assertIsNone(tr._install_parent_roctx(st))
        self.assertTrue(any("no parent window" in e for e in st.errors))
        self.assertEqual(tr._ACTIVE, {})

    def test_start_cb_installs_and_survives_partial_failure(self):
        patcher, _calls, originals, _tp = self._engine()

        class BoomTp:                        # wrapper install fails midway,
            def __init__(self):              # restore of the same keys works
                for alias in ("mp_model_forward", "mp_model_forward_embedding",
                              "mp_model_forward_lm_head_argmax"):
                    setattr(self, alias, originals[alias])

            def __setattr__(self, k, v):
                if (k == "mp_model_forward_lm_head_argmax"
                        and v is tr.mp_model_forward_mtp_head):
                    raise RuntimeError("alias slot poisoned")
                object.__setattr__(self, k, v)
        bt = BoomTp()
        st = tr.WindowState(0, 4)
        st.spec = {"backend": "roctx", "trace_dir": "/tmp/x"}
        st.gen = FakeGen([], model=FakeModel())
        with patcher:
            sys.modules["exllamav3.model"].model_tp = bt
            try:
                with mock.patch.dict(tr._ACTIVE, {"rx": FakeRoctx(), "handles": []},
                                     clear=True):
                    self.assertIs(tr._start_cb(st), False)     # fail closed
                    self.assertTrue(any("install failed" in e for e in st.errors))
                    # ranks were started, so a stop dispatch was still issued:
                    self.assertEqual([d[1] for d in st.gen.model.dispatches],
                                     ["tp_trace_roctx_start_worker",
                                      "tp_trace_roctx_stop_worker"])
                    pend = tr._ACTIVE["handles"]               # the pseudo stop
                    self.assertEqual([h[1] for h in pend],     # worker / undo would
                                     ["mp_model_forward", "mp_model_forward_embedding",
                                      "mp_model_forward_lm_head_argmax"])
                    tr._restore_roctx_handles(pend)            # restore these
                    for alias in originals:                    # exact originals
                        self.assertIs(bt.__dict__[alias], originals[alias])
            finally:
                tr._ACTIVE.clear()

    def test_none_and_torch_start_cb_install_nothing(self):
        patcher, _calls, originals, tpmod = self._engine()

        class Gen(FakeGen):
            def __init__(self):
                super().__init__([], model=FakeModel())
                self.draft_model = _mod("draft"); self.draft_model.modules = []
            def iterate_gen(self, results):
                return "x"
        for backend in ("none", "torch"):
            with self.subTest(backend=backend), patcher, \
                    mock.patch.dict(tr._ACTIVE, {"rx": FakeRoctx()}, clear=True):
                st = tr.WindowState(0, 4)
                st.spec = {"backend": backend, "trace_dir": "/tmp/x"}
                st.gen = Gen()
                self.assertTrue(tr._start_cb(st))
                for alias in originals:                        # NONE/torch: no hook
                    self.assertIs(tpmod.__dict__[alias], originals[alias])
                self.assertNotIn("iterate_gen", st.gen.__dict__)   # no shadows added
                self.assertEqual(st.errors, [])


class RoctxGateTests(unittest.TestCase):
    def setUp(self):
        self.env = {k: os.environ.get(k) for k in
                    (tr.ROCTX_PAUSE_ENV, tr.ROCTX_ARMED_ENV, tr.ROCTX_LIB_ENV)}
        self.saved = dict(tr._ROCTX)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        tr._ROCTX.clear(); tr._ROCTX.update(self.saved)

    def _reset(self):
        tr._ROCTX.clear(); tr._ROCTX.update({"controls": None, "status": "inactive"})

    def test_inactive_without_flag(self):
        os.environ.pop(tr.ROCTX_PAUSE_ENV, None)
        self._reset()
        tr._roctx_gate_env()
        self.assertEqual(tr._ROCTX["status"], "inactive")
        self.assertIsNone(tr._ROCTX["controls"])

    def test_pause_once_per_process_then_rearm(self):
        rx = FakeRoctx()
        os.environ.pop(tr.ROCTX_ARMED_ENV, None)
        self._reset()
        with mock.patch.object(tr, "_roctx_controls", lambda: rx):
            os.environ[tr.ROCTX_PAUSE_ENV] = "1"
            tr._roctx_gate_env()
            self.assertEqual(rx.ops, ["pause"])               # exactly one pause
            os.environ[tr.ROCTX_PAUSE_ENV] = "armed-not"      # status guard
            self._reset()                                     # emulate 2nd module copy
            os.environ[tr.ROCTX_PAUSE_ENV] = "1"
            tr._roctx_gate_env()
        self.assertEqual(rx.ops, ["pause"])                   # NO double pause
        self.assertEqual(tr._ROCTX["status"], "paused-earlier")

    def test_child_import_gates_before_any_torch_import(self):
        # subprocess mirrors the spawn child: flag set, __mp_main__ re-import of
        # this module must pause EARLY, before torch is importable-by-accident
        repo = str(Path(__file__).resolve().parents[3])
        code = (
            "import sys, os, json, types;"
            "sys.path.insert(0, %r);"
            "os.environ[%r] = '1';"
            "ops = [];"
            "stub = types.ModuleType('rocm_tools.rdna2.profile_stages');"
            "stub.RoctxControls = type('RoctxControls', (), {"
            "    '__init__': lambda self, *a, **k: None,"
            "    'pause': lambda self: ops.append('pause')});"
            "sys.modules['rocm_tools.rdna2.profile_stages'] = stub;"
            "import rocm_tools.rdna2.tp_trace_run as t;"
            "print(json.dumps({'status': t._ROCTX['status'], 'ops': ops,"
            "                  'torch_loaded': 'torch' in sys.modules,"
            "                  'armed': os.environ.get(t.ROCTX_ARMED_ENV) == str(os.getpid())}))"
            % (repo, tr.ROCTX_PAUSE_ENV))
        env = {k: v for k, v in os.environ.items()
               if k not in (tr.ROCTX_ARMED_ENV, tr.ROCTX_PAUSE_ENV)}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(rec["ops"], ["pause"])               # gated at module import
        self.assertIn(rec["status"], ("paused", "paused-late"))
        self.assertFalse(rec["torch_loaded"])                 # module stays torch-free
        self.assertTrue(rec["armed"])                         # pid-guarded marker set


class DispatchSelectionTests(unittest.TestCase):
    def test_backend_selects_worker_names(self):
        for backend, sname, tname in (("torch", "tp_trace_start_worker",
                                       "tp_trace_stop_worker"),
                                      ("roctx", "tp_trace_roctx_start_worker",
                                       "tp_trace_roctx_stop_worker"),
                                      ("none", "tp_trace_none_start_worker",
                                       "tp_trace_none_stop_worker")):
            with self.subTest(backend = backend):
                st = tr.WindowState(0, 4)
                st.spec = {"backend": backend}
                st.gen = FakeGen([])
                self.assertTrue(tr._start_cb(st))
                self.assertEqual(st.gen.model.dispatches[0][1], sname)
                tr._stop_cb(st)
                self.assertEqual(st.gen.model.dispatches[1][1], tname)

    def test_roctx_stop_failure_undoes_parent_window(self):
        rx = FakeRoctx()
        m = RoctxWorkerTests.Mod("k")
        st = tr.WindowState(0, 4)
        st.spec = {"backend": "roctx"}
        st.gen = FakeGen([], model=FakeModel(fail=True))
        saved = (dict(tr._ROCTX), dict(tr._ACTIVE))
        tr._ROCTX.update({"controls": rx, "status": "paused"})
        handles = []
        tr._tag_module_forwards_roctx({"modules": [m]}, rx, handles)
        tr._ACTIVE.update({"rx": rx, "handles": handles})
        try:
            tr._stop_cb(st)                # dispatch raises inside; must NOT propagate
            self.assertEqual(rx.ops[-2:], ["pop", "pause"])   # parent re-gated locally
            self.assertNotIn("forward", m.__dict__)           # parent wrappers restored
            self.assertEqual(tr._ACTIVE, {})
            self.assertTrue(any("stop dispatch failed" in e for e in st.errors))
        finally:
            tr._ROCTX.clear(); tr._ROCTX.update(saved[0])
            tr._ACTIVE.clear(); tr._ACTIVE.update(saved[1])


class VerdictTests(unittest.TestCase):
    def test_verdicts(self):
        r = lambda **kw: kw
        self.assertEqual(tr._gpu_execution_verdict(
            "torch", [r(captured_device_kernels=0), r(captured_device_kernels=0)]),
            "CPU_only_trace_no_device_kernels_do_not_claim_gpu_execution")
        self.assertEqual(tr._gpu_execution_verdict(
            "torch", [r(captured_device_kernels=0), r(captured_device_kernels=17)]),
            "device_kernels_observed")
        self.assertEqual(tr._gpu_execution_verdict(
            "torch", [r(error="x"), r(captured_device_kernels=None)]), "unverified")
        self.assertEqual(tr._gpu_execution_verdict("torch", []), "unverified")
        self.assertEqual(tr._gpu_execution_verdict("roctx", []),
                         "external_rocprofv3_csv_required")
        self.assertEqual(tr._gpu_execution_verdict("none", []),
                         "none_control_no_observation_by_design")


class HookRoctxTests(unittest.TestCase):
    def test_roctx_window_never_touches_torch_profiler(self):
        gen = FakeGen([3,2, 2,1, 1,0])
        st = tr.WindowState(0, 8)
        st.spec = {"backend": "roctx"}
        st.orig = fake_orig([])
        starts, stops = [], []
        def on_start():
            starts.append(1); return True
        def on_stop():
            stops.append(1)
        hook = tr.make_iterate_hook(st, on_start, on_stop)
        with mock.patch("torch.profiler.record_function",
                        side_effect=AssertionError("torch.profiler used in roctx mode")):
            for _ in range(3):
                hook(gen)
        self.assertEqual((starts, stops), ([1], [1]))


class HookNoneTests(unittest.TestCase):
    def test_none_window_never_touches_torch_profiler(self):
        gen = FakeGen([3,2, 2,1, 1,0])
        st = tr.WindowState(0, 8)
        st.spec = {"backend": "none"}
        st.orig = fake_orig([])
        starts, stops = [], []
        def on_start():
            starts.append(1); return True
        def on_stop():
            stops.append(1)
        hook = tr.make_iterate_hook(st, on_start, on_stop)
        with mock.patch("torch.profiler.record_function",
                        side_effect=AssertionError("torch.profiler used in none mode")):
            for _ in range(3):
                hook(gen)
        self.assertEqual((starts, stops), ([1], [1]))


class NoneWorkerTests(unittest.TestCase):
    class Mod:
        def __init__(self, key):
            self.key = key
        def forward(self, x, params=None):
            return 42

    def test_none_workers_sync_only_and_install_nothing(self):
        m = self.Mod("k")
        syncs = []
        saved = dict(tr._ACTIVE)
        try:
            with mock.patch.object(tr, "_sync_device", lambda idx: syncs.append(idx)):
                out = tr.tp_trace_none_start_worker(
                    {"device": 0, "modules": [m], "output_device": 1}, {"range": "W"})
                self.assertNotIn("error", out)
                self.assertEqual(out.get("synced"), True)
                self.assertNotIn("forward", m.__dict__)      # NO wrapper installed
                stop = tr.tp_trace_none_stop_worker({"device": 0})
                self.assertNotIn("error", stop)
                self.assertGreaterEqual(stop["monotonic_elapsed_s"], 0.0)
                self.assertEqual(tr._ACTIVE, {})             # window fully closed
            self.assertEqual(syncs, [0, 0])                  # exactly the 2 boundary syncs
            out2 = tr.tp_trace_none_stop_worker({"device": 0})
            self.assertIn("error", out2)                     # stop without start: fail-closed
        finally:
            tr._ACTIVE.clear()
            tr._ACTIVE.update(saved)

    def test_none_start_failure_leaves_no_state(self):
        def boom(idx):
            raise RuntimeError("no hip")
        with mock.patch.object(tr, "_sync_device", boom):
            out = tr.tp_trace_none_start_worker({"device": 1}, {})
        self.assertIn("error", out)
        self.assertEqual(tr._ACTIVE, {})                     # nothing leaked to restore


class DecodeAdmissionTests(unittest.TestCase):
    """run() validates decode-window admission BEFORE touching the filesystem
    or the native stack."""

    def _hargs(self, mode="mtp", new_tokens="256", batch="1"):
        return ["--", "--model", "M", "--prompts-json", "no-such.json",
                "--execution", "tp", "--mode", mode,
                "--new-tokens", new_tokens, "--batch-size", batch,
                "--power-socket", "s", "--output", "o.json"]

    def test_mtp_admitted_in_decode_mode(self):
        # passes every check (room: 64+64+2*4+1=137 <= 256): fails later only
        # on the missing prompt file, proving no ValueError was raised
        with self.assertRaises((FileNotFoundError, OSError)):
            tr.run(["--trace-dir", "/tmp/tp-trace-admission", "--trace-backend", "none",
                    "--decode-start-tokens", "64"] + self._hargs())

    def test_mixed_window_stays_ar_only(self):
        with self.assertRaisesRegex(ValueError, "mixed"):
            tr.run(["--trace-dir", "/tmp/tp-trace-admission"] + self._hargs())

    def test_batch_gt1_rejected_in_decode_mode(self):
        with self.assertRaisesRegex(ValueError, "batch-size 1"):
            tr.run(["--trace-dir", "/tmp/tp-trace-admission", "--trace-backend", "none",
                    "--decode-start-tokens", "64"] + self._hargs(batch="2"))

    def test_room_uses_mtp_burst_allowance(self):
        # AR needs only start+window+1=129; MTP needs 137
        with self.assertRaisesRegex(ValueError, "new-tokens"):
            tr.run(["--trace-dir", "/tmp/tp-trace-admission", "--trace-backend", "none",
                    "--decode-start-tokens", "64", "--decode-window-tokens", "64"]
                   + self._hargs(mode="mtp", new_tokens="136"))
        with self.assertRaisesRegex(ValueError, "new-tokens"):
            tr.run(["--trace-dir", "/tmp/tp-trace-admission", "--trace-backend", "none",
                    "--decode-start-tokens", "64", "--decode-window-tokens", "64"]
                   + self._hargs(mode="ar", new_tokens="128"))
        with self.assertRaises((FileNotFoundError, OSError)):   # 129 AR: admitted
            tr.run(["--trace-dir", "/tmp/tp-trace-admission", "--trace-backend", "none",
                    "--decode-start-tokens", "64", "--decode-window-tokens", "64"]
                   + self._hargs(mode="ar", new_tokens="129"))

    def test_nonpositive_decode_args_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            tr.run(["--trace-dir", "/tmp/tp-trace-admission",
                    "--decode-start-tokens", "0"] + self._hargs())


class ParserTests(unittest.TestCase):
    def test_decode_defaults_and_none_choice(self):
        wap = tr.build_parser().parse_args(["--trace-dir", "d"])
        self.assertIsNone(wap.decode_start_tokens)      # unset => legacy mixed window
        self.assertEqual(wap.decode_window_tokens, 64)
        self.assertEqual(wap.trace_iterations, 8)
        wap = tr.build_parser().parse_args(
            ["--trace-dir", "d", "--trace-backend", "none",
             "--decode-start-tokens", "64"])
        self.assertEqual(wap.trace_backend, "none")
        self.assertEqual(wap.decode_start_tokens, 64)


class TraceReportIntegrationTests(unittest.TestCase):
    def test_run_writes_backend_manifest_marks_report_and_fails_rank_error(self):
        import tempfile
        import types
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as td:
                base = Path(td)
                prompt = base / "prompts.json"
                prompt.write_text(json.dumps({"prompts": [{
                    "ids": [[1, 2]], "sha": tr.tp_run.prompt_sha256([1, 2]),
                    "language": "ja", "repeat": 1, "timed": True}]}))
                output = base / "run.json"
                state = tr.WindowState(0, 8)
                state.window_calls = [1]
                state.records = {"start": [], "stop": [
                    {"device": 0, "pid": 101}, {"device": 1, "pid": 102}]}
                if fail:
                    state.records["stop"][0]["pause_error"] = "injected pause failure"
                fake_exl = types.SimpleNamespace(Generator=type("Generator", (), {
                    "iterate": lambda self: []}))
                def fake_run(args, prompts):
                    output.write_text(json.dumps({"complete": True}))
                    return 0
                with mock.patch.dict(sys.modules, {"exllamav3": fake_exl}), \
                     mock.patch.object(tr, "WindowState", return_value=state), \
                     mock.patch.object(tr, "_roctx_gate_env"), \
                     mock.patch.object(tr.tp_run, "run", side_effect=fake_run):
                    code = tr.run(["--trace-dir", str(base / "trace"),
                                   "--trace-backend", "roctx", "--",
                                   "--model", "fake", "--prompts-json", str(prompt),
                                   "--execution", "tp", "--mode", "ar",
                                   "--power-socket", "fake", "--output", str(output)])
                manifest = json.loads((base / "trace/tp-trace-manifest.json").read_text())
                self.assertEqual(code, int(fail))
                self.assertEqual(manifest["complete"], not fail)
                self.assertTrue(manifest["profiler"]["no_torch_profiler"])
                self.assertTrue(json.loads(output.read_text())["profiled_diagnostic"])

    def test_run_decode_window_none_backend_manifest(self):
        import tempfile
        import types
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(tr.ROCTX_PAUSE_ENV, None)     # none must NOT gate ROCtx
            base = Path(td)
            prompt = base / "prompts.json"
            prompt.write_text(json.dumps({"prompts": [{
                "ids": [[1, 2]], "sha": tr.tp_run.prompt_sha256([1, 2]),
                "language": "ja", "repeat": 1, "timed": True}]}))
            output = base / "run.json"
            # window actually observed: started at 64 pre-iterate tokens (AR would
            # be exact; here an MTP burst made end 131 => additional 67, overshoot 3)
            state = tr.DecodeWindowState(0, 64, 64)
            state.window_calls = [67, 68]
            state.window_iterates = 2
            state.start_actual = 64
            state.end_actual = 131
            state.records = {"start": [], "stop": [
                {"device": 0, "pid": 101, "backend": "none", "synced": True,
                 "monotonic_elapsed_s": 0.5},
                {"device": 1, "pid": 102, "backend": "none", "synced": True,
                 "monotonic_elapsed_s": 0.5}]}
            fake_exl = types.SimpleNamespace(Generator=type("Generator", (), {
                "iterate": lambda self: []}))
            def fake_run(args, prompts):
                output.write_text(json.dumps({"complete": True}))
                return 0
            with mock.patch.dict(sys.modules, {"exllamav3": fake_exl}), \
                 mock.patch.object(tr, "DecodeWindowState", return_value=state), \
                 mock.patch.object(tr.tp_run, "run", side_effect=fake_run):
                code = tr.run(["--trace-dir", str(base / "trace"),
                               "--trace-backend", "none",
                               "--decode-start-tokens", "64",
                               "--decode-window-tokens", "64", "--",
                               "--model", "fake", "--prompts-json", str(prompt),
                               "--execution", "tp", "--mode", "mtp",
                               "--power-socket", "fake", "--output", str(output)])
            self.assertIsNone(os.environ.get(tr.ROCTX_PAUSE_ENV))
            manifest = json.loads((base / "trace/tp-trace-manifest.json").read_text())
            self.assertEqual(code, 0, manifest["errors"])
            self.assertEqual(manifest["complete"], True)
            self.assertEqual(manifest["backend"], "none")
            self.assertEqual(manifest["harness"]["mode"], "mtp")    # admitted now
            win = manifest["window"]
            self.assertEqual(win["mode"], "decode")
            dec = win["decode"]
            self.assertEqual(dec["requested_start_tokens"], 64)
            self.assertEqual(dec["actual_start_tokens"], 64)
            self.assertEqual(dec["actual_end_tokens"], 131)
            self.assertEqual(dec["actual_additional_tokens"], 67)
            self.assertEqual(dec["overshoot_tokens"], 3)
            self.assertEqual(dec["truncated_at_group_end"], False)
            self.assertTrue(manifest["profiler"]["no_torch_profiler"])
            self.assertTrue(manifest["profiler"]["no_roctx"])
            self.assertEqual(manifest["gpu_execution"],
                             "none_control_no_observation_by_design")
            self.assertTrue(json.loads(output.read_text())["profiled_diagnostic"])


if __name__ == "__main__":
    unittest.main(verbosity = 2)
