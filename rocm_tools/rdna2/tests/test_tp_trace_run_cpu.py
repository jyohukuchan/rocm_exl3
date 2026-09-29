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
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from rocm_tools.rdna2 import tp_trace_run as tr


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
                                       "tp_trace_roctx_stop_worker")):
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


if __name__ == "__main__":
    unittest.main(verbosity = 2)
