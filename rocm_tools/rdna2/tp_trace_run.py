#!/usr/bin/env python3
"""Bounded per-rank torch-profiler trace wrapper for the tp_run TP2 harness.

DIAGNOSTIC TOOL: numbers inside the trace window are PROFILED throughput
(profiler/record_function overhead), marked profiled_diagnostic in the
manifest -- never clean benchmark figures. The tp_run report is untouched.

The harness (--execution tp --mode ar: frozen prompts, power policy, RAM
audits, cleanup) runs UNMODIFIED: this wrapper parses its own args, then
verbatim tp_run args after '--', and hooks Generator.iterate boundaries.
Warmup group(s) run unprofiled; at the first iterate of the FIRST TIMED
group a dispatch starts torch.profiler (CPU + CUDA->HIP activities) INSIDE
EACH RANK's own process (the pseudo output rank runs inline in the parent,
so the parent's profiler also covers everything it executes), syncing each
rank's own device at that boundary. For --trace-iterations iterate calls the
parent emits a record_function per iterate while each rank wraps its
top-level module.forward calls in record_function("KEY seq=SEQLEN") (seq
distinguishes prefill from decode). The window then STOPS (also at group end
or on error): profilers export compressed Chrome traces named per
rank/device/pid, forward wrappers are restored, and a manifest with
rank/device/PID, window, files and captured-kernel count lands in
--trace-dir; tp_run continues unprofiled to its own audits and normal
unload. Model load/warmup/MTP never enter the window (first version AR-only;
no per-token syncs beyond the two boundary syncs; no Magpie).

Usage (same externally selected PYTHONPATH/native stack as tp_run):
    python3 -m rocm_tools.rdna2.tp_trace_run --trace-dir RUN/q-trace \
        --trace-iterations 8 [--trace-backend torch|roctx] -- \
        -m MODEL --prompts-json q-prompts.json --execution tp --mode ar \
        --new-tokens 32 --power-socket SOCK --output RUN/q-tp-traced.json

--trace-backend torch (default) uses torch.profiler per rank. PROVEN CAVEAT
(container probe, torch-profile-probe.json): requesting CPU+CUDA on this
ROCm stack can yield CPU-ONLY events (categories user_annotation/Trace/
cpu_op, zero kernels); the manifest then verdicts gpu_execution =
CPU_only_trace, so API success is never read as GPU-execution evidence.
--trace-backend roctx instead runs NO torch.profiler: it gates GPU
collection with the reviewed profile_stages.RoctxControls ctypes shim.
The env flag TP_TRACE_ROCTX_PAUSE=1 is set before Model.load; the
multiprocessing-SPAWN child re-import of this module (as __mp_main__) runs
the import-time hook, creating RoctxControls and calling pause() in every
rank before any torch/CUDA import (no effect when the flag is absent, so
the module stays CPU-safe for other importers). GPU kernels are collected
by the EXTERNAL rocprofv3 attach (root's SDK-first LD_PRELOAD + Triton
DEEPBIND bootstrap, which may pause earlier too: harmless). Rank start
syncs its own device, resumes, pushes an outer window range and installs
nested per-module "KEY seq=N" ROCtx markers; rank stop syncs, pops, pauses
and restores the wrappers (profile_stages' exception-safe order). GPU
execution is NOT claimed from inside this tool: the manifest records
backend=roctx + rank/device/PID/window gate states and requires root's
external CSV parse for kernel evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import shutil
import sys
import time
from pathlib import Path

from rocm_tools.rdna2 import tp_run   # CPU-safe (its native imports are lazy)

_MISSING = object()

# Per-process profiler/wrapper state: each spawned rank re-imports this module
# when the pickled function is resolved, so _ACTIVE is ONE dict per rank.
_ACTIVE: dict = {}

# ROCtx gating (reviewed shim: profile_stages.RoctxControls). The flag is set
# by the trace main BEFORE Model.load; multiprocessing spawn children
# re-import this module as __mp_main__ during start-up fixup, where this hook
# pauses collection early - before those processes import torch/CUDA. No-op
# (and torch-free) for every other importer when the flag is absent.
ROCTX_PAUSE_ENV = "TP_TRACE_ROCTX_PAUSE"
ROCTX_LIB_ENV = "TP_TRACE_ROCTX_LIB"      # optional per-run shim path override
ROCTX_ARMED_ENV = "TP_TRACE_ROCTX_PAUSE_ARMED_PID"   # process-local (pid-guarded)
_ROCTX: dict = {"controls": None, "status": "inactive"}


def _roctx_controls():
    """Fresh RoctxControls for this process (path overridable for tests)."""
    from rocm_tools.rdna2.profile_stages import RoctxControls
    lib = os.environ.get(ROCTX_LIB_ENV)
    return RoctxControls(lib) if lib else RoctxControls()


def _roctx_gate_env():
    if _ROCTX["status"] != "inactive":
        return                                  # this module copy already gated
    if os.environ.get(ROCTX_PAUSE_ENV) != "1":
        return                                  # torch mode / plain imports: no effect
    if os.environ.get(ROCTX_ARMED_ENV) == str(os.getpid()):
        # THIS process already paused via an earlier module copy (spawn fixup +
        # by-name import): fresh controls for the window's resume/push, no
        # second pause. The pid guard keeps the parent's marker from muting a
        # child that inherits PAUSE=1 but has not gated yet.
        try:
            _ROCTX["controls"] = _roctx_controls()
            _ROCTX["status"] = "paused-earlier"
        except Exception as e:
            _ROCTX["status"] = f"error: {e!r}"
        return
    late = "torch" in sys.modules               # gated after torch/CUDA init: load
    try:                                        # kernels stay in the external trace
        _ROCTX["controls"] = _roctx_controls()  # (root's bootstrap may pause earlier
        _ROCTX["controls"].pause()              #  too: a second pause is the same state)
        os.environ[ROCTX_ARMED_ENV] = str(os.getpid())
        _ROCTX["status"] = "paused-late" if late else "paused"
    except Exception as e:
        _ROCTX["status"] = f"error: {e!r}"


_roctx_gate_env()


def _sync_device(idx):
    import torch
    torch.cuda.synchronize(idx)


@contextlib.contextmanager
def _roctx_range(rx, label):
    rx.push(label)
    try:
        yield
    finally:
        rx.pop()


# pure wrapper logic (CPU-testable without torch / exllamav3)

def split_argv(argv):
    if "--" not in argv:
        raise ValueError("usage: tp_trace_run [wrapper args] -- [tp_run args]")
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def first_timed_group(prompts, batch_size):
    """First group whose first prompt is timed; mirrors tp_run's grouping."""
    for gi in range(0, len(prompts), batch_size):
        if prompts[gi].get("timed"):
            return gi // batch_size
    raise ValueError("prompt file has no timed group to trace")


class WindowState:
    """iterate-boundary bookkeeping: a new group is exactly a 0 -> positive
    step of num_remaining_jobs(); trace at most one window."""

    def __init__(self, window_group, max_iterates):
        self.window_group = window_group
        self.max_iterates = max_iterates
        self.group = -1
        self.rem_after = 0
        self.calls = 0
        self.active = False
        self.done = False
        self.want_start = False
        self.want_stop = False
        self.window_iterates = 0
        self.window_calls = []          # [call_index, ...] inside the window
        self.errors = []
        self.gen = None                 # bound by the hook on first iterate
        self.orig = None                # the class function being wrapped
        self.records = {}
        self.spec = {}

    def pre_iterate(self, rem_before):
        self.calls += 1
        if rem_before > 0 and self.rem_after == 0:
            self.group += 1
            if (not self.active and not self.done
                    and self.group == self.window_group):
                self.want_start = True

    def post_iterate(self, rem_after):
        self.rem_after = rem_after
        if self.active:
            self.window_calls.append(self.calls)
            self.window_iterates += 1
            if rem_after == 0 or self.window_iterates >= self.max_iterates:
                self.want_stop = True


def build_parser():
    ap = argparse.ArgumentParser(
        description="bounded per-rank torch.profiler trace around one tp_run "
                    "timed group (diagnostic only; pass tp_run args after '--')",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--trace-dir", required=True,
                    help="output dir for per-rank Chrome traces (.json.gz) + manifest")
    ap.add_argument("--trace-iterations", type=int, default=8,
                    help="max Generator.iterate calls inside the window")
    ap.add_argument("--trace-backend", choices=("torch", "roctx"), default="torch",
                    help="torch = per-rank torch.profiler Chrome traces (manifest "
                         "verdicts cpu_only when 0 device kernels are seen, as the "
                         "container probe did); roctx = NO torch.profiler: "
                         "profile_stages.RoctxControls gating + markers for an "
                         "EXTERNAL rocprofv3 GPU collection (root launches it)")
    return ap


# rank-side workers: top-level + picklable; run in EACH rank's own process
# (the pseudo output rank runs inline in the parent process)

def _tag_module_forwards(local_context, handles):
    """Wrap each top-level module's forward in record_function('KEY seq=N').
    Appends (module, previous_instance_forward) to the CALLER's handles list,
    so a partial wrap is still fully restorable."""
    import torch
    for m in local_context.get("modules") or []:
        orig = m.forward
        key = str(getattr(m, "key", None) or type(m).__name__)

        def call(*a, _o=orig, _k=key, **kw):
            x = a[0] if a else next(iter(kw.values()), None)
            seq = x.shape[1] if torch.is_tensor(x) and x.dim() >= 2 else "?"
            with torch.profiler.record_function(f"{_k} seq={seq}"):
                return _o(*a, **kw)
        handles.append((m, m.__dict__.get("forward", _MISSING)))
        m.forward = call


def _restore_module_forwards(handles):
    for m, prev in handles or []:
        if prev is _MISSING:
            m.__dict__.pop("forward", None)
        else:
            m.forward = prev


def _device_index(local_context):
    dev = local_context.get("device")
    return getattr(dev, "index", dev)


def tp_trace_start_worker(local_context, spec):
    import torch
    idx = _device_index(local_context)
    out = {"device": idx, "pid": os.getpid(), "started_unix": time.time(),
           "role": "parent-pseudo" if idx == local_context.get("output_device") else "spawned"}
    prof = None
    handles = []
    try:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA])   # CUDA -> HIP
        torch.cuda.synchronize(idx)                              # boundary sync
        prof.__enter__()
        _tag_module_forwards(local_context, handles)
        _ACTIVE.update(prof=prof, handles=handles, idx=idx,
                       t0=out["started_unix"], spec=spec)
        out["tagged_modules"] = len(handles)
    except Exception as e:                       # leave the rank exactly as found
        _restore_module_forwards(handles)
        try:
            if prof is not None:
                prof.__exit__(None, None, None)
        except Exception:
            pass
        out["error"] = repr(e)
    return out


def tp_trace_stop_worker(local_context):
    import torch
    idx = _device_index(local_context)
    out = {"device": idx, "pid": os.getpid()}
    st = dict(_ACTIVE)
    prof = st.get("prof")
    if prof is None:
        out["error"] = "no active profiler on this rank (start failed there)"
        _ACTIVE.clear()
        return out
    try:
        torch.cuda.synchronize(idx)               # boundary sync
        out["window_s"] = round(time.time() - st["t0"], 3)
        raw = Path(st["spec"]["trace_dir"]) / \
            f"tp-trace-rank{idx}-pid{out['pid']}.json"
        gz = Path(str(raw) + ".gz")
        try:
            prof.__exit__(None, None, None)       # device events land at exit
            kernels = None
            try:
                cu = torch.autograd.DeviceType.CUDA
                kernels = sum(1 for e in prof.events()
                              if getattr(e, "device_type", None) == cu)
            except Exception:
                pass
            prof.export_chrome_trace(str(raw))
            with open(raw, "rb") as fin, gzip.open(gz, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            out.update(trace_file=str(gz), trace_bytes=gz.stat().st_size,
                       captured_device_kernels=kernels)
        finally:
            raw.unlink(missing_ok=True)
            _ACTIVE.pop("prof", None)
    except Exception as e:
        out["error"] = repr(e)
    finally:
        _restore_module_forwards(_ACTIVE.pop("handles", None))
        _ACTIVE.clear()
    return out


def _tag_module_forwards_roctx(local_context, rx, handles):
    """ROCtx twin of _tag_module_forwards: nested push/pop markers keyed
    KEY seq=N around each top-level forward. Duck-typed on the input tensor,
    so no torch import happens in roctx mode."""
    for m in local_context.get("modules") or []:
        orig = m.forward
        key = str(getattr(m, "key", None) or type(m).__name__)

        def call(*a, _o=orig, _k=key, _rx=rx, **kw):
            x = a[0] if a else next(iter(kw.values()), None)
            try:
                seq = x.shape[1] if x.dim() >= 2 else "?"
            except Exception:
                seq = "?"
            with _roctx_range(_rx, f"{_k} seq={seq}"):
                return _o(*a, **kw)
        handles.append((m, m.__dict__.get("forward", _MISSING)))
        m.forward = call


def tp_trace_roctx_start_worker(local_context, spec):
    """In-rank window open (profile_stages order): sync own GPU -> resume ->
    push the outer rank/window range -> install nested per-module markers.
    The pause that makes this window the ONLY traced activity happened at
    import time (ROCTX_PAUSE_ENV hook, before torch/CUDA); nothing here can
    retroactively gate this rank's load/warmup, so the gate state is REPORTED,
    not fixed up."""
    idx = _device_index(local_context)
    out = {"device": idx, "pid": os.getpid(), "backend": "roctx",
           "gate": _ROCTX["status"], "started_unix": time.time()}
    rx = _ROCTX.get("controls")
    handles = []
    try:
        if rx is None:                             # gate errored: last-resort fresh controls
            rx = _roctx_controls()
            _ROCTX["controls"] = rx
            out["gate"] = "unpaused-at-start"      # resume has no pause behind it here
        _sync_device(idx)
        rx.resume()
        out["resumed"] = True
        label = f"{spec.get('range', 'tp_trace_window')} rank{idx} pid{out['pid']}"
        rx.push(label)
        out["range"] = label
        _tag_module_forwards_roctx(local_context, rx, handles)
        _ACTIVE.update(rx=rx, handles=handles, idx=idx, t0=out["started_unix"])
        out["tagged_modules"] = len(handles)
    except Exception as e:
        _restore_module_forwards(handles)
        try:
            if out.get("range"):
                rx.pop()
            if out.get("resumed"):
                rx.pause()                         # re-gate: a half-open window must not leak
        except Exception:
            pass
        out["error"] = repr(e)
    return out


def tp_trace_roctx_stop_worker(local_context):
    """In-rank window close: sync own GPU -> pop -> pause -> restore wrappers.
    pop/pause/restore are exception-safe (profile_stages contract): they run
    even after the sync failed, and no GPU-kernel observation is claimed here
    (the external rocprofv3 CSV is the only kernel evidence)."""
    idx = _device_index(local_context)
    out = {"device": idx, "pid": os.getpid(), "backend": "roctx"}
    st = dict(_ACTIVE)
    rx = st.get("rx")
    if rx is None:
        out["error"] = "no active roctx window on this rank (start failed there)"
        _ACTIVE.clear()
        return out
    try:
        try:
            _sync_device(idx)
            out["window_s"] = round(time.time() - st["t0"], 3)
        finally:
            try:
                rx.pop()
                out["popped"] = True
            except Exception as e:
                out["pop_error"] = repr(e)
            try:
                rx.pause()
                out["paused_after_window"] = True
            except Exception as e:
                out["pause_error"] = repr(e)
    except Exception as e:
        out["error"] = repr(e)
    finally:
        _restore_module_forwards(st.get("handles") or [])
        _ACTIVE.clear()
    return out


_WORKERS = {"torch": (tp_trace_start_worker, tp_trace_stop_worker),
            "roctx": (tp_trace_roctx_start_worker, tp_trace_roctx_stop_worker)}


# parent orchestration: iterate hook + dispatch of the rank workers

def make_iterate_hook(state, on_start, on_stop):
    def iterate(self, *a, **kw):
        state.gen = self
        try:
            state.pre_iterate(self.num_remaining_jobs())
            if state.want_start:
                state.want_start = False
                if on_start() is not False:
                    state.active = True
                else:
                    state.done = True
            if state.active and state.spec.get("backend", "torch") == "torch":
                import torch
                with torch.profiler.record_function(
                        f"tp_trace:Generator.iterate#{state.calls}"):
                    items = state.orig(self, *a, **kw)
            else:
                items = state.orig(self, *a, **kw)
        except BaseException:
            if state.active:
                try:
                    on_stop()
                except Exception as e:
                    state.errors.append(f"stop on error: {e!r}")
                state.active = False
                state.done = True
            raise
        state.post_iterate(self.num_remaining_jobs())
        if state.want_stop:
            state.want_stop = False
            state.active = False
            state.done = True
            try:
                on_stop()
            except Exception as e:
                state.errors.append(f"stop: {e!r}")
        return items
    return iterate


def _start_cb(state):
    model = getattr(state.gen, "model", None)
    if model is None or not getattr(model, "loaded_tp", False):
        state.errors.append("trace start: generator exposes no loaded_tp model")
        return False
    start_fn = _WORKERS[state.spec.get("backend", "torch")][0]
    try:
        state.records["start"] = model.tp_worker_dispatch_wait_multi(
            model.active_devices, start_fn, (state.spec,))
        return True
    except Exception as e:
        state.errors.append(f"trace start dispatch failed: {e!r}")
        return False


def _parent_roctx_undo(state, why):
    """Best-effort LOCAL (parent = pseudo-rank) window teardown when the stop
    dispatch never reached it: an open resume/push would otherwise leak every
    remaining group into the external trace."""
    rx = _ACTIVE.get("rx")                       # only if THIS process opened a window
    try:
        if rx is not None:
            rx.pop()
            rx.pause()
    except Exception as e:
        state.errors.append(f"parent roctx teardown ({why}): {e!r}")
    _restore_module_forwards(_ACTIVE.pop("handles", None))
    _ACTIVE.clear()
    state.errors.append(why)


def _stop_cb(state):
    model = state.gen.model
    backend = state.spec.get("backend", "torch")
    stop_fn = _WORKERS[backend][1]
    try:
        state.records["stop"] = model.tp_worker_dispatch_wait_multi(
            model.active_devices, stop_fn, ())
    except Exception as e:
        why = f"trace stop dispatch failed: {e!r}"
        if backend == "roctx":
            _parent_roctx_undo(state, why)
        else:
            state.errors.append(why)


def _gpu_execution_verdict(backend, stop_records):
    """NEVER read profiler API success as GPU-execution evidence: the actual
    container probe requested CPU+CUDA and still produced CPU-only events
    (torch-profile-probe.json). torch mode: verdict from the per-rank device
    kernel counts. roctx mode: this tool observes no kernels at all; the
    external rocprofv3 CSV parse is the only kernel evidence."""
    if backend == "roctx":
        return "external_rocprofv3_csv_required"
    ks = [r.get("captured_device_kernels") for r in stop_records or []
          if r and "error" not in r]
    if any(isinstance(k, int) and k > 0 for k in ks):
        return "device_kernels_observed"
    if ks and all(k == 0 for k in ks):
        return "CPU_only_trace_no_device_kernels_do_not_claim_gpu_execution"
    return "unverified"


def run(argv=None):
    wargv, hargv = split_argv(sys.argv[1:] if argv is None else list(argv))
    wap = build_parser().parse_args(wargv)
    args = tp_run.build_parser().parse_args(hargv)
    tp_run.validate_args(args)
    if args.execution != "tp":
        raise ValueError("tp_trace_run traces --execution tp only")
    if args.mode != "ar":
        raise ValueError("first version is AR-only (--mode ar): the parent-"
                         "process profiler would mix MTP drafts into the window")
    if wap.trace_backend == "roctx":
        # BEFORE any import that can touch CUDA and long before Model.load:
        # the env flag makes every spawn child gate itself at __mp_main__
        # re-import; _roctx_gate_env() pauses THIS (parent = output rank)
        # process now, so load/warmup kernels never enter the external trace.
        os.environ[ROCTX_PAUSE_ENV] = "1"
        _roctx_gate_env()
    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    try:
        import exllamav3                       # native stack via external PYTHONPATH
    except ImportError as e:
        print(f"error: native exllamav3/torch stack not importable: {e}",
              file=sys.stderr)
        return 2
    prompts = tp_run.validate_prompts(prompts, args.batch_size)
    trace_dir = Path(wap.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    state = WindowState(first_timed_group(prompts, args.batch_size),
                        wap.trace_iterations)
    state.spec = {"trace_dir": str(trace_dir),
                  "trace_iterations": wap.trace_iterations,
                  "backend": wap.trace_backend,
                  "range": f"tp_trace_window group{state.window_group} "
                           f"first{wap.trace_iterations}iterates"}
    state.orig = exllamav3.Generator.iterate
    exllamav3.Generator.iterate = make_iterate_hook(state,
                                                    lambda: _start_cb(state),
                                                    lambda: _stop_cb(state))
    rc = 1
    try:
        rc = tp_run.run(args, prompts)         # normal load/warmup/audits/unload
    finally:
        exllamav3.Generator.iterate = state.orig
        if state.active:                       # safety net; window normally ends inside
            try:
                _stop_cb(state)
            except Exception as e:
                state.errors.append(f"stop after run: {e!r}")
        if wap.trace_backend == "torch":
            try:
                import torch
                pinfo = {"activities": ["CPU", "CUDA(HIP)"], "torch": torch.__version__,
                         "hip": str(getattr(torch.version, "hip", None))}
            except Exception as e:
                pinfo = {"unavailable": repr(e)}
        else:
            pinfo = {"backend": "roctx", "no_torch_profiler": True,
                     "shim_pause_env": ROCTX_PAUSE_ENV,
                     "parent_gate": _ROCTX["status"],
                     "external_gpu_trace_required": True,
                     "external_gpu_trace_owner": "root's rocprofv3 launch "
                                                 "(SDK-first LD_PRELOAD, DEEPBIND)",
                     "kernel_observation": "none claimed by this tool: parse the "
                                           "external rocprofv3 CSV against the "
                                           "window/marker records for GPU evidence"}
        stop_records = state.records.get("stop") or []
        both = {r.get("device") for r in stop_records if r.get("pid") and "error" not in r}
        for stage in ("start", "stop"):
            for record in state.records.get(stage) or []:
                for key in ("error", "pop_error", "pause_error"):
                    if record.get(key):
                        state.errors.append(f"{stage} rank {record.get('device')}: {key}: {record[key]}")
        if not state.window_calls or len(stop_records) != 2 or both != {0, 1}:
            state.errors.append("a bounded window on both TP ranks was not captured")
        mp = trace_dir / "tp-trace-manifest.json"
        try:
            report_path = Path(args.output)
            if report_path.exists():
                report = json.loads(report_path.read_text())
                report.update(profiled_diagnostic=True, trace_manifest=str(mp))
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        except Exception as e:
            state.errors.append(f"marking profiled report: {e!r}")
        rc = rc or (1 if state.errors else 0)
        manifest = {
            "profiled_diagnostic": True,
            "complete": rc == 0,
            "backend": wap.trace_backend,
            "note": "throughput inside the traced window includes profiling overhead: "
                    "diagnostic comm-vs-compute evidence, NOT benchmark figures; see "
                    "the tp_run report (also marked profiled_diagnostic) for the run itself",
            "tp_run_output": str(args.output),
            "harness": {"model": str(args.model), "prompts": str(args.prompts_json),
                        "execution": args.execution, "mode": args.mode,
                        "batch_size": args.batch_size, "new_tokens": args.new_tokens},
            "trace_iterations": wap.trace_iterations,
            "window": {"group": state.window_group,
                       "iterate_calls": state.window_calls,
                       "iterates_traced": len(state.window_calls)},
            "gpu_execution": _gpu_execution_verdict(wap.trace_backend,
                                                    state.records.get("stop")),
            "profiler": pinfo,
            "ranks": (state.records.get("stop") or []),
            "start_records": state.records.get("start"),
            "errors": state.errors,
            "exit_code": rc}
        mp = trace_dir / "tp-trace-manifest.json"
        mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        both = {r.get("device") for r in manifest["ranks"]
                if r.get("pid") and "error" not in r}
        print(f"trace manifest: {mp}\nwindow={manifest['window']} "
              f"traced_ranks={sorted(both, key=str)} errors={state.errors or 'none'}",
              flush=True)
    return rc


if __name__ == "__main__":
    try:
        sys.exit(run())
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
