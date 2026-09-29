#!/usr/bin/env python3
"""Bounded per-rank torch-profiler trace wrapper for the tp_run TP2 harness.

DIAGNOSTIC TOOL: numbers inside the trace window are PROFILED throughput
(profiler/record_function overhead), marked profiled_diagnostic in the
manifest -- never clean benchmark figures. The tp_run report is untouched.

The harness (--execution tp; frozen prompts, power policy, RAM audits,
cleanup) runs UNMODIFIED: this wrapper parses its own args, then
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
unload. Model load/warmup never enter the window (no per-token syncs
beyond the two boundary syncs; Job.new_tokens is host-side metadata; no
Magpie).

DECODE-ONLY WINDOW (--decode-start-tokens N, the default --trace-iterations
cap does NOT apply): --mode mtp is admitted (capture is coarse by design in
this milestone: target module wrappers exist, draft kernels are NOT yet
separately labelled). On the first timed group the wrapper watches the
single ACTIVE job's new_tokens (batch1 enforced) and starts at the first
iterate whose PRE-iterate count is already >= N -- so prefill/first-token
and warmup groups are structurally excluded -- then collects until the count
reaches actual_start + --decode-window-tokens (default 64). MTP acceptance
bursts may overshoot; the ACTUAL start/end counts, iterates and overshoot
land in the manifest's window.decode section. --new-tokens is validated to
fit start+window+2*draft_bursts+1 so the window always ends before the
final (EOS) iterate; a group that drains first is reported as truncated
(error, fail-closed).

--trace-backend none is the observer-overhead control: the SAME window
detection and per-rank boundary device syncs run, but NO profiler, NO ROCtx
pause/resume/markers and NO wrappers are installed, and the manifest records
per-rank monotonic elapsed time for the window.

Usage (same externally selected PYTHONPATH/native stack as tp_run):
    python3 -m rocm_tools.rdna2.tp_trace_run --trace-dir RUN/q-trace \
        [--trace-backend torch|roctx|none] [--decode-start-tokens 64 \
        --decode-window-tokens 64] -- \
        -m MODEL --prompts-json q-prompts.json --execution tp --mode mtp \
        --new-tokens 256 --power-socket SOCK --output RUN/q-tp-traced.json
Legacy mixed window (AR-only): pass --trace-iterations N and NO
--decode-start-tokens.

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
nested markers; rank stop syncs, pops, pauses and restores every hook
(profile_stages' exception-safe order). MILESTONE-2 ATTRIBUTION
(roctx-only; torch/none paths unchanged): per-module "KEY seq=N" ranges are
extended by a BOUNDED class traversal of Module.modules (no descent into
Linear subtrees, never a wrapper per expert): attn/attn.qsa, gdn, moe,
moe.shared (only where the shared-expert MLP is called from Python; fused
paths stay under moe), ple(+ple.prefetch), ngram(forward/_stage/
_gather_rows, thread-local so the prefetch worker thread labels itself),
hc(+hc.mix/hc.apply_ for site forms), norm, embed. Each rank's comm
backend gets tp.<op> ranges with numel x element_size metadata (no .item(),
no sync). The parent (pseudo output rank) additionally labels its draft
model tree "draft.*", marks gen.iterate_draftmodel_mtp_gen / iterate_gen
as phase=mtp.draft / phase=target.verify_or_ar, and swaps model_tp's
worker-function aliases for top-level picklable wrappers that label each
dispatch phase=target.forward / phase=mtp.borrowed_embedding /
phase=mtp.borrowed_head (originals in model_tp_fn are never overwritten;
aliases restore on success, install failure and stop-dispatch failure).
Labels are INCLUSIVE CPU ranges: root derives exclusive GPU attribution
from the HIP correlation of each kernel to its launching API and the
markers enclosing THAT call, so shared target embedding/head kernels
dispatched through a borrowed phase count as MTP work. GPU execution is
NOT claimed from inside this tool: the manifest records backend=roctx +
rank/device/PID/window gate states and requires root's external CSV parse
for kernel evidence.
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

    def read_tokens(self, gen):
        """legacy mixed window has no token gating (host metadata only)."""
        return None

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

    def pre_iterate(self, rem_before, tokens=None):        # tokens: legacy mode ignores
        self.calls += 1
        if rem_before > 0 and self.rem_after == 0:
            self.group += 1
            if (not self.active and not self.done
                    and self.group == self.window_group):
                self.want_start = True

    def post_iterate(self, rem_after, tokens=None):
        self.rem_after = rem_after
        if self.active:
            self.window_calls.append(self.calls)
            self.window_iterates += 1
            if rem_after == 0 or self.window_iterates >= self.max_iterates:
                self.want_stop = True


class DecodeWindowState:
    """DECODE-ONLY window on the first timed group (root's batch1 run):
    STARTS only at the first iterate whose PRE-iterate Job.new_tokens is
    already >= start_tokens -- prefill and its first token are never inside
    -- and STOPS after the first iterate whose token count reaches
    start_actual + window_tokens. The legacy --trace-iterations cap does NOT
    apply here. MTP acceptance bursts may overshoot either bound; the ACTUAL
    start/end counts are recorded, never the requested ones. Job selection
    is exact: >1 simultaneously active job is ambiguous and refused, never
    guessed. Tokens are host-side job metadata (Job.new_tokens ints): no
    per-iterate RPC and no device sync here."""

    def __init__(self, window_group, start_tokens, window_tokens):
        self.window_group = window_group
        self.start_tokens = start_tokens
        self.window_tokens = window_tokens
        self.group = -1
        self.rem_after = 0
        self.calls = 0
        self.armed = False
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
        self.start_actual = None        # tokens generated BEFORE first traced iterate
        self.end_actual = None          # tokens generated AFTER the last traced iterate
        self.truncated = False          # group drained before the window completed

    def read_tokens(self, gen):
        """min over the group's active jobs (batch1: the single job). None
        while the group is prefilling (job still pending)."""
        jobs = list(getattr(gen, "active_jobs", ()) or ())
        if not jobs:
            return None
        if len(jobs) > 1:
            if not self.done:
                self.errors.append(
                    f"decode window: {len(jobs)} active jobs - ambiguous "
                    "job selection; requires --batch-size 1. Refusing")
                self.done = True
            return None
        return int(jobs[0].new_tokens)

    def pre_iterate(self, rem_before, tokens=None):
        self.calls += 1
        if rem_before > 0 and self.rem_after == 0:
            self.group += 1
            if (not self.armed and not self.done
                    and self.group == self.window_group):
                self.armed = True
        if (self.armed and not self.active and not self.done
                and not self.want_start
                and tokens is not None and tokens >= self.start_tokens):
            self.want_start = True
            self.start_actual = tokens

    def post_iterate(self, rem_after, tokens=None):
        self.rem_after = rem_after
        if not self.active:
            return
        self.window_calls.append(self.calls)
        self.window_iterates += 1
        if tokens is not None:
            self.end_actual = tokens
        if tokens is not None and tokens >= self.start_actual + self.window_tokens:
            self.want_stop = True                       # window complete (>= requested)
        elif rem_after == 0:
            self.truncated = True                       # drained mid-window (unsafe config)
            self.want_stop = True


def build_parser():
    ap = argparse.ArgumentParser(
        description="bounded per-rank torch.profiler trace around one tp_run "
                    "timed group (diagnostic only; pass tp_run args after '--')",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--trace-dir", required=True,
                    help="output dir for per-rank Chrome traces (.json.gz) + manifest")
    ap.add_argument("--trace-iterations", type=int, default=8,
                    help="LEGACY mixed window: max Generator.iterate calls "
                         "inside the window, starting at the timed group's "
                         "FIRST iterate (prefill included). Ignored when "
                         "--decode-start-tokens is set")
    ap.add_argument("--decode-start-tokens", type=int, default=None,
                    help="DECODE-ONLY window: tokens the selected (batch1) "
                         "job must ALREADY have generated before the window's "
                         "first iterate (root uses 64). Unset keeps the legacy "
                         "first-iterate mixed window")
    ap.add_argument("--decode-window-tokens", type=int, default=64,
                    help="DECODE-ONLY window: additional tokens to trace; the "
                         "stop fires at the first iterate whose count reaches "
                         "actual_start + N (MTP bursts may overshoot; the "
                         "ACTUAL counts are recorded)")
    ap.add_argument("--trace-backend", choices=("torch", "roctx", "none"),
                    default="torch",
                    help="torch = per-rank torch.profiler Chrome traces (manifest "
                         "verdicts cpu_only when 0 device kernels are seen, as the "
                         "container probe did); roctx = NO torch.profiler: "
                         "profile_stages.RoctxControls gating + markers for an "
                         "EXTERNAL rocprofv3 GPU collection (root launches it); "
                         "none = observer-overhead control: the same window "
                         "detection and per-rank boundary synchronize ONLY - no "
                         "profiler, no ROCtx calls, no wrappers installed")
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


# ---- roctx-only nested task/phase attribution (milestone 2) ----
# Labels are INCLUSIVE CPU ranges; root maps GPU kernels to them through the
# HIP runtime Correlation_Id of the launching API call, never by GPU/CPU time
# overlap. Class matching is by MRO name so engine subclasses stay covered
# without importing exllamav3 here (the module must stay torch-free).

_ROCTX_TAG_CLASSES = {          # class name -> (label base, wrapped attrs)
    "Attention": ("attn", ("forward",)),
    "GatedDeltaNet": ("gdn", ("forward",)),
    "BlockSparseMLP": ("moe", ("forward",)),
    "PLELayer": ("ple", ("forward", "prefetch")),
    "Embedding": ("embed", ("forward",)),
    "NGramEmbedding": ("ngram", ("forward", "_stage", "_gather_rows")),
    "GatedResidual": ("hc", ("forward", "mix", "apply_")),
    "RMSNorm": ("norm", ("forward",)),
    "GatedRMSNorm": ("norm", ("forward",)),
    "LayerNorm": ("norm", ("forward",)),
}

_ROCTX_COMM_OPS = ("all_reduce", "broadcast", "gather", "gather_small",
                   "fwd_barrier")


def _seq_meta(args, kw):
    """Decode seq len from the first positional arg, duck-typed (2-D+ ->
    shape[1], 1-D -> numel, else '?'). No torch dependency."""
    x = args[0] if args else next(iter(kw.values()), None)
    try:
        return x.shape[1] if x.dim() >= 2 else x.numel()
    except Exception:
        return "?"


def _tensor_meta(a):
    """'numel x bytes = total' metadata string for a tensor-like arg, or
    None. numel()/element_size() only: never touches values (no .item(), no
    sync)."""
    try:
        n, e = a.numel(), a.element_size()
        return "%dx%dB=%dB" % (n, e, n * e)
    except Exception:
        return None


def _roctx_attr(m, attr):
    """Resolve an inherited attribute without shadowing it: forward/mix/...
    may live on the class, so plain hasattr+getattr is required."""
    fn = getattr(m, attr, None)
    return fn if callable(fn) else None


def _tag_wrap_roctx(rx, m, attr, label_base, handles, key=None):
    """Wrap one instance attribute in a thread-local ROCtx range (roctx's
    push/pop stack is per thread, which is what lets ngram _stage/_gather_rows
    tagged here label their ThreadPoolExecutor worker). Appends an
    (obj, attr, previous) handle; skips targets already wrapped (dedup)."""
    if any(h[0] is m and h[1] == attr for h in handles):
        return False
    orig = _roctx_attr(m, attr)
    if orig is None:
        return False
    key = str(key if key is not None else getattr(m, "key", None)
              or type(m).__name__)

    def call(*a, _o=orig, _rx=rx, _lbl="%s %s" % (label_base, key), **kw):
        with _roctx_range(_rx, "%s seq=%s" % (_lbl, _seq_meta(a, kw))):
            return _o(*a, **kw)
    handles.append((m, attr, m.__dict__.get(attr, _MISSING)))
    setattr(m, attr, call)
    return True


def _restore_roctx_handles(handles):
    """Undo (obj, attr, previous) handles in reverse install order: instance
    shadows are deleted when they shadowed nothing, module/class attributes
    are restored by value. Runs on success AND exception teardown paths."""
    for obj, attr, prev in reversed(handles or []):
        if prev is _MISSING:
            obj.__dict__.pop(attr, None)
        else:
            setattr(obj, attr, prev)


def _walk_roctx_tree(module):
    """Bounded class traversal: yields the tree like Module.__iter__ but does
    NOT descend into Linear subtrees (per-expert quant buffers) so tagging
    stays at layer-scale labels, not thousands of expert wrappers."""
    yield module
    if any(c.__name__ == "Linear" for c in type(module).__mro__):
        return
    for sub in getattr(module, "modules", None) or []:
        yield from _walk_roctx_tree(sub)


def _tag_nested_roctx(modules, rx, handles, prefix=""):
    """Class-based nested labels (see _ROCTX_TAG_CLASSES). Module forwards
    already wrapped by the top-level pass are deduplicated away; only extra
    methods (ple.prefetch, hc.mix/apply_, ngram._stage/_gather_rows) still
    add ranges. Returns the number of wrapped attrs."""
    n = 0
    for top in modules or []:
        for m in _walk_roctx_tree(top):
            spec = next((_ROCTX_TAG_CLASSES[c.__name__]
                         for c in type(m).__mro__ if c.__name__ in
                         _ROCTX_TAG_CLASSES), None)
            if spec is None:
                continue
            base, attrs = spec
            if base == "attn" and getattr(m, "qsa_indexer", None) is not None:
                base = "attn.qsa"        # QSA vs full attention
            for attr in attrs:
                lbl = base if attr == "forward" else "%s.%s" % (base, attr)
                if _tag_wrap_roctx(rx, m, attr, prefix + lbl, handles):
                    n += 1
            se = getattr(m, "shared_experts", None)   # BlockSparseMLP
            if se is not None and _tag_wrap_roctx(
                    rx, se, "forward", prefix + "moe.shared", handles):
                n += 1
                # Fused shared-expert paths bypass Python entirely: those
                # kernels then stay under "moe", never claimed as split here.
    return n


def _tag_backend_roctx(backend, rx, handles):
    """Comm labels per rank: tp.<op> ranges carrying tensor metadata
    (numel x element_size of the first tensor args) so root can size the
    collective without any value read. Args/returns/exceptions pass through."""
    n = 0
    for op in _ROCTX_COMM_OPS:
        orig = _roctx_attr(backend, op)
        if orig is None or any(h[0] is backend and h[1] == op
                               for h in handles):
            continue

        def call(*a, _o=orig, _rx=rx, _op=op, **kw):
            meta = [m for m in (_tensor_meta(x) for x in a[:2]) if m]
            label = "tp.%s%s" % (_op, (" " + "+".join(meta)) if meta else "")
            with _roctx_range(_rx, label):
                return _o(*a, **kw)
        handles.append((backend, op, backend.__dict__.get(op, _MISSING)))
        setattr(backend, op, call)
        n += 1
    return n


def _tag_module_forwards_roctx(local_context, rx, handles, prefix=""):
    """ROCtx twin of _tag_module_forwards: nested push/pop markers keyed
    KEY seq=N around each top-level forward. Duck-typed on the input tensor,
    so no torch import happens in roctx mode. Handles are (module, "forward",
    previous) triples, restorable with _restore_roctx_handles."""
    for m in local_context.get("modules") or []:
        orig = m.forward
        key = str(getattr(m, "key", None) or type(m).__name__)

        def call(*a, _o=orig, _k=prefix + key, _rx=rx, **kw):
            seq = _seq_meta(a, kw)
            with _roctx_range(_rx, f"{_k} seq={seq}"):
                return _o(*a, **kw)
        handles.append((m, "forward", m.__dict__.get("forward", _MISSING)))
        m.forward = call


def _roctx_worker_phase(label, orig_name, local_context, args, kwargs):
    """Shared body of the worker wrappers: resolve the ORIGINAL canonical
    model_tp_fn function at call time (its alias in model_tp is what we
    replace; the canonical module is never touched) and label the pass with
    this rank's own window range. No rx (window closed/failed here) -> plain
    passthrough, so a stale hook can never crash a worker."""
    from exllamav3.model import model_tp_fn
    orig = getattr(model_tp_fn, orig_name)
    rx = _ACTIVE.get("rx")
    if rx is None:
        return orig(local_context, *args, **kwargs)
    with _roctx_range(rx, label):
        return orig(local_context, *args, **kwargs)


# Top-level (PICKLABLE by module+qualname: the child unpickles the replaced
# model_tp alias by reference, and the pseudo-rank runs it inline in the
# parent). _start_cb swaps model_tp's star-imported aliases for these during
# the active window ONLY; shared target embedding/head kernels dispatched
# through a borrowed phase count as MTP work even when their nested module
# label says target (root's exclusive attribution follows the phase range).

def mp_model_forward_target(local_context, *args, **kwargs):
    """phase wrapper: full target forward pass (prefill chunks and decode)."""
    return _roctx_worker_phase("phase=target.forward", "mp_model_forward",
                               local_context, args, kwargs)


def mp_model_forward_mtp_embedding(local_context, *args, **kwargs):
    """phase wrapper: MTP-borrowed target embedding dispatch."""
    return _roctx_worker_phase("phase=mtp.borrowed_embedding",
                               "mp_model_forward_embedding",
                               local_context, args, kwargs)


def mp_model_forward_mtp_head(local_context, *args, **kwargs):
    """phase wrapper: MTP-borrowed target lm_head argmax dispatch."""
    return _roctx_worker_phase("phase=mtp.borrowed_head",
                               "mp_model_forward_lm_head_argmax",
                               local_context, args, kwargs)


_ROCTX_WORKER_ALIASES = (
    ("mp_model_forward", mp_model_forward_target),
    ("mp_model_forward_embedding", mp_model_forward_mtp_embedding),
    ("mp_model_forward_lm_head_argmax", mp_model_forward_mtp_head),
)


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
        out["tagged_modules"] = len(handles)
        nested = _tag_nested_roctx(local_context.get("modules"), rx, handles)
        comm = _tag_backend_roctx(local_context.get("backend"), rx, handles)
        _ACTIVE.update(rx=rx, handles=handles, idx=idx, t0=out["started_unix"])
        out["tagged_nested"] = nested
        out["tagged_comm"] = comm
    except Exception as e:
        _restore_roctx_handles(handles)
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
        _restore_roctx_handles(st.get("handles") or [])
        _ACTIVE.clear()
    return out


def tp_trace_none_start_worker(local_context, spec):
    """Observer-overhead control, SAME boundary as the traced modes: sync
    this rank's own device and open the bookkeeping window - NO profiler,
    NO ROCtx pause/resume/markers, NO module wrappers (nothing is installed,
    so nothing needs restoring either). Elapsed uses the monotonic clock."""
    idx = _device_index(local_context)
    out = {"device": idx, "pid": os.getpid(), "backend": "none",
           "started_unix": time.time()}
    try:
        _sync_device(idx)                              # boundary sync only
        out["synced"] = True
        _ACTIVE.update(idx=idx, t0=time.monotonic(), spec=spec)
    except Exception as e:
        _ACTIVE.clear()
        out["error"] = repr(e)
    return out


def tp_trace_none_stop_worker(local_context):
    idx = _device_index(local_context)
    out = {"device": idx, "pid": os.getpid(), "backend": "none"}
    st = dict(_ACTIVE)
    if "t0" not in st:
        out["error"] = "no active none-control window on this rank (start failed there)"
        _ACTIVE.clear()
        return out
    try:
        _sync_device(idx)                              # boundary sync only
        out["monotonic_elapsed_s"] = round(time.monotonic() - st["t0"], 6)
    except Exception as e:
        out["error"] = repr(e)
    finally:
        _ACTIVE.clear()
    return out


_WORKERS = {"torch": (tp_trace_start_worker, tp_trace_stop_worker),
            "roctx": (tp_trace_roctx_start_worker, tp_trace_roctx_stop_worker),
            "none": (tp_trace_none_start_worker, tp_trace_none_stop_worker)}


# parent orchestration: iterate hook + dispatch of the rank workers

def make_iterate_hook(state, on_start, on_stop):
    def iterate(self, *a, **kw):
        state.gen = self
        try:
            state.pre_iterate(self.num_remaining_jobs(),
                              state.read_tokens(self))
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
        state.post_iterate(self.num_remaining_jobs(), state.read_tokens(self))
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


def _install_parent_roctx(state):
    """Parent-side (output rank) attribution the rank workers cannot see:
    the draft model runs in the PARENT and is not among the rank
    local_context modules, the iterate-phase labels are instance shadows of
    the generator, and the model_tp alias swap labels dispatched worker
    passes. All handles append to the parent's _ACTIVE["handles"]: the
    pseudo-rank stop worker restores them on success, _parent_roctx_undo
    restores them if the stop dispatch never returns. Worker-phase labels
    are pushed from EACH worker's own _ACTIVE rx (or passthrough if none)."""
    rx = _ACTIVE.get("rx")
    if rx is None:
        state.errors.append(
            "roctx attribution: no parent window (output-rank start failed); "
            "draft/phase labels NOT installed")
        return
    handles = _ACTIVE.setdefault("handles", [])
    gen = state.gen
    draft = getattr(gen, "draft_model", None)
    if draft is not None:
        mods = getattr(draft, "modules", None) or []
        _tag_module_forwards_roctx({"modules": mods}, rx, handles,
                                   prefix="draft.")
        _tag_nested_roctx(mods, rx, handles, prefix="draft.")
    for meth, label in (("iterate_draftmodel_mtp_gen", "phase=mtp.draft"),
                        ("iterate_gen", "phase=target.verify_or_ar")):
        orig = getattr(gen, meth, None)
        if not callable(orig) or any(h[0] is gen and h[1] == meth
                                     for h in handles):
            continue

        def call(*a, _o=orig, _rx=rx, _lbl=label, **kw):
            with _roctx_range(_rx, _lbl):
                return _o(*a, **kw)
        handles.append((gen, meth, gen.__dict__.get(meth, _MISSING)))
        setattr(gen, meth, call)
    from exllamav3.model import model_tp
    for alias, wrapper in _ROCTX_WORKER_ALIASES:
        prev = getattr(model_tp, alias, _MISSING)
        if prev is wrapper:                          # already installed
            continue
        handles.append((model_tp, alias, prev))
        setattr(model_tp, alias, wrapper)


def _start_cb(state):
    model = getattr(state.gen, "model", None)
    if model is None or not getattr(model, "loaded_tp", False):
        state.errors.append("trace start: generator exposes no loaded_tp model")
        return False
    start_fn = _WORKERS[state.spec.get("backend", "torch")][0]
    try:
        state.records["start"] = model.tp_worker_dispatch_wait_multi(
            model.active_devices, start_fn, (state.spec,))
    except Exception as e:
        state.errors.append(f"trace start dispatch failed: {e!r}")
        return False
    if state.spec.get("backend") == "roctx":
        try:
            _install_parent_roctx(state)
        except Exception as e:
            state.errors.append(f"roctx attribution install failed: {e!r}")
            try:
                _stop_cb(state)              # close ranks + undo parent
            except Exception as e2:
                state.errors.append(f"stop after failed install: {e2!r}")
            return False
    return True


def _parent_roctx_undo(state, why):
    """Best-effort LOCAL (parent = pseudo-rank) window teardown when the stop
    dispatch never reached it: an open resume/push would otherwise leak every
    remaining group into the external trace. Restores EVERY parent handle:
    rank/nested/comm wrappers plus the draft/gen/worker-alias hooks installed
    by _install_parent_roctx."""
    rx = _ACTIVE.get("rx")                       # only if THIS process opened a window
    try:
        if rx is not None:
            rx.pop()
            rx.pause()
    except Exception as e:
        state.errors.append(f"parent roctx teardown ({why}): {e!r}")
    _restore_roctx_handles(_ACTIVE.pop("handles", None))
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
    if backend == "none":
        return "none_control_no_observation_by_design"
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
    decode_mode = wap.decode_start_tokens is not None
    if not decode_mode and args.mode != "ar":
        raise ValueError("legacy mixed window (--decode-start-tokens unset) is "
                         "AR-only: its first-iterate window includes prefill. "
                         "Use --decode-start-tokens for the decode-only window, "
                         "which admits --mode mtp (coarse capture: draft kernels "
                         "are not yet separately labelled)")
    if decode_mode:
        if wap.decode_start_tokens < 1 or wap.decode_window_tokens < 1:
            raise ValueError("--decode-start-tokens/--decode-window-tokens must be positive")
        if args.batch_size != 1:
            raise ValueError("decode-only window requires --batch-size 1 "
                             "(single Job.new_tokens drives start/stop)")
        # window must COMPLETE strictly before the final (EOS) iterate: allow
        # for MTP acceptance bursts (up to draft_tokens+1 tokens per iterate)
        # at BOTH the start crossing and the stop crossing, plus 1 token of
        # slack so the eos/queue-drain housekeeping iterate is outside the window
        burst = args.draft_tokens if args.mode == "mtp" else 0
        required = (wap.decode_start_tokens + wap.decode_window_tokens
                    + 2 * burst + 1)
        if args.new_tokens < required:
            raise ValueError(
                f"--new-tokens {args.new_tokens} leaves no room for the decode "
                f"window: need >= start({wap.decode_start_tokens}) + "
                f"window({wap.decode_window_tokens}) + 2*draft_burst"
                f"({2 * burst}) + 1 = {required} so the window ends before the "
                "EOS iterate")
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

    if decode_mode:
        state = DecodeWindowState(first_timed_group(prompts, args.batch_size),
                                  wap.decode_start_tokens,
                                  wap.decode_window_tokens)
        rng = (f"tp_decode_window group{state.window_group} "
               f"start{wap.decode_start_tokens}+window{wap.decode_window_tokens}")
    else:
        state = WindowState(first_timed_group(prompts, args.batch_size),
                            wap.trace_iterations)
        rng = (f"tp_trace_window group{state.window_group} "
               f"first{wap.trace_iterations}iterates")
    state.spec = {"trace_dir": str(trace_dir),
                  "trace_iterations": wap.trace_iterations,
                  "backend": wap.trace_backend,
                  "window_mode": "decode" if decode_mode else "mixed",
                  "decode_start_tokens": wap.decode_start_tokens if decode_mode else None,
                  "decode_window_tokens": wap.decode_window_tokens if decode_mode else None,
                  "range": rng}
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
        elif wap.trace_backend == "roctx":
            pinfo = {"backend": "roctx", "no_torch_profiler": True,
                     "shim_pause_env": ROCTX_PAUSE_ENV,
                     "parent_gate": _ROCTX["status"],
                     "external_gpu_trace_required": True,
                     "external_gpu_trace_owner": "root's rocprofv3 launch "
                                                 "(SDK-first LD_PRELOAD, DEEPBIND)",
                     "kernel_observation": "none claimed by this tool: parse the "
                                           "external rocprofv3 CSV against the "
                                           "window/marker records for GPU evidence",
                     "task_phase_attribution": {
                         "labels": "inclusive CPU ranges (nested module tags, "
                                   "tp.<op> comm, phase=..., draft.*); root maps "
                                   "kernels via the HIP runtime Correlation_Id of "
                                   "the launching API call, never GPU/CPU overlap",
                         "worker_phases": ["phase=target.forward",
                                           "phase=mtp.borrowed_embedding",
                                           "phase=mtp.borrowed_head",
                                           "phase=mtp.draft",
                                           "phase=target.verify_or_ar"],
                         "note": "shared target embedding/head kernels dispatched "
                                 "through a borrowed phase count as MTP work even "
                                 "when their nested module label says target"}}
        else:
            pinfo = {"backend": "none", "control": "observer overhead",
                     "no_torch_profiler": True, "no_roctx": True,
                     "boundary": "window detection + per-rank device sync only; "
                                 "no wrappers/profilers/markers installed",
                     "elapsed_clock": "per-rank time.monotonic "
                                      "(monotonic_elapsed_s in rank records)"}
        stop_records = state.records.get("stop") or []
        both = {r.get("device") for r in stop_records if r.get("pid") and "error" not in r}
        for stage in ("start", "stop"):
            for record in state.records.get(stage) or []:
                for key in ("error", "pop_error", "pause_error"):
                    if record.get(key):
                        state.errors.append(f"{stage} rank {record.get('device')}: {key}: {record[key]}")
        if not state.window_calls or len(stop_records) != 2 or both != {0, 1}:
            state.errors.append("a bounded window on both TP ranks was not captured")
        if decode_mode and state.truncated:
            state.errors.append("decode window truncated at group end: the job "
                                "never reached the requested additional tokens "
                                "inside the timed group (room check violated?)")
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
        window_info = {"group": state.window_group,
                       "mode": "decode" if decode_mode else "mixed",
                       "iterate_calls": state.window_calls,
                       "iterates_traced": len(state.window_calls)}
        if decode_mode:
            addl = (state.end_actual - state.start_actual
                    if None not in (state.end_actual, state.start_actual) else None)
            window_info["decode"] = {
                "requested_start_tokens": wap.decode_start_tokens,
                "requested_window_tokens": wap.decode_window_tokens,
                "actual_start_tokens": state.start_actual,
                "actual_end_tokens": state.end_actual,
                "actual_additional_tokens": addl,
                "overshoot_tokens": (max(0, addl - wap.decode_window_tokens)
                                     if addl is not None else None),
                "truncated_at_group_end": state.truncated}
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
            "window": window_info,
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
