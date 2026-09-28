#!/usr/bin/env python3
"""Opt-in (--observe-transfers) transfer/sync instrumentation for profile_stages.

Profiling-side ONLY: this wraps the *aliases* the engine module namespaces hold
(``exllamav3.modules.module.to_device`` and ``exllamav3.util.tensor.to_device``,
the Qwen3 cross-device paths) and ``torch.cuda.synchronize`` at RUNTIME, from
inside the profiling tool. The engine source is never edited; ``uninstall()``
restores the original objects in a finally so no patched function can outlive
the run. With the observer OFF (the default), nothing is patched and behavior
is byte-identical to the Phase-2 harness.

What is counted, per armed window (stage_window arms/resets on entry and
snapshots after the window's closing synchronize, so model load, warmup and the
unprofiled gap are EXCLUDED):

  * CUDA->CUDA moves through the instrumented aliases: per (src,dst,mode)
    count, bytes (numel*element_size of the source), and the HOST WALL of the
    actual copy call. Mode direct/bounced is inferred from before/after deltas
    of ``exllamav3.util.device_copy.stats`` -- needs_bounce is NEVER called an
    extra time and nothing extra is synchronized. probe-triggered copies are
    flagged because their host wall also contains the P2P probe (plus its own
    syncs).
  * Host->device and device->host uploads as SEPARATE counts/bytes (D/H CPU
    params), so layer-boundary param traffic is visible without pretending it
    is inter-GPU DMA.
  * Explicit ``torch.cuda.synchronize`` invocations per device, with host wall
    sums/max. The device argument is resolved (torch.cuda.current_device())
    ONLY when called with an unknown current device (None / bare "cuda"),
    never eagerly per wrapper installation.

Honesty constraints baked into the output (see LEGEND):
  * durations are copy-call / sync-call HOST WALL: they may include waiting
    for pending compute and profiler/shim overhead -- NOT pure DMA time.
  * only moves that go through the instrumented aliases are observed; engine
    or torch transfers that bypass these call sites (plain .to()/.copy_(),
    driver-internal copies, SDMA activity) are NOT counted. Never claim "all
    driver transfers observed".
  * the window's closing boundary synchronize happens while armed; this
    measurement-boundary overhead is distinct from model-internal syncs. The opening sync
    happens before arming, so each armed window contributes exactly
    len(active_devices) closing syncs plus whatever the body called.

Call-site bookkeeping is deliberately defensive: classification and
aggregation errors are dropped rather than changing engine behavior, and an
exception from the ORIGINAL copy call propagates without being recorded as a
transfer (a failed copy moved nothing). Sync calls ARE recorded even when they
raise -- the host waited either way.

CPU-testable by construction: no torch/exllamav3 import at module scope; the
torch module, the alias targets and the device_copy stats dict are injected
(profile_stages resolves the real ones at runtime; the unit tests inject fakes).
"""

from __future__ import annotations

import importlib
import time

KIND_C2C = "cuda_to_cuda"
KIND_H2D = "host_to_device"
KIND_D2H = "device_to_host"
KIND_OTHER = "other"
KIND_NOOP = "noop"

AGG_FIELDS = ("count", "bytes", "wall_ns_sum", "wall_ns_max")


def _new_agg(bytes_known: bool = True) -> dict:
    agg = {f: 0 for f in AGG_FIELDS}
    agg["bytes_known"] = bytes_known
    return agg


def _merge_agg(dst: dict, src: dict) -> None:
    dst["count"] += src["count"]
    dst["wall_ns_sum"] += src["wall_ns_sum"]
    dst["wall_ns_max"] = max(dst["wall_ns_max"], src["wall_ns_max"])
    dst["bytes_known"] = dst["bytes_known"] and src["bytes_known"]
    dst["bytes"] += src["bytes"]


def _new_state() -> dict:
    return {
        "pairs": {},        # (src_idx, dst_idx, mode) -> agg (+probes)
        "h2d": _new_agg(),
        "d2h": _new_agg(),
        "other": _new_agg(),
        "syncs": {},        # "cuda:N" / "cpu" / ... -> agg
        "stats_delta": {"direct": 0, "bounced": 0, "probes": 0},
    }


def _fmt_dev(idx) -> str:
    return f"cuda:{idx}" if idx is not None else "cuda(?)"


class TransferObserver:
    """Wraps to_device aliases + torch.cuda.synchronize; armed per window."""

    #: (module path, attribute) pairs whose CURRENT value is wrapped at
    #: install() and restored at uninstall().
    ENGINE_TARGETS = (("exllamav3.modules.module", "to_device"),
                      ("exllamav3.util.tensor", "to_device"))
    STATS_MODULE = "exllamav3.util.device_copy"

    LEGEND = {
        "duration": ("*_wall_ns values are HOST WALL of the actual copy/sync "
                     "call: may include pending compute wait, the P2P probe and "
                     "submission cost -- NOT pure DMA time"),
        "coverage": ("only moves through the instrumented to_device aliases "
                     "(exllamav3.modules.module / exllamav3.util.tensor) and "
                     "explicit torch.cuda.synchronize calls are observed; "
                     "engine/torch transfers that bypass these call sites "
                     "(.to/.copy_/driver-internal/SDMA) are NOT counted -- "
                     "this is NOT a claim that all driver transfers were observed"),
        "mode": ("direct/bounced inferred from before/after deltas of "
                 "exllamav3.util.device_copy.stats around the original call; "
                 "needs_bounce is never called an extra time and no extra "
                 "synchronize is issued; 'unknown' when the stats dict was not "
                 "available"),
        "window": ("armed only inside profiled stage windows (load/warmup/gap "
                   "excluded); the closing boundary synchronize counts, the "
                   "opening one happens before arming"),
        "bytes": ("source numel*element_size per call; host_to_device/"
                  "device_to_host are separate counts (D/H param uploads), not "
                  "inter-GPU DMA"),
    }

    def __init__(self, torch_mod, targets = None, stats = None, cuda = None):
        """torch_mod: the torch module (used only for device normalization).
        targets: optional list of (module_obj, attr_name); default resolves the
        engine aliases via importlib at install(). stats: optional dict override
        (default resolves device_copy.stats)."""
        self._torch = torch_mod
        self._cuda = cuda if cuda is not None else torch_mod.cuda
        self._targets = None if targets is None else list(targets)
        self._stats = stats
        self._saved: list[tuple] = []
        self.installed = False
        self.recording = False
        self.windows_observed = 0
        self._win = _new_state()
        self._totals = _new_state()

    # ------------------------------------------------------------------ setup
    def install(self) -> list[str]:
        """Swap the alias attributes (and torch.cuda.synchronize) for wrappers.
        Any failure restores everything already patched before re-raising."""
        if self.installed:
            raise RuntimeError("TransferObserver.install() called twice")
        targets = self._targets
        if targets is None:
            targets = [(importlib.import_module(name), attr)
                       for name, attr in self.ENGINE_TARGETS]
        if self._stats is None:
            try:
                self._stats = getattr(importlib.import_module(self.STATS_MODULE), "stats")
            except Exception:
                self._stats = None       # mode falls back to "unknown"
        try:
            for mod, attr in targets:
                orig = getattr(mod, attr)
                if not callable(orig):
                    raise TypeError(f"{getattr(mod, '__name__', mod)}.{attr} is not callable")
                setattr(mod, attr, self._make_to_device_wrapper(orig))
                self._saved.append((mod, attr, orig))
            orig_sync = self._cuda.synchronize
            self._cuda.synchronize = self._make_sync_wrapper(orig_sync)
            self._saved.append((self._cuda, "synchronize", orig_sync))
        except Exception as e:
            leftovers = self.uninstall()      # best-effort restore, then report
            detail = f" (after partial install: {'; '.join(leftovers)})" if leftovers else ""
            raise RuntimeError(f"TransferObserver install failed{detail}: {e!r}") from e
        self.installed = True
        return [f"{getattr(mod, '__name__', '?')}.{attr}"
                for mod, attr, _ in self._saved]

    def uninstall(self) -> list[str]:
        """Restore every saved attribute (never raises); returns error strings
        for anything that could NOT be restored, so the caller can record them
        as cleanup failures instead of swallowing them."""
        self.recording = False
        errors: list[str] = []
        while self._saved:
            mod, attr, orig = self._saved.pop()      # reverse order
            try:
                setattr(mod, attr, orig)
            except Exception as e:
                errors.append(f"restore {getattr(mod, '__name__', '?')}.{attr}: {e!r}")
        self.installed = False
        return errors

    # ----------------------------------------------------------- window state
    def begin_window(self) -> None:
        if not self.installed:
            raise RuntimeError("observer not installed")
        if self.recording:
            raise RuntimeError("begin_window() while a window is already armed "
                               "(stage windows must not nest)")
        self._win = _new_state()
        self.recording = True

    def end_window(self) -> dict:
        if not self.recording:
            raise RuntimeError("end_window() without begin_window()")
        self.recording = False
        snap = self._render(self._win)
        self._merge_into(self._totals, self._win)
        self._win = _new_state()
        self.windows_observed += 1
        return snap

    def totals(self) -> dict:
        out = self._render(self._totals)
        out["windows_observed"] = self.windows_observed
        return out

    # ------------------------------------------------------------- internals
    def _stats_now(self) -> dict:
        if self._stats is None:
            return {}
        try:
            return {k: int(self._stats.get(k, 0)) for k in ("direct", "bounced", "probes")}
        except Exception:
            return {}

    @staticmethod
    def _stats_delta(before: dict, after: dict) -> dict:
        keys = set(before) | set(after)
        return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in keys}

    def _classify_move(self, t, device) -> tuple[str, int | None, int | None]:
        """Pure-metadata classification (no GPU calls, no needs_bounce):
        mirrors what the ORIGINAL to_device will decide to do."""
        if device is None:                      # original: emulate tensor.to(None) no-op
            return KIND_NOOP, None, None
        src = t.device
        dev = self._torch.device(device)
        if src == dev:
            return KIND_NOOP, None, None
        st, dt = getattr(src, "type", None), getattr(dev, "type", None)
        si, di = getattr(src, "index", None), getattr(dev, "index", None)
        if st == "cuda" and dt == "cuda":
            if si is not None and si == di:     # same physical device
                return KIND_NOOP, None, None
            return KIND_C2C, si, di
        if st == "cuda" and dt == "cpu":
            return KIND_D2H, si, None
        if st == "cpu" and dt == "cuda":
            return KIND_H2D, None, di
        return KIND_OTHER, None, None

    def _mode_from_delta(self, delta: dict) -> str:
        if delta.get("bounced", 0) > 0:
            return "bounced"
        if delta.get("direct", 0) > 0:
            return "direct"
        return "unknown"

    def _record_move(self, kind, src, dst, nbytes, delta, wall_ns) -> None:
        for key in ("direct", "bounced", "probes"):
            self._win["stats_delta"][key] += int(delta.get(key, 0))
        if kind == KIND_NOOP:
            return
        probes = int(delta.get("probes", 0))
        if kind == KIND_C2C:
            mode = self._mode_from_delta(delta)
            agg = self._win["pairs"].setdefault((src, dst, mode), _new_agg())
            if "probes" not in agg:
                agg["probes"] = 0
        else:
            agg = {"host_to_device": self._win["h2d"],
                   "device_to_host": self._win["d2h"]}.get(kind, self._win["other"])
        agg["count"] += 1
        agg["wall_ns_sum"] += wall_ns
        agg["wall_ns_max"] = max(agg["wall_ns_max"], wall_ns)
        if nbytes is None:
            agg["bytes_known"] = False
        else:
            agg["bytes"] += nbytes
        if kind == KIND_C2C and probes:
            agg["probes"] += probes

    def _sync_key(self, device) -> str:
        """Per-device attribution; torch.cuda.current_device() is queried only
        when the call itself does not pin the device (None / bare 'cuda')."""
        if device is None:
            return f"cuda:{self._cuda.current_device()}"
        if isinstance(device, int):
            return f"cuda:{device}"
        dev = device
        if getattr(dev, "type", None) is None:
            dev = self._torch.device(device)
        if getattr(dev, "type", None) == "cuda":
            idx = getattr(dev, "index", None)
            if idx is None:
                idx = self._cuda.current_device()
            return f"cuda:{idx}"
        t = getattr(dev, "type", None)
        return t if isinstance(t, str) else "unresolved"

    def _record_sync(self, key, wall_ns) -> None:
        agg = self._win["syncs"].setdefault(key, _new_agg(bytes_known = False))
        agg["count"] += 1
        agg["wall_ns_sum"] += wall_ns
        agg["wall_ns_max"] = max(agg["wall_ns_max"], wall_ns)

    def _make_to_device_wrapper(self, orig):
        def wrapper(*args, **kwargs):
            if not self.recording:
                return orig(*args, **kwargs)
            kind, src, dst, nbytes = KIND_OTHER, None, None, None
            try:
                t = args[0] if args else kwargs["t"]
                device = args[1] if len(args) > 1 else kwargs.get("device")
                kind, src, dst = self._classify_move(t, device)
                if kind != KIND_NOOP:
                    nbytes = int(t.numel()) * int(t.element_size())
            except Exception:
                kind, src, dst, nbytes = KIND_OTHER, None, None, None
            before = self._stats_now()
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)      # exceptions propagate: a failed
            wall_ns = (time.perf_counter() - t0) * 1e9   # copy is NOT recorded
            delta = self._stats_delta(before, self._stats_now())
            try:
                self._record_move(kind, src, dst, nbytes, delta, wall_ns)
            except Exception:
                pass                          # bookkeeping never breaks the engine
            return out
        wrapper.__name__ = f"transfer_observed_{getattr(orig, '__name__', 'to_device')}"
        wrapper._observer_original = orig
        return wrapper

    def _make_sync_wrapper(self, orig):
        def wrapper(*args, **kwargs):
            if not self.recording:
                return orig(*args, **kwargs)
            device = args[0] if args else kwargs.get("device")
            try:
                key = self._sync_key(device)
            except Exception:
                key = "unresolved"
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                wall_ns = (time.perf_counter() - t0) * 1e9   # a raising sync still waited
                try:
                    self._record_sync(key, wall_ns)
                except Exception:
                    pass
        wrapper.__name__ = "transfer_observed_synchronize"
        wrapper._observer_original = orig
        return wrapper

    # ---------------------------------------------------------------- output
    @staticmethod
    def _agg_out(agg: dict) -> dict:
        out = {"count": agg["count"],
               "bytes": agg["bytes"] if agg.get("bytes_known", True) else None,
               "host_wall_ns_sum": agg["wall_ns_sum"],
               "host_wall_ns_max": agg["wall_ns_max"]}
        return out

    def _merge_into(self, dst: dict, src: dict) -> None:
        for key, agg in src["pairs"].items():
            d = dst["pairs"].setdefault(key, _new_agg())
            if "probes" not in d:
                d["probes"] = 0
            _merge_agg(d, agg)
            d["probes"] += agg.get("probes", 0)
        for name in ("h2d", "d2h", "other"):
            _merge_agg(dst[name], src[name])
        for key, agg in src["syncs"].items():
            d = dst["syncs"].setdefault(key, _new_agg(bytes_known = False))
            _merge_agg(d, agg)
        for k, v in src["stats_delta"].items():
            dst["stats_delta"][k] += v

    def _render(self, state: dict) -> dict:
        pairs = []
        c2c_count = c2c_bytes = 0
        for (src, dst, mode) in sorted(state["pairs"], key = lambda k: tuple(str(x) for x in k)):
            agg = state["pairs"][(src, dst, mode)]
            out = self._agg_out(agg)
            out.update({"src": _fmt_dev(src), "dst": _fmt_dev(dst), "mode": mode,
                        "probe_triggered_calls": agg.get("probes", 0)})
            pairs.append(out)
            c2c_count += agg["count"]
            if agg.get("bytes_known", True):
                c2c_bytes += agg["bytes"]
        syncs = {key: self._agg_out(agg) for key, agg in sorted(state["syncs"].items())}
        return {
            "cuda_to_cuda": {"count": c2c_count,
                             "bytes": c2c_bytes if all(a["bytes_known"] for a in state["pairs"].values()) else None,
                             "pairs": pairs},
            "host_to_device": self._agg_out(state["h2d"]),
            "device_to_host": self._agg_out(state["d2h"]),
            "other_moves": self._agg_out(state["other"]),
            "device_copy_stats_delta": dict(state["stats_delta"]),
            "synchronizes": {"count": sum(a["count"] for a in syncs.values()),
                             "per_device": syncs},
            "legend": dict(self.LEGEND),
        }
