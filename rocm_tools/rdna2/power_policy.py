# rocm_tools/rdna2/power_policy.py
"""Opt-in energy-saving power-policy CLIENT for V620 benchmarks (no root here).

Promotes the runs/qwen38/benchmark_phase_switch.py hook pattern into a reusable
adapter. Speaks the helper (power_switch_server.py) protocol unchanged: one JSON
line {"mode": auto|profile_peak, "label": str} -> one event whose "after" must
confirm the mode. This client never spawns sudo per transition and never touches
sysfs; a missing helper or unconfirmed response fails loudly so a run is never
silently labelled policy-controlled.

Policy (energy saving; MTP speculation now authorized):
  max_batch_size == 1: auto during prefill; profile_peak BEFORE the first drafting
      computation of the first decode iteration (MTP / ordinary draft / dflash run
      earlier than target iterate_gen inside Generator.iterate; plain AR too); auto
      restored at queue-drained BEFORE on_queue_drained() housekeeping.
  max_batch_size  > 1: profile_peak on enter, no per-job toggles, auto on exit.

torch is injected (CPU tests fake it). Per-device torch.cuda.synchronize() precedes
each transition and is recorded SEPARATELY from the helper RPC (~6ms up / ~5ms down
measured; the ~16ms sync is NOT switching cost).
"""
from __future__ import annotations

import json
import os
import socket
import time

MODES = ("auto", "profile_peak")
_ACTIVE = None  # entered adapter currently owning the policy, if any


class PowerPolicyError(RuntimeError):
    """Helper missing / protocol violation / adapter misuse. Fail closed."""


def attach(generator, torch, configured_batch_size, used_devices, socket_path):
    """Return a context-managed adapter for `generator` (enter it to activate)."""
    return PowerPolicyAdapter(generator, torch, configured_batch_size,
                              used_devices, socket_path)


class PowerPolicyAdapter:
    def __init__(self, generator, torch, configured_batch_size, used_devices, socket_path):
        if type(configured_batch_size) is not int or configured_batch_size < 1:
            raise PowerPolicyError("configured_batch_size must be an int >= 1")
        self.gen = generator
        self.torch = torch
        self.batch = configured_batch_size
        self.devices = list(used_devices)
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise PowerPolicyError("used_devices must be nonempty and unique")
        self.socket_path = str(socket_path)
        self.records = []          # per-transition dicts; sync_ms vs rpc_ms separate
        self.applied = None        # last mode the helper confirmed via response["after"]
        self._sock = None
        self._fh = None
        self._saved = []           # (obj, name, bound original, had_own) for restore-on-exit
        self._in_decode = False
        self._failed = False

    # -- socket RPC ----------------------------------------------------------
    def _connect(self):
        if not os.path.exists(self.socket_path):
            raise PowerPolicyError(
                f"power helper socket '{self.socket_path}' is missing; start the privileged "
                "power_switch_server.py first. This client never spawns sudo and never writes sysfs.")
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(self.socket_path)
            self._fh = self._sock.makefile("rwb", buffering=0)
        except OSError as e:
            self._drop_connection()
            raise PowerPolicyError(f"cannot connect power helper at '{self.socket_path}': {e!r}") from e

    def switch(self, mode, label):
        """Sync the used GPUs, ask the helper for `mode`, verify, and record. Raises PowerPolicyError."""
        try:
            return self._switch(mode, label)
        except BaseException:
            self._failed = True
            raise

    def _switch(self, mode, label):
        if mode not in MODES:
            raise PowerPolicyError(f"invalid mode {mode!r}, expected one of {MODES}")
        if self._fh is None:
            raise PowerPolicyError("power helper is not connected (enter the adapter first)")
        start = time.perf_counter_ns()
        for dev in self.devices:
            self.torch.cuda.synchronize(dev)
        sync_ms = (time.perf_counter_ns() - start) / 1e6
        t = time.perf_counter_ns()
        try:
            self._sock.sendall((json.dumps({"mode": mode, "label": str(label)}) + "\n").encode())
            line = self._fh.readline()
        except OSError as e:
            self._drop_connection()
            raise PowerPolicyError(f"power helper link failed during {mode!r}: {e!r}") from e
        rpc_ms = (time.perf_counter_ns() - t) / 1e6
        if not line:
            self._drop_connection()
            raise PowerPolicyError("power helper disconnected mid-transition; policy state is unknown")
        try:
            rec = json.loads(line)
        except ValueError as e:
            raise PowerPolicyError(f"power helper sent invalid JSON ({e}); refusing to trust it") from e
        after = rec.get("after") if isinstance(rec, dict) else None
        if not isinstance(after, list) or len(after) != len(self.devices) or any(v != mode for v in after):
            raise PowerPolicyError(f"helper did not confirm mode {mode!r} (after={after!r}); "
                                   "this run must NOT be labelled policy-controlled")
        rec.update(mode=mode, label=str(label), sync_ms=sync_ms, rpc_ms=rpc_ms,
                   boundary_total_ms=(time.perf_counter_ns() - start) / 1e6)
        self.records.append(rec)
        self.applied = mode
        return rec

    def _drop_connection(self):
        fh, sock = self._fh, self._sock
        self._sock = self._fh = None
        try:
            if fh: fh.close()
        finally:
            if sock: sock.close()

    # -- generator hooks (batch == 1 only) ------------------------------------
    def _first_compute_attr(self):
        g = self.gen
        if getattr(g, "draft_model", None):
            if getattr(g, "dflash_draft", False):
                return "iterate_draftmodel_dflash_gen"   # drafts before target iterate_gen
            if getattr(g, "mtp_draft", False):
                return "iterate_draftmodel_mtp_gen"      # MTP runs earlier inside Generator.iterate
            return "iterate_draftmodel_gen"
        if getattr(g, "ngram_match_min", 0):
            return "iterate_ngram_gen"
        return "iterate_gen"

    def _install_hooks(self):
        targets = [(self._first_compute_attr(), self._before_compute),
                   ("on_queue_drained", self._before_drained)]
        for name, hook in targets:
            orig = getattr(self.gen, name, None)
            if not callable(orig):
                raise PowerPolicyError(f"Generator API changed? missing method '{name}'; "
                                       "refusing to guess where decode starts")
            had_own = name in self.gen.__dict__   # pre-existing instance override? keep it on restore
            setattr(self.gen, name, (lambda o, h: lambda *a, **k: h(o, *a, **k))(orig, hook))
            self._saved.append((self.gen, name, orig, had_own))

    def _before_compute(self, orig, *args, **kwargs):
        jobs = getattr(self.gen, "active_jobs", ())
        if not self._in_decode and jobs and all(j.is_prefill_done() for j in jobs):
            self.switch("profile_peak", "before-first-decode")
            self._in_decode = True
        return orig(*args, **kwargs)

    def _before_drained(self, orig, *args, **kwargs):
        if self._in_decode:
            self._in_decode = False
            self.switch("auto", "after-final-decode")    # restore BEFORE housekeeping work
        return orig(*args, **kwargs)

    def _restore_hooks(self):
        errors = []
        while self._saved:
            obj, name, orig, had_own = self._saved.pop()
            try:
                if had_own:
                    setattr(obj, name, orig)
                else:
                    delattr(obj, name)   # leave no instance shadow over the class method
            except Exception as e:
                errors.append(f"restore {name}: {e!r}")
        return errors

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None:
            raise PowerPolicyError("another generator already owns the power policy; "
                                   "only one adapter may be attached at a time")
        _ACTIVE = self
        try:
            self._connect()
            if self.batch > 1:
                self.switch("profile_peak", "attach-batch-gt1")  # keep through all inference
            else:
                self.switch("auto", "attach-batch1")
                self._install_hooks()
        except BaseException as exc:
            self._failed = True
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def close(self):
        self._drop_connection()

    def __exit__(self, exc_type, exc, tb):
        errors = self._restore_hooks()
        try:
            if self._fh is not None and (self.applied == "profile_peak" or self._in_decode or self._failed):
                self.switch("auto", "context-exit")
        except Exception as e:
            errors.append(f"return-to-auto: {e!r}")
        finally:
            self._in_decode = False
            try:
                self.close()
            except Exception as e:
                errors.append(f"socket close: {e!r}")
            global _ACTIVE
            if _ACTIVE is self:
                _ACTIVE = None
        if errors:
            self._failed = True
            msg = "power-policy cleanup failed: " + "; ".join(errors)
            if exc is None:
                raise PowerPolicyError(msg)
            if hasattr(exc, "add_note"):  # surface alongside, never hiding, the original error
                exc.add_note(msg)
            else:
                exc.power_policy_cleanup_error = msg
        return False

    # -- reporting ------------------------------------------------------------
    def summary(self):
        """Switch cost = rpc_ms; sync_ms is queue-drain bookkeeping, not switching cost."""
        return {"policy": "phase-switched" if self.batch == 1 else "held-profile-peak",
                "transitions": [(r["mode"], r["label"], r["rpc_ms"], r["sync_ms"]) for r in self.records],
                "rpc_ms_by_mode": {m: [r["rpc_ms"] for r in self.records if r["mode"] == m] for m in MODES},
                "sync_ms_total": sum(r["sync_ms"] for r in self.records),
                "helper_unverified": self._failed or not self.records}
