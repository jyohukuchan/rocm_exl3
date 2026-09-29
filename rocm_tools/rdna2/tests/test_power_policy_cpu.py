#!/usr/bin/env python3
"""CPU-only tests for rocm_tools/rdna2/power_policy.py.

Fakes: a Unix-socket helper speaking the power_switch_server.py JSON-line
protocol, a duck-typed torch (records synchronize calls) and a duck-typed
Generator. NOTHING HERE TOUCHES A GPU, sysfs or sudo; they prove the client's
policy ordering and fail-closed behavior only.

Run from the repo root:
    python3 -m unittest rocm_tools.rdna2.tests.test_power_policy_cpu
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2 import power_policy


class FakeHelper(threading.Thread):
    """Protocol-compatible stand-in for power_switch_server.py."""

    def __init__(self, path, responder=None):
        super().__init__(daemon=True)
        self.path, self.responder = str(path), responder
        self.requests = []
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(str(path))
        self.srv.listen(1)

    def run(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            with conn:
                with conn.makefile("rwb", buffering=0) as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        req = json.loads(line)
                        self.requests.append(req)
                        reply = self.responder(req) if self.responder else {
                            "label": req["label"], "mode": req["mode"],
                            "after": [req["mode"], req["mode"]], "both_write_ms": 0.5}
                        f.write((json.dumps(reply) + "\n").encode())

    def stop(self):
        try:
            self.srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.srv.close()
        self.join(timeout=1)
        os.unlink(self.path)   # a Unix socket file outlives its listener unless removed


class FakeTorch:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.cuda = SimpleNamespace(synchronize=lambda dev: self.events.append(("sync", dev)))


class FakeJob:
    def __init__(self, done):
        self.done = done

    def is_prefill_done(self):
        return self.done


class FakeGenerator:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.active_jobs = kw.get("active_jobs", [])
        self.calls = []
        self.probe = lambda: None

    def _m(self, name):
        def hook(results=None, draft_tokens=None):
            self.calls.append((name, self.probe()))
        return hook

    def iterate_gen(self, results=None, draft_tokens=None): self._m("iterate_gen")(results)
    def iterate_draftmodel_gen(self, results=None): self._m("iterate_draftmodel_gen")(results)
    def iterate_draftmodel_mtp_gen(self, results=None): self._m("iterate_draftmodel_mtp_gen")(results)
    def iterate_draftmodel_dflash_gen(self, results=None): self._m("iterate_draftmodel_dflash_gen")(results)
    def on_queue_drained(self): self._m("on_queue_drained")()


class PowerPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sock = Path(self.tmp.name) / "power-switch.sock"
        self.helper = FakeHelper(self.sock)
        self.helper.start()

    def tearDown(self):
        self.helper.stop()
        self.tmp.cleanup()

    # ---- single-batch phase switching ---------------------------------------
    def test_ar_switches_at_prefill_boundary_and_back_on_drain(self):
        gen = FakeGenerator()
        events = []
        pol = power_policy.attach(gen, FakeTorch(events), 1, [0, 1], self.sock)
        gen.probe = lambda: pol.applied
        with pol:
            gen.active_jobs = [FakeJob(False)]
            gen.iterate_gen([])                       # mid-prefill: stay auto
            self.assertEqual(pol.applied, "auto")
            gen.active_jobs = [FakeJob(True)]
            gen.iterate_gen([])                       # decode: switch BEFORE target compute
            self.assertEqual(gen.calls[-1], ("iterate_gen", "profile_peak"))
            gen.iterate_gen([])                       # once only, no per-iteration RPC
            self.assertEqual(len(pol.records), 2)
            gen.on_queue_drained()                    # auto restored BEFORE housekeeping
            self.assertEqual(gen.calls[-1], ("on_queue_drained", "auto"))
        self.assertEqual(pol.applied, "auto")         # already drained; exit adds no RPC
        self.assertEqual([r["label"] for r in pol.records],
                         ["attach-batch1", "before-first-decode", "after-final-decode"])
        self.assertEqual(events[:2], [("sync", 0), ("sync", 1)])   # sync precedes the RPC
        self.assertEqual(self.helper.requests, [
            {"mode": "auto", "label": "attach-batch1"},
            {"mode": "profile_peak", "label": "before-first-decode"},
            {"mode": "auto", "label": "after-final-decode"}])

    def test_mtp_switches_before_first_drafting_computation(self):
        gen = FakeGenerator(draft_model=object(), mtp_draft=True)
        pol = power_policy.attach(gen, FakeTorch(), 1, [0, 1], self.sock)
        gen.probe = lambda: pol.applied
        with pol:
            gen.active_jobs = [FakeJob(True)]
            gen.iterate_draftmodel_mtp_gen([])        # MTP runs earlier than iterate_gen
            self.assertEqual(gen.calls[-1], ("iterate_draftmodel_mtp_gen", "profile_peak"))
            gen.iterate_gen([])                       # target verify: already peaked, no RPC
            self.assertEqual(gen.calls[-1], ("iterate_gen", "profile_peak"))
        self.assertEqual([r["mode"] for r in pol.records], ["auto", "profile_peak", "auto"])

    def test_dflash_and_ordinary_draft_select_right_entry_point(self):
        for kw, attr in [({"draft_model": object(), "dflash_draft": True}, "iterate_draftmodel_dflash_gen"),
                         ({"draft_model": object()}, "iterate_draftmodel_gen")]:
            with self.subTest(attr=attr):
                gen = FakeGenerator(**kw)
                with power_policy.attach(gen, FakeTorch(), 1, [0, 1], self.sock) as pol:
                    self.assertIn(attr, [entry[1] for entry in pol._saved])

    # ---- batch > 1: hold profile_peak, no toggling ---------------------------
    def test_batch_gt1_holds_peak_on_attach_and_restores_on_exit(self):
        gen = FakeGenerator()
        pol = power_policy.attach(gen, FakeTorch(), 2, [0, 1], self.sock)
        with pol:
            self.assertEqual(pol._saved, [])          # no per-job hooks installed
            gen.active_jobs = [FakeJob(True), FakeJob(True)]
            gen.iterate_gen([])                       # untouched engine methods
            gen.on_queue_drained()                    # no toggling mid-run
            self.assertEqual(len(pol.records), 1)
        self.assertEqual([r["label"] for r in pol.records], ["attach-batch-gt1", "context-exit"])
        self.assertEqual(gen.calls, [("iterate_gen", None), ("on_queue_drained", None)])

    # ---- cleanup & errors -----------------------------------------------------
    def test_exception_in_body_still_restores_methods_and_auto(self):
        gen = FakeGenerator()
        pol = power_policy.attach(gen, FakeTorch(), 1, [0, 1], self.sock)
        gen.probe = lambda: pol.applied
        with self.assertRaises(ValueError):
            with pol:
                gen.active_jobs = [FakeJob(True)]
                gen.iterate_gen([])
                raise ValueError("boom")
        self.assertNotIn("on_queue_drained", gen.__dict__)   # instance shadow removed
        self.assertEqual(gen.on_queue_drained.__func__, FakeGenerator.on_queue_drained)
        self.assertEqual([r["mode"] for r in pol.records], ["auto", "profile_peak", "auto"])

    def test_missing_helper_fails_loudly_without_touching_generator(self):
        gen = FakeGenerator()
        pol = power_policy.attach(gen, FakeTorch(), 1, [0, 1], Path(self.tmp.name) / "absent.sock")
        with self.assertRaises(power_policy.PowerPolicyError):
            with pol:
                pass
        self.assertEqual(pol._saved, [])

    def test_unconfirmed_response_never_labels_policy_controlled(self):
        self.helper.stop()
        bad = FakeHelper(self.sock, responder=lambda req: {"mode": req["mode"], "after": ["auto", "auto"]})
        bad.start()
        self.helper = bad
        gen = FakeGenerator()
        pol = power_policy.attach(gen, FakeTorch(), 1, [0, 1], self.sock)
        with self.assertRaises(power_policy.PowerPolicyError):
            with pol:
                gen.active_jobs = [FakeJob(True)]
                gen.iterate_gen([])

    def test_double_attach_rejected_and_owner_slot_freed(self):
        gen = FakeGenerator()
        first = power_policy.attach(gen, FakeTorch(), 1, [0, 1], self.sock)
        with first:
            with self.assertRaises(power_policy.PowerPolicyError):
                with power_policy.attach(gen, FakeTorch(), 1, [0, 1], self.sock):
                    pass
        with power_policy.attach(gen, FakeTorch(), 1, [0, 1], self.sock):   # slot released
            pass

    def test_partial_device_acknowledgement_is_rejected(self):
        self.helper.stop()
        self.helper = FakeHelper(self.sock, responder=lambda req: {"after": [req["mode"]]})
        self.helper.start()
        pol = power_policy.attach(FakeGenerator(), FakeTorch(), 1, [0, 1], self.sock)
        self.assertTrue(pol.summary()["helper_unverified"])
        with self.assertRaises(power_policy.PowerPolicyError):
            with pol:
                pass
        self.assertTrue(pol.summary()["helper_unverified"])
        self.assertIsNone(power_policy._ACTIVE)

    def test_records_keep_rpc_and_sync_separate(self):
        pol = power_policy.attach(FakeGenerator(), FakeTorch(), 1, [0, 1], self.sock)
        with pol:
            pol.switch("profile_peak", "manual")
        r = pol.records[0]
        for key in ("rpc_ms", "sync_ms", "boundary_total_ms"):
            self.assertGreaterEqual(r[key], 0.0)
        s = pol.summary()
        self.assertEqual(s["helper_unverified"], False)
        self.assertEqual(len(s["rpc_ms_by_mode"]["profile_peak"]), 1)


if __name__ == "__main__":
    unittest.main()
