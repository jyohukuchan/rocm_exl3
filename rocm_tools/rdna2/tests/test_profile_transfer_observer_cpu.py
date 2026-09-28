"""CPU-only tests for profile_transfer_observer (no GPU, no exllamav3 needed).

The observer is dependency-injected: the torch module (only its .device
normalizer is used), the alias targets and the device_copy stats dict are
passed in, so these tests run with REAL torch device objects for
classification but fake tensor/cuda/module stand-ins for everything
GPU-facing. Covers: install / uninstall restoration (including
partial-install rollback), pass-through when disarmed, CUDA->CUDA
direct/bounced/probe attribution via stats deltas WITHOUT calling
needs_bounce, H2D/D2H separate counts, same-device/noop exclusion, failed
copy not counted / failed sync still counted, per-window arm/reset/snapshot
isolation, totals merge, explicit-sync keying (current device resolved only
when the call itself leaves it unknown), state-machine refusals, and the
honest HOST WALL / coverage-limit legend.

Run from the repo root:
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2.profile_transfer_observer import TransferObserver


class FakeTensor:
    """Duck-typed tensor: device + sizes only (what the observer reads)."""

    def __init__(self, device, numel = 1024, element_size = 2):
        self.device = torch.device(device)
        self._numel, self._esize = numel, element_size

    def numel(self):
        return self._numel

    def element_size(self):
        return self._esize


class FakeCuda:
    def __init__(self, current = 0, fail = False):
        self.sync_log = []
        self.current_calls = 0
        self._current = current
        self._fail = fail
        self.synchronize = self._sync          # instance attr: patchable

    def _sync(self, device = None):
        self.sync_log.append(device)
        if self._fail:
            raise RuntimeError("synthetic synchronize failure")

    def current_device(self):
        self.current_calls += 1
        return self._current


def make_orig(stats, mode_box, probe_first = False, raise_exc = False):
    """Stands in for the real device_copy.to_device: increments the stats the
    same way on a genuine cross-device move (mode_box[0] switches the verdict
    between windows WITHOUT reinstalling) and returns a marker object so call
    forwarding is testable. probe_first mimics the one-time P2P probe riding
    on the first cross-device call."""
    calls = []
    probed = [False]

    def orig(t, device, non_blocking = False):
        calls.append((t, device, non_blocking))
        if raise_exc:
            raise RuntimeError("synthetic copy failure")
        if device is None:
            return "SENTINEL"              # engine behavior: tensor.to(None) no-op
        src, dst = t.device, torch.device(device)
        if (src.type == "cuda" and dst.type == "cuda"
                and src.index != (dst.index if dst.index is not None else 0)):
            if probe_first and not probed[0]:
                stats["probes"] += 1
                probed[0] = True
            stats[mode_box[0]] += 1
        return "SENTINEL"
    return orig, calls


def build_observer(mode = "direct", probe_first = False, fail = False):
    stats = {"direct": 0, "bounced": 0, "probes": 0}
    mode_box = [mode]
    orig, calls = make_orig(stats, mode_box, probe_first = probe_first,
                            raise_exc = fail)
    ns1 = types.SimpleNamespace(to_device = orig)
    ns2 = types.SimpleNamespace(to_device = orig)
    cuda = FakeCuda()
    obs = TransferObserver(torch, targets = [(ns1, "to_device"), (ns2, "to_device")],
                           stats = stats, cuda = cuda)
    obs.mode_box = mode_box          # tests may switch the verdict between windows
    return obs, (ns1, ns2), cuda, stats, calls


class TestInstallRestore(unittest.TestCase):
    def test_install_swaps_all_originals(self):
        obs, (ns1, ns2), cuda, stats, _ = build_observer()
        orig = ns1.to_device
        orig_sync = cuda.synchronize
        wrapped = obs.install()
        self.assertTrue(obs.installed)
        self.assertIsNot(ns1.to_device, orig)
        self.assertIsNot(ns2.to_device, orig)
        self.assertIsNot(cuda.synchronize, orig_sync)
        self.assertIs(ns1.to_device._observer_original, orig)
        self.assertIs(cuda.synchronize._observer_original, orig_sync)
        self.assertEqual(len(wrapped), 3)            # two aliases + synchronize
        obs.uninstall()

    def test_uninstall_restores_identity_and_reports_nothing(self):
        obs, (ns1, ns2), cuda, _, _ = build_observer()
        orig = ns1.to_device
        orig_sync = cuda.synchronize
        obs.install()
        errors = obs.uninstall()
        self.assertEqual(errors, [])
        self.assertFalse(obs.installed)
        self.assertIs(ns1.to_device, orig)
        self.assertIs(ns2.to_device, orig)
        self.assertIs(cuda.synchronize, orig_sync)

    def test_double_install_refuses(self):
        obs, *_ = build_observer()
        obs.install()
        with self.assertRaises(RuntimeError):
            obs.install()
        obs.uninstall()

    def test_partial_install_failure_rolls_back(self):
        stats = {"direct": 0, "bounced": 0, "probes": 0}
        orig, _ = make_orig(stats, ["direct"])
        good = types.SimpleNamespace(to_device = orig)
        broken = types.SimpleNamespace()               # no to_device attribute
        cuda = FakeCuda()
        orig_sync = cuda.synchronize
        obs = TransferObserver(torch, targets = [(good, "to_device"),
                                                 (broken, "to_device")],
                               stats = stats, cuda = cuda)
        with self.assertRaises(RuntimeError):
            obs.install()
        self.assertFalse(obs.installed)
        self.assertFalse(obs.recording)
        self.assertIs(good.to_device, orig)            # rolled back, not left patched
        self.assertIs(cuda.synchronize, orig_sync)     # never patched


class TestPassThroughDisarmed(unittest.TestCase):
    def test_installed_but_disarmed_calls_original_untouched(self):
        obs, (ns1, _), cuda, stats, calls = build_observer()
        obs.install()
        out = ns1.to_device(FakeTensor("cuda:0"), "cuda:1", non_blocking = True)
        self.assertEqual(out, "SENTINEL")
        self.assertEqual(calls[0][2], True)            # kwarg forwarded verbatim
        cuda.synchronize(torch.device("cuda:0"))
        snap = obs.totals()
        self.assertEqual(snap["cuda_to_cuda"]["count"], 0)
        self.assertEqual(snap["synchronizes"]["count"], 0)
        self.assertEqual(snap["device_copy_stats_delta"],
                         {"direct": 0, "bounced": 0, "probes": 0})
        obs.uninstall()


class TestWindowAccounting(unittest.TestCase):
    def run_in_window(self, fn, **kw):
        obs, (ns1, _), cuda, stats, _ = build_observer(**kw)
        obs.install()
        obs.begin_window()
        fn(ns1, cuda, stats)
        return obs, obs.end_window()

    def test_direct_cuda_to_cuda_pair(self):
        obs, snap = self.run_in_window(
            lambda ns1, cuda, stats: ns1.to_device(FakeTensor("cuda:0", 512, 4), "cuda:1"))
        c2c = snap["cuda_to_cuda"]
        self.assertEqual(c2c["count"], 1)
        self.assertEqual(c2c["bytes"], 2048)
        self.assertEqual(c2c["pairs"][0]["src"], "cuda:0")
        self.assertEqual(c2c["pairs"][0]["dst"], "cuda:1")
        self.assertEqual(c2c["pairs"][0]["mode"], "direct")
        self.assertGreater(c2c["pairs"][0]["host_wall_ns_sum"], 0.0)   # HOST WALL
        self.assertLessEqual(c2c["pairs"][0]["host_wall_ns_max"],
                             c2c["pairs"][0]["host_wall_ns_sum"])
        self.assertEqual(snap["device_copy_stats_delta"]["direct"], 1)
        self.assertEqual(snap["device_copy_stats_delta"]["bounced"], 0)
        obs.uninstall()

    def test_bounced_and_probe_attribution(self):
        # both calls bounce; the first one also pays the one-time P2P probe
        def fn(ns1, cuda, stats):
            ns1.to_device(FakeTensor("cuda:1"), "cuda:0")
            ns1.to_device(FakeTensor("cuda:1", 64, 1), "cuda:0")
        obs, snap = self.run_in_window(fn, mode = "bounced", probe_first = True)
        c2c = snap["cuda_to_cuda"]
        self.assertEqual(c2c["count"], 2)
        self.assertEqual(c2c["bytes"], 2048 + 64)
        self.assertEqual(len(c2c["pairs"]), 1)         # same (src,dst,mode) aggregate
        pair = c2c["pairs"][0]
        self.assertEqual(pair["mode"], "bounced")
        self.assertEqual(pair["probe_triggered_calls"], 1)
        self.assertEqual(snap["device_copy_stats_delta"],
                         {"direct": 0, "bounced": 2, "probes": 1})
        obs.uninstall()

    def test_h2d_d2h_are_separate_counts(self):
        def fn(ns1, cuda, stats):
            ns1.to_device(FakeTensor("cpu", 100, 1), "cuda:0")     # H2D upload
            ns1.to_device(FakeTensor("cuda:0", 100, 1), "cpu")     # D2H readback
        obs, snap = self.run_in_window(fn)
        self.assertEqual(snap["host_to_device"]["count"], 1)
        self.assertEqual(snap["host_to_device"]["bytes"], 100)
        self.assertEqual(snap["device_to_host"]["count"], 1)
        self.assertEqual(snap["cuda_to_cuda"]["count"], 0)         # not inter-GPU DMA
        self.assertEqual(snap["device_copy_stats_delta"]["direct"], 0)
        obs.uninstall()

    def test_same_device_and_none_are_not_recorded(self):
        def fn(ns1, cuda, stats):
            ns1.to_device(FakeTensor("cuda:0"), "cuda:0")
            ns1.to_device(FakeTensor("cuda:0"), None)
        obs, snap = self.run_in_window(fn)
        self.assertEqual(snap["cuda_to_cuda"]["count"], 0)
        self.assertEqual(snap["host_to_device"]["count"], 0)
        self.assertEqual(snap["other_moves"]["count"], 0)
        obs.uninstall()

    def test_failed_copy_propagates_and_is_not_recorded(self):
        def fn(ns1, cuda, stats):
            with self.assertRaises(RuntimeError):
                ns1.to_device(FakeTensor("cuda:0"), "cuda:1")
        obs, snap = self.run_in_window(fn, fail = True)
        self.assertEqual(snap["cuda_to_cuda"]["count"], 0)         # nothing moved
        self.assertEqual(snap["device_copy_stats_delta"]["direct"], 0)
        obs.uninstall()

    def test_windows_are_isolated_and_totals_merge(self):
        obs, (ns1, _), cuda, stats, _ = build_observer(mode = "bounced")
        obs.install()
        obs.begin_window()
        ns1.to_device(FakeTensor("cuda:0"), "cuda:1")              # bounced now
        w1 = obs.end_window()
        self.assertEqual([p["mode"] for p in w1["cuda_to_cuda"]["pairs"]], ["bounced"])
        obs.mode_box[0] = "direct"                                 # verdict changes
        obs.begin_window()
        ns1.to_device(FakeTensor("cuda:0"), "cuda:1")              # direct now
        w2 = obs.end_window()
        self.assertEqual([p["mode"] for p in w2["cuda_to_cuda"]["pairs"]], ["direct"])
        tot = obs.totals()
        self.assertEqual(tot["windows_observed"], 2)
        self.assertEqual({p["mode"] for p in tot["cuda_to_cuda"]["pairs"]},
                         {"bounced", "direct"})
        self.assertEqual(tot["device_copy_stats_delta"],
                         {"direct": 1, "bounced": 1, "probes": 0})
        obs.uninstall()

    def test_state_machine_refusals(self):
        obs, *_ = build_observer()
        with self.assertRaises(RuntimeError):
            obs.begin_window()                                     # not installed
        obs.install()
        with self.assertRaises(RuntimeError):
            obs.end_window()                                       # not armed
        obs.begin_window()
        with self.assertRaises(RuntimeError):
            obs.begin_window()                                     # no nesting
        obs.end_window()
        obs.uninstall()


class TestSyncObservation(unittest.TestCase):
    def test_per_device_keying_and_lazy_current_resolution(self):
        obs, (ns1, _), cuda, stats, _ = build_observer()
        obs.install()
        obs.begin_window()
        base = cuda.current_calls
        cuda.synchronize(torch.device("cuda:1"))                   # explicit: no resolve
        cuda.synchronize(2)                                        # int: no resolve
        cuda.synchronize(torch.device("cuda"))                     # bare: resolve
        cuda.synchronize()                                         # None: resolve
        snap = obs.end_window()
        self.assertEqual(cuda.current_calls, base + 2)             # only the two unknowns
        devs = snap["synchronizes"]["per_device"]
        self.assertEqual(devs["cuda:1"]["count"], 1)
        self.assertEqual(devs["cuda:2"]["count"], 1)
        self.assertEqual(devs["cuda:0"]["count"], 2)               # FakeCuda current = 0
        self.assertEqual(snap["synchronizes"]["count"], 4)
        self.assertGreaterEqual(devs["cuda:1"]["host_wall_ns_sum"], 0.0)
        obs.uninstall()

    def test_failed_sync_is_still_counted(self):
        stats = {"direct": 0, "bounced": 0, "probes": 0}
        cuda = FakeCuda(fail = True)
        orig, _ = make_orig(stats, ["direct"])
        obs = TransferObserver(torch,
                               targets = [(types.SimpleNamespace(to_device = orig),
                                           "to_device")],
                               stats = stats, cuda = cuda)
        obs.install()
        obs.begin_window()
        with self.assertRaises(RuntimeError):
            cuda.synchronize(torch.device("cuda:1"))
        snap = obs.end_window()
        self.assertEqual(snap["synchronizes"]["per_device"]["cuda:1"]["count"], 1)
        obs.uninstall()

    def test_disarmed_sync_passthrough_no_accounting(self):
        obs, (ns1, _), cuda, stats, _ = build_observer()
        obs.install()
        cuda.synchronize(torch.device("cuda:0"))
        self.assertEqual(cuda.sync_log, [torch.device("cuda:0")])  # original executed
        self.assertEqual(obs.totals()["synchronizes"]["count"], 0)
        obs.uninstall()


class TestHonestLabels(unittest.TestCase):
    def test_unknown_bytes_remain_unknown_across_calls_and_windows(self):
        obs, *_ = build_observer()
        obs.install()
        try:
            obs.begin_window()
            obs._record_move("cuda_to_cuda", 0, 1, None, {"direct": 1}, 10)
            obs._record_move("cuda_to_cuda", 0, 1, 16, {"direct": 1}, 10)
            self.assertIsNone(obs.end_window()["cuda_to_cuda"]["bytes"])
            obs.begin_window()
            obs._record_move("cuda_to_cuda", 0, 1, 32, {"direct": 1}, 10)
            self.assertEqual(obs.end_window()["cuda_to_cuda"]["bytes"], 32)
            self.assertIsNone(obs.totals()["cuda_to_cuda"]["bytes"])
            self.assertEqual(obs.totals()["cuda_to_cuda"]["count"], 3)
        finally:
            obs.uninstall()

    def test_legend_states_host_wall_and_coverage_limits(self):
        legend = TransferObserver.LEGEND
        self.assertIn("HOST WALL", legend["duration"])
        self.assertIn("NOT pure DMA", legend["duration"])
        self.assertIn("NOT a claim that all driver transfers were observed",
                      legend["coverage"])
        self.assertIn("needs_bounce is never called an extra time", legend["mode"])
        self.assertIn("armed only inside profiled stage windows", legend["window"])

    def test_every_snapshot_carries_the_legend(self):
        obs, *_ = build_observer()
        obs.install()
        obs.begin_window()
        snap = obs.end_window()
        self.assertIn("duration", snap["legend"])
        self.assertIn("coverage", obs.totals()["legend"])
        obs.uninstall()


if __name__ == "__main__":
    unittest.main()
