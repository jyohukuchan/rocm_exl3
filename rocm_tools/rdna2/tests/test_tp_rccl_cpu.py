#!/usr/bin/env python3
"""CPU-only tests for the ROCm/RCCL tensor-parallel collective backend
(exllamav3/model/model_tp_rccl.py) and its dispatch in model_tp_fn.init_pg.

Two layers of evidence:

  1. A fake torch.distributed that blocks like the real synchronous API: one
     thread per simulated rank, collectives rendezvous on condition variables.
     This exercises rank mapping with a nontrivial active_devices order,
     contribution semantics (zeros, never uninitialized/NaN), float32
     retention, staging of noncontiguous tensors, gather ordering with uneven
     and zero widths and an output-only rank, and close / partial-init safety.
  2. Real torch.distributed (gloo, CPU tensors) end to end: an in-process
     world-1 construct/close/re-init, and a two-process scenario in spawned
     subprocesses hitting the exact dist API the backend uses
     (init_process_group, broadcast, all_reduce, send, recv, barrier,
     destroy_process_group).

NOTHING HERE TOUCHES A GPU: host torch is CPU-only, gloo runs on CPU tensors,
and the native extension is never imported (model_tp_rccl calls no ext/pg_*).
The fakes prove the logic; gloo proves the distributed API usage. RCCL on HIP
hardware is NOT claimed or tested here -- that is the root 2x gfx1030 step.

Run from the repo root (or anywhere):
    python3 -m pytest -q -p no:cacheprovider rocm_tools/rdna2/tests/test_tp_rccl_cpu.py
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from collections import deque
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import torch
import torch.distributed as dist

# The real exllamav3/__init__ imports the native extension, which does not exist
# on this host. Load modules under test through an alias package whose root
# __path__ points at exllamav3/, so relative imports resolve to the REAL light
# modules (util) while heavy deps (ext, native backends) receive stubs.
ALIAS = "tp_rccl_tests"

# A value exactly representable in float32 but rounded to 1.0 by bfloat16: any
# precision downcast in the reduction path changes it.
FP32_PROBE = 1.0 + 2.0 ** -9


def _alias_root():
    if ALIAS not in sys.modules:
        root = types.ModuleType(ALIAS)
        root.__path__ = [str(REPO / "exllamav3")]
        sys.modules[ALIAS] = root
        model = types.ModuleType(ALIAS + ".model")
        model.__path__ = [str(REPO / "exllamav3" / "model")]
        sys.modules[ALIAS + ".model"] = model
    return sys.modules[ALIAS]


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def load_rccl():
    _alias_root()
    name = ALIAS + ".model.model_tp_rccl"
    if name not in sys.modules:
        # `from ..util import log_tp` then resolves through the alias path to
        # the real, torch-only util package.
        _load_file(name, REPO / "exllamav3" / "model" / "model_tp_rccl.py")
    return sys.modules[name]


# ---------------------------------------------------------------------------
# fake torch.distributed: per-rank views over one shared rendezvous state.
# Collectives block until peers arrive, matching the real synchronous API so
# that scenario call order cannot mask a bug.
# ---------------------------------------------------------------------------

class _FakeWork:
    def wait(self):
        pass


class _GroupState:
    RENDEZVOUS_S = 30

    def __init__(self):
        self.cv = threading.Condition()
        self.world_size = None
        self.ar_bufs = []          # tensors pending in the current all_reduce round
        self.ar_round = 0
        self.barrier_count = 0
        self.barrier_round = 0
        self.bcast = {}            # src_rank -> list[payload bytes]
        self.mailboxes = {}        # (src_rank, dst_rank) -> list[payload bytes]
        self.allreduce_calls = []  # (rank, dtype, is_contiguous)
        self.broadcast_calls = []  # (rank, src_rank)
        self.sends = []            # (src_rank, dst_rank, nbytes)
        self.recvs = []            # (dst_rank, src_rank, nbytes)
        self.barriers = []         # timeout kwargs passed to dist.barrier
        self.destroy_count = 0


def _wire_bytes(t):
    return t.detach().reshape(-1).view(torch.uint8).clone()


def _write_wire(payload, dst):
    assert dst.is_contiguous(), "fake wire writes only into contiguous buffers"
    assert payload.numel() == dst.numel() * dst.element_size(), "wire size mismatch"
    dst.reshape(-1).view(torch.uint8).copy_(payload)


class FakeDist:
    """One instance per simulated rank; ranks share a _GroupState."""

    def __init__(self, state):
        self.state = state
        self.rank = None
        self.initialized = False
        self.init_kwargs = None
        self._bcast_pos = {}   # src_rank -> broadcasts from src consumed HERE
        self._recv_pos = {}    # src_rank -> sends from src to me consumed HERE

    def init_process_group(self, backend = None, rank = None, world_size = None,
                           init_method = None, timeout = None, **kw):
        self.rank = rank
        with self.state.cv:
            self.state.world_size = world_size
        self.initialized = True
        self.init_kwargs = dict(backend = backend, rank = rank, world_size = world_size,
                                init_method = init_method, timeout = timeout)

    def is_initialized(self):
        return self.initialized

    def get_world_size(self, group = None):
        return self.state.world_size

    def get_rank(self, group = None):
        return self.rank

    def destroy_process_group(self, group = None):
        with self.state.cv:
            self.state.destroy_count += 1
        self.initialized = False

    def barrier(self, group = None, async_op = False, timeout = None):
        st = self.state
        with st.cv:
            st.barriers.append(timeout)
            my = st.barrier_round
            st.barrier_count += 1
            if st.barrier_count == st.world_size:
                st.barrier_count = 0
                st.barrier_round += 1
                st.cv.notify_all()
            else:
                ok = st.cv.wait_for(lambda: st.barrier_round > my, timeout = st.RENDEZVOUS_S)
                assert ok, "barrier rendezvous timeout"

    def broadcast(self, tensor, src = None, group = None, async_op = False):
        st = self.state
        st.broadcast_calls.append((self.rank, src))
        payload = None
        with st.cv:
            if self.rank == src:
                st.bcast.setdefault(src, []).append(_wire_bytes(tensor))
                st.cv.notify_all()
            else:
                i = self._bcast_pos.get(src, 0)
                self._bcast_pos[src] = i + 1
                ok = st.cv.wait_for(lambda: len(st.bcast.get(src, [])) > i,
                                    timeout = st.RENDEZVOUS_S)
                assert ok, "broadcast rendezvous timeout"
                payload = st.bcast[src][i]
        if payload is not None:
            _write_wire(payload, tensor)
        return _FakeWork()

    def all_reduce(self, tensor, op = None, group = None, async_op = False):
        st = self.state
        st.allreduce_calls.append((self.rank, tensor.dtype, tensor.is_contiguous()))
        with st.cv:
            my = st.ar_round
            st.ar_bufs.append(tensor)
            if len(st.ar_bufs) == st.world_size:
                total = None
                for t in st.ar_bufs:
                    tc = t.detach().clone(memory_format = torch.contiguous_format)
                    total = tc if total is None else total + tc
                for t in st.ar_bufs:
                    t.copy_(total)
                st.ar_bufs = []
                st.ar_round += 1
                st.cv.notify_all()
            else:
                ok = st.cv.wait_for(lambda: st.ar_round > my, timeout = st.RENDEZVOUS_S)
                assert ok, "all_reduce rendezvous timeout"
        return _FakeWork()

    def send(self, tensor, dst = None, group = None, async_op = False):
        st = self.state
        st.sends.append((self.rank, dst, tensor.numel() * tensor.element_size()))
        with st.cv:
            st.mailboxes.setdefault((self.rank, dst), []).append(_wire_bytes(tensor))
            st.cv.notify_all()
        return _FakeWork()

    def recv(self, tensor, src = None, group = None, async_op = False):
        st = self.state
        st.recvs.append((self.rank, src, tensor.numel() * tensor.element_size()))
        with st.cv:
            i = self._recv_pos.get(src, 0)
            self._recv_pos[src] = i + 1
            ok = st.cv.wait_for(lambda: len(st.mailboxes.get((src, self.rank), [])) > i,
                                timeout = st.RENDEZVOUS_S)
            assert ok, "recv rendezvous timeout"
            payload = st.mailboxes[(src, self.rank)][i]
        _write_wire(payload, tensor)
        return _FakeWork()


class ExplodingDist:
    """Any collective the CPU helper might attempt raises."""
    def __getattr__(self, name):
        raise AssertionError(f"CPU helper must not touch torch.distributed (called {name})")


class ThreadLocalDist:
    """Proxy installed as model_tp_rccl.dist during a scenario: dispatches each
    call to the fake bound to the calling thread's rank."""
    def __init__(self):
        self._local = threading.local()

    def set(self, fd):
        self._local.fd = fd

    def __getattr__(self, name):
        fd = getattr(self._local, "fd", None)
        if fd is None:
            raise AssertionError(f"no fake dist bound to thread {threading.current_thread().name}")
        return getattr(fd, name)


@contextlib.contextmanager
def use_dist(module, fake):
    old = module.dist
    module.dist = fake
    try:
        yield fake
    finally:
        module.dist = old


@contextlib.contextmanager
def patch(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield value
    finally:
        setattr(obj, name, old)


def make_split(rccl, active, out_dev, warmup = False):
    """
    Construct one fake-dist-backed TPBackendRCCL per device in `active`.

    The RCCL warmup collective rendezvouses all ranks, so rank constructors run
    concurrently on their own threads when warmup=True; with warmup=False (the
    default, to keep logic scenarios focused) construction is sequential and
    the warmup hook is stubbed -- real warmup behavior is covered by
    test_real_class_warmup_reduces_once_per_rank and the gloo subprocess runs.
    """
    state = _GroupState()
    fds = {dev: FakeDist(state) for dev in active}
    bes = {}
    errors = {}

    cls = rccl.TPBackendRCCL
    if not warmup:
        cls = type("NoWarmupRCCL", (cls,), {"mp_warmup_rccl": lambda self: None})

    tl = ThreadLocalDist()

    def build(dev):
        tl.set(fds[dev])
        try:
            bes[dev] = cls(
                device = dev,
                active_devices = active,
                output_device = out_dev,
                init_method = "tcp://127.0.0.1:1234",
                master = (dev == out_dev),
                uuid = "test-uuid",
                timeout_s = 30.0,
                close_barrier_timeout_s = 5.0,
            )
        except BaseException as e:
            errors[dev] = e

    with use_dist(rccl, tl):
        if warmup:
            threads = [threading.Thread(target = build, args = (dev,)) for dev in active]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout = 30)
            assert not errors, errors
            assert not any(t.is_alive() for t in threads), "warmup rendezvous hung"
        else:
            for dev in active:
                build(dev)
            assert not errors, errors
    return bes, fds, state, tl


def run_threads(seqs):
    """seqs: list of (name, fn). Runs each on a thread; returns errors dict."""
    errors = {}

    def target(name, fn):
        try:
            fn()
        except BaseException as e:
            errors[name] = e

    threads = {}
    for name, fn in seqs:
        threads[name] = threading.Thread(target = target, args = (name, fn), name = name)
    for t in threads.values():
        t.start()
    for t in threads.values():
        t.join(timeout = 30)
    hung = [n for n, t in threads.items() if t.is_alive()]
    if hung:
        raise AssertionError(f"threads did not complete: {hung}")
    return errors


# ---------------------------------------------------------------------------
# fake-dist logic tests
# ---------------------------------------------------------------------------

class TestRCCLFakeCollectives(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rccl = load_rccl()

    def setUp(self):
        self.assertFalse(dist.is_initialized(), "test must not leak a real process group")

    def _run(self, bes, fds, tl, seqs):
        def wrap(dev):
            def f():
                tl.set(fds[dev])
                seqs[dev](bes[dev])
            return f
        with use_dist(self.rccl, tl):
            errors = run_threads([(f"rank-dev{dev}", wrap(dev)) for dev in seqs])
        if errors:
            name, e = next(iter(errors.items()))
            raise AssertionError(f"{name} raised: {e!r}") from e

    def test_cpu_helper_touches_no_gpu_no_dist_no_native(self):
        rccl = self.rccl
        with use_dist(rccl, ExplodingDist()):
            be = rccl.TPBackendRCCL(
                device = -1, active_devices = [2, 7], output_device = 7,
                init_method = "tcp://127.0.0.1:1", master = False, uuid = "u")
            self.assertEqual(be.rank, -1)
            self.assertIsNone(be.run_cpu_reduce_jobs())
            self.assertIsNone(be.end_cpu_reduce_jobs())
            for op in (lambda: be.fwd_barrier(),
                       lambda: be.broadcast(torch.zeros(2), 2),
                       lambda: be.all_reduce(torch.zeros(2)),
                       lambda: be.gather(torch.zeros(2), None, [2], 2, [2]),
                       lambda: be.gather_small(torch.zeros(2), None, [2], 2, [2])):
                with self.assertRaises(AssertionError):
                    op()
            be.close()
            be.close()  # repeat close safe

    def test_rank_mapping_and_init_args(self):
        active = [5, 2, 7]  # nontrivial order, output device last (in-process pseudo-worker)
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 7)
        self.assertEqual([bes[d].rank for d in active], [0, 1, 2])
        for dev, fd in fds.items():
            self.assertEqual(fd.init_kwargs["backend"], "nccl")  # RCCL joins as "nccl" on HIP
            self.assertEqual(fd.init_kwargs["rank"], bes[dev].rank)
            self.assertEqual(fd.init_kwargs["world_size"], 3)
            tmo = fd.init_kwargs["timeout"]
            self.assertIsInstance(tmo, timedelta)
            self.assertGreater(tmo.total_seconds(), 0)
            self.assertLessEqual(tmo.total_seconds(), 3600)  # finite, not the 30-min default
        with use_dist(self.rccl, tl):
            tl.set(fds[5])
            with self.assertRaises(ValueError):
                bes[5].broadcast(torch.zeros(1), src_device = 9)  # device not in the split

    def test_real_class_warmup_reduces_once_per_rank(self):
        bes, fds, st, tl = make_split(self.rccl, [0, 1], out_dev = 1, warmup = True)
        self.assertEqual(len(st.allreduce_calls), 2)   # exactly the two constructor warmups
        self.assertTrue(all(dt == torch.float32 for (_, dt, _) in st.allreduce_calls))
        self.assertTrue(all(c for (_, _, c) in st.allreduce_calls))

    def test_broadcast_maps_src_device_to_src_rank(self):
        active = [5, 2, 7]
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 7)
        val = torch.tensor([FP32_PROBE, -3.25], dtype = torch.float32)
        got = {dev: torch.zeros(2, dtype = torch.float32) for dev in active}

        def seq_src(be):
            be.broadcast(val.clone(), src_device = 2)

        def seq_recv(dev):
            def f(be):
                be.broadcast(got[dev], src_device = 2)
            return f

        self._run(bes, fds, tl, {2: seq_src, 5: seq_recv(5), 7: seq_recv(7)})
        for dev in (5, 7):
            self.assertTrue(torch.equal(got[dev], val), f"device {dev} got {got[dev]}")
        self.assertEqual({src for (_, src) in st.broadcast_calls}, {1})  # device 2 -> rank 1

    def test_all_reduce_float32_kept_poisoned_non_contributor_contributes_zeros(self):
        active = [3, 4]
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 4)
        mine = torch.tensor([FP32_PROBE, 0.5], dtype = torch.float32)
        theirs = torch.full((2,), float("nan"), dtype = torch.float32)  # must contribute zeros

        self._run(bes, fds, tl, {
            3: lambda be: be.all_reduce(mine, True),
            4: lambda be: be.all_reduce(theirs, False),
        })
        expect = torch.tensor([FP32_PROBE, 0.5])
        self.assertTrue(torch.equal(mine, expect), f"float32 precision lost: {mine}")
        self.assertTrue(torch.equal(theirs, expect), f"result not delivered: {theirs}")
        self.assertTrue(all(dt == torch.float32 for (_, dt, _) in st.allreduce_calls))
        self.assertNotIn(torch.bfloat16, {dt for (_, dt, _) in st.allreduce_calls})

    def test_all_reduce_noncontiguous_staged(self):
        active = [0, 1]
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 1)
        base = torch.zeros((1, 1, 8), dtype = torch.float32)
        view = base[..., ::2]                       # width 4, noncontiguous
        view.fill_(FP32_PROBE)
        other = torch.ones((1, 1, 4), dtype = torch.float32)
        self._run(bes, fds, tl, {
            0: lambda be: be.all_reduce(view),
            1: lambda be: be.all_reduce(other),
        })
        self.assertTrue(torch.equal(view, torch.full((1, 1, 4), FP32_PROBE + 1.0)))
        self.assertTrue(torch.equal(other, torch.full((1, 1, 4), FP32_PROBE + 1.0)))
        # what actually reached dist.all_reduce was always contiguous (staged)
        self.assertTrue(all(c for (_, _, c) in st.allreduce_calls))

    def test_all_reduce_single_rank_world(self):
        bes, fds, st, tl = make_split(self.rccl, [0], out_dev = 0)
        with use_dist(self.rccl, tl):
            tl.set(fds[0])
            t = torch.full((3,), 2.5)
            bes[0].all_reduce(t, True)
            self.assertTrue(torch.equal(t, torch.full((3,), 2.5)))
            z = torch.full((3,), 7.0)
            bes[0].all_reduce(z, False)
            self.assertTrue(torch.equal(z, torch.zeros(3)))
        self.assertEqual(st.allreduce_calls, [])    # world-1 needs no collective at all

    def test_gather_reordered_uneven_zero_widths_output_only_noncontig_out(self):
        # rank map: dev1->0, dev0->1, dev2->2; the output device holds NO slice of its own
        active = [1, 0, 2]
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 2)
        gather_devices = [1, 2, 0]                  # contributor order != active order
        ldims = [2, 0, 3]                           # dev1: 2 cols, dev2 (out): 0, dev0: 3 cols
        out_base = torch.zeros((1, 2, 10), dtype = torch.float32)
        out = out_base[..., ::2]                    # width 5 (== sum ldims), noncontiguous
        c1 = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        c0 = torch.tensor([[[10.0, 11.0, 12.0], [10.0, 11.0, 12.0]]])
        self._run(bes, fds, tl, {
            1: lambda be: be.gather(c1.clone(), None, gather_devices, 2, ldims),
            0: lambda be: be.gather(c0.clone(), None, gather_devices, 2, ldims),
            2: lambda be: be.gather(torch.zeros((1, 2, 0), dtype = torch.float32), out,
                                    gather_devices, 2, ldims),
        })
        self.assertTrue(torch.equal(out[..., 0:2], c1))
        self.assertTrue(torch.equal(out[..., 2:5], c0))
        self.assertTrue((out_base[..., 1::2] == 0).all(), "strided gaps must be untouched")
        self.assertEqual(sorted(st.sends), sorted([(0, 2, c1.numel() * 4), (1, 2, c0.numel() * 4)]))
        self.assertEqual([src for (_, src, _) in st.recvs], [0, 1])  # gather_devices order

    def test_gather_devices_accepts_tensor_and_zero_width_sends_nothing(self):
        active = [1, 0]
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 0)
        gd = torch.tensor([1, 0])
        own = torch.tensor([[[42], [43]], [[44], [45]]])
        out = torch.empty((2, 2, 1), dtype = torch.long)
        self._run(bes, fds, tl, {
            1: lambda be: be.gather_small(torch.zeros((2, 2, 0), dtype = torch.long),
                                          None, gd, 0, [0, 1]),
            0: lambda be: be.gather_small(own, out, gd, 0, [0, 1]),
        })
        self.assertEqual(st.sends, [])              # zero-width contributor sent nothing
        self.assertTrue(torch.equal(out, own))

    def test_gather_small_two_contributors_one_p2p_pair(self):
        active = [1, 0]                             # reversed device/rank mapping
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 0)
        vals = {1: torch.tensor([[1.5]], dtype = torch.float16),
                0: torch.tensor([[2.5]], dtype = torch.float16)}
        out = torch.empty((1, 2), dtype = torch.float16)
        self._run(bes, fds, tl, {
            1: lambda be: be.gather_small(vals[1], None, [1, 0], 0, [1, 1]),
            0: lambda be: be.gather_small(vals[0], out, [1, 0], 0, [1, 1]),
        })
        self.assertTrue(torch.equal(out, torch.tensor([[1.5, 2.5]])))
        self.assertEqual(len(st.sends), 1)          # one send/recv pair, no per-rank storm
        self.assertEqual(len(st.recvs), 1)

    def test_nonparticipant_and_bad_inputs_raise(self):
        active = [1, 9]                             # out dev 1 -> rank 0; dev 9 -> rank 1
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 1)

        def dev9_calls(be):
            with self.assertRaises(AssertionError):     # neither out nor contributor
                be.gather(torch.zeros((1, 1, 1)), None, [1], 1, [1])
            with self.assertRaises(AssertionError):     # contributor width != ldims entry
                be.gather(torch.ones((1, 1, 3)), None, [1, 9], 1, [1, 2])
            with self.assertRaises(AssertionError):     # gather_devices must not be None
                be.gather(torch.ones((1, 1, 2)), None, None, 1, [2])

        def out_calls(be):
            with self.assertRaises(AssertionError):     # out width != sum(ldims)
                be.gather(torch.ones((1, 1, 1)), torch.zeros((1, 1, 4)), [1, 9], 1, [1, 2])
            with self.assertRaises(AssertionError):     # out rank must supply out tensor
                be.gather(torch.ones((1, 1, 1)), None, [1, 9], 1, [1, 2])

        self._run(bes, fds, tl, {9: dev9_calls, 1: out_calls})
        self.assertEqual(st.sends, [])

    def test_fwd_barrier_hits_group_barrier(self):
        active = [0, 1]
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 1)
        self._run(bes, fds, tl, {0: lambda be: be.fwd_barrier(),
                                 1: lambda be: be.fwd_barrier()})
        self.assertEqual(st.barriers.count(None), 2)

    def test_close_bounded_idempotent_and_blocks_later_ops(self):
        active = [0, 1]
        bes, fds, st, tl = make_split(self.rccl, active, out_dev = 1)
        self._run(bes, fds, tl, {0: lambda be: be.close(),
                                 1: lambda be: be.close()})
        bes[0].close()                              # repeat close is a no-op
        self.assertEqual(st.destroy_count, 2)
        close_tmos = [t for t in st.barriers if t is not None]
        self.assertEqual(len(close_tmos), 2)        # one bounded drain barrier per rank
        self.assertTrue(all(isinstance(t, timedelta) and t.total_seconds() <= 5.0
                            for t in close_tmos))
        with use_dist(self.rccl, tl):
            tl.set(fds[0])
            with self.assertRaises(AssertionError):
                bes[0].broadcast(torch.zeros(1), 0)

    def test_close_survives_barrier_error(self):
        bes, fds, st, tl = make_split(self.rccl, [0], out_dev = 0)
        def boom(*a, **kw):
            raise RuntimeError("simulated: peer already gone")
        with use_dist(self.rccl, tl):
            tl.set(fds[0])
            fds[0].barrier = boom
            bes[0].close()
        self.assertEqual(st.destroy_count, 1)       # teardown proceeds past a failing barrier

    def test_close_survives_no_per_op_timeout_support(self):
        bes, fds, st, tl = make_split(self.rccl, [0], out_dev = 0)
        def legacy_barrier(*a, **kw):
            if "timeout" in kw:
                raise TypeError("barrier() got an unexpected keyword argument 'timeout'")
        with use_dist(self.rccl, tl):
            tl.set(fds[0])
            fds[0].barrier = legacy_barrier
            bes[0].close()
        self.assertEqual(st.destroy_count, 1)       # legacy torch: skip barrier, still destroy

    def test_partial_init_failure_creates_no_group(self):
        rccl = self.rccl
        st = _GroupState()
        fd = FakeDist(st)
        def bad_init(*a, **kw):
            raise RuntimeError("simulated rendezvous failure")
        fd.init_process_group = bad_init
        with use_dist(rccl, fd):
            with self.assertRaises(RuntimeError):
                rccl.TPBackendRCCL(device = 0, active_devices = [0], output_device = 0,
                                   init_method = "tcp://127.0.0.1:1", master = False,
                                   uuid = "u", timeout_s = 5.0)
        self.assertFalse(fd.initialized)
        self.assertEqual(st.destroy_count, 0)

    def test_warmup_failure_still_closes_cleanly(self):
        rccl = self.rccl
        leaked = []

        class Leak(rccl.TPBackendRCCL):
            def mp_warmup_rccl(self):
                leaked.append(self)
                raise RuntimeError("simulated RCCL comm init failure")

        st = _GroupState()
        fd = FakeDist(st)
        tl = ThreadLocalDist()
        tl.set(fd)
        with use_dist(rccl, tl):
            with self.assertRaises(RuntimeError):
                Leak(device = 0, active_devices = [0], output_device = 0,
                     init_method = "tcp://127.0.0.1:1", master = False, uuid = "u",
                     timeout_s = 5.0, close_barrier_timeout_s = 2.0)
        be = leaked[0]
        self.assertTrue(fd.initialized)             # group exists despite ctor raising
        self.assertTrue(be.pg_active and not be.closed)
        with use_dist(rccl, tl):
            be.close()
            be.close()
        self.assertEqual(st.destroy_count, 1)       # exactly one teardown
        self.assertFalse(fd.initialized)


# ---------------------------------------------------------------------------
# init_pg / worker dispatch: real model_tp_fn with stubbed heavy deps
# ---------------------------------------------------------------------------

class TestInitPgDispatch(unittest.TestCase):

    shared_calls = []

    @classmethod
    def setUpClass(cls):
        _alias_root()
        if ALIAS + ".util" not in sys.modules:
            importlib.import_module(ALIAS + ".util")
        cls.rccl = load_rccl()

    def _forbidden_ext(self):
        def make(name):
            def f(*a, **k):
                raise AssertionError(f"native ext.{name} must not be called")
            return f
        names = ("pg_init_context", "pg_barrier", "pg_broadcast", "pg_broadcast_ll",
                 "pg_all_reduce_cpu", "pg_gather", "pg_gather_small",
                 "run_cpu_reduce_jobs", "end_cpu_reduce_jobs")
        return types.SimpleNamespace(**{n: make(n) for n in names})

    def _stub_env(self):
        calls = {"NCCL": [], "Native": []}
        TestInitPgDispatch.shared_calls = shared_calls = []

        ext = types.ModuleType("ext")
        ext.exllamav3_ext = self._forbidden_ext()
        sys.modules[ALIAS + ".ext"] = ext

        backend = types.ModuleType("model_tp_backend")

        class RecNCCL:
            def __init__(self, **kw):
                calls["NCCL"].append(kw)
            def close(self):
                pass

        class RecNative:
            def __init__(self, **kw):
                calls["Native"].append(kw)
            def close(self):
                pass

        backend.TPBackendNCCL = RecNCCL
        backend.TPBackendNative = RecNative
        sys.modules[ALIAS + ".model.model_tp_backend"] = backend

        shared = types.ModuleType("model_tp_shared")

        class RecConsumer:
            def __init__(self, producer_imp, device = None, pin_memory = False):
                shared_calls.append(("consumer", device, pin_memory))
            def close(self):
                pass

        shared.SMProducer = object
        shared.SMConsumer = RecConsumer
        sys.modules[ALIAS + ".model.model_tp_shared"] = shared

        tok = types.ModuleType("tokenizer")
        emb = types.ModuleType("tokenizer.mm_embedding")
        emb.recv_embeddings = lambda *a, **k: None
        tok.mm_embedding = emb
        sys.modules[ALIAS + ".tokenizer"] = tok
        sys.modules[ALIAS + ".tokenizer.mm_embedding"] = emb

        sys.modules.pop(ALIAS + ".model.model_tp_fn", None)
        fn = _load_file(ALIAS + ".model.model_tp_fn",
                        REPO / "exllamav3" / "model" / "model_tp_fn.py")
        # fn must have bound the REAL RCCL class from the alias tree
        self.assertIs(fn.TPBackendRCCL, self.rccl.TPBackendRCCL)
        # neutralize process-level side effects of the worker entry point
        fn.install_parent_death_signal = lambda: False
        fn.signal = types.SimpleNamespace(signal = lambda *a: None, SIGINT = 2, SIG_IGN = 1)
        return fn, calls

    def _cuda_probe(self, rec, raise_on_call = False):
        def set_device(d):
            rec.append(("set_device", d))
            if raise_on_call:
                raise AssertionError("cuda touched on the CPU helper")
        def synchronize(*a):
            rec.append(("synchronize",))
            if raise_on_call:
                raise AssertionError("cuda synchronize on the CPU helper")
        return set_device, synchronize

    def test_hip_nccl_selects_rccl_cuda_build_keeps_native_nccl(self):
        fn, calls = self._stub_env()
        rec = []
        sd, sy = self._cuda_probe(rec)
        args = {"type": "nccl", "init_method": "tcp://127.0.0.1:9", "uuid": "abc"}
        sentinel = lambda **kw: types.SimpleNamespace(kind = "rccl", **kw)
        with patch(torch.version, "hip", "6.2"), \
             patch(fn, "TPBackendRCCL", sentinel), \
             patch(torch.cuda, "set_device", sd):
            ctx = fn.init_pg(0, [0, 1], 1, args)
            self.assertEqual(ctx["backend"].kind, "rccl")
            self.assertEqual(ctx["backend"].init_method, "tcp://127.0.0.1:9")
            self.assertEqual(ctx["backend"].uuid, "abc")
            self.assertEqual(ctx["backend"].output_device, 1)
        with patch(torch.version, "hip", None), \
             patch(torch.cuda, "set_device", sd):
            ctx = fn.init_pg(0, [0, 1], 1, args)
            self.assertEqual(len(calls["NCCL"]), 1)       # CUDA path: TPBackendNCCL, unchanged
            self.assertEqual(calls["NCCL"][0]["device"], 0)
        with patch(torch.cuda, "set_device", sd):
            ctx = fn.init_pg(-1, [0, 1], 1,
                             {"type": "native", "init_method": "x", "uuid": "y"})
            self.assertEqual(len(calls["Native"]), 1)     # native path untouched
            self.assertTrue(calls["Native"][0]["cpu"])

    def test_cpu_helper_through_init_pg_no_dist_no_cuda_no_native(self):
        fn, calls = self._stub_env()
        rec = []
        sd, sy = self._cuda_probe(rec, raise_on_call = True)
        with patch(torch.version, "hip", "6.2"), \
             patch(torch.cuda, "set_device", sd), \
             patch(torch.cuda, "synchronize", sy):
            ctx = fn.init_pg(-1, [0, 1], 1,
                             {"type": "nccl", "init_method": "tcp://127.0.0.1:9", "uuid": "abc"})
            be = ctx["backend"]
            self.assertIsInstance(be, self.rccl.TPBackendRCCL)
            self.assertFalse(dist.is_initialized())
            be.run_cpu_reduce_jobs()
            be.end_cpu_reduce_jobs()
            be.close()
        self.assertEqual(rec, [])

    def test_worker_quit_lifecycle_cpu_helper(self):
        fn, calls = self._stub_env()
        rec = []
        sd, sy = self._cuda_probe(rec, raise_on_call = True)

        class Conn:
            def __init__(self, msgs):
                self.msgs = deque(msgs)
            def poll(self, t):
                return bool(self.msgs)
            def recv(self):
                return self.msgs.popleft()

        with patch(torch.version, "hip", "6.2"), \
             patch(torch.cuda, "set_device", sd), \
             patch(torch.cuda, "synchronize", sy):
            fn.mp_model_worker(Conn(["quit"]), -1, [0, 1], 1,
                               {"type": "nccl", "init_method": "tcp://x:1", "uuid": "u"},
                               {"shm_name": "arena", "buffer_size": 16}, 0.0)
        self.assertEqual(rec, [])
        self.assertEqual(TestInitPgDispatch.shared_calls[-1], ("consumer", -1, False))

    def test_worker_quit_lifecycle_gpu_rank(self):
        fn, calls = self._stub_env()
        rec = []
        closed = []

        class Conn:
            def __init__(self, msgs):
                self.msgs = deque(msgs)
            def poll(self, t):
                return bool(self.msgs)
            def recv(self):
                return self.msgs.popleft()

        class FakeBe:
            def __init__(self, **kw):
                pass
            def close(self):
                closed.append(1)

        sd, sy = self._cuda_probe(rec)
        with patch(torch.version, "hip", "6.2"), \
             patch(fn, "TPBackendRCCL", FakeBe), \
             patch(torch.cuda, "set_device", sd), \
             patch(torch.cuda, "synchronize", sy):
            fn.mp_model_worker(Conn(["quit"]), 1, [0, 1], 1,
                               {"type": "nccl", "init_method": "tcp://x:1", "uuid": "u"},
                               {"shm_name": "arena", "buffer_size": 16}, 0.0)
        self.assertEqual(rec, [("set_device", 1), ("synchronize",)])
        self.assertEqual(TestInitPgDispatch.shared_calls[-1], ("consumer", 1, True))
        self.assertEqual(closed, [1])


# ---------------------------------------------------------------------------
# real torch.distributed (gloo) end to end, CPU tensors
# ---------------------------------------------------------------------------

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def gloo_child(rank, init_method):
    """Runs in a spawned subprocess (see TestRCCLGlooTwoRanks); returns 0 on pass.
    active_devices is [1, 0] with output device 0: dev1 -> rank 0, dev0 -> rank 1
    (reversed mapping, output device last as the loader arranges it)."""
    rccl = load_rccl()
    active = [1, 0]
    device = active[rank]
    be = rccl.TPBackendRCCL(
        device = device, active_devices = active, output_device = 0,
        init_method = init_method, master = (rank == 1), uuid = "gloo-test",
        dist_backend = "gloo", timeout_s = 90.0, close_barrier_timeout_s = 10.0)
    assert be.rank == rank

    be.fwd_barrier()

    # broadcast fp32 (precision probe) and int64 from device 1 (== rank 0)
    if rank == 0:
        b1 = torch.tensor([FP32_PROBE, 0.5], dtype = torch.float32)
        b2 = torch.tensor([1234567890123], dtype = torch.int64)
    else:
        b1 = torch.zeros(2, dtype = torch.float32)
        b2 = torch.zeros(1, dtype = torch.int64)
    be.broadcast(b1, src_device = 1)
    be.broadcast(b2, src_device = 1)
    assert torch.equal(b1, torch.tensor([FP32_PROBE, 0.5])), b1
    assert torch.equal(b2, torch.tensor([1234567890123])), b2

    # all_reduce: rank 0 contributes through a noncontiguous view; rank 1 is a
    # poisoned NaN buffer with contribution=False and must still receive the sum
    if rank == 0:
        base = torch.zeros((1, 1, 4), dtype = torch.float32)
        x = base[..., ::2]                    # width 2, noncontiguous
        x.fill_(FP32_PROBE)
        be.all_reduce(x)
        assert torch.equal(x, torch.full((1, 1, 2), FP32_PROBE)), x
    else:
        x = torch.full((1, 1, 2), float("nan"), dtype = torch.float32)
        be.all_reduce(x, False)
        assert torch.equal(x, torch.full((1, 1, 2), FP32_PROBE)), x
    be.fwd_barrier()

    # gather: contributor dev1 sends 2 cols, output dev0 copies its own 3 cols,
    # assembled into a NONCONTIGUOUS output view; contributor listed first
    gd = [1, 0]
    ldims = [2, 3]
    own = {0: torch.arange(2, dtype = torch.float32).view(1, 1, 2),
           1: torch.arange(3, dtype = torch.float32).view(1, 1, 3) + 10}
    if rank == 0:
        be.gather(own[0], None, gd, 0, ldims)
    else:
        out_base = torch.full((1, 1, 10), float("nan"), dtype = torch.float32)
        out = out_base[..., 0:10:2]                 # width 5, noncontiguous
        be.gather(own[1], out, gd, 0, ldims)
        assert torch.equal(out[..., 0:2], own[0]), out
        assert torch.equal(out[..., 2:5], own[1]), out
        assert torch.isnan(out_base[..., 1:11:2]).all(), "strided gaps must be untouched"

    # gather_small: argmax-style width-1 slices
    v = torch.tensor([[3.25 + rank]], dtype = torch.float16)
    out_v = torch.empty((1, 2), dtype = torch.float16) if rank == 1 else None
    be.gather_small(v, out_v, gd, 0, [1, 1])
    if rank == 1:
        assert torch.equal(out_v, torch.tensor([[3.25, 4.25]])), out_v

    # zero-width contributor: no wire traffic at all, out rank assembles locally
    if rank == 0:
        be.gather_small(torch.zeros((1, 0), dtype = torch.float32), None, gd, 0, [0, 1])
    else:
        out0 = torch.empty((1, 1), dtype = torch.float32)
        be.gather_small(torch.ones((1, 1), dtype = torch.float32), out0, gd, 0, [0, 1])
        assert torch.equal(out0, torch.ones(1, 1)), out0

    be.close()
    be.close()
    assert not dist.is_initialized()
    print("GLOO_CHILD_OK", rank, flush = True)
    return 0


class TestRCCLGlooTwoRanks(unittest.TestCase):

    def test_two_process_gloo_scenario(self):
        port = _free_port()
        init_method = f"tcp://127.0.0.1:{port}"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(TESTS_DIR), str(REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            "import test_tp_rccl_cpu as T\n"
            "sys.exit(T.gloo_child(**json.loads(sys.argv[1])))\n"
        )
        procs = []
        try:
            for rank in (0, 1):
                payload = json.dumps({"rank": rank, "init_method": init_method})
                procs.append(subprocess.Popen(
                    [sys.executable, "-c", script, payload],
                    env = env, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True))
            for p in procs:
                out, err = p.communicate(timeout = 240)
                self.assertEqual(p.returncode, 0,
                                 f"child rc={p.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
                self.assertIn("GLOO_CHILD_OK", out)
        finally:
            for p in procs:
                if p.poll() is None:
                    p.kill()


class TestRCCLGlooWorldOne(unittest.TestCase):

    def setUp(self):
        self.assertFalse(dist.is_initialized())

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_world1_lifecycle_close_twice_and_reinit(self):
        rccl = load_rccl()
        with tempfile.TemporaryDirectory() as td:
            common = dict(active_devices = [0], output_device = 0, master = True, uuid = "u",
                          dist_backend = "gloo", timeout_s = 30.0)
            be = rccl.TPBackendRCCL(device = 0,
                                    init_method = "file://" + os.path.join(td, "store1"), **common)
            self.assertTrue(dist.is_initialized())
            t = torch.full((2,), 1.25)
            be.all_reduce(t)
            self.assertTrue(torch.equal(t, torch.full((2,), 1.25)))
            z = torch.full((2,), 9.0)
            be.all_reduce(z, False)
            self.assertTrue(torch.equal(z, torch.zeros(2)))
            b = torch.tensor([FP32_PROBE])
            be.broadcast(b, src_device = 0)          # world-1 broadcast: own values
            self.assertTrue(torch.equal(b, torch.tensor([FP32_PROBE])))
            out = torch.empty(1, 1, 2)
            be.gather(torch.arange(2.0).view(1, 1, 2), out, [0], 0, [2])
            self.assertTrue(torch.equal(out, torch.arange(2.0).view(1, 1, 2)))
            be.fwd_barrier()
            be.run_cpu_reduce_jobs()
            be.end_cpu_reduce_jobs()
            be.close()
            be.close()
            self.assertFalse(dist.is_initialized())
            # teardown must not poison the process: the next split can initialize again
            be2 = rccl.TPBackendRCCL(device = 0,
                                     init_method = "file://" + os.path.join(td, "store2"), **common)
            be2.close()


if __name__ == "__main__":
    unittest.main()
