#!/usr/bin/env python3
"""CPU-only tests for the EXL3_NGRAM_MLOCK=1 opt-in RAM residency guarantee:
exllamav3/util/mlock.py (pure-ctypes mlock/munlock owner) and its hook in
exllamav3/modules/ngram_embedding.py (_load_ram_tables / unload / tp_import).

Nothing heavier than libc is real: the modules under test are loaded under a
private alias package (the real exllamav3 package, which JIT-builds the native
extension, is never imported), against recording fakes for libc, the safetensors
handles, the TP consumer and the tensors themselves. The native extension,
torch.distributed and any GPU surface are never imported. No syscall is
actually issued except in one tiny anonymous-mmap integration test (3 pages,
skipped if the host memlock allowance can't cover it).

Covered contracts:

  * mlock.py: default-off is a true no-op (no libc touched); only exactly
    "1" opts in; merged/deduped page-aligned ranges (overlap, touching,
    nested, unaligned head/tail rounding); all-or-nothing lock with
    partial-lock rollback; RLIMIT_MEMLOCK hint + errno on failure; explicit
    rejection of non-CPU / non-contiguous / address-less / interface-less
    tensors (zero syscalls); tensor references held while locked and dropped
    only after the last unlock; idempotent unlock; unlock partial failure
    keeps ownership for retry; double-lock rejected; non-Linux / libc-unloadable
    fail explicitly when opted in.
  * ngram_embedding.py: lock happens only after _load_ram_tables materialized
    the tables, once per slab (single tensor and multi-shard slab), with the
    range exactly covering the live tables[0]; default-off leaves _ram_lock
    None and makes no syscall; lock failure aborts the RAM load (no silent
    continue, no stale lock); reloading without unload unlocks the stale lock
    first; unload drains prefetch, then unlocks, then drops tables, and is
    idempotent (a second unload costs nothing); disk modes and the TP parent's
    tp_parent_defer metadata path never lock (the parent never materializes);
    the TP owner tp_import shares the hook, and a tp_import failure AFTER the
    lock unlocks before unwinding.

Run from the repo root (or anywhere):
    python3 -m pytest -q rocm_tools/rdna2/tests
"""

from __future__ import annotations

import ctypes
import errno
import importlib.util
import math
import mmap
import os
import resource
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch   # host CPU torch only: tensors for the slab paths, no CUDA calls

ALIAS = "ngram_mlock_cpu_tests"
PS = resource.getpagesize()
BASE = 0x7F0000000000        # page-aligned synthetic host address for the fakes


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


# ---------------------------------------------------------------------------
# fakes: libc syscalls, tensors, disk handles, TP consumer, host-memory guard
# ---------------------------------------------------------------------------

class FakeLibc:
    """Recording mlock/munlock stand-in. Failures are scripted by start address
    and reported through the real per-thread errno slot, exactly like a
    use_errno CDLL would leave it."""

    def __init__(self):
        self.calls = []               # (op, addr, nbytes) in call order
        self.mlock_fails = set()
        self.munlock_fails = set()
        self.fail_errno = errno.ENOMEM

    def _fire(self, op, addr, nbytes, fails):
        self.calls.append((op, addr, nbytes))
        if addr in fails:
            ctypes.set_errno(self.fail_errno)
            return -1
        return 0

    def mlock(self, addr, nbytes):
        return self._fire("mlock", addr, nbytes, self.mlock_fails)

    def munlock(self, addr, nbytes):
        return self._fire("munlock", addr, nbytes, self.munlock_fails)


class FakeTensor:
    """CPU-tensor surface ONLY as far as the locker reads it: device,
    contiguity and the storage byte span."""

    def __init__(self, ptr, nbytes, esize = 2, contiguous = True,
                 device = "cpu", dtype = torch.float16):
        self.device = torch.device(device)
        self.dtype = dtype
        self._ptr, self._nbytes, self._esize = ptr, nbytes, esize
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self._nbytes // self._esize

    def element_size(self):
        return self._esize


class FakeHandle:
    """DiskTensorHandle stand-in with the same metadata math; records reads and
    closes so the tp_import fd-hygiene contract is checkable."""

    def __init__(self, key, filename, abs_offset, shape, dtype):
        self.key = key
        self.filename = filename
        self.abs_offset = abs_offset
        self.shape = list(shape)
        self.dtype = dtype
        self.num_rows = shape[0]
        esize = torch.empty(0, dtype = dtype).element_size()
        self.row_bytes = math.prod(shape[1:]) * esize if len(shape) > 1 else esize
        self.reads = 0
        self.closes = 0

    def read_range(self, start, end):
        self.reads += 1
        return torch.empty((end - start, *self.shape[1:]), dtype = self.dtype)

    def close(self):
        self.closes += 1


class FakeConsumer:
    def __init__(self, tensors, fail = None):
        self.tensors = dict(tensors)
        self.fail = fail

    def recv(self, ref, cuda = False):
        if self.fail is not None:
            raise self.fail
        return self.tensors[ref]


class FakeModuleBase:
    """exllamav3.modules.Module stand-in: the attribute surface NGramEmbedding
    touches without a device, cache or graph stack."""

    def __init__(self, config, key, qmap):
        self.config = config
        self.key = key
        self.qmap = qmap
        self.device = None
        self.caps = {}


CHECK_MEMORY_CALLS = []


def _stub(name, attrs = None, path = None):
    m = types.ModuleType(name)
    if path is not None:
        m.__path__ = [str(path)]
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def install_alias_stubs():
    """Private alias package rooted at exllamav3/ so the REAL mlock.py,
    ngram_codec.py and ngram_embedding.py load without ever importing the real
    exllamav3 (its __init__ JIT-builds the native extension)."""
    if ALIAS in sys.modules:
        return sys.modules[ALIAS + ".modules.ngram_embedding"]
    _stub(ALIAS, path = REPO / "exllamav3")
    _stub(ALIAS + ".model", path = REPO / "exllamav3" / "model")
    _stub(ALIAS + ".loader", path = REPO / "exllamav3" / "loader")
    _stub(ALIAS + ".util", path = REPO / "exllamav3" / "util")
    _stub(ALIAS + ".modules", {"Module": FakeModuleBase}, path = REPO / "exllamav3" / "modules")
    _stub(ALIAS + ".ext", {"exllamav3_ext": types.SimpleNamespace()})
    _stub(ALIAS + ".model.config", {"Config": object})
    _stub(ALIAS + ".loader.safetensors", {"DiskTensorHandle": FakeHandle})
    _stub(ALIAS + ".util.memory", {
        "check_host_memory": lambda nbytes, what: CHECK_MEMORY_CALLS.append((nbytes, what))})
    _stub(ALIAS + ".modules.quant", path = REPO / "exllamav3" / "modules" / "quant")
    _stub(ALIAS + ".modules.quant.exl3_lib",
          path = REPO / "exllamav3" / "modules" / "quant" / "exl3_lib")
    _load_file(ALIAS + ".util.mlock", REPO / "exllamav3" / "util" / "mlock.py")
    _load_file(ALIAS + ".modules.quant.exl3_lib.ngram_codec",
               REPO / "exllamav3" / "modules" / "quant" / "exl3_lib" / "ngram_codec.py")
    mod = _load_file(ALIAS + ".modules.ngram_embedding",
                     REPO / "exllamav3" / "modules" / "ngram_embedding.py")
    assert "exllamav3" not in getattr(mod, "__package__", ""), "must not leak into real package"
    return mod


NE = install_alias_stubs()
ML = sys.modules[ALIAS + ".util.mlock"]
NGRAM_KEY = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"


def span(addr, nbytes, page = PS):
    start = addr & -page
    end = (addr + nbytes + page - 1) & -page
    return (start, end - start)


# ---------------------------------------------------------------------------
# exllamav3/util/mlock.py against a fake libc (plus one real-mmap test)
# ---------------------------------------------------------------------------

class MlockUtilTests(unittest.TestCase):

    def setUp(self):
        self.libc = FakeLibc()
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(ML.ENV_VAR, None)

    def lock(self):
        return ML.MlockedRanges(libc = self.libc, page_size = PS)

    # ---- opt-in switch ----

    def test_only_exactly_1_opts_in(self):
        for v, want in (("1", True), ("0", False), ("", False), ("2", False), ("true", False)):
            with self.subTest(v = v):
                os.environ[ML.ENV_VAR] = v
                self.assertEqual(ML.mlock_enabled(), want)
        self.assertFalse(ML.mlock_enabled())          # unset

    def test_default_off_is_a_true_noop(self):
        with mock.patch.object(ML, "_load_libc",
                               side_effect = AssertionError("libc loaded with the switch off")):
            self.assertIsNone(ML.lock_tables_if_requested([FakeTensor(BASE, 8192)], "t"))

    def test_opt_in_routes_through_lock_tables_if_requested(self):
        os.environ[ML.ENV_VAR] = "1"
        with mock.patch.object(ML, "_load_libc", return_value = self.libc):
            lock = ML.lock_tables_if_requested([FakeTensor(BASE, 8192)], "t")
        self.assertIsNotNone(lock)
        self.assertEqual(self.libc.calls, [("mlock", BASE, 8192)])
        lock.unlock()

    # ---- range merging ----

    def test_merged_page_ranges(self):
        m = ML.merged_page_ranges
        self.assertEqual(m([(BASE, 4096), (BASE + 4096, 4096)], PS), [(BASE, 8192)])
        self.assertEqual(m([(BASE + 4096, 4096), (BASE, 4096)], PS), [(BASE, 8192)])  # unordered in
        self.assertEqual(m([(BASE, 4096), (BASE, 4096)], PS), [(BASE, 4096)])         # duplicate
        self.assertEqual(m([(BASE, 8192), (BASE + 2048, 1024)], PS), [(BASE, 8192)])  # nested
        self.assertEqual(m([(BASE + 2000, 8 * 4096 - 3000)], PS), [(BASE, 8 * 4096)])  # rounding
        self.assertEqual(m([(BASE, 8192), (BASE + 10 * 4096, 4096)], PS),
                         [(BASE, 8192), (BASE + 10 * 4096, 4096)])                     # gap kept
        self.assertEqual(m([(BASE, 0)], PS), [])                                        # empty skipped
        with self.assertRaises(ValueError):
            m([(0, 4096)], PS)
        with self.assertRaises(ValueError):
            m([(BASE, 4096)], PS - 1)

    def test_lock_uses_one_call_per_merged_run(self):
        ts = [FakeTensor(BASE + 2048, 2048), FakeTensor(BASE + 4096, 4096),
              FakeTensor(BASE + 65536, 4096)]
        lock = self.lock().lock_tensors(ts, what = "table")
        self.assertEqual(self.libc.calls, [("mlock", BASE, 8192), ("mlock", BASE + 65536, 4096)])
        self.assertEqual(lock.locked_ranges, [(BASE, 8192), (BASE + 65536, 4096)])
        self.assertEqual(lock.locked_bytes, 12288)
        self.assertTrue(lock.is_locked)

    # ---- references / lifecycle ----

    def test_references_held_while_locked_then_dropped(self):
        t = FakeTensor(BASE, 4096)
        ref = weakref.ref(t)
        lock = self.lock().lock_tensors([t], what = "table")
        del t
        self.assertIsNotNone(ref())                    # locked -> storage kept alive
        lock.unlock()
        self.assertIsNone(ref())                        # unlocked -> references released
        lock.unlock()                                   # idempotent: no extra syscalls
        self.assertEqual(self.libc.calls, [("mlock", BASE, 4096), ("munlock", BASE, 4096)])

    def test_double_lock_rejected(self):
        lock = self.lock().lock_tensors([FakeTensor(BASE, 4096)], what = "table")
        with self.assertRaisesRegex(ML.MlockError, "already holds"):
            lock.lock_tensors([FakeTensor(BASE + 4096, 4096)], what = "table")
        with self.assertRaisesRegex(ML.MlockError, "no table tensors"):
            self.lock().lock_tensors([], what = "table")
        lock.unlock()

    # ---- explicit rejections: zero syscalls, nothing locked ----

    def test_unsupported_tensors_rejected(self):
        cases = {
            "cuda-resident": FakeTensor(BASE, 4096, device = "cuda:0"),
            "non-contiguous": FakeTensor(BASE, 4096, contiguous = False),
            "no host address": FakeTensor(0, 4096),
            "not a tensor": FakeHandle("k", "f", 0, [8, 160], torch.float16),
        }
        for label, t in cases.items():
            with self.subTest(case = label):
                lock = self.lock()
                with self.assertRaises(ML.MlockError):
                    lock.lock_tensors([t], what = "table")
                self.assertFalse(lock.is_locked)
                with self.assertRaisesRegex(ML.MlockError, "empty"):
                    lock.lock_tensors([FakeTensor(BASE, 0)], what = "table")
        self.assertEqual(self.libc.calls, [])

    # ---- failure handling ----

    def test_first_mlock_failure_raises_with_rlimit_hint(self):
        self.libc.mlock_fails = {BASE}
        with self.assertRaises(ML.MlockError) as cm:
            self.lock().lock_tensors([FakeTensor(BASE, 4096)], what = "my table")
        msg = str(cm.exception)
        self.assertIn("mlock(0x", msg)
        self.assertIn(os.strerror(errno.ENOMEM), msg)
        self.assertIn("my table", msg)
        self.assertIn("RLIMIT_MEMLOCK", msg)             # hint; the limit is never changed
        self.assertEqual(self.libc.calls, [("mlock", BASE, 4096)])

    def test_partial_lock_rolls_back_locked_ranges(self):
        t1, t2, t3 = (FakeTensor(BASE, 4096), FakeTensor(BASE + 4096, 4096),
                      FakeTensor(BASE + 65536, 4096))       # t1+t2 merge to one run; t3 fails
        self.libc.mlock_fails = {BASE + 65536}
        lock = self.lock()
        with self.assertRaises(ML.MlockError):
            lock.lock_tensors([t1, t2, t3], what = "table")
        self.assertEqual(self.libc.calls, [("mlock", BASE, 8192), ("mlock", BASE + 65536, 4096),
                                           ("munlock", BASE, 8192)])
        self.assertFalse(lock.is_locked)                 # nothing half-locked left behind
        self.assertEqual(lock._tensors, ())              # rollback released the held references

    def test_rollback_failure_keeps_ownership_for_retry(self):
        t1, t2 = FakeTensor(BASE, 4096), FakeTensor(BASE + 65536, 4096)
        keep = weakref.ref(t1)
        self.libc.mlock_fails = {BASE + 65536}
        self.libc.munlock_fails = {BASE}
        lock = self.lock()
        with self.assertRaisesRegex(ML.MlockError, "could not be rolled back"):
            lock.lock_tensors([t1, t2], what = "table")
        del t1, t2
        self.assertTrue(lock.is_locked)                  # the stranded range is still OWNED...
        self.assertIsNotNone(keep())                     # ...with its tensor reference held
        self.libc.munlock_fails = set()
        lock.unlock()                                    # ...so a later unlock completes
        self.assertFalse(lock.is_locked)
        self.assertIsNone(keep())

    def test_unlock_partial_failure_retries_only_the_failed_range(self):
        lock = self.lock().lock_tensors(
            [FakeTensor(BASE, 4096), FakeTensor(BASE + 65536, 4096)], what = "table")
        self.libc.munlock_fails = {BASE + 65536}
        with self.assertRaisesRegex(ML.MlockError, "remain owned"):
            lock.unlock()
        self.assertEqual(lock.locked_ranges, [(BASE + 65536, 4096)])
        self.assertEqual(self.libc.calls, [("mlock", BASE, 4096), ("mlock", BASE + 65536, 4096),
                                           ("munlock", BASE + 65536, 4096), ("munlock", BASE, 4096)])
        self.libc.munlock_fails = set()
        lock.unlock()
        self.assertEqual(self.libc.calls[-1], ("munlock", BASE + 65536, 4096))
        self.assertFalse(lock.is_locked)

    # ---- platform guards (opted-in failures must be explicit) ----

    def test_non_linux_and_missing_libc_fail_explicitly(self):
        for label, patcher in (
            ("platform", mock.patch.object(sys, "platform", "win32")),
            ("cdll", mock.patch.object(ctypes, "CDLL", side_effect = OSError("no libc.so.6"))),
        ):
            with self.subTest(case = label), patcher, \
                 mock.patch.object(ML, "_libc", None):
                with self.assertRaises(ML.MlockError):
                    ML._load_libc()
                self.assertIsNone(ML._libc)              # failure not cached as a good handle

    # ---- tiny real-kernel integration: anonymous mmap, 3 pages ----

    def test_real_mmap_mlock_mincore_munlock(self):
        if sys.platform != "linux":
            self.skipTest("mlock residency is a Linux guarantee")
        soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
        if 0 <= soft < 3 * PS:
            self.skipTest(f"host RLIMIT_MEMLOCK soft={soft} cannot cover 3 pages")

        class MmapTensor:
            device = "cpu"
            def __init__(self, addr, nbytes):
                self.p, self.n = addr, nbytes
            def is_contiguous(self): return True
            def data_ptr(self): return self.p
            def numel(self): return self.n
            def element_size(self): return 1

        npages = 3
        mm = mmap.mmap(-1, npages * PS)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        lock = None
        try:
            try:
                lock = ML.MlockedRanges().lock_tensors([MmapTensor(addr, npages * PS)],
                                                       what = "test mapping")
            except ML.MlockError as e:                   # host allowance is a launch concern
                self.skipTest(f"mlock not permitted here: {e}")
            self.assertEqual(lock.locked_ranges, [span(addr, npages * PS)])
            lib = ML._load_libc()
            lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
            lib.mincore.restype = ctypes.c_int
            vec = ctypes.create_string_buffer(npages)
            self.assertEqual(lib.mincore(addr & -PS, npages * PS, vec), 0)
            self.assertTrue(all(b & 1 for b in vec.raw), "mlocked pages must be RAM-resident")
            lock.unlock()
            self.assertFalse(lock.is_locked)
            lock = None
        finally:
            if lock is not None:
                lock.unlock()
            mm.close()


# ---------------------------------------------------------------------------
# NGramEmbedding integration: shared lock hook, unload ordering, TP import
# ---------------------------------------------------------------------------

class NgramMlockIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.libc = FakeLibc()
        CHECK_MEMORY_CALLS.clear()
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(ML.ENV_VAR, None)
        self.lib = mock.patch.object(ML, "_load_libc", return_value = self.libc)
        self.lib.start()
        self.addCleanup(self.lib.stop)

    def opt_in(self, value = "1"):
        if value is None:
            os.environ.pop(ML.ENV_VAR, None)
        else:
            os.environ[ML.ENV_VAR] = value

    def bare(self, mode = "fp16_ram", keys = ("t.weight",), rows = 8):
        mod = NE.NGramEmbedding(config = None, key = NGRAM_KEY, ngram_size = 3,
                                heads_per_ngram = 8, ple_embed_dim = 2560,
                                eos_token_id = 42, stream_from_disk = mode.endswith("_disk"))
        mod.device = torch.device("cpu")
        mod.mode = mode
        mod.num_rows = rows
        mod.rows_per_shard = rows // len(keys)
        mod._table_keys = list(keys)
        return mod

    def exported(self, mode):
        return {
            "cls": NE.NGramEmbedding,
            "kwargs": {"key": NGRAM_KEY, "ngram_size": 3, "heads_per_ngram": 8,
                       "ple_embed_dim": 2560, "eos_token_id": 42, "out_dtype": torch.float16},
            "mode": mode, "K": None, "num_rows": 8, "rows_per_shard": 8,
            "handles": [(NGRAM_KEY + ".weight", "/models/t-00001.safetensors", 4096,
                         [8, 160], "torch.float16")],
            "row_dtype": "torch.float16",
            "head_offsets": "off", "head_vocab_sizes": "sz", "layer_multipliers": "mul",
            "head_bias": None, "device": 0,
        }

    def consumer(self):
        return FakeConsumer({"off": torch.zeros(16, dtype = torch.long),
                             "sz": torch.ones(16, dtype = torch.long),
                             "mul": torch.ones(16, dtype = torch.long)})

    # ---- regular RAM loader path ----

    def test_default_off_leaves_module_unlocked(self):
        self.opt_in("0")
        mod = self.bare()
        mod._load_ram_tables(lambda i, k: FakeTensor(BASE + i * 8192, 8192))
        self.assertIsNotNone(mod.tables)
        self.assertIsNone(mod._ram_lock)
        self.assertEqual(self.libc.calls, [])

    def test_opt_in_locks_the_slab_after_tables_materialize(self):
        self.opt_in()
        mod = self.bare()
        t = FakeTensor(BASE + 512, 8192)                # unaligned pointer -> rounded span
        mod._load_ram_tables(lambda i, k: t)
        self.assertEqual(self.libc.calls, [("mlock", *span(t.data_ptr(), 8192))])
        self.assertIsInstance(mod._ram_lock, ML.MlockedRanges)
        self.assertEqual(mod._ram_lock.locked_ranges, [span(t.data_ptr(), 8192)])
        mod.unload()
        self.assertEqual(self.libc.calls[-1], ("munlock", *span(t.data_ptr(), 8192)))

    def test_sharded_table_locks_one_slab_range(self):
        self.opt_in()
        shards = [torch.zeros((4, 160), dtype = torch.float16) for _ in range(2)]
        mod = self.bare(keys = ("t.shard_0.weight", "t.shard_1.weight"), rows = 8)
        mod._load_ram_tables(lambda i, k: shards[i])
        self.assertEqual(CHECK_MEMORY_CALLS, [(8 * 160 * 2, mock.ANY)])   # guarded, then merged
        self.assertEqual(len(mod.tables), 1)                              # ONE contiguous slab
        self.assertEqual(self.libc.calls, [("mlock", *span(mod.tables[0].data_ptr(), 2560))])
        mod.unload()

    def test_unload_drains_unlocks_then_drops_and_is_idempotent(self):
        self.opt_in()
        mod = self.bare()
        mod._load_ram_tables(lambda i, k: FakeTensor(BASE, 8192))
        events = []
        orig_drain = mod._drain_prefetch
        orig_unlock = mod._ram_lock.unlock
        mod._drain_prefetch = lambda: (events.append("drain"), orig_drain())[1]
        mod._ram_lock.unlock = lambda: (events.append("unlock"), orig_unlock())[1]
        mod.unload()
        self.assertEqual(events, ["drain", "unlock"])     # prefetch drains BEFORE munlock
        self.assertIsNone(mod.tables)                     # tables dropped only after unlock
        self.assertIsNone(mod._ram_lock)
        before = list(self.libc.calls)
        self.assertEqual([c[0] for c in before], ["mlock", "munlock"])
        mod.unload()                                      # repeated unload: no-op, no raise
        self.assertEqual(self.libc.calls, before)

    def test_unlock_never_loaded_module_is_clean(self):
        self.opt_in()
        self.bare().unload()                              # fresh module: no state, no syscalls
        self.assertEqual(self.libc.calls, [])

    def test_lock_failure_aborts_the_ram_load_loudly(self):
        self.opt_in()
        self.libc.mlock_fails = {BASE}
        mod = self.bare()
        with self.assertRaises(ML.MlockError) as cm:      # no silent continue
            mod._load_ram_tables(lambda i, k: FakeTensor(BASE, 8192))
        self.assertIn("RLIMIT_MEMLOCK", str(cm.exception))
        self.assertIsNone(mod._ram_lock)                  # and nothing half-locked is owned
        mod.unload()                                      # unload still works on the failed load
        self.assertEqual(self.libc.calls, [("mlock", BASE, 8192)])

    def test_reload_without_unload_releases_the_stale_lock(self):
        self.opt_in()
        mod = self.bare()
        mod._load_ram_tables(lambda i, k: FakeTensor(BASE, 8192))
        mod._load_ram_tables(lambda i, k: FakeTensor(BASE + 65536, 8192))
        self.assertEqual(self.libc.calls,
                         [("mlock", BASE, 8192), ("munlock", BASE, 8192),
                          ("mlock", BASE + 65536, 8192)])
        mod.unload()
        self.assertEqual(self.libc.calls[-1], ("munlock", BASE + 65536, 8192))

    # ---- TP: owner import shares the hook; parent / disk paths stay out of it ----

    def test_tp_import_owner_ram_locks_the_imported_table(self):
        self.opt_in()
        module = NE.NGramEmbedding.tp_import(
            {"consumer": self.consumer(), "device": torch.device("cpu")},
            self.exported("fp16_ram"), {})
        self.assertEqual(CHECK_MEMORY_CALLS, [(8 * 160 * 2, mock.ANY)])   # guarded once
        addr, nbytes = int(module.tables[0].data_ptr()), 8 * 160 * 2
        self.assertEqual(self.libc.calls, [("mlock", *span(addr, nbytes))])
        module.unload()                                                   # owner retires cleanly
        self.assertEqual(self.libc.calls[-1], ("munlock", *span(addr, nbytes)))

    def test_tp_import_failure_after_lock_unlocks_before_unwinding(self):
        self.opt_in()
        boom = RuntimeError("arena dead")
        ctx = {"consumer": FakeConsumer({}, fail = boom), "device": torch.device("cpu")}
        with self.assertRaises(RuntimeError):
            NE.NGramEmbedding.tp_import(ctx, self.exported("fp16_ram"), {})
        ops = [c[0] for c in self.libc.calls]
        self.assertEqual(ops, ["mlock", "munlock"])       # no stale lock survives the raise

    def test_tp_import_disk_mode_never_locks(self):
        self.opt_in()
        module = NE.NGramEmbedding.tp_import(
            {"consumer": self.consumer(), "device": torch.device("cpu")},
            self.exported("fp16_disk"), {})
        self.assertIsNone(module.tables)
        self.assertIsNone(module._ram_lock)
        self.assertEqual(self.libc.calls, [])
        handle = module.handles[0]
        module.unload()
        self.assertEqual(self.libc.calls, [])
        self.assertEqual(handle.closes, 1)                # disk-mode handles still closed

    def test_tp_parent_defer_placeholder_never_materializes_or_locks(self):
        self.opt_in()
        grabbed = []

        class Stc:
            def has_tensor(self, name):
                return name == NGRAM_KEY + ".weight"
            def get_tensor_meta(self, name):
                return {name: {"shape": [8, 160]}}
            def get_tensor(self, name, device, **kw):
                grabbed.append(name)
                if name == NGRAM_KEY + ".weight":
                    return torch.zeros((8, 160), dtype = torch.float16)
                return torch.ones(16, dtype = torch.int64)      # the small hash params
            def get_tensor_handle(self, name):
                return FakeHandle(name, "/models/t.safetensors", 4096, [8, 160], torch.float16)
            def release_file(self, fn):
                pass
            tensor_file_map = {}

        cfg = types.SimpleNamespace(stc = Stc(), infer_params = None)
        mod = NE.NGramEmbedding(config = cfg, key = NGRAM_KEY, ngram_size = 3,
                                heads_per_ngram = 8, ple_embed_dim = 2560,
                                eos_token_id = 42, stream_from_disk = False)
        mod.load(torch.device("cpu"), tp_parent_defer = True)
        self.assertTrue(mod._tp_deferred)
        self.assertIsNone(mod.tables)                     # metadata-only: nothing materialized
        self.assertNotIn(NGRAM_KEY + ".weight", grabbed)  # the table tensor was never read
        self.assertIsNone(mod._ram_lock)
        self.assertEqual(self.libc.calls, [])
        mod.unload()


if __name__ == "__main__":
    unittest.main()
