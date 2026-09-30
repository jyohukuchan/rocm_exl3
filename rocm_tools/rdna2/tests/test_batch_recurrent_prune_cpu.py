#!/usr/bin/env python3
"""CPU-only tests for the opt-in batched prune_stranded() in
exllamav3/cache/recurrent.py (EXL3_BATCH_RECURRENT_PRUNE=1, default off).

No GPU, no torch import, no exllamav3 package execution: the REAL recurrent.py is
loaded under a private alias package (same stub pattern as test_tp_lifecycle_cpu.py;
the real util/memory.py imports torch at module level, so that one leaf is stubbed,
and _stashed_bytes' deferred `import torch` resolves against a fake torch module
whose Tensor is a plain weak-referenceable byte-witness class). Behavior fakes:

  * FakeModel mirrors the tp_dispatch_all contract -- records every parent call and
    runs the dispatched worker function inline against per-rank local_context
    registries, exactly like the in-process pseudo workers elsewhere in this suite;
  * FakePagetable answers is_resumable() from a key set;
  * note_freed / malloc_trim are observed through the loaded module's globals, so
    byte accounting, aggregation and the release-before-trim ORDER of the real
    code are under test.

Covered properties: zero/one/multiple stranded, live entries kept, exactly one TP
dispatch for many handles, worker refs gone before note_freed (weakref witness,
with the legacy per-entry path shown keeping them alive), handle aliases kept live
then deleted once, no duplicate parent byte accounting, LS path aggregates and
trims only after the dropped refs are released, bulk-delete precondition asserts
before any mutation, and the legacy default behavior unchanged.

Run from the repo root (or anywhere):
    python3 -m unittest rocm_tools.rdna2.tests.test_batch_recurrent_prune_cpu -v
or as part of the suite:
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
import unittest
import weakref
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

BATCH_ENV = "EXL3_BATCH_RECURRENT_PRUNE"

# recurrent.py is loaded under an alias package rooted at exllamav3/ so its relative
# imports resolve without executing exllamav3/__init__ (which imports torch). Nothing
# is ever installed under "exllamav3".
ALIAS = "batch_recurrent_prune_cpu_tests"


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


def install_alias_and_load():
    if ALIAS in sys.modules:
        return sys.modules[ALIAS + ".cache.recurrent"]
    root = types.ModuleType(ALIAS)
    root.__path__ = [str(REPO / "exllamav3")]
    sys.modules[ALIAS] = root
    cache_pkg = types.ModuleType(ALIAS + ".cache")
    cache_pkg.__path__ = [str(REPO / "exllamav3" / "cache")]
    sys.modules[ALIAS + ".cache"] = cache_pkg
    util_pkg = types.ModuleType(ALIAS + ".util")
    util_pkg.__path__ = [str(REPO / "exllamav3" / "util")]
    sys.modules[ALIAS + ".util"] = util_pkg

    # constants.py is stdlib-only: run the real file so PAGE_SIZE is not a duplicated constant
    _load_file(ALIAS + ".constants", REPO / "exllamav3" / "constants.py")

    memory = types.ModuleType(ALIAS + ".util.memory")
    memory.malloc_trim = lambda: None          # replaced per test with a recording probe
    sys.modules[ALIAS + ".util.memory"] = memory

    mod = _load_file(ALIAS + ".cache.recurrent", REPO / "exllamav3" / "cache" / "recurrent.py")
    assert mod.__name__.startswith(ALIAS)
    assert not getattr(mod, "__package__", "").startswith("exllamav3"), "must not leak into real package"
    return mod


rc = install_alias_and_load()


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeTensor:
    """Host-memory witness: _stashed_bytes only asks isinstance(torch.Tensor) and
    numel() * element_size(); weakref-able, and freed by refcount like a real tensor
    when the last reference drops."""
    def __init__(self, nbytes):
        self._nbytes = nbytes

    def numel(self):
        return self._nbytes

    def element_size(self):
        return 1


@contextlib.contextmanager
def fake_torch():
    """Make the deferred `import torch` inside _stashed_bytes resolve to a fake
    whose Tensor is FakeTensor. Restores any pre-existing real torch entry after."""
    t = types.ModuleType("torch")
    t.Tensor = FakeTensor
    with mock.patch.dict(sys.modules, {"torch": t}):
        yield


@contextlib.contextmanager
def env_batch(value):
    """EXL3_BATCH_RECURRENT_PRUNE=value for the duration; value None = unset key."""
    new = dict(os.environ)
    new.pop(BATCH_ENV, None)
    if value is not None:
        new[BATCH_ENV] = value
    with mock.patch.dict(os.environ, new, clear=True):
        yield


class FakePagetable:
    def __init__(self, resumable=()):
        self.resumable = set(resumable)
        self.referenced_pages = {}
        self.unreferenced_pages = {}

    def is_resumable(self, key):
        return key in self.resumable


class FakeModel:
    """tp_dispatch_all contract: records each parent call, then runs the dispatched
    function inline on every rank's local_context like the real fan-out (one parent
    call = one worker call per rank). in_dispatch lets accounting tests tell parent
    calls apart from worker calls."""
    def __init__(self, loaded_tp, n_ranks=2):
        self.loaded_tp = loaded_tp
        self.ranks = [{"recurrent_cache": {}} for _ in range(n_ranks)]
        self.dispatches = []
        self.in_dispatch = False

    def tp_dispatch_all(self, func, args):
        self.dispatches.append((func, tuple(args)))
        self.in_dispatch = True
        try:
            for ctx in self.ranks:
                func(ctx, *args)
        finally:
            self.in_dispatch = False


class FakeState:
    """Stands in for a recurrent state object passed to put()."""
    def __init__(self, stashed):
        self._stashed = stashed

    def stash(self):
        return self._stashed


def seed_tp(cache, model, key, handle, nbytes, tensors=None):
    """Metadata-only parent entry + real (fake) host tensors in every rank, as stash()
    does under loaded_tp."""
    cache[key] = {"position": 0, "checkpoint_size": nbytes, "tp_handle": handle}
    for ctx in model.ranks:
        ctx["recurrent_cache"][handle] = tensors if tensors is not None else [FakeTensor(nbytes)]


def seed_ls(cache, key, nbytes):
    """Local-path entry: the dict itself holds the state (what stash() returns when
    the model is not loaded_tp)."""
    st = {"position": 0, "checkpoint_size": nbytes, "layer0": [FakeTensor(nbytes)]}
    cache[key] = st
    return st


# ---------------------------------------------------------------------------
# note_freed observation
# ---------------------------------------------------------------------------

class NoteFreedRecorder:
    """Replace the loaded module's note_freed and tag calls by phase, so tests can
    assert the parent does NOT account TP bytes and workers account exactly once.
    With in_dispatch False the recorder behaves like the direct-call case."""
    def __init__(self, model):
        self.model = model
        self.calls = []          # (nbytes, in_dispatch)

    def __enter__(self):
        self._patch = mock.patch.object(rc, "note_freed", self._rec)
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)

    def _rec(self, nbytes):
        self.calls.append((nbytes, self.model.in_dispatch))

    def parent(self):
        return [c for c in self.calls if not c[1]]

    def worker(self):
        return [c for c in self.calls if c[1]]


class Base(unittest.TestCase):
    def setUp(self):
        self.orig_freed = rc._freed_bytes
        rc._freed_bytes = 0
        self.addCleanup(setattr, rc, "_freed_bytes", self.orig_freed)

    def make_cache(self, loaded_tp, pt, max_size=1 << 40, n_ranks=2):
        model = FakeModel(loaded_tp, n_ranks=n_ranks)
        cache = rc.RecurrentCache(model, max_size=max_size)
        cache.pagetable = pt
        return cache, model


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class LegacyDefaultTests(Base):

    def test_default_env_off_keeps_per_entry_tp_path(self):
        cache, model = self.make_cache(True, FakePagetable(["live"]))
        for i, key in enumerate(["s1", "s2", "s3"]):
            seed_tp(cache, model, key, handle=10 + i, nbytes=100 + i)
        seed_tp(cache, model, "live", handle=99, nbytes=7)
        with env_batch(None), fake_torch(), NoteFreedRecorder(model) as freed:
            self.assertEqual(cache.prune_stranded(), 3)
        self.assertEqual([f for f, _ in model.dispatches], [rc.mp_cache_recurrent_del] * 3)
        self.assertEqual([a for _, a in model.dispatches],
                         [(id(cache), h) for h in (10, 11, 12)])
        self.assertEqual(cache.metrics["stash_pruned"], 3)
        self.assertEqual(list(cache.keys()), ["live"])
        self.assertEqual(cache.current_size, 7)
        for ctx in model.ranks:
            self.assertEqual(set(ctx["recurrent_cache"]), {99})
        # legacy accounting: one parent note_freed per popped entry (checkpoint_size),
        # one worker note_freed per rank per entry
        self.assertEqual(freed.parent(), [(100, False), (101, False), (102, False)])
        self.assertEqual(len(freed.worker()), 6)

    def test_env_values_other_than_one_stay_legacy(self):
        for value in (None, "", "0", "2", "yes"):
            with self.subTest(value=value):
                cache, model = self.make_cache(True, FakePagetable(["live"]))
                seed_tp(cache, model, "s1", 1, 10)
                seed_tp(cache, model, "live", 2, 5)
                with env_batch(value), fake_torch():
                    self.assertEqual(cache.prune_stranded(), 1)
                self.assertEqual([f for f, _ in model.dispatches], [rc.mp_cache_recurrent_del])

    def test_put_eviction_path_unchanged_under_batch_env(self):
        # opt-in batch applies to prune_stranded only: LRU eviction keeps the old
        # per-entry mp_cache_recurrent_del dispatch
        cache, model = self.make_cache(True, None, max_size=100)
        seed_tp(cache, model, "old", handle=1, nbytes=60)
        cache.update_total_size()
        with env_batch("1"), fake_torch():
            cache.put("new", FakeState({"position": 0, "checkpoint_size": 80, "tp_handle": 2}))
        self.assertEqual([f for f, _ in model.dispatches], [rc.mp_cache_recurrent_del])
        self.assertEqual(model.dispatches[0][1], (id(cache), 1))
        self.assertEqual(cache.metrics["stash_evictions"], 1)
        self.assertEqual(cache.metrics["stash_pruned"], 0)
        self.assertEqual(list(cache.keys()), ["new"])
        self.assertEqual(cache.current_size, 80)


class BatchTpTests(Base):

    def test_zero_stranded_is_a_no_op(self):
        cache, model = self.make_cache(True, FakePagetable(["a", "b"]))
        seed_tp(cache, model, "a", 1, 10)
        seed_tp(cache, model, "b", 2, 20)
        with env_batch("1"), NoteFreedRecorder(model) as freed:
            self.assertEqual(cache.prune_stranded(), 0)
        self.assertEqual(model.dispatches, [])
        self.assertEqual(freed.calls, [])
        self.assertEqual(cache.metrics["stash_pruned"], 0)
        self.assertEqual(set(cache.keys()), {"a", "b"})
        no_pt = rc.RecurrentCache(FakeModel(True))
        self.assertEqual(no_pt.prune_stranded(), 0)

    def test_single_stranded_one_bulk_dispatch(self):
        cache, model = self.make_cache(True, FakePagetable(["live"]))
        seed_tp(cache, model, "dead", 1, 10)
        seed_tp(cache, model, "live", 2, 20)
        with env_batch("1"), fake_torch(), NoteFreedRecorder(model) as freed:
            self.assertEqual(cache.prune_stranded(), 1)
        self.assertEqual(len(model.dispatches), 1)
        func, args = model.dispatches[0]
        self.assertIs(func, rc.mp_cache_recurrent_del_bulk)
        self.assertEqual(args[0], id(cache))
        self.assertEqual(args[1], [1])
        self.assertIsInstance(args[1], list)          # picklable payload
        for ctx in model.ranks:
            self.assertEqual(set(ctx["recurrent_cache"]), {2})
        self.assertEqual(freed.parent(), [])                    # parent accounts nothing under TP
        self.assertEqual(freed.worker(), [(10, True), (10, True)])  # once per rank, actual bytes
        self.assertEqual(cache.metrics["stash_pruned"], 1)
        self.assertEqual(cache.current_size, 20)

    def test_many_stranded_exactly_one_dispatch(self):
        cache, model = self.make_cache(True, FakePagetable(["live1", "live2"]))
        for i in range(8):
            seed_tp(cache, model, f"d{i}", handle=i, nbytes=1000 + i)
        seed_tp(cache, model, "live1", 100, 7)
        seed_tp(cache, model, "live2", 101, 11)
        with env_batch("1"), fake_torch(), NoteFreedRecorder(model) as freed:
            self.assertEqual(cache.prune_stranded(), 8)
        self.assertEqual(len(model.dispatches), 1)
        func, (cache_id, handles) = model.dispatches[0]
        self.assertIs(func, rc.mp_cache_recurrent_del_bulk)
        self.assertEqual(cache_id, id(cache))
        self.assertEqual(handles, list(range(8)))     # dedup'd, pop-order preserved
        self.assertEqual(list(cache.keys()), ["live1", "live2"])
        self.assertEqual(cache.metrics["stash_pruned"], 8)
        self.assertEqual(cache.current_size, 18)
        for ctx in model.ranks:
            self.assertEqual(set(ctx["recurrent_cache"]), {100, 101})
        self.assertEqual(freed.parent(), [])
        expected = sum(1000 + i for i in range(8))
        self.assertEqual(freed.worker(), [(expected, True)] * len(model.ranks))

    def test_release_before_note_freed_weakref_witness(self):
        # the measured defect: legacy keeps each popped stash bound while note_freed
        # runs, so a threshold-triggered malloc_trim sees the pages still pinned.
        # The bulk worker must free everything first, then account once.
        def run(batched):
            cache, model = self.make_cache(True, FakePagetable(["live"]), n_ranks=1)
            seed_tp(cache, model, "live", 50, 3)
            for i in range(3):
                seed_tp(cache, model, f"d{i}", handle=i, nbytes=10)
            refs = [weakref.ref(model.ranks[0]["recurrent_cache"][i][0]) for i in range(3)]
            liveness = []
            orig_note_freed = rc.note_freed
            def spy(nbytes):
                liveness.append([r() is not None for r in refs])
                orig_note_freed(nbytes)
            with (env_batch("1" if batched else None), fake_torch(),
                  mock.patch.object(rc, "note_freed", spy)):
                self.assertEqual(cache.prune_stranded(), 3)
            return liveness
        legacy = run(batched=False)
        batch = run(batched=True)
        self.assertTrue(any(alive for call in legacy for alive in call),
                        "legacy documents refs still alive during note_freed")
        self.assertEqual(len(batch), 1)                       # a single aggregated call
        self.assertEqual(batch[0], [False, False, False])     # every tensor freed before it

    def test_alias_survivor_keeps_handle_then_deleted_once(self):
        cache, model = self.make_cache(True, FakePagetable(["alias_live"]))
        shared = {"position": 0, "checkpoint_size": 10, "tp_handle": 5}
        cache["stranded_alias"] = shared
        cache["alias_live"] = shared
        for ctx in model.ranks:
            ctx["recurrent_cache"][5] = [FakeTensor(10)]
        pt = cache.pagetable
        with env_batch("1"), fake_torch():
            self.assertEqual(cache.prune_stranded(), 1)
            self.assertEqual(model.dispatches, [])                    # handle still live -> no delete
            for ctx in model.ranks:
                self.assertEqual(set(ctx["recurrent_cache"]), {5})    # state intact for the survivor
            self.assertEqual(list(cache.keys()), ["alias_live"])
            pt.resumable.clear()
            self.assertEqual(cache.prune_stranded(), 1)               # deleted once, later
        self.assertEqual([a for _, a in model.dispatches], [(id(cache), [5])])
        for ctx in model.ranks:
            self.assertEqual(ctx["recurrent_cache"], {})
        self.assertEqual(cache.metrics["stash_pruned"], 2)

    def test_two_stranded_keys_one_handle_deduplicated(self):
        cache, model = self.make_cache(True, FakePagetable([]))
        shared = {"position": 0, "checkpoint_size": 10, "tp_handle": 5}
        cache["k1"] = shared
        cache["k2"] = shared
        for ctx in model.ranks:
            ctx["recurrent_cache"][5] = [FakeTensor(10)]
        with env_batch("1"), fake_torch(), NoteFreedRecorder(model) as freed:
            self.assertEqual(cache.prune_stranded(), 2)               # count kept per key
        self.assertEqual([a for _, a in model.dispatches], [(id(cache), [5])])  # one handle, not two
        self.assertEqual(cache.metrics["stash_pruned"], 2)
        for ctx in model.ranks:
            self.assertEqual(ctx["recurrent_cache"], {})
        self.assertEqual(freed.worker(), [(10, True)] * len(model.ranks))   # bytes counted once per rank


class BatchLsTests(Base):

    def test_aggregates_once_with_no_duplicate_bytes(self):
        cache, model = self.make_cache(False, FakePagetable(["live"]))
        a = seed_ls(cache, "d1", 100)
        cache["d2"] = a                                   # alias of the same stashed dict
        seed_ls(cache, "d3", 50)
        seed_ls(cache, "live", 7)
        del a
        with env_batch("1"), NoteFreedRecorder(model) as freed:
            self.assertEqual(cache.prune_stranded(), 3)
        self.assertEqual(freed.calls, [(150, False)])     # 100 (once, dedup'd by id) + 50
        self.assertEqual(model.dispatches, [])            # local path never dispatches
        self.assertEqual(list(cache.keys()), ["live"])
        self.assertEqual(cache.metrics["stash_pruned"], 3)
        self.assertEqual(cache.current_size, 7)

    def test_trim_fires_only_after_dropped_refs_released(self):
        cache, model = self.make_cache(False, FakePagetable(["live"]))
        for i in range(3):
            seed_ls(cache, f"d{i}", 400)
        seed_ls(cache, "live", 7)
        dead = ("d0", "d1", "d2")
        # plain dicts are not weakref-able; the inner state tensors are, and those are
        # what actually pin host memory (the dicts only hold them)
        refs = [weakref.ref(cache[k]["layer0"][0]) for k in dead]
        trims = []
        with (env_batch("1"),
              mock.patch.object(rc, "_TRIM_THRESHOLD", 1200),
              mock.patch.object(rc, "malloc_trim", lambda: trims.append(
                  [r() is not None for r in refs]))):
            self.assertEqual(cache.prune_stranded(), 3)
        self.assertEqual(len(trims), 1)                   # one aggregated note_freed crossing 1200
        self.assertEqual(trims[0], [False, False, False])  # every dropped state gone when trim ran

    def test_release_before_note_freed_ls_witness(self):
        # legacy LS path note_freed runs while `popped` still binds the entry in
        # turn; the batched path must call note_freed with nothing pinned
        def run(batched):
            cache, model = self.make_cache(False, FakePagetable([]))
            for i in range(2):
                seed_ls(cache, f"d{i}", 10)
            refs = [weakref.ref(cache[k]["layer0"][0]) for k in ("d0", "d1")]
            liveness = []
            orig = rc.note_freed
            def spy(nbytes):
                liveness.append([r() is not None for r in refs])
                orig(nbytes)
            with env_batch("1" if batched else None), mock.patch.object(rc, "note_freed", spy):
                cache.prune_stranded()
            return liveness
        self.assertTrue(any(run(batched=False)))          # the entry still being counted is bound
        self.assertEqual(run(batched=True), [[False, False]])


class BulkWorkerContractTests(Base):

    def test_missing_handle_asserts_before_mutation(self):
        ctx = {"recurrent_cache": {1: [FakeTensor(10)], 2: [FakeTensor(20)]}}
        before = dict(ctx["recurrent_cache"])
        with fake_torch(), NoteFreedRecorder(FakeModel(False)) as freed:
            with self.assertRaises(AssertionError):
                rc.mp_cache_recurrent_del_bulk(ctx, 9, [2, 99])    # 99 never existed
        self.assertEqual(ctx["recurrent_cache"], before)           # nothing partially deleted
        self.assertEqual(freed.calls, [])

    def test_duplicate_handles_assert_before_mutation(self):
        ctx = {"recurrent_cache": {1: [FakeTensor(10)]}}
        with fake_torch(), NoteFreedRecorder(FakeModel(False)) as freed:
            with self.assertRaises(AssertionError):
                rc.mp_cache_recurrent_del_bulk(ctx, 9, [1, 1])
        self.assertEqual(set(ctx["recurrent_cache"]), {1})
        self.assertEqual(freed.calls, [])

    def test_happy_path_sums_actual_bytes_once(self):
        # nested lists + non-tensor entries go through the real _stashed_bytes recursion
        ctx = {"recurrent_cache": {1: [[FakeTensor(30), FakeTensor(40)]], 2: [FakeTensor(50), "meta"]}}
        with fake_torch(), NoteFreedRecorder(FakeModel(False)) as freed:
            rc.mp_cache_recurrent_del_bulk(ctx, 9, [1, 2])
        self.assertEqual(ctx["recurrent_cache"], {})
        self.assertEqual(freed.calls, [(120, False)])              # a single note_freed, actual bytes


class EquivalenceTests(Base):

    def test_batch_and_legacy_agree_on_observable_state(self):
        # (aliasing scenarios are excluded by design: legacy re-dispatches a shared
        # handle per key and fails on the second pop; the batch path's dedup/liveness
        # handling is pinned by the alias tests above)
        def run(batched):
            cache, model = self.make_cache(True, FakePagetable(["live"]))
            for i in range(3):
                seed_tp(cache, model, f"d{i}", handle=10 + i, nbytes=100 + i)
            seed_tp(cache, model, "live", 99, nbytes=7)
            with env_batch("1" if batched else None), fake_torch():
                n = cache.prune_stranded()
            return n, cache, model
        with mock.patch.object(rc, "note_freed", lambda nbytes: None):
            n_l, cache_l, model_l = run(False)
            n_b, cache_b, model_b = run(True)
        self.assertEqual(n_l, n_b)
        self.assertEqual(cache_l.metrics, cache_b.metrics)
        self.assertEqual(list(cache_l.keys()), list(cache_b.keys()))
        self.assertEqual(cache_l.current_size, cache_b.current_size)
        for ctx_l, ctx_b in zip(model_l.ranks, model_b.ranks):
            self.assertEqual(set(ctx_l["recurrent_cache"]), set(ctx_b["recurrent_cache"]))
        # the whole point of the batch: fewer parent round-trips
        self.assertEqual(len(model_l.dispatches), 3)
        self.assertEqual(len(model_b.dispatches), 1)


if __name__ == "__main__":
    unittest.main(verbosity = 2)
