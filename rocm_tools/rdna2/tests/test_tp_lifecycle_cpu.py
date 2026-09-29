#!/usr/bin/env python3
"""CPU-only tests for the tensor-parallel coordinator lifecycle in
exllamav3/model/model_tp.py (Model_TPMixin).

Everything heavier than a pipe is faked INSIDE an alias-loaded copy of the
module (private package namespace; the real exllamav3 package is never
imported or patched, so this file is safe to collect alongside other suites):
multiprocessing Process/Pipe, the shared-memory SMProducer and the pseudo
output-rank conn/child are replaced with recording stubs. The native
extension and torch.distributed are never imported, and nothing touches a
GPU.

Covered lifecycle contracts:

  * create_tp_context partial-failure teardown (Process.start failure,
    pseudo-worker constructor failure, producer failure): already-spawned
    children are quit/joined/terminated with bounded waits, all conns close,
    the arena unlinks, the atexit hook drops, state resets for a retry, the
    ORIGINAL exception wins and cleanup shortfalls are attached to it
    (tp_cleanup_errors) instead of being swallowed.
  * atexit hook registered before the first spawn.
  * destroy_tp_context: idempotent (second call a no-op), tolerates missing
    pseudo conn / producer, guards child.join so one raise cannot strand the
    rest of the teardown, reports pseudo quit/close failures, producer close
    failures and survived-terminate children as an aggregate RuntimeError
    (.tp_teardown_errors) after ALL resources were processed, resets
    loaded_tp and the pending acks/refs while retaining tp_output_device for
    the load error path.
  * Only owned, started, real children are joined/terminated: never the
    PseudoChild (this process), never a child carrying the current PID.
  * Dead/missing/never-started selected workers fail the preflight BEFORE
    fan-out into the blocking pseudo-worker collective, with informative
    errors and zero commands dispatched; the all-alive fast path stays a
    couple of is_alive() polls per rank per step.
  * _load_tp failure (module load raise, worker-returned exception,
    abandoned generator): distribution producer closed, deferred STC bracket
    aborted (config stays reusable), in-flight module unloaded, arena and
    context destroyed, original error preserved, and the model reloadable
    afterwards.
  * Inference DISPATCH_TIMEOUT stays 20; the load path uses the separately
    configurable LOAD_DISPATCH_TIMEOUT and restores afterwards.
  * The uncommitted Qwen tp_dispatch_lm_head_argmax semantics (return_max,
    vocab_size crop argument) are preserved.

Real RCCL/NCCL rendezvous on hardware is out of scope here (that is the GPU
verification step).

Run from the repo root (or anywhere):
    python3 -m pytest -q -p no:cacheprovider rocm_tools/rdna2/tests/test_tp_lifecycle_cpu.py
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
import torch.multiprocessing  # noqa: F401  (create_tp_context calls torch.multiprocessing.set_sharing_strategy)

# model_tp is loaded under an alias package rooted at exllamav3/ so its
# relative imports resolve without executing the real exllamav3/__init__
# (which imports the native extension). Heavy leaves get stub modules; the
# torch-only util package is the real one. Nothing is ever installed under
# "exllamav3".
ALIAS = "tp_lifecycle_cpu_tests"


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
# Worker-function sentinels. model_tp star-imports model_tp_fn, so THESE
# identities are what its dispatch code sends over the pipes; tests compare
# by identity. Module-level defs so forward/prefill's ForkingPickler.dumps of
# (fn, args) resolves them by qualified name.
# ---------------------------------------------------------------------------

def mp_model_worker(*a, **k):
    raise AssertionError("worker loop must never run in CPU tests")

def mp_cpu_reduce(ctx): pass
def mp_model_forward(ctx, x, params, last_kv, prefill, single_idx): pass
def mp_model_append(ctx, exported): pass
def mp_model_append_gather(ctx): pass
def mp_model_forward_lm_head_argmax(*a): pass
def mp_set_plan(ctx, plan, active_devices): pass
def mp_set_consumer(ctx, producer_imp): pass
def mp_close_consumer(ctx): pass
def mp_cache_page_copy(*a): pass
def mp_cpu_cache_init(*a): pass
def mp_cpu_cache_store(*a): pass
def mp_cpu_cache_fetch(*a): pass
def mp_rotate_cache_pages(*a): pass
def touch_device_measure_vram(*a): pass


VRAM_RESULT = (1 << 30, 1 << 31)        # (free, total) acked by touch_device_measure_vram
FWD_RESULT = torch.tensor([[4.0]])      # non-None pseudo decode output (forward_tp asserts it)


def default_ack(msg):
    """What a healthy spawned worker would answer a command with."""
    if isinstance(msg, tuple) and msg[0] is touch_device_measure_vram:
        return VRAM_RESULT
    return None


def default_inline(fn, args):
    """What the in-process pseudo worker 'executes' a command to."""
    if fn is touch_device_measure_vram:
        return VRAM_RESULT
    if fn is mp_model_forward:
        return None if args[3] else FWD_RESULT   # prefill returns None, forward x
    return None


# ---------------------------------------------------------------------------
# CPU-only torch shim: model_tp resolves torch.* through its own namespace,
# and this host has no accelerator, so torch.device(int) / Tensor.to(int)
# would raise. The proxy accepts int device specs; everything else is real
# torch. (Never patching the real torch module for module_tp consumers.)
# ---------------------------------------------------------------------------

class FakeDevice:
    def __init__(self, index):
        self.index = index
        self.type = "cuda"

    def __repr__(self):
        return f"FakeDevice(cuda:{self.index})"


class TorchProxy:
    def __init__(self):
        pass

    def __getattr__(self, name):
        if name == "device":
            return self._device
        return getattr(torch, name)

    @staticmethod
    def _device(spec, *args):
        if isinstance(spec, int):
            return FakeDevice(spec)
        return torch.device(spec, *args)


class FakeSMProducer:
    def __init__(self, shm_name=None, buffer_size=0):
        self.buffer_size = buffer_size
        self.closed = 0
        self.sends = 0
        self.close_raises = False

    def export(self):
        return {"shm": id(self), "buffer_size": self.buffer_size}

    def send(self, tensor, cache_id=None):
        self.sends += 1
        return {"arena": id(self), "seq": self.sends}

    def clear(self):
        pass

    def close(self):
        self.closed += 1
        if self.close_raises:
            raise FileNotFoundError("arena already unlinked")


class FakeConn:
    """One end of a fake duplex pipe. The parent end records writes, reacts
    to 'quit' via an on_send hook, and auto-answers commands through a
    result_fn (mimicking an acking worker). dead models a closed peer: poll
    reports readable (EOF) and recv raises."""

    def __init__(self, name=""):
        self.name = name
        self.sent = []
        self.sent_bytes = 0
        self.inbox = deque()
        self.poll_timeouts = []
        self.closed = False
        self.dead = False
        self.on_send = None
        self.result_fn = None

    def send(self, msg):
        if self.closed:
            raise OSError("send on closed connection")
        self.sent.append(msg)
        if self.on_send is not None:
            self.on_send(msg)
        if self.result_fn is not None:
            self.inbox.append(self.result_fn(msg))

    def send_bytes(self, payload):
        if self.closed:
            raise OSError("send_bytes on closed connection")
        self.sent_bytes += 1
        if self.on_send is not None:
            self.on_send(("bytes", payload))
        if self.result_fn is not None:
            self.inbox.append(self.result_fn(("bytes", payload)))

    def poll(self, timeout=None):
        self.poll_timeouts.append(timeout)
        return bool(self.inbox) or self.dead

    def recv(self):
        if self.inbox:
            return self.inbox.popleft()
        if self.dead:
            raise EOFError("connection closed by peer")
        raise AssertionError(f"{self.name}: recv with no data queued")

    def close(self):
        self.closed = True


class FakeChild:
    """multiprocessing.Process stand-in with scripted start/join/terminate."""

    _next_pid = 100_000

    def __init__(self, target=None, args=(), **kwargs):
        self.target = target
        self.args = args
        self.device = args[1] if len(args) > 1 else None
        self.pid = None
        self._alive = False
        self.started = False
        self.join_timeouts = []
        self.terminate_count = 0
        self.alive_checks = 0
        self.kill_ready = False
        self.start_exc = None
        self.join_raises = False
        self._join_raised = False
        self.exits_on_quit = True
        self.survives_terminate = False

    def start(self):
        if self.start_exc is not None:
            raise self.start_exc
        self.started = True
        self._alive = True
        FakeChild._next_pid += 1
        self.pid = FakeChild._next_pid

    def is_alive(self):
        self.alive_checks += 1
        return self._alive

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if self.join_raises and not self._join_raised:
            self._join_raised = True
            raise RuntimeError("simulated join failure")
        if self.kill_ready:
            self._alive = False

    def terminate(self):
        self.terminate_count += 1
        if not self.survives_terminate:
            self.kill_ready = True


class FakePseudoParentConn:
    """PseudoParentConn stand-in: send() executes the command inline, exactly
    like the real conn can block inside collectives, through a swappable
    hook. Real class (not a factory function) so the source's isinstance
    checks keep working against the patched module attribute."""

    def __init__(self, device, active_devices, output_device, backend_args, producer, dbg_t0_):
        self.device = device
        self.active_devices = list(active_devices)
        self.sent = []
        self.result = None
        self.poll_timeouts = []
        self.quit_count = 0
        self.close_count = 0
        self.quit_raises = False
        self.inline = default_inline

    def send(self, msg):
        if msg == "quit":
            self.sent.append("quit")
            return
        self.sent.append(msg)
        fn, args = msg
        self.result = self.inline(fn, args)

    def poll(self, timeout=None):
        self.poll_timeouts.append(timeout)
        return True

    def recv(self):
        r = self.result
        self.result = None
        return r

    def close(self):
        self.close_count += 1

    def quit(self):
        self.quit_count += 1
        if self.quit_raises:
            raise RuntimeError("simulated pseudo backend close failure")
        self.close()


class FakePseudoChildConn:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class FakePseudoChild:
    def __init__(self):
        self.join_timeouts = []
        self.terminate_count = 0

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def terminate(self, *a, **k):
        self.terminate_count += 1


class Harness:
    """Binds fake Process/Pipe/SMProducer/pseudo classes into the loaded
    model_tp module for one test; uninstalls afterwards. Pseudo classes are
    patched with per-test SUBCLASSES so isinstance() in the source stays
    coherent with the instances create_tp_context produces."""

    def __init__(self, test, tp, mod, children_cfg=None, pseudo_raise=None,
                 inline_fn=None, arena_close_raises=False):
        self.test = test
        self.tp = tp
        self.mod = mod
        self.children_cfg = children_cfg or {}
        self.children = {}          # device -> FakeChild (spawned only)
        self.parent_conns = {}      # device -> FakeConn (spawned only)
        self.child_conns = {}
        self.producers = []         # creation order: arena first, then any others
        self.pseudo = None
        self.pseudo_raise = pseudo_raise
        self.inline_fn = inline_fn
        self.arena_close_raises = arena_close_raises
        self.start_hook_states = []  # was the atexit hook installed at each start()?
        self._pending_pair = None
        self._saved = {}

        harness = self

        class _Pseudo(FakePseudoParentConn):
            def __init__(s, *a, **k):
                if harness.pseudo_raise is not None:
                    raise harness.pseudo_raise
                super().__init__(*a, **k)
                if harness.inline_fn is not None:
                    s.inline = harness.inline_fn
                harness.pseudo = s

        class _Producer(FakeSMProducer):
            def __init__(s, *a, **k):
                super().__init__(*a, **k)
                s.close_raises = harness.arena_close_raises
                harness.producers.append(s)

        for name, value in (
            ("Process", self._fake_process),
            ("Pipe", self._fake_pipe),
            ("SMProducer", _Producer),
            ("PseudoParentConn", _Pseudo),
            ("PseudoChildConn", FakePseudoChildConn),
            ("PseudoChild", FakePseudoChild),
        ):
            self._saved[name] = getattr(mod, name)
            setattr(mod, name, value)

    def uninstall(self):
        for name, value in self._saved.items():
            setattr(self.mod, name, value)

    # -- factories -----------------------------------------------------------

    def _fake_pipe(self):
        pair = (FakeConn("parent"), FakeConn("child"))
        self._pending_pair = pair
        return pair

    def _fake_process(self, target=None, args=(), **kwargs):
        c = FakeChild(target=target, args=args, **kwargs)
        parent, child_conn = self._pending_pair
        self._pending_pair = None
        cfg = self.children_cfg.get(c.device, {})
        c.start_exc = cfg.get("start_exc")
        c.join_raises = cfg.get("join_raises", False)
        c.exits_on_quit = cfg.get("exits_on_quit", True)
        c.survives_terminate = cfg.get("survives_terminate", False)
        parent.result_fn = cfg.get("result_fn", default_ack)

        def on_send(msg):
            if msg == "quit" and c.exits_on_quit:
                c._alive = False
        parent.on_send = on_send

        bound_start = c.start

        def start():
            self.start_hook_states.append(self.test._hook_registered(self.tp))
            return bound_start()
        c.start = start

        self.children[c.device] = c
        self.parent_conns[c.device] = parent
        self.child_conns[c.device] = child_conn
        return c

    # -- scenario helpers ------------------------------------------------------

    def create(self, active=(1, 0), out=0):
        """Drive the real create_tp_context with fakes installed. `active`
        must already carry the output device last, like _load_tp arranges."""
        self.tp.active_devices = list(active)
        self.tp.tp_output_device = out
        self.tp.create_tp_context("native")

    def arena(self):
        return self.producers[0] if self.producers else None

    def assert_all_conns_closed(self):
        for d in self.children:
            self.test.assertTrue(self.parent_conns[d].closed, f"parent conn {d} open")
            self.test.assertTrue(self.child_conns[d].closed, f"child conn {d} open")


class FakeSTC:
    def __init__(self):
        self.deferred_mode = False
        self.begins = self.ends = self.aborts = self.closes = 0

    def begin_deferred_load(self, arena: bool = True):
        assert not self.deferred_mode, "stc stuck in deferred mode (missing abort)"
        self.deferred_mode = True
        self.begins += 1

    def end_deferred_load(self):
        assert self.deferred_mode
        self.deferred_mode = False
        self.ends += 1

    def abort_deferred_load(self):
        self.deferred_mode = False
        self.aborts += 1

    def close(self):
        self.closes += 1


class FakeConfig:
    def __init__(self, vocab_size=128):
        self.stc = FakeSTC()
        self.vocab_size = vocab_size


class FakeModule:
    def __init__(self, name, defer=False, load_exc=None, export_exc=None,
                 logits_output=False):
        self.name = name
        self.defer = defer
        self.load_exc = load_exc
        self.export_exc = export_exc
        self.caps = {"logits_output": logits_output}
        self.loads = self.exports = self.unloads = 0

    def make_tp_allocation(self, tp_options):
        return []

    def can_defer_load(self):
        return self.defer

    def load(self, device, tp_parent_defer=False):
        self.loads += 1
        if self.load_exc is not None:
            raise self.load_exc

    def tp_export(self, plan, producer):
        self.exports += 1
        if self.export_exc is not None:
            raise self.export_exc
        return {"cls": self.name}

    def unload(self):
        self.unloads += 1


class FakeProgress:
    def __init__(self, text, count, transient=True):
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n):
        self.updates.append(n)


class FakeAllocator:
    def __init__(self, components, num_tokens, output_num_tokens, dev_limits):
        self.components = components
        self.max_mem = None

    def initial_split(self, max_mem):
        self.max_mem = list(max_mem)

    def print_split(self):
        pass

    def compile_tp_plan(self):
        return {"fake": "plan"}


# ---------------------------------------------------------------------------
# base case: private alias package + stub modules, load the real model_tp.py
# ---------------------------------------------------------------------------

def install_alias_stubs():
    if ALIAS not in sys.modules:
        root = types.ModuleType(ALIAS)
        root.__path__ = [str(REPO / "exllamav3")]
        sys.modules[ALIAS] = root
        model = types.ModuleType(ALIAS + ".model")
        model.__path__ = [str(REPO / "exllamav3" / "model")]
        sys.modules[ALIAS + ".model"] = model

    if ALIAS + ".util" not in sys.modules:
        importlib.import_module(ALIAS + ".util")  # real package: torch-only deps

    stub_memory = types.ModuleType(ALIAS + ".util.memory")
    stub_memory.touch_device_measure_vram = touch_device_measure_vram
    sys.modules[ALIAS + ".util.memory"] = stub_memory

    stub_progress = types.ModuleType(ALIAS + ".util.progress")
    stub_progress.ProgressBar = FakeProgress
    sys.modules[ALIAS + ".util.progress"] = stub_progress

    stub_config = types.ModuleType(ALIAS + ".model.config")
    stub_config.Config = FakeConfig
    sys.modules[ALIAS + ".model.config"] = stub_config

    stub_tok_pkg = types.ModuleType(ALIAS + ".tokenizer")
    stub_tok_pkg.__path__ = [str(REPO / "exllamav3" / "tokenizer")]
    sys.modules[ALIAS + ".tokenizer"] = stub_tok_pkg
    stub_tok = types.ModuleType(ALIAS + ".tokenizer.mm_embedding")
    stub_tok.send_embeddings = lambda producer, ies: []
    sys.modules[ALIAS + ".tokenizer.mm_embedding"] = stub_tok

    stub_fn = types.ModuleType(ALIAS + ".model.model_tp_fn")
    stub_fn.torch = torch
    for fn in (
        mp_model_worker, mp_cpu_reduce, mp_model_forward, mp_model_append,
        mp_model_append_gather, mp_model_forward_lm_head_argmax, mp_set_plan,
        mp_set_consumer, mp_close_consumer, mp_cache_page_copy,
        mp_cpu_cache_init, mp_cpu_cache_store, mp_cpu_cache_fetch,
        mp_rotate_cache_pages,
    ):
        setattr(stub_fn, fn.__name__, fn)
    stub_fn.SMProducer = FakeSMProducer
    stub_fn.PseudoParentConn = FakePseudoParentConn
    stub_fn.PseudoChildConn = FakePseudoChildConn
    stub_fn.PseudoChild = FakePseudoChild
    sys.modules[ALIAS + ".model.model_tp_fn"] = stub_fn

    mod = _load_file(ALIAS + ".model.model_tp",
                     REPO / "exllamav3" / "model" / "model_tp.py")
    assert mod.__name__.startswith(ALIAS)
    assert "exllamav3" not in getattr(mod, "__package__", ""), "must not leak into real package"
    return mod


class LifecycleTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mod = install_alias_stubs()
        cls.cleanupper = cls.mod.cleanupper

    def setUp(self):
        self.harnesses = []
        self.baseline_hooks = list(self.cleanupper.atexit_fns)
        self.patch_mod("TPAllocator", FakeAllocator)
        self.patch_mod("torch", TorchProxy())

    def tearDown(self):
        for h in self.harnesses:
            h.uninstall()
        # No test may leak children, conns, the arena or the atexit hook.
        self.assertEqual(self.cleanupper.atexit_fns, self.baseline_hooks,
                         "atexit hook list leaked")

    def patch_mod(self, name, value):
        p = patch.object(self.mod, name, value)
        p.start()
        self.addCleanup(p.stop)

    def make(self, children_cfg=None, pseudo_raise=None, inline_fn=None,
             arena_close_raises=False):
        tp = self.mod.Model_TPMixin()
        tp.active_devices = []
        h = Harness(self, tp, self.mod, children_cfg, pseudo_raise, inline_fn,
                    arena_close_raises)
        self.harnesses.append(h)
        return tp, h

    def _hook_registered(self, tp):
        return tp.destroy_tp_context in self.cleanupper.atexit_fns

    def _load_gen(self, tp, modules, cfg=None, generator=True, active=(1, 0),
                  out=0, callback=None):
        cfg = cfg if cfg is not None else FakeConfig()
        return cfg, tp._load_tp(
            progressbar=False,
            reserve_per_device=None,
            use_per_device=None,
            active_devices=list(active),
            max_chunk_size=64,
            max_output_size=8,
            max_output_factor=4,
            callback_sync=callback,
            generator=generator,
            tp_output_device=out,
            config=cfg,
            modules=modules,
            dev_limits=None,
            tp_backend="native",
            verbose=False,
            tp_options={},
        )

    def assert_state_reset(self, tp):
        self.assertEqual(tp.mp_children, [])
        self.assertEqual(tp.mp_parent_conn, [])
        self.assertEqual(tp.mp_child_conn, [])
        self.assertIsNone(tp.tp_producer)
        self.assertEqual(tp.tp_pending_acks, [])
        self.assertIsNone(tp.tp_pending_refs)
        self.assertFalse(tp.loaded_tp)
        self.assertFalse(self._hook_registered(tp))


# ---------------------------------------------------------------------------
# create_tp_context
# ---------------------------------------------------------------------------

class TestCreateContext(LifecycleTestCase):

    def test_successful_create_and_destroy_balances_hook(self):
        tp, h = self.make()
        h.create()
        self.assertTrue(self._hook_registered(tp))
        self.assertEqual(h.start_hook_states, [True, True],
                         "atexit hook must be installed before the first spawn")
        self.assertEqual(len(tp.mp_children), 3)      # dev 0, 1 + CPU slot
        self.assertIsInstance(tp.mp_parent_conn[0], FakePseudoParentConn)
        self.assertIsInstance(tp.mp_children[0], FakePseudoChild)
        self.assertEqual(h.arena().buffer_size, 64 * 1024 ** 2)
        tp.destroy_tp_context()
        self.assert_state_reset(tp)
        self.assertEqual(h.arena().closed, 1)

    def test_unknown_backend_raises_before_any_resource(self):
        tp, h = self.make()
        tp.active_devices = [1, 0]
        tp.tp_output_device = 0
        with self.assertRaisesRegex(ValueError, "Unkwown backend"):
            tp.create_tp_context("nope")
        self.assert_state_reset(tp)
        self.assertEqual(h.producers, [])
        self.assertEqual(h.children, {})

    def test_partial_spawn_failure_cleans_and_allows_retry(self):
        boom = OSError(11, "Resource temporarily unavailable")
        tp, h = self.make(children_cfg={-1: {"start_exc": boom}})
        with self.assertRaises(OSError) as cm:
            h.create()
        self.assertIs(cm.exception, boom)             # original error wins
        self.assert_state_reset(tp)
        dev1 = h.children[1]
        self.assertIn("quit", h.parent_conns[1].sent) # spawned child got quit
        self.assertFalse(dev1.is_alive())
        self.assertEqual(dev1.terminate_count, 0)     # exited gracefully
        self.assertTrue(h.parent_conns[1].closed)
        self.assertTrue(h.child_conns[1].closed)
        self.assertEqual(h.arena().closed, 1)         # arena unlinked once
        never_started = h.children[-1]
        self.assertEqual(never_started.pid, None)
        self.assertEqual(never_started.join_timeouts, [])   # join would raise
        self.assertEqual(never_started.terminate_count, 0)
        self.assertIsNone(getattr(boom, "tp_cleanup_errors", None))
        # retry with healthy spawn
        h.children_cfg.clear()
        h.create()
        self.assertTrue(self._hook_registered(tp))
        tp.destroy_tp_context()
        self.assert_state_reset(tp)

    def test_pseudo_constructor_failure_cleans_children(self):
        tp, h = self.make(pseudo_raise=RuntimeError("simulated PG init failure"))
        with self.assertRaises(RuntimeError) as cm:
            h.create()
        self.assertIn("PG init", str(cm.exception))
        self.assertIsNone(h.pseudo)                   # object never materialized
        self.assertIn("quit", h.parent_conns[1].sent)
        self.assertTrue(h.parent_conns[1].closed)
        self.assertEqual(h.arena().closed, 1)
        self.assert_state_reset(tp)

    def test_producer_failure_leaves_nothing_behind(self):
        class NoShm(FakeSMProducer):
            def __init__(self, *a, **k):
                raise MemoryError("shm refused")
        tp, h = self.make()
        saved = self.mod.SMProducer
        self.mod.SMProducer = NoShm
        try:
            tp.active_devices = [1, 0]
            tp.tp_output_device = 0
            with self.assertRaises(MemoryError):
                tp.create_tp_context("native")
        finally:
            self.mod.SMProducer = saved
        self.assert_state_reset(tp)
        self.assertEqual(tp.tp_backend, "native")     # backend choice is retained

    def test_teardown_failure_during_create_cleanup_is_attached_not_swallowed(self):
        boom = OSError(11, "no fork")
        tp, h = self.make(children_cfg={-1: {"start_exc": boom}},
                          arena_close_raises=True)
        with self.assertRaises(OSError) as cm:
            h.create()
        self.assertIs(cm.exception, boom)             # original still propagates
        attached = getattr(boom, "tp_cleanup_errors", None)
        self.assertTrue(attached, "cleanup shortfall must be visible on the error")
        self.assertIn("producer", " ".join(attached))
        self.assert_state_reset(tp)                   # and cleanup still ran
        self.assertEqual(h.arena().closed, 1)


# ---------------------------------------------------------------------------
# destroy_tp_context
# ---------------------------------------------------------------------------

class TestDestroyContext(LifecycleTestCase):

    def test_destroy_on_fresh_object_is_safe_and_repeatable(self):
        tp, h = self.make()
        tp.destroy_tp_context()
        tp.destroy_tp_context()
        self.assert_state_reset(tp)

    def test_destroy_missing_pseudo_conn_and_producer(self):
        tp, h = self.make()
        # half-built context: real children spawned, pseudo slot never
        # filled, no arena producer
        c = FakeChild(args=(None, 1))
        c.start()
        parent, child_conn = FakeConn("p1"), FakeConn("c1")
        parent.on_send = lambda m: setattr(c, "_alive", False) if m == "quit" else None
        c2 = FakeChild(args=(None, 2))
        c2.start()
        parent2, child_conn2 = FakeConn("p2"), FakeConn("c2")
        parent2.on_send = lambda m: setattr(c2, "_alive", False) if m == "quit" else None
        tp.mp_children = [None, c, c2]
        tp.mp_parent_conn = [None, parent, parent2]
        tp.mp_child_conn = [None, child_conn, child_conn2]
        tp.tp_output_device = 0        # pseudo conn absent in that slot
        tp.tp_producer = None
        tp.destroy_tp_context()        # must not raise on the missing pieces
        self.assertFalse(c.is_alive())
        self.assertFalse(c2.is_alive())
        self.assertEqual(c.join_timeouts, [self.mod.CHILD_JOIN_TIMEOUT])
        self.assertTrue(parent.closed and child_conn.closed)
        self.assertTrue(parent2.closed and child_conn2.closed)
        self.assert_state_reset(tp)

    def test_repeated_destroy_reaps_once(self):
        tp, h = self.make(children_cfg={1: {"exits_on_quit": False},
                                        -1: {"exits_on_quit": False}})
        h.create()
        tp.destroy_tp_context()
        dev1, cpu = h.children[1], h.children[-1]
        self.assertEqual(dev1.terminate_count, 1)
        self.assertEqual(dev1.join_timeouts,
                         [self.mod.CHILD_JOIN_TIMEOUT, self.mod.CHILD_TERMINATE_JOIN_TIMEOUT])
        self.assertEqual(cpu.terminate_count, 1)
        self.assertEqual(h.pseudo.quit_count, 1)
        self.assertEqual(h.pseudo.close_count, 1)
        self.assertEqual(h.arena().closed, 1)
        h.assert_all_conns_closed()
        tp.destroy_tp_context()        # second call: pure no-op
        self.assertEqual(dev1.terminate_count, 1)
        self.assertEqual(h.pseudo.quit_count, 1)
        self.assertEqual(h.arena().closed, 1)

    def test_bounded_join_terminate_join_of_owned_real_children_only(self):
        tp, h = self.make(children_cfg={1: {"exits_on_quit": False}})
        h.create()
        pseudo_child = tp.mp_children[0]
        self.assertIsInstance(pseudo_child, FakePseudoChild)
        tp.destroy_tp_context()
        # pseudo rank: never joined, never terminated (it IS this process)
        self.assertEqual(pseudo_child.join_timeouts, [])
        self.assertEqual(pseudo_child.terminate_count, 0)
        dev1 = h.children[1]
        self.assertEqual(dev1.join_timeouts,
                         [self.mod.CHILD_JOIN_TIMEOUT, self.mod.CHILD_TERMINATE_JOIN_TIMEOUT])
        self.assertTrue(all(t > 0 for t in dev1.join_timeouts))   # both bounded

    def test_child_with_current_pid_is_left_alone_but_conns_closed(self):
        tp, h = self.make()
        h.create()
        dev1 = h.children[1]
        dev1.pid = os.getpid()         # mislabelled pseudo: never touch it
        tp.destroy_tp_context()
        self.assertEqual(dev1.terminate_count, 0)
        self.assertEqual(dev1.join_timeouts, [])
        self.assertEqual(h.parent_conns[1].sent, [])   # no quit write either
        self.assertTrue(h.parent_conns[1].closed)
        self.assertTrue(h.child_conns[1].closed)
        # but the real CPU helper still got quit and reaped
        self.assertIn("quit", h.parent_conns[-1].sent)
        self.assertFalse(h.children[-1].is_alive())

    def test_first_join_raises_but_rest_of_teardown_completes(self):
        tp, h = self.make(children_cfg={1: {"join_raises": True}})
        h.create()
        with self.assertRaises(RuntimeError) as cm:
            tp.destroy_tp_context()
        msg = str(cm.exception)
        self.assertIn("joining worker device 1", msg)
        self.assertIn("simulated join failure", msg)
        dev1 = h.children[1]
        # join failure assumed stuck → terminate → bounded re-reap happened
        self.assertEqual(dev1.terminate_count, 1)
        self.assertEqual(dev1.join_timeouts,
                         [self.mod.CHILD_JOIN_TIMEOUT, self.mod.CHILD_TERMINATE_JOIN_TIMEOUT])
        # the OTHER child still got quit and reaped, conns closed, pseudo quit,
        # arena unlinked: one join exception cannot strand resources
        self.assertIn("quit", h.parent_conns[-1].sent)
        self.assertFalse(h.children[-1].is_alive())
        h.assert_all_conns_closed()
        self.assertEqual(h.pseudo.quit_count, 1)
        self.assertEqual(h.arena().closed, 1)
        self.assert_state_reset(tp)
        self.assertTrue(getattr(cm.exception, "tp_teardown_errors", None))
        tp.destroy_tp_context()        # idempotent even after an erroring first call

    def test_pseudo_quit_failure_still_closes_and_is_reported(self):
        tp, h = self.make()
        h.create()
        h.pseudo.quit_raises = True
        with self.assertRaises(RuntimeError) as cm:
            tp.destroy_tp_context()
        self.assertIn("pseudo-worker quit", str(cm.exception))
        self.assertEqual(h.pseudo.close_count, 1)     # fallback close still ran
        self.assertFalse(h.children[1].is_alive())
        self.assertFalse(h.children[-1].is_alive())
        h.assert_all_conns_closed()
        self.assertEqual(h.arena().closed, 1)
        self.assert_state_reset(tp)

    def test_producer_close_failure_is_reported(self):
        tp, h = self.make()
        h.create()
        h.arena().close_raises = True
        with self.assertRaises(RuntimeError) as cm:
            tp.destroy_tp_context()
        self.assertIn("producer", str(cm.exception))
        h.assert_all_conns_closed()
        self.assert_state_reset(tp)

    def test_child_surviving_terminate_is_a_reported_leak(self):
        tp, h = self.make(children_cfg={1: {"exits_on_quit": False,
                                            "survives_terminate": True}})
        h.create()
        with self.assertRaises(RuntimeError) as cm:
            tp.destroy_tp_context()
        self.assertIn("survived terminate", str(cm.exception))
        self.assertIn("device 1", str(cm.exception))
        # the rest was still cleaned
        self.assertFalse(h.children[-1].is_alive())
        h.assert_all_conns_closed()
        self.assert_state_reset(tp)

    def test_multiple_teardown_failures_aggregate(self):
        tp, h = self.make(children_cfg={1: {"exits_on_quit": False,
                                            "survives_terminate": True}})
        h.create()
        h.pseudo.quit_raises = True
        h.arena().close_raises = True
        with self.assertRaises(RuntimeError) as cm:
            tp.destroy_tp_context()
        joined = " ".join(cm.exception.tp_teardown_errors)
        self.assertIn("pseudo-worker quit", joined)
        self.assertIn("survived terminate", joined)
        self.assertIn("producer", joined)
        self.assert_state_reset(tp)
        h.assert_all_conns_closed()

    def test_loaded_tp_reset_on_destroy_output_device_retained(self):
        tp, h = self.make()
        h.create()
        tp.loaded_tp = True
        tp.destroy_tp_context()
        self.assertFalse(tp.loaded_tp)                # restart permitted
        self.assertEqual(tp.tp_output_device, 0)      # retained for load error paths
        tp.active_devices = [1, 0]
        tp.create_tp_context("native")                # direct retry works
        tp.destroy_tp_context()
        self.assert_state_reset(tp)

    def test_pending_refs_and_acks_reset_even_when_drain_fails(self):
        tp, h = self.make()
        h.create()
        tp.tp_pending_acks = [1, -1]
        tp.tp_pending_refs = ("pinned args", object())
        h.parent_conns[1].dead = True                 # drain raises EOFError
        with self.assertRaises(RuntimeError) as cm:
            tp.destroy_tp_context()
        self.assertIn("draining deferred acks", str(cm.exception))
        self.assertEqual(tp.tp_pending_acks, [])
        self.assertIsNone(tp.tp_pending_refs)         # refs dropped regardless
        h.assert_all_conns_closed()
        self.assert_state_reset(tp)

    def test_dead_child_skips_quit_send_and_destroy_is_silent(self):
        tp, h = self.make()
        h.create()
        h.children[1]._alive = False
        tp.destroy_tp_context()                       # quiet, everything reaped
        self.assertEqual(h.children[1].terminate_count, 0)
        self.assertEqual(h.children[1].join_timeouts, [self.mod.CHILD_JOIN_TIMEOUT])
        h.assert_all_conns_closed()


# ---------------------------------------------------------------------------
# dead/missing worker preflight
# ---------------------------------------------------------------------------

class TestWorkerPreflight(LifecycleTestCase):

    def _dead_setup(self):
        tp, h = self.make()
        h.create()
        tripped = []

        def spy_inline(fn, args):
            tripped.append(fn)
            return default_inline(fn, args)
        h.pseudo.inline = spy_inline
        h.children[1]._alive = False
        return tp, h, tripped

    def test_forward_preflights_before_pseudo_collective(self):
        tp, h, tripped = self._dead_setup()
        with self.assertRaisesRegex(RuntimeError, "device 1 worker is no longer alive"):
            tp.forward_tp(torch.zeros(4), {}, 0, [])
        self.assertEqual(tripped, [])                 # blocking call never entered
        self.assertEqual(h.pseudo.sent, [])
        self.assertEqual(h.parent_conns[1].sent, [])  # no partial fan-out
        self.assertEqual(h.parent_conns[1].sent_bytes, 0)
        self.assertEqual(h.parent_conns[-1].sent, [])
        tp.destroy_tp_context()

    def test_prefill_preflights_before_pseudo_collective(self):
        tp, h, tripped = self._dead_setup()
        with self.assertRaisesRegex(RuntimeError, "no longer alive"):
            tp.prefill_tp(torch.zeros(4), {}, 0, [])
        self.assertEqual(tripped, [])
        self.assertEqual(h.pseudo.sent, [])
        tp.destroy_tp_context()

    def test_single_dispatch_checks_child_and_pseudo_before_send(self):
        tp, h, tripped = self._dead_setup()
        for device in (1, 0):
            with self.assertRaisesRegex(RuntimeError, "no longer alive"):
                tp.tp_worker_dispatch_single(device, mp_cpu_reduce, ())
        self.assertEqual(tripped, [])
        self.assertEqual(h.parent_conns[1].sent, [])
        self.assertEqual(h.pseudo.sent, [])
        tp.destroy_tp_context()

    def test_dispatch_multi_refuses_before_any_send(self):
        tp, h, _ = self._dead_setup()
        with self.assertRaisesRegex(RuntimeError, "no longer alive"):
            tp.tp_worker_dispatch_multi([1, 0], mp_set_plan,
                                        ({"fake": "plan"}, [1, 0]))
        self.assertEqual(h.parent_conns[1].sent, [])
        self.assertEqual(h.pseudo.sent, [])
        tp.destroy_tp_context()

    def test_missing_and_never_started_selected_workers_are_not_healthy(self):
        tp, h = self.make()
        h.create()
        # slot cleared: dispatch must name it, not block
        tp.mp_children[1] = None
        with self.assertRaisesRegex(RuntimeError, r"device 1 worker missing"):
            tp.tp_worker_dispatch(1, mp_cpu_reduce, ())
        # never-started child (no pid): same
        tp.mp_children[1] = FakeChild(args=(h.child_conns[1], 1))
        with self.assertRaisesRegex(RuntimeError, "never started"):
            tp.tp_worker_dispatch(1, mp_cpu_reduce, ())
        tp.destroy_tp_context()   # slot skipped by reapers, conns still closed
        h.assert_all_conns_closed()

    def test_no_context_dispatch_is_informative_not_indexerror(self):
        tp, h = self.make()
        with self.assertRaisesRegex(RuntimeError, "no TP context"):
            tp.tp_worker_dispatch(1, mp_cpu_reduce, ())

    def test_child_owned_by_this_pid_exits_preflight_but_forward_proceeds(self):
        # a child carrying THIS pid is not polled for liveness (we would never
        # waitpid our own pid): forward must proceed instead of failing
        tp, h = self.make()
        h.create()
        h.children[1].pid = os.getpid()
        h.children[1]._alive = False
        out = tp.forward_tp(torch.zeros(4), {}, 0, [])
        self.assertIs(out, FWD_RESULT)
        tp.destroy_tp_context()

    def test_fast_path_overhead_is_a_handful_of_polls(self):
        tp, h = self.make()
        h.create()
        tp.forward_tp(torch.zeros(4), {}, 0, [])
        self.assertEqual(h.parent_conns[1].sent_bytes, 1)   # exactly one fan-out
        # second forward drains the first pass's acks then fans out again
        out2 = tp.forward_tp(torch.zeros(4), {}, 0, [])
        self.assertIs(out2, FWD_RESULT)
        self.assertEqual(h.parent_conns[1].sent_bytes, 2)
        # constant, small number of is_alive() polls per step, not a re-scan of
        # every slot every send: ≤2 for the GPU rank, ≤3/forward for the CPU
        # helper (entry preflight, its own dispatch, pseudo's re-check)
        self.assertLessEqual(h.children[1].alive_checks, 5)
        self.assertLessEqual(h.children[-1].alive_checks, 7)
        tp.destroy_tp_context()
        self.assertEqual(tp.tp_pending_acks, [])
        self.assertIsNone(tp.tp_pending_refs)

    def test_result_error_shapes(self):
        boom = RuntimeError("worker-side fault")
        tp, h = self.make(children_cfg={1: {"result_fn": lambda msg: boom}})
        h.create()
        # worker-returned exception re-raised as-is
        with self.assertRaises(RuntimeError) as cm:
            tp.tp_worker_dispatch_single(1, mp_cpu_reduce, ())
        self.assertIs(cm.exception, boom)
        # dead peer (EOF on recv) becomes a named RuntimeError
        h.parent_conns[1].result_fn = None
        h.parent_conns[1].dead = True
        with self.assertRaisesRegex(RuntimeError, "device 1 died while a result"):
            tp.tp_worker_result(1)
        # silent-but-alive worker → TimeoutError naming device and timeout
        h.parent_conns[1].dead = False
        tp.tp_dispatch_timeout = 0.5
        with self.assertRaisesRegex(TimeoutError, "after 0.5s waiting for worker on device 1"):
            tp.tp_worker_result(1)
        # a dead worker discovered at timeout time is mentioned too
        h.children[1]._alive = False
        with self.assertRaisesRegex(TimeoutError, "is no longer alive"):
            tp.tp_worker_result(1)
        tp.destroy_tp_context()


# ---------------------------------------------------------------------------
# _load_tp failure cleanup / restart
# ---------------------------------------------------------------------------

class TestLoadTpLifecycle(LifecycleTestCase):

    def test_successful_load_and_unload(self):
        tp, h = self.make()
        mods = [FakeModule("a"), FakeModule("b", defer=True, logits_output=True)]
        cfg, gen = self._load_gen(tp, mods)
        outs = list(gen)
        self.assertEqual(outs, [(0, 2), (1, 2), (2, 2)])
        self.assertTrue(tp.loaded_tp)
        self.assertEqual(tp.active_devices, [1, 0])          # output moved last
        self.assertEqual((cfg.stc.begins, cfg.stc.ends, cfg.stc.closes), (1, 1, 1))
        self.assertEqual(cfg.stc.aborts, 0)
        for m in mods:
            self.assertEqual((m.loads, m.exports, m.unloads), (1, 1, 1))
        # distribution producer closed, arena producer untouched
        self.assertEqual(h.producers[1].closed, 1)
        self.assertEqual(h.arena().closed, 0)
        # pseudo saw the whole command sequence, gather included
        seq = [msg[0] for msg in h.pseudo.sent if not isinstance(msg, str)]
        self.assertEqual(seq[-2:], [mp_model_append_gather, mp_close_consumer])
        self.assertEqual(tp.tp_dispatch_timeout, self.mod.DISPATCH_TIMEOUT)
        tp.unload_tp()
        self.assert_state_reset(tp)
        self.assertIsNone(tp.tp_output_device)
        self.assertEqual(h.arena().closed, 1)
        h.assert_all_conns_closed()
        tp.unload_tp()                                       # repeat-safe

    def test_failed_module_load_cleans_and_allows_restart(self):
        tp, h = self.make()
        boom = RuntimeError("stloader fault mid module load")
        m_ok = FakeModule("ok")
        m_bad = FakeModule("bad", defer=True, load_exc=boom)
        cfg, gen = self._load_gen(tp, [m_ok, m_bad])
        with self.assertRaises(RuntimeError) as cm:
            for _ in gen:
                pass
        self.assertIs(cm.exception, boom)                    # original preserved
        self.assert_state_reset(tp)
        # deferred bracket: begin fired, end could not → aborted, stc usable
        self.assertEqual((cfg.stc.begins, cfg.stc.ends, cfg.stc.aborts), (1, 0, 1))
        cfg.stc.begin_deferred_load()                        # stc really usable again
        cfg.stc.end_deferred_load()
        # the in-flight module was unloaded; earlier modules already were
        self.assertEqual((m_ok.unloads, m_bad.unloads), (1, 1))
        # temporary distribution producer closed AND arena producer closed via destroy
        self.assertEqual(h.producers[1].closed, 1)
        self.assertEqual(h.arena().closed, 1)
        # children quit+reaped, conns closed, atexit hook dropped
        for d in h.children:
            self.assertFalse(h.children[d].is_alive())
        h.assert_all_conns_closed()
        self.assertEqual(tp.tp_dispatch_timeout, self.mod.DISPATCH_TIMEOUT)
        self.assertIsNone(getattr(boom, "tp_cleanup_errors", None))  # cleanup silent
        # RESTART on the same object
        mods2 = [FakeModule("r1"), FakeModule("r2", defer=True)]
        cfg2, gen2 = self._load_gen(tp, mods2)
        list(gen2)
        self.assertTrue(tp.loaded_tp)
        self.assertEqual(
            sum(1 for f in self.cleanupper.atexit_fns if f == tp.destroy_tp_context), 1)
        tp.unload_tp()
        self.assert_state_reset(tp)

    def test_worker_exception_during_load_propagates_and_cleans(self):
        def rf(msg):
            if isinstance(msg, tuple) and msg[0] is mp_model_append:
                return RuntimeError("worker OOM during append")
            return default_ack(msg)
        tp, h = self.make(children_cfg={1: {"result_fn": rf}})
        cfg, gen = self._load_gen(tp, [FakeModule("m0")])
        with self.assertRaisesRegex(RuntimeError, "worker OOM during append"):
            list(gen)
        self.assert_state_reset(tp)
        self.assertEqual(h.arena().closed, 1)
        self.assertEqual(h.producers[1].closed, 1)
        for d in h.children:
            self.assertFalse(h.children[d].is_alive())

    def test_abandoned_generator_cleans_like_an_exception(self):
        tp, h = self.make()
        mods = [FakeModule("m0"), FakeModule("m1")]
        cfg, gen = self._load_gen(tp, mods, generator=True)
        first = next(gen)                                   # mid-load...
        self.assertEqual(first, (0, 2))
        gen.close()                                         # ...abandoned
        self.assert_state_reset(tp)
        self.assertEqual(h.arena().closed, 1)
        self.assertEqual(h.producers[1].closed, 1)
        h.assert_all_conns_closed()

    def test_partial_create_failure_inside_load_not_double_destroyed(self):
        boom = OSError(11, "no fork")
        tp, h = self.make(children_cfg={-1: {"start_exc": boom}})
        cfg, gen = self._load_gen(tp, [FakeModule("m0")])
        with self.assertRaises(OSError) as cm:
            list(gen)                                       # create self-cleaned
        self.assertIs(cm.exception, boom)
        self.assertEqual(h.arena().closed, 1)               # not closed twice
        self.assertEqual(h.pseudo.quit_count, 1)            # quit exactly once
        self.assert_state_reset(tp)
        self.assertIsNone(getattr(boom, "tp_cleanup_errors", None))

    def test_cleanup_failure_during_load_is_attached_to_original(self):
        boom = RuntimeError("module import failed")
        tp, h = self.make(arena_close_raises=True)
        cfg, gen = self._load_gen(tp, [FakeModule("m", load_exc=boom)])
        with self.assertRaises(RuntimeError) as cm:
            list(gen)
        self.assertIs(cm.exception, boom)
        attached = getattr(boom, "tp_cleanup_errors", None)
        self.assertTrue(attached)
        self.assertIn("producer", " ".join(attached))
        self.assert_state_reset(tp)


# ---------------------------------------------------------------------------
# timeouts
# ---------------------------------------------------------------------------

class TestTimeoutSeparation(LifecycleTestCase):

    def test_inference_default_untouched_load_separately_configurable(self):
        self.assertEqual(self.mod.DISPATCH_TIMEOUT, 20)
        self.assertGreater(self.mod.LOAD_DISPATCH_TIMEOUT, self.mod.DISPATCH_TIMEOUT)
        saved = os.environ.get("EXL3_TP_LOAD_TIMEOUT")
        try:
            os.environ.pop("EXL3_TP_LOAD_TIMEOUT", None)
            self.assertEqual(self.mod._load_dispatch_timeout(), 180.0)
            os.environ["EXL3_TP_LOAD_TIMEOUT"] = "45"
            self.assertEqual(self.mod._load_dispatch_timeout(), 45.0)
            for bad in ("abc", "0", "-5", "nan", "inf", ""):
                os.environ["EXL3_TP_LOAD_TIMEOUT"] = bad
                self.assertEqual(self.mod._load_dispatch_timeout(), 180.0)
        finally:
            os.environ.pop("EXL3_TP_LOAD_TIMEOUT", None)
            if saved is not None:
                os.environ["EXL3_TP_LOAD_TIMEOUT"] = saved

    def test_load_polls_use_long_timeout_inference_restores(self):
        tp, h = self.make()
        with patch.object(self.mod, "LOAD_DISPATCH_TIMEOUT", 99.0):
            _, gen = self._load_gen(tp, [FakeModule("m0")])
            list(gen)
        self.assertTrue(h.parent_conns[1].poll_timeouts)
        self.assertEqual(set(h.parent_conns[1].poll_timeouts), {99.0})
        self.assertEqual(set(h.pseudo.poll_timeouts), {99.0})
        self.assertEqual(h.parent_conns[-1].poll_timeouts, [])   # not polled during load
        self.assertEqual(tp.tp_dispatch_timeout, self.mod.DISPATCH_TIMEOUT)
        # inference path uses the short default again
        tp.forward_tp(torch.zeros(4), {}, 0, [])
        self.assertEqual(h.pseudo.poll_timeouts[-1], self.mod.DISPATCH_TIMEOUT)
        tp.unload_tp()
        self.assertEqual(set(h.parent_conns[-1].poll_timeouts), {self.mod.DISPATCH_TIMEOUT})
        self.assertEqual(h.parent_conns[1].poll_timeouts[-1], self.mod.DISPATCH_TIMEOUT)


# ---------------------------------------------------------------------------
# uncommitted Qwen argmax semantics preserved
# ---------------------------------------------------------------------------

class TestArgmaxQwenSemantics(LifecycleTestCase):

    def _plan(self, spans):
        return {d: {"lm_head": (a, b, None)} for d, (a, b) in spans.items()}

    def test_master_only_single_path_return_max_and_vocab_arg(self):
        tp, h = self.make()
        h.create()
        tp.plan = self._plan({1: (0, 0), 0: (0, 8)})
        tp.config = FakeConfig(vocab_size=123)
        v, i = torch.tensor([[1.5]]), torch.tensor([[7]])
        h.pseudo.inline = lambda fn, args: (v, i)
        got = tp.tp_dispatch_lm_head_argmax(({"arena": 1}, {}), return_max=True)
        self.assertIs(got[0], i)
        self.assertIs(got[1], v)
        fn, args = h.pseudo.sent[-1]
        self.assertIs(fn, mp_model_forward_lm_head_argmax)
        self.assertEqual(args, ({"arena": 1}, {}, 0, None, None, 123))
        self.assertIs(tp.tp_dispatch_lm_head_argmax(({"arena": 1}, {})), i)
        tp.destroy_tp_context()

    def test_multi_rank_gather_selects_winner_and_draft_conf(self):
        tp, h = self.make()
        h.create()
        tp.plan = self._plan({1: (0, 64), 0: (64, 128)})
        tp.config = FakeConfig(vocab_size=128)
        all_vals = torch.tensor([[5.0, 9.0]])
        all_inds = torch.tensor([[3, 101]])
        h.pseudo.inline = lambda fn, args: (all_vals, all_inds)
        # CPU-only host: .to(int) needs no accelerator here (the gather already
        # produced output-device tensors); make ordinal specs identity
        real_to = torch.Tensor.to

        def cpu_to(self, spec, *a, **k):
            return self if isinstance(spec, int) else real_to(self, spec, *a, **k)
        p = patch.object(torch.Tensor, "to", cpu_to)
        p.start()
        self.addCleanup(p.stop)
        argmax, conf = tp.tp_dispatch_lm_head_argmax(({"arena": 1}, {}), return_max=True)
        self.assertEqual(int(argmax), 101)
        self.assertAlmostEqual(float(conf), 9.0)
        fn, args = h.parent_conns[1].sent[-1]
        self.assertIs(fn, mp_model_forward_lm_head_argmax)
        self.assertEqual(args[2:], (0, [0, 1], [1, 1], 128))    # offset, gd, ldims, vocab
        tp.destroy_tp_context()


if __name__ == "__main__":
    unittest.main()
