#!/usr/bin/env python3
"""CPU-only queue-contract test for mp_rotate_cache_pages (TP idle-defrag ack race).

The worker uploads rotation indices from the pinned shared arena with a
non-blocking H2D and launches cache_rotate kernels on the current stream. The
parent reuses (clears and rewrites) that arena as soon as the dispatch acks,
i.e. when this function returns. Unless the function drains the stream first,
the upload executes against the NEXT forward's bytes (poison) and rotates the
cache with garbage indices.

Nothing is imported from exllamav3: the real function source is extracted with
ast and exec'd against fakes. No GPU, no native extension, no package stubs.

Run: python3 -m pytest -q -p no:cacheprovider rocm_tools/rdna2/tests/test_tp_defrag_cpu.py
"""

import ast
from functools import lru_cache
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "exllamav3" / "model" / "model_tp_fn.py"
WORKER_DEV = 1  # explicit non-zero device rank: catches a dropped/implicit argument
PLANE_SPECS = [[("K", 64), ("V", 32), ("QSA-side", 16)], [("K2", 128)]]  # differing page_bytes


class Plane:  # what mp_rotate_cache_pages touches: cache[0].shape, .device, .dtype
    def __init__(self, tag, page_bytes):
        self.tag, self.page_bytes = tag, page_bytes
        self.device, self.dtype, self.shape = WORKER_DEV, "int32", (2,)

    def __getitem__(self, i):
        return self


class FakeStream:
    def __init__(self):
        self.queue, self.synchronized, self.sync_error = [], 0, None

    def synchronize(self):
        self.synchronized += 1
        if self.sync_error:
            raise self.sync_error
        while self.queue:
            self.queue.pop(0)()  # FIFO; each op reads shared state at EXECUTION time


def make_world():
    world = {"host": list(range(8)),  # mutable shared-arena bytes
             "gpu": {"indices": None},  # "device" mirror, filled only when uploaded
             "rotated": [], "stream_calls": []}
    stream = FakeStream()

    class _Cuda:
        def current_stream(self, device):
            world["stream_calls"].append(device)
            return stream

    world["t"] = type("FakeTorch", (), {"cuda": _Cuda(),
                                        "empty": staticmethod(lambda *a, **k: object())})()

    class Consumer:
        def recv(self, desc, cuda=False):
            assert cuda and desc["shape"][0] == len(world["host"])
            stream.queue.append(lambda: world["gpu"].update(
                indices=list(world["host"])))  # DELAYED read of the arena
            return world["gpu"]

    ext = type("E", (), {})()
    ext.cache_rotate = lambda cache, order, temp: stream.queue.append(
        lambda: world["rotated"].append(
            (cache.tag, cache.page_bytes, tuple(order["indices"]))))
    world["ext"], world["stream"] = ext, stream

    kv_modules = []
    for s in PLANE_SPECS:
        planes = [Plane(t, p) for t, p in s]
        layer = type("L", (), {"get_tensors": staticmethod(lambda p=planes: list(p))})()
        kv_modules.append(type("M", (), {"tp_cache_lookup": {7: layer}})())
    world["ctx"] = {"inf_consumer": Consumer(), "kv_modules": kv_modules,
                    "device": WORKER_DEV}
    return world


def load_fn(world, legacy=False):
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "mp_rotate_cache_pages")
    if legacy:  # reconstruct the pre-fix body: last stmt is the stream sync
        assert "synchronize" in ast.dump(fn.body[-1]), "sync is no longer the final statement"
        fn.body.pop()
    ns = {"torch": world["t"], "ext": world["ext"], "lru_cache": lru_cache}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<frozen>", "exec"), ns)
    return ns["mp_rotate_cache_pages"]


def test_fixed_drains_arena_before_ack():
    w = make_world()
    load_fn(w)(w["ctx"], 7, {"shape": (8,)})
    assert w["stream"].synchronized == 1 and not w["stream"].queue, \
        "function returned while the pinned-arena upload was still queued"
    assert len(w["rotated"]) == 4 and all(o == tuple(range(8)) for _, _, o in w["rotated"])
    w["host"][:] = [99] * 8  # parent rewrites the arena for the next forward
    w["stream"].synchronize()
    assert all(o == tuple(range(8)) for _, _, o in w["rotated"]), "poison leaked after ack"


def test_legacy_loses_the_arena_race():
    w = make_world()
    load_fn(w, legacy=True)(w["ctx"], 7, {"shape": (8,)})
    assert w["stream"].queue and w["gpu"]["indices"] is None, \
        "fake must not execute uploads eagerly"
    w["host"][:] = [77, 7, 300, 12, 0, 1, 2, 3]  # next forward's input token IDs
    w["stream"].synchronize()  # GPU catches up only after the ack
    assert all(o == tuple(w["host"]) for _, _, o in w["rotated"]), \
        "pre-fix code must observe the poisoned indices (models the GPU write fault)"


def test_planes_device_and_sync_error():
    w = make_world()
    load_fn(w)(w["ctx"], 7, {"shape": (8,)})
    assert w["stream_calls"] == [WORKER_DEV], "sync must name the worker device explicitly"
    assert {t: p for t, p, _ in w["rotated"]} == {"K": 64, "V": 32, "QSA-side": 16, "K2": 128}
    w = make_world()
    w["stream"].sync_error = RuntimeError("hipError")
    try:
        load_fn(w)(w["ctx"], 7, {"shape": (8,)})
    except RuntimeError:
        pass
    else:
        raise AssertionError("synchronize failure must propagate, not slip into the ack")


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__]))
