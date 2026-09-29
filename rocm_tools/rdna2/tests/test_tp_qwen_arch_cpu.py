#!/usr/bin/env python3
"""CPU-only tests for the Qwen3.8-Flash-Next tensor-parallel architecture port
(upstream c9abcd54 / d6943498 / 63b062d / 9d18a7f / 42e035d / 002176e, with the
Engram-RAM TP redesign) plus the staged model_tp_fn.py hunks.

The real exllamav3 package cannot be imported on a CPU-only box (pydantic / the
compiled extension), so the modules under test are loaded from source with the
heavy siblings faked:

    real (from file):  util/misc, util/device_copy, util/tensor,
                       model/model_tp_alloc, loader/safetensors, modules/module,
                       modules/ngram_embedding, modules/ple, modules/qsa_indexer,
                       modules/quant/exl3_lib/ngram_codec
    faked:             exllamav3_ext, model/config, tokenizer/mm_embedding,
                       util/memory (counting check_host_memory), the Linear /
                       RMSNorm children of PLE / QSA (duck-typed)

COLLECTION ISOLATION: every harness module - fake AND real-file-loaded - is
registered under a private per-process prefix (_tp_qwen_arch_cpu_harness_<pid>),
never under the real "exllamav3" dotted name. The sources use relative imports
only, which resolve through the loaded module's __package__, so the whole tree
stays inside the private namespace. This file must leave the real namespace
untouched in EITHER collection order: pytest imports all collected files before
running a single test, and a previous version of this harness shadowed
"exllamav3.modules" at collection time, breaking
tests/test_tp_export_attention.py ("cannot import name 'Linear' from
'exllamav3.modules'"). CollectionIsolationTest below guards the invariant.

Proven properties (the Engram-RAM contract first and foremost):
  * the TP parent NEVER materializes the n-gram table (payload-deferred load:
    only aux parameters are read, the table payload is not);
  * exactly ONE rank (the PLE-layer owner) imports the table; stub ranks import
    nothing and carry no submodule / recurrent registration; PLELayer.tp_import
    fails closed BEFORE any payload read on a missing/invalid global plan, a
    split or replicated placement, or a per-device slice that disagrees with
    the global plan (proven with zero read_range calls and zero consumer recv);
  * RAM stays RAM: the owner reads every row of every shard out of the
    safetensors file (real DiskTensorHandle) into host RAM through the same
    guarded slab path as --ngram_ram; a metadata-based check_host_memory fires
    before ANY owner read (single-shard included); no silent disk downgrade;
    read handles are closed on both the success and the failure path
    (DiskTensorHandle has no destructor), and disk-mode owner handles close on
    unload; the exported mode is recorded verbatim
    (no silent trellis_ram -> trellis_disk downgrade); a table requested in
    disk mode keeps streaming through the importer's OWN handles, so nothing
    goes stale when the producer / parent collection close;
  * an owner-imported table fetches row-identical results to a plain
    single-process --ngram_ram load, and to a disk-mode load;
  * the placeholder module fails closed on forward / prefetch, and an import
    with an invalid mode, or on a rank the plan does not own, fails closed;
  * no table bytes ever pass through the shared-memory producer (sends stay
    aux-sized; a table-sized send would blow the fake arena's cap assertion);
  * the parent's module.unload() leaves the owner's imported table intact;
  * Module.tp_collect / tp_single_owner collapse a whole-on-one-rank module's
    trailing all_reduce to a broadcast, and the TPAllocator treats a module
    max_devices as a HARD cap on top of the user dev_limits (63b062d);
  * QSAIndexer and HyperHead export/import metadata round-trips (9d18a7f /
    c9abcd54);
  * the staged qwen-tp-fn.patch carries the prefetch and vocab-limited argmax
    hunks.

GPU-side behavior (collectives, kernels) is hardware-validated separately;
these tests assert the plumbing, placement and memory-residence contracts only.

Run from the repo root (or anywhere):
    python3 -m pytest -q rocm_tools/rdna2/tests/test_tp_qwen_arch_cpu.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch   # host CPU torch: no CUDA calls anywhere in this file

REPO = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# module-under-test loader: real source files under a PRIVATE package prefix
#
# Every harness module registers under _H ("_tp_qwen_arch_cpu_harness_<pid>"),
# never under the real "exllamav3" dotted name. The sources use relative imports
# exclusively, and relative import resolution goes through the loaded module's
# __package__ (derived from its registered dotted name), so the fakes and the
# real files stay inside _H.*. This file must never shadow or replace any
# "exllamav3*" / "exllamav3_ext" sys.modules entry: pytest collects all test
# files before running any, and an import-time namespace takeover here would
# break the real-package tests collected later in the same process (e.g.
# tests/test_tp_export_attention.py: "cannot import name 'Linear' from
# 'exllamav3.modules'").
# ---------------------------------------------------------------------------

_H = "_tp_qwen_arch_cpu_harness_" + str(os.getpid())
_INSTALLED = []
_MEM_FAKE = None

# Snapshot whatever real-package state exists at collection time of THIS file
# (e.g. tests/test_tp_export_attention.py collected first imports the real
# exllamav3 in the same process). The isolation test proves this file never
# replaced, removed or shadowed any of it.
_REAL_NS_AT_IMPORT = {k: sys.modules.get(k) for k in list(sys.modules)
                      if k == "exllamav3" or k.startswith("exllamav3.")}


def _install(name, mod):
    assert name == _H or name.startswith(_H + "."), \
        f"harness module escapes the private namespace: {name}"
    assert getattr(mod, "__name__", name) == name, \
        f"harness module identity mismatch for {name}"
    sys.modules[name] = mod
    _INSTALLED.append(name)
    return mod


def _mk_pkg(name, path = None):
    mod = _install(name, types.ModuleType(name))
    mod.__path__ = path or []
    return mod


def _load_real(dotted, relpath):
    parent, _, leaf = dotted.rpartition(".")
    spec = importlib.util.spec_from_file_location(dotted, REPO / relpath)
    mod = _install(dotted, importlib.util.module_from_spec(spec))
    if parent:
        setattr(sys.modules[parent], leaf, mod)
    spec.loader.exec_module(mod)
    return mod


class _FakeExt:
    """Stands in for exllamav3_ext: none of the tested paths may call it."""
    def __getattr__(self, item):
        raise AssertionError(f"exllamav3_ext.{item} must not be called in CPU tests")


class FakeConfig:
    class _IP:
        def __init__(self, stream):
            self.ngram_stream_from_disk = stream

    def __init__(self, stc, stream_from_disk = True):
        self.stc = stc
        self.infer_params = self._IP(stream_from_disk)


class NullConfig(FakeConfig):
    def __init__(self):
        super().__init__(stc = None)


class FakeLinear:
    """Duck-typed loaded Linear: the tp_export/tp_import round-trip carries the key."""

    def __init__(self, config = None, key = None, *args, **kwargs):
        self.key = key
        self.device = None
        self.kwargs = kwargs

    def load(self, device, **kwargs):
        self.device = device

    def unload(self):
        self.device = None

    def optimizer_targets(self):
        return []

    def storage_size(self):
        return 65536

    def weights_numel(self):
        return 32768

    def tp_export(self, plan, producer):
        assert self.device is not None, "export before load"
        return {"cls": FakeLinear, "kwargs": {"key": self.key}}

    @staticmethod
    def tp_import(local_context, exported, plan):
        m = FakeLinear(key = exported["kwargs"]["key"])
        m.device = local_context["device"]
        return m

    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        return FakeLinear.tp_import(local_context, exported, plan)


class FakeRMSNorm(FakeLinear):
    def tp_export(self, plan, producer):
        return {"cls": FakeRMSNorm, "kwargs": {"key": self.key}}

    @staticmethod
    def tp_import(local_context, exported, plan):
        m = FakeRMSNorm(key = exported["kwargs"]["key"])
        m.device = local_context["device"]
        return m


def _install_stack():
    global _MEM_FAKE
    _mk_pkg(_H)
    ext = _mk_pkg(f"{_H}.ext")
    ext.exllamav3_ext = _FakeExt()
    # the loaded sources use both "import ..ext.exllamav3_ext as ext" (leaf-module
    # dict lookup) and "from ..ext import exllamav3_ext" (package-attribute /
    # submodule-import path): make every leaf registered by name also an
    # attribute of its parent package, keeping both spellings inside _H.
    util = _mk_pkg(f"{_H}.util")
    util.Timer = _load_real(f"{_H}.util.misc", "exllamav3/util/misc.py").Timer
    _load_real(f"{_H}.util.device_copy", "exllamav3/util/device_copy.py")
    _load_real(f"{_H}.util.tensor", "exllamav3/util/tensor.py")
    _MEM_FAKE = _install(f"{_H}.util.memory", types.ModuleType(f"{_H}.util.memory"))
    _MEM_FAKE.checks = []
    _MEM_FAKE.check_host_memory = lambda nbytes, what: _MEM_FAKE.checks.append((nbytes, what))
    _mk_pkg(f"{_H}.model")
    cfg = _install(f"{_H}.model.config", types.ModuleType(f"{_H}.model.config"))
    cfg.Config = FakeConfig
    cfg.NullConfig = NullConfig
    cfg.no_default = object()
    _load_real(f"{_H}.model.model_tp_alloc", "exllamav3/model/model_tp_alloc.py")
    _mk_pkg(f"{_H}.tokenizer")
    mme = _install(f"{_H}.tokenizer.mm_embedding",
                   types.ModuleType(f"{_H}.tokenizer.mm_embedding"))
    mme.FIRST_MM_EMBEDDING_INDEX = 1 << 40
    _mk_pkg(f"{_H}.loader")
    _load_real(f"{_H}.loader.safetensors", "exllamav3/loader/safetensors.py")
    mods = _mk_pkg(f"{_H}.modules")
    mods.Module = _load_real(f"{_H}.modules.module", "exllamav3/modules/module.py").Module
    lin = _install(f"{_H}.modules.linear", types.ModuleType(f"{_H}.modules.linear"))
    lin.Linear = FakeLinear
    rms = _install(f"{_H}.modules.rmsnorm", types.ModuleType(f"{_H}.modules.rmsnorm"))
    rms.RMSNorm = FakeRMSNorm
    _mk_pkg(f"{_H}.modules.quant")
    _mk_pkg(f"{_H}.modules.quant.exl3_lib")
    _load_real(f"{_H}.modules.quant.exl3_lib.ngram_codec",
               "exllamav3/modules/quant/exl3_lib/ngram_codec.py")
    _load_real(f"{_H}.modules.ngram_embedding", "exllamav3/modules/ngram_embedding.py")
    _load_real(f"{_H}.modules.ple", "exllamav3/modules/ple.py")
    _load_real(f"{_H}.modules.qsa_indexer", "exllamav3/modules/qsa_indexer.py")
    # `from . import Module` (ngram / faked linear+rmsnorm siblings) resolves via
    # the package object's attributes, not __path__: keep the namespace sealed so
    # nothing here can ever pull in a real-package file through a stray import.
    for name in _INSTALLED:
        assert name == _H or name.startswith(_H + "."), name


_install_stack()

# Bind the loaded classes directly from the private harness namespace: a plain
# "from exllamav3.... import X" here would hit the REAL package if it had been
# imported by another test module collected earlier in the same process, and
# the whole suite is designed around the harness classes.
_TPALLOC = sys.modules[f"{_H}.model.model_tp_alloc"]
TPAllocation, TPAllocator = _TPALLOC.TPAllocation, _TPALLOC.TPAllocator
Module = sys.modules[f"{_H}.modules.module"].Module
NGramEmbedding = sys.modules[f"{_H}.modules.ngram_embedding"].NGramEmbedding
_ple = sys.modules[f"{_H}.modules.ple"]
PLELayer, PLELayerState = _ple.PLELayer, _ple.PLELayerState
QSAIndexer = sys.modules[f"{_H}.modules.qsa_indexer"].QSAIndexer
ROW_DIM = sys.modules[f"{_H}.modules.quant.exl3_lib.ngram_codec"].ROW_DIM
DiskTensorHandle = sys.modules[f"{_H}.loader.safetensors"].DiskTensorHandle

CPU = torch.device("cpu")
PLE_KEY = "model.layers.0.ple"
NGRAM_KEY = f"{PLE_KEY}.ple_embedding.ngram_embedding"

# ---------------------------------------------------------------------------
# fakes: tensor collection over real table files, producer/consumer, backend
# ---------------------------------------------------------------------------

ROWS_PER_SHARD = 64
N_SHARDS = 2
WORDS = 21   # words_per_row(K=2)


class TableFixture:
    """A small trellis layout (default: 2 shards) written as real bytes on disk, plus
    the aux parameter tensors a converted ngram_embedding.safetensors carries."""

    def __init__(self, tmp, n_shards = N_SHARDS):
        self.n_shards = n_shards
        self.contents = []
        self.table_files = {}   # key -> DiskTensorHandle over a real file
        for i in range(n_shards):
            data = torch.arange(ROWS_PER_SHARD * WORDS, dtype = torch.int32) \
                .remainder(997).to(torch.int16).view(ROWS_PER_SHARD, WORDS)
            fn = os.path.join(tmp, f"ngram_shard_{i}.bin")
            data.numpy().tofile(fn)
            key = f"{NGRAM_KEY}.shard_{i}.trellis"
            self.table_files[key] = DiskTensorHandle(
                key = key, filename = fn, abs_offset = 0,
                shape = [ROWS_PER_SHARD, WORDS], dtype = torch.int16)
            self.contents.append(data)
        self.full = torch.cat(self.contents, dim = 0)
        self.num_rows = ROWS_PER_SHARD * n_shards
        self.aux = {
            f"{NGRAM_KEY}.head_offsets": torch.arange(0, 8, dtype = torch.int64) * 100,
            f"{NGRAM_KEY}.head_vocab_sizes": torch.full((8,), 50, dtype = torch.int64),
            f"{NGRAM_KEY}.layer_multipliers":
                torch.tensor([3, 5, 7, 11, 13, 17, 19, 23], dtype = torch.int64),
            f"{NGRAM_KEY}.head_bias": torch.zeros(8, ROW_DIM, dtype = torch.float16),
        }


class FakeSTC:
    """SafetensorsCollection stand-in over in-memory tensors + real table files.

    loaded_keys records every payload read: the deferred TP parent must list
    ONLY aux keys (the table payload never materializes in the parent)."""

    def __init__(self, fixture):
        self.fixture = fixture
        self.tensors = dict(fixture.aux)
        self.tensors[f"{PLE_KEY}.conv1d.weight"] = \
            torch.zeros(256, 1, 3, dtype = torch.float16)
        self.loaded_keys = []
        self.closed = False
        self.tensor_file_map = {}

    def has_tensor(self, key):
        return key in self.tensors or key in self.fixture.table_files

    def get_tensor_meta(self, key):
        src = self.fixture.table_files.get(key)
        shape = list(src.shape) if src is not None else list(self.tensors[key].shape)
        return {key: {"shape": shape}}

    def get_tensor_sizes(self, key):
        src = self.fixture.table_files.get(key)
        if src is not None:
            n = math.prod(src.shape) * torch.empty(0, dtype = src.dtype).element_size()
            return [n]
        if key in self.tensors:
            t = self.tensors[key]
            return [t.numel() * t.element_size()]
        return []

    def get_tensor(self, key, device = None, allow_bf16 = False, no_defer = False,
                   optional = False, float2half = False):
        self.loaded_keys.append(key)
        if key in self.fixture.table_files:      # a real single-process RAM load
            return self.fixture.table_files[key].read_range(0, ROWS_PER_SHARD)
        t = self.tensors.get(key)
        if t is None and not optional:
            raise ValueError(f"tensor {key} not in fake collection")
        return t.clone() if t is not None else None

    def get_tensor_handle(self, key, optional = False):
        return self.fixture.table_files[key]

    def release_file(self, filename):
        pass

    def find_stc(self, key):
        return self

    def close(self):
        self.closed = True


class FakeProducer:
    """Arena-shaped producer: records every send's byte size and serializes
    into one scratch buffer the FakeConsumer 'maps'. close() wipes it, proving
    imports hold no views into the shared arena."""

    LIMIT = 1 << 16   # anything table-shaped would blow this cap immediately

    def __init__(self):
        self.arena = bytearray(1 << 24)
        self.next_offset = 0
        self.sends = []           # (nbytes, shape)
        self.closed = False

    def send(self, tensor, cache_id = None):
        if tensor is None:
            return {"method": "none_tensor"}
        nbytes = tensor.element_size() * tensor.numel()
        assert nbytes <= self.LIMIT, \
            f"tp_export pushed a {nbytes}-byte tensor through the producer: " \
            "the n-gram table payload must never travel this way"
        self.sends.append((nbytes, tuple(tensor.shape)))
        off = self.next_offset
        src = tensor.cpu().contiguous().view(torch.uint8).flatten().numpy().tobytes()
        self.arena[off: off + nbytes] = src
        self.next_offset += (nbytes + 127) // 128 * 128
        return {"method": "buffer", "offset": off, "nbytes": nbytes,
                "dtype": str(tensor.dtype), "shape": tuple(tensor.shape),
                "cache_id": None}

    def clear(self):
        self.next_offset = 0

    def close(self):
        self.arena = bytearray(len(self.arena))   # unlink + wipe
        self.closed = True


class FakeConsumer:
    def __init__(self, producer):
        self.producer = producer
        self.recvs = 0

    def recv(self, imp, cuda = False, slice_dim = None, first = None, last = None):
        self.recvs += 1
        if imp["method"] == "none_tensor":
            return None
        dtype = getattr(torch, imp["dtype"].split(".")[1])
        n = imp["nbytes"]
        raw = torch.frombuffer(bytes(self.producer.arena[imp["offset"]: imp["offset"] + n]),
                               dtype = torch.uint8)
        t = raw.view(dtype).view(*imp["shape"])
        if slice_dim is not None:
            t = t.narrow(slice_dim, first, last - first)
        return t.clone(memory_format = torch.contiguous_format)


class FakeBackend:
    def __init__(self):
        self.collects = []   # ("broadcast", src) / ("all_reduce", contribution)

    def broadcast(self, tensor, src_device):
        self.collects.append(("broadcast", src_device))

    def all_reduce(self, tensor, contribution = True):
        self.collects.append(("all_reduce", contribution))


def make_ngram(cfg, stream_from_disk):
    return NGramEmbedding(
        config = cfg,
        key = NGRAM_KEY,
        ngram_size = 3,
        heads_per_ngram = 4,
        ple_embed_dim = 8 * ROW_DIM,
        eos_token_id = 151_643,
        stream_from_disk = stream_from_disk,
    )


def make_ple(cfg):
    return PLELayer(
        config = cfg, key = PLE_KEY, layer_idx = -1, hidden_size = 128, hc_mult = 2,
        ple_embed_dim = 8 * ROW_DIM, ngram_size = 3, heads_per_ngram = 4,
        eos_token_id = 151_643, conv_kernel_size = 3, rms_norm_eps = 1e-6,
        stream_from_disk = False,
    )


def owner_ctx(plan, consumer):
    return {"device": CPU, "consumer": consumer, "plan": plan, "active_devices": [0, 1]}


FULL_PLAN = [{PLE_KEY: (0, 1, "layer")}, {PLE_KEY: (1, 1, "layer")}]
UIDS = torch.tensor([0, ROWS_PER_SHARD - 1, ROWS_PER_SHARD, ROWS_PER_SHARD * N_SHARDS - 1])


class NgramTPExportImportTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix = "tp_ngram_")

    def _parent(self, stream_from_disk, defer):
        fixture = TableFixture(self.tmp)
        cfg = FakeConfig(FakeSTC(fixture))
        ng = make_ngram(cfg, stream_from_disk)
        ng.load(CPU, **({"tp_parent_defer": True} if defer else {}))
        return fixture, cfg, ng

    def test_parent_defers_ram_payload(self):
        """TP-parent RAM load: metadata + aux only. This is the 31-GiB-per-parent
        clone the Engram-RAM requirement forbids; it must not happen."""
        fixture, cfg, ng = self._parent(stream_from_disk = False, defer = True)
        self.assertTrue(ng._tp_deferred)
        self.assertIsNone(ng.tables)
        self.assertIsNone(ng.handles)
        self.assertEqual(ng.mode, "trellis_ram")        # requested mode RECORDED, not coerced
        self.assertEqual(set(cfg.stc.loaded_keys), set(fixture.aux),
                         "deferred parent must read aux params and NOTHING else")
        self.assertIsNotNone(ng.head_offsets)
        self.assertEqual(ng.num_rows, fixture.num_rows)
        self.assertEqual(ng.rows_per_shard, ROWS_PER_SHARD)
        # the placeholder fails closed on anything that needs the table:
        with self.assertRaises(RuntimeError):
            ng.forward(torch.zeros((1, 2 + ng.context_len), dtype = torch.long), {})
        with self.assertRaises(RuntimeError):
            ng.prefetch(torch.zeros((1, 2 + ng.context_len), dtype = torch.long))
        # ... but export from metadata works fine:
        exported = ng.tp_export(plan = {}, producer = FakeProducer())
        self.assertEqual(len(exported["handles"]), N_SHARDS)

    def test_export_is_metadata_only_and_records_mode(self):
        for stream, want in ((False, "trellis_ram"), (True, "trellis_disk")):
            fixture, cfg, ng = self._parent(stream_from_disk = stream, defer = not stream)
            prod = FakeProducer()
            exported = ng.tp_export(plan = {}, producer = prod)
            self.assertEqual(exported["mode"], want,
                             "tp_export must record RAM vs disk exactly as requested")
            keys, fns, offs, shapes, dts = zip(*exported["handles"])
            self.assertEqual(list(keys), [f"{NGRAM_KEY}.shard_{i}.trellis" for i in range(2)])
            self.assertTrue(all(f.startswith(self.tmp) for f in fns),
                            "export carries the source file locations")
            self.assertTrue(all(dt == "torch.int16" for dt in dts))
            # no payload ever rides the producer: every send stays aux-sized
            self.assertLess(max(n for n, _ in prod.sends), 1 << 12)

    def test_owner_ram_import_reads_table_once_from_files(self):
        fixture, cfg, ng = self._parent(stream_from_disk = False, defer = True)
        prod = FakeProducer()
        exported = ng.tp_export(plan = {}, producer = prod)
        cons = FakeConsumer(prod)
        _MEM_FAKE.checks.clear()
        reads = []
        opened = []
        orig = DiskTensorHandle.read_range
        DiskTensorHandle.read_range = lambda self, a, b: (
            reads.append((self.key, a, b)), opened.append(self), orig(self, a, b))[2]
        try:
            m = NGramEmbedding.tp_import(owner_ctx(FULL_PLAN, cons), exported, FULL_PLAN[0])
        finally:
            DiskTensorHandle.read_range = orig
        self.assertEqual(m.mode, "trellis_ram")
        self.assertIsNotNone(m.tables)
        self.assertIsNone(m.handles, "a RAM owner must not keep streaming handles")
        self.assertEqual(m.tables[0].shape, (fixture.num_rows, WORDS))
        self.assertEqual(m.rows_per_shard, fixture.num_rows)     # single-store slab
        # every shard read fully, exactly once:
        self.assertEqual(sorted(reads), sorted(
            [(f"{NGRAM_KEY}.shard_{i}.trellis", 0, ROWS_PER_SHARD) for i in range(N_SHARDS)]))
        # host-memory guards: the metadata-based pre-read check (whole table size)
        # fired BEFORE any read_range, and the sharded-slab guard of the normal
        # --ngram_ram path is still in place:
        self.assertGreaterEqual(len(_MEM_FAKE.checks), 2)
        self.assertEqual(_MEM_FAKE.checks[0][0],
                         fixture.num_rows * WORDS * 2)
        self.assertIn("TP owner rank", _MEM_FAKE.checks[0][1])
        self.assertIn("held in RAM (--ngram_ram)", _MEM_FAKE.checks[1][1])
        # read_range keeps a cached fd per handle and DiskTensorHandle has no
        # destructor: the import must have closed every handle it opened:
        self.assertEqual(len(opened), N_SHARDS)
        self.assertTrue(all(h.fd is None for h in opened),
                        "RAM tp_import leaked an open file handle")
        # byte-exact table contents:
        self.assertTrue(torch.equal(m.tables[0], fixture.full))
        # ... and row-identical to a plain single-process --ngram_ram load:
        ref_cfg = FakeConfig(FakeSTC(TableFixture(self.tmp)))
        ref = make_ngram(ref_cfg, stream_from_disk = False)
        ref.load(CPU)           # no defer flag: the LS path, unchanged
        self.assertTrue(torch.equal(m.fetch_rows(UIDS), ref.fetch_rows(UIDS)))

    def test_imports_fail_closed_and_no_stale_handles(self):
        fixture, cfg, ng = self._parent(stream_from_disk = False, defer = True)
        prod = FakeProducer()
        exported = ng.tp_export(plan = {}, producer = prod)
        cons = FakeConsumer(prod)
        m = NGramEmbedding.tp_import(owner_ctx(FULL_PLAN, cons), exported, FULL_PLAN[0])
        bad = dict(exported)
        bad["mode"] = "trellis_maybe"
        with self.assertRaises(AssertionError):
            NGramEmbedding.tp_import(owner_ctx(FULL_PLAN, cons), bad, FULL_PLAN[0])
        with self.assertRaises(AssertionError):
            # a rank the plan does not own may never import a second table copy
            NGramEmbedding.tp_import(
                owner_ctx(FULL_PLAN, cons), exported, FULL_PLAN[1])
        # end of loading: the producer arena is wiped and the parent collection
        # closed. The owner's imported state must be independent of both.
        prod.close()
        cfg.stc.close()
        self.assertFalse(any(prod.arena[: 4096]), "sanity: the arena really was wiped")
        self.assertTrue(torch.equal(m.tables[0], fixture.full))
        self.assertTrue(torch.equal(m.head_offsets,
                                    fixture.aux[f"{NGRAM_KEY}.head_offsets"]))
        rows = m.fetch_rows(UIDS)
        self.assertEqual(rows.shape, (4, ROW_DIM))
        # after unload the placeholder cannot fake an export either:
        ng.unload()
        self.assertIsNone(ng.mode)
        self.assertIsNone(ng._table_keys)
        with self.assertRaises(AssertionError):
            ng.tp_export(plan = {}, producer = FakeProducer())

    def test_disk_mode_import_streams_from_own_handles(self):
        """A caller that requested disk streaming keeps disk streaming (upstream
        semantics): the importer builds its own handles, materializes nothing,
        and fetches row-identical results through preads that outlive the parent."""
        fixture, cfg, ng = self._parent(stream_from_disk = True, defer = False)
        self.assertEqual(ng.mode, "trellis_disk")
        self.assertIsNone(ng.tables)
        prod = FakeProducer()
        exported = ng.tp_export(plan = {}, producer = prod)
        self.assertEqual(exported["mode"], "trellis_disk")
        m = NGramEmbedding.tp_import(owner_ctx(FULL_PLAN, FakeConsumer(prod)),
                                     exported, FULL_PLAN[0])
        self.assertIsNone(m.tables, "a disk import must not materialize the table")
        self.assertEqual(len(m.handles), N_SHARDS)
        prod.close()
        cfg.stc.close()
        ref_cfg = FakeConfig(FakeSTC(TableFixture(self.tmp)))
        ref = make_ngram(ref_cfg, stream_from_disk = False)
        ref.load(CPU)
        self.assertTrue(torch.equal(m.fetch_rows(UIDS), ref.fetch_rows(UIDS)),
                        "disk-mode results must match the RAM reference")

    def test_disk_owner_handles_closed_on_unload(self):
        """read_rows caches an fd per streaming handle and DiskTensorHandle has
        no destructor: the disk-mode owner's unload() must close them (and the
        parent stc re-close stays idempotent)."""
        fixture, cfg, ng = self._parent(stream_from_disk = True, defer = False)
        prod = FakeProducer()
        exported = ng.tp_export(plan = {}, producer = prod)
        m = NGramEmbedding.tp_import(owner_ctx(FULL_PLAN, FakeConsumer(prod)),
                                     exported, FULL_PLAN[0])
        m.fetch_rows(UIDS)                       # touches both shards -> fds open
        hs = list(m.handles)
        self.assertTrue(all(h.fd is not None for h in hs))
        m.unload()
        self.assertTrue(all(h.fd is None for h in hs),
                        "disk-mode unload leaked streaming fds")
        self.assertIsNone(m.handles)

    def test_ram_owner_guard_fires_before_any_read(self):
        """A failing check_host_memory (metadata-based) aborts the owner import
        with ZERO payload reads - no partial slab, no disk fallback."""
        fixture, cfg, ng = self._parent(stream_from_disk = False, defer = True)
        prod = FakeProducer()
        exported = ng.tp_export(plan = {}, producer = prod)
        events = []
        real_check = _MEM_FAKE.check_host_memory
        def guard(nbytes, what):
            events.append(("check", nbytes, what))
            raise RuntimeError("host memory guard: would leave less than the reserve")
        orig = DiskTensorHandle.read_range
        _MEM_FAKE.check_host_memory = guard
        DiskTensorHandle.read_range = \
            lambda self, a, b: (events.append(("read", self.key)), orig(self, a, b))[1]
        try:
            with self.assertRaises(RuntimeError):
                NGramEmbedding.tp_import(owner_ctx(FULL_PLAN, FakeConsumer(prod)),
                                         exported, FULL_PLAN[0])
        finally:
            DiskTensorHandle.read_range = orig
            _MEM_FAKE.check_host_memory = real_check
        self.assertEqual([e[0] for e in events], ["check"],
                         "owner import must not read_range before the memory guard passes")
        self.assertEqual(events[0][1], fixture.num_rows * WORDS * 2)

    def test_ram_owner_read_error_closes_handles(self):
        """A failing disk read mid-import must still close the handles whose
        shards were already read (try/finally, not just the success path)."""
        fixture, cfg, ng = self._parent(stream_from_disk = False, defer = True)
        prod = FakeProducer()
        exported = ng.tp_export(plan = {}, producer = prod)
        opened = []
        calls = {"n": 0}
        orig = DiskTensorHandle.read_range
        def flaky(self, a, b):
            opened.append(self)
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("simulated read failure on shard 1")
            return orig(self, a, b)
        DiskTensorHandle.read_range = flaky
        try:
            with self.assertRaises(OSError):
                NGramEmbedding.tp_import(owner_ctx(FULL_PLAN, FakeConsumer(prod)),
                                         exported, FULL_PLAN[0])
        finally:
            DiskTensorHandle.read_range = orig
        self.assertEqual(len(opened), 2)
        self.assertTrue(all(h.fd is None for h in opened),
                        "the first shard's fd must be closed even when a later read fails")

    def test_ram_owner_single_shard_guarded_before_read(self):
        """The pre-read guard covers SINGLE-shard tables too (the normal loader
        path has no guard there); check precedes read, table lands in RAM, every
        handle closed afterwards."""
        fixture = TableFixture(self.tmp, n_shards = 1)
        cfg = FakeConfig(FakeSTC(fixture))
        ng = make_ngram(cfg, stream_from_disk = False)
        ng.load(CPU, tp_parent_defer = True)
        prod = FakeProducer()
        exported = ng.tp_export(plan = {}, producer = prod)
        self.assertEqual(len(exported["handles"]), 1)
        events = []
        real_check = _MEM_FAKE.check_host_memory
        orig = DiskTensorHandle.read_range
        _MEM_FAKE.check_host_memory = lambda n, w: events.append(("check", n, w))
        DiskTensorHandle.read_range = \
            lambda self, a, b: (events.append(("read", self, a, b)), orig(self, a, b))[1]
        try:
            m = NGramEmbedding.tp_import(owner_ctx(FULL_PLAN, FakeConsumer(prod)),
                                         exported, FULL_PLAN[0])
        finally:
            DiskTensorHandle.read_range = orig
            _MEM_FAKE.check_host_memory = real_check
        self.assertEqual([e[0] for e in events], ["check", "read"])
        self.assertEqual(events[0][1], fixture.num_rows * WORDS * 2)
        _, h, a, b = events[1]
        self.assertEqual((a, b), (0, ROWS_PER_SHARD))
        self.assertEqual(m.tables[0].shape, (ROWS_PER_SHARD, WORDS))
        self.assertIsNone(h.fd, "single-shard import must close its read handle")

    def test_normal_ls_load_path_unchanged(self):
        """No defer flag -> exactly the pre-TP behavior: the module loads the
        table into RAM itself with no deferral bookkeeping."""
        fixture = TableFixture(self.tmp)
        cfg = FakeConfig(FakeSTC(fixture))
        ng = make_ngram(cfg, stream_from_disk = False)
        ng.load(CPU)
        self.assertFalse(ng._tp_deferred)
        self.assertEqual(ng.mode, "trellis_ram")
        self.assertIsNotNone(ng.tables)
        self.assertTrue(ng._tp_deferred is False)


class PleTPPlacementTest(unittest.TestCase):
    """PLELayer owner/stub placement (42e035d) with the RAM-preserving
    n-gram import (this fork's redesign) behind it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix = "tp_ple_")

    def _loaded_parent(self):
        fixture = TableFixture(self.tmp)
        cfg = FakeConfig(FakeSTC(fixture))
        ple = make_ple(cfg)
        ple.load(CPU, tp_parent_defer = True)
        # the TP contract for recurrent models: cache states exist before export
        state = PLELayerState(ple, max_batch_size = 2, max_history = 4, cache_id = 999)
        ple.recurrent_layers.append(state)
        return fixture, cfg, ple

    def test_allocation_is_single_device(self):
        fixture, cfg, ple = self._loaded_parent()
        tpa, = ple.make_tp_allocation({})
        self.assertEqual(tpa.max_devices, 1)
        self.assertEqual(tpa.channels_to_split, 1)
        self.assertEqual(tpa.channel_unit, "layer")
        self.assertGreater(tpa.storage_to_split, 0)

    def test_owner_and_stub_single_table_ownership(self):
        fixture, cfg, ple = self._loaded_parent()
        prod = FakeProducer()
        exported = ple.tp_export(plan = {}, producer = prod)
        cons_owner, cons_stub = FakeConsumer(prod), FakeConsumer(prod)
        reads = []
        orig = DiskTensorHandle.read_range
        DiskTensorHandle.read_range = \
            lambda self, a, b: (reads.append(self.key), orig(self, a, b))[1]
        try:
            with patch("torch.cuda.synchronize", lambda: None):
                owner = PLELayer.tp_import(owner_ctx(FULL_PLAN, cons_owner), exported, FULL_PLAN[0])
                stub = PLELayer.tp_import(owner_ctx(FULL_PLAN, cons_stub), exported, FULL_PLAN[1])
        finally:
            DiskTensorHandle.read_range = orig
        # one table, read once, on the owner only:
        self.assertEqual(sorted(reads), sorted(f"{NGRAM_KEY}.shard_{i}.trellis"
                                               for i in range(N_SHARDS)))
        self.assertEqual(cons_stub.recvs, 0, "the stub must recv no submodule payload")
        self.assertFalse(owner.stub)
        self.assertTrue(stub.stub)
        self.assertIsNotNone(owner.ple_embedding.tables)
        self.assertTrue(torch.equal(owner.ple_embedding.tables[0], fixture.full))
        self.assertIsNone(stub.ple_embedding)
        self.assertEqual(stub.modules, [], "stub carries no submodules")
        self.assertEqual(stub.recurrent_layers, [])
        self.assertNotIn("recurrent_cache", stub.caps)   # never registered as a cache module
        self.assertNotIn("prefetch_ids", stub.caps)      # no prefetch without a table
        self.assertEqual(owner.tp_owner, 0)
        self.assertEqual(stub.tp_owner, 0)
        self.assertEqual(len(owner.recurrent_layers), 1)
        self.assertIn(999, owner.tp_recurrent_lookup)
        self.assertEqual(owner.modules[0].mode, "trellis_ram")
        # collective contract: owner broadcasts its full stack, stub receives it
        backend = FakeBackend()
        x = torch.ones((1, 2, 2, 128))
        out = stub.forward(x, {"backend": backend})
        self.assertEqual(backend.collects, [("broadcast", 0)])
        self.assertTrue(torch.equal(out, x),
                        "stub returns the stack it received from the owner")

    def test_parent_unload_does_not_touch_owner_table(self):
        fixture, cfg, ple = self._loaded_parent()
        prod = FakeProducer()
        exported = ple.tp_export(plan = {}, producer = prod)
        with patch("torch.cuda.synchronize", lambda: None):
            owner = PLELayer.tp_import(owner_ctx(FULL_PLAN, FakeConsumer(prod)),
                                       exported, FULL_PLAN[0])
        table = owner.ple_embedding.tables[0]
        ple.unload()            # the parent placeholder and its (empty) state go away
        self.assertIsNone(ple.ple_embedding.tables)
        self.assertTrue(torch.equal(owner.ple_embedding.tables[0], table))
        self.assertTrue(torch.equal(table, fixture.full))

    def test_stub_import_rejects_split_plans(self):
        fixture, cfg, ple = self._loaded_parent()
        prod = FakeProducer()
        exported = ple.tp_export(plan = {}, producer = prod)
        with self.assertRaises(AssertionError):
            PLELayer.tp_import(owner_ctx(FULL_PLAN, FakeConsumer(prod)), exported,
                               {PLE_KEY: (0, 1, "heads")})
        with self.assertRaises(AssertionError):
            PLELayer.tp_import(owner_ctx(FULL_PLAN, FakeConsumer(prod)), exported,
                               {PLE_KEY: (0, 2, "layer")})

    # -- global-plan fail-closed gates: no payload read, no producer recv --------

    def _exported_parent(self):
        fixture = TableFixture(self.tmp)
        cfg = FakeConfig(FakeSTC(fixture))
        ple = make_ple(cfg)
        ple.load(CPU, tp_parent_defer = True)
        ple.recurrent_layers.append(
            PLELayerState(ple, max_batch_size = 2, max_history = 4, cache_id = 999))
        prod = FakeProducer()
        return fixture, prod, ple.tp_export(plan = {}, producer = prod)

    def _assert_gate_rejects(self, ctx, plan, exported, cons):
        reads = []
        orig = DiskTensorHandle.read_range
        DiskTensorHandle.read_range = \
            lambda self, a, b: (reads.append(self.key), orig(self, a, b))[1]
        try:
            with self.assertRaises(AssertionError):
                PLELayer.tp_import(ctx, exported, plan)
        finally:
            DiskTensorHandle.read_range = orig
        self.assertEqual(reads, [], "a rejected import may not read the table payload")
        self.assertEqual(cons.recvs, 0, "a rejected import may not consume producer payloads")

    def test_missing_global_plan_rejected(self):
        fixture, prod, exported = self._exported_parent()
        cons = FakeConsumer(prod)
        # the current tp_single_owner path returns None here; silently proceeding
        # would load a table on a rank whose placement cannot be verified at all
        self._assert_gate_rejects(
            {"device": CPU, "consumer": cons}, FULL_PLAN[0], exported, cons)

    def test_replicated_owners_rejected(self):
        fixture, prod, exported = self._exported_parent()
        cons = FakeConsumer(prod)
        gplan = [{PLE_KEY: (0, 1, "layer")}, {PLE_KEY: (0, 1, "layer")}]
        ctx = {"device": CPU, "consumer": cons, "plan": gplan, "active_devices": [0, 1]}
        self._assert_gate_rejects(ctx, gplan[0], exported, cons)

    def test_no_owner_rejected(self):
        fixture, prod, exported = self._exported_parent()
        cons = FakeConsumer(prod)
        gplan = [{PLE_KEY: (1, 1, "layer")}, {PLE_KEY: (2, 2, "layer")}]
        ctx = {"device": CPU, "consumer": cons, "plan": gplan, "active_devices": [0, 1]}
        self._assert_gate_rejects(ctx, gplan[0], exported, cons)

    def test_malformed_other_rank_plan_rejected(self):
        fixture, prod, exported = self._exported_parent()
        cons = FakeConsumer(prod)
        gplan = [{PLE_KEY: (0, 1, "layer")}, {PLE_KEY: (1, 3, "heads")}]
        ctx = {"device": CPU, "consumer": cons, "plan": gplan, "active_devices": [0, 1]}
        self._assert_gate_rejects(ctx, gplan[0], exported, cons)

    def test_slice_disagreement_with_global_plan_rejected(self):
        fixture, prod, exported = self._exported_parent()
        cons = FakeConsumer(prod)
        # rank 1 claims the owner slice locally, but the global plan puts the layer
        # on rank 0: accepting it would clone the table
        ctx = {"device": 1, "consumer": cons, "plan": FULL_PLAN, "active_devices": [0, 1]}
        self._assert_gate_rejects(ctx, FULL_PLAN[0], exported, cons)

    def test_owner_rank_stubbed_by_local_plan_rejected(self):
        fixture, prod, exported = self._exported_parent()
        cons = FakeConsumer(prod)
        # inverse: the global owner rank is handed an empty (stub) slice - the
        # table would end up loaded by NOBODY and the broadcast would hang
        ctx = {"device": 0, "consumer": cons, "plan": FULL_PLAN, "active_devices": [0, 1]}
        self._assert_gate_rejects(ctx, FULL_PLAN[1], exported, cons)


class ModuleCollectivesTest(unittest.TestCase):
    """d6943498: all_reduce collapses to a byte-exact broadcast for modules the
    plan placed whole on one rank."""

    class _M(Module):
        def optimizer_targets(self):
            return []

        def forward(self, x, params, out_dtype = None):
            return x

    def _ctx(self, plan, devices):
        return {"plan": plan, "active_devices": devices}

    def test_tp_single_owner(self):
        M = self._M
        whole = [{"a": (0, 4, "heads")}, {"a": (4, 4, "heads")}]
        self.assertEqual(M.tp_single_owner(self._ctx(whole, [0, 1]), "a"), 0)
        split = [{"a": (0, 2, "heads")}, {"a": (2, 4, "heads")}]
        self.assertIsNone(M.tp_single_owner(self._ctx(split, [0, 1]), "a"))
        two_keys = [{"a": (0, 4, "h"), "b": (0, 4, "h")},
                    {"a": (4, 4, "h"), "b": (4, 4, "h")}]
        self.assertEqual(M.tp_single_owner(self._ctx(two_keys, [0, 1]), "a", "b"), 0)
        split_owners = [{"a": (0, 4, "h"), "b": (4, 4, "h")},
                        {"a": (4, 4, "h"), "b": (0, 4, "h")}]
        self.assertIsNone(M.tp_single_owner(self._ctx(split_owners, [0, 1]), "a", "b"),
                          "parts owned by different ranks must keep the all_reduce")
        self.assertIsNone(M.tp_single_owner({}, "a"), "no plan -> conservative all_reduce")
        self.assertIsNone(M.tp_single_owner(self._ctx(whole, None), "a"))

    def test_tp_collect_collapses_to_broadcast(self):
        m = self._M(config = None, key = "k", qmap = None)
        self.assertIsNone(m.tp_owner, "class default: no owner -> all_reduce")
        be = FakeBackend()
        m.tp_owner = 1
        m.tp_collect(be, torch.zeros(2), contribution = False)
        self.assertEqual(be.collects, [("broadcast", 1)],
                         "single-owner module broadcasts instead of summing zeros")
        be.collects.clear()
        m.tp_owner = None
        m.tp_reduce = True
        m.tp_collect(be, torch.zeros(2), False)
        self.assertEqual(be.collects[0], ("all_reduce", False))
        m.tp_collect(be, torch.zeros(2))
        self.assertEqual(be.collects[1], ("all_reduce", True))


class AllocatorMaxDevicesTest(unittest.TestCase):
    """63b062d (model_tp_alloc hunk): a module-enforced max_devices is a HARD
    cap, on top of whatever the user set for the component type."""

    def _owners(self, max_devices, dev_limits, whole, mem = (8 << 30, 8 << 30)):
        comps = [TPAllocation(
            key = "attn0", channel_width = 1, channel_unit = "heads",
            storage_to_split = 1 << 30, overhead_to_split = 0,
            channels_to_split = 4, limit_key = "attn", max_devices = max_devices)]
        alloc = TPAllocator(comps, num_tokens = 64, output_num_tokens = 8,
                            dev_limits = dev_limits)
        alloc.initial_split(list(mem))
        plan = alloc.compile_tp_plan()
        owners = [d for d in range(len(plan))
                  if plan[d]["attn0"][1] > plan[d]["attn0"][0]]
        if whole:   # the owner holds the whole module, not a partial band
            self.assertEqual(plan[owners[0]]["attn0"][:2], (0, 4))
        return owners

    def test_hard_cap_beats_dev_limit(self):
        # the pre-fix code took dev_limits["attn"]=2 and SPLIT the layer anyway
        self.assertEqual(len(self._owners(1, {"attn": 2}, whole = True)), 1)

    def test_cap_without_dev_limit(self):
        self.assertEqual(len(self._owners(1, {}, whole = True)), 1)

    def test_no_cap_splits_across_devices(self):
        self.assertEqual(len(self._owners(None, {}, whole = False)), 2)

    def test_dev_limit_still_respected(self):
        self.assertEqual(len(self._owners(None, {"attn": 1}, whole = True)), 1)

    def test_three_devices_min_of_both(self):
        self.assertEqual(len(self._owners(2, {"attn": 3}, whole = False,
                                          mem = (8 << 30,) * 3)), 2)


class QSAIndexerTPTest(unittest.TestCase):
    """9d18a7f: the indexer travels whole with its attention layer."""

    def _idx(self, cfg):
        return QSAIndexer(config = cfg, key = "attn.indexer", hidden_size = 128,
                          n_heads = 16, kv_heads = 1, head_dim = 128,
                          token_budget = 2048, compress_ratio = 4,
                          rms_norm_eps = 1e-6)

    def test_export_import_roundtrip(self):
        cfg = FakeConfig(stc = None)
        idx = self._idx(cfg)
        for m in idx.modules:
            m.load(CPU)
        idx.device = CPU
        prod = FakeProducer()
        exported = idx.tp_export(plan = {}, producer = prod)
        self.assertEqual(exported["kwargs"]["n_heads"], 16)
        self.assertEqual(exported["kwargs"]["kv_heads"], 1)
        self.assertEqual(exported["kwargs"]["token_budget"], 2048)
        self.assertEqual(exported["kwargs"]["compress_ratio"], 4)
        self.assertEqual(exported["kwargs"]["rms_norm_eps"], 1e-6)
        for name in QSAIndexer._tp_submodules:
            self.assertIn(name, exported)
        plan = {}   # indexer submodule keys are NOT in the plan: whole imports
        m = QSAIndexer.tp_import({"device": CPU, "consumer": None}, exported, plan)
        self.assertEqual(m.key, "attn.indexer")
        self.assertEqual(m.n_heads, 16)
        self.assertEqual(m.kv_heads, 1)
        self.assertEqual(m.block_topk, 2048 // 4)
        self.assertEqual(m.compress_ratio, 4)
        self.assertEqual(m.rms_norm_eps, 1e-6)
        self.assertEqual(m.device, CPU)
        self.assertEqual([x.key for x in m.modules],
                         ["attn.indexer.index_qk_proj", "attn.indexer.q_layernorm",
                          "attn.indexer.k_layernorm"])
        self.assertEqual(m.storage_size(), 65536)

    def test_parent_constructs_children_from_config(self):
        cfg = FakeConfig(stc = None)
        idx = self._idx(cfg)
        self.assertIsInstance(idx.index_qk_proj, FakeLinear)
        self.assertIsInstance(idx.q_layernorm, FakeRMSNorm)
        self.assertEqual(idx.index_qk_proj.key, "attn.indexer.index_qk_proj")


class HyperHeadMeanExportTest(unittest.TestCase):
    """c9abcd54: the parameterless-mean flag rides the HyperHead export
    (imported heads without it would silently change the stream collapse)."""

    def test_mean_key_in_hunk(self):
        src = (REPO / "exllamav3/modules/hyperconnections.py").read_text()
        self.assertIn('"mean": self.mean,', src)

    def test_tp_import_forwards_kwargs(self):
        src = (REPO / "exllamav3/modules/hyperconnections.py").read_text()
        self.assertIn("HyperHead(config = None, **exported[\"kwargs\"])", src)


class FnPatchPayloadTest(unittest.TestCase):
    """The staged model_tp_fn.py hunks (9d18a7f prefetch + 002176e vocab-limited
    argmax): the patch file must exist as provenance, and the working tree must
    actually carry the payload - model_tp.py's tp_dispatch_lm_head_argmax passes
    vocab_size positionally and depends on the argmax hunk being live."""

    PATCH_CANDIDATES = (
        Path("/home/homelab1/datapool/rocm-exl3-rdna2/runs/tp2/qwen-tp-fn.patch"),
        Path("/work/runs/tp2/qwen-tp-fn.patch"),   # same file inside the v620 pair container
    )
    FN = REPO / "exllamav3/model/model_tp_fn.py"

    def _patch_text(self):
        for p in self.PATCH_CANDIDATES:
            if p.exists():
                return p.read_text()
        return None

    def test_patch_covers_prefetch_and_vocab(self):
        text = self._patch_text()
        if text is None:
            self.skipTest("qwen-tp-fn.patch not visible at host or /work paths")
        self.assertIn('module.caps.get("prefetch_ids")', text)
        self.assertIn("single_idx is None", text)
        self.assertIn("vocab_size: int = -1", text)
        self.assertIn('-float("inf")', text)
        self.assertIn("gather_devices: list[int] | None", text)

    def test_fn_working_tree_carries_the_payload(self):
        # If this fails while the patch check above passes, the hunks still need
        # to be applied (git apply qwen-tp-fn.patch); model_tp.py + fn must land
        # together or the draft TP argmax raises TypeError.
        src = self.FN.read_text()
        self.assertIn('module.caps.get("prefetch_ids")', src)
        self.assertIn("vocab_size: int = -1", src)
        self.assertIn('-float("inf")', src)


class TPPlumbingSourceTest(unittest.TestCase):
    """Cheap structural checks that the d6943498 wiring landed in every fork
    module with a trailing TP collective (the GPU-behavior side runs under
    tests/test_tp_export_attention.py on the hardware box)."""

    MODULES_REDUCE = ["attn", "sliding_attn", "gated_delta_net", "mamba2", "dsv4",
                      "mlp", "block_sparse_mlp", "transformer"]

    def test_tp_collect_wired(self):
        for name in self.MODULES_REDUCE:
            src = (REPO / f"exllamav3/modules/{name}.py").read_text()
            self.assertIn("self.tp_collect(", src, f"{name}.py lost tp_collect wiring")
            self.assertNotIn('params["backend"].all_reduce(', src,
                             f"{name}.py still calls all_reduce directly")

    def test_tp_owner_wired(self):
        for name in self.MODULES_REDUCE:
            src = (REPO / f"exllamav3/modules/{name}.py").read_text()
            self.assertIn("tp_single_owner(", src, f"{name}.py lost tp_owner wiring")

    def test_supports_tp_enabled(self):
        src = (REPO / "exllamav3/architecture/qwen4_exp.py").read_text()
        self.assertIn('"supports_tp": True', src)
        draft = (REPO / "exllamav3/architecture/qwen4_exp_mtp.py").read_text()
        self.assertIn('"supports_tp": False', draft)   # draft stays unsplit by design
        self.assertIn("lm_head_argmax(state, params)", draft)

    def test_mtp_target_dispatch_wired(self):
        src = (REPO / "exllamav3/modules/arch_specific/qwen4_exp_mtp.py").read_text()
        self.assertIn("mp_model_forward_embedding", src)
        self.assertIn("loaded_tp", src)
        model_tp = (REPO / "exllamav3/model/model_tp.py").read_text()
        self.assertIn("tp_parent_defer", model_tp)
        self.assertIn("return_max", model_tp)
        model = (REPO / "exllamav3/model/model.py").read_text()
        self.assertIn("def lm_head_argmax", model)


class CollectionIsolationTest(unittest.TestCase):
    """This file fakes the exllamav3 stack at COLLECTION time (module import).
    pytest imports every collected file before running any test, so a previous
    version's sys.modules takeover of the real "exllamav3" names broke files
    importing the genuine package in the same process (e.g.
    tests/test_tp_export_attention.py: "cannot import name 'Linear' from
    'exllamav3.modules'"). Everything harness-owned must live under the private
    _H prefix, in either collection order, and the real namespace must be
    observable-untouched by this file."""

    def test_harness_registers_only_private_names(self):
        self.assertTrue(_H.startswith("_tp_qwen_arch_cpu_harness_"))
        for name in _INSTALLED:
            self.assertTrue(name == _H or name.startswith(_H + "."),
                            f"harness module outside the private prefix: {name}")
            self.assertNotIn("exllamav3", name.split("."),
                             f"real package name segment in {name}")

    def test_nothing_registered_under_real_namespace(self):
        # every sys.modules key under the REAL names must be foreign to the
        # harness: never an _H module, never one of our fake classes' homes
        ours = {id(sys.modules[n]) for n in _INSTALLED}
        for k, m in list(sys.modules.items()):
            if k == "exllamav3" or k.startswith("exllamav3."):
                self.assertNotIn(id(m), ours,
                                 f"sys.modules[{k!r}] is a harness module (shadowing!)")
                for attr in ("Linear", "Module", "NGramEmbedding", "PLELayer"):
                    v = getattr(m, attr, None)
                    self.assertNotIn(v, (FakeLinear, FakeRMSNorm, Module,
                                         NGramEmbedding, PLELayer, PLELayerState),
                                     f"real namespace attribute {k}.{attr} is a harness fake")

    def test_real_namespace_untouched_since_import(self):
        # whatever exllamav3 entries existed when this file was collected must
        # be the very same objects now (order: attention tests collected first
        # in the container) - this file neither replaced any of them. Keys that
        # no longer exist belong to a peer test's own fake stack that cleaned
        # up after itself; this file never removes anything it did not install.
        for k, v in _REAL_NS_AT_IMPORT.items():
            if k in sys.modules:
                self.assertIs(sys.modules[k], v, f"this file replaced sys.modules[{k!r}]")

    def test_classes_under_test_come_from_the_harness(self):
        for cls in (NGramEmbedding, PLELayer, PLELayerState, QSAIndexer, Module,
                    TPAllocation, TPAllocator, DiskTensorHandle):
            self.assertTrue(cls.__module__ == _H or cls.__module__.startswith(_H + "."),
                            f"{cls.__name__} came from {cls.__module__}, not the harness")

    def test_harness_leaves_no_exllamav3_keys_behind(self):
        # every real-namespace key that exists in this process (ours, or a peer
        # test's own fake stack) must resolve to a foreign module, never to a
        # module this harness loaded: map our private leaves by their on-disk
        # source file and reject any exllamav3 key backed by the same object.
        ours = {id(sys.modules[n]) for n in _INSTALLED}
        for k, m in list(sys.modules.items()):
            if k == "exllamav3" or k.startswith("exllamav3.") or k == "exllamav3_ext":
                self.assertNotIn(id(m), ours,
                                 f"sys.modules[{k!r}] is the very object this harness "
                                 "loaded under the private prefix (namespace takeover)")


def tearDownModule():
    # Nothing was ever registered under the real "exllamav3" namespace, so there
    # is nothing to restore for other test modules. Drop the private harness tree
    # so a re-import in the same process (unlikely under pytest) starts fresh.
    for name in reversed(_INSTALLED):
        if name == _H or name.startswith(_H + "."):
            sys.modules.pop(name, None)
    _INSTALLED.clear()


if __name__ == "__main__":
    unittest.main(verbosity = 2)
