#!/usr/bin/env python3
"""CPU-only tests for the Phase-5 benchmark-harness additions.

Covers:
  * bench.py --ngram-ram: CLI wiring (legacy default untouched),
    process_memory_snapshot()/memory_delta() host accounting;
  * multi_gpu n-gram residence helpers: find_ngram_modules (duck-typed walk),
    describe_ngram_module (actual mode / tensor residence / metadata bytes /
    prefetch counters) and collect_ngram_state fail-closed verdicts when
    --ngram-ram was requested but the live tables did NOT end up host-RAM-
    backed (disk modes, missing tensors, non-host residence), plus the
    tableless no-op note;
  * multi_gpu.audit_placement on HYBRID models: GDN/PLE recurrent layer states
    audited locally (allocation, owning-module locality, cross-CUDA tensors
    rejected, deliberate host-side holders recorded), QSA indexer planes
    (raw_k/pooled) included in the cache-layer tensor audit, recurrent layer
    counts recorded per device, a GDN/PLE-only device no longer held to the
    ordinary-attention KV count -- while the validated dense Qwen3 D/M verdicts
    (messages, counters, progression) are regression-checked to be unchanged.

Everything runs against duck-typed fakes plus real torch.device objects; the
stub tensors deliberately expose ONLY numel()/element_size()/device (no
.item()): the audit must read placement metadata, never GPU values, and would
crash here if it tried. NOTHING HERE CLAIMS A GPU EVER RAN -- these fakes
prove the checks reject false assumptions; hardware evidence comes only from
the container runs.

Run from the repo root (or anywhere):
    python3 -m pytest -q rocm_tools/rdna2/tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch   # host CPU torch: only for torch.device objects, no CUDA calls

from rocm_tools.rdna2 import bench
from rocm_tools.rdna2 import multi_gpu

D0 = torch.device("cuda:0")
D1 = torch.device("cuda:1")
D5 = torch.device("cuda:5")
DCPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# stubs (same shapes as test_ls_split_cpu.py; metadata-only tensors)
# ---------------------------------------------------------------------------

class MetaT:
    """Stub tensor carrying ONLY placement metadata. No .item(): if any audited
    code path tried a GPU value sync it would AttributeError right here."""

    def __init__(self, device, numel = 1024, esize = 2):
        self.device = device
        self._numel = numel
        self._esize = esize

    def numel(self):
        return self._numel

    def element_size(self):
        return self._esize


class FakeCacheLayer:
    """Dense KV layer: .k/.v (+get_tensors fallback)."""

    def __init__(self, device, k = ..., v = ...):
        self.device = device
        self.k = MetaT(device) if k is ... else k
        self.v = MetaT(device) if v is ... else v

    def get_tensors(self):
        out = [t for t in (self.k, self.v) if t is not None]
        return out or None


class FakeQsaCacheLayer(FakeCacheLayer):
    """CacheLayer_qsa shape: fp16 k/v PLUS the indexer planes raw_k/pooled."""

    def __init__(self, device, raw_k = ..., pooled = ...):
        super().__init__(device)
        self.raw_k = MetaT(device, 256, 2) if raw_k is ... else raw_k
        self.pooled = MetaT(device, 64, 2) if pooled is ... else pooled

    def get_tensors(self):
        out = [t for t in (self.k, self.v, self.raw_k, self.pooled) if t is not None]
        return out or None


class FakeRecState:
    """GDNLayerState/PLELayerState shape: .device + named tensor holders."""

    def __init__(self, device, conv_state = ..., recurrent_state = ..., id_state = None):
        self.device = device
        self.conv_state = (MetaT(device, 512, 2) if conv_state is ... else conv_state)
        self.recurrent_state = (MetaT(device, 4096, 4) if recurrent_state is ...
                                else recurrent_state)
        self.id_state = id_state


class FakeModule:
    def __init__(self, key, device = None, caps = None, subs = None,
                 cache_layers = None, recurrent_layers = None, layer_idx = None):
        self.key = key
        self.device = device
        self.caps = caps or {}
        self.modules = subs or []
        self.cache_layers = cache_layers or []
        self.recurrent_layers = recurrent_layers or []
        self.layer_idx = layer_idx


class FakeModel:
    def __init__(self, modules, active, output):
        self.modules = modules
        self.active_devices = active
        self.output_device = output


class FakeNgram:
    """NGramEmbedding shape (post-load): mode + tables or handles + stats."""

    def __init__(self, key, mode, device, tables = None, handles = None,
                 num_rows = 4096, rows_per_shard = 4096, stats = None):
        self.key = key
        self.mode = mode
        self.device = device
        self.tables = tables
        self.handles = handles
        self.num_rows = num_rows
        self.rows_per_shard = rows_per_shard
        self.prefetch_stats = stats if stats is not None else {
            "hit": 0, "miss": 0, "retired": 0}


class FakeHandle:
    def __init__(self, filename, row_bytes = 320):
        self.filename = filename
        self.row_bytes = row_bytes


def embed_head(dev_e = D0, dev_h = D1):
    return (FakeModule("model.embed_tokens", dev_e, layer_idx = -1),
            FakeModule("lm_head", dev_h, caps = {"logits_output": True}, layer_idx = 36))


def dense_block(i, dev):
    """Validated Qwen3 shape: attn (kv_cache + cache layer) + mlp."""
    subs = [FakeModule(f"model.layers.{i}.self_attn", dev, caps = {"kv_cache": True},
                       cache_layers = [FakeCacheLayer(dev)]),
            FakeModule(f"model.layers.{i}.mlp", dev)]
    return FakeModule(f"model.layers.{i}", dev, subs = subs, layer_idx = i)


def gdn_block(i, dev, with_state = True):
    """Hybrid GDN shape: linear_attn submodule with recurrent_cache caps and
    an attached GDNLayerState-like object (allocated on the block device)."""
    state = [FakeRecState(dev)] if with_state else []
    subs = [FakeModule(f"model.layers.{i}.linear_attn", dev,
                        caps = {"recurrent_cache": True}, recurrent_layers = state),
            FakeModule(f"model.layers.{i}.mlp", dev)]
    return FakeModule(f"model.layers.{i}", dev, subs = subs, layer_idx = i)


def qsa_block(i, dev):
    """Hybrid QSA shape: ordinary attention whose cache layer carries the
    indexer planes (CacheLayer_qsa)."""
    subs = [FakeModule(f"model.layers.{i}.self_attn", dev, caps = {"kv_cache": True},
                       cache_layers = [FakeQsaCacheLayer(dev)]),
            FakeModule(f"model.layers.{i}.mlp", dev)]
    return FakeModule(f"model.layers.{i}", dev, subs = subs, layer_idx = i)


def ple_top(i, dev, ngram = None):
    """PLELayer shape: TOP-level recurrent_cache module (negative layer_idx)
    with a nested NGramEmbedding submodule."""
    subs = [ngram] if ngram is not None else []
    state = [FakeRecState(dev, conv_state = MetaT(dev, 512, 2),
                          recurrent_state = None, id_state = MetaT(DCPU, 3, 8))]
    return FakeModule(f"model.language_model.layers.{i}.ple", dev,
                      caps = {"recurrent_cache": True}, subs = subs,
                      recurrent_layers = state, layer_idx = -(i + 1))


# ---------------------------------------------------------------------------
# bench.py: --ngram-ram CLI + host process accounting
# ---------------------------------------------------------------------------

class NgramRamCliTests(unittest.TestCase):
    def test_legacy_default_unchanged(self):
        args = bench.build_parser().parse_args(["-m", "d"])
        self.assertIs(args.ngram_ram, False)

    def test_flag_parses_true_and_coexists_with_split(self):
        args = bench.build_parser().parse_args(
            ["-m", "d", "--use-per-device", "12", "14", "--ngram-ram"])
        self.assertIs(args.ngram_ram, True)
        self.assertEqual(args.use_per_device, [12.0, 14.0])


class ProcessMemoryTests(unittest.TestCase):
    def test_snapshot_reads_real_ints_on_linux(self):
        snap = bench.process_memory_snapshot()
        self.assertEqual(sorted(snap),
                         ["major_faults", "minor_faults", "rss_bytes", "rss_peak_bytes"])
        self.assertIsInstance(snap["major_faults"], int)
        self.assertIsInstance(snap["minor_faults"], int)
        self.assertGreaterEqual(snap["major_faults"], 0)
        self.assertGreater(snap["rss_peak_bytes"], 0)
        self.assertGreater(snap["rss_bytes"], 0)

    def test_snapshot_is_monotonic_on_fault_counters(self):
        a = bench.process_memory_snapshot()
        b = bench.process_memory_snapshot()
        self.assertGreaterEqual(b["minor_faults"], a["minor_faults"])

    def test_memory_delta_arithmetic(self):
        a = {"rss_bytes": 100, "rss_peak_bytes": 200, "major_faults": 5, "minor_faults": 50}
        b = {"rss_bytes": 150, "rss_peak_bytes": 250, "major_faults": 7, "minor_faults": 80}
        self.assertEqual(bench.memory_delta(a, b),
                         {"rss_bytes": 50, "rss_peak_bytes": 50,
                          "major_faults": 2, "minor_faults": 30})

    def test_memory_delta_none_safe(self):
        self.assertIsNone(bench.memory_delta(None, {"x": 1}))
        self.assertIsNone(bench.memory_delta({"x": 1}, None))
        self.assertEqual(bench.memory_delta({"x": 1}, {"x": None}), {"x": None})


# ---------------------------------------------------------------------------
# multi_gpu: n-gram residence measured from the live objects
# ---------------------------------------------------------------------------

class NgramHelpersTests(unittest.TestCase):
    def ngram_model(self, ngram):
        ple = ple_top(0, D0, ngram = ngram)
        emb, head = embed_head()
        return FakeModel([emb, ple, head], [0, 1], D1)

    def test_find_walks_nested_and_ignores_plain_modules(self):
        ng = FakeNgram("...layers.0.ple.ple_embedding.ngram_embedding",
                       "trellis_ram", D0, tables = [MetaT(DCPU, 100, 2)])
        found = multi_gpu.find_ngram_modules(self.ngram_model(ng))
        self.assertEqual([m.key for m in found], [ng.key])

    def test_disk_mode_describes_handles_not_tables(self):
        ng = FakeNgram("ple.ngram_embedding", "trellis_disk", D0, tables = None,
                       handles = [FakeHandle("/models/t-00001.safetensors")])
        rec = multi_gpu.describe_ngram_module(ng)
        self.assertEqual(rec["mode"], "trellis_disk")
        self.assertFalse(rec["ram_backed"])
        self.assertEqual(rec["num_disk_handles"], 1)
        self.assertEqual(rec["disk_handle_files"], ["/models/t-00001.safetensors"])
        self.assertEqual(rec["table_tensors"], [])
        self.assertIsNone(rec["table_ram_bytes"])

    def test_describe_records_actual_residence_and_bytes(self):
        ng = FakeNgram("ple.ngram_embedding", "fp16_ram", D0,
                       tables = [MetaT(DCPU, 1000, 2), MetaT(DCPU, 500, 2)],
                       stats = {"hit": 4, "miss": 1, "retired": 0})
        rec = multi_gpu.describe_ngram_module(ng)
        self.assertTrue(rec["ram_backed"])
        self.assertEqual(rec["table_residence"], ["cpu"])
        self.assertEqual(rec["table_ram_bytes"], 1000 * 2 + 500 * 2)
        self.assertEqual(rec["prefetch_stats"], {"hit": 4, "miss": 1, "retired": 0})

    def test_prefetch_stats_copied_not_aliased(self):
        ng = FakeNgram("k", "trellis_ram", D0, tables = [MetaT(DCPU, 1, 2)])
        rec = multi_gpu.describe_ngram_module(ng)
        ng.prefetch_stats["hit"] = 99
        self.assertEqual(rec["prefetch_stats"]["hit"], 0)

    def test_collect_require_ram_ok_on_real_ram_tables(self):
        ng = FakeNgram("ple.ngram_embedding", "trellis_ram", D0,
                       tables = [MetaT(DCPU, 8192, 2)])
        st = multi_gpu.collect_ngram_state(self.ngram_model(ng), require_ram = True)
        self.assertTrue(st["ok"], st["problems"])
        self.assertEqual(st["tables_found"], 1)
        self.assertTrue(st["ram_backed"])
        self.assertEqual(st["total_table_ram_bytes"], 8192 * 2)

    def test_collect_require_ram_fails_on_disk_streamed_table(self):
        ng = FakeNgram("ple.ngram_embedding", "fp16_disk", D0, tables = None,
                       handles = [FakeHandle("/models/t.safetensors")])
        st = multi_gpu.collect_ngram_state(self.ngram_model(ng), require_ram = True)
        self.assertFalse(st["ok"])
        self.assertTrue(any("fp16_disk" in p and "not" in p for p in st["problems"]),
                        st["problems"])

    def test_collect_require_ram_fails_on_non_host_residence(self):
        # RAM-labelled mode whose tensors are NOT on the host: the process RSS
        # assumption the flag makes would be false -> must fail closed.
        ng = FakeNgram("ple.ngram_embedding", "trellis_ram", D0,
                       tables = [MetaT(D0, 8192, 2)])
        st = multi_gpu.collect_ngram_state(self.ngram_model(ng), require_ram = True)
        self.assertFalse(st["ok"])
        self.assertTrue(any("not host-resident" in p for p in st["problems"]),
                        st["problems"])

    def test_collect_require_ram_fails_on_empty_ram_tables(self):
        ng = FakeNgram("ple.ngram_embedding", "trellis_ram", D0, tables = [])
        st = multi_gpu.collect_ngram_state(self.ngram_model(ng), require_ram = True)
        self.assertFalse(st["ok"])
        self.assertTrue(any("no table tensors" in p for p in st["problems"]),
                        st["problems"])

    def test_collect_tableless_model_is_recorded_noop(self):
        emb, head = embed_head()
        st = multi_gpu.collect_ngram_state(FakeModel([emb, head], [0, 1], D1),
                                          require_ram = True)
        self.assertTrue(st["ok"])           # nothing to force, nothing to fail
        self.assertEqual(st["tables_found"], 0)
        self.assertTrue(any("no-op" in n for n in st["notes"]), st["notes"])

    def test_collect_without_requirement_never_fails_disk_modes(self):
        # legacy default (no flag): disk-streamed tables are recorded, not an
        # error -- behavior before Phase 5 is untouched.
        ng = FakeNgram("ple.ngram_embedding", "trellis_disk", D0, tables = None,
                       handles = [FakeHandle("/models/t.safetensors")])
        st = multi_gpu.collect_ngram_state(self.ngram_model(ng), require_ram = False)
        self.assertTrue(st["ok"], st["problems"])
        self.assertEqual(st["modules"][0]["mode"], "trellis_disk")
        self.assertIs(st["ram_backed"], False)

    def test_residence_evidence_never_cites_reserved_as_capacity(self):
        ng = FakeNgram("k", "trellis_ram", D0, tables = [MetaT(DCPU, 1, 2)])
        st = multi_gpu.collect_ngram_state(FakeModel([ng], [0], D0), require_ram = True)
        self.assertIn("NOT treated as evidence", st["residence_evidence"])


# ---------------------------------------------------------------------------
# audit_placement: hybrid (GDN/QSA/PLE) models
# ---------------------------------------------------------------------------

class HybridAuditTests(unittest.TestCase):
    def test_computation_state_on_cpu_or_meta_is_rejected(self):
        for device in (torch.device("cpu"), torch.device("meta")):
            model = self.hybrid_model()
            model.modules[1].modules[0].recurrent_layers[0].recurrent_state = MetaT(device)
            self.assertFalse(multi_gpu.audit_placement(model, [0, 1])["ok"])

    def test_empty_recurrent_storage_is_rejected(self):
        model = self.hybrid_model()
        state = model.modules[1].modules[0].recurrent_layers[0]
        state.conv_state = state.recurrent_state = None
        self.assertFalse(multi_gpu.audit_placement(model, [0, 1])["ok"])

    def test_missing_qsa_plane_is_rejected(self):
        model = self.hybrid_model()
        model.modules[2].modules[0].cache_layers[0].pooled = None
        self.assertFalse(multi_gpu.audit_placement(model, [0, 1])["ok"])

    def hybrid_model(self):
        """embed -> [gdn0, qsa1, ple2-ish top] on cuda:0 | [gdn3, qsa4] on cuda:1
        -> head. GDN/PLE blocks carry local recurrent states; QSA layers carry
        the indexer planes."""
        emb, head = embed_head()
        mods = [emb,
                gdn_block(0, D0),
                qsa_block(1, D0),
                ple_top(2, D0),
                gdn_block(3, D1),
                qsa_block(4, D1),
                head]
        return FakeModel(mods, [0, 1], D1)

    def test_hybrid_split_passes_and_records_recurrent_counts(self):
        a = multi_gpu.audit_placement(self.hybrid_model(), [0, 1])
        self.assertTrue(a["ok"], a["problems"])
        self.assertEqual(a["recurrent_layers_per_device"], {"cuda:0": 2, "cuda:1": 1})
        self.assertEqual(a["total_recurrent_layers"], 3)
        self.assertEqual(a["cache_layers_per_device"], {"cuda:0": 1, "cuda:1": 1})
        # per-record allocation evidence: devices + metadata bytes, no sync
        gdn_rec = [r for r in a["recurrent_layers"]
                   if r["module_key"].endswith("layers.3.linear_attn")][0]
        self.assertEqual(gdn_rec["device"], "cuda:1")
        self.assertEqual(gdn_rec["device_index"], 1)
        self.assertEqual(gdn_rec["tensor_devices"],
                         {"conv_state": "cuda:1", "recurrent_state": "cuda:1"})
        self.assertEqual(gdn_rec["total_bytes"], 512 * 2 + 4096 * 4)

    def test_ple_host_side_id_state_recorded_not_rejected(self):
        a = multi_gpu.audit_placement(self.hybrid_model(), [0, 1])
        ple_rec = [r for r in a["recurrent_layers"]
                   if r["module_key"].endswith(".ple")][0]
        self.assertEqual(ple_rec["tensor_devices"]["id_state"], "cpu")
        self.assertTrue(a["ok"], a["problems"])   # host-side by design, but recorded

    def test_qsa_planes_included_in_cache_layer_audit(self):
        a = multi_gpu.audit_placement(self.hybrid_model(), [0, 1])
        qsa_recs = [c for c in a["cache_layers"] if "self_attn" in c["module_key"]]
        self.assertEqual(len(qsa_recs), 2)
        for rec in qsa_recs:
            self.assertEqual(rec["n_tensors"], 4)
            self.assertEqual(set(rec["tensor_devices"]),
                             {"k", "v", "raw_k", "pooled"})

    def test_qsa_plane_on_wrong_device_rejected(self):
        m = self.hybrid_model()
        qsa = m.modules[5].modules[0]           # cuda:1 attention
        qsa.cache_layers = [FakeQsaCacheLayer(D1, raw_k = MetaT(D0, 256, 2))]
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("raw_k" in p for p in a["problems"]), a["problems"])

    def test_recurrent_state_remote_from_owning_module_rejected(self):
        m = self.hybrid_model()
        m.modules[1].modules[0].recurrent_layers = [FakeRecState(D1)]   # GDN on D0
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("owning module" in p for p in a["problems"]), a["problems"])

    def test_recurrent_state_tensor_cross_device_rejected(self):
        m = self.hybrid_model()
        gdn = m.modules[1].modules[0]
        gdn.recurrent_layers = [FakeRecState(D0, conv_state = MetaT(D1, 512, 2))]
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("cuda:1" in p and "recurrent layer state" in p
                            for p in a["problems"]), a["problems"])

    def test_unallocated_recurrent_state_rejected(self):
        m = self.hybrid_model()
        gdn = m.modules[1].modules[0]
        gdn.recurrent_layers = [FakeRecState(None, conv_state = None,
                                             recurrent_state = None)]
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("device is None" in p and "recurrent" in p
                            for p in a["problems"]), a["problems"])

    def test_state_outside_split_reported_without_crash(self):
        m = self.hybrid_model()
        gdn = m.modules[1].modules[0]
        gdn.recurrent_layers = [FakeRecState(D5)]
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("cuda:5" in p and "not one of the expected" in p
                            for p in a["problems"]), a["problems"])
        self.assertTrue(all(r["device_index"] is None or r["device_index"] in (0, 1)
                            for r in a["recurrent_layers"]))

    def test_gdn_only_device_not_held_to_ordinary_attention_count(self):
        # The false-positive this task exists to kill: cuda:1 legitimately owns
        # only GDN layers (zero KV cache layers) with local recurrent states.
        emb, head = embed_head()
        mods = [emb, dense_block(0, D0), dense_block(1, D0), gdn_block(2, D1), head]
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D1), [0, 1])
        self.assertTrue(a["ok"], a["problems"])
        self.assertEqual(a["cache_layers_per_device"], {"cuda:0": 2, "cuda:1": 0})
        self.assertEqual(a["recurrent_layers_per_device"], {"cuda:0": 0, "cuda:1": 1})
        self.assertFalse(any("owns NO cache layers" in p for p in a["problems"]))

    def test_recurrent_capable_device_without_state_rejected(self):
        # counterpart rule: if recurrent states exist anywhere, a device whose
        # modules advance state must own them locally.
        emb, head = embed_head()
        mods = [emb, dense_block(0, D0), gdn_block(1, D0), gdn_block(2, D1, with_state = False),
                head]
        # with_state=False removes the state objects, so total falls to 1 (D0)
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D1), [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("owns NO recurrent layer states" in p for p in a["problems"]),
                        a["problems"])


# ---------------------------------------------------------------------------
# D/M regression: dense Qwen3 verdicts / messages are byte-identical
# ---------------------------------------------------------------------------

class DenseRegressionTests(unittest.TestCase):
    def dense_model(self, strip_cache_on_d1 = False):
        emb, head = embed_head()
        mods = [emb] + [dense_block(i, D0 if i < 3 else D1) for i in range(6)] + [head]
        if strip_cache_on_d1:
            for blk in mods[4:7]:
                blk.modules[0].cache_layers = []
        return FakeModel(mods, [0, 1], D1)

    def test_valid_dense_split_ok_with_zero_recurrent_keys(self):
        a = multi_gpu.audit_placement(self.dense_model(), [0, 1])
        self.assertTrue(a["ok"], a["problems"])
        self.assertEqual(a["transformer_modules_per_device"], {"cuda:0": 3, "cuda:1": 3})
        self.assertEqual(a["cache_layers_per_device"], {"cuda:0": 3, "cuda:1": 3})
        self.assertEqual(a["recurrent_layers_per_device"], {"cuda:0": 0, "cuda:1": 0})
        self.assertEqual(a["total_recurrent_layers"], 0)
        self.assertEqual(a["recurrent_layers"], [])
        self.assertEqual(a["kv_cache_modules_per_device"], {"cuda:0": 3, "cuda:1": 3})

    def test_kv_device_missing_cache_layers_still_exact_old_message(self):
        a = multi_gpu.audit_placement(self.dense_model(strip_cache_on_d1 = True), [0, 1])
        self.assertFalse(a["ok"])
        self.assertIn("device cuda:1 owns NO cache layers although 3 cache layer(s) "
                      "exist -- KV pages are not local to the devices whose attention "
                      "layers need them", a["problems"])

    def test_cache_layer_records_gained_bytes_without_changing_devices(self):
        a = multi_gpu.audit_placement(self.dense_model(), [0, 1])
        for rec in a["cache_layers"]:
            self.assertEqual(rec["tensor_devices"], {"k": "cuda:0", "v": "cuda:0"}
                             if rec["device"] == "cuda:0" else
                             {"k": "cuda:1", "v": "cuda:1"})
            self.assertEqual(rec["total_bytes"], 2 * 1024 * 2)


if __name__ == "__main__":
    unittest.main()
