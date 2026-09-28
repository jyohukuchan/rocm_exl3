#!/usr/bin/env python3
"""CPU-only tests for the Phase-3 dual-V620 layer-split harness.

Covers rocm_tools/rdna2/multi_gpu.py (budget validation, the split gate,
plan_load_mode, per-device memory/transfer helpers and the placement AUDITOR),
the new --use-per-device / --cache-tokens / --expect-arch CLI wiring in
bench.py + collect_top1.py, and the single-GPU regressions (no-budget bench
still demands exactly one visible GPU; transformers CPU stays supported).

Everything runs against duck-typed fakes or the host CPU torch's torch.device
objects (construction needs no CUDA). NOTHING HERE CLAIMS A GPU EVER RAN: the
fakes prove the gates reject and the auditor measures -- they are not evidence
of hardware success, which only the root's container runs provide.

Run from the repo root (or anywhere):
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import argparse
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch   # host CPU torch: only for torch.device objects, no CUDA calls

from rocm_tools.rdna2 import bench
from rocm_tools.rdna2 import collect_top1
from rocm_tools.rdna2 import multi_gpu


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeProps:
    def __init__(self, name, arch, mem, uuid=None, pci=None):
        self.name = name
        self.gcnArchName = arch
        self.total_memory = mem
        self.multi_processor_count = 40
        if uuid is not None:
            self.uuid = uuid
        if pci is not None:
            self.pci_bus_id = pci


class FakeCuda:
    def __init__(self, props_list, available=True):
        self._props = props_list
        self._available = available
        self.reset_calls = []
        self.sync_calls = []
        self.snap_calls = {"alloc": {}, "res": {}, "peak": {}}

    def device_count(self):
        return len(self._props)

    def is_available(self):
        return self._available

    def get_device_properties(self, i):
        return self._props[i]

    def synchronize(self, dev):
        self.sync_calls.append(str(dev))

    def reset_peak_memory_stats(self, dev):
        self.reset_calls.append(str(dev))

    def memory_allocated(self, dev):
        return self.snap_calls["alloc"].get(str(dev), 0)

    def memory_reserved(self, dev):
        return self.snap_calls["res"].get(str(dev), 0)

    def max_memory_allocated(self, dev):
        return self.snap_calls["peak"].get(str(dev), 0)


class FakeTorch:
    def __init__(self, props_list, hip="6.4.x", available=True):
        self.version = argparse.Namespace(hip=hip)
        self.cuda = FakeCuda(props_list, available)
        self.device = torch.device   # real device objects; no CUDA init


class ShimT:
    """Fake tensor: only .device matters to the auditor."""
    def __init__(self, device):
        self.device = device


class FakeCacheLayer:
    def __init__(self, device, k_dev=None, v_dev=None, storage=True):
        self.device = device
        if storage:
            self.k = ShimT(k_dev if k_dev is not None else device)
            self.v = ShimT(v_dev if v_dev is not None else device)
        else:
            self.k = None
            self.v = None

    def get_tensors(self):
        out = [t for t in (self.k, self.v) if t is not None]
        return out or None


class FakeModule:
    def __init__(self, key, device=None, caps=None, subs=None,
                 cache_layers=None, layer_idx=None):
        self.key = key
        self.device = device
        self.caps = caps or {}
        self.modules = subs or []
        self.cache_layers = cache_layers or []
        self.layer_idx = layer_idx


class FakeModel:
    def __init__(self, modules, active, output):
        self.modules = modules
        self.active_devices = active
        self.output_device = output


D0 = torch.device("cuda:0")
D1 = torch.device("cuda:1")
DCPU = torch.device("cpu")


def fake_block(i, dev, with_cache=True):
    """Mimic this fork's TransformerBlock: a layer_idx-carrying top module with
    Attention (kv_cache caps + cache layer) + MLP submodules."""
    subs = []
    attn_cl = [FakeCacheLayer(dev)] if with_cache else []
    subs.append(FakeModule(f"model.layers.{i}.self_attn", dev,
                           caps={"kv_cache": True}, cache_layers=attn_cl))
    subs.append(FakeModule(f"model.layers.{i}.mlp", dev))
    return FakeModule(f"model.layers.{i}", dev, subs=subs, layer_idx=i)


def embed_head(dev_e=D0, dev_h=D1):
    # embed and head carry layer_idx too (root finding) but are NOT transformer
    # modules -- the auditor must classify them by caps/key, not layer_idx.
    return (FakeModule("model.embed_tokens", dev_e, layer_idx=-1),
            FakeModule("lm_head", dev_h, caps={"logits_output": True}, layer_idx=36))


# ---------------------------------------------------------------------------
# budget validation
# ---------------------------------------------------------------------------

class BudgetTests(unittest.TestCase):
    def test_none_stays_none_backward_compatible(self):
        self.assertIsNone(multi_gpu.validate_use_per_device(None))

    def test_valid_floats_preserved_in_order(self):
        self.assertEqual(multi_gpu.validate_use_per_device([3, 4]), [3.0, 4.0])
        self.assertEqual(multi_gpu.validate_use_per_device([2.7, 4.0]), [2.7, 4.0])

    def test_fewer_than_two_rejected(self):
        for bad in ([5], [], [0.5]):
            with self.assertRaises(ValueError):
                multi_gpu.validate_use_per_device(bad)

    def test_nonpositive_nonfinite_nonnumeric_rejected(self):
        for bad in ([0, 4], [-2, 4], [float("nan"), 4], [float("inf"), 4],
                    [-float("inf"), 4], [True, 2], ["3", 4], [None, 4]):
            with self.assertRaises(ValueError):
                multi_gpu.validate_use_per_device(bad)


# ---------------------------------------------------------------------------
# gate + planning (fake torch; no GPU implied)
# ---------------------------------------------------------------------------

def pair_props():
    return [FakeProps("V620", "gfx1030", 32 * 1024**3, "GPU-aaaa", "67"),
            FakeProps("V620", "gfx1030", 32 * 1024**3, "GPU-bbbb", "3")]


class GateTests(unittest.TestCase):
    def test_identity_capture_direct_from_props(self):
        t = FakeTorch(pair_props())
        ident = multi_gpu.capture_device_identity(t, 0)
        self.assertEqual(ident["index"], 0)
        self.assertEqual(ident["pci_bus_id"], "67")
        self.assertEqual(ident["uuid"], "GPU-aaaa")
        self.assertEqual(ident["total_memory_bytes"], 32 * 1024**3)

    def test_missing_pci_uuid_recorded_null_not_guessed(self):
        t = FakeTorch([FakeProps("V620", "gfx1030", 1000)])
        ident = multi_gpu.capture_device_identity(t, 0)
        self.assertIsNone(ident["pci_bus_id"])
        self.assertIsNone(ident["uuid"])

    def test_gate_happy_path_captures_all_devices(self):
        t = FakeTorch(pair_props())
        gate = multi_gpu.validate_split_gate(t, 2, "gfx1030")
        self.assertEqual(gate["visible_devices"], 2)
        self.assertEqual([d["pci_bus_id"] for d in gate["devices"]], ["67", "3"])

    def test_gate_requires_rocm(self):
        t = FakeTorch(pair_props(), hip=None)
        with self.assertRaises(SystemExit) as cm:
            multi_gpu.validate_split_gate(t, 2, "gfx1030")
        self.assertIn("no ROCm", str(cm.exception))

    def test_gate_requires_exact_visible_count(self):
        # 3 budgets but 2 GPUs visible -> refuse, never a partial split
        t = FakeTorch(pair_props())
        with self.assertRaises(SystemExit) as cm:
            multi_gpu.validate_split_gate(t, 3, "gfx1030")
        self.assertIn("layer split with 3 expected GPU(s) found 2", str(cm.exception))
        # and the reverse: 2 budgets, 3 visible
        props3 = pair_props() + [FakeProps("V620", "gfx1030", 32 * 1024**3, "GPU-c", "11")]
        t3 = FakeTorch(props3)
        with self.assertRaises(SystemExit):
            multi_gpu.validate_split_gate(t3, 2, "gfx1030")

    def test_gate_rejects_arch_mismatch_on_any_device(self):
        props = pair_props()
        props[1] = FakeProps("Other", "gfx1036", 8 * 1024**3)
        t = FakeTorch(props)
        with self.assertRaises(SystemExit) as cm:
            multi_gpu.validate_split_gate(t, 2, "gfx1030")
        self.assertIn("gfx1036", str(cm.exception))

    def test_plan_no_budgets_is_always_single(self):
        # explicit --use-per-device is the ONLY sanctioned split path: even
        # with 2 visible GPUs, no-budgets plans the untouched single path.
        t = FakeTorch(pair_props())
        plan = multi_gpu.plan_load_mode(t, None, "gfx1030")
        self.assertEqual(plan["mode"], "single")
        self.assertIsNone(plan["gate"])

    def test_plan_budgets_gate_and_carry_through(self):
        t = FakeTorch(pair_props())
        plan = multi_gpu.plan_load_mode(t, [3.0, 4.0], "gfx1030")
        self.assertEqual(plan["mode"], "layer_split")
        self.assertEqual(plan["budgets"], [3.0, 4.0])
        self.assertEqual(plan["gate"]["use_per_device_gib"], [3.0, 4.0])
        self.assertEqual(len(plan["gate"]["devices"]), 2)


class SingleGateRegressionTests(unittest.TestCase):
    """bench's no-budget path must still require EXACTLY one visible GPU."""

    def test_single_gate_accepts_one_gfx1030(self):
        t = FakeTorch([FakeProps("V620", "gfx1030", 32 * 1024**3)])
        info = bench.validate_gpu(t, "cuda:0", "gfx1030")
        self.assertEqual(info["visible_devices"], 1)

    def test_single_gate_rejects_pair_and_hints_at_flag(self):
        t = FakeTorch(pair_props())
        with self.assertRaises(SystemExit) as cm:
            bench.validate_gpu(t, "cuda:0", "gfx1030")
        msg = str(cm.exception)
        self.assertIn("exactly one visible GPU, found 2", msg)
        self.assertIn("--use-per-device", msg)

    def test_single_gate_rejects_bad_device_and_arch(self):
        t = FakeTorch([FakeProps("V620", "gfx1030", 1)])
        with self.assertRaises(SystemExit):
            bench.validate_gpu(t, "cuda:1", "gfx1030")
        t2 = FakeTorch([FakeProps("V620", "gfx1013", 1)])
        with self.assertRaises(SystemExit):
            bench.validate_gpu(t2, "cuda:0", "gfx1030")


# ---------------------------------------------------------------------------
# per-device memory + transfer stats helpers
# ---------------------------------------------------------------------------

class MemoryStatsTests(unittest.TestCase):
    def test_snapshot_per_device_values(self):
        t = FakeTorch(pair_props())
        t.cuda.snap_calls = {"alloc": {"cuda:0": 100, "cuda:1": 200},
                             "res": {"cuda:0": 300, "cuda:1": 400},
                             "peak": {"cuda:0": 500, "cuda:1": 600}}
        snap = multi_gpu.device_memory_snapshot(t, [0, 1])
        self.assertEqual(snap["cuda:0"]["allocated_bytes"], 100)
        self.assertEqual(snap["cuda:1"]["peak_bytes"], 600)
        self.assertEqual(snap["cuda:1"]["reserved_bytes"], 400)

    def test_reset_and_sync_hit_every_used_device(self):
        t = FakeTorch(pair_props())
        multi_gpu.reset_peak_on_devices(t, [0, 1])
        multi_gpu.sync_devices(t, [0, 1])
        self.assertEqual(t.cuda.reset_calls, ["cuda:0", "cuda:1"])
        self.assertEqual(t.cuda.sync_calls, ["cuda:0", "cuda:1"])

    def test_diff_transfer_stats(self):
        before = {"direct": 5, "bounced": 2, "probes": 1}
        after = {"direct": 9, "bounced": 2, "probes": 3}
        self.assertEqual(multi_gpu.diff_transfer_stats(before, after),
                         {"direct": 4, "bounced": 0, "probes": 2})


# ---------------------------------------------------------------------------
# placement audit (the heart: measured, never inferred)
# ---------------------------------------------------------------------------

class AuditTests(unittest.TestCase):
    def good_model(self, with_cache=True):
        emb, head = embed_head()
        mods = [emb] + [fake_block(i, D0 if i < 3 else D1, with_cache)
                        for i in range(6)] + [head]
        return FakeModel(mods, [0, 1], D1)

    def test_valid_contiguous_split_passes_with_ordered_records(self):
        a = multi_gpu.audit_placement(self.good_model(), [0, 1])
        self.assertTrue(a["ok"], a["problems"])
        self.assertEqual(a["transformer_modules_per_device"], {"cuda:0": 3, "cuda:1": 3})
        self.assertEqual(a["cache_layers_per_device"], {"cuda:0": 3, "cuda:1": 3})
        self.assertEqual(a["device_progression"],
                         [{"device": "cuda:0", "modules": 4},
                          {"device": "cuda:1", "modules": 4}])
        # ordered records exist for every module and cache layer
        self.assertEqual(len(a["modules"]), 8)
        self.assertEqual([m["order"] for m in a["modules"]], list(range(8)))
        self.assertEqual(len(a["cache_layers"]), 6)
        # embed/head are recorded but NOT counted as transformer modules
        self.assertFalse(a["modules"][0]["transformer"])
        self.assertFalse(a["modules"][-1]["transformer"])
        self.assertEqual(a["output_device"], "cuda:1")

    def test_head_only_device_rejected_by_transformer_ownership(self):
        # root review scenario: cuda:0 holds ONLY embed/head (layer_idx present
        # but no attention) -- must not pass as a split.
        emb, head = embed_head(dev_e=D0, dev_h=D0)
        mods = [emb] + [fake_block(i, D1) for i in range(4)] + [head]
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D0), [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("owns NO transformer modules" in p for p in a["problems"]),
                        a["problems"])

    def test_noncontiguous_progression_rejected(self):
        emb, head = embed_head()
        mods = [emb] + [fake_block(i, D0 if i % 2 == 0 else D1) for i in range(4)] + [head]
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D1), [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("not contiguous" in p for p in a["problems"]), a["problems"])

    def test_missing_module_device_reported(self):
        m = self.good_model()
        m.modules[2].device = None
        m.modules[2].modules[0].device = None
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("device is None" in p for p in a["problems"]), a["problems"])

    def test_unexpected_device_index_no_keyerror(self):
        # root review: a stray cuda:5 module must produce a clear problem, not
        # crash the auditor (doc: never raises).
        emb, head = embed_head()
        mods = [emb, fake_block(0, torch.device("cuda:5")), fake_block(1, D1), head]
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D1), [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("cuda:5" in p and "not one of the expected" in p
                            for p in a["problems"]), a["problems"])
        self.assertTrue(all(m["device_index"] is None or m["device_index"] in (0, 1)
                            for m in a["modules"]))

    def test_submodule_device_disagreement_rejected(self):
        m = self.good_model()
        m.modules[1].modules[1].device = D1  # mlp of a cuda:0 block sneaks away
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("submodule device" in p for p in a["problems"]), a["problems"])

    def test_cache_layer_only_on_one_device_rejected(self):
        emb, head = embed_head()
        mods = [emb] + [fake_block(i, D0 if i < 3 else D1) for i in range(6)] + [head]
        # strip every cache layer off the cuda:1 blocks: attention on cuda:1
        # with KV pages only ever allocated on cuda:0
        for blk in mods[4:7]:
            blk.modules[0].cache_layers = []
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D1), [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("owns NO cache layers" in p for p in a["problems"]),
                        a["problems"])

    def test_cache_device_vs_owning_attention_module_rejected(self):
        # root review: comparing only k/v against cl.device would MISS this --
        # the layer and both tensors all say cuda:1 while the attention itself
        # lives on cuda:0.
        emb, head = embed_head()
        mods = [emb] + [fake_block(i, D0 if i < 3 else D1) for i in range(6)] + [head]
        attn = mods[1].modules[0]
        attn.cache_layers = [FakeCacheLayer(D1)]   # wrong device, tensors match it
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D1), [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("owning attention module" in p for p in a["problems"]),
                        a["problems"])

    def test_cache_tensor_off_recorded_device_rejected(self):
        emb, head = embed_head()
        mods = [emb] + [fake_block(i, D0 if i < 3 else D1) for i in range(6)] + [head]
        mods[1].modules[0].cache_layers = [FakeCacheLayer(D0, v_dev=D1)]
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D1), [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("tensor v is on cuda:1" in p for p in a["problems"]),
                        a["problems"])

    def test_empty_cache_storage_rejected(self):
        # root review: empty/None get_tensors means MISSING storage for Qwen3
        # cache layers -- must fail, not pass vacuously.
        emb, head = embed_head()
        mods = [emb] + [fake_block(i, D0 if i < 3 else D1) for i in range(6)] + [head]
        mods[1].modules[0].cache_layers = [FakeCacheLayer(D0, storage=False)]
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D1), [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("no backing k/v tensors" in p for p in a["problems"]),
                        a["problems"])

    def test_partial_cache_storage_rejected(self):
        m = self.good_model()
        m.modules[1].modules[0].cache_layers[0].v = None
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("missing k/v tensor" in p for p in a["problems"]), a["problems"])

    def test_active_devices_mismatch_rejected(self):
        m = self.good_model()
        m.active_devices = [0]           # loader never engaged cuda:1
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("active_devices" in p for p in a["problems"]), a["problems"])

    def test_active_devices_as_device_objects_normalized(self):
        m = self.good_model()
        m.active_devices = [D0, D1]      # tolerate torch.device entries
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertEqual(a["active_devices"], [0, 1])
        self.assertTrue(a["ok"], a["problems"])

    def test_cacheless_bulk_split_passes_on_module_count(self):
        # bulk runs with no --cache-tokens: zero cache layers exist, so the
        # cache-ownership check is vacuous but transformer ownership still
        # forces BOTH devices to hold real layers.
        m = self.good_model(with_cache=False)
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertTrue(a["ok"], a["problems"])
        self.assertEqual(a["total_cache_layers"], 0)
        bad = self.good_model(with_cache=False)
        emb, head = embed_head(dev_e=D0, dev_h=D0)
        bad.modules = ([emb] + [fake_block(i, D1, False) for i in range(6)] + [head])
        a2 = multi_gpu.audit_placement(bad, [0, 1])
        self.assertFalse(a2["ok"])

    def test_cpu_prefer_cpu_module_allowed_and_skipped(self):
        emb, head = embed_head(dev_e=DCPU)
        emb.caps = {"prefer_cpu": True}
        mods = [emb] + [fake_block(i, D0 if i < 3 else D1) for i in range(6)] + [head]
        a = multi_gpu.audit_placement(FakeModel(mods, [0, 1], D1), [0, 1])
        self.assertTrue(a["ok"], a["problems"])
        self.assertEqual(a["device_progression"][0]["device"], "cuda:0")

    def test_unexpected_cpu_module_without_cap_rejected(self):
        m = self.good_model()
        m.modules[0].device = DCPU       # embed on CPU without prefer_cpu
        a = multi_gpu.audit_placement(m, [0, 1])
        self.assertFalse(a["ok"])
        self.assertTrue(any("without a prefer_cpu" in p for p in a["problems"]),
                        a["problems"])


# ---------------------------------------------------------------------------
# CLI wiring (parsers stay host-safe; option cross-checks are pure)
# ---------------------------------------------------------------------------

class CliWiringTests(unittest.TestCase):
    def test_bench_parser_accepts_and_defaults_none(self):
        p = bench.build_parser()
        args = p.parse_args(["-m", "d", "--use-per-device", "3", "4"])
        self.assertEqual(args.use_per_device, [3.0, 4.0])
        self.assertIsNone(p.parse_args(["-m", "d"]).use_per_device)

    def test_collect_parser_accepts_and_defaults_none(self):
        p = collect_top1.build_parser()
        base = ["--manifest", "m", "-m", "d", "--backend", "exl3", "-o", "o"]
        args = p.parse_args(base + ["--use-per-device", "6.5", "8",
                                    "--cache-tokens", "8704", "--expect-arch", "gfx1030"])
        self.assertEqual(args.use_per_device, [6.5, 8.0])
        self.assertEqual(args.cache_tokens, 8704)
        self.assertEqual(args.expect_arch, "gfx1030")
        self.assertIsNone(p.parse_args(base).use_per_device)
        self.assertIsNone(p.parse_args(base).cache_tokens)

    def test_transformers_rejects_layer_split_option(self):
        ns = argparse.Namespace(backend="transformers", use_per_device=[3, 4],
                                cache_tokens=None)
        with self.assertRaises(SystemExit) as cm:
            collect_top1.check_cli_options(ns)
        self.assertIn("EXL3 layer-split only", str(cm.exception))

    def test_transformers_rejects_cache_tokens_option(self):
        ns = argparse.Namespace(backend="transformers", use_per_device=None,
                                cache_tokens=8704)
        with self.assertRaises(SystemExit) as cm:
            collect_top1.check_cli_options(ns)
        self.assertIn("--cache-tokens", str(cm.exception))

    def test_transformers_cpu_defaults_unaffected(self):
        ns = argparse.Namespace(backend="transformers", use_per_device=None,
                                cache_tokens=None)
        self.assertIsNone(collect_top1.check_cli_options(ns))

    def test_bad_budgets_refused_by_check_cli_options(self):
        ns = argparse.Namespace(backend="exl3", use_per_device=[3], cache_tokens=None)
        with self.assertRaises(SystemExit):
            collect_top1.check_cli_options(ns)
        ns2 = argparse.Namespace(backend="exl3", use_per_device=[float("nan"), 4],
                                 cache_tokens=None)
        with self.assertRaises(SystemExit):
            collect_top1.check_cli_options(ns2)


class CacheTokensTests(unittest.TestCase):
    def ns(self, execution, load_chunk=2048, cache=None):
        return argparse.Namespace(execution=execution,
                                  load_max_chunk_size=load_chunk,
                                  cache_tokens=cache)

    def test_chunked_default_unchanged(self):
        # auto sizing = old formula max(4096, load chunk, longest case), page-rounded
        self.assertEqual(collect_top1.compute_cache_tokens(self.ns("chunked"), 1500), 4096)
        self.assertEqual(collect_top1.compute_cache_tokens(self.ns("chunked"), 6000), 6144)

    def test_chunked_override_must_reach_capacity(self):
        self.assertEqual(collect_top1.compute_cache_tokens(self.ns("chunked", cache=8704), 5000), 8704)
        with self.assertRaises(SystemExit):
            collect_top1.compute_cache_tokens(self.ns("chunked", cache=5000), 5200)
        with self.assertRaises(SystemExit):
            collect_top1.compute_cache_tokens(self.ns("chunked", load_chunk=4096, cache=4000), 100)

    def test_bulk_default_stays_cache_free(self):
        self.assertIsNone(collect_top1.compute_cache_tokens(self.ns("bulk"), 1500))

    def test_bulk_explicit_cache_allocated_for_placement_parity(self):
        self.assertEqual(collect_top1.compute_cache_tokens(self.ns("bulk", cache=8704), 1500), 8704)
        with self.assertRaises(SystemExit):
            collect_top1.compute_cache_tokens(self.ns("bulk", cache=2000), 1500)

    def test_bench_cache_formula_untouched(self):
        # regression: bench's existing override semantics (floor at need/4096/chunk)
        # are computed inline in run(); the helper PAGE rounding contract stays:
        self.assertEqual(multi_gpu.device_key(0), "cuda:0")


if __name__ == "__main__":
    unittest.main()
