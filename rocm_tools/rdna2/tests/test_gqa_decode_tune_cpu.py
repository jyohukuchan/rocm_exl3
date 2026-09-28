#!/usr/bin/env python3
"""CPU-only tests for the gfx1030 FP16 BC decode-attention tuning helper.

No GPU, no exllamav3 import (bc_attn pulls in the compiled extension), no
mocked model execution: the helper's real eligibility / geometry / signature /
alignment code runs against synthetic decision inputs and plain integers --
the part whose mistakes would either silently mis-shape a CUDA-graph slot or
promise an alignment the launch does not honor.

The device probes are driven through their documented caches (_is_hip,
_arch_by_index) so decode_tune_enabled runs end-to-end on a CPU host; nothing
touches a driver.

Run from the repo root (or anywhere):
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
HELPER = REPO / "exllamav3" / "rocm_py" / "gqa_decode_tune.py"
BC_ATTN = REPO / "exllamav3" / "modules" / "attention_fn" / "bc_attn.py"

try:
    import torch
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False


def load_helper():
    """Load the module from its file without importing the exllamav3
    package (top level imports stdlib only, by design)."""
    spec = importlib.util.spec_from_file_location("gdt_under_test", str(HELPER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gdt = load_helper()   # no torch needed for the pure surface


def cdiv(a, b):
    return -(-a // b)


def decision(**over):
    """The proven-eligible decode shape (D: Qwen3-8B 4bpw), as _configure's
    eligibility inputs; override one field per negative test."""
    d = dict(is_hip = True, gcn_arch_name = "gfx1030", env_on = True,
             q_len = 1, head_dim = 128, num_q_heads = 32, num_kv_heads = 8,
             k_bits = 0, v_bits = 0, gate_mode = 0, qsa = False)
    d.update(over)
    return d


class EligibilityTests(unittest.TestCase):

    def test_verified_shapes_are_eligible(self):
        self.assertTrue(gdt.decode_tune_eligible(**decision()))                          # D: kvh 8
        self.assertTrue(gdt.decode_tune_eligible(**decision(num_kv_heads = 4)))          # M: kvh 4

    def test_other_shapes_decline(self):
        negatives = [
            dict(q_len = 2), dict(q_len = 16),
            dict(head_dim = 64), dict(head_dim = 256),
            dict(num_q_heads = 16), dict(num_q_heads = 64),
            dict(num_kv_heads = 2), dict(num_kv_heads = 16),
            dict(k_bits = 4), dict(v_bits = 4), dict(k_bits = 0, v_bits = 2),
            dict(gate_mode = 1), dict(gate_mode = 2), dict(gate_mode = 3),
            dict(qsa = True),
        ]
        for over in negatives:
            self.assertFalse(gdt.decode_tune_eligible(**decision(**over)),
                             f"{over} must decline")

    def test_other_builds_devices_and_env_decline(self):
        self.assertFalse(gdt.decode_tune_eligible(**decision(is_hip = False)))
        self.assertFalse(gdt.decode_tune_eligible(**decision(env_on = False)))
        for arch in ("gfx1031", "gfx1035", "gfx1100", "gfx1151", "gfx942", "", "sm_86"):
            self.assertFalse(gdt.decode_tune_eligible(**decision(gcn_arch_name = arch)), arch)

    def test_arch_accepts_exact_and_colon_suffix_only(self):
        self.assertTrue(gdt.arch_supported("gfx1030"))
        self.assertTrue(gdt.arch_supported("gfx1030:xnack-"))
        self.assertFalse(gdt.arch_supported("gfx1031"))      # unmeasured sibling
        self.assertFalse(gdt.arch_supported("gfx10300"))     # not a prefix match
        self.assertFalse(gdt.arch_supported(""))


class GeometryTests(unittest.TestCase):

    def test_next_power_of_2(self):
        self.assertEqual(gdt.next_power_of_2(1), 1)
        self.assertEqual(gdt.next_power_of_2(3), 4)
        self.assertEqual(gdt.next_power_of_2(8), 8)

    def test_narrowing_gives_4_rows_for_d_and_8_for_m(self):
        # q_len 1 -> block_m 1 -> original block_h 16; group = 32//8 = 4 (D) or
        # 32//4 = 8 (M).
        self.assertEqual(gdt.narrowed_block_h(16, 4), 4)
        self.assertEqual(gdt.narrowed_block_h(16, 8), 8)

    def test_narrowing_never_exceeds_and_never_inflates(self):
        self.assertEqual(gdt.narrowed_block_h(2, 8), 2)     # min keeps the original
        self.assertEqual(gdt.narrowed_block_h(16, 5), 8)    # partial groups pad to pow2

    def test_h_blocks_stays_one_so_the_cpp_grid_derivation_matches(self):
        # attention.cpp re-derives programs with the ORIGINAL block_h = 16
        # (h_blocks = cdiv(group, 16) = 1); the tuned Python geometry must land
        # on the same program count, else the launched grid and the partial
        # buffers disagree with the baked kernel indexing.
        for bsz in (1, 2, 8):
            for kvh in (4, 8):
                group = 32 // kvh
                block_m = 1
                orig_h = max(16 // block_m, 1)
                tuned_h = gdt.narrowed_block_h(orig_h, group)
                self.assertEqual(cdiv(group, orig_h), 1)
                self.assertEqual(cdiv(group, tuned_h), 1, "h_blocks must stay 1")
                self.assertEqual(bsz * kvh * cdiv(group, tuned_h),
                                 bsz * kvh * cdiv(group, orig_h),
                                 "program count must be unchanged")
                self.assertEqual(block_m * tuned_h, group,
                                 "the decode tile rows cover the group exactly")


class HintTests(unittest.TestCase):
    """The AOT signature rewrite: exactly the promised entries gain ':16',
    everything the launch cannot guarantee (block table width, seqlens,
    num_splits, dead scale/sink args, constexprs) stays unhinted."""

    SPLIT_SIG = {
        "q": "*fp16", "k_cache": "*fp16", "v_cache": "*fp16",
        "block_table": "*i32", "cache_seqlens": "*i32", "out": "*fp16",
        "partial_o": "*fp32", "partial_ml": "*fp32",
        "k_scales": "*fp16", "v_scales": "*fp16", "h32": "*fp16",
        "split_len": "i32", "num_pages_per_seq": "i32", "num_splits": "i32",
        "sinks": "*fp32", "QCK": "constexpr", "BLOCK_H": "constexpr",
    }
    COMBINE_SIG = {
        "partial_o": "*fp32", "partial_ml": "*fp32", "out": "*fp16", "h32": "*fp16",
        "num_splits": "i32", "sinks": "*fp32", "QCV": "constexpr",
    }

    def test_split_signature_hints_only_the_proven_args(self):
        out = gdt.with_alignment_hints(self.SPLIT_SIG, gdt.SPLIT_ALIGNED_ARGS)
        for name in ("q", "k_cache", "v_cache", "out", "partial_o", "partial_ml",
                     "split_len"):
            self.assertTrue(out[name].endswith(":16"), name)
        for name in ("block_table", "cache_seqlens", "num_pages_per_seq", "num_splits",
                     "k_scales", "v_scales", "h32", "sinks", "QCK", "BLOCK_H"):
            self.assertFalse(out[name].endswith(":16"), name)

    def test_combine_signature_hints_only_the_proven_args(self):
        out = gdt.with_alignment_hints(self.COMBINE_SIG, gdt.COMBINE_ALIGNED_ARGS)
        self.assertEqual(out["partial_o"], "*fp32:16")
        self.assertEqual(out["partial_ml"], "*fp32:16")
        self.assertEqual(out["out"], "*fp16:16")
        self.assertEqual(out["h32"], "*fp16")           # dead/fp16-dummy args stay unhinted
        self.assertEqual(out["num_splits"], "i32")      # arbitrary value: no promise
        self.assertEqual(out["sinks"], "*fp32")

    def test_idempotent_and_non_destructive(self):
        once = gdt.with_alignment_hints(self.SPLIT_SIG, gdt.SPLIT_ALIGNED_ARGS)
        twice = gdt.with_alignment_hints(once, gdt.SPLIT_ALIGNED_ARGS)
        self.assertEqual(once, twice)
        self.assertEqual(self.SPLIT_SIG["q"], "*fp16")   # input dict untouched
        self.assertNotIn(":16:16", " ".join(twice.values()))


class AlignmentTests(unittest.TestCase):

    def test_pointer_set_must_all_be_16_divisible(self):
        self.assertTrue(gdt.pointers_aligned([0x0, 16, 512, 0x7f000_0000]))
        self.assertFalse(gdt.pointers_aligned([16, 24]))
        self.assertFalse(gdt.pointers_aligned([512, 2]))

    def test_empty_set_promises_nothing(self):
        # callers pass the exact pointer set the hints mark; an empty list is a
        # wiring bug, not a vacuous promise
        self.assertFalse(gdt.pointers_aligned([]))

    def test_fp16_element_offset_breaks_the_promise(self):
        if not HAVE_TORCH:
            self.skipTest("torch not importable on this host")
        base = torch.empty((1024,), dtype = torch.float)
        self.assertEqual(base.data_ptr() % 16, 0)                 # allocator base
        self.assertTrue(gdt.pointers_aligned([base.data_ptr()]))
        half = base[2:5].view(torch.half)                         # 2 fp32 elems = 8B in
        self.assertEqual(half.data_ptr() % 16, 8)                 # -> must be declined
        self.assertFalse(gdt.pointers_aligned([half.data_ptr()]))


@unittest.skipUnless(HAVE_TORCH, "torch not importable on this host")
class WrapperTests(unittest.TestCase):
    """decode_tune_enabled end-to-end on a CPU host: env switch read from
    os.environ, build/device probed through their caches (no driver call)."""

    def run_enabled(self, device, hip, arch_cache, **shape):
        with mock.patch.dict(os.environ, {"EXL3_ROCM_GQA_TUNE": "1"}):
            old_hip, old_cache = gdt._is_hip, gdt._arch_by_index
            gdt._is_hip, gdt._arch_by_index = hip, dict(arch_cache)
            try:
                return gdt.decode_tune_enabled(device, **shape)
            finally:
                gdt._is_hip, gdt._arch_by_index = old_hip, old_cache

    SHAPE = dict(q_len = 1, head_dim = 128, num_q_heads = 32, num_kv_heads = 8,
                 k_bits = 0, v_bits = 0, gate_mode = 0, qsa = False)

    def test_gfx1030_cached_arch_enables_without_touching_a_driver(self):
        self.assertTrue(self.run_enabled(torch.device("cuda", 7), True,
                                         {7: "gfx1030:xnack-"}, **self.SHAPE))
        self.assertTrue(self.run_enabled(torch.device("cuda", 7), True,
                                         {7: "gfx1030"}, q_len = 1, head_dim = 128,
                                         num_q_heads = 32, num_kv_heads = 4,
                                         k_bits = 0, v_bits = 0, gate_mode = 0, qsa = False))

    def test_other_arch_cached_declines(self):
        self.assertFalse(self.run_enabled(torch.device("cuda", 7), True,
                                          {7: "gfx1151"}, **self.SHAPE))
        self.assertFalse(self.run_enabled(torch.device("cuda", 7), True, {7: ""},
                                          **self.SHAPE))

    def test_non_hip_build_and_unprobeable_devices_decline(self):
        self.assertFalse(self.run_enabled(torch.device("cuda", 7), False,
                                          {7: "gfx1030"}, **self.SHAPE))
        self.assertFalse(self.run_enabled(torch.device("cpu"), True, {}, **self.SHAPE))

    def test_env_optout_short_circuits_the_wrapper(self):
        with mock.patch.dict(os.environ, {"EXL3_ROCM_GQA_TUNE": "0"}):
            old_hip, old_cache = gdt._is_hip, gdt._arch_by_index
            gdt._is_hip, gdt._arch_by_index = True, {7: "gfx1030"}
            try:
                self.assertFalse(gdt.decode_tune_enabled(
                    torch.device("cuda", 7), **self.SHAPE))
            finally:
                gdt._is_hip, gdt._arch_by_index = old_hip, old_cache

    def test_env_semantics(self):
        for v, want in ((None, True), ("1", True), ("0", False), ("", False),
                        ("false", False), ("False", False), (" 0 ", False), ("2", True)):
            env = {} if v is None else {"EXL3_ROCM_GQA_TUNE": v}
            self.assertEqual(gdt.env_enabled(env), want, repr(v))


class WiringTests(unittest.TestCase):
    """Source-level guards that _configure uses the helper as contracted:
    the hint decision is checked BEFORE the signature compiles, both tuned
    signatures are hinted together, and the narrowing feeds the same
    block_rows that sizes the scratch buffers."""

    SRC = BC_ATTN.read_text(encoding = "utf-8")

    def test_helper_wired_into_configure(self):
        self.assertIn("from ...rocm_py import gqa_decode_tune", self.SRC)
        self.assertIn("gqa_decode_tune.decode_tune_enabled", self.SRC)
        self.assertIn("gqa_decode_tune.narrowed_block_h(block_h, group_size)", self.SRC)
        self.assertEqual(self.SRC.count("with_alignment_hints"), 2)   # split + combine
        self.assertNotIn("tune_align", self.SRC)

    def test_hip_branch_gates_the_helper_import(self):
        # CUDA builds must not import rocm_py from this module at all
        conf = self.SRC[self.SRC.index("def _configure"):]
        self.assertLess(conf.index('getattr(torch.version, "hip", None)'),
                        conf.index("from ...rocm_py import gqa_decode_tune"))

    def test_cache_pointers_gate_tuning_before_geometry(self):
        conf = self.SRC[self.SRC.index("def _configure"):]
        conf = conf[:conf.index("def _qsa_sparse_geometry")]
        self.assertLess(conf.index("self.cache_k.data_ptr()"),
                        conf.index("block_h = gqa_decode_tune.narrowed_block_h"),
                        "an unaligned cache must retain the original geometry, "
                        "which is fixed before any pointer check could revoke it")

    def test_allocation_and_compile_order_is_untouched(self):
        # Statics still come AFTER the kernels compile (no runtime-order change
        # for untuned CUDA / other shapes): xp.zero_() and the buffer fetches
        # keep their original position between kernel compile and registration.
        conf = self.SRC[self.SRC.index("def _configure"):]
        conf = conf[:conf.index("def _qsa_sparse_geometry")]
        self.assertLess(conf.index("k_split = _compile_kernel"),
                        conf.index("# Static intermediates"))
        self.assertLess(conf.index("k_update = _compile_kernel(dev, _paged_kv_update_kernel"),
                        conf.index("R = bsz * q_len"))

    def test_owned_statics_asserted_before_registration(self):
        conf = self.SRC[self.SRC.index("def _configure"):]
        conf = conf[:conf.index("def _qsa_sparse_geometry")]
        for token in ("partial_ml = g_tensor_cache.get_bucketed", "not 16B aligned",
                      "self.bc.configure_slot("):
            self.assertIn(token, conf)
        self.assertLess(conf.index("partial_ml = g_tensor_cache.get_bucketed"),
                        conf.index("not 16B aligned"))
        self.assertLess(conf.index("not 16B aligned"),
                        conf.index("self.bc.configure_slot("),
                        "the alignment fail-clear must precede slot registration")

    def test_narrowed_tile_flows_into_scratch_sizes(self):
        conf = self.SRC[self.SRC.index("def _configure"):]
        self.assertLess(conf.index("block_h = gqa_decode_tune.narrowed_block_h"),
                        conf.index("block_rows = block_m * block_h"))
        self.assertIn("pn_o = programs * splits_cap * block_rows * hd_pad", conf)

    def test_helper_top_level_is_stdlib_only(self):
        src = HELPER.read_text(encoding = "utf-8")
        for line in src.splitlines():
            if line.startswith(("import ", "from ")) and "future" not in line:
                self.fail(f"top-level import outside stdlib: {line!r}")

    def test_helper_is_loadable_standalone(self):
        mod = load_helper()                             # no exllamav3 package init
        self.assertEqual(mod.ENV_SWITCH, "EXL3_ROCM_GQA_TUNE")
        self.assertEqual(list(mod.SPLIT_ALIGNED_ARGS)[-1], "split_len")


if __name__ == "__main__":
    unittest.main()
