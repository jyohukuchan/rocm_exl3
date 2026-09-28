#!/usr/bin/env python3
"""CPU-only tests for summarize_rocprof.py (no GPU, no torch, no ROCm).

These exercise the REAL helper behavior end-to-end: classification of the
exact kernel-name shapes the pilot traces show (mangled ``_ZL25exl3_...``
symbols whose length digits sit before ``exl3``, ``[clone .intern...]`` and
``(anonymous namespace)`` wrappers, triton python names, Tensile tiles, the
``__amd_rocclr_copyBuffer`` driver kernel), interval-union math, marker
parsing, window assignment with boundary-crossing detection, duplicate
file/row rejection, per-process isolation, empty/error refusals, known
synthetic windows with hand-checked numbers, aggregation medians, and the
optional --run-json cross-check. Fixture trees are written as real CSVs in
temporary directories and run through summarize_rocprof.build_summary /
main, not through string inspection.

Run from the repo root (or anywhere):
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2 import summarize_rocprof as sr

MARKER_HEADER = ["Domain", "Function", "Process_Id", "Thread_Id",
                 "Correlation_Id", "Start_Timestamp", "End_Timestamp"]
KERNEL_HEADER = ["Kind", "Agent_Id", "Queue_Id", "Stream_Id", "Thread_Id",
                "Dispatch_Id", "Kernel_Id", "Kernel_Name", "Correlation_Id",
                "Start_Timestamp", "End_Timestamp", "LDS_Block_Size",
                "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count",
                "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
                "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z"]
COPY_HEADER = ["Kind", "Agent_Id", "Thread_Id", "Memory_Copy_Size",
               "Start_Timestamp", "End_Timestamp", "Correlation_Id"]


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", newline = "", encoding = "utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def marker_row(domain, fn, pid, s, e, cid = 1):
    return [domain, fn, str(pid), str(pid), str(cid), str(s), str(e)]


def kernel_row(pid, name, s, e, dispatch = 1):
    return ["KERNEL_DISPATCH", "Agent 1", "1", "0", str(pid), str(dispatch),
            str(dispatch), name, str(dispatch), str(s), str(e),
            "0", "0", "120", "0", "128", "32", "1", "1", "512", "16", "1"]


def copy_row(pid, s, e, size = 4096, kind = "MEMCPY_ACTIVITY"):
    return [kind, "Agent 1", str(pid), str(size), str(s), str(e), str(s)]

DECODE_M1 = "stage=decode;context=2048;repeat=1;tokens=4"

def marker_decode(pid, s = 1000, e = 2000):
    return marker_row(sr.MARKER_RANGE_DOMAIN, DECODE_M1, pid, s, e)

def kernel_exl3(pid, s = 1100, e = 1400, dispatch = 1):
    return kernel_row(pid, "void exl3_gemv_dot_kernel<6>(x)", s, e, dispatch)


class TestNormalizeAndClassify(unittest.TestCase):
    """Real names captured from the pilot traces (see runs/timing-breakdown)."""

    def test_moe_prefill_histogram_and_sort_primitives(self):
        self.assertEqual(sr.classify_kernel("void at::cuda::kernelHistogram1D<long>(...)"),
                         (sr.CAT_OTHER, "histogram"))
        self.assertEqual(sr.classify_kernel("void rocprim::ROCPRIM_400200_NS::detail::trampoline_kernel<sort>(...)"),
                         (sr.CAT_OTHER, "parallel_primitives"))
        self.assertEqual(sr.classify_kernel("__amd_rocclr_gwsInit"),
                         (sr.CAT_OTHER, "runtime_support"))

    def test_mangled_exl3_gemv_family_length_prefix(self):
        # The digits sit right before "exl3"; a word-boundary pattern anchored
        # there would fail. Length-prefix folding must recover the base name.
        cases = {
            "_ZL25exl3_gemv_mr_had_in_multiPK6__halfPvPKlS1_P16Exl3GemvMrParamsPKmPS_iiiiii.intern.8f6de1e0":
                (sr.CAT_DEQUANT, "hadamard"),          # had_in BEFORE fused gemv
            "_ZL26exl3_gemv_mr_had_in_singlePK6__halfPKtPvS1_PS_S1_P16Exl3GemvMrParamsii.intern.8f6dec6a":
                (sr.CAT_DEQUANT, "hadamard"),
            "_ZL28exl3_gemv_rdna_had_in_kernelPK6__halfPS_S1_i.intern.586386e8f97fd309":
                (sr.CAT_DEQUANT, "hadamard"),
            "_ZL34exl3_gemv_rdna_had_out_half_kernelPK6__halfPS_S1_i.intern.586386e8f97fd309":
                (sr.CAT_DEQUANT, "hadamard"),          # had_out too
            "_ZL22exl3_gemv_mr_dot_multiILi3ELb0ELi0ELi4ELi1EEvP16Exl3GemvMrParamsPKmS3_PK6__halfiiii":
                (sr.CAT_DEQUANT, "fused_quantized_gemv_gemm"),
            "_ZL23exl3_gemv_mr_dot_singleILi3ELb1ELi0ELi8ELi1EEvP16Exl3GemvMrParamsiii.intern.8f6dec6a":
                (sr.CAT_DEQUANT, "fused_quantized_gemv_gemm"),
        }
        for name, want in cases.items():
            self.assertEqual(sr.classify_kernel(name), want, msg = name)

    def test_demangled_exl3_and_reconstruct(self):
        self.assertEqual(sr.classify_kernel(
            "void exl3_gemv_dot_kernel<6, false, 0, 4>(__half const*, unsigned short const*, void*, i...)"),
            (sr.CAT_DEQUANT, "fused_quantized_gemv_gemm"))
        self.assertEqual(sr.classify_kernel(
            "void exl3_mgemv_dot_kernel_splitk<3, true, 0, 4>(__half const*, void*, long const*, __ha...)"),
            (sr.CAT_DEQUANT, "fused_quantized_gemv_gemm"))
        self.assertEqual(sr.classify_kernel("reconstruct_kernel<4, 1>"),
                         (sr.CAT_DEQUANT, "standalone_reconstruction"))
        # reconstruct* is standalone reconstruction even when it carries the
        # hadamard codebook (the reconstruct rule precedes the hadamard one).
        self.assertEqual(sr.classify_kernel("void reconstruct_had_kernel<4, 2>(half*...)"),
                         (sr.CAT_DEQUANT, "standalone_reconstruction"))
        self.assertEqual(sr.classify_kernel("void had_hf_r_128_kernel<true, true>(const half*...)"),
                         (sr.CAT_DEQUANT, "hadamard"))

    def test_moe_gather_is_other_before_generic_moe(self):
        self.assertEqual(sr.classify_kernel("void exl3_moe_gather_kernel(...)"),
                         (sr.CAT_OTHER, "moe_support"))
        self.assertEqual(sr.classify_kernel("exl3_moe_scatter_kernel"),
                         (sr.CAT_OTHER, "moe_support"))
        self.assertEqual(sr.classify_kernel("moe_split_collect_add_kernel"),
                         (sr.CAT_OTHER, "moe_support"))
        # actual fused MoE compute stays dequant_related/fused
        self.assertEqual(sr.classify_kernel("exl3_moe_kernel_k1_n128_cb2"),
                         (sr.CAT_DEQUANT, "fused_quantized_moe"))
        self.assertEqual(sr.classify_kernel("exl3_moe_coop_kernel_a_k2"),
                         (sr.CAT_DEQUANT, "fused_quantized_moe"))

    def test_routing_gemv_is_routing_not_gemm(self):
        self.assertEqual(sr.classify_kernel(
            "routing_gemv_kernel(__half const*, __half const*, __half*, int, int)"),
            (sr.CAT_OTHER, "routing"))
        self.assertEqual(sr.classify_kernel(
            "routing_std_topk_kernel(__half const*, long*, __half*, __hip_bfloat16 const*, __half con...)"),
            (sr.CAT_OTHER, "routing"))
        self.assertEqual(sr.classify_kernel("routing_sel_norm_kernel"),
                         (sr.CAT_OTHER, "routing"))

    def test_attention_kernels(self):
        for name in ("_paged_attn_decode_split_kernel",
                     "_paged_attn_decode_combine_kernel",
                     "_paged_attn_prefill_kernel"):
            self.assertEqual(sr.classify_kernel(name),
                             (sr.CAT_ATTENTION, "paged_attention"), msg = name)
        self.assertEqual(sr.classify_kernel("_paged_kv_update_kernel"),
                         (sr.CAT_ATTENTION, "kv_cache_update"))
        self.assertEqual(sr.classify_kernel("void paged_kv_update_vec8_kernel(...)"),
                         (sr.CAT_ATTENTION, "kv_cache_update"))
        self.assertEqual(sr.classify_kernel("void kv_cache_update_kernel_paged(...)"),
                         (sr.CAT_ATTENTION, "kv_cache_update"))
        # rope applies the QK norms (attn.py passes q_norm/k_norm into it)
        self.assertEqual(sr.classify_kernel(
            "void rope_kernel<2, false>(__half const*, __half*, __half const*, __half*, float const*,...)"),
            (sr.CAT_ATTENTION, "rope_qknorm"))
        self.assertEqual(sr.classify_kernel("void fused_qk_norm_kernel(...)"),
                         (sr.CAT_ATTENTION, "rope_qknorm"))

    def test_generic_norm_stays_other(self):
        self.assertEqual(sr.classify_kernel(
            "void rms_norm_kernel<0, float, __half, __half, float>(float const*, __half const*, __hal...)"),
            (sr.CAT_OTHER, "generic_norm"))
        self.assertEqual(sr.classify_kernel("void rms_norm_kernel<2, float, __half, __half, float>(...)"),
                         (sr.CAT_OTHER, "generic_norm"))

    def test_softmax_is_not_attention(self):
        # attention softmax is fused inside the _paged_attn kernels; a
        # standalone softmax belongs to the sampler path, never attention.
        self.assertEqual(sr.classify_kernel(
            "void at::native::cunn_SoftMaxForward<4, float, float, float, at::native::SoftMaxForwardEpilogue>(...)"),
            (sr.CAT_OTHER, "sampling"))
        self.assertEqual(sr.classify_kernel("void softmax_warp_backward(...)"),
                         (sr.CAT_OTHER, "sampling"))

    def test_sampling_activation_copy_dense_and_fallback(self):
        self.assertEqual(sr.classify_kernel(
            "void (anonymous namespace)::fs_partial_argmax_kernel<__half, 0>(__half const*, __half co...)"),
            (sr.CAT_OTHER, "sampling"))
        self.assertEqual(sr.classify_kernel(
            "(anonymous namespace)::fs_finalize_kernel(ValIdx const*, unsigned long*, int) [clone .intern.8f6de]"),
            (sr.CAT_OTHER, "sampling"))
        self.assertEqual(sr.classify_kernel("void argmax_sample_kernel(...)"),
                         (sr.CAT_OTHER, "sampling"))
        self.assertEqual(sr.classify_kernel("void act_mul_kernel_h<0>(__half const*, __half const*, __half*, float, unsigned long)"),
                         (sr.CAT_OTHER, "activation"))
        # driver copy kernel is a REAL GPU kernel in the kernel trace
        self.assertEqual(sr.classify_kernel("__amd_rocclr_copyBuffer"),
                         (sr.CAT_OTHER, "driver_copy_kernel"))
        self.assertEqual(sr.classify_kernel(
            "Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_..."),
            (sr.CAT_OTHER, "dense_gemm"))
        self.assertEqual(sr.classify_kernel("void hgemm_f16acc_kernel(...)"),
                         (sr.CAT_OTHER, "dense_gemm"))
        # aten torch ops with informative subcategory
        self.assertEqual(sr.classify_kernel(
            "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, st..."),
            (sr.CAT_OTHER, "aten_torch_op"))
        self.assertEqual(sr.classify_kernel(
            "void at::native::unrolled_elementwise_kernel<at::native::FillFunctor<c10::Half>, std::ar..."),
            (sr.CAT_OTHER, "memory_fill"))
        self.assertEqual(sr.classify_kernel("mystery_unseen_engine_kernel"),
                         (sr.CAT_OTHER, "other_unclassified"))

    def test_primary_categories_are_mutually_exclusive(self):
        # every rule in the table maps to exactly one primary category
        for _kind, _pat, cat, _sub in sr.CLASSIFICATION:
            self.assertIn(cat, sr.CATEGORIES)

    def test_interval_union_math(self):
        self.assertEqual(sr.interval_union_ns([]), 0)
        self.assertEqual(sr.interval_union_ns([(10, 20)]), 10)
        self.assertEqual(sr.interval_union_ns([(10, 20), (15, 30)]), 20)   # overlap once
        self.assertEqual(sr.interval_union_ns([(0, 100), (20, 30)]), 100)  # nested
        self.assertEqual(sr.interval_union_ns([(10, 20), (20, 30)]), 20)   # touching
        self.assertEqual(sr.interval_union_ns([(10, 20), (30, 40)]), 20)   # disjoint
        self.assertEqual(sr.interval_union_ns([(5, 5), (10, 13)]), 3)      # zero-length skipped
        # copies + kernels together must union, not sum: [100,300] u [400,500]
        self.assertEqual(sr.interval_union_ns([(100, 300), (200, 250), (400, 500)]), 300)
        self.assertEqual(sr.interval_union_ns([(100, 300), (200, 250), (400, 500)]),
                         200 + 100)     # nested [200,250] adds nothing to the sum


class SyntheticTraceMixin:
    """Build real rocprofv3-shaped trace trees in temp dirs."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.trace_dir = Path(self._tmp.name) / "marker-trace"
        self.addCleanup(self._tmp.cleanup)

    def write_unit(self, pid, markers, kernels, copies = None, subdir = "uuidA"):
        d = self.trace_dir / subdir
        d.mkdir(parents = True, exist_ok = True)
        write_csv(d / f"{pid}_marker_api_trace.csv", MARKER_HEADER, markers)
        write_csv(d / f"{pid}_kernel_trace.csv", KERNEL_HEADER, kernels)
        if copies is not None:
            write_csv(d / f"{pid}_memory_copy_trace.csv", COPY_HEADER, copies)

    def summarize(self, run = None):
        return sr.build_summary(self.trace_dir, run)


# ---------------------------------------------------------------------------
# happy path: one synthetic known window with hand-checked numbers
#
#   pid 1001:
#     prefill window [1000, 5000]        wall 4000 ns
#         kernel exl3_gemv      [1100, 4000]  dur 2900
#     decode window  [20000, 24000]      wall 4000 ns, tokens 4
#         fused gemv A          [20100, 20600]  500
#         fused gemv B          [20300, 21000]  700  (A|B union 900, overlap 300)
#         rms_norm              [21500, 21700]  200
#         paged attn            [22000, 23000]  1000
#         driver copy kernel    [23100, 23200]  100
#         memory copy (HtoD)    [21400, 21450]  50
#     outside kernel            [7000, 7500]   500
#
#   decode: kernel_sum 2500, kernel_union 2200, overlap 300, busy 2250,
#           non_gpu 1750, pct (denominator 2500): dequant 48%, attn 40%,
#           other 12%, per-token: sum 625, dequant 300.
# ---------------------------------------------------------------------------

class TestSyntheticKnownWindow(SyntheticTraceMixin, unittest.TestCase):
    DECODE = "stage=decode;context=2048;repeat=1;tokens=4"
    PREFILL = "stage=prefill;context=2048;repeat=1;tokens=1"

    def build(self):
        markers = [
            marker_row(sr.MARKER_CONTROL_DOMAIN, "roctxProfilerPause", 1001, 500, 550),
            marker_row(sr.MARKER_CONTROL_DOMAIN, "roctxProfilerResume", 1001, 900, 960),
            marker_row(sr.MARKER_RANGE_DOMAIN, self.PREFILL, 1001, 1000, 5000),
            marker_row(sr.MARKER_CONTROL_DOMAIN, "roctxProfilerPause", 1001, 5100, 5150),
            marker_row(sr.MARKER_CONTROL_DOMAIN, "roctxProfilerResume", 1001, 19000, 19050),
            marker_row(sr.MARKER_RANGE_DOMAIN, self.DECODE, 1001, 20000, 24000),
            marker_row(sr.MARKER_CONTROL_DOMAIN, "roctxProfilerPause", 1001, 24500, 24550),
        ]
        kernels = [
            kernel_row(1001, "void exl3_gemv_dot_kernel<6, false, 0, 4>(x)", 1100, 4000, 1),
            kernel_row(1001, "_ZL22exl3_gemv_mr_dot_multiILi3Ev", 20100, 20600, 2),
            kernel_row(1001, "_ZL22exl3_gemv_mr_dot_multiILi3Ev", 20300, 21000, 3),
            kernel_row(1001, "void rms_norm_kernel<0>(x)", 21500, 21700, 4),
            kernel_row(1001, "_paged_attn_decode_split_kernel", 22000, 23000, 5),
            kernel_row(1001, "__amd_rocclr_copyBuffer", 23100, 23200, 6),
            kernel_row(1001, "void exl3_gemv_dot_kernel<6, false, 0, 4>(x)", 7000, 7500, 7),
        ]
        copies = [copy_row(1001, 21400, 21450)]
        self.write_unit(1001, markers, kernels, copies)

    def test_happy_window_numbers(self):
        self.build()
        res = self.summarize()
        self.assertTrue(res["ok"], res["errors"])
        self.assertEqual(len(res["windows"]), 2)
        decode = next(w for w in res["windows"] if w["stage"] == "decode")
        prefill = next(w for w in res["windows"] if w["stage"] == "prefill")

        self.assertEqual(decode["marker_wall_ns"], 4000)
        self.assertEqual(decode["kernel_count"], 5)
        self.assertEqual(decode["kernel_sum_ns"], 2500)
        self.assertEqual(decode["kernel_union_ns"], 2200)
        self.assertEqual(decode["kernel_overlap_ns"], 300)     # sum - union, no double count
        self.assertEqual(decode["copy_count"], 1)
        self.assertTrue(decode["copy_measured"])
        self.assertEqual(decode["copy_sum_ns"], 50)
        self.assertEqual(decode["copy_union_ns"], 50)
        self.assertEqual(decode["gpu_busy_union_ns"], 2250)    # union(kernels, copies)
        self.assertEqual(decode["non_gpu_interval_ns"], 1750)  # wall - busy
        self.assertNotIn("anomaly", decode)

        cats = decode["categories"]
        self.assertEqual(cats[sr.CAT_DEQUANT]["kernel_sum_ns"], 1200)
        self.assertEqual(cats[sr.CAT_ATTENTION]["kernel_sum_ns"], 1000)
        self.assertEqual(cats[sr.CAT_OTHER]["kernel_sum_ns"], 300)  # norm + driver copy
        # percentages are over the total KERNEL SUM (2500), never over wall
        self.assertEqual(cats[sr.CAT_DEQUANT]["pct_of_window_kernel_sum"], 48.0)
        self.assertEqual(cats[sr.CAT_ATTENTION]["pct_of_window_kernel_sum"], 40.0)
        self.assertEqual(cats[sr.CAT_OTHER]["pct_of_window_kernel_sum"], 12.0)
        # subcategories: had_in kernels were NOT billed as fused gemv in this
        # fixture (none present), but fused gemv subcategory carries both hits
        self.assertEqual(cats[sr.CAT_DEQUANT]["subcategories"]
                         ["fused_quantized_gemv_gemm"]["kernel_sum_ns"], 1200)

        # decode per-token normalization by marker.tokens = 4
        self.assertEqual(decode["per_token"]["kernel_sum_per_token_ns"], 625.0)
        self.assertEqual(cats[sr.CAT_DEQUANT]["per_token_ns"], 300.0)

        # prefill: whole-prompt total; per-input-token optional via context
        self.assertEqual(prefill["kernel_sum_ns"], 2900)
        self.assertEqual(prefill["per_token"]["per_input_token_kernel_sum_ns"],
                         round(2900 / 2048, 3))

        # control markers counted + ignored, never selected
        fileinfo = next(iter(res["files"].values()))
        self.assertEqual(fileinfo["control_markers_ignored"], 5)
        self.assertEqual(fileinfo["selected_ranges"], 2)

        # outside kernel counted separately, absent from stage totals
        self.assertEqual(len(res["outside_selected_ranges"]), 1)
        out = res["outside_selected_ranges"][0]
        self.assertEqual(out["kernels_outside_count"], 1)
        self.assertEqual(out["kernels_outside_sum_ns"], 500)
        self.assertNotIn("kernel_sum_ns", {"2900 + 2500": 0})  # (self-check, trivial)
        # neither window total includes the outside kernel:
        self.assertEqual(prefill["kernel_sum_ns"] + decode["kernel_sum_ns"], 5400)

    def test_kernel_name_table_and_unclassified(self):
        self.build()
        res = self.summarize()
        decode = next(w for w in res["windows"] if w["stage"] == "decode")
        names = {r["name"]: r for r in decode["per_kernel"]}
        self.assertEqual(names["_ZL22exl3_gemv_mr_dot_multiILi3Ev"]["count"], 2)
        self.assertEqual(names["_ZL22exl3_gemv_mr_dot_multiILi3Ev"]["kernel_sum_ns"], 1200)
        self.assertEqual(names["__amd_rocclr_copyBuffer"]["category"], sr.CAT_OTHER)
        self.assertEqual(res["unclassified_kernel_names"], [])


class TestAggregation(SyntheticTraceMixin, unittest.TestCase):
    def test_median_distribution_across_repeats(self):
        markers = [marker_row(sr.MARKER_RANGE_DOMAIN,
                              f"stage=decode;context=2048;repeat={r};tokens=4",
                              1001, 20000 * r, 20000 * r + 1000) for r in (1, 2, 3)]
        # dequant durations 100 / 300 / 200 -> median 200
        kernels = [kernel_row(1001, "void exl3_gemv_dot_kernel<6>(x)",
                              20000 * r + 10, 20000 * r + 10 + dur)
                   for r, dur in ((1, 100), (2, 300), (3, 200))]
        self.write_unit(1001, markers, kernels)
        res = self.summarize()
        self.assertTrue(res["ok"], res["errors"])
        agg = res["aggregates"]["stage=decode;context=2048"]
        self.assertEqual(agg["windows_n"], 3)
        self.assertEqual(agg["repeats"], [1, 2, 3])
        self.assertFalse(agg["multiple_processes"])
        dequant = agg["categories"][sr.CAT_DEQUANT]["per_window"]
        self.assertEqual(dequant["median_ns"], 200)
        self.assertEqual(dequant["min_ns"], 100)
        self.assertEqual(dequant["max_ns"], 300)
        self.assertEqual(dequant["values_ns"], [100, 200, 300])
        # per decode token across windows: durations/4 -> median 50
        self.assertEqual(agg["categories"][sr.CAT_DEQUANT]["per_decode_token_median_ns"], 50.0)

    def test_median_never_mixes_processes(self):
        # two processes in one trace dir, DIFFERENT repeats: allowed, pooled
        # group flagged, per-process blocks authoritative
        self.write_unit(1001, [marker_row(sr.MARKER_RANGE_DOMAIN,
                                          "stage=decode;context=2048;repeat=1;tokens=4",
                                          1001, 1000, 2000)],
                        [kernel_row(1001, "void exl3_gemv_dot_kernel<6>(x)", 1100, 1400)],
                        subdir = "uuidA")
        self.write_unit(2002, [marker_row(sr.MARKER_RANGE_DOMAIN,
                                          "stage=decode;context=2048;repeat=2;tokens=4",
                                          2002, 5000, 6000)],
                        [kernel_row(2002, "void exl3_gemv_dot_kernel<6>(x)", 5100, 5700)],
                        subdir = "uuidB")
        res = self.summarize()
        self.assertTrue(res["ok"], res["errors"])
        agg = res["aggregates"]["stage=decode;context=2048"]
        self.assertTrue(agg["multiple_processes"])
        self.assertEqual(agg["pids"], [1001, 2002])
        self.assertIn("1001", agg["per_process"])
        self.assertIn("2002", agg["per_process"])
        self.assertEqual(agg["per_process"]["1001"]["categories"][sr.CAT_DEQUANT]
                         ["per_window"]["median_ns"], 300)
        self.assertEqual(agg["per_process"]["2002"]["categories"][sr.CAT_DEQUANT]
                         ["per_window"]["median_ns"], 600)

    def test_same_range_from_two_processes_is_rejected(self):
        for pid, sub in ((1001, "uuidA"), (2002, "uuidB")):
            self.write_unit(pid, [marker_row(sr.MARKER_RANGE_DOMAIN,
                                             "stage=decode;context=2048;repeat=1;tokens=4",
                                             pid, 1000, 2000)],
                            [kernel_row(pid, "void exl3_gemv_dot_kernel<6>(x)", 1100, 1400)],
                            subdir = sub)
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("duplicate selected range" in e for e in res["errors"]), res["errors"])


class TestRefusals(SyntheticTraceMixin, unittest.TestCase):
    def marker_ok(self):
        return marker_decode(1001)

    def kernel_ok(self):
        return kernel_exl3(1001)

    def test_missing_kernel_file_rejected(self):
        d = self.trace_dir / "uuidA"
        d.mkdir(parents = True)
        write_csv(d / "1001_marker_api_trace.csv", MARKER_HEADER, [self.marker_ok()])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("missing kernel" in e for e in res["errors"]), res["errors"])

    def test_missing_marker_file_rejected(self):
        d = self.trace_dir / "uuidA"
        d.mkdir(parents = True)
        write_csv(d / "1001_kernel_trace.csv", KERNEL_HEADER, [self.kernel_ok()])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("missing marker" in e for e in res["errors"]), res["errors"])

    def test_no_trace_files_at_all(self):
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("no rocprofv3" in e for e in res["errors"]))

    def test_no_selected_ranges_rejected(self):
        # only control markers and an unselected range: nothing to summarize
        markers = [
            marker_row(sr.MARKER_CONTROL_DOMAIN, "roctxProfilerPause", 1001, 100, 120),
            marker_row(sr.MARKER_RANGE_DOMAIN, "some_other_range", 1001, 300, 400),
        ]
        self.write_unit(1001, markers, [self.kernel_ok()])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("no selected stage ranges" in e for e in res["errors"]), res["errors"])

    def test_empty_marker_range_rejected(self):
        self.write_unit(1001, [marker_row(sr.MARKER_RANGE_DOMAIN,
                                          "stage=decode;context=2048;repeat=1;tokens=4",
                                          1001, 1000, 1000)],
                        [self.kernel_ok()])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("empty marker range" in e for e in res["errors"]), res["errors"])

    def test_nonpositive_timestamps_rejected(self):
        self.write_unit(1001, [marker_row(sr.MARKER_RANGE_DOMAIN,
                                          "stage=decode;context=2048;repeat=1;tokens=4",
                                          1001, 1000, 2000)],
                        [kernel_row(1001, "void exl3_gemv_dot_kernel<6>(x)", 0, 1400),
                         kernel_row(1001, "void rms_norm_kernel<0>(x)", 1100, 1400, 2)])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("non-positive timestamp" in e for e in res["errors"]), res["errors"])

    def test_end_before_start_rejected(self):
        self.write_unit(1001, [self.marker_ok()],
                        [kernel_row(1001, "void exl3_gemv_dot_kernel<6>(x)", 1500, 1100)])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("before start" in e for e in res["errors"]), res["errors"])

    def test_duplicate_kernel_rows_rejected(self):
        row = self.kernel_ok()
        self.write_unit(1001, [self.marker_ok()], [row, list(row)])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("duplicate kernel row" in e for e in res["errors"]), res["errors"])

    def test_duplicate_marker_rows_rejected(self):
        row = self.marker_ok()
        self.write_unit(1001, [row, list(row)], [self.kernel_ok()])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("duplicate marker row" in e for e in res["errors"]), res["errors"])

    def test_duplicate_files_rejected(self):
        self.write_unit(1001, [self.marker_ok()], [self.kernel_ok()])
        dup = self.trace_dir / "uuidA" / "1001_extra_kernel_trace.csv"
        write_csv(dup, KERNEL_HEADER, [self.kernel_ok()])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("duplicate kernel trace file" in e for e in res["errors"]),
                        res["errors"])

    def test_overlapping_windows_in_process_rejected(self):
        markers = [
            marker_row(sr.MARKER_RANGE_DOMAIN, "stage=decode;context=2048;repeat=1;tokens=4",
                       1001, 1000, 3000),
            marker_row(sr.MARKER_RANGE_DOMAIN, "stage=decode;context=2048;repeat=2;tokens=4",
                       1001, 2500, 4000),
        ]
        self.write_unit(1001, markers, [self.kernel_ok()])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("overlap in process" in e for e in res["errors"]), res["errors"])

    def test_boundary_crossing_kernel_rejected_not_clipped(self):
        markers = [marker_row(sr.MARKER_RANGE_DOMAIN,
                              "stage=decode;context=2048;repeat=1;tokens=4", 1001, 1000, 2000)]
        kernels = [
            kernel_row(1001, "void exl3_gemv_dot_kernel<6>(x)", 1100, 1400),        # inside
            kernel_row(1001, "void rms_norm_kernel<0>(x)", 1900, 2100, 2),           # crosses end
            kernel_row(1001, "void _paged_attn_decode_split_kernel(x)", 900, 1150, 3),  # crosses start
            kernel_row(1001, "__amd_rocclr_copyBuffer", 800, 2500, 4),               # spans whole
        ]
        self.write_unit(1001, markers, kernels)
        res = self.summarize()
        self.assertFalse(res["ok"])
        crossings = [e for e in res["errors"] if "CROSSES" in e]
        self.assertEqual(len(crossings), 3)
        # the crossing kernels were NOT silently clipped into the window:
        w = res["windows"][0]
        self.assertEqual(w["kernel_count"], 1)
        self.assertEqual(w["kernel_sum_ns"], 300)

    def test_boundary_crossing_copy_rejected(self):
        self.write_unit(
            1001,
            [marker_row(sr.MARKER_RANGE_DOMAIN,
                        "stage=decode;context=2048;repeat=1;tokens=4", 1001, 1000, 2000)],
            [self.kernel_ok()],
            [copy_row(1001, 1500, 2500)])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("memory copy" in e and "CROSSES" in e for e in res["errors"]),
                        res["errors"])

    def test_zero_kernels_in_selected_window_rejected(self):
        # a window with NO kernel inside is not "zero GPU time" -- it is a
        # broken capture; the outside kernel stays outside.
        self.write_unit(
            1001,
            [marker_row(sr.MARKER_RANGE_DOMAIN,
                        "stage=decode;context=2048;repeat=1;tokens=4", 1001, 1000, 2000)],
            [kernel_row(1001, "void exl3_gemv_dot_kernel<6>(x)", 5000, 5200)])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("zero GPU kernels inside" in e for e in res["errors"]), res["errors"])

    def test_fake_zero_gpu_trace_not_success(self):
        # window present but the kernel file has zero kernel rows at all
        self.write_unit(
            1001,
            [marker_row(sr.MARKER_RANGE_DOMAIN,
                        "stage=decode;context=2048;repeat=1;tokens=4", 1001, 1000, 2000)],
            [])
        res = self.summarize()
        self.assertFalse(res["ok"])

    def test_marker_process_id_mismatch_rejected(self):
        self.write_unit(1001, [marker_row(sr.MARKER_RANGE_DOMAIN,
                                          "stage=decode;context=2048;repeat=1;tokens=4",
                                          9999, 1000, 2000)],
                        [self.kernel_ok()])
        res = self.summarize()
        self.assertFalse(res["ok"])
        self.assertTrue(any("mixing processes" in e or "!= filename PID" in e
                            for e in res["errors"]), res["errors"])

    def test_memory_copy_file_absent_is_not_zero(self):
        # tolerated: windows still summarize; copies reported NOT MEASURED
        self.write_unit(1001, [self.marker_ok()], [self.kernel_ok()])
        res = self.summarize()
        self.assertTrue(res["ok"], res["errors"])
        self.assertTrue(res["copy_trace"].startswith("absent"))
        w = res["windows"][0]
        self.assertFalse(w["copy_measured"])
        self.assertIsNone(w["copy_sum_ns"])       # not 0 -- never measured
        self.assertIsNone(w["copy_union_ns"])
        # GPU busy is kernels-only union in that case
        self.assertEqual(w["gpu_busy_union_ns"], 300)
        # and the remainder is honestly labeled: may include UNTRACED SDMA/
        # memory copies, never "pure CPU compute", never "GPU idle"
        self.assertIn("SDMA", w["non_gpu_note"])
        self.assertIn("UNTRACED", w["non_gpu_note"])
        self.assertIn("never GPU idle", w["non_gpu_note"])
        self.assertIn("SDMA", res["legend"]["non_gpu_interval_ns"])
        self.assertIn("Never labeled pure CPU", res["legend"]["non_gpu_interval_ns"])

    def test_measured_copy_note_drops_sdma_caveat(self):
        self.write_unit(1001, [self.marker_ok()], [self.kernel_ok()],
                        [copy_row(1001, 1100, 1150)])
        res = self.summarize()
        self.assertTrue(res["ok"], res["errors"])
        self.assertIn("traced copies", res["windows"][0]["non_gpu_note"])


def minimal_run_json(trace_dir_marker_wall_ns = 1000):
    """profile_stages-shaped artifact matching the TestRefusals fixtures."""
    return {
        "format": sr.PROFILE_FORMAT, "ok": True, "errors": [], "cleanup_failures": [],
        "params": {"repeats": 1, "decode_start": 96, "decode_tokens": 4},
        "provenance": {"decode_start": 96, "decode_tokens": 4},
        "runs": [{
            "kind": "timed_decode_job", "context": 2048, "repeat": 1,
            "decode_start": 96, "decode_tokens": 4, "ids_sha256": "ab" * 32,
            "prefill_window": None,
            "decode_window": {"roctx_traced": True, "stage": "decode", "tokens": 4,
                              "wall_s": trace_dir_marker_wall_ns / 1e9},
            "job_result": {"prompt_tokens": 2048, "cached_tokens": 0,
                           "new_tokens": 100, "eos_reason": "max_new_tokens"},
            "sequence": {"generated_ids_sha256": "cd" * 32},
        }],
    }


class TestRunJsonCrossCheck(SyntheticTraceMixin, unittest.TestCase):
    def happy_trace(self):
        self.write_unit(1001, [marker_decode(1001)], [kernel_exl3(1001)])

    def test_matching_run_json_ok_and_wall_delta_reported(self):
        self.happy_trace()
        run = minimal_run_json(1000)  # wall_s == marker wall
        res = self.summarize(run = run)
        self.assertTrue(res["ok"], res["errors"])
        cc = res["run_json_check"]
        self.assertEqual(cc["expected_from_run"], 1)
        self.assertEqual(cc["found_in_trace"], 1)
        self.assertEqual(cc["windows"][0]["wall_delta_pct"], 0.0)
        self.assertEqual(cc["problems"], [])

    def test_wall_mismatch_is_reported_not_rewritten(self):
        self.happy_trace()
        run = minimal_run_json(1200)  # marker 1000 ns vs recorded 1.2 us
        res = self.summarize(run = run)
        delta = res["run_json_check"]["windows"][0]["wall_delta_pct"]
        self.assertAlmostEqual(delta, -16.667, places = 2)

    def test_token_count_mismatch_is_problem(self):
        self.happy_trace()
        run = minimal_run_json()
        run["runs"][0]["decode_window"]["tokens"] = 5
        res = self.summarize(run = run)
        self.assertFalse(res["ok"])
        self.assertTrue(any("marker tokens=4" in e for e in res["errors"]), res["errors"])

    def test_missing_profiled_window_is_problem(self):
        self.happy_trace()
        run = minimal_run_json()
        run["runs"][0]["repeat"] = 7
        res = self.summarize(run = run)
        self.assertFalse(res["ok"])
        self.assertTrue(any("missing from trace" in e for e in res["errors"]), res["errors"])

    def test_run_json_with_errors_forces_nonzero(self):
        self.happy_trace()
        run = minimal_run_json()
        run["ok"] = False
        run["errors"] = ["aborted: boom"]
        res = self.summarize(run = run)
        self.assertFalse(res["ok"])
        self.assertTrue(any("run-json" in e for e in res["errors"]), res["errors"])


class TestCli(SyntheticTraceMixin, unittest.TestCase):
    def test_help_runs_on_cpu(self):
        for module in (sr,):
            with self.assertRaises(SystemExit) as cm:
                module.main(["--help"])
            self.assertEqual(cm.exception.code, 0)

    def test_main_exit_codes_and_output_file(self):
        self.write_unit(1001, [marker_decode(1001)], [kernel_exl3(1001)])
        out = Path(self._tmp.name) / "summary.json"
        self.assertEqual(sr.main(["--trace-dir", str(self.trace_dir),
                                  "--output", str(out)]), 0)
        data = json.loads(out.read_text())
        self.assertTrue(data["ok"])
        self.assertEqual(data["format"], sr.SUMMARY_FORMAT)
        # broken tree -> nonzero, errors written to the JSON too
        bad = Path(self._tmp.name) / "empty-trace"
        bad.mkdir()
        out2 = Path(self._tmp.name) / "summary2.json"
        self.assertEqual(sr.main(["--trace-dir", str(bad), "--output", str(out2)]), 1)
        self.assertFalse(json.loads(out2.read_text())["ok"])

    def test_main_rejects_wrong_run_json_format(self):
        out = Path(self._tmp.name) / "summary.json"
        rj = Path(self._tmp.name) / "run.json"
        rj.write_text(json.dumps({"format": "something/else"}))
        self.assertEqual(sr.main(["--trace-dir", str(self.trace_dir),
                                  "--output", str(out), "--run-json", str(rj)]), 1)
        self.assertFalse(json.loads(out.read_text())["ok"])


if __name__ == "__main__":
    unittest.main()
