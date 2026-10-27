#!/usr/bin/env python3
"""CPU-only tests for the EXL3 INT8 W8A8 prefill path (exl3.py).

Covers, without any GPU or built extension:
  * _exact_int32_gemm is bit-exact against an int64 reference (the emulation
    the fp32 fallback relies on until ext.igemm_recon exists).
  * reconstruct_hgemm routes to the int8 path exactly when EXL3_INT8_PREFILL
    is on, the tensor is eligible, and no graph capture is in flight.
  * graph capture ("capture" in params) and the env-off switch keep the
    regular fp16 reconstruct+hgemm path (igemm never called).

Module loading mirrors tests/test_gdn_rdna2_dispatch.py: the real exl3.py is
executed standalone against stub parent packages; torch itself is real
(host CPU build is enough -- no CUDA device is touched).

Run from the repo root:
    python3 -m unittest rocm_tools.rdna2.tests.test_int8_prefill_cpu -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

EXL3_PATH = ROOT / "exllamav3" / "modules" / "quant" / "exl3.py"


def _load_exl3(ext_namespace):
    made = {}

    def stub(pkg_name, attrs=None):
        mod = types.ModuleType(pkg_name)
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        made[pkg_name] = sys.modules.get(pkg_name, ...)
        sys.modules[pkg_name] = mod
        return mod

    try:
        stub("exllamav3")
        stub("exllamav3.modules")
        stub("exllamav3.modules.quant")
        stub("exllamav3.model")
        stub("exllamav3.model.config", {"Config": object})
        stub("exllamav3.modules.quant.exl3_lib")
        stub("exllamav3.modules.quant.exl3_lib.quantize", {
            "preapply_had_l": lambda *a: None,
            "preapply_had_r": lambda *a: None,
            "had_k": 128,
            "had_n": 128,
        })
        stub("exllamav3.ext", {"exllamav3_ext": ext_namespace})
        stub("exllamav3.util", {"profile_opt": None})
        stub("exllamav3.util.tensor", {"g_tensor_cache": None})
        # exl3.py does both `from ...util import profile_opt` (package attr)
        # and `from ...util.tensor import g_tensor_cache` (submodule).
        spec = importlib.util.spec_from_file_location(
            "exllamav3.modules.quant.exl3_under_test", str(EXL3_PATH))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, prev in made.items():
            if prev is ...:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class _Calls:
    def __init__(self):
        self.names = []


def _ext_stub(calls, with_igemm=True):
    def had_r_128(a, b, s, sv, scale):
        calls.names.append("had_r_128")
        b.copy_(a)

    def reconstruct(w, trellis, K, mcg, mul1):
        calls.names.append("reconstruct")
        torch.nn.init.uniform_(w, -1.0, 1.0)

    def reconstruct_slice(w, trellis, K, mcg, mul1, n_start):
        calls.names.append("reconstruct_slice")
        torch.nn.init.uniform_(w, -1.0, 1.0)

    def hgemm_recon(a, b, c):
        calls.names.append("hgemm_recon")
        c.copy_(torch.zeros_like(c))

    ns = {
        "had_r_128": had_r_128,
        "reconstruct": reconstruct,
        "reconstruct_slice": reconstruct_slice,
        "hgemm_recon": hgemm_recon,
    }
    if with_igemm:
        def igemm_recon(a, b, c):
            calls.names.append("igemm_recon")
            c.fill_(7)

        ns["igemm_recon"] = igemm_recon
    return types.SimpleNamespace(**ns)


def _fake_linear(mod, scale=1.0):
    lin = mod.LinearEXL3.__new__(mod.LinearEXL3)
    lin.trellis = types.SimpleNamespace(is_cuda=True, device=torch.device("cpu"))
    lin.in_features = 256
    lin.out_features = 256
    lin.K = 3
    lin.mcg = False
    lin.mul1 = True
    lin.suh = torch.full((256,), 0.5, dtype=torch.float16) * scale
    lin.svh = torch.full((256,), 0.25, dtype=torch.float16) * scale
    lin.bias = None
    lin._fused_reconstruct = False
    lin.key = None
    lin.config = types.SimpleNamespace(
        infer_params=types.SimpleNamespace(no_reconstruct=False))
    lin.default_out_dtype = torch.float16
    lin.out_dtype = None
    return lin


class Int8PrefillCpuTest(unittest.TestCase):
    def test_exact_int32_gemm_matches_int64_reference(self):
        calls = _Calls()
        mod = _load_exl3(_ext_stub(calls))
        gen = torch.Generator().manual_seed(0)
        for (m, k, n) in [(17, 700, 33), (5, 2000, 9), (1, 512, 128)]:
            a8 = torch.randint(-128, 128, (m, k), generator=gen).to(torch.int8)
            w8 = torch.randint(-128, 128, (k, n), generator=gen).to(torch.int8)
            # extremes: every lane sees -128/127 products
            a8[0, :] = -128
            a8[-1, :] = 127
            w8[:, 0] = -128
            w8[:, -1] = 127
            got = mod._exact_int32_gemm(a8, w8)
            want = (a8.to(torch.int64) @ w8.to(torch.int64)).to(torch.int32)
            self.assertEqual(got.dtype, torch.int32)
            self.assertTrue(torch.equal(got, want),
                            f"mismatch at {(m, k, n)}")

    def test_int8_path_taken_when_eligible(self):
        calls = _Calls()
        mod = _load_exl3(_ext_stub(calls))
        with patch.object(mod, "int8_prefill", True):
            lin = _fake_linear(mod)
            x = torch.randn(200, 256, dtype=torch.float16)
            y = lin.forward(x, {}, None)
            self.assertIn("igemm_recon", calls.names)
            self.assertNotIn("hgemm_recon", calls.names)
            self.assertTrue(torch.isfinite(y.float()).all())

    def test_capture_stays_on_fp16_path(self):
        calls = _Calls()
        mod = _load_exl3(_ext_stub(calls))
        with patch.object(mod, "int8_prefill", True):
            lin = _fake_linear(mod)
            x = torch.randn(200, 256, dtype=torch.float16)
            y = lin.forward(x, {"capture": object()}, None)
            self.assertIn("hgemm_recon", calls.names)
            self.assertNotIn("igemm_recon", calls.names)
            self.assertTrue(torch.isfinite(y.float()).all())

    def test_env_off_stays_on_fp16_path(self):
        calls = _Calls()
        mod = _load_exl3(_ext_stub(calls))
        with patch.object(mod, "int8_prefill", False):
            lin = _fake_linear(mod)
            x = torch.randn(200, 256, dtype=torch.float16)
            y = lin.forward(x, {}, None)
            self.assertIn("hgemm_recon", calls.names)
            self.assertNotIn("igemm_recon", calls.names)
            self.assertTrue(torch.isfinite(y.float()).all())

    def test_ablate_w_measures_weight_grid_only(self):
        calls = _Calls()
        mod = _load_exl3(_ext_stub(calls, with_igemm=True))
        with patch.object(mod, "int8_prefill", True), \
             patch.object(mod, "_int8_ablate", "w"):
            lin = _fake_linear(mod)
            x = torch.randn(200, 256, dtype=torch.float16)
            y = lin.forward(x, {}, None)
            self.assertNotIn("igemm_recon", calls.names)
            self.assertTrue(torch.isfinite(y.float()).all())

    def test_ablate_a_measures_activation_grid_only(self):
        calls = _Calls()
        mod = _load_exl3(_ext_stub(calls, with_igemm=True))
        with patch.object(mod, "int8_prefill", True), \
             patch.object(mod, "_int8_ablate", "a"):
            lin = _fake_linear(mod)
            x = torch.randn(200, 256, dtype=torch.float16)
            y = lin.forward(x, {}, None)
            self.assertNotIn("igemm_recon", calls.names)
            self.assertTrue(torch.isfinite(y.float()).all())

    def test_default_path_applies_fp32_scales_with_original_svh(self):
        calls = _Calls()
        ns = _ext_stub(calls, with_igemm=True)
        seen = {}
        orig_had = ns.had_r_128
        def had(a, b, s, sv, scale):
            seen.setdefault("svh_args", []).append(sv)
            return orig_had(a, b, s, sv, scale)
        ns.had_r_128 = had
        mod = _load_exl3(ns)
        with patch.object(mod, "int8_prefill", True), \
             patch.object(mod, "_int8_ablate", ""):
            lin = _fake_linear(mod)
            x = torch.randn(200, 256, dtype=torch.float16)
            y = lin.forward(x, {}, None)
            self.assertIn("igemm_recon", calls.names)
            self.assertTrue(torch.isfinite(y.float()).all())
            # Output Hadamard(s) must carry the ORIGINAL svh: folding the
            # weight scales into an fp16 svh copy cost +0.058 macro KLD.
            self.assertTrue(seen["svh_args"])
            for sv in seen["svh_args"]:
                if sv is not None:
                    self.assertIs(sv, lin.svh)

    def test_no_igemm_backend_uses_exact_emulation(self):
        calls = _Calls()
        mod = _load_exl3(_ext_stub(calls, with_igemm=False))
        with patch.object(mod, "int8_prefill", True):
            lin = _fake_linear(mod)
            x = torch.randn(200, 256, dtype=torch.float16)
            real_emu = mod._exact_int32_gemm
            with patch.object(mod, "_exact_int32_gemm",
                              wraps=real_emu) as emu:
                y = lin.forward(x, {}, None)
                self.assertEqual(emu.call_count, 1)
            self.assertNotIn("hgemm_recon", calls.names)
            self.assertTrue(torch.isfinite(y.float()).all())


if __name__ == "__main__":
    unittest.main()
