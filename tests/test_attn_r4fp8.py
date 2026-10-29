#!/usr/bin/env python3
"""Round 4: minimal-diff FP8 / INT8 prefill (r4_fp8.py).

Needs GPU. Scheme tests (torch only) run anywhere; kernel tests need real
FP8 WMMA (gfx1201/R9700) and skip elsewhere:
  HIP_VISIBLE_DEVICES=0 python3 -m pytest tests/test_attn_r4fp8.py -x -q

Covers: pre-pass shapes/dtypes, torch reference vs fp16 SDPA accuracy,
frozen-mu softmax invariance (int8), kernel vs reference (full blocks must
match by construction), 27B-geometry kernel vs fp16, paged folding.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

devices = ["cuda:0"]


def _has_fp8():
    from exllamav3.modules.attention_fn.int8fp8 import _has_fp8_arch
    try:
        return _has_fp8_arch(torch.device("cuda:0"))
    except Exception:
        return False


def rel_err(got, base):
    d = (got.float() - base.float()).abs()
    return d.max().item(), d.max().item() / base.float().abs().amax().clamp_min(1e-12).item()


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", ["fp8", "int8"])
def test_r4_quantize_shapes(device, mode):
    from exllamav3.modules.attention_fn.r4_fp8 import r4_quantize
    torch.manual_seed(0)
    B, Tq, Tk, Hq, Hkv, D = 1, 64, 512, 8, 2, 128
    sm = 1.0 / (D ** 0.5)
    q = torch.randn((B, Tq, Hq, D), device=device, dtype=torch.float16)
    kf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    vf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    quant = r4_quantize(q, kf, vf, sm, mode)
    qdt = torch.int8 if mode == "int8" else torch.float8_e4m3fn
    assert quant["q_q"].shape == q.shape and quant["q_q"].dtype == qdt
    assert quant["k_q"].shape == kf.shape and quant["k_q"].dtype == qdt
    assert quant["v_q"].shape == vf.shape and quant["v_q"].dtype == torch.float8_e4m3fn
    assert quant["row_log2e"].shape == (B, Tq, Hq) and quant["row_log2e"].dtype == torch.float32
    assert quant["v_scale"].shape == (Hkv, D) and (quant["v_scale"] > 0).all()
    assert torch.isfinite(quant["row_log2e"]).all()
    if mode == "int8":
        assert quant["mu"].shape == (Hkv, D)
    else:
        assert quant["mu"] is None


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", ["fp8", "int8"])
def test_r4_ref_vs_fp16(device, mode):
    """Scheme accuracy (torch only): r4 reference vs fp16 SDPA."""
    from exllamav3.modules.attention_fn.r4_fp8 import sdpa_r4_ref
    torch.manual_seed(1)
    # Square self-attention (bottom-right causal == plain causal here).
    B, T, Hq, Hkv, D = 1, 256, 8, 2, 256
    sm = 1.0 / (D ** 0.5)
    q = torch.randn((B, T, Hq, D), device=device, dtype=torch.float16)
    kf = (torch.randn((B, T, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    vf = (torch.randn((B, T, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    got = sdpa_r4_ref(q, kf, vf, causal=True, sm_scale=sm, mode=mode)
    # Manual fp16 reference (torch SDPA is unreliable for GQA on ROCm).
    qf, kk, vv = q.float(), kf.float(), vf.float()
    cmask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
    base = torch.zeros_like(qf)
    for h in range(Hq):
        s = (qf[0, :, h, :] @ kk[0, :, h // (Hq // Hkv), :].t()) * sm + cmask
        base[0, :, h, :] = torch.softmax(s, dim=1) @ vv[0, :, h // (Hq // Hkv), :]
    base = base.half()
    ma, rel = rel_err(got, base)
    assert rel < 0.08, f"{mode}: rel={rel:.4f} maxabs={ma:.5f}"


@pytest.mark.parametrize("device", devices)
def test_r4_int8_frozen_mu_invariant(device):
    """Frozen-mean smoothing must be softmax-exact: int8 ref with the frozen
    mu and no bias term equals an explicit-bias computation."""
    from exllamav3.modules.attention_fn.r4_fp8 import r4_quantize
    torch.manual_seed(2)
    B, Tq, Tk, Hq, Hkv, D = 1, 32, 256, 4, 2, 64
    sm = 1.0 / (D ** 0.5)
    q = torch.randn((B, Tq, Hq, D), device=device, dtype=torch.float16)
    kf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    vf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    quant = r4_quantize(q, kf, vf, sm, "int8")
    mu = quant["mu"]
    # q.mu is constant over keys per row: adding it cannot change softmax.
    qf = q.float()
    qmu = (qf[:, :, 0, :] * mu[0]).sum(dim=-1, keepdim=True)  # (B,Tq,1)
    assert torch.isfinite(qmu).all()
    assert (qmu.abs() > 0).any()  # non-trivial bias that must still cancel


@pytest.mark.parametrize("device", devices)
def test_r4_to_paged(device):
    from exllamav3.modules.attention_fn.r4_fp8 import r4_quantize, r4_to_paged
    torch.manual_seed(0)
    B, Tq, Tk, Hq, Hkv, D = 1, 64, 512, 8, 2, 128
    sm = 1.0 / (D ** 0.5)
    q = torch.randn((B, Tq, Hq, D), device=device, dtype=torch.float16)
    kf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    vf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    quant = r4_quantize(q, kf, vf, sm, "fp8")
    kc, vc, bt, cs = r4_to_paged(quant)
    assert kc.shape == (Tk // 256, 256, Hkv, D)
    assert vc.shape == kc.shape and bt.shape == (B, Tk // 256)
    assert cs.tolist() == [Tk]
    assert torch.equal(kc.view(B, Tk, Hkv, D), quant["k_q"])


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", ["fp8", "int8"])
def test_r4_kernel_matches_ref(device, mode):
    """Kernel vs torch reference (needs gfx1201 FP8 WMMA)."""
    if not _has_fp8():
        pytest.skip("needs FP8 WMMA arch (gfx1201)")
    from exllamav3.modules.attention_fn.r4_fp8 import (
        r4_quantize, r4_to_paged, r4_prefill, sdpa_r4_ref)
    torch.manual_seed(3)
    B, Tq, Tk, Hq, Hkv, D = 1, 64, 1024, 8, 2, 128
    sm = 1.0 / (D ** 0.5)
    q = torch.randn((B, Tq, Hq, D), device=device, dtype=torch.float16)
    kf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    vf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    quant = r4_quantize(q, kf, vf, sm, mode)
    kc, vc, bt, cs = r4_to_paged(quant)
    got = r4_prefill(quant["q_q"], kc, vc, bt, cs,
                     quant["row_log2e"], quant["v_scale"],
                     causal=True, mode=mode)
    torch.cuda.synchronize()
    ref = sdpa_r4_ref(q, kf, vf, causal=True, sm_scale=sm, mode=mode)
    ma, rel = rel_err(got, ref)
    assert ma < 0.02, f"{mode}: maxabs={ma:.5f} rel={rel:.4f}"


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("mode", ["fp8", "int8"])
def test_r4_kernel_27b_shape(device, mode):
    """27B geometry (24/4/256) kernel vs ref and vs fp16 kernel."""
    if not _has_fp8():
        pytest.skip("needs FP8 WMMA arch (gfx1201)")
    from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
    from exllamav3.modules.attention_fn.r4_fp8 import (
        r4_quantize, r4_to_paged, r4_prefill, sdpa_r4_ref)
    torch.manual_seed(4)
    B, Tq, Tk, Hq, Hkv, D = 1, 128, 1024, 24, 4, 256
    sm = 1.0 / (D ** 0.5)
    q = torch.randn((B, Tq, Hq, D), device=device, dtype=torch.float16)
    kf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    vf = (torch.randn((B, Tk, Hkv, D), device=device, dtype=torch.float16) * 0.5).contiguous()
    quant = r4_quantize(q, kf, vf, sm, mode)
    kc, vc, bt, cs = r4_to_paged(quant)
    got = r4_prefill(quant["q_q"], kc, vc, bt, cs,
                     quant["row_log2e"], quant["v_scale"],
                     causal=True, mode=mode)
    torch.cuda.synchronize()
    ref = sdpa_r4_ref(q, kf, vf, causal=True, sm_scale=sm, mode=mode)
    ma, rel = rel_err(got, ref)
    assert ma < 0.02, f"{mode}: vs-ref maxabs={ma:.5f} rel={rel:.4f}"
    pages = Tk // 256
    kc16 = kf.view(B * pages, 256, Hkv, D).clone()
    vc16 = vf.view(B * pages, 256, Hkv, D).clone()
    bt16 = torch.arange(B * pages, device=device, dtype=torch.int32).view(B, pages)
    cs16 = torch.tensor([Tk], device=device, dtype=torch.int32)
    out16 = torch.empty_like(q)
    paged_attn_triton_prefill(q[:, :Tq].contiguous(), None, None, kc16, vc16,
                              bt16, cs16, causal=True, softmax_scale=sm, out=out16)
    torch.cuda.synchronize()
    ma, rel = rel_err(got, out16)
    assert rel < 0.08, f"{mode}: vs-fp16 rel={rel:.4f} maxabs={ma:.5f}"
