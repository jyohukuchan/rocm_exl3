#!/usr/bin/env python3
"""Opt-in INT8-QK / FP8-PV prefill attention (Sage-style) vs fp16.

Needs GPU. Run: HIP_VISIBLE_DEVICES=1 python3 -m pytest tests/test_attn_int8fp8.py -x -q
Covers: torch reference vs fp16 SDPA (random, GQA, head_dim 256, causal and
not, long KV), ablation paths (nosmooth / per-tensor-k / fp16pv), Triton
kernel vs reference over a paged fp16 cache, and the dispatch contract
(off by default -> None, on -> close to the in-repo fp16 Triton prefill).
A real-Q/K/V capture from Qwen3.8-27B (eval/prof/capture_qkv_27b.py) is used
when EXL3_INT8FP8_QKV_PATH points at it, otherwise that test skips.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

torch.set_printoptions(precision=5, sci_mode=False, linewidth=200)

devices = ["cuda:0"]

PAGE_SIZE = 256


def sdpa_bottom_right(q, k, v, causal):
    """fp16 SDPA baseline with bottom-right-aligned causal mask (what paged
    prefill with cache_seqlens means: query row m sees keys <= past + m)."""
    bsz, tq, hq, dim = q.shape
    tk, hkv = k.shape[1], k.shape[2]
    g = hq // hkv
    qh = q.float().transpose(1, 2).reshape(bsz * hq, tq, dim)
    kh = k.float().transpose(1, 2).repeat_interleave(g, dim=1).reshape(bsz * hq, tk, dim)
    vh = v.float().transpose(1, 2).repeat_interleave(g, dim=1).reshape(bsz * hq, tk, dim)
    mask = None
    if causal:
        q_abs = (tk - tq) + torch.arange(tq, device=q.device)
        mask = (torch.arange(tk, device=q.device)[None, :] <= q_abs[:, None])
        mask = mask.unsqueeze(0).expand(bsz * hq, -1, -1)
    return torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, attn_mask=mask).reshape(bsz, hq, tq, dim).transpose(1, 2).half()


def rel_err(got, base):
    d = (got.float() - base.float()).abs()
    return d.max().item(), d.max().item() / base.float().abs().amax().clamp_min(1e-12).item()


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("causal", [True, False])
@torch.inference_mode()
def test_ref_vs_fp16_random(device, causal):
    from exllamav3.modules.attention_fn.int8fp8 import sdpa_int8fp8_ref
    torch.manual_seed(0)
    bsz, tq, tk, hq, hkv, dim = 1, 64, 1024, 8, 2, 256
    q = torch.randn((bsz, tq, hq, dim), generator=torch.Generator(device=device).manual_seed(0),
                    device=device, dtype=torch.float16)
    k = torch.randn((bsz, tk, hkv, dim), generator=torch.Generator(device=device).manual_seed(1),
                    device=device, dtype=torch.float16) * 0.5
    v = torch.randn((bsz, tk, hkv, dim), generator=torch.Generator(device=device).manual_seed(2),
                    device=device, dtype=torch.float16) * 0.5
    base = sdpa_bottom_right(q, k, v, causal)
    got = sdpa_int8fp8_ref(q, k, v, causal=causal)
    mx, rel = rel_err(got, base)
    print(f"causal={causal} default maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    assert rel < 0.08, f"default INT8QK/FP8PV too far from fp16: rel={rel:.4f}"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_ref_target_geometry(device):
    """Qwen3.8-27B full-attention geometry: 24 q heads, 4 kv heads, dim 256."""
    from exllamav3.modules.attention_fn.int8fp8 import sdpa_int8fp8_ref
    torch.manual_seed(7)
    bsz, tq, tk, hq, hkv, dim = 1, 128, 4096, 24, 4, 256
    g0 = torch.Generator(device=device).manual_seed(0)
    q = torch.randn((bsz, tq, hq, dim), generator=g0, device=device, dtype=torch.float16)
    k = torch.randn((bsz, tk, hkv, dim), generator=torch.Generator(device=device).manual_seed(1),
                    device=device, dtype=torch.float16) * 0.5
    v = torch.randn((bsz, tk, hkv, dim), generator=torch.Generator(device=device).manual_seed(2),
                    device=device, dtype=torch.float16) * 0.5
    base = sdpa_bottom_right(q, k, v, True)
    got = sdpa_int8fp8_ref(q, k, v, causal=True)
    mx, rel = rel_err(got, base)
    print(f"target-geometry default maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    assert rel < 0.08, f"target geometry too far from fp16: rel={rel:.4f}"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_ref_ablations(device):
    from exllamav3.modules.attention_fn.int8fp8 import sdpa_int8fp8_ref
    bsz, tq, tk, hq, hkv, dim = 1, 64, 1024, 8, 2, 256
    q = torch.randn((bsz, tq, hq, dim), generator=torch.Generator(device=device).manual_seed(0),
                    device=device, dtype=torch.float16)
    k = torch.randn((bsz, tk, hkv, dim), generator=torch.Generator(device=device).manual_seed(1),
                    device=device, dtype=torch.float16) * 0.5
    v = torch.randn((bsz, tk, hkv, dim), generator=torch.Generator(device=device).manual_seed(2),
                    device=device, dtype=torch.float16) * 0.5
    base = sdpa_bottom_right(q, k, v, True)
    errs = {}
    for name, ab in [("default", frozenset()),
                     ("fp16pv", frozenset({"fp16pv"})),
                     ("nosmooth", frozenset({"nosmooth"})),
                     ("per-tensor-k", frozenset({"per-tensor-k"}))]:
        got = sdpa_int8fp8_ref(q, k, v, causal=True, ablate=ab)
        mx, rel = rel_err(got, base)
        errs[name] = rel
        print(f"ablate {name:12s} maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    # PV in fp16 must beat (or match) PV in FP8: finer math, same QK.
    assert errs["fp16pv"] <= errs["default"] + 1e-6
    assert errs["fp16pv"] < 0.02
    # Smoothing / granularity must not blow up on structureless data.
    assert errs["nosmooth"] < 0.08
    assert errs["per-tensor-k"] < 0.08


def _paged_inputs(device, tq, tk_past, hq, hkv, dim, seed=0):
    """Empty paged fp16 cache + tq new rows; kernel appends then attends."""
    pages = (tq + tk_past + PAGE_SIZE - 1) // PAGE_SIZE
    k_cache = torch.zeros((pages, PAGE_SIZE, hkv, dim), device=device, dtype=torch.float16)
    v_cache = torch.zeros((pages, PAGE_SIZE, hkv, dim), device=device, dtype=torch.float16)
    block_table = torch.arange(pages, device=device, dtype=torch.int32).view(1, pages)
    cache_seqlens = torch.tensor([tk_past], device=device, dtype=torch.int32)
    g = torch.Generator(device=device).manual_seed(seed)
    if tk_past:
        kp = torch.randn((1, tk_past, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
        vp = torch.randn((1, tk_past, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
        # pre-fill the cache pages directly (contiguous page 0..)
        k_cache.view(-1, hkv, dim)[:tk_past].copy_(kp[0])
        v_cache.view(-1, hkv, dim)[:tk_past].copy_(vp[0])
    k = torch.randn((1, tq, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    v = torch.randn((1, tq, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    q = torch.randn((1, tq, hq, dim), generator=g, device=device, dtype=torch.float16)
    return q, k, v, k_cache, v_cache, block_table, cache_seqlens


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_kernel_vs_ref_paged(device):
    from exllamav3.modules.attention_fn.int8fp8 import (
        sdpa_int8fp8_ref, paged_attn_triton_prefill_int8fp8)
    tq, tk_past, hq, hkv, dim = 256, 7936, 24, 4, 256  # 8192 total KV
    q, k, v, kc, vc, bt, cs = _paged_inputs(device, tq, tk_past, hq, hkv, dim)
    past_k = kc.view(-1, hkv, dim)[:tk_past].unsqueeze(0)
    past_v = vc.view(-1, hkv, dim)[:tk_past].unsqueeze(0)
    full_k = torch.cat([past_k, k], dim=1)
    full_v = torch.cat([past_v, v], dim=1)
    ref = sdpa_int8fp8_ref(q, full_k, full_v, causal=True)
    for pv in (0, 1):
        got = paged_attn_triton_prefill_int8fp8(
            q, k, v, kc.clone(), vc.clone(), bt, cs, causal=True, pv_mode=pv)
        mx, rel = rel_err(got, ref)
        print(f"paged kernel pv={pv} vs ref maxabs={mx:.5f} rel={rel:.4f}", flush=True)
        # Kernel and reference share the exact block algorithm; only int
        # rounding may differ. Absolute threshold: at 8K KV the fp16 outputs
        # themselves are small, so a relative metric misleads here.
        assert mx < 0.01, f"kernel (pv={pv}) disagrees with reference: maxabs={mx:.5f}"
    # in-repo fp16 Triton prefill as the production baseline for the same pages
    from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
    fp16 = paged_attn_triton_prefill(
        q, k, v, kc.clone(), vc.clone(), bt, cs, causal=True)
    mx, rel = rel_err(ref, fp16)
    print(f"ref vs in-repo-fp16 maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    assert rel < 0.08


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_dispatch_opt_in(device, monkeypatch):
    """Off by default (returns None, fp16 path untouched); on request the new
    mode serves the same paged call and stays near the fp16 Triton kernel."""
    from exllamav3.modules.attention_fn.common import AttnArgs
    from exllamav3.modules.attention_fn.int8fp8 import fn_triton_paged_attn_prefill_int8fp8
    from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
    tq, tk_past, hq, hkv, dim = 64, 960, 8, 2, 128
    q, k, v, kc, vc, bt, cs = _paged_inputs(device, tq, tk_past, hq, hkv, dim)
    args = AttnArgs(1, tq, hq, dim, tq, hkv, q, k, v, kc, vc, True,
                    1.0 / (dim ** 0.5), None, None, None, 0.0, bt, cs)
    monkeypatch.delenv("EXL3_ATTN_INT8FP8", raising=False)
    assert fn_triton_paged_attn_prefill_int8fp8(args) is None
    monkeypatch.setenv("EXL3_ATTN_INT8FP8", "1")
    got = fn_triton_paged_attn_prefill_int8fp8(args)
    assert got is not None
    fp16 = paged_attn_triton_prefill(q, k, v, kc, vc, bt, cs, causal=True)
    mx, rel = rel_err(got, fp16)
    print(f"dispatch on-mode vs fp16 maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    assert rel < 0.08


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_dispatch_nocache(device, monkeypatch):
    """Cache-less prefill (model.forward bulk) folds K/V into temporary
    pages and stays near the in-repo fp16 nocache kernel."""
    from exllamav3.modules.attention_fn.common import AttnArgs
    from exllamav3.modules.attention_fn.int8fp8 import fn_triton_paged_attn_prefill_int8fp8
    from exllamav3.modules.attention_fn.triton_paged import fn_triton_attn_nocache
    bsz, tq, hq, hkv, dim = 1, 64, 8, 2, 128
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn((bsz, tq, hq, dim), generator=g, device=device, dtype=torch.float16)
    k = torch.randn((bsz, tq, hkv, dim), generator=g, device=device, dtype=torch.float16)
    v = torch.randn((bsz, tq, hkv, dim), generator=g, device=device, dtype=torch.float16)
    args = AttnArgs(bsz, tq, hq, dim, tq, hkv, q, k, v, None, None, True,
                    1.0 / (dim ** 0.5), None, None, -1, 0.0, None, None)
    monkeypatch.delenv("EXL3_ATTN_INT8FP8", raising=False)
    assert fn_triton_paged_attn_prefill_int8fp8(args) is None
    monkeypatch.setenv("EXL3_ATTN_INT8FP8", "1")
    got = fn_triton_paged_attn_prefill_int8fp8(args)
    assert got is not None
    fp16 = fn_triton_attn_nocache(args)
    assert fp16 is not None
    mx, rel = rel_err(got, fp16)
    print(f"nocache on-mode vs fp16 maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    assert rel < 0.08


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_real_qkv_capture(device):
    """Real Q/K/V from Qwen3.8-27B full-attention layers (capture with
    eval/prof/capture_qkv_27b.py). Skips when the capture is absent."""
    path = os.environ.get("EXL3_INT8FP8_QKV_PATH", "/tmp/i8f8_qkv_27b.pt")
    if not os.path.exists(path):
        pytest.skip(f"no capture at {path}; run eval/prof/capture_qkv_27b.py")
    from exllamav3.modules.attention_fn.int8fp8 import sdpa_int8fp8_ref
    cap = torch.load(path, map_location=device, weights_only=True)
    worst = 0.0
    for i, layer in enumerate(cap["layers"][:4]):
        q = layer["q"].to(device).half().contiguous()
        k = layer["k"].to(device).half().contiguous()
        v = layer["v"].to(device).half().contiguous()
        base = sdpa_bottom_right(q, k, v, True)
        got = sdpa_int8fp8_ref(q, k, v, causal=True)
        mx, rel = rel_err(got, base)
        print(f"layer {cap['layer_idx'][i]} real-qkv default maxabs={mx:.5f} rel={rel:.4f}", flush=True)
        worst = max(worst, rel)
    assert worst < 0.08, f"real-data error too large: rel={worst:.4f}"
