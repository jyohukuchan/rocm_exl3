#!/usr/bin/env python3
"""Round-2 cached path: CacheLayer_i8f8 + prefill kernel reading it directly.

Needs GPU. Run: HIP_VISIBLE_DEVICES=0 python3 -m pytest tests/test_attn_i8f8_cached.py -x -q
(V620 OK for correctness: FP8 PV falls back to the int8/fp16 proxies; the
torch reference below always uses the exact E4M3 grid.)

Covers: quantize-on-write exactness (odd chunks, page crossings, 1-token
appends) vs fp16, cached kernel vs the round-1 torch reference (full blocks
must match round 1 by construction), partial-block/stage handling across
append calls, dispatch (opt-in runs the cached kernel and counts it;
unset/decode fall back to the fp16 dequant path), copy_page, storage size.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

torch.set_printoptions(precision=5, sci_mode=False, linewidth=200)

devices = ["cuda:0"]

PAGE_SIZE = 256
QB = 32


def _layer(device, max_tokens=2048, hkv=2, dim=128):
    from exllamav3.cache.i8f8 import CacheLayer_i8f8
    attn = SimpleNamespace(num_kv_heads=hkv, head_dim=dim)
    layer = CacheLayer_i8f8(None, attn, 0, max_tokens)
    layer.alloc(torch.device(device))
    return layer


def _append(layer, k, v, seqlens, bt):
    layer.update_kv_direct(seqlens, bt, k, v, k.shape[1])
    return seqlens + k.shape[1]


def rel_err(got, base):
    d = (got.float() - base.float()).abs()
    return d.max().item(), d.max().item() / base.float().abs().amax().clamp_min(1e-12).item()


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_update_roundtrip_fp16(device):
    """Odd-sized appends across block and page boundaries: get_kv (dequant +
    stage overlay) must reproduce the fp16 rows up to quantization noise."""
    torch.manual_seed(0)
    hkv, dim = 2, 128
    layer = _layer(device, 1024, hkv, dim)
    pages = 4
    bt = torch.arange(pages, device=device, dtype=torch.int32).view(1, pages)
    cs = torch.tensor([0], device=device, dtype=torch.int32)
    g = torch.Generator(device=device).manual_seed(0)
    kept_k, kept_v = [], []
    for L in (100, 7, 250, 1, 64, 300):
        k = torch.randn((1, L, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
        v = torch.randn((1, L, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
        kept_k.append(k)
        kept_v.append(v)
        cs = _append(layer, k, v, cs, bt)
    full_k = torch.cat(kept_k, dim=1)
    full_v = torch.cat(kept_v, dim=1)
    T = int(cs.item())
    assert T == sum((100, 7, 250, 1, 64, 300))
    k2, v2 = layer.get_kv(cs, bt)
    got_k = k2.view(-1, hkv, dim)[:T].unsqueeze(0)
    got_v = v2.view(-1, hkv, dim)[:T].unsqueeze(0)
    mx, rel = rel_err(got_k, full_k)
    print(f"dequant-K maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    assert rel < 0.06, f"K roundtrip too lossy: rel={rel:.4f}"
    mx, rel = rel_err(got_v, full_v)
    print(f"dequant-V maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    assert rel < 0.06, f"V roundtrip too lossy: rel={rel:.4f}"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_update_block_exactness(device):
    """Complete 32-blocks in the cache must equal the torch block quantizer
    applied to the same fp16 rows (no compounding across appends)."""
    from exllamav3.cache.i8f8 import quantize_blocks
    torch.manual_seed(1)
    hkv, dim = 2, 64
    layer = _layer(device, 512, hkv, dim)
    bt = torch.arange(2, device=device, dtype=torch.int32).view(1, 2)
    cs = torch.tensor([0], device=device, dtype=torch.int32)
    g = torch.Generator(device=device).manual_seed(1)
    k = torch.randn((1, 96, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    v = torch.randn((1, 96, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    cs = _append(layer, k[:, :40], v[:, :40], cs, bt)   # stage holds 8
    assert int(layer.stage_len[0].item()) == 8
    cs = _append(layer, k[:, 40:], v[:, 40:], cs, bt)   # completes blocks 1,2
    assert int(layer.stage_len[0].item()) == 0
    ref = quantize_blocks(k[:, :96].reshape(3, 32, hkv, dim),
                          v[:, :96].reshape(3, 32, hkv, dim))
    rk, rsk, rmu, rvq, rsv = ref
    assert torch.equal(layer.k_q[0, :96], rk.reshape(96, hkv, dim))
    assert torch.equal(layer.k_sc[0, :3], rsk)
    assert torch.equal(layer.k_mu[0, :3], rmu)
    assert torch.equal(layer.v_q[0, :96], rvq.reshape(96, hkv, dim))
    assert torch.equal(layer.v_sc[0, :3], rsv)


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_kernel_vs_ref_cached(device, monkeypatch):
    """Cached kernel over the packed cache vs the round-1 torch reference
    over the same fp16 rows. Full blocks must match round 1 by construction."""
    from exllamav3.modules.attention_fn.int8fp8 import sdpa_int8fp8_ref
    from exllamav3.modules.attention_fn.i8f8_cached import (
        paged_attn_triton_prefill_i8f8_cached, CALLS)
    monkeypatch.setenv("EXL3_ATTN_INT8FP8", "1")
    hq, hkv, dim = 8, 2, 128
    layer = _layer(device, 2048, hkv, dim)
    pages = 2048 // PAGE_SIZE
    bt = torch.arange(pages, device=device, dtype=torch.int32).view(1, pages)
    past, tq = 960, 64   # total 1024 = 32 blocks, no tail
    g = torch.Generator(device=device).manual_seed(2)
    kp = torch.randn((1, past, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    vp = torch.randn((1, past, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    k = torch.randn((1, tq, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    v = torch.randn((1, tq, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    q = torch.randn((1, tq, hq, dim), generator=g, device=device, dtype=torch.float16)
    cs = torch.tensor([0], device=device, dtype=torch.int32)
    cs = _append(layer, kp, vp, cs, bt)
    cs_past = cs.clone()
    cs = _append(layer, k, v, cs, bt)
    full_k = torch.cat([kp, k], dim=1)
    full_v = torch.cat([vp, v], dim=1)
    ref = sdpa_int8fp8_ref(q, full_k, full_v, causal=True)
    before = CALLS["cached"]
    for pv in (0, 1):
        out = paged_attn_triton_prefill_i8f8_cached(
            q, tq, layer.get_qkv(), bt, cs_past, causal=True, pv_mode=pv)
        mx, rel = rel_err(out, ref)
        print(f"cached kernel pv={pv} vs round-1 ref maxabs={mx:.5f} rel={rel:.4f}", flush=True)
        assert mx < 0.02, f"cached kernel (pv={pv}) disagrees with ref: maxabs={mx:.5f}"
    assert CALLS["cached"] == before  # wrapper does not count; dispatch does


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_kernel_partial_tail(device, monkeypatch):
    """Appends leaving a staged tail across several calls: kernel vs ref."""
    from exllamav3.modules.attention_fn.int8fp8 import sdpa_int8fp8_ref
    from exllamav3.modules.attention_fn.i8f8_cached import paged_attn_triton_prefill_i8f8_cached
    monkeypatch.setenv("EXL3_ATTN_INT8FP8", "1")
    hq, hkv, dim = 8, 2, 128
    layer = _layer(device, 2048, hkv, dim)
    bt = torch.arange(8, device=device, dtype=torch.int32).view(1, 8)
    cs = torch.tensor([0], device=device, dtype=torch.int32)
    g = torch.Generator(device=device).manual_seed(3)
    kept_k, kept_v = [], []
    for L in (100, 7, 250):
        k = torch.randn((1, L, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
        v = torch.randn((1, L, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
        kept_k.append(k)
        kept_v.append(v)
        cs = _append(layer, k, v, cs, bt)
    full_k = torch.cat(kept_k, dim=1)
    full_v = torch.cat(kept_v, dim=1)
    tq = 37  # causal prefill chunk ending mid-block (total 357+37=394, tail 10)
    q = torch.randn((1, tq, hq, dim), generator=g, device=device, dtype=torch.float16)
    k = torch.randn((1, tq, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    v = torch.randn((1, tq, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    cs_past = cs.clone()
    cs = _append(layer, k, v, cs, bt)
    ref = sdpa_int8fp8_ref(q, torch.cat([full_k, k], dim=1), torch.cat([full_v, v], dim=1),
                           causal=True)
    for pv in (0, 1):
        out = paged_attn_triton_prefill_i8f8_cached(
            q, tq, layer.get_qkv(), bt, cs_past, causal=True, pv_mode=pv)
        mx, rel = rel_err(out, ref)
        print(f"tail kernel pv={pv} vs ref maxabs={mx:.5f} rel={rel:.4f}", flush=True)
        assert mx < 0.02, f"tail path (pv={pv}) disagrees: maxabs={mx:.5f}"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_dispatch_cached_and_fallback(device, monkeypatch):
    """Dispatch with a CacheLayer_i8f8: env on -> cached kernel (counted) and
    near the fp16 kernel; env off or decode -> fp16 dequant fallback."""
    from exllamav3.modules.attention_fn.dispatch import attn_dispatch
    from exllamav3.modules.attention_fn.i8f8_cached import CALLS
    from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
    hq, hkv, dim = 8, 2, 128
    layer = _layer(device, 2048, hkv, dim)
    bt = torch.arange(8, device=device, dtype=torch.int32).view(1, 8)
    g = torch.Generator(device=device).manual_seed(4)
    past, tq = 960, 64
    kp = torch.randn((1, past, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    vp = torch.randn((1, past, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    k = torch.randn((1, tq, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    v = torch.randn((1, tq, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    q = torch.randn((1, tq, hq, dim), generator=g, device=device, dtype=torch.float16)
    # The past must already live in the cache (as in real use): pre-fill it.
    layer.update_kv_direct(torch.tensor([0], device=device, dtype=torch.int32),
                           bt, kp, vp, past)
    cs = torch.tensor([past], device=device, dtype=torch.int32)
    n0 = CALLS["cached"]
    monkeypatch.setenv("EXL3_ATTN_INT8FP8", "1")
    got = attn_dispatch(q, k, v, cache=layer, cache_idx=None, cache_instance=None,
                        causal=True, sm_scale=1.0 / (dim ** 0.5),
                        block_table=bt, cache_seqlens=cs, dispatch_cache={})
    assert got is not None
    assert CALLS["cached"] == n0 + 1, "cached kernel did not run (silent fallback?)"
    # fp16 reference over an equivalent fp16 cache with the same rows
    from exllamav3.cache.fp16 import CacheLayer_fp16
    from types import SimpleNamespace as NS
    ref_layer = CacheLayer_fp16(None, NS(num_kv_heads=hkv, head_dim=dim), 0, 2048)
    ref_layer.alloc(torch.device(device))
    ref_layer.k.view(-1, hkv, dim)[:past].copy_(kp[0])
    ref_layer.v.view(-1, hkv, dim)[:past].copy_(vp[0])
    fp16 = paged_attn_triton_prefill(q, k, v, ref_layer.k, ref_layer.v, bt,
                                    torch.tensor([past], device=device, dtype=torch.int32),
                                    causal=True)
    mx, rel = rel_err(got, fp16)
    print(f"dispatch cached vs fp16 maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    assert rel < 0.08
    # Env off -> dequant fallback, cached counter untouched, still correct
    monkeypatch.delenv("EXL3_ATTN_INT8FP8", raising=False)
    n1 = CALLS["cached"]
    cs2 = torch.tensor([past + tq], device=device, dtype=torch.int32)
    q2 = torch.randn((1, 1, hq, dim), generator=g, device=device, dtype=torch.float16)
    k2 = torch.randn((1, 1, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    v2 = torch.randn((1, 1, hkv, dim), generator=g, device=device, dtype=torch.float16) * 0.5
    got2 = attn_dispatch(q2, k2, v2, cache=layer, cache_idx=None, cache_instance=None,
                         causal=True, sm_scale=1.0 / (dim ** 0.5),
                         block_table=bt, cache_seqlens=cs2, dispatch_cache={})
    assert got2 is not None
    assert CALLS["cached"] == n1, "cached kernel must not run with the switch off"
    assert got2.shape == (1, 1, hq, dim)


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_cached_real_qkv_capture(device):
    """Real 27B Q/K/V through the packed cache + cached kernel vs fp16.
    Skips when the capture is absent (same path as the round-1 test)."""
    path = os.environ.get("EXL3_INT8FP8_QKV_PATH", "/tmp/i8f8_qkv_27b.pt")
    if not os.path.exists(path):
        pytest.skip(f"no capture at {path}; run eval/prof/capture_qkv_27b.py")
    from exllamav3.modules.attention_fn.i8f8_cached import (
        paged_attn_triton_prefill_i8f8_cached, CALLS)
    cap = torch.load(path, map_location=device, weights_only=True)
    n0 = CALLS["cached"]
    worst = 0.0
    for i, layer in enumerate(cap["layers"][:4]):
        hq = layer["q"].shape[2]
        hkv, dim = layer["k"].shape[2], layer["k"].shape[3]
        tq, tk = layer["q"].shape[1], layer["k"].shape[1]
        cl = _layer(device, ((tk + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE, hkv, dim)
        pages = cl.num_pages
        bt = torch.arange(pages, device=device, dtype=torch.int32).view(1, pages)
        cs0 = torch.tensor([0], device=device, dtype=torch.int32)
        kk = layer["k"].to(device).half().contiguous()
        vv = layer["v"].to(device).half().contiguous()
        qq = layer["q"].to(device).half().contiguous()
        cl.update_kv_direct(cs0, bt, kk, vv, tk)
        got = paged_attn_triton_prefill_i8f8_cached(
            qq, tq, cl.get_qkv(), bt, cs0, causal=True)
        base = torch.nn.functional.scaled_dot_product_attention(
            qq.float().transpose(1, 2).reshape(1 * hq, tq, dim),
            kk.float().transpose(1, 2).repeat_interleave(hq // hkv, dim=1).reshape(1 * hq, tk, dim),
            vv.float().transpose(1, 2).repeat_interleave(hq // hkv, dim=1).reshape(1 * hq, tk, dim),
            is_causal=True).reshape(1, hq, tq, dim).transpose(1, 2).half()
        mx, rel = rel_err(got, base)
        print(f"layer {cap['layer_idx'][i]} cached-vs-fp16 maxabs={mx:.5f} rel={rel:.4f}", flush=True)
        worst = max(worst, rel)
    assert CALLS["cached"] == n0  # wrapper-level call; dispatch counts separately
    assert worst < 0.08, f"real-data cached error too large: rel={worst:.4f}"


@pytest.mark.parametrize("device", devices)
def test_copy_page_and_storage(device):
    layer = _layer(device, 1024, 2, 64)
    layer2 = _layer(device, 1024, 2, 64)
    n = layer.storage_size()
    pages, S, H, D = 4, PAGE_SIZE, 2, 64
    expect = pages * S * H * D * 2 + pages * 8 * H * (4 + 2 * D * 2) + 2 * pages * QB * H * D * 2 + pages * 4
    assert n == expect, f"{n} != {expect}"
    # ~0.53x of fp16 (2*P*S*H*D*2 bytes)
    assert n < 2 * pages * S * H * D * 2
    layer.copy_page(layer, 0, 1, 100)
    assert int(layer.stage_len[1].item()) == 100 % QB
    assert torch.equal(layer.k_q[1, :100], layer.k_q[0, :100])
    layer.free()
    layer2.free()
    assert layer.k_q is None
