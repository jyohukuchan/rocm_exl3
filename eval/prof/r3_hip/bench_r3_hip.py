#!/usr/bin/env python3
"""Round-3 HIP prototype: correctness + speed vs Triton fp16 / Triton cached.

Builds eval/prof/r3_hip/hip_i8f8_attn.hip via torch.utils.cpp_extension and:
  1. unit-tests the RNE E4M3 converter vs torch .to(float8_e4m3fn),
  2. checks the HIP kernel vs the round-1 torch reference (random + odd sizes
     with staged tails, causal/non-causal),
  3. benches fp16 Triton prefill vs Triton cached vs HIP cached (interleaved).

Usage: python3 bench_r3_hip.py [--tk 8192 32768] [--tq 2048] [--iters 7]
Env: HIP_VISIBLE_DEVICES must isolate ONE gfx1201 (rule: one GPU per process).
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, "/src")

import torch

torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
print(f"device: {props.name} {props.gcnArchName}", flush=True)

from torch.utils.cpp_extension import load

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hip_i8f8_attn.hip")
hip = load(name="hip_i8f8_attn", sources=[SRC],
           extra_cflags=["-O3"],
           extra_cuda_cflags=["-O3", "--offload-arch=gfx1201"],
           verbose=False)
print("hip ext loaded", flush=True)

from types import SimpleNamespace

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
from exllamav3.modules.attention_fn.i8f8_cached import (
    paged_attn_triton_prefill_i8f8_cached)
from exllamav3.modules.attention_fn.int8fp8 import sdpa_int8fp8_ref
from exllamav3.cache.i8f8 import CacheLayer_i8f8

PAGE, QB = 256, 32


def _layer(device, max_tokens, hkv, dim):
    attn = SimpleNamespace(num_kv_heads=hkv, head_dim=dim)
    layer = CacheLayer_i8f8(None, attn, 0, max_tokens)
    layer.alloc(torch.device(device))
    return layer


def test_e4m3():
    torch.manual_seed(0)
    # P-path ranges (P/sp never exceeds ~448; torch emits 0x7F past ~460,
    # a quirk outside every real input, so cap the top range at 460).
    for lo, hi in ((0.0, 1.0), (0.0, 448.0), (0.0, 0.01), (400.0, 460.0),
                   (1e-4, 2e-3), (2.0, 3.0)):
        x = (torch.rand(65536, device="cuda:0") * (hi - lo) + lo).float()
        got = hip.test_e4m3(x)
        ref = x.to(torch.float8_e4m3fn).view(torch.uint8)
        bad = int((got != ref).sum())
        print(f"e4m3 [{lo},{hi}]: mismatches={bad}/65536", flush=True)
        assert bad == 0, f"E4M3 converter diverges in [{lo},{hi}]"


def hip_run(q, layer, bt, past, causal=True):
    # prototype kernel is B=1 with 3-D q/out views (no copy: row 0 of batch 1)
    out = torch.empty_like(q)
    hip.prefill(q[0], *layer.get_qkv(), bt[0].contiguous(), out[0],
                q.shape[1] + int(past.item()), int(past.item()),
                float(1.0 / (q.shape[3] ** 0.5)), causal)
    torch.cuda.synchronize()
    return out


@torch.inference_mode()
def test_correct(Tk=1024, Tq=64, hq=8, hkv=2, dim=256, causal=True, seed=0):
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(seed)
    layer = _layer(dev, ((Tk + PAGE - 1) // PAGE) * PAGE, hkv, dim)
    pages = layer.num_pages
    bt = torch.arange(pages, device=dev, dtype=torch.int32).view(1, pages)
    past = Tk - Tq
    kp = torch.randn((1, past, hkv, dim), generator=g, device=dev, dtype=torch.float16) * 0.5
    vp = torch.randn((1, past, hkv, dim), generator=g, device=dev, dtype=torch.float16) * 0.5
    k = torch.randn((1, Tq, hkv, dim), generator=g, device=dev, dtype=torch.float16) * 0.5
    v = torch.randn((1, Tq, hkv, dim), generator=g, device=dev, dtype=torch.float16) * 0.5
    q = torch.randn((1, Tq, hq, dim), generator=g, device=dev, dtype=torch.float16)
    cs = torch.tensor([0], device=dev, dtype=torch.int32)
    if past:
        layer.update_kv_direct(cs, bt, kp, vp, past)
    cs_past = torch.tensor([past], device=dev, dtype=torch.int32)
    layer.update_kv_direct(cs_past, bt, k, v, Tq)
    got = hip_run(q, layer, bt, cs_past, causal)
    full_k = torch.cat([kp, k], dim=1)
    full_v = torch.cat([vp, v], dim=1)
    ref = sdpa_int8fp8_ref(q, full_k, full_v, causal=causal)
    d = (got.float() - ref.float()).abs()
    mx = d.max().item()
    rel = mx / ref.float().abs().amax().item()
    print(f"correct Tk={Tk} Tq={Tq} causal={causal}: maxabs={mx:.5f} rel={rel:.4f}", flush=True)
    # Triton cached on the same inputs (sanity: should also match ref closely)
    tri = paged_attn_triton_prefill_i8f8_cached(
        q, Tq, layer.get_qkv(), bt, cs_past.clone(), causal=causal)
    d2 = (tri.float() - ref.float()).abs()
    print(f"  triton-cached vs ref: maxabs={d2.max().item():.5f}", flush=True)
    assert mx < 0.03, f"HIP kernel diverges: maxabs={mx:.5f}"
    return mx


def time_fn(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2], min(ts), max(ts)


def bench(Tk, Tq, iters, rounds, hq=24, hkv=4, dim=256):
    dev = "cuda:0"
    SM = 1.0 / (dim ** 0.5)
    pages = Tk // PAGE
    kc = torch.randn((pages, PAGE, hkv, dim), device=dev, dtype=torch.float16) * 0.5
    vc = torch.randn((pages, PAGE, hkv, dim), device=dev, dtype=torch.float16) * 0.5
    bt = torch.arange(pages, device=dev, dtype=torch.int32).view(1, pages)
    cs = torch.tensor([Tk - Tq], device=dev, dtype=torch.int32)
    q = torch.randn((1, Tq, hq, dim), device=dev, dtype=torch.float16)
    k = torch.randn((1, Tq, hkv, dim), device=dev, dtype=torch.float16) * 0.5
    v = torch.randn((1, Tq, hkv, dim), device=dev, dtype=torch.float16) * 0.5
    layer = _layer(dev, Tk, hkv, dim)
    past = Tk - Tq
    if past:
        layer.update_kv_direct(torch.tensor([0], device=dev, dtype=torch.int32), bt,
                               kc.view(1, Tk, hkv, dim)[:, :past].contiguous(),
                               vc.view(1, Tk, hkv, dim)[:, :past].contiguous(), past)
    layer.update_kv_direct(cs, bt, k, v, Tq)
    out = torch.empty_like(q)
    a = paged_attn_triton_prefill(q, k, v, kc.clone(), vc.clone(), bt, cs,
                                  causal=True, softmax_scale=SM)
    c = hip_run(q, layer, bt, cs, True)
    d = (a.float() - c.float()).abs()
    print(f"Tk={Tk} Tq={Tq}: hip-vs-fp16 maxabs={d.max().item():.5f} "
          f"rel={d.max().item() / a.float().abs().amax().item():.4f}", flush=True)
    fns = {
        "fp16": lambda: paged_attn_triton_prefill(q, k, v, kc, vc, bt, cs,
                                                 causal=True, softmax_scale=SM),
        "cached-tri": lambda: paged_attn_triton_prefill_i8f8_cached(
            q, Tq, layer.get_qkv(), bt, cs, causal=True),
        "cached-hip": lambda: hip.prefill(q[0], *layer.get_qkv(), bt[0].contiguous(), out[0],
                                          Tk, past, SM, True),
    }
    res = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            med, mn, mx = time_fn(fn, iters)
            res[name].append((med, mn, mx))
    for name, vals in res.items():
        meds = [x[0] for x in vals]
        detail = " ".join(f"{m:.2f}[{mn:.2f}-{mx:.2f}]" for m, mn, mx in vals)
        print(f"Tk={Tk} Tq={Tq} {name:12s} overall_med={st.median(meds):.2f}ms "
              f"rounds={detail}", flush=True)
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", type=int, nargs="+", default=[8192, 32768, 131072])
    ap.add_argument("--tq", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    test_e4m3()
    # correctness: odd sizes (tails), small + 27B-like head_dim
    test_correct(1024, 64)
    test_correct(357 + 37, 37)
    test_correct(1024, 64, causal=False)
    test_correct(2048, 128, hq=24, hkv=4, dim=256, seed=1)
    if not args.skip_bench:
        with torch.inference_mode():
            for Tk in args.tk:
                for Tq in args.tq:
                    if Tq > Tk:
                        continue
                    bench(Tk, Tq, args.iters, args.rounds)
    print("R3_HIP_OK")


if __name__ == "__main__":
    main()
