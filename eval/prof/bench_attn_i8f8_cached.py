#!/usr/bin/env python3
"""Round-2 microbench: fp16 prefill vs round-1 on-the-fly vs cached i8f8.

27B full-attention geometry (B=1, 24/4/256). For each (Tk, Tq): fills an fp16
paged cache with the past, then times fp16 / round-1 (update+attend) /
cached (quantize-on-write + cached kernel, also reported split). Interleaved
A/B rounds; prints median/min/max + correctness vs fp16 + CALLS counters
(proof the cached path ran) + bytes/token.

Usage: bench_attn_i8f8_cached.py [--tk 8192 32768 131072] [--tq 2048 8192]
       [--iters 7] [--rounds 3]
Env: HIP_VISIBLE_DEVICES isolates one GPU (report R9700 numbers only).
"""
import argparse
import statistics as st
import sys

sys.path.insert(0, "/src")

import torch

torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
print(f"device: {props.name} {props.gcnArchName} "
      f"{props.total_memory / 1e9:.1f}GB sm={props.multi_processor_count}", flush=True)
free, total = torch.cuda.mem_get_info()
print(f"cuda free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB", flush=True)
import triton
print(f"triton: {triton.__version__}", flush=True)

from types import SimpleNamespace

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
from exllamav3.modules.attention_fn.int8fp8 import paged_attn_triton_prefill_int8fp8
from exllamav3.modules.attention_fn.i8f8_cached import (
    paged_attn_triton_prefill_i8f8_cached, CALLS)
from exllamav3.cache.i8f8 import CacheLayer_i8f8

B, Hq, Hkv, D, PAGE = 1, 24, 4, 256, 256
SM = 1.0 / (D ** 0.5)
torch.manual_seed(0)


def build(Tk, Tq):
    pages = Tk // PAGE
    kc = torch.randn((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    vc = torch.randn((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    bt = torch.arange(pages, device="cuda:0", dtype=torch.int32).view(1, pages)
    cs = torch.tensor([Tk - Tq], device="cuda:0", dtype=torch.int32)
    q = torch.randn((B, Tq, Hq, D), device="cuda:0", dtype=torch.float16)
    k = torch.randn((B, Tq, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    v = torch.randn((B, Tq, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    return q, k, v, kc, vc, bt, cs


def make_i8f8(Tk, Tq, kc, vc, bt):
    layer = CacheLayer_i8f8(None, SimpleNamespace(num_kv_heads=Hkv, head_dim=D), 0, Tk)
    layer.alloc(torch.device("cuda:0"))
    past = Tk - Tq
    if past:
        layer.update_kv_direct(torch.tensor([0], device="cuda:0", dtype=torch.int32), bt,
                               kc.view(1, Tk, Hkv, D)[:, :past].contiguous(),
                               vc.view(1, Tk, Hkv, D)[:, :past].contiguous(), past)
    return layer


def time_fn(fn, iters):
    for _ in range(3):
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


def bytes_per_token():
    # K int8 + mu fp16/32 + sc fp32/32 + V fp8 + vsc fp16/32, per kv head
    per_head = D * 1 + D * 2 / 32 + 4 / 32 + D * 1 + D * 2 / 32
    return Hkv * per_head, Hkv * D * 2 * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", type=int, nargs="+", default=[8192, 32768, 131072])
    ap.add_argument("--tq", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    bpt_new, bpt_fp16 = bytes_per_token()
    print(f"bytes/token/layer: cached={bpt_new:.1f} fp16={bpt_fp16:.1f} "
          f"ratio={bpt_new / bpt_fp16:.3f}", flush=True)
    with torch.inference_mode():
        for Tk in args.tk:
            for Tq in args.tq:
                if Tq > Tk:
                    continue
                q, k, v, kc, vc, bt, cs = build(Tk, Tq)
                past = Tk - Tq
                layer = make_i8f8(Tk, Tq, kc, vc, bt)
                a = paged_attn_triton_prefill(q, k, v, kc.clone(), vc.clone(), bt, cs,
                                              causal=True, softmax_scale=SM)
                b = paged_attn_triton_prefill_int8fp8(q, k, v, kc.clone(), vc.clone(), bt, cs,
                                                      causal=True)
                # Steady state: the new rows are already in the cache (the
                # kernel is a pure reader); same for the timed runs below.
                layer.update_kv_direct(cs, bt, k, v, Tq)
                c = paged_attn_triton_prefill_i8f8_cached(
                    q, Tq, layer.get_qkv(), bt, cs.clone(), causal=True)
                for name, x in (("round1", b), ("cached", c)):
                    d = (a.float() - x.float()).abs()
                    rel = d.max().item() / a.float().abs().amax().item()
                    print(f"Tk={Tk} Tq={Tq}: {name}-vs-fp16 maxabs={d.max().item():.5f} "
                          f"rel={rel:.4f} cached_calls={CALLS['cached']}", flush=True)
                fns = {
                    "fp16": lambda: paged_attn_triton_prefill(
                        q, k, v, kc, vc, bt, cs, causal=True, softmax_scale=SM),
                    "round1": lambda: paged_attn_triton_prefill_int8fp8(
                        q, k, v, kc, vc, bt, cs, causal=True),
                    "cached": lambda: paged_attn_triton_prefill_i8f8_cached(
                        q, Tq, layer.get_qkv(), bt, cs, causal=True),
                    "cached-update": lambda: layer.update_kv_direct(cs, bt, k, v, Tq),
                }
                res = {name: [] for name in fns}
                for _ in range(args.rounds):
                    for name, fn in fns.items():
                        med, mn, mx = time_fn(fn, args.iters)
                        res[name].append((med, mn, mx))
                for name, vals in res.items():
                    meds = [x[0] for x in vals]
                    detail = " ".join(f"{m:.2f}[{mn:.2f}-{mx:.2f}]" for m, mn, mx in vals)
                    print(f"Tk={Tk} Tq={Tq} {name:14s} overall_med={st.median(meds):.2f}ms "
                          f"rounds={detail}", flush=True)
                del q, k, v, kc, vc, a, b, c, layer
                torch.cuda.empty_cache()
    print("BENCH_OK")


if __name__ == "__main__":
    main()
