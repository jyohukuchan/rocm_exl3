#!/usr/bin/env python3
"""Attention-kernel microbench: in-repo fp16 Triton prefill vs opt-in
INT8-QK / FP8-PV prefill (Qwen3.8-27B full-attention geometry).

Usage: bench_attn_int8fp8.py [--tk 8192 32768 131072] [--iters 7] [--rounds 3]
Env: HIP_VISIBLE_DEVICES isolates one GPU. Run fp16 and int8fp8 interleaved
in the same process; reports median/min/max per round plus overall median.
Intended for R9700 (gfx1201); also runs on V620 (FP8 falls back to proxy).
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
import triton
print(f"triton: {triton.__version__}", flush=True)

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
from exllamav3.modules.attention_fn.int8fp8 import paged_attn_triton_prefill_int8fp8

B, Tq, Hq, Hkv, D, PAGE = 1, 1024, 24, 4, 256, 256
SM = 1.0 / (D ** 0.5)
torch.manual_seed(0)


def build(Tk):
    pages = Tk // PAGE
    kc = torch.randn((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    vc = torch.randn((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    bt = torch.arange(pages, device="cuda:0", dtype=torch.int32).view(1, pages)
    cs = torch.tensor([Tk - Tq], device="cuda:0", dtype=torch.int32)
    q = torch.randn((B, Tq, Hq, D), device="cuda:0", dtype=torch.float16)
    k = torch.randn((B, Tq, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    v = torch.randn((B, Tq, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    return q, k, v, kc, vc, bt, cs


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", type=int, nargs="+", default=[8192, 32768, 131072])
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    with torch.inference_mode():
        for Tk in args.tk:
            q, k, v, kc, vc, bt, cs = build(Tk)
            a = paged_attn_triton_prefill(q, k, v, kc.clone(), vc.clone(), bt, cs,
                                          causal=True, softmax_scale=SM)
            b = paged_attn_triton_prefill_int8fp8(q, k, v, kc.clone(), vc.clone(), bt, cs,
                                                 causal=True)
            d = (a.float() - b.float()).abs()
            rel = d.max().item() / a.float().abs().amax().item()
            print(f"Tk={Tk}: int8fp8-vs-fp16 maxabs={d.max().item():.5f} rel={rel:.4f}",
                  flush=True)
            from exllamav3.modules.attention_fn.int8fp8 import _has_fp8_arch
            print(f"Tk={Tk}: has_fp8_arch={_has_fp8_arch(q.device)}", flush=True)
            fns = {
                "fp16": lambda: paged_attn_triton_prefill(
                    q, k, v, kc, vc, bt, cs, causal=True, softmax_scale=SM),
                "i8f8-pv0": lambda: paged_attn_triton_prefill_int8fp8(
                    q, k, v, kc, vc, bt, cs, causal=True, pv_mode=0),
                "i8f8-auto": lambda: paged_attn_triton_prefill_int8fp8(
                    q, k, v, kc, vc, bt, cs, causal=True),
            }
            try:
                b2 = paged_attn_triton_prefill_int8fp8(q, k, v, kc.clone(), vc.clone(), bt, cs,
                                                      causal=True, pv_mode=2)
                d2 = (a.float() - b2.float()).abs()
                print(f"Tk={Tk}: pv2-vs-fp16 maxabs={d2.max().item():.5f} "
                      f"rel={d2.max().item() / a.float().abs().amax().item():.4f}", flush=True)
                fns["i8f8-pv2"] = lambda: paged_attn_triton_prefill_int8fp8(
                    q, k, v, kc, vc, bt, cs, causal=True, pv_mode=2)
                del b2
            except Exception as e:
                print(f"Tk={Tk}: pv2 unavailable: {type(e).__name__}", flush=True)
            res = {name: [] for name in fns}
            for _ in range(args.rounds):
                for name, fn in fns.items():
                    med, mn, mx = time_fn(fn, args.iters)
                    res[name].append((med, mn, mx))
            for name, vals in res.items():
                meds = [x[0] for x in vals]
                detail = " ".join(f"{m:.2f}[{mn:.2f}-{mx:.2f}]" for m, mn, mx in vals)
                print(f"Tk={Tk} {name:9s} overall_med={st.median(meds):.2f}ms "
                      f"rounds={detail}", flush=True)
            del q, k, v, kc, vc, a, b
            torch.cuda.empty_cache()
    print("BENCH_OK")


if __name__ == "__main__":
    main()
