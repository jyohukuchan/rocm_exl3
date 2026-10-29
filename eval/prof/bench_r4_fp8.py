#!/usr/bin/env python3
"""Round 4 bench: fp16 Triton prefill vs r4 all-FP8 (A) / INT8-QK+FP8-PV (B).

27B full-attention geometry (B=1, 24/4/256, causal). The r4 kernel reads a
pre-quantized paged cache (pure reader); the quantize pre-pass is timed
separately from the kernel so kernel speed is isolated. Interleaved A/B
rounds; prints median/min/max + correctness vs the r4 torch reference and
vs fp16.

Usage: bench_r4_fp8.py [--tk 8192 32768 131072] [--tq 2048 8192]
       [--iters 7] [--rounds 3] [--modes fp8 int8] [--smoke]
       [--block-m N] [--block-n N] [--warps N] [--stages N]
       [--splits N] [--extra-knobs '{"kpack": 2}']
Env: HIP_VISIBLE_DEVICES isolates one GPU (R9700 numbers only).
"""
import argparse
import json
import statistics as st
import sys
import time

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

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
from exllamav3.modules.attention_fn.r4_fp8 import (
    r4_quantize, r4_to_paged, r4_prefill, sdpa_r4_ref)

B, Hq, Hkv, D, PAGE = 1, 24, 4, 256, 256
SM = 1.0 / (D ** 0.5)
torch.manual_seed(0)


def build(Tk, Tq):
    q = torch.randn((B, Tq, Hq, D), device="cuda:0", dtype=torch.float16)
    kf = (torch.randn((B, Tk, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5).contiguous()
    vf = (torch.randn((B, Tk, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5).contiguous()
    # fp16 paged cache holding the FULL window (timed run appends the Tq new
    # rows over identical values; correctness subsets read it directly).
    past = Tk - Tq
    pages = Tk // PAGE
    kc = torch.empty((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16)
    vc = torch.empty((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16)
    kc.view(1, Tk, Hkv, D).copy_(kf)
    vc.view(1, Tk, Hkv, D).copy_(vf)
    bt = torch.arange(pages, device="cuda:0", dtype=torch.int32).view(1, pages)
    cs = torch.tensor([past], device="cuda:0", dtype=torch.int32)
    k = kf[:, past:].contiguous()
    v = vf[:, past:].contiguous()
    return q, k, v, kf, vf, kc, vc, bt, cs


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


def rel_err(got, base):
    d = (got.float() - base.float()).abs()
    return d.max().item(), (d.max() / base.float().abs().amax().clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", nargs="+", type=int, default=[8192, 32768, 131072])
    ap.add_argument("--tq", nargs="+", type=int, default=[2048, 8192])
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["fp8", "int8"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--block-m", type=int, default=None)
    ap.add_argument("--block-n", type=int, default=None)
    ap.add_argument("--warps", type=int, default=None)
    ap.add_argument("--stages", type=int, default=None)
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--extra-knobs", type=str, default=None)
    ap.add_argument("--fp16-block-m", type=int, default=None)
    ap.add_argument("--fp16-block-n", type=int, default=None)
    ap.add_argument("--fp16-warps", type=int, default=None)
    ap.add_argument("--fp16-stages", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.tk, args.tq, args.iters, args.rounds = [1024], [128], 3, 1
    extra = json.loads(args.extra_knobs) if args.extra_knobs else {}
    kw = dict(block_m=args.block_m, block_n=args.block_n, num_warps=args.warps,
              num_stages=args.stages, num_splits=args.splits, extra_launch=extra)
    kw = {k: v for k, v in kw.items() if v is not None or k == "extra_launch"}
    fkw = dict(block_m=args.fp16_block_m, block_n=args.fp16_block_n,
               num_warps=args.fp16_warps, num_stages=args.fp16_stages)
    fkw = {k: v for k, v in fkw.items() if v is not None}

    for Tk in args.tk:
        for Tq in args.tq:
            assert Tk % PAGE == 0 and Tq <= Tk
            q, k, v, kf, vf, kc, vc, bt, cs = build(Tk, Tq)
            out_fp16 = torch.empty_like(q)

            def fp16_fn():
                paged_attn_triton_prefill(q, k, v, kc, vc, bt, cs, causal=True,
                                          softmax_scale=SM, out=out_fp16, **fkw)

            # Pre-quantize once (outside timing); kernel inputs are views.
            # Warm up once so CUDA init is not billed to the first mode.
            _w = r4_quantize(q[:, :1], kf[:, :1], vf[:, :1], SM, args.modes[0])
            torch.cuda.synchronize()
            quants = {}
            for mode in args.modes:
                tq_ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    quant = r4_quantize(q, kf, vf, SM, mode)
                    torch.cuda.synchronize()
                    tq_ts.append((time.perf_counter() - t0) * 1000)
                t_q = sorted(tq_ts)[1]
                kc4, vc4, bt4, cs4 = r4_to_paged(quant)
                quants[mode] = (quant, kc4, vc4, bt4, cs4, t_q)

            # Correctness on a 128-row subset (full-window scores would OOM
            # the torch reference at large Tk). Also warms up / compiles.
            fp16_fn()
            torch.cuda.synchronize()
            print(f"Tk={Tk} Tq={Tq}:", flush=True)
            n_sub = min(Tq, 128)
            q_sub = q[:, :n_sub].contiguous()
            cs_full = torch.tensor([Tk], device="cuda:0", dtype=torch.int32)
            out_sub = torch.empty_like(q_sub)
            paged_attn_triton_prefill(q_sub, None, None, kc, vc, bt, cs_full,
                                      causal=True, softmax_scale=SM, out=out_sub)
            torch.cuda.synchronize()
            for mode in args.modes:
                quant, kc4, vc4, bt4, cs4, t_q = quants[mode]
                got = r4_prefill(quant["q_q"][:, :n_sub].contiguous(), kc4, vc4, bt4, cs4,
                                 quant["row_log2e"][:, :n_sub].contiguous(),
                                 quant["v_scale"],
                                 causal=True, mode=mode, **kw)
                torch.cuda.synchronize()
                ref = sdpa_r4_ref(q_sub, kf, vf, causal=True, sm_scale=SM, mode=mode)
                mk, rk = rel_err(got, ref)
                mf, rf = rel_err(got, out_sub)
                mr, rr = rel_err(ref, out_sub)
                print(f"  r4-{mode}: vs-ref maxabs={mk:.5f} rel={rk:.4f} "
                      f"| vs-fp16 maxabs={mf:.4f} rel={rf:.4f} | ref-vs-fp16 rel={rr:.4f} "
                      f"(amax got={got.float().abs().amax():.5f} ref={ref.float().abs().amax():.5f} "
                      f"fp16={out_sub.float().abs().amax():.5f}) | quant-pass={t_q:.2f}ms",
                      flush=True)

            # Interleaved A/B timing rounds.
            for r in range(args.rounds):
                fmed, fmn, fmx = time_fn(fp16_fn, args.iters)
                print(f"  [r{r}] fp16 med={fmed:.2f} min={fmn:.2f} max={fmx:.2f}ms", flush=True)
                for mode in args.modes:
                    quant, kc4, vc4, bt4, cs4, t_q = quants[mode]
                    out = torch.empty_like(q)

                    def kfn(_q=(quant, kc4, vc4, bt4, cs4, out)):
                        _quant, _kc, _vc, _bt, _cs, _o = _q
                        r4_prefill(_quant["q_q"], _kc, _vc, _bt, _cs,
                                   _quant["row_log2e"], _quant["v_scale"],
                                   causal=True, mode=mode, **kw, out=_o)

                    med, mn, mx = time_fn(kfn, args.iters)
                    e2e = med + t_q
                    print(f"  [r{r}] r4-{mode} kernel med={med:.2f} min={mn:.2f} max={mx:.2f}ms "
                          f"(x{fmed / med:.2f} vs fp16) e2e~{e2e:.2f}ms",
                          flush=True)
            # free before the next shape
            del q, k, v, kf, vf, kc, vc, bt, cs, out_fp16, quants
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
