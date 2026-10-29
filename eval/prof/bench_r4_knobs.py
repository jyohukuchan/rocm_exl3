#!/usr/bin/env python3
"""Round 4 kernel knob sweep (single process, shared data/compiles).

Sweeps BLOCK_M/BLOCK_N, num_warps/stages and AMD knobs (kpack,
matrix_instr_nonkdim, waves_per_eu, schedule_hint) for the r4 fp8/int8
kernels at Tk=32768/Tq=2048 (27B geometry), interleaved against the fp16
default. --gcn prints the AMDGCN instruction mix for fp16 vs r4.

Usage: bench_r4_knobs.py [--modes fp8 int8] [--gcn] [--verify]
Env: HIP_VISIBLE_DEVICES isolates one GPU (R9700 numbers only).
"""
import argparse
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
from exllamav3.modules.attention_fn.r4_fp8 import (
    r4_quantize, r4_to_paged, r4_prefill, _r4_prefill_kernel)
from exllamav3.modules.attention_fn import triton_paged as TP

B, Hq, Hkv, D, PAGE = 1, 24, 4, 256, 256
SM = 1.0 / (D ** 0.5)
Tk, Tq = 32768, 2048
torch.manual_seed(0)

OPS = ["v_wmma", "v_mfma", "mfma", "wmma", "ds_read", "ds_write",
       "buffer_load", "buffer_store", "flat_load", "flat_store",
       "v_dot", "v_fma", "v_cvt", "v_perm", "v_permlane",
       "s_barrier", "s_waitcnt"]


def mix(gcn):
    lines = gcn.splitlines()
    code = [l for l in lines if l.strip() and not l.strip().startswith((".", "/", ";"))]
    m = {op: sum(1 for l in code if op in l) for op in OPS}
    return {"total_lines": len(lines), "code_lines": len(code),
            "mix": {k: v for k, v in m.items() if v}}


def build():
    q = torch.randn((B, Tq, Hq, D), device="cuda:0", dtype=torch.float16)
    kf = (torch.randn((B, Tk, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5).contiguous()
    vf = (torch.randn((B, Tk, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5).contiguous()
    past = Tk - Tq
    pages = Tk // PAGE
    kc = torch.empty((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16)
    vc = torch.empty((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16)
    kc.view(1, Tk, Hkv, D).copy_(kf)
    vc.view(1, Tk, Hkv, D).copy_(vf)
    bt = torch.arange(pages, device="cuda:0", dtype=torch.int32).view(1, pages)
    cs = torch.tensor([past], device="cuda:0", dtype=torch.int32)
    return q, kf, vf, kc, vc, bt, cs


def time_fn(fn, iters=5):
    for _ in range(2):
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


CONFIGS = [
    ("base", {}),
    ("stages=1", {"num_stages": 1}),
    ("stages=3", {"num_stages": 3}),
    ("kpack=2", {"extra_launch": {"kpack": 2}}),
    ("nonkdim=16", {"extra_launch": {"matrix_instr_nonkdim": 16}}),
    ("waves=2", {"extra_launch": {"waves_per_eu": 2}}),
    ("sched=attn", {"extra_launch": {"schedule_hint": "attention"}}),
    ("BN=64", {"block_n": 64}),
    ("BN=64+S1", {"block_n": 64, "num_stages": 1}),
    ("BN=64+kpack2", {"block_n": 64, "extra_launch": {"kpack": 2}}),
    ("BM=128", {"block_m": 128}),
    ("BM=128+BN=64", {"block_m": 128, "block_n": 64}),
    ("BM=128+BN=64+S1", {"block_m": 128, "block_n": 64, "num_stages": 1}),
    ("warps=4", {"num_warps": 4}),
    ("BM=256", {"block_m": 256}),
    ("BM=256+S1", {"block_m": 256, "num_stages": 1}),
    ("BM=256+BN=64+S1", {"block_m": 256, "block_n": 64, "num_stages": 1}),
]


FP16_CONFIGS = [
    ("base", {}),
    ("BM=128", {"block_m": 128}),
    ("BM=128+BN=64", {"block_m": 128, "block_n": 64}),
    ("BM=128+S1", {"block_m": 128, "num_stages": 1}),
    ("BM=128+BN=64+S1", {"block_m": 128, "block_n": 64, "num_stages": 1}),
    ("BM=128+W4", {"block_m": 128, "num_warps": 4}),
    ("BM=128+BN=64+W4", {"block_m": 128, "block_n": 64, "num_warps": 4}),
    ("BN=64", {"block_n": 64}),
    ("W4", {"num_warps": 4}),
    ("S1", {"num_stages": 1}),
    ("BM=256", {"block_m": 256}),
    ("BM=256+S1", {"block_m": 256, "num_stages": 1}),
    ("BM=256+BN=64+S1", {"block_m": 256, "block_n": 64, "num_stages": 1}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["fp8", "int8"])
    ap.add_argument("--gcn", action="store_true")
    ap.add_argument("--fp16sweep", action="store_true")
    ap.add_argument("--verify", type=str, default=None,
                    help="fp8|int8: correctness of the BM128+BN64 config vs ref/fp16")
    args = ap.parse_args()

    q, kf, vf, kc, vc, bt, cs = build()
    out_fp16 = torch.empty_like(q)

    def fp16_fn():
        paged_attn_triton_prefill(q, kf[:, Tk - Tq:].contiguous(),
                                  vf[:, Tk - Tq:].contiguous(),
                                  kc, vc, bt, cs, causal=True,
                                  softmax_scale=SM, out=out_fp16)

    quants = {}
    for mode in args.modes:
        quant = r4_quantize(q, kf, vf, SM, mode)
        kc4, vc4, bt4, cs4 = r4_to_paged(quant)
        quants[mode] = (quant, kc4, vc4, bt4, cs4)
    torch.cuda.synchronize()

    if args.gcn:
        # GCN mix at the 27B tile (BM64/BN32/W8/S2), fp16 vs r4.
        print("-- gcn mix (BM64/BN32/W8/S2) --", flush=True)
        hd_pad = 256
        grid = (Tq // 64, B * Hq, 1)
        ker = TP._paged_attn_prefill_kernel.run(
            q, kc, vc, bt, cs, out_fp16, q, q, q, q, q, q, q, q,
            1, False, 0, 0, 0, Tq, Tq, Hq, Hkv, bt.shape[1], 256, D, hd_pad,
            float(SM), True, -1, -1, False, False, 0.0, False, False, 64, 32,
            grid=grid, warmup=True, num_warps=8, num_stages=2)
        m = mix(ker.asm["amdgcn"])
        print(f"fp16: total={m['total_lines']} " + " ".join(f"{k}={v}" for k, v in m["mix"].items()), flush=True)
        quant, kc4, vc4, bt4, cs4 = quants["fp8" if "fp8" in quants else args.modes[0]]
        out = torch.empty_like(q)
        q_blocks = triton.cdiv(Tq, 64)
        ker = _r4_prefill_kernel.run(
            quant["q_q"], kc4, vc4, bt4, cs4, quant["row_log2e"], quant["v_scale"], out,
            quant["q_q"], quant["q_q"],
            1, False, Tq, 0, Hq, Hkv, bt4.shape[1], 256, D, hd_pad,
            0 if quant["q_q"].dtype != torch.int8 else 1, True, 64, 32,
            grid=(q_blocks, B * Hq, 1), warmup=True, num_warps=8, num_stages=2)
        m = mix(ker.asm["amdgcn"])
        print(f"r4: total={m['total_lines']} " + " ".join(f"{k}={v}" for k, v in m["mix"].items()), flush=True)
        return

    if args.verify:
        from exllamav3.modules.attention_fn.r4_fp8 import sdpa_r4_ref
        mode = args.verify
        quant, kc4, vc4, bt4, cs4 = quants[mode]
        n_sub = 128
        q_sub = q[:, :n_sub].contiguous()
        cs_full = torch.tensor([Tk], device="cuda:0", dtype=torch.int32)
        out_sub = torch.empty_like(q_sub)
        paged_attn_triton_prefill(q_sub, None, None, kc, vc, bt, cs_full,
                                  causal=True, softmax_scale=SM, out=out_sub)
        got = r4_prefill(quant["q_q"][:, :n_sub].contiguous(), kc4, vc4, bt4, cs4,
                         quant["row_log2e"][:, :n_sub].contiguous(), quant["v_scale"],
                         causal=True, mode=mode,
                         block_m=128, block_n=64)
        torch.cuda.synchronize()
        ref = sdpa_r4_ref(q_sub, kf, vf, causal=True, sm_scale=SM, mode=mode)

        def rel(g, b):
            d = (g.float() - b.float()).abs()
            return d.max().item(), (d.max() / b.float().abs().amax().clamp_min(1e-12)).item()
        mk, rk = rel(got, ref)
        mf, rf = rel(got, out_sub)
        print(f"verify r4-{mode} BM128+BN64: vs-ref maxabs={mk:.5f} rel={rk:.4f} "
              f"vs-fp16 maxabs={mf:.4f} rel={rf:.4f}", flush=True)
        return

    if args.fp16sweep:
        f0, _, _ = time_fn(fp16_fn)
        print(f"fp16 default med={f0:.2f}ms", flush=True)
        for name, cfg in FP16_CONFIGS:
            try:
                def ffn(_cfg=cfg):
                    paged_attn_triton_prefill(q, kf[:, Tk - Tq:].contiguous(),
                                              vf[:, Tk - Tq:].contiguous(),
                                              kc, vc, bt, cs, causal=True,
                                              softmax_scale=SM, out=out_fp16, **_cfg)
                med, mn, mx = time_fn(ffn)
                print(f"  {name:16s} med={med:7.2f} min={mn:.2f} max={mx:.2f}ms (x{f0 / med:.3f})",
                      flush=True)
            except Exception as ex:
                print(f"  {name:16s} FAIL {type(ex).__name__}: {str(ex)[:100]}", flush=True)
        return

    fmed, _, _ = time_fn(fp16_fn)
    print(f"fp16 default med={fmed:.2f}ms", flush=True)
    for mode in args.modes:
        print(f"-- {mode} --", flush=True)
        quant, kc4, vc4, bt4, cs4 = quants[mode]
        out = torch.empty_like(q)
        for name, cfg in CONFIGS:
            cfg = dict(cfg)
            try:
                def kfn(_cfg=cfg):
                    r4_prefill(quant["q_q"], kc4, vc4, bt4, cs4,
                               quant["row_log2e"], quant["v_scale"],
                               causal=True, mode=mode, out=out, **_cfg)
                med, mn, mx = time_fn(kfn)
                print(f"  {name:14s} med={med:7.2f} min={mn:.2f} max={mx:.2f}ms (x{fmed / med:.3f})",
                      flush=True)
            except Exception as ex:
                print(f"  {name:14s} FAIL {type(ex).__name__}: {str(ex)[:100]}", flush=True)
        # re-time fp16 to track drift/contention
        fmed2, _, _ = time_fn(fp16_fn)
        print(f"  fp16 re-check med={fmed2:.2f}ms", flush=True)


if __name__ == "__main__":
    main()
