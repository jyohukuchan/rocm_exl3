#!/usr/bin/env python3
"""AMDGCN codegen check: fp16 prefill vs round-1 on-the-fly vs cached kernel.

Builds the 27B geometry (B=1, Tq=1024, 24/4/256, Tk=4096, causal), compiles
each kernel via .run(warmup=False) and reports GCN line counts + instruction
mix (WMMA kinds, LDS, global). Run on V620 (pv 0/1) and R9700 (pv 0/1/2).

Usage: codegen_dump_i8f8.py [--out JSON]
"""
import json
import sys

sys.path.insert(0, "/src")

import torch

torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
print(f"device: {props.name} {props.gcnArchName}", flush=True)

import triton

from exllamav3.modules.attention_fn import triton_paged as TP
from exllamav3.modules.attention_fn import int8fp8 as R1
from exllamav3.modules.attention_fn import i8f8_cached as C2
from exllamav3.cache.i8f8 import CacheLayer_i8f8
from types import SimpleNamespace

B, Tq, Hq, Hkv, D, PAGE = 1, 1024, 24, 4, 256, 256
Tk = 4096
SM = 1.0 / (D ** 0.5)
torch.manual_seed(0)

OPS = ["v_wmma", "v_mfma", "mfma", "wmma", "ds_read", "ds_write", "buffer_load",
       "buffer_store", "flat_load", "flat_store", "global_load", "global_store",
       "v_dot", "v_fma", "v_cvt", "s_barrier", "s_waitcnt"]


def mix(gcn):
    lines = gcn.splitlines()
    code = [l for l in lines if l.strip() and not l.strip().startswith((".", "/", ";"))]
    m = {op: sum(1 for l in code if op in l) for op in OPS}
    return {"total_lines": len(lines), "code_lines": len(code),
            "mix": {k: v for k, v in m.items() if v}}


def build_fp16():
    pages = Tk // PAGE
    kc = torch.randn((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    vc = torch.randn((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    bt = torch.arange(pages, device="cuda:0", dtype=torch.int32).view(1, pages)
    cs = torch.tensor([Tk - Tq], device="cuda:0", dtype=torch.int32)
    q = torch.randn((B, Tq, Hq, D), device="cuda:0", dtype=torch.float16)
    k = torch.randn((B, Tq, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    v = torch.randn((B, Tq, Hkv, D), device="cuda:0", dtype=torch.float16) * 0.5
    return q, k, v, kc, vc, bt, cs


def dump_fp16():
    q, k, v, kc, vc, bt, cs = build_fp16()
    out = torch.empty_like(q)
    hd_pad = 256
    grid = (Tq // 64, B * Hq, 1)
    ker = TP._paged_attn_prefill_kernel.run(
        q, kc, vc, bt, cs, out,
        q, q, q, q, q,   # partial_o, partial_ml, k/v_scales, h32 placeholders
        q, q, q,          # k_new, v_new, sinks placeholders
        1, False, 0, 0, 0,   # num_splits, IS_SPLIT, NEW_KV, QCK, QCV
        Tq, Tq, Hq, Hkv,
        bt.shape[1], 256, D, hd_pad, float(SM),
        True, -1, -1, False, False, 0.0, False, False, 64, 32,
        grid=grid, warmup=False, num_warps=8, num_stages=2,
    )
    return ker.asm["amdgcn"]


def dump_r1(pv):
    q, k, v, kc, vc, bt, cs = build_fp16()
    out = torch.empty_like(q)
    gs = torch.zeros((1, Hkv), dtype=torch.float32, device="cuda:0")
    grid = (Tq // 64, B * Hq)
    ker = R1._i8f8_prefill_kernel.run(
        q, kc, vc, bt, cs, out, gs,
        Tq, Tq, Hq, Hkv, bt.shape[1],
        256, D, 256, float(SM),
        True, pv, True, False, 64, 32,
        grid=grid, warmup=False, num_warps=8, num_stages=2,
    )
    return ker.asm["amdgcn"]


def dump_cached(pv):
    q, k, v, kc, vc, bt, cs = build_fp16()
    layer = CacheLayer_i8f8(None, SimpleNamespace(num_kv_heads=Hkv, head_dim=D), 0, Tk)
    layer.alloc(torch.device("cuda:0"))
    layer.update_kv_direct(torch.tensor([0], device="cuda:0", dtype=torch.int32),
                           bt, kc.view(1, Tk, Hkv, D)[:, :Tk - Tq].contiguous(),
                           vc.view(1, Tk, Hkv, D)[:, :Tk - Tq].contiguous(), Tk - Tq)
    packed = layer.get_qkv()
    vq_fp8 = packed[3].view(torch.float8_e4m3fn)
    out = torch.empty_like(q)
    grid = (Tq // 64, B * Hq)
    ker = C2._i8f8c_prefill_kernel.run(
        q, packed[0], packed[1], packed[2], vq_fp8, packed[4],
        packed[5], packed[6], packed[7],
        bt, torch.tensor([Tk - Tq], device="cuda:0", dtype=torch.int32), out,
        Tq, Tq, Hq, Hkv, bt.shape[1],
        256, D, 256, 32, 8, float(SM),
        True, pv, 64, 32,
        grid=grid, warmup=False, num_warps=8, num_stages=2,
    )
    return ker.asm["amdgcn"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = {"device": f"{props.name} {props.gcnArchName}"}
    jobs = [("fp16", lambda: dump_fp16()),
            ("round1-pv0", lambda: dump_r1(0)),
            ("round1-pv1", lambda: dump_r1(1)),
            ("round1-pv2", lambda: dump_r1(2)),
            ("cached-pv0", lambda: dump_cached(0)),
            ("cached-pv1", lambda: dump_cached(1)),
            ("cached-pv2", lambda: dump_cached(2))]
    for name, fn in jobs:
        try:
            rep[name] = mix(fn())
            m = rep[name]
            ops = " ".join(f"{k}={v}" for k, v in m["mix"].items())
            print(f"{name:12s} total={m['total_lines']} code={m['code_lines']} {ops}", flush=True)
        except Exception as e:
            rep[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"{name:12s} COMPILE-FAIL {type(e).__name__}", flush=True)
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
