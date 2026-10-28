#!/usr/bin/env python3
"""Fused w8 quant kernel: bitwise vs torch scalar path + speed."""
import os
import sys
import statistics

sys.path.insert(0, "/src/w8a8_lib")
sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
import torch
from exllamav3.rocm_py.moe_int8 import quant_w8_slab

torch.cuda.set_device(0)
g = torch.Generator(device="cuda:0").manual_seed(11)
dev = "cuda:0"
with torch.inference_mode():
    for B, K, N in [(128, 2048, 768), (8, 2048, 768), (128, 768, 2048)]:
        wfp = (torch.randn((B, K, N), generator=g, device=dev) * 2).half()
        s0 = (torch.rand((B,), generator=g, device=dev) * 0.03 + 0.005)
        w8t = torch.empty((B, K, N), dtype=torch.int8, device=dev)
        quant_w8_slab(wfp, s0, w8t)
        torch.cuda.synchronize()
        # torch scalar reference per expert
        w8r = torch.empty_like(w8t)
        for b in range(B):
            s = float(s0[b])
            w8r[b].copy_((wfp[b].float() / s).round().clamp(-128, 127).to(torch.int8))
        d = (w8t.float() - w8r.float()).abs()
        print(f"B={B} K={K} N={N}: maxabs={d.max().item()} nbad={(d>0).sum().item()}", flush=True)
        # mechanism probe on first shape only
        if B == 128 and K == 2048:
            r = (1.0 / s0)
            r32 = r.float()
            w8m = torch.empty_like(w8r)
            for b in range(B):
                w8m[b].copy_((wfp[b].float() * float(r32[b])).round().clamp(-128, 127).to(torch.int8))
            dm = (w8r.float() - w8m.float()).abs()
            print(f"   scalar-div vs scalar-recip-mul: maxabs={dm.max().item()} nbad={(dm>0).sum().item()}", flush=True)
            w8d = torch.empty_like(w8r)
            st = s0.float()
            for b in range(B):
                w8d[b].copy_((wfp[b].float() / st[b]).round().clamp(-128, 127).to(torch.int8))
            dd = (w8r.float() - w8d.float()).abs()
            print(f"   scalar-div vs tensor-div(same vals): maxabs={dd.max().item()} nbad={(dd>0).sum().item()}", flush=True)
        # speed
        for _ in range(3):
            quant_w8_slab(wfp, s0, w8t)
        torch.cuda.synchronize()
        ts = []
        for _ in range(10):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            quant_w8_slab(wfp, s0, w8t)
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        print(f"   median {statistics.median(ts):.2f}ms", flush=True)
print("done", flush=True)
