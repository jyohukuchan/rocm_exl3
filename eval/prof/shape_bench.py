#!/usr/bin/env python3
"""GEMM-level shootout on dense-27B prefill shapes: igemm_recon (int8->int32)
vs hgemm_recon (fp16), plus the int8-path overhead pieces (had, quant,
dequant) modeled with torch ops. Decides C2 by measurement.
Usage: shape_bench.py
Env: HIP_VISIBLE_DEVICES.
"""
import os
import sys
import time

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
import torch
from exllamav3.ext import exllamav3_ext as ext

torch.cuda.set_device(0)

SHAPES = [
    ("down", 5120, 17408),   # placeholder, replaced below per-M
]


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


g = torch.Generator(device="cuda:0").manual_seed(7)
CASES = []
for M in (2048, 8192):
    for name, K, N in [("gate/up", 5120, 17408), ("down", 17408, 5120),
                       ("q", 5120, 12288), ("o", 6144, 5120),
                       ("gdn_qkv", 5120, 10240), ("gdn_z", 5120, 6144),
                       ("gdn_out", 6144, 5120), ("k/v", 5120, 1024),
                       ("head_slice", 5120, 32768)]:
        CASES.append((M, name, K, N))

print(f"{'M':>5} {'name':<10} {'K':>6} {'N':>6} {'igemm':>8} {'hg16':>8} "
      f"{'hg32':>8} (ms)", flush=True)
with torch.inference_mode():
    for M, name, K, N in CASES:
        a8 = torch.randint(-100, 100, (M, K), generator=g, device="cuda:0",
                           dtype=torch.int32).to(torch.int8)
        w8 = torch.randint(-100, 100, (K, N), generator=g, device="cuda:0",
                           dtype=torch.int32).to(torch.int8)
        c = torch.empty((M, N), dtype=torch.int32, device="cuda:0")
        xh = torch.randn((M, K), generator=g, device="cuda:0", dtype=torch.float16)
        w = torch.randn((K, N), generator=g, device="cuda:0", dtype=torch.float16)
        y16 = torch.empty((M, N), dtype=torch.float16, device="cuda:0")
        y32 = torch.empty((M, N), dtype=torch.float32, device="cuda:0")
        t_ig = bench(lambda: ext.igemm_recon(a8, w8, c))
        t_h16 = bench(lambda: ext.hgemm_recon(xh, w, y16))
        t_h32 = bench(lambda: ext.hgemm_recon(xh, w, y32))
        print(f"{M:>5} {name:<10} {K:>6} {N:>6} {t_ig:8.3f} {t_h16:8.3f} "
              f"{t_h32:8.3f}", flush=True)
print("done", flush=True)
