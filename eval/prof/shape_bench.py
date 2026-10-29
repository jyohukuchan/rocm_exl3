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
                       ("q", 5120, 6144), ("o", 6144, 5120),
                       ("gdn_qkv", 5120, 4608), ("gdn_out", 3072, 5120),
                       ("head_slice", 5120, 32768)]:
        CASES.append((M, name, K, N))

print(f"{'M':>5} {'name':<10} {'K':>6} {'N':>6} {'igemm':>8} {'hgemm':>8} "
      f"{'hadKN':>8} {'quantW':>8} {'deq':>8} (ms)", flush=True)
with torch.inference_mode():
    for M, name, K, N in CASES:
        a8 = torch.randint(-100, 100, (M, K), generator=g, device="cuda:0",
                           dtype=torch.int32).to(torch.int8)
        w8 = torch.randint(-100, 100, (K, N), generator=g, device="cuda:0",
                           dtype=torch.int32).to(torch.int8)
        c = torch.empty((M, N), dtype=torch.int32, device="cuda:0")
        xh = torch.randn((M, K), generator=g, device="cuda:0", dtype=torch.float16)
        w = torch.randn((K, N), generator=g, device="cuda:0", dtype=torch.float16)
        y = torch.empty((M, N), dtype=torch.float16, device="cuda:0")
        t_ig = bench(lambda: ext.igemm_recon(a8, w8, c))
        t_hg = bench(lambda: ext.hgemm_recon(xh, w, y))
        x = torch.empty_like(xh)
        t_had = bench(lambda: ext.had_r_128(xh, x, None, None, 1.0))
        t_qw = bench(lambda: (w.float() / 0.02).round().clamp(-128, 127).to(torch.int8))
        sa = torch.rand((M, 1), generator=g, device="cuda:0") + 0.01
        t_deq = bench(lambda: (c.float() * sa).to(y.dtype))
        print(f"{M:>5} {name:<10} {K:>6} {N:>6} {t_ig:8.3f} {t_hg:8.3f} "
              f"{t_had:8.3f} {t_qw:8.3f} {t_deq:8.3f}", flush=True)
print("done", flush=True)
