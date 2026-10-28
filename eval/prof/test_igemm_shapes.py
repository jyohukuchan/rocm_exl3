#!/usr/bin/env python3
"""igemm_batched vs per-batch recon across M (padding-relevant shapes)."""
import os
import sys

sys.path.insert(0, "/src/w8a8_lib")
sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
import torch
from exllamav3.ext import exllamav3_ext as ext

torch.cuda.set_device(0)
g = torch.Generator(device="cuda:0").manual_seed(9)
with torch.inference_mode():
    for B, M, K, N in [(128, 64, 2048, 768), (128, 64, 768, 2048),
                       (128, 17, 2048, 768), (128, 92, 2048, 768),
                       (128, 33, 2048, 768), (32, 64, 2048, 768),
                       (127, 64, 2048, 768)]:
        dev = "cuda:0"
        A = torch.randint(-128, 127, (B, M, K), generator=g, device=dev,
                          dtype=torch.int32).to(torch.int8)
        W = torch.randint(-128, 127, (B, K, N), generator=g, device=dev,
                          dtype=torch.int32).to(torch.int8)
        # zero out trailing rows of odd batches (padding pattern)
        for b in range(B):
            c = (b * 37) % M + 1
            if c < M:
                A[b, c:] = 0
        Cb = torch.empty((B, M, N), dtype=torch.int32, device=dev)
        ext.igemm_batched(A, W, Cb)
        torch.cuda.synchronize()
        bad = 0
        for b in range(B):
            Cr = torch.empty((M, N), dtype=torch.int32, device=dev)
            ext.igemm_recon(A[b], W[b], Cr)
            if not torch.equal(Cb[b], Cr):
                bad += 1
                if bad <= 2:
                    d = (Cb[b].float() - Cr.float()).abs()
                    print(f"  M={M} K={K} N={N} batch {b}: maxdiff={d.max().item()} nbad={(d>0).sum().item()}")
        print(f"B={B} M={M} K={K} N={N}: {'OK' if bad==0 else f'{bad} BAD BATCHES'}", flush=True)
print("done", flush=True)
