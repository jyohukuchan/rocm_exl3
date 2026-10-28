#!/usr/bin/env python3
"""igemm_batched correctness: strided-batched int8 GEMM vs per-batch igemm_recon."""
import os
import sys

sys.path.insert(0, "/src/w8a8_lib")
sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
import torch

import exllamav3_ext as ext

assert hasattr(ext, "igemm_batched"), "igemm_batched missing from ext"
torch.cuda.set_device(0)
g = torch.Generator(device="cuda:0").manual_seed(7)

for B, M, K, N in [(3, 5, 64, 48), (128, 160, 2048, 768), (4, 1, 2048, 768)]:
    with torch.inference_mode():
        A = torch.randint(-128, 127, (B, M, K), generator=g, device="cuda:0",
                          dtype=torch.int32).to(torch.int8)
        W = torch.randint(-128, 127, (B, K, N), generator=g, device="cuda:0",
                          dtype=torch.int32).to(torch.int8)
        Cb = torch.empty((B, M, N), dtype=torch.int32, device="cuda:0")
        ext.igemm_batched(A, W, Cb)
        ok = True
        for b in range(B):
            Cr = torch.empty((M, N), dtype=torch.int32, device="cuda:0")
            ext.igemm_recon(A[b], W[b], Cr)
            if not torch.equal(Cb[b], Cr):
                d = (Cb[b].float() - Cr.float()).abs().max().item()
                print(f"MISMATCH B={B} M={M} K={K} N={N} batch {b} maxdiff={d}")
                ok = False
        print(f"B={B} M={M} K={K} N={N}: {'OK' if ok else 'FAIL'}", flush=True)
print("done", flush=True)
