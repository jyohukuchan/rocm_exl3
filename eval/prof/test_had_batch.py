#!/usr/bin/env python3
"""had_r_128_batch vs per-expert had_r_128 loop at grouped-MoE scales."""
import os
import sys

sys.path.insert(0, "/src/w8a8_lib")
sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
import torch
from exllamav3.ext import exllamav3_ext as ext

torch.cuda.set_device(0)
g = torch.Generator(device="cuda:0").manual_seed(5)
with torch.inference_mode():
    for B, cmax, N, E in [(8, 32, 768, 8), (128, 17, 768, 128),
                          (128, 92, 768, 128), (128, 160, 768, 128),
                          (128, 92, 2048, 128)]:
        dev = "cuda:0"
        x = torch.randn((B * cmax, N), generator=g, device=dev).half()
        table = torch.randn((E, N), generator=g, device=dev).half()
        ids = torch.randperm(E, generator=g, device=dev)[:B].to(torch.int64)
        y1 = x.clone()
        ext.had_r_128_batch(y1, y1, None, table, ids, cmax, 1.0)
        y2 = x.clone()
        for b in range(B):
            r = y2[b * cmax:(b + 1) * cmax]
            ext.had_r_128(r, r, None, table[ids[b].item()], 1.0)
        d = (y1.float() - y2.float()).abs()
        print(f"B={B} cmax={cmax} N={N}: maxabs={d.max().item():.3e}", flush=True)
print("done", flush=True)
