#!/usr/bin/env python3
"""Fused epilogue kernels vs torch+had sequence (correctness + speed)."""
import os
import statistics
import sys

sys.path.insert(0, "/src/w8a8_lib")
sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.rocm_py.moe_int8 import gu_epilog_had, dquant_rowwise, d_epilog_had

torch.cuda.set_device(0)
g = torch.Generator(device="cuda:0").manual_seed(2)
dev = "cuda:0"
with torch.inference_mode():
    for B, M, N in [(130, 150, 768), (128, 64, 768), (130, 150, 2048)]:
        R = B * M
        E = 128
        cg = torch.randint(-2**20, 2**20, (R, N), generator=g, device=dev, dtype=torch.int32)
        cu = torch.randint(-2**20, 2**20, (R, N), generator=g, device=dev, dtype=torch.int32)
        sa = torch.rand((R, 1), generator=g, device=dev) * 0.05 + 0.001
        s0 = torch.rand((B,), generator=g, device=dev) * 0.03 + 0.005
        svg = torch.randint(0, 2, (E, N), generator=g, device=dev, dtype=torch.int32).float() * 2 - 1
        svg = svg.half()
        ids = torch.arange(B, device=dev) % E
        ids = ids.to(torch.int64)
        # reference: torch epilogue + native had
        gr = ((cg.float() * sa) * s0.repeat_interleave(M).unsqueeze(1)).half()
        ur = ((cu.float() * sa) * s0.repeat_interleave(M).unsqueeze(1)).half()
        ext.had_r_128_batch(gr, gr, None, svg, ids, M, 1.0)
        ext.had_r_128_batch(ur, ur, None, svg, ids, M, 1.0)
        # fused
        gf = torch.empty_like(gr)
        uf = torch.empty_like(ur)
        s0f = s0.float()
        gu_epilog_had(cg, cu, sa, s0f, s0f, svg, svg, ids, M, gf, uf)
        torch.cuda.synchronize()
        dg = (gf.float() - gr.float()).abs()
        du = (uf.float() - ur.float()).abs()
        print(f"gu B={B} M={M} N={N}: g maxabs={dg.max().item():.3e} u maxabs={du.max().item():.3e}", flush=True)
        # dquant
        a = (torch.randn((R, N), generator=g, device=dev) * 3).half()
        sadr = a.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        xdr = (a.float() / sadr).round().clamp(-128, 127).to(torch.int8)
        sadf = torch.empty((R, 1), dtype=torch.float, device=dev)
        xdf = torch.empty((R, N), dtype=torch.int8, device=dev)
        dquant_rowwise(a, sadf, xdf)
        torch.cuda.synchronize()
        print(f"   dquant: sad {((sadf - sadr).abs().max().item()):.3e} xd {((xdf.float() - xdr.float()).abs().max().item())}", flush=True)
        # d_epilog
        cd = torch.randint(-2**20, 2**20, (R, N), generator=g, device=dev, dtype=torch.int32)
        dr = ((cd.float() * sadr) * s0.repeat_interleave(M).unsqueeze(1)).half()
        ext.had_r_128_batch(dr, dr, None, svg, ids, M, 1.0)
        df = torch.empty_like(dr)
        d_epilog_had(cd, sadr, s0f, svg, ids, M, df)
        torch.cuda.synchronize()
        dd = (df.float() - dr.float()).abs()
        print(f"   d_epilog: maxabs={dd.max().item():.3e}", flush=True)
    # speed on the big shape
    B, M, N = 130, 150, 768
    R = B * M
    cg = torch.randint(-2**20, 2**20, (R, N), device=dev, dtype=torch.int32)
    cu = torch.randint(-2**20, 2**20, (R, N), device=dev, dtype=torch.int32)
    sa = torch.rand((R, 1), device=dev) * 0.05 + 0.001
    s0 = torch.rand((B,), device=dev).float() * 0.03 + 0.005
    svg = (torch.randint(0, 2, (E, N), device=dev, dtype=torch.int32).float() * 2 - 1).half()
    ids = (torch.arange(B, device=dev) % E).to(torch.int64)
    gf = torch.empty((R, N), dtype=torch.half, device=dev)
    uf = torch.empty_like(gf)
    for _ in range(3):
        gu_epilog_had(cg, cu, sa, s0, s0, svg, svg, ids, M, gf, uf)
    torch.cuda.synchronize()

    def med(fn, iters=10):
        ts = []
        for _ in range(iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return round(statistics.median(ts), 3)

    print("gu_epilog_had:", med(lambda: gu_epilog_had(cg, cu, sa, s0, s0, svg, svg, ids, M, gf, uf)), "ms", flush=True)
print("done", flush=True)
