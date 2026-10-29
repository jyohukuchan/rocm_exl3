#!/usr/bin/env python3
"""Synthetic Flash-Next-shape stage decomposition (no model load).

Times every grouped-MoE INT8 stage except trellis reconstruct at the
Flash-Next expert shapes (gu K=2560 N=640, d K=640 N=2048, 512 experts,
top10) for uniform routing at each T. Weight reconstruct is projected from
the measured q30b per-slab rate (same trellis format, stated in doc).
Compares against fp16 hgemm_batched (production-like, padded).
Usage: moe_breakdown_synth_fn.py [OUT.json]
Env: HIP_VISIBLE_DEVICES.
"""
import json
import math
import os
import statistics
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch

torch.cuda.set_device(0)
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.rocm_py.moe_int8 import (
    quant_w8_slab, dquant_rowwise, gu_epilog_had, d_epilog_had)

OUT = sys.argv[1] if len(sys.argv) > 1 else None
H, I, E, TOPK = 2560, 640, 512, 10
CAP = 160
TLIST = [512, 2048, 8192]


def med(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return round(statistics.median(ts), 4)


dev = torch.device("cuda:0")
g = torch.Generator(device="cuda:0").manual_seed(5)
res = {}
with torch.inference_mode():
    for T in TLIST:
        r = {}
        U = T * TOPK // E
        ns = max(1, (U + CAP - 1) // CAP)
        cmax = (U + ns - 1) // ns
        B = E * ns
        r["uniform"] = {"rows_per_expert": U, "B": B, "cmax": cmax,
                        "assign": T * TOPK, "padded": B * cmax}
        # activation input had + torch quant (per token)
        y = torch.randn((T, H), generator=g, device=dev, dtype=torch.half)
        xh = torch.empty_like(y)
        su = torch.randn((H,), generator=g, device=dev, dtype=torch.half)
        r["act_had_ms"] = med(lambda: ext.had_r_128(y, xh, su, None, 1.0))
        sa = xh.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        r["act_quant_torch_ms"] = med(
            lambda: (xh.float() / sa).round().clamp(-128, 127).to(torch.int8))
        # sort/bincount/gather torch ops
        sel = torch.randint(0, E, (T, TOPK), generator=g, device=dev)
        flat = sel.reshape(-1)
        r["sort_ms"] = med(lambda: flat.argsort())
        r["bincount_ms"] = med(lambda: torch.bincount(flat, minlength=E + 1))
        tok = torch.randint(0, T, (B * cmax,), device=dev)
        a8e = torch.randint(-100, 100, (T * TOPK, H), generator=g, device=dev,
                            dtype=torch.int32).to(torch.int8)
        r["gather_ms"] = med(lambda: a8e[tok].view(B, cmax, -1))
        for tag, K, N in (("gu", H, I), ("d", I, H)):
            Wfp = torch.randn((B, K, N), generator=g, device=dev,
                              dtype=torch.float16)
            s0 = torch.full((B,), 0.02, dtype=torch.float32, device=dev)
            W8 = torch.empty((B, K, N), dtype=torch.int8, device=dev)
            r[f"wquant_{tag}_ms"] = med(lambda: quant_w8_slab(Wfp, s0, W8))
            Xin = torch.randint(-100, 100, (B, cmax, K), generator=g, device=dev,
                                dtype=torch.int32).to(torch.int8)
            C32 = torch.empty((B, cmax, N), dtype=torch.int32, device=dev)
            r[f"igemm_{tag}_ms"] = med(lambda: ext.igemm_batched(Xin, W8, C32))
            Xh = torch.randn((B, cmax, K), generator=g, device=dev,
                             dtype=torch.float16)
            Wh = torch.randn((B, K, N), generator=g, device=dev,
                             dtype=torch.float16)
            Yh = torch.empty((B, cmax, N), dtype=torch.float16, device=dev)
            r[f"hgemm_batched_{tag}_ms"] = med(
                lambda: ext.hgemm_batched(Xh, Wh, Yh))
            del Xh, Wh, Yh, Xin, C32, Wfp, W8
            torch.cuda.empty_cache()
        # fused epilogues (synthetic tables)
        Ng, Nd = I, H
        e_ids = torch.arange(E, device=dev).repeat_interleave(ns).to(torch.int64)
        Cg = torch.empty((B * cmax, Ng), dtype=torch.int32, device=dev)
        Cu = torch.empty_like(Cg)
        SA = torch.empty((B * cmax, 1), dtype=torch.float, device=dev).fill_(0.02)
        gg = torch.empty((B * cmax, Ng), dtype=torch.half, device=dev)
        uu = torch.empty_like(gg)
        s0e = torch.full((B,), 0.02, dtype=torch.float32, device=dev)
        svg = torch.randn((E, Ng), dtype=torch.half, device=dev)
        r["epigu_ms"] = med(lambda: gu_epilog_had(Cg, Cu, SA, s0e, s0e, svg, svg,
                                                  e_ids, cmax, gg, uu))
        aout = torch.empty_like(gg)
        import torch.nn.functional as F
        r["silu_ms"] = med(lambda: aout.copy_(F.silu(gg)))
        ss = torch.empty((B * cmax, 1), dtype=torch.float, device=dev)
        XX = torch.empty((B * cmax, I), dtype=torch.int8, device=dev)
        r["dquant_ms"] = med(lambda: dquant_rowwise(gg, ss, XX))
        Cd = torch.empty((B * cmax, Nd), dtype=torch.int32, device=dev)
        svd = torch.randn((E, Nd), dtype=torch.half, device=dev)
        dd = torch.empty((B * cmax, Nd), dtype=torch.half, device=dev)
        r["epid_ms"] = med(lambda: d_epilog_had(Cd, ss, s0e, svd, e_ids,
                                                cmax, dd))
        out = torch.zeros((T, H), dtype=torch.float, device=dev)
        df = torch.randn((B * cmax, Nd), device=dev, dtype=torch.float)
        def _acc():
            for b in range(B):
                r0 = b * cmax
                out.index_add_(0, tok[r0:r0 + cmax], df[r0:r0 + cmax])
        r["indexadd_ms"] = med(_acc)
        print(f"T={T}: " + json.dumps(r), flush=True)
        res[str(T)] = r
print(json.dumps(res, indent=1))
if OUT:
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT, flush=True)
