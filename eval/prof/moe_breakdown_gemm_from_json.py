#!/usr/bin/env python3
"""GEMM-level real-vs-uniform timing from a routing JSON (no model load).
Usage: moe_breakdown_gemm_from_json.py ROUTING.json [OUT.json]
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

from exllamav3.ext import exllamav3_ext as ext

torch.cuda.set_device(0)
R = json.load(open(sys.argv[1]))
OUT = sys.argv[2] if len(sys.argv) > 2 else None
T = R["T"]
mid = R["mid_stats"]
E = len(mid["counts"])
cap = 160
TOPK = mid["assign"] // mid["T"]

# shapes: infer from uniform padded? stored in gemm? recompute via model key
# Caller passes shapes via env MOE_BD_SHAPES=Kg,Ng,Kd,Nd
Kg, Ng, Kd, Nd = [int(x) for x in os.environ["MOE_BD_SHAPES"].split(",")]


def entries(counts):
    ec = []
    for x in counts:
        if x == 0:
            continue
        s = max(1, math.ceil(x / cap))
        base, rem = divmod(x, s)
        ec += [base + (1 if i < rem else 0) for i in range(s)]
    return ec


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
g = torch.Generator(device="cuda:0").manual_seed(9)
U = T * TOPK // E
ns = max(1, (U + cap - 1) // cap)
cmax_u = (U + ns - 1) // ns
B_u = E * ns if U else 0
ec_real = entries(mid["counts"])
B_r = len(ec_real)
cmax_r = max(ec_real)
out = {}
with torch.inference_mode():
    for cfgname, B, cmax, ec in (("uniform", B_u, cmax_u, [cmax_u] * B_u),
                                 ("real", B_r, cmax_r, ec_real)):
        r = {"B": B, "cmax": cmax, "padded": B * cmax, "assign": T * TOPK}
        for tag, K, N in (("gu", Kg, Ng), ("d", Kd, Nd)):
            A8 = torch.randint(-100, 100, (B, cmax, K), generator=g, device=dev,
                               dtype=torch.int32).to(torch.int8)
            W8 = torch.randint(-100, 100, (B, K, N), generator=g, device=dev,
                               dtype=torch.int32).to(torch.int8)
            Cb = torch.empty((B, cmax, N), dtype=torch.int32, device=dev)
            r[f"igemm_{tag}_ms"] = med(lambda: ext.igemm_batched(A8, W8, Cb))
            del A8, W8, Cb
            torch.cuda.empty_cache()
            Xh = torch.randn((B, cmax, K), generator=g, device=dev,
                             dtype=torch.float16)
            Wh = torch.randn((B, K, N), generator=g, device=dev,
                             dtype=torch.float16)
            Yh = torch.empty((B, cmax, N), dtype=torch.float16, device=dev)
            r[f"hgemm_batched_{tag}_ms"] = med(
                lambda: ext.hgemm_batched(Xh, Wh, Yh))
            r[f"hgemm_exact_{tag}_ms"] = med(
                lambda: ext.hgemm_exact_group(Xh, Wh, Yh, ec))
            del Xh, Wh, Yh
            torch.cuda.empty_cache()
        print(f"{cfgname}: " + json.dumps(r), flush=True)
        out[cfgname] = r
print(json.dumps(out, indent=1))
if OUT:
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)
