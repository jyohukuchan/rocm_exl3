#!/usr/bin/env python3
"""B: rows/expert sweep - fp16 vs int8 GEMM-only throughput and crossover.

Synthetic weights/activations at the REAL expert shapes of q30b (H=2048,I=768)
and flash-next (H=2560,I=640). Compares:
  single: ext.hgemm_recon (fp16) vs ext.igemm_recon (int8->int32)
  batched: ext.hgemm_exact_group vs ext.igemm_batched (B=8 full slabs)
  dense-ideal: dense-size GEMM at T rows (MoE-FLOP-equivalent N=3*topk*I)
CUDA-event median. No model load needed.
Usage: moe_breakdown_gemm.py [OUT.json]
Env: HIP_VISIBLE_DEVICES selects GPU.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch

torch.cuda.set_device(0)
from exllamav3.ext import exllamav3_ext as ext

ITERS = 20
WARM = 5


def med(fn, iters=ITERS):
    for _ in range(WARM):
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
    return statistics.median(ts)


MODELS = {
    "q30b": {"H": 2048, "I": 768, "E": 128, "topk": 8},
    "flash-next": {"H": 2560, "I": 640, "E": 512, "topk": 10},
}
ROWS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
DENSE_T = [512, 2048, 8192]

res = {"models": {}, "dense_ideal": {}}
g = torch.Generator(device="cuda:0").manual_seed(7)

with torch.inference_mode():
    for mname, m in MODELS.items():
        H, I = m["H"], m["I"]
        out = {"gu": [], "d": []}
        # gate/up: (R,H)x(H,I); down: (R,I)x(I,H)
        for tag, K, N in (("gu", H, I), ("d", I, H)):
            for R in ROWS:
                a8 = torch.randint(-100, 100, (R, K), generator=g,
                                   device="cuda:0", dtype=torch.int32).to(torch.int8)
                w8 = torch.randint(-100, 100, (K, N), generator=g,
                                   device="cuda:0", dtype=torch.int32).to(torch.int8)
                c32 = torch.empty((R, N), dtype=torch.int32, device="cuda:0")
                xh = torch.randn((R, K), generator=g, device="cuda:0", dtype=torch.float16)
                w = torch.randn((K, N), generator=g, device="cuda:0", dtype=torch.float16)
                y = torch.empty((R, N), dtype=torch.float16, device="cuda:0")
                t_i = med(lambda: ext.igemm_recon(a8, w8, c32))
                t_h = med(lambda: ext.hgemm_recon(xh, w, y))
                # achieved throughput on the math (2*R*K*N), int8 DP4A vs fp16
                flops = 2 * R * K * N
                out[tag].append({
                    "rows": R, "fp16_ms": round(t_h, 4), "int8_ms": round(t_i, 4),
                    "speedup": round(t_h / t_i, 3) if t_i > 0 else None,
                    "fp16_tflops": round(flops / t_h / 1e9, 3) if t_h > 0 else None,
                    "int8_tflops": round(flops / t_i / 1e9, 3) if t_i > 0 else None,
                })
                print(f"{mname} {tag} R={R:5d} fp16={t_h:.4f}ms int8={t_i:.4f}ms "
                      f"su={t_h / t_i:.2f}x", flush=True)
        # batched: B=8 experts, full slabs (no padding) at cmax=R
        out["batched"] = []
        for tag, K, N in (("gu", H, I), ("d", I, H)):
            for R in [16, 64, 160, 512]:
                B = 8
                A8 = torch.randint(-100, 100, (B, R, K), generator=g,
                                   device="cuda:0", dtype=torch.int32).to(torch.int8)
                W8 = torch.randint(-100, 100, (B, K, N), generator=g,
                                   device="cuda:0", dtype=torch.int32).to(torch.int8)
                Cb = torch.empty((B, R, N), dtype=torch.int32, device="cuda:0")
                Xh = torch.randn((B, R, K), generator=g, device="cuda:0", dtype=torch.float16)
                W = torch.randn((B, K, N), generator=g, device="cuda:0", dtype=torch.float16)
                Y = torch.empty((B, R, N), dtype=torch.float16, device="cuda:0")
                counts = [R] * B
                t_i = med(lambda: ext.igemm_batched(A8, W8, Cb), iters=40)
                t_h = med(lambda: ext.hgemm_exact_group(Xh, W, Y, counts), iters=40)
                # fp16 per-expert loop (what the compat fallback does)
                def loop_h():
                    Y.zero_()
                    for b in range(B):
                        ext.hgemm_recon(Xh[b, :R], W[b], Y[b, :R])
                t_hl = med(loop_h, iters=40)
                # int8 per-expert loop (prototype path)
                def loop_i():
                    for b in range(B):
                        ext.igemm_recon(A8[b], W8[b], Cb[b])
                t_il = med(loop_i, iters=40)
                print(f"    loop-check {tag} cmax={R}: fp16loop={t_hl:.4f}ms int8loop={t_il:.4f}ms",
                      flush=True)
                print(f"{mname} batch {tag} B=8 cmax={R:4d} fp16={t_h:.4f}ms "
                      f"int8={t_i:.4f}ms su={t_h / t_i:.2f}x", flush=True)
                out["batched"].append({"tag": tag, "B": B, "cmax": R,
                                       "fp16_ms": round(t_h, 4), "int8_ms": round(t_i, 4),
                                       "speedup": round(t_h / t_i, 3),
                                       "fp16loop_ms": round(t_hl, 4),
                                       "int8loop_ms": round(t_il, 4)})
        res["models"][mname] = out
    # dense-ideal: MoE-FLOP-equivalent dense GEMM, N = 3*topk*I
    for mname, m in MODELS.items():
        H, I, topk = m["H"], m["I"], m["topk"]
        N = 3 * topk * I
        rows = []
        for T in DENSE_T:
            a8 = torch.randint(-100, 100, (T, H), generator=g,
                               device="cuda:0", dtype=torch.int32).to(torch.int8)
            w8 = torch.randint(-100, 100, (H, N), generator=g,
                               device="cuda:0", dtype=torch.int32).to(torch.int8)
            c32 = torch.empty((T, N), dtype=torch.int32, device="cuda:0")
            xh = torch.randn((T, H), generator=g, device="cuda:0", dtype=torch.float16)
            w = torch.randn((H, N), generator=g, device="cuda:0", dtype=torch.float16)
            y = torch.empty((T, N), dtype=torch.float16, device="cuda:0")
            t_i = med(lambda: ext.igemm_recon(a8, w8, c32))
            t_h = med(lambda: ext.hgemm_recon(xh, w, y))
            print(f"dense-ideal {mname} T={T:5d} K={H} N={N} fp16={t_h:.3f}ms "
                  f"int8={t_i:.3f}ms su={t_h / t_i:.2f}x", flush=True)
            rows.append({"T": T, "K": H, "N": N, "fp16_ms": round(t_h, 4),
                         "int8_ms": round(t_i, 4), "speedup": round(t_h / t_i, 3)})
        res["dense_ideal"][mname] = rows

out_path = sys.argv[1] if len(sys.argv) > 1 else None
# rows/expert mapping for real chunk sizes
for mname, m in MODELS.items():
    mp = {}
    for T in [512, 2048, 8192, 32768]:
        mp[str(T)] = round(T * m["topk"] / m["E"], 1)
    res["models"][mname]["rows_per_expert_map"] = mp
print(json.dumps(res, indent=1))
if out_path:
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out_path, flush=True)
