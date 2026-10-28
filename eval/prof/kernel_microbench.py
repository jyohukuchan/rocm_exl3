#!/usr/bin/env python3
"""Kernel-level microbench for the INT8-prefill cost model (CUDA events).

Times individual ops on REAL model tensors (no profiler needed):
  reconstruct (fp16 trellis decode) / reconstruct_had_slice (fused) /
  hgemm_recon (fp16) / igemm_recon (int8) / had_r_128 /
  activation quant (amax+div+round) / w8 quant / int32 epilogue scale.

Usage: kernel_microbench.py MODEL_DIR [OUT.json]
Env: HIP_VISIBLE_DEVICES selects GPU. Single-GPU only.
"""

import json
import os
import statistics
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch

ITERS = 20


def med(fn, iters=ITERS):
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
    return round(statistics.median(ts), 3)


def main():
    model_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    torch.cuda.set_device(0)
    from exllamav3.model import Config, Model
    from exllamav3.ext import exllamav3_ext as ext
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load(device=torch.device("cuda:0"))
    print(f"loaded {model_dir}", flush=True)
    res = {}

    # --- find one MoE expert triple + one dense linear ---------------------
    moe = gate0 = up0 = dn0 = dense = None
    seen = set()

    def visit(mod):
        nonlocal moe, dense
        cn = type(mod).__name__
        if cn == "BlockSparseMLP" and moe is None:
            try:
                if len(mod.gates) and len(mod.ups) and len(mod.downs):
                    moe = mod
            except TypeError:
                pass
        if cn == "LinearEXL3" and dense is None:
            dense = mod

    # exllamav3 modules: walk via registered submodule lists is awkward;
    # use gc-free approach: model holds decoder blocks in a known attr.
    stack = [model]
    while stack:
        m = stack.pop()
        visit(m)
        try:
            attrs = vars(m).values()
        except TypeError:
            continue
        for attr in attrs:
            if attr is None or attr is m:
                continue
            cn = type(attr).__name__
            if "exllamav3" in type(attr).__module__:
                if id(attr) not in seen:
                    seen.add(id(attr))
                    stack.append(attr)
            elif isinstance(attr, (list, tuple)):
                for x in attr:
                    if "exllamav3" in type(x).__module__:
                        if id(x) not in seen:
                            seen.add(id(x))
                            stack.append(x)
    print(f"moe found={moe is not None} dense found={dense is not None}",
          flush=True)
    if moe is not None:
        gate0, up0, dn0 = moe.gates[0].inner, moe.ups[0].inner, moe.downs[0].inner
        res["expert_shapes"] = {
            "gate": [gate0.in_features, gate0.out_features],
            "up": [up0.in_features, up0.out_features],
            "down": [dn0.in_features, dn0.out_features],
        }
    if dense is not None:
        res["dense_shape"] = [dense.in_features, dense.out_features]

    with torch.inference_mode():
        # --- expert gate triple (rows=128 ~ 2048tok/128exp*8topk) ---------
        if moe is not None:
            k, n = gate0.in_features, gate0.out_features
            rows = 128
            dev = gate0.trellis.device
            w = torch.empty((k, n), dtype=torch.half, device=dev)
            res["recon_expert_ms"] = med(
                lambda: ext.reconstruct(w, gate0.trellis, gate0.K,
                                        gate0.mcg, gate0.mul1))
            a = torch.randn((rows, k), dtype=torch.half, device=dev)
            xh = torch.empty_like(a)
            res["had_in_expert_ms"] = med(
                lambda: ext.had_r_128(a, xh, gate0.suh, None, 1.0))
            w8 = (w.float() / 0.02).round().clamp(-128, 127).to(torch.int8)
            sa = a.float().abs().amax(dim=1, keepdim=True) / 127.0
            a8 = (a.float() / sa).round().clamp(-128, 127).to(torch.int8)
            c32 = torch.empty((rows, n), dtype=torch.int32, device=dev)
            res["igemm_expert_ms"] = med(lambda: ext.igemm_recon(a8, w8, c32))
            c16 = torch.empty((rows, n), dtype=torch.half, device=dev)
            res["hgemm_expert_ms"] = med(
                lambda: ext.hgemm_recon(xh, w, c16))
            res["actquant_expert_ms"] = med(
                lambda: (xh.float() / (xh.float().abs().amax(
                    dim=1, keepdim=True).clamp_min(1e-12) / 127.0)).round()
                .clamp(-128, 127).to(torch.int8))
            res["epilog_expert_ms"] = med(
                lambda: (c32.float() * sa * 0.02).to(torch.half))
        # --- dense linear at prefill rows ----------------------------------
        if dense is not None:
            k, n = dense.in_features, dense.out_features
            dev = dense.trellis.device
            for rows in (512, 2048):
                w = torch.empty((k, n), dtype=torch.half, device=dev)
                tag = f"rows{rows}"
                res[f"recon_dense_{tag}_ms"] = med(
                    lambda: ext.reconstruct(w, dense.trellis, dense.K,
                                            dense.mcg, dense.mul1))
                a = torch.randn((rows, k), dtype=torch.half, device=dev)
                xh = torch.empty_like(a)
                res[f"had_dense_{tag}_ms"] = med(
                    lambda: ext.had_r_128(a, xh, dense.suh, None, 1.0))
                c16 = torch.empty((rows, n), dtype=torch.half, device=dev)
                res[f"hgemm_dense_{tag}_ms"] = med(
                    lambda: ext.hgemm_recon(xh, w, c16))
                w8 = (w.float() / 0.02).round().clamp(-128, 127).to(torch.int8)
                sa = a.float().abs().amax(dim=1, keepdim=True) / 127.0
                a8 = (a.float() / sa).round().clamp(-128, 127).to(torch.int8)
                c32 = torch.empty((rows, n), dtype=torch.int32, device=dev)
                res[f"igemm_dense_{tag}_ms"] = med(
                    lambda: ext.igemm_recon(a8, w8, c32))

    print(json.dumps(res, indent=1), flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", out, flush=True)


main()
