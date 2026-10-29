#!/usr/bin/env python3
"""Weight reconstruction cost on a REAL int8-grid (s0) model (dense 27B).

Times trellis->fp16 (ext.reconstruct) vs trellis->int8 via s0
(reconstruct + quant_w8_slab) on real linears. Also activation-side
Hadamard (had_r_128) and dense GEMM fp16 vs int8 for reference.
Usage: moe_breakdown_recon.py [OUT.json]
Env: HIP_VISIBLE_DEVICES.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else None
torch.cuda.set_device(0)
from exllamav3.model import Config, Model
from exllamav3.rocm_py.moe_int8 import quant_w8_slab
from exllamav3.ext import exllamav3_ext as ext

config = Config.from_directory("/work/models/qwen38-27b-exl3-305-int8grid")
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))
print("loaded 27b-int8grid", flush=True)

inners = []


def _walk(m, seen):
    if type(m).__name__ == "LinearEXL3":
        if getattr(m, "s0", None) is not None:
            inners.append(m)
        return
    try:
        attrs = vars(m).values()
    except TypeError:
        return
    for a in attrs:
        if a is None or a is m or id(a) in seen:
            continue
        if "exllamav3" in type(a).__module__:
            seen.add(id(a))
            _walk(a, seen)
        elif isinstance(a, (list, tuple)):
            for x in a:
                if "exllamav3" in type(x).__module__ and id(x) not in seen:
                    seen.add(id(x))
                    _walk(x, seen)


_walk(model, set())
print(f"s0 linears: {len(inners)}", flush=True)
# one gate/up/down-sized + one qkv-sized linear
picks = []
seen_shapes = set()
for m in inners:
    sh = (m.in_features, m.out_features)
    if sh not in seen_shapes:
        seen_shapes.add(sh)
        picks.append(m)
    if len(picks) >= 4:
        break
print("shapes:", [(m.in_features, m.out_features) for m in picks], flush=True)


def med(fn, iters=20):
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


res = {}
with torch.inference_mode():
    # global clock warmup on the biggest slab before any timing
    _wm = picks[-1]
    _wk, _wn = _wm.in_features, _wm.out_features
    _wdev = _wm.trellis.device
    _ww = torch.empty((_wk, _wn), dtype=torch.half, device=_wdev)
    _w8 = torch.empty((_wk, _wn), dtype=torch.int8, device=_wdev)
    _s0 = _wm.s0.to(dtype=torch.float32, device=_wdev)
    for _ in range(20):
        ext.reconstruct(_ww, _wm.trellis, _wm.K, _wm.mcg, _wm.mul1)
        quant_w8_slab(_ww.unsqueeze(0), _s0, _w8.unsqueeze(0))
    torch.cuda.synchronize()
    for m in picks:
        k, n = m.in_features, m.out_features
        dev = m.trellis.device
        s0 = m.s0.to(dtype=torch.float32, device=dev)
        d = {}
        w = torch.empty((k, n), dtype=torch.half, device=dev)
        d["recon_fp16_ms"] = med(
            lambda: ext.reconstruct(w, m.trellis, m.K, m.mcg, m.mul1))
        w8 = torch.empty((k, n), dtype=torch.int8, device=dev)
        d["quant_only_ms"] = med(
            lambda: quant_w8_slab(w.unsqueeze(0), s0, w8.unsqueeze(0)))
        def _rq():
            ext.reconstruct(w, m.trellis, m.K, m.mcg, m.mul1)
            quant_w8_slab(w.unsqueeze(0), s0, w8.unsqueeze(0))
        d["recon_int8_via_s0_ms"] = med(_rq)
        d["s0_value"] = float(m.s0_value)
        # activation-side Hadamard at T=2048
        T = 2048
        a = torch.randn((T, k), dtype=torch.half, device=dev)
        xh = torch.empty_like(a)
        d["had_act_T2048_ms"] = med(lambda: ext.had_r_128(a, xh, m.suh, None, 1.0))
        # GEMM fp16 vs int8 at T=2048
        sa = torch.full((T, 1), 0.02, dtype=torch.float, device=dev)
        a8 = torch.randint(-100, 100, (T, k), device=dev, dtype=torch.int32).to(torch.int8)
        c32 = torch.empty((T, n), dtype=torch.int32, device=dev)
        w8b = torch.randint(-100, 100, (k, n), device=dev, dtype=torch.int32).to(torch.int8)
        y = torch.empty((T, n), dtype=torch.half, device=dev)
        d["igemm_T2048_ms"] = med(lambda: ext.igemm_recon(a8, w8b, c32))
        d["hgemm_T2048_ms"] = med(lambda: ext.hgemm_recon(xh, w, y))
        res[f"{k}x{n}"] = d
        print(f"{k}x{n}: " + json.dumps(d), flush=True)
print(json.dumps(res, indent=1))
if OUT:
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT, flush=True)
