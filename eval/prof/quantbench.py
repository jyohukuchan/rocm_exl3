#!/usr/bin/env python3
"""Quant variant speed + exactness on a real [128,2048,768] fp16 slab.
Usage: quantbench.py MODEL_DIR
Variants: fp32-div-tensor (current), fp32-div-pyscalar, fp32-mul-inv,
fp16-div, fp16-mul-inv. Reports ms + maxabs vs fp32-div-tensor.
"""
import os
import statistics
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
os.environ["EXL3_INT8_PREFILL"] = "1"
import torch

model_dir = sys.argv[1]
torch.cuda.set_device(0)
from exllamav3.model import Config, Model

config = Config.from_directory(model_dir)
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))
print("loaded", flush=True)

mlp = None
stack = [model]
seen = set()
while stack:
    m = stack.pop()
    if type(m).__name__ == "BlockSparseMLP":
        mlp = m
        break
    try:
        attrs = vars(m).values()
    except TypeError:
        continue
    for attr in attrs:
        if attr is None or attr is m:
            continue
        if "exllamav3" in type(attr).__module__ and id(attr) not in seen:
            seen.add(id(attr))
            stack.append(attr)
        elif isinstance(attr, (list, tuple)):
            for x in attr:
                if "exllamav3" in type(x).__module__ and id(x) not in seen:
                    seen.add(id(x))
                    stack.append(x)
from exllamav3.ext import exllamav3_ext as ext
dev = mlp.device

with torch.inference_mode():
    C = mlp._int8_moe_grouped_cache()
    tG = C["it"]["g"]
    E, K, N = C["E"], tG["K"], tG["N"]
    Wfp = torch.empty((E, K, N), dtype=torch.half, device=dev)
    ext.reconstruct_batch(Wfp, tG["ptrs"], tG["bK"], tG["mcg"], tG["mul1"])
    torch.cuda.synchronize()
    s0t = C["s0g"]  # [E,1,1] fp32 cuda
    s0py = float(s0t[43, 0, 0])
    inv = (1.0 / s0t)
    invh = inv.half()
    s0h = s0t.half()
    W8 = torch.empty((E, K, N), dtype=torch.int8, device=dev)

    def med(fn, iters=10):
        for _ in range(3):
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

    outs = {}
    outs["fp32-div-tensor"] = (med(lambda: W8.copy_((Wfp.float() / s0t).round().clamp(-128, 127).to(torch.int8))), W8.clone())
    outs["fp32-mul-inv"] = (med(lambda: W8.copy_((Wfp.float() * inv).round().clamp(-128, 127).to(torch.int8))), W8.clone())
    outs["fp16-div-tensor"] = (med(lambda: W8.copy_((Wfp / s0h).round().clamp(-128, 127).to(torch.int8))), W8.clone())
    outs["fp16-mul-inv"] = (med(lambda: W8.copy_((Wfp * invh).round().clamp(-128, 127).to(torch.int8))), W8.clone())
    ref = outs["fp32-div-tensor"][1]
    for k, (ms, v) in outs.items():
        d = (v.float() - ref.float()).abs().max().item()
        print(f"{k}: {ms}ms maxabs-vs-ref={d}", flush=True)
print("done", flush=True)
