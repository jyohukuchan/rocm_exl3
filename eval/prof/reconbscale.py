#!/usr/bin/env python3
"""reconstruct_batch batch-size dependence for one trellis.
Usage: reconbscale.py MODEL_DIR
Compares reconstruct (reference) vs reconstruct_batch at B=1,2,4,8,32,128
for expert 43's up trellis (fp16 outputs, before quant).
"""
import os
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
    tU = C["it"]["u"]
    K, N = tU["K"], tU["N"]
    ui = mlp.ups[43].inner
    ref = torch.empty((K, N), dtype=torch.half, device=dev)
    ext.reconstruct(ref, ui.trellis, ui.K, ui.mcg, ui.mul1)
    torch.cuda.synchronize()
    p43 = ui.trellis.data_ptr()
    p0 = mlp.ups[0].inner.trellis.data_ptr()
    for B in (1, 2, 4, 8, 32, 128):
        ptrs = torch.tensor([p43] + [p0] * (B - 1), dtype=torch.int64, device=dev)
        W = torch.empty((B, K, N), dtype=torch.half, device=dev)
        ext.reconstruct_batch(W, ptrs, tU["bK"], tU["mcg"], tU["mul1"])
        torch.cuda.synchronize()
        d = (W[0].float() - ref.float()).abs()
        print(f"B={B}: maxabs={d.max().item():.3e} nbad={(d > 0).sum().item()}", flush=True)
print("done", flush=True)
