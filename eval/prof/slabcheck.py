#!/usr/bin/env python3
"""Full-slab vs per-expert weight recon for ALL 128 experts (gate/up/down).
Usage: slabcheck.py MODEL_DIR
"""
import os
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
os.environ["EXL3_INT8_PREFILL"] = "1"
os.environ["EXL3_INT8_MOE_GROUPED"] = "1"
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
print("mlp:", mlp.key, flush=True)
dev = mlp.device

with torch.inference_mode():
    C = mlp._int8_moe_grouped_cache()
    E = C["E"]
    for tag, attr in (("g", "gates"), ("u", "ups"), ("d", "downs")):
        t = C["it"][tag]
        K, N = t["K"], t["N"]
        inners = [getattr(mlp, attr)[e].inner for e in range(E)]
        ptrs = t["ptrs"]
        s0 = C["s0" + tag].view(E)
        Wfp = torch.empty((E, K, N), dtype=torch.half, device=dev)
        W8 = torch.empty((E, K, N), dtype=torch.int8, device=dev)
        mlp._recon_int8_slab(W8, Wfp, ptrs, t["bK"], t["mcg"], t["mul1"], s0.view(E, 1, 1))
        torch.cuda.synchronize()
        bad = []
        for e in range(E):
            ref = mlp._recon_int8(inners[e], s0[e])
            dd = (W8[e].float() - ref.float()).abs().max().item()
            if dd != 0:
                bad.append((e, dd))
        print(f"{tag}: {len(bad)} mismatched experts of {E}" +
              (f" e.g. {bad[:8]}" if bad else ""), flush=True)
print("done", flush=True)
