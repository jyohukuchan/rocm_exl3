#!/usr/bin/env python3
"""Grouped-vs-proto battery: determinism + trigger hunt on one MoE layer.
Usage: battery.py MODEL_DIR
Sweeps T, magnitude, seeds; reports per-case maxabs (grouped vs proto)
and grouped-vs-grouped (determinism). Prints failing-case stats.
Env: EXL3_INT8_PREFILL=1, HIP_VISIBLE_DEVICES.
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
assert mlp is not None
print("mlp:", mlp.key, flush=True)
dev = mlp.device
E = mlp.num_local_experts or mlp.num_experts
H = mlp.expert_size

with torch.inference_mode():
    for T, mag, seed in [(64, 1.0, 1), (256, 1.0, 2), (512, 1.0, 3),
                         (256, 10.0, 4), (256, 0.1, 5), (2048, 1.0, 6)]:
        g = torch.Generator(device="cuda:0").manual_seed(seed)
        y = (torch.randn((T, H), generator=g, device=dev) * mag).half()
        sel = torch.randint(0, E, (T, 8), generator=g, device=dev)
        w = torch.rand((T, 8), generator=g, device=dev).half()
        o1 = mlp._forward_int8_moe_grouped(y, T, sel, w, {})
        o2 = mlp._forward_int8_moe_grouped(y, T, sel, w, {})
        op = mlp._forward_int8_moe(y, T, sel, w, {})
        dg = (o1.float() - o2.float()).abs().max().item()
        dp = (o1.float() - op.float()).abs()
        print(f"T={T} mag={mag}: grouped-det={dg:.2e} grouped-vs-proto maxabs={dp.max().item():.3e} mean={dp.mean().item():.3e}",
              flush=True)
        if dp.max().item() > 1e-3:
            rm = dp.max(dim=1).values
            big = (rm > 1e-3).nonzero().flatten()
            print(f"   {len(big)} rows diverge; e.g. row {big[0].item()}: max={rm[big[0]].item():.3e}", flush=True)
            # which experts serve row r0, and their counts?
            r0 = big[0].item()
            print(f"   row {r0} experts: {sel[r0].tolist()}", flush=True)
print("done", flush=True)
