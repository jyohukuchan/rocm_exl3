#!/usr/bin/env python3
"""INT8 grouped-vs-prototype equivalence with fake s0 (TIMING build only).

Verifies which kernels the timed grouped path uses: gu should match the
prototype closely (fixed), d is known-buggy (factorial128 residual ~6.2).
Numerics invalid (constant s0, foreign su); only the grouped-vs-proto
DIVERGENCE pattern matters.
Usage: moe_breakdown_equiv.py MODEL_DIR [T]
Env: HIP_VISIBLE_DEVICES.
"""
import os
import sys

sys.path.insert(0, "/src")
os.environ["EXL3_INT8_PREFILL"] = "1"
os.environ["EXL3_INT8_MOE_GROUPED"] = "1"
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch

MODEL_DIR = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 256
torch.cuda.set_device(0)
from exllamav3.model import Config, Model
from exllamav3.modules import block_sparse_mlp as M

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))


def _fake_state(self):
    gi = self.gates[0].inner
    n = len(self.gates)
    s = [0.02] * n
    return (gi.suh, s, s, s)


M.BlockSparseMLP._int8_moe_state = _fake_state
print("fake s0 installed (equivalence pattern only)", flush=True)

mlps = []


def _walk(m, seen):
    if type(m).__name__ == "BlockSparseMLP":
        mlps.append(m)
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
mlp = mlps[8]
E = len(mlp.gates)
print("mlp:", mlp.key, "E=", E, flush=True)
g = torch.Generator(device="cuda:0").manual_seed(3)
dev = mlp.device
with torch.inference_mode():
    y = torch.randn((T, mlp.expert_size), dtype=torch.half, device=dev)
    sel = torch.randint(0, E, (T, 8), generator=g, device=dev)
    w = torch.rand((T, 8), generator=g, device=dev).half()
    op = mlp._forward_int8_moe(y, T, sel, w, {})
    torch.cuda.synchronize()
    og = mlp._forward_int8_moe_grouped(y, T, sel, w, {})
    torch.cuda.synchronize()
    d = (op.float() - og.float()).abs()
    print(f"T={T} grouped-vs-proto: maxabs={d.max().item():.4f} "
          f"mean={d.mean().item():.6f}", flush=True)
    print(f"proto out scale: maxabs={op.float().abs().max().item():.4f} "
          f"meanabs={op.float().abs().mean().item():.6f}", flush=True)
    big = (d > 0.1).nonzero()
    print(f"rows>0.1: {big.shape[0]}", flush=True)
print("done", flush=True)
