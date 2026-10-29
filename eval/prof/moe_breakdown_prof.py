#!/usr/bin/env python3
"""Capture EXL3_INT8_MOE_GROUPED per-stage profile lines (fake s0, timing only).
Usage: moe_breakdown_prof.py MODEL_DIR T
Env: HIP_VISIBLE_DEVICES.
"""
import os
import sys

sys.path.insert(0, "/src")
os.environ["EXL3_INT8_PREFILL"] = "1"
os.environ["EXL3_INT8_MOE_GROUPED"] = "1"
os.environ["EXL3_INT8_MOE_PROFILE"] = "1"
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch

MODEL_DIR, T = sys.argv[1], int(sys.argv[2])
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
g = torch.Generator(device="cuda:0").manual_seed(99)
with torch.inference_mode():
    ids = torch.randint(0, config.vocab_size, (1, T), generator=g, device="cuda:0")
    _ = model.forward(ids, {})
    torch.cuda.synchronize()
    print("=== timed ===", flush=True)
    ids = torch.randint(0, config.vocab_size, (1, T), generator=g, device="cuda:0")
    _ = model.forward(ids, {})
    torch.cuda.synchronize()
print("done", flush=True)
