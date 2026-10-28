#!/usr/bin/env python3
"""Grouped-vs-prototype INT8 MoE: numerical equivalence + timing.
Runs one full prefill forward, saves each MoE layer output + timings to OUT.pt.
Usage: equiv_grouped.py MODEL_DIR L SEED OUT.pt
Env: EXL3_INT8_PREFILL=1, EXL3_INT8_MOE_GROUPED=0/1, HIP_VISIBLE_DEVICES.
"""
import os
import sys
import time

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
os.environ["EXL3_INT8_PREFILL"] = "1"
import torch

model_dir, L, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
torch.cuda.set_device(0)
from exllamav3.model import Config, Model
from exllamav3.modules import block_sparse_mlp as M

config = Config.from_directory(model_dir)
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))
print(f"loaded {model_dir} grouped={M.INT8_MOE_GROUPED}", flush=True)

which = "grouped" if (M.INT8_MOE_GROUPED and hasattr(M.ext, "igemm_batched")) else "proto"
fn = (M.BlockSparseMLP._forward_int8_moe_grouped if which == "grouped"
      else M.BlockSparseMLP._forward_int8_moe)
caps = []


def cap(self, y, bsz, sel, w, params):
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    r = fn(self, y, bsz, sel, w, params)
    t1.record()
    caps.append((self.key, r.detach().float().cpu(), t0, t1))
    return r


if which == "grouped":
    M.BlockSparseMLP._forward_int8_moe_grouped = cap
else:
    M.BlockSparseMLP._forward_int8_moe = cap

g = torch.Generator(device="cuda:0").manual_seed(seed)
with torch.inference_mode():
    ids = torch.randint(0, config.vocab_size, (1, L), generator=g, device="cuda:0")
    _ = model.forward(ids, {})
    torch.cuda.synchronize()
    caps.clear()
    ids = torch.randint(0, config.vocab_size, (1, L), generator=g, device="cuda:0")
    t0 = time.perf_counter()
    _ = model.forward(ids, {})
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

torch.cuda.synchronize()
per_layer = [(k, v, round(a.elapsed_time(b), 2)) for k, v, a, b in caps]
tot = sum(t for _, _, t in per_layer)
torch.save({"which": which, "outs": [(k, v) for k, v, _ in per_layer],
            "times": [(k, t) for k, _, t in per_layer],
            "moe_total_ms": round(tot, 1), "wall_s": round(wall, 3)}, out)
print(f"which={which} wall={wall:.3f}s moe_total={tot:.1f}ms saved {out}", flush=True)
