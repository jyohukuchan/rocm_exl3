#!/usr/bin/env python3
"""Same-process proto-vs-grouped A/B on IDENTICAL inputs.
Flips M.INT8_MOE_GROUPED between two forwards; captures per-layer MoE
outputs for both. Eliminates cross-process / cross-token confounds.
Usage: ab_sameproc.py MODEL_DIR L SEED OUT.pt
Env: EXL3_INT8_PREFILL=1, HIP_VISIBLE_DEVICES.
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
print("loaded", flush=True)
assert hasattr(M.ext, "igemm_batched")

caps = {}
orig_g = M.BlockSparseMLP._forward_int8_moe_grouped
orig_p = M.BlockSparseMLP._forward_int8_moe


def cap(which, fn):
    def inner(self, y, bsz, sel, w, params):
        r = fn(self, y, bsz, sel, w, params)
        caps.setdefault(which, []).append((self.key, r.detach().float().cpu()))
        return r
    return inner


M.BlockSparseMLP._forward_int8_moe_grouped = cap("g", orig_g)
M.BlockSparseMLP._forward_int8_moe = cap("p", orig_p)
# NOTE: forward() reads the module-global flag at call time; flip it directly.
g = torch.Generator(device="cuda:0").manual_seed(seed)
results = {}
with torch.inference_mode():
    ids = torch.randint(0, config.vocab_size, (1, L), generator=g, device="cuda:0")
    M.INT8_MOE_GROUPED = True
    _ = model.forward(ids, {})
    torch.cuda.synchronize()
    for which in ("g", "p"):
        caps.clear()
        M.INT8_MOE_GROUPED = (which == "g")
        t0 = time.perf_counter()
        _ = model.forward(ids, {})
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        outs = caps[which]
        results[which] = {"outs": outs, "wall": round(wall, 3)}
        print(f"{which}: wall={wall:.3f}s layers={len(outs)}", flush=True)

go = dict(results["g"]["outs"])
po = dict(results["p"]["outs"])
comp = []
for k in go:
    d = (go[k].float() - po[k].float()).abs()
    comp.append((k, d.max().item()))
comp.sort(key=lambda x: -x[1])
print("worst:", [(k, "%.2e" % v) for k, v in comp[:5]], flush=True)
print("best:", [(k, "%.2e" % v) for k, v in comp[-3:]], flush=True)
torch.save({"comp": comp,
            "wall_g": results["g"]["wall"], "wall_p": results["p"]["wall"]}, out)
print("wrote", out, flush=True)
