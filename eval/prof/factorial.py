#!/usr/bin/env python3
"""Factorial isolation: padding x duplicate-picks for grouped-vs-proto.
One MoE layer, hand-built routing. Case matrix:
  A: no pad (every active expert exactly R rows), unique picks
  B: padded (counts vary, cmax waste), unique picks
  C: no pad, duplicate picks (token twice in one expert)
  D: padded + duplicates
Usage: factorial.py MODEL_DIR
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
H = mlp.expert_size
E = mlp.num_local_experts or mlp.num_experts

g = torch.Generator(device="cuda:0").manual_seed(11)
with torch.inference_mode():
    # 8 experts (0..7), top-4 per token, T=64 tokens. Counts engineered.
    T, K8 = 64, 4
    y = torch.randn((T, H), generator=g, device=dev).half()
    base_w = torch.rand((T, K8), generator=g, device=dev).half()

    def run(sel, tag):
        w = base_w.clone()
        o1 = mlp._forward_int8_moe_grouped(y, T, sel, w, {})
        o2 = mlp._forward_int8_moe_grouped(y, T, sel, w, {})
        op = mlp._forward_int8_moe(y, T, sel, w, {})
        d = (o1.float() - op.float()).abs()
        dd = (o1.float() - o2.float()).abs()
        print(f"{tag}: g-vs-p maxabs={d.max().item():.3e} mean={d.mean().item():.3e} | g-det={dd.max().item():.2e}",
              flush=True)

    # A: exact 32 rows/expert, experts 0..7, unique (T*4=256 = 8*32)
    selA = torch.empty((T, K8), dtype=torch.long, device=dev)
    for t in range(T):
        for k in range(K8):
            selA[t, k] = (t + k * T) % 8 if False else (t * K8 + k) * 8 // 256
    # simpler: round-robin token t to experts {(4t+k) mod 8}, counts even 32
    for t in range(T):
        for k in range(K8):
            selA[t, k] = (4 * t + k) % 8
    run(selA, "A nopad unique")

    # B: skewed counts (expert 0 gets 64, expert 7 gets 4), unique
    selB = selA.clone()
    # move 28 of expert-7's rows to expert... ensure uniqueness per row:
    # rows where sel has 7: reassign half of them to expert 0 only if row lacks 0
    for t in range(T):
        row = selB[t]
        if 7 in row and 0 not in row:
            row[row == 7] = 0
            if len(set(row.tolist())) < 4:
                break
    run(selB, "B padded unique")

    # C: no pad + duplicate (token 0 picks expert 3 twice; keep counts even by swap)
    selC = selA.clone()
    selC[0, 0] = 3
    selC[0, 1] = 3  # duplicate
    run(selC, "C nopad dup")

    # D: padded + duplicate
    selD = selB.clone()
    selD[1, 0] = 5
    selD[1, 1] = 5
    run(selD, "D padded dup")
print("done", flush=True)
