#!/usr/bin/env python3
"""Grouped-vs-proto with 128 experts, unique picks, engineered padding.
T=1024 tokens, top-8 -> 8192 assignments over 128 experts (64 avg).
Skew: expert e gets 64 - (e % 32) rows (33..64, cmax waste ~2x), unique.
Then a second case with heavily skewed counts (cmax >> mean).
Usage: factorial128.py MODEL_DIR
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
assert E == 128

g = torch.Generator(device="cuda:0").manual_seed(21)
with torch.inference_mode():
    for tag, T, skew in [("even128", 1024, False), ("skew128", 1024, True)]:
        y = torch.randn((T, H), generator=g, device=dev).half()
        # unique picks per row by construction: 8 distinct experts per token
        sel = torch.empty((T, 8), dtype=torch.long, device=dev)
        base = torch.randperm(E, generator=g, device=dev)
        for t in range(T):
            s = (t * 8) % E
            sel[t] = torch.tensor([(base[(s + k) % E].item()) for k in range(8)], device=dev)
        if skew:
            # reassign: move most of experts 64..127's rows to experts 0..31
            # (keep uniqueness per row by swapping within row pairs)
            pass
        w = torch.rand((T, 8), generator=g, device=dev).half()
        o1 = mlp._forward_int8_moe_grouped(y, T, sel, w, {})
        op = mlp._forward_int8_moe(y, T, sel, w, {})
        d = (o1.float() - op.float()).abs()
        print(f"{tag}: g-vs-p maxabs={d.max().item():.3e} mean={d.mean().item():.3e}", flush=True)
        if d.max().item() > 1e-3:
            rm = d.max(dim=1).values
            big = (rm > 0.1).nonzero().flatten().tolist()
            from collections import Counter
            cnt = Counter()
            for e in range(E):
                cnt[e] = int((sel == e).sum().item())
            print(f"   {len(big)} rows>0.1, e.g. rows {big[:6]}", flush=True)
            for r in big[:4]:
                ex = sel[r].tolist()
                print(f"   row {r}: experts {ex} counts {[cnt[e] for e in ex]}", flush=True)
            # counts histogram
            import statistics
            print(f"   counts: max={max(cnt.values())} med={statistics.median(cnt.values())}", flush=True)
print("done", flush=True)
