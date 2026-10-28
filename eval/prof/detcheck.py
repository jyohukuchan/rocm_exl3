#!/usr/bin/env python3
"""In-process determinism: reconstruct / reconstruct_batch / quant chain.
Usage: detcheck.py MODEL_DIR
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
    st = mlp._int8_moe_state()
    s0u = st[2][43]
    # 1. reconstruct twice
    w1 = torch.empty((K, N), dtype=torch.half, device=dev)
    w2 = torch.empty((K, N), dtype=torch.half, device=dev)
    ext.reconstruct(w1, ui.trellis, ui.K, ui.mcg, ui.mul1)
    ext.reconstruct(w2, ui.trellis, ui.K, ui.mcg, ui.mul1)
    torch.cuda.synchronize()
    print("reconstruct x2:", (w1.float() - w2.float()).abs().max().item(), flush=True)
    # 2. reconstruct_batch B=1 twice, same Wfp buffer reuse + fresh buffer
    p = torch.tensor([ui.trellis.data_ptr()], dtype=torch.int64, device=dev)
    a1 = torch.empty((1, K, N), dtype=torch.half, device=dev)
    a2 = torch.empty((1, K, N), dtype=torch.half, device=dev)
    ext.reconstruct_batch(a1, p, tU["bK"], tU["mcg"], tU["mul1"])
    ext.reconstruct_batch(a2, p, tU["bK"], tU["mcg"], tU["mul1"])
    torch.cuda.synchronize()
    print("recon_batch B=1 x2 (fresh bufs):", (a1.float() - a2.float()).abs().max().item(), flush=True)
    ext.reconstruct_batch(a1, p, tU["bK"], tU["mcg"], tU["mul1"])
    torch.cuda.synchronize()
    print("recon_batch B=1 3rd call vs 2nd:", (a1.float() - a2.float()).abs().max().item(), flush=True)
    print("recon_batch B=1 vs reconstruct:", (a1[0].float() - w1.float()).abs().max().item(), flush=True)
    # 3. quant chain twice on same Wfp
    s0t = C["s0u"][43:44]
    q1 = (a1.float() / s0t).round().clamp(-128, 127).to(torch.int8)
    q2 = (a1.float() / s0t).round().clamp(-128, 127).to(torch.int8)
    torch.cuda.synchronize()
    print("quant x2:", (q1.float() - q2.float()).abs().max().item(), flush=True)
    qp = (w1.float() / s0u).round().clamp(-128, 127).to(torch.int8)
    print("quant(batch-Wfp,s0-tensor) vs quant(recon-W,s0-pyfloat):",
          (q1[0].float() - qp.float()).abs().max().item(), flush=True)
print("done", flush=True)
