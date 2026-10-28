#!/usr/bin/env python3
"""Final discriminator: W8U1 vs w8pu in-process; empty-C vs zeroed-C.
Usage: final43.py MODEL_DIR
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
from exllamav3.ext import exllamav3_ext as ext
dev = mlp.expert_size and mlp.device
H = mlp.expert_size

g = torch.Generator(device="cuda:0").manual_seed(21)
with torch.inference_mode():
    C = mlp._int8_moe_grouped_cache()
    T = 1024
    y = torch.randn((T, H), generator=g, device=dev).half()
    sel = torch.empty((T, 8), dtype=torch.long, device=dev)
    E = mlp.num_local_experts or mlp.num_experts
    base = torch.randperm(E, generator=g, device=dev)
    for t in range(T):
        s = (t * 8) % E
        sel[t] = torch.tensor([(base[(s + k) % E].item()) for k in range(8)], device=dev)
    # shared rotation+quant (same as method)
    xh = torch.empty_like(y)
    ext.had_r_128(y, xh, C["su"], None, 1.0)
    sa = xh.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    a8 = (xh.float() / sa).round().clamp(-128, 127).to(torch.int8)
    # expert 43 rows
    e = 43
    from exllamav3.util.tensor import buffered_interleaved_arange
    flat_token = buffered_interleaved_arange(T, 8, device=dev)
    order = sel.reshape(-1).argsort()
    token_sorted = flat_token[order]
    counts = torch.bincount(sel.reshape(-1), minlength=E).tolist()
    starts = [0]
    for c in counts:
        starts.append(starts[-1] + c)
    top = token_sorted[starts[e]:starts[e] + counts[e]]
    c = len(top)
    a8e = a8.index_select(0, top)
    # weights both ways, IN THIS PROCESS
    ui = mlp.ups[e].inner
    st = mlp._int8_moe_state()
    s0u = st[2][e]
    w8pu = mlp._recon_int8(ui, s0u)
    tU = C["it"]["u"]
    Wf = torch.empty((1, tU["K"], tU["N"]), dtype=torch.half, device=dev)
    W8U1 = torch.empty((1, tU["K"], tU["N"]), dtype=torch.int8, device=dev)
    mlp._recon_int8_slab(W8U1, Wf, tU["ptrs"][e:e+1], tU["bK"], tU["mcg"], tU["mul1"], C["s0u"][e:e+1])
    print("W8U1 vs w8pu:", (W8U1[0].float() - w8pu.float()).abs().max().item(), flush=True)
    # GEMM: empty-C vs zeroed-C vs per-batch
    Xb = a8e.view(1, -1, tU["K"])
    Cr = torch.empty((c, tU["N"]), dtype=torch.int32, device=dev)
    ext.igemm_recon(a8e, w8pu, Cr)
    Ce = torch.empty((1, c, tU["N"]), dtype=torch.int32, device=dev)
    ext.igemm_batched(Xb, W8U1, Ce)
    Cz = torch.zeros((1, c, tU["N"]), dtype=torch.int32, device=dev)
    ext.igemm_batched(Xb, W8U1, Cz)
    print("batched(empty-C) vs recon:", (Ce[0].float() - Cr.float()).abs().max().item(), flush=True)
    print("batched(zeroed-C) vs recon:", (Cz[0].float() - Cr.float()).abs().max().item(), flush=True)
    print("batched(empty) vs batched(zeroed):", (Ce[0].float() - Cz[0].float()).abs().max().item(), flush=True)
    # also gate for control
    gi = mlp.gates[e].inner
    s0g = st[1][e]
    w8pg = mlp._recon_int8(gi, s0g)
    tG = C["it"]["g"]
    WfG = torch.empty((1, tG["K"], tG["N"]), dtype=torch.half, device=dev)
    W8G1 = torch.empty((1, tG["K"], tG["N"]), dtype=torch.int8, device=dev)
    mlp._recon_int8_slab(W8G1, WfG, tG["ptrs"][e:e+1], tG["bK"], tG["mcg"], tG["mul1"], C["s0g"][e:e+1])
    print("W8G1 vs w8pg:", (W8G1[0].float() - w8pg.float()).abs().max().item(), flush=True)
    Xg = a8e.view(1, -1, tG["K"])
    Crg = torch.empty((c, tG["N"]), dtype=torch.int32, device=dev)
    ext.igemm_recon(a8e, w8pg, Crg)
    Ceg = torch.empty((1, c, tG["N"]), dtype=torch.int32, device=dev)
    ext.igemm_batched(Xg, W8G1, Ceg)
    print("gate batched(empty-C) vs recon:", (Ceg[0].float() - Crg.float()).abs().max().item(), flush=True)
print("done", flush=True)
