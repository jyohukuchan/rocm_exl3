#!/usr/bin/env python3
"""Expert-43 deep dive: run grouped (with DUMP=1), then recompute expert 43
proto-style from the same inputs and compare every intermediate.
Usage: analyze43.py MODEL_DIR
"""
import os
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
os.environ["EXL3_INT8_PREFILL"] = "1"
os.environ["EXL3_INT8_MOE_GROUPED"] = "1"
os.environ["EXL3_INT8_MOE_DUMP"] = "1"
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
H = mlp.expert_size
E = mlp.num_local_experts or mlp.num_experts

g = torch.Generator(device="cuda:0").manual_seed(21)
with torch.inference_mode():
    T = 1024
    y = torch.randn((T, H), generator=g, device=dev).half()
    sel = torch.empty((T, 8), dtype=torch.long, device=dev)
    base = torch.randperm(E, generator=g, device=dev)
    for t in range(T):
        s = (t * 8) % E
        sel[t] = torch.tensor([(base[(s + k) % E].item()) for k in range(8)], device=dev)
    w = torch.rand((T, 8), generator=g, device=dev).half()
    out_g = mlp._forward_int8_moe_grouped(y, T, sel, w, {})

D = torch.load("/tmp/moe_dump.pt", map_location="cpu", weights_only=False)
tok_pad, w_pad, df = D["tok_pad"], D["w_pad"], D["df"]
SAslab, Xgu = D["SAslab"], D["Xgu"]
active, cmax = D["active"], D["cmax"]
counts, starts = D["counts"], D["starts"]
print(f"B={len(active)} cmax={cmax}", flush=True)

C = mlp._int8_moe_grouped_cache()
su = C["su"]
# recompute a8/sa from y (same as method)
y_d, sel_d, w_d = D["y"].to(dev).half(), D["sel"].to(dev), D["w"].to(dev).half()
xh = torch.empty_like(y_d)
ext.had_r_128(y_d, xh, su, None, 1.0)
sa = xh.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
a8 = (xh.float() / sa).round().clamp(-128, 127).to(torch.int8)

e = 43
b = active.index(e)
c = counts[e]
r0 = b * cmax
rows = tok_pad[r0:r0 + c]
print(f"expert 43: batch {b} count {c} rows[:8]={rows[:8].tolist()}", flush=True)
# 1. gather check
xg = Xgu.view(-1, Xgu.shape[-1])[r0:r0 + c].to(dev)
ref_x = a8[rows.to(dev)]
print("Xgu[43] vs a8[rows]:", (xg.float() - ref_x.float()).abs().max().item(), flush=True)
sg = SAslab.view(-1, 1)[r0:r0 + c].to(dev)
ref_s = sa[rows.to(dev)]
print("SAslab[43] vs sa[rows]:", (sg.float() - ref_s.float()).abs().max().item(), flush=True)
wg = w_pad[r0:r0 + c]
# proto rows for expert 43: token_sorted segment
from exllamav3.util.tensor import buffered_interleaved_arange
flat_token = buffered_interleaved_arange(T, 8, device=dev)
order = sel_d.reshape(-1).argsort()
token_sorted = flat_token[order]
weight_sorted = w_d.reshape(-1)[order]
s = starts[e]
top = token_sorted[s:s + c]
wgt = weight_sorted[s:s + c]
print("tok rows match proto top:", torch.equal(rows.to(dev), top), flush=True)
print("wgt match:", torch.equal(wg.to(dev).half(), wgt), flush=True)
# 2. proto contribution for expert 43 gate..down (reuse module helper for full proto? just gate+up+silu+down)
gi, ui, di = mlp.gates[e].inner, mlp.ups[e].inner, mlp.downs[e].inner
st = mlp._int8_moe_state()
s0g, s0u, s0d = st[1][e], st[2][e], st[3][e]
a8e = a8.index_select(0, top)
sae = sa.index_select(0, top)
Wg8 = mlp._recon_int8(gi, s0g)
Cg0 = torch.empty((c, gi.out_features), dtype=torch.int32, device=dev)
ext.igemm_recon(a8e, Wg8, Cg0)
gp = (Cg0.float() * sae * s0g).to(torch.half)
ext.had_r_128(gp, gp, None, gi.svh, 1.0)
# grouped gate slice from dump? Cg not saved; compare via df at the end.
# 3. df slice vs proto full contribution
Wu8 = mlp._recon_int8(ui, s0u)
Cu0 = torch.empty((c, ui.out_features), dtype=torch.int32, device=dev)
ext.igemm_recon(a8e, Wu8, Cu0)
up = (Cu0.float() * sae * s0u).to(torch.half)
ext.had_r_128(up, up, None, ui.svh, 1.0)
ap = torch.empty_like(up)
mlp.activation_fn_call(gp, up, ap, mlp.act_limit)
with torch.inference_mode():
    sadp = ap.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    a8dp = (ap.float() / sadp).round().clamp(-128, 127).to(torch.int8)
Wd8 = mlp._recon_int8(di, s0d)
Cd0 = torch.empty((c, di.out_features), dtype=torch.int32, device=dev)
ext.igemm_recon(a8dp, Wd8, Cd0)
dp = (Cd0.float() * sadp * s0d).to(torch.half)
ext.had_r_128(dp, dp, None, di.svh, 1.0)
dp = dp.to(torch.float).mul_(wgt.unsqueeze(1))
df43 = df.view(-1, df.shape[-1])[r0:r0 + c].to(dev)
print("df[43] vs proto contrib:", (df43.float() - dp.float()).abs().max().item(), flush=True)
print("done", flush=True)
