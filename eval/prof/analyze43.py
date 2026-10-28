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

# B=1 grouped-style chain for expert 43 (same inputs) vs proto
tG, tU, tD = C["it"]["g"], C["it"]["u"], C["it"]["d"]
Xb = a8[top].view(1, -1, tG["K"])
SAb = sa[top].view(1, -1, 1)
Wf = torch.empty((1, tG["K"], tG["N"]), dtype=torch.half, device=dev)
mlp._recon_int8_slab(
    (W8g1 := torch.empty((1, tG["K"], tG["N"]), dtype=torch.int8, device=dev)),
    Wf, tG["ptrs"][43:44], tG["bK"], tG["mcg"], tG["mul1"], C["s0g"][43:44])
Cg1 = torch.empty((1, c, tG["N"]), dtype=torch.int32, device=dev)
ext.igemm_batched(Xb, W8g1, Cg1)
gg1 = (Cg1.float() * SAb * C["s0g"][43:44]).to(torch.half).view(-1, tG["N"])
ext.had_r_128_batch(gg1, gg1, None, C["svg"][43:44],
                    torch.tensor([0], dtype=torch.int64, device=dev), c, 1.0)
print("B=1 gate vs proto:", (gg1.float() - gp.float()).abs().max().item(), flush=True)
D2 = torch.load("/tmp/moe_dump.pt", map_location="cpu", weights_only=False)
if D2["Cg"] is not None:
    Cg43 = D2["Cg"].view(-1, D2["Cg"].shape[-1])[43 * cmax:43 * cmax + c].to(dev)
    print("Cg[43]-full vs Cg-B=1:",
          (Cg43.float() - Cg1.float()).abs().max().item(), flush=True)
# B=1 up + silu for expert 43
WfU = torch.empty((1, tU["K"], tU["N"]), dtype=torch.half, device=dev)
W8U1 = torch.empty((1, tU["K"], tU["N"]), dtype=torch.int8, device=dev)
mlp._recon_int8_slab(W8U1, WfU, tU["ptrs"][43:44], tU["bK"], tU["mcg"], tU["mul1"], C["s0u"][43:44])
Cu1 = torch.empty((1, c, tU["N"]), dtype=torch.int32, device=dev)
ext.igemm_batched(Xb, W8U1, Cu1)
ug1 = (Cu1.float() * SAb * C["s0u"][43:44]).to(torch.half).view(-1, tU["N"])
ext.had_r_128_batch(ug1, ug1, None, C["svu"][43:44],
                    torch.tensor([0], dtype=torch.int64, device=dev), c, 1.0)
ag1 = torch.empty_like(ug1)
mlp.activation_fn_call(gg1, ug1, ag1, mlp.act_limit)
if D2["a"] is not None:
    a43 = D2["a"].view(-1, D2["a"].shape[-1])[43 * cmax:43 * cmax + c].to(dev)
    print("a[43]-full vs a-B=1:",
          (a43.float() - ag1.float()).abs().max().item(), flush=True)
if D2["d"] is not None:
    d43 = D2["d"].view(-1, D2["d"].shape[-1])[43 * cmax:43 * cmax + c].to(dev)
    print("d[43]-full: range [%s, %s] mean %s" % (
        d43.float().min().item(), d43.float().max().item(), d43.float().mean().item()),
        flush=True)
# B=1 down chain for expert 43 from ag1 (verified == full a[43])
with torch.inference_mode():
    sad1 = ag1.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    Xd1 = (ag1.float() / sad1).round().clamp(-128, 127).to(torch.int8)
tD = C["it"]["d"]
WfD = torch.empty((1, tD["K"], tD["N"]), dtype=torch.half, device=dev)
W8D1 = torch.empty((1, tD["K"], tD["N"]), dtype=torch.int8, device=dev)
mlp._recon_int8_slab(W8D1, WfD, tD["ptrs"][43:44], tD["bK"], tD["mcg"], tD["mul1"], C["s0d"][43:44])
Cd1 = torch.empty((1, c, tD["N"]), dtype=torch.int32, device=dev)
ext.igemm_batched(Xd1.view(1, -1, tD["K"]), W8D1, Cd1)
dg1 = (Cd1.float() * sad1.view(1, -1, 1) * C["s0d"][43:44]).to(torch.half).view(-1, tD["N"])
ext.had_r_128_batch(dg1, dg1, None, C["svd"][43:44],
                    torch.tensor([0], dtype=torch.int64, device=dev), c, 1.0)
print("B=1 down vs proto:", (dg1.float() - dp.float()).abs().max().item(), flush=True)
# HEAD-ON: up-branch scales + epilogue for expert 43
ui43 = mlp.ups[43].inner
print("svu[43] vs ups[43].svh:",
      (C["svu"][43].float() - ui43.svh.float()).abs().max().item(), flush=True)
print("svg[43] vs gates[43].svh:",
      (C["svg"][43].float() - mlp.gates[43].inner.svh.float()).abs().max().item(), flush=True)
print("svd[43] vs downs[43].svh:",
      (C["svd"][43].float() - mlp.downs[43].inner.svh.float()).abs().max().item(), flush=True)
st43 = mlp._int8_moe_state()
print("s0u[43] cache vs state:", float(C["s0u"][43, 0, 0]) - float(st43[2][43]), flush=True)
print("s0d[43] cache vs state:", float(C["s0d"][43, 0, 0]) - float(st43[3][43]), flush=True)
print("s0g[43] cache vs state:", float(C["s0g"][43, 0, 0]) - float(st43[1][43]), flush=True)
print("ug1(B=1 up) vs proto up:", (ug1.float() - up.float()).abs().max().item(), flush=True)
print("Cu1(B=1 up GEMM) vs proto Cu0:",
      (Cu1.float() - Cu0.float()).abs().max().item(), flush=True)
if D2["Cd"] is not None:
    Cd43 = D2["Cd"].view(-1, D2["Cd"].shape[-1])[43 * cmax:43 * cmax + c].to(dev)
    print("Cd[43]-full vs Cd-B=1:",
          (Cd43.float() - Cd1.float()).abs().max().item(), flush=True)
    print("d[43]-full vs d-B=1:",
          (d43.float() - dg1.float()).abs().max().item(), flush=True)
print("done", flush=True)
