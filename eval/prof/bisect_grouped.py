#!/usr/bin/env python3
"""Bisect grouped-vs-prototype divergence stage by stage (one MoE layer).
Usage: bisect_grouped.py MODEL_DIR  (env EXL3_INT8_PREFILL=1)
Compares: W8 slabs, C slabs, g/u, a, d, out between the two paths.
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
from exllamav3.modules import block_sparse_mlp as M

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
assert mlp is not None
print("mlp:", mlp.key, flush=True)

from exllamav3.ext import exllamav3_ext as ext

T, H, top_k, E = 256, mlp.expert_size, 8, mlp.num_local_experts or mlp.num_experts
g = torch.Generator(device="cuda:0").manual_seed(3)
dev = mlp.device
with torch.inference_mode():
    y = torch.randn((T, H), dtype=torch.half, device=dev)
    sel = torch.randint(0, E, (T, top_k), generator=g, device=dev)
    w = torch.rand((T, top_k), generator=g, device=dev).half()

    out_p = mlp._forward_int8_moe(y, T, sel, w, {})
    out_g = mlp._forward_int8_moe_grouped(y, T, sel, w, {})
    d = (out_p.float() - out_g.float()).abs()
    print(f"out: maxabs={d.max().item():.3e} mean={d.mean().item():.3e}", flush=True)

    # Stage 1: shared rotation + quant (duplicated here for inspection)
    C = mlp._int8_moe_grouped_cache()
    xh = torch.empty_like(y)
    ext.had_r_128(y, xh, C["su"], None, 1.0)
    sa = xh.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    a8 = (xh.float() / sa).round().clamp(-128, 127).to(torch.int8)

    # Stage 2: weight slabs vs per-expert recon (expert 0 of gate)
    gi = mlp.gates[0].inner
    s0g = C["s0g"][0, 0, 0].item()
    w8p = mlp._recon_int8(gi, torch.tensor(s0g, device=dev))
    print(f"proto Wg8[0]: shape {tuple(w8p.shape)} range [{w8p.min().item()},{w8p.max().item()}]")

    import statistics

    def med(fn, iters=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        return round(statistics.median(ts), 3)

    E = C["E"]
    t = C["it"]["g"]
    K, N = t["K"], t["N"]
    ptrs = t["ptrs"][:E].contiguous()
    s0 = C["s0g"][:E]
    Wfp = torch.empty((E, K, N), dtype=torch.half, device=dev)
    W8 = torch.empty((E, K, N), dtype=torch.int8, device=dev)
    print("recon_batch alone:", med(lambda: ext.reconstruct_batch(Wfp, ptrs, t["bK"], t["mcg"], t["mul1"])), flush=True)
    ext.reconstruct_batch(Wfp, ptrs, t["bK"], t["mcg"], t["mul1"])
    torch.cuda.synchronize()
    print("quant alone:", med(lambda: W8.copy_((Wfp.float() / s0).round().clamp(-128, 127).to(torch.int8))), flush=True)
    print("slab helper:", med(lambda: mlp._recon_int8_slab(W8, Wfp, ptrs, t["bK"], t["mcg"], t["mul1"], s0)), flush=True)
    # slab vs per-expert for expert 0
    mlp._recon_int8_slab(W8, Wfp, ptrs, t["bK"], t["mcg"], t["mul1"], s0)
    torch.cuda.synchronize()
    d = (W8[0].float() - w8p.float()).abs()
    print(f"W8 slab[0] vs proto: maxabs={d.max().item():.3e}", flush=True)

    # Quant variants: speed + bitwise vs fp32 reference
    ref = W8.clone()
    s0h = s0.half()
    def q_fp16():
        with torch.inference_mode():
            W8.copy_((Wfp / s0h).round().clamp(-128, 127).to(torch.int8))
    print("quant fp16:", med(q_fp16), flush=True)
    torch.cuda.synchronize()
    dq = (W8.float() - ref.float()).abs()
    print(f"fp16-quant vs fp32-quant: maxabs={dq.max().item():.3e}", flush=True)

    # Stage 3: input gather + epilogue vs prototype pieces for expert 0
    # (recompute grouping here)
    from exllamav3.util.tensor import buffered_interleaved_arange
    num_ex = E
    flat_token = buffered_interleaved_arange(T, top_k, device=dev)
    order = sel.reshape(-1).argsort()
    token_sorted = flat_token[order]
    weight_sorted = w.reshape(-1)[order]
    counts = torch.bincount(sel.reshape(-1), minlength=num_ex).tolist()
    starts, acc = [0], 0
    for c in counts:
        acc += c
        starts.append(acc)
    ef = next(e for e in range(num_ex) if counts[e] > 0)
    print(f"bisect expert: {ef} rows={counts[ef]}", flush=True)
    e0rows = token_sorted[starts[ef]:starts[ef] + counts[ef]]
    # proto epilogue for expert ef gate (recompute)
    gi = mlp.gates[ef].inner
    s0g = C[f"s0g"][ef, 0, 0].item()
    a8e = a8.index_select(0, e0rows)
    sae = sa.index_select(0, e0rows)
    Cg0 = torch.empty((len(e0rows), gi.out_features), dtype=torch.int32, device=dev)
    ext.igemm_recon(a8e, w8p, Cg0)
    gp = (Cg0.float() * sae * torch.tensor(s0g, device=dev)).to(torch.half)
    ext.had_r_128(gp, gp, None, gi.svh, 1.0)
    # grouped epilogue slice for expert 0 (batch position of expert 0)
    # rebuild the slab path pieces for expert 0 only
    Xb = a8[e0rows]
    SAb = sa[e0rows]
    Wfp0 = torch.empty((1, K, N), dtype=torch.half, device=dev)
    ext.reconstruct_batch(Wfp0, ptrs[ef:ef+1], t["bK"], t["mcg"], t["mul1"])
    W80 = (Wfp0.float() / s0[ef]).round().clamp(-128, 127).to(torch.int8)
    CgB = torch.empty((1, len(e0rows), N), dtype=torch.int32, device=dev)
    ext.igemm_batched(Xb.view(1, -1, K), W80, CgB)
    gg = (CgB.float() * SAb.view(1, -1, 1) * s0[ef]).to(torch.half)
    gg2 = gg.view(-1, N)
    ext.had_r_128_batch(gg2, gg2, None, C["svg"][ef:ef+1], torch.tensor([0], dtype=torch.int64, device=dev), len(e0rows), 1.0)
    dg = (gg2.float() - gp.float()).abs()
    print(f"gate epilogue grouped[ef] vs proto: maxabs={dg.max().item():.3e}", flush=True)

    # Full up + silu + down chain for expert ef, grouped-style vs proto
    ui = mlp.gates[ef].inner  # placeholder, replaced below
    ui = mlp.ups[ef].inner
    di = mlp.downs[ef].inner
    s0u = C["s0u"][ef, 0, 0].item()
    s0d = C["s0d"][ef, 0, 0].item()
    # proto up
    w8pu = mlp._recon_int8(ui, torch.tensor(s0u, device=dev))
    Cu0 = torch.empty((len(e0rows), ui.out_features), dtype=torch.int32, device=dev)
    ext.igemm_recon(a8e, w8pu, Cu0)
    up = (Cu0.float() * sae * torch.tensor(s0u, device=dev)).to(torch.half)
    ext.had_r_128(up, up, None, ui.svh, 1.0)
    ap = torch.empty_like(up)
    mlp.activation_fn_call(gp, up, ap, mlp.act_limit)
    # grouped up
    t2 = C["it"]["u"]
    WfpU = torch.empty((1, t2["K"], t2["N"]), dtype=torch.half, device=dev)
    ext.reconstruct_batch(WfpU, t2["ptrs"][ef:ef+1], t2["bK"], t2["mcg"], t2["mul1"])
    W8U = (WfpU.float() / C["s0u"][ef]).round().clamp(-128, 127).to(torch.int8)
    CuB = torch.empty((1, len(e0rows), t2["N"]), dtype=torch.int32, device=dev)
    ext.igemm_batched(Xb.view(1, -1, t2["K"]), W8U, CuB)
    ug = (CuB.float() * SAb.view(1, -1, 1) * C["s0u"][ef]).to(torch.half)
    ug2 = ug.view(-1, t2["N"])
    ext.had_r_128_batch(ug2, ug2, None, C["svu"][ef:ef+1], torch.tensor([0], dtype=torch.int64, device=dev), len(e0rows), 1.0)
    du = (ug2.float() - up.float()).abs()
    print(f"up epilogue grouped vs proto: maxabs={du.max().item():.3e}", flush=True)
    ag = torch.empty_like(ug2)
    mlp.activation_fn_call(gg2, ug2, ag, mlp.act_limit)
    da = (ag.float() - ap.float()).abs()
    print(f"silu grouped vs proto: maxabs={da.max().item():.3e}", flush=True)
    # proto down
    with torch.inference_mode():
        sadp = ap.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        a8dp = (ap.float() / sadp).round().clamp(-128, 127).to(torch.int8)
    w8pd = mlp._recon_int8(di, torch.tensor(s0d, device=dev))
    Cd0 = torch.empty((len(e0rows), di.out_features), dtype=torch.int32, device=dev)
    ext.igemm_recon(a8dp, w8pd, Cd0)
    dp = (Cd0.float() * sadp * torch.tensor(s0d, device=dev)).to(torch.half)
    ext.had_r_128(dp, dp, None, di.svh, 1.0)
    # grouped down
    with torch.inference_mode():
        sadg = ag.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        XdB = (ag.float() / sadg).round().clamp(-128, 127).to(torch.int8)
    t3 = C["it"]["d"]
    WfpD = torch.empty((1, t3["K"], t3["N"]), dtype=torch.half, device=dev)
    ext.reconstruct_batch(WfpD, t3["ptrs"][ef:ef+1], t3["bK"], t3["mcg"], t3["mul1"])
    W8D = (WfpD.float() / C["s0d"][ef]).round().clamp(-128, 127).to(torch.int8)
    CdB = torch.empty((1, len(e0rows), t3["N"]), dtype=torch.int32, device=dev)
    ext.igemm_batched(XdB.view(1, -1, t3["K"]), W8D, CdB)
    dg2 = (CdB.float() * sadg.view(1, -1, 1) * C["s0d"][ef]).to(torch.half)
    dg2v = dg2.view(-1, t3["N"])
    ext.had_r_128_batch(dg2v, dg2v, None, C["svd"][ef:ef+1], torch.tensor([0], dtype=torch.int64, device=dev), len(e0rows), 1.0)
    dd = (dg2v.float() - dp.float()).abs()
    print(f"down epilogue grouped vs proto: maxabs={dd.max().item():.3e}", flush=True)
print("done", flush=True)
