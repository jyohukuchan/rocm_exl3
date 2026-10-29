#!/usr/bin/env python3
"""D: real-text routing skew + grouped GEMM timing (real vs uniform).

1. Tokenize a long real text, run fp16 forward, capture selected_experts/layer.
2. Stats: per-expert counts max/mean/CV, cmax/B/padded with ROW_CAP splitting.
3. Time grouped GEMMs (igemm_batched int8 vs hgemm_batched fp16-padded and
   hgemm_exact_group fp16-exact) at the REAL config vs UNIFORM config (same T),
   gate/up and down separately.
4. Time the full grouped INT8 forward (fake s0, timing only) with real sel vs
   uniform sel (same y, uniform weights) on one middle layer.

Usage: moe_breakdown_routing.py MODEL_DIR T [OUT.json] [TEXT]
Env: HIP_VISIBLE_DEVICES.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ["EXL3_INT8_PREFILL"] = "1"
os.environ["EXL3_INT8_MOE_GROUPED"] = "1"
import torch

MODEL_DIR = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
OUT = sys.argv[3] if len(sys.argv) > 3 else None
TEXT = (sys.argv[4] if len(sys.argv) > 4
        else "/src/eval/eval_texts/pride_prejudice_mod.txt")

torch.cuda.set_device(0)
from exllamav3.model import Config, Model
from exllamav3.modules import block_sparse_mlp as M
from exllamav3.ext import exllamav3_ext as ext

SPLIT = None
if os.environ.get("MOE_BD_SPLIT"):
    SPLIT = [float(x) for x in os.environ["MOE_BD_SPLIT"].split(",")]

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
if SPLIT:
    model.load(use_per_device=SPLIT, max_chunk_size=T)
else:
    model.load(device=torch.device("cuda:0"))
print(f"loaded {MODEL_DIR} split={SPLIT}", flush=True)

mlps = []


def _walk(m, seen):
    if type(m).__name__ == "BlockSparseMLP":
        mlps.append(m)
        return
    try:
        attrs = vars(m).values()
    except TypeError:
        return
    for a in attrs:
        if a is None or a is m or id(a) in seen:
            continue
        if "exllamav3" in type(a).__module__:
            seen.add(id(a))
            _walk(a, seen)
        elif isinstance(a, (list, tuple)):
            for x in a:
                if "exllamav3" in type(x).__module__ and id(x) not in seen:
                    seen.add(id(x))
                    _walk(x, seen)


_walk(model, set())
E = mlps[0].num_local_experts or mlps[0].num_experts
TOPK = mlps[0].num_experts_per_tok
H = mlps[0].expert_size
print(f"layers={len(mlps)} E={E} topk={TOPK} H={H}", flush=True)

from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
text = open(TEXT).read()
ids = tk(text, return_tensors="pt")["input_ids"][0]
print(f"text chars={len(text)} tokens={ids.numel()}", flush=True)
OFF = int(os.environ.get("MOE_BD_OFF", "0"))
if ids.numel() < OFF + T:
    reps = (OFF + T + ids.numel() - 1) // ids.numel()
    ids = ids.repeat(reps)
ids = ids[OFF:OFF + T]
ids = ids.unsqueeze(0).to("cuda:0")

SEL = {}


def _wrap(self):
    rf = self.routing_fn

    def wrapper(bsz, cfg, z, p):
        sel, w = rf(bsz, cfg, z, p)
        SEL[self.key] = (sel.detach().to("cpu"), w.detach().to("cpu"))
        return sel, w

    self.routing_fn = wrapper


for m in mlps:
    _wrap(m)

with torch.inference_mode():
    _ = model.forward(ids, {})
    torch.cuda.synchronize()
    print("warmup done, recapturing", flush=True)
    SEL.clear()
    _ = model.forward(ids, {})
    torch.cuda.synchronize()
print(f"captured {len(SEL)} layers", flush=True)

cap = int(os.environ.get("EXL3_INT8_MOE_ROW_CAP", "160"))


def stats_for(counts, E_, cap_):
    import math
    c = counts.float()
    mean = c.mean().item()
    std = c.std().item()
    mx = c.max().item()
    cv = std / mean if mean else 0.0
    nsplit = [(math.ceil(int(x) / cap_) if int(x) > 0 else 0)
              for x in counts.tolist()]
    B = sum(nsplit)
    cmax = 0
    for x, s in zip(counts.tolist(), nsplit):
        if s:
            cmax = max(cmax, (int(x) + s - 1) // s)
    padded = B * cmax
    return {"max": mx, "mean": round(mean, 2), "cv": round(cv, 3),
            "B": B, "cmax": cmax, "padded": padded,
            "assign": int(counts.sum().item())}


per_layer = {}
for k, (sel, w) in SEL.items():
    t, kk = sel.shape
    flat = sel.reshape(-1)
    counts = torch.bincount(flat, minlength=E)
    per_layer[k] = {"T": t, **stats_for(counts, E, cap),
                    "counts": counts.tolist()}
uni_counts = torch.full((E,), (T * TOPK) // E, dtype=torch.long)
uni = {"T": T, **stats_for(uni_counts, E, cap)}
print("UNIFORM: " + json.dumps(uni), flush=True)
agg_keys = ["max", "cv", "B", "cmax", "padded"]
agg = {}
for ak in agg_keys:
    vs = [v[ak] for v in per_layer.values()]
    agg[ak] = {"median": round(statistics.median(vs), 3),
               "mean": round(statistics.mean(vs), 3),
               "max": max(vs)}
print("REAL(layers agg): " + json.dumps(agg), flush=True)
# show one mid layer histogram summary
midk = sorted(per_layer)[len(per_layer) // 2]
print(f"mid {midk}: " + json.dumps({k: per_layer[midk][k] for k in agg_keys}),
      flush=True)
# skew sanity: histogram of mid layer + random-ids control
mc = torch.tensor(per_layer[midk]["counts"])
qs = torch.quantile(mc.float(), torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
print(f"mid counts quantiles 0/50/90/99/100: {[round(float(x),1) for x in qs]} "
      f"zeros={(mc == 0).sum().item()}", flush=True)
_gr = torch.Generator(device="cpu").manual_seed(0)
ctr = torch.randint(0, E, (T * TOPK,)).bincount(minlength=E)
cc = ctr.float()
print(f"RANDOM-control: max={ctr.max().item()} mean={cc.mean().item():.1f} "
      f"cv={cc.std().item()/cc.mean().item():.3f}", flush=True)


def med(fn, iters=30):
    for _ in range(5):
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
    return round(statistics.median(ts), 4)


mid = mlps[len(mlps) // 2]
dev = mid.device
torch.cuda.set_device(dev)
g = torch.Generator(device=dev).manual_seed(9)
Kg, Ng = mid.gates[0].inner.in_features, mid.gates[0].inner.out_features
Kd, Nd = mid.downs[0].inner.in_features, mid.downs[0].inner.out_features
print(f"shapes gu=({Kg},{Ng}) d=({Kd},{Nd})", flush=True)

gemmres = {}
NOGEMM = os.environ.get("MOE_BD_NOGEMM", "0") == "1"
with torch.inference_mode():
    for cfgname, cfg in ((("uniform", uni), ("real", per_layer[midk]))
                          if not NOGEMM else ()):
        B, cmax = cfg["B"], cfg["cmax"]
        if cfgname == "real":
            counts = torch.tensor(per_layer[midk]["counts"][:E], device=dev)
            # expand per-entry counts with cap splitting
            import math
            entry_counts = []
            for x in counts.tolist():
                if x == 0:
                    continue
                s = max(1, math.ceil(x / cap))
                base, rem = divmod(x, s)
                entry_counts += [base + (1 if i < rem else 0) for i in range(s)]
            assert len(entry_counts) == B, (len(entry_counts), B)
        else:
            entry_counts = [cmax] * B
        r = {"B": B, "cmax": cmax, "padded": B * cmax,
             "assign": T * TOPK}
        for tag, K, N in (("gu", Kg, Ng), ("d", Kd, Nd)):
            A8 = torch.randint(-100, 100, (B, cmax, K), generator=g, device=dev,
                               dtype=torch.int32).to(torch.int8)
            W8 = torch.randint(-100, 100, (B, K, N), generator=g, device=dev,
                               dtype=torch.int32).to(torch.int8)
            Cb = torch.empty((B, cmax, N), dtype=torch.int32, device=dev)
            Xh = torch.randn((B, cmax, K), generator=g, device=dev,
                             dtype=torch.float16)
            Wh = torch.randn((B, K, N), generator=g, device=dev,
                             dtype=torch.float16)
            Yh = torch.empty((B, cmax, N), dtype=torch.float16, device=dev)
            r[f"igemm_{tag}_ms"] = med(lambda: ext.igemm_batched(A8, W8, Cb))
            r[f"hgemm_batched_{tag}_ms"] = med(
                lambda: ext.hgemm_batched(Xh, Wh, Yh))
            r[f"hgemm_exact_{tag}_ms"] = med(
                lambda: ext.hgemm_exact_group(Xh, Wh, Yh, entry_counts))
        print(f"{cfgname}: " + json.dumps(r), flush=True)
        gemmres[cfgname] = r

# full grouped INT8 forward, real sel vs uniform sel (fake s0, timing only)
if NOGEMM:
    t_real, t_uni = None, None
else:
    def _fake_state(self):
        gi = self.gates[0].inner
        n = len(self.gates)
        s = [0.02] * n
        return (gi.suh, s, s, s)

    M.BlockSparseMLP._int8_moe_state = _fake_state
    with torch.inference_mode():
        y = torch.randn((T, H), device=dev, dtype=torch.half)
        sel_real, _ = SEL[mid.key]
        sel_real = sel_real.to(dev)
        w_uni = torch.full((T, TOPK), 1.0 / TOPK, device=dev, dtype=torch.half)
        # uniform sel: round robin
        base = torch.arange(T * TOPK, device=dev) % E
        sel_uni = base.view(T, TOPK)
        t_real = med(lambda: mid._forward_int8_moe_grouped(y, T, sel_real, w_uni, {}),
                     iters=10)
        t_uni = med(lambda: mid._forward_int8_moe_grouped(y, T, sel_uni, w_uni, {}),
                    iters=10)
        print(f"grouped-fwd mid layer: real_sel={t_real}ms uniform_sel={t_uni}ms "
              f"ratio={t_real / t_uni:.3f}x", flush=True)

res = {"T": T, "text": TEXT, "uniform": uni, "layers_agg": agg,
       "mid_key": midk, "mid_stats": per_layer[midk],
       "gemm": gemmres,
       "grouped_fwd_ms": {"real": t_real, "uniform": t_uni}}
print(json.dumps({k: v for k, v in res.items() if k != "mid_stats"}, indent=1))
if OUT:
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT, flush=True)
