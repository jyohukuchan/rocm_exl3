#!/usr/bin/env python3
"""A: one-MoE-layer forward breakdown, fp16 path vs int8 grouped path (part 1)."""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/src")

MODE = sys.argv[2] if len(sys.argv) > 2 else "fp16"
if MODE == "int8":
    os.environ["EXL3_INT8_PREFILL"] = "1"
    os.environ["EXL3_INT8_MOE_GROUPED"] = "1"
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

import torch

MODEL_DIR = sys.argv[1]
TLIST = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [512, 2048, 8192]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
LAYER_IDX = int(sys.argv[5]) if len(sys.argv) > 5 else 8
SPLIT = None
if os.environ.get("MOE_BD_SPLIT"):
    SPLIT = [float(x) for x in os.environ["MOE_BD_SPLIT"].split(",")]

torch.cuda.set_device(0)
from exllamav3.model import Config, Model
from exllamav3.modules import block_sparse_mlp as M

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
if SPLIT:
    model.load(use_per_device=SPLIT, max_chunk_size=max(TLIST))
else:
    model.load(device=torch.device("cuda:0"))
print(f"loaded {MODEL_DIR} mode={MODE} split={SPLIT}", flush=True)


def _sync_all():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)

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
print(f"n_moe_layers={len(mlps)}", flush=True)
E = mlps[0].num_local_experts or mlps[0].num_experts
H = mlps[0].expert_size
TOPK = mlps[0].num_experts_per_tok
print(f"E={E} H={H} topk={TOPK}", flush=True)

if MODE == "int8":
    def _fake_state(self):
        gi = self.gates[0].inner
        n = len(self.gates)
        s = [0.02] * n
        return (gi.suh, s, s, s)

    M.BlockSparseMLP._int8_moe_state = _fake_state
    print("fake s0 state installed (TIMING ONLY, numerics invalid)", flush=True)

LAYER_MS = {}
ROUTER_MS = {}
_orig_fwd = M.BlockSparseMLP.forward


def _fwd(self, x, params, out_dtype=None):
    if not getattr(self, "_rt_wrapped", False):
        self._rt_wrapped = True
        rf = self.routing_fn

        def wrapper(bsz, cfg, z, p):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            r = rf(bsz, cfg, z, p)
            b.record()
            ROUTER_MS.setdefault(self.key, []).append((a, b))
            return r

        self.routing_fn = wrapper
    if SPLIT:
        _sync_all()
        t0 = time.perf_counter()
        r = _orig_fwd(self, x, params, out_dtype)
        _sync_all()
        LAYER_MS.setdefault(self.key, []).append((time.perf_counter() - t0) * 1000)
        return r
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    r = _orig_fwd(self, x, params, out_dtype)
    e1.record()
    LAYER_MS.setdefault(self.key, []).append((e0, e1))
    return r


def _summarize():
    def _tot(v):
        if v and isinstance(v[0], (int, float)):
            return round(sum(v), 3)
        return round(sum(a.elapsed_time(b) for a, b in v), 3)
    lm = {k: _tot(v) for k, v in LAYER_MS.items()}
    rm = {k: _tot(v) for k, v in ROUTER_MS.items()}
    return lm, rm


if MODE in ("fp16", "int8"):
    M.BlockSparseMLP.forward = _fwd
    res = {}
    NREPS = int(os.environ.get("MOE_BD_REPS", "3"))
    for T in TLIST:
        per_rep_layer, per_rep_router, walls = [], [], []
        g = torch.Generator(device="cuda:0").manual_seed(99)
        for r in range(NREPS + 1):
            LAYER_MS.clear()
            ROUTER_MS.clear()
            ids = torch.randint(0, config.vocab_size, (1, T), generator=g,
                                device="cuda:0")
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model.forward(ids, {})
            _sync_all()
            walls.append(time.perf_counter() - t0)
            if r == 0:
                print(f"T={T} warmup: layers={len(LAYER_MS)} wall={walls[-1]:.2f}s",
                      flush=True)
                continue
            lm, rm = _summarize()
            per_rep_layer.append(sum(lm.values()))
            per_rep_router.append(sum(rm.values()))
        n = len(mlps)
        res[str(T)] = {
            "wall_s": round(statistics.median(walls[1:]), 3),
            "moe_total_ms": round(statistics.median(per_rep_layer), 2),
            "moe_per_layer_ms": round(statistics.median(per_rep_layer) / n, 3),
            "router_total_ms": round(statistics.median(per_rep_router), 2),
            "router_per_layer_ms": round(statistics.median(per_rep_router) / n, 3),
            "n_layers": n,
        }
        print(f"T={T}: " + json.dumps(res[str(T)]), flush=True)
    print(json.dumps(res, indent=1))
    if OUT:
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", OUT, flush=True)

if MODE == "stages":
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.rocm_py.moe_int8 import (
        quant_w8_slab, dquant_rowwise, gu_epilog_had, d_epilog_had)
    mlp = mlps[min(LAYER_IDX, len(mlps) - 1)]
    print("stage mlp:", mlp.key, flush=True)
    dev = mlp.device
    torch.cuda.set_device(dev)

    def _fs(self):
        gi = self.gates[0].inner
        n = len(self.gates)
        s = [0.02] * n
        return (gi.suh, s, s, s)

    M.BlockSparseMLP._int8_moe_state = _fs
    C = mlp._int8_moe_grouped_cache()
    assert C is not None, "grouped cache build failed"
    print(f"cache E={C['E']} Kg={C['it']['g']['K']} Ng={C['it']['g']['N']} "
          f"Kd={C['it']['d']['K']} Nd={C['it']['d']['N']}", flush=True)

    def med(fn, iters=20):
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

    g = torch.Generator(device="cuda:0").manual_seed(5)
    res = {}
    cap = int(os.environ.get("EXL3_INT8_MOE_ROW_CAP", "160"))
    with torch.inference_mode():
        for T in TLIST:
            r = {}
            y = torch.randn((T, H), generator=g, device=dev, dtype=torch.half)
            xh = torch.empty_like(y)
            su = C["su"]
            r["act_had_ms"] = med(lambda: ext.had_r_128(y, xh, su, None, 1.0))
            sa = xh.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
            r["act_quant_torch_ms"] = med(
                lambda: (xh.float() / sa).round().clamp(-128, 127).to(torch.int8))
            U = T * TOPK // E
            nsplit = max(1, (U + cap - 1) // cap)
            cmax = (U + nsplit - 1) // nsplit
            B = E * nsplit
            r["uniform"] = {"rows_per_expert": U, "B": B, "cmax": cmax,
                            "assign": T * TOPK, "padded": B * cmax}
            sel = torch.randint(0, E, (T, TOPK), generator=g, device=dev)
            flat = sel.reshape(-1)
            r["sort_ms"] = med(lambda: flat.argsort())
            r["bincount_ms"] = med(lambda: torch.bincount(flat, minlength=E + 1))
            a8e = torch.randint(-100, 100, (T * TOPK, H), generator=g, device=dev,
                                dtype=torch.int32).to(torch.int8)
            tok = torch.randint(0, T, (B * cmax,), device=dev)
            r["gather_ms"] = med(lambda: a8e[tok].view(B, cmax, -1))
            for tag in ("g", "u", "d"):
                t = C["it"][tag]
                K, N = t["K"], t["N"]
                e_ids = torch.arange(E, device=dev).repeat_interleave(nsplit)
                ptrs = t["ptrs"][e_ids].contiguous()
                s0 = torch.full((B,), 0.02, dtype=torch.float32, device=dev)
                Wfp = torch.empty((B, K, N), dtype=torch.half, device=dev)
                W8 = torch.empty((B, K, N), dtype=torch.int8, device=dev)
                r[f"recon_{tag}_ms"] = med(
                    lambda: ext.reconstruct_batch(Wfp, ptrs, t["bK"], t["mcg"], t["mul1"]))
                ext.reconstruct_batch(Wfp, ptrs, t["bK"], t["mcg"], t["mul1"])
                torch.cuda.synchronize()
                r[f"wquant_{tag}_ms"] = med(lambda: quant_w8_slab(Wfp, s0, W8))
                Xin = torch.randint(-100, 100, (B, cmax, K), generator=g, device=dev,
                                    dtype=torch.int32).to(torch.int8)
                C32 = torch.empty((B, cmax, N), dtype=torch.int32, device=dev)
                r[f"igemm_{tag}_ms"] = med(lambda: ext.igemm_batched(Xin, W8, C32))
                Xh = torch.randn((B, cmax, K), generator=g, device=dev,
                                 dtype=torch.float16)
                Wh = torch.randn((B, K, N), generator=g, device=dev, dtype=torch.float16)
                Yh = torch.empty((B, cmax, N), dtype=torch.float16, device=dev)
                r[f"hgemm_{tag}_ms"] = med(
                    lambda: ext.hgemm_exact_group(Xh, Wh, Yh, [cmax] * B))
                y2 = torch.empty((B * cmax, N), dtype=torch.half, device=dev)
                r[f"had_post_{tag}_ms"] = med(
                    lambda: ext.had_r_128(Yh.view(B * cmax, N), y2, None,
                                          mlp.downs[0].inner.svh, 1.0))
            # fused INT8 epilogues at gu/down shapes
            Ng = C["it"]["g"]["N"]
            Nd = C["it"]["d"]["N"]
            Cg = torch.empty((B * cmax, Ng), dtype=torch.int32, device=dev)
            Cu = torch.empty_like(Cg)
            SA = torch.empty((B * cmax, 1), dtype=torch.float, device=dev).fill_(0.02)
            gg = torch.empty((B * cmax, Ng), dtype=torch.half, device=dev)
            uu = torch.empty_like(gg)
            act_ids = e_ids.to(torch.int64)
            s0e = torch.full((B,), 0.02, dtype=torch.float32, device=dev)
            Etab = 128 if E <= 128 else E
            svg = torch.randn((Etab, Ng), dtype=torch.half, device=dev)
            r["epigu_ms"] = med(lambda: gu_epilog_had(Cg, Cu, SA, s0e, s0e, svg, svg,
                                                      act_ids, cmax, gg, uu))
            aa = torch.randn((B * cmax, Ng), generator=g, device=dev, dtype=torch.half)
            aout = torch.empty_like(aa)
            r["silu_ms"] = med(
                lambda: mlp.activation_fn_call(gg, uu, aout, mlp.act_limit))
            ss = torch.empty((B * cmax, 1), dtype=torch.float, device=dev)
            XX = torch.empty((B * cmax, C["it"]["d"]["K"]),
                             dtype=torch.int8, device=dev)
            r["dquant_ms"] = med(lambda: dquant_rowwise(aa, ss, XX))
            Cd = torch.empty((B * cmax, Nd), dtype=torch.int32, device=dev)
            s0d = torch.full((B,), 0.02, dtype=torch.float32, device=dev)
            svd = torch.randn((Etab, Nd), dtype=torch.half, device=dev)
            dd = torch.empty((B * cmax, Nd), dtype=torch.half, device=dev)
            r["epid_ms"] = med(lambda: d_epilog_had(Cd, ss, s0d, svd, act_ids,
                                                    cmax, dd))
            # scatter/combine
            out = torch.zeros((T, H), dtype=torch.float, device=dev)
            df = torch.randn((B * cmax, Nd), device=dev, dtype=torch.float)
            ww = torch.rand((B * cmax,), device=dev)
            def _acc():
                for b in range(B):
                    r0 = b * cmax
                    out.index_add_(0, tok[r0:r0 + cmax], df[r0:r0 + cmax])
            r["indexadd_ms"] = med(_acc)
            print(f"T={T}: " + json.dumps(r), flush=True)
            res[str(T)] = r
    print(json.dumps(res, indent=1))
    if OUT:
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", OUT, flush=True)

