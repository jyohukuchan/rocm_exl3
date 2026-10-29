#!/usr/bin/env python3
"""Round 4 dot-level sweep on gfx1201: full QK^T GEMM per dtype.

C[4096, 8192] = A[4096, 256] @ B[8192, 256].T in attention tile order: each
program owns 64 query rows and streams all 256 key panels (BN=128), so every
dot reads different data (no reassociation/CSE possible) and the pattern is
exactly an attention QK^T pass minus softmax. Reports sustained TMAC/s per
dtype (fp16/int8/fp8) + AMD knob sweep + AMDGCN mix.

Usage: bench_r4_dotsweep.py [--knobs] [--bm 32] [--out JSON]
Env: HIP_VISIBLE_DEVICES isolates one GPU (R9700 numbers only).
"""
import argparse
import json
import sys

sys.path.insert(0, "/src")

import torch
import triton
import triton.language as tl

torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
print(f"device: {props.name} {props.gcnArchName} "
      f"{props.total_memory / 1e9:.1f}GB sm={props.multi_processor_count}", flush=True)
print(f"triton: {triton.__version__}", flush=True)

RQ, RK, RD = 4096, 8192, 256
BN = 128

OPS = ["v_wmma", "v_mfma", "mfma", "wmma", "ds_read", "ds_write",
       "buffer_load", "buffer_store", "flat_load", "flat_store",
       "v_dot", "v_fma", "v_cvt", "v_perm", "v_permlane",
       "s_barrier", "s_waitcnt"]


def mix(gcn):
    lines = gcn.splitlines()
    code = [l for l in lines if l.strip() and not l.strip().startswith((".", "/", ";"))]
    m = {op: sum(1 for l in code if op in l) for op in OPS}
    return {"total_lines": len(lines), "code_lines": len(code),
            "mix": {k: v for k, v in m.items() if v}}


@triton.jit
def _qk_gemm_kernel(a, b, c,
                    BM: tl.constexpr, BN: tl.constexpr, RD: tl.constexpr,
                    RK: tl.constexpr, RQ: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, RD)
    at = tl.load(a + offs_m[:, None] * RD + offs_k[None, :],
                 mask=offs_m[:, None] < RQ)
    for n0 in range(0, RK, BN):
        offs_n = n0 + tl.arange(0, BN)
        bt = tl.load(b + offs_n[None, :] * RD + offs_k[:, None])
        s = tl.dot(at, bt).to(tl.float32)
        tl.store(c + offs_m[:, None] * RK + offs_n[None, :], s,
                 mask=offs_m[:, None] < RQ)


def _macs(bm):
    return RQ * RK * RD


def run_case(dtype, bm, warps, stages, extra):
    dev = "cuda:0"
    if dtype == "fp16":
        a = (torch.randn((RQ, RD), device=dev, dtype=torch.float16) * 0.5)
        b = (torch.randn((RK, RD), device=dev, dtype=torch.float16) * 0.5)
    elif dtype == "int8":
        a = torch.randint(-4, 4, (RQ, RD), device=dev, dtype=torch.int8)
        b = torch.randint(-4, 4, (RK, RD), device=dev, dtype=torch.int8)
    else:
        a = (torch.randn((RQ, RD), device=dev, dtype=torch.float16) * 2).to(torch.float8_e4m3fn)
        b = (torch.randn((RK, RD), device=dev, dtype=torch.float16) * 2).to(torch.float8_e4m3fn)
    c = torch.empty((RQ, RK), dtype=torch.float32, device=dev)
    ker = _qk_gemm_kernel
    grid = (triton.cdiv(RQ, bm),)
    try:
        k = ker.run(a, b, c, bm, BN, RD, RK, RQ, grid=grid,
                    warmup=True, num_warps=warps, num_stages=stages, **extra)
        ker[grid](a, b, c, bm, BN, RD, RK, RQ,
                  num_warps=warps, num_stages=stages, **extra)
    except Exception as e:
        s = str(e)
        return {"error": (s[-300:] if len(s) > 300 else s).replace("\n", " | ")}
    for _ in range(2):
        ker[grid](a, b, c, bm, BN, RD, RK, RQ,
                  num_warps=warps, num_stages=stages, **extra)
    torch.cuda.synchronize()
    import time as _time
    t0 = _time.perf_counter()
    for _ in range(10):
        ker[grid](a, b, c, bm, BN, RD, RK, RQ,
                  num_warps=warps, num_stages=stages, **extra)
    torch.cuda.synchronize()
    t_wall = (_time.perf_counter() - t0) / 10
    sev, eev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(10):
        sev.record()
        ker[grid](a, b, c, bm, BN, RD, RK, RQ,
                  num_warps=warps, num_stages=stages, **extra)
        eev.record()
        eev.synchronize()
        ts.append(sev.elapsed_time(eev) * 1e-6)  # s
    ts.sort()
    t = ts[len(ts) // 2]
    out = {"med_ms": t_wall * 1e3, "event_ms": t * 1e3,
           "tmac_s": _macs(bm) / t_wall / 1e12,
           "c_absmax": float(c.abs().amax().item())}
    try:
        out["gcn"] = mix(k.asm["amdgcn"])
    except Exception as ex:
        out["gcn_error"] = str(ex)[:120]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", action="store_true")
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = {"device": f"{props.name} {props.gcnArchName}",
           "geom": f"C[{RQ},{RK}] K={RD} BM={args.bm} BN={BN}"}
    print(f"geom: {rep['geom']}", flush=True)
    base = {"num_warps": 8, "num_stages": 2}
    for dtype in ("fp16", "int8", "fp8"):
        r = run_case(dtype, args.bm, base["num_warps"], base["num_stages"], {})
        rep[f"{dtype}-base"] = r
        t = r.get("tmac_s")
        g = (r.get("gcn") or {}).get("mix", {})
        print(f"  {dtype}: " + (f"{t:.2f} TMAC/s wall={r['med_ms']:.3f}ms ev={r.get('event_ms', -1):.3f}ms "
                                f"absmax={r['c_absmax']:.1f} "
                                f"wmmma={g.get('v_wmma', 0)} cvt={g.get('v_cvt', 0)} "
                                f"lds={g.get('ds_read', 0) + g.get('ds_write', 0)} "
                                f"bar={g.get('s_barrier', 0)} lines={(r.get('gcn') or {}).get('total_lines', 0)}"
                                if t else f"FAIL {r.get('error')}"),
              flush=True)
    b = {d: rep.get(f"{d}-base", {}).get("tmac_s") for d in ("fp16", "int8", "fp8")}
    if all(b.values()):
        print(f"RATIO int8/fp16={b['int8'] / b['fp16']:.2f} fp8/fp16={b['fp8'] / b['fp16']:.2f}", flush=True)

    if args.knobs:
        variants = [
            {"kpack": 2},
            {"matrix_instr_nonkdim": 16},
            {"waves_per_eu": 2},
            {"schedule_hint": "attention"},
            {"num_stages": 3},
            {"num_stages": 1},
        ]
        for dtype in ("fp8", "int8", "fp16"):
            print(f"-- knobs {dtype} --", flush=True)
            for extra in variants:
                extra = dict(extra)
                s = extra.pop("num_stages", base["num_stages"])
                r = run_case(dtype, args.bm, base["num_warps"], s, extra)
                rep[f"{dtype}-knob-{extra}-s{s}"] = r
                t = r.get("tmac_s")
                print(f"  {extra} s={s}: " + (f"{t:.2f} TMAC/s wall={r['med_ms']:.3f}ms ev={r.get('event_ms', -1):.3f}ms" if t else f"FAIL {r.get('error')}"),
                      flush=True)
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
