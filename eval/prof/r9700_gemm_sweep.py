#!/usr/bin/env python3
"""R9700 (gfx1201, RDNA4 WMMA) vs V620 (gfx1030, RDNA2 dot4) GEMM sweep.

INT8 (int8xint8->int32) vs FP16 (fp16xfp16->fp16 out, fp32 accumulate),
GEMM-only, synthetic data. Uses torch + Triton backends ONLY so the same
script runs on gfx1201 WITHOUT the exllamav3 C++ ext (not built there).

Backends:
  fp16_torch : torch.matmul (hipBLASLt/rocBLAS, fp32 accumulate)
  int8_torch : torch._int_mm (hipBLASLt; may be unsupported -> recorded)
  fp16_triton / int8_triton : minimal tl.dot GEMM (fp32 / int32 accumulate)
  fp8_probe  : one-shape torch fp8-e4m3 matmul probe (trivial only)

Shapes:
  q30b MoE expert: gu (R,2048)x(2048,768), d (R,768)x(768,2048)
  flash-next expert: gu (R,2560)x(2560,640), d (R,640)x(640,2560)
  dense 27B: gate/up (M,5120)x(5120,17408), down (M,17408)x(17408,5120)
  grouped proxy: uniform bmm over B=8 experts at real rows/expert
    (q30b T=2048->128, 8192->512; flash T=2048->40, 8192->160)

Timing: CUDA events, warmup, interleaved fp16/int8 rounds, median+min/max.
Usage: r9700_gemm_sweep.py [--out JSON] [--quick]
Env: HIP_VISIBLE_DEVICES selects GPU. ONE GPU per process (mixing V620+R9700
in one process segfaults).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, "/src")

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception as e:  # noqa: BLE001
    HAVE_TRITON = False
    TRITON_ERR = repr(e)

GPU = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None


def contention_snapshot():
    try:
        out = subprocess.run(["rocm-smi", "--showpids"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip()[:2000]
    except Exception as e:  # noqa: BLE001
        return f"<rocm-smi unavailable: {e}>"


# ---------------- timing ----------------

def time_fn(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    n = len(ts)
    med = ts[n // 2] if n % 2 else 0.5 * (ts[n // 2 - 1] + ts[n // 2])
    return {"med_ms": med, "min_ms": ts[0], "max_ms": ts[-1],
            "iters": iters, "warmup": warmup}


def iters_for(M):
    if M <= 64:
        return 50
    if M <= 512:
        return 30
    if M <= 2048:
        return 15
    return 10


def tflops(M, K, N, ms):
    return (2 * M * K * N / ms / 1e9) if ms and ms > 0 else None


# ---------------- Triton kernels ----------------

if HAVE_TRITON:
    # NOTE: pointer-bump form (a_ptrs += BK*s_ak) miscompiles on this
    # stack (triton 3.7 + ROCm 7.2, gfx1030: multi-tile K gives garbage
    # while single-tile is exact). Recompute pointers from base each
    # iteration instead (verified exact).
    @triton.jit
    def _gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                     s_am, s_ak, s_bk, s_bn, s_cm, s_cn,
                     STORE_FP16: tl.constexpr, ACC_DTYPE: tl.constexpr,
                     OUT_DTYPE: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_am = (pid_m * BM + tl.arange(0, BM)) % M
        offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=ACC_DTYPE)
        for k in range(0, tl.cdiv(K, BK)):
            ko = k * BK
            a = tl.load(a_ptr + (offs_am[:, None] * s_am
                                 + (ko + offs_k)[None, :] * s_ak),
                        mask=(ko + offs_k)[None, :] < K, other=0.0)
            b = tl.load(b_ptr + ((ko + offs_k)[:, None] * s_bk
                                 + offs_bn[None, :] * s_bn),
                        mask=(ko + offs_k)[:, None] < K, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=OUT_DTYPE)
        offs_cm = pid_m * BM + tl.arange(0, BM)
        offs_cn = pid_n * BN + tl.arange(0, BN)
        c_ptrs = c_ptr + s_cm * offs_cm[:, None] + s_cn * offs_cn[None, :]
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if STORE_FP16:
            tl.store(c_ptrs, acc.to(tl.float16), mask=mask)
        else:
            tl.store(c_ptrs, acc.to(tl.int32), mask=mask)

    def triton_gemm(a, b, out_fp16):
        M, K = a.shape
        _, N = b.shape
        BM, BN, BK = 64, 64, 64
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        c = torch.empty((M, N), device=a.device,
                        dtype=torch.float16 if out_fp16 else torch.int32)
        _gemm_kernel[grid](a, b, c, M, N, K,
                           a.stride(0), a.stride(1),
                           b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1),
                           out_fp16,
                           tl.int32 if not out_fp16 else tl.float32,
                           tl.int32 if not out_fp16 else tl.float32,
                           BM, BN, BK, num_warps=4)
        return c

    def triton_gcn_snippet():
        """Best-effort: find compiled AMDGCN and grep WMMA/dot ISA."""
        try:
            caches = getattr(_gemm_kernel, "device_caches", None)
            if caches is None:
                return {"note": "no device_caches attr"}
            variants = []
            for table in caches.values():
                elems = table if isinstance(table, (tuple, list)) \
                    else [table]
                for e in elems:
                    items = e.items() if isinstance(e, dict) else [(None, e)]
                    for _, ker in items:
                        asm = getattr(ker, "asm", None)
                        if asm and "amdgcn" in asm:
                            gcn = asm["amdgcn"]
                            hits = sorted(set(
                                ln.strip() for ln in gcn.splitlines()
                                if ("wmma" in ln or "dot4" in ln
                                    or "v_dot" in ln or "mfma" in ln)))
                            variants.append({
                                "n_gcn_lines": len(gcn.splitlines()),
                                "n_isa_hits": len(hits),
                                "isa_hits": hits[:12],
                            })
            if not variants:
                return {"note": "no amdgcn found in device_caches"}
            return {"n_variants": len(variants),
                    "variants": variants[:6]}
        except Exception as e:  # noqa: BLE001
            return {"note": f"gcn inspect failed: {e!r}"}


# ---------------- single GEMM sweep ----------------

def bench_single(M, K, N, g, out):
    dev = "cuda:0"
    it = iters_for(M)
    a8 = torch.randint(-100, 100, (M, K), generator=g, device=dev,
                       dtype=torch.int32).to(torch.int8).contiguous()
    w8 = torch.randint(-100, 100, (K, N), generator=g, device=dev,
                       dtype=torch.int32).to(torch.int8).contiguous()
    xh = torch.randn((M, K), generator=g, device=dev,
                     dtype=torch.float16).contiguous()
    w = torch.randn((K, N), generator=g, device=dev,
                    dtype=torch.float16).contiguous()
    row = {"M": M, "K": K, "N": N}
    # fp16 torch (interleaved rounds vs int8 below)
    t_h = time_fn(lambda: torch.matmul(xh, w), it)
    row["fp16_torch_ms"] = round(t_h["med_ms"], 4)
    row["fp16_torch_min"] = round(t_h["min_ms"], 4)
    row["fp16_torch_max"] = round(t_h["max_ms"], 4)
    row["fp16_tflops"] = round(tflops(M, K, N, t_h["med_ms"]), 3)
    # int8 torch
    try:
        c32 = torch._int_mm(a8, w8)  # warmup-compile check
        torch.cuda.synchronize()
        t_i = time_fn(lambda: torch._int_mm(a8, w8), it)
        row["int8_torch_ms"] = round(t_i["med_ms"], 4)
        row["int8_torch_min"] = round(t_i["min_ms"], 4)
        row["int8_torch_max"] = round(t_i["max_ms"], 4)
        row["int8_tops"] = round(tflops(M, K, N, t_i["med_ms"]), 3)
        row["int8_fp16_ratio"] = round(t_h["med_ms"] / t_i["med_ms"], 3)
    except Exception as e:  # noqa: BLE001
        row["int8_torch_ms"] = None
        row["int8_torch_err"] = f"{type(e).__name__}: {str(e)[:160]}"
    # triton fp16/int8 (one warmup correctness spot-check per (K,N))
    if HAVE_TRITON:
        try:
            yh = triton_gemm(xh, w, True)
            torch.cuda.synchronize()
            ref = torch.matmul(xh, w)
            ok_h = torch.allclose(yh.float(), ref.float(),
                                  rtol=2e-2, atol=2e-2)
            row["triton_fp16_correct"] = bool(ok_h)
            t_th = time_fn(lambda: triton_gemm(xh, w, True), it)
            row["fp16_triton_ms"] = round(t_th["med_ms"], 4)
            row["fp16_triton_tflops"] = round(
                tflops(M, K, N, t_th["med_ms"]), 3)
        except Exception as e:  # noqa: BLE001
            row["fp16_triton_ms"] = None
            row["fp16_triton_err"] = f"{type(e).__name__}: {str(e)[:160]}"
        try:
            yi = triton_gemm(a8, w8, False)
            torch.cuda.synchronize()
            if row.get("int8_torch_ms") is not None:
                ok_i = torch.equal(yi, torch._int_mm(a8, w8))
            else:
                ok_i = bool((yi.float() == (a8.float() @ w8.float())).all())
            row["triton_int8_correct"] = bool(ok_i)
            t_ti = time_fn(lambda: triton_gemm(a8, w8, False), it)
            row["int8_triton_ms"] = round(t_ti["med_ms"], 4)
            row["int8_triton_tops"] = round(
                tflops(M, K, N, t_ti["med_ms"]), 3)
            if row.get("fp16_triton_ms"):
                row["triton_ratio"] = round(
                    t_th["med_ms"] / t_ti["med_ms"], 3)
        except Exception as e:  # noqa: BLE001
            row["int8_triton_ms"] = None
            row["int8_triton_err"] = f"{type(e).__name__}: {str(e)[:160]}"
    else:
        row["triton_note"] = TRITON_ERR
    flops = 2 * M * K * N
    row["gflop"] = round(flops / 1e9, 3)
    out.append(row)
    del a8, w8, xh, w
    torch.cuda.empty_cache()
    return row


def bench_grouped(r, K, N, g, B=8):
    """Uniform B-expert proxy: fp16 bmm vs int8 per-expert loop."""
    dev = "cuda:0"
    it = iters_for(r)
    Xh = torch.randn((B, r, K), generator=g, device=dev,
                     dtype=torch.float16).contiguous()
    W = torch.randn((B, K, N), generator=g, device=dev,
                    dtype=torch.float16).contiguous()
    A8 = torch.randint(-100, 100, (B, r, K), generator=g, device=dev,
                       dtype=torch.int32).to(torch.int8).contiguous()
    W8 = torch.randint(-100, 100, (B, K, N), generator=g, device=dev,
                       dtype=torch.int32).to(torch.int8).contiguous()
    row = {"rows_per_expert": r, "K": K, "N": N, "B": B}
    t_h = time_fn(lambda: torch.bmm(Xh, W), it)
    row["fp16_bmm_ms"] = round(t_h["med_ms"], 4)
    row["fp16_tflops"] = round(tflops(B * r, K, N, t_h["med_ms"]), 3)
    # int8: per-expert _int_mm loop (torch has no int8 bmm)
    try:
        for b in range(B):
            torch._int_mm(A8[b], W8[b])
        torch.cuda.synchronize()

        def loop_i():
            for b in range(B):
                torch._int_mm(A8[b], W8[b])

        t_i = time_fn(loop_i, it)
        row["int8_loop_ms"] = round(t_i["med_ms"], 4)
        row["int8_tops"] = round(tflops(B * r, K, N, t_i["med_ms"]), 3)
        row["ratio"] = round(t_h["med_ms"] / t_i["med_ms"], 3)
        row["int8_backend"] = "torch._int_mm loop"
    except Exception as e:  # noqa: BLE001
        row["int8_loop_ms"] = None
        row["int8_loop_err"] = f"{type(e).__name__}: {str(e)[:160]}"
        if HAVE_TRITON:
            try:
                def loop_t():
                    for b in range(B):
                        triton_gemm(A8[b], W8[b], False)

                loop_t()
                torch.cuda.synchronize()
                t_i = time_fn(loop_t, it)
                row["int8_loop_ms"] = round(t_i["med_ms"], 4)
                row["int8_tops"] = round(
                    tflops(B * r, K, N, t_i["med_ms"]), 3)
                row["ratio"] = round(t_h["med_ms"] / t_i["med_ms"], 3)
                row["int8_backend"] = "triton int8 loop"
            except Exception as e2:  # noqa: BLE001
                row["int8_triton_err"] = f"{type(e2).__name__}: {str(e2)[:160]}"
    del Xh, W, A8, W8
    torch.cuda.empty_cache()
    return row


def fp8_probe(g):
    M, K, N = 512, 5120, 17408
    dev = "cuda:0"
    row = {"M": M, "K": K, "N": N}
    try:
        a = torch.randn((M, K), device=dev, dtype=torch.float16).to(
            torch.float8_e4m3fn).contiguous()
        b = torch.randn((K, N), device=dev, dtype=torch.float16).to(
            torch.float8_e4m3fn).contiguous()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        t = time_fn(lambda: torch.matmul(a, b), 10)
        row["fp8_ms"] = round(t["med_ms"], 4)
        row["fp8_tflops"] = round(tflops(M, K, N, t["med_ms"]), 3)
        row["out_dtype"] = str(c.dtype)
    except Exception as e:  # noqa: BLE001
        row["fp8_ms"] = None
        row["fp8_err"] = f"{type(e).__name__}: {str(e)[:200]}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="smoke: subset of shapes")
    args = ap.parse_args()
    torch.cuda.set_device(0)
    g = torch.Generator(device="cuda:0").manual_seed(7)
    res = {
        "gpu": {"name": GPU.name if GPU else None,
                "arch": getattr(GPU, "gcnArchName", None) if GPU else None,
                "total_memory_MB": (GPU.total_memory // (1024 * 1024)
                                    if GPU else None)},
        "torch": torch.__version__,
        "triton": triton.__version__ if HAVE_TRITON else None,
        "hip_visible": os.environ.get("HIP_VISIBLE_DEVICES"),
        "contention_before": contention_snapshot(),
    }
    print(f"gpu={res['gpu']} torch={res['torch']} "
          f"triton={res['triton']} HIP={res['hip_visible']}", flush=True)
    print("KFD before:\n" + res["contention_before"], flush=True)

    ROWS = [32, 128, 1024] if args.quick else \
        [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    MODELS = {"q30b": {"H": 2048, "I": 768},
              "flash-next": {"H": 2560, "I": 640}}
    res["single"] = {}
    with torch.inference_mode():
        for mname, m in MODELS.items():
            H, I = m["H"], m["I"]
            for tag, K, N in (("gu", H, I), ("d", I, H)):
                key = f"{mname}.{tag}"
                rows = []
                for R in ROWS:
                    row = bench_single(R, K, N, g, rows)
                    fi = row.get("fp16_torch_ms")
                    ii = row.get("int8_torch_ms")
                    su = row.get("int8_fp16_ratio")
                    tr = row.get("triton_ratio")
                    print(f"{key} R={R:5d} fp16={fi}ms "
                          f"int8={ii}ms su={su}x triton_su={tr}x "
                          f"fp16={row.get('fp16_tflops')}TF "
                          f"int8={row.get('int8_tops')}TOPS",
                          flush=True)
                res["single"][key] = rows
        # dense 27B
        DM = [512, 2048] if args.quick else [512, 2048, 8192]
        res["dense27b"] = {}
        for tag, K, N in (("gate_up", 5120, 17408),
                          ("down", 17408, 5120)):
            rows = []
            for M in DM:
                row = bench_single(M, K, N, g, rows)
                print(f"dense27b.{tag} M={M:5d} "
                      f"fp16={row.get('fp16_torch_ms')}ms "
                      f"int8={row.get('int8_torch_ms')}ms "
                      f"su={row.get('int8_fp16_ratio')}x "
                      f"fp16={row.get('fp16_tflops')}TF "
                      f"int8={row.get('int8_tops')}TOPS",
                      flush=True)
            res["dense27b"][tag] = rows
        # grouped proxy at real rows/expert (uniform, B=8)
        res["grouped"] = []
        G = [("q30b.gu", 128, 2048, 768), ("q30b.gu", 512, 2048, 768),
             ("flash.gu", 40, 2560, 640), ("flash.gu", 160, 2560, 640)]
        if args.quick:
            G = G[:1]
        for name, r, K, N in G:
            row = bench_grouped(r, K, N, g)
            row["name"] = name
            res["grouped"].append(row)
            print(f"grouped {name} r={r} B=8 fp16={row.get('fp16_bmm_ms')}ms "
                  f"int8={row.get('int8_loop_ms')}ms "
                  f"ratio={row.get('ratio')}x "
                  f"({row.get('int8_backend')})", flush=True)
        # triton ISA check (after kernels ran)
        if HAVE_TRITON:
            res["triton_gcn"] = triton_gcn_snippet()
            print("triton_gcn: " + json.dumps(res["triton_gcn"])[:1500],
                  flush=True)
        else:
            res["triton_gcn"] = {"note": TRITON_ERR}
        res["fp8_probe"] = fp8_probe(g)
        print("fp8_probe: " + json.dumps(res["fp8_probe"]), flush=True)
    res["contention_after"] = contention_snapshot()
    print("KFD after:\n" + res["contention_after"], flush=True)
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
