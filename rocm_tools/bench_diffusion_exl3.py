#!/usr/bin/env python3
"""Measure EXL3 reconstruction amortization, not diffusion model quality.

Uses synthetic packed 3INST weights and FP16 activations. No model download or
quantizer is involved. Run on an idle GPU with a prebuilt extension.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3


def measure(fn, repeats, iterations):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def packed(k, n, bits, device):
    t = torch.randint(0, 65536, (k // 16, n // 16, 16 * bits),
                      device=device, dtype=torch.int32).short()
    su = torch.sign(torch.randn(k, device=device)).half()
    sv = torch.sign(torch.randn(n, device=device)).half()
    return t, su, sv


def validate(bits, device):
    # Independent FP32 Sylvester reference for both weight-basis transforms.
    k, n = 256, 384
    t, su, sv = packed(k, n, bits, device)
    wh = torch.empty(k, n, device=device, dtype=torch.float16)
    ext.reconstruct(wh, t, bits, False, False)
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= 128 ** 0.5
    ref = torch.einsum("ij,bjn->bin", h, wh.float().view(k // 128, 128, n)).reshape(k, n)
    ref = torch.einsum("kbi,ij->kbj", ref.view(k, n // 128, 128), h).reshape(k, n)
    ref *= su.float()[:, None] * sv.float()[None, :]
    fused = torch.empty_like(wh)
    ext.reconstruct_had_slice(fused, t, su, sv, bits, False, False, 0)
    err = ((fused.float() - ref).abs().max() / ref.abs().max()).item()
    if not torch.isfinite(fused).all() or err > 0.002:
        raise RuntimeError(f"Reconstruction correctness failed: {err}")
    return {"weight_max_error_over_ref_max": err, "limit": 0.002}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--bits", type=int, choices=range(2, 9), default=4)
    ap.add_argument("--rows", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    ap.add_argument("--shapes", nargs="+", default=["4096x4096", "4096x12288"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if min(args.rows + [args.repeats, args.iterations]) <= 0:
        ap.error("rows, repeats and iterations must be positive")
    shapes = [tuple(map(int, s.split("x"))) for s in args.shapes]
    if any(len(s) != 2 or any(d <= 0 or d % 128 for d in s) for s in shapes):
        ap.error("shapes must be KxN with positive multiples of 128")
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261018)
    device = torch.device("cuda", args.device)
    props = torch.cuda.get_device_properties(device)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic EXL3 Linear; no image/video quality or end-to-end measurement",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "extension_path": ext.__file__,
        "extension_sha256": hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
        "extension_build_commit": "unknown; pre-existing binary, not rebuilt for this probe",
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": props.name, "arch": getattr(props, "gcnArchName", None),
        "initial_free_total_bytes": list(torch.cuda.mem_get_info(device)),
        "environment": {k: v for k, v in os.environ.items() if k.startswith(
            ("EXL3_", "ROCM", "HIP_VISIBLE", "CUDA_VISIBLE", "LD_LIBRARY", "LD_PRELOAD"))},
        "parameters": {**vars(args), "output": str(args.output)},
        "method": "HIP events, eager launches, 4 warmups, median of repeated groups; hot weights; no graphs",
        "correctness": validate(args.bits, device), "results": [],
    }
    for k, n in shapes:
        t, su, sv = packed(k, n, args.bits, device)
        linear = LinearEXL3(None, k, n, suh=su, svh=sv, trellis=t)
        w = torch.empty(k, n, device=device, dtype=torch.float16)
        wh = torch.empty_like(w)

        def reconstruct():
            ext.reconstruct(wh, t, args.bits, False, False)

        def reconstruct_fused():
            ext.reconstruct_had_slice(w, t, su, sv, args.bits, False, False, 0)

        reconstruct()
        reconstruct_fused()
        rec = measure(reconstruct, args.repeats, args.iterations)
        fused_rec = measure(reconstruct_fused, args.repeats, args.iterations)
        for m in args.rows:
            x = torch.randn(m, k, device=device, dtype=torch.float16) * 0.01
            xh = torch.empty_like(x)
            y = torch.empty(m, n, device=device, dtype=torch.float16)

            def cached_gemm():
                ext.hgemm_recon(x, w, y)

            def fused_path():
                reconstruct_fused()
                cached_gemm()

            def separate_path():
                ext.had_r_128(x, xh, su, None, 1.0)
                reconstruct()
                ext.hgemm_recon(xh, wh, y)
                ext.had_r_128(y, y, None, sv, 1.0)

            def forward():
                return linear.forward(x, {})

            def torch_gemm():
                torch.mm(x, w, out=y)

            cached_gemm()
            ref = y.clone()
            actual = forward()
            rel = ((actual.float() - ref.float()).norm() / ref.float().norm()).item()
            if not torch.isfinite(actual).all() or rel > 0.005:
                raise RuntimeError(f"Forward correctness failed for {(m,k,n)}: {rel}")
            del actual, ref
            timings = {name: measure(fn, args.repeats, args.iterations) for name, fn in [
                ("cached_fp16_gemm", cached_gemm), ("torch_fp16_mm", torch_gemm),
                ("fused_reconstruct_gemm", fused_path), ("separate_hadamard_path", separate_path),
                ("linear_forward", forward),
            ]}
            # Baseline buffers above remain allocated: report *incremental* forward
            # allocation only, never mislabel this as whole-model peak VRAM.
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
            out = forward()
            torch.cuda.synchronize()
            incremental = torch.cuda.max_memory_allocated(device) - before
            del out
            row = {
                "m": m, "k": k, "n": n, "bits": args.bits,
                "packed_bytes": t.numel() * t.element_size(),
                "fp16_weight_bytes": w.numel() * w.element_size(),
                "incremental_forward_peak_bytes": incremental,
                "forward_relative_l2_error": rel,
                "plain_reconstruct": rec, "fused_reconstruct": fused_rec,
                "timings": timings,
                "forward_over_cached": timings["linear_forward"]["median_ms"] / timings["cached_fp16_gemm"]["median_ms"],
            }
            report["results"].append(row)
            print(json.dumps({"shape": [m,k,n], "forward_over_cached": row["forward_over_cached"],
                              "ms": {name: value["median_ms"] for name, value in timings.items()}}), flush=True)
            del x, xh, y
        del t, su, sv, linear, w, wh
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
