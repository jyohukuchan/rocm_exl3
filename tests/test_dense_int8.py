#!/usr/bin/env python3
"""Unit tests for dense INT8 Triton helpers (C2): bitwise contracts for
quant kernels, tolerance contract for the fp32 epilogue. Needs GPU.
Run: HIP_VISIBLE_DEVICES=1 python3 tests/test_dense_int8.py"""
import os
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")
import torch

torch.cuda.set_device(0)
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.rocm_py.dense_int8 import (
    dense_epilog_had, dquant_rowwise, quant_w8_slab)

g = torch.Generator(device="cuda:0").manual_seed(1234)
ok = True


def check(name, cond, info=""):
    global ok
    print(f"{'PASS' if cond else 'FAIL'} {name} {info}", flush=True)
    ok = ok and cond


with torch.inference_mode():
    # dquant vs torch reference: scales agree to ~1 ulp (Triton lowers the
    # /127 scale division to recip-mul, torch uses true div); xd may differ
    # by +-1 only on exact-.5-boundary elements.
    a = torch.randn((257, 5120), generator=g, device="cuda:0", dtype=torch.float16)
    sa = torch.empty((257, 1), dtype=torch.float32, device="cuda:0")
    xd = torch.empty((257, 5120), dtype=torch.int8, device="cuda:0")
    dquant_rowwise(a, sa, xd)
    sa_t = a.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    xd_t = (a.float() / sa_t).round().clamp(-128, 127).to(torch.int8)
    rel = ((sa - sa_t).abs() / sa_t.clamp_min(1e-30)).max().item()
    check("dquant sa ~1ulp", rel < 2e-7, f"rel={rel:.2e}")
    xd_bad = (xd != xd_t)
    check("dquant xd boundary-only", xd_bad.sum().item() / xd.numel() < 1e-5
          and (xd.float() - xd_t.float()).abs().max().item() <= 1,
          f"frac={xd_bad.sum().item() / xd.numel():.2e}")

    # w8quant bitwise vs torch reference (s0 model formula)
    w = torch.randn((512, 768), generator=g, device="cuda:0", dtype=torch.float16)
    s0 = torch.tensor([0.0234], dtype=torch.float32, device="cuda:0")
    w8 = torch.empty((1, 512, 768), dtype=torch.int8, device="cuda:0")
    quant_w8_slab(w.unsqueeze(0), s0, w8)
    w8_t = (w.float() / 0.0234).round().clamp(-128, 127).to(torch.int8)
    check("w8quant bitwise", torch.equal(w8.squeeze(0), w8_t))

    # dense epilogue vs torch chain (tolerance: fp32 vs fp16-had rounding).
    # Realistic magnitudes: pre-had values O(1), post-had O(10).
    R, N = 256, 4608
    c = torch.randint(-20000, 20000, (R, N), generator=g, device="cuda:0",
                      dtype=torch.int32)
    sa2 = torch.rand((R, 1), generator=g, device="cuda:0", dtype=torch.float32) * 0.02 + 0.005
    ws = torch.tensor([0.02], dtype=torch.float32, device="cuda:0")
    svh = torch.randn(N, generator=g, device="cuda:0", dtype=torch.float16)
    y = torch.empty((R, N), dtype=torch.float16, device="cuda:0")
    dense_epilog_had(c, sa2, ws, svh, y, N)
    y_t = (c.float() * sa2 * 0.02).to(torch.float16)
    ext.had_r_128(y_t, y_t, None, svh, 1.0)
    d = (y.float() - y_t.float()).abs()
    scale = y_t.float().abs().amax().item()
    check("epilog tolerance", d.max().item() < 5e-3 * scale,
          f"maxabs={d.max().item():.6f} scale={scale:.3f}")

    # strided-y slice path (lm_head): compact c + strided y give same slice
    SN, NF = 256, 1024
    cs = torch.randint(-20000, 20000, (R, SN), generator=g, device="cuda:0",
                       dtype=torch.int32)
    svs = torch.randn(SN, generator=g, device="cuda:0", dtype=torch.float16)
    ybig = torch.zeros((R, NF), dtype=torch.float16, device="cuda:0")
    dense_epilog_had(cs, sa2, ws, svs, ybig, NF, 128)
    ref = torch.empty((R, SN), dtype=torch.float16, device="cuda:0")
    dense_epilog_had(cs, sa2, ws, svs, ref, SN, 0)
    got = ybig[:, 128:128 + SN]
    check("epilog strided==compact", torch.equal(got, ref))
    check("epilog slice untouched", ybig[:, :128].abs().max().item() == 0.0
          and ybig[:, 384:].abs().max().item() == 0.0)

print("ALL_OK" if ok else "FAILURES", flush=True)
sys.exit(0 if ok else 1)
