# SPDX-License-Identifier: MIT
"""Fused INT8 weight quantize for the grouped MoE path (RDNA2 W8A8 work).

Single-pass Triton kernel: fp16 slab / per-expert fp32 step -> round ->
clamp -> int8 slab. Replaces a multi-pass torch chain (fp16->fp32 temp,
broadcast divide, round, clamp, convert) that runs at ~150 GB/s effective
on this ROCm build (16.7ms for a 128x2048x768 slab).

Numerical contract: bitwise identical to the per-expert prototype's
``(w.float() / s0_python_float).round().clamp(-128, 127).to(torch.int8)``.
The divisor MUST stay a scalar: tensor division selects a different kernel
whose last-ulp behavior differs on .5-boundary elements (found via expert
43's up weights). Each Triton program handles one (expert, chunk) tile and
loads its expert's step as a scalar.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rint_half_even(x):
    # Round-half-to-even matching torch.round (banker's). tl.math has no
    # rint on this backend, and half-up (+0.5 floor) differs on exact .5,
    # which real grid-exact weights do hit (see expert-43 saga).
    fx = tl.floor(x)
    d = x - fx
    is_even = (fx * 0.5) == tl.floor(fx * 0.5)
    q = tl.where(d < 0.5, fx, tl.where(d > 0.5, fx + 1.0, tl.where(is_even, fx, fx + 1.0)))
    # Non-finite inputs never occur here (finite fp16 / positive fp32), but
    # keep torch.round semantics if they do (round(inf) = inf -> clamps).
    q = tl.where(x == float("inf"), float("inf"), q)
    q = tl.where(x == float("-inf"), float("-inf"), q)
    return q


@triton.jit
def _w8quant_kernel(
    WFP,
    S0,
    W8,
    KN,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    cid = tl.program_id(1)
    offs = cid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < KN
    # Reciprocal-multiply, NOT true division: torch's scalar division on
    # this build lowers to x * (1/s) (verified: scalar-div == explicit
    # recip-mul bitwise, while tensor division uses true fdiv and differs
    # on .5-boundary elements). Match the prototype's kernel selection.
    r = 1.0 / tl.load(S0 + b)
    x = tl.load(WFP + b * KN + offs, mask).to(tl.float32)
    q = _rint_half_even(x * r)
    q = tl.clamp(q, -128, 127).to(tl.int8)
    tl.store(W8 + b * KN + offs, q, mask)


def quant_w8_slab(wfp: torch.Tensor, s0: torch.Tensor, w8: torch.Tensor) -> None:
    """wfp [B, K, N] half, s0 [B] fp32 CUDA, w8 [B, K, N] int8 out."""
    assert wfp.is_cuda and w8.is_cuda and s0.is_cuda
    B, K, N = wfp.shape
    assert w8.shape == (B, K, N) and s0.numel() == B
    assert wfp.is_contiguous() and w8.is_contiguous() and s0.is_contiguous()
    KN = K * N
    BLOCK = 4096
    grid = (B, triton.cdiv(KN, BLOCK))
    _w8quant_kernel[grid](wfp, s0, w8, KN, BLOCK)
