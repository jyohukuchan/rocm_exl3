# SPDX-License-Identifier: MIT
"""Private RDNA MoE prefill candidates; native FP16/FP32 rounding is retained."""

import torch
import triton
import triton.language as tl
from triton.language import core


@triton.jit
def _butterflies(value, columns: tl.constexpr, GROUPS: tl.constexpr):
    for shift in tl.static_range(7):
        step = 1 << shift
        indices = tl.broadcast_to(columns[None, :] ^ step, (GROUPS, 128))
        peer = tl.gather(value, indices, axis=1)
        value = tl.where((columns[None, :] & step) != 0, -value, value) + peer
    return value


@triton.jit(do_not_specialize=["ROWS", "ROWS_PER_EXPERT"])
def _hadamard(
    X,
    Y,
    TABLE,
    IDS,
    ROWS,
    COLS: tl.constexpr,
    ROWS_PER_EXPERT,
    PRE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    transform = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    row = transform // (COLS // 128)
    chunk = transform % (COLS // 128)
    c = tl.arange(0, 128)
    offset = row[:, None] * COLS + chunk[:, None] * 128 + c[None, :]
    valid = row < ROWS
    expert = tl.load(IDS + row // ROWS_PER_EXPERT, valid, 0)
    scale = tl.load(
        TABLE + expert[:, None] * COLS + chunk[:, None] * 128 + c[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)
    value = tl.load(X + offset, valid[:, None], 0).to(tl.float32)
    if PRE:
        value = (value * scale).to(X.dtype.element_ty).to(tl.float32)
    value = _butterflies(value, c, GROUPS)
    value = (value * 0.088388347648).to(Y.dtype.element_ty).to(tl.float32)
    if not PRE:
        value = value * scale
    tl.store(Y + offset, value, valid[:, None])


def hadamard_batch(x, out, table, ids, rows_per_expert, pre, groups=4):
    rows, cols = x.shape
    assert not (
        pre and x.dtype == torch.float32
    ), "FP32 pre-scale uses native FMA contraction"
    assert x.dtype in (torch.float16, torch.float32) and out.dtype == x.dtype
    assert x.is_contiguous() and out.is_contiguous() and cols % 128 == 0
    assert table.dtype == torch.float16 and table.is_contiguous()
    assert ids.dtype == torch.int64 and ids.is_contiguous()
    _hadamard[(triton.cdiv(rows * (cols // 128), groups),)](
        x,
        out,
        table,
        ids,
        rows,
        cols,
        rows_per_expert,
        pre,
        groups,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit(do_not_specialize=["EXPERTS", "CMAX", "SENTINEL"])
def _gather_input(
    X,
    TOKENS,
    META,
    Y,
    EXPERTS,
    CMAX,
    COLS: tl.constexpr,
    SENTINEL,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    expert = row // CMAX
    r = row % CMAX
    start = tl.load(META + EXPERTS + expert)
    count = tl.load(META + 2 * EXPERTS + expert)
    source = tl.where(r < count, start + r, SENTINEL)
    token = tl.load(TOKENS + source)
    c = tl.arange(0, BLOCK)
    value = tl.load(X + token * COLS + c, c < COLS, 0)
    tl.store(Y + row * COLS + c, value, c < COLS)


def gather_input(x, tokens, meta, experts, cmax, sentinel):
    out = torch.empty((experts, cmax, x.shape[1]), device=x.device, dtype=x.dtype)
    _gather_input[(experts * cmax,)](
        x,
        tokens,
        meta,
        out,
        experts,
        cmax,
        x.shape[1],
        sentinel,
        triton.next_power_of_2(x.shape[1]),
        num_warps=4,
    )
    return out


@triton.jit(do_not_specialize=["EXPERTS", "CMAX", "SENTINEL"])
def _gather_pre_hadamard(
    X,
    TOKENS,
    META,
    SG,
    SU,
    IDS,
    G,
    U,
    EXPERTS,
    CMAX,
    COLS: tl.constexpr,
    SENTINEL,
    GROUPS: tl.constexpr,
):
    transform = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    row = transform // (COLS // 128)
    chunk = transform % (COLS // 128)
    b, r = row // CMAX, row % CMAX
    valid = row < EXPERTS * CMAX
    start = tl.load(META + EXPERTS + b, valid, 0)
    count = tl.load(META + 2 * EXPERTS + b, valid, 0)
    source = tl.where(r < count, start + r, SENTINEL)
    token = tl.load(TOKENS + source, valid, 0)
    expert = tl.load(IDS + b, valid, 0)
    c = tl.arange(0, 128)
    col = chunk[:, None] * 128 + c[None, :]
    value = tl.load(X + token[:, None] * COLS + col, valid[:, None], 0).to(tl.float32)
    sg = tl.load(SG + expert[:, None] * COLS + col, valid[:, None], 0).to(tl.float32)
    su = tl.load(SU + expert[:, None] * COLS + col, valid[:, None], 0).to(tl.float32)
    vg = (value * sg).to(tl.float16).to(tl.float32)
    vu = (value * su).to(tl.float16).to(tl.float32)
    vg = _butterflies(vg, c, GROUPS) * 0.088388347648
    vu = _butterflies(vu, c, GROUPS) * 0.088388347648
    offset = row[:, None] * COLS + col
    tl.store(G + offset, vg, valid[:, None])
    tl.store(U + offset, vu, valid[:, None])


def gather_pre_hadamard(
    x, tokens, meta, ids, scales_g, scales_u, experts, cmax, sentinel
):
    assert x.dtype == torch.float16 and x.shape[1] % 128 == 0
    shape = (experts, cmax, x.shape[1])
    gate, up = torch.empty(shape, device=x.device, dtype=x.dtype), torch.empty(
        shape, device=x.device, dtype=x.dtype
    )
    _gather_pre_hadamard[(triton.cdiv(experts * cmax * (x.shape[1] // 128), 4),)](
        x,
        tokens,
        meta,
        scales_g,
        scales_u,
        ids,
        gate,
        up,
        experts,
        cmax,
        x.shape[1],
        sentinel,
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return gate, up


# HIP's h2exp lowers to the half-precision OCML entry, preserving its half
# rounding before the native half add/reciprocal/multiply sequence.
@core.extern
def _exp_half(value, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [value],
        {(core.dtype("fp16"),): ("__ocml_exp_f16", core.dtype("fp16"))},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit(do_not_specialize=["ROWS", "CMAX"])
def _post_silu_pre(
    G, U, SG, SU, SD, IDS, OUT, ROWS, COLS: tl.constexpr, CMAX, GROUPS: tl.constexpr
):
    transform = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    row = transform // (COLS // 128)
    chunk = transform % (COLS // 128)
    valid = row < ROWS
    c = tl.arange(0, 128)
    col = chunk[:, None] * 128 + c[None, :]
    offset = row[:, None] * COLS + col
    expert = tl.load(IDS + row // CMAX, valid, 0)
    g = tl.load(G + offset, valid[:, None], 0).to(tl.float32)
    u = tl.load(U + offset, valid[:, None], 0).to(tl.float32)
    g = (_butterflies(g, c, GROUPS) * 0.088388347648).to(tl.float16).to(tl.float32)
    u = (_butterflies(u, c, GROUPS) * 0.088388347648).to(tl.float16).to(tl.float32)
    sg = tl.load(SG + expert[:, None] * COLS + col, valid[:, None], 0).to(tl.float32)
    su = tl.load(SU + expert[:, None] * COLS + col, valid[:, None], 0).to(tl.float32)
    sd = tl.load(SD + expert[:, None] * COLS + col, valid[:, None], 0).to(tl.float32)
    g = (g * sg).to(tl.float16)
    u = (u * su).to(tl.float16).to(tl.float32)
    exponential = _exp_half(-g).to(tl.float32)
    denominator = (1.0 + exponential).to(tl.float16).to(tl.float32)
    reciprocal = tl.div_rn(1.0, denominator).to(tl.float16).to(tl.float32)
    silu = (g.to(tl.float32) * reciprocal).to(tl.float16).to(tl.float32)
    activation = (silu * u).to(tl.float16).to(tl.float32)
    activation = (activation * sd).to(tl.float16).to(tl.float32)
    result = _butterflies(activation, c, GROUPS) * 0.088388347648
    tl.store(OUT + offset, result, valid[:, None])


def post_silu_pre(g, u, post_g, post_u, pre_d, ids, cmax):
    assert (
        g.dtype == u.dtype == torch.float16 and g.is_contiguous() and u.is_contiguous()
    )
    rows, cols = g.shape
    assert cols % 128 == 0
    out = torch.empty_like(g)
    _post_silu_pre[(triton.cdiv(rows * (cols // 128), 4),)](
        g,
        u,
        post_g,
        post_u,
        pre_d,
        ids,
        out,
        rows,
        cols,
        cmax,
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
