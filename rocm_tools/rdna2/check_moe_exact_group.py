# SPDX-License-Identifier: MIT
"""Validate original-shape grouped GEMM and expert-order accumulation."""

import json
import sys
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.rocm_py.moe_group_backend import ensure_group_backend

assert ensure_group_backend(ext)
from exllamav3.rocm_py.moe_prefill import ordered_accumulate


def main():
    records = []
    for device in (0, 1):
        with torch.cuda.device(device):
            torch.manual_seed(71)
            for dtype in (
                () if "--ordered-only" in sys.argv else (torch.half, torch.float)
            ):
                for k, n in ((2560, 640), (640, 2560)):
                    counts = [193, 129, 157]
                    b = len(counts)
                    m = max(counts)
                    x = torch.randn((b, m, k), device="cuda", dtype=torch.half)
                    w = torch.randn((b, k, n), device="cuda", dtype=torch.half) * 0.02
                    expected = torch.zeros((b, m, n), device="cuda", dtype=dtype)
                    actual = torch.empty_like(expected)
                    for i, count in enumerate(counts):
                        ext.hgemm_recon(x[i, :count], w[i], expected[i, :count])
                    ext.hgemm_exact_group(x, w, actual, counts)
                    assert torch.equal(actual, expected), (
                        device,
                        k,
                        n,
                        dtype,
                        float((actual - expected).abs().max()),
                    )
                    records.append(
                        {
                            "device": device,
                            "operation": "original_shape_group_gemm",
                            "dtype": str(dtype),
                            "k": k,
                            "n": n,
                            "exact": True,
                        }
                    )
            rows, k, e, cols = 33, 10, 16, 2560
            global_ids = torch.stack(
                [torch.randperm(e * 2, device="cuda")[:k] for _ in range(rows)]
            )
            experts = torch.where(global_ids < e, global_ids, e).reshape(-1)
            order = experts.argsort()
            inverse = torch.empty_like(order).scatter_(
                0, order, torch.arange(rows * k, device="cuda")
            )
            counts = torch.bincount(experts, minlength=e + 1)
            starts = counts.cumsum(0) - counts
            base = starts + torch.arange(e + 1, device="cuda") * 3
            kind = torch.where(torch.arange(e + 1, device="cuda") % 3 != 0, 2, 1)
            kind[-1] = 0
            tables = torch.stack((base, starts, kind))
            scratch = torch.randn((int(base[-1] + counts[-1]) + 1, cols), device="cuda")
            weights = torch.randn(rows * k, device="cuda", dtype=torch.half)
            initial = torch.randn((rows, cols), device="cuda")
            expected = initial.clone()
            actual = initial.clone()
            tokens = torch.arange(rows, device="cuda").repeat_interleave(k)[order]
            for expert in range(e):
                if int(kind[expert]) != 2:
                    continue
                start, count, offset = (
                    int(starts[expert]),
                    int(counts[expert]),
                    int(base[expert]),
                )
                value = (
                    scratch[offset : offset + count]
                    .clone()
                    .mul_(weights[start : start + count, None])
                )
                expected.index_add_(0, tokens[start : start + count], value)
            ordered_accumulate(actual, scratch, experts, inverse, tables, weights, k, e)
            assert torch.equal(actual, expected), float((actual - expected).abs().max())
            records.append(
                {
                    "device": device,
                    "operation": "expert_order_accumulation",
                    "exact": True,
                }
            )
    Path("/work/runs/context-batch/moe-exact-group-check.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
