# SPDX-License-Identifier: MIT
"""Exhaustively compare all finite FP16 SiLU inputs to the native engine."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.rocm_py.moe_prefill import _silu_half


@triton.jit
def candidate(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    tl.store(Y + i, _silu_half(x), i < N)


def main():
    rows = []
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.half)
    values = values[torch.isfinite(values)]
    assert values.numel() == 63488
    for device in (0, 1):
        with torch.cuda.device(device):
            x = values.to("cuda").view(1, -1)
            ones = torch.ones_like(x)
            expected, actual = torch.empty_like(x), torch.empty_like(x)
            ext.silu_mul(x, ones, expected, 0.0)
            candidate[(triton.cdiv(x.numel(), 256),)](x, actual, x.numel(), 256)
            difference = actual.view(torch.int16) != expected.view(torch.int16)
            record = {
                "device": device,
                "finite_half_inputs": x.numel(),
                "different_bits": int(difference.sum()),
                "exact": torch.equal(
                    actual.view(torch.int16), expected.view(torch.int16)
                ),
            }
            print(json.dumps(record), flush=True)
            rows.append(record)
            assert record["exact"], record
    Path("/work/runs/context-batch/moe-silu-precision.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
