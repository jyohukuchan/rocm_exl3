#!/usr/bin/env python3
"""
Bitwise gate for the EXL3 tile quantizer.

Quantizes fixed pseudo-random tiles for every (K, codebook) pair and prints one
SHA-256 per configuration. Two extension builds are bit-identical iff their
outputs agree line for line:

    PYTHONPATH=<lib_a>:/src python rocm_tools/qt_bitwise.py > a.txt
    PYTHONPATH=<lib_b>:/src python rocm_tools/qt_bitwise.py > b.txt
    diff a.txt b.txt

This is the fast gate for quantizer changes (seconds); the partial-conversion
qtensors comparison (gate2_partial.sh) and the full-model SHA-256 are the
end-to-end gates on top of it.
"""
import hashlib
import sys

import torch

from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles


def main() -> int:
    if not torch.cuda.is_available():
        print(" !! a GPU is required", file = sys.stderr)
        return 1
    torch.manual_seed(1234)
    for K in range(1, 9):
        for cb in ("plain", "mcg", "mul1"):
            tiles = torch.randn(4096, 256, device = "cuda:0")
            scales = torch.randint(1, 8, (4096, 1), device = "cuda:0").float()
            tiles = (tiles * scales).contiguous()
            quant_args = {"K": K}
            if cb != "plain":
                quant_args[cb] = True
            q, idx = quantize_tiles(tiles, quant_args)
            h = hashlib.sha256(
                q.cpu().numpy().tobytes() + idx.cpu().numpy().tobytes()
            ).hexdigest()
            print(f"K{K} {cb} {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
