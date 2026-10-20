#!/usr/bin/env python3
"""Compare canonical/N-major CPU experts over a layer stack, not one warm expert."""
import argparse
import json
import statistics
import time

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_cpu_layout import copy_trellis


def main():
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("--layers", type = int, default = 40)
    ap.add_argument("--experts", type = int, default = 5)
    ap.add_argument("--threads", type = int, default = 24)
    ap.add_argument("--bits", type = int, default = 5)
    ap.add_argument("--iters", type = int, default = 10)
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(53)
    h, f = 2048, 512
    x = torch.randn(1, h).half()
    idx = torch.arange(args.experts, dtype = torch.int32).view(1, -1)
    weights = torch.full(idx.shape, 1 / args.experts, dtype = torch.float16)
    scales_h = [torch.full((h,), 0.02, dtype = torch.float16)] * args.experts
    scales_f = [torch.full((f,), 0.02, dtype = torch.float16)] * args.experts
    layers = {0: [], 2: []}
    try:
        for _ in range(args.layers):
            mats = [torch.randint(-32768, 32767, shape, dtype = torch.int16)
                    for shape in ([(h // 16, f // 16, args.bits * 16)] * (2 * args.experts) +
                                  [(f // 16, h // 16, args.bits * 16)] * args.experts)]
            for order in layers:
                packed = []
                for src in mats:
                    dst = torch.empty_like(src)
                    copy_trellis(dst, src, order)
                    packed.append(dst)
                e = args.experts
                handle = ext.exl3_moe_cpu_make_layer(
                    packed[:e], scales_h, scales_f, packed[e:2 * e], scales_h, scales_f,
                    packed[2 * e:], scales_f, scales_h, [], [], [], 0, 0.0, order)
                layers[order].append(handle)
        outputs = [torch.empty(1, h) for _ in range(args.layers)]

        def sweep(order):
            for handle, out in zip(layers[order], outputs):
                ext.exl3_moe_cpu_forward(handle, x, idx, weights, out, args.threads)

        sweep(0)
        baseline = [v.clone() for v in outputs]
        sweep(2)
        assert all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(baseline, outputs))
        samples = {0: [], 2: []}
        for order in (0, 2, 2, 0, 0, 2):
            sweep(order)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                sweep(order)
            samples[order].append((time.perf_counter() - t0) * 1e6 / args.iters)
        medians = {o: statistics.median(v) for o, v in samples.items()}
        print(json.dumps(dict(config = vars(args), us_per_stack = samples, medians = medians,
                              speedup = medians[0] / medians[2], bit_equal = True)))
    finally:
        for handles in layers.values():
            for handle in handles:
                ext.exl3_moe_cpu_free_layer(handle)


if __name__ == "__main__":
    main()
