#!/usr/bin/env python3
"""A/B compressed-word sharing and physical wave counts with fixed logical sums."""
import argparse
import hashlib
import json
import os
import statistics

import torch
from exllamav3.ext import exllamav3_ext as ext


def main():
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("--bits", type = int, default = 4)
    ap.add_argument("--rows", type = int, default = 1)
    ap.add_argument("--k", type = int, default = 2048)
    ap.add_argument("--n", type = int, default = 6144)
    ap.add_argument("--copies", type = int, default = 4)
    ap.add_argument("--iters", type = int, default = 50)
    args = ap.parse_args()
    os.environ["EXL3_GEMV_MAX_M"] = "8"
    os.environ["EXL3_GEMV_MR_M1"] = "1"
    torch.manual_seed(81)
    k, n, m = args.k, args.n, args.rows
    a = torch.randn(m, k, device = "cuda").half()
    bs = [torch.randint(-32768, 32767, (k // 16, n // 16, args.bits * 16),
                        dtype = torch.int16, device = "cuda") for _ in range(args.copies)]
    su = torch.full((k,), 0.02, dtype = torch.float16, device = "cuda")
    sv = torch.full((n,), 0.2, dtype = torch.float16, device = "cuda")
    ah = torch.empty_like(a)
    outs = [torch.empty(m, n, device = "cuda") for _ in bs]

    def call():
        for b, c in zip(bs, outs):
            ext.exl3_gemm(a, b, c, su, ah, sv, -1, False, True, 0)

    configs = [(0, 0), (1, 0), (0, 4), (0, 8), (0, 16), (1, 4), (1, 8), (1, 16)]
    graphs, baseline = {}, None
    for words, waves in configs:
        os.environ["EXL3_GEMV_WORD_TILE"] = str(words)
        if waves:
            os.environ["EXL3_GEMV_PHYSICAL_WARPS"] = str(waves)
        else:
            os.environ.pop("EXL3_GEMV_PHYSICAL_WARPS", None)
        call()
        if baseline is None:
            baseline = [c.clone() for c in outs]
        torch.cuda.synchronize()
        assert all(torch.isfinite(c).all() and torch.equal(c.view(torch.int32), ref.view(torch.int32))
                   for c, ref in zip(outs, baseline))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[(words, waves)] = graph
    samples = {key: [] for key in configs}
    for sweep in range(6):
        for key in (configs if sweep % 2 == 0 else list(reversed(configs))):
            graph = graphs[key]
            for _ in range(5):
                graph.replay()
            start, end = [torch.cuda.Event(enable_timing = True) for _ in range(2)]
            start.record()
            for _ in range(args.iters):
                graph.replay()
            end.record()
            end.synchronize()
            samples[key].append(start.elapsed_time(end) * 1000 / args.iters / args.copies)
    def fingerprint(tensors):
        digest = hashlib.sha256()
        for tensor in tensors:
            digest.update(tensor.cpu().numpy().tobytes())
        return digest.hexdigest()

    print(json.dumps(dict(config = vars(args), device = torch.cuda.get_device_properties(0).gcnArchName,
                          inputs_sha256 = fingerprint([a, su, sv, *bs]),
                          outputs_sha256 = fingerprint(baseline),
                          bit_equal = True, us = {str(key): vals for key, vals in samples.items()},
                          medians = {str(key): statistics.median(vals) for key, vals in samples.items()})))


if __name__ == "__main__":
    main()
