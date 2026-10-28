#!/usr/bin/env python3
"""Prefill throughput benchmark: normal EXL3 vs INT8 paths.

Loads one EXL3 model on a single GPU, runs full-sequence forwards
(prefill, no KV reuse across runs) at fixed token lengths, reports tok/s.

Usage:
  bench_prefill.py --model DIR --out OUT.json [--lengths 512,2048,8192]
                   [--reps 3] [--device 0] [--exl3-int8-prefill 0/1]
                   [--profile-moe 0/1] [--max-cases 0]
Env passthrough: EXL3_INT8_PREFILL / EXL3_INT8_MOE_PROFILE are read by
exllamav3 itself; this script only sets defaults when the flags say so.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--exl3-int8-prefill", type=int, default=-1)
    ap.add_argument("--profile-moe", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.exl3_int8_prefill >= 0:
        os.environ["EXL3_INT8_PREFILL"] = str(args.exl3_int8_prefill)
    if args.profile_moe:
        os.environ["EXL3_INT8_MOE_PROFILE"] = "1"
    sys.path.insert(0, "/src")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(args.device))
    torch.cuda.set_device(0)
    from exllamav3.model import Config, Model

    lengths = [int(x) for x in args.lengths.split(",")]
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(device=torch.device("cuda:0"))
    vocab = config.vocab_size
    print(f"loaded {args.model} vocab={vocab}", flush=True)

    g = torch.Generator(device="cuda:0").manual_seed(1234)
    results = {"model": args.model, "lengths": {}}
    with torch.inference_mode():
        for L in lengths:
            # Fresh random prompt per call (same pattern as the KLD dump loop):
            # each forward is an independent full prefill, no KV reuse.
            def one_forward():
                ids = torch.randint(0, vocab, (1, L), generator=g, device="cuda:0")
                return model.forward(ids, {})
            # Warmup (also triggers lazy init / autotune / MoE engagement logs)
            for _ in range(2):
                _ = one_forward()
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                _ = one_forward()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            best = min(ts)
            results["lengths"][str(L)] = {
                "seconds_best": best,
                "seconds_all": ts,
                "tps": L / best,
            }
            print(f"L={L}: {L / best:.0f} tok/s (best of {args.reps}: {best:.3f}s)", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
