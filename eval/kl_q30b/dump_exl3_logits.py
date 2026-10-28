#!/usr/bin/env python3
"""Dump per-position FP32 logits for the KLD cases with exllamav3.

One forward per case (bulk, no chunking, temperature 1): row i = raw FP32
logits after consuming token i. Output layout matches the llama.cpp
reference (ref-llama/): <case>.f32 = rows*151936 little-endian float32,
plus a manifest.json with per-case input SHA.

Usage:
  dump_exl3_logits.py --model DIR --refdir REF --outdir OUT [--device 0]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--refdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--limit-cases", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, "/src")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(args.device))
    torch.cuda.set_device(0)
    from exllamav3.model import Config, Model

    refdir = Path(args.refdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ids = sorted((refdir / "ids").glob("*.txt"))
    if args.limit_cases:
        ids = ids[:args.limit_cases]

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    model.load(device=torch.device("cuda:0"))
    print(f"loaded {args.model}", flush=True)

    manifest = {"model": args.model, "engine": "exllamav3", "dtype": "exl3", "cases": []}
    with torch.inference_mode():
        for idf in ids:
            case = idf.stem
            toks = list(map(int, idf.read_text().split()))
            inp = torch.tensor([toks], dtype=torch.long, device="cuda:0")
            y = model.forward(inp, {})
            if isinstance(y, (tuple, list)):
                y = y[0]
            assert y.dim() == 3 and y.shape[0] == 1 and y.shape[1] == len(toks), \
                f"{case}: unexpected logits shape {tuple(y.shape)} for {len(toks)} tokens"
            lg = y[0].float().contiguous().cpu().numpy()
            assert lg.shape[1] == 151936, f"{case}: vocab {lg.shape[1]}"
            nf = int((lg != lg).sum() + (abs(lg) == float("inf")).sum())
            (outdir / f"{case}.f32").write_bytes(lg.astype("<f4").tobytes())
            manifest["cases"].append({
                "id": case,
                "logits_file": f"{case}.f32",
                "shape": list(lg.shape),
                "nonfinite_count": nf,
                "positions": list(range(len(toks))),
                "input_token_ids_sha256": "sha256:" + hashlib.sha256(
                    json.dumps(toks, separators=(",", ":")).encode()).hexdigest(),
            })
            print(f"{case} rows={len(toks)} nonfinite={nf}", flush=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", outdir, flush=True)


if __name__ == "__main__":
    main()
