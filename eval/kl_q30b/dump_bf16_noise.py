#!/usr/bin/env python3
"""Noise-floor dump: exllamav3 BF16 source model, 1-2 cases, 2-GPU split.

Same .f32 + manifest layout as dump_exl3_logits.py. Used to separate
cross-engine noise (llama.cpp vs exllamav3) from quantization loss.
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
    ap.add_argument("--cases", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, "/src")
    # Both V620s: no HIP_VISIBLE_DEVICES restriction here (caller ensures it).
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    from exllamav3.model import Config, Model

    refdir = Path(args.refdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    for i, n in model.load_gen(use_per_device=[28, 28], max_chunk_size=128,
                               max_output_size=8, progressbar=True, verbose=True):
        if i % 8 == 0:
            print(f"module {i}/{n}", flush=True)
    print("LOAD OK", flush=True)

    manifest = {"model": args.model, "engine": "exllamav3",
                "dtype": "bf16", "cases": []}
    with torch.inference_mode():
        for case in args.cases.split(","):
            toks = list(map(int, (refdir / "ids" / f"{case}.txt").read_text().split()))
            inp = torch.tensor([toks], dtype=torch.long, device="cuda:0")
            y = model.forward(inp, {})
            if isinstance(y, (tuple, list)):
                y = y[0]
            assert y.dim() == 3 and y.shape[0] == 1 and y.shape[1] == len(toks), \
                f"{case}: unexpected logits shape {tuple(y.shape)}"
            lg = y[0].float().contiguous().cpu().numpy()
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
