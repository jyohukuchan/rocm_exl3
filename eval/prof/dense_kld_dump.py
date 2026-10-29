#!/usr/bin/env python3
"""Dump full-vocab FP32 logits for KLD cases-v1.json through an EXL3 model.

Produces a candidate dir compatible with the GSQ compare.py protocol:
  <id>.f32          - row-major fp32 logits, shape [rows, W] (W = model head
                      width, padded with -1e30 to 248320 when narrower; KL
                      only covers the valid ids so padding is inert)
  manifest.json    - cases [{id, token_count, input_token_ids_sha256,
                      positions, shape, dtype, layout, nonfinite_count,
                      logits_file, raw_path}]

Usage: dense_kld_dump.py MODEL_DIR CASES_JSON OUT_DIR [PAD_W]
Env: HIP_VISIBLE_DEVICES, EXL3_INT8_PREFILL=0/1 (selects fp16 vs int8 path).
Token ids come straight from the manifest (no tokenizer involved).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

import torch

model_dir, cases_json, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
pad_w = int(sys.argv[4]) if len(sys.argv) > 4 else 248320

os.makedirs(out_dir, exist_ok=True)
torch.cuda.set_device(0)
from exllamav3.model import Config, Model

config = Config.from_directory(model_dir)
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))
print(f"loaded {model_dir}", flush=True)

cases = json.loads(open(cases_json).read())["cases"]
manifest = {"schema": "exl3-kld-dump-v1", "engine": "rocm-exl3",
            "model": model_dir,
            "int8_prefill": os.environ.get("EXL3_INT8_PREFILL", "0") == "1",
            "cases": []}
with torch.inference_mode():
    for c in cases:
        ids = torch.tensor([c["token_ids"]], dtype=torch.long, device="cuda:0")
        logits = model.forward(ids, {}).float().cpu().numpy()[0]
        assert (logits != logits).sum() == 0, f"nonfinite in {c['id']}"
        n, w = logits.shape
        if w < pad_w:
            import numpy as np
            logits = np.pad(logits, ((0, 0), (0, pad_w - w)),
                            constant_values=-1e30)
        fn = c["id"] + ".f32"
        logits.astype("<f4").tofile(os.path.join(out_dir, fn))
        h = hashlib.sha256(
            json.dumps(c["token_ids"], separators=(",", ":")).encode()).hexdigest()
        manifest["cases"].append({
            "id": c["id"], "token_count": n,
            "input_token_ids_sha256": h,
            "positions": list(range(n)),
            "shape": [n, logits.shape[1]], "dtype": "f32",
            "layout": "row-major", "nonfinite_count": 0,
            "logits_file": fn, "raw_path": fn})
        print(f"{c['id']}: rows={n} w={logits.shape[1]}", flush=True)
with open(os.path.join(out_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote", out_dir, flush=True)
