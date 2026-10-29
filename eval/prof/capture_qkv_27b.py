#!/usr/bin/env python3
"""Capture real Q/K/V from Qwen3.8-27B full-attention layers (q heads == 24).

Usage: capture_qkv_27b.py MODEL_DIR CASES_JSON OUT_PT [MAX_TOK]
Hooks attn_dispatch (no-cache prefill path) and saves fp16 q/k/v per
full-attention layer for the longest case truncated to MAX_TOK (default 512).
No tokenizer needed (ids come from the cases manifest).
Env: HIP_VISIBLE_DEVICES must isolate one GPU.
"""
import os
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

import json

import torch

model_dir, cases_json, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
max_tok = int(sys.argv[4]) if len(sys.argv) > 4 else 512

torch.cuda.set_device(0)
import exllamav3.modules.attn as attn_mod
from exllamav3.model import Config, Model

orig_dispatch = attn_mod.attn_dispatch
records = []


def spy(q, k, v, **kw):
    if q.shape[2] == 24:  # full-attention layers only (24 q heads)
        records.append({"q": q.detach().float().cpu(),
                        "k": k.detach().float().cpu(),
                        "v": v.detach().float().cpu()})
    return orig_dispatch(q, k, v, **kw)


attn_mod.attn_dispatch = spy

config = Config.from_directory(model_dir)
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))
print(f"loaded {model_dir}", flush=True)

cases = json.loads(open(cases_json).read())["cases"]
case = max(cases, key=lambda c: len(c["token_ids"]))
ids = torch.tensor([case["token_ids"][:max_tok]], dtype=torch.long, device="cuda:0")
print(f"case {case['id']} rows={ids.shape[1]}", flush=True)
with torch.inference_mode():
    logits = model.forward(ids, {})
print(f"logits {tuple(logits.shape)} layers-captured={len(records)}", flush=True)
assert len(records) == 16, f"expected 16 full-attention layers, got {len(records)}"

torch.save({"layers": [{"q": r["q"].half(), "k": r["k"].half(), "v": r["v"].half()}
                       for r in records],
            "layer_idx": list(range(len(records))),
            "case_id": case["id"], "rows": ids.shape[1],
            "model": model_dir},
           out_path)
print(f"wrote {out_path}", flush=True)
