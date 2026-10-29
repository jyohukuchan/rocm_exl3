#!/usr/bin/env python3
"""Dense fp16 vs INT8 prefill A/B on IDENTICAL inputs.

Two separate processes (EXL3_INT8_PREFILL is read at import): run once with
=0 and once with =1, same MODEL_DIR/L/SEED, then compare the saved tail
logits with dense_ab_cmp.py (or a torch one-liner).

Usage: dense_int8_ab.py MODEL_DIR L SEED OUT_PREFIX
Env: EXL3_INT8_PREFILL=0/1, HIP_VISIBLE_DEVICES, LAST_TOKENS_ONLY=N (0=full).
Writes OUT_PREFIX.json {wall_s, tok_s, int8, tail stats} and OUT_PREFIX.pt
(last 256 logit rows, fp32 CPU).
"""
import json
import os
import sys
import time

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

import torch

model_dir, L, seed, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
last_only = int(os.environ.get("LAST_TOKENS_ONLY", "0"))
int8 = os.environ.get("EXL3_INT8_PREFILL", "0") == "1"

torch.cuda.set_device(0)
from exllamav3.model import Config, Model

config = Config.from_directory(model_dir)
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))
print(f"loaded {model_dir} int8={int8}", flush=True)

g = torch.Generator(device="cuda:0").manual_seed(seed)
with torch.inference_mode():
    ids = torch.randint(0, config.vocab_size, (1, L), generator=g, device="cuda:0")
    params = {"last_tokens_only": last_only} if last_only else {}
    _ = model.forward(ids, params)  # warmup (also prints engage lines)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model.forward(ids, params)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

tail = logits[0, -256:, :].float().cpu()
torch.save(tail, outp + ".pt")
stats = {
    "int8": int8,
    "L": L,
    "wall_s": round(wall, 3),
    "tok_s": round(L / wall, 1),
    "tail_absmax": round(tail.abs().max().item(), 6),
    "tail_mean": round(tail.mean().item(), 6),
    "last_top5": torch.topk(logits[0, -1, :], 5).indices.cpu().tolist(),
}
with open(outp + ".json", "w") as f:
    json.dump(stats, f, indent=1)
print(json.dumps(stats), flush=True)
print("wrote", outp + ".json/.pt", flush=True)
