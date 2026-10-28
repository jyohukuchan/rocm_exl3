#!/usr/bin/env python3
"""Profile target: load (i), warmup, then N timed full prefill forwards."""
import os, sys, time
sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch
from exllamav3.model import Config, Model

model_dir, Ls, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.cuda.set_device(0)
config = Config.from_directory(model_dir)
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))
print("loaded", model_dir, flush=True)
g = torch.Generator(device="cuda:0").manual_seed(99)
with torch.inference_mode():
    for L in [int(x) for x in Ls.split(",")]:
        ids = torch.randint(0, config.vocab_size, (1, L), generator=g, device="cuda:0")
        _ = model.forward(ids, {})  # warmup
        torch.cuda.synchronize()
        for r in range(reps):
            ids = torch.randint(0, config.vocab_size, (1, L), generator=g, device="cuda:0")
            t0 = time.perf_counter()
            _ = model.forward(ids, {})
            torch.cuda.synchronize()
            print(f"FW L={L} rep={r} {time.perf_counter()-t0:.3f}s", flush=True)
print("DONE", flush=True)
