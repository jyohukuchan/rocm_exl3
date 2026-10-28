#!/usr/bin/env python3
"""Stage-level prefill breakdown via monkeypatched forwards + CUDA events.

exllamav3 Modules are not torch.nn.Modules (no forward hooks), so this wraps
the Python forward methods of blocks / attention / MLP-MOE and records a CUDA
event pair around each call. Strictly nested single-stream execution makes the
event pairs exact device times.

Usage:
  stage_profile.py MODEL_DIR LENGTHS REPS [OUT.json]

Env: HIP_VISIBLE_DEVICES selects the GPU (single-GPU only).
Per-forward-mean ms: layer_total / attn / mlp(MoE or dense) /
norms_resid (= layer - attn - mlp), dispatch_gap (wall - sum layers).
"""

import json
import os
import sys
import time

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch

STACK = []          # block idx stack (strictly nested)
TIMES = {}          # (kind, idx) -> [ms, ...] per rep (filled per rep)


def wrap(cls, kind):
    orig = cls.forward

    def fwd(self, *args, **kwargs):
        if kind == "layer":
            idx = fwd.counter[0]
            fwd.counter[0] += 1
            STACK.append(idx)
            key = ("layer", idx)
        else:
            key = (kind, STACK[-1] if STACK else -1)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        out = orig(self, *args, **kwargs)
        e1.record()
        TIMES.setdefault(key, []).append((e0, e1))
        if kind == "layer":
            STACK.pop()
        return out

    fwd.counter = [0]
    cls.forward = fwd


def main():
    model_dir, Ls, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out = sys.argv[4] if len(sys.argv) > 4 else None
    torch.cuda.set_device(0)
    from exllamav3.model import Config, Model
    from exllamav3.modules import transformer as T
    from exllamav3.modules import attn as A
    from exllamav3.modules import block_sparse_mlp as M
    from exllamav3.modules import mlp as P

    for cls in (T.TransformerBlock, T.ParallelDecoderBlock):
        wrap(cls, "layer")
    for cls in (A.Attention,):
        # other attention subclasses inherit this forward; wrap base only
        wrap(cls, "attn")
    for cls in (M.BlockSparseMLP, P.GatedMLP):
        wrap(cls, "mlp")

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load(device=torch.device("cuda:0"))
    print(f"loaded {model_dir}", flush=True)

    g = torch.Generator(device="cuda:0").manual_seed(99)
    result = {}
    with torch.inference_mode():
        for L in [int(x) for x in Ls.split(",")]:
            key = str(L)
            acc = {}
            walls = []
            for r in range(reps + 1):  # rep 0 = warmup
                TIMES.clear()
                STACK.clear()
                for cls in (T.TransformerBlock, T.ParallelDecoderBlock):
                    cls.forward.counter = [0]
                ids = torch.randint(0, config.vocab_size, (1, L),
                                    generator=g, device="cuda:0")
                t0 = time.perf_counter()
                _ = model.forward(ids, {})
                torch.cuda.synchronize()
                walls.append(time.perf_counter() - t0)
                if r == 0:
                    nl = len([k for k in TIMES if k[0] == "layer"])
                    na = len([k for k in TIMES if k[0] == "attn"])
                    nm = len([k for k in TIMES if k[0] == "mlp"])
                    print(f"L={L} warmup calls: layer={nl} attn={na} mlp={nm}",
                          flush=True)
                    continue
                for (kind, idx), pairs in TIMES.items():
                    acc.setdefault(kind, []).append(
                        sum(e0.elapsed_time(e1) for e0, e1 in pairs))
            sums = {k: round(sum(v) / max(1, reps), 2)
                    for k, v in acc.items()}
            lay = sums.get("layer", 0)
            att = sums.get("attn", 0)
            mlp = sums.get("mlp", 0)
            wall_mean = sum(walls[1:]) / max(1, reps)
            means = {"layer": lay, "attn": att, "mlp": mlp,
                     "norms_resid": round(lay - att - mlp, 2),
                     "wall_s": round(wall_mean, 3),
                     "dispatch_gap_ms": round(wall_mean * 1000 - lay, 2)}
            result[key] = {"mean": means}
            print(f"L={L}: " + json.dumps(means), flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
        print("wrote", out, flush=True)


main()
