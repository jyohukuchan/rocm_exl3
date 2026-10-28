#!/usr/bin/env python3
"""Dense-model stage profile: per-key linear timings + block/attn/GDN/MLP totals.

Wraps TransformerBlock/ParallelDecoderBlock (layer), Attention (full-attn
total incl. qkv/o), GatedDeltaNet (linear-attn total incl. projections),
GatedMLP (dense MLP total) and every LinearEXL3 (by module key). Buckets are
assigned offline from key substrings; warmup prints the full key list so the
mapping stays data-driven.

Usage: stage_profile_dense.py MODEL_DIR LENGTHS REPS [OUT.json]
Env: HIP_VISIBLE_DEVICES selects the GPU (single-GPU only).
"""
import json
import os
import sys
import time

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch

STACK = []
TIMES = {}


def wrap(cls, kind):
    orig = cls.forward

    def fwd(self, *args, **kwargs):
        if kind == "layer":
            idx = fwd.counter[0]
            fwd.counter[0] += 1
            STACK.append(idx)
            key = ("layer", idx)
        elif kind == "lin":
            key = ("lin", getattr(self, "key", "?"))
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
    # Skip the full-vocab logit materialization (4 GB at 8192 rows) for long
    # contexts: only the lm_head rows change, everything else runs full.
    # Compensate wall/lm_head with the standalone lm_head bench. 0 = full.
    last_only = int(os.environ.get("LAST_TOKENS_ONLY", "0"))
    torch.cuda.set_device(0)
    from exllamav3.model import Config, Model
    from exllamav3.modules import transformer as T
    from exllamav3.modules import attn as A
    from exllamav3.modules import gated_delta_net as G
    from exllamav3.modules import mlp as P
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from exllamav3.modules.linear import Linear

    for cls in (T.TransformerBlock, T.ParallelDecoderBlock):
        wrap(cls, "layer")
    wrap(A.Attention, "attn")
    wrap(G.GatedDeltaNet, "gdn")
    wrap(P.GatedMLP, "mlp")
    wrap(LinearEXL3, "lin")
    if Linear is not LinearEXL3:
        wrap(Linear, "lin")

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
                params = {"last_tokens_only": last_only} if last_only else {}
                _ = model.forward(ids, params)
                torch.cuda.synchronize()
                walls.append(time.perf_counter() - t0)
                if r == 0:
                    nl = len([k for k in TIMES if k[0] == "layer"])
                    na = len([k for k in TIMES if k[0] == "attn"])
                    ng = len([k for k in TIMES if k[0] == "gdn"])
                    nm = len([k for k in TIMES if k[0] == "mlp"])
                    keys = sorted({k[1] for k in TIMES if k[0] == "lin"})
                    print(f"L={L} warmup: layer={nl} attn={na} gdn={ng} mlp={nm} "
                          f"lin_keys={len(keys)}", flush=True)
                    for k in keys:
                        print(f"  lin: {k}", flush=True)
                    continue
                for (kind, idx), pairs in TIMES.items():
                    ms = sum(e0.elapsed_time(e1) for e0, e1 in pairs)
                    if kind == "lin":
                        acc.setdefault(f"lin:{idx}", []).append(ms)
                    else:
                        acc.setdefault(kind, []).append(ms)
            sums = {k: round(sum(v) / max(1, reps), 2)
                    for k, v in acc.items()}
            wall_mean = sum(walls[1:]) / max(1, reps)
            sums["wall_s"] = round(wall_mean, 3)
            sums["toks"] = round(L / wall_mean, 1)
            result[key] = sums
            big = {k: v for k, v in sums.items() if not k.startswith("lin:")}
            print(f"L={L}: " + json.dumps(big), flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
        print("wrote", out, flush=True)


main()
