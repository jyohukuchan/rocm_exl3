#!/usr/bin/env python3
"""Chunked-prefill profile via the serving Generator (TTFT + buckets).

Drives text-only prompts of length L through a real Generator with
max_chunk_size C (KV cache carried across chunks), recording the same
per-module CUDA-event timings as stage_profile_dense. TTFT ~= prefill.

Usage: chunked_profile.py MODEL_DIR L CHUNK REPS [OUT.json]
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
    from exllamav3.model import Config, Model
    from exllamav3 import Cache, Tokenizer, Generator, Job
    from exllamav3.modules import transformer as T
    from exllamav3.modules import attn as A
    from exllamav3.modules import gated_delta_net as G
    from exllamav3.modules import mlp as P
    from exllamav3.modules.quant.exl3 import LinearEXL3

    model_dir, L, CH, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    out = sys.argv[5] if len(sys.argv) > 5 else None
    torch.cuda.set_device(0)

    for cls in (T.TransformerBlock, T.ParallelDecoderBlock):
        wrap(cls, "layer")
    wrap(A.Attention, "attn")
    wrap(G.GatedDeltaNet, "gdn")
    wrap(P.GatedMLP, "mlp")
    wrap(LinearEXL3, "lin")

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=L + 2048, max_batch_size=1)
    model.load(device=torch.device("cuda:0"))
    print(f"loaded {model_dir}", flush=True)
    tokenizer = Tokenizer.from_config(config)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    max_chunk_size=CH)

    # Text-only prompt of exactly L tokens (repeat + truncate; no vision).
    enc = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. " * 64,
        add_bos=True, encode_special_tokens=True)
    try:
        unit = [int(x) for x in enc]
    except (TypeError, ValueError):
        unit = [int(x) for x in enc.view(-1).tolist()]
    ids = (unit * (L // len(unit) + 1))[:L]
    assert len(ids) == L, (len(ids), len(unit))

    result = {}
    with torch.inference_mode():
        for r in range(reps + 1):
            TIMES.clear()
            STACK.clear()
            for cls in (T.TransformerBlock, T.ParallelDecoderBlock):
                cls.forward.counter = [0]
            job = Job(input_ids=torch.tensor([ids]),
                        max_new_tokens=1)
            gen.enqueue(job)
            t0 = time.perf_counter()
            ttft = None
            while gen.num_remaining_jobs():
                for res in gen.iterate():
                    if ttft is None and res.get("text"):
                        ttft = time.perf_counter() - t0
            torch.cuda.synchronize()
            if r == 0:
                print(f"L={L} CH={CH} warmup TTFT={ttft:.2f}s", flush=True)
                continue
            acc = {}
            for (kind, idx), pairs in TIMES.items():
                ms = sum(e0.elapsed_time(e1) for e0, e1 in pairs)
                acc.setdefault(f"{kind}:{idx}" if kind == "lin" else kind, []).append(ms)
            sums = {k: round(sum(v) / 1, 2) for k, v in acc.items()}
            sums["ttft_s"] = round(ttft, 3)
            sums["toks"] = round(L / ttft, 1)
            result[f"run{r}"] = sums
            big = {k: v for k, v in sums.items() if not k.startswith("lin:")}
            print(f"L={L} run{r}: " + json.dumps(big), flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
        print("wrote", out, flush=True)


main()
