#!/usr/bin/env python3
"""Round-2 E2E check: fp16 cache vs round-1 on-the-fly vs packed i8f8 cache.

Chunked prefill (2048) + greedy decode through the Generator, 32K needle
(passkey retrieval) and 2K sanity. Compares text, passkey presence, and the
decoded-position logits (KLD/top1). Always EXL3_HGEMM_FP16_OUT=1.

Arms:
  fp16   : CacheLayer_fp16,  EXL3_ATTN_INT8FP8 unset
  round1 : CacheLayer_fp16,  EXL3_ATTN_INT8FP8=1 (on-the-fly)
  cached : CacheLayer_i8f8,  EXL3_ATTN_INT8FP8=1 (packed cache + cached kernel)

Asserts the cached kernel actually ran (CALLS counter) in the cached arm.
NOTE: on gfx1030 (V620) the cached kernel uses the int8-proxy PV (no FP8
WMMA); the exact E4M3 path is validated by kernel-vs-ref on R9700.

Usage: needle_i8f8_cached.py MODEL_DIR OUT_JSON [--arms fp16,round1,cached]
       [--ctx 32768] . Needs one GPU with ~24GB free.
"""
import json
import os
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

import time

import torch

MODEL_DIR = sys.argv[1]
OUT_JSON = sys.argv[2]

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--arms", default="fp16,round1,cached")
_ap.add_argument("--ctx", type=int, default=32768)
_a, _ = _ap.parse_known_args()
ARMS = [a.strip() for a in _a.arms.split(",")]
CTX = _a.ctx
PASSKEY_A = "847291"
PASSKEY_B = "391577"

torch.cuda.set_device(0)
print(f"device: {torch.cuda.get_device_properties(0).name}", flush=True)

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, TopPSampler
from exllamav3.cache import CacheLayer_fp16, CacheLayer_i8f8

config = Config.from_directory(MODEL_DIR)
tokenizer = Tokenizer.from_config(config)

filler = ("The grass is green. The sky is blue. The sun is bright. "
          "We see birds flying in the sky. ")


def enc(s):
    t = tokenizer.encode(s)
    if isinstance(t, torch.Tensor):
        t = t.tolist()
    if isinstance(t, list) and len(t) == 1 and isinstance(t[0], list):
        t = t[0]
    assert all(isinstance(x, int) for x in t), type(t)
    return t


def build_prompt(ctx, passkey):
    needle = f"Remember this: the magic passkey number is {passkey}. "
    question = "What is the magic passkey number? Answer with the number only. "
    f_ids = enc(filler)
    n_ids = enc(needle)
    q_ids = enc(question)
    target = ctx - len(q_ids)
    body = []
    while len(body) < target - len(n_ids):
        body += f_ids
    insert = len(body) // 2
    ids = body[:insert] + n_ids + body[insert:]
    ids = (ids + f_ids * ((target - len(ids)) // len(f_ids) + 1))[:target]
    return (ids + q_ids)[:ctx]


def run_case(n_ctx, passkey, arm, steps=30):
    from exllamav3.modules.attention_fn.i8f8_cached import CALLS
    if arm in ("round1", "cached"):
        os.environ["EXL3_ATTN_INT8FP8"] = "1"
    else:
        os.environ.pop("EXL3_ATTN_INT8FP8", None)
    prompt = build_prompt(n_ctx, passkey)
    model = Model.from_config(config)
    lt = CacheLayer_i8f8 if arm == "cached" else CacheLayer_fp16
    cache = Cache(model, max_num_tokens=n_ctx + 256, layer_type=lt)
    model.load(device=torch.device("cuda:0"))
    n0 = CALLS["cached"]
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                          max_chunk_size=2048)
    job = Job(input_ids=torch.tensor([prompt], dtype=torch.long, device="cpu"),
              max_new_tokens=steps,
              sampler=TopPSampler(top_p=0, temperature=0.0),
              return_logits=True, seed=0)
    generator.enqueue(job)
    t0 = time.time()
    first_s = None
    texts, logits = [], []
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if first_s is None:
                first_s = time.time() - t0
            if "text" in r:
                texts.append(r["text"])
            if "logits" in r and r["logits"] is not None:
                lg = torch.as_tensor(r["logits"])
                if lg.ndim == 3:
                    logits.append(lg.cpu())
    dt = time.time() - t0
    text = "".join(texts)
    ran_cached = CALLS["cached"] - n0
    model.unload()
    del generator, cache, model
    torch.cuda.empty_cache()
    cat = torch.cat(logits, dim=1) if logits else None
    print(f"arm={arm} ctx={n_ctx}: cached_calls={ran_cached} "
          f"time_to_first={first_s:.1f}s total={dt:.1f}s ok={passkey in text}",
          flush=True)
    if arm == "cached":
        assert ran_cached > 0, "cached kernel never ran (silent fallback?)"
    return {"text": text, "time_to_first_result_s": first_s, "total_s": dt,
            "logit_shapes": [tuple(l.shape) for l in logits],
            "logits": cat, "passkey_ok": passkey in text,
            "cached_calls": ran_cached}


def kld_top1(a, b):
    # Full-vocab KLD is NaN-poisoned by -inf logits (unused ids); condition
    # each position on the common finite set and renormalize (same mask both
    # sides), which keeps the comparison principled.
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n].double(), b[:, :n].double()
    with torch.inference_mode():
        mask = torch.isfinite(a) & torch.isfinite(b)
        a = torch.where(mask, a, torch.tensor(-1e30, dtype=torch.float64))
        b = torch.where(mask, b, torch.tensor(-1e30, dtype=torch.float64))
        pa, pb = a.log_softmax(-1), b.log_softmax(-1)
        kld = (pa.exp() * (pa - pb)).sum(-1)
        kld = torch.nan_to_num(kld, nan=0.0, posinf=0.0, neginf=0.0)
        ta, tb = a.argmax(-1), b.argmax(-1)
        agree = (ta == tb).double()
    return kld, agree


os.environ["EXL3_HGEMM_FP16_OUT"] = "1"
out = {"model": MODEL_DIR, "arms": ARMS, "ctx": CTX}
import pathlib
ldir = str(OUT_JSON) + ".logits"
os.makedirs(ldir, exist_ok=True)
for tag, n_ctx, pk, steps in (("c32k", CTX, PASSKEY_A, 30), ("c2k", 2048, PASSKEY_B, 80)):
    results = {arm: run_case(n_ctx, pk, arm, steps) for arm in ARMS}
    entry = {"decode_steps": steps}
    for arm, r in results.items():
        if r["logits"] is not None:
            p = os.path.join(ldir, f"{tag}_{arm}.pt")
            torch.save(r["logits"].half(), p)
            entry[f"{arm}_logits_file"] = p
    for arm, r in results.items():
        entry[f"{arm}_text"] = r["text"]
        entry[f"{arm}_ok"] = r["passkey_ok"]
        entry[f"{arm}_time_to_first_s"] = r["time_to_first_result_s"]
        entry[f"{arm}_total_s"] = r["total_s"]
        entry[f"{arm}_cached_calls"] = r["cached_calls"]
    if "fp16" in results:
        for arm in ARMS:
            if arm == "fp16" or results["fp16"]["logits"] is None or results[arm]["logits"] is None:
                continue
            kld, agree = kld_top1(results["fp16"]["logits"], results[arm]["logits"])
            entry[f"{arm}_kld_mean"] = kld.mean().item()
            entry[f"{arm}_kld_max"] = kld.amax().item()
            entry[f"{arm}_top1"] = agree.mean().item()
            print(f"{tag}: fp16-vs-{arm} KLD mean={entry[f'{arm}_kld_mean']:.6f} "
                  f"max={entry[f'{arm}_kld_max']:.4f} top1={entry[f'{arm}_top1']:.4f}",
                  flush=True)
    out[tag] = entry
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=1)
print(f"wrote {OUT_JSON}", flush=True)
