#!/usr/bin/env python3
"""Long-context check for the opt-in INT8/FP8 prefill attention.

Production path (chunked prefill with fp16 paged cache + greedy decode)
under both modes (EXL3_ATTN_INT8FP8 unset vs =1, always EXL3_HGEMM_FP16_OUT=1):

Phase A (32K needle): prefill a 32768-token prompt in 2048-token chunks,
greedy-decode 30 tokens with return_logits; compares generated text,
passkey presence, and the 30 decoded-position logits (KLD/top1).

Phase B (2K needle): same at 2048 tokens (fast sanity).

Usage: needle_attn_32k.py MODEL_DIR OUT_JSON. Needs one GPU with ~24GB free.
"""
import json
import os
import sys

sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")

import time

import torch

MODEL_DIR = sys.argv[1]
OUT_JSON = sys.argv[2]
PASSKEY_A = "847291"
PASSKEY_B = "391577"

torch.cuda.set_device(0)

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, TopPSampler

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


def run_case(n_ctx, passkey, mode, steps=30):
    if mode:
        os.environ["EXL3_ATTN_INT8FP8"] = "1"
    else:
        os.environ.pop("EXL3_ATTN_INT8FP8", None)
    prompt = build_prompt(n_ctx, passkey)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=n_ctx + 256)
    model.load(device=torch.device("cuda:0"))
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
    model.unload()
    del generator, cache, model
    torch.cuda.empty_cache()
    cat = torch.cat(logits, dim=1) if logits else None  # (1, N, V)
    return {"text": text, "time_to_first_result_s": first_s, "total_s": dt,
            "logit_shapes": [tuple(l.shape) for l in logits],
            "logits": cat,
            "passkey_ok": passkey in text}


os.environ["EXL3_HGEMM_FP16_OUT"] = "1"
res = {}
for tag, n_ctx, pk, steps in (("c32k", 32768, PASSKEY_A, 30), ("c2k", 2048, PASSKEY_B, 80)):
    r0 = run_case(n_ctx, pk, False, steps)
    r1 = run_case(n_ctx, pk, True, steps)
    entry = {"fp16_text": r0["text"], "int8fp8_text": r1["text"],
             "fp16_ok": r0["passkey_ok"], "int8fp8_ok": r1["passkey_ok"],
             "fp16_time_to_first_s": r0["time_to_first_result_s"],
             "int8fp8_time_to_first_s": r1["time_to_first_result_s"],
             "fp16_total_s": r0["total_s"], "int8fp8_total_s": r1["total_s"],
             "decode_steps": steps}
    if r0["logits"] is not None and r1["logits"] is not None:
        a, b = r0["logits"].double(), r1["logits"].double()
        n = min(a.shape[1], b.shape[1])
        a, b = a[:, :n], b[:, :n]
        entry["logit_event_shapes"] = r0["logit_shapes"]
        with torch.inference_mode():
            pa, pb = a.log_softmax(-1), b.log_softmax(-1)
            kld = (pa.exp() * (pa - pb)).sum(-1)
            kld = torch.nan_to_num(kld, nan=0.0, posinf=0.0, neginf=0.0)
            d = (a - b).abs()
            d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            entry.update({"logit_rows": n,
                          "logit_maxabs": d.max().item(), "logit_meanabs": d.mean().item(),
                          "logit_kld_mean": kld.mean().item(),
                          "logit_kld_max": kld.max().item(),
                          "logit_top1_agree": (a.argmax(-1) == b.argmax(-1)).float().mean().item()})
    res[tag] = entry
    print(json.dumps({tag: {k: v for k, v in entry.items()}}, indent=1), flush=True)

json.dump(res, open(OUT_JSON, "w"), indent=1)
print(f"wrote {OUT_JSON}", flush=True)
