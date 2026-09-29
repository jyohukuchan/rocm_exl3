#!/usr/bin/env python3
"""Reusable Qwen3.8 Flash-Next AR/MTP evaluation CLI for the V620 pair.

Target EXL3 3.05bpw+head5 split across the pair with Engram tables in RAM; the MTP head
(mtp_bits3, already inside the checkpoint) is constructed and kept on cuda:1 in
BOTH modes so AR controls share the exact device layout and residency of MTP
runs. Power policy is delegated to the reviewed rocm_tools.rdna2.power_policy
CLIENT (root starts the privileged Unix-socket helper; a missing socket fails
loudly). Timing mirrors the accepted runs/qwen38-mtp sweep: ONE delivery event
per iterate whose cumulative job.new_tokens changed (MTP multi-token bursts are
never duplicated per streaming event), observed stream rate from the first..last
cumulative events, legacy engine rate kept separately. First-run JIT is a cold
path: warmup prompts are those flagged timed=false; comparisons are left to
root. torch / exllamav3 / exllamav3_ext load LAZILY inside run() so prompt
validation and this module's unit tests stay CPU-safe. JSON is written in a
finally block, both models unload and the power context closes even on error;
exit is nonzero unless every job completed exactly.

Usage (native stack via PYTHONPATH):
    python3 -m rocm_tools.rdna2.qwen_mtp_run -m MODEL --prompts-json P.json \
        --mode mtp --power-socket SOCK --output OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import struct
import sys
import time
from pathlib import Path

MODES = ("ar", "mtp")
RESULT_KEYS = ("time_prefill", "time_generate", "new_tokens", "prompt_tokens",
               "cached_tokens", "eos_reason", "accepted_draft_tokens", "rejected_draft_tokens")


def prompt_sha256(ids_row):
    """sha256 over little-endian int64 bytes, matching the sweep's tensor.to_bytes()."""
    return hashlib.sha256(b"".join(struct.pack("<q", v) for v in ids_row)).hexdigest()


def validate_prompts(payload, batch_size):
    """Normalize + verify a frozen-prompt file; returns entries with metadata preserved.

    Accepts the sweep artifact shape {"prompts":[{"language","repeat","timed",
    "ids":[[token_ids]],"sha":hex}]} or the same shape written cleanly. Raises
    ValueError on: non-integer/empty/multi-row ids, sha mismatch (truncated
    or retokenized prompts), duplicates within one process (they would be cache
    hits), groups that do not cover --batch-size, or mixed timed flags inside a
    batch group (warmup must never dilute a measured batch).
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("prompts"), list) \
            or not payload["prompts"]:
        raise ValueError("prompts-json must be {'prompts': [ ... ]} with at least one entry")
    out, seen = [], {}
    for i, p in enumerate(payload["prompts"]):
        if not isinstance(p, dict):
            raise ValueError(f"prompt {i}: must be an object")
        ids = p.get("ids")
        if not (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)):
            raise ValueError(f"prompt {i}: ids must be [[token_ids]] with exactly one row")
        row = ids[0]
        if not row:
            raise ValueError(f"prompt {i}: empty token id row")
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"prompt {i}: ids must be plain integers, got {type(v).__name__}")
            if not 0 <= v < 2 ** 63:
                raise ValueError(f"prompt {i}: id {v} outside nonnegative int64 range")
        if type(p.get("timed")) is not bool:
            raise ValueError(f"prompt {i}: timed must be an explicit boolean")
        sha = prompt_sha256(row)
        if p.get("sha") != sha:
            raise ValueError(f"prompt {i}: sha mismatch (file={p.get('sha')!r} computed={sha}); "
                             "ids are truncated or not the frozen tokenization")
        if sha in seen:
            raise ValueError(f"prompt {i}: duplicate of prompt {seen[sha]}; a repeated prompt "
                             "inside one process is a cache hit, not a fresh measurement")
        seen[sha] = i
        out.append({**p, "language": p.get("language"), "repeat": p.get("repeat"),
                    "ids": row, "sha": sha})
    if len(out) % batch_size:
        raise ValueError(f"{len(out)} prompts do not cover --batch-size {batch_size} evenly; "
                         "a partial group would dilute batched timing")
    for g in range(0, len(out), batch_size):
        if len({p["timed"] for p in out[g:g + batch_size]}) > 1:
            raise ValueError(f"group {g // batch_size}: mixed timed flags; warmup and measured "
                             "jobs must not share a batch group")
    return out


def delivery_rate(events):
    """Observed stream tps from raw cumulative (t_s, tokens_so_far) events.

    Uses (last_cum - first_cum) / (last_t - first_t) so a multi-token first
    iterate (MTP burst) is not amortized as if tokens arrived one by one;
    deliberately NOT (N-1)/duration. Returns None when no interval exists.
    """
    if len(events) < 2:
        return None
    (t0, c0), (t1, c1) = events[0], events[-1]
    return None if t1 <= t0 else (c1 - c0) / (t1 - t0)


def build_parser():
    ap = argparse.ArgumentParser(
        description="Qwen3.8 Flash-Next AR/MTP evaluation on the V620 pair (frozen prompts, "
                    "policy-switched, JSON report even on failure)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-m", "--model", required=True, help="model dir holding target + mtp component")
    ap.add_argument("--prompts-json", required=True, help="frozen prompt ids file (see docstring)")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--draft-tokens", type=int, default=4, help="draft-window ceiling")
    ap.add_argument("--dynamic-draft", action="store_true", help="let the calibrator shrink drafts")
    ap.add_argument("--draft-confidence", type=float, default=0.4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--cache-tokens", type=int, default=8704)
    ap.add_argument("--max-chunk-size", type=int, default=2048)
    ap.add_argument("--use-per-device", type=float, nargs=2, default=[28, 28], metavar=("GiB0", "GiB1"),
                    help="load budget per GPU in GiB")
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--power-socket", required=True, help="power_switch_server.py Unix socket")
    ap.add_argument("--output", required=True, help="report JSON path (written even on failure)")
    ap.add_argument("--validate-finite", action="store_true",
                    help="explicit per-forward isfinite checks; validation-only, never steady performance")
    return ap


def validate_args(args):
    for name in ("draft_tokens", "batch_size", "cache_tokens", "max_chunk_size", "new_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if any(not math.isfinite(b) or b <= 0 for b in args.use_per_device):
        raise ValueError("--use-per-device budgets must be finite and positive")
    if not math.isfinite(args.draft_confidence) or not 0 < args.draft_confidence <= 1:
        raise ValueError("--draft-confidence must be finite and in (0, 1]")
    if args.cache_tokens % 256 or args.max_chunk_size % 256:
        raise ValueError("cache and chunk capacities must be multiples of 256")
    if args.mode == "ar" and args.dynamic_draft:
        raise ValueError("--dynamic-draft requires --mode mtp; AR has no draft proposals")
    return args


def _raise_nofile(target=65536):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < target:
        raise ValueError(f"nofile hard limit {hard} cannot support required {target}")
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return {"rlimit_nofile_soft": soft, "rlimit_nofile_hard": hard}


def run(args, prompts):
    """GPU-side evaluation; returns the exit code (0 only when fully complete).

    Never calls os._exit; the report JSON is written even on failure and both
    models unload via finally.
    """
    import torch
    import exllamav3_ext
    from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
    from exllamav3.generator.sampler import ArgmaxSampler
    from rocm_tools.rdna2 import multi_gpu, power_policy
    from rocm_tools.rdna2.common import model_fingerprint, git_commit

    native_path = Path(exllamav3_ext.__file__).resolve()
    report = {"complete": False, "validation_only": args.validate_finite,
              "cli": {k: v for k, v in vars(args).items()},
              "native": {"path": str(native_path),
                         "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest()},
              "model_fingerprint": model_fingerprint(args.model),
              "repo_git_commit": git_commit(),
              "runtime_env": {k: v for k, v in os.environ.items()
                              if k.startswith("EXL3_") or k in ("PYTHONPATH", "HSA_ENABLE_SDMA",
                                  "HSA_ENABLE_PEER_SDMA", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
              "power_policy_socket": str(args.power_socket), **_raise_nofile(),
              "prompts": [{**{k: v for k, v in p.items() if k != "ids"},
                           "ids_sha256": p["sha"], "n_tokens": len(p["ids"])} for p in prompts],
              "runs": [], "groups": []}
    model = draft = policy = None
    try:
        if torch.cuda.device_count() != 2 or any(
                torch.cuda.get_device_properties(d).gcnArchName.split(":")[0] != "gfx1030"
                for d in (0, 1)):
            raise RuntimeError("requires exactly 2 gfx1030 GPUs (the V620 pair)")
        cfg = Config.from_directory(args.model)
        cfg.infer_params.ngram_stream_from_disk = False
        model = Model.from_config(cfg)
        draft = Model.from_config(cfg, component="mtp")   # resident even in AR control runs
        tok = Tokenizer.from_config(cfg)
        if any(v >= tok.actual_vocab_size for p in prompts for v in p["ids"]):
            raise ValueError("prompt contains a token ID outside the tokenizer vocabulary")
        for gi in range(0, len(prompts), args.batch_size):
            needed = sum(((len(p["ids"]) + args.new_tokens + args.draft_tokens + 255) // 256) * 256
                         for p in prompts[gi:gi + args.batch_size])
            if needed > args.cache_tokens:
                raise ValueError(f"group requires {needed} cache tokens, capacity is {args.cache_tokens}")
        cache = Cache(model, max_num_tokens=args.cache_tokens, max_batch_size=args.batch_size,
                      max_history=args.draft_tokens)
        dcache = Cache(draft, max_num_tokens=args.cache_tokens, max_batch_size=args.batch_size)
        draft.load(device="cuda:1", max_chunk_size=args.max_chunk_size, progressbar=False)
        report["allocated_bytes_after_draft_load"] = {f"cuda:{d}": torch.cuda.memory_allocated(d) for d in (0, 1)}
        model.load(use_per_device=args.use_per_device, max_chunk_size=args.max_chunk_size, progressbar=False)
        report["allocated_bytes_after_load"] = {f"cuda:{d}": torch.cuda.memory_allocated(d) for d in (0, 1)}
        for kind, audit in (("placement", multi_gpu.audit_placement(model, [0, 1])),
                            ("ngram", multi_gpu.collect_ngram_state(model, require_ram=True))):
            report[kind] = audit
            if not audit["ok"]:
                raise RuntimeError(f"{kind} audit failed: {audit.get('problems')}")
        kwargs = dict(model=model, cache=cache, tokenizer=tok, max_batch_size=args.batch_size,
                      max_chunk_size=args.max_chunk_size, ngram_match_min=0, record_draft_stats=True)
        if args.mode == "mtp":
            kwargs.update(draft_model=draft, draft_cache=dcache, num_draft_tokens=args.draft_tokens,
                          dynamic_draft_tokens=args.dynamic_draft, draft_confidence=args.draft_confidence)
        else:
            kwargs["num_draft_tokens"] = 0   # AR: no active draft configuration
        gen = Generator(**kwargs)
        report["generator"] = {"mode": args.mode, "mtp_draft": gen.mtp_draft,
                               "num_draft_tokens": gen.num_draft_tokens, "dynamic_draft": gen.dynamic_draft,
                               "ngram_match_min": gen.ngram_match_min, "record_draft_stats": gen.record_draft_stats}
        report["cache_capacity"] = {"main": {"num_slots": cache.num_slots, "max_num_tokens": cache.max_num_tokens,
                                             "max_history": cache.max_history},
                                    "draft": {"num_slots": dcache.num_slots, "max_num_tokens": dcache.max_num_tokens}}
        assert bool(gen.mtp_draft) == (args.mode == "mtp")
        assert gen.num_draft_tokens == (args.draft_tokens if args.mode == "mtp" else 0)
        assert gen.ngram_match_min == 0 and gen.record_draft_stats
        assert gen.dynamic_draft == args.dynamic_draft
        if args.mode == "mtp":
            assert gen.draft_model is draft and gen.draft_cache is dcache
        forward_checks = {"target": 0, "draft": 0}
        if args.validate_finite:
            def wrap(module, key):
                orig = module.forward
                def call(input_ids, params=None, **kw):
                    y = orig(input_ids, params, **kw)
                    if isinstance(y, torch.Tensor) and not torch.isfinite(y).all().item():
                        raise RuntimeError(f"{key} forward produced non-finite output")
                    forward_checks[key] += 1
                    return y
                module.forward = call
            wrap(model, "target")
            wrap(draft, "draft")
        with power_policy.attach(gen, torch, args.batch_size, [0, 1], args.power_socket) as policy:
            for gi in range(0, len(prompts), args.batch_size):
                states = []
                for p in prompts[gi:gi + args.batch_size]:
                    job = Job(input_ids=torch.tensor([p["ids"]], dtype=torch.long),
                              max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                              sampler=ArgmaxSampler())
                    gen.enqueue(job)
                    states.append({"prompt": p, "job": job, "prev": 0, "events": [],
                                   "ids_out": [], "text": [], "last": None})
                t0, unix0 = time.perf_counter(), time.time()
                by_job = {id(s["job"]): s for s in states}
                while gen.num_remaining_jobs():
                    items = gen.iterate()
                    t = time.perf_counter() - t0   # sample AFTER progress, like the accepted sweep
                    for s in states:
                        n = int(s["job"].new_tokens)
                        if n > s["prev"]:
                            s["events"].append([t, n])
                            s["prev"] = n
                    for item in items:
                        if item.get("stage") == "error":
                            raise RuntimeError(f"job error: {item}")
                        s = by_job.get(id(item.get("job")))
                        if item.get("stage") == "streaming" and s is not None:
                            s["text"].append(item.get("text", ""))
                            if item.get("token_ids") is not None:
                                s["ids_out"].extend(item["token_ids"].reshape(-1).tolist())
                            if item.get("eos"):
                                s["last"] = item
                wall = time.perf_counter() - t0
                row_refs = []
                for s in states:
                    p, last, ev = s["prompt"], s["last"], s["events"]
                    if last is None:
                        raise RuntimeError(f"prompt sha={p['sha']} finished without an eos result; incomplete")
                    if (last.get("new_tokens") != args.new_tokens
                            or last.get("prompt_tokens") != len(p["ids"])
                            or last.get("cached_tokens") != 0):
                        raise RuntimeError(f"prompt sha={p['sha']}: new_tokens={last.get('new_tokens')} "
                                           f"prompt_tokens={last.get('prompt_tokens')} "
                                           f"cached_tokens={last.get('cached_tokens')} violate exact/"
                                           f"cache-miss requirements")
                    if not ev or ev[-1][1] != args.new_tokens:
                        raise RuntimeError(f"prompt sha={p['sha']}: delivery events {ev[-1:]} never reached "
                                           f"{args.new_tokens} cumulative tokens")
                    row = {"mode": args.mode, "language": p["language"], "repeat": p["repeat"],
                           "timed": p["timed"], "ids_sha256": p["sha"],
                           **{k: last.get(k) for k in RESULT_KEYS}}
                    row.update(draft_stats=s["job"].draft_stats, delivery_events=ev,
                               first_delivery_s=ev[0][0], stream_window_s=ev[-1][0],
                               tokens=s["ids_out"], text="".join(s["text"]),
                               observed_decode_tps=delivery_rate(ev),
                               legacy_engine_decode_tps=(args.new_tokens - 1) / last["time_generate"]
                               if last.get("time_generate") else None)
                    report["runs"].append(row)
                    row_refs.append(row["ids_sha256"])
                    print(f"{args.mode} {p['language']} rep={p['repeat']} timed={p['timed']} "
                          f"tokens={row['new_tokens']} engine_tps={row['legacy_engine_decode_tps']:.3f} "
                          f"observed_tps={row['observed_decode_tps']}", flush=True)
                report["groups"].append({"group": gi // args.batch_size, "timed": prompts[gi]["timed"],
                                         "jobs": len(states), "wall_s": wall,
                                         "wall_start_unix_s": unix0,
                                         "total_new_tokens": len(states) * args.new_tokens,
                                         "ids_sha256": row_refs})
        report["power_policy"] = policy.summary()
        report["finite_forward_counts"] = forward_checks if args.validate_finite else None
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED: {report['error']}", file=sys.stderr)
    finally:
        cleanup = []
        if policy is not None:
            report["power_policy"] = policy.summary()
        for name, obj in (("draft", draft), ("model", model)):
            if obj is not None:
                try:
                    obj.unload()
                except Exception as e:
                    cleanup.append(f"unload {name}: {e!r}")
        report["cleanup_errors"] = cleanup
        if cleanup:
            report["complete"] = False
        try:
            report["peak_allocated_bytes"] = {
                f"cuda:{d}": torch.cuda.max_memory_allocated(d)
                for d in range(min(2, torch.cuda.device_count()))}
        except Exception as e:
            report["complete"] = False
            cleanup.append(f"memory reporting: {e!r}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["complete"] else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        prompts = validate_prompts(json.loads(Path(args.prompts_json).read_text(encoding="utf-8")),
                                   args.batch_size)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        return run(args, prompts)
    except ImportError as e:
        print(f"error: native exllamav3/torch stack not importable (it is selected externally "
              f"via PYTHONPATH): {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
