#!/usr/bin/env python3
"""End-to-end throughput for the single V620 (gfx1030), backend=exl3 only.

Basis: rocm_tools/bench_model.py (measurement conventions: median of repeats,
5% noise floor flag, regenerate prompts per repeat so the prefix cache misses)
and the examples Generator/Job API -- hardened for Phase 2 acceptance:

  * Hard GPU identity gate: torch.version.hip must be set, exactly one GPU
    must be visible, and its gcnArchName must equal --expect-arch (gfx1030).
    Any mismatch exits nonzero before loading anything.
  * Explicit device: the model loads with model.load(device=...) on --device,
    and only cuda:0 is accepted. --max-chunk-size bounds prefill chunking.
  * Speculative decoding is disabled and asserted impossible: no draft model,
    n-gram drafting off, generator.num_draft_tokens == 0.
  * Exact fresh inputs: prompts are exactly --contexts tokens of fresh random
    in-vocab IDs from a seeded CPU generator, never reused => prefix-cache
    miss. Every job must report prompt_tokens == requested length and
    cached_tokens == 0; a cache hit or short prompt fails that row (reported,
    nonzero exit) instead of silently emitting inflated speed.
  * Exact generated length (bench mode): Job(min_new_tokens=max_new_tokens)
    suppresses stop tokens until the requested length is produced (the sampler
    option that supports the requested length), and the job must report
    new_tokens == --new-tokens with eos_reason == "max_new_tokens". A job that
    finishes for any other reason fails the run -- no misleading numbers.
  * smoke mode: same inputs/enforcement, one run per job, but an early EOS is
    ALLOWED and labeled (early_stop=True + eos_reason), since smoke checks that
    the stack runs, not speed.
  * Missing job results or "stage: error" results fail truthfully.
  * Peak device memory per timed job via torch.cuda.reset_peak_memory_stats /
    max_memory_allocated, plus allocated-after-load baseline.
  * Per-run rows + median/spread summaries go to --json-out together with the
    repo git commit, model fingerprint and full env block (including
    EXL3_ROCM_* switches, so conservative vs optimized executions are
    distinguishable post-hoc).
  * Normal cleanup: model.unload() runs in a finally block; a teardown failure
    is printed, recorded in "failures" and forces a nonzero exit -- never
    silently swallowed. No os._exit: JSON is written and fsynced before normal
    process exit. NOTE for the orchestrator: if native teardown segfaults after
    the JSON file and RESULT line were emitted (this fork documents such
    crashes on some RDNA builds; see bench_model.py header), judge by the
    artifacts.

Examples
--------
    # smoke: one run per job, early EOS allowed but labeled
    # (EXL3_BC_ATTN=0: runtime agent found the default EXL3 BC attention hangs
    #  module load on this stack; the switch is recorded in the artifact env)
    EXL3_BC_ATTN=0 /opt/venv/bin/python rocm_tools/rdna2/bench.py \
        -m /work/models/qwen3-8b-exl3-4bpw --mode smoke \
        --json-out /work/phase0/bench_smoke.json

    # bench: warmup 1 + 5 timed repeats, prefill+decode at 512/2048/8192, out 256
    EXL3_BC_ATTN=0 /opt/venv/bin/python rocm_tools/rdna2/bench.py \
        -m /work/models/qwen3-8b-exl3-4bpw --mode bench \
        --contexts 512 2048 8192 --new-tokens 256 --repeats 5 --warmup 1 \
        --seed 1234 --json-out /work/phase0/bench_qwen3_8b_exl3.json
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rocm_tools.rdna2.common import (
    BENCH_FORMAT,
    git_commit,
    model_fingerprint,
    now_utc,
    python_env,
    repo_root,
    rocm_patch_env,
    round_up_page,
    torch_env,
    visible_gpu_env,
    write_json,
)

NOISE_FLOOR = 0.05   # same convention as bench_model.py


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog = "bench.py",
        description = "Single-V620 exl3 end-to-end prefill/decode throughput + smoke.",
    )
    ap.add_argument("-m", "--model-dir", required = True)
    ap.add_argument("--mode", choices = ("bench", "smoke"), default = "bench",
                    help = "bench: strict length enforcement, median of repeats; "
                           "smoke: one run per job, early EOS allowed but labeled")
    ap.add_argument("-d", "--device", default = "cuda:0",
                    help = "must be cuda:0 (single-GPU box; validated)")
    ap.add_argument("--contexts", type = int, nargs = "+", default = [512, 2048, 8192],
                    help = "total context sizes; each job gets a fresh prompt of exactly N tokens")
    ap.add_argument("--new-tokens", type = int, default = 256,
                    help = "generated tokens for decode runs (bench mode requires exactly this many)")
    ap.add_argument("--repeats", type = int, default = 5, help = "timed repeats per job (bench)")
    ap.add_argument("--warmup", type = int, default = 1, help = "untimed warmup runs per job (bench)")
    ap.add_argument("--max-chunk-size", type = int, default = 2048,
                    help = "prefill chunk size for model.load()/Generator")
    ap.add_argument("--cache-tokens", type = int, default = None,
                    help = "override KV cache token count (default: auto, >= max(ctx)+out)")
    ap.add_argument("--seed", type = int, default = 1234)
    ap.add_argument("--expect-arch", default = "gfx1030")
    ap.add_argument("--json-out", default = "rdna2_bench_results.json")
    return ap


def validate_gpu(torch, device_str: str, expect_arch: str) -> dict:
    """The Phase-2 environment gate. Raises SystemExit with a clear message."""
    if getattr(torch.version, "hip", None) is None:
        raise SystemExit(" !! FATAL: torch build has no ROCm (torch.version.hip is None); "
                         "run inside the rocm-exl3-rdna2 container venv")
    if device_str not in ("cuda:0", "cuda"):
        raise SystemExit(f" !! FATAL: --device must be cuda:0 (got {device_str!r})")
    n = torch.cuda.device_count()
    if n != 1:
        raise SystemExit(f" !! FATAL: expected exactly one visible GPU, found {n}. "
                         f"Check CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES "
                         f"(env: {visible_gpu_env()})")
    if not torch.cuda.is_available():
        raise SystemExit(" !! FATAL: torch.cuda.is_available() is False for the single visible device")
    props = torch.cuda.get_device_properties(0)
    gcn = getattr(props, "gcnArchName", "") or ""
    if gcn != expect_arch:
        raise SystemExit(f" !! FATAL: gcnArchName is {gcn!r}, expected {expect_arch!r}: "
                         f"this harness only measures the {expect_arch} V620")
    return {
        "name": props.name,
        "gcnArchName": gcn,
        "total_memory_bytes": getattr(props, "total_memory", None),
        "multi_processor_count": getattr(props, "multi_processor_count", None),
        "torch_version_hip": torch.version.hip,
        "visible_devices": n,
    }


def fresh_prompt(torch, n: int, vocab: int, rng) -> "torch.Tensor":
    """
    Exactly n random in-vocab IDs on CPU (Job requires system-memory input ids),
    from an advancing seeded generator so no two runs share a prompt => the
    prefix cache misses every timed run. Range mirrors bench_model.py (5%-95%
    of actual vocab) to avoid special/BOS/EOS pieces inside the prompt.
    """
    lo, hi = int(vocab * 0.05), int(vocab * 0.95)
    return torch.randint(lo, hi, (1, n), dtype = torch.long, generator = rng)


def run_job(generator, Job, ids, max_new: int, min_new: int, seed: int) -> dict:
    """
    Drive one Job to completion through the standard examples loop
    (enqueue -> iterate until num_remaining_jobs()==0). Returns the final eos
    result dict; raises RuntimeError on a job error or a drained queue that
    never produced one. Never returns a partial 'success'.
    """
    from exllamav3.generator.sampler import ArgmaxSampler

    job = Job(input_ids = ids, max_new_tokens = max_new, min_new_tokens = min_new,
              sampler = ArgmaxSampler(), seed = seed)
    generator.enqueue(job)
    final = None
    err = None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            stage = r.get("stage")
            if stage == "error":
                err = r.get("error")
            elif stage == "streaming" and r.get("eos"):
                final = r
    if final is None:
        raise RuntimeError(f"job produced no completion result (error={err!r})")
    return final


def summarize(samples: list[float]) -> dict:
    med = statistics.median(samples)
    spread = (max(samples) - min(samples)) / med if med else 0.0
    return {
        "n": len(samples),
        "median": med,
        "min": min(samples),
        "max": max(samples),
        "spread_rel": spread,
        "flag_exceeds_noise_floor": spread > NOISE_FLOOR,
    }


def run(args) -> int:
    # GPU-facing imports happen only after argparse (so --help works on the host).
    import torch
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer

    gpu = validate_gpu(torch, args.device, args.expect_arch)
    if args.new_tokens < 1 or not args.contexts:
        raise SystemExit(" !! FATAL: --new-tokens must be >= 1 and --contexts non-empty")
    contexts = sorted(set(args.contexts))
    repeats = max(1, args.repeats) if args.mode == "bench" else 1
    warmups = max(0, args.warmup) if args.mode == "bench" else 0
    dev = torch.device(args.device)

    runs: list[dict] = []
    failures: list[str] = []
    load_s = None
    mem_after_load = None
    cache_tokens = None
    model = None

    # Everything that touches the device is inside try; model.unload() runs in
    # finally and its failure is recorded (never swallowed silently).
    try:
        config = Config.from_directory(args.model_dir)
        model = Model.from_config(config)
        tokenizer = Tokenizer.from_config(config)
        vocab = int(tokenizer.actual_vocab_size)

        need = max(contexts) + args.new_tokens + 256
        # Floor 4096 (and max_chunk_size): model.load runs a dummy forward at
        # max_chunk_size and asserts cache capacity -- see bench_model.py.
        cache_tokens = round_up_page(max(args.cache_tokens or 0, need, 4096, args.max_chunk_size))
        cache = Cache(model, max_num_tokens = cache_tokens)   # BEFORE model.load()

        t0 = time.time()
        model.load(device = args.device, max_chunk_size = args.max_chunk_size, progressbar = False)
        load_s = time.time() - t0
        mem_after_load = torch.cuda.memory_allocated(dev)
        print(f" -- loaded {args.model_dir} in {load_s:.1f}s on {args.device}; "
              f"cache {cache_tokens} tokens; gpu {gpu['name']} ({gpu['gcnArchName']}, "
              f"hip {gpu['torch_version_hip']})", flush = True)

        generator = Generator(
            model = model, cache = cache, tokenizer = tokenizer,
            max_batch_size = 1,
            max_chunk_size = args.max_chunk_size,
            ngram_match_min = 0,          # explicit: no speculative decoding
        )
        # Assert, don't assume: spec decode must be structurally impossible here.
        spec_off = (generator.draft_model is None and generator.draft_cache is None
                    and generator.ngram_match_min == 0 and generator.num_draft_tokens == 0)
        if not spec_off:
            raise SystemExit(" !! FATAL: generator has draft/n-gram spec decoding active; "
                             "this harness must measure plain AR decode")

        rng = torch.Generator().manual_seed(args.seed)

        def one_job(phase: str, ctx: int, rep: int, timed: bool) -> dict:
            """Run one measured/checked job; appends its row to runs; raises on problems."""
            ids = fresh_prompt(torch, ctx, vocab, rng)
            sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            deliv = args.new_tokens if phase == "decode" else 1
            # bench mode: stop tokens suppressed until the full requested length
            # is delivered (Job min_new_tokens), so exact-length is enforceable.
            min_new = deliv if (args.mode == "bench" and phase == "decode") else 0
            if timed:
                torch.cuda.reset_peak_memory_stats(dev)
            wall0 = time.perf_counter()
            res = run_job(generator, Job, ids, max_new = deliv, min_new = min_new,
                          seed = args.seed + rep)
            wall = time.perf_counter() - wall0
            row = {
                "phase": phase, "context": ctx, "repeat": rep, "timed": timed,
                "ids_sha256": sha,
                "seed": args.seed + rep,
                "prompt_tokens_expected": ctx,
                "prompt_tokens": res.get("prompt_tokens"),
                "cached_tokens": res.get("cached_tokens"),
                "requested_new_tokens": deliv,
                "new_tokens": res.get("new_tokens"),
                "eos_reason": res.get("eos_reason"),
                "ttft_ms": res.get("time_prefill", 0.0) * 1000.0,
                "time_generate_s": res.get("time_generate"),
                "wall_s": wall,
                "peak_mem_bytes": (torch.cuda.max_memory_allocated(dev) if timed else None),
            }
            problems = []
            if row["prompt_tokens"] != ctx:
                problems.append(f"prompt_tokens {row['prompt_tokens']} != requested {ctx} "
                                f"(short requested token count)")
            if row["cached_tokens"]:
                problems.append(f"PREFIX CACHE HIT (cached_tokens={row['cached_tokens']}): "
                                f"this row does not measure a cold prefill")
            if row["new_tokens"] is None or row["new_tokens"] < 1:
                problems.append(f"new_tokens={row['new_tokens']}")
            if phase == "decode":
                if row["new_tokens"] != deliv:
                    problems.append(f"new_tokens {row['new_tokens']} != requested {deliv}")
                early = bool(row["eos_reason"] != "max_new_tokens")
                row["early_stop"] = early
                if early:
                    if args.mode == "bench":
                        problems.append(f"decode stopped early: eos_reason={row['eos_reason']!r} "
                                        f"in bench mode")
                    else:
                        row["label"] = (f"SMOKE: ended at eos_reason={row['eos_reason']!r} "
                                        f"after {row['new_tokens']}/{deliv} tokens (labeled, not a failure)")
            if row["ttft_ms"] <= 0:
                problems.append("time_prefill missing/zero -- TTFT unmeasurable")
            if phase == "prefill":
                row["prefill_tps"] = ctx / (res["time_prefill"] + 1e-10)
            else:
                tg = res["time_generate"] or 0.0
                nt = row["new_tokens"]
                row["tpot_ms"] = (tg / max(nt - 1, 1)) * 1000.0
                row["decode_tps"] = max(nt - 1, 1) / (tg + 1e-10)
                row["e2e_tps"] = nt / (res["time_prefill"] + tg + 1e-10)
            row["problems"] = problems
            runs.append(row)
            if problems:
                raise AssertionError("; ".join(problems))
            return row

        for phase in ("prefill", "decode"):
            for ctx in contexts:
                label = f"{phase}@{ctx}"
                try:
                    for rep in range(warmups):
                        one_job(phase, ctx, rep, timed = False)
                    rates = []
                    for rep in range(warmups, warmups + repeats):
                        r = one_job(phase, ctx, rep, timed = True)
                        rates.append(r["prefill_tps"] if phase == "prefill" else r["decode_tps"])
                    med = statistics.median(rates)
                    spread = (max(rates) - min(rates)) / med if med else 0.0
                    flag = "  <- exceeds 5% noise floor, rerun" if spread > NOISE_FLOOR else ""
                    print(f" -- {label:16} ok  median {med:10.1f} t/s   spread {spread:5.1%}{flag}",
                          flush = True)
                except Exception as e:
                    failures.append(f"{label}: {e}")
                    print(f" -- {label:16} FAIL: {e}", flush = True)
                    if not isinstance(e, (AssertionError, RuntimeError, SystemExit)):
                        traceback.print_exc()
    finally:
        if model is not None:
            try:
                model.unload()
            except Exception as e:
                msg = f"model.unload() during cleanup: {e!r}"
                failures.append(msg)
                print(f" !! {msg}", file = sys.stderr)

    # ---- summary (after cleanup so teardown failures are in the artifact) ----
    summary = []
    for phase in ("prefill", "decode"):
        for ctx in contexts:
            rows = [r for r in runs if r["timed"] and r["phase"] == phase and r["context"] == ctx]
            if not rows:
                continue
            entry = {"phase": phase, "context": ctx, "repeats": len(rows),
                     "ttft_ms": summarize([r["ttft_ms"] for r in rows])}
            if phase == "prefill":
                entry["prefill_tps"] = summarize([r["prefill_tps"] for r in rows])
            else:
                entry["tpot_ms"] = summarize([r["tpot_ms"] for r in rows])
                entry["decode_tps"] = summarize([r["decode_tps"] for r in rows])
                entry["e2e_tps"] = summarize([r["e2e_tps"] for r in rows])
            entry["peak_mem_bytes_max"] = max(r["peak_mem_bytes"] for r in rows)
            summary.append(entry)

    exit_code = 0 if not failures else 1
    out = {
        "format": BENCH_FORMAT,
        "mode": args.mode,
        "created_utc": now_utc(),
        "tool": "rocm_tools/rdna2/bench.py",
        "ok": exit_code == 0,
        "failures": failures,
        "params": vars(args),
        "seed": args.seed,
        "commit": git_commit(repo_root()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "model_fingerprint": model_fingerprint(args.model_dir),
        "load_s": load_s,
        "cache_tokens": cache_tokens,
        "memory": {"after_load_allocated_bytes": mem_after_load},
        "gpu_gate": gpu,
        "spec_decode": {"enabled": False, "asserted": True,
                        "draft_model": None, "ngram_match_min": 0},
        "sampler": "ArgmaxSampler (greedy, deterministic; min_new_tokens=max_new_tokens in bench mode)",
        "fresh_prompt": "exactly N random IDs in [5%,95%) of actual vocab per run from seeded CPU rng",
        "formulas": {
            "ttft": "job time_prefill (first prefill start -> first token)",
            "tpot": "time_generate / (new_tokens - 1)",
            "prefill_tps": "context_tokens / time_prefill",
            "decode_tps": "(new_tokens - 1) / time_generate",
            "e2e_tps": "new_tokens / (time_prefill + time_generate)",
            "spread_rel": "(max - min) / median over timed repeats",
        },
        "env": {
            "python": python_env(),
            "torch": torch_env(),
            "visible_gpu": visible_gpu_env(),
            "exl3_rocm_switches": rocm_patch_env(),
        },
        "runs": runs,
        "summary": summary,
    }
    write_json(args.json_out, out)
    print(f"\n  RESULT {'ok' if exit_code == 0 else 'FAIL'}: "
          f"{len(runs)} jobs run, {len(failures)} failed groups; "
          f"JSON: {args.json_out} (written before normal exit)", flush = True)
    sys.stdout.flush()
    return exit_code


def main(argv = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        print(f" !! bench aborted: {e}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
