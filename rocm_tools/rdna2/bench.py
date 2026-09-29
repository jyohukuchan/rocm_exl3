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
  * Per-token inter-token latency (ITL): generator.iterate() completions are
    timestamped with time.perf_counter and increases in job.new_tokens are
    observed (batch=1 and spec decode off are asserted above, so plain AR
    advances +1 per round). The first token is excluded (its dt carries
    prefill/TTFT), so N generated tokens yield N-1 positive interval
    samples. Each timed decode run stores the raw samples plus nearest-rank
    p50/p95; the summary merges the raw samples of all timed runs into one
    distribution. This is the empirical per-token distribution, distinct
    from the per-job mean TPOT (time_generate/(new_tokens-1)). No extra
    torch.cuda.synchronize per token: it would perturb what is measured.
    Multi-token jumps are impossible for plain AR and are reported as
    itl_bursts that fail the row, never averaged into fake ITL samples.
  * smoke mode: same inputs/enforcement, one run per job, but an early EOS is
    ALLOWED and labeled (early_stop=True + eos_reason), since smoke checks that
    the stack runs, not speed.
  * Missing job results or "stage: error" results fail truthfully.
  * Peak device memory per timed job via torch.cuda.reset_peak_memory_stats /
    max_memory_allocated, plus allocated-after-load baseline.

Phase 3 -- dual-V620 layer split (rocm_tools/rdna2/multi_gpu.py):
  * EXL3 autosplit, NOT tensor parallelism: --use-per-device GIB GIB ... is
    the only sanctioned split path here. It loads through the official
    model.load(use_per_device=[...], max_chunk_size=...) API WITHOUT a device
    argument (the engine forbids combining them). At least two finite positive
    GiB budgets, mapping 1:1 to visible order (cuda:0, cuda:1, ...). Without
    the flag the single-GPU path is untouched: exactly one visible GPU is
    still required, and on a multi-device box the run refuses with a pointer
    to --use-per-device instead of silently switching modes.
  * Split-mode gate (before loading): ROCm torch, visible device count EQUALS
    the budget count, and EVERY visible device's gcnArchName equals
    --expect-arch. PCI/uuid/name/memory/index of each device is captured into
    the artifact. No monkeypatching anywhere.
  * After the split load the placement is AUDITED, not assumed: every module's
    and cache layer's actual device attribute is inspected recursively
    (modules, cache layers, their backing k/v tensors, the loader's
    active_devices), both devices must own real transformer modules (by
    capability/key pattern, not layer_idx) and cache layers, and the device
    progression in forward order must be contiguous. The ordered placement
    record and per-device memory land in the artifact ("layer_split"); an
    audit failure aborts measurement (nonzero, JSON written) -- throughput
    numbers are never emitted for an unverified split.
  * In split mode torch.cuda.synchronize + reset_peak_memory_stats run on ALL
    used devices at job boundaries only (never per token), and every job row
    carries per-device peak/allocated/reserved plus the delta of the engine's
    device_copy transfer counters (direct/bounced/probes) for that job.
    All single-GPU output keys are preserved unchanged.
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

Phase 5 -- Qwen3.8-Flash-Next (qwen4_exp) benchmark trustworthiness:
  * --ngram-ram forces the PLE n-gram embedding table into HOST RAM by
    setting config.infer_params.ngram_stream_from_disk = False before load
    (the model_init --ngram_ram lever; engine default otherwise stays
    EXL3_NGRAM_STREAM=1 disk-streamed, unchanged when the flag is absent).
    The assumption is then VERIFIED, not trusted: every NGramEmbedding in
    the live model is duck-typed (multi_gpu.collect_ngram_state) and its
    actual load mode ("trellis_ram"/"fp16_ram" vs the disk modes), table
    tensor residence (str(tensor.device)) and metadata bytes, disk-handle
    count/filenames and cumulative prefetch_stats are recorded. Requesting
    --ngram-ram on a table-bearing model whose tables did NOT end up
    host-RAM-resident fails the run closed (nonzero, JSON written, no
    throughput measured). Tableless models get a recorded no-op note, not a
    failure. Residence is evidenced by the tensors themselves plus process
    RSS/page-fault snapshots (resource.getrusage + /proc/self/statm, taken
    before load / after load / after inference); CUDA "reserved" is
    allocator state on the board and is NEVER equated with board total or
    used as residence evidence. The four load modes return identical
    results for the same table contents (per exllamav3/modules/
    ngram_embedding.py), so this measures the same model, not a different
    one.
  * The placement audit understands the hybrid architecture: GDN recurrent
    modules and their local conv_state/recurrent_state tensors, PLE layers
    with host-side id_state, and the QSA indexer planes (raw_k/pooled) on
    attention cache layers are inspected through direct attributes only (no
    arbitrary-graph traversal, no GPU .item sync), recurrent layer counts
    are recorded per device, and a device holding only GDN/PLE layers is
    not rejected for missing ordinary KV cache layers (dense Qwen3 D/M
    verdicts are unchanged).
  * Still NO speculation anywhere: draft model/n-gram drafting stay off and
    generator.num_draft_tokens == 0 is asserted; --ngram-ram changes table
    STORAGE only, never decoding.

Examples
--------
    # smoke: one run per job, early EOS allowed but labeled
    # (plain defaults: EXL3 BC attention ON is verified working on this stack;
    #  EXL3_BC_ATTN=0 is a diagnostic switch only -- see rdna2/README.md.
    #  Keep other GPU workloads off the card while measuring; the untimed
    #  warmup runs absorb cold JIT/module-load time so it stays out of medians)
    /opt/venv/bin/python rocm_tools/rdna2/bench.py \
        -m /work/models/qwen3-8b-exl3-4bpw --mode smoke \
        --json-out /work/phase0/bench_smoke.json

    # bench: warmup 1 + 5 timed repeats, prefill+decode at 512/2048/8192, out 256
    # (decode rows also carry raw per-token ITL samples + p50/p95)
    /opt/venv/bin/python rocm_tools/rdna2/bench.py \
        -m /work/models/qwen3-8b-exl3-4bpw --mode bench \
        --contexts 512 2048 8192 --new-tokens 256 --repeats 5 --warmup 1 \
        --seed 1234 --json-out /work/phase0/bench_qwen3_8b_exl3.json

    # Phase 3: explicit GiB budgets forcing the split point (root-verified for
    # Qwen3-8B exl3-4bpw on the V620 pair: [3, 4] -> 21/15 transformer layers,
    # cache k/v tensors local). Placement is audited after load; the run
    # aborts nonzero if either device owns no real transformer modules.
    /opt/venv/bin/python rocm_tools/rdna2/bench.py \
        -m /work/models/qwen3-8b-exl3-4bpw --mode bench \
        --use-per-device 3 4 \
        --json-out /work/phase3/bench_ls_qwen3_8b.json

    # Phase 5: Qwen3.8-Flash-Next on the V620 pair with the PLE n-gram table
    # held in host RAM (sets config.infer_params.ngram_stream_from_disk=False
    # before load; actual mode/residence/bytes, prefetch stats and RSS +
    # major/minor fault deltas land in the JSON; the run fails closed if a
    # table-bearing model did NOT end up RAM-backed). Hybrid GDN/QSA
    # placement is audited with per-device recurrent-layer state counts.
    /opt/venv/bin/python rocm_tools/rdna2/bench.py \
        -m /work/models/qwen3.8-flash-next-exl3 --mode bench \
        --use-per-device 12 14 --ngram-ram \
        --json-out /work/phase5/bench_ls_qwen38.json
"""

from __future__ import annotations

import argparse
import hashlib
import math
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
from rocm_tools.rdna2 import multi_gpu   # stdlib-only at import (CPU-safe)

NOISE_FLOOR = 0.05   # same convention as bench_model.py


class _SkipJobs(Exception):
    """Internal control flow: load happened but MUST not be measured (e.g.
    placement audit failed). The try/finally below still unloads the model and
    the JSON artifact is still written with the failures recorded."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog = "bench.py",
        description = "exl3 end-to-end prefill/decode throughput + smoke "
                      "(single gfx1030 V620, or audited layer split across "
                      "multiple visible V620s).",
    )
    ap.add_argument("-m", "--model-dir", required = True)
    ap.add_argument("--mode", choices = ("bench", "smoke"), default = "bench",
                    help = "bench: strict length enforcement, median of repeats; "
                           "smoke: one run per job, early EOS allowed but labeled")
    ap.add_argument("-d", "--device", default = "cuda:0",
                    help = "must be cuda:0 (single-GPU box; validated; unused in layer-split mode)")
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
    ap.add_argument("--use-per-device", type = float, nargs = "+", default = None,
                    metavar = "GIB",
                    help = "EXL3 layer split (not TP): explicit GiB budgets passed to the official "
                           "model.load(use_per_device=[...]) autosplit API WITHOUT a device arg. "
                           "At least two finite positive values, one per visible GPU in cuda:0, "
                           "cuda:1... order, and the visible device count must equal it. Without "
                           "this flag the single-GPU path is unchanged (exactly one visible GPU "
                           "required; multi-device boxes are refused with a pointer to this flag)")
    ap.add_argument("--ngram-ram", action = "store_true",
                    help = "hold the PLE n-gram embedding table in host RAM: sets "
                           "config.infer_params.ngram_stream_from_disk = False before load "
                           "(the model_init --ngram_ram lever; equivalent to EXL3_NGRAM_STREAM=0). "
                           "Legacy default is untouched without this flag (engine env default, "
                           "disk-streamed). The ACTUAL table mode/residence/bytes and prefetch "
                           "stats are audited from the live modules after load; requesting this "
                           "flag on a table-bearing model that did not end up RAM-backed fails "
                           "the run (nonzero, JSON written). Storage location only: the four "
                           "load modes return identical results, so decoding and this harness's "
                           "spec-decode assertions are unaffected.")
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
                         f"(env: {visible_gpu_env()})"
                         + (f" -- for an audited multi-GPU layer split pass "
                            f"--use-per-device GIB GIB ... instead" if n > 1 else ""))
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


def run_job(generator, Job, ids, max_new: int, min_new: int,
            seed: int) -> tuple[dict, list]:
    """
    Drive one Job to completion through the standard examples loop
    (enqueue -> iterate until num_remaining_jobs()==0). Returns
    (final_eos_result, token_events), where token_events is a list of
    (time.perf_counter() taken immediately after each generator.iterate()
    completes, job.new_tokens) pairs, recorded only when the cumulative token
    count grew. batch=1 with spec decode asserted off means plain AR advances
    job.new_tokens by exactly 1 per round; anything else is surfaced later as
    an itl_burst, not smeared across fake per-token samples. Deliberately no
    extra torch.cuda.synchronize per token: iterate() has already synced on
    its own output path, and adding a sync would perturb the latency being
    measured. Raises RuntimeError on a job error or a drained queue that never
    produced a completion result. Never returns a partial 'success'.
    """
    from exllamav3.generator.sampler import ArgmaxSampler

    job = Job(input_ids = ids, max_new_tokens = max_new, min_new_tokens = min_new,
              sampler = ArgmaxSampler(), seed = seed)
    generator.enqueue(job)
    final = None
    err = None
    events: list[tuple[float, int]] = []
    prev_new = 0          # Job.new_tokens starts at 0 (this harness sets no prefix_token)
    while generator.num_remaining_jobs():
        results = generator.iterate()
        t_done = time.perf_counter()
        n = int(job.new_tokens)
        if n > prev_new:
            events.append((t_done, n))
            prev_new = n
        for r in results:
            stage = r.get("stage")
            if stage == "error":
                err = r.get("error")
            elif stage == "streaming" and r.get("eos"):
                final = r
    if final is None:
        raise RuntimeError(f"job produced no completion result (error={err!r})")
    return final, events


def observed_timing(events, wall_start: float) -> dict:
    """Actual first-token latency and subsequent token time, using the ITL clock.

    The generator's legacy time_first_token is set before its first decode
    forward, so time_generate can include a slow first-token initialization.
    Keep those legacy fields for comparison, and expose both observed stages.
    """
    if not events or events[0][0] <= wall_start:
        raise ValueError("missing or nonpositive first-token observation")
    decode_s = events[-1][0] - events[0][0]
    return {
        "first_token_wall_ms": (events[0][0] - wall_start) * 1000,
        "decode_observed_s": decode_s,
        "decode_observed_tps": ((len(events) - 1) / decode_s
                                if len(events) > 1 and decode_s > 0 else None),
    }


def percentile(samples: list[float], q: float) -> float:
    """
    Nearest-rank percentile: the smallest sample value v such that at least
    q percent of the samples are <= v (q in percent, 0 < q <= 100).
    Deterministic and interpolation-free, so the result is always an actually
    observed sample -- which is what "p50/p95 of the raw ITL samples" means
    here. q*len is an exact integer product divided by 100, so an exact rank
    never wobbles across the ceil boundary.
    """
    if not samples:
        raise ValueError("percentile() needs at least one sample")
    if not (0.0 < q <= 100.0):
        raise ValueError(f"q must be in (0, 100], got {q}")
    ordered = sorted(samples)
    idx = math.ceil(q * len(ordered) / 100.0) - 1      # 1-based rank -> 0-based index
    return ordered[min(max(idx, 0), len(ordered) - 1)]


def token_intervals(events: list[tuple[float, int]]) -> tuple[list[float], list[dict]]:
    """
    Turn run_job's (iterate-completion timestamp, cumulative job.new_tokens)
    observations into (raw ITL samples in ms, burst records).

    The first observation anchors the first generated token: the wall time up
    to it includes prefill/TTFT, so it contributes NO sample. Every later
    strictly positive interval of a +1 increment is one ITL sample (plain AR,
    N tokens -> N-1 samples). An increment of != 1 tokens cannot be split
    into honest per-token latencies, so it is recorded as a burst dict and
    contributes NO averaged fake sample; the caller must report bursts rather
    than label them as ITLs.
    """
    samples_ms: list[float] = []
    bursts: list[dict] = []
    prev_t = None
    prev_n = 0
    for t, n in events:
        delta = n - prev_n
        if delta == 1 and prev_t is not None:
            dt = t - prev_t
            if dt > 0:
                samples_ms.append(dt * 1000.0)
        elif delta != 1:
            bursts.append({
                "from_new_tokens": prev_n,
                "to_new_tokens": n,
                "delta": delta,
                "dt_ms": (None if prev_t is None else (t - prev_t) * 1000.0),
            })
        prev_t, prev_n = t, n
    return samples_ms, bursts


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


def process_memory_snapshot() -> dict:
    """
    Host-process memory/fault accounting for the --ngram-ram assumption check.
    rss_peak_bytes is resource.getrusage(RUSAGE_SELF).ru_maxrss (Linux kB ->
    bytes; peak over the process lifetime); rss_bytes is the CURRENT resident
    set from /proc/self/statm (pages * page size); major/minor_faults are the
    cumulative rusage counters. Anything unreadable stays None -- never
    guessed, never inferred from CUDA allocator state (reserved/allocated on
    the board says nothing about host RAM residence or board capacity).
    """
    out = {"rss_bytes": None, "rss_peak_bytes": None,
           "major_faults": None, "minor_faults": None}
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        out["rss_peak_bytes"] = int(ru.ru_maxrss) * 1024
        out["major_faults"] = int(ru.ru_majflt)
        out["minor_faults"] = int(ru.ru_minflt)
        try:
            with open("/proc/self/statm", "r") as f:
                resident_pages = int(f.read().split()[1])
            out["rss_bytes"] = resident_pages * resource.getpagesize()
        except Exception:
            pass
    except Exception:
        pass
    return out


def memory_delta(before: dict | None, after: dict | None) -> dict | None:
    """Per-key int difference of two process_memory_snapshot() dicts;
    None where either side is missing/non-integer. Pure arithmetic (CPU-testable)."""
    if not before or not after:
        return None
    out: dict = {}
    for k in after:
        a, b = after.get(k), before.get(k)
        out[k] = (a - b) if isinstance(a, int) and isinstance(b, int) else None
    return out


def run(args) -> int:
    # Budget validation happens at the CLI boundary, before any GPU import.
    try:
        budgets = multi_gpu.validate_use_per_device(args.use_per_device)
    except ValueError as e:
        raise SystemExit(f" !! FATAL: {e}")
    if budgets is not None and args.device not in ("cuda:0", "cuda"):
        raise SystemExit(f" !! FATAL: --device must stay cuda:0 with --use-per-device "
                         f"(layer split spans all visible devices; --device is unused there)")

    # GPU-facing imports happen only after argparse (so --help works on the host).
    import torch
    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.util import device_copy

    if args.new_tokens < 1 or not args.contexts:
        raise SystemExit(" !! FATAL: --new-tokens must be >= 1 and --contexts non-empty")
    contexts = sorted(set(args.contexts))
    repeats = max(1, args.repeats) if args.mode == "bench" else 1
    warmups = max(0, args.warmup) if args.mode == "bench" else 0
    dev = torch.device(args.device)

    # Load mode from the CLI budgets ONLY (explicit --use-per-device is the
    # single sanctioned split path):
    #   budgets given -> gated autosplit with those exact budgets,
    #                   model.load WITHOUT device, placement audited
    #   no budgets    -> untouched single-GPU path + gate (exactly 1 visible)
    plan = multi_gpu.plan_load_mode(torch, budgets, args.expect_arch)
    split = plan["mode"] == "layer_split"
    if split:
        gpu = plan["gate"]
        used_idx = list(range(gpu["visible_devices"]))
    else:
        gpu = validate_gpu(torch, args.device, args.expect_arch)
        used_idx = [0]

    runs: list[dict] = []
    failures: list[str] = []
    load_s = None
    mem_after_load = None
    mem_after_load_per_device = None
    placement = None
    copy_stats_after_load = None
    cache_tokens = None
    model = None
    ngram_stream_cfg = None
    ngram_state = None
    ngram_final_modules = None
    host_mem_before_load = None
    host_mem_after_load = None
    host_mem_after_inference = None

    # Everything that touches the device is inside try; model.unload() runs in
    # finally and its failure is recorded (never swallowed silently).
    try:
        config = Config.from_directory(args.model_dir)
        # Phase 5: --ngram-ram = the model_init --ngram_ram lever. Set the
        # load-time option BEFORE the load; without the flag the config (and
        # the engine's EXL3_NGRAM_STREAM env default) stays untouched. This
        # changes table STORAGE only -- all four NGramEmbedding load modes
        # return identical results -- so it is not a model-changing option.
        if args.ngram_ram:
            infer_params = getattr(config, "infer_params", None)
            if infer_params is None:
                raise SystemExit(" !! FATAL: --ngram-ram: this Config exposes no "
                                 "infer_params, so ngram_stream_from_disk=False cannot "
                                 "be set before load")
            infer_params.ngram_stream_from_disk = False
        ngram_stream_cfg = getattr(getattr(config, "infer_params", None),
                                   "ngram_stream_from_disk", None)
        model = Model.from_config(config)
        tokenizer = Tokenizer.from_config(config)
        vocab = int(tokenizer.actual_vocab_size)

        need = max(contexts) + args.new_tokens + 256
        # Floor 4096 (and max_chunk_size): model.load runs a dummy forward at
        # max_chunk_size and asserts cache capacity -- see bench_model.py.
        cache_tokens = round_up_page(max(args.cache_tokens or 0, need, 4096, args.max_chunk_size))
        cache = Cache(model, max_num_tokens = cache_tokens)   # BEFORE model.load()

        t0 = time.time()
        host_mem_before_load = process_memory_snapshot()
        if split:
            # Official layer-split autosplit API: NO device argument (the engine
            # asserts device and use_per_device are mutually exclusive).
            load_kwargs = {"max_chunk_size": args.max_chunk_size, "progressbar": False}
            if plan["budgets"] is not None:
                load_kwargs["use_per_device"] = plan["budgets"]
            model.load(**load_kwargs)
        else:
            model.load(device = args.device, max_chunk_size = args.max_chunk_size,
                       progressbar = False)
        load_s = time.time() - t0
        mem_after_load = torch.cuda.memory_allocated(dev)
        copy_stats_after_load = dict(device_copy.stats)
        host_mem_after_load = process_memory_snapshot()
        if split:
            multi_gpu.sync_devices(torch, used_idx)
            mem_after_load_per_device = multi_gpu.device_memory_snapshot(torch, used_idx)
            placement = multi_gpu.audit_placement(model, used_idx)
            tp = placement["transformer_modules_per_device"]
            print(f" -- loaded {args.model_dir} in {load_s:.1f}s as LAYER SPLIT "
                  f"(budgets {plan['budgets']} GiB over "
                  f"{len(used_idx)}x {gpu['expect_arch']}); transformer modules per device: {tp}; "
                  f"cache layers per device: {placement['cache_layers_per_device']}; "
                  f"recurrent layer states per device: {placement['recurrent_layers_per_device']}; "
                  f"placement audit: {'OK' if placement['ok'] else 'FAILED'}", flush = True)
            if not placement["ok"]:
                for p in placement["problems"]:
                    print(f"    AUDIT FAIL: {p}", file = sys.stderr, flush = True)
        else:
            print(f" -- loaded {args.model_dir} in {load_s:.1f}s on {args.device}; "
                  f"cache {cache_tokens} tokens; gpu {gpu['name']} ({gpu['gcnArchName']}, "
                  f"hip {gpu['torch_version_hip']})", flush = True)

        if split and not placement["ok"]:
            # Fail-closed: never measure an unverified split. The audit and the
            # JSON artifact still land; nonzero exit via "failures"; the
            # finally below still runs model.unload() on the normal path.
            failures.append("layer-split placement audit failed: "
                            + "; ".join(placement["problems"])
                            + " -- no throughput jobs were run")
            raise _SkipJobs()

        # ---- Phase 5: n-gram table residence, VERIFIED from the live modules
        # (never assumed, never inferred from CUDA reserved memory). Recording
        # happens in both load modes; the fail-closed check only when
        # --ngram-ram was requested: a table-bearing model that did not end up
        # host-RAM-backed invalidates the RAM assumption the run is testing.
        ngram_state = multi_gpu.collect_ngram_state(model, require_ram = args.ngram_ram)
        if args.ngram_ram or ngram_state["tables_found"]:
            print(f" -- ngram table(s): {ngram_state['tables_found']}; modes "
                  f"{[m['mode'] for m in ngram_state['modules']]}; RAM bytes "
                  f"{ngram_state['total_table_ram_bytes']}; residence read from the actual "
                  f"table tensors (NOT from CUDA reserved); audit: "
                  f"{'OK' if ngram_state['ok'] else 'FAILED'}", flush = True)
            for note in ngram_state["notes"]:
                print(f"    note: {note}", flush = True)
        if args.ngram_ram and not ngram_state["ok"]:
            for p in ngram_state["problems"]:
                print(f"    NGRAM-RAM FAIL: {p}", file = sys.stderr, flush = True)
            failures.append("ngram-ram audit failed: "
                            + "; ".join(ngram_state["problems"])
                            + " -- no throughput jobs were run")
            raise _SkipJobs()

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
            # ---- job boundary: peak stats reset (ALL used devices in split
            # mode; the single old call is kept verbatim for one GPU) ----
            if timed:
                if split:
                    multi_gpu.sync_devices(torch, used_idx)
                    multi_gpu.reset_peak_on_devices(torch, used_idx)
                else:
                    torch.cuda.reset_peak_memory_stats(dev)
            stats0 = dict(device_copy.stats)
            wall_start_unix_s = time.time()
            wall0 = time.perf_counter()
            res, events = run_job(generator, Job, ids, max_new = deliv, min_new = min_new,
                                  seed = args.seed + rep)
            wall = time.perf_counter() - wall0
            # ---- job boundary end: sync every used device (split only), then
            # read peak/allocated/reserved per GPU + transfer-counter delta.
            # NO per-token synchronization anywhere: these calls sit outside
            # the measured wall window's token loop (wall is already closed
            # for single-GPU semantics; the sync only settles device state for
            # the stats read, never the ITL samples). ----
            if split:
                multi_gpu.sync_devices(torch, used_idx)
            mem_after_job = multi_gpu.device_memory_snapshot(torch, used_idx)
            row = {
                "phase": phase, "context": ctx, "repeat": rep, "timed": timed,
                "wall_start_unix_s": wall_start_unix_s,
                **observed_timing(events, wall0),
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
                # old key preserved: single GPU -> identical expression as
                # before; split -> the max per-device peak (timed rows only)
                "peak_mem_bytes": ((max(v["peak_bytes"] for v in mem_after_job.values())
                                    if timed else None) if split
                                   else (torch.cuda.max_memory_allocated(dev) if timed else None)),
                # Phase 3 additions (both modes; keys are device strings):
                "memory_after_job": mem_after_job,
                "device_copy_delta": multi_gpu.diff_transfer_stats(
                    stats0, dict(device_copy.stats)),
            }
            # Raw per-token ITL (decode rows only). Mean TPOT below stays
            # exactly as before; these are the actual per-token intervals.
            itl_samples_ms: list[float] = []
            itl_bursts: list[dict] = []
            if phase == "decode":
                itl_samples_ms, itl_bursts = token_intervals(events)
                row["itl_samples_ms"] = itl_samples_ms
                row["itl_n"] = len(itl_samples_ms)
                row["itl_p50_ms"] = percentile(itl_samples_ms, 50.0) if itl_samples_ms else None
                row["itl_p95_ms"] = percentile(itl_samples_ms, 95.0) if itl_samples_ms else None
                row["itl_bursts"] = itl_bursts
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
                if itl_bursts:
                    problems.append(
                        f"unexpected multi-token increments during decode: {itl_bursts} "
                        f"(batch=1 plain AR must advance job.new_tokens by exactly 1 per "
                        f"iterate(); bursts are reported, not averaged into ITL samples)")
                elif row["new_tokens"] is not None and row["new_tokens"] >= 1:
                    expected_itl = row["new_tokens"] - 1
                    if len(itl_samples_ms) != expected_itl:
                        last_obs = events[-1][1] if events else 0
                        problems.append(
                            f"ITL sample count {len(itl_samples_ms)} != new_tokens-1 "
                            f"({expected_itl}); observed {len(events)} token-progress "
                            f"events, last job.new_tokens={last_obs} -- per-token "
                            f"interval coverage incomplete")
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
                    itl_merged = []
                    for rep in range(warmups, warmups + repeats):
                        r = one_job(phase, ctx, rep, timed = True)
                        rates.append(r["prefill_tps"] if phase == "prefill" else r["decode_tps"])
                        itl_merged.extend(r.get("itl_samples_ms", []))
                    med = statistics.median(rates)
                    spread = (max(rates) - min(rates)) / med if med else 0.0
                    flag = "  <- exceeds 5% noise floor, rerun" if spread > NOISE_FLOOR else ""
                    itl_txt = ""
                    if phase == "decode":
                        itl_txt = (f"   itl n={len(itl_merged)} "
                                   f"p50 {percentile(itl_merged, 50.0):7.2f} "
                                   f"p95 {percentile(itl_merged, 95.0):7.2f} ms"
                                   if itl_merged else "   itl n=0 (no intervals)")
                    print(f" -- {label:16} ok  median {med:10.1f} t/s   spread {spread:5.1%}{flag}{itl_txt}",
                          flush = True)
                except Exception as e:
                    failures.append(f"{label}: {e}")
                    print(f" -- {label:16} FAIL: {e}", flush = True)
                    if not isinstance(e, (AssertionError, RuntimeError, SystemExit)):
                        traceback.print_exc()

        # Phase 5: post-inference evidence -- prefetch counters moved how far?
        # Table tensors unchanged (RAM modes never free them mid-run)? RSS and
        # major/minor fault deltas separate "load allocated the table" from
        # "inference is paging it".
        host_mem_after_inference = process_memory_snapshot()
        if ngram_state is not None:
            ngram_final_modules = multi_gpu.collect_ngram_state(
                model, require_ram = False)["modules"]
    except _SkipJobs:
        pass   # audit already recorded the failure in "failures"; run cleanup
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
                     "ttft_ms": summarize([r["ttft_ms"] for r in rows]),
                     "first_token_wall_ms": summarize([r["first_token_wall_ms"] for r in rows])}
            if phase == "prefill":
                entry["prefill_tps"] = summarize([r["prefill_tps"] for r in rows])
            else:
                entry["tpot_ms"] = summarize([r["tpot_ms"] for r in rows])
                entry["decode_tps"] = summarize([r["decode_tps"] for r in rows])
                entry["decode_observed_tps"] = summarize(
                    [r["decode_observed_tps"] for r in rows if r["decode_observed_tps"] is not None])
                entry["e2e_tps"] = summarize([r["e2e_tps"] for r in rows])
                merged_itl = [s for r in rows for s in r["itl_samples_ms"]]
                entry["itl_ms"] = {
                    "n": len(merged_itl),
                    "p50": percentile(merged_itl, 50.0) if merged_itl else None,
                    "p95": percentile(merged_itl, 95.0) if merged_itl else None,
                    "note": "merged raw per-token intervals of all timed runs "
                            "(empirical distribution; distinct from tpot_ms, "
                            "the per-job mean time_generate/(new_tokens-1))",
                }
            entry["peak_mem_bytes_max"] = max(r["peak_mem_bytes"] for r in rows)
            # Phase 3 addition: per-device aggregate of the timed rows' peaks
            # (single-GPU runs report the same value under "cuda:0").
            pmax: dict = {}
            for r in rows:
                for k, v in (r.get("memory_after_job") or {}).items():
                    pmax[k] = max(pmax.get(k, 0), v["peak_bytes"])
            entry["peak_mem_bytes_max_per_device"] = pmax
            if split:
                entry["device_copy_delta_total"] = {
                    kk: sum((r.get("device_copy_delta") or {}).get(kk, 0) for r in rows)
                    for kk in ("direct", "bounced", "probes")}
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
        "memory": {"after_load_allocated_bytes": mem_after_load,
                   "after_load_per_device": mem_after_load_per_device},
        "gpu_gate": gpu,
        "load_mode": plan["mode"],
        "layer_split": ({
            "enabled": True,
            "use_per_device_gib": plan["budgets"],
            "budget_semantics": "GiB per visible GPU (engine converts "
                                 "int(gib*1024**3)), applied ON TOP of memory already "
                                 "allocated at load() time; maps 1:1 to visible order",
            "load_call": "model.load(use_per_device=..., max_chunk_size=...) -- "
                         "no device argument (engine forbids combining them)",
            "gate": gpu,
            "placement_audit": placement,
            "memory_after_load_per_device": mem_after_load_per_device,
            "device_copy_stats_after_load": copy_stats_after_load,
            "sync_policy": "torch.cuda.synchronize + reset_peak_memory_stats on all "
                           "used devices at job boundaries only; no per-token sync",
        } if split else {
            "enabled": False,
            "use_per_device_gib": None,
            "placement_audit": None,
        }),
        "spec_decode": {"enabled": False, "asserted": True,
                        "draft_model": None, "ngram_match_min": 0},
        # Phase 5: the --ngram-ram assumption, checked against the live engine
        # objects instead of trusted. after_load holds the per-module records
        # (mode/residence/bytes/prefetch counters) + verdict; after_inference_
        # modules re-reads the cumulative prefetch counters once jobs have run;
        # process_memory brackets the run with RSS + major/minor faults so the
        # RAM allocation (load delta) and any paging (inference delta) are both
        # visible. Table residence NEVER uses CUDA reserved-vs-board-total:
        # reserved is allocator state on the GPU, not host RAM evidence.
        "ngram_ram": {
            "requested": bool(args.ngram_ram),
            "legacy_default_untouched": not args.ngram_ram,
            "config_ngram_stream_from_disk": ngram_stream_cfg,
            "after_load": ngram_state,
            "after_inference_modules": ngram_final_modules,
            "process_memory": {
                "before_load": host_mem_before_load,
                "after_load": host_mem_after_load,
                "after_inference": host_mem_after_inference,
                "delta_load": memory_delta(host_mem_before_load, host_mem_after_load),
                "delta_inference": memory_delta(host_mem_after_load,
                                                host_mem_after_inference),
            },
            "semantics": "ngram_stream_from_disk is a STORAGE-location switch "
                         "(disk-streamed rows vs host-RAM table); all four "
                         "NGramEmbedding load modes return identical results, so "
                         "no decoding behavior or model identity changes and the "
                         "spec-decode assertions above are unaffected",
        },
        "sampler": "ArgmaxSampler (greedy, deterministic; min_new_tokens=max_new_tokens in bench mode)",
        "fresh_prompt": "exactly N random IDs in [5%,95%) of actual vocab per run from seeded CPU rng",
        "formulas": {
            "first_token_wall_ms": "observed first token after generator.iterate minus job wall start; includes first decode forward",
            "decode_observed_tps": "observed inter-token count / elapsed time after first token; excludes first-token initialization",
            "ttft": "legacy generator time_prefill (first prefill start -> entry to first decode); actual first-token latency is first_token_wall_ms",
            "tpot": "time_generate / (new_tokens - 1)",
            "itl": "raw per-token intervals: positive ms between successive "
                   "generator.iterate() completions (time.perf_counter; no extra "
                   "per-token cuda.synchronize), observed via job.new_tokens "
                   "increments; the first token is excluded (its dt carries "
                   "prefill/TTFT), so plain AR with N new_tokens gives N-1 samples; "
                   "a multi-token increment is reported as itl_bursts and fails the "
                   "decode row instead of being averaged into fake ITL samples",
            "itl_p50_p95": "nearest-rank percentiles of the raw samples, per timed "
                           "run (itl_p50_ms/itl_p95_ms) and over the merged timed-run "
                           "distribution in summary (itl_ms.p50/p95); not medians "
                           "of per-run medians",
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
