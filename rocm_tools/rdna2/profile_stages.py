#!/usr/bin/env python3
"""Bounded per-stage GPU profiling harness for the single-GPU RDNA boxes.

This tool does NOT optimize anything and does NOT touch the engine. It drives
the production Generator/Job flow exactly as bench.py does, and marks two
small measurement windows per timed job with ROCtx ranges so that an
externally-attached `rocprofv3 --kernel-trace --memory-copy-trace
--marker-trace --output-format csv` produces a trace in which the ONLY traced
GPU activity sits inside named stage windows. The CPU-side artifact this tool
writes (wall clocks, hashes, decode positions) is what summarize_rocprof.py
cross-checks against the marker CSVs.

Measurement design (why it looks like this):

  * Reproducible inputs. The profiled prompts are NOT freshly random: the
    whole baseline bench artifact's CPU RNG prompt stream is replayed in the
    recorded runs order (same seed, same vocab, same bench.fresh_prompt) and
    EVERY row's ids_sha256 must match the baseline. Only phase=decode rows are
    then used: repeat 0 (warmup, never profiled) and repeats 1..--repeats as
    timed runs. Same input per GPU across machines; fresh-per-run prompts
    keep the prefix cache cold.
  * Gated tracing. roctxProfilerPause(0) is issued before model load/warmup,
    so cold JIT / weight upload / warmup kernels are never in the trace. Each
    timed window is: sync -> resume -> range-push -> enqueue/iterate -> sync
    -> clock stop -> range-pop -> pause. Synchronizing at both boundaries
    means every traced kernel lies wholly inside a marker window
    (summarize_rocprof.py rejects anything crossing a boundary).
  * Bounded scope, per job. One decode job per measured repeat with
    min_new_tokens = max_new_tokens = decode_start + decode_tokens:
      - prefill window: enqueue + iterate until new_tokens == 1  (tokens=1),
      - unprofiled gap: iterate until new_tokens == decode_start  (NOT traced;
        recorded as unprofiled progress, never as a profiler window),
      - decode window: iterate until new_tokens == decode_start + K (tokens=K).
    No per-token synchronizations and no per-token profiler scopes: extra
    syncs would perturb what is measured (same convention as bench.py's ITL).
  * Truthful validation. Every timed job must report prompt_tokens == the
    requested context, cached_tokens == 0 (no prefix hits), exact +1 token
    increments inside the profiled windows, new_tokens == decode_total and
    eos_reason == "max_new_tokens". Output identity is captured and STORED
    (generated-id list + sha256s) while the job is still alive, before any
    cleanup drops references, so root can diff cross-GPU artifacts. Logit
    finiteness is NOT re-checked per step: that is the already-passed quality
    gates' job, and per-step device reads would perturb the window.
  * --no-roctx is the matched observer-overhead control: IDENTICAL flow,
    timing and synchronizations, but pause/resume/push/pop are no-ops and no
    marker exists in any trace. Records carry "roctx_traced": false and
    "profiled": false so they can never be mislabeled as profiler windows.
  * Cleanup honesty: model.unload() + sync, then local references are dropped
    (del model/generator/cache + gc, best-effort engine cache releases guarded
    per call) only AFTER nothing of the loaded resources is used anymore; the
    generated sequences are already copied out above. Teardown failures are
    recorded in the artifact and force a nonzero exit. There is no os._exit
    and no device reset anywhere: the JSON is written (fsynced) in a finally
    before normal process exit, including on error paths.

Runtime bootstrap is root's responsibility and is intentionally NOT patched
here: rocprofv3 must be launched with
LD_PRELOAD=librocprofiler-sdk.so:/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1
(SDK FIRST, preserving the host ROCr), plus root's triton._C.libtriton
RTLD_DEEPBIND pre-load before exllamav3 imports. Those live in root's wrapper,
not in this harness or the engine.

Examples
--------
    # profiled run (root launches this under rocprofv3, serially per GPU):
    /opt/venv/bin/python rocm_tools/rdna2/profile_stages.py \
        -m /work/models/qwen3-8b-exl3-4bpw \
        --baseline-json /work/phase0/bench_qwen3_8b_exl3.json \
        --contexts 2048 8192 --repeats 3 \
        --expect-arch gfx1030 --output /work/profile/profile_stages.json

    # matched control (no markers, no tracing; same flow + timings):
    /opt/venv/bin/python rocm_tools/rdna2/profile_stages.py \
        -m ... --baseline-json ... --no-roctx --output .../control.json
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rocm_tools.rdna2 import bench  # CPU-safe import (torch lives inside functions)
from rocm_tools.rdna2.common import (
    BENCH_FORMAT,
    canonical_bytes,
    git_commit,
    model_fingerprint,
    now_utc,
    python_env,
    read_json,
    repo_root,
    rocm_patch_env,
    round_up_page,
    sha256_file,
    sha256_hex,
    torch_env,
    visible_gpu_env,
    write_json,
)

PROFILE_FORMAT = "rdna2-profile-stages/1"

# Root's fixed ROCm installation; the roctx shim lives here on both boxes.
ROCTX_LIB_DEFAULT = "/opt/rocm/core-7.14/lib/librocprofiler-sdk-roctx.so"

MAX_CHUNK_SIZE = 2048          # bounds prefill chunking (same as the baseline)
DEFAULT_CACHE_TOKENS = 8704    # FP16 cache; page-rounded 8192 ctx + 160 + slack


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog = "profile_stages.py",
        description = ("Bounded EXL3 prefill/decode stage profiler: replays the baseline "
                       "bench's exact prompts and wraps two synchronized GPU windows per "
                       "timed decode job in ROCtx markers for rocprofv3. Control mode "
                       "(--no-roctx) runs the identical flow with no-op markers."),
    )
    ap.add_argument("-m", "--model-dir", required = True)
    ap.add_argument("--baseline-json", required = True,
                    help = "previous bench.py artifact (rdna2-bench/1); its full CPU RNG "
                           "prompt stream is replayed and every ids_sha256 asserted")
    ap.add_argument("--contexts", type = int, nargs = "+", default = [2048, 8192],
                    help = "decode-phase contexts to profile (must exist in the baseline)")
    ap.add_argument("--repeats", type = int, default = 3,
                    help = "timed repeats per context, baseline timed rows rep 1..N (1..5); "
                           "baseline rep 0 supplies the warmup prompt")
    ap.add_argument("--decode-start", type = int, default = 96,
                    help = "unprofiled tokens generated before the decode window opens "
                           "(decode window = new_tokens decode_start -> decode_start+K)")
    ap.add_argument("--decode-tokens", type = int, default = 64,
                    help = "tokens inside the profiled decode window")
    ap.add_argument("--expect-arch", choices = ("gfx1030", "gfx1201"), default = "gfx1030")
    ap.add_argument("--output", default = "profile_stages_results.json",
                    help = "JSON artifact path (written in finally, fsynced)")
    ap.add_argument("--no-roctx", dest = "roctx", action = "store_false", default = True,
                    help = "control mode: identical flow, no-op marker/pause/resume calls")
    return ap


class RoctxError(RuntimeError):
    pass


class RoctxControls:
    """ctypes ROCtx shim (the probe-verified signatures from roctx_probe.py).

    Four entry points only: profiler pause/resume (argument 0 = the global
    domain, return code 0 on success) and range push/pop with byte labels.
    pop/pause are exception-safe by contract: stage_window forces them even
    after a body failure.
    """

    traced = True

    def __init__(self, lib_path: str = ROCTX_LIB_DEFAULT):
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise RoctxError(f"cannot load roctx shim {lib_path!r}: {e}") from e
        self._lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        self._lib.roctxRangePushA.restype = ctypes.c_int
        self._lib.roctxRangePop.argtypes = []
        self._lib.roctxRangePop.restype = ctypes.c_int
        self._lib.roctxProfilerPause.argtypes = [ctypes.c_uint64]
        self._lib.roctxProfilerPause.restype = ctypes.c_int
        self._lib.roctxProfilerResume.argtypes = [ctypes.c_uint64]
        self._lib.roctxProfilerResume.restype = ctypes.c_int

    def pause(self) -> None:
        if self._lib.roctxProfilerPause(0) != 0:
            raise RoctxError("roctxProfilerPause(0) returned nonzero")

    def resume(self) -> None:
        if self._lib.roctxProfilerResume(0) != 0:
            raise RoctxError("roctxProfilerResume(0) returned nonzero")

    def push(self, label: str) -> None:
        if self._lib.roctxRangePushA(label.encode("utf-8")) < 0:
            raise RoctxError(f"roctxRangePushA({label!r}) returned negative")

    def pop(self) -> None:
        if self._lib.roctxRangePop() < 0:
            raise RoctxError("roctxRangePop() returned negative")

    def close(self) -> None:
        pass   # the shim stays loaded for the process lifetime; nothing to release.


class NoopControls:
    """--no-roctx control: same call sites, zero profiler effect."""

    traced = False

    def pause(self) -> None: pass
    def resume(self) -> None: pass
    def push(self, label: str) -> None: pass
    def pop(self) -> None: pass
    def close(self) -> None: pass


@contextlib.contextmanager
def stage_window(controls, label: str, record: dict, sync):
    """Exception-safe bounded window, IDENTICAL in both modes:

        sync -> resume -> push -> [body] -> sync -> clock-stop -> pop -> pause

    With NoopControls the resume/push/pop/pause calls do nothing, but the two
    synchronizes and the wall clock still bracket the body, so a --no-roctx
    run is a matched observer-overhead control of the profiled run. The record
    gains wall_s (host perf_counter across the synchronized body) and, on any
    teardown hiccup, window_errors -- pop/pause/sync failures after a body
    exception are recorded, never silently swallowed.
    """
    sync()
    controls.resume()
    controls.push(label)
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        try:
            sync()
        except Exception as e:
            record.setdefault("window_errors", []).append(f"end-sync: {e!r}")
        record["wall_s"] = time.perf_counter() - t0
        for name, fn in (("pop", controls.pop), ("pause", controls.pause)):
            try:
                fn()
            except Exception as e:
                record.setdefault("window_errors", []).append(f"{name}: {e!r}")


class TokenTracker:
    """One advancing Job's token progress, split into named segments.

    Every generator.iterate() completion is timestamped (perf_counter, taken
    immediately after iterate() returns -- no extra per-token synchronize) and
    observed increases of job.new_tokens are recorded per segment. Plain AR
    with batch=1 must advance +1 per iterate(); anything else is stored as a
    burst (never smeared into fake per-token samples) and fails the row.
    Zero-progress iterates are counted.
    """

    def __init__(self, job):
        self.job = job
        self.segments: dict[str, dict] = {}

    def begin(self, name: str) -> None:
        self.segments[name] = {
            "t0": time.perf_counter(), "prev_n": int(self.job.new_tokens),
            "events_ns": [], "zero_iters": 0, "bursts": [],
        }

    def note(self, name: str) -> None:
        seg = self.segments[name]
        n = int(self.job.new_tokens)
        delta = n - seg["prev_n"]
        if delta > 0:
            seg["events_ns"].append((time.perf_counter() - seg["t0"]) * 1e9)
        else:
            seg["zero_iters"] += 1
        if delta > 1:
            seg["bursts"].append({"from_new_tokens": seg["prev_n"],
                                  "to_new_tokens": n, "delta": delta})
        seg["prev_n"] = n

    def strict_increments(self, name: str) -> list[str]:
        """Problems for a window that must advance by exactly +1 per token."""
        seg = self.segments[name]
        problems = []
        if seg["bursts"]:
            problems.append(f"multi-token increments in {name}: {seg['bursts']}")
        if seg["zero_iters"]:
            problems.append(f"{seg['zero_iters']} no-progress iterate() call(s) in {name}")
        return problems

    def summary(self, name: str) -> dict:
        seg = self.segments[name]
        return {"events_n": len(seg["events_ns"]),
                "zero_iters": seg["zero_iters"],
                "bursts": list(seg["bursts"]),
                "token_times_ns": list(seg["events_ns"])}


def load_and_validate_baseline(path: str, contexts: list[int], repeats: int,
                               decode_total: int) -> dict:
    """Structural checks on the baseline artifact (host-side, pre-GPU).

    Fails loudly (SystemExit) unless the artifact carries, for every requested
    context, exactly one rep-0 warmup decode row plus timed decode rows rep
    1..repeats, all problem-free -- those rows define the profiled runs.
    """
    m = read_json(path)
    errors: list[str] = []
    if m.get("format") != BENCH_FORMAT:
        errors.append(f"baseline format is {m.get('format')!r}, expected {BENCH_FORMAT!r}")
    if m.get("mode") != "bench":
        errors.append(f"baseline mode is {m.get('mode')!r}; strict timed rows require 'bench'")
    runs = m.get("runs")
    if not isinstance(runs, list) or not runs:
        errors.append("baseline runs[] missing/empty -- nothing to replay")
    params = m.get("params") or {}
    if not isinstance(params.get("seed"), int):
        errors.append("baseline params.seed missing (prompt stream not reproducible)")
    if int(params.get("max_chunk_size", MAX_CHUNK_SIZE)) != MAX_CHUNK_SIZE:
        errors.append(f"baseline max_chunk_size {params.get('max_chunk_size')} != {MAX_CHUNK_SIZE}")
    if int(params.get("new_tokens", 0) or 0) < decode_total:
        errors.append(f"baseline new_tokens {params.get('new_tokens')} < decode_total {decode_total}")
    if not errors:
        by: dict[tuple, list] = {}
        for row in runs:
            by.setdefault((row.get("phase"), row.get("context"), row.get("timed")),
                          []).append(row)
        for ctx in contexts:
            warm = by.get(("decode", ctx, False), [])
            if len(warm) != 1 or warm[0].get("repeat") != 0:
                errors.append(f"baseline has no single phase=decode context={ctx} rep0 "
                              f"warmup row (found {len(warm)})")
            timed = [r for r in by.get(("decode", ctx, True), [])
                     if isinstance(r.get("repeat"), int) and 1 <= r["repeat"] <= repeats]
            if len(timed) != repeats:
                errors.append(f"baseline has {len(timed)} timed decode rows rep1..{repeats} "
                              f"for context={ctx}, expected {repeats}")
            for r in by.get(("decode", ctx, False), []) + by.get(("decode", ctx, True), []):
                if r.get("problems"):
                    errors.append(f"baseline decode row rep={r.get('repeat')} ctx={ctx} "
                                  f"carries problems: {r['problems']}")
    if errors:
        raise SystemExit(" !! FATAL: baseline artifact unusable: " + "; ".join(errors))
    return m


def regenerate_prompts(baseline: dict, torch, vocab: int) -> list[dict]:
    """Replay the ENTIRE baseline CPU RNG prompt stream in recorded order.

    Same seed, same fresh_prompt, one call per baseline run in artifact order.
    Returns one entry per row with the regenerated ids tensor and its sha256
    (identical construction to bench.py: sha256 over the numpy bytes). Every
    hash MUST equal the baseline's ids_sha256; any mismatch is fatal -- the
    profiled inputs would then not be the inputs the baseline measured.
    """
    seed = int(baseline["params"]["seed"])
    rng = torch.Generator().manual_seed(seed)
    out: list[dict] = []
    mismatches: list[str] = []
    for i, row in enumerate(baseline["runs"]):
        ids = bench.fresh_prompt(torch, int(row["context"]), vocab, rng)
        sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        if sha != row.get("ids_sha256"):
            mismatches.append(f"run[{i}] phase={row.get('phase')} ctx={row.get('context')} "
                              f"rep={row.get('repeat')}: regenerated {sha[:16]}.. != "
                              f"baseline {str(row.get('ids_sha256'))[:16]}..")
        out.append({"row": row, "ids": ids, "ids_sha256": sha})
    if mismatches:
        raise AssertionError(
            f"prompt-stream replay failed for {len(mismatches)}/{len(out)} rows "
            f"(seed={seed}, vocab={vocab}): " + "; ".join(mismatches[:6]))
    return out


def run_timed_decode_job(generator, Job, ids, ctx: int, rep: int, seed: int,
                         controls, decode_start: int, decode_tokens: int,
                         sync) -> dict:
    """One full decode job with two bounded windows around/inside it.

    Flow ('iterate until' = while the job.new_tokens target is unreached and
    the generator queue is not drained; every iterate() completion is
    timestamped via the TokenTracker, no extra per-token syncs):
      prefill window  : enqueue -> new_tokens == 1        [marker tokens=1]
      unprofiled gap  : -> new_tokens == decode_start     (never traced)
      decode window   : -> new_tokens == decode_start+K   [marker tokens=K]
    The job completes on the last decode-window token (min_new == max_new),
    so the completion result is captured within this call. The generated
    sequence is copied out (list + hashes) BEFORE any cleanup can drop the
    job references.
    """
    from exllamav3.generator.sampler import ArgmaxSampler

    total = decode_start + decode_tokens
    prefill_label = f"stage=prefill;context={ctx};repeat={rep};tokens=1"
    decode_label = f"stage=decode;context={ctx};repeat={rep};tokens={decode_tokens}"
    job = Job(input_ids = ids, max_new_tokens = total, min_new_tokens = total,
              sampler = ArgmaxSampler(), seed = seed)
    tracker = TokenTracker(job)

    final = None
    errors_seen: list = []

    def drive_one() -> None:
        nonlocal final
        for r in generator.iterate():
            stage = r.get("stage")
            if stage == "error":
                errors_seen.append(r.get("error"))
            elif stage == "streaming" and r.get("eos"):
                final = r

    def new_window(stage: str, label: str, tokens: int) -> dict:
        return {"stage": stage, "context": ctx, "repeat": rep, "tokens": tokens,
                "marker_label": label, "profiled": bool(controls.traced),
                "roctx_traced": bool(controls.traced)}

    # -- prefill window: enqueue happens INSIDE the window (TTFT kernels).
    # A long prompt is consumed in max_chunk_size chunks, so iterate() rounds
    # with zero new tokens are EXPECTED here (8192/2048 -> several chunk
    # iters before the first token). Strict +1 is decode-only; for prefill we
    # assert no bursts and that the window closes at new_tokens == 1.
    pf = new_window("prefill", prefill_label, 1)
    with stage_window(controls, prefill_label, pf, sync):
        generator.enqueue(job)
        tracker.begin("prefill")
        while job.new_tokens < 1 and generator.num_remaining_jobs() > 0:
            drive_one()
            tracker.note("prefill")
    pf["new_tokens_at_close"] = int(job.new_tokens)

    # -- unprofiled gap: NOT a profiler window and never labeled as one.
    gap = {"profiled": False, "roctx_traced": False,
           "note": f"unprofiled advance to new_tokens=={decode_start} "
                   f"(no markers, no tracing, no per-token syncs)"}
    tracker.begin("gap")
    gap_t0 = time.perf_counter()
    while job.new_tokens < decode_start and generator.num_remaining_jobs() > 0:
        drive_one()
        tracker.note("gap")
    gap["wall_s"] = time.perf_counter() - gap_t0
    gap.update(tracker.summary("gap"))

    # -- decode window: exactly decode_tokens one-token advances.
    dc = new_window("decode", decode_label, decode_tokens)
    with stage_window(controls, decode_label, dc, sync):
        tracker.begin("decode")
        while job.new_tokens < total and generator.num_remaining_jobs() > 0:
            drive_one()
            tracker.note("decode")

    problems: list[str] = []
    if errors_seen:
        problems.append(f"job error result(s): {errors_seen}")
    if final is None:
        problems.append("job produced no completion result")
    else:
        for key, want in (("prompt_tokens", ctx), ("cached_tokens", 0),
                          ("new_tokens", total), ("eos_reason", "max_new_tokens")):
            got = final.get(key)
            if got != want:
                problems.append(f"{key}={got!r} != {want!r}")
        if (final.get("time_prefill") or 0) <= 0:
            problems.append("completion result time_prefill missing/zero")
    if pf.get("window_errors"):
        problems.append(f"prefill window teardown errors: {pf['window_errors']}")
    if dc.get("window_errors"):
        problems.append(f"decode window teardown errors: {dc['window_errors']}")
    # Chunked prefill may take several zero-token iterations before its first token.
    if tracker.segments["prefill"]["bursts"] or tracker.summary("prefill")["events_n"] != 1:
        problems.append("prefill must produce exactly one token without a burst")
    problems += tracker.strict_increments("decode")
    if tracker.segments["gap"]["bursts"]:
        problems.append(f"multi-token increments in unprofiled gap: "
                        f"{tracker.segments['gap']['bursts']}")
    if int(job.new_tokens) != total:
        problems.append(f"job.new_tokens={job.new_tokens} at end != {total}")

    # Copy the output identity out NOW, before cleanup can release the job.
    seq = {"sequence_len": None, "generated_ids": None,
           "generated_ids_sha256": None, "sequence_sha256": None}
    if final is not None:
        try:
            full = job.sequences[0].sequence_ids.torch().reshape(-1).cpu()
            ids_list = [int(t) for t in full.tolist()]
            seq["sequence_len"] = len(ids_list)
            seq["sequence_sha256"] = sha256_hex(full.numpy().tobytes())
            if len(ids_list) >= total:
                gen = ids_list[-total:]
                seq["generated_ids"] = gen
                seq["generated_ids_sha256"] = sha256_hex(canonical_bytes(gen))
            else:
                problems.append(f"sequence_ids shorter than generated count: "
                                f"{len(ids_list)} < {total}")
        except Exception as e:
            problems.append(f"sequence capture failed: {e!r}")

    pf.update(tracker.summary("prefill"))
    dc.update(tracker.summary("decode"))
    return {
        "kind": "timed_decode_job", "context": ctx, "repeat": rep, "seed": seed,
        "ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "decode_start": decode_start, "decode_tokens": decode_tokens,
        "prefill_window": pf, "unprofiled_gap": gap, "decode_window": dc,
        "job_result": {
            "prompt_tokens": final.get("prompt_tokens") if final else None,
            "cached_tokens": final.get("cached_tokens") if final else None,
            "new_tokens": final.get("new_tokens") if final else None,
            "eos_reason": final.get("eos_reason") if final else None,
            "time_prefill_s": final.get("time_prefill") if final else None,
            "time_generate_s": final.get("time_generate") if final else None,
        },
        "sequence": seq,
        "problems": problems,
    }


def run_warmup_decode_job(generator, Job, ids, ctx: int, seed: int,
                          decode_start: int, decode_tokens: int, sync) -> dict:
    """Untimed, UNTRACED warmup through the production flow.

    Same job parameters as the timed runs (min_new == max_new == decode_total)
    so the same length regimes are exercised before anything is measured. The
    profiler stays paused for the whole warmup (pause taken before model
    load), so warmup kernels never enter the trace, and no marker is pushed.
    """
    from exllamav3.generator.sampler import ArgmaxSampler

    total = decode_start + decode_tokens
    job = Job(input_ids = ids, max_new_tokens = total, min_new_tokens = total,
              sampler = ArgmaxSampler(), seed = seed)
    sync()
    t0 = time.perf_counter()
    generator.enqueue(job)
    final = None
    errs: list = []
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            stage = r.get("stage")
            if stage == "error":
                errs.append(r.get("error"))
            elif stage == "streaming" and r.get("eos"):
                final = r
    sync()
    wall = time.perf_counter() - t0
    problems: list[str] = []
    if errs:
        problems.append(f"warmup job error result(s): {errs}")
    if final is None:
        problems.append("warmup job produced no completion result")
    else:
        if final.get("prompt_tokens") != ctx:
            problems.append(f"warmup prompt_tokens {final.get('prompt_tokens')} != {ctx}")
        if final.get("cached_tokens"):
            problems.append(f"warmup prefix cache hit (cached_tokens={final.get('cached_tokens')})")
        if final.get("new_tokens") != total:
            problems.append(f"warmup new_tokens {final.get('new_tokens')} != {total}")
        if final.get("eos_reason") != "max_new_tokens":
            problems.append(f"warmup eos_reason={final.get('eos_reason')!r}")
    return {
        "kind": "warmup_decode_job", "profiled": False, "roctx_traced": False,
        "note": "warmup is never measured and never traced (profiler paused before load)",
        "context": ctx, "ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "seed": seed, "wall_s": wall, "problems": problems,
    }


def release_gpu_resources(torch, names: dict, cleanup_failures: list) -> None:
    """Best-effort post-unload release, run ONLY when nothing is used anymore.

    Root's verified pilot sequence (minus any engine *implementation* edits --
    these are existing public helpers, each guarded independently so a missing
    or renamed hook is recorded, not fatal): drop the local references, gc,
    then ask the two known caches (module-level tensor cache, BC-attention
    kernel cache) to release, empty the torch allocator and synchronize.
    Every failure is recorded into cleanup_failures and forces nonzero exit.
    """
    try:
        gc.collect()
    except Exception as e:                          # pragma: no cover
        cleanup_failures.append(f"gc.collect(): {e!r}")
    try:
        from exllamav3.util.tensor import g_tensor_cache
        g_tensor_cache.drop_all()
    except Exception as e:
        cleanup_failures.append(f"g_tensor_cache.drop_all(): {e!r}")
    try:
        from exllamav3.modules.attention_fn import bc_attn as _bc
        kc = getattr(_bc, "_kernel_cache", None)
        if kc is not None and hasattr(kc, "clear"):
            kc.clear()
    except Exception as e:
        cleanup_failures.append(f"bc_attn._kernel_cache.clear(): {e!r}")
    for name in list(names):
        names[name] = None                          # drop the held references
    try:
        gc.collect()
    except Exception as e:                          # pragma: no cover
        cleanup_failures.append(f"gc.collect(): {e!r}")
    try:
        if torch is not None:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        cleanup_failures.append(f"empty_cache/synchronize: {e!r}")


def run(args) -> int:
    # ---- argparse already done by main; nothing GPU-facing until here ----
    if not (1 <= args.repeats <= 5):
        raise SystemExit(f" !! FATAL: --repeats must be 1..5 (got {args.repeats})")
    if args.decode_start < 1 or args.decode_tokens < 1:
        raise SystemExit(" !! FATAL: --decode-start/--decode-tokens must be >= 1")
    if not args.contexts:
        raise SystemExit(" !! FATAL: --contexts empty")
    contexts = sorted(set(int(c) for c in args.contexts))
    decode_total = args.decode_start + args.decode_tokens

    result: dict = {
        "format": PROFILE_FORMAT,
        "tool": "rocm_tools/rdna2/profile_stages.py",
        "created_utc": now_utc(),
        "ok": False,
        "errors": [],
        "cleanup_failures": [],
        "mode": {"roctx": bool(args.roctx),
                 "control_mode": not args.roctx,
                 "note": ("--no-roctx matched control: identical flow, timing and syncs, "
                          "no-op marker/pause/resume calls; windows are wall-clock only, "
                          "NOT profiler windows" if not args.roctx else
                          "rocprofv3 attached externally (SDK-first LD_PRELOAD, root's "
                          "bootstrap); only the stage windows are traced")},
        "params": vars(args),
        "provenance": {
            "commit": git_commit(repo_root()),
            "model_dir": str(Path(args.model_dir).resolve()),
            "model_fingerprint": model_fingerprint(args.model_dir),
            "env": {
                "python": python_env(),
                "torch": torch_env(),
                "visible_gpu": visible_gpu_env(),
                "exl3_switches": rocm_patch_env(),
                "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
            },
            "baseline": {"path": str(Path(args.baseline_json).resolve()),
                         "sha256": sha256_file(args.baseline_json)},
            "max_chunk_size": MAX_CHUNK_SIZE,
            "cache_tokens": None,
            "cache_tokens_source": None,
            "cache_layer_type": None,
            "decode_start": args.decode_start,
            "decode_tokens": args.decode_tokens,
            "decode_total": decode_total,
            "decode_positions_per_repeat": [
                {"context": c, "unprofiled_new_tokens": [0, args.decode_start],
                 "profiled_decode_new_tokens": [args.decode_start, decode_total]}
                for c in contexts],
            "loaded_extension_path": None,
            "exllamav3_init_path": None,
            "roctx_lib": ROCTX_LIB_DEFAULT if args.roctx else None,
            "roctx_paused_before_load": False,
        },
        "gpu_gate": None,
        "prompt_replay": {"runs": None, "ids_sha256_all_match": None},
        "runs": [],
    }
    held: dict = {}          # model/generator/cache/... refs for cleanup
    torch = None
    model = None
    controls = NoopControls()
    try:
        baseline = load_and_validate_baseline(args.baseline_json, contexts,
                                              args.repeats, decode_total)
        result["provenance"]["baseline"].update({
            "seed": baseline["params"].get("seed"),
            "commit": baseline.get("commit"),
            "params": {k: baseline["params"].get(k) for k in
                       ("contexts", "repeats", "warmup", "new_tokens", "max_chunk_size",
                        "cache_tokens", "mode")},
            "runs_total": len(baseline["runs"]),
        })

        # Pause the profiler BEFORE any GPU work (torch/HIP init, load, warmup).
        if args.roctx:
            controls = RoctxControls()
            controls.pause()
        result["provenance"]["roctx_paused_before_load"] = bool(args.roctx)

        # GPU-facing imports happen only here, after argparse + validation,
        # so --help works on a CPU-only host.
        import torch
        held["torch"] = torch
        from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
        import exllamav3
        import exllamav3.ext as _ext_mod
        ext_mod = getattr(_ext_mod, "exllamav3_ext", None)
        result["provenance"]["loaded_extension_path"] = getattr(ext_mod, "__file__", None)
        result["provenance"]["exllamav3_init_path"] = getattr(exllamav3, "__file__", None)

        result["gpu_gate"] = bench.validate_gpu(torch, "cuda:0", args.expect_arch)
        dev = torch.device("cuda:0")
        sync = lambda: torch.cuda.synchronize(dev)

        config = Config.from_directory(args.model_dir)
        model = Model.from_config(config)
        held["model"] = model
        tokenizer = Tokenizer.from_config(config)
        vocab = int(tokenizer.actual_vocab_size)

        # Reproduce the whole baseline prompt stream BEFORE loading anything
        # heavy: a hash mismatch is a provenance failure, not a GPU failure.
        replay = regenerate_prompts(baseline, torch, vocab)
        result["prompt_replay"] = {"runs": len(replay), "ids_sha256_all_match": True,
                                   "seed": baseline["params"]["seed"], "vocab": vocab}
        by_key = {(e["row"]["phase"], e["row"]["context"], e["row"]["repeat"]): e
                  for e in replay}
        missing = [(ctx, rep) for ctx in contexts for rep in range(args.repeats + 1)
                   if ("decode", ctx, rep) not in by_key]
        if missing:
            raise AssertionError(f"baseline decode rows missing after replay: {missing}")

        cache_tokens = baseline.get("cache_tokens")
        result["provenance"]["cache_tokens_source"] = "baseline artifact"
        if not cache_tokens:
            cache_tokens = DEFAULT_CACHE_TOKENS
            result["provenance"]["cache_tokens_source"] = "default (FP16, page-rounded)"
        cache_tokens = round_up_page(int(cache_tokens))
        need = round_up_page(max(contexts) + decode_total + 256)
        if cache_tokens < need:
            raise AssertionError(f"cache_tokens {cache_tokens} < required {need} "
                                 f"(max context + decode_total + 256 slack)")
        result["provenance"]["cache_tokens"] = cache_tokens
        cache = Cache(model, max_num_tokens = cache_tokens)   # BEFORE model.load()
        held["cache"] = cache
        result["provenance"]["cache_layer_type"] = cache.layer_type.__name__
        if cache.layer_type.__name__ != "CacheLayer_fp16":
            raise AssertionError(f"expected FP16 cache, got {cache.layer_type.__name__}")

        t0 = time.perf_counter()
        model.load(device = "cuda:0", max_chunk_size = MAX_CHUNK_SIZE, progressbar = False)
        result["provenance"]["load_s"] = time.perf_counter() - t0
        print(f" -- loaded {args.model_dir}; cache {cache_tokens} tokens; "
              f"profiler {'paused (roctx)' if args.roctx else 'not attached (control)'}",
              flush = True)

        generator = Generator(
            model = model, cache = cache, tokenizer = tokenizer,
            max_batch_size = 1,
            max_chunk_size = MAX_CHUNK_SIZE,
            ngram_match_min = 0,          # explicit: no speculative decoding
        )
        held["generator"] = generator
        spec_off = (generator.draft_model is None and generator.draft_cache is None
                    and generator.ngram_match_min == 0 and generator.num_draft_tokens == 0)
        if not spec_off:
            raise SystemExit(" !! FATAL: generator has draft/n-gram spec decoding active; "
                             "this harness must measure plain AR decode")

        for ctx in contexts:
            warm = by_key[("decode", ctx, 0)]
            wrec = run_warmup_decode_job(generator, Job, warm["ids"], ctx,
                                         int(warm["row"]["seed"]),
                                         args.decode_start, args.decode_tokens, sync)
            result["runs"].append(wrec)
            if wrec["problems"]:
                raise AssertionError(f"warmup ctx={ctx} failed: {wrec['problems']}")
            print(f" -- warmup@{ctx:5} ok  wall {wrec['wall_s']:7.2f}s (unprofiled/untraced)",
                  flush = True)

            for rep in range(1, args.repeats + 1):
                e = by_key[("decode", ctx, rep)]
                rec = run_timed_decode_job(generator, Job, e["ids"], ctx, rep,
                                           int(e["row"]["seed"]), controls,
                                           args.decode_start, args.decode_tokens, sync)
                result["runs"].append(rec)
                if rec["problems"]:
                    raise AssertionError(f"decode@{ctx} rep{rep}: {rec['problems']}")
                pw, dw = rec["prefill_window"], rec["decode_window"]
                print(f" -- decode@{ctx:5} rep{rep} ok  prefill "
                      f"{pw['wall_s'] * 1000:8.2f} ms  decode({args.decode_tokens}tok) "
                      f"{dw['wall_s'] * 1000:8.2f} ms", flush = True)
    except SystemExit as e:
        result["errors"].append(f"FATAL: {e}")
    except Exception as e:
        traceback.print_exc()
        result["errors"].append(f"aborted: {e!r}")
    finally:
        # Normal cleanup, in root's verified order: unload + sync first, then
        # drop references and best-effort release caches. Sequences/hashes are
        # already copied out of the job in run_timed_decode_job. A teardown
        # failure is recorded and forces a nonzero exit -- never swallowed.
        try:
            controls.close()
        except Exception as e:
            result["cleanup_failures"].append(f"controls.close(): {e!r}")
        if model is not None:
            try:
                if torch is not None:
                    torch.cuda.synchronize()
                model.unload()
            except Exception as e:
                result["cleanup_failures"].append(f"model.unload()/sync: {e!r}")
        # Release the actual local references as well as the tracking dictionary.
        generator = cache = model = config = tokenizer = None
        held.clear()
        try:
            release_gpu_resources(torch, held, result["cleanup_failures"])
        except Exception as e:
            result["cleanup_failures"].append(f"release_gpu_resources: {e!r}")
        result["ok"] = (not result["errors"] and not result["cleanup_failures"]
                        and result["gpu_gate"] is not None)
        write_json(args.output, result)   # fsynced via common.write_json

    print(f"\n  RESULT {'ok' if result['ok'] else 'FAIL'}: "
          f"{len(result['runs'])} jobs recorded; errors={len(result['errors'])} "
          f"cleanup_failures={len(result['cleanup_failures'])}; JSON: {args.output}",
          flush = True)
    sys.stdout.flush()
    return 0 if result["ok"] else 1


def main(argv = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        print(f" !! profile_stages aborted: {e}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
