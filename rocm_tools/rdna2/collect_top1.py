#!/usr/bin/env python3
"""Teacher-forced top-1 collection for one backend, over a frozen token manifest.

This is the accuracy primitive for Phase 0-2 on the single V620: at each manifest
position p of a case, record the argmax token ID of the teacher-forced next-token
distribution after consuming ids[0..p]. Only manifest tokens are ever fed to the
model -- no generated continuation participates -- so both backends see exactly
the same literal input sequence and the artifact is a small list of integers
(top-1 agreement is the Phase 0-2 criterion; full-vocab KLD would need logit rows
and is intentionally not collected here).

EVIDENCE STATUS: every artifact produced here is CANDIDATE evidence. Bulk-vs-chunk
agreement (or cross-backend agreement) localizes/absolves kernel-path divergence at
identical inputs; it does NOT prove independently correct outputs and this file must
never be labeled ground truth. Ground truth for the Qwen3.8 Flash Next Phase-5 work
comes from the root orchestrator's independent testing of the actual model.

Backends
--------
exl3, execution=bulk (default):
    model.forward(ids, {"attn_mode": "flash_attn_nc"})
    One cache-free full-sequence forward per case -> logits for every position.
    Same route eval/ppl.py uses. Exercises prefill/GEMM kernels end-to-end.

exl3, execution=chunked --chunk-size C:
    model.forward(chunk, {"attn_mode": "flash_attn", "cache": cache,
                          "past_len": s, "batch_shape": (1, cache_max_seq_len)})
    The exact cached-call pattern of eval/perf.py and the generator's full-chain
    prefill forward (job.py MTP branch). K/V go through the paged Cache; C=1 is
    the autoregressive decode route (single-token forward + cached attention),
    C>1 is bulk-style cached prefill in C-token steps. Chunked-vs-bulk disagreement
    localizes prefill-vs-decode kernel divergence at identical inputs.

Hybrid recurrent models (Qwen4Exp / Qwen3.8-Flash-Next: GDN + PLE + QSA)
------------------------------------------------------------------------
Supported on BOTH exl3 routes, through the engine's own recurrent machinery
(no reimplementation, no engine edits):

  * bulk (cache-free): the flash_attn_nc forward carries no cache/batch_shape,
    so Qwen4ExpModel.prepare_inputs -> recurrent_util.prepare_for_recurrence is
    a deliberate no-op and the model runs STATELESS over the whole sequence:
    GDN scans from a zero initial state (gated_delta_net_fn with
    recurrent_state=None), and PLELayer._history builds the n-gram hashing
    history as [eos]x(ngram_size-1) + ids for the full literal case -- exactly
    the EOS-padded sequence start a fresh cleared state carries at position 0.

  * chunked (cached): per case the collector obtains ONE cleared state
    (Cache.get_new_state -> GDNState(clear=True) zeroes every GDN conv/delta
    slot and fills the PLE id-history slot with eos), passes that same live
    state in params["recurrent_states"] for EVERY chunk, and verifies after each
    forward that advance_recurrent_states moved state.position to exactly the
    chunk end -- so chunks NEVER silently re-run from zero state. PLE history
    is the REAL preceding-token context across chunk boundaries: the engine
    reads the carried window from the state slot and writes
    id_state[:ctx] = cat(prev_window, chunk_ids)[-ctx:] after every forward;
    the collector independently re-derives the expected window at every
    boundary (ple_carry_update vs ple_reference_history, CPU-side) and fails
    the case on a mismatch. Between independent cases the state slot is
    released in a finally block (also on exceptions) and the cache's state pool
    is rebuilt (Cache.reset_states), so no case can inherit another's state.
    No speculative decoding, MTP, or forced synthetic tokens anywhere in this
    collector: params NEVER set "recurrent_history", so states advance
    destructively without history writes and every harvested row conditions on
    the literal manifest prefix only.

exl3 n-gram table residency (--ngram-ram):
    Set config.infer_params.ngram_stream_from_disk = False BEFORE model.load()
    so every NGramEmbedding (PLE) table lands in system RAM instead of being
    streamed from disk (model_init's --ngram_ram is the engine-side precedent).
    The ACTUAL post-load mode of every table (trellis_ram/fp16_ram vs
    trellis_disk/fp16_disk) is recorded in execution.ngram either way; if RAM
    was requested and any table is not actually RAM-resident, collection
    refuses to run rather than silently measuring the streamed route.

transformers (unquantized source reference, e.g. the Qwen3-8B BF16 dir):
    model(input_ids, use_cache=False)["logits"]
    --dtype auto (default) loads the checkpoint in its declared torch_dtype, so a
    BF16 source gives a native BF16 reference (both BF16 GEMM and full-model BF16
    generation are verified working on this gfx1030 stack); fp16/bf16/fp32 are
    explicit overrides. Requested dtype, checkpoint-declared dtype, effective
    parameter dtype and the model config's ACTUAL attention implementation are
    all recorded in the artifact. from_pretrained is passed the real kwargs
    (dtype= on current Transformers, torch_dtype= fallback) -- signature
    inspection is not used because **kwargs would hide them and silently drop
    options. --device cpu is fully supported for this backend (large BF16
    references that do not fit the V620 can be measured on host RAM; slower,
    but numerically valid; recorded honestly in metadata).

Output: JSON with per-case top-1 IDs, manifest/case/input hashes, model and
config fingerprints, environment (python/torch/HIP/GPU, EXL3_ROCM_* switches that
distinguish conservative vs optimized executions) and all run parameters.
Runtime note (updated 2026-09-29): the earlier "default EXL3 BC attention HANGS
at module load" claim was a MISATTRIBUTION -- it coincided with a concurrent GPU
probe on the same card. Default BC attention works on this stack (isolated warm
plain and fresh-cache plain runs generate correctly and exit 0), so normal
commands do NOT set EXL3_BC_ATTN=0; that switch remains a diagnostic for
bisecting the attention path under contention only. The switch set actually in
effect is captured into every artifact's env block.
Phase 3: exl3 collections can run on the gfx1030 PAIR as an audited LAYER SPLIT
(official model.load(use_per_device=[GiB,...]) autosplit API, no device
argument -- not tensor parallelism). --use-per-device is EXL3-only and refused
on the transformers backend; --cache-tokens sizes the KV cache allocated before
load so a split places identically to an equally sized bench run (bulk stays
cache-free by default; if forced, cache_max_seq_len and uses_kv_cache are
labeled separately). The gate (ROCm, visible count == budget count, EVERY
device gcnArchName == --expect-arch) plus the verified placement audit (per
module/cache layer + backing k/v tensors, contiguous device progression) are
recorded in the artifact under execution.layer_split; collection refuses to run
on an unverified split.
total_positions is ALWAYS relative to the full manifest: running a subset with
--limit-cases yields collected < total, which marks the artifact incomplete and
makes comparison refuse it -- two identically truncated artifacts cannot pass.
Non-finite logit rows are flagged per position and also mark the result
incomplete. Per-case exceptions are recorded truthfully, the partial artifact is
written, and the process exits nonzero. Cleanup (model.unload() for exl3) runs
in a finally block and its errors are recorded into the artifact as failures --
never silently swallowed, never via os._exit; models are released through the
normal path. Recurrent-state teardown (slot release + pool reset) runs in a
finally block too; a failed release is recorded as an error (fail-closed).

Examples
--------
    # transformers reference from the unquantized BF16 source (native dtype)
    /opt/venv/bin/python rocm_tools/rdna2/collect_top1.py \
        --manifest /work/phase0/manifest_qwen3_8b.json \
        --backend transformers -m /work/models/qwen3-8b-bf16 \
        --device cuda:0 -o /work/phase0/top1_ref_transformers.json

    # ...or on host RAM when the reference does not fit the V620 (e.g. 30B)
    /opt/venv/bin/python rocm_tools/rdna2/collect_top1.py \
        --manifest /work/phase0/manifest_qwen3_30b.json \
        --backend transformers -m /work/models/qwen3-30b-a3b-bf16 \
        --device cpu -o /work/phase0/top1_ref_30b_cpu.json

    # EXL3 bulk-prefill candidate (default BC attention ON -- verified working)
    /opt/venv/bin/python rocm_tools/rdna2/collect_top1.py \
        --manifest /work/phase0/manifest_qwen3_8b.json \
        --backend exl3 -m /work/models/qwen3-8b-exl3-4bpw \
        --execution bulk -o /work/phase0/top1_cand_exl3_bulk.json

    # EXL3 autoregressive (decode-route) candidate
    /opt/venv/bin/python rocm_tools/rdna2/collect_top1.py \
        --manifest /work/phase0/manifest_qwen3_8b.json \
        --backend exl3 -m /work/models/qwen3-8b-exl3-4bpw \
        --execution chunked --chunk-size 1 \
        -o /work/phase0/top1_cand_exl3_chunked1.json

    # Phase 3: audited LAYER SPLIT over both gfx1030 V620s. --cache-tokens 8704
    # (both executions) so the split point matches an identically sized bench
    # run; the placement audit lands in execution.layer_split and collection
    # refuses to run on an unverified split.
    /opt/venv/bin/python rocm_tools/rdna2/collect_top1.py \
        --manifest /work/phase0/manifest_qwen3_8b.json \
        --backend exl3 -m /work/models/qwen3-8b-exl3-4bpw \
        --use-per-device 3 4 --cache-tokens 8704 \
        --execution chunked --chunk-size 1 \
        -o /work/phase3/top1_ls_exl3_chunked1.json

    # Phase-5 prerequisite: Qwen3.8-Flash-Next (Qwen4Exp: GDN + PLE), cached
    # chunked teacher forcing with live recurrent-state carry, PLE n-gram table
    # pinned to system RAM before load (actual residency recorded; a disk-mode
    # table under --ngram-ram refuses the run):
    /opt/venv/bin/python rocm_tools/rdna2/collect_top1.py \
        --manifest /work/phase5/manifest_qwen38_next.json \
        --backend exl3 -m /work/models/qwen38-next-exl3 \
        --execution chunked --chunk-size 1 --ngram-ram \
        -o /work/phase5/top1_cand_exl3_chunked1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rocm_tools.rdna2.common import (
    DTYPE_KEYS,
    POSITION_SEMANTICS,
    TOP1_FORMAT,
    git_commit,
    model_fingerprint,
    now_utc,
    python_env,
    read_json,
    repo_root,
    resolve_dtype_choice,
    rocm_patch_env,
    round_up_page,
    torch_env,
    validate_manifest,
    visible_gpu_env,
    write_json,
)
from rocm_tools.rdna2 import multi_gpu   # stdlib-only at import (CPU-safe)


def check_cli_options(args) -> None:
    """Backend cross-checks that need no GPU/torch: validate budgets once and
    refuse EXL3-only options (--use-per-device, --cache-tokens, --ngram-ram) on
    the transformers backend before anything loads."""
    try:
        budgets = multi_gpu.validate_use_per_device(args.use_per_device)
    except ValueError as e:
        raise SystemExit(f" !! FATAL: {e}")
    multi_gpu.reject_split_on_transformers(args.backend, budgets)
    if args.backend == "transformers" and getattr(args, "cache_tokens", None) is not None:
        raise SystemExit(" !! FATAL: --cache-tokens is an exl3 paged-KV-cache option; the "
                         "transformers backend runs use_cache=False and never allocates one")
    if getattr(args, "cache_tokens", None) is not None and args.cache_tokens < 1:
        raise SystemExit(f" !! FATAL: --cache-tokens must be >= 1 (got {args.cache_tokens})")
    if args.backend == "transformers" and getattr(args, "ngram_ram", False):
        raise SystemExit(" !! FATAL: --ngram-ram is an exl3 option controlling the PLE "
                         "n-gram table's residency (streamed-from-disk vs RAM); the "
                         "transformers backend has no EXL3 n-gram table")


def compute_cache_tokens(args, max_case_len: int) -> int | None:
    """
    KV-cache capacity to allocate BEFORE model.load() (cache tensors are
    allocated with the layers; in a layer split they land on the devices that
    own the attention modules).

      chunked: required = max(4096, load_max_chunk_size, max_case_len); the
               default (no --cache-tokens) is exactly the old auto size.
      bulk:    no cache by default (cache-free whole-sequence forward). If
               --cache-tokens IS given, allocate it anyway so the split
               placement matches an identically sized bench run; the bulk
               forward still ignores the cache (uses_kv_cache stays False,
               the allocation is recorded separately as cache_max_seq_len).
      --cache-tokens below the required capacity is refused, never clamped.
    """
    required = max(4096, args.load_max_chunk_size, max_case_len)
    if args.cache_tokens is not None:
        if args.cache_tokens < required:
            raise SystemExit(
                f" !! FATAL: --cache-tokens {args.cache_tokens} is below the required capacity "
                f"{required} (max(4096, load-max-chunk-size {args.load_max_chunk_size}, longest "
                f"case {max_case_len})) -- raise it, don't under-allocate")
        return round_up_page(args.cache_tokens)
    if args.execution == "chunked":
        return round_up_page(required)
    return None


# ---------------------------------------------------------------------------
# Pure, CPU-testable helpers for hybrid recurrent models (no torch needed)
# ---------------------------------------------------------------------------

def make_bulk_params() -> dict:
    """
    Params for the exl3 bulk route. Cache-free BY CONSTRUCTION: with no
    "cache"/"batch_shape"/"cache_seqlens" and recurrent_states absent, Qwen4's
    prepare_inputs -> recurrent_util.prepare_for_recurrence takes its no-op
    branch and every recurrent module runs stateless over the full literal
    sequence (GDN scans from zero state; PLELayer._history uses the
    [eos]x(ngram_size-1) + ids padding). Same dict a nonrecurrent bulk run
    always used.
    """
    return {"attn_mode": "flash_attn_nc"}


def ple_reference_history(ids, ctx: int, eos_id: int) -> list:
    """
    Ground-truth PLE token-history for a full stateless sequence: the exact
    hashing history PLELayer builds via _history(ids), i.e. the (ngram_size-1)
    context slots before token 0 filled with the PLE eos token, followed by the
    literal ids. The window the n-gram hash sees at global position p is
    ple_reference_history(ids, ctx, eos)[p : p + ctx + 1].
    """
    return [eos_id] * ctx + list(ids)


def ple_carry_update(carry, chunk_ids, ctx: int, eos_id: int) -> list:
    """
    The PLELayer id-history state write (non-speculative branch,
    id_state[slot, :ctx] = cat(carried_window, chunk)[-ctx:]), in pure Python.
    carry=None means a fresh cleared slot, whose context PLELayerState.clear()
    fills with the PLE eos token -- the EOS-padded sequence start. Chaining
    this across a case's chunks must reproduce ple_reference_history's windows
    at every position: this is the "real previous-token context across chunk
    boundaries, never a zero-restarted chunk" property the chunked recurrent
    route relies on and that the runtime verify_carry check re-derives.
    """
    hist = (list(carry) if carry is not None else [eos_id] * ctx) + list(chunk_ids)
    return hist[-ctx:] if ctx else []


def collect_ngram_table_modes(model) -> dict:
    """
    Walk the model's module tree after load() and record every NGramEmbedding's
    ACTUAL storage mode: None = not loaded yet, "*_disk" = streamed from disk,
    "*_ram" = resident in system RAM (see exllamav3/modules/ngram_embedding.py).
    Duck-typed by class name so the walk itself is CPU-testable against fakes;
    cycle-guarded.
    """
    modes: dict = {}
    seen: set = set()
    stack = list(getattr(model, "modules", []))
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        if type(m).__name__ == "NGramEmbedding":
            modes[str(getattr(m, "key", f"<unnamed_{len(modes)}>"))] = getattr(m, "mode", "<no mode attr>")
        stack.extend(getattr(m, "modules", []) or [])
    return modes


def ngram_residency_report(table_modes: dict, ram_requested: bool) -> tuple:
    """
    Summarize actual n-gram table residency and enforce a --ngram-ram request.
    Returns (report, error); error is not None exactly when a RAM request was
    made and the tables as found could not honor it: a disk-streamed table
    (the default streaming route), a mode we don't recognize, a table whose
    mode was never set, or NO n-gram table at all (the request would be
    vacuous -- almost certainly the wrong model directory). Without a request
    nothing is refused; the observed modes are still recorded (honest label of
    what actually ran). Never guesses residency from the config: the recorded
    values are the module attributes set by NGramEmbedding.load().
    """
    ram = sorted(k for k, m in table_modes.items() if isinstance(m, str) and m.endswith("_ram"))
    disk = sorted(k for k, m in table_modes.items() if isinstance(m, str) and m.endswith("_disk"))
    other = sorted(k for k, m in table_modes.items() if k not in ram and k not in disk)
    if not table_modes:
        residency = "none"
    elif not disk and not other:
        residency = "ram"
    elif not ram and not other:
        residency = "disk"
    else:
        residency = "mixed"
    report = {
        "ram_requested": bool(ram_requested),
        "tables": {k: table_modes[k] for k in sorted(table_modes)},
        "counts": {"total": len(table_modes), "ram": len(ram), "disk": len(disk), "other": len(other)},
        "residency": residency,
    }
    error = None
    if ram_requested:
        if not table_modes:
            error = ("--ngram-ram requested but the loaded model exposes no n-gram "
                     "(PLE) table: wrong model directory? refusing")
        elif residency != "ram":
            offenders = {"disk": disk, "unknown_or_unset_mode": other}
            error = ("--ngram-ram was requested but actual table residency is "
                     f"{residency!r}: refusing to collect labeled as RAM while the "
                     f"table(s) loaded otherwise: {offenders}")
    return report, error


def run_chunked_case(
    case,
    chunk: int,
    recurrent: bool,
    forward_chunk,
    rows_of,
    harvest_row,
    params_for_chunk,
    new_state=None,
    free_state=None,
    reset_pool=None,
    verify_carry=None,
    evidence: dict | None = None,
):
    """
    Teacher-forced chunked pass over ONE manifest case -- the chunked branch of
    the exl3 collector as a pure orchestration so CPU tests drive the exact
    same code path against fakes that honour the engine's contract.

    Injected callables:
      forward_chunk(s, e, params) -> logits for global positions s..e-1
      params_for_chunk(s, state)  -> the exl3 params dict (state is None when
                                     recurrent=False; the REAL implementation
                                     never sets "recurrent_history": no
                                     speculative/MTP state writes ever happen)
      rows_of(logits, s, e)       -> [(global_pos, row)] validating row<->position mapping
      harvest_row(row)            -> (top1, finite)
      recurrent=True additionally requires:
      new_state()                 -> Cache.get_new_state(): ONE cleared slot,
                                     GDN conv/delta state zeroed, PLE id context
                                     eos-filled (the EOS-padded sequence start)
      free_state(state)           -> return the slot to the pool
      reset_pool()                -> Cache.reset_states() barrier so no leaked slot can
                                     poison a later case even if freeing was botched
      verify_carry(state, e)      -> optional PLE-history re-derivation at every boundary;
                                     returns a note (recorded) or raises (case fails)

    State-carry contract (enforced here, mirroring recurrent_util): the same
    live state object is passed for EVERY chunk of the case, its position
    equals s before each forward and equals e after it (advance_recurrent_states
    runs inside model.forward) -- a chunk that silently ran from a zero state
    trips one of those checks. The state is released in a finally block on the
    success AND error paths, so independent cases never share dirty state.
    Returns (top1, nonfinite); exceptions propagate after cleanup.
    """
    if recurrent and (new_state is None or free_state is None or reset_pool is None):
        raise RuntimeError("recurrent chunked collection requires new_state, free_state and reset_pool")
    ev = evidence if evidence is not None else {}
    for k, empty in (("chunks", 0), ("states_created", 0), ("states_released", 0),
                     ("release_failures", []), ("carry_notes", [])):
        ev.setdefault(k, empty)
    L = case["len_ids"]
    positions = case["positions"]
    want = {p: j for j, p in enumerate(positions)}
    top1 = [None] * len(positions)
    nonfinite = []
    state = None
    try:
        if recurrent:
            state = new_state()
            ev["states_created"] += 1
            pos0 = getattr(state, "position", 0)
            if pos0 != 0:
                raise RuntimeError(f"newly allocated recurrent state starts at position {pos0}, not 0")
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            if recurrent:
                pos = getattr(state, "position", None)
                if pos != s:
                    raise RuntimeError(
                        f"recurrent state position {pos} != past_len {s} before chunk [{s},{e}): "
                        f"the state was not carried intact from the previous chunk")
            params = params_for_chunk(s, state)
            if "recurrent_history" in params:
                raise RuntimeError("recurrent_history must never be set by this collector "
                                   "(no speculative decoding / MTP verification)")
            logits = forward_chunk(s, e, params)
            if recurrent:
                pos = getattr(state, "position", None)
                if pos != e:
                    raise RuntimeError(
                        f"recurrent state position {pos} != consumed length {e} after chunk "
                        f"[{s},{e}) -- the engine did not advance the carried state (a "
                        f"zero-restarted chunk would quietly diverge from teacher forcing)")
                if verify_carry is not None:
                    note = verify_carry(state, e)
                    if note and note not in ev["carry_notes"]:
                        ev["carry_notes"].append(note)   # dedup: one note per distinct problem
            try:
                for p, row in rows_of(logits, s, e):
                    j = want.pop(p, None)
                    if j is None:
                        continue
                    top1[j], ok = harvest_row(row)
                    if not ok:
                        nonfinite.append(p)
            finally:
                del logits
            del params
            ev["chunks"] += 1
        if want:
            missing = sorted(want)
            raise RuntimeError(f"chunked run produced no logits for positions "
                               f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
    finally:
        if recurrent:
            if state is not None:
                try:
                    free_state(state)
                    ev["states_released"] += 1
                except Exception as fe:
                    ev["release_failures"].append(f"state.free(): {fe!r}")
            try:
                reset_pool()
            except Exception as re_:
                ev["release_failures"].append(f"cache.reset_states(): {re_!r}")
    return top1, nonfinite


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog = "collect_top1.py",
        description = "Teacher-forced top-1 collection over a frozen manifest.",
    )
    ap.add_argument("--manifest", required = True, help = "Manifest JSON (rocm_tools/rdna2/manifest.py output)")
    ap.add_argument("-m", "--model-dir", required = True, help = "Model directory for this backend")
    ap.add_argument("--backend", required = True, choices = ("exl3", "transformers"))
    ap.add_argument("-o", "--output", required = True, help = "Top-1 result JSON path to write")
    ap.add_argument("-d", "--device", default = "cuda:0",
                    help = "cuda:0 for exl3; transformers also accepts cpu (large reference "
                           "that cannot fit the V620)")
    ap.add_argument("--execution", choices = ("bulk", "chunked"), default = "bulk",
                    help = "exl3 only: bulk = cache-free whole-sequence forward; "
                           "chunked = Model.forward with params/cache in C-token steps")
    ap.add_argument("--chunk-size", type = int, default = None,
                    help = "exl3 chunked: tokens per cached forward step; 1 = autoregressive decode route")
    ap.add_argument("--load-max-chunk-size", type = int, default = 2048,
                    help = "exl3: max_chunk_size passed to model.load() (bounds the load-time "
                           "dummy forward, incl. the split autosplit; also the sanity bound on "
                           "--chunk-size)")
    ap.add_argument("--use-per-device", type = float, nargs = "+", default = None,
                    metavar = "GIB",
                    help = "exl3 only: FORCE a layer split (not TP) through the official "
                           "model.load(use_per_device=[...]) autosplit API, without a device "
                           "argument. At least two finite positive GiB budgets, one per visible "
                           "GPU in cuda:0, cuda:1... order; the visible device count must equal "
                           "it and every device must match --expect-arch. Placement is audited "
                           "after load and recorded in the artifact")
    ap.add_argument("--cache-tokens", type = int, default = None,
                    help = "exl3 only: KV-cache capacity allocated BEFORE load (must be >= "
                           "max(4096, --load-max-chunk-size, longest case)). Chunked: defaults to "
                           "that auto size. Bulk: no cache by default; giving this flag allocates "
                           "the cache anyway so a split places identically to an identically "
                           "sized bench run -- bulk forward still ignores it (uses_kv_cache=false)")
    ap.add_argument("--ngram-ram", action = "store_true",
                    help = "exl3 only: BEFORE model.load(), set config.infer_params."
                           "ngram_stream_from_disk=False so every PLE n-gram table loads into "
                           "system RAM instead of streaming from disk. The ACTUAL post-load "
                           "residency of every table is recorded in execution.ngram; if RAM was "
                           "requested and any table is not resident there (disk/unknown/no "
                           "tables), collection refuses to run instead of mislabeling")
    ap.add_argument("--expect-arch", default = "gfx1030",
                    help = "exl3 layer split only: required gcnArchName of EVERY visible GPU")
    ap.add_argument("--dtype", choices = DTYPE_KEYS, default = "auto",
                    help = "transformers only: load dtype. auto (default) keeps the "
                           "checkpoint's declared torch_dtype so a BF16 source gives a "
                           "native BF16 reference; fp16/bf16/fp32 force the cast. What "
                           "was requested, declared and effective is recorded either way")
    ap.add_argument("--attn-implementation", default = None,
                    help = "transformers only: pass attn_implementation to from_pretrained "
                           "(e.g. eager for the conservative reference). The actually "
                           "selected implementation is read back from model config and "
                           "recorded; left unset = Transformers default")
    ap.add_argument("--limit-cases", default = None,
                    help = "comma-separated case_ids to run (debug aid; total_positions stays "
                           "the FULL manifest count, so the artifact is flagged incomplete "
                           "and comparison refuses it by design)")
    return ap


def _harvest_row(torch, logits_row, vocab):
    """Float-upcast, finite-check, argmax over [0, vocab) only. Runs on the row's
    own device (a vocab-length slice + fp32 upcast, on-device argmax); only the
    resulting int crosses to CPU, never a full-vocabulary copy."""
    row = logits_row[:vocab].float()
    finite = bool(torch.isfinite(row).all())
    top1 = int(torch.argmax(row).item())
    del row
    return top1, finite


# ---------------------------------------------------------------------------
# exl3
# ---------------------------------------------------------------------------

def collect_exl3(args, manifest, vocab, cleanup_errors: list):
    import torch
    from exllamav3 import Cache, Config, Model, Tokenizer
    from exllamav3.constants import PAGE_SIZE as EXL_PAGE_SIZE
    from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
    from exllamav3.util import device_copy

    if not args.device.startswith("cuda"):
        raise SystemExit(f" !! exl3 backend needs a ROCm/CUDA device, got --device {args.device!r} "
                         f"(use --backend transformers --device cpu for host-RAM references)")

    budgets = multi_gpu.validate_use_per_device(args.use_per_device)
    plan = multi_gpu.plan_load_mode(torch, budgets, args.expect_arch)
    split = plan["mode"] == "layer_split"
    if split and args.device not in ("cuda:0", "cuda"):
        raise SystemExit(f" !! FATAL: --device must stay cuda:0 with --use-per-device "
                         f"(the split spans all visible devices; --device is unused there)")

    all_cases = manifest["cases"]
    cases = all_cases
    if args.limit_cases:
        wanted = [s.strip() for s in args.limit_cases.split(",")]
        cases = [c for c in all_cases if c["case_id"] in wanted]
        if not cases:
            raise SystemExit(f" !! --limit-cases matched no manifest case ({wanted})")

    model = None
    placement = None
    mem_after_load = None
    try:
        config = Config.from_directory(args.model_dir)
        if getattr(args, "ngram_ram", False):
            # BEFORE load: NGramEmbedding.load() reads this when the module itself
            # leaves stream_from_disk=None (the model classes do) -- same as
            # model_init.py's engine-supported --ngram_ram path. Actual residency is
            # verified from the loaded modules below, never taken from this flag.
            config.infer_params.ngram_stream_from_disk = False
        tokenizer = Tokenizer.from_config(config)
        av = int(tokenizer.actual_vocab_size)
        if av != vocab:
            raise SystemExit(
                f" !! tokenizer.actual_vocab_size {av} != manifest valid_vocab_size {vocab}: "
                f"manifest and model do not share a tokenizer -- refusing (cross-tokenizer misalignment)")
        model = Model.from_config(config)

        # Hybrid recurrent model? (Qwen4Exp: GDN linear-attention states + PLE
        # n-gram token history). Both exl3 routes support it through the engine's
        # own preparation (see module docstring); nothing about the collection
        # protocol changes -- only the state plumbing inside model.forward().
        rec_layers = list(model.get_recurrent_layers())
        recurrent = bool(rec_layers)
        ple_layers = [m for m in rec_layers if getattr(m, "ple_embedding", None) is not None]
        ngram_ctx = ple_layers[0].ple_embedding.context_len if ple_layers else None
        ple_eos = ple_layers[0].ple_embedding.eos_token_id if ple_layers else None

        execution = args.execution
        chunk = args.chunk_size if execution == "chunked" else None
        if execution == "chunked":
            if not chunk or chunk < 1:
                raise SystemExit(" !! --execution chunked requires --chunk-size C >= 1 (C=1 autoregressive)")
            if chunk > args.load_max_chunk_size:
                raise SystemExit(f" !! --chunk-size {chunk} exceeds --load-max-chunk-size "
                                 f"{args.load_max_chunk_size}; raise the latter")

        # Cache must exist BEFORE load so cache tensors (incl. the recurrent-state
        # slot pool: GDN conv/delta state, PLE conv + id-history) get allocated
        # with the layers (in a split: on the devices owning their modules).
        # Bulk without --cache-tokens stays cache-free exactly as before.
        max_case_len = max(c["len_ids"] for c in cases)
        cache_max_seq_len = compute_cache_tokens(args, max_case_len)
        cache = None
        if cache_max_seq_len is not None:
            cache = Cache(model, max_num_tokens = cache_max_seq_len)
        if recurrent and execution == "chunked" and cache is None:
            raise SystemExit(" !! recurrent chunked collection requires a cache (state slots); "
                             "this is a bug in cache sizing")

        if split:
            load_kwargs = {"max_chunk_size": args.load_max_chunk_size}
            if plan["budgets"] is not None:
                load_kwargs["use_per_device"] = plan["budgets"]
            model.load(**load_kwargs)   # NO device argument -- engine autosplit
            expect_idx = list(range(plan["gate"]["visible_devices"]))
            multi_gpu.sync_devices(torch, expect_idx)
            copy_after_load = dict(device_copy.stats)
            mem_after_load = multi_gpu.device_memory_snapshot(torch, expect_idx)
            placement = multi_gpu.audit_placement(model, expect_idx)
            print(f" -- loaded {args.model_dir} as LAYER SPLIT (budgets "
                  f"{plan['budgets']} GiB over {len(expect_idx)}x "
                  f"{args.expect_arch}); transformer modules per device: "
                  f"{placement['transformer_modules_per_device']}, cache layers per device: "
                  f"{placement['cache_layers_per_device']}, audit: "
                  f"{'OK' if placement['ok'] else 'FAILED'}", flush = True)
            if not placement["ok"]:
                for p in placement["problems"]:
                    print(f"    AUDIT FAIL: {p}", file = sys.stderr)
            ids_device = multi_gpu.device_key(expect_idx[0])
        else:
            model.load(device = args.device, max_chunk_size = args.load_max_chunk_size)
            ids_device = args.device

        # Where did the n-gram table(s) ACTUALLY land? Recorded either way; a
        # --ngram-ram request that did not materialize refuses the run here
        # (before any collection) instead of labeling disk streaming as RAM.
        ngram_report, ngram_error = ngram_residency_report(
            collect_ngram_table_modes(model), bool(getattr(args, "ngram_ram", False)))
        if ngram_error:
            raise SystemExit(f" !! FATAL: {ngram_error}")
        if ngram_report["tables"]:
            print(f" -- n-gram tables: {ngram_report['residency']} "
                  f"({ngram_report['counts']}, requested_ram={ngram_report['ram_requested']})",
                  flush = True)
        if recurrent:
            print(f" -- recurrent model: {len(rec_layers)} recurrent state layers, "
                  f"{len(ple_layers)} PLE layers; route = "
                  f"{'stateless full-sequence' if execution == 'bulk' else 'one cleared state carried per case'}",
                  flush = True)

        rec_stats = {"chunks": 0, "states_created": 0, "states_released": 0,
                     "release_failures": [], "carry_notes": []}

        def run_case(case, ev):
            ids_t = torch.tensor([case["ids"]], dtype = torch.long, device = ids_device)
            positions = case["positions"]
            top1 = [None] * len(positions)
            nonfinite = []
            if execution == "bulk":
                logits = model.forward(ids_t, make_bulk_params())
                if logits.dim() != 3 or logits.shape[1] != case["len_ids"] or logits.shape[2] < vocab:
                    raise RuntimeError(f"unexpected bulk logits shape {tuple(logits.shape)} "
                                       f"(want (1, {case['len_ids']}, >={vocab}))")
                try:
                    for j, p in enumerate(positions):
                        top1[j], ok = _harvest_row(torch, logits[0, p, :], vocab)
                        if not ok:
                            nonfinite.append(p)
                finally:
                    del logits
            else:
                def params_for_chunk(s, state):
                    # Fresh dict per chunk: the Embedding module fills
                    # params["input_ids"] with THIS chunk's literal ids and
                    # PLELayer builds its hashing history as
                    # cat(id_state[slot, :ctx], chunk_ids) -- the carried window
                    # in the live state IS the real previous-token context
                    # (eos-padded at sequence start by GDNState's slot clear).
                    # recurrent_states=None below only when the model has no
                    # recurrent layers at all (the pre-existing route).
                    return {
                        "attn_mode": "flash_attn",
                        "cache": cache,
                        "past_len": s,
                        "batch_shape": (1, cache_max_seq_len),
                        "recurrent_states": [state] if state is not None else None,
                    }

                def forward_chunk(s, e, params):
                    return model.forward(ids_t[:, s:e], params)

                def rows_of(logits, s, e):
                    if (logits.dim() != 3 or logits.shape[1] != e - s
                            or logits.shape[2] < vocab):
                        raise RuntimeError(f"unexpected chunked logits shape "
                                           f"{tuple(logits.shape)} (need (1, {e - s}, >={vocab})); "
                                           f"row-to-position mapping (row i = global s+i) unverified")
                    return ((s + i, logits[0, i, :]) for i in range(e - s))

                def verify_carry(state, e):
                    """Re-derive the PLE id-history the NEXT chunk will hash with
                    (pure ple_carry_update chain over the literal manifest ids)
                    and compare it to what the state slot ACTUALLY carries. A
                    mismatch means chunk boundaries lost token context -- hard
                    case failure; unreadable state internals are a note (no GPU
                    semantics can be checked from them)."""
                    if not ple_layers:
                        return None
                    notes = []
                    for pl in ple_layers:
                        ng = pl.ple_embedding
                        ctx = int(ng.context_len)
                        eos_id = int(ng.eos_token_id)
                        prefix = case["ids"][:e]
                        mm = getattr(pl, "mm_token_id", None)
                        if mm is not None:
                            # PLELayer._prepare_ids substitutes embedding-alias ids
                            # with the literal placeholder BEFORE the state stores
                            # the window; the expected value must do the same
                            prefix = [mm if t >= FIRST_MM_EMBEDDING_INDEX else t
                                      for t in prefix]
                        expected = ple_carry_update(None, prefix, ctx, eos_id)
                        try:
                            layer_state = state.cache.get_recurrent_layer((pl.layer_idx, 0))
                            _, id_state = layer_state.get_state_tensors()
                            carried = [int(t) for t in id_state[state.slot, :ctx].tolist()]
                        except Exception as e_acc:
                            notes.append(f"{pl.key}: {e_acc!r}")
                            continue
                        if carried != expected:
                            raise RuntimeError(
                                f"PLE token history desync at {pl.key} after position {e}: "
                                f"state carries {carried}, expected {expected} from the "
                                f"literal manifest prefix (EOS-padded sequence start)")
                    return ("PLE carry not externally verifiable: "
                            + "; ".join(notes)) if notes else None

                top1, nonfinite = run_chunked_case(
                    case, chunk, recurrent,
                    forward_chunk = forward_chunk,
                    rows_of = rows_of,
                    harvest_row = lambda row: _harvest_row(torch, row, vocab),
                    params_for_chunk = params_for_chunk,
                    new_state = cache.get_new_state if recurrent else None,
                    free_state = (lambda st: st.free()) if recurrent else None,
                    reset_pool = cache.reset_states if recurrent else None,
                    verify_carry = verify_carry if recurrent else None,
                    evidence = ev,
                )
            del ids_t
            return top1, nonfinite

        results = {}
        errors = []
        collected = 0
        audit_failed = split and placement is not None and not placement["ok"]
        if audit_failed:
            # Fail-closed, but auditable: zero positions collected makes the
            # artifact incomplete -> nonzero exit; the failed placement record
            # still lands in execution.layer_split.placement_audit.
            errors.append("layer-split placement audit failed: "
                          + "; ".join(placement["problems"])
                          + " -- no cases were collected on an unverified split")
        for case in ([] if audit_failed else cases):
            cid = case["case_id"]
            ev = {}
            try:
                top1, nonfinite = run_case(case, ev)
                if nonfinite:
                    errors.append(f"case {cid}: non-finite logit rows at "
                                  f"{len(nonfinite)} position(s)")
                results[cid] = {
                    "status": "ok",
                    "case_id": cid,
                    "ids_sha256": case["ids_sha256"],
                    "len_ids": case["len_ids"],
                    "positions": case["positions"],
                    "n_positions": len(case["positions"]),
                    "top1": top1,
                    "nonfinite_positions": sorted(nonfinite),
                }
                collected += len(case["positions"])
            except SystemExit:
                raise
            except Exception as e:
                results[cid] = {
                    "status": "error",
                    "case_id": cid,
                    "error": repr(e),
                    "traceback_tail": traceback.format_exc().splitlines()[-6:],
                }
                errors.append(f"case {cid}: {e!r}")
            for k in ("chunks", "states_created", "states_released"):
                rec_stats[k] += ev.get(k, 0)
            rec_stats["release_failures"] += [f"case {cid}: {f}" for f in ev.get("release_failures", [])]
            rec_stats["carry_notes"] += [f"case {cid}: {f}" for f in ev.get("carry_notes", [])]
            if recurrent and results[cid]["status"] == "ok":
                # one state per case: creation and release counts must stay in balance
                if ev.get("states_created", 0) != ev.get("states_released", 0):
                    errors.append(f"case {cid}: recurrent state slot accounting unbalanced "
                                  f"(created {ev.get('states_created')}, released {ev.get('states_released')})")
            print(f" -- exl3/{execution}: case {cid:20} -> {results[cid]['status']} "
                  f"({results[cid].get('n_positions', 0)} positions)", flush = True)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        errors += [f"recurrent-state release failed: {f}" for f in rec_stats["release_failures"]]

        execution_info = {
            "mode": execution,
            "chunk_size": chunk,
            "attn_mode": "flash_attn_nc" if execution == "bulk" else "flash_attn",
            "uses_kv_cache": execution == "chunked",
            "cache_max_seq_len": cache_max_seq_len,
            "cache_requested_tokens": args.cache_tokens,
            "exl3_page_size": EXL_PAGE_SIZE,
            "load_max_chunk_size": args.load_max_chunk_size,
            "input_ids_device": ids_device,
            "load_mode": plan["mode"],
            "layer_split": ({
                "enabled": True,
                "use_per_device_gib": plan["budgets"],
                "budget_semantics": "GiB per visible GPU (engine converts int(gib*1024**3)), "
                                     "applied ON TOP of memory already allocated at load() time; "
                                     "maps 1:1 to visible order",
                "load_call": "model.load(use_per_device=..., max_chunk_size=...) -- no device "
                             "argument (engine forbids combining them)",
                "gate": plan["gate"],
                "placement_audit": placement,
                "memory_after_load_per_device": mem_after_load,
                "device_copy_stats_after_load": copy_after_load,
                "device_copy_stats_at_end": dict(device_copy.stats),
            } if split else {
                "enabled": False,
                "use_per_device_gib": None,
                "placement_audit": None,
            }),
            "dtype_label": "EXL3 quantized weights; logits upcast per row for argmax",
            "teacher_forcing": "manifest literal IDs only; no generated continuation",
            "evidence_status": ("candidate teacher-forced top-1 evidence; bulk/chunk agreement "
                                "localizes kernel divergence but proves neither route "
                                "independently correct -- NOT ground truth"),
            "recurrent": ({
                "model_recurrent": True,
                "recurrent_state_layers": len(rec_layers),
                "ple_layers": len(ple_layers),
                "ple_context_len": ngram_ctx,
                "ple_eos_token_id": ple_eos,
                "route": ("stateless_full_sequence" if execution == "bulk"
                          else "one_cleared_state_carried_per_case"),
                "state_lifecycle": {
                    "states_created": rec_stats["states_created"],
                    "states_released": rec_stats["states_released"],
                    "release_failures": rec_stats["release_failures"],
                    "chunks": rec_stats["chunks"],
                    "slot_pool_reset_between_cases": execution == "chunked",
                },
                "ple_history": ("engine_carried_window_verified_per_boundary"
                                if execution == "chunked" and ple_layers else
                                "stateless_eos_padded_full_sequence"),
                "carry_notes": rec_stats["carry_notes"],
                "speculative_or_mtp": False,
                "recurrent_history_param_ever_set": False,
            } if recurrent else None),
            "ngram": ngram_report,
        }
        return results, errors, collected, execution_info

    finally:
        # Normal cleanup belongs here and must not swallow problems: teardown errors
        # land in cleanup_errors, which run() folds into the artifact's error list,
        # marking the result incomplete (fail-closed). No os._exit anywhere.
        if model is not None:
            try:
                model.unload()
            except Exception as e:
                msg = f"model.unload() during cleanup: {e!r}"
                cleanup_errors.append(msg)
                print(f" !! {msg}", file = sys.stderr)


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------

def collect_transformers(args, manifest, vocab, cleanup_errors: list):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    # Declared checkpoint dtype, read straight from config.json (for the "auto"
    # resolution and for honest labeling of whatever actually loaded).
    declared_dtype = None
    cfg_path = Path(args.model_dir) / "config.json"
    if cfg_path.is_file():
        declared_dtype = json.loads(cfg_path.read_text(encoding = "utf-8")).get("torch_dtype")
    choice, dtype_note = resolve_dtype_choice(args.dtype, declared_dtype)
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                   "fp32": torch.float32}[choice]

    cases = manifest["cases"]
    if args.limit_cases:
        wanted = [s.strip() for s in args.limit_cases.split(",")]
        cases = [c for c in cases if c["case_id"] in wanted]
        if not cases:
            raise SystemExit(f" !! --limit-cases matched no manifest case ({wanted})")

    device = torch.device(args.device)
    model = None
    used_dtype_kw = None
    used_attn_kw = False
    effective_dtype = None
    effective_attn = None
    results: dict = {}
    errors: list = []
    collected = 0
    try:
        # from_pretrained takes these options through **kwargs, so signature
        # inspection cannot see them -- pass them for real, and fall back only
        # on TypeError (unexpected keyword). dtype replaced torch_dtype in
        # current Transformers; the ACTUAL attention implementation is read
        # back from the model config afterwards, which is authoritative.
        attempts = [("dtype", True), ("dtype", False), ("torch_dtype", True), ("torch_dtype", False)]
        last_type_error = None
        for dtype_kw, with_attn in attempts:
            load_kwargs = {dtype_kw: torch_dtype}
            if with_attn and args.attn_implementation:
                load_kwargs["attn_implementation"] = args.attn_implementation
            try:
                model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
                used_dtype_kw = dtype_kw
                used_attn_kw = bool(with_attn and args.attn_implementation
                                    and "attn_implementation" in load_kwargs)
                break
            except TypeError as e:
                last_type_error = e
        if model is None:
            raise last_type_error
        if args.attn_implementation and not used_attn_kw:
            print(f" !! this transformers build rejected attn_implementation="
                  f"{args.attn_implementation!r}; recording what the config actually selected",
                  file = sys.stderr)

        model.to(device)
        model.eval()
        effective_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        effective_attn = getattr(model.config, "_attn_implementation", None) \
            or getattr(model.config, "attn_implementation", None)

        for case in cases:
            cid = case["case_id"]
            try:
                ids_t = torch.tensor([case["ids"]], dtype = torch.long, device = device)
                with torch.inference_mode():
                    out = model(input_ids = ids_t, use_cache = False)
                logits = out["logits"]
                if (logits.shape[0] != 1 or logits.shape[1] != case["len_ids"]
                        or logits.shape[-1] < vocab):
                    raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)} "
                                       f"for ids (1, {case['len_ids']}) with vocab >= {vocab}")
                top1 = [None] * len(case["positions"])
                nonfinite = []
                for j, p in enumerate(case["positions"]):
                    top1[j], ok = _harvest_row(torch, logits[0, p, :], vocab)
                    if not ok:
                        nonfinite.append(p)
                del logits, out
                if nonfinite:
                    errors.append(f"case {cid}: non-finite logit rows at "
                                  f"{len(nonfinite)} position(s)")
                results[cid] = {
                    "status": "ok",
                    "case_id": cid,
                    "ids_sha256": case["ids_sha256"],
                    "len_ids": case["len_ids"],
                    "positions": case["positions"],
                    "n_positions": len(case["positions"]),
                    "top1": top1,
                    "nonfinite_positions": sorted(nonfinite),
                }
                collected += len(case["positions"])
            except Exception as e:
                results[cid] = {
                    "status": "error",
                    "case_id": cid,
                    "error": repr(e),
                    "traceback_tail": traceback.format_exc().splitlines()[-6:],
                }
                errors.append(f"case {cid}: {e!r}")
            print(f" -- transformers: case {cid:20} -> {results[cid]['status']} "
                  f"({results[cid].get('n_positions', 0)} positions)", flush = True)

        converted = (declared_dtype is not None and effective_dtype != str(declared_dtype))
        execution_info = {
            "mode": "hf_forward_no_cache",
            "device_requested": args.device,
            "device_effective": str(device),
            "dtype_requested": args.dtype,
            "dtype_resolved": choice,
            "dtype_resolution_note": dtype_note,
            "dtype_kwarg_used": used_dtype_kw,
            "dtype_effective": effective_dtype,
            "dtype_converted_from_checkpoint": converted,
            "checkpoint_declared_torch_dtype": declared_dtype,
            "attn_implementation_requested": args.attn_implementation,
            "attn_implementation_kwarg_passed": used_attn_kw,
            "attn_implementation_effective": effective_attn,
            "dtype_label": (
                f"{effective_dtype} converted from declared {declared_dtype}" if converted
                else f"native {effective_dtype}" + (f" (declared {declared_dtype})"
                                                    if declared_dtype else " (checkpoint declares no dtype)")),
            "transformers_version": transformers.__version__,
            "teacher_forcing": "manifest literal IDs only; use_cache=False; no generated continuation",
            "evidence_status": ("candidate teacher-forced top-1 evidence; agreement proves no "
                                "route independently correct -- NOT ground truth"),
        }
        return results, errors, collected, execution_info

    finally:
        # Release the HF model on the normal path; errors here are surfaced and
        # recorded (-> incomplete artifact -> nonzero exit), not swallowed.
        try:
            model = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception as e:
            msg = f"transformers cleanup: {e!r}"
            cleanup_errors.append(msg)
            print(f" !! {msg}", file = sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def cuda_devices():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"count": 0}
        devs = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devs.append({
                "index": i, "name": p.name,
                "gcnArchName": getattr(p, "gcnArchName", None),
                "total_memory_bytes": getattr(p, "total_memory", None),
            })
        return {"count": len(devs), "devices": devs}
    except Exception as e:
        return {"error": repr(e)}


def run(args) -> int:
    check_cli_options(args)   # refuses EXL3-only options on transformers, validates budgets
    manifest = read_json(args.manifest)
    errors = validate_manifest(manifest)
    if errors:
        print(" !! manifest invalid, refusing to collect:", file = sys.stderr)
        for e in errors:
            print(f"    {e}", file = sys.stderr)
        return 1
    vocab = manifest["valid_vocab_size"]
    # total_positions is ALWAYS the full-manifest count, independent of any
    # --limit-cases subset that actually ran.
    full_total = sum(len(c["positions"]) for c in manifest["cases"])

    cleanup_errors: list[str] = []
    if args.backend == "exl3":
        results, case_errors, collected, execution_info = collect_exl3(args, manifest, vocab, cleanup_errors)
    else:
        results, case_errors, collected, execution_info = collect_transformers(args, manifest, vocab, cleanup_errors)

    errors_list = list(case_errors) + list(cleanup_errors)
    if args.limit_cases and collected != full_total:
        errors_list.append(f"--limit-cases used: {collected}/{full_total} positions of the "
                           f"FULL manifest collected; this artifact is partial by design")
    if collected != full_total:
        errors_list.append(f"incomplete: collected {collected}/{full_total} positions")
    complete = (collected == full_total and not errors_list)

    out = {
        "format": TOP1_FORMAT,
        "created_utc": now_utc(),
        "tool": "rocm_tools/rdna2/collect_top1.py",
        "backend": args.backend,
        "device": args.device,
        "model_dir": str(Path(args.model_dir).resolve()),
        "model_fingerprint": model_fingerprint(args.model_dir),
        "repo_git_commit": git_commit(repo_root()),
        "env": {
            "python": python_env(),
            "torch": torch_env(),
            "visible_gpu": visible_gpu_env(),
            "exl3_rocm_switches": rocm_patch_env(),
            "cuda_devices": cuda_devices(),
        },
        "params": vars(args),
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "position_semantics": POSITION_SEMANTICS,
        "valid_vocab_size": vocab,
        "config_vocab_size": manifest.get("config_vocab_size"),
        "execution": execution_info,
        "cases": results,
        "total_positions": full_total,
        "collected_positions": collected,
        "complete": complete,
        "errors": errors_list,
    }
    write_json(args.output, out)
    print(f" -- wrote {args.output}: complete={complete} "
          f"positions={collected}/{full_total} errors={len(errors_list)}")
    if not complete:
        for e in errors_list:
            print(f"    ERR {e}", file = sys.stderr)
        return 1
    return 0


def main(argv = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        print(f" !! collection failed: {e}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
