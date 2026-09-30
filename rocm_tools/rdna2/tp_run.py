#!/usr/bin/env python3
"""TP2-vs-layer-split evaluation CLI for the V620 pair (Phase 6 harness).

One frozen-prompt job set, run either as the official layer-split load
(same engine API as qwen_mtp_run: Model.load(use_per_device=[GiB, GiB]), no
device) or as 2-way tensor parallelism (Model.load(..., tensor_p=True,
tp_backend='nccl', tp_output_device='cuda:1') -- 'tensor_parallel' is the CLI
wording, tensor_p is this fork's keyword). The MTP head stays UNSHARDED on
cuda:1 in both executions, loaded before the target, so the only A/B variable
is target placement. Dense Qwen3-8B / MoE Qwen3-30B-A3B checkpoints define no
'mtp' component: --mode ar runs them without constructing (or even
importing) a draft; --mode mtp fails before any construction. Qwen3.8 keeps
the native mtp.* component resident even in AR controls, exactly like
qwen_mtp_run, for device-layout parity. There is NO standalone-head override
here: the MTP config/weights always come from --model.

Placement truth: after a TP load the parent model.modules are unloaded shells
(the loader exports each CPU-loaded module to the ranks and unloads it), so
multi_gpu.audit_placement on the PARENT is wrong for TP. The TP audit
therefore interrogates each worker's ACTUAL local_context via
model.tp_worker_dispatch_wait_multi(model.active_devices, tp_audit_rank, ())
-- a top-level, picklable function that runs inside every rank and reports
its PID, device, gfx arch (suffix-normalized, full string preserved), the
module shards it imported WITH EVIDENCE (stub flags, linear in/out features,
quant type, weight presence, head counts, num_local_experts and routing
ranges), plan shard slices and expert ownership (per-expert keys AND
unit='experts' rows), cache/recurrent state geometry (per-TENSOR devices
checked, not just the layer label; host-side PLE id_state allowed, foreign
CUDA rejected), per-rank torch allocation/peak, its local PLE/NGram
residence (mode, CPU table bytes/tensor count/disk handles), and its own
/proc RSS/PSS/VmSwap. The parent aggregates (never trusting budgets or the
pseudo output rank's pipe alone): a rank counts as computing only from
imported-module evidence -- a nonempty plan row or a transformer flag on a
stub is not proof -- active_devices must be exactly {0,1} (one rank can
never be labelled TP2), unexpected/duplicate rank records reject, every
module must sit on its rank's device, and the output rank's PID must BE the
parent PID (so parent memory is attributed once, not double-counted, via a
per-PID RSS/PSS sum). Engram keys are captured from the CONSTRUCTED parent
tree BEFORE loading and the aggregator then requires exactly ONE physical
CPU-RAM owner per expected key: a silently absent table (upstream hides it
by disk-streaming), mode None, RAM mode without tensors, unknown/zero byte
counts, tables resident on CUDA despite a RAM mode, disk handles, duplicated
or unplanned owners, and NO non-faulting mincore proof that the table's own
pages are fully resident (process-wide VmSwap>0 with resident table pages is
a diagnostic note, not a false failure; VmSwap alone never passes) (a
separate task implements the single-owner RAM mode; the plan's "distinguish
PLE module replication from physical table duplication" is enforced from the
actual tensors and per-PID RSS/PSS, never from CUDA reserved). A tableless
D/M checkpoint records the n-gram check as a no-op instead of failing. The
parent model shells are additionally scanned so a shell that still holds
table tensors counts as a second owner. TP power-policy phase boundaries use
a torch facade whose cuda.synchronize dispatches tp_sync_rank INTO each
owning rank (the parent cannot drain a spawned context); LS keeps plain
torch and every routed sync is reported.

KV cache policy: the selected production setting is KEY 5-BIT / VALUE 4-BIT
quantized caches for BOTH target and draft, built as
Cache(model, ..., layer_type=CacheLayer_quant, k_bits=5, v_bits=4) (exported
by exllamav3.cache). QSA attention modules auto-map CacheLayer_quant to
CacheLayer_qsa_quant inside the engine and keep the FP16 raw_k/pooled
indexer planes; GDN/PLE recurrent state is NOT KV and stays at its current
FP32/BF16 types. --cache-fp16 is an EXPLICIT diagnostic baseline only
(CacheLayer_fp16, no bits kwargs); it never becomes the default and bits
overrides are rejected with it. The requested policy plus the ACTUAL loaded
cache state are captured in the report: every worker/parent cache-layer
record now carries the layer class name, k_bits/v_bits and the named packed
qk/qv/sk/sv tensors (shape/dtype/device/bytes) alongside any FP16 planes,
and validate_cache_alignment fails closed when an observed full-attention
cache class or bit width does not match the request (a silent engine-side
fallback to FP16 is a hard error; a GDN-only draft with zero KV layers is
legitimate and recorded as such, never fabricated as quantized).

Validation (--validate-finite) never masquerades as throughput: report
"validation_only" gates consumption, the MAIN RETURN LOGITS are checked on
the parent's model.forward, an optional picklable installer wraps each rank
module's forward INSIDE the worker to isfinite every actually-returned
tensor (unobserved inner values are never asserted), and per-rank check
counts are collected AFTER inference (never at install time) and must be
positive on every rank. The full worker audit is re-collected after
inference and before unload so reported per-rank peaks/PSS are real; the
top-level peak_allocated_bytes for TP is explicitly labelled
parent_process_only instead of passing the parent's cuda:0 stats off as the
child rank's peak. Timing never syncs per token; it mirrors qwen_mtp_run:
one burst-aware delivery sample per iterate that advanced job.new_tokens,
engine rate kept separately, cache hits / short outputs / argmax-length
violations fail closed. Fixed power policy is delegated to the reviewed
power_policy client: batch1 prefill auto / peak from the first
draft+verify+decode compute / auto on drain; batch>1 held peak; the context
manager restores on every exit path. Workers are drained and models unloaded
even when initialization failed halfway (a partially-spawned TP context is
torn down explicitly because Model.unload() only does so once loaded_tp was
set, and if the engine's own destroy raises, the harness best-effort joins
and terminates ONLY its real child processes with bounded waits -- never
the in-process pseudo rank); JSON is written in a finally block; no
os._exit, and nothing outside this process tree is ever timed out or killed.

--capacity-only separates memory CAPACITY from actual long-context inference:
it keeps the frozen-prompt capacity gate, the target+draft cache construction,
the loads and the existing placement / actual-cache / Engram audits, then stops
BEFORE any Generator is constructed -- no prefill/decode, no power-phase
transitions, no runs/groups. The report carries capacity_only true,
validation_only true and complete true (exit code still reflects cleanup
failures: the shared finally drains workers and unloads models, and a cleanup
error flips complete to false). Per-rank torch facts are recorded as LOAD-only
allocator high-water marks (weights + KV caches + recurrent state as loaded),
NEVER disguised as post-inference peaks, in report["capacity_report"] for TP
and LS alike: allocation evidence only, not a usable-runtime-context claim.

Batched throughput: every group additionally reports group_throughput_metrics
derived from the per-job delivery events the loop already sampled (pure CPU,
no extra GPU synchronization): a common decode window from max(first delivery)
to min(last delivery), cumulative tokens counted with a step function at each
boundary, aggregate tps = counted tokens / window duration (deliberately NOT a
mean or sum of per-job tps), per-job window tps, summed input tokens, the
engine-measured time_prefill semantics from job.py (first-prefill-start ->
first token, so max(time_prefill) approximates the prefill makespan, not a
wall-clock phase boundary), engine TTFT min/max, end-to-end total outputs /
group wall, and cross-job medians of the per-job engine/observed decode tps.
Groups bind to absolute report row indices (run_indices) beside the hashes.
An invalid overlap yields null window fields, never bogus rates.

Usage (native stack via PYTHONPATH, root starts the privileged power helper):
    python3 -m rocm_tools.rdna2.tp_run -m MODEL --prompts-json P.json \
        --execution tp --mode mtp --power-socket SOCK --output OUT.json
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import resource
import statistics
import sys
import time
import types
from pathlib import Path

# multi_gpu is stdlib-only (CPU-safe), so importing its residence vocabulary at
# module scope keeps the pure-data aggregators testable without the lazy stack.
from rocm_tools.rdna2.multi_gpu import NGRAM_RAM_MODES
# Frozen-prompt validation, hashing, burst-aware rate math, result keys and the
# nofile raise are SHARED with the reviewed qwen_mtp_run CLI: identical artifact
# semantics, imported rather than duplicated. qwen_mtp_run itself stays
# CPU-importable (native imports are lazy inside its run()).
from rocm_tools.rdna2.qwen_mtp_run import (MODES, RESULT_KEYS, _raise_nofile,
                                           delivery_rate, prompt_sha256, validate_prompts)

EXECUTIONS = ("ls", "tp")
TP_BACKEND = "nccl"                      # ROCm/RCCL path; 'native' full port is NOT assumed
TP_OUTPUT_DEVICE = "cuda:1"              # logits gathered where the unsharded MTP head lives
DRAFT_DEVICE = "cuda:1"
EXPECT_ARCH = "gfx1030"
PAGE = 256
# Selected production KV policy (user-mandated going forward): 5-bit keys / 4-bit
# values via CacheLayer_quant; QSA attention auto-maps to CacheLayer_qsa_quant
# inside the engine. --cache-fp16 is an explicit diagnostic baseline ONLY.
DEFAULT_KV_K_BITS = 5
DEFAULT_KV_V_BITS = 4
# Cache-layer classes an ACTUAL K5/V4 quantized request may legitimately load
# (the requested CacheLayer_quant itself, plus the QSA variant Attention maps
# to). Anything else under a quant request is a silent fallback and fails.
KV_QUANT_CLASSES = frozenset({"CacheLayer_quant", "CacheLayer_qsa_quant"})
KV_FP16_CLASSES = frozenset({"CacheLayer_fp16", "CacheLayer_qsa"})
# MoE expert-parallel plan keys carry the owned expert index; count what a rank
# actually holds rather than trusting the requested split.
_EXPERT_KEY_RE = re.compile(r"(?:^|\.)experts?[._](\d+)")
# Per-module facts that PROVE a rank imported real, non-stub shards: head
# splits, owned-expert counts and routing ranges. Stubs carry the flags but no
# loaded weights, so the aggregator requires positive evidence, not a plan row.
_FACT_INTS = ("num_q_heads", "num_kv_heads", "num_local_experts",
              "routing_first", "routing_last")
_LINEAR_WEIGHT_ATTRS = ("weight", "q_weight", "weight_q", "weight_t", "embedding")
_LINEAR_QTYPE_ATTRS = ("qtype", "quant_type", "quantization", "quant")
_LINEAR_CAP = 8           # representative geometries only; counts stay aggregate
_LAYER_GEOM_CAP = 4


def build_parser():
    ap = argparse.ArgumentParser(
        description="TP2 vs layer-split evaluation on the V620 pair (frozen prompts, "
                    "worker-side TP audit, JSON report even on failure)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-m", "--model", required=True, help="model dir (target; + mtp component where the "
                    "checkpoint has one -- the draft always comes from here, no standalone override)")
    ap.add_argument("--prompts-json", required=True, help="frozen prompt ids file (same shape as qwen_mtp_run)")
    ap.add_argument("--execution", choices=EXECUTIONS, required=True,
                    help="ls = official layer-split autosplit load; tp = 2-way tensor parallel over both GPUs")
    ap.add_argument("--mode", choices=MODES, required=True,
                    help="mtp requires the checkpoint's own mtp component; dense/MoE models without one "
                         "run ar (no draft is constructed at all)")
    ap.add_argument("--draft-tokens", type=int, default=4, help="draft-window ceiling")
    ap.add_argument("--dynamic-draft", action="store_true", help="let the calibrator shrink drafts")
    ap.add_argument("--draft-confidence", type=float, default=0.4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--cache-tokens", type=int, default=8704)
    ap.add_argument("--cache-k-bits", type=int, default=DEFAULT_KV_K_BITS,
                    help="quantized KV cache key bits (2..8); the selected default is 5 (K5/V4). "
                         "Ignored only by the explicit --cache-fp16 diagnostic baseline")
    ap.add_argument("--cache-v-bits", type=int, default=DEFAULT_KV_V_BITS,
                    help="quantized KV cache value bits (2..8); the selected default is 4 (K5/V4). "
                         "Ignored only by the explicit --cache-fp16 diagnostic baseline")
    ap.add_argument("--cache-fp16", action="store_true",
                    help="DIAGNOSTIC ONLY: load FP16 KV caches instead of the default K5/V4 quant "
                         "selection; never the default, and incompatible with bit overrides")
    ap.add_argument("--max-chunk-size", type=int, default=2048)
    ap.add_argument("--use-per-device", type=float, nargs=2, default=[28, 28], metavar=("GiB0", "GiB1"),
                    help="load budget per GPU in GiB")
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--power-socket", required=True, help="power_switch_server.py Unix socket")
    ap.add_argument("--output", required=True, help="report JSON path (written even on failure)")
    ap.add_argument("--validate-finite", action="store_true",
                    help="explicit isfinite checks (parent logits + per-rank forwards); validation-only, "
                         "never steady performance")
    ap.add_argument("--capacity-only", action="store_true", default=False,
                    help="load target+draft caches, run the existing placement/actual-cache/Engram "
                         "audits, record LOAD-only allocator facts, then exit BEFORE any Generator or "
                         "inference: allocation evidence only, never a usable-runtime-context claim")
    return ap


def validate_args(args):
    """All argument/shape/capacity math is checked before any model is constructed."""
    if args.execution not in EXECUTIONS:
        raise ValueError(f"--execution must be one of {EXECUTIONS}, got {args.execution!r}")
    for name in ("draft_tokens", "batch_size", "cache_tokens", "max_chunk_size", "new_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if any(not math.isfinite(b) or b <= 0 for b in args.use_per_device):
        raise ValueError("--use-per-device budgets must be finite and positive "
                         "(a 0 would silently exclude a device from the split)")
    if not math.isfinite(args.draft_confidence) or not 0 < args.draft_confidence <= 1:
        raise ValueError("--draft-confidence must be finite and in (0, 1]")
    if args.cache_tokens % PAGE or args.max_chunk_size % PAGE:
        raise ValueError("cache and chunk capacities must be multiples of 256")
    if args.mode == "ar" and args.dynamic_draft:
        raise ValueError("--dynamic-draft requires --mode mtp; AR has no draft proposals")
    # KV cache selection: K5/V4 quant is the default and the only production
    # policy; CacheLayer_quant itself asserts 2..8, so reject early with a
    # CLI-facing message. --cache-fp16 is diagnostic-only and takes no bits.
    for name, val in (("--cache-k-bits", args.cache_k_bits),
                      ("--cache-v-bits", args.cache_v_bits)):
        if not 2 <= val <= 8:
            raise ValueError(f"{name} must be in 2..8 (CacheLayer_quant asserts the same range)")
    if args.cache_fp16 and (args.cache_k_bits != DEFAULT_KV_K_BITS
                            or args.cache_v_bits != DEFAULT_KV_V_BITS):
        raise ValueError("--cache-fp16 is the diagnostic FP16 baseline; --cache-k-bits/"
                         "--cache-v-bits only apply to the default K5/V4 quant selection")
    return args


def requested_cache(args):
    """The cache policy this invocation demands, captured verbatim in the
    report so requested-vs-observed can be checked against the ACTUAL loaded
    layers (no silent fallback)."""
    if args.cache_fp16:
        return {"policy": "fp16-diagnostic", "layer_type": "CacheLayer_fp16",
                "k_bits": None, "v_bits": None,
                "note": "explicit diagnostic baseline; the selected policy remains K5/V4 quant"}
    return {"policy": "quant", "layer_type": "CacheLayer_quant",
            "k_bits": args.cache_k_bits, "v_bits": args.cache_v_bits,
            "note": "user-selected K5/V4 KV for target AND draft; QSA attention auto-maps to "
                    "CacheLayer_qsa_quant (FP16 raw_k/pooled indexer planes retained); GDN "
                    "recurrent state is not KV and stays FP32/BF16"}


def required_cache_tokens(prompts, new_tokens, draft_tokens):
    """Page-rounded sum of prompt + generated + draft-window per job in one group."""
    return sum(((len(p["ids"]) + new_tokens + draft_tokens + PAGE - 1) // PAGE) * PAGE
               for p in prompts)


def _runtime_libraries():
    """Actual loaded runtime paths; catches duplicate HSA/SDK copies that env alone misses."""
    names = ("libhsa-runtime64", "libamdhip64", "librccl", "librocprofiler-sdk")
    found = {name: set() for name in names}
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            for name in names:
                if name in Path(fields[5]).name:
                    found[name].add(fields[5])
        return {name: sorted(paths) for name, paths in found.items()}
    except OSError as e:
        return {"error": repr(e)}


def _proc_mem():
    """Own-process RSS/PSS/VmSwap (KB) plus minor/major fault counters from
    /proc (missing fields are None, never guessed)."""
    out = {"vm_rss_kb": None, "pss_kb": None, "vm_swap_kb": None,
           "minflt": None, "majflt": None}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                for field, key in (("VmRSS:", "vm_rss_kb"), ("VmSwap:", "vm_swap_kb")):
                    if line.startswith(field):
                        try:
                            out[key] = int(line.split()[1])
                        except (IndexError, ValueError):
                            pass
    except OSError:
        pass
    try:
        with open("/proc/self/smaps_rollup", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Pss:"):
                    out["pss_kb"] = int(line.split()[1])
                    break
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/self/stat", "r", encoding="utf-8") as f:
            # fields after the parenthesised comm: state=+0, minflt=+7, majflt=+9
            after = f.read().rsplit(")", 1)[-1].split()
        out["minflt"] = int(after[7])
        out["majflt"] = int(after[9])
    except (OSError, ValueError, IndexError):
        pass
    return out


# ---------------------------------------------------------------------------
# Table-page residency: mincore(2) evidence, NOT process-wide VmSwap guesses.
# A 30 GiB Engram table in a 40 GiB-RSS process whose ZFS-reclaimable host
# happens to have tens of MB of unrelated pages swapped out is RAM-resident
# for our purposes iff the TABLE's own pages are. mincore reports which pages
# of a mapping have RAM backing WITHOUT faulting them in, without mlock, and
# without touching inference state; the kernel vector costs 1 byte per page
# (~8 MB for a 31 GiB table) and is never copied to or from the table.
# ---------------------------------------------------------------------------

_LIBC = None


def _mincore_libc():
    global _LIBC
    if _LIBC is None:
        lib = ctypes.CDLL("libc.so.6", use_errno=True)
        lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        lib.mincore.restype = ctypes.c_int
        _LIBC = lib
    return _LIBC


def _page_span(address, nbytes, page_size):
    """Page-aligned span covering [address, address+nbytes): unaligned heads
    and tails each cost one extra page."""
    aligned = (address // page_size) * page_size
    return aligned, (address + nbytes - 1) // page_size - aligned // page_size + 1


def _mincore_range(address, nbytes, page_size=None):
    """Residency facts for one byte range; 'error' is set (and all_resident
    is False) whenever residency cannot be PROVEN -- fail closed."""
    p = int(page_size or resource.getpagesize())
    res = {"page_size": p, "bytes": int(nbytes), "pages_total": None,
           "pages_resident": None, "pages_nonresident": None,
           "all_resident": False, "error": None}
    try:
        address, nbytes = int(address), int(nbytes)
        if address <= 0 or nbytes <= 0:
            raise ValueError(f"invalid range address={address} bytes={nbytes}")
        aligned, total = _page_span(address, nbytes, p)
        res.update(aligned_start=aligned, pages_total=total)
        vec = ctypes.create_string_buffer(total)
        ctypes.set_errno(0)
        rc = _mincore_libc().mincore(ctypes.c_void_p(aligned),
                                     ctypes.c_size_t(total * p), vec)
        if rc != 0:
            raise OSError(f"mincore rc={rc}: {os.strerror(ctypes.get_errno())}")
        resident = sum(b & 1 for b in vec.raw)
        res.update(pages_resident=resident, pages_nonresident=total - resident,
                   all_resident=resident == total)
    except (OSError, ValueError, TypeError) as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def _table_residency(t):
    """mincore probe of one tensor's actual storage: CPU-contiguous required
    (a view's byte range is only meaningful for contiguous layouts). Nothing
    is read through, faulted in, or copied; metadata + one kernel query."""
    dev = str(getattr(t, "device", None))
    if dev != "cpu":
        return {"error": f"table tensor is not a CPU tensor (device {dev})"}
    if not bool(getattr(t, "is_contiguous", lambda: True)()):
        return {"error": "table tensor view is non-contiguous; byte range unprovable"}
    try:
        addr, nbytes = int(t.data_ptr()), int(t.numel()) * int(t.element_size())
    except Exception as e:
        return {"error": f"storage handle unreadable: {e!r}"}
    return _mincore_range(addr, nbytes)


def measure_ngram_residency(tables):
    """Aggregate mincore evidence over every table tensor of one n-gram module."""
    per, tot, res_n, err = [], 0, 0, None
    for i, t in enumerate(tables or []):
        r = _table_residency(t)
        r["index"] = i
        per.append(r)
        if r.get("error"):
            err = err or r["error"]
            continue
        tot += r["pages_total"]
        res_n += r["pages_resident"]
    return {"probe": "mincore", "tensors": per, "pages_total": tot,
            "pages_resident": res_n,
            "pages_nonresident": (tot - res_n) if err is None else None,
            "all_resident": bool(tables) and err is None and tot > 0 and res_n == tot,
            "error": err}


def attach_ngram_residency(ng_records, ngram_modules):
    """Attach measured residency to each describe-style record (matched by key).
    Records without table tensors get an explicit None: absence of evidence is
    never treated as residency."""
    by_key = {}
    for m in ngram_modules:
        by_key.setdefault(str(getattr(m, "key", "?")), []).append(m)
    seen = {}
    for rec in ng_records:
        key = str(rec.get("key"))
        if not rec.get("num_table_tensors"):
            rec["residency"] = None
            continue
        i = seen.get(key, 0)
        mods = by_key.get(key) or []
        m = mods[i] if i < len(mods) else None
        if m is not None:
            seen[key] = i + 1
        tables = list(getattr(m, "tables", None) or []) if m is not None else []
        if tables:
            rec["residency"] = measure_ngram_residency(tables)
        else:
            rec["residency"] = {"probe": "mincore", "tensors": [], "pages_total": 0,
                                "pages_resident": 0, "pages_nonresident": 0,
                                "all_resident": False,
                                "error": ("live module exposes no table tensors"
                                          if m is not None else "no matching live module")}
    return ng_records


def _dev_index(value):
    """Normalize 'cuda:1' / 1 / torch.device(1)-ish values to an int index or None."""
    idx = getattr(value, "index", None)
    if isinstance(idx, int):
        return idx
    s = str(value)
    tail = s.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _summarize_plan(dev_plan, sample_cap=64, expert_cap=256):
    """Bounded summary of one device's TP plan slice map {key: (begin, end, unit)}.

    MoE ownership shows up two ways: per-expert '.experts.N' keys AND a whole
    module row whose channel_unit is 'experts' (expert parallelism picks a
    routing range per module key). Record both; the worker's actual
    num_local_experts/routing facts are the physical proof."""
    dev_plan = dev_plan or {}
    slices, width_by_unit, experts, expert_units = {}, {}, set(), {}
    for key in sorted(dev_plan, key=str):
        spec = dev_plan[key]
        try:
            beg, end, unit = int(spec[0]), int(spec[1]), str(spec[2])
        except (TypeError, ValueError, IndexError):
            continue
        if end > beg:
            slices[key] = [beg, end, unit]
            width_by_unit[unit] = width_by_unit.get(unit, 0) + (end - beg)
            me = _EXPERT_KEY_RE.search(str(key))
            if me:
                experts.add(int(me.group(1)))
            if unit == "experts":
                expert_units[str(key)] = [beg, end]
    return {"plan_entries": len(dev_plan), "nonempty_slices": len(slices),
            "width_by_unit": width_by_unit,
            "nonempty_sample": dict(list(slices.items())[:sample_cap]),
            "expert_ids_owned": sorted(experts)[:expert_cap],
            "expert_count_owned": len(experts),
            "expert_unit_owned": dict(list(expert_units.items())[:sample_cap]),
            "expert_unit_count": len(expert_units)}


def _module_facts(m, tree, linear_cap=_LINEAR_CAP):
    """What this rank ACTUALLY imported for one top-level module: stub flag,
    attention-head/expert-parallel splits, and up to linear_cap representative
    linear geometries with quant type and weight presence. Attribute-level,
    primitives only, picklable."""
    facts = {"stub": bool(getattr(m, "stub", False))}
    for a in _FACT_INTS:
        v = getattr(m, a, None)
        if isinstance(v, int) and not isinstance(v, bool):
            facts[a] = v
    lines, count = [], 0
    for sm in tree:
        in_f, out_f = getattr(sm, "in_features", None), getattr(sm, "out_features", None)
        if not (isinstance(in_f, int) and isinstance(out_f, int)):
            continue
        count += 1
        if len(lines) >= linear_cap:
            continue
        weight = next((wa for wa in _LINEAR_WEIGHT_ATTRS
                       if getattr(sm, wa, None) is not None), None)
        qtype = next((str(getattr(sm, qa)) for qa in _LINEAR_QTYPE_ATTRS
                      if getattr(sm, qa, None) is not None), None)
        lines.append({"name": str(getattr(sm, "key", type(sm).__name__)),
                      "type": type(sm).__name__,
                      "in_features": in_f, "out_features": out_f,
                      "qtype": qtype, "weight_present": weight is not None,
                      "weight_attr": weight})
    facts["linear_count"] = count
    facts["linear"] = lines
    return facts


def _cache_layer_audit_tensors(cl):
    """Named backing tensors of one KV cache layer FOR THE CACHE AUDIT.
    multi_gpu._cache_layer_tensors probes k/v/raw_k/pooled first, so a quant
    layer's packed pages would fall through to anonymous t0..t3; name qk/qv/
    sk/sv here (plus the QSA FP16 planes when present) so the report shows
    exactly what a K5/V4 layer actually allocated. Everything else keeps the
    shared resolver's behavior."""
    from rocm_tools.rdna2 import multi_gpu
    tensors = {}
    for attr in ("qk", "qv", "sk", "sv"):
        t = getattr(cl, attr, None)
        if t is not None:
            tensors[attr] = t
    if tensors:
        for attr in ("raw_k", "pooled"):
            t = getattr(cl, attr, None)
            if t is not None:
                tensors[attr] = t
        return tensors
    return multi_gpu._cache_layer_tensors(cl)


def _layer_geometry(layers, tensors_fn, cap=_LAYER_GEOM_CAP):
    """Aggregate count + byte total with up to `cap` representative per-layer
    tensor metadata (device/shape/dtype/bytes) -- attribute level only, no
    .item(), no GPU sync; picklable. Cache layers additionally carry the
    ACTUAL layer class name and k_bits/v_bits, and the aggregates
    `cls_counts` / `bits_variants` are UNCAPACITATED so the requested-vs-
    observed cache audit sees every layer, not only the sampled ones."""
    sample, total, all_known, bytes_total = [], 0, True, 0
    cls_counts, bits_variants = {}, {}
    for cl in layers:
        total += 1
        cls = type(cl).__name__
        cls_counts[cls] = cls_counts.get(cls, 0) + 1
        bits = []
        for attr in ("k_bits", "v_bits"):
            v = getattr(cl, attr, None)
            bits.append(int(v) if isinstance(v, int) and not isinstance(v, bool) else None)
        bits_variants.setdefault(cls, set()).add(tuple(bits))
        tensors = tensors_fn(cl)
        info = {}
        for name, t in sorted(tensors.items()):
            nbytes = None
            try:
                nbytes = int(t.numel()) * int(t.element_size())
            except Exception:
                pass
            if nbytes is None:
                all_known = False
            else:
                bytes_total += nbytes
            shape = None
            try:
                shape = [int(x) for x in t.shape]
            except Exception:
                pass
            info[name] = {"device": str(getattr(t, "device", None)), "shape": shape,
                          "dtype": str(getattr(t, "dtype", None)), "bytes": nbytes}
        if len(sample) < cap:
            sample.append({"device": str(getattr(cl, "device", None)),
                           "device_index": _dev_index(getattr(cl, "device", None)),
                           "cls": cls, "k_bits": bits[0], "v_bits": bits[1],
                           "n_tensors": len(tensors), "tensors": info})
    return {"layers_total": total,
            "bytes_total": bytes_total if (total and all_known) else None,
            "cls_counts": cls_counts,
            # None-vs-int tuples are not directly comparable; sort on a safe key.
            "bits_variants": {c: [list(b) for b in
                                  sorted(v, key=lambda b: [-1 if x is None else x for x in b])]
                              for c, v in sorted(bits_variants.items())},
            "sample": sample}


# ---------------------------------------------------------------------------
# Functions that run INSIDE each TP rank (top level here = picklable by
# reference; spawn'ed workers import this module, which is CPU-safe).
# ---------------------------------------------------------------------------

def tp_audit_rank(local_context):
    """Interrogate this worker's ACTUAL local_context and return a picklable record.

    Everything is read from the live objects in this process: own PID, claimed
    device and gfx arch, per-rank torch allocator stats, /proc RSS/PSS/VmSwap,
    the module shards this rank imported, this device's plan slices/expert
    ownership, cache and recurrent state geometry, and an OBSERVATIONAL n-gram
    residence snapshot (the cross-rank single-owner verdict belongs to the
    parent aggregator, not to one rank's view)."""
    import torch
    from types import SimpleNamespace
    from rocm_tools.rdna2 import multi_gpu

    device = local_context.get("device")
    rec = {"device": device, "rank": local_context.get("rank"),
           "world_size": local_context.get("world_size"),
           "active_devices": list(local_context.get("active_devices") or []),
           "output_device": local_context.get("output_device"),
           "pid": os.getpid()}
    try:
        props = torch.cuda.get_device_properties(device)
        rec["gcnArchName"] = getattr(props, "gcnArchName", "") or ""
        rec["device_name"] = props.name
    except Exception as e:
        rec["gcnArchName"] = None
        rec["device_props_error"] = repr(e)
    # On HIP builds the engine maps tp_backend='nccl' onto its RCCL backend; record
    # the ACTUAL backend class the rank joined, not just the requested string.
    backend = local_context.get("backend")
    rec["backend_class"] = type(backend).__name__ if backend is not None else None
    for key, fn in (("torch_allocated_bytes", torch.cuda.memory_allocated),
                    ("torch_peak_bytes", torch.cuda.max_memory_allocated),
                    ("torch_reserved_bytes", torch.cuda.memory_reserved)):
        try:
            rec[key] = int(fn(device))
        except Exception:
            rec[key] = None
    rec["proc_mem"] = _proc_mem()
    rec["runtime_libraries"] = _runtime_libraries()

    plan = local_context.get("plan")
    dev_plan = None
    if isinstance(plan, dict):
        dev_plan = plan.get(device)
    elif isinstance(plan, (list, tuple)):
        try:
            dev_plan = plan[device]
        except (IndexError, TypeError):
            dev_plan = None
    rec["plan"] = _summarize_plan(dev_plan)

    modules = list(local_context.get("modules") or [])
    mods = []
    for m in modules:
        tree = list(multi_gpu._iter_module_tree(m))
        cache_layers = [cl for sm in tree for cl in (getattr(sm, "cache_layers", None) or [])]
        rec_layers = [rl for sm in tree for rl in (getattr(sm, "recurrent_layers", None) or [])]
        mods.append({
            "key": getattr(m, "key", None),
            "type": type(m).__name__,
            "device": str(getattr(m, "device", None)),
            "device_index": _dev_index(getattr(m, "device", None)),
            "caps": {k: bool(v) for k, v in (getattr(m, "caps", None) or {}).items()},
            "transformer": multi_gpu._is_transformer(m),
            "kv_cache_modules": sum(1 for sm in tree if multi_gpu._caps(sm).get("kv_cache")),
            "recurrent_cache_modules": sum(1 for sm in tree if multi_gpu._caps(sm).get("recurrent_cache")),
            "cache_layers": _layer_geometry(cache_layers, _cache_layer_audit_tensors),
            "recurrent_layers": _layer_geometry(rec_layers, multi_gpu._recurrent_layer_tensors),
            "facts": _module_facts(m, tree),
        })
    rec["modules"] = mods
    # require_ram=False: per-rank facts only; the parent owns the policy verdict.
    # Table-page residency is measured HERE, inside the owning process, with
    # non-faulting mincore on the tensors' own byte ranges.
    holder = SimpleNamespace(modules=modules)
    rec["ngram"] = multi_gpu.collect_ngram_state(holder, require_ram=False)
    attach_ngram_residency(rec["ngram"]["modules"], multi_gpu.find_ngram_modules(holder))
    return rec


def tp_install_finite_hook(local_context):
    """Worker-side validation installer (picklable; only ever dispatched with
    --validate-finite). Wraps every rank module's forward to isfinite the tensor it
    ACTUALLY returned -- unobserved inner values are never asserted. Raises inside
    the rank on the first non-finite output, which tp_worker_result re-raises here."""
    import torch
    state = {"checks": 0, "wrapped": []}
    local_context["_tp_run_finite"] = state

    def make(orig, key):
        def call(*args, **kwargs):
            y = orig(*args, **kwargs)
            t = y if isinstance(y, torch.Tensor) else (
                y[0] if isinstance(y, (tuple, list)) and y and isinstance(y[0], torch.Tensor) else None)
            if t is not None:
                if not torch.isfinite(t).all().item():
                    raise RuntimeError(f"TP rank forward {key} produced non-finite output")
                state["checks"] += 1
            return y
        return call

    for m in local_context.get("modules") or []:
        orig = getattr(m, "forward", None)
        if orig is None or getattr(m, "_tp_run_finite_wrapped", False):
            continue
        m.forward = make(orig, str(getattr(m, "key", type(m).__name__)))
        m._tp_run_finite_wrapped = True
        state["wrapped"].append(str(getattr(m, "key", type(m).__name__)))
    return {"device": local_context.get("device"), "wrapped": len(state["wrapped"])}


def tp_finite_hook_stats(local_context):
    """Collect per-rank forward check counts installed by tp_install_finite_hook."""
    state = local_context.get("_tp_run_finite")
    if not isinstance(state, dict):
        return {"device": local_context.get("device"), "checks": None,
                "note": "finite hook not installed on this rank"}
    return {"device": local_context.get("device"), "checks": int(state.get("checks", 0)),
            "wrapped_modules": list(state.get("wrapped", []))}


def tp_cpu_helper_meta(local_context):
    """Optional metadata from the device -1 CPU helper slot (native-reduction
    worker); never treated as a GPU owner. Best-effort: dispatch may fail."""
    return {"device": local_context.get("device"), "pid": os.getpid(),
            "backend": type(local_context.get("backend")).__name__,
            "proc_mem": _proc_mem()}


def tp_sync_rank(local_context):
    """Run INSIDE the rank process and synchronize its own device: a parent
    torch.cuda.synchronize(dev) cannot drain a spawned rank's CUDA context, so
    TP power-phase boundaries dispatch this instead."""
    import torch
    dev = local_context.get("device")
    torch.cuda.synchronize(dev)
    return {"device": dev, "pid": os.getpid()}


class _TPPowerTorch:
    """Minimal torch-like facade handed to power_policy for TP runs only.

    power_policy's sole torch use is cuda.synchronize(dev) at phase-boundary
    transitions; for TP each call becomes an in-rank tp_sync_rank dispatch so
    the ACTUAL owning process (spawned rank or in-process pseudo rank) drains
    its queues, and every sync is recorded for the report. LS keeps plain
    torch. No per-token path calls this: the timing loop is untouched."""

    def __init__(self, model):
        self.model = model
        self.syncs = []
        self.cuda = types.SimpleNamespace(synchronize=self.synchronize)

    def synchronize(self, dev):
        idx = _dev_index(dev)
        active = [d if not hasattr(d, "index") else d.index
                  for d in (getattr(self.model, "active_devices", None) or [])]
        if idx is None or idx not in active:
            raise RuntimeError(f"cannot route power sync for device {dev!r} to a TP rank "
                               f"(active_devices {active}) -- refusing to label an unsynced "
                               "boundary as policy-controlled")
        worker = self.model.tp_worker_dispatch_single(idx, tp_sync_rank, ())
        self.syncs.append({"device": idx, "worker": worker})
        return worker


# ---------------------------------------------------------------------------
# Engram (n-gram table) ownership: exactly one physical CPU-RAM owner per
# expected key; everything else fails closed. Shared by the TP aggregator and
# the LS path, pure data in / pure verdicts out.
# ---------------------------------------------------------------------------

def validate_ngram_ownership(observed, expected_keys):
    """observed: [(device, pid, ngram_record)] across every rank (or one process).
    expected_keys: Engram keys captured from the parent model BEFORE loading.

    Per expected key require exactly one physical CPU-RAM owner; reject: key
    exposed by no rank, mode None (never loaded), RAM mode with no table
    tensors, unknown/zero byte counts, table tensors resident on CUDA despite
    a RAM mode (e.g. cuda:1 under trellis_ram), MISSING/FAILED/PARTIAL mincore
    page-residency evidence (process-wide VmSwap alone never passes), disk
    modes/handles, and duplicate owners. Tables whose key was NOT expected are
    extra owners and fail too. With an empty expected set (real D/M
    checkpoints) nothing is required and nothing may appear as an owner."""
    problems, owners, replicated = [], {}, []
    by_key = {}
    for dev, pid, ng in observed:
        by_key.setdefault(str(ng.get("key")), []).append((dev, pid, ng))
    expected = sorted({str(k) for k in (expected_keys or ())})

    def owner_entry(dev, pid, ng):
        return {"device": dev, "pid": pid, "bytes": ng.get("table_ram_bytes"),
                "tables": ng.get("num_table_tensors"), "residency": ng.get("residency")}

    for key in expected:
        recs = by_key.pop(key, [])
        if not recs:
            problems.append(f"expected n-gram table {key!r} is exposed by NO rank: silently "
                            "absent (upstream hides the missing Engram by disk-streaming); "
                            "Engram RAM fails closed")
            continue
        found = []
        for dev, pid, ng in recs:
            mode = ng.get("mode")
            tables = ng.get("num_table_tensors") or 0
            handles = ng.get("num_disk_handles") or 0
            resid = sorted({str(r) for r in (ng.get("table_residence") or [])})
            nbytes = ng.get("table_ram_bytes")
            if mode is None:
                problems.append(f"n-gram {key!r} on {dev}: mode is None (table never loaded)")
            elif str(mode).endswith("_disk") or handles:
                problems.append(f"n-gram {key!r} on {dev}: disk-offloaded (mode={mode!r}, "
                                f"handles={handles}) -- Engram RAM was demanded; upstream "
                                "silent disk streaming must not pass")
            elif mode not in NGRAM_RAM_MODES:
                problems.append(f"n-gram {key!r} on {dev}: unknown mode {mode!r}")
            elif not tables:
                problems.append(f"n-gram {key!r} on {dev}: RAM mode {mode!r} but no table "
                                "tensors are present")
            elif any(r != "cpu" for r in resid):
                problems.append(f"n-gram {key!r} on {dev}: RAM-mode table tensors are not "
                                f"host-resident: {resid}")
            elif not isinstance(nbytes, int) or nbytes <= 0:
                problems.append(f"n-gram {key!r} on {dev}: unknown/zero table byte count "
                                f"{nbytes!r}; physical RAM residence unproven")
            elif not isinstance(res := ng.get("residency"), dict) or res.get("error") is not None \
                    or not res.get("tensors"):
                problems.append(
                    f"n-gram {key!r} on {dev}: NO mincore page-residency evidence"
                    + (f" (probe failed: {res.get('error')})" if isinstance(res, dict)
                       and res.get("error") else "")
                    + "; process-wide VmSwap/RSS alone never proves the table itself is "
                      "RAM-resident, so this fails closed")
            elif not res.get("all_resident"):
                problems.append(f"n-gram {key!r} on {dev}: table pages NOT fully resident in "
                                f"RAM ({res.get('pages_resident')}/{res.get('pages_total')} "
                                "resident by mincore); Engram RAM fails closed")
            else:
                found.append(owner_entry(dev, pid, ng))
        if len(found) > 1:
            problems.append(f"n-gram table {key!r} has {len(found)} RAM owners "
                            f"{[(o['device'], o['bytes']) for o in found]} -- the physical "
                            "table is duplicated across ranks instead of one owner in RAM")
        elif len(found) == 1:
            owners[key] = found
    for key, recs in sorted(by_key.items()):
        for dev, pid, ng in recs:
            mode = ng.get("mode")
            handles = ng.get("num_disk_handles") or 0
            unplanned_owner = ((ng.get("num_table_tensors") or 0) > 0
                               and str(mode) in NGRAM_RAM_MODES and not handles)
            if unplanned_owner or str(mode).endswith("_disk") or handles:
                problems.append(f"unexpected n-gram table {key!r} on device {dev} "
                                f"(mode={mode!r}, handles={handles}): not exposed by the "
                                "pre-load checkpoint scan -- extra/duplicate owner")
            else:
                replicated.append((dev, key, mode))
    return problems, owners, replicated


def _module_has_real_work(m):
    """Evidence that a rank ACTUALLY computes with this module: a loaded linear
    (weight present), an attention-head or expert split, a routing range, or
    allocated cache/recurrent tensors. A nonempty plan row or a transformer
    flag on a stub is explicitly NOT evidence."""
    if m.get("stub") or (m.get("facts") or {}).get("stub"):
        return False
    f = m.get("facts") or {}
    if any(l.get("weight_present") for l in f.get("linear") or []):
        return True
    if f.get("num_q_heads") or f.get("num_kv_heads") or f.get("num_local_experts"):
        return True
    if (isinstance(f.get("routing_first"), int) and isinstance(f.get("routing_last"), int)
            and f["routing_last"] > f["routing_first"]):
        return True
    for key in ("cache_layers", "recurrent_layers"):
        for cl in (m.get(key) or {}).get("sample", []):
            if cl.get("n_tensors"):
                return True
    return False


# ---------------------------------------------------------------------------
# KV cache selection audit: the REQUESTED policy (K5/V4 quant by default, or
# the explicit FP16 diagnostic baseline) is checked against the cache layers
# ACTUALLY constructed, from every observation source (per-rank worker audits
# for TP, parent-side walks for LS and for the always-unsharded draft).
# Pure data in / verdicts out -- unit-testable without the native stack.
# ---------------------------------------------------------------------------

def _merge_cache_geometries(geometries, sample_cap=4):
    """Collapse per-module _layer_geometry dicts into aggregate facts, keeping
    up to sample_cap representative per-layer tensor records (device/shape/
    dtype/bytes of the actual qk/qv/sk/sv pages) so the cache section of the
    report shows WHAT was loaded, not only WHICH counts."""
    layers_total, bytes_total, cls_counts, bits_variants = 0, 0, {}, {}
    known_bytes, reps = True, []
    for g in geometries or []:
        count = int(g.get("layers_total") or 0)
        if not count:
            continue  # Embedding/recurrent-only modules have no KV allocation.
        layers_total += count
        bt = g.get("bytes_total")
        if isinstance(bt, int):
            bytes_total += bt
        else:
            known_bytes = False
        for c, n in (g.get("cls_counts") or {}).items():
            cls_counts[c] = cls_counts.get(c, 0) + int(n)
        for c, variants in (g.get("bits_variants") or {}).items():
            bits_variants.setdefault(c, set()).update(tuple(v) for v in variants)
        for s in g.get("sample") or []:
            if len(reps) < sample_cap:
                reps.append(s)
    return {"layers_total": layers_total,
            "bytes_total": bytes_total if (known_bytes and layers_total) else None,
            "cls_counts": {c: cls_counts[c] for c in sorted(cls_counts)},
            "bits_variants": {c: [list(b) for b in
                                  sorted(v, key=lambda b: [-1 if x is None else x for x in b])]
                              for c, v in sorted(bits_variants.items())},
            "representative_layers": reps}


def validate_cache_alignment(geometries, request, *, label, require_kv):
    """Compare ACTUAL cache-layer geometry records against the requested
    policy; returns (problems, notes, observed-summary). Fail closed on any
    silent fallback (FP16 classes under a quant request, unknown classes,
    wrong bit widths, quant classes without k_bits/v_bits evidence) and on a
    require_kv model (the full-attention target) that shows NO cache layers
    at all -- a quantized global-attn cache must be DEMONSTRATED, not
    assumed. A GDN/recurrent-only draft legitimately holds zero KV layers
    (its state is not KV); that is recorded as a note, never quantified."""
    agg = _merge_cache_geometries(geometries)
    observed = {"label": label, "request": request, **agg}
    problems, notes = [], []
    if not agg["layers_total"]:
        if require_kv:
            problems.append(
                f"{label}: requested {request['layer_type']} KV but NO cache layers were "
                "observed -- the quantized global-attention cache must be demonstrated, "
                "never assumed")
        else:
            notes.append(
                f"{label}: zero KV cache layers observed -- legitimate for a GDN/recurrent-only "
                "draft (recurrent state is NOT KV and stays FP32/BF16); recorded as-is, not "
                "fabricated as quantized")
        return problems, notes, observed
    if not agg["cls_counts"]:
        problems.append(f"{label}: {agg['layers_total']} cache layer(s) observed without any "
                        "class/bits metadata -- the audit cannot prove what was actually loaded")
        return problems, notes, observed
    if request["policy"] == "quant":
        want = (request["k_bits"], request["v_bits"])
        for cls, n in agg["cls_counts"].items():
            if cls not in KV_QUANT_CLASSES:
                problems.append(
                    f"{label}: requested {want[0]}-bit K / {want[1]}-bit V quantized KV but "
                    f"{n} '{cls}' layer(s) are actually loaded -- silent fallback to a "
                    "different cache type is rejected (no --cache-fp16 was given)")
                continue
            variants = agg["bits_variants"].get(cls) or []
            if not variants:
                problems.append(f"{label}: '{cls}' exposes no k_bits/v_bits evidence")
            for kb, vb in variants:
                if kb is None or vb is None:
                    problems.append(f"{label}: '{cls}' layer without k_bits/v_bits evidence "
                                    f"(got {kb!r}/{vb!r}, requested {want[0]}/{want[1]})")
                elif (kb, vb) != want:
                    problems.append(f"{label}: '{cls}' actually built with k_bits={kb} "
                                    f"v_bits={vb}, requested {want[0]}/{want[1]}")
    else:
        for cls, n in agg["cls_counts"].items():
            if cls not in KV_FP16_CLASSES:
                problems.append(f"{label}: diagnostic FP16 baseline requested but {n} "
                                f"'{cls}' layer(s) are actually loaded")
                continue
            for kb, vb in agg["bits_variants"].get(cls) or []:
                if kb is not None or vb is not None:
                    problems.append(f"{label}: diagnostic FP16 baseline requested but '{cls}' "
                                    f"reports k_bits={kb} v_bits={vb}")
    return problems, notes, observed


def run_cache_runtime_audit(request, target_geometries, draft_geometries):
    """One requested-vs-actual audit over BOTH caches' real layer records:
    target (full attention -- KV must be demonstrated) and draft (zero KV is
    legitimate for GDN-only heads). Returns (problems, notes, observed)."""
    t_probs, t_notes, t_obs = validate_cache_alignment(
        target_geometries, request, label="target full-attention KV", require_kv=True)
    d_probs, d_notes, d_obs = validate_cache_alignment(
        draft_geometries, request, label="draft", require_kv=False)
    return t_probs + d_probs, t_notes + d_notes, {"target": t_obs, "draft": d_obs}


def _cache_geometries_from_modules(module_records):
    """cache_layers geometry dicts from TP rank module records (already the
    workers' actual observations)."""
    out = []
    for r in module_records:
        for m in r.get("modules") or []:
            g = m.get("cache_layers")
            if isinstance(g, dict):
                out.append(g)
    return out


def _cache_geometries_from_model(model):
    """Parent-side equivalent of the worker capture, for the LS target and the
    (never-sharded) draft whose cache tensors live in THIS process."""
    from rocm_tools.rdna2 import multi_gpu
    out = []
    for m in (getattr(model, "modules", None) or []):
        cache_layers = [cl for sm in multi_gpu._iter_module_tree(m)
                        for cl in (getattr(sm, "cache_layers", None) or [])]
        if cache_layers:
            out.append(_layer_geometry(cache_layers, _cache_layer_audit_tensors))
    return out


# ---------------------------------------------------------------------------
# Parent-side aggregation: the ONLY place TP cross-rank policy is decided.
# ---------------------------------------------------------------------------

def aggregate_tp_audit(rank_records, expected_devices, output_device, parent_pid,
                       cpu_meta=None, expect_arch=EXPECT_ARCH, expected_ngram_keys=()):
    """Pure-data verdict over tp_audit_rank records (unit-testable with fakes).

    Fails closed when: a rank record is missing/duplicated/unexpected (records
    from devices outside the request included), a claimed gfx arch differs
    (suffix-normalized, full string preserved in the record), a module or ANY
    of its cache/recurrent tensors lives on a device other than its rank's own
    (host-side id_state excepted), a GPU rank owns no EVIDENCED computation
    (loaded weights / head-expert splits / allocated state -- a plan row or a
    transformer flag on a stub proves nothing), the in-process pseudo output
    rank is not the parent PID itself (its memory must be attributed exactly
    once), or the Engram tables do not end up with exactly one physical CPU-RAM
    owner per pre-load-expected key (missing, mode None, tensorless RAM mode,
    unknown/zero bytes, CUDA-resident tables, disk streaming, duplicated or
    unplanned owners all reject; the owner additionally needs non-faulting
    mincore evidence that ALL of the table's own pages are resident -- a
    process-wide VmSwap>0 with fully-resident table pages is a reported
    diagnostic note, not a failure). A checkpoint with no expected and no observed table (dense /
    MoE D/M) records a no-op instead of failing."""
    problems, notes = [], []
    by_device = {}
    for rec in rank_records or []:
        d = rec.get("device")
        if d in by_device:
            problems.append(f"duplicate audit records for device {d}")
        by_device[d] = rec
    for i in expected_devices:
        if i not in by_device:
            problems.append(f"no audit record from device {i} -- rank unobserved, placement unproven")
    for d in by_device:
        if d not in expected_devices:
            problems.append(f"audit record for unexpected device {d} (expected {list(expected_devices)}) "
                            "-- the run requested a different split than the ranks report")

    all_recs = [by_device[i] for i in expected_devices if i in by_device]
    arch_base = str(expect_arch).split(":", 1)[0]
    observed_ngrams = []
    for rec in all_recs:
        dev = rec["device"]
        arch = str(rec.get("gcnArchName") or "")
        if arch.split(":", 1)[0] != arch_base:
            problems.append(f"device {dev} reports gcnArchName {rec.get('gcnArchName')!r}, "
                            f"expected base {arch_base!r}")
        for m in rec.get("modules") or []:
            if m.get("device_index") is not None and m["device_index"] != dev:
                problems.append(f"device {dev}: module {m.get('key')} ({m.get('type')}) is on "
                                f"{m.get('device')}, not on this rank's device")
            for gk in ("cache_layers", "recurrent_layers"):
                for cl in (m.get(gk) or {}).get("sample", []):
                    if cl.get("device_index") is not None and cl["device_index"] != dev:
                        problems.append(f"device {dev}: state tensors of module {m.get('key')} "
                                        f"live on {cl.get('device')}")
                    for tname, tinfo in (cl.get("tensors") or {}).items():
                        tdev = str((tinfo or {}).get("device"))
                        if tname == "id_state":
                            if tdev != "cpu":
                                problems.append(f"device {dev}: {m.get('key')} id_state is on "
                                                f"{tdev}, but PLE id history must stay host-side")
                            continue
                        tidx = _dev_index(tdev)
                        if tidx is not None and tidx != dev:
                            problems.append(f"device {dev}: {m.get('key')} state tensor "
                                            f"{tname!r} lives on {tdev}, not this rank")
        if not any(_module_has_real_work(m) for m in rec.get("modules") or []):
            problems.append(f"device {dev} owns NO provable computation (need loaded weights, "
                            f"head/expert splits or allocated state; empty-plan/stub modules "
                            f"only) -- both GPUs must run non-trivial TP work")
        for ng in (rec.get("ngram") or {}).get("modules", []):
            observed_ngrams.append((dev, rec.get("pid"), ng))

    ng_problems, ng_owners, ng_replicated = validate_ngram_ownership(
        observed_ngrams, expected_ngram_keys)
    problems.extend(ng_problems)
    for key, found in ng_owners.items():
        o = found[0]
        rank = by_device.get(o["device"]) or {}
        swap = (rank.get("proc_mem") or {}).get("vm_swap_kb")
        if isinstance(swap, int) and swap > 0:
            notes.append(f"n-gram owner {key!r} (device {o['device']}, PID {o['pid']}) process "
                         f"reports VmSwap {swap} KB: process-wide diagnostic only -- the TABLE's "
                         "own pages are mincore-proven fully resident, which is the accepted "
                         "evidence (this run must not claim every process page is in RAM)")
    if not observed_ngrams and not expected_ngram_keys:
        notes.append("no n-gram table expected or found on any rank: tableless checkpoint "
                     "(dense/MoE D/M); RAM-owner audit is a recorded no-op, not a failure")

    pids = {}
    for rec in all_recs:
        pids.setdefault(rec.get("pid"), []).append(rec.get("device"))
    out_rec = by_device.get(output_device)
    if out_rec is not None and out_rec.get("pid") != parent_pid:
        problems.append(f"output device {output_device} runs in PID {out_rec.get('pid')} but the "
                        f"parent is PID {parent_pid}: the in-process pseudo rank IS the parent, so "
                        "reporting otherwise means the audit cannot attribute memory correctly")
    mem = {"unique_pids": {}, "totals": {"vm_rss_kb": 0, "pss_kb": 0, "vm_swap_kb": 0},
           "torch_by_rank": []}
    for pid, devs in sorted(pids.items(), key=lambda kv: str(kv[0])):
        rep = by_device[devs[0]]
        pm = rep.get("proc_mem") or {}
        for k in mem["totals"]:
            v = pm.get(k)
            if isinstance(v, int):
                mem["totals"][k] += v
        mem["unique_pids"][str(pid)] = {"devices": sorted(d if d is not None else -99 for d in devs),
                                        "proc_mem": pm,
                                        "note": "counted once even though it also owns the "
                                                "parent-side shells" if pid == parent_pid else None}
        if len(devs) > 1:
            notes.append(f"PID {pid} spans devices {devs}; its RSS/PSS summed once, not per rank")
        mem["torch_by_rank"].append({
            "device": rep.get("device"), "pid": rep.get("pid"),
            "allocated_bytes": rep.get("torch_allocated_bytes"),
            "peak_bytes": rep.get("torch_peak_bytes"),
            "reserved_bytes": rep.get("torch_reserved_bytes")})
    if mem["totals"]["vm_swap_kb"]:
        notes.append(f"ranks hold {mem['totals']['vm_swap_kb']} KB swapped out (VmRSS-based "
                     "residence is degraded toward swap; see per-PID records)")

    audit = {
        "ok": not problems,
        "expected_devices": list(expected_devices),
        "output_device": output_device,
        "parent_pid": parent_pid,
        "arch_expected": expect_arch,
        "problems": problems,
        "notes": notes,
        "ranks": all_recs,
        "ngram_expected_keys": list(expected_ngram_keys or ()),
        "ngram_ram_owners": ng_owners,
        "ngram_replicated_copies": ng_replicated,
        "memory": mem,
        "cpu_helper": cpu_meta,
    }
    return audit


# ---------------------------------------------------------------------------
# --capacity-only evidence: a PURE dict transform over what the load-time
# audits already captured (unit-testable without the native stack).
# ---------------------------------------------------------------------------

LOAD_ONLY_SEMANTICS = (
    "allocator bytes sampled at LOAD time by the capacity probe; no Generator and no "
    "inference ever ran, so peak_bytes is the weight+cache load high-water mark, never a "
    "post-inference peak")


def build_capacity_report(args, prompts, report):
    """Assemble the --capacity-only allocation-evidence section for TP and LS.

    The run reached here only after the frozen-prompt capacity gate, the
    target+draft cache construction, both loads and the existing
    placement / actual-cache / Engram audits passed -- those verdicts live in
    their normal report sections and are referenced here. Every allocator
    fact is labelled LOAD-only: this is allocation evidence, NOT a claim of
    usable runtime-context capacity (long-context inference, post-inference
    peaks and throughput were never measured). Group token math re-states the
    prompt-capacity requirement (already enforced fail-closed before any load;
    this never bypasses it)."""
    groups = []
    for gi in range(0, len(prompts), args.batch_size):
        chunk = prompts[gi:gi + args.batch_size]
        needed = required_cache_tokens(chunk, args.new_tokens, args.draft_tokens)
        groups.append({"group": gi // args.batch_size, "jobs": len(chunk),
                       "required_cache_tokens": needed,
                       "cache_token_capacity": args.cache_tokens,
                       "fits": needed <= args.cache_tokens})

    per_rank = []
    audit = report.get("tp_audit")
    if audit:
        # TP: each worker self-reported its OWN allocator state during the
        # post-load audit (the parent cannot see a spawned rank's context).
        for r in (audit.get("memory") or {}).get("torch_by_rank") or []:
            entry = dict(r)
            entry["scope"] = "tp_worker_self_reported"
            entry["semantics"] = LOAD_ONLY_SEMANTICS
            per_rank.append(entry)
    for dev, snap in sorted((report.get("memory_snapshot") or {}).items()):
        entry = dict(snap)
        entry["device"] = dev
        entry["scope"] = "parent_process_only"
        entry["semantics"] = LOAD_ONLY_SEMANTICS
        per_rank.append(entry)

    return {
        "capacity_only": True,
        "evidence": "load_allocation_only",
        "execution_actual": (report.get("execution") or {}).get("actual"),
        "requested": {
            "cache_tokens": args.cache_tokens,
            "batch_size": args.batch_size,
            "new_tokens": args.new_tokens,
            "draft_tokens": args.draft_tokens,
            "max_chunk_size": args.max_chunk_size,
            "load_budget_gib_per_device": list(args.use_per_device),
            "cache": requested_cache(args),
        },
        "prompt_capacity_groups": groups,
        "cache_capacity": report.get("cache_capacity"),
        "mtp_residency": report.get("mtp_residency"),
        "allocator_peak_semantics": LOAD_ONLY_SEMANTICS,
        "per_rank_load_memory": per_rank,
        "parent_allocated_after_draft_load_bytes": report.get("allocated_bytes_after_draft_load"),
        "parent_allocated_after_load_bytes": report.get("allocated_bytes_after_load"),
        "device_memory_snapshot": report.get("memory_snapshot"),
        "audits": {
            "tp_audit_ok": audit.get("ok") if audit else None,
            "placement_ok": (report.get("placement") or {}).get("ok"),
            "ngram_ok": (report.get("ngram") or {}).get("ok"),
            "cache_runtime_audit_observed": (report.get("cache") or {}).get("observed") is not None,
        },
        "not_claimed": (
            "usable runtime-context capacity: no Generator, prefill or decode ran, so "
            "nothing here proves long-context inference works at these cache sizes; read "
            "as allocation evidence only"),
    }


# ---------------------------------------------------------------------------
# Group throughput from the per-job delivery events / engine rows the run
# loop ALREADY recorded. Pure CPU math (stdlib only, no GPU synchronization,
# no new engine calls); every derived rate is a token COUNT over a measured
# interval, never an average or sum of individual tps figures.
# ---------------------------------------------------------------------------

def _num(v):
    """A genuinely measured finite number (bools are never arithmetic evidence)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _events_valid(events):
    """Nonempty [[t_s, cumulative_tokens], ...] with numeric fields and
    nondecreasing sample times -- the exact shape the run loop appends."""
    if not isinstance(events, list) or not events:
        return False
    prev = None
    for e in events:
        if not (isinstance(e, (list, tuple)) and len(e) == 2):
            return False
        if not _num(e[0]) or not _num(e[1]):
            return False
        if prev is not None and e[0] < prev:
            return False
        prev = e[0]
    return True


def _cum_at(events, boundary):
    """Step-function delivered count: the cumulative value of the LAST event
    with t <= boundary (0 if none). An MTP multi-token burst counts wholly at
    its sample instant -- it is never amortized across time."""
    val = 0
    for t, cum in events:
        if t <= boundary:
            val = cum
        else:
            break
    return val


def group_throughput_metrics(rows, *, wall_s=None, run_index_offset=0):
    """One batch group's batched-throughput metrics over the report rows of
    exactly that group (bound to ABSOLUTE indices in report["runs"] through
    run_index_offset -- hashes alone cannot address rows once repeats run).

    Common decode window: start = max(first delivery per job), end = min(last
    delivery per job) -- the interval over which EVERY job was demonstrably
    still streaming. Delivered tokens are the cumulative step-function counts
    at <= each boundary; aggregate_decode_tps = counted token delta / window
    duration. Per-job rates are never averaged or summed into the aggregate.
    For a batch of one the window collapses onto the job's own first/last
    delivery, so aggregate_decode_tps == that row's burst-aware
    delivery_rate (= observed_decode_tps): batch1 equivalence is structural.

    An invalid overlap (no rows, missing/garbage events, or end <= start,
    e.g. a job whose deliveries all land after every other job finished)
    yields null window fields plus an overlap_note instead of bogus rates;
    the engine-measured fields are independent.

    Engine semantics (job.py, inspected): time_prefill spans the job's FIRST
    PREFILL START to its first token -- it excludes queue wait, so
    prefill_makespan_s_engine = max(time_prefill) approximates the group
    prefill makespan from per-job prefill starts as ENGINE-MEASURED time, not
    a wall-clock phase boundary; time_generate spans first token to last
    token. Per-job decode medians are medians ACROSS jobs of
    (new_tokens-1)/time_generate and of burst-aware delivery_rate -- central
    tendency only, never a group aggregate. end_to_end divides summed
    new_tokens by the harness group wall (prefill + decode + overhead)."""
    jobs, invalid_rows = [], []
    for pos, row in enumerate(rows):
        ev = row.get("delivery_events")
        if not _events_valid(ev):
            invalid_rows.append(run_index_offset + pos)
            ev = []
        jobs.append({"run_index": run_index_offset + pos,
                     "ids_sha256": row.get("ids_sha256"),
                     "events": ev,
                     "new_tokens": row.get("new_tokens"),
                     "prompt_tokens": row.get("prompt_tokens"),
                     "time_prefill": row.get("time_prefill"),
                     "time_generate": row.get("time_generate")})

    w_start = w_end = w_dur = None
    overlap_note = None
    if not jobs:
        overlap_note = "no rows in group"
    elif invalid_rows:
        overlap_note = (f"rows {invalid_rows} recorded no/invalid delivery events; the common "
                        "window is undefined, rates stay null")
    else:
        start = max(j["events"][0][0] for j in jobs)
        end = min(j["events"][-1][0] for j in jobs)
        if end <= start:
            overlap_note = (f"invalid common decode overlap: max(first delivery)={start} >= "
                            f"min(last delivery)={end}; refusing to emit window rates")
        else:
            w_start, w_end, w_dur = start, end, end - start

    per_job_window = []
    for j in jobs:
        toks = rate = None
        if w_dur is not None:
            toks = _cum_at(j["events"], w_end) - _cum_at(j["events"], w_start)
            rate = toks / w_dur
        per_job_window.append({"run_index": j["run_index"], "ids_sha256": j["ids_sha256"],
                               "tokens_in_window": toks, "tps": rate})
    at_start = sum(_cum_at(j["events"], w_start) for j in jobs) if w_dur is not None else None
    at_end = sum(_cum_at(j["events"], w_end) for j in jobs) if w_dur is not None else None

    prompt_vals = [j["prompt_tokens"] for j in jobs]
    input_total = sum(prompt_vals) if prompt_vals and all(_num(v) for v in prompt_vals) else None
    out_vals = [j["new_tokens"] for j in jobs]
    total_out = sum(out_vals) if out_vals and all(_num(v) for v in out_vals) else None
    e2e_tps = total_out / wall_s if (total_out is not None and _num(wall_s) and wall_s > 0) else None

    # Engine TTFT / prefill makespan (see docstring for the exact semantics).
    prefill_vals = [j["time_prefill"] for j in jobs if _num(j["time_prefill"])]
    ttft = {"min": min(prefill_vals) if prefill_vals else None,
            "max": max(prefill_vals) if prefill_vals else None,
            "jobs": len(prefill_vals)}
    makespan = max(prefill_vals) if prefill_vals and len(prefill_vals) == len(jobs) else None
    firsts = [j["events"][0][0] for j in jobs if j["events"]]

    per_job_rates, eng_rates, obs_rates = [], [], []
    for j in jobs:
        nt, tg = j["new_tokens"], j["time_generate"]
        eng = (nt - 1) / tg if _num(nt) and _num(tg) and tg > 0 and nt >= 1 else None
        obs = delivery_rate(j["events"]) if j["events"] and len(j["events"]) >= 2 else None
        if eng is not None:
            eng_rates.append(eng)
        if obs is not None:
            obs_rates.append(obs)
        per_job_rates.append({"run_index": j["run_index"], "ids_sha256": j["ids_sha256"],
                              "engine_tps": eng, "observed_tps": obs})
    median_note = None
    if jobs and (len(eng_rates) < len(jobs) or len(obs_rates) < len(jobs)):
        median_note = (f"engine median over {len(eng_rates)}/{len(jobs)} jobs, observed median "
                       f"over {len(obs_rates)}/{len(jobs)} jobs; jobs without usable timing are "
                       "excluded, never imputed")

    return {
        "jobs": len(jobs),
        "run_indices": [j["run_index"] for j in jobs],
        "overlap_valid": w_dur is not None,
        "overlap_note": overlap_note,
        "common_window": {
            "start_s": w_start, "end_s": w_end, "duration_s": w_dur,
            "delivered_at_start": at_start, "delivered_at_end": at_end,
            "token_delta": (at_end - at_start) if at_start is not None else None,
            "aggregate_decode_tps": ((at_end - at_start) / w_dur) if w_dur is not None else None,
        },
        "per_job_window_tps": per_job_window,
        "input_tokens_total": input_total,
        "prefill_makespan_s_engine": makespan,
        "ttft_engine_s": ttft,
        "first_delivery_s": {"min": min(firsts) if firsts else None,
                             "max": max(firsts) if firsts else None},
        "end_to_end": {"total_new_tokens": total_out, "wall_s": wall_s, "aggregate_tps": e2e_tps},
        "per_job_decode_tps": per_job_rates,
        "decode_median_tps": {"engine": statistics.median(eng_rates) if eng_rates else None,
                              "observed": statistics.median(obs_rates) if obs_rates else None,
                              "engine_jobs": len(eng_rates), "observed_jobs": len(obs_rates),
                              "note": median_note},
    }


# ---------------------------------------------------------------------------
# GPU-side run
# ---------------------------------------------------------------------------

class _CapacityProbeComplete(Exception):
    """--capacity-only control flow, raised after the load-time audits instead
    of constructing a Generator. NOT an error: the dedicated handler leaves the
    report marked complete, and the shared finally block still drains workers
    and unloads models -- a cleanup failure flips complete and forces exit 1."""


def run(args, prompts):
    """TP-vs-LS evaluation; returns the exit code (0 only when fully complete).

    Never calls os._exit; the report JSON is written even on failure, workers
    are drained, models unload via finally, and the power context restores on
    every exit path. With --capacity-only the run stops after the load-time
    placement/actual-cache/Engram audits: no Generator, no inference, no power
    phase, no runs/groups -- only allocation evidence (see
    build_capacity_report)."""
    import torch
    import exllamav3_ext
    from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job
    # CacheLayer_quant/CacheLayer_fp16 are exported by exllamav3.cache (verified in
    # its __init__); the engine itself maps QSA attention to CacheLayer_qsa_quant.
    from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
    from exllamav3.generator.sampler import ArgmaxSampler
    from rocm_tools.rdna2 import multi_gpu, power_policy
    from rocm_tools.rdna2.common import model_fingerprint, git_commit

    native_path = Path(exllamav3_ext.__file__).resolve()
    report = {"complete": False, "validation_only": bool(args.validate_finite or args.capacity_only),
              "capacity_only": bool(args.capacity_only),
              "cli": {k: v for k, v in vars(args).items()},
              "cache": {"requested": requested_cache(args), "observed": None, "notes": []},
              "execution": {"requested": args.execution,
                            "tp_backend": TP_BACKEND if args.execution == "tp" else None,
                            "tp_output_device": TP_OUTPUT_DEVICE if args.execution == "tp" else None},
              "native": {"path": str(native_path),
                         "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest()},
              "model_fingerprint": model_fingerprint(args.model),
              "repo_git_commit": git_commit(),
              "runtime_env": {k: v for k, v in os.environ.items()
                              if k.startswith(("EXL3_", "NCCL_", "TORCH_NCCL_")) or k in ("PYTHONPATH", "HSA_ENABLE_SDMA",
                                  "LD_PRELOAD", "LD_LIBRARY_PATH", "AMD_SERIALIZE_KERNEL",
                                  "HSA_DISABLE_COREDUMP_ON_EXCEPTION",
                                  "HSA_ENABLE_PEER_SDMA", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
              "runtime_libraries": _runtime_libraries(),
              "power_policy_socket": str(args.power_socket), **_raise_nofile(),
              "prompts": [{**{k: v for k, v in p.items() if k != "ids"},
                           "ids_sha256": p["sha"], "n_tokens": len(p["ids"])} for p in prompts],
              "runs": [], "groups": []}
    model = draft = policy = None
    power_torch = None
    cleanup_errors = []
    try:
        if torch.cuda.device_count() != 2 or any(
                torch.cuda.get_device_properties(d).gcnArchName.split(":")[0] != EXPECT_ARCH
                for d in (0, 1)):
            raise RuntimeError(f"requires exactly 2 {EXPECT_ARCH} GPUs (the V620 pair)")
        cfg = Config.from_directory(args.model)
        cfg.infer_params.ngram_stream_from_disk = False
        has_mtp = "mtp" in (getattr(cfg, "model_classes", None) or {})
        report["mtp_residency"] = {"component_present": has_mtp, "requested_mode": args.mode,
                                   "draft_device": DRAFT_DEVICE}
        if args.mode == "mtp" and not has_mtp:
            raise ValueError(f"--mode mtp requires an 'mtp' component in {args.model}; this "
                             "checkpoint (dense Qwen3-8B / MoE Qwen3-30B-A3B) has none -- use "
                             "--mode ar (no draft is constructed for it)")
        tok = Tokenizer.from_config(cfg)
        if any(v >= tok.actual_vocab_size for p in prompts for v in p["ids"]):
            raise ValueError("prompt contains a token ID outside the tokenizer vocabulary")
        for gi in range(0, len(prompts), args.batch_size):
            needed = required_cache_tokens(prompts[gi:gi + args.batch_size],
                                           args.new_tokens, args.draft_tokens)
            if needed > args.cache_tokens:
                raise ValueError(f"group {gi // args.batch_size} requires {needed} cache tokens, "
                                 f"capacity is {args.cache_tokens}")
        model = Model.from_config(cfg)
        if args.execution == "tp" and not (getattr(model, "caps", None) or {}).get("supports_tp"):
            raise ValueError(f"{cfg.architecture} does not declare supports_tp yet: TP2 depends on "
                             "the in-progress backend/architecture task; run --execution ls for the "
                             "accepted layer-split comparison meanwhile")
        # Capture expected Engram keys from the CONSTRUCTED parent module tree BEFORE
        # any load turns it into TP shells: this is what a Qwen3.8 checkpoint must end
        # up exposing exactly one physical CPU-RAM owner for. Dense/MoE D/M yield an
        # empty set here and stay a recorded no-op.
        expected_ngram_keys = sorted({str(getattr(m, "key", None) or type(m).__name__)
                                      for m in multi_gpu.find_ngram_modules(model)})
        report["ngram_expected_keys"] = expected_ngram_keys
        draft = Model.from_config(cfg, component="mtp") if has_mtp else None
        report["mtp_residency"]["resident"] = draft is not None
        # SAME cache kwargs for target and draft: K5/V4 quant by default (the
        # user-selected policy), FP16 only when explicitly demanded as a
        # diagnostic baseline. Bits never reach CacheLayer_fp16, and vice versa.
        cache_kwargs = ({"layer_type": CacheLayer_fp16} if args.cache_fp16 else
                        {"layer_type": CacheLayer_quant, "k_bits": args.cache_k_bits,
                         "v_bits": args.cache_v_bits})
        cache = Cache(model, max_num_tokens=args.cache_tokens, max_batch_size=args.batch_size,
                      max_history=args.draft_tokens, **cache_kwargs)
        dcache = (Cache(draft, max_num_tokens=args.cache_tokens, max_batch_size=args.batch_size,
                        **cache_kwargs) if draft is not None else None)
        expected_cls = CacheLayer_fp16 if args.cache_fp16 else CacheLayer_quant
        for cname, c in (("target", cache), ("draft", dcache)):
            if c is None:
                continue
            got = getattr(c, "layer_type", None)
            if got is not expected_cls:
                raise RuntimeError(
                    f"{cname} Cache constructed layer_type "
                    f"{getattr(got, '__name__', got)!r} although {expected_cls.__name__} "
                    "was requested -- no silent fallback")
        if draft is not None:
            draft.load(device=DRAFT_DEVICE, max_chunk_size=args.max_chunk_size, progressbar=False)
        report["allocated_bytes_after_draft_load"] = {f"cuda:{d}": torch.cuda.memory_allocated(d)
                                                      for d in (0, 1)}
        if args.execution == "tp":
            model.load(use_per_device=args.use_per_device, max_chunk_size=args.max_chunk_size,
                       progressbar=False, tensor_p=True, tp_backend=TP_BACKEND,
                       tp_output_device=TP_OUTPUT_DEVICE)
        else:
            model.load(use_per_device=args.use_per_device, max_chunk_size=args.max_chunk_size,
                       progressbar=False)
        report["allocated_bytes_after_load"] = {f"cuda:{d}": torch.cuda.memory_allocated(d)
                                                for d in (0, 1)}
        report["execution"]["loaded_tp"] = bool(getattr(model, "loaded_tp", False))
        report["execution"]["actual"] = "tp2" if report["execution"]["loaded_tp"] else "layer_split"
        report["execution"]["actual_backend"] = getattr(model, "tp_backend", None)
        out_idx = _dev_index(getattr(model, "output_device", None))
        report["execution"]["output_device"] = out_idx

        if args.execution == "tp":
            if not report["execution"]["loaded_tp"]:
                raise RuntimeError("TP load finished without setting loaded_tp -- the run would "
                                   "have measured the placement it did not ask for")
            if out_idx != _dev_index(TP_OUTPUT_DEVICE):
                raise RuntimeError(f"model.output_device is {out_idx}, expected cuda:1: TP logits "
                                   "must land next to the unsharded MTP draft")
            expected = [i if not hasattr(i, "index") else i.index
                        for i in (getattr(model, "active_devices", None) or [])]
            if sorted(expected) != [0, 1]:
                raise RuntimeError(f"--execution tp requires tensor parallelism across BOTH "
                                   f"V620s: model.active_devices is {sorted(set(expected))}, "
                                   "expected exactly [0, 1] -- a single-rank or duplicated "
                                   "layout must never be labelled TP2")
            finite_installs = None
            if args.validate_finite and not args.capacity_only:
                # --capacity-only never runs a forward, so a hook could only ever
                # report zero checks: skipped, and the artifact says so via
                # capacity_only=true, rather than installing an inert wrapper.
                finite_installs = model.tp_worker_dispatch_wait_multi(
                    expected, tp_install_finite_hook, ())
            rank_records = model.tp_worker_dispatch_wait_multi(expected, tp_audit_rank, ())
            cpu_meta = None
            try:
                cpu_meta = model.tp_worker_dispatch_wait_multi([-1], tp_cpu_helper_meta, ())[0]
            except Exception as e:
                cpu_meta = {"absent": True, "error": repr(e)}
            audit = aggregate_tp_audit(rank_records, expected, _dev_index(TP_OUTPUT_DEVICE),
                                       os.getpid(), cpu_meta=cpu_meta,
                                       expected_ngram_keys=expected_ngram_keys)
            if report["execution"]["actual_backend"] not in (None, TP_BACKEND):
                audit["problems"].append(f"tp_backend is {report['execution']['actual_backend']!r}, "
                                         f"requested {TP_BACKEND!r}")
                audit["ok"] = False
            # parent-side shell scan: unloaded shells must not retain tables, and
            # the pseudo rank's in-process tables ARE counted through its own record.
            shell_recs = [multi_gpu.describe_ngram_module(m)
                          for m in multi_gpu.find_ngram_modules(model)]
            stray = [s for s in shell_recs if s["num_table_tensors"] or s["num_disk_handles"]]
            if stray:
                audit["problems"].append(f"parent TP shells still hold n-gram table(s)/handles: "
                                         f"{[(s['key'], s['mode'], s['num_table_tensors'], s['num_disk_handles']) for s in stray]}")
                audit["ok"] = False
            audit["finite_hook_installs"] = finite_installs
            audit["plan_from_parent"] = _summarize_plan_all(model)
            # ACTUAL loaded KV cache vs the requested K5/V4 (or diagnostic FP16)
            # policy: rank records carry each worker's real cache-layer class,
            # bits and qk/qv/sk/sv geometry; the parent-side walk covers the
            # unsharded draft on cuda:1.
            cache_probs, cache_notes, cache_obs = run_cache_runtime_audit(
                report["cache"]["requested"],
                _cache_geometries_from_modules(rank_records),
                _cache_geometries_from_model(draft) if draft is not None else [])
            report["cache"]["observed"] = cache_obs
            report["cache"]["notes"] = cache_notes
            if cache_probs:
                audit["problems"].extend(cache_probs)
                audit["ok"] = False
            report["tp_audit"] = audit
            if not audit["ok"]:
                raise RuntimeError(f"TP worker audit failed: {audit['problems']}")
        else:
            report["placement"] = multi_gpu.audit_placement(model, [0, 1])
            if not report["placement"]["ok"]:
                raise RuntimeError(f"placement audit failed: {report['placement'].get('problems')}")
            ng = multi_gpu.collect_ngram_state(model, require_ram=True)
            # BEFORE inference: mincore-prove every expected table page is RAM-
            # resident in THIS process (LS owner == parent). Never faults pages.
            attach_ngram_residency(ng.get("modules") or [],
                                   multi_gpu.find_ngram_modules(model))
            ng_problems, ng_owners, ng_replicated = validate_ngram_ownership(
                [(None, os.getpid(), rec) for rec in ng.get("modules") or []],
                expected_ngram_keys)
            ng_notes = list(ng.get("notes") or [])
            owner_swap = (_proc_mem().get("vm_swap_kb") or 0) if ng_owners else 0
            if owner_swap > 0:
                ng_notes.append(f"Engram owner process reports VmSwap {owner_swap} KB: "
                                "process-wide diagnostic only; the table's own pages are "
                                "mincore-proven fully resident")
            ng = {**ng, "expected_keys": expected_ngram_keys, "ram_owners": ng_owners,
                  "replicated_copies": ng_replicated, "notes": ng_notes,
                  "problems": list(ng.get("problems") or []) + ng_problems}
            ng["ok"] = not ng["problems"]
            report["ngram"] = ng
            if not ng["ok"]:
                raise RuntimeError(f"ngram audit failed: {ng['problems']}")
            # LS: the target's and the draft's cache tensors all live in THIS
            # process, so the requested-vs-actual KV audit reads their real
            # layer objects (same geometry capture as the TP workers).
            cache_probs, cache_notes, cache_obs = run_cache_runtime_audit(
                report["cache"]["requested"],
                _cache_geometries_from_model(model),
                _cache_geometries_from_model(draft) if draft is not None else [])
            report["cache"]["observed"] = cache_obs
            report["cache"]["notes"] = cache_notes
            if cache_probs:
                raise RuntimeError(f"cache runtime audit failed: {cache_probs}")
        report["memory_snapshot"] = multi_gpu.device_memory_snapshot(torch, [0, 1])

        if args.capacity_only:
            # Allocation evidence ONLY: the frozen-prompt capacity gate, both
            # cache constructions, the loads and the full placement/actual-
            # cache/Engram audit above all ran exactly as in a measured run --
            # then STOP. No Generator is constructed, no prefill/decode runs,
            # no power phase engages, and no runs/groups exist to carry speed
            # fields. The allocator facts below are LOAD-time samples, never
            # post-inference peaks. finally cleanup still runs; a cleanup
            # failure flips complete and the exit code.
            cap = {"main": {"num_slots": cache.num_slots, "max_num_tokens": cache.max_num_tokens,
                            "max_history": cache.max_history}}
            if dcache is not None:
                cap["draft"] = {"num_slots": dcache.num_slots, "max_num_tokens": dcache.max_num_tokens}
            report["cache_capacity"] = cap
            report["capacity_report"] = build_capacity_report(args, prompts, report)
            report["complete"] = True
            raise _CapacityProbeComplete()

        kwargs = dict(model=model, cache=cache, tokenizer=tok, max_batch_size=args.batch_size,
                      max_chunk_size=args.max_chunk_size, ngram_match_min=0,
                      record_draft_stats=True)
        if args.mode == "mtp":
            kwargs.update(draft_model=draft, draft_cache=dcache, num_draft_tokens=args.draft_tokens,
                          dynamic_draft_tokens=args.dynamic_draft, draft_confidence=args.draft_confidence)
        else:
            kwargs["num_draft_tokens"] = 0   # AR (and every no-MTP model): no active draft config
        gen = Generator(**kwargs)
        report["generator"] = {"mode": args.mode, "mtp_draft": gen.mtp_draft,
                               "num_draft_tokens": gen.num_draft_tokens, "dynamic_draft": gen.dynamic_draft,
                               "ngram_match_min": gen.ngram_match_min, "record_draft_stats": gen.record_draft_stats}
        cap = {"main": {"num_slots": cache.num_slots, "max_num_tokens": cache.max_num_tokens,
                        "max_history": cache.max_history}}
        if dcache is not None:
            cap["draft"] = {"num_slots": dcache.num_slots, "max_num_tokens": dcache.max_num_tokens}
        report["cache_capacity"] = cap
        assert bool(gen.mtp_draft) == (args.mode == "mtp")
        assert gen.num_draft_tokens == (args.draft_tokens if args.mode == "mtp" else 0)
        assert gen.ngram_match_min == 0 and gen.record_draft_stats
        assert gen.dynamic_draft == (args.dynamic_draft and args.mode == "mtp")
        if args.mode == "mtp":
            assert gen.draft_model is draft and gen.draft_cache is dcache  # passed draft attached
        if has_mtp and args.mode == "ar":
            report["mtp_residency"]["ar_parity"] = ("native mtp head kept resident on cuda:1 "
                                                    "exactly like qwen_mtp_run AR controls")
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
            if draft is not None:
                wrap(draft, "draft")
        power_torch = _TPPowerTorch(model) if args.execution == "tp" else torch
        with power_policy.attach(gen, power_torch, args.batch_size, [0, 1],
                                 args.power_socket) as policy:
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
                    t = time.perf_counter() - t0   # sample AFTER progress; NEVER a per-token sync
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
                row_start = len(report["runs"])       # absolute index of this group's first row
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
                    row = {"execution": args.execution, "mode": args.mode,
                           "language": p["language"], "repeat": p["repeat"],
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
                    print(f"{args.execution} {args.mode} {p['language']} rep={p['repeat']} "
                          f"timed={p['timed']} tokens={row['new_tokens']} "
                          f"engine_tps={row['legacy_engine_decode_tps']:.3f} "
                          f"observed_tps={row['observed_decode_tps']}", flush=True)
                report["groups"].append({"group": gi // args.batch_size, "timed": prompts[gi]["timed"],
                                         "jobs": len(states), "wall_s": wall,
                                         "wall_start_unix_s": unix0,
                                         "total_new_tokens": len(states) * args.new_tokens,
                                         "ids_sha256": row_refs,
                                         # ABSOLUTE report["runs"] indices bind every metric below
                                         # to actual rows, not only to prompt hashes.
                                         "run_indices": list(range(row_start, len(report["runs"]))),
                                         "throughput": group_throughput_metrics(
                                             report["runs"][row_start:], wall_s=wall,
                                             run_index_offset=row_start)})
        if args.execution == "tp":
            # AFTER inference, BEFORE unload: the only real per-rank peaks/PSS
            # (parent allocator stats cannot see a spawned rank's context) and
            # the ACTUAL finite-hook counters (hooks saw the forwards they ran).
            final_ranks = model.tp_worker_dispatch_wait_multi(expected, tp_audit_rank, ())
            final_stats = None
            if args.validate_finite:
                final_stats = model.tp_worker_dispatch_wait_multi(
                    expected, tp_finite_hook_stats, ())
                checks = [s.get("checks") for s in (final_stats or [])]
                if not checks or any(not isinstance(c, int) or c <= 0 for c in checks):
                    raise RuntimeError(f"--validate-finite: per-rank finite-check counts {checks} "
                                       "must be positive on every rank (hook saw no forwards?)")
            report["tp_final_audit"] = {
                "ranks": [{k: r.get(k) for k in ("device", "pid", "gcnArchName",
                                                 "torch_allocated_bytes", "torch_peak_bytes",
                                                 "torch_reserved_bytes", "proc_mem", "runtime_libraries")}
                          for r in final_ranks],
                "finite_hook_stats": final_stats}
            final_audit = aggregate_tp_audit(final_ranks, expected,
                                             _dev_index(TP_OUTPUT_DEVICE), os.getpid(),
                                             expected_ngram_keys=expected_ngram_keys)
            fc_probs, fc_notes, fc_obs = run_cache_runtime_audit(
                report["cache"]["requested"],
                _cache_geometries_from_modules(final_ranks),
                _cache_geometries_from_model(draft) if draft is not None else [])
            report["cache"]["observed_post_inference"] = fc_obs
            report["cache"]["notes_post_inference"] = fc_notes
            if fc_probs:
                final_audit["problems"].extend(fc_probs)
                final_audit["ok"] = False
            report["tp_final_audit"].update({k: final_audit[k] for k in (
                "ok", "problems", "notes", "ngram_ram_owners")})
            if not final_audit["ok"]:
                raise RuntimeError(f"post-inference TP audit failed: {final_audit['problems']}")
        else:
            # LS: re-probe the owner's table pages AFTER inference too (the pre-
            # load pass already ran). Same non-faulting mincore query.
            post = []
            for m in multi_gpu.find_ngram_modules(model):
                if getattr(m, "tables", None):
                    post.append({"key": str(getattr(m, "key", "?")),
                                 **measure_ngram_residency(m.tables)})
            report["ngram"]["residency_post_inference"] = post
            lost = [p for p in post
                    if not p["all_resident"] or not p["pages_total"]]
            if lost:
                raise RuntimeError(
                    "post-inference Engram residency lost: "
                    + json.dumps([{"key": p["key"], "error": p["error"],
                                   "pages_resident": p["pages_resident"],
                                   "pages_total": p["pages_total"]} for p in lost]))
        report["power_policy"] = policy.summary()
        report["finite_forward_counts"] = forward_checks if args.validate_finite else None
        report["complete"] = True
    except _CapacityProbeComplete:
        # Control flow, not a failure: capacity_only/validation_only/complete
        # and capacity_report are already set; the normal report keys stay at
        # their init values (runs/groups empty -- no speed fields were ever
        # produced). finally still owns cleanup, and cleanup_errors there
        # override complete so the return code reflects a failed teardown.
        pass
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILED: {report['error']}", file=sys.stderr)
    finally:
        if policy is not None:
            report["power_policy"] = policy.summary()
        if isinstance(power_torch, _TPPowerTorch):
            report["power_rank_syncs"] = power_torch.syncs
        _drain_tp_workers(model, cleanup_errors)
        for name, obj in (("draft", draft), ("model", model)):
            if obj is not None:
                try:
                    obj.unload()
                except Exception as e:
                    cleanup_errors.append(f"unload {name}: {e!r}")
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["complete"] = False
        try:
            parent_peaks = {f"cuda:{d}": torch.cuda.max_memory_allocated(d)
                            for d in range(min(2, torch.cuda.device_count()))}
            if report["execution"].get("loaded_tp"):
                rank_peaks = ({f"cuda:{r.get('device')}": r.get("torch_peak_bytes")
                               for r in report.get("tp_final_audit", {}).get("ranks", [])}
                              if report.get("tp_final_audit") else None)
                report["peak_allocated_bytes"] = {
                    "scope": "parent_process_only",
                    "note": "a spawned rank's CUDA context is invisible to this process's "
                            "allocator stats; real per-rank peaks are in tp_final_audit.ranks "
                            "(post-inference worker query), not in parent_devices",
                    "parent_devices": parent_peaks,
                    "tp_rank_peak_bytes": rank_peaks}
            else:
                report["peak_allocated_bytes"] = parent_peaks
        except Exception as e:
            report["complete"] = False
            cleanup_errors.append(f"memory reporting: {e!r}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["complete"] else 1


def _summarize_plan_all(model):
    """Bounded parent-side view of model.plan (the exact split the workers were
    told to import) for the artifact; workers remain the source of truth."""
    plan = getattr(model, "plan", None)
    if isinstance(plan, dict):
        return {str(d): _summarize_plan(v) for d, v in sorted(plan.items(), key=lambda kv: str(kv[0]))}
    if isinstance(plan, (list, tuple)):
        return {str(d): _summarize_plan(v) for d, v in enumerate(plan) if isinstance(v, dict)}
    return None


def _drain_tp_workers(model, cleanup_errors):
    """Spawned ranks and the pseudo output conn must be drained even when TP init
    failed BEFORE Model.load set loaded_tp (Model.unload()/unload_tp() is a no-op
    in that state). Joins/terminates only the workers THIS run spawned, with
    bounded waits; nothing unrelated is touched and no os._exit shortcut is
    taken."""
    if model is None:
        return
    children = getattr(model, "mp_children", None)
    if not children or getattr(model, "loaded_tp", False):
        return                      # a fully loaded model is torn down by unload()
    try:
        model.destroy_tp_context()
        return
    except Exception as e:
        cleanup_errors.append(f"tp worker drain: destroy_tp_context: {e!r}")
    # destroy_tp_context itself failed: best-effort manual teardown so the leak is
    # at least bounded, and record it -- never silently claim a clean exit.
    out_dev = getattr(model, "tp_output_device", None)
    self_pid = os.getpid()
    for idx, child in enumerate(children):
        if child is None:
            continue
        # The pseudo output rank is THIS process; terminating it would kill the
        # harness (and the PID check catches a mislabelled pseudo child too).
        if idx == out_dev or type(child).__name__ == "PseudoChild" \
                or getattr(child, "pid", None) == self_pid:
            continue
        try:
            child.join(timeout=2)
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)
        except Exception as e:
            cleanup_errors.append(f"tp child teardown (slot {idx}): {e!r}")
    for idx, conn in enumerate(getattr(model, "mp_parent_conn", None) or []):
        if conn is None or idx == out_dev or type(conn).__name__ == "PseudoParentConn":
            continue
        try:
            conn.close()
        except Exception as e:
            cleanup_errors.append(f"tp conn close (slot {idx}): {e!r}")


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
