#!/usr/bin/env python3
"""CPU-only summarizer for the bounded stage traces produced by rocprofv3.

Input: a rocprofv3 output tree (--trace-dir, any nesting: rocprofv3 writes
its CSVs under <uuid> subdirectories as <pid>_kernel_trace.csv etc.), plus
optionally the profile_stages JSON artifact. Output: one JSON summary of GPU
time per measured stage window. Nothing here touches a GPU, ROCm or torch --
it is pure CSV data processing and the CPU tests cover the logic.

What it does

  1. Discovers trace files. Recursively finds ``*_kernel_trace.csv``,
     ``*_marker_api_trace.csv`` and ``*_memory_copy_trace.csv`` and groups
     them by process, keyed on (parent directory, PID from the filename
     prefix ``<pid>_``). Ranges seen from different processes are never
     mixed into one timeline or one median.
  2. Reads the selected stage ranges from the marker API trace: rows whose
     Domain is MARKER_CORE_RANGE_API whose Function carries
     ``stage=<prefill|decode>;context=N;repeat=R;tokens=K``.
     MARKER_CONTROL_API rows (roctxProfilerPause/Resume) are profiler
     control: counted and ignored, never stages.
  3. Classifies every kernel into exactly one PRIMARY category --
     dequant_related, attention_related, other_gpu -- with informative
     subcategories. Ordered, first-match rules informed by the real traces
     (mangled ``_ZL25exl3_gemv_mr_had_in_multi...`` symbols with a
     length-digit BEFORE "exl3" (a word-boundary regex would fail),
     demangled ``void rms_norm_kernel<...>`` names, triton python names,
     rocBLAS/Tensile ``Cijk_...`` tiles, and ``__amd_rocclr_copyBuffer``
     which is an actual GPU kernel -- not a memory-copy-trace event).
  4. Computes per window: marker wall span; kernel time per category and
     subcategory; the UNION of kernel intervals (never double-counting
     overlapping kernels); memory-copy sum and union (an absent copy file is
     reported as NOT MEASURED, never as zero); GPU busy = union of kernels
     and copies; kernel overlap = sum - union; the non-GPU remainder of the
     window; and a PER-AGENT breakdown keyed by the rocprof Agent_Id recorded
     in the CSV (an agent id is NOT a HIP device index -- root maps it via
     agent_info trace metadata): per-agent kernel count/sum/union and
     category sums, plus per-agent copy totals bucketed by Source_Agent_Id
     (falling back to Agent_Id) when the copy CSV supplies them. Per-agent
     unions are computed WITHIN one agent; the window-level kernel_union_ns
     is the cross-agent union -- never conflate or sum the two. Copy rows
     keep Direction/src/dst/bytes RAW as the CSV supplies them (Direction
     labels have been observed wrong; no inference, and bytes is None when
     the CSV has no size column -- never fabricated). Percentages are
     relative to the window's total kernel time -- never silently to wall.
     Decode windows normalize per generated token (marker ``tokens``);
     prefill reports the whole-prompt total plus an optional per-input-token
     figure derived from ``context``.
  5. Aggregates across repeats per (stage, context): median category
     duration per window and per decode token, plus min/max/distribution.
     Raw windows are always preserved. Global medians mix processes only
     when the whole trace-dir holds a single process; otherwise per-process
     blocks are authoritative and the pooled group says so.
  6. With --run-json, cross-checks the harness artifact against the trace:
     recorded wall time vs marker wall, marker tokens vs recorded tokens,
     repeats and 64-token decode counts, cached_tokens/new_tokens/eos_reason,
     and profiled jobs missing from the trace (or markers missing from the
     run artifact).

Refusals -- any of these yields ok:false, listed errors and a nonzero exit;
we never summarize a broken capture as success: missing kernel or marker
trace file; no selected ranges; empty (zero-length) marker ranges;
non-positive or end-before-start timestamps; duplicate files or duplicate
rows; the same (stage,context,repeat) seen twice (in one process or across
processes); two stage windows overlapping within a process; any kernel or
copy CROSSTING a stage boundary (never silently clipped -- the harness
synchronizes at every boundary, so a crossing means the capture or the
harness is broken); a selected window containing zero GPU kernels; and a
trace with no GPU events at all (fake zero-GPU success). Kernels/copies
outside every selected range are counted separately and are never part of
stage totals.

Labels stay honest: the non-GPU remainder is reported as exactly that. When
the memory-copy trace is present it covers CPU submission, queue waits and
profiler overhead; when it is ABSENT (profiling runs with --kernel-trace
--marker-trace only), GPU busy covers observed kernels only and the
remainder may additionally include untraced SDMA/memory-copy activity. It is
never labeled pure CPU compute and never labeled GPU idle. Fused quantized
GEMV/GEMM/MoE is one category and is never called pure dequant or split into
invented shares. Kernel (device) time and marker (host API) time are
separate fields, never summed.

Run (host, CPU only):
    python3 rocm_tools/rdna2/summarize_rocprof.py \
        --trace-dir /work/profile/marker-trace --output profile_summary.json \
        [--run-json /work/profile/profile_stages.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable

PROFILE_FORMAT = "rdna2-profile-stages/1"
SUMMARY_FORMAT = "rdna2-profile-summary/1"

KERNEL_SUFFIX = "kernel_trace.csv"
MARKER_SUFFIX = "marker_api_trace.csv"
COPY_SUFFIX = "memory_copy_trace.csv"

CAT_DEQUANT = "dequant_related"
CAT_ATTENTION = "attention_related"
CAT_OTHER = "other_gpu"
CATEGORIES = (CAT_DEQUANT, CAT_ATTENTION, CAT_OTHER)

MARKER_RANGE_DOMAIN = "MARKER_CORE_RANGE_API"
MARKER_CONTROL_DOMAIN = "MARKER_CONTROL_API"
STAGES = ("prefill", "decode")
LABEL_RE = re.compile(r"stage=(?P<stage>[a-z]+);context=(?P<context>\d+);"
                      r"repeat=(?P<repeat>\d+);tokens=(?P<tokens>\d+)")

# ---------------------------------------------------------------------------
# kernel-name classification
#
# Ordered rules, FIRST MATCH WINS, so the categories are mutually exclusive:
#
#   * MoE data movement (exl3_moe_gather/scatter/reduce, moe_split_*,
#     moe_bias_add, moe_flag) is checked BEFORE the fused-MoE match and is
#     other_gpu/moe_support -- only exl3_moe's actual compute is fused work.
#   * routing_* (incl. routing_gemv) is other_gpu/routing BEFORE any gemv
#     family pattern; it is expert selection, not quantized projection.
#   * reconstruct* is standalone weight reconstruction/dequant.
#   * Hadamard transforms (had_* family AND the had_in/had_out rotation
#     kernels embedded in the GEMV paths, e.g. exl3_gemv_mr_had_in_multi)
#     are dequant/hadamard and MUST match BEFORE the fused-GEMV family so
#     transform time is not counted as fused compute.
#   * exl3_(m)gemv/exl3_(m)gemm are the fused quantized GEMV/GEMM kernels --
#     this is where the quantized QKV/out projections live (per spec they
#     belong to the dequant group, as fused work, not as "pure dequant").
#   * Attention: triton _paged_attn_* split/combine/prefill kernels, the
#     CUDA attn_chunked*/attn_reduce_* kernels, _paged_kv_update* /
#     paged_kv_update_vec8 / kv_cache_update kernels (KV writes are part of
#     attention here), rope_kernel (which applies the QK norms: attn.py
#     passes q_norm/k_norm pointers into it), and any explicitly qk-norm
#     named kernel.
#   * Generic norm (rms_norm/gated_rms_norm = transformer block norms),
#     activation, sampling, dense GEMM (Tensile Cijk_*/hgemm), aten ops,
#     memory fill and the __amd_rocclr_copyBuffer driver copy kernel are
#     other_gpu with informative subcategories.
#   * Softmax is deliberately NOT an attention pattern: on this stack the
#     attention softmax is fused inside the _paged_attn kernels, so a
#     standalone softmax kernel belongs to the sampler path. Never classify
#     unrelated softmax as attention by substring.
#   * Anything unmatched stays other_gpu/other_unclassified and is listed
#     per name for root review.
#
# (kind, pattern, category, subcategory); kind in {startswith, in, re}.
CLASSIFICATION: tuple[tuple[str, str, str, str], ...] = (
    # -- MoE plumbing is other, BEFORE the fused-MoE compute match ----------
    ("re", r"exl3_moe_(?:gather|scatter|reduce|reduction|split|cpu)",
     CAT_OTHER, "moe_support"),
    ("re", r"^(?:moe_split_|moe_bias_add|moe_flag_)", CAT_OTHER, "moe_support"),
    ("startswith", "routing_", CAT_OTHER, "routing"),

    # -- dequant_related: actual quantized-weight handling -------------------
    ("startswith", "reconstruct", CAT_DEQUANT, "standalone_reconstruction"),
    ("startswith", "had_", CAT_DEQUANT, "hadamard"),
    ("in", "hadamard", CAT_DEQUANT, "hadamard"),
    ("re", r"had_(?:in|out)", CAT_DEQUANT, "hadamard"),   # BEFORE fused gemv
    ("re", r"exl3_(?:m?gemv|m?gemm)", CAT_DEQUANT, "fused_quantized_gemv_gemm"),
    ("re", r"exl3_moe", CAT_DEQUANT, "fused_quantized_moe"),

    # -- attention_related ---------------------------------------------------
    ("in", "_paged_attn", CAT_ATTENTION, "paged_attention"),
    ("in", "attn_chunked", CAT_ATTENTION, "paged_attention"),
    ("in", "attn_reduce", CAT_ATTENTION, "attention_combine"),
    ("in", "_paged_kv_update", CAT_ATTENTION, "kv_cache_update"),
    ("in", "paged_kv_update_vec8", CAT_ATTENTION, "kv_cache_update"),
    ("in", "kv_cache_update", CAT_ATTENTION, "kv_cache_update"),
    ("in", "rope_kernel", CAT_ATTENTION, "rope_qknorm"),   # applies QK norms
    ("re", r"qk_?norm", CAT_ATTENTION, "rope_qknorm"),
    ("re", r"\brope\b", CAT_ATTENTION, "rope_qknorm"),

    # -- other_gpu: informative subcategories --------------------------------
    ("in", "rms_norm", CAT_OTHER, "generic_norm"),
    ("in", "layernorm", CAT_OTHER, "generic_norm"),
    ("startswith", "norm_", CAT_OTHER, "generic_norm"),
    ("re", r"^(?:act_mul|activation_|softcap_|deinterleave_qg|silu|swiglu|gelu)",
     CAT_OTHER, "activation"),
    ("re", r"^(?:argmax_sample|gumbel|fs_|dry_penalty|apply_rep_pens|apply_pres_freq|"
            r"apply_logit_bitmask|adaptivep|sampling_)", CAT_OTHER, "sampling"),
    ("re", r"(?i)softmax", CAT_OTHER, "sampling"),          # never attention
    ("re", r"^(?:__amd_rocclr_copy|copy2d_kernel)|copyBuffer",
     CAT_OTHER, "driver_copy_kernel"),
    ("in", "FillFunctor", CAT_OTHER, "memory_fill"),
    ("re", r"^(?:fill_kernel|vectorized_fill)", CAT_OTHER, "memory_fill"),
    ("re", r"^(?:Cijk_|tensile|hipblas|hgemm|igemm)", CAT_OTHER, "dense_gemm"),
    ("re", r"(?:^|::)gemm|gemv", CAT_OTHER, "dense_gemm"),  # exl3/routing already matched
    ("in", "kernelHistogram1D", CAT_OTHER, "histogram"),
    ("startswith", "rocprim::", CAT_OTHER, "parallel_primitives"),
    ("startswith", "__amd_rocclr_", CAT_OTHER, "runtime_support"),
    ("in", "ngram", CAT_OTHER, "ngram_support"),
    ("in", "quantize", CAT_OTHER, "quant_support"),
    ("in", "at::native::", CAT_OTHER, "aten_torch_op"),
)

KERNEL_KINDS = ("KERNEL_DISPATCH", "KERNEL")


def _norm_kernel_name(name: str) -> str:
    """Fold a Kernel_Name into a classification subject.

    Handles the shapes real traces show: a demangling length prefix before
    the identifier (``_ZL25exl3_gemv...`` -- the digits sit right before
    ``exl3`` so a word-boundary pattern anchored on ``exl3`` would fail),
    ``.intern.<hex>`` and ``[clone .intern...]`` suffixes, the demangled
    ``void `` prefix and the trailing argument list.
    """
    n = (name or "").strip()
    # The anonymous-namespace wrapper carries parentheses itself; drop it
    # BEFORE trimming the trailing argument list, whose own parens would
    # otherwise make the trim greedy over the whole name.
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"\s*\[clone\b.*$", "", n)
    n = re.sub(r"\.intern\.[0-9A-Za-z]+$", "", n)
    n = re.sub(r"\(.*\)$", "", n)
    if n.startswith("void "):
        n = n[len("void "):]
    m = re.match(r"^_ZL?(\d+)", n)
    if m:
        ln = int(m.group(1))
        ident = n[m.end():m.end() + ln]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident):
            n = ident
    return n


def classify_kernel(name: str) -> tuple[str, str]:
    """Return (primary category, subcategory) for a Kernel_Name value."""
    n = _norm_kernel_name(name)
    for kind, pattern, cat, sub in CLASSIFICATION:
        if kind == "startswith":
            hit = n.startswith(pattern)
        elif kind == "in":
            hit = pattern in n
        else:
            hit = re.search(pattern, n) is not None
        if hit:
            return cat, sub
    return CAT_OTHER, "other_unclassified"


def interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    """Nanoseconds covered by the union of [start, end] GPU intervals.

    Overlapping/concurrent intervals count once, which is what keeps GPU
    busy and the sum-minus-union overlap figure honest instead of
    double-counting concurrent work.
    """
    iv = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    total = 0
    cur_s = cur_e = None
    for s, e in iv:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


# ---------------------------------------------------------------------------
# trace parsing (CPU only; tolerant of rocprofv3 nesting, strict on data)
# ---------------------------------------------------------------------------

def _read_rows(path: Path) -> tuple[list[dict], list[str]]:
    """Parse one CSV into dicts; returns (rows, errors). Raw row order kept;
    ragged rows and unreadable files are errors."""
    errors: list[str] = []
    try:
        with open(path, newline = "", encoding = "utf-8") as f:
            rd = csv.reader(f)
            try:
                header = [h.strip() for h in next(rd)]
            except StopIteration:
                return [], [f"{path.name}: empty file"]
            idx = {h: i for i, h in enumerate(header)}
            rows = []
            for lineno, raw in enumerate(rd, start = 2):
                if not raw or all(not c.strip() for c in raw):
                    continue
                if len(raw) != len(header):
                    errors.append(f"{path.name} line {lineno}: ragged row "
                                  f"({len(raw)} cols vs {len(header)})")
                    continue
                rows.append({h: raw[i].strip() for h, i in idx.items()})
        return rows, errors
    except OSError as e:
        return [], [f"{path}: unreadable: {e}"]


def _pos_ns(value: str, where: str, errors: list[str]) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        errors.append(f"{where}: timestamp {value!r} is not an integer")
        return None
    if v <= 0:
        errors.append(f"{where}: non-positive timestamp {v}")
        return None
    return v


class FileSet:
    """One process's rocprofv3 trace files (keyed by parent dir + PID)."""

    def __init__(self, key: tuple[str, int]):
        self.key = key
        self.kernel: Path | None = None
        self.marker: Path | None = None
        self.copy: Path | None = None     # absent is tolerated

    @property
    def pid(self) -> int:
        return self.key[1]


def discover_files(trace_dir: Path) -> tuple[list[FileSet], list[str]]:
    errors: list[str] = []
    sets: dict[tuple, FileSet] = {}
    for path in sorted(trace_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        kind = None
        for suffix, attr in ((KERNEL_SUFFIX, "kernel"),
                             (MARKER_SUFFIX, "marker"),
                             (COPY_SUFFIX, "copy")):
            if name.endswith(suffix):
                kind = attr
                break
        if kind is None:
            continue
        m = re.match(r"(\d+)_", name)
        if not m:
            errors.append(f"{path}: trace file without a <pid>_ filename "
                          f"prefix -- cannot attribute a process")
            continue
        key = (str(path.parent.resolve()), int(m.group(1)))
        fs = sets.setdefault(key, FileSet(key))
        if getattr(fs, kind) is not None:
            errors.append(f"duplicate {kind} trace file for process pid={key[1]} "
                          f"in {key[0]}: {getattr(fs, kind).name} and {name}")
            continue
        setattr(fs, kind, path)
    out = []
    for key, fs in sets.items():
        missing = [label for label, p in (("kernel", fs.kernel), ("marker", fs.marker))
                   if p is None]
        if missing:
            errors.append(f"process pid={key[1]} in {key[0]}: missing "
                          f"{' and '.join(missing)} trace file -- refusing to "
                          f"summarize a partial capture")
            continue
        out.append(fs)
    return out, errors


def parse_markers(fs: FileSet) -> tuple[list[dict], dict, list[str]]:
    """Selected stage ranges from MARKER_CORE_RANGE_API; control markers are
    counted and ignored. Rejects bad timestamps, duplicate rows, unknown
    stages, empty ranges, Process_Id/file-prefix mismatches."""
    errors: list[str] = []
    rows, rerr = _read_rows(fs.marker)
    errors += rerr
    if rows and not {"Domain", "Function", "Process_Id", "Start_Timestamp",
                     "End_Timestamp"} <= set(rows[0]):
        return [], {}, [f"{fs.marker.name}: unexpected marker columns "
                        f"{sorted(rows[0])}"]
    stats = {"rows": len(rows), "control_ignored": 0, "unselected_ignored": 0}
    windows: list[dict] = []
    seen: set[tuple] = set()
    for i, row in enumerate(rows):
        sig = tuple(sorted(row.items()))
        if sig in seen:
            errors.append(f"{fs.marker.name}: duplicate marker row {i}")
            continue
        seen.add(sig)
        domain = row.get("Domain", "")
        if domain == MARKER_CONTROL_DOMAIN:
            stats["control_ignored"] += 1          # roctxProfiler* control
            continue
        if domain != MARKER_RANGE_DOMAIN:
            stats["unselected_ignored"] += 1
            continue
        fn = row.get("Function", "")
        m = LABEL_RE.search(fn)
        if not m:
            stats["unselected_ignored"] += 1       # range not selected for timing
            continue
        if m.group("stage") not in STAGES:
            errors.append(f"{fs.marker.name}: row {i} unknown stage "
                          f"{m.group('stage')!r} in {fn!r}")
            continue
        where = f"{fs.marker.name} row {i}"
        s = _pos_ns(row.get("Start_Timestamp", ""), where, errors)
        e = _pos_ns(row.get("End_Timestamp", ""), where, errors)
        if s is None or e is None:
            continue
        if e < s:
            errors.append(f"{where}: marker end {e} before start {s}")
            continue
        if e == s:
            errors.append(f"{where}: empty marker range [{s},{e}] for {fn!r}")
            continue
        pid_raw = row.get("Process_Id", "")
        try:
            pid = int(pid_raw)
        except ValueError:
            errors.append(f"{where}: Process_Id {pid_raw!r} is not an integer")
            continue
        if pid != fs.pid:
            errors.append(f"{where}: marker Process_Id {pid} != filename PID "
                          f"{fs.pid} -- refusing to mix processes")
            continue
        tokens = int(m.group("tokens"))
        if tokens < 1:
            errors.append(f"{where}: marker tokens={tokens} < 1 for {fn!r}")
            continue
        windows.append({
            "process": fs.key, "pid": pid, "label": fn,
            "stage": m.group("stage"), "context": int(m.group("context")),
            "repeat": int(m.group("repeat")), "tokens": tokens,
            "start_ns": s, "end_ns": e,
        })
    return windows, stats, errors


def parse_kernels(fs: FileSet) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    rows, rerr = _read_rows(fs.kernel)
    errors += rerr
    if rows and not {"Kind", "Kernel_Name", "Start_Timestamp",
                     "End_Timestamp"} <= set(rows[0]):
        return [], [f"{fs.kernel.name}: unexpected kernel columns "
                    f"{sorted(rows[0])}"]
    out: list[dict] = []
    seen: set[tuple] = set()
    for i, row in enumerate(rows):
        kind = row.get("Kind", "")
        if kind not in KERNEL_KINDS:
            continue                       # agent info or other record kinds
        sig = tuple(sorted(row.items()))
        if sig in seen:
            errors.append(f"{fs.kernel.name}: duplicate kernel row {i}")
            continue
        seen.add(sig)
        where = f"{fs.kernel.name} row {i}"
        s = _pos_ns(row.get("Start_Timestamp", ""), where, errors)
        e = _pos_ns(row.get("End_Timestamp", ""), where, errors)
        if s is None or e is None:
            continue
        if e < s:
            errors.append(f"{where}: kernel end {e} before start {s}")
            continue
        name = row.get("Kernel_Name", "")
        cat, sub = classify_kernel(name)
        out.append({"process": fs.key, "start_ns": s, "end_ns": e,
                    "name": name, "category": cat, "subcategory": sub,
                    "agent": row.get("Agent_Id", ""),
                    "stream": row.get("Stream_Id", "")})
    return out, errors


COPY_SRC_ADDR_KEYS = ("Src_Address", "Src_Addr", "Source_Address")
COPY_DST_ADDR_KEYS = ("Dst_Address", "Dst_Addr", "Destination_Address")
COPY_SIZE_KEYS = ("Memory_Copy_Size", "Size")


def _raw_field(row: dict, keys: tuple[str, ...]):
    """First non-empty value among candidate column names, else None. Used to
    preserve CSV-supplied fields verbatim without fabricating absent data."""
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return None


def parse_copies(fs: FileSet) -> tuple[list[dict], list[str]]:
    """Memory-copy-trace rows. An absent file is NOT zero copy time; it is
    reported as not measured by the caller (fs.copy is None).

    Raw-field preservation for the two-agent split: Agent_Id, Source_Agent_Id,
    Destination_Agent_Id and Direction are kept exactly as the CSV supplies
    them (empty string when the column is absent). Direction labels have been
    observed WRONG on real traces (GPU->GPU copies labeled HOST_TO_DEVICE), so
    nothing here ever infers a host/CPU endpoint from Direction -- agent
    attribution downstream uses only the explicit agent-id fields. bytes is
    the size column when present, None when the CSV supplies none (the
    two-GPU copy trace showed no bytes at all) -- never fabricated as 0."""
    if fs.copy is None:
        return [], []
    errors: list[str] = []
    rows, rerr = _read_rows(fs.copy)
    errors += rerr
    if rows and not {"Start_Timestamp", "End_Timestamp"} <= set(rows[0]):
        return [], [f"{fs.copy.name}: unexpected memory-copy columns "
                    f"{sorted(rows[0])}"]
    out: list[dict] = []
    seen: set[tuple] = set()
    for i, row in enumerate(rows):
        sig = tuple(sorted(row.items()))
        if sig in seen:
            errors.append(f"{fs.copy.name}: duplicate memory-copy row {i}")
            continue
        seen.add(sig)
        where = f"{fs.copy.name} row {i}"
        s = _pos_ns(row.get("Start_Timestamp", ""), where, errors)
        e = _pos_ns(row.get("End_Timestamp", ""), where, errors)
        if s is None or e is None:
            continue
        if e < s:
            errors.append(f"{where}: copy end {e} before start {s}")
            continue
        kind = row.get("Kind", "")
        ctype = row.get("Copy_Type") or row.get("Location_Type") or ""
        raw_size = _raw_field(row, COPY_SIZE_KEYS)
        size: int | None = None
        if raw_size is not None:
            try:
                size = int(raw_size)
            except ValueError:
                errors.append(f"{where}: copy size {raw_size!r} is not an integer")
        out.append({"process": fs.key, "start_ns": s, "end_ns": e, "kind": kind,
                    "copy_type": f"{kind} {ctype}".strip(), "bytes": size,
                    "agent": row.get("Agent_Id", ""),
                    "source_agent": row.get("Source_Agent_Id", ""),
                    "dest_agent": row.get("Destination_Agent_Id", ""),
                    "direction": row.get("Direction", ""),
                    "src_addr": _raw_field(row, COPY_SRC_ADDR_KEYS),
                    "dst_addr": _raw_field(row, COPY_DST_ADDR_KEYS)})
    return out, errors


def check_marker_windows(windows: list[dict]) -> list[str]:
    """Reject duplicate selected ranges (same stage/context/repeat seen twice
    -- in one process or across processes) and stage windows that overlap
    within a process. The harness synchronizes at every boundary, so ranges
    must be unique and disjoint per process."""
    errors: list[str] = []
    by_key: dict[tuple, dict] = {}
    for w in windows:
        k = (w["stage"], w["context"], w["repeat"])
        if k in by_key:
            other = by_key[k]
            errors.append(
                f"duplicate selected range {k}: seen in process {other['process']} "
                f"[{other['start_ns']},{other['end_ns']}] and process {w['process']} "
                f"[{w['start_ns']},{w['end_ns']}] -- same ranges from different "
                f"processes must not be mixed; re-run with one trace dir per profile")
            continue
        by_key[k] = w
    for pid_key in sorted({w["process"] for w in windows}, key = str):
        ws = sorted((w for w in windows if w["process"] == pid_key),
                    key = lambda w: w["start_ns"])
        for a, b in zip(ws, ws[1:]):
            if b["start_ns"] < a["end_ns"]:
                errors.append(
                    f"stage windows overlap in process {pid_key}: "
                    f"{a['stage']}@{a['context']}r{a['repeat']} "
                    f"[{a['start_ns']},{a['end_ns']}] vs "
                    f"{b['stage']}@{b['context']}r{b['repeat']} "
                    f"[{b['start_ns']},{b['end_ns']}] -- the harness synchronizes "
                    f"at every boundary; overlapping windows mean a broken capture")
    return errors


def assign_to_windows(windows: list[dict], events: list[dict],
                      etype: str) -> tuple[dict[int, list[dict]], dict, list[str]]:
    """Bucket GPU events into selected stage windows, per process.

    All measured GPU activity of a stage must lie WHOLLY inside its marker
    (the window is bounded by synchronizes). Anything that crosses a
    boundary is an error -- never silently clipped. Events outside every
    selected range are tallied separately and never enter stage totals.
    """
    errors: list[str] = []
    by_window: dict[int, list[dict]] = {i: [] for i in range(len(windows))}
    outside: dict[tuple, list[dict]] = {}
    for ev in events:
        hits = [i for i, w in enumerate(windows)
                if w["process"] == ev["process"]
                and ev["start_ns"] >= w["start_ns"] and ev["end_ns"] <= w["end_ns"]]
        if len(hits) > 1:
            errors.append(f"{etype} {ev.get('name', ev.get('kind', ''))!r} "
                          f"[{ev['start_ns']},{ev['end_ns']}] inside two "
                          f"overlapping stage windows")
            continue
        if hits:
            by_window[hits[0]].append(ev)
            continue
        crossing = [w for w in windows if w["process"] == ev["process"]
                    and ev["start_ns"] < w["end_ns"] and ev["end_ns"] > w["start_ns"]]
        if crossing:
            w = crossing[0]
            errors.append(
                f"{etype} {ev.get('name', ev.get('kind', ''))!r} "
                f"[{ev['start_ns']},{ev['end_ns']}] CROSSES the "
                f"{w['stage']}@{w['context']}r{w['repeat']} window "
                f"[{w['start_ns']},{w['end_ns']}] -- boundaries must synchronize; "
                f"refusing to clip")
            continue
        outside.setdefault(ev["process"], []).append(ev)
    return by_window, outside, errors


def summarize_window(w: dict, kernels: list[dict], copies: list[dict],
                     copy_measured: bool) -> dict:
    """Per-window GPU numbers. Kernel (device) time and marker wall (host
    API span) stay in separate fields and are never summed together."""
    ksum = sum(ev["end_ns"] - ev["start_ns"] for ev in kernels)
    kunion = interval_union_ns([(ev["start_ns"], ev["end_ns"]) for ev in kernels])
    csum = sum(ev["end_ns"] - ev["start_ns"] for ev in copies)
    cunion = interval_union_ns([(ev["start_ns"], ev["end_ns"]) for ev in copies])
    busy = interval_union_ns([(ev["start_ns"], ev["end_ns"]) for ev in kernels]
                             + [(ev["start_ns"], ev["end_ns"]) for ev in copies])
    wall = w["end_ns"] - w["start_ns"]
    tokens = w["tokens"]
    if copy_measured:
        non_gpu_note = ("marker wall minus GPU busy (kernels and traced copies): "
                        "CPU submission, queue wait and profiler overhead; NOT "
                        "labeled pure CPU compute and NOT GPU idle")
    else:
        non_gpu_note = ("marker wall minus kernel busy: memory-copy trace is absent "
                        "(NOT MEASURED), so this remainder may include UNTRACED "
                        "SDMA/memory-copy activity as well as CPU submission, queue "
                        "wait and profiler overhead -- never pure CPU compute, "
                        "never GPU idle")
    cats: dict[str, dict] = {}
    names: dict[str, dict] = {}
    per_agent: dict[str, dict] = {}

    def agent_bucket(akey: str) -> dict:
        return per_agent.setdefault(akey, {
            "kernel_count": 0, "kernel_sum_ns": 0, "_kint": [], "categories": {},
            "copy_count": 0, "copy_sum_ns": 0, "_cint": [],
            "_copy_bytes": 0, "_copy_bytes_missing": 0})

    for ev in kernels:
        dur = ev["end_ns"] - ev["start_ns"]
        c = cats.setdefault(ev["category"],
                            {"kernel_sum_ns": 0, "kernel_count": 0, "kernels": [],
                             "subcategories": {}})
        c["kernel_sum_ns"] += dur
        c["kernel_count"] += 1
        c["kernels"].append(ev)
        sub = c["subcategories"].setdefault(ev["subcategory"],
                                           {"kernel_sum_ns": 0, "kernel_count": 0})
        sub["kernel_sum_ns"] += dur
        sub["kernel_count"] += 1
        nm = names.setdefault(ev["name"],
                              {"name": ev["name"], "category": ev["category"],
                               "subcategory": ev["subcategory"],
                               "count": 0, "kernel_sum_ns": 0})
        nm["count"] += 1
        nm["kernel_sum_ns"] += dur
        a = agent_bucket(ev.get("agent") or "unknown")
        a["kernel_count"] += 1
        a["kernel_sum_ns"] += dur
        a["_kint"].append((ev["start_ns"], ev["end_ns"]))
        ac = a["categories"].setdefault(
            ev["category"], {"kernel_count": 0, "kernel_sum_ns": 0, "_kint": []})
        ac["kernel_count"] += 1
        ac["kernel_sum_ns"] += dur
        ac["_kint"].append((ev["start_ns"], ev["end_ns"]))
    for c in cats.values():
        c["kernel_union_ns"] = interval_union_ns(
            [(ev["start_ns"], ev["end_ns"]) for ev in c["kernels"]])
        del c["kernels"]
        c["pct_of_window_kernel_sum"] = (round(100.0 * c["kernel_sum_ns"] / ksum, 3)
                                         if ksum else None)
        c["per_token_ns"] = round(c["kernel_sum_ns"] / tokens, 3)
        for s in c["subcategories"].values():
            s["pct_of_window_kernel_sum"] = (round(100.0 * s["kernel_sum_ns"] / ksum, 3)
                                             if ksum else None)

    # Per-agent copy totals: bucketed by the explicit Source_Agent_Id field
    # (Agent_Id fallback). Direction labels are NEVER consulted for agent
    # attribution -- they have been observed wrong (GPU->GPU labeled
    # HOST_TO_DEVICE). bytes stays None when the CSV supplies no size column.
    for ev in copies:
        a = agent_bucket(ev.get("source_agent") or ev.get("agent") or "unknown")
        a["copy_count"] += 1
        a["copy_sum_ns"] += ev["end_ns"] - ev["start_ns"]
        a["_cint"].append((ev["start_ns"], ev["end_ns"]))
        if ev.get("bytes") is None:
            a["_copy_bytes_missing"] += 1
        else:
            a["_copy_bytes"] += ev["bytes"]
    per_agent_out: dict[str, dict] = {}
    for akey in sorted(per_agent, key = lambda k: (-per_agent[k]["kernel_sum_ns"], k)):
        a = per_agent[akey]
        a["kernel_union_ns"] = interval_union_ns(a.pop("_kint"))
        for ac in a["categories"].values():
            ac["kernel_union_ns"] = interval_union_ns(ac.pop("_kint"))
        a["copy_union_ns"] = interval_union_ns(a.pop("_cint")) if copy_measured else None
        if not copy_measured:
            a["copy_sum_ns"] = None
            a["copy_bytes"] = None
        else:
            a["copy_bytes"] = (None if a["copy_count"] and a["_copy_bytes_missing"]
                               else a["_copy_bytes"])
        a.pop("_copy_bytes")
        a.pop("_copy_bytes_missing")
        per_agent_out[akey] = a
    out = {
        "stage": w["stage"], "context": w["context"], "repeat": w["repeat"],
        "tokens": tokens, "label": w["label"],
        "process": list(w["process"]), "pid": w["pid"],
        "marker_start_ns": w["start_ns"], "marker_end_ns": w["end_ns"],
        "marker_wall_ns": wall,
        "kernel_count": len(kernels), "kernel_sum_ns": ksum,
        "kernel_union_ns": kunion, "kernel_overlap_ns": ksum - kunion,
        "copy_measured": copy_measured,
        "copy_count": len(copies),
        "copy_sum_ns": csum if copy_measured else None,
        "copy_union_ns": cunion if copy_measured else None,
        "gpu_busy_union_ns": busy,
        "non_gpu_interval_ns": wall - busy,
        "non_gpu_note": non_gpu_note,
        "categories": cats,
        "per_agent": per_agent_out,
        "per_agent_note": ("keyed by the rocprof Agent_Id in the CSV (an agent id is NOT a "
                           "HIP device index; root maps it via agent_info trace metadata, "
                           "e.g. Location_Id/Drm_Render_Minor). *_union_ns are computed "
                           "WITHIN this agent; the window-level kernel_union_ns/gpu_busy_"
                           "union_ns are the CROSS-agent union -- never conflate them and "
                           "never sum per-agent unions into a global busy figure. Copies are "
                           "bucketed by Source_Agent_Id (fallback Agent_Id), never by the "
                           "unreliable Direction label; copy_bytes is None when the CSV "
                           "supplies no size for some rows and copy_* fields are None when "
                           "the copy trace was not measured at all"),
        "per_kernel": sorted(names.values(), key = lambda r: -r["kernel_sum_ns"]),
        "per_token": {
            "kernel_sum_per_token_ns": round(ksum / tokens, 3),
            "gpu_busy_per_token_ns": round(busy / tokens, 3),
        },
    }
    if w["stage"] == "prefill" and w["context"] > 0:
        out["per_token"]["per_input_token_kernel_sum_ns"] = round(ksum / w["context"], 3)
        out["per_token"]["note"] = ("prefill total is the whole prompt; "
                                    "per-input-token is optional, derived from context")
    else:
        out["per_token"]["note"] = ("decode: normalized by marker tokens "
                                    f"({tokens} generated tokens)")
    if out["kernel_overlap_ns"] < 0 or out["non_gpu_interval_ns"] < 0:
        out["anomaly"] = ("negative overlap or non-GPU interval -- the event set "
                          "is not consistent with a synchronized window")
    return out


def _dist(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "median_ns": statistics.median(vals),
            "min_ns": min(vals), "max_ns": max(vals), "values_ns": sorted(vals)}


def aggregate(windows: list[dict]) -> dict:
    """Medians/distribution per (stage, context). When a trace dir holds
    several processes the pooled group is flagged ambiguous and the
    per-process blocks are authoritative: a median is never derived by
    mismatching processes."""
    groups: dict[tuple, list[dict]] = {}
    for w in windows:
        groups.setdefault((w["stage"], w["context"]), []).append(w)
    out: dict[str, dict] = {}
    for (stage, ctx), ws in sorted(groups.items()):
        per_proc: dict[int, list[dict]] = {}
        for w in ws:
            per_proc.setdefault(w["pid"], []).append(w)

        def block(g: list[dict]) -> dict:
            cats = {}
            for cat in CATEGORIES:
                entry = {"per_window": dist_of(
                    [w["categories"].get(cat, {}).get("kernel_sum_ns", 0) for w in g])}
                if stage == "decode":
                    entry["per_decode_token_median_ns"] = statistics.median(
                        [w["categories"].get(cat, {}).get("per_token_ns", 0.0) for w in g])
                cats[cat] = entry
            return {
                "windows_n": len(g),
                "repeats": sorted(w["repeat"] for w in g),
                "marker_wall_ns": dist_of([w["marker_wall_ns"] for w in g]),
                "kernel_sum_ns": dist_of([w["kernel_sum_ns"] for w in g]),
                "gpu_busy_union_ns": dist_of([w["gpu_busy_union_ns"] for w in g]),
                "non_gpu_interval_ns": dist_of([w["non_gpu_interval_ns"] for w in g]),
                "categories": cats,
            }

        def dist_of(vals):
            return _dist(vals)

        entry = block(ws)
        entry["stage"], entry["context"] = stage, ctx
        entry["pids"] = sorted(per_proc)
        entry["multiple_processes"] = len(per_proc) > 1
        if entry["multiple_processes"]:
            entry["pooled_note"] = ("trace dir contains more than one process: the "
                                    "pooled medians above mix GPUs/processes; use "
                                    "per_process blocks for comparisons")
        entry["per_process"] = {str(pid): block(g) for pid, g in sorted(per_proc.items())}
        out[f"stage={stage};context={ctx}"] = entry
    return out


# ---------------------------------------------------------------------------
# run-json cross-check (optional)
# ---------------------------------------------------------------------------

def windows_from_run(run: dict) -> tuple[list[dict], "int | None"]:
    """Stage windows as the profile_stages artifact recorded them (traced
    windows only; --no-roctx control windows have no marker to match).
    Returns (windows, decode_tokens)."""
    out = []
    decode_tokens = ((run.get("provenance") or {}).get("decode_tokens")
                     or (run.get("params") or {}).get("decode_tokens"))
    for rec in run.get("runs", []):
        if rec.get("kind") != "timed_decode_job":
            continue
        for key in ("prefill_window", "decode_window"):
            w = rec.get(key) or {}
            if not w.get("roctx_traced"):
                continue
            out.append({
                "stage": w.get("stage"), "context": rec.get("context"),
                "repeat": rec.get("repeat"), "tokens": w.get("tokens"),
                "wall_s": w.get("wall_s"),
                "ids_sha256": rec.get("ids_sha256"),
                "generated_ids_sha256": (rec.get("sequence") or {}).get(
                    "generated_ids_sha256"),
                "job_result": rec.get("job_result") or {},
            })
    return out, decode_tokens


def cross_check(run: dict, trace_windows: list[dict]) -> tuple[dict, list[str]]:
    """Compare harness artifact to parsed trace; report deltas, never fix."""
    problems: list[str] = []
    expected, run_decode_tokens = windows_from_run(run)
    by_key: dict[tuple, dict] = {}
    for w in trace_windows:
        k = (w["stage"], w["context"], w["repeat"])
        if k in by_key:
            problems.append(f"ambiguous trace window {k}: seen from process "
                            f"{by_key[k]['pid']} and {w['pid']}")
            continue
        by_key[k] = w
    checks: list[dict] = []
    seen_keys: set[tuple] = set()
    for exp in expected:
        k = (exp["stage"], exp["context"], exp["repeat"])
        seen_keys.add(k)
        w = by_key.get(k)
        if w is None:
            problems.append(f"profiled stage window missing from trace: {k}")
            continue
        wall_s = exp["wall_s"]
        delta_pct = None
        if isinstance(wall_s, (int, float)) and wall_s > 0:
            delta_pct = 100.0 * (w["marker_wall_ns"] / 1e9 - wall_s) / wall_s
        if exp["tokens"] is not None and w["tokens"] != exp["tokens"]:
            problems.append(f"{k}: marker tokens={w['tokens']} != run record "
                            f"tokens={exp['tokens']}")
        jr = exp["job_result"]
        if k[0] == "decode":
            exp_tok = jr.get("new_tokens")
            ds = run.get("provenance", {}).get("decode_start")
            dt = run.get("provenance", {}).get("decode_tokens")
            if ds is not None and dt is not None and exp_tok is not None \
                    and exp_tok != ds + dt:
                problems.append(f"{k}: job new_tokens {exp_tok} != decode_start+tokens "
                                f"{ds + dt}")
        checks.append({
            "stage": k[0], "context": k[1], "repeat": k[2],
            "marker_wall_ns": w["marker_wall_ns"], "run_wall_s": wall_s,
            "wall_delta_pct": round(delta_pct, 3) if delta_pct is not None else None,
            "marker_tokens": w["tokens"], "run_tokens": exp["tokens"],
            "ids_sha256": exp["ids_sha256"],
            "generated_ids_sha256": exp["generated_ids_sha256"],
            "prompt_tokens": jr.get("prompt_tokens"),
            "cached_tokens": jr.get("cached_tokens"),
            "new_tokens": jr.get("new_tokens"),
            "eos_reason": jr.get("eos_reason"),
        })
        if jr.get("cached_tokens"):
            problems.append(f"{k}: run record cached_tokens={jr['cached_tokens']} "
                            f"(prefix cache hit inside a profiled run)")
        if k[0] == "decode" and jr.get("eos_reason") not in (None, "max_new_tokens"):
            problems.append(f"{k}: eos_reason={jr.get('eos_reason')!r} in a profiled "
                            f"decode window")
    for k, w in by_key.items():
        if k not in seen_keys:
            problems.append(f"selected range in trace without a run-json record: {k}")
    per_stage_ctx: dict[tuple, int] = {}
    for w in trace_windows:
        per_stage_ctx[(w["stage"], w["context"])] = \
            per_stage_ctx.get((w["stage"], w["context"]), 0) + 1
    n_repeats = ((run.get("params") or {}).get("repeats"))
    for k, cnt in sorted(per_stage_ctx.items(), key = str):
        if n_repeats and k[0] == "decode" and cnt != int(n_repeats):
            problems.append(f"trace has {cnt} windows for {k}, run params repeats="
                            f"{n_repeats}")
    deltas = [abs(c["wall_delta_pct"]) for c in checks
              if c["wall_delta_pct"] is not None]
    return {"windows": checks,
            "expected_from_run": len(expected), "found_in_trace": len(by_key),
            "max_abs_wall_delta_pct": round(max(deltas), 3) if deltas else None,
            "run_decode_tokens": run_decode_tokens,
            "problems": problems}, problems


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def build_summary(trace_dir: Path, run: dict | None) -> dict:
    result: dict = {
        "format": SUMMARY_FORMAT,
        "tool": "rocm_tools/rdna2/summarize_rocprof.py",
        "created_utc": None,
        "trace_dir": str(trace_dir.resolve()),
        "ok": False,
        "errors": [],
    }
    files, errors = discover_files(trace_dir)
    windows_all: list[dict] = []
    kernels_all: list[dict] = []
    copies_all: list[dict] = []
    per_file: dict = {}
    copy_present = False
    for fs in files:
        ws, mstats, merr = parse_markers(fs)
        ks, kerr = parse_kernels(fs)
        cs, cerr = parse_copies(fs)
        errors += merr + kerr + cerr
        windows_all += ws
        kernels_all += ks
        copies_all += cs
        if fs.copy is not None:
            copy_present = True
        per_file[f"pid{fs.pid}@{Path(fs.kernel).parent.name}"] = {
            "process": list(fs.key),
            "kernel_file": str(fs.kernel), "marker_file": str(fs.marker),
            "copy_file": str(fs.copy) if fs.copy else None,
            "marker_rows": mstats.get("rows", 0),
            "control_markers_ignored": mstats.get("control_ignored", 0),
            "unselected_ranges_ignored": mstats.get("unselected_ignored", 0),
            "selected_ranges": len(ws),
            "kernel_rows": len(ks), "copy_rows": len(cs),
        }

    if not files:
        errors.append(f"{trace_dir}: no rocprofv3 kernel/marker trace files found")
    if not windows_all:
        errors.append("no selected stage ranges (MARKER_CORE_RANGE_API rows with "
                      "stage=...;context=...;repeat=...;tokens=...) -- nothing to "
                      "summarize; refusing empty output as success")
    if not kernels_all and not errors:
        errors.append("no GPU kernel events in any trace file -- refusing to "
                      "summarize fake zero-GPU events as success")
    errors += check_marker_windows(windows_all)

    by_window, outside_k, kerr = assign_to_windows(windows_all, kernels_all, "kernel")
    by_window_c, outside_c, cerr = assign_to_windows(windows_all, copies_all,
                                                     "memory copy")
    errors += kerr + cerr

    win_records = []
    for i, w in enumerate(windows_all):
        rec = summarize_window(w, by_window.get(i, []), by_window_c.get(i, []),
                               copy_present)
        if not by_window.get(i):
            rec["error"] = ("selected stage window contains zero GPU kernels -- "
                            "empty or mislabeled capture, not a zero-time stage")
            errors.append(f"{rec['stage']}@{rec['context']}r{rec['repeat']} "
                          f"(pid {rec['pid']}): zero GPU kernels inside the marker "
                          f"window")
        if rec.get("anomaly"):
            errors.append(f"{rec['stage']}@{rec['context']}r{rec['repeat']}: "
                          f"{rec['anomaly']}")
        win_records.append(rec)

    outside = []
    for key, evs in sorted(outside_k.items(), key = str):
        outside.append({"process": list(key), "kernels_outside_count": len(evs),
                        "kernels_outside_sum_ns": sum(e["end_ns"] - e["start_ns"]
                                                      for e in evs),
                        "note": "outside every selected range; NOT part of any "
                                "stage total"})
    for key, evs in sorted(outside_c.items(), key = str):
        outside.append({"process": list(key), "copies_outside_count": len(evs),
                        "copies_outside_sum_ns": sum(e["end_ns"] - e["start_ns"]
                                                     for e in evs),
                        "note": "memory copies outside selected ranges; not in "
                                "stage totals"})

    name_rows: dict[str, dict] = {}
    for ev in kernels_all:
        nm = name_rows.setdefault(ev["name"], {"name": ev["name"],
                                               "category": ev["category"],
                                               "subcategory": ev["subcategory"],
                                               "total_count": 0, "total_sum_ns": 0})
        nm["total_count"] += 1
        nm["total_sum_ns"] += ev["end_ns"] - ev["start_ns"]
    unmatched = sorted(nm["name"] for nm in name_rows.values()
                       if nm["subcategory"] == "other_unclassified")

    result.update({
        "copy_trace": ("measured" if copy_present
                       else "absent -- NOT MEASURED (never reported as zero copy time)"),
        "files": per_file,
        "windows": win_records,
        "aggregates": aggregate(win_records),
        "kernel_name_table": sorted(name_rows.values(),
                                    key = lambda r: -r["total_sum_ns"]),
        "unclassified_kernel_names": unmatched,
        "outside_selected_ranges": outside,
        "legend": {
            "categories": "mutually exclusive primary categories: dequant_related, "
                          "attention_related, other_gpu (first matching rule wins)",
            "fused_quantized_gemv_gemm / fused_quantized_moe": "dequant and tensor-core "
                "compute fused in one kernel; reported as ONE category, never as pure "
                "dequant and never split into invented shares",
            "percentages": "pct_of_window_kernel_sum: denominator is the window's "
                           "TOTAL KERNEL SUM (never wall)",
            "non_gpu_interval_ns": "marker wall minus GPU busy. With the memory-copy "
                "trace present: CPU submission, queue wait and profiler overhead. "
                "With it absent (profiling uses --kernel-trace --marker-trace only): "
                "may ALSO include untraced SDMA/memory-copy activity. GPU busy in "
                "that case covers observed kernels only. Never labeled pure CPU "
                "compute, never labeled GPU idle",
            "copy_measured": "memory-copy-trace events. An ABSENT copy CSV means "
                "NOT MEASURED (profiling may run with --kernel-trace --marker-trace "
                "only) and is never reported as zero copy time; __amd_rocclr_"
                "copyBuffer etc. in the KERNEL trace are real GPU kernels and are "
                "classified separately (other_gpu/driver_copy_kernel)",
            "kernel vs marker time": "kernel_sum/union/busy are GPU time; "
                                     "marker_wall_ns is the host API range -- kept as "
                                     "separate fields, never summed",
            "per_agent": "per-window, per-rocprof-Agent_Id kernel count/sum/union and "
                "category sums. Agent ids are trace agent ids, NOT HIP device indices "
                "(root maps via agent_info Location_Id / Drm_Render_Minor metadata). "
                "Per-agent unions are that agent's own busy; the window-level "
                "kernel_union_ns/gpu_busy_union_ns are the cross-agent union -- the two "
                "are never conflated and per-agent busy is never summed into a global "
                "figure (on a healthy split the per-agent unions overlap in time and "
                "exceed the global union)",
            "copy_raw_fields": "memory-copy rows preserve Agent_Id, Source_Agent_Id, "
                "Destination_Agent_Id, Direction, src/dst addresses and size exactly as "
                "the CSV supplies them (empty string / None when absent). Direction "
                "labels have been observed WRONG on real two-GPU traces (GPU->GPU "
                "labeled HOST_TO_DEVICE): no host/CPU endpoint is ever inferred from "
                "Direction, and per-agent copy attribution uses only the explicit agent "
                "id fields. bytes is None when no size column is supplied -- never "
                "fabricated as 0. Root may disable the copy trace when it is unstable; "
                "its absence is NOT MEASURED, never zero",
        },
    })
    if run is not None:
        cc, cc_problems = cross_check(run, win_records)
        result["run_json_check"] = cc
        if run.get("errors"):
            cc_problems = list(cc_problems) + [f"run-json errors: {run['errors']}"]
        if run.get("cleanup_failures"):
            cc_problems = list(cc_problems) + \
                [f"run-json cleanup_failures: {run['cleanup_failures']}"]
        if run.get("ok") is False:
            cc_problems = list(cc_problems) + ["run-json reports ok=false"]
        result["run_json_check"]["problems"] = cc_problems
        errors.extend(f"run-json: {p}" for p in cc_problems)

    result["errors"] = errors
    result["ok"] = not errors
    return result


def main(argv = None) -> int:
    ap = argparse.ArgumentParser(
        prog = "summarize_rocprof.py",
        description = ("CPU-only summarizer for rocprofv3 kernel/memory-copy/marker "
                       "CSV traces: classifies GPU time per bounded stage window. "
                       "Never runs on the GPU."),
    )
    ap.add_argument("--trace-dir", required = True,
                    help = "rocprofv3 output tree (from --kernel-trace "
                           "--memory-copy-trace --marker-trace --output-format csv)")
    ap.add_argument("--output", required = True, help = "summary JSON path")
    ap.add_argument("--run-json", default = None,
                    help = "optional profile_stages artifact to cross-check marker "
                           "wall vs recorded wall, token counts and job results")
    args = ap.parse_args(argv)

    errors: list[str] = []
    run = None
    if args.run_json:
        try:
            run = json.loads(Path(args.run_json).read_text(encoding = "utf-8"))
        except Exception as e:
            errors.append(f"--run-json unreadable: {e}")
            run = None
        else:
            if not isinstance(run, dict) or run.get("format") != PROFILE_FORMAT:
                errors.append(f"--run-json format is "
                              f"{run.get('format')!r}, expected {PROFILE_FORMAT!r}")
                run = None
    td = Path(args.trace_dir)
    if not td.is_dir():
        errors.append(f"--trace-dir {td} is not a directory")

    if errors:
        out = {"format": SUMMARY_FORMAT,
               "tool": "rocm_tools/rdna2/summarize_rocprof.py",
               "trace_dir": str(td), "ok": False, "errors": errors}
    else:
        out = build_summary(td, run)

    dst = Path(args.output)
    dst.parent.mkdir(parents = True, exist_ok = True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text(json.dumps(out, indent = 2, default = str) + "\n", encoding = "utf-8")
    tmp.replace(dst)
    for e in out["errors"]:
        print(f" !! {e}", file = sys.stderr)
    print(f"  {'OK' if out['ok'] else 'FAIL'}: {len(out.get('windows', []))} "
          f"selected window(s), {len(out.get('errors', []))} error(s); "
          f"JSON: {args.output}", flush = True)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
