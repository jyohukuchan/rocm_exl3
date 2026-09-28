"""Shared, stdlib-only helpers for the single-V620 (gfx1030) measurement harness.

This module must import on a CPU-only host with no ROCm extension build: it
deliberately never imports torch or exllamav3 at module scope. Anything GPU
facing lives behind lazy imports in the CLIs that need it (manifest.py and
collect_top1.py load the tokenizer/models; bench.py loads a model). Comparison
and validation -- the logic the CPU tests cover -- is pure data processing.

Artifacts are plain JSON. Two schemas:

  manifest  (rdna2-token-manifest/1): literal token IDs per case, selected
            positions, valid vocab size, input hashes, provenance.
  top1      (rdna2-top1/1): teacher-forced argmax token ID per (case, position)
            for one backend/model run, plus manifest digest, env and params.

Position semantics (identical in both schemas, see POSITION_SEMANTICS):
a position p in a case of length L means "the next-token distribution after
consuming ids[0..p]" -- i.e. row p of the logits the model returns for the
prefilled prefix. 0 <= p <= L-1; row L-1 predicts the continuation token
(EOS-style) after the full case. Both backends return logits for every input
row (eval/ppl.py: model.forward(ids, {"attn_mode":"flash_attn_nc"}) sliced
[:, :-1, :] against ids[:, 1:]; Transformers CausalLMOutput["logits"] uses the
same alignment), so no shifting is applied at collection time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

MANIFEST_FORMAT = "rdna2-token-manifest/1"
TOP1_FORMAT = "rdna2-top1/1"
BENCH_FORMAT = "rdna2-bench/1"

POSITION_SEMANTICS = (
    "position p = next-token distribution after consuming ids[0..p] "
    "(row p of model logits; 0 <= p <= len(ids)-1; row len(ids)-1 predicts "
    "the token after the case)"
)

# exllamav3.constants.PAGE_SIZE. Kept as a literal so host tools import nothing
# GPU-facing; collect_top1.py re-derives the real value once exllamav3 is loaded.
PAGE_SIZE = 256


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys = True, ensure_ascii = False,
                      separators = (",", ":")).encode("utf-8")


def read_json(path: str | Path):
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding = "utf-8") as f:
        json.dump(obj, f, indent = 2, sort_keys = False, ensure_ascii = False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def round_up_page(n: int) -> int:
    return -(-n // PAGE_SIZE) * PAGE_SIZE


# ---------------------------------------------------------------------------
# hashing of manifest pieces
# ---------------------------------------------------------------------------

def case_ids_sha256(ids: list[int]) -> str:
    return sha256_hex(canonical_bytes(list(ids)))


def manifest_digest(manifest: dict) -> str:
    """Canonical hash over the full manifest content except its own digest field."""
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return sha256_hex(canonical_bytes(body))


# ---------------------------------------------------------------------------
# position selection (deterministic, no RNG)
# ---------------------------------------------------------------------------

def allocate_positions(lengths: list[int], total: int) -> list[int]:
    """
    Split `total` selected positions over cases of the given lengths, proportionally
    to case length (largest-remainder), clamped to [1, length] per case. Deterministic:
    same inputs always give the same per-case counts.
    """
    if not lengths:
        raise ValueError("no cases")
    capacity = sum(lengths)
    if total > capacity:
        raise ValueError(f"requested {total} positions but cases only yield {capacity}")
    if total < len(lengths):
        raise ValueError(f"requested {total} positions for {len(lengths)} cases (need >= 1 per case)")
    s = sum(lengths)
    raw = [total * L / s for L in lengths]
    counts = [max(1, min(L, math.floor(r))) for r, L in zip(raw, lengths)]
    remainder = total - sum(counts)
    # largest fractional part first; ties by case index for determinism
    order = sorted(range(len(lengths)), key = lambda i: (-(raw[i] - math.floor(raw[i])), i))
    while remainder > 0:
        progressed = False
        for i in order:
            if counts[i] < lengths[i]:
                counts[i] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:  # cannot happen while remainder <= capacity - sum(min 1)
            raise ValueError("position allocation stalled")
    if sum(counts) != total:
        raise ValueError("position allocation mismatch")
    return counts


def select_positions(length: int, count: int) -> list[int]:
    """
    `count` distinct, strictly increasing positions in [0, length-1], evenly spread:
    p_j = floor((j + 0.5) * length / count). Deterministic; no RNG involved.
    """
    if not 1 <= count <= length:
        raise ValueError(f"count {count} outside [1, length {length}]")
    return [int((j + 0.5) * length // count) for j in range(count)]


# ---------------------------------------------------------------------------
# manifest / result validation (pure data checks; the CPU tests cover this)
# ---------------------------------------------------------------------------

def validate_manifest(m) -> list[str]:
    """Return a list of human-readable problems; empty list means the manifest is sound."""
    errors: list[str] = []
    if not isinstance(m, dict):
        return ["manifest is not a JSON object"]
    if m.get("format") != MANIFEST_FORMAT:
        errors.append(f"manifest format is {m.get('format')!r}, expected {MANIFEST_FORMAT!r}")
    stored = m.get("manifest_sha256")
    if not stored or not isinstance(stored, str):
        errors.append("manifest_sha256 missing")
    elif stored != manifest_digest(m):
        errors.append("manifest_sha256 does not match manifest content (file was edited or truncated)")
    vocab = m.get("valid_vocab_size")
    if not isinstance(vocab, int) or vocab <= 0:
        errors.append("valid_vocab_size missing or not a positive integer")
    if not isinstance(m.get("cases"), list) or not m.get("cases"):
        errors.append("cases missing or empty")
        return errors
    seen_ids: set = set()
    for case in m["cases"]:
        cid = case.get("case_id")
        ctx = f"case {cid!r}"
        if cid in seen_ids:
            errors.append(f"{ctx}: duplicate case_id")
        seen_ids.add(cid)
        ids = case.get("ids")
        if not isinstance(ids, list) or not ids:
            errors.append(f"{ctx}: ids missing or empty")
            continue
        if any(not isinstance(t, int) or isinstance(t, bool) for t in ids):
            errors.append(f"{ctx}: ids must be plain integers (literal token IDs)")
            continue
        if isinstance(vocab, int) and any(t < 0 or t >= vocab for t in ids):
            errors.append(f"{ctx}: token ID outside [0, valid_vocab_size) -- possible cross-tokenizer mismatch")
        if case_ids_sha256(ids) != case.get("ids_sha256"):
            errors.append(f"{ctx}: ids_sha256 does not match ids content")
        pos = case.get("positions")
        if not isinstance(pos, list) or not pos:
            errors.append(f"{ctx}: positions missing or empty")
            continue
        if any(not isinstance(p, int) or isinstance(p, bool) for p in pos):
            errors.append(f"{ctx}: positions must be integers")
            continue
        if pos != sorted(set(pos)):
            errors.append(f"{ctx}: positions must be sorted and free of duplicates")
        bad = [p for p in pos if p < 0 or p >= len(ids)]
        if bad:
            errors.append(f"{ctx}: {len(bad)} position(s) outside [0, len(ids)-1] (first: {bad[0]})")
    if isinstance(m.get("total_positions"), int):
        n = sum(len(c.get("positions") or []) for c in m["cases"])
        if n != m["total_positions"]:
            errors.append(f"total_positions {m['total_positions']} != summed positions {n}")
    return errors


def _validate_top1_body(m: dict, errors: list[str]) -> int:
    """Shared body checks for a top1 result file; returns collected-position count."""
    cases = m.get("cases")
    if not isinstance(cases, dict) or not cases:
        errors.append("cases missing or empty (no results in file)")
        return 0
    vocab = m.get("valid_vocab_size")
    n_collected = 0
    for cid, case in cases.items():
        ctx = f"case {cid!r}"
        if not isinstance(case, dict):
            errors.append(f"{ctx}: not an object")
            continue
        status = case.get("status")
        if status == "error":
            errors.append(f"{ctx}: collection errored: {case.get('error', 'no detail')}")
            continue
        if status != "ok":
            errors.append(f"{ctx}: unknown status {status!r}")
            continue
        pos = case.get("positions")
        top1 = case.get("top1")
        if not isinstance(pos, list) or not isinstance(top1, list):
            errors.append(f"{ctx}: positions/top1 missing")
            continue
        if len(pos) != len(top1):
            errors.append(f"{ctx}: {len(top1)} top1 values for {len(pos)} positions")
            continue
        if case.get("nonfinite_positions"):
            errors.append(
                f"{ctx}: {len(case['nonfinite_positions'])} position(s) had non-finite logits "
                f"(first: {case['nonfinite_positions'][0]}) -- result is not trustworthy")
        bad = [p for p in pos if not isinstance(p, int) or isinstance(p, bool)]
        if bad:
            errors.append(f"{ctx}: non-integer position values")
        bad = [t for t in top1 if not isinstance(t, int) or isinstance(t, bool)]
        if bad:
            errors.append(f"{ctx}: non-integer top1 values")
        elif isinstance(vocab, int) and any(t < 0 or t >= vocab for t in top1):
            errors.append(f"{ctx}: top1 token ID outside [0, valid_vocab_size)")
        n_collected += len(pos)
    return n_collected


def validate_top1_result(m) -> list[str]:
    """Structural checks for a top1 result file (self-consistency only; the
    cross-file manifest/position checks live in compare_top1.compare_files)."""
    errors: list[str] = []
    if not isinstance(m, dict):
        return ["result is not a JSON object"]
    if m.get("format") != TOP1_FORMAT:
        errors.append(f"format is {m.get('format')!r}, expected {TOP1_FORMAT!r}")
    if not m.get("manifest_sha256"):
        errors.append("manifest_sha256 missing (result not tied to a token manifest)")
    n = _validate_top1_body(m, errors)
    if m.get("complete") is not True:
        errors.append(f"result flagged incomplete (complete={m.get('complete')!r})")
    if m.get("errors"):
        errors.append(f"result carries {len(m['errors'])} error record(s)")
    total = m.get("total_positions")
    if isinstance(total, int) and n < total:
        errors.append(f"incomplete: {n} positions collected out of {total}")
    return errors


# ---------------------------------------------------------------------------
# environment / provenance capture (cheap, no CUDA init unless asked)
# ---------------------------------------------------------------------------

def git_commit(root: Path | None = None) -> str | None:
    root = root or repo_root()
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output = True, text = True, timeout = 10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def python_env() -> dict:
    import platform
    import sys
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "argv0": os.path.basename(sys.argv[0]),
    }


def torch_env() -> dict:
    """torch version strings without touching CUDA (no device is initialised)."""
    try:
        import torch
    except Exception as e:
        return {"torch": None, "torch_import_error": repr(e)}
    return {
        "torch": torch.__version__,
        "torch_version_hip": getattr(getattr(torch, "version", None), "hip", None),
        "torch_version_cuda": getattr(getattr(torch, "version", None), "cuda", None),
    }


def visible_gpu_env() -> dict:
    return {k: os.environ[k] for k in
            ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
            if k in os.environ}


def rocm_patch_env() -> dict:
    """Engine and ROCr transfer switches affecting measured behavior."""
    transfer_keys = {"HSA_ENABLE_SDMA", "HSA_ENABLE_PEER_SDMA", "EXLLAMA_NO_P2P_COPY"}
    return {k: v for k, v in os.environ.items() if k.startswith("EXL3_ROCM") or
            k.startswith("EXL3_") or k in transfer_keys}


def model_fingerprint(model_dir: str) -> dict:
    """Identity of a model directory: file listing + config content. No model load."""
    d = Path(model_dir)
    fp: dict = {"dir": str(d.resolve()), "name": d.name, "exists": d.is_dir()}
    if not d.is_dir():
        return fp
    entries = []
    for p in sorted(d.iterdir(), key = lambda p: p.name):
        if p.is_file():
            entries.append(f"{p.name}|{p.stat().st_size}")
    fp["files_fingerprint_sha256"] = sha256_hex("\n".join(entries).encode("utf-8"))
    for cfg in ("config.json", "generation_config.json"):
        if (d / cfg).is_file():
            fp[f"{cfg}_sha256"] = sha256_file(d / cfg)
    return fp


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# dtype decision (pure function so CPU tests cover the "auto" logic)
# ---------------------------------------------------------------------------

DTYPE_KEYS = ("auto", "fp16", "bf16", "fp32")
_DTYPE_FROM_DECLARED = {
    "bfloat16": "bf16", "torch.bfloat16": "bf16",
    "float16": "fp16", "torch.float16": "fp16", "half": "fp16",
    "float32": "fp32", "torch.float32": "fp32", "float": "fp32",
}


def resolve_dtype_choice(requested: str, declared: str | None) -> tuple[str, str]:
    """
    Which load dtype the transformers backend should use. "auto" keeps the
    reference in the checkpoint's native dtype (BF16 source -> BF16 reference,
    i.e. truly unquantized-and-unconverted); an explicit request wins over the
    declaration. Returns (choice, note) with the note recorded in artifacts so
    metadata describes what actually happened rather than what was assumed.
    """
    if requested != "auto":
        return requested, "explicit --dtype"
    if declared:
        key = _DTYPE_FROM_DECLARED.get(declared.lower())
        if key:
            return key, f"resolved from checkpoint torch_dtype={declared}"
    return "fp16", f"checkpoint declares {declared!r}; fell back to fp16"
