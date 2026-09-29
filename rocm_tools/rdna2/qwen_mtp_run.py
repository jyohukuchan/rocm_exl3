#!/usr/bin/env python3
"""Reusable Qwen3.8 Flash-Next AR/MTP evaluation CLI for the V620 pair.

Target EXL3 3.05bpw+head5 split across the pair with Engram tables in RAM; the MTP head
(by default mtp_bits3, already inside the checkpoint; see --mtp-model for a standalone
independently quantized head) is constructed and kept on cuda:1 in
BOTH modes so AR controls share the exact device layout and residency of MTP
runs. Optionally --mtp-model points ONLY the mtp component at a standalone directory of
independently quantized mtp.* shards (util/convert_mtp.py output) plus a compatible
config.json: the target config, weight split, embed_tokens/lm_head, tokenizer and Engram
stay exactly as read from --model, and Generator attach_to still binds the target's
embedding/head into the draft (so the flag also selects the resident draft in AR
controls). Attachment-relevant fields -- architecture, hidden_size, vocab_size, hc_mult,
MTP layer count -- are diffed against the target config and fail loudly BEFORE any model
construction or GPU weight load; quantization bits/qmap may differ freely, that is the
whole point. The chosen directory is reported under report["mtp_model"] with its OWN
fingerprint (same directory-listing/config semantics as model_fingerprint, PLUS full
SHA256 of every MTP *.safetensors and a combined digest -- the 85 GB target is never
rehashed; over-cap directories degrade to name|size listings and say so) and the
converter's source metadata when present. The top-level model_fingerprint keeps its
original meaning: the TARGET checkpoint only.

Power policy is delegated to the reviewed rocm_tools.rdna2.power_policy
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
    optionally add --mtp-model MTP_DIR (a directory holding a compatible config.json and
    ONLY independently quantized mtp.* .safetensors, e.g. from util/convert_mtp.py) to
    draft with that head instead of the one inside MODEL; everything else -- target
    weights, embed/lm_head, tokenizer, Engram -- is read from --model exactly as before.
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


# Config fields the draft's attach_to() binding physically depends on. The MTP head
# borrows the target's embed_tokens/lm_head and consumes the trunk's pre-collapse
# hyper-connection stream stack, so identity and widths must agree; the draft's own
# quantization (bits/mtp_bits/qmap) is deliberately NOT compared -- independently
# quantized MTP weights are the whole point of --mtp-model.
MTP_COMPAT_FIELDS = ("architecture", "hidden_size", "vocab_size", "hc_mult",
                     "mtp_num_hidden_layers")

# A standalone MTP export is ~1.5 GB; content-hashing it is cheap enough to do every
# run. Pointing --mtp-model at the 85 GB target directory instead must never trigger a
# whole-target rehash, so above this cap the fingerprint degrades to a name|bytes listing
# and the report says so.
MTP_STRONG_HASH_MAX_BYTES = 8 * 1024 ** 3


def read_safetensors_header(path):
    """Safetensors JSON header via stdlib only (tensor data bytes are never read)."""
    with open(path, "rb") as f:
        blob = f.read(8)
        if len(blob) != 8:
            raise ValueError(f"{path.name}: not a safetensors file (short header)")
        (hlen,) = struct.unpack("<Q", blob)
        if not 0 < hlen <= 1 << 28:
            raise ValueError(f"{path.name}: implausible safetensors header length {hlen}")
        raw = f.read(hlen)
    if len(raw) != hlen:
        raise ValueError(f"{path.name}: truncated safetensors header")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path.name}: unreadable safetensors header ({e})") from None
    if not isinstance(header, dict):
        raise ValueError(f"{path.name}: safetensors header is not a JSON object")
    return header


def inspect_mtp_directory(directory):
    """Pre-GPU sanity pass and strong fingerprint of a standalone MTP directory.

    Fails with an actionable ValueError when the directory is missing, holds no
    .safetensors files, or holds none with "mtp.*" tensor keys (the prefix
    util/convert_mtp.py exports). Otherwise returns per-file streaming SHA256 over the
    full bytes of every *.safetensors (standalone MTP shards are ~1.5 GB; above
    MTP_STRONG_HASH_MAX_BYTES the content hash is skipped and the listing stays
    name|bytes-only, so pointing --mtp-model at the whole 85 GB target directory can
    never rehash it), a combined digest over the sorted "name|bytes|sha256-or-dash"
    lines, tensor-key counts, and any converter-written __metadata__ per file. Only
    this directory's weight files are hashed; the target checkpoint is covered by the
    separate top-level model_fingerprint entry.
    """
    d = Path(directory)
    if not d.is_dir():
        raise ValueError(f"--mtp-model directory not found: {d}")
    files = sorted((p for p in d.glob("*.safetensors") if p.is_file()), key=lambda p: p.name)
    if not files:
        raise ValueError(f"--mtp-model directory {d} holds no *.safetensors files; write the "
                         "independently quantized mtp.* tensors here with util/convert_mtp.py "
                         "together with a compatible config.json")
    listing, metadata, mtp_count, other_count = [], {}, 0, 0
    for p in files:
        header = read_safetensors_header(p)
        md = header.get("__metadata__")
        if isinstance(md, dict) and md:
            metadata[p.name] = {str(k): str(v) for k, v in md.items()}
        for key in header:
            if key in ("__metadata__", "_header_offset"):
                continue
            if key.startswith("mtp.") or key == "mtp":
                mtp_count += 1
            else:
                other_count += 1
        listing.append({"name": p.name, "bytes": p.stat().st_size, "sha256": None})
    if not mtp_count:
        raise ValueError(f"--mtp-model directory {d}: its {len(files)} *.safetensors file(s) carry "
                         f"{other_count} tensor keys but no 'mtp.*' ones -- this is not an exported "
                         "MTP head (util/convert_mtp.py output) and cannot serve as --mtp-model")
    total = sum(e["bytes"] for e in listing)
    content_hashed = total <= MTP_STRONG_HASH_MAX_BYTES
    if content_hashed:
        for e in listing:
            h = hashlib.sha256()
            with open(d / e["name"], "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            e["sha256"] = h.hexdigest()
    combined = hashlib.sha256("\n".join(
        f"{e['name']}|{e['bytes']}|{e['sha256'] or '-'}" for e in listing).encode("utf-8")).hexdigest()
    coverage = ("SHA256 over the full bytes of every *.safetensors file listed here, combined by "
                "SHA256 over the sorted name|bytes|sha256 lines" if content_hashed else
                f"listing only (name|bytes, combined SHA256 over those lines): {total} bytes "
                f"exceed the {MTP_STRONG_HASH_MAX_BYTES} byte content-hash cap, so this directory "
                "was NOT rehashed to avoid duplicating the target checkpoint's cost")
    return {"directory": str(d.resolve()), "files": listing, "total_bytes": total,
            "content_hashed": content_hashed, "combined_sha256": combined,
            "mtp_tensor_count": mtp_count, "non_mtp_tensor_count": other_count,
            "source_metadata": metadata or None, "coverage": coverage}


def check_mtp_config_compatibility(target_cfg, mtp_cfg):
    """Fail clearly, before any model construction or GPU weight load, when a standalone
    MTP directory cannot be attached to this target.

    Compares the fields the attach binding depends on (MTP_COMPAT_FIELDS) plus the
    presence of an "mtp" component model class in the MTP directory config. Returns the
    compared values as a report-friendly dict; raises ValueError listing every mismatch.
    """
    def field(cfg, name):
        try:
            return getattr(cfg, name)
        except AttributeError:
            return None

    bad = []
    values = {}
    for name in MTP_COMPAT_FIELDS:
        t, m = field(target_cfg, name), field(mtp_cfg, name)
        values[name] = {"target": t, "mtp": m}
        if t != m:
            bad.append(f"{name}: target={t!r} mtp-directory={m!r}")
    if "mtp" not in getattr(mtp_cfg, "model_classes", {}):
        bad.append("model_classes: the MTP directory config.json defines no 'mtp' component "
                   "(text_config.mtp_num_hidden_layers must be > 0)")
    if bad:
        raise ValueError("--mtp-model is not attachable to this target: " + "; ".join(bad)
                         + "; the target's embed_tokens/lm_head are shared with the draft, so "
                           "architecture/hidden_size/vocab_size/hc_mult/mtp layer count must match "
                           "(quantization bits may differ)")
    return values


def build_parser():
    ap = argparse.ArgumentParser(
        description="Qwen3.8 Flash-Next AR/MTP evaluation on the V620 pair (frozen prompts, "
                    "policy-switched, JSON report even on failure)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-m", "--model", required=True, help="model dir holding target + mtp component")
    ap.add_argument("--mtp-model", default=None, metavar="MTP_DIR",
                    help="directory of an independently quantized MTP head (compatible config.json + "
                         "mtp.* .safetensors, e.g. written by util/convert_mtp.py); ONLY the draft "
                         "component config/weights are read from it. Target weights, split, "
                         "embed_tokens/lm_head, tokenizer and Engram always come from --model "
                         "(Generator attach_to binds them into the draft). Default: --model, i.e. the "
                         "mtp.* tensors inside the target checkpoint; applies in ar mode too, where "
                         "the draft stays resident for layout parity")
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
    if args.mtp_model is not None and not args.mtp_model.strip():
        raise ValueError("--mtp-model must be a directory path; omit it to use --model's own mtp.* tensors")
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
              "mtp_model": {"source": "custom-directory" if args.mtp_model is not None
                            else "target-checkpoint",
                            "directory": str(Path(args.mtp_model or args.model).resolve())},
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
        mtp_cfg = cfg
        if args.mtp_model is not None:
            # Independently quantized standalone head: sanity-pass and fingerprint the directory,
            # build a config for the mtp component ONLY, and verify the fields Generator.attach_to
            # binds against -- all before any model is constructed or any GPU weight is loaded.
            mtp_weights = inspect_mtp_directory(args.mtp_model)
            mtp_dir = mtp_weights["directory"]   # absolute, resolved once, reported as-is
            mtp_cfg = Config.from_directory(mtp_dir)
            mtp_cfg.infer_params.ngram_stream_from_disk = False
            mtp_compat = check_mtp_config_compatibility(cfg, mtp_cfg)
            report["mtp_model"].update({
                "directory": mtp_dir,
                "model_fingerprint": model_fingerprint(mtp_dir),
                "weights_fingerprint": {k: mtp_weights[k] for k in
                                        ("files", "total_bytes", "content_hashed", "combined_sha256",
                                         "mtp_tensor_count", "non_mtp_tensor_count", "coverage")},
                "source_metadata": mtp_weights["source_metadata"],
                "config_compatibility": mtp_compat,
                "attachment": "target embed_tokens/lm_head bound by Generator.attach_to; target "
                              "weights/split, tokenizer and Engram come from --model unchanged",
            })
        model = Model.from_config(cfg)
        draft = Model.from_config(mtp_cfg, component="mtp")   # resident even in AR control runs
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
