#!/usr/bin/env python3
"""Create the fixed token manifest for single-V620 Phase 0-2 measurement.

Tokenizes the 8 fixed examples (see examples.py for the set and its provenance)
with the *model's own tokenizer*, then freezes literal token IDs, selected
probe positions, input hashes and the valid-vocab boundary into one JSON file.
Every later step (reference/candidate top-1 collection, comparison) consumes
this manifest; nothing downstream re-tokenizes text, which is what keeps the
two backends from silently misaligning across tokenizers.

Position semantics: a position p in a case of length L is the next-token
distribution after consuming ids[0..p] (row p of the teacher-forced logits;
0 <= p <= L-1). Positions are allocated proportionally to case length and
evenly spread inside each case; selection is deterministic (no RNG).

Runs anywhere the repo package imports (the exllamav3 tokenizer is pure
Python/HF-side, but importing exllamav3 touches the built extension, so in
practice run this inside the rocm-exl3-rdna2 container):

    /opt/venv/bin/python rocm_tools/rdna2/manifest.py \
        -m /work/models/qwen3-8b-exl3-4bpw \
        -o /work/phase0/manifest_qwen3_8b.json \
        --positions 1024

Exits nonzero if the requested positions don't fit or the manifest fails
self-validation. No model weights are loaded; only config + tokenizer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rocm_tools.rdna2.common import (
    MANIFEST_FORMAT,
    POSITION_SEMANTICS,
    allocate_positions,
    case_ids_sha256,
    git_commit,
    manifest_digest,
    now_utc,
    python_env,
    repo_root,
    select_positions,
    sha256_text,
    validate_manifest,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog = "manifest.py",
        description = __doc__.splitlines()[0],
    )
    ap.add_argument("-m", "--model-dir", required = True,
                    help = "Model directory whose tokenizer defines the literal IDs "
                           "(EXL3 dir for exl3 runs; the HF dir and the EXL3 dir converted "
                           "from it share a tokenizer -- still use one of them consistently)")
    ap.add_argument("-o", "--output", required = True, help = "Manifest JSON path to write")
    ap.add_argument("-p", "--positions", type = int, default = 1024,
                    help = "Total probe positions to select across all cases (default 1024)")
    ap.add_argument("--add-bos", action = "store_true",
                    help = "Prepend BOS token if the tokenizer has one (Qwen3 has none; "
                           "default off, and the choice is recorded in the manifest)")
    return ap


def run(args: argparse.Namespace) -> int:
    # exllamav3 (and torch) load lazily so --help works on a host without the build.
    from exllamav3 import Config, Tokenizer

    from rocm_tools.rdna2.examples import get_cases

    root = repo_root()
    cases_src = get_cases(root)
    if len(cases_src) != 8:
        print(f" !! expected 8 fixed examples, found {len(cases_src)}", file = sys.stderr)
        return 1

    config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(config)
    valid_vocab = int(tokenizer.actual_vocab_size)
    config_vocab = int(getattr(config, "vocab_size", 0) or 0)

    records = []
    lengths = []
    for spec in cases_src:
        ids_t = tokenizer.encode(spec["text"], add_bos = args.add_bos)
        ids = [int(t) for t in ids_t.reshape(-1).tolist()]
        if not ids:
            print(f" !! case {spec['case_id']} tokenized empty", file = sys.stderr)
            return 1
        # Literal IDs must live inside the valid-vocab range; anything else means we
        # are feeding padding/special values the argmax comparison can't interpret.
        out_of_range = [t for t in ids if t < 0 or t >= valid_vocab]
        if out_of_range:
            print(f" !! case {spec['case_id']}: {len(out_of_range)} token IDs outside "
                  f"[0, {valid_vocab}) -- refusing", file = sys.stderr)
            return 1
        lengths.append(len(ids))
        rec = {
            "case_id": spec["case_id"],
            "language": spec["language"],
            "kind": spec["kind"],
            "provenance": spec["provenance"],
            "text_sha256": sha256_text(spec["text"]),
            "add_bos": bool(args.add_bos),
            "len_ids": len(ids),
            "ids": ids,
            "ids_sha256": case_ids_sha256(ids),
        }
        for extra in ("source_file", "source_file_sha256"):
            if extra in spec:
                rec[extra] = spec[extra]
        records.append(rec)

    capacity = sum(lengths)
    if args.positions > capacity:
        print(f" !! requested {args.positions} positions but the 8 fixed examples only "
              f"yield {capacity} with this tokenizer; lower --positions", file = sys.stderr)
        return 1
    counts = allocate_positions(lengths, args.positions)
    for rec, count in zip(records, counts):
        rec["positions"] = select_positions(rec["len_ids"], count)

    manifest = {
        "format": MANIFEST_FORMAT,
        "created_utc": now_utc(),
        "tool": "rocm_tools/rdna2/manifest.py",
        "model_dir": str(Path(args.model_dir).resolve()),
        "repo_git_commit": git_commit(root),
        "python": python_env(),
        "requested_positions": args.positions,
        "total_positions": sum(len(r["positions"]) for r in records),
        "position_semantics": POSITION_SEMANTICS,
        "valid_vocab_size": valid_vocab,
        "config_vocab_size": config_vocab,
        "tokenizer": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "cases": records,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)

    errors = validate_manifest(manifest)
    if errors:
        print(" !! manifest failed self-validation:", file = sys.stderr)
        for e in errors:
            print(f"    {e}", file = sys.stderr)
        return 1

    write_json(args.output, manifest)
    print(f" -- manifest: {args.output}")
    print(f" -- valid_vocab_size={valid_vocab} config_vocab_size={config_vocab} "
          f"positions={manifest['total_positions']} (requested {args.positions})")
    print(f" -- digest: {manifest['manifest_sha256']}")
    for rec in records:
        print(f"    {rec['case_id']:20} {rec['language']:>2} len={rec['len_ids']:5} "
              f"positions={len(rec['positions']):5}")
    return 0


def main(argv = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f" !! manifest creation failed: {e}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
