#!/usr/bin/env python3
"""Frozen-input, full-vocabulary BF16-relative conversion quality probe.

Collect each backend in a separate process; compare on CPU. This is a small
regression probe, not a benchmark of general model capability or long context.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def tokenizer_hashes(model):
    root = Path(model)
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json")
    return {name: sha(root / name) for name in names if (root / name).is_file()}


def runtime_fingerprint():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted((root / "exllamav3").rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(bytes.fromhex(sha(p)))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    ext = sys.modules.get("exllamav3_ext")
    native = getattr(ext, "__file__", None)
    return {"repo_base_commit": commit, "engine_python_sha256": h.hexdigest(),
            "collector_sha256": sha(Path(__file__)),
            "native_sha256": sha(Path(native)) if native else None}


def freeze(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    cases = []
    for case in json.loads(Path(args.prompts).read_text()):
        ids = tok.encode(case["text"], add_special_tokens=False)[:args.tokens]
        if len(ids) < 2:
            raise ValueError("Each case needs at least two tokens")
        cases.append({"id": case["id"], "ids": ids})
    manifest = {"cases": cases, "vocab": 1 + max(tok.get_vocab().values()),
                "prompts_sha256": sha(Path(args.prompts)), "tokens_cap": args.tokens,
                "tokenizer_sha256": tokenizer_hashes(args.model)}
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


@torch.inference_mode()
def collect(args):
    manifest = json.loads(Path(args.manifest).read_text())
    vocab = manifest["vocab"]
    tokenizer_files = tokenizer_hashes(args.model)
    if manifest.get("tokenizer_sha256", tokenizer_files) != tokenizer_files:
        raise ValueError("Model tokenizer files differ from the frozen manifest")
    torch.manual_seed(0)
    torch.set_num_threads(8)
    if args.backend == "transformers":
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
        parameter_dtype = str(next(model.parameters()).dtype)
        forward = lambda ids: model(input_ids=ids, use_cache=False).logits
    else:
        from exllamav3 import Config, Model, Tokenizer
        config = Config.from_directory(args.model)
        tok = Tokenizer.from_config(config)
        if tok.actual_vocab_size != vocab:
            raise ValueError(f"Tokenizer vocab mismatch: {tok.actual_vocab_size} != {vocab}")
        model = Model.from_config(config)
        max_length = max(len(case["ids"]) for case in manifest["cases"])
        model.load(use_per_device=[28], max_chunk_size=max_length, max_output_size=max_length)
        parameter_dtype = "EXL3 weights / engine FP16 activations"
        forward = lambda ids: model.forward(ids, {"attn_mode": "flash_attn_nc"})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for case in manifest["cases"]:
        ids = torch.tensor([case["ids"]], dtype=torch.long, device="cuda:0")
        logits = forward(ids)[0, :-1, :vocab].float().cpu()
        if logits.shape != (len(case["ids"]) - 1, vocab) or not torch.isfinite(logits).all():
            raise RuntimeError(f"Invalid logits for {case['id']}: {logits.shape}")
        torch.save(logits, output / (case["id"] + ".pt"))
        print(f"Collected {case['id']}: {len(logits)} positions", flush=True)
    metadata = {"model": args.model, "backend": args.backend,
                "tokenizer_sha256": tokenizer_files, "runtime": runtime_fingerprint(),
                "parameter_dtype": parameter_dtype, "manifest_sha256": sha(Path(args.manifest)),
                "torch": torch.__version__, "hip": torch.version.hip,
                "gpu": torch.cuda.get_device_properties(0).gcnArchName,
                "wall_collection_s": time.monotonic() - start,
                "weights_sha256": {p.name: sha(p) for p in sorted(Path(args.model).glob("*.safetensors"))}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def compare(args):
    torch.set_num_threads(8)
    manifest = json.loads(Path(args.manifest).read_text())
    paths = [Path(args.reference), Path(args.baseline), Path(args.candidate)]
    metadata = [json.loads((p / "metadata.json").read_text()) for p in paths]
    if any(m["manifest_sha256"] != sha(Path(args.manifest)) for m in metadata):
        raise ValueError("Frozen inputs differ between collections")
    tokenizers = [m.get("tokenizer_sha256") for m in metadata]
    if any(t is None for t in tokenizers) or any(t != tokenizers[0] for t in tokenizers):
        raise ValueError("Collection tokenizer fingerprints are missing or differ")
    cases = []
    for case in manifest["cases"]:
        logits = [torch.load(p / (case["id"] + ".pt"), weights_only=True) for p in paths]
        if len({tuple(x.shape) for x in logits}) != 1:
            raise ValueError("Logit shapes differ")
        targets = torch.tensor(case["ids"][1:])
        sums = dict.fromkeys(("bf16_to_baseline_kl", "bf16_to_candidate_kl",
                             "baseline_to_candidate_kl", "reference_nll", "baseline_nll",
                             "candidate_nll", "baseline_candidate_top1_agreement"), 0.0)
        for start in range(0, len(targets), 16):
            stop = min(start + 16, len(targets))
            lp = [torch.log_softmax(x[start:stop].double(), dim=-1) for x in logits]
            p = [x.exp() for x in lp]
            for key, a, b in (("bf16_to_baseline_kl", 0, 1), ("bf16_to_candidate_kl", 0, 2),
                              ("baseline_to_candidate_kl", 1, 2)):
                sums[key] += float((p[a] * (lp[a] - lp[b])).sum())
            for key, x in zip(("reference_nll", "baseline_nll", "candidate_nll"), lp):
                sums[key] += float(-x.gather(1, targets[start:stop, None]).sum())
            sums["baseline_candidate_top1_agreement"] += float(
                (logits[1][start:stop].argmax(-1) == logits[2][start:stop].argmax(-1)).sum())
        cases.append({"id": case["id"], "positions": len(targets),
                      **{k: v / len(targets) for k, v in sums.items()}})
    total = sum(c["positions"] for c in cases)
    means = {k: sum(c[k] * c["positions"] for c in cases) / total for k in sums}
    delta = means["bf16_to_candidate_kl"] - means["bf16_to_baseline_kl"]
    # Practical screening gate, explicitly not a universal quality guarantee.
    allowance = max(0.001, means["bf16_to_baseline_kl"] * 0.10)
    case_checks = {c["id"]: (c["bf16_to_candidate_kl"] - c["bf16_to_baseline_kl"]
                             <= max(0.002, c["bf16_to_baseline_kl"] * 0.20)) for c in cases}
    result = {"manifest_sha256": sha(Path(args.manifest)), "positions": total,
              "metadata": metadata, "cases": cases, "means": means,
              "screening_gate": {"max_mean_kl_increase_nats": allowance,
                                  "actual_mean_kl_increase_nats": delta,
                                  "per_case_limit": "max(0.002 nats, 20% of baseline KL)",
                                  "per_case_passed": case_checks,
                                  "passed": delta <= allowance and all(case_checks.values())},
              "limitations": "Six short text/code cases; no vision, MTP, long-context or capability evaluation."}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"means": means, "gate": result["screening_gate"]}, indent=2))
    if not result["screening_gate"]["passed"]:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("freeze", "collect", "compare"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model")
    ap.add_argument("--prompts")
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--backend", choices=("transformers", "exl3"))
    ap.add_argument("--reference")
    ap.add_argument("--baseline")
    ap.add_argument("--candidate")
    ap.add_argument("--output")
    args = ap.parse_args()
    {"freeze": freeze, "collect": collect, "compare": compare}[args.action](args)


if __name__ == "__main__":
    main()
