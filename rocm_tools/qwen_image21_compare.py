#!/usr/bin/env python3
"""Audit matched QwenImage21 results and summarize storage/quality/timings.

Image differences are diagnostic, not a substitute for reviewing the six
generation/editing criteria. Timing ratios use only the dedicated benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm_tools.qwen_image21 import read_json
from rocm_tools.qwen_image21_suite import CASES, matching_report, checkpoint_path, precision_label


def image_pixels(path, measurement):
    image = Image.open(path)
    if (image.mode != measurement["image_mode"] or list(image.size) != measurement["image_size"]
            or hashlib.sha256(image.tobytes()).hexdigest() != measurement.get("pixel_sha256")):
        raise ValueError(f"PNG does not match its recorded output: {path}")
    if not measurement.get("finite_final_latents") or not measurement.get("finite_vae_output"):
        raise ValueError(f"Missing finite-output verification: {path}")
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    composited = rgba[..., :3] * alpha + (1.0 - alpha)
    return composited, alpha


def paired_reports(left, right):
    # A weight reconstruction budget has no BF16 equivalent; everything else
    # controlling sampling, placement and VAE must agree.
    excluded = {"reconstruction_workspace_bytes"}
    a = {k: v for k, v in left["settings"].items() if k not in excluded}
    b = {k: v for k, v in right["settings"].items() if k not in excluded}
    if a != b or left["source"] != right["source"] or left["versions"] != right["versions"]:
        raise ValueError("BF16 and EXL3 settings/source/runtime differ")
    if left["gpu"] != right["gpu"] or left["arch"] != right["arch"]:
        raise ValueError("BF16 and EXL3 hardware differs")
    if left["text_encoding"] != right["text_encoding"]:
        raise ValueError("BF16 and EXL3 prompt encodings differ")
    # Quality baselines can predate bookkeeping-only CLI changes. Match the
    # model sources and dependencies; benchmarks additionally require the
    # current identical CLI through matching_report(repeats=3).
    for key in ("runtime_source_sha256", "dependency_pins"):
        if left["implementation"].get(key) != right["implementation"].get(key):
            raise ValueError("BF16 and EXL3 model implementations differ")
    if (left.get("allocator_configuration") is None
            or left["allocator_configuration"] != right.get("allocator_configuration")):
        raise ValueError("BF16 and EXL3 allocator settings differ or lack provenance")


def attach_manual_review(report, review):
    unknown = set(review["cases"]) - set(report["cases"])
    if unknown:
        raise ValueError(f"Review contains unknown cases: {sorted(unknown)}")
    for name, entry in review["cases"].items():
        case = report["cases"][name]
        if any(entry.get(key) != case[key] for key in ("bf16_pixels_sha256", "exl3_pixels_sha256")):
            raise ValueError(f"Review refers to different generated images: {name}")
        if entry.get("verdict") not in ("pass", "fail") or not entry.get("observation", "").strip():
            raise ValueError(f"Review needs a verdict and observation: {name}")
        case["manual_review"] = entry["verdict"]
        case["manual_observation"] = entry["observation"]
    report["manual_review_required"] = any(c["manual_review"] == "pending" for c in report["cases"].values())
    report["quality_screen_passed"] = (None if report["manual_review_required"] else
                                       all(c["manual_review"] == "pass" for c in report["cases"].values()))
    report["reviewer"] = review.get("reviewer")


def compare(args):
    cases = read_json(CASES)["evaluation"]
    checkpoint = checkpoint_path(args.root, args.variant, args.bits)
    manifest = read_json(checkpoint / "quantization.json")
    report = {"variant": args.variant, "source": manifest["source"], "cases": {},
              "manual_review_required": True,
              "storage": {key: manifest[key] for key in (
                  "source_state_bytes", "quantized_state_bytes", "unquantized_state_bytes",
                  "state_size_ratio", "effective_bits_per_source_element")}}
    report["storage"]["safetensors_bytes"] = sum(p.stat().st_size for p in checkpoint.glob("*.safetensors"))
    report["storage"]["source_safetensors_bytes"] = sum(p.stat().st_size for p in (args.model / "transformer").glob("*.safetensors"))
    report["storage"]["file_size_ratio"] = report["storage"]["safetensors_bytes"] / report["storage"]["source_safetensors_bytes"]
    report["storage"]["target_met"] = report["storage"]["file_size_ratio"] <= 0.4
    for case in cases:
        refs = [args.root / args.variant / "bf16" / ref / "run-0.png" for ref in case.get("reference_cases", [])]
        pair = {}
        for precision in ("bf16", "exl3"):
            directory = args.root / args.variant / precision_label(precision, args.bits) / case["id"]
            result = read_json(directory / "report.json")
            if not matching_report(result, case, refs, args.variant, args.model, precision,
                                   checkpoint=checkpoint, offload="none"):
                raise ValueError(f"Case report is inconsistent: {directory}")
            pair[precision] = result
        paired_reports(pair["bf16"], pair["exl3"])
        left = image_pixels(args.root / args.variant / "bf16" / case["id"] / "run-0.png",
                            pair["bf16"]["measurements"][0])
        right = image_pixels(args.root / args.variant / "exl3" / case["id"] / "run-0.png",
                             pair["exl3"]["measurements"][0])
        expected_calls = pair["exl3"]["measurements"][0]["actual_steps"]
        observed_layers = [name for name, layer in pair["exl3"]["linear_shapes"].items()
                           if layer["weight_dtype"] == "exl3" and sum(layer["inputs"].values()) == expected_calls]
        if set(observed_layers) != set(manifest["layers"]):
            raise ValueError(f"Not all EXL3 layers were executed in {case['id']}")
        report["cases"][case["id"]] = {
            "review_criterion": case["review"], "manual_review": "pending",
            "cli_sha256": {p: pair[p]["implementation"]["cli_sha256"] for p in ("bf16", "exl3")},
            "rgb_mae_white_background_255": float(np.abs(left[0] - right[0]).mean() * 255),
            "alpha_mae_255": float(np.abs(left[1] - right[1]).mean() * 255),
            "bf16_alpha_extrema": [float(left[1].min()), float(left[1].max())],
            "exl3_alpha_extrema": [float(right[1].min()), float(right[1].max())],
            "executed_exl3_layers": len(observed_layers),
            "bf16_pixels_sha256": pair["bf16"]["measurements"][0]["pixel_sha256"],
            "exl3_pixels_sha256": pair["exl3"]["measurements"][0]["pixel_sha256"],
        }
    benchmark_case = next(c for c in cases if c["id"] == "composition")
    pair = {}
    for precision in ("bf16", "exl3"):
        result = read_json(args.root / args.variant / "benchmark" / precision_label(precision, args.bits) / "report.json")
        if not matching_report(result, benchmark_case, [], args.variant, args.model, precision,
                               repeats=3, warmup=1, checkpoint=checkpoint, offload="text-encoder"):
            raise ValueError(f"Benchmark does not match current implementation/settings: {precision}")
        pair[precision] = result
    paired_reports(pair["bf16"], pair["exl3"])
    report["benchmark"] = {}
    for precision, result in pair.items():
        runs = [run for run in result["measurements"] if not run["warmup"]]
        report["benchmark"][precision] = {
            "pipeline_seconds_median": statistics.median(run["pipeline_seconds"] for run in runs),
            "denoising_ms_median": statistics.median(run["denoising_ms"] for run in runs),
            "peak_allocated_bytes_max": max(run["peak_allocated_bytes"] for run in runs),
            "peak_reserved_bytes_max": max(run["peak_reserved_bytes"] for run in runs),
        }
    bf16, exl3 = (report["benchmark"][p] for p in ("bf16", "exl3"))
    report["benchmark"]["denoising_time_ratio"] = exl3["denoising_ms_median"] / bf16["denoising_ms_median"]
    report["benchmark"]["pipeline_time_ratio"] = exl3["pipeline_seconds_median"] / bf16["pipeline_seconds_median"]
    report["benchmark"]["peak_allocated_ratio"] = exl3["peak_allocated_bytes_max"] / bf16["peak_allocated_bytes_max"]
    report["benchmark"]["denoising_target_met"] = report["benchmark"]["denoising_time_ratio"] <= 1.3
    if args.reviews:
        attach_manual_review(report, read_json(args.reviews))
        report["review_file_sha256"] = hashlib.sha256(args.reviews.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"storage": report["storage"], "benchmark": report["benchmark"]}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("base", "turbo"), required=True)
    ap.add_argument("--bits", type=int, choices=range(1, 9), default=4)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reviews", type=Path, help="Attach explicit image-hash-bound manual review decisions")
    compare(ap.parse_args())


if __name__ == "__main__":
    main()
