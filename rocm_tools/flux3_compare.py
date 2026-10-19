#!/usr/bin/env python3
"""Validate matched FLUX.3 measurements and export a concise comparison."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def matched(reports, keys):
    for key in keys:
        if any(r[key] != reports[0][key] for r in reports[1:]):
            raise ValueError(f"Comparison conditions differ: {key}")


def compare(root, bits):
    precisions = [f"exl3-{b}bit" for b in bits]
    benchmark = {p: read(root / "benchmark" / p / "composition/report.json") for p in ["bf16", *precisions]}
    reports = list(benchmark.values())
    matched(reports, ("source", "versions", "implementation", "gpu", "arch", "caption", "images", "timing_scope"))
    setting_keys = ("width", "height", "seed", "steps", "vae_dtype", "vae_tiling", "vae_tile_sample_min_size",
                    "vae_tile_overlap_factor", "text_encoder_offload", "encoding_cache", "attention_query_chunk")
    matched([r["settings"] for r in reports], setting_keys)
    speed = {}
    for precision, report in benchmark.items():
        if report["measurement_purpose"] != "benchmark" or report["nmse"] is not None:
            raise ValueError("Instrumented measurements cannot be used as speed benchmarks")
        rows = report["measurements"]
        if [r["warmup"] for r in rows] != [True, False, False, False]:
            raise ValueError("Benchmark needs one warmup and three measured runs")
        measured = rows[1:]
        if any(r["actual_steps"] != 16 or not r["finite_final_latents"] or not r["finite_vae_output"] for r in rows):
            raise ValueError("Incomplete or nonfinite benchmark")
        hashes = sorted({r["pixel_sha256"] for r in measured})
        speed[precision] = {
            "pipeline_seconds": statistics.median(r["pipeline_seconds"] for r in measured),
            "denoising_seconds": statistics.median(r["denoising_seconds"] for r in measured),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in measured),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in measured),
            "pixel_hashes": hashes, "repeated_image_hashes_equal": len(hashes) == 1,
        }
        if precision != "bf16" and report["quantization"]["bits"] != int(precision.split("-")[1][:-3]):
            raise ValueError("Benchmark bit width differs from its label")
    for row in speed.values():
        row["pipeline_relative_to_bf16"] = row["pipeline_seconds"] / speed["bf16"]["pipeline_seconds"]
        row["denoising_relative_to_bf16"] = row["denoising_seconds"] / speed["bf16"]["denoising_seconds"]
    nmse, cases = {}, ("text", "composition", "multi_ref")
    all_reports = []
    per_case = {}
    for case in cases:
        rows = {p: read(root / "nmse" / p / case / "report.json") for p in precisions}
        per_case[case] = rows
        matched(list(rows.values()), ("source", "versions", "implementation", "gpu", "arch", "caption", "images", "settings"))
        baseline = rows[precisions[0]]["nmse"]
        for precision, report in rows.items():
            metric = report["nmse"]
            if report["measurement_purpose"] != "nmse" or metric["candidate"] != "exl3":
                raise ValueError("Expected teacher-trajectory EXL3 NMSE measurements")
            if report["precision"] != "bf16":
                raise ValueError("NMSE teacher must be BF16")
            if metric["candidate_quantization"]["bits"] != int(precision.split("-")[1][:-3]):
                raise ValueError("NMSE bit width differs from its label")
            if len(metric["steps"]) != 16 or metric["conditioning_sha256"] != baseline["conditioning_sha256"]:
                raise ValueError("NMSE conditioning or step counts differ")
            for reference, current in zip(baseline["steps"], metric["steps"]):
                matched([reference, current], ("latent_sha256", "reference_prediction_sha256", "timestep", "target_timestep", "elements"))
            if not math.isclose(metric["reference_squared_sum"], baseline["reference_squared_sum"], rel_tol=1e-12):
                raise ValueError("NMSE reference energies differ")
            layers = {name: value for name, value in report["candidate_linear_layers"].items() if value["precision"] == "exl3"}
            if len(layers) != 258 or any(value["calls"] != (0 if name.startswith("mode_blocks.image_cond.") and not report["images"] else 16)
                                         for name, value in layers.items()):
                raise ValueError("NMSE did not execute all expected EXL3 layers")
            all_reports.append(report)
    matched(all_reports, ("source", "versions", "implementation", "gpu", "arch"))
    for precision in precisions:
        rows = [per_case[c][precision]["nmse"] for c in cases]
        error = math.fsum(r["squared_error_sum"] for r in rows)
        energy = math.fsum(r["reference_squared_sum"] for r in rows)
        if not math.isfinite(error) or not math.isfinite(energy) or error < 0 or energy <= 0:
            raise ValueError("NMSE requires finite errors and positive reference energy")
        nmse[precision] = {"nmse": error / energy, "squared_error_sum": error, "reference_squared_sum": energy,
                           "cases": {c: per_case[c][precision]["nmse"]["nmse"] for c in cases}}
    self_report = read(root / "nmse/bf16/composition/report.json")
    if self_report["nmse"]["candidate"] != "bf16-self" or len(self_report["nmse"]["steps"]) != 16:
        raise ValueError("Missing BF16 self-comparison")
    first = per_case["composition"][precisions[0]]
    matched([self_report, *all_reports, *reports], ("source", "versions", "implementation", "gpu", "arch"))
    matched([self_report, first], ("source", "versions", "caption", "images", "settings", "gpu", "arch"))
    if self_report["nmse"]["conditioning_sha256"] != first["nmse"]["conditioning_sha256"]:
        raise ValueError("Self-comparison conditioning differs")
    for reference, current in zip(first["nmse"]["steps"], self_report["nmse"]["steps"]):
        matched([reference, current], ("latent_sha256", "reference_prediction_sha256", "timestep", "target_timestep"))
    manifests = {p: read(root / p / "quantization.json") for p in precisions}
    matched(list(manifests.values()), ("source", "calibration", "source_state_numel", "unquantized_sha256"))
    for precision in precisions:
        if benchmark[precision]["quantization"] != manifests[precision]:
            raise ValueError("Benchmark checkpoint differs from the current model")
        if any(per_case[c][precision]["nmse"]["candidate_quantization"] != manifests[precision] for c in cases):
            raise ValueError("NMSE checkpoint differs from the current model")
    return {"source": reports[0]["source"], "gpu": reports[0]["gpu"], "arch": reports[0]["arch"],
            "benchmark_settings": reports[0]["settings"], "benchmark": speed, "nmse": nmse,
            "nmse_cases": list(cases), "bf16_self_nmse": self_report["nmse"]["nmse"],
            "teacher_latents_predictions_and_conditioning_identical_across_bits": True,
            "models": {p: {k: m[k] for k in ("bits", "effective_bits_per_source_element", "quantized_state_bytes", "unquantized_state_bytes")}
                       for p, m in manifests.items()}}


def image_grid(root, output, labels, cases, tile=384):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new("RGB", (len(labels) * tile, 40 + len(cases) * (tile + 30)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * tile + 10, 8), label, fill="black", font=font)
        for row, case in enumerate(cases):
            path = root / "quality" / label / case
            report = read(path / "report.json")
            with Image.open(path / "run-0.png") as original:
                if hashlib.sha256(original.tobytes()).hexdigest() != report["measurements"][0]["pixel_sha256"]:
                    raise ValueError(f"Image/report hash mismatch: {path}")
                image = original.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            y = 40 + row * (tile + 30)
            draw.text((column * tile + 10, y), case, fill="black", font=font)
            canvas.paste(image, (column * tile, y + 30))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--images", action="store_true")
    args = parser.parse_args()
    result = compare(args.root, args.bits)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = ["| Precision | Denoising s | Pipeline s | Peak allocated GiB | NMSE |", "|---|---:|---:|---:|---:|"]
    for label, row in result["benchmark"].items():
        score = "—" if label == "bf16" else f"{result['nmse'][label]['nmse']:.9g}"
        rows.append(f"| {label} | {row['denoising_seconds']:.3f} | {row['pipeline_seconds']:.3f} | {row['peak_allocated_bytes']/1024**3:.3f} | {score} |")
    text = "\n".join(rows) + f"\n\nBF16 self NMSE: {result['bf16_self_nmse']:.9g}\n"
    (args.output / "comparison.md").write_text(text)
    if args.images:
        image_grid(args.root, args.output / "multibit.png", ["bf16", *result["nmse"]], result["nmse_cases"])
        image_grid(args.root, args.output / "quality-4bit.png", ["bf16", "exl3-4bit"],
                   ["photo", "text", "composition", "edit", "multi_ref", "photo_2048"])
    print(text, end="")


if __name__ == "__main__":
    main()
