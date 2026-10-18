#!/usr/bin/env python3
"""Run fixed calibration/evaluation cases and matched BF16/EXL3 benchmarks.

Each invocation uses a fresh process. Completed outputs are reused only when
their source, prompt, seed, shape, reference pixels and sampling settings match.
Run timing benchmarks with no other GPU workload.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm_tools.qwen_image21 import image_identity, images_from_paths, read_json, source

DRIVER = Path(__file__).with_name("qwen_image21.py")
CASES = Path(__file__).with_name("qwen_image21_cases.json")


def expected_settings(case, refs, variant, model):
    sigmas = read_json(model / "model_index.json").get("sample_sigmas")
    return {
        "prompt": case["prompt"], "seed": case["seed"], "width": case["width"],
        "height": case["height"], "image": image_identity(images_from_paths(refs)),
        "negative_prompt": None, "true_cfg_scale": 1.0, "use_kv_cache": True,
        "sample_sigmas": sigmas,
    }, len(sigmas) if sigmas else 40


def matching_report(report, case, refs, variant, model, precision, *, repeats=1, warmup=0,
                    checkpoint=None, offload=None, calibration=False):
    if report.get("nmse") is not None or report.get("measurement_purpose") == "nmse":
        return False
    expected, steps = expected_settings(case, refs, variant, model)
    if report.get("source") != source(variant) or report.get("precision") != precision:
        return False
    if repeats > 1:
        if report.get("measurement_purpose") != "benchmark":
            return False
        # A previously measured implementation must not silently pass as the
        # current benchmark after changing the CLI or EXL3 runtime.
        identity = report.get("implementation", {})
        if identity.get("cli_sha256") != hashlib.sha256(DRIVER.read_bytes()).hexdigest():
            return False
        for name in ("linear.py", "checkpoint.py"):
            file = DRIVER.resolve().parents[1] / "exllamav3/diffusion" / name
            if identity.get("exl3_diffusion_runtime_sha256", {}).get(name) != hashlib.sha256(file.read_bytes()).hexdigest():
                return False
    settings = report.get("settings", {})
    if any(settings.get(key) != value for key, value in expected.items()):
        return False
    if offload is not None and settings.get("offload") != offload:
        return False
    if bool(report.get("calibration_instrumented")) != calibration:
        return False
    if precision == "exl3" and report.get("quantization") != read_json(checkpoint / "quantization.json"):
        return False
    runs = report.get("measurements", [])
    for run in runs:
        times = [run.get("pipeline_seconds"), run.get("denoising_ms")]
        times += run.get("transformer_ms", [])
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in times):
            return False
        if len(run.get("transformer_ms", [])) != steps or len(run.get("timesteps", [])) != steps:
            return False
        if run.get("cache_modes") != ["extract"] + ["cached"] * (steps - 1):
            return False
    return (len(runs) == repeats + warmup
            and [run.get("warmup") for run in runs] == [True] * warmup + [False] * repeats
            and all(run.get("actual_steps") == steps and run.get("image_size") == [case["width"], case["height"]]
                    for run in runs))


def execute(arguments, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(DRIVER), *map(str, arguments)]
    print(json.dumps({"command": command, "log": str(log)}), flush=True)
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_generation(args, case, refs, output, *, precision="bf16", encoding=None,
                   capture=None, repeats=1, warmup=0, offload="model"):
    report_file = output / "report.json"
    checkpoint = args.root / f"{args.variant}-exl3-4bit"
    expected_offload = "none" if encoding else offload
    if report_file.exists():
        if matching_report(read_json(report_file), case, refs, args.variant, args.model, precision,
                           repeats=repeats, warmup=warmup, checkpoint=checkpoint,
                           offload=None if capture else expected_offload, calibration=bool(capture)):
            if not all((output / f"run-{i}.png").exists() for i in range(warmup + repeats)):
                raise ValueError(f"Missing PNG in completed run {output}")
            if capture and not (capture / "calibration.json").exists():
                raise ValueError(f"Missing calibration samples for completed run {output}")
            print(f"reuse {output}", flush=True)
            return
        raise ValueError(f"Existing report has different settings: {output}; use a new root or inspect it")
    arguments = ["generate", "--variant", args.variant, "--model", args.model, "--output", output,
                 "--width", case["width"], "--height", case["height"], "--seed", case["seed"],
                 "--warmup", warmup, "--repeats", repeats, "--offload", offload]
    arguments.extend(["--purpose", "benchmark" if repeats > 1 else "quality"])
    arguments.extend(["--encoding", encoding] if encoding else ["--prompt", case["prompt"]])
    if refs:
        arguments.extend(["--images", *refs])
    if capture:
        arguments.extend(["--capture", capture])
    if precision == "exl3":
        arguments.extend(["--exl3", checkpoint])
    execute(arguments, output.with_suffix(".log"))
    if not matching_report(read_json(report_file), case, refs, args.variant, args.model, precision,
                           repeats=repeats, warmup=warmup, checkpoint=checkpoint,
                           offload=expected_offload, calibration=bool(capture)):
        raise RuntimeError(f"Generated report failed settings verification: {report_file}")


def run_encoding(args, case, refs):
    directory = args.root / args.variant / "encodings" / case["id"]
    if (directory / "encoding.json").exists():
        record = read_json(directory / "encoding.json")
        if (record["source"] != source(args.variant)
                or record["prompts"] != {"positive": case["prompt"]}
                or record["original_images"] != image_identity(images_from_paths(refs))
                or record["output_resolution"] != 1024
                or not (directory / "embeddings.safetensors").exists()):
            raise ValueError(f"Mismatched encoding cache: {directory}")
    else:
        arguments = ["encode", "--variant", args.variant, "--model", args.model,
                     "--prompt", case["prompt"], "--output", directory]
        if refs:
            arguments.extend(["--images", *refs])
        execute(arguments, directory.with_suffix(".log"))
    return directory


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("calibrate", "convert", "evaluate", "benchmark", "nmse"))
    ap.add_argument("--variant", choices=("base", "turbo"), required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--precision", choices=("bf16", "exl3", "both"), default="both")
    ap.add_argument("--case", action="append", help="Optional evaluation case selection")
    ap.add_argument("--offload", choices=("model", "text-encoder"), default="model")
    ap.add_argument("--checkpoint", type=Path, help="NMSE candidate; defaults to the suite 4bit checkpoint")
    ap.add_argument("--nmse-self", action="store_true", help="NMSE BF16 repeat-forward control")
    ap.add_argument("--nmse-output", type=Path, help="Fresh directory for NMSE reports and summary")
    args = ap.parse_args()
    if args.nmse_self and args.checkpoint:
        ap.error("--nmse-self and --checkpoint are mutually exclusive")
    if args.phase != "nmse" and (args.nmse_self or args.checkpoint or args.nmse_output):
        ap.error("NMSE options require the nmse phase")
    cases = read_json(CASES)
    precisions = ("bf16", "exl3") if args.precision == "both" else (args.precision,)
    if args.phase == "calibrate":
        for case in cases["calibration"]:
            refs = [args.root / f"{args.variant}-{ref}" / "run-0.png" for ref in case.get("reference_cases", [])]
            capture = args.root / "calibration" / f"{args.variant}-{case['id'].split('-')[-1]}"
            run_generation(args, case, refs, args.root / f"{args.variant}-{case['id']}",
                           capture=capture, offload=args.offload)
    elif args.phase == "convert":
        target = args.root / f"{args.variant}-exl3-4bit"
        calibration = [args.root / "calibration" / f"{args.variant}-{c['id'].split('-')[-1]}" for c in cases["calibration"]]
        if (target / "quantization.json").exists():
            manifest = read_json(target / "quantization.json")
            if (manifest.get("source") != source(args.variant) or manifest.get("bits") != 4
                    or manifest.get("complete") is not True
                    or manifest.get("calibration") != [hashlib.sha256((p / "calibration.json").read_bytes()).hexdigest() for p in calibration]):
                raise ValueError("Completed conversion differs from this suite")
            print(f"reuse {target}", flush=True)
        else:
            execute(["convert", "--variant", args.variant, "--model", args.model,
                     "--calibration", *calibration, "--bits", 4, "--output", target],
                    args.root / f"{args.variant}-convert.log")
    else:
        selected = [c for c in cases["evaluation"] if not args.case or c["id"] in args.case]
        if args.case and set(args.case) != {c["id"] for c in selected}:
            ap.error("Unknown evaluation case")
        if args.phase == "benchmark":
            selected = [c for c in cases["evaluation"] if c["id"] == "composition"]
        nmse_reports = {}
        nmse_directory = args.nmse_output or args.root / args.variant / "nmse" / ("bf16-self" if args.nmse_self else "exl3")
        if args.phase == "nmse" and nmse_directory.exists():
            raise ValueError("NMSE output already exists; use a fresh --nmse-output to avoid mixing experiments")
        for case in selected:
            refs = [args.root / args.variant / "bf16" / ref / "run-0.png" for ref in case.get("reference_cases", [])]
            if not all(p.exists() for p in refs):
                raise ValueError("Generate the BF16 reference cases before dependent editing cases")
            encoding = run_encoding(args, case, refs) if args.phase in ("evaluate", "nmse") else None
            if args.phase == "nmse":
                directory = nmse_directory / case["id"]
                arguments = ["generate", "--variant", args.variant, "--model", args.model,
                             "--encoding", encoding, "--output", directory, "--purpose", "diagnostic",
                             "--width", case["width"], "--height", case["height"], "--seed", case["seed"]]
                arguments.extend(["--nmse-self"] if args.nmse_self else
                                 ["--nmse-exl3", args.checkpoint or args.root / f"{args.variant}-exl3-4bit"])
                if refs:
                    arguments.extend(["--images", *refs])
                execute(arguments, directory.with_suffix(".log"))
                report = read_json(directory / "report.json")
                nmse_reports[case["id"]] = report["nmse"]
                continue
            for precision in precisions:
                directory = (args.root / args.variant / precision / case["id"] if args.phase == "evaluate" else
                             args.root / args.variant / "benchmark" / precision)
                run_generation(args, case, refs, directory, precision=precision, encoding=encoding,
                               repeats=3 if args.phase == "benchmark" else 1,
                               warmup=1 if args.phase == "benchmark" else 0,
                               offload="text-encoder" if args.phase == "benchmark" else "model")

        if nmse_reports:
            from rocm_tools.qwen_image21 import write_json
            error = math.fsum(r["squared_error_sum"] for r in nmse_reports.values())
            signal = math.fsum(r["reference_squared_sum"] for r in nmse_reports.values())
            write_json(nmse_directory / "summary.json", {
                "source": source(args.variant), "nmse": error / signal,
                "squared_error_sum": error, "reference_squared_sum": signal,
                "aggregation": "global ratio of sums over all selected cases/steps/target elements; not mean case NMSE",
                "cases": nmse_reports,
            })
            print(json.dumps({"variant": args.variant, "nmse": error / signal}), flush=True)


if __name__ == "__main__":
    main()
