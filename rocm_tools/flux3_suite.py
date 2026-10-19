#!/usr/bin/env python3
"""Fixed FLUX.3 EXL3 calibration, generation, benchmarking and NMSE cases."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

CLI = Path(__file__).with_name("flux3.py")
CASES = Path(__file__).with_name("flux3_cases.json")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def invoke(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)


def run_case(args, case, *, precision="bf16", purpose="quality", capture=False, nmse=False):
    source_cases = {c["id"]: c for c in read(CASES)["calibration" if capture else "evaluation"]}
    refs = []
    for reference in case.get("reference_cases", []):
        parent = args.root / ("calibration-runs" if capture else "bf16") / reference
        if not (parent / "report.json").exists():
            run_case(args, source_cases[reference], capture=capture)
        refs.append(parent / "run-0.png")
    directory = args.root / ("nmse" if nmse else "benchmark" if purpose == "benchmark" else "calibration-runs" if capture else "") / precision / case["id"]
    if capture:
        directory = args.root / "calibration-runs" / case["id"]
    completed = directory / "report.json"
    expected_purpose = "nmse" if nmse else "calibration" if capture else purpose
    if completed.exists():
        report = read(completed)
        settings = report["settings"]
        if (report["source"] != read(CLI.with_name("flux3_sources.json"))["model"]
                or report["caption"] != case["caption"]
                or (settings["width"], settings["height"], settings["seed"]) != (case["width"], case["height"], case["seed"])
                or report["measurement_purpose"] != expected_purpose
                or settings["encoding_cache"] != bool(nmse or args.cached)):
            raise ValueError(f"Existing report has different inputs or purpose: {completed}")
        if precision.startswith("exl3-"):
            quant = report["nmse"]["candidate_quantization"] if nmse else report["quantization"]
            if quant["bits"] != int(precision.split("-")[1][:-3]):
                raise ValueError(f"Existing report uses another bit width: {completed}")
            if quant != read(args.root / precision / "quantization.json"):
                raise ValueError(f"Existing report uses another checkpoint: {completed}")
        if purpose == "benchmark" and [r["warmup"] for r in report["measurements"]] != [True, False, False, False]:
            raise ValueError(f"Existing benchmark does not have one warmup and three measurements: {completed}")
        print("REUSE", completed, flush=True)
        return report
    command = [sys.executable, CLI, "generate", "--model", args.model, "--output", directory,
               "--width", case["width"], "--height", case["height"], "--seed", case["seed"],
               "--purpose", "diagnostic" if nmse or capture else purpose, "--device", args.device]
    if refs:
        command += ["--images", *refs]
    if nmse or args.cached:
        encoding = args.root / "encoding" / case["id"]
        if not (encoding / "encoding.json").exists():
            encode = [sys.executable, CLI, "encode", "--model", args.model, "--output", encoding,
                      "--caption", case["caption"], "--device", args.device]
            if refs:
                encode += ["--images", *refs]
            invoke(encode, encoding / "encode.log")
        command += ["--encoding", encoding]
    else:
        command += ["--caption", case["caption"]]
    if capture:
        command += ["--capture", args.root / "calibration" / case["id"]]
    if precision.startswith("exl3-"):
        bits = int(precision.split("-")[1][:-3])
        command += ["--nmse-exl3" if nmse else "--exl3", args.root / f"exl3-{bits}bit"]
    elif nmse:
        command += ["--nmse-self"]
    if purpose == "benchmark":
        command += ["--warmup", "1", "--repeats", "3"]
    invoke(command, directory / "generate.log")
    report = read(completed)
    if any(not r["finite_final_latents"] or not r["finite_vae_output"] for r in report["measurements"]):
        raise ValueError("Nonfinite generation report")
    return report


def summarize(root):
    summary = {"quality": {}, "benchmark": {}, "nmse": {}}
    for report_path in sorted(root.rglob("report.json")):
        report = read(report_path)
        relative = report_path.relative_to(root).as_posix()
        purpose = report["measurement_purpose"]
        if purpose == "nmse":
            label = report_path.parent.parent.name
            group = summary["nmse"].setdefault(label, {"cases": [], "squared_error_sum": 0, "reference_squared_sum": 0})
            metric = report["nmse"]
            group["cases"].append({"case": report_path.parent.name, "nmse": metric["nmse"], "report": relative})
            group["squared_error_sum"] += metric["squared_error_sum"]
            group["reference_squared_sum"] += metric["reference_squared_sum"]
        elif purpose in ("quality", "benchmark"):
            rows = [r for r in report["measurements"] if not r["warmup"]]
            summary[purpose][relative] = {
                "gpu": report["gpu"], "precision": report["precision"], "runs": len(rows),
                "pipeline_seconds_median": statistics.median(r["pipeline_seconds"] for r in rows),
                "denoising_seconds_median": statistics.median(r["denoising_seconds"] for r in rows),
                "peak_allocated_bytes_max": max(r["peak_allocated_bytes"] for r in rows),
                "pixel_hashes": sorted({r["pixel_sha256"] for r in rows}),
            }
    for value in summary["nmse"].values():
        value["nmse"] = value["squared_error_sum"] / value["reference_squared_sum"]
    write(root / "results.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "convert", "evaluate", "benchmark", "nmse", "nmse-self", "summarize"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=range(1, 9), nargs="+", default=[4])
    parser.add_argument("--precision", choices=("bf16", "exl3"), default="exl3")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--cached", action="store_true", help="Cache only text encoding; reference VAE encoding stays in pipeline timing")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    frozen = args.root / "cases.json"
    if frozen.exists() and read(frozen) != read(CASES):
        raise ValueError("Fixed cases changed; use a new root")
    write(frozen, read(CASES))
    all_cases = read(CASES)
    if args.command == "calibrate":
        for case in all_cases["calibration"]:
            run_case(args, case, capture=True)
    elif args.command == "convert":
        captures = [args.root / "calibration" / c["id"] for c in all_cases["calibration"]]
        for bits in args.bits:
            checkpoint = args.root / f"exl3-{bits}bit"
            if (checkpoint / "quantization.json").exists():
                print("Checkpoint exists", checkpoint, flush=True)
                continue
            invoke([sys.executable, CLI, "convert", "--model", args.model, "--calibration", *captures,
                    "--bits", bits, "--output", checkpoint, "--device", args.device], args.root / f"convert-{bits}bit.log")
    elif args.command != "summarize":
        selected = args.cases or (["composition"] if args.command in ("benchmark", "nmse-self") else
                                  ["text", "composition", "multi_ref"] if args.command == "nmse" else
                                  [c["id"] for c in all_cases["evaluation"]])
        case_map = {c["id"]: c for c in all_cases["evaluation"]}
        unknown = set(selected) - set(case_map)
        if unknown:
            parser.error(f"Unknown case IDs: {sorted(unknown)}")
        for case_id in selected:
            labels = ["bf16"] if args.precision == "bf16" or args.command == "nmse-self" else [f"exl3-{b}bit" for b in args.bits]
            for label in labels:
                run_case(args, case_map[case_id], precision=label,
                         purpose="benchmark" if args.command == "benchmark" else "quality",
                         nmse=args.command in ("nmse", "nmse-self"))
    summarize(args.root)


if __name__ == "__main__":
    main()
