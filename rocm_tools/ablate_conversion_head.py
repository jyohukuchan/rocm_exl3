#!/usr/bin/env python3
"""Toggle fast_head_capture at a Qwen3.5 pre-norm checkpoint and rebuild.

The calibration-Hessian mode is inherited unchanged from the source run. Its
decoder weights/states and uncalibrated side-model tensors are copied, not
requantized. Only the final norm/head run through the normal converter. A save
hook stops after the new head tensors are on disk; normal compilation then
assembles a complete artifact for byte-level comparison.

This is an attribution probe, not a full-conversion timing benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from exllamav3.conversion import convert_model as conversion
from exllamav3.conversion.compile import compile_model
from exllamav3.ext import exllamav3_ext
from exllamav3.modules import Linear


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class HeadSaved(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-work", type=Path, required=True)
    parser.add_argument("--checkpoint", default="ckpt_old", choices=("ckpt", "ckpt_old"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fast-head", choices=("on", "off"), required=True)
    cli = parser.parse_args()
    source = cli.source_work.resolve()
    work = cli.work.resolve()
    output = cli.output.resolve()
    if work.exists() or output.exists():
        raise FileExistsError("Use fresh work/output directories; source artifacts are never overwritten")
    source_args = json.loads((source / "args.json").read_text())
    config_json = json.loads((Path(source_args["in_dir"]) / "config.json").read_text())
    if config_json.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("This checkpoint probe is validated only for dense Qwen3.5")
    checkpoint = source / cli.checkpoint
    job = json.loads((checkpoint / "job.json").read_text())
    expected_model = Path(source_args["out_dir"]) / "model.safetensors"
    expected_hash = sha(expected_model)
    source_modules = {p.name: sha(p) for p in sorted((source / "qtensors").glob("*.safetensors"))}
    checkpoint_hashes = {name: sha(checkpoint / name) for name in
                         ("job.json", "state.safetensors", "original_input_ids.safetensors")}

    work.mkdir(parents=True)
    shutil.copytree(source / "qtensors", work / "qtensors")
    shutil.copytree(checkpoint, work / "ckpt")
    saved_args = dict(source_args, work_dir=str(work), out_dir=str(output),
                      fast_head_capture=cli.fast_head == "on", max_module=None)
    saved_args.setdefault("hessian_fp16", False)
    (work / "args.json").write_text(json.dumps(saved_args, indent=2) + "\n")
    args, state, ok, error = conversion.prepare(conversion.parser.parse_args(["-w", str(work), "-r"]))
    if not ok:
        raise RuntimeError(error)
    if args["hessian_fp16"] != bool(source_args.get("hessian_fp16", False)):
        raise AssertionError("Hessian mode must not change after decoder calibration")

    models = []
    rewritten = []
    get_base = conversion.get_base_model
    save = conversion.save_tensor

    def capture_model(run_args):
        result = get_base(run_args)
        model = result[1]
        if not isinstance(model.modules[-1], Linear):
            raise ValueError("Expected a terminal Linear head")
        if job["next_module_idx"] != len(model.modules) - 2 or not model.modules[-2].key.endswith(".norm"):
            raise ValueError("Checkpoint must precede the final norm/head pair")
        models.append(result)
        return result

    def stop_after_head(tensor, filename, run_args):
        save(tensor, filename, run_args)
        if filename.startswith("qtensors/"):
            rewritten.append(Path(filename).name)
        if models and filename == f"qtensors/{models[0][1].modules[-1].key}.safetensors":
            raise HeadSaved()

    conversion.get_base_model = capture_model
    conversion.save_tensor = stop_after_head
    start = time.monotonic()
    try:
        conversion.main(args, state)
    except HeadSaved:
        print(" -- Ablation: terminal head saved; keeping copied side-model tensors", flush=True)
    else:
        raise RuntimeError("Did not stop at the expected head")
    finally:
        conversion.get_base_model = get_base
        conversion.save_tensor = save
    tail_seconds = time.monotonic() - start

    config, model, mtp, vision, tokenizer, _ = models[0]
    expected_rewrites = [m.key + ".safetensors" for m in model.modules[-2:]]
    if rewritten != expected_rewrites:
        raise AssertionError(f"Unexpected rewritten modules: {rewritten}")
    started_compile = time.monotonic()
    compile_model(args, model, config, tokenizer, mtp, vision)
    compile_seconds = time.monotonic() - started_compile
    module_hashes = {p.name: sha(p) for p in sorted((work / "qtensors").glob("*.safetensors"))}
    if module_hashes.keys() != source_modules.keys():
        raise AssertionError("Compiled module set differs from the source")
    differences = [name for name in module_hashes if module_hashes[name] != source_modules[name]]
    actual_hash = sha(output / "model.safetensors")
    config_equal = (output / "config.json").read_bytes() == (expected_model.parent / "config.json").read_bytes()
    result = {
        "source_work": str(source), "source_model": str(expected_model),
        "checkpoint": cli.checkpoint, "checkpoint_sha256": checkpoint_hashes,
        "next_module_idx": job["next_module_idx"], "source_fast_head_capture": bool(source_args.get("fast_head_capture", False)),
        "fast_head_capture": args["fast_head_capture"], "hessian_fp16": args["hessian_fp16"],
        "rewritten_modules": rewritten, "differing_modules": differences,
        "module_files_checked": len(module_hashes), "output_model": str(output / "model.safetensors"),
        "expected_model_sha256": expected_hash, "model_sha256": actual_hash,
        "model_bitwise_equal": actual_hash == expected_hash, "config_bitwise_equal": config_equal,
        "tail_conversion_seconds": tail_seconds, "compile_seconds": compile_seconds,
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_properties(0).gcnArchName,
        "native_sha256": sha(exllamav3_ext.__file__), "probe_sha256": sha(__file__),
        "scope": "Final norm/head recomputed with all calibration rows; decoder/uncalibrated side tensors reused. Not a full conversion benchmark.",
    }
    (work / "ablation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["model_bitwise_equal"] or not config_equal:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
