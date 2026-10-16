"""Run the authorized download -> quantize -> validate -> publish workflow."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pid", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--modules-per-process", type=int, default=8)
    args = parser.parse_args()
    if args.modules_per_process < 1:
        parser.error("--modules-per-process must be positive")
    if not args.resume and args.download_pid is None:
        parser.error("Specify --download-pid for a new job, or --resume")
    root = Path(__file__).resolve().parents[2]
    data = Path("/home/homelab1/datapool")
    source = data / "ai_models/safetensors/Hy-Memory-30B-A3B-BF16"
    run = data / "rocm-exl3-rdna2/runs/hy-memory-30b-a3b-4bit-base"
    output = data / "rocm-exl3-rdna2/models/Hy-Memory-30B-A3B-exl3-4.08bpw"
    installed = data / "ai_models/safetensors/Hy-Memory-30B-A3B-exl3-4.08bpw"
    run.mkdir(parents=True, exist_ok=True)
    state = {"started_at": datetime.now(timezone.utc).isoformat(), "stages": []}
    if args.resume:
        state = json.loads((run / "pipeline-status.json").read_text())
        state.pop("error", None)
        state["resumed_at"] = datetime.now(timezone.utc).isoformat()

    def status(stage, **extra):
        state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        (run / "pipeline-status.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({"stage": stage, **extra}), flush=True)

    def execute(stage, command, append=False):
        status(stage, command=command)
        start = time.monotonic()
        with (run / f"{stage}.log").open("a" if append else "w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["stages"].append({"name": stage, "wall_s": time.monotonic() - start})

    docker = ["docker", "exec", "-w", "/src", "-e", "PYTHONUNBUFFERED=1",
              "rocm-exl3-r9700-conv", "python"]
    source_c = "/models/safetensors/Hy-Memory-30B-A3B-BF16"
    run_c = "/work/runs/hy-memory-30b-a3b-4bit-base"
    output_c = "/work/models/Hy-Memory-30B-A3B-exl3-4.08bpw"
    try:
        if not args.resume:
            status("waiting_for_download")
            while True:
                try:
                    os.kill(args.download_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(15)
        index = json.loads((source / "model.safetensors.index.json").read_text())
        expected = set(index["weight_map"].values())
        assert len(expected) == 51 and all((source / name).is_file() for name in expected), "Download incomplete"
        if not args.resume:
            execute("prepare", docker + ["doc/hy_memory_quantization/prepare.py", source_c,
                                          "--output", run_c, "--check-shapes"])
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            (run / "conversion-engine-revision.txt").write_text(revision + "\n")
        # Recycle the process at persisted module boundaries. This bounds allocator
        # growth on large MoEs without changing the saved FP32 calibration states,
        # per-module random seeds, recipe or quantization code.
        checkpoint = run / "checkpoint/ckpt/job.json"
        fresh = not args.resume
        while True:
            next_idx = 0 if fresh else json.loads(checkpoint.read_text())["next_module_idx"]
            limit = next_idx + args.modules_per_process - 1
            if fresh:
                command = docker + ["convert.py", "-i", source_c, "-o", output_c,
                                    "-w", run_c + "/checkpoint", "-rcp", run_c + "/recipe-4bit-base.json",
                                    "-hb", "8", "-cb", "mul1", "-cr", "250", "-cc", "2048",
                                    "-cd", run_c + "/calibration.safetensors", "-d", "0", "-ss", "4096"]
            else:
                command = docker + ["convert.py", "-w", run_c + "/checkpoint", "-r"]
            command += ["-cpi", "1"]
            if limit < 50:  # Embedding + 48 blocks + norm + head; final process compiles too.
                command += ["--max_module", str(limit)]
            execute("convert", command, append=(args.resume or not fresh))
            if (output / "quantization_config.json").is_file():
                break
            new_idx = json.loads(checkpoint.read_text())["next_module_idx"]
            assert new_idx > next_idx, "Conversion did not advance its checkpoint"
            fresh = False
        execute("validate", docker + ["doc/hy_memory_quantization/validate.py", output_c,
                                       "--recipe", run_c + "/recipe-4bit-base.json",
                                       "--report", run_c + "/validation.json"])
        execute("publish", ["/home/homelab1/.local/share/pipx/venvs/huggingface-hub/bin/python",
                            "doc/hy_memory_quantization/publish.py", str(output),
                            "--source-dir", str(source), "--run-dir", str(run), "--publish"])
        # Make the same files available through the standard model directory without a copy.
        installed.mkdir(exist_ok=False)
        for path in output.iterdir():
            if path.is_file():
                os.link(path, installed / path.name)
        status("complete", installed=str(installed))
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
