#!/usr/bin/env python3
"""Pinned Qwen-Image-2.1 BF16 baseline with separately cached text encoding.

The split avoids retaining an 8B text encoder alongside the DiT on a 32GB GPU.
Cached-prompt pipeline timing is explicitly separate from text encoding/load.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import types
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SOURCES = Path(__file__).with_name("qwen_image21_sources.json")
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source(variant):
    return read_json(SOURCES)["models"][variant]


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub", "safetensors")}


def configure_inference_allocator():
    # EXL3 normally enables expandable segments on import. Keep the same
    # allocator for pure Diffusers BF16 and EXL3 in matched comparisons.
    os.environ.setdefault("EXL3_EXPANDABLE_SEGMENTS", "0")
    if os.environ["EXL3_EXPANDABLE_SEGMENTS"] != "0":
        import exllamav3  # Explicit opt-in applies the initializer to both paths.
    return {name: os.environ.get(name) for name in (
        "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "EXL3_EXPANDABLE_SEGMENTS")}


def implementation_identity():
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

    return {"cli_sha256": SCRIPT_SHA256, "dependency_pins": read_json(SOURCES),
            "exl3_diffusion_runtime_sha256": {
                name: hashlib.sha256((Path(__file__).resolve().parents[1] / "exllamav3/diffusion" / name).read_bytes()).hexdigest()
                for name in ("linear.py", "checkpoint.py")
            },
            "runtime_source_sha256": {
                cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                for cls in (QwenImage21Pipeline, QwenImage21Transformer2DModel)
            }}


def download(args):
    from huggingface_hub import snapshot_download

    spec = source(args.variant)
    target = args.root.expanduser() / spec["repo_id"].split("/")[-1]
    snapshot_download(**spec, local_dir=target, max_workers=4,
                      allow_patterns=["*.json", "*.safetensors", "processor/*", "LICENSE", "README.md"])
    write_json(target / "exl3_source.json", spec)
    print(target)


def verify_source(args):
    record = args.model / "exl3_source.json"
    if not record.exists() or read_json(record) != source(args.variant):
        raise ValueError("Model source is not pinned; fetch with this tool's download command")


def images_from_paths(paths):
    from PIL import Image

    return [Image.open(p).convert("RGBA") for p in paths] if paths else None


def image_identity(images):
    return [{"size": list(im.size), "mode": im.mode,
             "sha256": hashlib.sha256(im.tobytes()).hexdigest()} for im in (images or [])]


def make_pipeline(model_path, text_encoder=None, transformer=None, vae=None):
    from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLProcessor

    index = read_json(model_path / "model_index.json")
    return QwenImage21Pipeline(
        processor=Qwen3VLProcessor.from_pretrained(model_path / "processor", local_files_only=True),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_path / "scheduler", local_files_only=True),
        text_encoder=text_encoder, transformer=transformer, vae=vae,
        sample_sigmas=index.get("sample_sigmas"),
    )


def preprocess_references(pipe, images, output_resolution):
    # Identical reference resize as the pinned pipeline before text encoding.
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    if images is None:
        return None
    output = []
    for im in images:
        width, height, _ = calculate_dimensions(output_resolution**2, im.width / im.height)
        output.append(pipe.image_processor.resize(im, width=width, height=height))
    return output


def encode(args):
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3VLForConditionalGeneration

    allocator = configure_inference_allocator()
    verify_source(args)
    torch.cuda.set_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model / "text_encoder", dtype=torch.bfloat16,
        device_map={"": args.device}, local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    pipe = make_pipeline(args.model, text_encoder=encoder)
    original_images = images_from_paths(args.images)
    images = preprocess_references(pipe, original_images, args.output_resolution)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    tensors, prompts, times = {}, {"positive": args.prompt}, {}
    if args.negative_prompt is not None:
        prompts["negative"] = args.negative_prompt
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for label, prompt in prompts.items():
            torch.cuda.synchronize()
            started = time.perf_counter()
            values = pipe._get_qwen_prompt_embeds(prompt, images, torch.device(args.device))
            torch.cuda.synchronize()
            times[label] = time.perf_counter() - started
            for key, value in zip(("embeds", "mask", "image_mask"), values):
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite text encoder output: {label}.{key}")
                tensors[f"{label}.{key}"] = value.contiguous().cpu()
    save_file(tensors, args.output / "embeddings.safetensors")
    write_json(args.output / "encoding.json", {
        "source": source(args.variant), "versions": versions(), "prompts": prompts,
        "implementation": implementation_identity(),
        "allocator_configuration": allocator,
        "embeddings_sha256": hashlib.sha256((args.output / "embeddings.safetensors").read_bytes()).hexdigest(),
        "original_images": image_identity(original_images), "encoded_images": image_identity(images),
        "output_resolution": args.output_resolution, "load_seconds": load_seconds,
        "encode_seconds": times, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "note": "Text encoding measured separately; not steady-state full pipeline latency",
    })
    print(json.dumps({"load_seconds": load_seconds, "encode_seconds": times}), flush=True)


def generate(args):
    import torch
    from PIL import ImageStat
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21
    from safetensors.torch import load_file

    allocator = configure_inference_allocator()
    verify_source(args)
    torch.cuda.set_device(args.device)
    initial_free, initial_total = torch.cuda.mem_get_info()
    if args.purpose == "benchmark" and initial_total - initial_free > 512 * 1024**2:
        raise RuntimeError("Benchmark GPU is already in use; wait for other GPU work to finish")
    cached = read_json(args.encoding / "encoding.json") if args.encoding else None
    if cached:
        if cached["source"] != source(args.variant):
            raise ValueError("Cached encoding belongs to a different model revision")
        if cached["versions"]["diffusers"] != versions()["diffusers"] or cached["versions"]["transformers"] != versions()["transformers"]:
            raise ValueError("Cached encoding library versions differ from current runtime")
        if cached.get("implementation", {}).get("runtime_source_sha256") != implementation_identity()["runtime_source_sha256"]:
            raise ValueError("Cached encoding Diffusers source differs from current runtime")
        if cached.get("embeddings_sha256") != hashlib.sha256((args.encoding / "embeddings.safetensors").read_bytes()).hexdigest():
            raise ValueError("Cached embedding checksum mismatch")
    images = images_from_paths(args.images)
    if cached and image_identity(images) != cached["original_images"]:
        raise ValueError("Reference images differ from cached text encoding")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    quantization = None
    if cached or args.exl3 or args.offload == "text-encoder":
        if args.exl3:
            from exllamav3.diffusion.checkpoint import load_transformer
            transformer, quantization = load_transformer(args.exl3, device=args.device,
                                                         workspace_bytes=args.workspace_mib * 1024**2)
            if quantization["source"] != source(args.variant):
                raise ValueError("EXL3 checkpoint belongs to a different source revision")
        else:
            transformer = QwenImage21Transformer2DModel.from_pretrained(
                args.model / "transformer", torch_dtype=torch.bfloat16,
                device_map={"": args.device}, local_files_only=True,
            ).eval()
        vae = AutoencoderKLQwenImage21.from_pretrained(
            args.model / "vae", torch_dtype=torch.bfloat16,
            device_map={"": args.device}, local_files_only=True,
        ).eval()
        pipe = make_pipeline(args.model, transformer=transformer, vae=vae)
        if not cached:
            from accelerate import cpu_offload
            from transformers import Qwen3VLForConditionalGeneration
            encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model / "text_encoder", dtype=torch.bfloat16,
                local_files_only=True, attn_implementation="sdpa",
            ).eval()
            # Keep DiT resident and stream only the high-precision text encoder
            # one module at a time, so it does not dictate the VRAM peak.
            cpu_offload(encoder, execution_device=torch.device(args.device), offload_buffers=True)
            pipe.register_modules(text_encoder=encoder)
    else:
        pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.enable_model_cpu_offload(device=args.device)
        transformer, vae = pipe.transformer, pipe.vae
    vae.enable_tiling()
    tensors = load_file(args.encoding / "embeddings.safetensors", device=args.device) if cached else None

    def cached_encode(self, prompt, image=None, device=None):
        text = prompt if isinstance(prompt, str) else prompt[0] if len(prompt) == 1 else None
        label = next((name for name, value in cached["prompts"].items() if value == text), None)
        if label is None or image_identity(image) != cached["encoded_images"]:
            raise ValueError("Pipeline requested prompt/reference data absent from the encoding cache")
        return tuple(tensors[f"{label}.{key}"].to(device) for key in ("embeds", "mask", "image_mask"))

    if cached:
        pipe._get_qwen_prompt_embeds = types.MethodType(cached_encode, pipe)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    if args.variant == "turbo" and args.steps is not None:
        raise ValueError("Turbo uses saved sample_sigmas; do not override --steps")
    kwargs = {
        "prompt": cached["prompts"]["positive"] if cached else args.prompt, "image": images,
        "negative_prompt": cached["prompts"].get("negative") if cached else args.negative_prompt,
        "true_cfg_scale": args.cfg, "width": args.width, "height": args.height,
        "output_resolution": cached["output_resolution"] if cached else args.output_resolution,
        "use_kv_cache": not args.no_kv_cache,
    }
    if args.variant == "base":
        kwargs["num_inference_steps"] = args.steps or 40
    if args.cfg > 1 and kwargs["negative_prompt"] is None:
        raise ValueError("CFG > 1 requires a negative prompt")

    expected_steps = len(pipe.config.sample_sigmas) if pipe.config.sample_sigmas else kwargs["num_inference_steps"]
    capture = None
    if args.capture:
        from exllamav3.diffusion.calibration import ActivationCapture
        steps = len(pipe.config.sample_sigmas) if pipe.config.sample_sigmas else kwargs["num_inference_steps"]
        capture = ActivationCapture(transformer, [0, steps // 2, steps - 1], args.capture_rows)

    nmse = None
    nmse_quantization = None
    if args.nmse_exl3 or args.nmse_self:
        from rocm_tools.qwen_image21_nmse import OutputNMSE
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
        if args.nmse_exl3:
            from exllamav3.diffusion.checkpoint import load_transformer
            candidate, nmse_quantization = load_transformer(
                args.nmse_exl3, device=args.device, workspace_bytes=args.workspace_mib * 1024**2)
            if nmse_quantization["source"] != source(args.variant):
                raise ValueError("NMSE candidate source revision differs from BF16")
        else:
            candidate = transformer
        nmse = OutputNMSE(candidate.forward, lambda: QwenImage21KVCache(len(transformer.transformer_blocks)))

    shape_stats, calls, step_ends = {}, [], []
    handles = []
    linear_types = (torch.nn.Linear,)
    if quantization:
        from exllamav3.diffusion.linear import EXL3DiffusionLinear
        linear_types += (EXL3DiffusionLinear,)
    for name, module in transformer.named_modules():
        if isinstance(module, linear_types):
            is_quantized = not isinstance(module, torch.nn.Linear)
            stat = shape_stats[name] = {"in": module.in_features, "out": module.out_features,
                                        "weight_dtype": "exl3" if is_quantized else str(module.weight.dtype),
                                        "inputs": {}}

            def record(mod, inputs, stat=stat):
                x = inputs[0]
                signature = f"{list(x.shape)}:{x.dtype}"
                stat["inputs"][signature] = stat["inputs"].get(signature, 0) + 1

            handles.append(module.register_forward_pre_hook(record))

    def start_transformer(module, inputs, kw):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        calls.append({"start": start, "timestep": kw["timestep"].detach(), "mode": kw.get("kv_cache_mode")})

    def end_transformer(module, inputs, output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        calls[-1]["end"] = end

    handles.extend((transformer.register_forward_pre_hook(start_transformer, with_kwargs=True),
                    transformer.register_forward_hook(end_transformer)))

    if nmse:
        handles.append(transformer.register_forward_hook(nmse.observe, with_kwargs=True))

    def step_callback(pipeline, index, timestep, data):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        step_ends.append(event)
        if capture:
            capture.step = index + 1
        if nmse:
            nmse.step = index + 1
        if index + 1 == expected_steps and not torch.isfinite(data["latents"]).all():
            raise RuntimeError("Nonfinite final diffusion latents")
        return data

    original_decode = vae.decode

    def checked_decode(*values, **decode_kwargs):
        decoded = original_decode(*values, **decode_kwargs)
        sample = decoded[0] if isinstance(decoded, tuple) else decoded.sample
        if not torch.isfinite(sample).all():
            raise RuntimeError("Nonfinite VAE output")
        return decoded

    vae.decode = checked_decode
    records = []
    try:
        with torch.inference_mode():
            for run in range(args.warmup + args.repeats):
                calls.clear()
                step_ends.clear()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                output = pipe(**kwargs, generator=torch.Generator("cpu").manual_seed(args.seed),
                              callback_on_step_end=step_callback).images[0]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                record = {
                    "warmup": run < args.warmup, "pipeline_seconds": elapsed,
                    "denoising_ms": calls[0]["start"].elapsed_time(step_ends[-1]),
                    "transformer_ms": [call["start"].elapsed_time(call["end"]) for call in calls],
                    "timesteps": [call["timestep"].float().cpu().tolist() for call in calls],
                    "cache_modes": [call["mode"] for call in calls],
                    "actual_steps": len(step_ends),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "image_mode": output.mode, "image_size": list(output.size),
                    "finite_final_latents": True, "finite_vae_output": True,
                    "rgb_stddev": ImageStat.Stat(output.convert("RGB")).stddev,
                    "pixel_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                }
                if output.mode == "RGBA":
                    histogram = output.getchannel("A").histogram()
                    pixels = output.width * output.height
                    record["alpha"] = {"extrema": list(output.getchannel("A").getextrema()),
                                       "transparent_fraction": histogram[0] / pixels,
                                       "opaque_fraction": histogram[255] / pixels}
                output.save(args.output / f"run-{run}.png")
                records.append(record)
                print(json.dumps({k: v for k, v in record.items() if k not in ("transformer_ms", "timesteps", "cache_modes")}), flush=True)
                del output
    finally:
        for handle in handles:
            handle.remove()
        if capture:
            capture.close()
        vae.decode = original_decode
    if capture:
        capture.save(args.capture, {"source": source(args.variant), "prompt": kwargs["prompt"],
                                    "negative_prompt": kwargs["negative_prompt"], "cfg": args.cfg,
                                    "seed": args.seed, "width": args.width, "height": args.height,
                                    "images": image_identity(images), "versions": versions(),
                                    "sample_sigmas": pipe.config.sample_sigmas,
                                    "actual_steps": records[-1]["actual_steps"],
                                    "use_kv_cache": not args.no_kv_cache})
    write_json(args.output / "report.json", {
        "source": source(args.variant), "versions": versions(),
        "implementation": implementation_identity(),
        "allocator_configuration": allocator,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(), "arch": torch.cuda.get_device_properties(args.device).gcnArchName,
        "precision": "exl3" if quantization else "bf16", "quantization": quantization,
        "text_encoding": cached, "model_load_seconds": load_seconds,
        "settings": {**kwargs, "image": image_identity(images), "seed": args.seed,
                     "sample_sigmas": pipe.config.sample_sigmas, "vae_tiling": True,
                     "reconstruction_workspace_bytes": args.workspace_mib * 1024**2 if quantization else None,
                     "offload": "none" if cached else "text-encoder" if quantization else args.offload},
        "measurements": records, "linear_shapes": shape_stats,
        "nmse": ({**nmse.report(), "candidate": "exl3" if args.nmse_exl3 else "bf16-self",
                  "quantization": nmse_quantization,
                  "measurement_code_sha256": hashlib.sha256(
                      Path(__file__).with_name("qwen_image21_nmse.py").read_bytes()).hexdigest()}
                 if nmse else None),
        "calibration_instrumented": bool(capture),
        "measurement_purpose": "nmse" if nmse else "calibration" if capture else args.purpose,
        "initial_gpu_free_total_bytes": [initial_free, initial_total],
        "timing_scope": ("NMSE instrumented: includes candidate execution; not a speed/VRAM benchmark." if nmse else "Cached-prompt pipeline; encoding and model loading are separate." if cached else
                         "Full pipeline including text encoding, model CPU offload transfers and VAE; excludes initial loading and PNG save."),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--variant", choices=("base", "turbo"), required=True)
    fetch.add_argument("--root", type=Path, default=Path("~/datapool/ai_models/safetensors"))
    fetch.set_defaults(fn=download)
    converter = sub.add_parser("convert")
    converter.add_argument("--variant", choices=("base", "turbo"), required=True)
    converter.add_argument("--model", type=Path, required=True)
    converter.add_argument("--calibration", type=Path, nargs="+", required=True)
    converter.add_argument("--output", type=Path, required=True)
    converter.add_argument("--bits", type=int, choices=range(1, 9), default=4)
    converter.add_argument("--device", default="cuda:0")

    def convert(args):
        import torch
        from exllamav3.diffusion.checkpoint import convert_transformer
        verify_source(args)
        torch.cuda.set_device(args.device)
        convert_transformer(args.model, args.output, args.calibration, bits=args.bits, device=args.device)

    converter.set_defaults(fn=convert)
    for name, fn in (("encode", encode), ("generate", generate)):
        parser = sub.add_parser(name)
        parser.add_argument("--variant", choices=("base", "turbo"), required=True)
        parser.add_argument("--model", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--images", type=Path, nargs="*", default=[])
        parser.set_defaults(fn=fn)
        if name == "encode":
            parser.add_argument("--prompt", required=True)
            parser.add_argument("--negative-prompt")
            parser.add_argument("--output-resolution", type=int, default=1024)
        else:
            inputs = parser.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--encoding", type=Path)
            inputs.add_argument("--prompt")
            parser.add_argument("--negative-prompt")
            parser.add_argument("--output-resolution", type=int, default=1024)
            parser.add_argument("--width", type=int, default=1024)
            parser.add_argument("--height", type=int, default=1024)
            parser.add_argument("--steps", type=int)
            parser.add_argument("--cfg", type=float, default=1.0)
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--no-kv-cache", action="store_true")
            parser.add_argument("--warmup", type=int, default=0)
            parser.add_argument("--repeats", type=int, default=1)
            parser.add_argument("--capture", type=Path, help="Save timestep activation samples; timings are then diagnostic only")
            parser.add_argument("--capture-rows", type=int, default=128)
            nmse_options = parser.add_mutually_exclusive_group()
            nmse_options.add_argument("--nmse-exl3", type=Path, help="Compare this checkpoint on the BF16 trajectory; diagnostic only")
            nmse_options.add_argument("--nmse-self", action="store_true", help="BF16 repeat-forward numerical floor on the same trajectory")
            parser.add_argument("--exl3", type=Path, help="EXL3 DiT checkpoint; DiT stays GPU resident")
            parser.add_argument("--offload", choices=("model", "text-encoder"), default="model",
                                help="For BF16: model-level offload or stream only text encoder. EXL3 always uses text-encoder offload unless embeddings are cached.")
            parser.add_argument("--workspace-mib", type=int, default=256)
            parser.add_argument("--purpose", choices=("diagnostic", "quality", "benchmark"), default="diagnostic")
    args = ap.parse_args()
    if args.command == "generate" and (args.repeats < 1 or args.warmup < 0):
        ap.error("repeats must be positive and warmup nonnegative")
    if args.command == "generate" and args.capture and (args.warmup != 0 or args.repeats != 1):
        ap.error("calibration capture requires warmup=0 and repeats=1; benchmark separately")
    if args.command == "generate" and (args.nmse_exl3 or args.nmse_self):
        if (args.exl3 or args.capture or not args.encoding or args.warmup != 0
                or args.repeats != 1 or args.purpose != "diagnostic" or args.cfg != 1):
            ap.error("NMSE requires cached encoding, BF16 teacher, CFG=1, warmup=0, repeats=1, diagnostic purpose, no calibration")
    args.fn(args)


if __name__ == "__main__":
    main()
