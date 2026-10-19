#!/usr/bin/env python3
"""Pinned FLUX.3 Image Klein: BF16/EXL3 generation, calibration and output NMSE."""

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SOURCES = Path(__file__).with_name("flux3_sources.json")
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub", "safetensors")}


def implementation():
    from diffusers import Flux3Transformer2DModel
    from diffusers.modular_pipelines.flux3 import before_denoise, encoders, denoise, decoders, prompt_upsampler

    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), SOURCES, Path(inspect.getfile(Flux3Transformer2DModel))]
    files += [Path(m.__file__) for m in (before_denoise, encoders, denoise, decoders, prompt_upsampler)]
    files += list((root / "exllamav3/diffusion").glob("*.py"))
    return {p.name: SCRIPT_SHA256 if p.resolve() == Path(__file__).resolve() else digest(p) for p in files}


def source():
    return read_json(SOURCES)["model"]


def verify_source(path, component="model"):
    record = Path(path) / "exl3_source.json"
    if not record.exists() or read_json(record) != read_json(SOURCES)[component]:
        raise ValueError(f"{component} source is not pinned; download with this CLI")


def load_images(paths):
    from PIL import Image

    if len(paths) > 10:
        raise ValueError("FLUX.3 takes at most ten reference images")
    return [Image.open(p).convert("RGB") for p in paths]


def image_identity(images):
    return [{"size": list(im.size), "mode": im.mode,
             "pixel_sha256": hashlib.sha256(im.tobytes()).hexdigest()} for im in images]


def download(args):
    from huggingface_hub import snapshot_download

    spec = read_json(SOURCES)[args.component]
    args.output.mkdir(parents=True, exist_ok=True)
    patterns = ["*.json", "*.jinja", "README.md", "LICENSE*", "NOTICE*", "model.safetensors"]
    if args.component == "model":
        patterns = ["config.json", "manifest.json", "modular_model_index.json", "README.md", "LICENSE.md",
                    "NOTICE.md", "text_encoder/*", "vae/*"]
        if not args.components_only:
            patterns.append("transformer/*")
    snapshot_download(**spec, local_dir=args.output, allow_patterns=patterns, max_workers=4)
    write_json(args.output / "exl3_source.json", spec)


def configure(device):
    os.environ.setdefault("EXL3_EXPANDABLE_SEGMENTS", "0")
    import torch

    torch.cuda.set_device(device)
    if torch.version.hip and torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx10"):
        os.environ["USE_HUB_KERNELS"] = "NO"


def configure_upsampler_kernels():
    """FLA's BF16 dot intrinsic cannot compile for the tested gfx1030 GPU."""
    import torch

    if not torch.version.hip:
        return "transformers-default"
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if not arch.startswith("gfx10"):
        return "transformers-default"
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    # Transformers exposes its readable Torch fallback through functools.wraps.
    # Disable just these optional kernel wrappers, without modifying site-packages.
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        setattr(module, name, inspect.unwrap(getattr(module, name)))
    return "transformers-torch-reference-gfx10"


def upsample_caption(model, prompt, images, boxes=None):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from diffusers.modular_pipelines.flux3.prompt_upsampler import prepare_references, upsample_prompt

    verify_source(model, "upsampler")
    kernel_policy = configure_upsampler_kernels()
    torch.cuda.reset_peak_memory_stats()
    loaded = time.perf_counter()
    encoder = Qwen3_5ForConditionalGeneration.from_pretrained(
        model, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()},
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - loaded
    start = time.perf_counter()
    with torch.inference_mode():
        caption = upsample_prompt(encoder, processor, prompt,
                                  images=prepare_references(images) if images else [], boxes=boxes)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    del encoder, processor
    gc.collect()
    torch.cuda.empty_cache()
    return caption, {"source": read_json(SOURCES)["upsampler"], "request": prompt, "kernel_policy": kernel_policy,
                     "boxes": boxes, "seconds": seconds, "model_load_seconds": load_seconds,
                     "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                     "caption": caption}


def upsample(args):
    configure(args.device)
    images = load_images(args.images)
    caption, record = upsample_caption(args.model, args.prompt, images,
                                        read_json(args.boxes) if args.boxes else None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "caption.txt").write_text(caption + "\n")
    write_json(args.output / "caption.json", {**record, "images": image_identity(images), "versions": versions()})
    print(caption, flush=True)


def caption_input(args, images):
    if args.caption:
        return args.caption, {"origin": "provided expanded caption"}
    if args.caption_file:
        caption = args.caption_file.read_text().strip()
        if not caption:
            raise ValueError("Empty caption file")
        # If it came from this CLI's upsampler, enforce the same reference images.
        record = args.caption_file.with_name("caption.json")
        if record.exists() and read_json(record).get("images") != image_identity(images):
            raise ValueError("Caption was expanded for different reference images")
        return caption, {"origin": "caption file", "sha256": digest(args.caption_file)}
    if args.prompt:
        if not args.upsampler:
            raise ValueError("--prompt needs --upsampler; use --caption/--caption-file for an expanded scene caption")
        return upsample_caption(args.upsampler, args.prompt, images,
                                read_json(args.boxes) if args.boxes else None)
    raise ValueError("Provide a prompt, expanded caption or encoding cache")


def load_encoder(model, device):
    import torch
    from accelerate import cpu_offload
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        model / "text_encoder", dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa",
    ).eval()
    cpu_offload(encoder, execution_device=torch.device(device), offload_buffers=True)
    processor = AutoProcessor.from_pretrained(model / "text_encoder", local_files_only=True)
    return encoder, processor


def encode_caption(encoder, processor, caption, device):
    from diffusers.modular_pipelines.flux3.encoders import Flux3DistilledTextEncoderStep

    spec = read_json(SOURCES)["text_encoder"]
    return Flux3DistilledTextEncoderStep._get_qwen3_vl_prompt_embeds(
        encoder, processor, caption, device=device, max_sequence_length=spec["max_tokens"],
        pad_multiple=spec["pad_multiple"], hidden_states_layers=tuple(spec["layers"]),
    )


def encoding_identity():
    from diffusers.modular_pipelines.flux3 import encoders

    return {"recipe": digest(SOURCES), "diffusers_encoder": digest(encoders.__file__),
            "encode_function": hashlib.sha256(inspect.getsource(encode_caption).encode()).hexdigest()}


def encode(args):
    import torch
    from safetensors.torch import save_file

    configure(args.device)
    verify_source(args.model)
    images = load_images(args.images)
    caption, expansion = caption_input(args, images)
    encoder, processor = load_encoder(args.model, args.device)
    start = time.perf_counter()
    with torch.inference_mode():
        embedding = encode_caption(encoder, processor, caption, args.device)
    torch.cuda.synchronize()
    if not torch.isfinite(embedding).all():
        raise ValueError("Nonfinite text encoder output")
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({"prompt_embeds": embedding.cpu().contiguous()}, args.output / "embeddings.safetensors")
    write_json(args.output / "encoding.json", {
        "source": source(), "caption": caption, "expansion": expansion, "images": image_identity(images),
        "seconds": time.perf_counter() - start, "versions": versions(), "implementation": implementation(),
        "encoding_implementation": encoding_identity(),
        "embeddings_sha256": digest(args.output / "embeddings.safetensors"),
    })


def make_pipeline(transformer, vae, device):
    from diffusers import Flux3DistilledModularPipeline
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.flux3.modular_blocks_flux3_distilled import Flux3DistilledAutoCoreDenoiseStep
    from diffusers.modular_pipelines.flux3.decoders import Flux3DecodeStep

    blocks = SequentialPipelineBlocks.from_blocks_dict({
        "denoise": Flux3DistilledAutoCoreDenoiseStep(), "decode": Flux3DecodeStep(),
    })
    pipe = Flux3DistilledModularPipeline(blocks=blocks)
    pipe.update_components(transformer=transformer, vae=vae)
    return pipe.to(device)


def encode_references(vae, images, device):
    if not images:
        return None
    from diffusers.modular_pipelines.flux3.modular_blocks_flux3 import Flux3VaeEncoderSequentialStep

    pipe = Flux3VaeEncoderSequentialStep().init_pipeline()
    pipe.update_components(vae=vae)
    pipe.to(device)
    return pipe(image=images, output="image_cond_latents")


def generate(args):
    import torch
    from PIL import ImageStat
    from diffusers import Flux3Transformer2DModel, AutoencoderKLFlux2
    from safetensors.torch import load_file
    from exllamav3.diffusion.calibration import ActivationCapture
    from exllamav3.diffusion.checkpoint import load_transformer
    from exllamav3.diffusion.linear import EXL3DiffusionLinear
    from exllamav3.diffusion.metrics import PredictionNMSE
    from exllamav3.diffusion.flux3_attention import configure_flux3_attention

    configure(args.device)
    verify_source(args.model)
    initial_free, initial_total = torch.cuda.mem_get_info()
    if args.purpose == "benchmark" and initial_total - initial_free > 512 * 1024**2:
        raise RuntimeError("Benchmark GPU is already in use")
    if (args.output / "report.json").exists():
        raise FileExistsError("Completed output already exists; use a new directory")
    images = load_images(args.images)
    cached = read_json(args.encoding / "encoding.json") if args.encoding else None
    if cached:
        if cached["source"] != source() or cached["images"] != image_identity(images):
            raise ValueError("Encoding cache source or reference images differ")
        if cached["versions"] != versions() or cached.get("encoding_implementation") != encoding_identity():
            raise ValueError("Encoding cache runtime differs")
        if cached["embeddings_sha256"] != digest(args.encoding / "embeddings.safetensors"):
            raise ValueError("Encoding cache checksum mismatch")
        caption, expansion = cached["caption"], cached["expansion"]
    else:
        caption, expansion = caption_input(args, images)
    started = time.perf_counter()
    quantization = None
    if args.exl3:
        transformer, quantization = load_transformer(args.exl3, device=args.device,
                                                     workspace_bytes=args.workspace_mib * 1024**2)
        if quantization["source"] != source() or quantization.get("transformer_type") != "Flux3Transformer2DModel":
            raise ValueError("EXL3 checkpoint belongs to another source model")
    else:
        transformer = Flux3Transformer2DModel.from_pretrained(
            args.model / "transformer", torch_dtype=torch.bfloat16,
            device_map={"": args.device}, local_files_only=True,
        ).eval()
    query_chunk_size = configure_flux3_attention(transformer, args.device, args.attention_query_chunk)
    # The reference VAE is FP32, including batch-normalization statistics.
    vae = AutoencoderKLFlux2.from_pretrained(
        args.model / "vae", torch_dtype=torch.float32, device_map={"": args.device}, local_files_only=True,
    ).eval()
    vae.enable_tiling()
    original_decode = vae.decode
    def checked_decode(*inputs, **kwargs):
        value = original_decode(*inputs, **kwargs)
        sample = value[0] if isinstance(value, tuple) else value.sample
        if not torch.isfinite(sample).all():
            raise RuntimeError("Nonfinite VAE output")
        return value
    vae.decode = checked_decode
    pipe = make_pipeline(transformer, vae, args.device)
    encoder, processor = (None, None) if cached else load_encoder(args.model, args.device)
    embedding = load_file(args.encoding / "embeddings.safetensors", device=args.device)["prompt_embeds"] if cached else None
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    candidate, candidate_quantization, nmse = None, None, None
    if args.nmse_exl3:
        candidate, candidate_quantization = load_transformer(args.nmse_exl3, device=args.device,
                                                             workspace_bytes=args.workspace_mib * 1024**2)
        if (candidate_quantization["source"] != source() or
                candidate_quantization.get("transformer_type") != "Flux3Transformer2DModel"):
            raise ValueError("NMSE candidate belongs to another source model")
        configure_flux3_attention(candidate, args.device, query_chunk_size)
        nmse = PredictionNMSE(candidate.forward)
    elif args.nmse_self:
        nmse = PredictionNMSE(transformer.forward)
    capture = ActivationCapture(transformer, [0, 8, 15], args.capture_rows, allow_uncalled=True) if args.capture else None
    stats, candidate_stats, calls, handles = {}, {}, [], []
    graphs = [(transformer, stats)] + ([(candidate, candidate_stats)] if candidate is not None else [])
    for graph, inventory in graphs:
        for name, module in graph.named_modules():
            if isinstance(module, (torch.nn.Linear, EXL3DiffusionLinear)):
                entry = inventory[name] = {"in": module.in_features, "out": module.out_features,
                                          "precision": "exl3" if isinstance(module, EXL3DiffusionLinear) else str(module.weight.dtype),
                                          "calls": 0}
                def record(mod, inputs, entry=entry):
                    entry["calls"] += 1
                handles.append(module.register_forward_pre_hook(record))

    def before(mod, inputs, kwargs):
        if capture:
            capture.step = len(calls)
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        calls.append({"start": event, "timestep": kwargs["timestep"].float().cpu().tolist(),
                      "target_timestep": kwargs["target_timestep"].float().cpu().tolist()})

    def after(mod, inputs, kwargs, output):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        calls[-1]["end"] = event

    handles.append(transformer.register_forward_pre_hook(before, with_kwargs=True))
    handles.append(transformer.register_forward_hook(after, with_kwargs=True))
    if nmse:
        handles.append(transformer.register_forward_hook(nmse.observe, with_kwargs=True))
    args.output.mkdir(parents=True, exist_ok=True)
    measurements = []
    try:
        with torch.inference_mode():
            for run in range(args.warmup + args.repeats):
                calls.clear()
                for entry in (*stats.values(), *candidate_stats.values()):
                    entry["calls"] = 0
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                current_embedding = embedding if cached else encode_caption(encoder, processor, caption, args.device)
                if not torch.isfinite(current_embedding).all():
                    raise RuntimeError("Nonfinite text encoder output")
                refs = encode_references(vae, images, args.device)
                if refs and not all(torch.isfinite(value).all() for value in refs):
                    raise RuntimeError("Nonfinite reference VAE latents")
                inputs = {"prompt_embeds": current_embedding, "width": args.width, "height": args.height,
                          "num_inference_steps": 16, "generator": torch.Generator("cpu").manual_seed(args.seed)}
                if refs:
                    inputs["image_cond_latents"] = refs
                result = pipe(**inputs, output=["images", "latents"])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                if not torch.isfinite(result["latents"]).all():
                    raise RuntimeError("Nonfinite final diffusion latents")
                image = result["images"][0]
                if len(calls) != 16:
                    raise RuntimeError(f"Expected the official 16 steps, got {len(calls)}")
                for inventory in (stats, candidate_stats):
                    for name, entry in inventory.items():
                        if entry["precision"] == "exl3":
                            expected = 0 if name.startswith("mode_blocks.image_cond.") and not images else 16
                            if entry["calls"] != expected:
                                raise RuntimeError(f"Incorrect EXL3 execution count: {name}")
                # Decode is checked before PIL conversion by a VAE hook below.
                measurement = {
                    "warmup": run < args.warmup, "pipeline_seconds": seconds,
                    "denoising_seconds": calls[0]["start"].elapsed_time(calls[-1]["end"]) / 1000,
                    "transformer_ms": [c["start"].elapsed_time(c["end"]) for c in calls],
                    "timesteps": [c["timestep"] for c in calls], "target_timesteps": [c["target_timestep"] for c in calls],
                    "actual_steps": len(calls), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "finite_final_latents": True,
                    "finite_vae_output": True, "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "image_size": list(image.size), "image_mode": image.mode,
                    "rgb_stddev": ImageStat.Stat(image.convert("RGB")).stddev,
                }
                image.save(args.output / f"run-{run}.png")
                measurements.append(measurement)
                print(json.dumps({k: v for k, v in measurement.items() if k not in
                                  ("transformer_ms", "timesteps", "target_timesteps")}), flush=True)
    finally:
        vae.decode = original_decode
        for handle in handles:
            handle.remove()
        if capture:
            capture.close()
    if capture:
        capture.save(args.capture, {"source": source(), "caption": caption, "images": image_identity(images),
                                   "seed": args.seed, "width": args.width, "height": args.height,
                                   "steps": 16, "versions": versions()})
    properties = torch.cuda.get_device_properties(args.device)
    write_json(args.output / "report.json", {
        "source": source(), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "initial_gpu_free_total_bytes": [initial_free, initial_total],
        "versions": versions(), "implementation": implementation(), "gpu": torch.cuda.get_device_name(),
        "arch": getattr(properties, "gcnArchName", f"sm_{properties.major}{properties.minor}"),
        "precision": "exl3" if quantization else "bf16", "quantization": quantization,
        "caption": caption, "expansion": expansion, "images": image_identity(images),
        "settings": {"width": args.width, "height": args.height, "seed": args.seed, "steps": 16,
                     "vae_dtype": "float32", "vae_tiling": True, "text_encoder_offload": "module",
                     "vae_tile_sample_min_size": vae.tile_sample_min_size,
                     "vae_tile_overlap_factor": vae.tile_overlap_factor,
                     "attention_query_chunk": query_chunk_size,
                     "encoding_cache": bool(cached), "workspace_mib": args.workspace_mib if quantization else None},
        "model_load_seconds": load_seconds, "measurements": measurements, "linear_layers": stats,
        "candidate_linear_layers": candidate_stats,
        "nmse": {**nmse.report(), "candidate": "bf16-self" if args.nmse_self else "exl3",
                 "candidate_quantization": candidate_quantization} if nmse else None,
        "measurement_purpose": "nmse" if nmse else "calibration" if capture else args.purpose,
        "timing_scope": "Includes VAE, references and text encoding unless cached; excludes loading, PUP and PNG save.",
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--component", choices=("model", "upsampler"), default="model")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--components-only", action="store_true")
    fetch.set_defaults(fn=download)
    convert = sub.add_parser("convert")
    convert.add_argument("--model", type=Path, required=True)
    convert.add_argument("--calibration", type=Path, nargs="+", required=True)
    convert.add_argument("--bits", type=int, choices=range(1, 9), default=4)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--device", default="cuda:0")
    def convert_model(args):
        from exllamav3.diffusion.checkpoint import convert_transformer
        configure(args.device)
        verify_source(args.model)
        convert_transformer(args.model, args.output, args.calibration, bits=args.bits, device=args.device)
    convert.set_defaults(fn=convert_model)
    for name, fn in (("upsample", upsample), ("encode", encode), ("generate", generate)):
        parser = sub.add_parser(name)
        parser.add_argument("--model", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--images", type=Path, nargs="*", default=[])
        parser.add_argument("--boxes", type=Path)
        inputs = parser.add_mutually_exclusive_group(required=True)
        inputs.add_argument("--prompt")
        if name != "upsample":
            inputs.add_argument("--caption", help="Full scene caption already expanded by an LLM")
            inputs.add_argument("--caption-file", type=Path)
            parser.add_argument("--upsampler", type=Path)
        parser.set_defaults(fn=fn)
        if name == "generate":
            inputs.add_argument("--encoding", type=Path)
            parser.add_argument("--exl3", type=Path)
            parser.add_argument("--width", type=int, default=1024)
            parser.add_argument("--height", type=int, default=1024)
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--workspace-mib", type=int, default=256)
            parser.add_argument("--attention-query-chunk", type=int,
                                help="Query slice size; default 1024 on gfx10, native elsewhere. 0 selects native SDPA.")
            parser.add_argument("--warmup", type=int, default=0)
            parser.add_argument("--repeats", type=int, default=1)
            parser.add_argument("--capture", type=Path)
            parser.add_argument("--capture-rows", type=int, default=128)
            nmse = parser.add_mutually_exclusive_group()
            nmse.add_argument("--nmse-exl3", type=Path)
            nmse.add_argument("--nmse-self", action="store_true")
            parser.add_argument("--purpose", choices=("diagnostic", "quality", "benchmark"), default="diagnostic")
    args = ap.parse_args()
    if args.command in ("encode", "generate") and args.boxes and not args.prompt:
        ap.error("--boxes needs --prompt and --upsampler; expanded captions must include their box table")
    if args.command == "generate":
        if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
            ap.error("width and height must be positive multiples of 16")
        if args.repeats < 1 or args.warmup < 0:
            ap.error("repeats must be positive; warmup must be nonnegative")
        if args.capture and (args.warmup != 0 or args.repeats != 1):
            ap.error("calibration requires warmup=0, repeats=1")
        if args.nmse_exl3 or args.nmse_self:
            if args.exl3 or args.capture or not args.encoding or args.warmup or args.repeats != 1 or args.purpose != "diagnostic":
                ap.error("NMSE requires cached encoding, BF16 teacher, one diagnostic run, no calibration")
    if args.command != "download":
        configure(args.device)
    args.fn(args)


if __name__ == "__main__":
    main()
