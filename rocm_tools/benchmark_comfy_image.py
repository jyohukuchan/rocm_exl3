"""Benchmark native ComfyUI BF16 implementations using pinned local source weights.

No server, UI, node-result cache, torch.compile, or approximate denoiser cache.
The sampler schedule is imported from the source recipe, not ComfyUI's defaults.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
COMFY_REV = '8f04fd6ecf4a548937971e840af2a985529fab76'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def load_shards(folder):
    from safetensors.torch import load_file
    result = {}
    for path in sorted(folder.glob('*.safetensors')):
        part = load_file(path)
        if set(result) & set(part):
            raise ValueError('Duplicate checkpoint tensor')
        result.update(part)
    if not result:
        raise ValueError(f'No weights in {folder}')
    return result


def flux3_native(state):
    """Invert the pinned official Diffusers conversion; verify with its forward map."""
    import torch
    from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers
    exact = {
        'timestep_embedder.linear_1.weight': 'time_in.in_layer.weight',
        'timestep_embedder.linear_2.weight': 'time_in.out_layer.weight',
        'target_timestep_embedder.linear_1.weight': 'ref_time_in.in_layer.weight',
        'target_timestep_embedder.linear_2.weight': 'ref_time_in.out_layer.weight',
        'proj_out.weight': 'final_layer.image.linear.weight',
        'norm_out.linear.weight': 'final_layer.image.adaLN_modulation.1.weight',
    }
    prefixes = {'x_embedders.': 'emb_in.', 'context_embedder.': 'txt_in.',
                'mode_blocks.': 'content_mode_blocks.', 'single_transformer_blocks.': 'single_blocks.'}
    suffixes = {'attn.to_q.weight': 'q_proj.weight', 'attn.to_k.weight': 'k_proj.weight',
                'attn.to_v.weight': 'v_proj.weight', 'attn.to_out.0.weight': 'attn_out.weight',
                'attn.norm_q.weight': 'norm.query_norm.scale', 'attn.norm_k.weight': 'norm.key_norm.scale',
                'ff.linear_in.weight': 'mlp_in.weight', 'ff.linear_out.weight': 'mlp_out.weight'}
    native = {}
    for key, value in state.items():
        if key in exact:
            name = exact[key]
            if key == 'norm_out.linear.weight':
                a, b = value.chunk(2, dim=0)
                value = torch.cat([b, a])
        else:
            name = key
            for old, new in prefixes.items():
                if name.startswith(old):
                    name = new + name[len(old):]
                    break
            name = name.replace('.linear.weight', '.lin.weight')
            for old, new in suffixes.items():
                if name.endswith(old):
                    name = name[:-len(old)] + new
                    break
        assert name not in native
        native[name] = value
    restored = convert_flux3_transformer_checkpoint_to_diffusers(dict(native))
    assert restored.keys() == state.keys()
    assert all(torch.equal(value, restored[k]) for k, value in state.items())
    return native


def qwen_native(state):
    """Use ComfyUI's fused SwiGLU format without changing any source value."""
    import torch
    from diffusers.loaders.single_file_utils import convert_qwen_image21_transformer_checkpoint_to_diffusers
    native = dict(state)
    for key in list(native):
        if key.endswith('img_mlp.gate_layer.weight'):
            up = key.replace('gate_layer', 'proj')
            native[key.replace('gate_layer', 'gate_up')] = torch.cat([native.pop(key), native.pop(up)])
    restored = convert_qwen_image21_transformer_checkpoint_to_diffusers(dict(native))
    assert restored.keys() == state.keys()
    assert all(torch.equal(value, restored[k]) for k, value in state.items())
    return native


def schedule(source, flux, turbo):
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    if flux:
        scheduler = FlowMatchEulerDiscreteScheduler(shift=6.93)
        scheduler.set_timesteps(sigmas=np.linspace(1, 1 / 16, 16))
    else:
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(source / 'scheduler/scheduler_config.json'))
        mu = calculate_shift(1024, scheduler.config.base_image_seq_len, scheduler.config.max_image_seq_len,
                             scheduler.config.base_shift, scheduler.config.max_shift)
        sigmas = read(source / 'model_index.json')['sample_sigmas'] if turbo else np.linspace(1, 1 / 40, 40)
        scheduler.set_timesteps(sigmas=sigmas, mu=mu)
    return scheduler.sigmas.cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', choices=('qwen-base', 'qwen-turbo', 'flux3'), required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--comfy', type=Path, required=True)
    ap.add_argument('--vae-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    git = ['git', '-c', f'safe.directory={args.comfy}', '-C', str(args.comfy)]
    assert subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip() == COMFY_REV
    assert not subprocess.check_output([*git, 'diff', '--stat'], text=True).strip()
    sys.path.insert(0, str(args.comfy))
    from comfy.cli_args import args as comfy_args
    comfy_args.use_pytorch_cross_attention = True
    comfy_args.bf16_unet = True
    # Built-in automatic VRAM management; no non-ROCm dynamic allocator patch.
    import torch
    import comfy.sd as sd
    import comfy.sample as sample
    import comfy.samplers as samplers
    import comfy.model_management as mm
    from safetensors.torch import load_file
    from PIL import Image, ImageStat
    import numpy as np

    logging.basicConfig(level=logging.INFO)
    free, total = torch.cuda.mem_get_info()
    assert total - free < 512 * 1024**2, 'GPU already in use'
    flux, turbo = args.family == 'flux3', args.family == 'qwen-turbo'
    pins = read(REPO / f'rocm_tools/{"flux3" if flux else "qwen_image21"}_sources.json')
    spec = pins['model'] if flux else pins['models']['turbo' if turbo else 'base']
    assert read(args.source / 'exl3_source.json') == spec
    cases = read(REPO / f'rocm_tools/{"flux3" if flux else "qwen_image21"}_cases.json')
    case = next(c for c in cases['evaluation'] if c['id'] == 'composition')
    caption, seed = case['caption' if flux else 'prompt'], case['seed']
    sigmas = schedule(args.source, flux, turbo)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    original = load_shards(args.source / 'transformer')
    count, elements = len(original), sum(x.numel() for x in original.values())
    native = flux3_native(original) if flux else qwen_native(original)
    del original
    model = sd.load_diffusion_model_state_dict(native, model_options={'dtype': torch.bfloat16})
    if model is None:
        raise RuntimeError('ComfyUI did not recognize the checkpoint')
    del native
    encoder_state = load_shards(args.source / 'text_encoder')
    clip = sd.load_text_encoder_state_dicts([encoder_state], clip_type=sd.CLIPType.FLUX3 if flux else sd.CLIPType.QWEN_IMAGE,
                                            model_options={'dtype': torch.bfloat16})
    del encoder_state
    vae_path = args.vae_root / ('flux3/ae.safetensors' if flux else 'qwen/vae/qwen_image_2.1_vae_bf16.safetensors')
    vae_state = load_file(vae_path)
    reference_vae = load_shards(args.source / 'vae')
    # The official Comfy Qwen VAE uses the Wan 2.2 namespace and BF16 storage;
    # upstream Diffusers uses an image-specific namespace and FP32 storage.
    # Compare every numeric tensor at the actual common inference dtype, retaining
    # duplicate multiplicities and ignoring names and singleton conv dimensions.
    vae_dtype = torch.float32 if flux else torch.bfloat16
    def payload(value):
        raw = value.to(vae_dtype).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return value.numel(), hashlib.sha256(raw).hexdigest()
    # The FP32 exports differ only in the BatchNorm training-step counter.
    # This counter is not read by evaluation-mode normalization; running mean
    # and variance, affine parameters and every other tensor must still match.
    ignored = 'bn.num_batches_tracked'
    counters = {name: state[ignored].item() if ignored in state else None
                for name, state in [('source', reference_vae), ('native', vae_state)]}
    source_payloads = Counter(payload(v) for k, v in reference_vae.items() if k != ignored)
    assert source_payloads == Counter(payload(v) for k, v in vae_state.items() if k != ignored), 'VAE inference tensors differ at inference dtype'
    vae_tensor_count = len(reference_vae) - int(ignored in reference_vae)
    del reference_vae
    vae = sd.VAE(sd=vae_state, dtype=torch.float32 if flux else torch.bfloat16)
    del vae_state
    vae.throw_exception_if_invalid()
    torch.set_num_threads(old_threads)
    gc.collect()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    graph = model.model.diffusion_model
    calls, steps = [], []

    def before(mod, inputs, kwargs):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        timestep = kwargs.get('timestep', inputs[1] if len(inputs) > 1 else None)
        calls.append({'start': event, 'timestep': timestep.detach()})

    def after(mod, inputs, kwargs, out):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        calls[-1]['end'] = event

    handles = [graph.register_forward_pre_hook(before, with_kwargs=True),
               graph.register_forward_hook(after, with_kwargs=True)]

    def step_callback(index, denoised, x, total_steps):
        steps.append(index)

    measurements = []
    try:
        with torch.inference_mode():
            for run in range(args.warmup + args.repeats):
                calls.clear()
                steps.clear()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                tokens = clip.tokenize(caption, prevent_empty_text=True)
                positive = clip.encode_from_tokens_scheduled(tokens)
                torch.cuda.synchronize()
                encoded = time.perf_counter()
                latent = torch.zeros((1, 128 if flux else 64, 64, 64), device='cpu')
                noise = sample.prepare_noise(latent, seed)
                # Euler integration with precisely the source model's saved/derived sigmas.
                result = sample.sample_custom(model, noise, 1.0, samplers.sampler_object('euler'), sigmas,
                                              positive, [], latent, callback=step_callback, seed=seed)
                torch.cuda.synchronize()
                sampled = time.perf_counter()
                if not torch.isfinite(result).all():
                    raise RuntimeError('Nonfinite diffusion output')
                # Match the existing Qwen CLI's 256px tiles / 64px overlap.
                # ComfyUI's untiled and default 1024px fallback exceeded 32 GiB
                # with its resident BF16 models in this environment.
                pixels = vae.decode(result) if flux else vae.decode_tiled(result, tile_x=16, tile_y=16, overlap=4)
                torch.cuda.synchronize()
                if not torch.isfinite(pixels).all():
                    raise RuntimeError('Nonfinite VAE output')
                array = (pixels[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                image = Image.fromarray(array)
                elapsed = time.perf_counter() - start
                assert len(calls) == len(sigmas) - 1 == len(steps)
                assert image.size == (1024, 1024)
                record = {'warmup': run < args.warmup, 'pipeline_seconds': elapsed,
                          'encoding_seconds': encoded - start, 'sampler_seconds_including_transfers': sampled - encoded,
                          'vae_and_postprocess_seconds': time.perf_counter() - sampled,
                          'denoising_seconds': calls[0]['start'].elapsed_time(calls[-1]['end']) / 1000,
                          'transformer_ms': [c['start'].elapsed_time(c['end']) for c in calls],
                          'timesteps': [c['timestep'].float().cpu().tolist() for c in calls], 'actual_steps': len(calls),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                          'finite_final_latents': True, 'finite_vae_output': True,
                          'conditioning_shape': list(positive[0][0].shape),
                          'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                          'image_size': list(image.size), 'image_mode': image.mode,
                          'rgb_stddev': ImageStat.Stat(image.convert('RGB')).stddev}
                image.save(args.output / f'run-{run}.png')
                measurements.append(record)
                print(json.dumps(record), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    report = {'engine': 'ComfyUI', 'engine_revision': COMFY_REV, 'source': spec, 'family': args.family,
              'source_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
              'gpu': torch.cuda.get_device_name(), 'arch': torch.cuda.get_device_properties(0).gcnArchName,
              'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'diffusers', 'comfy-kitchen', 'comfy-aimdo']},
              'implementation_sha256': digest(__file__), 'model_load_seconds': load_seconds,
              'weight_conversion': {'tensor_count': count, 'elements': elements,
                                    'official_converter_roundtrip_equal_all_tensors': True,
                                    'qwen_fused_swiglu': not flux, 'flux_fused_qkv': False,
                                    'vae_tensor_values_multiset_equal_at_runtime_dtype': True,
                                    'vae_verified_tensor_count': vae_tensor_count,
                                    'vae_training_only_batch_counters_not_compared': counters},
              'settings': {'width': 1024, 'height': 1024, 'batch': 1, 'seed': seed, 'caption': caption,
                           'cfg': 1.0, 'sampler': 'euler', 'sigmas': sigmas.tolist(),
                           'source_schedule_override': True, 'torch_compile': False,
                           'vae_dtype': str(vae.vae_dtype),
                           'vae_decode': 'native automatic tiling fallback' if flux else 'native tiled, 256px tiles / 64px overlap',
                           'attention': 'pytorch SDPA', 'conditioning_cached_between_runs': False,
                           'kv_cache': 'ComfyUI lossless default for Qwen, none for distilled FLUX3',
                           'memory_policy': 'ComfyUI automatic VRAM management', 'cpu_threads': old_threads},
              'vae_file': {'path': str(vae_path), 'sha256': digest(vae_path)},
              'measurements': measurements,
              'timing_scope': 'Headless native engine. Full fresh encoding, sampler including transfers, VAE and PIL; excludes initial load, weight conversion and PNG save. No node-output cache.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
