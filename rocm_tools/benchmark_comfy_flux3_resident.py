"""Diagnose FLUX.3 denoising variance with resident DiT and shared cached conditioning."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocm_tools.benchmark_comfy_image import COMFY_REV, load_shards, flux3_native, schedule, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comfy', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--encoding', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert subprocess.check_output(['git', '-c', f'safe.directory={args.comfy}', '-C', str(args.comfy),
                                    'rev-parse', 'HEAD'], text=True).strip() == COMFY_REV
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.comfy))
    from comfy.cli_args import args as comfy_args
    comfy_args.use_pytorch_cross_attention = True
    comfy_args.bf16_unet = True
    import torch
    import comfy.sd as sd
    import comfy.sample as sample
    import comfy.samplers as samplers
    import comfy.model_management as mm
    from safetensors.torch import load_file
    assert torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0] < 512 * 1024**2
    threads = torch.get_num_threads()
    torch.set_num_threads(8)
    model = sd.load_diffusion_model_state_dict(flux3_native(load_shards(args.source / 'transformer')),
                                               model_options={'dtype': torch.bfloat16})
    torch.set_num_threads(threads)
    sigmas = schedule(args.source, True, False)
    encoding = json.loads((args.encoding / 'encoding.json').read_text())
    assert encoding['source'] == json.loads((args.source / 'exl3_source.json').read_text())
    assert encoding['embeddings_sha256'] == digest(args.encoding / 'embeddings.safetensors')
    embeddings = load_file(args.encoding / 'embeddings.safetensors')['prompt_embeds']
    positive = [[embeddings, {}]]
    mm.load_models_gpu([model], force_full_load=True)
    calls = []

    def before(module, inputs):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        calls.append({'start': event})

    def after(module, inputs, output):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        calls[-1]['end'] = event

    graph = model.model.diffusion_model
    handles = [graph.register_forward_pre_hook(before), graph.register_forward_hook(after)]
    records = []
    try:
        with torch.inference_mode():
            for run in range(4):
                calls.clear()
                latent = torch.zeros((1, 128, 64, 64))
                noise = sample.prepare_noise(latent, 1703)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                result = sample.sample_custom(model, noise, 1, samplers.sampler_object('euler'), sigmas,
                                              positive, [], latent, seed=1703)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                assert len(calls) == 16 and torch.isfinite(result).all()
                record = {'warmup': run == 0, 'sampler_seconds': elapsed,
                          'denoising_seconds': calls[0]['start'].elapsed_time(calls[-1]['end']) / 1000,
                          'transformer_ms': [x['start'].elapsed_time(x['end']) for x in calls],
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'finite_final_latents': True, 'actual_steps': len(calls),
                          'latent_sha256': hashlib.sha256(result.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                records.append(record)
                print(json.dumps(record), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    report = {'engine': 'ComfyUI', 'engine_revision': COMFY_REV,
              'gpu': torch.cuda.get_device_name(), 'arch': torch.cuda.get_device_properties(0).gcnArchName,
              'source': encoding['source'], 'conditioning_shape': list(embeddings.shape),
              'conditioning_file_sha256': digest(args.encoding / 'embeddings.safetensors'),
              'settings': {'width': 1024, 'height': 1024, 'steps': 16, 'cfg': 1, 'seed': 1703,
                           'sigmas': sigmas.tolist(), 'dit': 'BF16 fully resident', 'vae': 'not loaded',
                           'text_encoder': 'not loaded; shared previously cached BF16 conditioning'},
              'implementation_sha256': digest(__file__), 'measurements': records,
              'scope': 'Additional resident-DiT diagnosis, separate from full-pipeline benchmark.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
