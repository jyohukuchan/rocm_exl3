"""Validate and archive fixed-case native ComfyUI vs rocm_exl3 measurements."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ('qwen-base', 'qwen-turbo', 'flux3')
PRECISIONS = ('bf16', 'exl3-4bit')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, family, engine, precision):
    path = root / engine / family / precision / 'report.json'
    report = json.loads(path.read_text())
    flux = family == 'flux3'
    pins = json.loads((REPO / f'rocm_tools/{"flux3" if flux else "qwen_image21"}_sources.json').read_text())
    source = pins['model'] if flux else pins['models'][family.removeprefix('qwen-')]
    assert report['source'] == source
    cases = json.loads((REPO / f'rocm_tools/{"flux3" if flux else "qwen_image21"}_cases.json').read_text())
    case = next(c for c in cases['evaluation'] if c['id'] == 'composition')
    settings = report['settings']
    prompt = settings['caption'] if engine == 'comfy' else report['caption'] if flux else settings['prompt']
    assert prompt == case['caption' if flux else 'prompt'] and settings['seed'] == case['seed']
    assert settings['width'] == settings['height'] == 1024
    assert 'R9700' in report['gpu'] and report['arch'].startswith('gfx1201')
    runs = report['measurements']
    assert len(runs) == 4 and [r['warmup'] for r in runs] == [True, False, False, False]
    expected_steps = 16 if flux else 8 if family == 'qwen-turbo' else 40
    assert all(r['actual_steps'] == expected_steps and len(r['transformer_ms']) == expected_steps
               and r['finite_final_latents'] and r['finite_vae_output'] and r['image_size'] == [1024, 1024]
               for r in runs)
    measured = [r for r in runs if not r['warmup']]
    timing = lambda r: r['denoising_seconds'] if 'denoising_seconds' in r else r['denoising_ms'] / 1000
    assert all(math.isfinite(t) and t > 0 for r in runs for t in
               [r['pipeline_seconds'], timing(r), *r['transformer_ms']])
    if engine == 'rocm' and not flux:
        assert all(r['cache_modes'] == ['extract'] + ['cached'] * (expected_steps - 1) for r in runs)
        assert report['settings']['true_cfg_scale'] == 1
    if engine == 'rocm':
        assert report['precision'] == ('bf16' if precision == 'bf16' else 'exl3')
        if precision != 'bf16':
            assert report['quantization']['bits'] == 4 and report['quantization']['complete']
    else:
        assert report['weight_conversion']['official_converter_roundtrip_equal_all_tensors']
        assert report['weight_conversion']['vae_tensor_values_multiset_equal_at_runtime_dtype']
    images = []
    from PIL import Image
    for i, r in enumerate(runs):
        image_path = path.with_name(f'run-{i}.png')
        with Image.open(image_path) as im:
            assert hashlib.sha256(im.tobytes()).hexdigest() == r['pixel_sha256']
        images.append({'path': str(image_path), 'file_sha256': sha(image_path), 'pixel_sha256': r['pixel_sha256']})
    result = {'family': family, 'engine': 'ComfyUI' if engine == 'comfy' else 'rocm_exl3', 'precision': precision,
              'source': source, 'steps': expected_steps, 'report_sha256': sha(path), 'images': images,
              'pipeline_seconds': statistics.median(r['pipeline_seconds'] for r in measured),
              'pipeline_samples_seconds': [r['pipeline_seconds'] for r in measured],
              'pipeline_range_seconds': [min(r['pipeline_seconds'] for r in measured), max(r['pipeline_seconds'] for r in measured)],
              'denoising_seconds': statistics.median(timing(r) for r in measured),
              'denoising_samples_seconds': [timing(r) for r in measured],
              'denoising_range_seconds': [min(timing(r) for r in measured), max(timing(r) for r in measured)],
              'transformer_seconds_sum': statistics.median(sum(r['transformer_ms']) / 1000 for r in measured),
              'peak_allocated_gib': statistics.median(r['peak_allocated_bytes'] / 1024**3 for r in measured),
              'repeated_measured_pixel_hashes_equal': len({r['pixel_sha256'] for r in measured}) == 1,
              'initialization_seconds_including_adapter_validation': report['model_load_seconds']}
    if engine == 'comfy':
        result['engine_revision'] = report['engine_revision']
        result['conditioning_shape'] = measured[0]['conditioning_shape']
        result['parts_seconds'] = {k: statistics.median(r[k] for r in measured) for k in
            ('encoding_seconds', 'sampler_seconds_including_transfers', 'vae_and_postprocess_seconds')}
        result['weight_conversion'] = report['weight_conversion']
    return result, path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = args.output / 'reports'
    reports.mkdir(exist_ok=True)
    rows = []
    for family in FAMILIES:
        for engine, precisions in [('comfy', ['bf16']), ('rocm', PRECISIONS)]:
            for precision in precisions:
                row, path = collect(args.root, family, engine, precision)
                with gzip.open(reports / f'{family}__{engine}__{precision}.json.gz', 'wb') as f:
                    f.write(path.read_bytes())
                rows.append(row)
        family_rows = [r for r in rows if r['family'] == family]
        comfy = next(r for r in family_rows if r['engine'] == 'ComfyUI')
        bf16 = next(r for r in family_rows if r['engine'] == 'rocm_exl3' and r['precision'] == 'bf16')
        for row in family_rows:
            row['pipeline_relative_to_comfy_bf16'] = row['pipeline_seconds'] / comfy['pipeline_seconds']
            row['denoising_relative_to_comfy_bf16'] = row['denoising_seconds'] / comfy['denoising_seconds']
            row['pipeline_relative_to_rocm_bf16'] = row['pipeline_seconds'] / bf16['pipeline_seconds']
            row['denoising_relative_to_rocm_bf16'] = row['denoising_seconds'] / bf16['denoising_seconds']
    result = {'device': 'single Radeon AI PRO R9700', 'width': 1024, 'height': 1024, 'batch': 1,
              'warmup': 1, 'measured_repeats': 3, 'statistic': 'median', 'rows': rows,
              'scope': 'Fixed composition input per model. Fresh encoding every run; excludes initial loading, PNG save and FLUX PUP. Different engine memory/VAE implementations remain native.',
              'limitations': ['One prompt and seed per model; not a broad workload average.',
                             'Headless ComfyUI native engine APIs, not HTTP/UI/node-cache latency.',
                             'Comfy native VAE automatic tiling vs rocm_exl3 VAE tiling; memory policies differ.',
                             'Denoising interval excludes text encoding and VAE, while Comfy sampler wall time also includes transfers.',
                             'Initial Comfy setup includes verified source weight adaptation; initialization times are not a load-speed ranking.',
                             'Identical source schedules and model values do not imply pixel-identical results across engine arithmetic and RNG implementations.']}
    resident_path = args.root / 'comfy-flux3-resident.json'
    if resident_path.exists():
        resident = json.loads(resident_path.read_text())
        runs = resident['measurements']
        assert [r['warmup'] for r in runs] == [True, False, False, False]
        assert all(r['actual_steps'] == 16 and r['finite_final_latents'] for r in runs)
        samples = [r['denoising_seconds'] for r in runs if not r['warmup']]
        result['resident_flux3_diagnosis'] = {
            'scope': resident['scope'], 'denoising_seconds': statistics.median(samples),
            'denoising_samples_seconds': samples,
            'conditioning_file_sha256': resident['conditioning_file_sha256'],
            'repeated_measured_latent_hashes_equal': len({r['latent_sha256'] for r in runs if not r['warmup']}) == 1,
            'note': 'Fully resident BF16 DiT, shared cached BF16 conditioning, no encoder/VAE. Separate from full-pipeline results.'}
        with gzip.open(reports / 'flux3__comfy__resident-diagnosis.json.gz', 'wb') as f:
            f.write(resident_path.read_bytes())
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = []
    for field, title in [('pipeline_seconds', '生成全体 秒'), ('denoising_seconds', 'Denoising 秒'),
                         ('peak_allocated_gib', 'Peak allocated GiB')]:
        lines += [f'## {title}', '', '| Model | ComfyUI BF16 | rocm BF16 | EXL3 4bit |',
                  '|---|---:|---:|---:|']
        for family in FAMILIES:
            selected = [r for r in rows if r['family'] == family]
            lines.append('| ' + family + ' | ' + ' | '.join(f'{r[field]:.3f}' for r in selected) + ' |')
        lines.append('')
    (args.output / 'comparison.md').write_text('\n'.join(lines))
    from PIL import Image, ImageDraw
    cell, label = 320, 32
    grid = Image.new('RGB', (3 * cell, 3 * (cell + 2 * label)), 'white')
    draw = ImageDraw.Draw(grid)
    for family_index, family in enumerate(FAMILIES):
        for index, row in enumerate(r for r in rows if r['family'] == family):
            image = Image.open(row['images'][1]['path']).convert('RGBA')
            background = Image.new('RGBA', image.size, 'white')
            background.alpha_composite(image)
            image = background.convert('RGB').resize((cell, cell), Image.Resampling.LANCZOS)
            x, y = index * cell, family_index * (cell + 2 * label)
            draw.text((x + 5, y + 5), f'{family} / {row["engine"]} {row["precision"]}', fill='black')
            draw.text((x + 5, y + label + 5), f'{row["pipeline_seconds"]:.2f}s', fill='black')
            grid.paste(image, (x, y + 2 * label))
    grid.save(args.output / 'comparison.png')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
