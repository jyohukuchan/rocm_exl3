"""Run serial, fixed-input R9700 image benchmarks in an existing GPU environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--engine', choices=('rocm', 'comfy'), default='rocm')
    ap.add_argument('--models', nargs='+', choices=('qwen-base', 'qwen-turbo', 'flux3'),
                    default=['qwen-base', 'qwen-turbo', 'flux3'])
    ap.add_argument('--precisions', nargs='+', choices=('bf16', 'exl3-3bit', 'exl3-4bit', 'exl3-5bit'),
                    default=['bf16', 'exl3-4bit'])
    ap.add_argument('--components', type=Path, default=Path('/models/safetensors'))
    ap.add_argument('--qwen-weights', type=Path, default=Path('/work/qwen-image21'))
    ap.add_argument('--flux-weights', type=Path, default=Path('/work/flux3/r9700'))
    ap.add_argument('--comfy-source', type=Path)
    ap.add_argument('--native-vae', type=Path)
    args = ap.parse_args()
    if args.engine == 'comfy' and (args.comfy_source is None or args.native_vae is None):
        ap.error('ComfyUI requires --comfy-source and --native-vae')
    for family in args.models:
        flux = family == 'flux3'
        variant = family.removeprefix('qwen-')
        source = args.components / ('FLUX.3-Image-Klein' if flux else
                                    'Qwen-Image-2.1' + ('-Turbo' if variant == 'turbo' else ''))
        cases = json.loads((REPO / f'rocm_tools/{"flux3" if flux else "qwen_image21"}_cases.json').read_text())
        case = next(c for c in cases['evaluation'] if c['id'] == 'composition')
        for precision in (['bf16'] if args.engine == 'comfy' else args.precisions):
            output = args.root / args.engine / family / precision
            if (output / 'report.json').exists():
                raise FileExistsError(f'Use a fresh output root or select unfinished tasks: {output}')
            output.parent.mkdir(parents=True, exist_ok=True)
            if args.engine == 'comfy':
                command = [sys.executable, '-u', str(REPO / 'rocm_tools/benchmark_comfy_image.py'),
                           '--family', family, '--source', str(source), '--comfy', str(args.comfy_source),
                           '--vae-root', str(args.native_vae), '--output', str(output)]
            else:
                command = [sys.executable, '-u', str(REPO / f'rocm_tools/{"flux3" if flux else "qwen_image21"}.py'),
                           'generate', '--model', str(source), '--output', str(output),
                           '--seed', str(case['seed']), '--width', '1024', '--height', '1024',
                           '--warmup', '1', '--repeats', '3', '--purpose', 'benchmark',
                           '--caption' if flux else '--prompt', case['caption' if flux else 'prompt']]
                if not flux:
                    command += ['--variant', variant, '--offload', 'text-encoder']
                if precision != 'bf16':
                    bits = int(precision.split('-')[1].removesuffix('bit'))
                    if bits not in (3, 4, 5):
                        raise ValueError('Expected BF16 or EXL3 3/4/5bit')
                    checkpoint = args.flux_weights / f'exl3-{bits}bit' if flux else args.qwen_weights / f'{variant}-exl3-{bits}bit'
                    command += ['--exl3', str(checkpoint)]
            log = output.with_suffix('.log')
            print(json.dumps({'model': family, 'precision': precision, 'engine': args.engine,
                              'command': command, 'log': str(log)}), flush=True)
            with log.open('w') as f:
                subprocess.run(command, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
            print(f'complete {args.engine}/{family}/{precision}', flush=True)


if __name__ == '__main__':
    main()
