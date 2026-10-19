"""Stage licensed FLUX.3 Image Klein EXL3 checkpoints without changing tensor data."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import re
import shutil

from rocm_tools.prepare_qwen_image21_hf_release import annotate_tensor_file, read, write, sha

ENGINE = Path(__file__).resolve().parents[1]
ENGINE_REV = '0d879e7a7b6aadd3c08508240ade6db87332bafe'
LICENSE_SHA = '100a3a5af8f1a2943dca0d127671bcaa93ad894a63330825219f1a0a44451f22'
NOTICE_SHA = 'a12f7eb0cab87ed210ffb0921425589a1c88abded6ba0c4285b4d32cbaf5f35c'
LICENSE_NAME = 'FLUX Kommunity License v1.0'


def prepare(args):
    pins = read(ENGINE / 'rocm_tools/flux3_sources.json')
    spec = pins['model']
    assert read(args.source_model / 'exl3_source.json') == spec
    assert sha(args.source_model / 'LICENSE.md') == LICENSE_SHA
    assert sha(args.source_model / 'NOTICE.md') == NOTICE_SHA
    if args.output.exists():
        raise FileExistsError('Use a new staging directory')
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', args.repo_id):
        raise ValueError('Expected namespace/repository')
    args.output.mkdir(parents=True)
    publisher = args.repo_id.split('/')[0]
    attribution = (args.source_model / 'NOTICE.md').read_text()
    notice = attribution + '\n' + (
        f'Modified by {publisher}, 2026-10-19, from {spec["repo_id"]}@{spec["revision"]}.\n'
        '258 transformer block Linear layers were quantized to EXL3 3/4/5bit. Other DiT tensors '
        'were retained at high precision and repackaged. Publication adds source/license/modification '
        'metadata only; evaluated tensor payload bytes are unchanged.\n'
        'This is an independent community conversion, not an official product of Black Forest Labs Inc. '
        'It has not been endorsed, approved or validated by Black Forest Labs Inc.\n'
        'Rights to use this model are granted directly by Black Forest Labs Inc. to recipients '
        'under the accompanying FLUX Kommunity License, not sublicensed by this publisher. '
        'No additional model license terms are imposed.\n'
    )
    shutil.copyfile(args.source_model / 'LICENSE.md', args.output / 'LICENSE.md')
    (args.output / 'NOTICE.md').write_text(notice)
    evidence = ENGINE / 'benchmarks/2026-10-19/flux3'
    for source_name, target_name in (
        ('comparison.json', 'evaluation.json'), ('comparison.md', 'benchmark.md'),
        ('multibit.png', 'comparison.png'), ('quality-4bit.png', 'quality-4bit.png'),
        ('multibit-visual-review.json', 'visual_review.json'),
        ('4bit-visual-review.json', 'visual_review_4bit.json'),
        ('r9700-environment.json', 'r9700-environment.json'),
        ('v620-environment.json', 'v620-environment.json'),
    ):
        shutil.copyfile(evidence / source_name, args.output / target_name)
    shutil.copyfile(ENGINE / 'rocm_tools/flux3_cases.json', args.output / 'evaluation_cases.json')
    shutil.copyfile(ENGINE / 'rocm_tools/flux3_sources.json', args.output / 'sources.json')
    cases = read(args.output / 'evaluation_cases.json')
    caption = next(c['caption'] for c in cases['evaluation'] if c['id'] == 'composition')
    (args.output / 'example-caption.txt').write_text(caption + '\n')
    evaluation = read(args.output / 'evaluation.json')
    assert evaluation['source'] == spec
    publication = {'source': spec, 'engine_repository': 'https://github.com/jyohukuchan/rocm_exl3_forme',
                   'engine_revision': ENGINE_REV, 'repository': args.repo_id,
                   'license_sha256': LICENSE_SHA, 'upstream_notice_sha256': NOTICE_SHA,
                   'release_tag': 'exl3-v1', 'header_only_repackaging': True, 'checkpoints': {}}
    for bits in (3, 4, 5):
        source = args.checkpoints_root / f'exl3-{bits}bit'
        original = read(source / 'quantization.json')
        assert original['complete'] and original['bits'] == bits and original['source'] == spec
        assert original['transformer_type'] == 'Flux3Transformer2DModel' and len(original['layers']) == 258
        target = args.output / f'{bits}bit'
        target.mkdir()
        shutil.copyfile(args.output / 'LICENSE.md', target / 'LICENSE.md')
        shutil.copyfile(args.output / 'NOTICE.md', target / 'NOTICE.md')
        config = source / 'transformer_config.json'
        assert sha(config) == original['transformer_config_sha256']
        shutil.copyfile(config, target / config.name)
        manifest, mapping = copy.deepcopy(original), {}
        metadata = {'source_model': spec['repo_id'], 'source_revision': spec['revision'],
                    'license': LICENSE_NAME, 'attribution': attribution.strip()}
        for record in manifest['layers'].values():
            metadata['MODIFICATION_NOTICE'] = (
                f'Modified by {publisher}: quantized from {spec["repo_id"]} to EXL3 {bits}bit '
                'and repackaged as a per-layer checkpoint. Publication changes metadata only; '
                'evaluated tensor payload unchanged. Independent community conversion.')
            name = record['file']
            if Path(name).name != name:
                raise ValueError('Expected a flat checkpoint filename')
            info = annotate_tensor_file(source / name, target / name, record['sha256'], metadata)
            mapping[name] = info
            record['evaluated_sha256'], record['sha256'] = record['sha256'], info['sha256']
        metadata['MODIFICATION_NOTICE'] = (
            f'Modified by {publisher}: retained high-precision subset of {spec["repo_id"]}, '
            'repackaged from original shards. Publication changes metadata only; evaluated tensor payload unchanged.')
        info = annotate_tensor_file(source / 'unquantized.safetensors', target / 'unquantized.safetensors',
                                    original['unquantized_sha256'], metadata)
        mapping['unquantized.safetensors'] = info
        manifest['unquantized_evaluated_sha256'] = original['unquantized_sha256']
        manifest['unquantized_sha256'] = info['sha256']
        manifest['_publication'] = {'license': LICENSE_NAME, 'modification_notice': notice,
                                    'evaluated_manifest_sha256': sha(source / 'quantization.json'),
                                    'engine_revision': ENGINE_REV}
        write(target / 'quantization.json', manifest)
        publication['checkpoints'][str(bits)] = {'files': mapping,
            'evaluated_manifest_sha256': sha(source / 'quantization.json'),
            'safetensors_bytes': sum(x['bytes'] for x in mapping.values())}
        print(f'{bits}bit: all 259 tensor payloads verified identical', flush=True)
    write(args.output / 'PUBLICATION.json', publication)
    weight_rows = '\n'.join(
        f'| [{b}bit]({b}bit) | {b} | {read(args.output / f"{b}bit/quantization.json")["effective_bits_per_source_element"]:.6f} | '
        f'{publication["checkpoints"][str(b)]["safetensors_bytes"]/1e9:.6f} | {evaluation["nmse"][f"exl3-{b}bit"]["nmse"]:.10f} |'
        for b in (3, 4, 5))
    case_rows = '\n'.join('| ' + c + ' | ' + ' | '.join(
        f'{evaluation["nmse"][f"exl3-{b}bit"]["cases"][c]:.10f}' for b in (3, 4, 5)) + ' |'
        for c in evaluation['nmse_cases'])
    replacements = {'SOURCE_ID': spec['repo_id'], 'SOURCE_REV': spec['revision'], 'ENGINE_REV': ENGINE_REV,
                    'REPO_ID': args.repo_id, 'REPO_NAME': args.repo_id.split('/')[1],
                    'WEIGHT_ROWS': weight_rows, 'CASE_ROWS': case_rows}
    card = (ENGINE / 'doc/flux3_hf/README.template.md').read_text()
    for key, value in replacements.items():
        card = card.replace('@' + key + '@', value)
    if re.search(r'@[A-Z_]+@', card):
        raise ValueError('Unresolved model-card placeholder')
    (args.output / 'README.md').write_text(card)
    from huggingface_hub import ModelCard
    parsed = ModelCard.load(str(args.output / 'README.md'))
    assert parsed.data.base_model == spec['repo_id'] and parsed.data.base_model_relation == 'quantized'
    assert parsed.data.license == 'other' and parsed.data.license_name == 'flux-kommunity'
    weight_hashes = {f'{b}bit/{name}': info['sha256'] for b, checkpoint in publication['checkpoints'].items()
                     for name, info in checkpoint['files'].items()}
    entries = []
    for p in sorted(args.output.rglob('*')):
        if p.is_file():
            relative = p.relative_to(args.output).as_posix()
            entries.append({'path': relative, 'bytes': p.stat().st_size,
                            'sha256': weight_hashes.get(relative) or sha(p)})
    write(args.output / 'FILE_MANIFEST.json', {'files': entries, 'manifest_excludes_itself': True})
    print(f'Ready: {args.repo_id}, {len(entries)+1} files; not uploaded', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--checkpoints-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
