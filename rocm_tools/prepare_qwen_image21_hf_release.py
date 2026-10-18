"""Prepare public EXL3 diffusion artifacts; no upload or repository creation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct

ENGINE = Path(__file__).resolve().parents[1]
ENGINE_REV = '37847ec9949aaf30fcf72ad6679da78968857d7a'
LICENSE_SHA = '8dc973f024ff95966bea25866efa443fd16776dcb1001e681e3d467ea572b28d'
ATTRIBUTION = ('Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, '
               'Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def header(f):
    prefix = f.read(8)
    if len(prefix) != 8:
        raise ValueError('Truncated safetensors prefix')
    size = struct.unpack('<Q', prefix)[0]
    if size > 100 * 1024**2:
        raise ValueError('Unexpected safetensors header size')
    raw = f.read(size)
    if len(raw) != size:
        raise ValueError('Truncated safetensors header')
    return prefix, raw, json.loads(raw)


def annotate_tensor_file(source, target, expected_sha, metadata):
    """Copy only the header with notices; verify original SHA and exact data bytes.

    The source file is never modified or hardlinked. Existing metadata and tensor
    descriptors/offsets survive. SHA validation completes before atomic rename.
    """
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError('Publication destination must be a new file')
    temporary = target.with_suffix('.safetensors.tmp')
    original_hash, data_hash = hashlib.sha256(), hashlib.sha256()
    try:
        with source.open('rb') as src, temporary.open('xb') as dst:
            prefix, raw, old = header(src)
            original_hash.update(prefix + raw)
            new = copy.deepcopy(old)
            new['__metadata__'] = {**new.get('__metadata__', {}), **metadata}
            encoded = json.dumps(new, ensure_ascii=False, separators=(',', ':')).encode()
            encoded += b' ' * (-len(encoded) % 8)
            dst.write(struct.pack('<Q', len(encoded)))
            dst.write(encoded)
            for block in iter(lambda: src.read(8 * 1024**2), b''):
                original_hash.update(block)
                data_hash.update(block)
                dst.write(block)
        if original_hash.hexdigest() != expected_sha:
            raise ValueError(f'Source checksum mismatch: {source.name}')
        published_hash, reread_data = hashlib.sha256(), hashlib.sha256()
        with temporary.open('rb') as f:
            prefix, raw, checked = header(f)
            published_hash.update(prefix + raw)
            assert {k:v for k,v in checked.items() if k != '__metadata__'} == {k:v for k,v in old.items() if k != '__metadata__'}
            for block in iter(lambda: f.read(8 * 1024**2), b''):
                published_hash.update(block)
                reread_data.update(block)
        if reread_data.digest() != data_hash.digest():
            raise ValueError('Publication changed tensor payload bytes')
        from safetensors import safe_open
        with safe_open(temporary, framework='np') as f:
            assert set(f.keys()) == set(old) - {'__metadata__'}
            assert f.metadata()['MODIFICATION_NOTICE'] == metadata['MODIFICATION_NOTICE']
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {'sha256': published_hash.hexdigest(), 'evaluated_sha256': expected_sha,
            'tensor_payload_sha256': data_hash.hexdigest(), 'tensor_payload_identical': True,
            'bytes': target.stat().st_size}


def prepare(args):
    pins = read(ENGINE / 'rocm_tools/qwen_image21_sources.json')
    spec = pins['models'][args.variant]
    assert read(args.source_model / 'exl3_source.json') == spec
    assert sha(args.source_model / 'LICENSE') == LICENSE_SHA
    if args.output.exists():
        raise FileExistsError('Use a new staging directory; existing releases are not overwritten')
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', args.repo_id):
        raise ValueError('Expected namespace/repository')
    args.output.mkdir(parents=True)
    publisher = args.repo_id.split('/')[0]
    notice = ATTRIBUTION + '\n\nBuilt with Qwen.\n\n' + (
        f'Modified by {publisher}, 2026-10-19. Derived from {spec["repo_id"]}@{spec["revision"]}.\n'
        'Eligible transformer block Linear weights were quantized to EXL3 3/4/5bit; '
        'other DiT tensors were retained at high precision and repackaged.\n'
        'Publication adds modification/source/license metadata without changing the evaluated tensor payloads.\n'
        'This is a community conversion. Research/evaluation, non-commercial use only under LICENSE.\n')
    shutil.copyfile(args.source_model / 'LICENSE', args.output / 'LICENSE')
    (args.output / 'Notice').write_text(notice)
    result = read(ENGINE / 'benchmarks/2026-10-19/qwen-image21-multibit-nmse.json')
    evaluation = {k:v for k,v in result.items() if k != 'models'}
    evaluation['model'] = result['models'][args.variant]
    write(args.output / 'evaluation.json', evaluation)
    cases = read(ENGINE / 'rocm_tools/qwen_image21_cases.json')
    write(args.output / 'evaluation_cases.json', cases)
    review = read(ENGINE / 'benchmarks/2026-10-19/qwen-image21-multibit-review.json')
    review['cases'] = {k:v for k,v in review['cases'].items() if k.startswith(args.variant + '/')}
    write(args.output / 'visual_review.json', review)
    shutil.copyfile(args.checkpoints_root / f'multibit-review-{args.variant}.png', args.output / 'comparison.png')
    publication = {'source': spec, 'engine_repository': 'https://github.com/jyohukuchan/rocm_exl3_forme',
                   'engine_revision': ENGINE_REV, 'repository': args.repo_id,
                   'license_sha256': LICENSE_SHA, 'release_tag': 'exl3-v1', 'header_only_repackaging': True, 'checkpoints': {}}
    for bits in (3, 4, 5):
        source = args.checkpoints_root / f'{args.variant}-exl3-{bits}bit'
        original = read(source / 'quantization.json')
        assert original['complete'] and original['bits'] == bits and original['source'] == spec
        assert len(original['layers']) == 224
        target = args.output / f'{bits}bit'
        target.mkdir()
        shutil.copyfile(args.source_model / 'LICENSE', target / 'LICENSE')
        (target / 'Notice').write_text(notice)
        shutil.copyfile(source / 'transformer_config.json', target / 'transformer_config.json')
        manifest = copy.deepcopy(original)
        mapping = {}
        for record in manifest['layers'].values():
            metadata = {
                'MODIFICATION_NOTICE': f'Modified by {publisher}: quantized from {spec["repo_id"]} to EXL3 {bits}bit and repackaged as a per-layer checkpoint. Publication adds metadata only; evaluated tensor payload unchanged.',
                'source_model': spec['repo_id'], 'source_revision': spec['revision'],
                'license': 'Qwen Research License Agreement (2026-09-20); non-commercial research/evaluation only',
                'attribution': ATTRIBUTION,
            }
            info = annotate_tensor_file(source / record['file'], target / record['file'], record['sha256'], metadata)
            mapping[record['file']] = info
            record['evaluated_sha256'] = record['sha256']
            record['sha256'] = info['sha256']
        metadata['MODIFICATION_NOTICE'] = (f'Modified by {publisher}: retained high-precision tensor subset from '
            f'{spec["repo_id"]}, repackaged from original shards. Publication adds metadata only; evaluated tensor payload unchanged.')
        info = annotate_tensor_file(source / 'unquantized.safetensors', target / 'unquantized.safetensors', original['unquantized_sha256'], metadata)
        mapping['unquantized.safetensors'] = info
        manifest['unquantized_evaluated_sha256'] = original['unquantized_sha256']
        manifest['unquantized_sha256'] = info['sha256']
        manifest['_publication'] = {
            'modification_notice': f'Modified by {publisher}: source DiT quantized to EXL3 {bits}bit; publication adds tensor metadata and updates file checksums. Numeric tensors unchanged from evaluation.',
            'license': 'Qwen Research License Agreement (2026-09-20)',
            'evaluated_manifest_sha256': sha(source / 'quantization.json'),
            'engine_revision': ENGINE_REV,
        }
        write(target / 'quantization.json', manifest)
        publication['checkpoints'][str(bits)] = {'files': mapping, 'evaluated_manifest_sha256': sha(source / 'quantization.json'),
                                               'safetensors_bytes': sum(x['bytes'] for x in mapping.values())}
        print(f'{args.variant} {bits}bit: 225 tensor payloads verified identical', flush=True)
    write(args.output / 'PUBLICATION.json', publication)
    measurements = evaluation['model']
    weight_rows = '\n'.join(
        f'| [{b}bit]({b}bit) | {b} | {measurements[str(b)]["effective_bits_per_source_element"]:.6f} | '
        f'{publication["checkpoints"][str(b)]["safetensors_bytes"]/1e9:.6f} | {measurements[str(b)]["nmse"]:.9f} |'
        for b in (3, 4, 5))
    case_rows = '\n'.join('| ' + c + ' | ' + ' | '.join(f'{measurements[str(b)]["cases"][c]["nmse"]:.9f}' for b in (3,4,5)) + ' |'
                          for c in ('text', 'composition', 'multi_ref'))
    replacements = {
        'SOURCE_ID': spec['repo_id'], 'SOURCE_REV': spec['revision'], 'ENGINE_REV': ENGINE_REV,
        'TITLE': 'EXL3 quantization of ' + spec['repo_id'].split('/')[1],
        'PUBLISHER': publisher, 'VARIANT': args.variant, 'REPO_ID': args.repo_id, 'REPO_NAME': args.repo_id.split('/')[1],
        'COMPONENT_DIR': 'image21-' + args.variant + '-components', 'WEIGHT_ROWS': weight_rows, 'CASE_ROWS': case_rows,
        'STEPS': '8' if args.variant == 'turbo' else '40', 'CAL_STEPS': '0/4/7' if args.variant == 'turbo' else '0/20/39',
        'SCHEDULE_NOTE': ('Turbo uses the **eight saved sigmas**, CFG 1 and prefix KV cache. Do not pass `--steps`.'
                          if args.variant == 'turbo' else 'The CLI defaults to **40 steps**, CFG 1 and prefix KV cache for the base model.'),
        'VISUAL_NOTE': ('At 3bit/4bit, the two-reference example changes the blue background to gray and changes layout; the 3bit cube is partly cropped. The 5bit example retains the blue background and is closer to BF16, with small differences.'
                        if args.variant == 'turbo' else 'The 3bit two-reference example loses the white sail and changes layout. The 5bit example preserves the sail and is closer to BF16, but is not pixel-identical. HELLO and the simple object arrangement remain recognizable at all tested bit widths.'),
    }
    card = (ENGINE / 'doc/qwen_image21_hf/README.template.md').read_text()
    for key, value in replacements.items():
        card = card.replace('@' + key + '@', value)
    if re.search(r'@[A-Z_]+@', card):
        raise ValueError('Unresolved model-card template placeholder')
    (args.output / 'README.md').write_text(card)
    from huggingface_hub import ModelCard
    parsed = ModelCard.load(str(args.output / 'README.md'))
    assert parsed.data.base_model == spec['repo_id'] and parsed.data.base_model_relation == 'quantized'
    assert parsed.data.license == 'other' and parsed.data.license_name == 'qwen-research'
    assert parsed.data.library_name is None
    entries = []
    # Weight hashes already independently verified during repackaging.
    weight_hashes = {f'{b}bit/{name}':info['sha256'] for b, checkpoint in publication['checkpoints'].items()
                     for name, info in checkpoint['files'].items()}
    for p in sorted(args.output.rglob('*')):
        if p.is_file():
            relative = p.relative_to(args.output).as_posix()
            entries.append({'path': relative, 'bytes': p.stat().st_size,
                            'sha256': weight_hashes.get(relative) or sha(p)})
    write(args.output / 'FILE_MANIFEST.json', {'files': entries, 'manifest_excludes_itself': True})
    print(json.dumps({'repository': args.repo_id, 'files':len(entries)+1,
                      'bytes_without_manifest':sum(e['bytes'] for e in entries), 'uploaded':False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('base','turbo'), required=True)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--checkpoints-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
