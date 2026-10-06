"""Prepare a JEV Hub release locally; performs no upload or repository creation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', default='jyohukuchan/JEV-27B-VL-exl3-4bpw')
    args = parser.parse_args()
    source, output = args.model.resolve(), args.output.resolve()
    if output.exists():
        parser.error('Output must be a new directory; existing releases are never overwritten')
    if len(args.repo_id.split('/')) != 2:
        parser.error('Repository ID must be namespace/name')
    engine = Path(__file__).resolve().parents[1]
    assets = engine/'doc/jev_hf_release'
    for relative in ('config.json', 'quantization_config.json', 'model.safetensors.index.json',
                     'decision_rows.safetensors', 'adapter_vllm/adapter_model.safetensors',
                     'adapter_vllm/decision_head.json', 'calibration.json', 'LICENSE', 'README.md'):
        if not (source/relative).is_file():
            parser.error(f'Missing required model file: {relative}')
    config = json.loads((source/'config.json').read_text())
    assert config['quantization_config']['quant_method'] == 'exl3'
    assert config['quantization_config']['bits'] == 4
    output.mkdir(parents=True)
    excluded = {'README.md', 'serve.sh', 'serve_decide.py'}
    linked = 0
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(source)
        dest = output/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.safetensors':
            os.link(path, dest)
            linked += 1
        else:
            shutil.copy2(path, dest)
    card = (assets/'README.md').read_text().replace(
        'jyohukuchan/JEV-27B-VL-exl3-4bpw', args.repo_id)
    (output/'README.md').write_text(card)
    for name in ('NOTICE', 'serve_exl3.py'):
        shutil.copy2(assets/name, output/name)
    shutil.copy2(source/'README.md', output/'SOURCE_MODEL_CARD.md')
    shutil.copy2(engine/'LICENSE', output/'LICENSE-code')
    evidence = engine/'doc/jev_latency_20261006'
    evaluation = {}
    for gpu in ('r9700', 'v620'):
        results = json.loads((evidence/f'{gpu}.json').read_text())
        evaluation[gpu] = {
            'text': {key:{'prompt_tokens':value['prompt_tokens'], 'http':value['http']}
                     for key,value in results['text'].items()},
            'browser': {key:results['browser'][key] for key in
                        ('episode_count', 'success_count', 'http', 'server', 'prompt_tokens', 'viewport')},
        }
    (output/'EVALUATION_SUMMARY.json').write_text(json.dumps(evaluation, indent=2)+'\n')
    metadata = {
        'source_model': 'autotrust/JEV-27B-VL',
        'source_revision': (source/'source_revision.txt').read_text().strip(),
        'engine': 'https://github.com/jyohukuchan/rocm_exl3',
        'tested_engine_revision': '7675fa26cdf52fbe5a90060a69c60c752855fd8f',
        'proposed_hub_repository': args.repo_id,
        'license': 'Apache-2.0 (weights), MIT (launcher)',
        'quantization': config['quantization_config'],
    }
    (output/'RELEASE_METADATA.json').write_text(json.dumps(metadata, indent=2)+'\n')
    # Check every header/key against the index before computing the release digest list.
    index = json.loads((output/'model.safetensors.index.json').read_text())['weight_map']
    actual = {}
    for filename in sorted(set(index.values())):
        with (output/filename).open('rb') as handle:
            size = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(size))
        actual.update({key:filename for key in header if key != '__metadata__'})
    assert actual == index, 'Weight shards and index differ'
    from huggingface_hub import ModelCard
    parsed = ModelCard.load(str(output/'README.md'))
    assert parsed.data.base_model_relation == 'quantized'
    assert parsed.data.base_model == 'autotrust/JEV-27B-VL'
    assert parsed.data.library_name is None
    entries = [{'path':str(path.relative_to(output)), 'bytes':path.stat().st_size,
                'sha256':sha256(path)} for path in sorted(output.rglob('*')) if path.is_file()]
    manifest = {'files':entries, 'bytes_without_manifest':sum(e['bytes'] for e in entries),
                'manifest_excludes_itself':True}
    (output/'FILE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output':str(output), 'proposed_repository':args.repo_id,
                      'files':len(entries)+1, 'hardlinked_weight_files':linked,
                      'indexed_tensors':len(actual),
                      'bytes_without_manifest':manifest['bytes_without_manifest'],
                      'uploaded':False}, indent=2))


if __name__ == '__main__':
    main()
