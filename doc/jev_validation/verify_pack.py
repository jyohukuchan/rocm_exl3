import collections
import hashlib
import json
import math
import pathlib
import struct
import torch
from safetensors import safe_open
from safetensors.torch import load_file

root = pathlib.Path('/work/runs/jev-20261006')
pack = pathlib.Path('/work/models/JEV-27B-VL-exl3-4bpw')
source = pathlib.Path('/models/safetensors/JEV-27B-VL')
rows = load_file(str(pack/'decision_rows.safetensors'))
index = json.loads((source/'model.safetensors.index.json').read_text())
with safe_open(source/index['weight_map']['lm_head.weight'], framework='pt', device='cpu') as handle:
    original_rows = torch.cat([handle.get_slice('lm_head.weight')[i:i+1] for i in rows['token_ids'].tolist()])
assert torch.equal(original_rows, rows['base_rows'])

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8*1024*1024):
            value.update(block)
    return value.hexdigest()

for relative in ['adapter_vllm/adapter_model.safetensors', 'adapter_vllm/adapter_config.json', 'adapter_vllm/decision_head.json', 'calibration.json']:
    assert digest(pack/relative) == digest(source/relative), relative

def headers(folder):
    result = {}
    for path in folder.glob('model*.safetensors'):
        with path.open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(length))
        result.update({key:value for key,value in header.items() if key != '__metadata__'})
    return result

original, stored = headers(source), headers(pack)
vision = {key:value for key,value in stored.items() if key.startswith('model.visual.')}
assert vision and all(value['dtype'] == 'BF16' for value in vision.values())
assert all(key in original and value['shape'] == original[key]['shape'] for key,value in vision.items())
quantization = json.loads((pack/'quantization_config.json').read_text())
trunk = [entry for name,entry in quantization['tensor_storage'].items() if '.layers.' in name and entry.get('quant_format') == 'exl3']
assert len(trunk) == 400 and all(entry['bits_per_weight'] == 4 for entry in trunk)
assert quantization['tensor_storage']['lm_head']['bits_per_weight'] == 6
mtp_quantized = [key for key in stored if key.startswith('mtp.') and key.endswith('.trellis')]
assert len(mtp_quantized) == 8
for key in mtp_quantized:
    value = stored[key]
    source_shape = original[key.removesuffix('.trellis')+'.weight']['shape']
    assert (value['data_offsets'][1]-value['data_offsets'][0])*8/math.prod(source_shape) == 6
parameters = sum(math.prod(value['shape']) for value in original.values())
def tensor_bytes(tensors):
    return sum(value['data_offsets'][1]-value['data_offsets'][0] for value in tensors.values())
weight_bytes = tensor_bytes(stored)
manifest = json.loads((root/'quantized-pack-manifest.json').read_text())
report = {
    'source_revision': (source/'source_revision.txt').read_text().strip(),
    'source_parameters': parameters,
    'source_tensor_bytes': tensor_bytes(original),
    'packed_weight_tensor_bytes': weight_bytes,
    'packed_weight_effective_bpw': weight_bytes*8/parameters,
    'all_pack_bytes': manifest['bytes'],
    'all_pack_effective_bpw': manifest['bytes']*8/parameters,
    'trunk_quantized_modules': len(trunk),
    'trunk_bits': 4,
    'generation_head_bits': 6,
    'vision_tensors': len(vision),
    'vision_tensor_bytes': tensor_bytes(vision),
    'vision_dtype': 'BF16',
    'mtp_tensors': len([key for key in original if 'mtp' in key]),
    'mtp_policy_bits': quantization['mtp_bits'],
    'mtp_quantized_projections': len(mtp_quantized),
    'mtp_actual_trellis_bits': 6,
    'decision_rows_exact_against_source': True,
    'decision_rows': {key:{'shape':list(value.shape), 'dtype':str(value.dtype)} for key,value in rows.items()},
    'adapter_and_calibration_exact': True,
    'manifest_file_count': len(manifest['files']),
}
(root/'quantized-integrity-report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
