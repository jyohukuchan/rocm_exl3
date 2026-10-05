"""Numerical checks for PEFT VL names, local projection slices and adapter isolation."""
import json
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from exllamav3.model.lora import LoRA
from exllamav3.modules.linear import Linear


def make_linear(key, *, i=6, o=8, fi=None, fo=None, start_i=0, start_o=0, alt=None):
    module = Linear(None, key, i, o, full_in_features=fi, full_out_features=fo,
                    first_in_feature=start_i, first_out_feature=start_o, alt_key=alt, pad_to=8)
    module.device = torch.device('cpu')
    return module


def adapter(tmp_path, weights):
    (tmp_path / 'adapter_config.json').write_text(json.dumps({'r': 2, 'lora_alpha': 4}))
    save_file(weights, tmp_path / 'adapter_model.safetensors')
    return str(tmp_path)


def test_sliced_vl_lora_matches_unsliced_peft_and_isolates_base(tmp_path):
    torch.manual_seed(42)
    a, b = torch.randn(2, 6), torch.randn(16, 2)
    path = 'model.language_model.layers.0.mlp.up_proj'
    targets = [make_linear(path + f'.slice.{j}', o=8, fi=6, fo=16, start_o=j*8, alt=path)
               for j in range(2)]
    model = SimpleNamespace(modules_dict={m.key: m for m in targets}, loaded_tp=False)
    directory = adapter(tmp_path, {'base_model.model.model.layers.0.mlp.up_proj.lora_A.weight': a,
                                   'base_model.model.model.layers.0.mlp.up_proj.lora_B.weight': b})
    lora = LoRA.from_directory(model, directory, strict=True, dtype=torch.float32)
    x = torch.randn(1, 3, 6)
    padded = torch.nn.functional.pad(x, (0, 2))
    out = []
    for m in targets:
        y = torch.zeros(1, 3, 8)
        m.apply_lora(padded, y, {'loras': (lora,)})
        out.append(y)
        base = torch.zeros_like(y)
        m.apply_lora(padded, base, {'loras': ()})
        assert torch.count_nonzero(base) == 0
    torch.testing.assert_close(torch.cat(out, -1), x @ a.T @ b.T * 2)
    lora.unload()
    assert all(not m.lora_a_tensors for m in targets)


def test_input_slices_sum_to_full_down_projection(tmp_path):
    torch.manual_seed(12)
    a, b = torch.randn(2, 12), torch.randn(6, 2)
    path = 'model.language_model.layers.0.mlp.down_proj'
    targets = [make_linear(path + f'.slice.{j}', i=6, o=6, fi=12, fo=6, start_i=j*6, alt=path)
               for j in range(2)]
    model = SimpleNamespace(modules_dict={m.key: m for m in targets}, loaded_tp=False)
    lora = LoRA.from_directory(model, adapter(tmp_path, {
        'base_model.model.model.layers.0.mlp.down_proj.lora_A.weight': a,
        'base_model.model.model.layers.0.mlp.down_proj.lora_B.weight': b}), strict=True, dtype=torch.float32)
    x = torch.randn(1, 3, 12)
    y = torch.zeros(1, 3, 6)
    for j, m in enumerate(targets):
        part = torch.zeros_like(y)  # also exercises trimmed, padded outputs
        m.apply_lora(torch.nn.functional.pad(x[..., j*6:(j+1)*6], (0,2)), part)
        y += part
    torch.testing.assert_close(y, x @ a.T @ b.T * 2)
    lora.enabled = False
    before = part.clone()
    m.apply_lora(torch.ones(1, 3, 8), part)
    torch.testing.assert_close(part, before)


def test_unmatched_strict_adapter_is_transactional(tmp_path):
    m = make_linear('lm_head', fi=6, fo=8)
    model = SimpleNamespace(modules_dict={m.key: m}, loaded_tp=False)
    weights = {f'base_model.model.{name}.lora_{half}.weight': value
               for name in ('lm_head', 'missing')
               for half, value in [('A', torch.ones(2,6)), ('B', torch.ones(8,2))]}
    with pytest.raises(ValueError, match='Unmatched'):
        LoRA.from_directory(model, adapter(tmp_path, weights), strict=True)
    assert not m.lora_a_tensors and not m.lora_b_tensors
