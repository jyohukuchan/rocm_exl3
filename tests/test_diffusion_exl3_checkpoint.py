"""Small real QwenImage21 graph: capture -> convert -> fresh checkpoint load."""

import json

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "QwenImage21Transformer2DModel"):
    pytest.skip("requires pinned QwenImage21 Diffusers", allow_module_level=True)

from exllamav3.diffusion.calibration import ActivationCapture
from exllamav3.diffusion.checkpoint import convert_transformer, load_transformer
from exllamav3.diffusion.linear import EXL3DiffusionLinear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU quantizer")
@torch.inference_mode()
def test_tiny_transformer_checkpoint(tmp_path):
    torch.manual_seed(32)
    model = diffusers.QwenImage21Transformer2DModel(
        num_layers=1, num_attention_heads=1, attention_head_dim=128,
        context_in_dim=128, mlp_ratio=1,
    ).to(device="cuda", dtype=torch.bfloat16).eval()
    source = {"repo_id": "test/tiny-qwen-image21", "revision": "test-only"}
    original = tmp_path / "original"
    original.mkdir()
    (original / "exl3_source.json").write_text(json.dumps(source))
    model.save_pretrained(original / "transformer")
    inputs = {
        "hidden_states": torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16),
        "encoder_hidden_states": torch.randn(1, 3, 128, device="cuda", dtype=torch.bfloat16),
        "timestep": torch.tensor([0.5], device="cuda", dtype=torch.bfloat16),
        "img_shapes": [[(1, 2, 2)]],
        "img_mask": torch.tensor([[False, False, False, True]], device="cuda"),
        "return_dict": False,
    }
    capture = ActivationCapture(model, [0], rows_per_call=16)
    model(**inputs)
    capture.close()
    calibration = tmp_path / "calibration"
    capture.save(calibration, {"source": source})
    checkpoint = tmp_path / "checkpoint"
    manifest = convert_transformer(original, checkpoint, [calibration], device="cuda:0")
    assert len(manifest["layers"]) == 7
    restored, _ = load_transformer(checkpoint, device="cuda:0", workspace_bytes=128 * 128 * 4)
    assert sum(isinstance(m, EXL3DiffusionLinear) for m in restored.modules()) == 7
    assert not any(t.is_meta for t in list(restored.parameters()) + list(restored.buffers()))
    output = restored(**inputs)[0]
    # The uncached transformer returns the joint text/image sequence; the
    # pipeline subsequently takes the last four (target image) tokens.
    assert output.shape == (1, 7, 64) and torch.isfinite(output).all()
    reloaded, _ = load_transformer(checkpoint, device="cuda:0", workspace_bytes=128 * 128 * 4)
    torch.testing.assert_close(reloaded(**inputs)[0], output, rtol=0, atol=0)
