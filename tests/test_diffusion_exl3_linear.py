"""GPU smoke tests: bounded reconstruction, BF16 range and serialization."""

import io

import pytest
import torch

from exllamav3.diffusion import EXL3DiffusionLinear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires EXL3 GPU extension")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [3, 4, 5])
@torch.inference_mode()
def test_sliced_linear_roundtrip_and_dtype(dtype, bits):
    from exllamav3.ext import exllamav3_ext as ext

    torch.manual_seed(123)
    k, n = 256, 384
    t = torch.randint(0, 65536, (k // 16, n // 16, 16 * bits), device="cuda", dtype=torch.int32).short()
    su = (torch.rand(k, device="cuda") + 0.5).half()
    sv = (torch.rand(n, device="cuda") + 0.5).half()
    bias = torch.randn(n, device="cuda", dtype=dtype)
    layer = EXL3DiffusionLinear(k, n, t, su.clone(), sv.clone(), bias,
                                workspace_bytes=4 * k * 128).to(dtype=dtype)
    assert torch.equal(layer.suh, su) and torch.equal(layer.svh, sv)
    assert layer.suh.dtype == torch.float16
    w = torch.empty(k, n, device="cuda", dtype=torch.float16)
    ext.reconstruct_had_slice(w, t, su, sv, bits, False, False, 0)
    x = torch.randn(2, 3, k, device="cuda", dtype=dtype)
    if dtype == torch.bfloat16:
        # These inputs cannot be represented in FP16, even though BF16 GEMM is valid.
        x *= 100000
    expected = torch.nn.functional.linear(x, w.to(dtype).T, bias)
    actual = layer(x)
    assert actual.dtype == dtype and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.02 if dtype == torch.float16 else 8192,
                               rtol=0.004 if dtype == torch.float16 else 0.02)
    stream = io.BytesIO()
    torch.save(layer.state_dict(), stream)
    stream.seek(0)
    restored = EXL3DiffusionLinear(k, n, torch.zeros_like(t), torch.zeros_like(su),
                                   torch.zeros_like(sv), torch.zeros_like(bias),
                                   workspace_bytes=4 * k * 128)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(restored(x), actual, atol=0, rtol=0)
    assert set(layer.state_dict()) == {"trellis", "suh", "svh", "bias"}


def test_workspace_minimum_and_shape_validation():
    k, n = 256, 384
    kwargs = dict(in_features=k, out_features=n,
                  trellis=torch.zeros(k // 16, n // 16, 64, dtype=torch.int16),
                  suh=torch.ones(k, dtype=torch.float16), svh=torch.ones(n, dtype=torch.float16))
    with pytest.raises(ValueError, match="workspace"):
        EXL3DiffusionLinear(**kwargs, workspace_bytes=4 * k * 128 - 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires EXL3 GPU quantizer")
@torch.inference_mode()
def test_calibrated_conversion_on_held_out_inputs():
    from exllamav3.diffusion.quantize import quantize_linear

    torch.manual_seed(171)
    original = torch.nn.Linear(128, 256, device="cuda", dtype=torch.bfloat16)
    calibration = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    held_out = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    quantized, stats = quantize_linear(original, calibration, bits=4, device="cuda:0")
    reference = original(held_out).float()
    output = quantized(held_out).float()
    relative_l2 = ((output - reference).norm() / reference.norm()).item()
    assert torch.isfinite(output).all()
    assert relative_l2 < 0.12, relative_l2
    assert stats["quantized_state_bytes"] < stats["source_weight_bytes"] * 0.4
    assert stats["calibration_rows"] == 512
    assert 0 <= stats["proxy_error"] < 0.02


@torch.inference_mode()
def test_capture_selects_timesteps_and_preserves_provenance(tmp_path):
    import json
    from safetensors.torch import load_file
    from exllamav3.diffusion.calibration import ActivationCapture

    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(128, 128)])
    capture = ActivationCapture(model, [0, 2], rows_per_call=4)
    for step in range(3):
        capture.step = step
        model.transformer_blocks[0](torch.full((8, 128), float(step + 1)))
    capture.close()
    capture.save(tmp_path, {"source": "test", "seed": 17})
    manifest = json.loads((tmp_path / "calibration.json").read_text())
    layer = manifest["layers"]["transformer_blocks.0"]
    data = load_file(tmp_path / layer["file"])["inputs"]
    assert data.shape == (8, 128)
    assert torch.equal(data[:4], torch.ones(4, 128))
    assert torch.equal(data[4:], torch.full((4, 128), 3.0))
    assert [call["step"] for call in layer["calls"]] == [0, 2]
    assert manifest["provenance"] == {"source": "test", "seed": 17}
