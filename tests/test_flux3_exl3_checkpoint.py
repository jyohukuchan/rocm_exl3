"""Real FLUX.3 graph, conditional-branch capture and EXL3 checkpoint reload."""

import json

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "Flux3Transformer2DModel"):
    pytest.skip("requires pinned FLUX.3 Diffusers", allow_module_level=True)

from exllamav3.diffusion.calibration import ActivationCapture
from exllamav3.diffusion.checkpoint import convert_transformer, load_transformer
from exllamav3.diffusion.linear import EXL3DiffusionLinear


@pytest.mark.parametrize("masked", [False, True])
@torch.inference_mode()
def test_query_sliced_attention_matches_native(masked):
    from diffusers.models.transformers.transformer_flux3 import Flux3Attention, Flux3PosEmbed
    from exllamav3.diffusion.flux3_attention import Flux3QuerySlicedAttnProcessor

    torch.manual_seed(42)
    attention = Flux3Attention(128, heads=2, dim_head=64).eval()
    hidden = torch.randn(1, 23, 128)
    rope = Flux3PosEmbed(theta=10000, axes_dim=[16, 16, 16, 16])(torch.randn(23, 4))
    mask = torch.ones(1, 1, 23, 23, dtype=torch.bool) if masked else None
    if masked:
        mask[..., 2::3] = False
    reference = attention(hidden, attention_mask=mask, image_rotary_emb=rope)
    attention.set_processor(Flux3QuerySlicedAttnProcessor(query_chunk_size=7))
    actual = attention(hidden, attention_mask=mask, image_rotary_emb=rope)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU quantizer")
@torch.inference_mode()
def test_flux3_conditional_checkpoint(tmp_path):
    torch.manual_seed(53)
    model = diffusers.Flux3Transformer2DModel(
        num_layers=1, num_single_layers=1, num_attention_heads=1,
        attention_head_dim=128, joint_attention_dim=128, timestep_channels=128,
        mlp_ratio=1, flow_map=True,
    ).to(device="cuda", dtype=torch.bfloat16).eval()
    original = tmp_path / "original"
    original.mkdir()
    source = {"repo_id": "test/tiny-flux3", "revision": "test-only"}
    (original / "exl3_source.json").write_text(json.dumps(source))
    model.save_pretrained(original / "transformer")
    inputs = {
        "hidden_states": torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16),
        "encoder_hidden_states": torch.randn(1, 3, 128, device="cuda", dtype=torch.bfloat16),
        "timestep": torch.tensor([0.8], device="cuda", dtype=torch.bfloat16),
        "target_timestep": torch.tensor([0.5], device="cuda", dtype=torch.bfloat16),
        "img_ids": torch.randn(4, 4, device="cuda"),
        "txt_ids": torch.randn(3, 4, device="cuda"), "return_dict": False,
    }
    captures = []
    for branch in ("text", "reference"):
        if branch == "reference":
            inputs.update(image_cond_hidden_states=torch.randn(1, 2, 128, device="cuda", dtype=torch.bfloat16),
                          image_cond_ids=torch.randn(2, 4, device="cuda"))
        capture = ActivationCapture(model, [0], rows_per_call=8, allow_uncalled=True)
        model(**inputs)
        capture.close()
        directory = tmp_path / branch
        capture.save(directory, {"source": source})
        captures.append(directory)
    # Text-only calibration is a valid partial capture, but cannot convert the reference stream alone.
    with pytest.raises(ValueError, match="all eligible"):
        convert_transformer(original, tmp_path / "incomplete", captures[:1])
    destination = tmp_path / "checkpoint"
    manifest = convert_transformer(original, destination, captures, device="cuda:0")
    assert manifest["transformer_type"] == "Flux3Transformer2DModel"
    assert len(manifest["layers"]) == 24
    restored, _ = load_transformer(destination, device="cuda:0", workspace_bytes=128 * 128 * 4)
    assert sum(isinstance(m, EXL3DiffusionLinear) for m in restored.modules()) == 24
    assert not any(t.is_meta for t in (*restored.parameters(), *restored.buffers()))
    output = restored(**inputs)[0]
    reference = model(**inputs)[0]
    assert output.shape == reference.shape == (1, 4, 128)
    assert torch.isfinite(output).all()
    assert (output.float() - reference.float()).norm() / reference.float().norm() < 0.2
    again, _ = load_transformer(destination, device="cuda:0", workspace_bytes=128 * 128 * 4)
    torch.testing.assert_close(again(**inputs)[0], output, atol=0, rtol=0)
    from rocm_tools.flux3 import make_pipeline
    from exllamav3.diffusion.metrics import PredictionNMSE

    # Exercise the actual 16-step modular flow-map loop, rather than just a forward call.
    vae = diffusers.AutoencoderKLFlux2(
        block_out_channels=(32, 32, 32, 32), layers_per_block=1, latent_channels=32,
        sample_size=32, norm_num_groups=32,
    ).to(device="cuda", dtype=torch.float32).eval()
    metric = PredictionNMSE(restored.forward)
    handle = model.register_forward_hook(metric.observe, with_kwargs=True)
    pipe = make_pipeline(model, vae, "cuda:0")
    result = pipe(prompt_embeds=inputs["encoder_hidden_states"], width=32, height=32,
                  num_inference_steps=16, generator=torch.Generator("cpu").manual_seed(21),
                  output=["images", "latents"])
    handle.remove()
    assert result["images"][0].size == (32, 32) and torch.isfinite(result["latents"]).all()
    measured = metric.report()
    assert len(measured["steps"]) == 16 and 0 <= measured["nmse"] < 0.1
    assert measured["steps"][-1]["target_timestep"] == [0.0]
