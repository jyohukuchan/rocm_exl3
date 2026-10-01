"""Mapped router must preserve original picks, their order and weights exactly."""
import os
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("EXL3_GPU_TESTS") != "1",
                                reason="GPU regression tests are opt-in")


@pytest.mark.parametrize("device", [0, 1])
@pytest.mark.parametrize("rows", [1, 5, 32, 512])
@pytest.mark.parametrize("ties", [False, True])
def test_mapped_router_changes_only_storage_ids(device, rows, ties):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    torch.cuda.set_device(device)
    torch.manual_seed(145)
    dev = f"cuda:{device}"
    hidden = torch.randn((rows, 2560), device=dev).half()
    if ties:
        hidden.zero_()
    gate = torch.randn((2560, 512), device=dev).half()
    gate_t = gate.T.contiguous()
    scales = (torch.rand(512, device=dev) + .5).bfloat16()
    bias = None if ties else torch.randn(512, device=dev).half()
    ids = torch.empty((rows, 10), dtype=torch.long, device=dev)
    mapped = torch.empty_like(ids)
    weights = torch.empty((rows, 10), dtype=torch.half, device=dev)
    mapped_weights = torch.empty_like(weights)
    scores = torch.empty((rows, 512), dtype=torch.half, device=dev)
    mapped_scores = torch.empty_like(scores)
    mapping = torch.randperm(512, device=dev).int()
    ext.routing_std(hidden, gate, scores, ids, weights, scales, gate_t, bias)
    ext.routing_std_mapped(hidden, gate, mapped_scores, mapped, mapped_weights,
                           scales, gate_t, bias, mapping)
    torch.cuda.synchronize(device)
    assert torch.equal(scores, mapped_scores)
    assert torch.equal(mapped, mapping[ids].long())
    assert torch.equal(weights.view(torch.int16), mapped_weights.view(torch.int16))
