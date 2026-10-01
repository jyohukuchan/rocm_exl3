"""GPU regression checks for reusable sparse-attention score workspaces.

Opt in with EXL3_GPU_TESTS=1 in the ROCm environment; ordinary CPU test runs
must not allocate a GPU or compile the extension.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("EXL3_GPU_TESTS") != "1",
                                reason="GPU regression tests are opt-in")


@pytest.mark.parametrize("width", [750, 16384])
def test_live_score_view_excludes_previous_request_tail(width):
    import torch
    from exllamav3.ext import exllamav3_ext as ext

    scores = torch.full((2, 32768), 100., dtype=torch.float16, device="cuda:0")
    scores[:, :width] = torch.linspace(0., 1., width, device=scores.device).half()
    selected = torch.empty((2, 512), dtype=torch.int32, device=scores.device)
    ext.dsa_topk(scores.narrow(1, 0, width), selected, 512, None, 0)
    torch.cuda.synchronize()
    assert bool(((selected >= 0) & (selected < width)).all())
    # Every selected entry beats the first entry outside the last 512, allowing
    # fp16 ties at the cutoff for the larger split-topk test.
    selected_scores = scores.gather(1, selected.long())
    cutoff = scores[0, width - 512]
    assert bool((selected_scores >= cutoff).all())


def test_pool_expansion_masks_invisible_selections_per_query():
    import torch
    from exllamav3.modules.attention_fn.dsa_triton import _dsa_pool_expand_kernel

    pools = torch.tensor([[0, 1, 1000, -1], [0, 1, 1000, -1]],
                         dtype=torch.int32, device="cuda:0")
    expanded = torch.empty((2, 32), dtype=torch.int32, device=pools.device)
    _dsa_pool_expand_kernel[(2, 1)](
        pools, expanded, 9, P=4, SEL=4, K_pad=32, KP_pool=4, TAIL=True,
        SEQ=2, MULTIROW=0, BLOCK=256)
    torch.cuda.synchronize()
    visible = torch.tensor([10, 11], device=pools.device)[:, None]
    assert bool(((expanded >= 0) & (expanded < visible) | (expanded == -1)).all())
    assert torch.equal(expanded[:, :8], torch.arange(8, device=pools.device).expand(2, -1))
    assert bool((expanded[:, 8:12] == -1).all())
    assert torch.equal(expanded[0, 16:18], torch.tensor([8, 9], device=pools.device))
    assert torch.equal(expanded[1, 16:19], torch.tensor([8, 9, 10], device=pools.device))
