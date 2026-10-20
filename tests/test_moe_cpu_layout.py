"""N-major CPU compute, arena replacement and streamed GPU canonicalization."""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_cpu_layout import copy_trellis, projection_tile_order, select_tile_order


@pytest.mark.parametrize("bits", [3, 5, 8])
def test_nmajor_cpu_and_stream_roundtrip(bits):
    torch.manual_seed(173 + bits)
    k, n, experts = 512, 640, 3
    up = [torch.randint(-32768, 32767, (k // 16, n // 16, bits * 16), dtype = torch.int16)
          for _ in range(experts)]
    down = [torch.randint(-32768, 32767, (n // 16, k // 16, bits * 16), dtype = torch.int16)
            for _ in range(experts)]
    layouts = []
    for order in (0, 2):
        mats = []
        for src in up + down:
            dst = torch.empty_like(src)
            # The same helper writes newly loaded experts and replaces arena weights.
            copy_trellis(dst, torch.zeros_like(src), order)
            copy_trellis(dst, src, order)
            mats.append(dst)
        layouts.append(mats)
    scales_k = [torch.full((k,), 0.02, dtype = torch.float16) for _ in range(experts)]
    scales_n = [torch.full((n,), 0.02, dtype = torch.float16) for _ in range(experts)]
    handles = [ext.exl3_moe_cpu_make_layer(
        [], [], [], mats[:experts], scales_k, scales_n,
        mats[experts:], scales_n, scales_k, [], [], [], 2, 0.0, order)
        for mats, order in zip(layouts, (0, 2))]
    try:
        for rows in (1, 3):
            x = torch.randn(rows, k).half()
            indices = torch.tensor([[0, 2]] * rows, dtype = torch.int32)
            weights = torch.full((rows, 2), 0.5, dtype = torch.float16)
            for threads in (1, 4):
                outputs = []
                for handle in handles:
                    out = torch.empty(rows, k)
                    ext.exl3_moe_cpu_forward(handle, x, indices, weights, out, threads)
                    assert torch.isfinite(out).all()
                    outputs.append(out)
                assert torch.equal(outputs[0].view(torch.int32), outputs[1].view(torch.int32))
    finally:
        for handle in handles:
            ext.exl3_moe_cpu_free_layer(handle)

    if not torch.cuda.is_available():
        return
    # Three expert payloads, two projections with unequal K/N and offsets.
    # Untouched prefix/suffix sentinels catch a bad projection stride.
    payloads = []
    for e in range(experts):
        payloads.append(torch.cat((layouts[1][e].flatten(), layouts[1][experts + e].flatten())))
    src = torch.cat(payloads).cuda()
    dst = torch.full_like(src, 123)
    ub = up[0].numel() * 2
    db = down[0].numel() * 2
    ext.moe_unswizzle_trellis(src, dst, experts, ub + db, 0, k // 16, n // 16, bits, 2)
    expected = torch.cat([torch.cat((up[e].flatten(), torch.full_like(down[e].flatten(), 123)))
                          for e in range(experts)])
    assert torch.equal(dst.cpu(), expected)
    ext.moe_unswizzle_trellis(src, dst, experts, ub + db, ub, n // 16, k // 16, bits, 2)
    expected = torch.cat([torch.cat((up[e].flatten(), down[e].flatten())) for e in range(experts)])
    assert torch.equal(dst.cpu(), expected)


def test_tile_order_selection():
    assert select_tile_order(ext, "0") == 0
    assert select_tile_order(ext, "2") == 2
    assert select_tile_order(ext, "2", False) == 0
    assert projection_tile_order(1, 8) == 0
    assert projection_tile_order(2, 8) == 2
    with pytest.raises(ValueError):
        select_tile_order(ext, "unknown")
