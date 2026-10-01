"""Exact GPU comparisons against the original full-raw QSA pooling kernel."""
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("EXL3_GPU_TESTS") != "1",
                                reason="GPU regression tests are opt-in")


@pytest.mark.parametrize("device", [0, 1])
@pytest.mark.parametrize("warps", [2, 4])
def test_compact_pool_is_bit_identical_through_wrap_and_rejection(device, warps):
    import torch
    from exllamav3.modules.attention_fn.qsa_triton import (
        _qsa_pool_update_kernel, _qsa_pool_update_compact_kernel,
        _qsa_raw_append_compact_kernel,
    )
    from exllamav3.modules.attention_fn.mla_triton import _mla_plane_update_kernel

    dev = f"cuda:{device}"
    torch.cuda.set_device(device)
    torch.manual_seed(481)
    bsz, pages, page, d, p, window = 2, 32, 256, 128, 4, 16
    bt = torch.randperm(bsz * pages, device=dev).int().reshape(bsz, pages)
    raw = torch.zeros((bsz * pages, page, d), dtype=torch.half, device=dev)
    tail = torch.zeros((bsz * pages, window, d), dtype=torch.half, device=dev)
    pooled = torch.zeros((bsz * pages, page // p, d), dtype=torch.half, device=dev)
    compact = torch.zeros_like(pooled)
    norm = (torch.randn(d, device=dev) * .05).half()
    freq = 1. / (10000000. ** (torch.arange(0, 64, 2, device=dev).float() / 64))
    pos = torch.zeros(bsz, dtype=torch.int32, device=dev)
    # Starts / ends inside pools, page crossings, many wraps per prefill,
    # mixed sequence positions, and discarded verification tokens.
    rounds = [(1, 0), (2, 0), (17, 0), (237, 0), (513, 0), (2048, 0),
              (5, 0), (1, 4), (8, 0), (3, 3), (4, 1)]
    for n, rewind in rounds:
        if rewind:
            pos -= torch.tensor([rewind, max(0, rewind - 1)], device=dev, dtype=pos.dtype)
        fresh = torch.randn((bsz, n, d), device=dev).half()
        _mla_plane_update_kernel[(bsz * n,)](
            fresh, raw, bt, pos, pages, n, page_size=page, D=d, DST_D=0, DST_OFF=0)
        kw = dict(page_size=page, P=p, D=d, ROPE_R=64, attn_factor=1.,
                  eps=1e-6, MAXPOOLS=1, num_warps=warps, num_stages=1)
        _qsa_pool_update_kernel[(bsz, n // p + 1)](
            raw.view(-1, d), pooled.view(-1, d), norm, freq, bt, pos, pages, n, **kw)
        _qsa_pool_update_compact_kernel[(bsz, n // p + 1)](
            tail.view(-1, d), compact.view(-1, d), norm, freq, bt, pos, pages, n,
            fresh, RAW_ROWS=window, **kw)
        _qsa_raw_append_compact_kernel[(bsz * n,)](
            fresh, tail, bt, pos, pages, n, page_size=page, D=d, RAW_ROWS=window)
        torch.cuda.synchronize(device)
        assert torch.equal(pooled.view(torch.int16), compact.view(torch.int16)), (n, rewind)
        assert torch.isfinite(compact).all()
        pos += n


@pytest.mark.parametrize("device", [0, 1])
def test_compact_cache_layout_copy_and_history(monkeypatch, device):
    import torch
    from exllamav3.cache import Cache, CacheLayer_quant
    from exllamav3.cache.qsa import CacheLayer_qsa_quant
    torch.cuda.set_device(device)

    attention = SimpleNamespace(num_kv_heads=2, head_dim=256,
                                qsa_indexer=SimpleNamespace(head_dim=128, compress_ratio=4))
    monkeypatch.delenv("EXL3_QSA_FULL_RAW", raising=False)
    layer = CacheLayer_qsa_quant(None, attention, 1, 1024, 5, 4, raw_history=4)
    larger = CacheLayer_qsa_quant(None, attention, 1, 1024, 5, 4, raw_history=48)
    assert layer.raw_rows == 16 and larger.raw_rows == 64
    assert layer.tp_export(None)["args"]["raw_rows"] == 16
    layer.alloc(torch.device(f"cuda:{device}"))
    layer.raw_k[0].copy_(torch.arange(16 * 128, device=layer.device).reshape(16, 128))
    layer.pooled[0].fill_(.25)
    layer.copy_page(layer, 0, 1, 256)
    assert torch.equal(layer.raw_k[0], layer.raw_k[1])
    assert torch.equal(layer.pooled[0], layer.pooled[1])
    layer.copy_page(layer, 0, 2, 28)
    assert torch.equal(layer.pooled[0, :7], layer.pooled[2, :7])
    with pytest.raises(AssertionError, match="pool boundary"):
        layer.copy_page(layer, 0, 2, 27)
    monkeypatch.setenv("EXL3_QSA_FULL_RAW", "1")
    full = CacheLayer_qsa_quant(None, attention, 1, 1024, 5, 4)
    assert full.raw_rows == 256
    assert full.storage_size() - layer.storage_size() == 4 * (256 - 16) * 128 * 2
    monkeypatch.delenv("EXL3_QSA_FULL_RAW", raising=False)
    attention.layer_idx = 0
    attention.cache_layers = []
    attention.cache_layer_type = lambda cls, kwargs: (CacheLayer_qsa_quant, kwargs)
    model = SimpleNamespace(config=None, cache_weakrefs={}, recurrent_state_cls=None,
                            get_cache_layers=lambda: [attention], get_recurrent_layers=lambda: [],
                            get_layer_instances=lambda i: [(i, 0)])
    cache = Cache(model, 1024, layer_type=CacheLayer_quant, k_bits=5, v_bits=4, max_history=48)
    assert cache.layers[0, 0].raw_rows == 64
    assert cache.prefix_alignment == 4
    assert cache.qsa_raw_window == 64
