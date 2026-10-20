"""Changing compressed-word supply or physical waves must preserve FP32 bits."""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason = "requires ROCm")


@pytest.mark.parametrize("bits", [3, 4, 6, 8])
@pytest.mark.parametrize("rows", [1, 3])
def test_word_tile_and_physical_waves(monkeypatch, bits, rows):
    monkeypatch.setenv("EXL3_GEMV_MAX_M", "8")
    monkeypatch.setenv("EXL3_GEMV_MR_M1", "1")
    monkeypatch.delenv("EXL3_GEMV_SPLITK_WARPS", raising = False)
    g = torch.Generator().manual_seed(32 + bits)
    k, n = 4096, 512  # baseline has 16 logical K partitions
    a = torch.randn(rows, k, generator = g).half().cuda()
    b = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), generator = g,
                      dtype = torch.int16).cuda()
    su = torch.full((k,), 0.02, dtype = torch.float16, device = "cuda")
    sv = torch.full((n,), 0.2, dtype = torch.float16, device = "cuda")
    ah = torch.empty_like(a)
    c = torch.empty(rows, n, device = "cuda")

    def call():
        ext.exl3_gemm(a, b, c, su, ah, sv, -1, False, True, 0)

    monkeypatch.setenv("EXL3_GEMV_WORD_TILE", "0")
    monkeypatch.delenv("EXL3_GEMV_PHYSICAL_WARPS", raising = False)
    call()
    baseline = c.clone()
    assert torch.isfinite(baseline).all()
    for words, waves in [(0, 4), (1, 4), (1, 16)]:
        monkeypatch.setenv("EXL3_GEMV_WORD_TILE", str(words))
        monkeypatch.setenv("EXL3_GEMV_PHYSICAL_WARPS", str(waves))
        call()
        assert torch.equal(c.view(torch.int32), baseline.view(torch.int32))
    # Both choices are ordinary captured args; later env changes cannot mutate them.
    monkeypatch.setenv("EXL3_GEMV_PHYSICAL_WARPS", "4")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    monkeypatch.setenv("EXL3_GEMV_WORD_TILE", "0")
    monkeypatch.setenv("EXL3_GEMV_PHYSICAL_WARPS", "16")
    graph.replay()
    assert torch.equal(c.view(torch.int32), baseline.view(torch.int32))
