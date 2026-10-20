"""Small Q2 regression: LDS admission, cyclic states and decoded weights."""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason = "requires ROCm")


@pytest.mark.parametrize("bits,length", [(2, 256), (2, 160), (3, 256), (4, 256)])
def test_quantizer_global_costs(monkeypatch, bits, length):
    monkeypatch.setenv("EXL3_QT_OPTIMIZED", "0")
    g = torch.Generator().manual_seed(27)
    x = torch.randn(2, length, generator = g).cuda()
    edges = 65536 >> bits
    costs = torch.empty(2, 2, edges, dtype = torch.float16, device = "cuda")
    history = torch.empty(2, length, edges, dtype = torch.int16, device = "cuda")
    outputs = []
    for _ in range(2):
        q = torch.empty_like(x)
        idx = torch.empty_like(x, dtype = torch.int16)
        ext.quantize_tiles(x, q, idx, costs, history, bits, False, True)
        outputs.append((q.cpu(), idx.cpu()))
    q, idx = outputs[0]
    assert torch.equal(q, outputs[1][0]) and torch.equal(idx, outputs[1][1])
    states = idx.long() & 65535
    # The next state's predecessor equals this state's low 16-K bits, including wraparound.
    assert torch.equal(torch.roll(states, -1, 1) >> bits, states & (edges - 1))
    product = (states * 0x83DCD12D) & 0xffffffff
    byte_sum = sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
    ref = ((887 * byte_sum - 452608).double() / 131072).half().float()
    assert torch.equal(q.view(torch.int32), ref.view(torch.int32))
    assert torch.isfinite(q).all()
    assert ((q - x.cpu()) ** 2).mean() < x.cpu().square().mean()
