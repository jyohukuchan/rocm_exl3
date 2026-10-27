"""igemm_recon must be a bit-exact int8->int32 row-major GEMM.

Run with EXL3_GPU_TESTS=1 and a built extension on the path, e.g.
PYTHONPATH=/src/w8a8_lib EXL3_GPU_TESTS=1 pytest <this file>.
"""
import os
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("EXL3_GPU_TESTS") != "1",
                                reason="GPU regression tests are opt-in")


@pytest.mark.parametrize("device", [0, 1])
@pytest.mark.parametrize("shape", [(32, 64, 48), (513, 2560, 5120), (256, 640, 256)])
def test_igemm_recon_exact(device, shape):
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    if getattr(ext, "igemm_recon", None) is None:
        pytest.skip("extension without igemm_recon")
    torch.cuda.set_device(device)
    torch.manual_seed(1234)
    dev = f"cuda:{device}"
    m, k, n = shape
    a8 = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=dev)
    b8 = torch.randint(-128, 127, (k, n), dtype=torch.int8, device=dev)
    c32 = torch.empty((m, n), dtype=torch.int32, device=dev)
    ext.igemm_recon(a8, b8, c32)
    torch.cuda.synchronize(device)
    # Exact CPU int64 reference narrowed to int32 (products fit int32 for
    # every EXL3 shape: K * 127 * 127 < 2**31).
    ref = (a8.cpu().to(torch.int64) @ b8.cpu().to(torch.int64)).to(torch.int32)
    assert torch.equal(c32.cpu(), ref)


def test_igemm_recon_rejects_non_contiguous():
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    if getattr(ext, "igemm_recon", None) is None:
        pytest.skip("extension without igemm_recon")
    dev = "cuda:0"
    a8 = torch.randint(-128, 127, (16, 32), dtype=torch.int8, device=dev)
    b8 = torch.randint(-128, 127, (32, 24), dtype=torch.int8, device=dev).t()
    assert not b8.is_contiguous()
    c32 = torch.empty((16, 24), dtype=torch.int32, device=dev)
    with pytest.raises(Exception):
        ext.igemm_recon(a8, b8, c32)
