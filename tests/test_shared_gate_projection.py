# GPU regression tests for the fused shared-gate projection ext.add_sigmoid_gate_proj.
#
# Reference is an independent torch FP64 dot -> sigmoid -> weighted add on the same
# half inputs/weights. Background (RDNA2, Qwen3.8 MoE shared expert gate):
# block_reduce_sum_broadcast_f (reduction.cuh) ran warp_reduce_sum_f (shfl_down, so
# only lane 0 holds the full sum) in warp 0 and then let EVERY lane write shared[0].
# The data race broadcast a partial sum on RDNA2: add_sigmoid_gate_proj(ones, ones,
# zeros, w=1/N) returned 0.5 at N=128 and 0.68993 at N=2560 instead of
# sigmoid(1) = 0.7310585786300049 (shared-gate-synthetic-baseline.json); the first
# real Qwen3.8 MoE row gave a gate scalar of 0.20132904 instead of 0.37406739
# (shared-gate-probe.json), explaining ~100% of the routed+shared MoE output error.
# N=1024/2048/4096 hid the bug for uniform input, so the suite also uses non-uniform
# random rows. The fused path serves rows <= 32 (larger batches use normalprojection
# and were always correct), so row counts stay within that range.

import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not torch.cuda.is_available():
    pytest.skip("add_sigmoid_gate_proj is a GPU kernel; skip on CPU without importing exllamav3", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext

device = "cuda:0"


def fp64_reference(x, y, z0, w):
    # z = z0 + x * sigmoid(y @ w), with dot/sigmoid/add computed in FP64 from the same half y/w
    gate = torch.sigmoid(y.double() @ w.double().reshape(-1))
    return z0.double() + x.double() * gate.unsqueeze(-1)


def run_kernel(x, y, z0, w):
    z = z0.clone()
    ext.add_sigmoid_gate_proj(x, y, z, w)  # in-place: z <- z + x * sigmoid(y @ w)
    return z


@pytest.mark.parametrize("dim", [128, 2048, 2560])
@torch.inference_mode()
def test_uniform_gate_baseline(dim):
    # Synthetic baseline: x=1, y=1, w=half(1/dim) makes the dot approximately 1.
    # Account for half rounding (notably at dim2560). Dims 128/2560 fail on the race;
    # 2048 is the baseline shape where uniform input previously hid it.
    rows = 4
    x = torch.ones(rows, dim, dtype = torch.float, device = device)
    y = torch.ones(rows, dim, dtype = torch.half, device = device)
    z = torch.zeros(rows, dim, dtype = torch.float, device = device)
    w = torch.full((dim, 1), 1.0 / dim, dtype = torch.half, device = device)
    out = run_kernel(x, y, z, w)
    torch.testing.assert_close(out, fp64_reference(x, y, z, w).float(), rtol = 1e-5, atol = 1e-6)


@pytest.mark.parametrize("dim", [128, 2048, 2560])
@torch.inference_mode()
def test_random_rows_match_fp64(dim):
    # Non-uniform rows defeat the uniform-input cancellation; nonzero z0 exercises the
    # weighted add; 32 rows covers the fused-path batch sizes (rows 1/8/16/32).
    torch.manual_seed(1234 + dim)
    rows = 32
    x = torch.randn(rows, dim, dtype = torch.float, device = device)
    y = torch.randn(rows, dim, dtype = torch.half, device = device)
    z0 = torch.randn(rows, dim, dtype = torch.float, device = device)
    w = (torch.randn(dim, 1, dtype = torch.float, device = device) * 0.05).to(torch.half)
    out = run_kernel(x, y, z0, w)
    ref = fp64_reference(x, y, z0, w).to(torch.float)
    torch.testing.assert_close(out, ref, rtol = 1e-4, atol = 1e-4)
    assert not torch.equal(out, z0), "gate contribution was skipped on every row"


@pytest.mark.parametrize("dim", [128, 2560])
@torch.inference_mode()
def test_repeated_calls_are_deterministic(dim):
    # The shared[0] race could also surface as launch-to-launch nondeterminism;
    # identical inputs must produce bit-identical outputs, matching the fp64 reference.
    torch.manual_seed(987 + dim)
    rows = 8
    x = torch.randn(rows, dim, dtype = torch.float, device = device)
    y = torch.randn(rows, dim, dtype = torch.half, device = device)
    z0 = torch.randn(rows, dim, dtype = torch.float, device = device)
    w = (torch.randn(dim, 1, dtype = torch.float, device = device) * 0.05).to(torch.half)
    first = run_kernel(x, y, z0, w)
    ref = fp64_reference(x, y, z0, w).to(torch.float)
    torch.testing.assert_close(first, ref, rtol = 1e-4, atol = 1e-4)
    for _ in range(9):
        assert torch.equal(run_kernel(x, y, z0, w), first)
