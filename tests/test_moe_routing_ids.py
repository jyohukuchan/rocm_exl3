"""Invalid router picks must land in the MoE sentinel bucket, never in bincount.

Regression test for the KLD dump crash (en_prose_129): a negative expert id
reached ``torch.bincount`` in the grouped INT8 MoE path and aborted the run
with "bincount only supports 1-d non-negative integral inputs". Out-of-range
picks (negative or >= num_experts) must map to the trailing sentinel bucket so
they contribute exact zeros. Loads the helper by file path so the test runs on
a CPU host without the CUDA/ROCm extension toolchain.
"""
import importlib.util
import os
import torch


def _load_helper():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "exllamav3", "modules", "moe_routing_ids.py")
    spec = importlib.util.spec_from_file_location("moe_routing_ids", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_invalid_picks_map_to_sentinel_replicated():
    mod = _load_helper()
    flat = torch.tensor([-1, 0, 3, 4, 127, 128], dtype=torch.long)
    local = mod.localize_expert_ids(flat, 128, 128, None)
    assert local.tolist() == [128, 0, 3, 4, 127, 128]
    # The downstream bincount must accept the mapped ids.
    counts = torch.bincount(local, minlength=128 + 1)
    assert len(counts) == 129
    assert counts[128].item() == 2


def test_sharded_picks_offset_and_sentinel():
    mod = _load_helper()
    # Global ids 10..13 on a shard holding [10, 11] (routing_first=10).
    flat = torch.tensor([9, 10, 11, 12], dtype=torch.long)
    local = mod.localize_expert_ids(flat, 2, 128, 10)
    assert local.tolist() == [2, 0, 1, 2]
