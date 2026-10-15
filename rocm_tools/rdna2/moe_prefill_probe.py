# SPDX-License-Identifier: MIT
"""Run the standard TP harness with bounded, untimed MoE validation/counters."""

import json
import os
import sys
import threading
from collections import Counter
from pathlib import Path

import torch
from exllamav3.modules import block_sparse_mlp as sparse
from exllamav3.modules import moe_batch_recon as batch
from exllamav3.ext import exllamav3_ext as ext
from rocm_tools.rdna2 import tp_run

_STATE = threading.local()
_STATS = Counter()
_CHECKED_LAYERS = set()
_LAYOUTS = {}
_SEEN_BATCH_LAYERS = set()
_original_batch = sparse.BlockSparseMLP._run_batch_recon
_original_group = batch.BatchReconLayer.run_group
_original_audit = tp_run.tp_audit_rank
_OUTPUT = Path(sys.argv[sys.argv.index("--output") + 1])
_VALIDATE = int(os.environ.get("EXL3_MOE_VALIDATE_GROUPS", "0"))


def run_batch(self, *args, **kwargs):
    previous = getattr(_STATE, "owner", None)
    _STATE.owner = self
    try:
        return _original_batch(self, *args, **kwargs)
    finally:
        _STATE.owner = previous


def run_group(
    self, y_ext, out_ext, tok_ext, w_ext, ids, starts, counts, ptrs=None, out_slab=None
):
    result = _original_group(
        self, y_ext, out_ext, tok_ext, w_ext, ids, starts, counts, ptrs, out_slab
    )
    if id(self) not in _SEEN_BATCH_LAYERS:
        _SEEN_BATCH_LAYERS.add(id(self))
        layout = {
            "g": self.dims_g,
            "u": self.dims_u,
            "d": self.dims_d,
            "cb_g": self.cb_g,
            "cb_u": self.cb_u,
            "cb_d": self.cb_d,
            "folded": self.folded,
            "interm_fp32": self.interm_fp32,
            "fast_gather": getattr(self, "fast_gather", False),
            "fast_guad": getattr(self, "fast_guad", False),
        }
        _LAYOUTS[repr(layout)] = layout
    _STATS["groups"] += 1
    _STATS["experts"] += len(ids)
    _STATS["real_rows"] += sum(counts)
    _STATS["padded_rows"] += len(ids) * max(counts)
    _STATS["folded"] = int(self.folded)
    if (
        _STATS["checked_groups"] < _VALIDATE
        and out_slab is not None
        and _STATE.owner.key not in _CHECKED_LAYERS
    ):
        owner = _STATE.owner
        for i, (expert, start, count) in enumerate(zip(ids, starts, counts)):
            x = y_ext.index_select(0, tok_ext[start : start + count])
            yh = torch.empty((count * 2, x.shape[1]), dtype=torch.half, device=x.device)
            interm = torch.empty(
                (count * 2, owner.intermediate_size_padded),
                dtype=owner.interm_dtype,
                device=x.device,
            )
            activation = (
                interm[:count]
                if owner.interm_dtype == torch.half
                else torch.empty_like(interm[:count], dtype=torch.half)
            )
            reference = torch.empty(
                (count, owner.expert_size), dtype=torch.float, device=x.device
            )
            owner.bc.run_single_expert_dq(x, expert, yh, interm, activation, reference)
            actual = out_slab[i * max(counts) : i * max(counts) + count]
            assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
            difference = (actual - reference).abs()
            scale = reference.abs().max().clamp_min(1e-6)
            relative = float(difference.max() / scale)
            rms = float(
                difference.square().mean().sqrt()
                / reference.square().mean().sqrt().clamp_min(1e-6)
            )
            _STATS["worst_scale_relative"] = max(
                _STATS["worst_scale_relative"], relative
            )
            _STATS["worst_rms_relative"] = max(_STATS["worst_rms_relative"], rms)
            _STATS["exact_experts"] += int(torch.equal(actual, reference))
            _STATS["checked_experts"] += 1
            assert relative <= 0.005 and rms <= 0.001, (relative, rms)
        _STATS["checked_groups"] += 1
        _CHECKED_LAYERS.add(owner.key)
        _STATS["checked_layers"] = len(_CHECKED_LAYERS)
    return result


sparse.BlockSparseMLP._run_batch_recon = run_batch
batch.BatchReconLayer.run_group = run_group


def audit(local_context):
    result = _original_audit(local_context)
    snapshot = {
        "device": result["device"],
        "stats": dict(_STATS),
        "batch_recon": sparse.BATCH_RECON,
        "folded_default": batch.RECON_FOLDED,
        "layouts": list(_LAYOUTS.values()),
    }
    _OUTPUT.with_name(
        f'{_OUTPUT.stem}-moe-rank-{result["device"]}-{os.getpid()}.json'
    ).write_text(json.dumps(snapshot, indent=2) + "\n")
    return result


tp_run.tp_audit_rank = audit


def main():
    argv = sys.argv[1:]
    if "--sweep-meta" in argv:
        i = argv.index("--sweep-meta")
        del argv[i : i + 2]
    status = tp_run.main(argv)
    if _OUTPUT.exists():
        report = json.loads(_OUTPUT.read_text())
        report["moe_experiment"] = [
            json.loads(p.read_text())
            for p in sorted(_OUTPUT.parent.glob(_OUTPUT.stem + "-moe-rank-*.json"))
        ]
        report["moe_validation_groups"] = _VALIDATE
        _OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
