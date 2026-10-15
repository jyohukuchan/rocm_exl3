# SPDX-License-Identifier: MIT
"""Run the standard TP harness with bounded, untimed MoE validation/counters."""

import json
import os
import sys
import statistics
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
_VALIDATE_SCHEDULER = int(os.environ.get("EXL3_MOE_VALIDATE_SCHEDULER", "0"))
_original_fused = ext.exl3_moe
_TUNE_GROUPS = int(os.environ.get("EXL3_MOE_TUNE_GROUPS", "0"))
_SCHEDULER_TUNES = []
_FUSED_PREFILL_CALLS = 0


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
            "exact_gemm": getattr(self, "exact_gemm", False),
            "fused_rows": _STATE.owner.fused_rows,
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


def run_fused(*args):
    global _FUSED_PREFILL_CALLS
    if (
        _TUNE_GROUPS
        and len(_SCHEDULER_TUNES) < _TUNE_GROUPS
        and args[0].shape[0] >= 1024
    ):
        ordinal = _FUSED_PREFILL_CALLS
        _FUSED_PREFILL_CALLS += 1
        if ordinal in (0, 23, 46):
            counts = args[2][: args[13].numel()].tolist()
            slots = sum(c for c in counts if args[32] <= c <= args[33])
            reference = list(args)
            reference[1] = torch.zeros_like(args[1])
            reference[30] = torch.empty_like(args[30])
            expected = None
            measurements = {}
            sms = ext.g_get_num_sms(args[0].device.index)
            for width in (8, 4, 2, 1):
                capacity = min(64, sms // width)
                for index in range(5, 9):
                    reference[index] = torch.empty(
                        (capacity, *args[index].shape[1:]),
                        device=args[index].device,
                        dtype=args[index].dtype,
                    )
                for _ in range(3):
                    _original_fused(*reference)
                values = []
                for _ in range(3):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    begin.record()
                    _original_fused(*reference)
                    end.record()
                    end.synchronize()
                    values.append(begin.elapsed_time(end))
                if expected is None:
                    expected = reference[30][:slots].clone()
                assert torch.equal(reference[30][:slots], expected), (
                    "scheduler values differ",
                    width,
                )
                measurements[str(width)] = {
                    "capacity": capacity,
                    "ms": values,
                    "median_ms": statistics.median(values),
                    "exact": True,
                }
            _SCHEDULER_TUNES.append(
                {
                    "ordinal": ordinal,
                    "rows": args[0].shape[0],
                    "fused_slots": slots,
                    "measurements": measurements,
                }
            )
    check = _STATS["scheduler_exact_checks"] < _VALIDATE_SCHEDULER
    original_capacity = ext.exl3_moe_max_concurrency(args[0].device.index)
    if check and args[30] is not None and args[5].shape[0] > original_capacity:
        assert len(args) == 35, "native fused MoE signature changed"
        reference = list(args)
        reference[1] = torch.zeros_like(args[1])
        reference[30] = torch.zeros_like(args[30])
        for index in range(5, 9):
            reference[index] = args[index][:original_capacity]
        _original_fused(*reference)
        result = _original_fused(*args)
        counts = args[2][: args[13].numel()].tolist()
        slots = sum(count for count in counts if args[32] <= count <= args[33])
        assert torch.equal(
            args[30][:slots], reference[30][:slots]
        ), "fused scheduling changed expert values"
        _STATS["scheduler_exact_checks"] += 1
        _STATS["scheduler_nonempty_checks"] += int(slots > 0)
        return result
    return _original_fused(*args)


if _VALIDATE_SCHEDULER or _TUNE_GROUPS:
    ext.exl3_moe = run_fused


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
        "scheduler_tunes": list(_SCHEDULER_TUNES),
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
