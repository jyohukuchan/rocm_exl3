#!/usr/bin/env python3
"""GPU probe: the EXL3_ROCM_MOE_MULTI_TOKEN slot-major MoE decode candidate.

Scope is a bounded numeric + latency PROXY on ONE real MoE block. This is not
an end-to-end improvement claim and a small-n single-layer probe is exactly
what it looks like: a gate for root to judge before adopting the route (the
frozen profile baseline -- code verify 87.4%, MoE+shared 5.9 ms/output -- is
the yardstick; the candidate remains experimental and default-OFF until real
model numbers beat it).

What is loaded: exactly one BlockSparseMLP through the module tree -- the
same "first MoE weight loading" path rocm_tools/rdna2/moe_rows_probe.py
uses. No full Model.load, no Engram/PLE tables, no cache, no calibration
corpus.

Three runs per (partition, rows), each fed the SAME y / selected_experts /
routing_weights tensors:

  ref        the v1.4.4 per-token row loop under EXL3_MGEMV=0 -- the
             cooperative kernel, the numerically validated reference (what
             masked TP R>1 falls back to today)
  orig       the v1.4.4 per-token row loop under the engine default (the
             barrier-free mgemv fast path) -- today's production behaviour
  cand       the slot-major candidate entry point when it fires, the row loop
             otherwise; rows 2..5 MUST show
             candidate_fired true (route evidence, so a silent no-op can
             never pose as a pass) and rows 1 MUST show it declined

TP simulation: shard partitions are built by SLICING the module's pointer
tables to the shard's expert range (SimML narrow views, zero copies) BEFORE
any rebasing happens -- min/max=-1/-1 plus pre-rebased local indices alone
would otherwise address the UNSLICED full-expert table and silently compute
the wrong experts. Original and candidate always run through the same shard
view. This simulates one rank's partial sum; collectives are not exercised.

Why the candidate zeroes and de-fuses when sharded: the cooperative kernel's
grouped reduce guards ``indices[...] >= 0`` and check_masked_2tok in
rocm_tools/mgemv_check.py demonstrates the pre-rebased -1-skip contract
THERE -- but that test disables MGEMV, so the barrier-free fast path's
masked numerics have NEVER been validated; this probe is that validation.
The fast path's reduce has no guard: its fallback exl3_mgemv_reduce_kernel
sums every slot row of C, and its fused epilogue's arrival counter
(red_target counts skipped slots too) never completes on a masked call and
never self-resets -- poisoning every later weighted call on the device. The
candidate therefore zeroes the gate/up/activation/down slot scratch and runs
its one weighted launch under EXL3_GEMV_FUSE_OUT=0 (guarded by a process
lock; restoring on every path, exception included). Staging is a bounded
module-owned 128-slot bundle (cleared by the patched unload); a native
bundle materializes only when the loaded cfg buffers are shorter than the
active slot count. EXL3_MGEMV=0 disables the candidate entirely: the
unchanged row loop owns that configuration.

Readouts per pair: finite, relative-L2, max-abs error, and OBSERVED bit
identity (reported, never demanded: the fast and cooperative paths are
different kernels and legitimate rounding differences are root's call).
Token classes (all picks in range / partially masked / all picks outside the
shard) are reported separately; the candidate's all-outside rows must be
exact zeros (contract check, not a tolerance). Stale-scratch exposure:
selection sets are reused recurrently (A, B, A) on one shard view and A's
two outputs must match bit for bit -- the same deterministic route rerun.
Latency: CUDA events, WARMUP warmups, SAMPLES samples of REPS repeats,
orig vs cand, ONLY for rows the candidate actually took (no invalid warm or
timed rows). CPU-side accounting records exl3_mgemm call counts (all /
weighted / weighted under EXL3_GEMV_FUSE_OUT=0 / num_tokens>1).

    PYTHONPATH=lib:/src EXL3_ROCM_MOE_MULTI_TOKEN=1 python -m \
        rocm_tools.rdna2.moe_multitoken_probe --model <exl3-dir> \
        [--device 0] [--layer-index 0] [--rows 1,2,3,4,5] -o out.json

--help is CPU-safe: torch/exllamav3 are imported lazily inside main().
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import os
import statistics
import sys
import traceback
from pathlib import Path

# Direct script execution is supported as a convenience, but append the
# repository only after PYTHONPATH entries.  The GPU command above uses
# ``python -m`` with an explicit PYTHONPATH so the selected native checkout
# remains authoritative when several /src trees exist.
_REPO_PATH = str(Path(__file__).resolve().parents[2])
if _REPO_PATH not in sys.path:
    sys.path.append(_REPO_PATH)

from rocm_tools.rdna2.common import (
    git_commit, model_fingerprint, now_utc, python_env, repo_root,
    rocm_patch_env, torch_env, visible_gpu_env, write_json,
)

PROBE_FORMAT = "rdna2-moe-multitoken-probe/1"
SEED = 1234
WARMUP, SAMPLES, REPS = 5, 5, 5
ROWS_DEFAULT = "1,2,3,4,5"
FUSE_ENV = "EXL3_GEMV_FUSE_OUT"
CONTROLLED_ENV_KEYS = (
    "EXL3_ROCM_MOE_MULTI_TOKEN", "EXL3_MGEMV", FUSE_ENV,
)
EXPECTED_MODEL = {
    "hidden_size": 2560,
    "intermediate_size": 640,
    "num_experts": 512,
    "top_k": 10,
    "expert_size": 2560,
}


def env_diff_keys(before: dict, after: dict) -> dict:
    """Return environment changes by key name only (never leak values)."""
    before_keys, after_keys = set(before), set(after)
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": sorted(k for k in before_keys & after_keys
                          if before[k] != after[k]),
    }
PROPOSED_ROUTE = (
    "rows 2..5, slots=R*top_k<=128, gfx1030, EXL3_MGEMV enabled: ONE "
    "slot-major exl3_mgemm triple (gate, up with per-slot A = "
    "repeat_interleave(y, top_k); down with num_tokens=R, flat (1,R*K) "
    "indices/weights, min/max=-1/-1). Shards pre-rebase on GPU (where "
    "in-range -> sel-min_expert else -1, positions preserved) so the "
    "barrier-free fast path accepts num_tokens>1 instead of declining to the "
    "cooperative kernel; sharded calls zero the gate/up outputs, the "
    "activation scratch and the R*K down rows (inactive slots contribute "
    "exact zeros), and that one weighted launch runs with "
    f"{FUSE_ENV}=0 (the fast-path fused reduce cannot complete with skipped "
    "slots and would poison its counters). All native args are narrowed to "
    "the active slot count; staging is a bounded module-owned 128-slot "
    "bundle, cleared by unload. Rows 1 / anything unsupported: unchanged "
    "per-token row loop. Expert allocation, top-k, routing, activation math "
    "and collectives are untouched."
)


def parse_rows(spec: str) -> list[int]:
    vals = []
    for tok in spec.split(","):
        if not tok.strip():
            continue
        n = int(tok)
        if n < 1:
            raise ValueError(f"--rows: {n} < 1")
        vals.append(n)
    if not vals:
        raise ValueError("--rows: empty")
    return sorted(set(vals))


def nonnegative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a finite non-negative number, got {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"expected a finite non-negative number, got {text!r}")
    return value


def t_sha256(t) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def pair_metrics(a, b) -> dict:
    """Cross-tensor numbers (fp32, same shape). Reported, never thresholded
    (the only hard numeric assertions are exact zeros on all-outside shard
    rows and same-route rerun determinism)."""
    d = (a - b).abs()
    scale = float(b.abs().max())
    bn, an = float(b.norm()), float(a.norm())
    import torch
    numeric_equal = bool(torch.equal(a, b))
    if a.dtype == torch.float32 and b.dtype == torch.float32:
        # ``a == b`` treats +0.0 and -0.0 as equal.  For the optional
        # bit-identity observation, compare the actual IEEE-754 bit patterns.
        abit = a.detach().contiguous().view(torch.int32)
        bbit = b.detach().contiguous().view(torch.int32)
        bit_diff = int((abit != bbit).sum())
        bitexact = bit_diff == 0
    else:
        bit_diff = int((a != b).sum())
        bitexact = numeric_equal
    return {
        "maxabs": float(d.max()),
        "rel_l2": float(d.norm() / bn) if bn > 0 else None,
        "maxabs_over_ref_scale": float(d.max()) / scale if scale > 0 else None,
        "relative_error": float(d.max()) / scale if scale > 0 else None,
        "cosine": float((a * b).sum() / (an * bn)) if an > 0 and bn > 0 else None,
        "numeric_equal": numeric_equal,
        "bitexact_observed": bitexact,
        "diff_elements": int((a != b).sum()),
        "bit_diff_elements": bit_diff,
        "elements": int(a.numel()),
    }


# ---------------------------------------------------------------------------
# TP-simulation views: pointer tables SLICED to the local expert range before
# any rebasing. narrow() offsets the table base by lo*8 bytes (view, no copy);
# a rebased local index m then resolves to the shard's own m-th expert
# weights. Scratch buffers are shared with the parent module: every run is
# sequential and clones its output, so there is no cross-run aliasing.
# ---------------------------------------------------------------------------

class SimML:
    def __init__(self, base, lo, hi):
        n = hi - lo
        self.ptrs_trellis = base.ptrs_trellis.narrow(0, lo, n)
        self.ptrs_suh = base.ptrs_suh.narrow(0, lo, n)
        self.ptrs_svh = base.ptrs_svh.narrow(0, lo, n)
        self.K = base.K
        self.mcg = base.mcg
        self.mul1 = base.mul1


class SimCfg:
    def __init__(self, base, lo, hi):
        self._base = base
        self.min_expert = lo
        self.max_expert = hi

    def __getattr__(self, name):
        return getattr(self._base, name)


class SimShard:
    def __init__(self, base, lo, hi):
        self._base = base
        self.experts_cfg = SimCfg(base.experts_cfg, lo, hi)
        self.multi_gate = SimML(base.multi_gate, lo, hi)
        self.multi_up = SimML(base.multi_up, lo, hi)
        self.multi_down = SimML(base.multi_down, lo, hi)

    def __getattr__(self, name):
        return getattr(self._base, name)


@contextlib.contextmanager
def env_set(name, value):
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def make_selections(torch, gen, E, K, lo, hi, rows):
    """(rows, K) int64 expert ids + token-class masks for one partition.

    Unsharded: plain random picks. Sharded: a deterministic pattern -- row 0
    all-inside, row 1 all-OUTSIDE (empty partial rank), row 2 a boundary row
    (lo-1 / lo / hi-1 / hi where within [0,E)), then all-inside -- so every
    masked case reaches the grouped reduce on every run.
    """
    rows_list = []

    def _rand_ids(pool, n):
        perm = torch.randperm(len(pool), generator=gen)[:n]
        return pool[perm]

    if lo is None:
        pool = torch.arange(E)
        for _ in range(rows):
            rows_list.append(_rand_ids(pool, K))
    else:
        in_ids = torch.arange(lo, hi)
        out_ids = torch.cat([torch.arange(0, lo), torch.arange(hi, E)])
        for i in range(rows):
            pat = i % 3
            if pat == 0 or rows == 1:
                rows_list.append(_rand_ids(in_ids, K))
            elif pat == 1:
                rows_list.append(_rand_ids(out_ids, K))
            else:
                bounds = [v for v in (lo, lo - 1, hi, hi - 1) if 0 <= v < E]
                rest = _rand_ids(in_ids, max(K - len(bounds), 0))
                pick = torch.tensor(bounds, dtype=torch.long)
                rows_list.append(torch.cat([pick, rest])[:K])
    sel = torch.stack(rows_list).long()
    if lo is None:
        all_in = torch.ones(rows, dtype=torch.bool)
        all_out = torch.zeros(rows, dtype=torch.bool)
    else:
        inr = (sel >= lo) & (sel < hi)
        all_in = inr.all(dim=1)
        all_out = (~inr).all(dim=1)
    partial = ~(all_in | all_out)
    return sel, all_in, all_out, partial


def instrument(ext, counts):
    """CPU-side accounting of exl3_mgemm launches (all / with weights /
    weighted under EXL3_GEMV_FUSE_OUT=0 / with num_tokens>1), including the
    native argument shapes. The wrapper is removed before timing so timed
    runs are clean."""
    orig = ext.exl3_mgemm

    def inner(*a, **k):
        counts["mgemm"] = counts.get("mgemm", 0) + 1
        if len(a) > 15:
            counts.setdefault("native_calls", []).append({
                "A_shape": list(a[0].shape) if hasattr(a[0], "shape") else None,
                "C_shape": list(a[2].shape) if hasattr(a[2], "shape") else None,
                "A_had_shape": list(a[4].shape) if hasattr(a[4], "shape") else None,
                "indices_shape": list(a[6].shape) if hasattr(a[6], "shape")
                else None,
                "num_tokens": int(a[15]),
            })
        if len(a) > 7 and a[7] is not None:
            counts["weighted"] = counts.get("weighted", 0) + 1
            if os.environ.get(FUSE_ENV) == "0":
                counts["weighted_fuse0"] = counts.get("weighted_fuse0", 0) + 1
        if len(a) > 15 and int(a[15]) > 1:
            counts["multitoken"] = counts.get("multitoken", 0) + 1
        return orig(*a, **k)

    ext.exl3_mgemm = inner
    return orig


def deinstrument(ext, orig):
    ext.exl3_mgemm = orig


def bench_events(torch, fn):
    """CUDA-event median ms/call: WARMUP warmups, SAMPLES samples of REPS
    repeats; synchronize only at sample boundaries."""
    with torch.inference_mode():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        batches = []
        for _ in range(SAMPLES):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(REPS):
                fn()
            e1.record()
            e1.synchronize()
            batches.append(e0.elapsed_time(e1) / REPS)
    return {"per_call_ms": statistics.median(batches),
            "min_ms": min(batches), "max_ms": max(batches),
            "samples_ms": batches,
            "warmup": WARMUP, "samples": SAMPLES, "reps_per_sample": REPS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="moe_multitoken_probe.py")
    ap.add_argument("--model", required=True, help="EXL3 model directory")
    ap.add_argument("--device", type=int, default=0, help="GPU ordinal (cuda:N)")
    ap.add_argument("--output", "-o", required=True, help="JSON report path")
    ap.add_argument("--layer-index", type=int, default=0,
                    help="index into the MoE layers found in the model tree")
    ap.add_argument("--rows", default=ROWS_DEFAULT,
                    help="comma-separated decode rows to replay (engine gate: 2..5 for the candidate)")
    ap.add_argument("--max-rel-l2", type=nonnegative_float, default=0.001,
                    help="maximum fired-candidate relative L2 error vs cooperative reference (default: 0.001)")
    ap.add_argument("--max-scaled-abs", type=nonnegative_float, default=0.005,
                    help="maximum fired-candidate max-abs/reference-scale error (default: 0.005)")
    args = ap.parse_args(argv)
    process_env_start = dict(os.environ)
    controlled_env_start = {
        name: os.environ.get(name) for name in CONTROLLED_ENV_KEYS
    }

    report = {
        "format": PROBE_FORMAT, "created_utc": now_utc(),
        "tool": "rocm_tools/rdna2/moe_multitoken_probe.py",
        "mode": "numeric+latency-proxy", "seed": SEED, "params": vars(args),
        "env": {"python": python_env(), "torch": torch_env(),
                "visible_gpu": visible_gpu_env(), "exl3_switches": rocm_patch_env(),
                "EXL3_ROCM_MOE_MULTI_TOKEN": os.environ.get("EXL3_ROCM_MOE_MULTI_TOKEN"),
                "EXL3_MGEMV": os.environ.get("EXL3_MGEMV"),
                FUSE_ENV: os.environ.get(FUSE_ENV)},
        "repo_git_commit": git_commit(repo_root()),
        "model_dir": str(Path(args.model).resolve()),
        "model_fingerprint": model_fingerprint(args.model),
        "proposed_route": PROPOSED_ROUTE,
        "numeric_acceptance": {
            "max_rel_l2": args.max_rel_l2,
            "max_scaled_abs": args.max_scaled_abs,
            "zero_reference_requires_maxabs": 0.0,
        },
        "known_unsupported": [
            "non-gfx1030 device or non-ROCm torch",
            "rows outside 2..5 or rows*top_k > 128",
            "gateless MoE, unloaded expert tables, or unsupported quant K outside 1..8",
            "non-half contiguous y/routing weights or non-int64 contiguous selections",
            "un-aligned hidden/intermediate/down widths",
            "EXL3_MGEMV=0 (candidate requires the barrier-free fast path)",
        ],
        "checks": {}, "runs": [], "errors": [],
        "candidate_refusals": [],
        "runtime_flags": {
            "controlled_start": dict(controlled_env_start),
            "candidate_enable": {
                "name": "EXL3_ROCM_MOE_MULTI_TOKEN",
                "required_value": "1",
                "value_at_start": os.environ.get("EXL3_ROCM_MOE_MULTI_TOKEN"),
            },
            "sharded_weighted_launch": {
                "name": FUSE_ENV,
                "required_value": "0",
                "scope": "candidate down launch only; core must restore it",
                "value_at_start": os.environ.get(FUSE_ENV),
            },
        },
        "limitations": [
            "single MoE layer, small-n: a proxy for a future trained-MTP win, "
            "NOT an E2E improvement claim",
            "TP is SIMULATED by sliced pointer-table shard views on one rank; "
            "no collectives, no tp_export/tp_import, no cross-rank allreduce",
            "orig-vs-ref on all-outside tokens may show the PRE-EXISTING R=1 "
            "fused-counter gap (the weighted reduce never fires when zero "
            "slots arrive, leaving a stale row): baseline behaviour reported "
            "for root, not a candidate regression",
            f"the candidate toggles {FUSE_ENV} process-globally around its "
            "own weighted launch only -- experimental, not assumed "
            "thread-safe; the ext re-reads the env per call, host-side, no sync",
            "graph capture bakes the env read at capture time; the candidate "
            "is eager-route code and replay uses the recorded kernel form",
            "the cooperative exl3_mgemm fallback is unchanged; the known "
            "cooperative-launch refusal risk is not exercised here",
            "CU count is NOT assumed anywhere (never hardware-verified)",
        ],
    }

    def add_check(name, ok, detail=""):
        report["checks"][name] = {"pass": bool(ok), "detail": detail}
        print(f" [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    route_env_snapshot = None
    mod = None
    try:
        rows_sweep = parse_rows(args.rows)
        if max(rows_sweep) > 5:
            raise RuntimeError(f"--rows max {max(rows_sweep)} > 5: the candidate gate "
                               "bounds rows at 5; trim the sweep")

        import torch
        from exllamav3 import Config, Model
        from exllamav3.ext import exllamav3_ext as ext
        from exllamav3.modules import block_sparse_mlp as bsm
        from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
        from exllamav3 import rocm_py
        from exllamav3.rocm_py import describe

        native_file = Path(ext.__file__).resolve()
        report["native_extension"] = {"path": str(native_file),
                                      "sha256": hashlib.sha256(native_file.read_bytes()).hexdigest()}
        report["applied_patches"] = describe().splitlines()
        if torch.version.hip is None:
            raise RuntimeError("not a ROCm torch build; the candidate is ROCm/gfx1030-only")

        status = rocm_py.moe_multi_token_status()
        report["moe_mt_status"] = status
        if not status["on"]:
            report["candidate_refusals"].append({"scope": "startup",
                                                  "reason": status["reason"]})
        fwd_impl = getattr(BlockSparseMLP.forward, "__name__", "?")
        add_check("mgemm_steer_active", fwd_impl == "_forward_mgemm_route",
                  f"BlockSparseMLP.forward={fwd_impl}")
        if fwd_impl != "_forward_mgemm_route":
            raise RuntimeError("default mgemm steer inactive; the candidate lives "
                               "under it (see describe above)")
        if not status["on"]:
            add_check("candidate_enabled", False, status["reason"])
            raise RuntimeError(f"candidate not wired: {status['reason']} -- run with "
                               "EXL3_ROCM_MOE_MULTI_TOKEN=1 on a gfx1030 host")
        add_check("candidate_enabled", True, status["reason"])
        if os.environ.get("EXL3_MGEMV") == "0":
            raise RuntimeError("EXL3_MGEMV=0 is set in the environment: orig/cand runs "
                               "would silently be the cooperative kernel, defeating the "
                               "fast-path validation this probe exists for")

        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        arch = getattr(props, "gcnArchName", "")
        report["device"] = {"index": args.device, "name": props.name, "gcnArchName": arch}
        if not arch.startswith("gfx1030"):
            raise RuntimeError(f"cuda:{args.device} is {arch!r}; the candidate gates to "
                               "gfx1030 only")
        if bsm.MAX_BSZN < max(rows_sweep):
            raise RuntimeError(f"block_sparse_mlp.MAX_BSZN={bsm.MAX_BSZN} < max --rows "
                               f"{max(rows_sweep)}: scratch was sized at load for the cap")

        config = Config.from_directory(args.model)
        model = Model.from_config(config)          # tree only; no Model.load
        moes = [m for m in model if isinstance(m, BlockSparseMLP)]
        if not moes:
            raise RuntimeError("no BlockSparseMLP found in the 'text' component")
        mod = moes[args.layer_index]
        report["target"] = {
            "key": mod.key, "moe_index": args.layer_index, "n_moes": len(moes),
            "num_experts": mod.num_experts, "top_k": mod.num_experts_per_tok,
            "hidden_size": mod.hidden_size,
            "intermediate_size": mod.intermediate_size,
            "expert_size": mod.expert_size,
            "gated": mod.gated, "activation_fn": mod.activation_fn,
            "shared_experts": mod.shared_experts is not None,
        }
        add_check("gated_experts", bool(mod.gated), f"gated={mod.gated}")
        if not mod.gated:
            raise RuntimeError("target MoE is gateless; the candidate is gated-expert only")
        if getattr(mod, "alt_residual_channel", False):
            raise RuntimeError("target MoE reads params['residual']: out of this probe's scope")
        with torch.inference_mode():
            mod.load(device)

        cfg = mod.experts_cfg
        E, K = mod.num_experts, mod.num_experts_per_tok
        Hi, I, Ho = cfg.yh.shape[-1], cfg.interm_a.shape[-1], cfg.out_d.shape[-1]
        report["cfg"] = {"min_expert": cfg.min_expert, "max_expert": cfg.max_expert,
                         "Hi": Hi, "I": I, "Ho": Ho, "H_out": cfg.out_bszn.shape[-1],
                         "yh_rows": cfg.yh.shape[0], "interm_rows": cfg.interm_g.shape[0],
                         "out_d_rows": cfg.out_d.shape[0], "out_bszn_rows": cfg.out_bszn.shape[0],
                         "bsm_MAX_BSZN": bsm.MAX_BSZN}
        actual_model = {
            "hidden_size": int(mod.hidden_size),
            "intermediate_size": int(mod.intermediate_size),
            "num_experts": int(E),
            "top_k": int(K),
            "expert_size": int(mod.expert_size),
        }
        report["expected_model"] = dict(EXPECTED_MODEL)
        report["actual_model"] = actual_model
        shape_ok = actual_model == EXPECTED_MODEL
        add_check("expected_model_shape", shape_ok,
                  f"expected {EXPECTED_MODEL}, got {actual_model}")
        if not shape_ok:
            raise RuntimeError(f"target MoE shape mismatch: expected {EXPECTED_MODEL}, "
                               f"got {actual_model}")
        if cfg.min_expert is not None and cfg.min_expert >= 0:
            raise RuntimeError("module loaded as an expert-range shard; this probe's "
                               "simulation assumes an unsharded load (min/max -1/-1)")
        if 5 * K > 128:
            raise RuntimeError(f"top_k={K}: rows 5 would exceed the 128-slot gate")
        add_check("output_scratch_covers_rows", cfg.out_bszn.shape[0] >= 5,
                  f"out_bszn rows {cfg.out_bszn.shape[0]} for max rows 5; "
                  "candidate native scratch may be separately expanded")
        add_check("aligned_fastpath_dims", Hi % 128 == 0 and I % 128 == 0 and Ho % 128 == 0,
                  f"Hi={Hi} I={I} Ho={Ho}")
        if cfg.out_bszn.shape[0] < 5:
            raise RuntimeError("loaded MoE output scratch does not cover 5 rows")
        if not (Hi % 128 == 0 and I % 128 == 0 and Ho % 128 == 0):
            raise RuntimeError(f"loaded MoE dimensions are not fast-path aligned: "
                               f"Hi={Hi} I={I} Ho={Ho}")

        partitions = [("full", None), ("half0", (0, E // 2)), ("half1", (E // 2, E))]
        report["partitions"] = [
            {"name": n, "range": None if prange is None else list(prange)}
            for n, prange in partitions
        ]

        gen = torch.Generator().manual_seed(SEED)
        # Imports, model construction, and load may legitimately establish
        # backend environment state.  The route audit starts only after that
        # setup, immediately before any numeric route checks.
        route_env_snapshot = dict(os.environ)
        report["runtime_flags"]["setup_env_diff"] = env_diff_keys(
            process_env_start, route_env_snapshot)

        def shard_view(part_lo_hi):
            if part_lo_hi is None:
                return mod
            lo, hi = part_lo_hi
            return SimShard(mod, lo, hi)

        def run_engine(mview, y, sel, rw):
            """The unchanged per-token route used as the original baseline."""
            with torch.inference_mode():
                rocm_py._moe_mgemm_rowloop(mview, y, sel, rw)
                return mview.experts_cfg.out_bszn[:y.shape[0]].detach().float().clone()

        def run_candidate(mview, y, sel, rw):
            """Returns (fired, output, refusal).

            A declined candidate is immediately run through the unchanged row
            loop.  This makes the output valid for R=1/unsupported shapes and
            records why no candidate timing is allowed for that case.
            """
            # The public step performs its own support gate.  Call it once;
            # only a refusal needs the separate explanatory gate read.  This
            # keeps the timed candidate path free of probe-side validation.
            with torch.inference_mode():
                fired = bool(rocm_py.moe_multi_token_step(mview, y, sel, rw))
                reason = None
                if not fired:
                    supported, reason = rocm_py.moe_multi_token_supported(
                        mview, y, sel, rw)
                    if supported:
                        reason = "candidate step declined after support gate"
                    rocm_py._moe_mgemm_rowloop(mview, y, sel, rw)
                output = mview.experts_cfg.out_bszn[:y.shape[0]].detach().float().clone()
                return fired, output, (None if fired else str(reason))

        def run_ref(mview, y, sel, rw):
            with torch.inference_mode():
                with env_set("EXL3_MGEMV", "0"):
                    rocm_py._moe_mgemm_rowloop(mview, y, sel, rw)
                return mview.experts_cfg.out_bszn[:y.shape[0]].detach().float().clone()

        def audit_route_environment():
            """Check controlled flags and all keys after the route window."""
            current = dict(os.environ)
            controlled_after = {
                name: current.get(name) for name in CONTROLLED_ENV_KEYS
            }
            diff = env_diff_keys(route_env_snapshot, current)
            report["runtime_flags"]["controlled_after"] = controlled_after
            report["runtime_flags"]["route_env_diff"] = diff
            controlled_ok = controlled_after == controlled_env_start
            route_ok = not any(diff.values())
            add_check("controlled_env_stable", controlled_ok,
                      "candidate EXL3 flags restored to process-start values")
            add_check("route_env_stable", route_ok,
                      "route-window environment unchanged; diff contains key names only")
            return controlled_ok and route_ok

        all_finite = True
        r1_declined = True
        r1_fallback_outputs = True
        fired_expected = True
        zero_rows_ok = True
        stale_ok = True
        candidate_numeric_ok = True
        candidate_launch_evidence = True
        sharded_fuse0_evidence = True
        inputs_cache = {}

        for pname, prange in partitions:
            lo, hi = (None, None) if prange is None else prange
            mview = shard_view(prange)
            for r in rows_sweep:
                y = (torch.randn((r, Hi), generator=gen) * 0.25).half().to(device).contiguous()
                sel_cpu, all_in, all_out, partial = make_selections(torch, gen, E, K, lo, hi, r)
                sel = sel_cpu.to(device).contiguous()
                rw_cpu = torch.softmax(torch.randn((r, K), generator=gen), dim=-1)
                rw = rw_cpu.half().contiguous().to(device)
                inputs_cache[(pname, r)] = (y, sel, rw)

                entry = {"partition": pname,
                         "range": [lo, hi] if lo is not None else None,
                         "rows": r,
                         "input_sha256": {"y": t_sha256(y), "sel": t_sha256(sel),
                                          "rw": t_sha256(rw)},
                         "token_classes": {"all_in": int(all_in.sum()),
                                           "all_out": int(all_out.sum()),
                                           "partial": int(partial.sum())}}

                oref = run_ref(mview, y, sel, rw)
                oorig = run_engine(mview, y, sel, rw)      # rows==1: also the loop
                fired, ocand, cand_refusal = run_candidate(mview, y, sel, rw)
                entry["candidate_fired"] = bool(fired)
                entry["candidate_refusal_reason"] = cand_refusal
                if cand_refusal is not None:
                    report["candidate_refusals"].append({
                        "partition": pname, "rows": r, "reason": cand_refusal,
                    })
                if r >= 2 and not fired:
                    fired_expected = False
                if r == 1:
                    if fired:
                        r1_declined = False
                    entry["cand_equals_loop_r1"] = bool(torch.equal(ocand, oorig))
                    r1_fallback_outputs = r1_fallback_outputs and entry["cand_equals_loop_r1"]

                nonfinite = {k: int((~t.isfinite()).sum())
                             for k, t in (("ref", oref), ("orig", oorig), ("cand", ocand))}
                all_finite = all_finite and all(v == 0 for v in nonfinite.values())
                entry["nonfinite"] = nonfinite
                entry["cand_vs_ref"] = pair_metrics(ocand, oref)
                entry["orig_vs_ref"] = pair_metrics(oorig, oref)
                entry["cand_vs_orig"] = pair_metrics(ocand, oorig)
                if fired:
                    ref_scale = float(oref.abs().max())
                    if ref_scale == 0.0:
                        numeric_gate = entry["cand_vs_ref"]["maxabs"] == 0.0
                        gate_reason = "reference output is zero; candidate maxabs must be exactly zero"
                    else:
                        rel_l2 = entry["cand_vs_ref"]["rel_l2"]
                        scaled_abs = entry["cand_vs_ref"]["maxabs_over_ref_scale"]
                        numeric_gate = (
                            rel_l2 is not None and scaled_abs is not None
                            and math.isfinite(rel_l2) and math.isfinite(scaled_abs)
                            and rel_l2 <= args.max_rel_l2
                            and scaled_abs <= args.max_scaled_abs)
                        gate_reason = (
                            f"rel_l2 <= {args.max_rel_l2} and scaled_abs <= "
                            f"{args.max_scaled_abs}")
                    entry["candidate_numeric_gate"] = {
                        "pass": bool(numeric_gate), "reference_scale": ref_scale,
                        "reason": gate_reason,
                    }
                    candidate_numeric_ok = candidate_numeric_ok and numeric_gate

                if lo is not None:
                    dev_mask = lambda b: b.to(device)
                    if int(all_in.sum()) > 0:
                        m_ = dev_mask(all_in)
                        entry["cand_vs_ref_allin_rows"] = pair_metrics(
                            ocand[m_].reshape(-1), oref[m_].reshape(-1))
                    if int(all_out.sum()) > 0:
                        m_ = dev_mask(all_out)
                        entry["cand_allout_row_zero"] = bool((ocand[m_] == 0.0).all())
                        entry["orig_allout_row_zero_frac"] = float(
                            (oorig[m_] == 0.0).float().mean())
                        zero_rows_ok = zero_rows_ok and entry["cand_allout_row_zero"]
                    if int(partial.sum()) > 0:
                        m_ = dev_mask(partial)
                        entry["cand_vs_ref_partial_rows"] = pair_metrics(
                            ocand[m_].reshape(-1), oref[m_].reshape(-1))

                # Recurrent reuse with changing selections on ONE shard view:
                # stale slot scratch must not leak (A, B, A -- A's rerun must
                # be bit-identical to the first A: same route, same input).
                if r >= 2:
                    sel2_cpu, _, _, _ = make_selections(torch, gen, E, K, lo, hi, r)
                    y2 = (torch.randn((r, Hi), generator=gen) * 0.25).half().to(device).contiguous()
                    sel2 = sel2_cpu.to(device).contiguous()
                    rw2 = torch.softmax(torch.randn((r, K), generator=gen),
                                        dim=-1).half().contiguous().to(device)
                    f_b, _ob, why_b = run_candidate(mview, y2, sel2, rw2)
                    f_a2, oa2, why_a2 = run_candidate(mview, y, sel, rw)
                    entry["stale_reuse"] = {"B_fired": bool(f_b), "A_rerun_fired": bool(f_a2),
                                            "A_rerun_bitidentical": pair_metrics(
                                                ocand, oa2)["bitexact_observed"],
                                            "B_refusal_reason": why_b,
                                            "A_rerun_refusal_reason": why_a2,
                                            "B_input_sha256": {
                                                "y": t_sha256(y2), "sel": t_sha256(sel2),
                                                "rw": t_sha256(rw2),
                                            }}
                    stale_ok = (stale_ok and entry["stale_reuse"]["A_rerun_bitidentical"]
                                and f_b and f_a2)

                # CPU launch accounting, clean env, one instrumented run each
                for kind in ("orig", "cand"):
                    counts = {}
                    orig_ext = instrument(ext, counts)
                    try:
                        with torch.inference_mode():
                            rocm_py._moe_mgemm_rowloop(mview, y, sel, rw) if kind == "orig" \
                                else rocm_py.moe_multi_token_step(mview, y, sel, rw)
                    finally:
                        deinstrument(ext, orig_ext)
                    entry.setdefault("launches", {})[kind] = counts

                c_launch = entry["launches"]["cand"]
                if fired:
                    slots = r * K
                    native_calls = c_launch.get("native_calls", [])
                    candidate_shape_ok = (
                        len(native_calls) == 3
                        and all(call["num_tokens"] == r
                                and call["A_shape"][0] == slots
                                and call["C_shape"][0] == slots
                                and call["A_had_shape"][0] == slots
                                and call["indices_shape"] == [1, slots]
                                for call in native_calls)
                    )
                    c_launch["candidate_native_shapes_ok"] = candidate_shape_ok
                    c_launch["candidate_slots"] = slots
                    candidate_launch_evidence = candidate_launch_evidence and (
                        c_launch.get("mgemm", 0) == 3 and
                        c_launch.get("multitoken", 0) == 3 and
                        candidate_shape_ok)
                    if lo is not None:
                        sharded_fuse0_evidence = sharded_fuse0_evidence and (
                            c_launch.get("weighted_fuse0", 0) >= 1)

                report["runs"].append(entry)
                print(f" -- {pname:>5} rows={r:<2} fired={entry['candidate_fired']} "
                      f"cand-vs-ref maxabs={entry['cand_vs_ref']['maxabs']:.3e} "
                      f"rel_l2={entry['cand_vs_ref']['rel_l2']} "
                      f"bitexact_obs={entry['cand_vs_ref']['bitexact_observed']} "
                      f"orig-vs-ref maxabs={entry['orig_vs_ref']['maxabs']:.3e}",
                      flush=True)

        add_check("all_finite", all_finite, "ref + cand routed outputs finite")
        add_check("candidate_fired_rows_2_5", fired_expected,
                  "every rows>=2 run took the slot-major route (route evidence, "
                  "not a silent loop fallthrough)")
        add_check("rowloop_used_for_r1", r1_declined,
                  "rows=1 declined the candidate")
        add_check("rowloop_output_for_r1", r1_fallback_outputs,
                  "rows=1 fallback output equals the unchanged row loop")
        add_check("cand_zero_all_outside_partial", zero_rows_ok,
                  "shard runs: all-outside tokens reduce to exact 0.0")
        add_check("candidate_numeric_tolerance", candidate_numeric_ok,
                  f"fired cand-vs-ref within rel_l2 <= {args.max_rel_l2}, "
                  f"scaled maxabs <= {args.max_scaled_abs}; zero references require exact zero")
        add_check("stale_scratch_reuse_bitidentical", stale_ok,
                  "A,B,A on one shard view: A's rerun bit-identical and fired each time")
        add_check("candidate_launch_evidence", candidate_launch_evidence,
                  "fired candidate calls made exactly 3 num_tokens>1 exl3_mgemm launches")
        add_check("sharded_fuse_out_flag_evidence", sharded_fuse0_evidence,
                  f"sharded fired calls observed {FUSE_ENV}=0 on weighted launch")
        if not audit_route_environment():
            raise RuntimeError("route environment changed before timing; refusing timing")
        if not candidate_numeric_ok:
            raise RuntimeError(
                "candidate numeric acceptance failed; refusing to collect timing for "
                "a numerically invalid route")

        # ---------------- latency proxy (candidate-fired rows only) ----------
        for pname, prange in partitions:
            for r in rows_sweep:
                if r < 2:
                    continue
                e_run = next((e for e in report["runs"]
                              if e["partition"] == pname and e["rows"] == r), None)
                if e_run is None or not e_run["candidate_fired"] \
                        or e_run["nonfinite"]["cand"] != 0:
                    continue          # no invalid warm/timed rows
                mview = shard_view(prange)
                y, sel, rw = inputs_cache[(pname, r)]
                tm = {}
                tm["orig"] = bench_events(
                    torch, lambda: rocm_py._moe_mgemm_rowloop(mview, y, sel, rw))
                tm["cand"] = bench_events(
                    torch, lambda: rocm_py.moe_multi_token_step(mview, y, sel, rw))
                with env_set("EXL3_MGEMV", "0"):
                    tm["ref_coop_context"] = bench_events(
                        torch, lambda: rocm_py._moe_mgemm_rowloop(mview, y, sel, rw))
                tm["speedup_cand_over_orig"] = (
                    tm["orig"]["per_call_ms"] / tm["cand"]["per_call_ms"]
                    if tm["cand"]["per_call_ms"] > 0 else None)
                e_run["timing_ms"] = tm
                print(f" -- time {pname:>5} rows={r:<2} orig={tm['orig']['per_call_ms']:.3f} "
                      f"cand={tm['cand']['per_call_ms']:.3f} "
                      f"ref={tm['ref_coop_context']['per_call_ms']:.3f} ms", flush=True)
        # Re-audit after CUDA-event timing so no temporary flag or backend
        # mutation can be hidden by the earlier numeric-only audit.
        if not audit_route_environment():
            raise RuntimeError("route environment changed during timing")
        report["bench_method"] = (
            f"{WARMUP} warmups, {SAMPLES} samples x {REPS} repeats, CUDA events "
            "per call, synchronize at sample boundaries only; route-level MoE "
            "step only (no router/shared expert/post-norm, no E2E)")

        report["pass"] = all(v["pass"] for v in report["checks"].values()) \
            and not report["errors"]
        print(f"\n == moe_multitoken_probe: {'PASS' if report['pass'] else 'FAIL'} "
              f"(rows {rows_sweep}, partitions {[p for p, _ in partitions]})", flush=True)
    except Exception as e:
        report["errors"].append(repr(e))
        report["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["pass"] = False
        print(f" !! probe failed: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        if mod is not None:
            try:
                with torch.inference_mode():
                    mod.unload()
            except Exception as e:
                report["errors"].append(f"unload: {e!r}")
                report["pass"] = False
        controlled_after = {
            name: os.environ.get(name) for name in CONTROLLED_ENV_KEYS
        }
        report["runtime_flags"]["controlled_after_final"] = controlled_after
        report["runtime_flags"]["controlled_restored_to_start"] = (
            controlled_after == controlled_env_start)
        if route_env_snapshot is not None:
            report["runtime_flags"].setdefault(
                "route_env_diff", env_diff_keys(route_env_snapshot, dict(os.environ)))
        if controlled_after != controlled_env_start:
            report["checks"]["controlled_env_stable"] = {
                "pass": False,
                "detail": "final controlled EXL3 flags differ from process-start values",
            }
            report["pass"] = False
        try:
            write_json(args.output, report)
            print(f" -- wrote {args.output}")
        except Exception as e:
            print(f" !! could not write {args.output}: {e}", file=sys.stderr)
            raise
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
