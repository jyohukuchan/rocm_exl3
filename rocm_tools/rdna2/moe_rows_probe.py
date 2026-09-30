#!/usr/bin/env python3
"""GPU diagnostic: bounded MoE decode-row expansion on the ROCm mgemm bszN route.

Loads ONE real BlockSparseMLP from an EXL3 directory (module tree built; only
the target's load() touches the checkpoint -- the rest of the model, the
Engram/PLE tables, and any Cache are never loaded) and replays the same
seeded rows through the paths the EXL3_ROCM_MOE_MGEMM_MAX_ROWS knob decides
between:

  A expanded : forward at the loaded Python cap (block_sparse_mlp.MAX_BSZN,
               raised once before load by the rocm_py mgemm steer; scratch
               sized at load) -- every row runs the per-token exl3_mgemv loop.
  B cap8     : MAX_BSZN temporarily back to the shipped 8; rows > 8 take the
               fused exl3_moe / dense tier the knob exists to avoid. Restored
               to the loaded cap in finally, never left raised.
  C per-row  : A's rows as independent 1-row calls, each output CLONED before
               the next call (the bszN result aliases the static out_bszn
               scratch and the shared-expert tail accumulates into it --
               rocm_tools/moe_check.py's trap), concatenated: row independence.

Every run carries call evidence (routed exl3_mgemm = loop, exl3_moe = tier),
and rows <= 8 are flagged same_route for A vs B, so a trivial 0.00 can never
pose as cross-path evidence. All paths run the module's own forward, which
includes shared experts exactly once (bc_sh_exp False, verified).

No numerical tolerance is asserted: rel_l2/maxabs/cosine/bitexact are
reported for the validation owner (root) to judge against the frozen
baseline; the probe fails only on non-finite output, shape/evidence
contradictions, an inactive steer or a cap below the sweep. Latency is
separate and unprofiled: WARMUP warmups, REPS forwards timed with ONE
synchronize outside the loop. Full TP (shards + cross-rank reduce) is
explicitly not_tested here.

Run with the engine knob covering the sweep (root selects 20; 24 for b4xd5):
    EXL3_ROCM_MOE_MGEMM_MAX_ROWS=20 /opt/venv/bin/python \
        /src/rocm_tools/rdna2/moe_rows_probe.py --model <dir> \
        [--layer-index 0] [--device 0] [--rows 1,4,8,9,12,15,16,20] -o out.json
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rocm_tools.rdna2.common import (
    git_commit, model_fingerprint, now_utc, python_env, repo_root,
    rocm_patch_env, torch_env, visible_gpu_env, write_json,
)

PROBE_FORMAT = "rdna2-moe-rows-probe/1"
SEED = 1234
FALLBACK_CAP = 8    # shipped block_sparse_mlp.MAX_BSZN without the knob
WARMUP, REPS, SAMPLES = 5, 5, 5


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


def pair_metrics(a, b) -> dict:
    """Cross-tensor numbers (fp32, same shape). Reported, never thresholded."""
    d = (a - b).abs()
    scale = float(b.abs().max())
    bn, an = float(b.norm()), float(a.norm())
    bit = (a == b)
    return {
        "maxabs": float(d.max()),
        "rel_l2": float(d.norm() / bn) if bn > 0 else None,
        "maxabs_over_ref_scale": float(d.max()) / scale if scale > 0 else None,
        "cosine": float((a * b).sum() / (an * bn)) if an > 0 and bn > 0 else None,
        "bitexact": bool(bit.all()),
        "diff_elements": int((~bit).sum()),
        "elements": int(a.numel()),
    }


def t_sha256(t) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def instrument(ext, counts: dict) -> dict:
    """Count routed exl3_mgemm calls (the selected_experts positional arg at
    index 6 is non-null only for the per-token bszN loop; shared-expert mgemm
    passes None) and exl3_moe launches (the tiered fallback). Restored before
    timing so the latency runs are clean."""
    state = {}
    def wrap(name, pred):
        orig = getattr(ext, name)
        state[name] = orig
        def inner(*a, **k):
            if pred(a, k):
                counts[name] = counts.get(name, 0) + 1
            return orig(*a, **k)
        setattr(ext, name, inner)
    wrap("exl3_mgemm", lambda a, k: len(a) > 6 and a[6] is not None)
    if hasattr(ext, "exl3_moe"):
        wrap("exl3_moe", lambda a, k: True)
    return state


def deinstrument(ext, state: dict) -> None:
    for name, orig in state.items():
        setattr(ext, name, orig)


def fwd_at_cap(mod, bsm, torch, x, cap):
    """One batched forward with the eligibility cap temporarily `cap`, output
    cloned before returning (static-buffer aliasing). Runs under
    inference_mode, as every real Model.forward pass does."""
    prev = bsm.MAX_BSZN
    bsm.MAX_BSZN = cap
    try:
        with torch.inference_mode():
            return mod.forward(x, {}).detach().float().clone()
    finally:
        bsm.MAX_BSZN = prev


def per_row_cat(mod, bsm, torch, x, cap):
    """x's rows as independent 1-row calls at the expanded cap, each cloned
    before the next call, concatenated along dim 1."""
    prev = bsm.MAX_BSZN
    bsm.MAX_BSZN = cap
    outs = []
    try:
        with torch.inference_mode():
            for i in range(x.shape[1]):
                outs.append(mod.forward(x[:, i:i + 1].contiguous(), {})
                            .detach().float().clone())
        return torch.cat(outs, dim=1)
    finally:
        bsm.MAX_BSZN = prev


def bench(torch, mod, bsm, x, cap):
    import statistics
    def body():
        prev = bsm.MAX_BSZN
        bsm.MAX_BSZN = cap
        try:
            mod.forward(x, {})
        finally:
            bsm.MAX_BSZN = prev
    batches = []
    with torch.inference_mode():
        for _ in range(WARMUP):
            body()
        for _ in range(SAMPLES):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPS):
                body()
            torch.cuda.synchronize()
            batches.append((time.perf_counter() - t0) * 1e3)
    per_call = [v / REPS for v in batches]
    return {"batch_ms": batches, "per_forward_ms": statistics.median(per_call),
            "min_ms": min(per_call), "max_ms": max(per_call),
            "warmup": WARMUP, "reps_per_sample": REPS, "samples": SAMPLES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="moe_rows_probe.py")
    ap.add_argument("--model", required=True, help="EXL3 model directory")
    ap.add_argument("--layer-index", type=int, default=0,
                    help="index into the MoE layers found in the model tree (negative OK)")
    ap.add_argument("--device", type=int, default=0, help="GPU ordinal (cuda:N)")
    ap.add_argument("--rows", default="1,4,8,9,12,15,16,20",
                    help="comma-separated flattened row counts to replay")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args(argv)

    report = {
        "format": PROBE_FORMAT, "created_utc": now_utc(), "tool": "rocm_tools/rdna2/moe_rows_probe.py",
        "mode": "numeric+bench", "seed": SEED, "params": vars(args),
        "env": {"python": python_env(), "torch": torch_env(), "visible_gpu": visible_gpu_env(),
                "exl3_switches": rocm_patch_env()},
        "repo_git_commit": git_commit(repo_root()),
        "model_dir": str(Path(args.model).resolve()), "model_fingerprint": model_fingerprint(args.model),
        "tp_expert_range": {"tested": False,
                            "note": "tp_export/tp_import shards + cross-rank reduce not exercised; root validates full TP"},
        "checks": {}, "rows_out": [], "bench": [], "errors": [],
    }

    def add_check(name, ok, detail=""):
        report["checks"][name] = {"pass": bool(ok), "detail": detail}
        print(f" [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    mod = None
    try:
        rows = parse_rows(args.rows)
        max_rows = max(rows)
        report["rows_requested"] = rows

        # Knob precondition, checked BEFORE importing the engine: the scratch
        # was sized at load from what EXL3_ROCM_MOE_MGEMM_MAX_ROWS selected at
        # import time, so the sweep must be covered by the engine, not the probe.
        knob_raw = os.environ.get("EXL3_ROCM_MOE_MGEMM_MAX_ROWS")
        if knob_raw is None or not knob_raw.strip():
            raise RuntimeError("EXL3_ROCM_MOE_MGEMM_MAX_ROWS unset: the engine loads cap 8; "
                               f"this sweep needs >= {max_rows} (run with ...=20 or =24)")
        try:
            knob = int(knob_raw.strip())
        except ValueError:
            raise RuntimeError(f"EXL3_ROCM_MOE_MGEMM_MAX_ROWS={knob_raw!r} is not an integer "
                               "(the engine aborts at import too; use 8..24)")
        if knob < max_rows:
            raise RuntimeError(f"EXL3_ROCM_MOE_MGEMM_MAX_ROWS={knob} < max --rows {max_rows}: "
                               "rows above the cap would not exercise expansion; raise the knob (<=24)")
        print(f" -- knob EXL3_ROCM_MOE_MGEMM_MAX_ROWS={knob}, sweep 1..{max_rows} covered", flush=True)

        import torch
        from exllamav3 import Config, Model
        from exllamav3.ext import exllamav3_ext as ext
        from exllamav3.modules import block_sparse_mlp as bsm
        from exllamav3.modules import mlp as mlp_mod
        from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
        from exllamav3.rocm_py import describe

        native_file = Path(ext.__file__).resolve()
        report["native_extension"] = {"path": str(native_file),
                                      "sha256": hashlib.sha256(native_file.read_bytes()).hexdigest()}
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        wanted = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
        if soft < wanted:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
        report["rlimit_nofile"] = resource.getrlimit(resource.RLIMIT_NOFILE)
        desc = describe()
        report["applied_patches"] = desc.splitlines()
        if torch.version.hip is None:
            raise RuntimeError("not a ROCm torch build; the mgemm steer (and this probe) "
                               "is ROCm-only. describe(): " + desc.replace("\n", " | "))
        fwd_impl = getattr(BlockSparseMLP.forward, "__name__", "?")
        steer_ok = fwd_impl == "_forward_mgemm_route"
        add_check("mgemm_steer_active", steer_ok, f"BlockSparseMLP.forward={fwd_impl}")
        if not steer_ok:
            raise RuntimeError(
                f"mgemm steer not active (forward={fwd_impl}): the per-token loop this probe "
                "measures exists only under the default steer (BSZN off, MGEMM_ROUTE on). "
                "describe(): " + desc.replace("\n", " | "))
        if bsm.MAX_BSZN != knob:
            raise RuntimeError(f"knob={knob} did not reach the engine (block_sparse_mlp.MAX_BSZN="
                               f"{bsm.MAX_BSZN}); the cap is applied at import under the default "
                               "steer only -- check describe() above")
        add_check("knob_applied_at_import", True,
                  f"block_sparse_mlp.MAX_BSZN at entry = {bsm.MAX_BSZN} (knob {knob})")
        mlp8 = mlp_mod.MAX_BSZN == FALLBACK_CAP
        add_check("mlp_cap_untouched", mlp8, f"mlp.MAX_BSZN = {mlp_mod.MAX_BSZN} "
                  "(shared-expert BC graphs + native checks must keep 8)")
        if not mlp8:
            raise RuntimeError(f"mlp.MAX_BSZN={mlp_mod.MAX_BSZN} != 8: native BC_GatedMLP graphs "
                               "are compiled for 8 rows; the invariant probed here is broken")

        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        report["device"] = {"index": args.device, "name": props.name,
                            "gcnArchName": getattr(props, "gcnArchName", None)}
        report["native_path"] = {"hip": torch.version.hip, "forward_impl": fwd_impl,
                                 "python_cap_at_entry": bsm.MAX_BSZN,
                                 "mlp_max_bszn": mlp_mod.MAX_BSZN,
                                 "coop_stub": "unreachable behind _BCProxy while the steer is active"}

        # Model tree only: nothing else's load() runs, so the full model and
        # the Engram/PLE tables are never touched.
        config = Config.from_directory(args.model)
        model = Model.from_config(config)
        moes = [m for m in model if isinstance(m, BlockSparseMLP)]
        if not moes:
            raise RuntimeError("no BlockSparseMLP found in the 'text' component")
        mod = moes[args.layer_index]
        parent = next((blk for blk in model.modules if getattr(blk, "mlp", None) is mod), None)
        report["target"] = {
            "key": mod.key, "moe_index": args.layer_index, "n_moes": len(moes),
            "parent_transformer": getattr(parent, "key", None),
            "parent_layer_idx": getattr(parent, "layer_idx", None),
            "num_experts": mod.num_experts, "top_k": mod.num_experts_per_tok,
            "hidden_size": mod.hidden_size, "expert_size": mod.expert_size,
            "gated": mod.gated, "activation_fn": mod.activation_fn,
            "shared_experts": mod.shared_experts is not None,
            "shared_gate": mod.shared_gate is not None,
        }
        if getattr(mod, "alt_residual_channel", False):
            raise RuntimeError("target MoE reads params['residual']; this probe's minimal "
                               "params cover standard-router + shared-expert models only")

        with torch.inference_mode():
            mod.load(device)

        cfg = mod.experts_cfg
        cap = bsm.MAX_BSZN
        report["cap"] = {"env_knob": knob, "at_load": cap,
                         "out_bszn_shape": tuple(cfg.out_bszn.shape),
                         "yh_shape": tuple(cfg.yh.shape),
                         "fallback_cap": FALLBACK_CAP,
                         "bszn_slots_expected": cap * mod.num_experts_per_tok}
        add_check("scratch_sized_to_cap",
                  cfg.out_bszn.shape[0] == cap and cfg.yh.shape[0] == cap * mod.num_experts_per_tok,
                  f"out_bszn {tuple(cfg.out_bszn.shape)}, yh {tuple(cfg.yh.shape)}")
        add_check("bszn_route_eligible", mod.bc is not None and mod.support_quant_paths,
                  f"bc={'yes' if mod.bc is not None else 'NONE'} "
                  f"support_quant_paths={mod.support_quant_paths}")
        add_check("shared_experts_once", mod.bc_sh_exp is False,
                  "bc_sh_exp False => the Python tail adds shared experts exactly once per path")

        # Value hash of the original quantized weights -- NOT the pointer
        # tables (data_ptr reshuffles every run). A hash gap is a provenance
        # note, not a measurement failure, so it stays out of errors.
        try:
            wh = hashlib.sha256()
            for t in (mod.routing_gate.inner.weight, mod.ups[0].inner.trellis,
                      mod.downs[-1].inner.trellis):
                wh.update(t.detach().cpu().contiguous().numpy().tobytes())
            report["weights_sha256_sample"] = wh.hexdigest()
        except Exception as e:
            report["weights_sha256_sample"] = None
            report["weights_sha256_note"] = f"hash skipped: {e!r}"

        gen = torch.Generator().manual_seed(SEED)
        x_full = torch.randn((1, max_rows, mod.hidden_size), generator=gen,
                             dtype=torch.float32).half().to(device)

        def run_all(xr):
            counts = {}
            state = instrument(ext, counts)
            try:
                counts.clear(); a = fwd_at_cap(mod, bsm, torch, xr, cap); ev_a = dict(counts)
                counts.clear(); b = fwd_at_cap(mod, bsm, torch, xr, FALLBACK_CAP); ev_b = dict(counts)
                counts.clear(); c = per_row_cat(mod, bsm, torch, xr, cap); ev_c = dict(counts)
            finally:
                deinstrument(ext, state)
            return a, b, c, {"expanded": ev_a, "cap8_fallback": ev_b, "per_row": ev_c}

        all_finite = all_shapes = all_evidence = True
        for r in rows:
            xr = x_full[:, :r].contiguous()
            a, b, c, ev = run_all(xr)
            entry = {"rows": r, "input_sha256": t_sha256(xr), "route_evidence": ev}
            for name, t in (("expanded", a), ("cap8_fallback", b), ("per_row_cat", c)):
                nf = int((~t.isfinite()).sum())
                all_finite = all_finite and nf == 0
                entry[f"out_{name}"] = {"sha256": t_sha256(t), "nonfinite": nf,
                                        "absmax": float(t.abs().max()), "shape": tuple(t.shape)}
            all_shapes = all_shapes and a.shape == b.shape == c.shape == (1, r, mod.expert_size)
            same_route = r <= FALLBACK_CAP
            entry["A_vs_B_same_route"] = same_route
            entry["A_vs_B"] = {**pair_metrics(a, b), "interpretation":
                               ("same route (rows <= fallback cap): any diff is nondeterminism"
                                if same_route else
                                "cross-route (per-token mgemv vs the >8-row tier): expect "
                                "accumulation-order scale; judged against the frozen baseline, "
                                "no probe-side tolerance")}
            entry["A_vs_C"] = {**pair_metrics(a, c), "interpretation":
                               "row independence: near-bitexact expected; evaluated, not asserted"}
            if not same_route:
                # expanded must show the loop; the cap-8 fallback of the same
                # rows must NOT -- else A vs B silently compared one branch
                all_evidence = all_evidence and (ev["expanded"].get("exl3_mgemm", 0) >= r
                                                 and ev["cap8_fallback"].get("exl3_mgemm", 0) == 0)
                entry["A_vs_B"]["zero_diff_cross_route_warn"] = entry["A_vs_B"]["maxabs"] == 0.0
            report["rows_out"].append(entry)
            print(f" -- rows={r:<3} A-vs-B maxabs={entry['A_vs_B']['maxabs']:.3e} "
                  f"rel_l2={entry['A_vs_B']['rel_l2']} bitexact_A_C={entry['A_vs_C']['bitexact']}",
                  flush=True)

        add_check("all_finite", all_finite, "every replayed output finite")
        add_check("shapes_match", all_shapes, "A/B/C agree on (1, rows, expert_size)")
        add_check("route_evidence", all_evidence,
                  "expanded ran the per-token loop; cap-8 fallback of rows>8 did not")

        # Latency: clean (uninstrumented) sync-loop, separate from numerics.
        for r in rows:
            xr = x_full[:, :r].contiguous()
            entry = {"rows": r, "same_route_ab": r <= FALLBACK_CAP,
                     "expanded": bench(torch, mod, bsm, xr, cap),
                     "cap8_fallback": bench(torch, mod, bsm, xr, FALLBACK_CAP)}
            report["bench"].append(entry)
            print(f" -- bench rows={r:<3} expanded={entry['expanded']['per_forward_ms']:.3f} ms "
                  f"cap8={entry['cap8_fallback']['per_forward_ms']:.3f} ms", flush=True)
        report["bench_method"] = (f"{WARMUP} warmups, {SAMPLES} samples of {REPS} forwards; "
                                  "synchronize at sample boundaries only, instrumentation removed; "
                                  "per_forward_ms is median(sample elapsed / forwards)")

        report["pass"] = all(v["pass"] for v in report["checks"].values()) and not report["errors"]
        print(f"\n == moe_rows_probe: {'PASS' if report['pass'] else 'FAIL'} "
              f"(cap {cap}, rows {rows})", flush=True)
    except Exception as e:
        report["errors"].append(repr(e))
        report["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
        report["pass"] = False
        print(f" !! probe failed: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        if mod is not None:
            try:
                mod.unload()
            except Exception as e:
                report["errors"].append(f"unload: {e!r}")
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
