#!/usr/bin/env python3
"""GPU regression probe: gated-MLP fp16 range balance (gfx1030).

Loads the EXL3 model, runs the reference manifest's problem cases (the three
code-heavy ones whose silu(g)*u intermediate overflowed to Inf/NaN before
the balance patch), and gates on:

  finite       every harvested logit row is finite (the original failure:
               d-nonfinite-trace.json, 24 Inf at model.layers.2, three cases
               all-NaN);
  once-only    the svh rescale (up /= 8, down *= 8) lands exactly once per
               load: repeating forwards leaves every svh absmax untouched
               (compounding would move it by 8x), and an unload/reload
               cycle restores the same absmax (a balance skipped on reload
               would differ by 8x the other way -- the old load_local-rescale
               design died precisely to the deferred refill, so the reload
               leg is the regression that matters);
  no-retention after unload + gc, weakrefs to the balanced inners are dead:
               the marker bookkeeping must not pin weight objects across
               reloads;
  agreement    teacher-forced top-1 over the SAME positions matches the
               frozen BF16 CPU oracle and the proven balanced artifact
               (d-candidate-balanced8-loaded: 94.6% overall vs BF16; the
               py_source cases 95.0-95.9%) at --min-agreement. This probe
               runs a case subset by design, so agreement is computed here,
               index-aligned per case -- not through compare_top1's
               full-manifest completeness gate, and never as generated-token
               equality.

With the patch disabled (EXL3_ROCM_MLP_RANGE_BALANCE=0) the finiteness and
agreement gates truthfully fail on the code cases -- that is the A/B signal;
the bookkeeping gates are reported skipped.

Bulk (cache-free) forward, same params as collect_top1 --execution bulk:
rows >> MAX_BSZN, so GatedMLP.dispatch takes the gate/up/down paths, and the
BC bszN graph would read the same svh storage through its pointer tables.
The guard runs at forward entry, before whichever branch dispatches.

Example (inside the rocm-exl3-rdna2 container, exclusive GPU):
    /opt/venv/bin/python /src/rocm_tools/rdna2/mlp_range_balance_probe.py \
        --manifest /work/runs/d-manifest.json \
        --model-dir /work/models/qwen3-8b-exl3-4bpw \
        --bf16-ref /work/runs/d-reference-bf16-cpu.json \
        --balanced-ref /work/runs/d-candidate-balanced8-loaded.json \
        -o /work/runs/d-mlp-range-balance-probe.json
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
import weakref
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rocm_tools.rdna2.common import (
    git_commit,
    model_fingerprint,
    now_utc,
    python_env,
    read_json,
    repo_root,
    rocm_patch_env,
    torch_env,
    validate_manifest,
    visible_gpu_env,
    write_json,
)

PROBE_FORMAT = "rdna2-mlp-range-balance-probe/1"
PROBLEM_CASES = ["py_source_01", "py_source_02", "py_source_03"]
BULK_ATTN = "flash_attn_nc"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog = "mlp_range_balance_probe.py")
    ap.add_argument("--manifest", required = True)
    ap.add_argument("--model-dir", required = True, help = "EXL3 model (backend exl3 only)")
    ap.add_argument("-o", "--output", required = True)
    ap.add_argument("--cases", default = ",".join(PROBLEM_CASES),
                    help = "comma-separated case_ids, or 'all' (default: the problem code cases)")
    ap.add_argument("--bf16-ref", default = None,
                    help = "top1 artifact to align against (e.g. the BF16 CPU oracle)")
    ap.add_argument("--balanced-ref", default = None,
                    help = "proven balanced candidate top1 artifact "
                           "(d-candidate-balanced8-loaded.json)")
    ap.add_argument("--min-agreement", type = float, default = 0.90)
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--load-max-chunk-size", type = int, default = 2048)
    ap.add_argument("--repeat-forwards", type = int, default = 2,
                    help = "extra passes over the first case to prove repeats do "
                           "not compound the balance")
    ap.add_argument("--no-reload", action = "store_true",
                    help = "skip the unload/reload cycle (not advised; the reload "
                           "leg is exactly what the old load_local rescale failed)")
    return ap


# ---------------------------------------------------------------------------
# GPU-facing helpers (lazy torch imports: this module stays import-clean)
# ---------------------------------------------------------------------------

def harvest(torch, logits, positions, vocab):
    """Per-position (top1, nonfinite), float-upcast row, argmax over [0, vocab)."""
    top1, nonfinite = [], []
    for p in positions:
        row = logits[0, p, :vocab].float()
        if not bool(torch.isfinite(row).all()):
            nonfinite.append(p)
        top1.append(int(torch.argmax(row).item()))
    return top1, nonfinite


def run_cases(torch, model, cases, vocab, device):
    per_case = {}
    all_finite = True
    for case in cases:
        ids_t = torch.tensor([case["ids"]], dtype = torch.long, device = device)
        logits = model.forward(ids_t, {"attn_mode": BULK_ATTN})
        if logits.dim() != 3 or logits.shape[1] != case["len_ids"] or logits.shape[2] < vocab:
            raise RuntimeError(f"unexpected bulk logits shape {tuple(logits.shape)} for "
                               f"case {case['case_id']} (want (1, {case['len_ids']}, >={vocab}))")
        top1, nonfinite = harvest(torch, logits, case["positions"], vocab)
        del logits, ids_t
        if nonfinite:
            all_finite = False
        per_case[case["case_id"]] = {
            "positions": case["positions"],
            "top1": top1,
            "nonfinite_positions": nonfinite,
        }
        print(f"    case {case['case_id']:18} {len(top1):4} positions, "
              f"non-finite rows: {len(nonfinite)}", flush = True)
    return per_case, all_finite


def snapshot_svh(model):
    """absmax of every GatedMLP up/down svh, keyed by linear key. A tiny
    reduction per tensor; the only way to *see* a compounded or lost balance
    from outside. Exact fp16 values round-trip through float, so comparisons
    are equality comparisons."""
    from exllamav3.modules.mlp import GatedMLP
    import torch
    snap = {}
    for m in model:
        if not isinstance(m, GatedMLP):
            continue
        for lin in list(m.ups) + list(m.downs):
            inner = lin.inner
            if inner is not None and inner.svh is not None:
                snap[lin.key] = float(torch.max(inner.svh.abs()))
    return snap


def grab_inner_refs(model):
    """Weakrefs to every GatedMLP up/down inner -- the retention witness."""
    from exllamav3.modules.mlp import GatedMLP
    refs = []
    for m in model:
        if isinstance(m, GatedMLP):
            for lin in list(m.ups) + list(m.downs):
                if lin.inner is not None:
                    refs.append((lin.key, weakref.ref(lin.inner)))
    return refs


def agree_with(ref_path, results):
    """Index-aligned top-1 agreement per case against a saved top1 artifact;
    cases the artifact lacks are skipped (subset run by design)."""
    out = {}
    if not ref_path:
        return out, None
    ref = read_json(ref_path)
    ref_cases = ref.get("cases", {})
    hits = total = 0
    for cid, res in results.items():
        rc = ref_cases.get(cid)
        if rc is None or rc.get("status") not in (None, "ok"):
            continue
        if rc.get("positions") != res["positions"]:
            raise RuntimeError(f"case {cid}: probe positions differ from reference {ref_path}; "
                               f"cannot align")
        diffs = [p for p, a, b in zip(res["positions"], res["top1"], rc["top1"]) if a != b]
        n = len(res["positions"])
        out[cid] = {"n": n, "agree": n - len(diffs), "first_diffs": diffs[:8]}
        hits += out[cid]["agree"]
        total += n
    return out, (hits / total if total else None)


def main(argv = None) -> int:
    args = build_parser().parse_args(argv)
    report = {
        "format": PROBE_FORMAT,
        "created_utc": now_utc(),
        "tool": "rocm_tools/rdna2/mlp_range_balance_probe.py",
        "params": vars(args),
        "env": {"python": python_env(), "torch": torch_env(),
                "visible_gpu": visible_gpu_env(), "exl3_rocm_switches": rocm_patch_env()},
        "repo_git_commit": git_commit(repo_root()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "model_fingerprint": model_fingerprint(args.model_dir),
        "checks": {},
        "errors": [],
    }

    import torch
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.modules.mlp import GatedMLP
    from exllamav3.rocm_py import mlp_range_balance as mrb

    manifest = read_json(args.manifest)
    verr = validate_manifest(manifest)
    if verr:
        print(f" !! manifest invalid: {verr}", file = sys.stderr)
        return 1
    vocab = manifest["valid_vocab_size"]
    if args.cases.strip().lower() == "all":
        cases = list(manifest["cases"])
    else:
        want = [s.strip() for s in args.cases.split(",") if s.strip()]
        cases = [c for c in manifest["cases"] if c["case_id"] in want]
        missing = sorted(set(want) - {c["case_id"] for c in cases})
        if missing:
            print(f" !! manifest has no cases {missing}", file = sys.stderr)
            return 1
    report["cases"] = [c["case_id"] for c in cases]
    report["manifest_sha256"] = manifest["manifest_sha256"]
    report["total_positions"] = sum(len(c["positions"]) for c in cases)

    patch_active = (bool(getattr(GatedMLP.forward, mrb.WRAPPER, False))
                    and mrb.is_hip_build()
                    and mrb.device_is_gfx1030(torch.device(args.device)))
    report["patch_active"] = patch_active
    print(f" -- range-balance patch: {'ACTIVE' if patch_active else 'ABSENT (A/B baseline run)'}")

    checks = report["checks"]

    def add_check(name, passed, detail, skipped = False):
        checks[name] = {"pass": True if skipped else bool(passed), "detail": detail}
        if skipped:
            checks[name]["skipped"] = True
        tag = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        print(f" [{tag}] {name}: {detail}", flush = True)

    def load_model():
        config = Config.from_directory(args.model_dir)
        tokenizer = Tokenizer.from_config(config)
        av = int(tokenizer.actual_vocab_size)
        if av != vocab:
            raise SystemExit(f" !! tokenizer.actual_vocab_size {av} != manifest "
                             f"valid_vocab_size {vocab}")
        model = Model.from_config(config)
        t0 = time.monotonic()
        model.load(device = args.device, max_chunk_size = args.load_max_chunk_size,
                   progressbar = False)
        print(f" -- model loaded in {time.monotonic() - t0:.1f}s", flush = True)
        return model

    model = None
    try:
        # -- pass 1: the guard balances at each module's first forward --------
        print(" -- pass 1 (first forward per module)", flush = True)
        model = load_model()
        results1, finite1 = run_cases(torch, model, cases, vocab, args.device)
        report["pass1"] = results1
        add_check("finite_pass1", finite1,
                  "all logit rows finite" if finite1 else
                  f"non-finite rows: "
                  f"{sum(len(r['nonfinite_positions']) for r in results1.values())}")
        snap1 = snapshot_svh(model)
        if patch_active:
            audit1 = mrb.audit(model)
            report["audit_pass1"] = audit1
            c = audit1["counts"]
            add_check("balanced_after_first_forward",
                      c["already"] == c["pairs"] and c["mixed"] == 0 and c["skip"] == 0,
                      f"counts={c} unbalanced={audit1['unbalanced_eligible'][:4]}")

        # -- repeated forwards: the balance must not compound ------------------
        drift_top1 = False
        repeat_finite = True
        for r in range(args.repeat_forwards):
            print(f" -- repeat pass {r + 1}/{args.repeat_forwards} (first case)", flush = True)
            cid = cases[0]["case_id"]
            res_r, fin_r = run_cases(torch, model, cases[:1], vocab, args.device)
            if not fin_r:
                repeat_finite = False
            if res_r[cid]["top1"] != results1[cid]["top1"]:
                drift_top1 = True
        snap_r = snapshot_svh(model)
        moved = {k: (snap1[k], snap_r[k]) for k in snap1 if snap1[k] != snap_r[k]}
        add_check("finite_repeats", repeat_finite,
                  "repeats stayed finite" if repeat_finite
                  else "a repeat produced non-finite rows")
        if patch_active:
            add_check("repeat_no_compounding", not moved,
                      f"every svh absmax identical after {args.repeat_forwards} repeats" if not moved
                      else f"{len(moved)} svh absmax moved: {dict(list(moved.items())[:5])}")
        else:
            add_check("repeat_no_compounding", True,
                      "skipped (patch absent: nothing to compound)", skipped = True)
        add_check("repeat_top1_stable", not drift_top1,
                  "top-1 identical across repeats" if not drift_top1
                  else "top-1 drifted on a repeat (nondeterminism or compounding)")

        # -- unload/reload: once-per-new-inner, and no retained weights -------
        refs = grab_inner_refs(model)
        model.unload()
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        alive = [k for k, wr in refs if wr() is not None]
        if patch_active:
            add_check("no_retention_after_unload", not alive,
                      f"all {len(refs)} balanced inners collected" if not alive
                      else f"{len(alive)} inners still alive: {alive[:5]}")
        else:
            add_check("no_retention_after_unload", True, "skipped (patch absent)", skipped = True)

        if args.no_reload:
            add_check("reload", True, "skipped (--no-reload)", skipped = True)
        else:
            print(" -- reload (fresh inner objects from checkpoint)", flush = True)
            model = load_model()
            results2, finite2 = run_cases(torch, model, cases, vocab, args.device)
            report["pass2"] = results2
            add_check("finite_after_reload", finite2,
                      "post-reload rows finite" if finite2
                      else f"non-finite rows after reload: "
                           f"{sum(len(r['nonfinite_positions']) for r in results2.values())}")
            alive2 = [k for k, wr in refs if wr() is not None]
            if patch_active:
                snap2 = snapshot_svh(model)
                diff = {k: (snap1[k], snap2[k]) for k in snap1 if snap1[k] != snap2[k]}
                add_check("reload_applied_once", not diff,
                          "svh absmax identical to the first load => balance applied "
                          "exactly once to fresh checkpoint data" if not diff
                          else f"{len(diff)} svh differ after reload (compounded or lost "
                               f"balance): {dict(list(diff.items())[:5])}")
                if alive2:
                    add_check("no_retention_after_unload", False,
                              f"{len(alive2)} old inners pinned again after reload: {alive2[:5]}")
                audit2 = mrb.audit(model)
                report["audit_pass2"] = audit2
                c2 = audit2["counts"]
                add_check("audit_after_reload",
                          c2["already"] == c2["pairs"] and c2["mixed"] == 0,
                          f"counts={c2}")
            same = all(results1[cid]["top1"] == results2[cid]["top1"] for cid in results1)
            add_check("reload_top1_stable", same,
                      "top-1 identical after unload/reload" if same
                      else "top-1 differs after reload (balance not reproducible)")

        # -- agreement vs saved references --------------------------------------
        agree_targets = (("bf16_oracle", args.bf16_ref), ("balanced_artifact", args.balanced_ref))
        have_any = False
        for label, path in agree_targets:
            if not path:
                continue
            have_any = True
            per_case, rate = agree_with(path, results1)
            report[f"agreement_{label}"] = {"per_case": per_case, "rate": rate,
                                            "reference": str(Path(path).resolve())}
            if rate is None:
                add_check(f"agreement_{label}", False, "no overlapping cases in reference")
            else:
                add_check(f"agreement_{label}", rate >= args.min_agreement,
                          f"top-1 {rate:.4f} vs --min-agreement {args.min_agreement}")
        if not have_any:
            add_check("agreement", False, "no reference artifacts given (--bf16-ref/--balanced-ref)")

        report["pass"] = all(v.get("pass", True) for v in checks.values())
        print(f"\n == mlp range-balance probe: {'PASS' if report['pass'] else 'FAIL'} "
              f"(checks={len(checks)}, patch_active={patch_active})")
    except Exception as e:
        report["errors"].append(repr(e))
        report["traceback_tail"] = traceback.format_exc().splitlines()[-10:]
        print(f" !! probe crashed: {e}", file = sys.stderr)
        traceback.print_exc()
        report["pass"] = False
    finally:
        if model is not None:
            try:
                model.unload()
            except Exception as e:
                report["errors"].append(f"cleanup model.unload(): {e!r}")
                report["pass"] = False
        try:
            write_json(args.output, report)
            print(f" -- wrote {args.output}")
        except Exception as e:
            print(f" !! could not write {args.output}: {e}", file = sys.stderr)
            raise
        sys.stdout.flush()
    return 0 if report.get("pass") and not report["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
