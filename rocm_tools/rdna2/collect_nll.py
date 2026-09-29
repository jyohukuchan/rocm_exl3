#!/usr/bin/env python3
"""Full-corpus next-token NLL/PPL over the SAME frozen manifest IDs.

collect_top1.py samples 1024 of the 2030 frozen tokens, so its PPL is not the
full corpus: this tool derives a DENSE manifest (per case every position
0..len(ids)-2, target ids[p+1]; final unlabeled token excluded; literal IDs
copied verbatim), runs the EXISTING collector in-process with _harvest_row
WRAPPED (top-1 sidecar kept; restored in finally) and harvests FP64
NLL = logsumexp(row[:valid_vocab]) - row[target] -- scalars only, no full-vocab
dumps. Fail-closed: a cursor over the expected flat (case, pos, target) order
stops labeling at ANY anomaly, and a pass also requires the collector artifact
complete plus per-case position/count/order/ids-hash agreement; --limit-cases
(silent flat-target misalignment risk) and <2-token cases are refused.
CANDIDATE evidence only -- reference-vs-candidate judgement is out of scope,
NOT ground truth. Dense manifest / top-1 sidecar / report land next to
--nll-out.
"""

import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rocm_tools.rdna2 import collect_top1   # CPU-safe at import (torch loads lazily)
from rocm_tools.rdna2.common import (
    POSITION_SEMANTICS, TOP1_FORMAT, canonical_bytes, manifest_digest, now_utc,
    read_json, sha256_hex, validate_manifest, write_json)

NLL_FORMAT = "rdna2-fullcorpus-nll/1"


def nll_row(torch, logits_row, vocab, target):
    """Stable FP64 NLL of one row; torch is passed in (never imported here), the
    slice stays on the row's device, only the scalar crosses to Python."""
    row = logits_row[:vocab].to(dtype = torch.float64)
    return float(torch.logsumexp(row, 0).item() - row[target].item())


def build_dense_manifest(src, src_path):
    """Dense plan + manifest in one pass: per case positions 0..len(ids)-2 with
    targets ids[p+1] (final unlabeled token excluded); literal IDs/hashes copied
    verbatim; source digest kept as provenance; dense digest recomputed. Cases
    with len_ids != len(ids) or <2 tokens are refused. Returns (dense, plan, errors)."""
    cases, plan, errs = [], [], []
    for c in src["cases"]:
        ids = c["ids"]
        if c.get("len_ids") != len(ids):
            errs.append(f"case {c['case_id']}: len_ids {c.get('len_ids')} != len(ids) {len(ids)}")
        elif len(ids) < 2:
            errs.append(f"case {c['case_id']}: only {len(ids)} token(s) -- no next-token coverage")
        else:
            cases.append(dict(c, positions = list(range(len(ids) - 1))))
            plan.extend((c["case_id"], p, ids[p + 1]) for p in range(len(ids) - 1))
    dense = {k: v for k, v in src.items() if k not in
             ("manifest_sha256", "created_utc", "tool", "requested_positions",
              "total_positions", "cases")}
    dense.update({"created_utc": now_utc(), "tool": "rocm_tools/rdna2/collect_nll.py",
                  "requested_positions": len(plan), "total_positions": len(plan),
                  "position_semantics": POSITION_SEMANTICS,
                  "source_manifest": {"path": str(Path(src_path).resolve()),
                                      "sha256": src["manifest_sha256"],
                                      "total_positions": src["total_positions"]},
                  "cases": cases})
    dense["manifest_sha256"] = manifest_digest(dense)
    return dense, plan, errs


class NllCursor:
    """Labels harvested rows in strict dense-plan order; once broken stays True,
    so no later row is ever labeled under a possibly-stale alignment."""

    def __init__(self, plan):
        self.plan, self.losses, self.problems, self.broken = plan, [], [], False

    def wrap(self, orig):
        def wrapper(torch, row, vocab):
            try:
                top1, finite = orig(torch, row, vocab)   # original top-1 behavior kept
                self.record(torch, row, vocab, finite)
            except Exception as exc:
                self._fail(f"harvest failed; alignment lost: {exc!r}")
                raise
            return top1, finite
        return wrapper

    def _fail(self, msg):
        self.problems.append(msg)
        self.broken = True   # fail-closed: stop labeling this run

    def record(self, torch, row, vocab, finite):
        if self.broken:
            return
        if len(self.losses) >= len(self.plan):
            return self._fail("harvest call beyond the dense plan (alignment lost)")
        cid, p, target = self.plan[len(self.losses)]
        if not finite:
            return self._fail(f"non-finite logit row at case {cid} pos {p}")
        if not 0 <= target < vocab:
            return self._fail(f"target {target} outside [0, {vocab}) at case {cid} pos {p}")
        nll = nll_row(torch, row, vocab, target)
        if not math.isfinite(nll):
            return self._fail(f"non-finite NLL at case {cid} pos {p}")
        self.losses.append({"case_id": cid, "pos": p, "target": target, "nll": nll})


def cross_check(cursor, dense, art, rc):
    """All alignment gates; an empty problem list is what licenses a PPL claim."""
    errs = [f"collector exited rc={rc}"] if rc != 0 else []
    if art is None:
        return errs + ["top-1 sidecar missing: nothing to align the cursor against"]
    expected = sum(len(c["positions"]) for c in dense["cases"])
    if art.get("format") != TOP1_FORMAT or art.get("manifest_sha256") != dense["manifest_sha256"]:
        errs.append("top-1 artifact not produced from the dense manifest (format/digest)")
    if not art.get("complete") or art.get("errors"):
        errs.append(f"collector artifact not complete, errors: {art.get('errors') or []}")
    if (art.get("total_positions"), art.get("collected_positions")) != (expected, expected):
        errs.append(f"collected {art.get('collected_positions')}/{art.get('total_positions')} "
                    f"positions != expected full-corpus labels {expected}")
    for case in dense["cases"]:
        cid, want = case["case_id"], case["positions"]
        got = (art.get("cases") or {}).get(cid) or {}
        top1 = got.get("top1") or []
        if (got.get("status") != "ok" or got.get("positions") != want
                or got.get("n_positions") != len(want) or got.get("len_ids") != case["len_ids"]
                or got.get("ids_sha256") != case["ids_sha256"]
                or len(top1) != len(want) or any(t is None for t in top1)):
            errs.append(f"case {cid}: dense manifest/collector alignment failed "
                        f"(status {got.get('status')!r})")
    if cursor.problems or cursor.broken:
        errs.append("NLL cursor: " + "; ".join(cursor.problems or ["broken without detail"]))
    if len(cursor.losses) != expected:
        return errs + [f"labeled {len(cursor.losses)} rows < expected {expected} "
                       "(partial corpus: no plausible PPL)"]
    for k, (cid, p, t) in enumerate(cursor.plan):
        l = cursor.losses[k]
        if (l["case_id"], l["pos"], l["target"]) != (cid, p, t):
            errs.append(f"cursor entry {k} misaligned against the dense plan")
            break
    return errs


def aggregate(losses):
    """Mean NLL and PPL (exp of mean) globally and per case, dense case order."""
    per, order = {}, []
    for l in losses:
        if l["case_id"] not in per:
            per[l["case_id"]] = []
            order.append(l["case_id"])
        per[l["case_id"]].append(l["nll"])

    def stat(vals):
        m = sum(vals) / len(vals)
        try:
            return m, math.exp(m)
        except OverflowError:
            return m, None

    m, ppl = stat([l["nll"] for l in losses])
    return {"n_labels": len(losses), "mean_nll": m, "perplexity": ppl,
            "per_case": [{"case_id": c, "n_labels": len(per[c]), "mean_nll": stat(per[c])[0],
                          "perplexity": stat(per[c])[1]} for c in order]}


def build_parser():
    ap = collect_top1.build_parser()
    ap.prog = "collect_nll.py"
    ap.description = ("Full-corpus next-token NLL/PPL: runs collect_top1.py in-process over a "
                      "derived dense manifest, harvesting scalar FP64 NLLs. --manifest stays the "
                      "SOURCE manifest; -o is forced to the top-1 sidecar next to --nll-out.")
    for act in ap._actions:
        if act.dest == "output":
            act.required = False
    ap.add_argument("--nll-out", required = True,
                    help = "NLL result JSON; dense manifest and top-1 sidecar go beside it")
    return ap


def run(args) -> int:
    src = read_json(args.manifest)
    if verr := validate_manifest(src):
        raise SystemExit(" !! source manifest invalid, refusing: " + "; ".join(verr))
    if args.limit_cases:
        raise SystemExit(" !! FATAL: --limit-cases could silently misalign the flat target "
                         "order; full-corpus NLL requires every manifest case")
    src_path = str(Path(args.manifest).resolve())
    dense, plan, perr = build_dense_manifest(src, src_path)
    if perr:
        raise SystemExit(" !! FATAL: dense labeling plan refused: " + "; ".join(perr))
    if dverr := validate_manifest(dense):
        raise SystemExit(f" !! FATAL: derived dense manifest failed validation: {dverr}")
    out = Path(args.nll_out)
    dense_path = out.with_name(out.stem + ".dense_manifest.json")
    top1_path = out.with_name(out.stem + ".top1.json")
    params = vars(args).copy()
    write_json(dense_path, dense)
    args.manifest, args.output = str(dense_path), str(top1_path)
    cursor = NllCursor(plan)
    orig = collect_top1._harvest_row
    collect_top1._harvest_row = cursor.wrap(orig)
    try:
        try:
            rc = collect_top1.run(args)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
        except Exception as e:
            cursor._fail(f"collector crashed: {e!r}")
            rc = 1
    finally:
        collect_top1._harvest_row = orig   # the collector is restored, always
    try:
        art = read_json(top1_path)
    except Exception:
        art = None
    errs = cross_check(cursor, dense, art, rc)
    a = art or {}
    report = {
        "format": NLL_FORMAT, "tool": "rocm_tools/rdna2/collect_nll.py",
        "created_utc": now_utc(), "repo_git_commit": a.get("repo_git_commit"),
        "semantics": "NLL at p = fp64 logsumexp(logits[p][:valid_vocab]) - logits[p][ids[p+1]]; "
                     "every position 0..len(ids)-2 per case, final unlabeled token excluded",
        "evidence_status": "candidate full-corpus NLL/PPL for this execution identity only; "
                           "reference-vs-candidate judgement out of scope -- NOT ground truth",
        "source_manifest": {"path": src_path, "sha256": src["manifest_sha256"],
                            "total_positions": src["total_positions"]},
        "dense_manifest": {"path": str(dense_path.resolve()), "sha256": dense["manifest_sha256"],
                           "expected_labels": len(plan)},
        "top1_sidecar": {"path": str(top1_path.resolve()), "collector_rc": rc,
                         "complete": a.get("complete"),
                         "collected_positions": a.get("collected_positions")},
        "identity": {k: a.get(k) for k in ("backend", "device", "model_dir", "model_fingerprint",
                                           "execution", "env")},
        "params": params, "valid_vocab_size": dense["valid_vocab_size"],
        "rows_counted": len(cursor.losses), "expected_labels": len(plan),
        "losses_sha256": sha256_hex(canonical_bytes(cursor.losses)), "losses": cursor.losses,
        "complete": not errs, "nll": aggregate(cursor.losses) if not errs else None,
        "problems": errs}
    write_json(out, report)
    n = report["nll"]
    print(f" -- NLL {'COMPLETE' if not errs else 'INCOMPLETE'}: "
          f"rows={len(cursor.losses)}/{len(plan)}"
          + (f" mean_nll={n['mean_nll']:.6f} ppl={n['perplexity']:.6f}" if n else "")
          + f" -> {out}")
    for e in errs:
        print(f"    ERR {e}", file = sys.stderr)
    return 0 if not errs else 1


def main(argv = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        print(f" !! NLL collection failed: {e}", file = sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
