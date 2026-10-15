# SPDX-License-Identifier: MIT
"""Compare aligned teacher-forced distributions using CPU FP64 reductions."""

import argparse
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    assert len(reference) == len(candidate) > 0
    kl, nll_ref, nll_candidate, top1_match = [], [], [], []
    for r, c in zip(reference, candidate):
        assert r["ids_sha256"] == c["ids_sha256"] and r["targets"] == c["targets"]
        assert r["logits"].shape == c["logits"].shape
        assert torch.isfinite(r["logits"]).all() and torch.isfinite(c["logits"]).all()
        for i, target in enumerate(r["targets"]):
            p = torch.log_softmax(r["logits"][i].double(), dim=-1)
            q = torch.log_softmax(c["logits"][i].double(), dim=-1)
            kl.append(float((p.exp() * (p - q)).sum()))
            nll_ref.append(float(-p[target]))
            nll_candidate.append(float(-q[target]))
            top1_match.append(int(p.argmax()) == int(q.argmax()))
    mean = lambda values: sum(values) / len(values)
    delta = mean(nll_candidate) - mean(nll_ref)
    gates = json.loads(args.gates.read_text())
    report = {
        "positions": len(kl),
        "mean_kl": mean(kl),
        "max_kl": max(kl),
        "mean_nll_reference": mean(nll_ref),
        "mean_nll_candidate": mean(nll_candidate),
        "mean_nll_delta": delta,
        "relative_ppl_delta": math.expm1(delta),
        "argmax_agreement": mean(top1_match),
        "per_position_kl": kl,
        "nll_reference": nll_ref,
        "nll_candidate": nll_candidate,
        "gates": gates,
        "teacher_forced": True,
        "unconstrained_output_agreement_claimed": False,
    }
    report["passed"] = (
        report["mean_kl"] <= gates["mean_kl_max"]
        and report["max_kl"] <= gates["max_kl_max"]
        and abs(delta) <= gates["absolute_mean_nll_delta_max"]
        and abs(report["relative_ppl_delta"])
        <= gates["absolute_relative_ppl_delta_max"]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if not isinstance(v, list)}, indent=2
        )
    )
    assert report["passed"], "candidate failed predeclared distribution/NLL gates"


if __name__ == "__main__":
    main()
