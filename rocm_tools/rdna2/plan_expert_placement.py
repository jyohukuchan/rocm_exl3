"""Offline fixed-quota expert placement candidates from activation profiles."""
import numpy as np


PUBLIC_DOMAINS = ("code_gen", "code_proj", "code_files", "code_doc", "cot", "dialogue", "instruct", "tools")


def public_frequency(path, domains=PUBLIC_DOMAINS):
    """Equal-domain layer shares from the public map's route hits, not REAP importance."""
    import gzip
    import json

    totals = {domain: np.zeros((48, 512), dtype=np.int64) for domain in domains}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["domain"] in totals:
                totals[row["domain"]] += np.array([row["layers"][str(i)]["hits"]
                                                   for i in range(48)], dtype=np.int64)
    shares = []
    for domain, hits in totals.items():
        denominator = hits.sum(axis=1, keepdims=True)
        if (denominator <= 0).any():
            raise ValueError(f"Missing activation hits for {domain}")
        shares.append(hits / denominator)
    return np.mean(shares, axis=0)


def frequency_assignment(frequency, quota):
    assigned = np.zeros(len(frequency), dtype=bool)
    counts = [0, 0]
    loads = [0., 0.]
    limits = [quota, len(frequency) - quota]
    for expert in np.argsort(-frequency, kind="stable"):
        rank = min((r for r in (0, 1) if counts[r] < limits[r]), key=lambda r: loads[r])
        assigned[expert] = rank == 0
        counts[rank] += 1
        loads[rank] += frequency[expert]
    return assigned


def storage_order(assigned):
    # Original ID order within each rank keeps import/debugging predictable.
    return np.flatnonzero(assigned).tolist() + np.flatnonzero(~assigned).tolist()


def event_matrix(events, num_experts=512):
    matrix = np.zeros((len(events), num_experts), dtype=np.float32)
    categories, case_work = {}, {}
    for i, event in enumerate(events):
        matrix[i] = np.bincount(event["ids"], minlength=num_experts)
        categories.setdefault(event["category"], set()).add(event["case"])
        case_work[event["case"]] = case_work.get(event["case"], 0) + len(event["ids"])
    weights = np.array([1. / len(categories) / len(categories[e["category"]]) /
                        case_work[e["case"]] for e in events], dtype=np.float32)
    return matrix, weights


def imbalance_cost(matrix, weights, assigned):
    difference = matrix @ assigned.astype(np.float32) - matrix.sum(axis=1) / 2
    return float(np.sum(np.abs(difference) * weights))


def improve_assignment(matrix, weights, assigned, iterations=50, candidates=8):
    assigned = assigned.copy()
    work = matrix.sum(axis=1) / 2
    difference = matrix @ assigned.astype(np.float32) - work
    cost = float(np.sum(np.abs(difference) * weights))
    if assigned.all() or not assigned.any():
        return assigned, cost
    for _ in range(iterations):
        gradient = matrix.T @ (np.sign(difference) * weights)
        left = np.flatnonzero(assigned)
        right = np.flatnonzero(~assigned)
        remove = left[np.argsort(-gradient[left], kind="stable")[:candidates]]
        add = right[np.argsort(gradient[right], kind="stable")[:candidates]]
        pairs = [(a, b) for a in remove for b in add]
        deltas = np.stack([matrix[:, b] - matrix[:, a] for a, b in pairs], axis=1)
        costs = (np.abs(difference[:, None] + deltas) * weights[:, None]).sum(axis=0)
        best = int(np.argmin(costs))
        if float(costs[best]) >= cost - 1e-9:
            break
        a, b = pairs[best]
        assigned[a], assigned[b] = False, True
        difference += deltas[:, best]
        cost = float(costs[best])
    return assigned, cost


def plan_orders(plan, frequency, events=None, seed_frequencies=()):
    """Create Qwen-style TP2 orders without changing each layer's expert quota.

    frequency is [layers, experts]; events is an optional list of per-layer
    event lists produced from an unpermuted decode/verify route capture.
    """
    import re

    frequency = np.asarray(frequency)
    if frequency.ndim != 2 or not np.isfinite(frequency).all() or (frequency < 0).any():
        raise ValueError("Frequency must be a finite nonnegative [layers, experts] matrix")
    if len(plan) != 2:
        raise ValueError("This planner requires a two-rank plan")
    num_experts = frequency.shape[1]
    orders, scores = {}, {}
    for key, (first, last, unit) in plan[0].items():
        if unit != "experts":
            continue
        if first != 0 or plan[1].get(key) != [last, num_experts, "experts"]:
            raise ValueError(f"Expected contiguous TP2 expert ownership for {key}")
        match = re.search(r"\.layers\.(\d+)\.mlp$", key)
        if match is None or int(match[1]) >= len(frequency):
            raise ValueError(f"No frequency row for {key}")
        layer = int(match[1])
        assigned = frequency_assignment(frequency[layer], last)
        if events is not None:
            if not events[layer]:
                raise ValueError(f"No decode events for {key}")
            matrix, weights = event_matrix(events[layer], num_experts)
            baseline = np.arange(num_experts) < last
            seeds = [baseline, assigned] + [frequency_assignment(seed[layer], last)
                                            for seed in seed_frequencies]
            candidates = [improve_assignment(matrix, weights, seed)
                          for seed in seeds]
            assigned, cost = min(candidates, key=lambda x: x[1])
            scores[key] = {"baseline": imbalance_cost(matrix, weights, baseline), "candidate": cost}
        orders[key] = storage_order(assigned)
    if not orders:
        raise ValueError("Plan contains no expert-parallel MoE layers")
    return {"version": 1, "num_experts": num_experts, "orders": orders,
            "training_imbalance": scores}


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON two-rank TP plan")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frequency", help="NPY [layers, experts] route hits or shares")
    source.add_argument("--public-map", help="Public expert_map_sequences.jsonl.gz")
    parser.add_argument("--domains", nargs="+", default=PUBLIC_DOMAINS)
    parser.add_argument("--save-frequency", help="Optionally save the input frequency matrix as NPY")
    parser.add_argument("--events", help="JSON per-layer decode/verify event lists (enables cooccurrence swaps)")
    parser.add_argument("--seed-frequency", action="append", default=[], help="Additional NPY swap starting points")
    parser.add_argument("--output", required=True, help="Output JSON for --tp-expert-order")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    events = json.loads(Path(args.events).read_text()) if args.events else None
    frequency = (np.load(args.frequency, allow_pickle=False) if args.frequency
                 else public_frequency(args.public_map, args.domains))
    if args.save_frequency:
        np.save(args.save_frequency, frequency)
    seeds = [np.load(path, allow_pickle=False) for path in args.seed_frequency]
    if any(seed.shape != frequency.shape or not np.isfinite(seed).all() or (seed < 0).any() for seed in seeds):
        parser.error("Seed frequencies must be nonnegative finite matrices matching the frequency shape")
    result = plan_orders(plan, frequency, events, seeds)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
