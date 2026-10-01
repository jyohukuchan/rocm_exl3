"""Permutation validation and constrained offline placement invariants."""
import importlib.util
import gzip
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(relative):
    spec = importlib.util.spec_from_file_location("placement_test", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


placement = load("exllamav3/util/expert_placement.py")
planner = load("rocm_tools/rdna2/plan_expert_placement.py")
profile = load("rocm_tools/rdna2/expert_route_profile.py")


@pytest.mark.parametrize("bad", [[0, 0, 2], [0, 1], [0, 1, 3], [False, 1, 2], ["0", 1, 2]])
def test_invalid_orders_rejected(bad):
    with pytest.raises(ValueError):
        placement.validate_order(bad, 3)


def test_inverse_preserves_selected_original_order(tmp_path):
    order = [4, 1, 5, 0, 3, 2]
    inverse = placement.inverse_order(order)
    picks = [5, 0, 3]
    assert [order[inverse[x]] for x in picks] == picks
    path = tmp_path / "order.json"
    path.write_text(json.dumps({"version": 1, "num_experts": 6, "orders": {"moe": order}}))
    assert placement.load_orders(path) == {"moe": tuple(order)}


def test_optimizer_keeps_quota_and_reduces_cooccurring_imbalance():
    events = [{"ids": [0, 1], "case": 0, "category": "chat", "rows": 1},
              {"ids": [2, 3], "case": 1, "category": "coding", "rows": 1}]
    matrix, weights = planner.event_matrix(events, 4)
    initial = np.array([True, True, False, False])
    result, cost = planner.improve_assignment(matrix, weights, initial)
    assert result.sum() == 2
    assert cost < planner.imbalance_cost(matrix, weights, initial)
    assert sorted(planner.storage_order(result)) == list(range(4))


def test_plan_preserves_ownership_and_rejects_mismatched_plan():
    key = "model.layers.0.mlp"
    plan = [{key: [0, 2, "experts"]}, {key: [2, 4, "experts"]}]
    events = [[{"ids": [0, 1], "case": 0, "category": "chat"},
               {"ids": [2, 3], "case": 1, "category": "coding"}]]
    result = planner.plan_orders(plan, [[1, 1, 1, 1]], events)
    order = result["orders"][key]
    assert sorted(order) == [0, 1, 2, 3]
    assert len(set(order[:2]) & {0, 1}) == 1
    assert result["training_imbalance"][key]["candidate"] == 0
    plan[1][key][0] = 1
    with pytest.raises(ValueError):
        planner.plan_orders(plan, [[1, 1, 1, 1]])


def test_capture_aggregation_excludes_prefill_tail(tmp_path):
    metadata = {str(layer): [{"offset": 0, "rows": 1, "position": 90},
                             {"offset": 1, "rows": 2, "position": 99}]
                for layer in range(48)}
    arrays = {f"ids_{layer}": np.tile(np.arange(10), (3, 1)) for layer in range(48)}
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez(tmp_path / "rank-0-case-000.npz", **arrays)
    events, hits = profile.collect_events(tmp_path, [{"category": "coding",
                                                     "response": {"usage": {"prompt_tokens": 100}}}])
    assert hits.sum() == 48 * 2 * 10
    assert all(len(layer) == 1 and layer[0]["rows"] == 2 for layer in events)


def test_public_domains_are_equal_weight_and_use_hits(tmp_path):
    path = tmp_path / "map.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        for domain, expert, count in (("a", 0, 100), ("b", 1, 1), ("b", 1, 1)):
            hits = [0] * 512
            hits[expert] = count
            stream.write(json.dumps({"domain": domain, "layers": {
                str(i): {"hits": hits, "importance": [999] * 512} for i in range(48)}}) + "\n")
    mix = planner.public_frequency(path, ("a", "b"))
    assert np.all(mix[:, :2] == .5)
    assert np.all(mix[:, 2:] == 0)


@pytest.mark.parametrize("fail_at", ["install", "flush"])
def test_profile_example_unloads_on_capture_failure(monkeypatch, tmp_path, fail_at):
    from rocm_tools.exl3_server import runtime

    unloaded = []

    def dispatch(devices, function, args):
        if function.__name__ == fail_at:
            raise RuntimeError("capture failed")

    model = SimpleNamespace(loaded_tp=True, plan=[{}, {}], active_devices=[0, 1],
                            tp_worker_dispatch_wait_multi=dispatch)
    result = SimpleNamespace(model=model, unload_models=lambda: unloaded.append(True))
    monkeypatch.setenv("EXL3_EXPERT_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(runtime, "load_runtime", lambda: result)
    bootstrap = runpy.run_path(str(ROOT / "examples/expert_profile_bootstrap.py"))
    with pytest.raises(RuntimeError, match="capture failed"):
        bootstrap["load_with_profile"]()
        result.unload_models()
    assert unloaded == [True]
