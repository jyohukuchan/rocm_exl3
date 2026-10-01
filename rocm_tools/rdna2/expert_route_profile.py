"""Opt-in target-MoE route recording for decode placement experiments.

GPU buffers are preallocated; recording adds GPU copies and is deliberately
excluded from performance measurements. Functions are importable by TP workers.
"""
import json
import re
from pathlib import Path


def flush(local_context):
    import numpy as np

    state = local_context.get("expert_route_profile")
    if state is None or state["case"] < 0:
        return None
    arrays, metadata = {}, {}
    for layer, count in enumerate(state["counts"]):
        if count:
            arrays[f"ids_{layer}"] = state["buffer"][layer, :count].cpu().numpy()
            metadata[str(layer)] = state["events"][layer]
    arrays["metadata"] = np.array(json.dumps(metadata))
    path = state["directory"] / f"rank-{local_context['device']}-case-{state['case']:03d}.npz"
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "layers": len(metadata), "rows": sum(state["counts"])}


def install(local_context, directory, max_rows=4096):
    import torch
    from rocm_tools.rdna2.multi_gpu import _iter_module_tree

    device = local_context["device"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with torch.cuda.device(device):
        state = {"directory": directory, "case": -1, "counts": [0] * 48,
                 "events": [[] for _ in range(48)],
                 "buffer": torch.empty((48, max_rows, 10), dtype=torch.int16, device=device)}
    local_context["expert_route_profile"] = state
    installed = []

    def make(original, layer):
        def routing(rows, cfg, y, params):
            selected, weights = original(rows, cfg, y, params)
            seqlens = params.get("cache_seqlens")
            position = int(seqlens[0]) if seqlens is not None else int(params.get("position", 0))
            if layer == 0 and position == 0 and rows > 8:
                flush(local_context)
                state["case"] += 1
                state["counts"] = [0] * 48
                state["events"] = [[] for _ in range(48)]
            if rows <= 8 and state["case"] >= 0:
                offset = state["counts"][layer]
                if offset + rows > max_rows:
                    raise RuntimeError("Expert route profile buffer exhausted")
                state["buffer"][layer, offset:offset + rows].copy_(selected)
                state["events"][layer].append({"offset": offset, "rows": rows, "position": position})
                state["counts"][layer] += rows
            return selected, weights
        return routing

    for module in local_context.get("modules") or []:
        for child in _iter_module_tree(module):
            match = re.search(r"\.layers\.(\d+)\.mlp$", getattr(child, "key", ""))
            if match and getattr(child, "routing_gate", None) is not None:
                if getattr(child, "expert_map", None) is not None:
                    raise ValueError("Capture original expert IDs without --tp-expert-order first")
                if child.num_experts != 512 or child.num_experts_per_tok != 10:
                    raise ValueError("This route recorder requires 512 experts and top-k 10")
                layer = int(match.group(1))
                child.routing_fn = make(child.routing_fn, layer)
                installed.append(layer)
    assert sorted(installed) == list(range(48)), installed
    return {"device": device, "layers": len(installed), "buffer_bytes": state["buffer"].numel() * 2}


def collect_events(directory, requests, rank=0):
    """Strip prefill tails and retain every target decode/verify computation.

    requests is the benchmark request list with response.usage.prompt_tokens,
    in capture order. Only one rank is read: replicated routers select the
    same global IDs, so combining both would double-count the work.
    """
    import numpy as np

    events = [[] for _ in range(48)]
    hits = np.zeros((48, 512), dtype=np.int64)
    for case, request in enumerate(requests):
        prompt_tokens = request["response"]["usage"]["prompt_tokens"]
        path = Path(directory) / f"rank-{rank}-case-{case:03d}.npz"
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            for layer in range(48):
                for event in metadata[str(layer)]:
                    if event["position"] < prompt_tokens - 1:
                        continue
                    offset, rows = event["offset"], event["rows"]
                    ids = data[f"ids_{layer}"][offset:offset + rows].ravel().astype(int)
                    hits[layer] += np.bincount(ids, minlength=512)
                    events[layer].append({"ids": ids.tolist(), "rows": rows,
                                          "case": case, "category": request["category"]})
    return events, hits


def main():
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--requests", required=True, help="Benchmark JSON with requests in capture order")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output-events", required=True)
    parser.add_argument("--output-hits", required=True)
    args = parser.parse_args()
    requests = json.loads(Path(args.requests).read_text())["requests"]
    events, hits = collect_events(args.profiles, requests, args.rank)
    Path(args.output_events).write_text(json.dumps(events) + "\n")
    np.save(args.output_hits, hits)


if __name__ == "__main__":
    main()
