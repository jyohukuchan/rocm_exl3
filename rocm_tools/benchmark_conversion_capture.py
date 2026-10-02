#!/usr/bin/env python3
"""FP32-Hessian finite-row validation and terminal-head capture equivalence."""
import argparse
import json
import time

import torch
from exllamav3 import Config, Model
from exllamav3.modules import Linear


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.manual_seed(731)
    torch.set_num_threads(8)
    # A non-finite activation excludes its token row, not an entire feature.
    probe = Linear(None, "finite_probe", 128, 128, qmap="finite")
    probe.device = torch.device("cuda:0")
    x = torch.randn(1, 128, 128, dtype=torch.float16, device="cuda:0")
    x[0, 1, 0] = float("nan")
    x[0, 3, 0] = float("inf")
    params = {"capture": {}}
    probe.capture_H(x, params)
    h = params["capture"]["finite"]
    assert h["H"].dtype == torch.float32 and torch.isfinite(h["H"]).all()
    assert h["count"] == 126 and h["inf_nan"].tolist() == [1, 1]
    finite_rows = x.view(128, 128).float()
    finite_rows = finite_rows[torch.isfinite(finite_rows).all(dim=1)]
    torch.testing.assert_close(h["H"], finite_rows.T @ finite_rows)
    del probe, x, h, params, finite_rows

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    head = model.modules[-1]
    assert isinstance(head, Linear) and head.qmap
    head.load(torch.device("cuda:0"))
    rows = [torch.randn(1, 2048, head.in_features, dtype=torch.float16, device="cuda:0")
            for _ in range(16)]
    captures = []
    head_times = {}
    for fast in (False, True):
        capture = {}
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i, row in enumerate(rows):
            y = head.forward(row, {"capture": capture, "capture_only_input": fast and i >= 5})
            assert y.shape[-1] == (head.in_features if fast and i >= 5 else head.out_features)
            del y
        torch.cuda.synchronize()
        head_times[str(fast)] = time.perf_counter() - start
        captures.append(capture[head.qmap])
    a, b = captures
    equal = torch.equal(a["H"], b["H"]) and torch.equal(a["inf_nan"], b["inf_nan"])
    assert equal and a["count"] == b["count"] == 16 * 2048
    result = {"torch": torch.__version__, "hip": torch.version.hip,
              "gpu": torch.cuda.get_device_properties(0).gcnArchName,
              "hessian_dtype": "float32",
              "nonfinite_rows_excluded": 2,
              "head_rows": 16, "head_reference_rows": 5, "head_capture_seconds": head_times,
              "head_hessian_bitwise_equal": equal, "head_captured_tokens": int(a["count"])}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
