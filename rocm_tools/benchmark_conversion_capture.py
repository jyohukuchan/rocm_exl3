#!/usr/bin/env python3
"""Rerunnable Hessian GEMM timing and terminal-head capture equivalence probe."""
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
    x = torch.randn(2048, 6144, device="cuda:0", dtype=torch.float16)
    xf = x.float()
    timings = {}
    values = []
    for name, fn in (("fp32", lambda: torch.mm(xf.T, xf)),
                     ("fp16_to_fp32", lambda: torch.mm(x.T, x, out_dtype=torch.float32))):
        for _ in range(3):
            y = fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            y = fn()
        torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - start) / 20
        assert torch.isfinite(y).all()
        values.append(y)
    relative = float((values[0] - values[1]).norm() / values[0].norm())
    del x, xf, y, values

    # Exercise workspace chunking and preserve the original non-finite-row policy.
    wide = Linear(None, "wide_probe", 9216, 128, qmap="wide")
    wide.device = torch.device("cuda:0")
    x = torch.randn(1, 128, 9216, dtype=torch.float16, device="cuda:0")
    x[0, 1, 0] = float("nan")
    x[0, 3, 0] = float("inf")
    wide_results = []
    for half_hessian in (False, True):
        params = {"capture": {}, "hessian_fp16": half_hessian}
        wide.capture_H(x, params)
        h = params["capture"]["wide"]
        assert h["count"] == 126 and torch.isfinite(h["H"]).all()
        assert h["inf_nan"].tolist() == [1, 1]
        wide_results.append(h["H"])
    wide_relative = float((wide_results[0] - wide_results[1]).norm() / wide_results[0].norm())
    assert wide_relative < 5e-5
    del wide_results, wide, x, h, params

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
              "gemm_shape": [2048, 6144], "gemm_seconds": timings,
              "hessian_relative_frobenius_error": relative,
              "wide_chunked_hessian_relative_error": wide_relative,
              "nonfinite_rows_excluded": 2,
              "head_rows": 16, "head_reference_rows": 5, "head_capture_seconds": head_times,
              "head_hessian_bitwise_equal": equal, "head_captured_tokens": int(a["count"])}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
