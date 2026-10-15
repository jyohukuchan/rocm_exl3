# SPDX-License-Identifier: MIT
"""Native-vs-candidate equality and alternating public-call HIP event timings."""

import json
import statistics
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.rocm_py.moe_prefill import gather_input, hadamard_batch


def time_calls(original, candidate, repetitions=5, iterations=40):
    for _ in range(10):
        original()
        candidate()
    graphs = []
    for call in (original, candidate):
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=torch.cuda.current_device())
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(iterations):
                call()
        graphs.append(graph)
    samples = [[], []]
    for repeat in range(repetitions):
        for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            call = (original, candidate)[which]
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            graphs[which].replay()
            end.record()
            end.synchronize()
            samples[which].append(begin.elapsed_time(end) * 1000 / iterations)
    return {
        "original_us": statistics.median(samples[0]),
        "candidate_us": statistics.median(samples[1]),
        "samples_us": samples,
    }


def main():
    rows = []
    for device in (0, 1):
        with torch.cuda.device(device):
            torch.manual_seed(31)
            ids = torch.tensor([3, 0, 2, 1], device="cuda", dtype=torch.long)
            for dtype in (torch.half, torch.float):
                for cols in (512, 2560):
                    for pre in (False, True):
                        if pre and dtype == torch.float32:
                            continue
                        x = torch.randn((4 * 193, cols), device="cuda", dtype=dtype)
                        scales = (
                            torch.randn((4, cols), device="cuda", dtype=torch.half)
                            * 0.15
                            + 1
                        ).half()
                        expected = torch.empty_like(x)
                        actual = torch.empty_like(x)
                        original = lambda: ext.had_r_128_batch(
                            x,
                            expected,
                            scales if pre else None,
                            None if pre else scales,
                            ids,
                            193,
                            1.0,
                        )
                        candidate = lambda: hadamard_batch(
                            x, actual, scales, ids, 193, pre
                        )
                        original()
                        candidate()
                        difference = (actual.float() - expected.float()).abs()
                        exact = torch.equal(actual, expected)
                        row = {
                            "device": device,
                            "dtype": str(dtype),
                            "cols": cols,
                            "pre": pre,
                            "exact": exact,
                            "maxabs": float(difference.max()),
                            "different": int(torch.count_nonzero(difference)),
                        }
                        if not exact:
                            rows.append(row)
                            print(json.dumps(row), flush=True)
                            raise AssertionError(row)
                        row.update(time_calls(original, candidate))
                        rows.append(row)
                        print(json.dumps(row), flush=True)
            # Ragged groups and padding sentinel: pure gather must be bit-exact.
            experts, cmax, cols = 4, 193, 2560
            counts = torch.tensor([193, 157, 148, 129], device="cuda", dtype=torch.long)
            starts = torch.cat(
                [torch.zeros(1, device="cuda", dtype=torch.long), counts.cumsum(0)[:-1]]
            )
            meta = torch.stack((ids, starts, counts, ids, ids, ids))
            total = int(counts.sum())
            x = torch.randn((513, cols), device="cuda", dtype=torch.half)
            x[-1].zero_()
            tokens = torch.randint(0, 512, (total + 1,), device="cuda")
            tokens[-1] = 512
            ar = torch.arange(cmax, device="cuda")
            source = torch.where(
                ar[None, :] < counts[:, None], starts[:, None] + ar[None, :], total
            ).reshape(-1)
            original = lambda: x.index_select(0, tokens.index_select(0, source)).view(
                experts, cmax, cols
            )
            candidate = lambda: gather_input(x, tokens, meta, experts, cmax, total)
            assert torch.equal(original(), candidate())
            row = {
                "device": device,
                "operation": "gather",
                "exact": True,
                **time_calls(original, candidate),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path("/work/runs/context-batch/moe-prefill-components.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
