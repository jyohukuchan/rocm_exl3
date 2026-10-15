# SPDX-License-Identifier: MIT
"""Test fused MoE gathers/Hadamards/activation against the original native chain."""

import json
from pathlib import Path

import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.rocm_py.moe_prefill import gather_pre_hadamard, post_silu_pre
from rocm_tools.rdna2.check_moe_prefill import time_calls


def main():
    rows = []
    for device in (0, 1):
        with torch.cuda.device(device):
            torch.manual_seed(42)
            b, cmax, hidden, intermediate = 4, 193, 2560, 640
            ids = torch.tensor([3, 0, 2, 1], device="cuda", dtype=torch.long)
            counts = torch.tensor([193, 157, 148, 129], device="cuda", dtype=torch.long)
            starts = torch.cat(
                [torch.zeros(1, device="cuda", dtype=torch.long), counts.cumsum(0)[:-1]]
            )
            meta = torch.stack((ids, starts, counts, ids, ids, ids))
            # A group is a strided slice of the layer-wide metadata slab.
            full_meta = torch.cat(
                (
                    torch.zeros((6, 1), device="cuda", dtype=torch.long),
                    meta,
                    torch.zeros((6, 2), device="cuda", dtype=torch.long),
                ),
                dim=1,
            )
            meta = full_meta[:, 1:5]
            ids = meta[0]
            total = int(counts.sum())
            x = torch.randn((513, hidden), device="cuda", dtype=torch.half)
            x[-1].zero_()
            tokens = torch.randint(0, 512, (total + 1,), device="cuda")
            tokens[-1] = 512
            sg = (torch.randn((4, hidden), device="cuda") * 0.7).half()
            su = (torch.randn((4, hidden), device="cuda") * 0.7).half()
            ar = torch.arange(cmax, device="cuda")
            source = torch.where(
                ar[None, :] < counts[:, None], starts[:, None] + ar[None, :], total
            ).reshape(-1)
            gref = torch.empty((b * cmax, hidden), device="cuda", dtype=torch.half)
            uref = torch.empty_like(gref)

            def original_gather():
                gathered = x.index_select(0, tokens.index_select(0, source))
                ext.had_r_128_batch(gathered, gref, sg, None, ids, cmax, 1.0)
                ext.had_r_128_batch(gathered, uref, su, None, ids, cmax, 1.0)

            candidate_gather = lambda: gather_pre_hadamard(
                x, tokens, meta, ids, sg, su, b, cmax, total
            )
            original_gather()
            g, u = candidate_gather()
            assert torch.equal(g.view_as(gref), gref) and torch.equal(
                u.view_as(uref), uref
            )
            row = {
                "device": device,
                "operation": "gather_dual_pre_hadamard",
                "exact": True,
                **time_calls(original_gather, candidate_gather),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            # Independent native post-Hadamards -> native half SiLU -> down pre-Hadamard.
            for multiplier in (0.02, 1.0, 5.0):
                g = (
                    torch.randn(
                        (b * cmax, intermediate), device="cuda", dtype=torch.half
                    )
                    * multiplier
                )
                u = torch.randn_like(g) * multiplier
                pg = (torch.randn((4, intermediate), device="cuda") * 0.7).half()
                pu = (torch.randn((4, intermediate), device="cuda") * 0.7).half()
                pd = (torch.randn((4, intermediate), device="cuda") * 0.7).half()
                gg, uu, act, expected = (torch.empty_like(g) for _ in range(4))

                def original_post():
                    ext.had_r_128_batch(g, gg, None, pg, ids, cmax, 1.0)
                    ext.had_r_128_batch(u, uu, None, pu, ids, cmax, 1.0)
                    ext.silu_mul(gg, uu, act, 0.0)
                    ext.had_r_128_batch(act, expected, pd, None, ids, cmax, 1.0)

                candidate_post = lambda: post_silu_pre(g, u, pg, pu, pd, ids, cmax)
                original_post()
                actual = candidate_post()
                difference = (actual.float() - expected.float()).abs()
                row = {
                    "device": device,
                    "operation": "post_hadamard_silu_down_pre",
                    "multiplier": multiplier,
                    "exact": torch.equal(actual, expected),
                    "maxabs": float(difference.max()),
                    "different": int(torch.count_nonzero(difference)),
                }
                print(json.dumps(row), flush=True)
                assert row["exact"], row
                if multiplier == 1.0:
                    row.update(time_calls(original_post, candidate_post))
                rows.append(row)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.Stream(device=device)):
                replay_output = candidate_post()
            g.normal_()
            u.normal_()
            original_post()
            graph.replay()
            assert torch.equal(
                replay_output, expected
            ), "graph replay retained stale inputs"
            rows.append(
                {"device": device, "operation": "changed_graph_inputs", "exact": True}
            )
    Path("/work/runs/context-batch/moe-prefill-fusion-components.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
