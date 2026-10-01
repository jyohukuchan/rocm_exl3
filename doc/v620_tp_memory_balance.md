# V620 TP2: balancing memory for larger cache capacity

2026-10-02 (Asia/Tokyo). Adjusting the existing target-model `-gs` budgets
reduced the memory imbalance without introducing CPU offload, splitting QSA
heads, changing quantization, or adding communication stages. The current
service retains a 512Ki cache and uses `-gs 28.375,27.625`.

## Why equal budgets do not produce equal whole-device usage

`-gs` supplies the TP target's allocation budgets. The packed MTP model and its
cache are already on output GPU 1 before the target loads; vision is on GPU 0.
With explicit budgets, the TP loader uses those numbers instead of subtracting
the preceding allocations from device free memory. Other runtime workspaces
also differ between ranks. `-gs 28,28` therefore does not mean the two devices
will have equal total VRAM usage.

The target allocator can shift expert and vocabulary slices and choose which
device owns whole QSA/PLE components. This is decided at model load. There is
no extra transfer or migration during each generation step. Whole components
make the relationship between budgets and real memory discontinuous: a larger
budget shift can move an entire layer and overshoot the desired balance.

## Actual measurements

V620 x2, Qwen3.8 Flash Next EXL3 3.05bpw, original 3-bit dynamic MTP max4,
confidence0.6, K5/V4, compact QSA raw rings, batch1, xhigh, vision enabled,
Engram in one locked RAM owner. All conditions used the same four cold coding
prompts; the first was warmup and three were timed. Each request had 8,344
uncached input tokens and generated exactly 256 tokens. Decode includes
reasoning tokens. No image was submitted.

GPU 0 is the vision/worker side (PCI43:00.0); GPU 1 is the MTP/output side
(PCI03:00.0). Numbers below are board-level inference peaks sampled every
0.2 seconds after readiness, including warmup; they exclude model loading.
Speed is the median of the three timed requests, using engine metrics.

| Cache capacity | Target budgets GiB | GPU0 / GPU 1 peak GiB | Prefill tok/s | Decode tok/s |
|---|---|---|---:|---:|
| 512Ki reference | 28 / 28 | 29.85 / 30.90 | 515.47 | 48.40 |
| 512Ki larger shift | 28.75 / 27.25 | 30.86 / 29.89 | 509.44 | 50.59 |
| **512Ki selected** | **28.375 / 27.625** | **30.47 / 30.28** | **507.16** | **50.16** |
| 768Ki capacity check | 28.5 / 27.5 | 31.70 / 31.35 | 511.90 | 48.97 |

At 512Ki, the peak difference fell from 1.049 to 0.199 GiB (about 81% less).
The largest device peak fell by 0.426 GiB. Prefill changed by -1.61% and decode
by +3.64%. Three repeats and differing MTP acceptance are insufficient to claim
a speedup; the observed speeds were comparable. All requests completed and
health remained normal.

The 768Ki result proves allocation plus the stated short inference workload,
not a full-768Ki input or long-context quality. Both cards stayed below their
31.984375 GiB physical capacity in the sampled workload, with about 0.28/0.63 GiB
remaining at their respective peaks. Larger or different workloads, especially
vision/concurrency, need their own peak checks.

See the [normalized evidence](../benchmarks/2026-10-02/tp-memory-balance.json)
for prompt hashes, all warmup/timed metrics, ranges, cache audits, and startup
Torch memory. One earlier reference startup timed out while loading a worker;
it is excluded. All rows above are complete successful runs.

## Applying the settings

Use the [existing server setup](../rocm_tools/exl3_server/README.md), keeping
the same model, MTP, cache, power helper and process limits. Change only:

```text
512Ki: -gs 28.375,27.625 -cs 524288 --context-limit 524288
768Ki: -gs 28.5,27.5     -cs 786432 --context-limit 786432
```

Budgets are ordered by the inference process's GPU IDs, not the host
`rocm-smi` index. They sum to the same 56 GiB target budget in these comparisons.
Restart the model to apply them. The 512Ki setting is the deployed default;
the 768Ki setting remains a tested capacity option.

Recheck the budgets when changing cache capacity, batch slots, draft placement,
or vision placement. The target cache cost is incorporated in each new TP plan,
while the separate MTP cache grows on GPU 1. A single fixed split should not be
assumed balanced for every context size. A first estimate is to shift half the
measured board-memory difference from the fuller device's target budget to the
other device, then inspect the actual plan/peaks and correct whole-layer jumps.
Reducing this difference cannot solve a shortage of total VRAM.

## Reproducing the coding workload

Start a fresh server for each condition with the same xhigh defaults. Read
`prompt_fixture` from the evidence JSON, construct each prompt as
`prefix.format(k=k) + repeat_line * repeat_count + suffix_separator + suffix`,
and submit `k=0,1,2,3` in that order to `/v1/chat/completions`:

```json
{
  "model": "qwen38-local",
  "messages": [{"role": "user", "content": "<constructed prompt>"}],
  "max_tokens": 256,
  "temperature": 0,
  "include_timings": false,
  "stream": false
}
```

Do not override thinking: the measured server defaults were enabled/xhigh.
Exclude `k=0` from speed medians, require zero cached tokens, and compare the
`exl3_metrics.prefill_tokens_per_second` and `output_tokens_per_second` fields.
Sample both cards' `mem_info_vram_used` sysfs counters during the workload.
