# 768Ki default: synthetic full-cache memory verification

2026-10-02 (Asia/Tokyo). The V620-pair service now defaults to 786,432 cache
tokens, target budgets 28.5/27.5 GiB, batch1, K5/V4, original 3-bit dynamic MTP
max4/confidence0.6, xhigh and vision enabled. OpenCode and LibreChat's configured
context limits were raised to match; the output limit remains 32,768.

## Method

This checks memory at the full window without evaluating a 768Ki-token prompt.
The normal target, MTP, vision and locked-RAM Engram were loaded first.

1. Every target and draft KV/QSA tensor was initialized over all 3,072 physical
   pages. Quantized scales were finite/nonzero; pooled keys had a finite pattern.
   The target's 12 layers and the draft's 1 layer each had 786,432-token capacity.
2. A real generator job allocated every page, with a unique physical block-table
   entry for each. CPU token IDs represented 786,395 synthetic prompt tokens;
   recurrent/cache metadata was advanced to 784,128, skipping those prefix
   computations. The skipped history is synthetic, not a valid language state.
3. The real model evaluated the final 2,048+218 prompt tokens using full-context
   sparse attention, target state export and the normal MTP prefill path.
4. Fixed MTP4 generated 32 tokens near the end of the cache. This exercises the
   maximum draft/verification window allowed by the production dynamic max4.
   The greatest target write end was 786,430, within the 786,432 allocation; the
   last two slots remain reserved by the generator's speculative bookkeeping.
5. Actual-vocabulary logits were checked for finiteness. Unused vocabulary
   padding is excluded because the sampler intentionally masks it with `-inf`.

The GPU probe took 11.72 seconds after normal model startup. It initialized all
pages; it did not alias every historical block-table entry to page 0.

## Results

| Check | Result |
|---|---|
| Physical pages allocated | 3,072 / 3,072 |
| Actual target + MTP cache bytes | 7,361,003,520 (6.85546875 GiB) |
| Real tail-prefill tokens | 2,266 |
| Generated tokens | 32 |
| Maximum target write end, exclusive | 786,430 |
| Finite actual-vocabulary logits checked | 7,938,464 |
| OOM / out-of-bounds failure | None |

Board memory was sampled every 50 ms during the probe:

| Inference GPU | Peak GiB | Physical capacity GiB | Remaining at sampled peak GiB |
|---|---:|---:|---:|
| GPU 0, vision/worker side | 31.4883 | 31.984375 | 0.4961 |
| GPU 1, MTP/output side | 31.1821 | 31.984375 | 0.8023 |

The preceding [ordinary coding workload](v620_tp_memory_balance.md) peaked
higher, at 31.70/31.35 GiB, and also passed. Torch peak/reserved bytes and actual
cache geometry are included in the [normalized evidence](../benchmarks/2026-10-02/full-cache-768ki-memory.json).
Sampling can miss short non-Torch allocations; successful native execution and
Torch allocator peaks complement these sampled counters.

This proves the configured cache allocation and near-end attention/MTP memory
paths for batch1 and the stated chunk/window sizes. It does not evaluate
768Ki-token comprehension, retrieval, accumulated real recurrent state, or
different batch/vision workloads. Synthetic output is not a quality result.

## Reproduce and deploy

The helper is [probe_cache_capacity.py](../rocm_tools/rdna2/probe_cache_capacity.py).
After the usual [server runtime setup](../rocm_tools/exl3_server/README.md), call
`exercise(loaded_runtime, report_path, pci_devices=(...))` from a diagnostic
bootstrap immediately after `load_runtime()` returns. PCI addresses must be
ordered by inference GPU IDs. Defaults are this pair's 43:00.0/03:00.0.
The helper uses the runtime's phase-switched power policy.

Always discard/unload this diagnostic runtime and start a fresh production
runtime afterwards: its KV contents were overwritten with synthetic history.
Production settings:

```text
-gs 28.5,27.5 -cq 5,4 -cs 786432 --context-limit 786432
-ambs 1 -chunk_size 2048 -ndt 4 -dds -dc 0.6
--max-output-tokens 32768
-ctk '{"enable_thinking":true,"reasoning_effort":"xhigh"}'
```

OpenCode advertises 786,432 context tokens, 753,664 input tokens and 32,768 output
tokens. LibreChat's local endpoint uses `maxContextTokens: 786432`. The server
continues reserving space for the configured speculative window and clamps
requested output lengths accordingly.
