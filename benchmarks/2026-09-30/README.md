# V620 TP2 benchmark results — 2026-09-30 (JST)

These are measured results for this experimental fork, not estimates or a promise for other machines. [Reproduction guide](../../doc/reproduce_v620.md) · [Fork changes](../../doc/fork_changes.md) · [Detailed investigation](../../doc/qwen38_v620_context_batch.md).

## Configuration

- Two AMD Radeon Pro V620 cards (`gfx1030`, 32 GiB each), tensor parallel 2.
- `turboderp/Qwen3.8-Flash-Next-exl3`, branch `3.05bpw_h5_ng5`, revision `69e33439ae950f17bcbe95c98f117d80f759ab6d`. Original packed 3-bit MTP, 5-bit head and Engram; no replacement MTP weights in these results. Acquire weights separately under their applicable license.
- K5/V4 QSA cache for both target and draft; recurrent state retains its original precision. One physical CPU RAM Engram table, 32,640,156,672 bytes.
- Each request has 8,192 input tokens and generates 256 tokens. Frozen synthetic filler plus Japanese/code requests; this is a capacity/throughput test, not a long-context comprehension or model-quality evaluation.
- Batch 1 uses dynamic draft length up to 4 (confidence 0.6), auto prefill and profile_peak draft/verify/decode. Batch 2–4 use fixed draft 1 and profile_peak throughout inference. Original GPU policies are restored after exit.
- Target load budgets 28/28 GiB; prefill chunk 2048; total cache capacity `8704 * batch` tokens. Router replication, MoE row threshold 20 and batched recurrent pruning enabled.
- One warmup group per language; two timed groups for batch 1–3 and three for the final batch 4 confirmation. Values below are medians.

## Results

All throughput values are **aggregate tok/s across the batch**. Per-request effective throughput is aggregate divided by batch.

| Batch | Prefill Japanese / code | Decode Japanese / code, including final drain | Decode Japanese / code, common generation window |
|---:|---:|---:|---:|
| 1 | 476.76 / 479.41 | 38.88 / 50.52 | 38.88 / 50.52 |
| 2 | 473.83 / 467.18 | 55.85 / 56.68 | 59.22 / 61.04 |
| 3 | 472.67 / 465.49 | 66.25 / 73.95 | 70.06 / 78.92 |
| 4 | **471.80 / 459.68** | **70.56 / 66.51** | **76.29 / 85.80** |

Batch 4 final-drain-inclusive ranges were 68.46–73.24 Japanese and 63.98–72.04 code. Dividing its median aggregate by 4 gives 17.64/16.63 tok/s per request. Common-window rates correspond to 19.07/21.45 tok/s per request. Latest first-delivery medians were 69.45/71.28 seconds for all four 8K requests. Draft acceptance was 70.57%/92.24%.

**Timing definitions:** conservative prefill = total input tokens / latest first delivery (includes first-token overhead). Full-span decode counts delivered tokens between the first and last deliveries, excluding the first burst and including final queue-drain delay. Common-window decode measures the overlap when every job is still generating, before later completion/drain. It is not the sum of independently measured job rates. Code generation is faster in the common window but pays a larger completion delay in these 256-token runs.

The earlier two-repeat batch4 result (66.89/58.81 tok/s including drain) remains documented separately; it is not pooled into the final three-repeat confirmation.

## Validation and memory

All 32 final batch 4 jobs completed (24 timed). There were no prefix-cache hits. Actual draft windows were1, TP/KV/single-owner RAM audits passed, and 24 prompts shared with the prior run produced identical token sequences. Engram was held with ordinary `mlock`, not copied or CUDA-host-registered: one process held 32,640,159,744 page-aligned bytes and returned to 0 on unload. This proves table residency, not that every page of every process avoided swap.

Both V620s reported profile_peak in all sampled inference observations and returned to auto afterward. Sampled board VRAM peaks were 27.47/28.72 GiB (1-second sampling can miss brief spikes); minimum sampled host MemAvailable was 3.66 GiB. The run used the recorded, process-local `EXL3_HOST_MEM_RESERVE_MB=0` load-time guard override because kernel MemAvailable did not include reclaimable ZFS ARC. Actual memory locking and before/after residency audits remained enabled. OS/ARC/swap settings were not changed; the normal library reserve is still enabled by default.

The earlier unlocked batch 2 and batch 4 attempts that failed RAM residency, and the additional attempt that failed the pre-load host-memory guard, are **not performance samples**. The successful earlier batch 1–3 runs passed boundary residency audits without mlock; final batch 4 additionally locked its table throughout.

Separately, batch 1 with fixed MTP4 completed **261,632 input +256 output tokens** in a 262,144-token cache (prefill 463.54 tok/s, observed decode 33.82 tok/s). The maximum practical context for batch 2–4 is **not established**; further tests were deferred. Allocation-only successes are not inference limits.

## Included evidence

- [results.json](results.json): medians, ranges, acceptance and conditions.
- [constraints.txt](constraints.txt): observed direct Python dependency versions (not a full system/transitive lockfile).
- [environment.json](environment.json): CPU, RAM, software/compiler versions and native SHA256. The tested HIP SDK 7.14 and Torch ROCm 7.2 wheel labels differ; this is the actual mixed stack, not a claim that all ROCm versions work.
- [provenance.json](provenance.json): engine commits and hashes of the compressed evidence/input manifests. No model weights or native binaries are included.
- `prompts-*.json.gz`: byte-for-byte frozen token-ID manifests after decompression, valid for the specified model/tokenizer revision. Do not reuse them for unrelated tokenizers.
- `long-prompts-cap262144-b1.json.gz`: exact input for the separately recorded batch1 long-context observation. It is not part of the default8K speed run.
- `reports/*.json.gz`: raw report structure with filesystem paths/GPU UUIDs normalized. Timing values, outputs, token IDs, hashes and audits are retained. Original and normalized hashes are distinguished. These are historical observations, not assertions about files on the reader's machine.

To independently recompute the final batch 4 summary without running inference:

```bash
RUN_DIR=$(mktemp -d)
gzip -dc benchmarks/2026-09-30/reports/batch4-8k-mtp1.json.gz > "$RUN_DIR/report.json"
python -m rocm_tools.rdna2.summarize_tp "$RUN_DIR/report.json"
```

This reads the recorded delivery events; it does not load a model, access GPUs, change power settings or download anything. New inference runs use the [reproduction guide](../../doc/reproduce_v620.md). The dated evidence stays separate from later improvements.
