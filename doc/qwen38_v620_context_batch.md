# Qwen3.8 Flash Next: V620 TP2 context / batch / MTP measurements

Status: batch4 speed confirmation complete, 2026-09-30. Further long-context tests are deferred at the user's request. Only completed runs below are usable evidence. Capacity allocation alone is not an inference result.

## Configuration and artifacts

- Two Radeon Pro V620 (gfx1030), tensor parallel 2; original EXL3 3.05bpw target and original packed 3-bit MTP, no model replacement/requantization.
- Model revision `69e33439ae950f17bcbe95c98f117d80f759ab6d`, `/work/models/qwen38-flash-next-exl3-3.05bpw`.
- Engram: one physical CPU RAM table (32,640,156,672 bytes), per-table mincore residency audit; no disk streaming. K5/V4 for both target and draft QSA cache; recurrent states retain their original types.
- New native `/work/lib-context-mr-bounds`, SHA256 `57afa48c9a61b9e7ba2721917bf011096d7cc947e43b2a8444ee8350f0820718`.
- Final batch1–3 source `/work/runs/context-batch/source-prune-native-fixed`; batch4 confirmation uses `/work/runs/context-batch/source-mlock-fixed` with the reviewed RAM-lock addition. Manifests record source hashes and validated commits; snapshots do not contain `.git`.
- Explicit environment: `EXL3_TP_REPLICATE_ROUTER=1`, `EXL3_ROCM_MOE_MGEMM_MAX_ROWS=20`, `EXL3_BATCH_RECURRENT_PRUNE=1`, `HSA_ENABLE_SDMA=0`, `LD_PRELOAD=libhsa-runtime64.so`.
- Batch1: auto prefill, profile_peak draft/verify/decode, auto idle. Batch>1: profile_peak inference, auto after exit.
- Host artifacts: `/path/to/rocm-exl3-data/runs/context-batch` (container `/work/runs/context-batch`). JSON files contain prompts/hashes, actual cache/RAM/TP audits, delivery events, and per-rank memory data. `*-process.json` contains command/exit/power restoration and sampled board VRAM peaks.

## Measurement definitions

Inputs are frozen synthetic repeated filler ending with Japanese or code requests. These measure capacity and speed, not long-context retrieval accuracy. Generation is 256 tokens unless a smoke test explicitly says 64.

Main decode metric is delivered-token aggregate throughput from the first delivery to the last, excluding the first burst and including final queue-drain delay. Divide aggregate by batch for average per-sequence throughput. The separately reported common window covers the interval when every job is still generating; it excludes later drain and can be substantially faster. Neither metric is a sum of independently measured job speeds.

Prefill throughput is total input tokens divided by the latest first-delivery time, so it conservatively includes first-token overhead. Engine-internal decode timing is not interchangeable with observed delivery timing.

## Final 8K measurements

Each language has one warmup; batch1–3 have two timed groups and the final batch4 confirmation has three. Input 8192 and output 256 tokens per sequence; cache 8704 tokens per sequence, target load budgets 28/28 GiB, prefill chunk 2048. Batch1 dynamic MTP max4/confidence0.6; batch2–4 fixed MTP1 based on the screening below.

| Batch | Draft setting | Japanese decode aggregate tok/s | Code decode aggregate tok/s | Japanese/code prefill aggregate tok/s |
|---|---|---:|---:|---:|
| 1 | dynamic max4 | 38.88 | 50.52 | 476.76 / 479.41 |
| 2 | fixed1 | 55.85 | 56.68 | 473.83 / 467.18 |
| 3 | fixed1 | 66.25 | 73.95 | 472.67 / 465.49 |
| 4 | fixed1 | 70.56 | 66.51 | 471.80 / 459.68 |

Batch1 run: `final-8k-b1-d4`; Japanese repetitions 38.74–39.02, code 50.45–50.58 tok/s. Do not infer a universal gain from a two-repeat median.

Batch2 run: `final-8k-b2-d1`; per-sequence averages Japanese27.93/code28.34 tok/s. Common-window aggregate Japanese59.22/code61.04 tok/s. All final TP/KV/RAM audits and power restoration passed.

Batch3 run: `final-8k-b3-d1`; per-sequence averages Japanese22.08/code24.65 tok/s. Common-window aggregate Japanese70.06/code78.92 tok/s. Final audits and power restoration passed. Earlier fixed-MTP2 baseline was Japanese34.71/code41.93 aggregate; the difference combines draft/history settings and code changes, not an isolated kernel speedup.

Batch4 final run: `batch4-8k-mtp1`, three timed groups per language.

| Batch4 metric | Japanese | Code |
|---|---:|---:|
| Full-span aggregate decode median tok/s | 70.56 | 66.51 |
| Range of the three timed groups | 68.46–73.24 | 63.98–72.04 |
| Full-span aggregate /4 tok/s | 17.64 | 16.63 |
| Common-window aggregate decode median tok/s | 76.29 | 85.80 |
| Common-window aggregate /4 tok/s | 19.07 | 21.45 |
| Prefill aggregate median tok/s | 471.80 | 459.68 |
| Latest first-delivery median seconds | 69.45 | 71.28 |
| MTP draft acceptance | 70.57% | 92.24% |

All32 jobs (24 timed) completed, each with8192 uncached input tokens and256 outputs, actual draft1. Final TP/K5V4/single-owner RAM audits passed; native/model fingerprints matched the prior run. The24 repeated prompts produced identical token sequences to the earlier successful mlocked run. Exactly one process held32,640,159,744 locked bytes and released them to0 on unload. Board VRAM peaks were27.47/28.72GiB. Both V620s remained `profile_peak` in every sampled inference observation and returned to `auto` after exit. Evidence: `batch4-final-audit.json`, main report and process/power reports.

Earlier successful `final-8k-b4-d1` had only two timed groups: full-span medians Japanese66.89/code58.81, ranges66.56–67.21/48.61–69.01. It is retained as historical variation, not silently pooled into the three-repeat confirmation. The preceding failed RAM-audit run is excluded; all24 outputs matched its successful mlocked retry.

The additional run used `EXL3_HOST_MEM_RESERVE_MB=0` only as a recorded load-time diagnostic override after the normal guard rejected a prior attempt (`confirm-8k-b4-d1-r3`: needed31128MiB, available24997MiB, reserve2048MiB). The guard uses kernel MemAvailable without accounting for reclaimable ZFS ARC. OS/ARC/swap settings were not changed. Actual mlock and before/after table-residency audits remained mandatory; the selected library/config default host reserve is unchanged. Minimum sampled host MemAvailable during inference was3.66GiB. The failure remain in the artifact directory.

## Long context (partial; further tests deferred)

The model configuration allows 262144 positions. Cache capacity is per sequence in this table; the CLI `--cache-tokens` takes the sum across the batch. A 512-token margin leaves room for 256 generated tokens and speculative/cache bookkeeping.

| Batch | Cache capacity per sequence | Actual input + output | Fixed draft | Chunk | Load budgets GiB | Status |
|---|---:|---:|---:|---:|---|---|
| 1 | 262144 | 261632 + 256 | 4 | 2048 | 30 / 29 | Complete |
| 2–4 | Pending | Pending | 4 | Pending | Pending | Deferred by user |

`long-b1-c262144-d4`: prefill 463.54 tok/s, first delivery 564.43 s, observed decode 33.82 tok/s (engine 31.24), acceptance 43.82%. Sampled device-global VRAM peaks 29.22/29.72 GiB. These include allocations outside Torch; Torch peak alone underestimates physical usage. Sampling interval is 1 s and may miss short spikes.

Earlier allocation-only tests loaded 262144 per sequence for batch1–3 and 196608 for batch4; batch4 at 212992 explicitly ran out of memory. These are not validated inference limits. A previous batch3 full-size runtime failed RCCL before the bounds fix and is excluded. Batch3/4 require full input and generation before recommendation.

## MTP1–4 screening

2K input, one warmup plus one timed group per language/window; fixed draft lengths 1–4, shared max-history4 allocation. Actual draft windows were checked. These are screening data, not repeated final rankings. Batch1–3 used the older native; batch4 below uses the corrected native and pruning enabled. The bounds fix preserves valid-row arithmetic, but the software difference is recorded rather than hidden.

| Batch / language | MTP1 | MTP2 | MTP3 | MTP4 |
|---|---:|---:|---:|---:|
| 1 Japanese | 37.89 | 35.45 | 33.20 | 33.08 |
| 1 code | 44.78 | 49.51 | 54.30 | 46.09 |
| 2 Japanese | 58.31 | 50.95 | 45.03 | 26.43 |
| 2 code | 63.09 | 59.79 | 66.76 | 40.85 |
| 3 Japanese | 67.98 | 39.77 | 36.35 | 33.31 |
| 3 code | 73.19 | 47.53 | 47.63 | 47.90 |
| 4 Japanese | 70.37 | 44.66 | 40.36 | 30.80 |
| 4 code | 75.98 | 56.36 | 52.82 | 46.34 |

Values are full-span aggregate decode tok/s. More draft tokens increase draft and verification work; low acceptance can outweigh saved target invocations. Japanese batch1 acceptance fell from 57.1% at MTP1 to 37.1% at MTP4. Some dense/shared multirow paths still have an eight-row limit; expanding the MoE proxy does not eliminate every larger-batch fallback. MTP3/4 are supported and exercised, but are not universally faster.

## Changes and validation

1. `ffdafad`: opt-in ROCm per-token MoE route threshold 8→20 (accepted range 8–24). Actual target MoE microbenchmark: 9 rows 4.713→1.409 ms, 20 rows 7.371→2.821 ms. These are module timings, not model-wide speedups. All finite; relative L2 about 0.00062–0.00065 across different routes.
2. `ab1bc11`: bound padded-row loads in RDNA multirow GEMV. A five-row tight scratch allocation was read as eight rows. The old kernel reproducibly faulted at an unmapped guard page; the fixed kernel passed 16 strict bit-pattern guard cases, 200 numeric cases, the formerly failing real prefix-reuse case, and batch4 MTP1–4. See [bounds report](rdna2_multirow_bounds_fix.md).
3. `3a6313a`: opt-in batching of recurrent checkpoint pruning. CPU suite: 693 passing tests. Same-input batch4 sweep produced identical output tokens in all 64 jobs with pruning off/on. Timed group wall total 333.53→316.19 s (5.2% less time; 5.48% more throughput). This includes varying defragmentation/trim costs and only one timed group per condition; do not attribute the entire gain to bulk deletion or generalize it.
4. Prefix-reuse regression: 16/16 outputs identical across prune off/on, actual draft4, 1792 cached tokens on reuse, finite/audits pass. Cold vs reused execution itself has existing cross-path numerical differences; that is distinct from the pruning change.

## Interrupted measurement

The initial final batch2 attempt failed the Engram residency audit (7,951,096 / 7,968,789 pages resident). Its supervisor state was incomplete when inspected; no benchmark process remained. Evidence is preserved in `interrupted-final-b2-ram-audit/` and excluded from performance results. The batch2 retry passed all final audits without changing OS, ARC, or swap settings. The initial final batch4 run completed generation but failed the post-inference residency audit (7,907,144 / 7,968,789 pages resident; about240.8MiB nonresident). It is preserved in `failed-final-b4-ram-audit/` and excluded from final performance results. `cecfba7` adds opt-in `EXL3_NGRAM_MLOCK=1`: ordinary mlock on existing CPU table pages, no copy or CUDA registration, default off, explicit failure if locking cannot be granted. Unload drains prefetch then unlocks before dropping tensors. Root verification: 721 CPU tests +79 subtests; a real three-page mlock/mincore/munlock and insufficient-limit probe; `mlock-b4-d4-finite` with 8 real jobs and per-rank finite checks (3527/3934), all final audits passed. Exactly one process held32,640,159,744 locked bytes and returned to0 on unload.

For these runs, the launcher grants only the benchmark process and inherited children a34GiB MEMLOCK limit using a PID/namespace-checked handshake before model import (`--memlock-gib 34`). It changes no global OS/ARC/swap settings or persistent container configuration. Batch1–3 earlier runs passed both boundary residency audits without mlock; later runs additionally hold the table locked throughout. Locking stabilizes the RAM condition; it is not claimed as a compute optimization.

## Reproduction

The public, portable entry point is [reproduce_v620.md](reproduce_v620.md). It uses the tracked power helper and direct TP runner rather than the original private `run_bench.py` container supervisor. No locally named container image, private artifact directory, or prebuilt native extension is required; install the documented ROCm dependencies and build this source.

The [dated evidence bundle](../benchmarks/2026-09-30/README.md) contains the exact frozen prompts, normalized raw successful reports, timing-event summaries, environment and provenance. The public summarizer can recalculate the metrics without loading a model. Paths earlier in this historical report describe the original environment; use the public guide for current commands. Failed/exploratory local run tags remain historical notes rather than shipped artifacts.
