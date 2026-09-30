# Rough changes on this branch relative to the fork parent

Parent: [`CarouselAether/rocm_exl3`](https://github.com/CarouselAether/rocm_exl3) at
`dd7a670065f37943f09a5eeb53818f38e9751472` (itself a ROCm port of
[`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) v1.5.0, validated
by the parent on `gfx1151`). This is a grouped summary, not a commit diary — `git log dd7a670065f3..HEAD`
and the per-report evidence links at the bottom are the detailed record.

What the parent documented (and what stopped being true here): the parent claimed **no
upstream C++/CUDA is modified**, only five upstream files differ outside the `rocm/`,
`rocm_py/`, `rocm_tools/` areas, and **tensor-parallel is unavailable on ROCm**. On this
branch all three statements changed.

## 1. Build & targeting (`setup.py`)

- `gfx1030` (RDNA2 / V620 / Navi 21) added to `SUPPORTED_GPU_ARCHS`.
- LDS budgets: `gfx1030` = 64 KiB per workgroup (its hardware limit); `gfx1201` (R9700,
  *measured*) corrected from 90 KiB to 64 KiB. Runtime still clamps to the device's real
  `sharedMemPerBlock`.
- The ROCm `hipcc` builder, backend selector and version gate (≥ 7.2.4) are the parent's,
  unchanged in spirit.

## 2. Native kernels — including the first upstream CUDA/C++ edit

- **`exllamav3/exllamav3_ext/reduction.cuh` (shared upstream header): modified.** A
  single-writer fix in `block_reduce_sum_broadcast_f` — only lane 0 writes the shared
  broadcast slot after `shfl_down`-based warp reduction, removing a latent race. This is
  the one upstream (non-`rocm/`) C++/CUDA file changed; CUDA is otherwise untouched code
  that this branch simply does not test.
- `exllamav3/exllamav3_ext/rocm/rdna_wmma.hip.h`: wave32 **SIMT/fdot2 matrix fallback** so
  the WMMA-shaped fragment API works on `gfx1030`, which has no WMMA encoding.
- `exllamav3/exllamav3_ext/rocm/quant/exl3_gemv_multirow_rdna.hip`: **padded-row bounds fix** —
  multirow GEMV used to read past valid rows (a 5-row allocation was touched as 8),
  reproducibly faulting on guard pages; fixed and verified with bit-pattern guard cases and
  real batch-4 MTP1–4 runs. Report:
  [rdna2_multirow_bounds_fix.md](rdna2_multirow_bounds_fix.md).

## 3. Tensor parallelism on the ROCm path (previously "not available")

- `exllamav3/model/model_tp.py`, `model_tp_fn.py`, `model_tp_alloc.py` substantially
  extended; new **`model_tp_rccl.py`**: RCCL-based collectives (torch process groups on
  ROCm) for 2-way TP. This is not the upstream CUDA `parallel/` kernels.
- **Std MoE router replication** (`EXL3_TP_REPLICATE_ROUTER=1`, default off): the small
  router is computed on each rank that owns routed experts, removing 2 broadcasts/layer —
  measured 194 → 98 collectives per body forward; ≈+10% AR decode, MTP code +12% engine
  (≈+4.7% observed; outputs/acceptance shift too, so it is not attributed to comms alone).
  Breakdown: [v620_tp_decode_optimization.md](v620_tp_decode_optimization.md).
- Validated on 2× V620 with Qwen3.8-Flash-Next only. Other rank counts/GPUs/models:
  untested.

## 4. Qwen3.8-Flash-Next, PLE/Engram and batch paths (shared Python)

- `exllamav3/modules/ngram_embedding.py` + new `exllamav3/util/mlock.py`: Engram n-gram
  table as **one physical CPU-RAM copy owned by a single TP rank**, with opt-in
  `EXL3_NGRAM_MLOCK=1` (plain `mlock` on existing pages, explicit failure if the limit is
  missing, unlock-on-unload ordering; residency audited via `mincore`).
- `exllamav3/cache/recurrent.py`: opt-in batched checkpoint pruning
  (`EXL3_BATCH_RECURRENT_PRUNE=1`); same-input batch-4 runs produced identical tokens with
  pruning off/on.
- `rocm_py` steering: `EXL3_ROCM_MOE_MGEMM_MAX_ROWS` (per-token MoE route threshold 8→
  opt-in up to 24; the V620 measurements use 20), plus updated docs of every switch in
  `exllamav3/rocm_py/__init__.py`.
- Qwen3.8 architecture plumbing: `architecture/qwen4_exp*.py`, `modules/ple.py`,
  `modules/qsa_indexer.py`, `modules/attn.py`, `gated_delta_net*`, `hyperconnections.py`,
  `block_sparse_mlp.py`, `mlp.py`, `module.py`, `model/model.py` and friends — TP/audit
  hooks, QSA cache mapping (K5/V4 with FP16 indexer planes; GDN recurrent states keep
  FP32/BF16), multirow bounds interactions.
- gfx1030 decode-attention tuning (`bc_attn.py` + `rocm_py/gqa_decode_tune.py`) and MLP
  FP16 overflow range balancing (`rocm_py/mlp_range_balance.py`) — both device/shape-gated,
  with opt-out flags available.
- `exllamav3/vendor/fla/*`: RDNA2 dispatch guards for the vendored FLA kernels.

## 5. Measurement harness & tests (`rocm_tools/`, `tests/`)

- New/expanded `rocm_tools/rdna2/` harness: frozen-token manifests, teacher-forced top-1
  comparison gate, bench with ITL sampling, layer-split placement auditor, stage profiles,
  `qwen_mtp_run.py` (MTP/layer-split runner), **`tp_run.py`** (TP2/LS batch runner with audits),
  `tp_trace_run.py`, power-policy pieces, and CPU-only test suites under
  `rocm_tools/rdna2/tests/`.
- `power_server.py` (user-started privileged power helper with policy restore) and
  `summarize_tp.py` (public metric summarizer) are part of this branch's public
  reproduction flow — see [reproduce_v620.md](reproduce_v620.md).
- New repo-level CPU/GPU test modules: `tests/test_qwen38_rdna2.py`,
  `test_conv_rdna2_dispatch.py`, `test_gdn_rdna2_dispatch.py`, `test_fla_rdna2_dispatch.py`,
  `test_shared_gate_projection.py`; `test_tp_export_attention.py` extended.

## 6. Documentation

- Validation reports under `doc/` (phase 2, V620 pair, TP decode, context/batch, R9700
  comparison, bounds fix, timing breakdown) and machine-readable validated configs
  (`qwen38_v620_tp_config.json`, `qwen38_v620_mtp_config.json`, `v620_pair_config.json` —
  documentation, not auto-loaded).
- Dated public benchmark bundle: [`benchmarks/2026-09-30/`](../benchmarks/2026-09-30/README.md).
- `README.md` rewritten for fork ancestry, honest verified matrix and the corrected TP /
  shared-code statements.

## What is *not* claimed here

- No CUDA-side revalidation: upstream CUDA code is unchanged apart from the shared
  `reduction.cuh` fix, and this branch does not run CUDA.
- RDNA3/3.5 coverage is inherited from the parent's `gfx1151` validation and was not
  independently rerun after these changes.
- R9700 results depend on documented comparison adapters/workarounds
  ([r9700_vs_v620.md](r9700_vs_v620.md)) — not general RDNA4 support.
- The bundled HTTP server was not newly validated for the TP/MTP paths (see README).

## Evidence index

- [rdna2_phase2_results.md](rdna2_phase2_results.md) — single V620, 2026-09-28
- [v620_pair_results.md](v620_pair_results.md) — 2×V620 layer split, 2026-09-29
- [v620_tp_decode_optimization.md](v620_tp_decode_optimization.md) — TP2 decode breakdown
- [qwen38_v620_context_batch.md](qwen38_v620_context_batch.md) — batch 1–4 + MTP1–4 + long context, 2026-09-30
- [qwen38_v620_tp_config.json](qwen38_v620_tp_config.json) — validated TP2 config snapshot
- [r9700_vs_v620.md](r9700_vs_v620.md) — R9700 workarounds & comparison
- [rdna2_multirow_bounds_fix.md](rdna2_multirow_bounds_fix.md) — bounds-fix detail

## OpenCode用HTTP API（2026-10-01）

- tool出力をOpenAI形式へ変換し、tool history、ID、reasoningを保持。
- incremental SSE argumentsとparallel tool calls、JSON Schema生成filterを統合。
- Qwen templateのparameter例・履歴を生成形式に合わせてrender。
- TP2/RCCL、元のMTP、K5/V4、Engram RAM＋mlock、power policyをHTTP loaderへ統合・監査。
- MTP中のfilter activation境界、AsyncGenerator終了時のqueue解放、複数候補usage集計を修正。
- OpenCodeの実read/write/shell往復、生成コードと14テスト、batch4動作を検証。

[日付付き検証記録](opencode_api_validation.md)と[API/OpenCode手順](../rocm_tools/exl3_server/README.md)。
