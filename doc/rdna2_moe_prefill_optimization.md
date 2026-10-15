# RDNA2 EXL3 MoE grouped reconstruction

The optional `EXL3_ROCM_BATCH_RECON=auto` route groups heavy-expert weight
reconstruction and surrounding work while calling the installed hipBLAS GEMM
for each expert at its original matrix shape. The ROCm default remains off
until an end-to-end improvement is established. See [controls](env_vars.md#exl3_rocm_batch_recon-default-0).

## Scope and numerical contract

Tested target: Qwen3.8 Flash Next EXL3 3.05bpw on two Radeon PRO V620 gfx1030
GPUs, true TP2/RCCL, H2560/I640, 512 experts (256/256 or257/255 per TP pair), top-k10,
three-bit mul1 weights, FP16 intermediates and SiLU. Triton3.7 is required.
No CPU expert split, latent projection or activation clamp is covered by auto.
Other models and unavailable dependencies retain the individual-expert route.
The general folded-reconstruction default remains unchanged, including streamed
CPU experts; only exact ROCm auto explicitly selects unfolded reconstruction.

The exact route keeps the fused threshold128, each expert's unpadded GEMM M,
FP16 rounding at activation-transform boundaries, and FP32 weighted accumulation
in increasing expert order. The optional host-only bridge forwards to the loaded
engine's `hgemm_recon` symbol. It avoids rebuilding the GPU core and introducing
unrelated compiler or kernel changes. The bridge must match the installed
Python/Torch ABI. No new model quantization is performed.

## Work attempted

The previous diagnostic timings are cumulative stream spans over7936 prefill
tokens/188 MoE forwards per GPU, including host gaps. They are not pure kernel
times or achievable savings. Each of these >=100ms areas was addressed:

| Area | Previous GPU0/GPU1 span | Candidate |
|---|---:|---|
| Fused EXL3 expert kernel | 3814/3921ms | Tried four CUs per expert instead of eight; rejected because changing K-partition boundaries changed FP16 partial sums. The original scheduler remains. |
| Weight reconstruction | 165/186ms | Reconstruct count-sorted groups through existing pointer-table batch APIs. Keep original GEMM shapes. |
| Activation Hadamards | 231/252ms | Fuse input gather with gate/up pre-transforms; fuse gate/up post-transforms, half SiLU and down pre-transform. Keep explicit half rounding and disable FMA contraction. |
| Weighted individual-expert scatter | 167/166ms | Store heavy outputs in deterministic slots and accumulate valid weighted contributions in expert order with one Triton launch. |
| Scratch/dispatch/preparation | 244/308ms | Pack IDs/starts/counts once per active layer and gather projection pointer tables once, replacing per-group transfers. Bound simultaneous live scratch slabs. |

These candidates can reduce individual launch counts without guaranteeing a
shorter total inference. Grouped slabs still carry padding and temporary traffic,
and preserving original GEMM shapes retains one BLAS launch per expert/projection.

## Validation

- Original-core exact route:128 teacher-forced output positions on literal frozen
  Japanese/code prompts; mean/max KL, NLL delta and relative PPL delta all zero.
  This is a numerical equivalence sample, not a general quality benchmark.
- Both GPUs: ragged original-shape GEMMs and ordered accumulation match exactly;
  Hadamard/activation fusion matches the native chain for multiple scales,
  negative signs, strided metadata and changed-input graph replays.
- All63488 finite FP16 inputs match the native half SiLU calculation bit for bit.
- Runtime pilot checks one group from each of47 layers per rank against the
  individual-expert reference, verifies finite outputs and the TP/KV/RAM audit.
- CPU regression suite:188 passed,1 skipped,59 subtests; optional bridge/policy
  tests:12 passed. GPU correctness tests run separately from performance trials.

The initially faster strided-batched/threshold32 candidate failed predeclared
teacher-forced gates (mean KL0.00165666, max KL0.02747185, relative PPL delta
+0.6999%). It is not selected by auto. A full native rebuild also changed free
output tokens versus the installed original core, so final comparisons preserve
the original core SHA256
`57afa48c9a61b9e7ba2721917bf011096d7cc947e43b2a8444ee8350f0820718`.
The V1 rebuilt-core MTP ON performance block is excluded because a separate GPU
test might have overlapped it; its numerical checks remain diagnostic.

## Reproduction

Build and enable the host bridge as described in [environment controls](env_vars.md).
Use `rocm_tools/rdna2/check_moe_exact_group.py`, `check_moe_prefill_fusion.py`,
`check_moe_metadata.py` and `check_moe_silu_precision.py` with the same engine,
Torch and native library path before using another runtime.

The final clean benchmark uses frozen Japanese/code8192-token prompts, one warmup
and two timed samples per language/block,256 output tokens, AR and dynamic MTP,
and OFF/ON/OFF blocks with an identical original GPU core. Model weights, router
replication, decode cap20, recurrent pruning, K5/V4 cache,8704-token capacity,
chunk2048,28/28GiB budgets,34GiB process memlock and one locked RAM Engram table
are fixed. Prefill uses the existing auto power policy; decode uses peak; cleanup
restores auto. HC fusion is off in both conditions. No GPU component tests overlap
these blocks. OS, ARC and swap settings are unchanged.

Raw commands, process/power/memory audits, numerical reports, rejected trials and
source snapshots are under
`/home/homelab1/coding-local/aiter-rdna2-artifacts/moe-prefill-optimization`.
Final performance results were collected by `run_sidecar_final.py`; all six
performance blocks and the47-layer pilot completed and passed cleanup/audits.

## Final decision: retain ROCm default off

| Mode / input | OFF prefill s | ON prefill s | Time reduction | OFF decode tok/s | ON decode tok/s |
|---|---:|---:|---:|---:|---:|
| AR / ja | 16.234 | 16.140 | +0.58% | 28.73 | 29.31 |
| AR / code | 16.527 | 16.420 | +0.65% | 28.73 | 27.94 |
| MTP / ja | 16.774 | 16.904 | -0.78% | 35.81 | 36.79 |
| MTP / code | 17.076 | 16.851 | +1.32% | 55.22 | 56.81 |

OFF is the mean of four timed samples from the surrounding two blocks; ON is
the mean of two timed samples. Positive reduction means shorter prefill time.
The mean reduction across four cases is0.44%. MTP Japanese prefill regressed
0.78%, and AR code observed decode throughput fell2.77%. The small differences
and run-to-run spread do not establish a reliable general acceleration. No
statistical significance is claimed. The predeclared promotion gates failed;
`EXL3_ROCM_BATCH_RECON` therefore remains0. `auto` provides the exact opt-in route.

All1536 generated token positions per block match the reference in both modes;
MTP draft acceptance/rejection histories match too. Model fingerprint, cache
format/capacity, native hash, TP/RAM audits and power restoration match. The
installed host bridge was copied alongside the original core in
`/work/lib-context-mr-bounds`; the original core hash is unchanged. In this
installation, setting `EXL3_ROCM_BATCH_RECON=auto` is sufficient with the existing
library path. Default and auto startup import checks both pass.

See [machine-readable results](rdna2_moe_prefill_results.json) for every timed
prefill sample, drift, observed decode rate and promotion gates. The complete
raw reports and runtime manifests remain in the artifact directory above.
