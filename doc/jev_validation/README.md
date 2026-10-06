# JEV-27B-VL validation artifacts, 2026-10-06

See [../jev.md](../jev.md) for the precision policy, runtime/API behavior,
hardware conditions, results and limitations.

- `cases*.json`: eight basic cases and four independent sensitive cases.
- `reference-bf16*.json`: original BF16 backbone with FP32 PEFT decision head.
- `quantized-{r9700,v620}*.json`: native single-GPU outputs, full-distribution
  comparisons and GPU/source/native-binary fingerprints.
- `quantized-http-*-results.json`: actual HTTP responses on each quantized GPU.
- `source-manifest.json`: downloaded revision and verified HF LFS digests.
- `quantized-pack-manifest.json`: all 39 output files with sizes/SHA256.
- `quantized-integrity-report.json`: actual storage/precision accounting and
  exact-source decision-row/adapter checks.
- `conversion-status.json`: original wrapper exit 127; retained unchanged.
  `conversion-artifact-verification.json` records separate completion/index
  evidence and explains the shell bookkeeping error.
- `canonical-pack-location.json`: host model directory and shared weight inodes.
- `previous-service-restored.json`: existing Qwen API service restored and
  localhost health response checked after releasing the validation GPUs.
- `http_verify.py` and `verify_pack.py`: validation scripts for the documented
  host containers. Paths refer to that host's `/work` and `/models` mounts.

These small cases establish runtime fidelity and execution coverage. They do
not measure broad decision accuracy, recalibration, reasoning quality, MTP
performance or the model's maximum usable context. The unquantized V620
baseline used serialized kernels; the final quantized runs did not.

Original logs and conversion working files remain at
`/home/homelab1/datapool/rocm-exl3-rdna2/runs/jev-20261006` and
`/home/homelab1/datapool/rocm-exl3-rdna2/build-jev27-vl-4bpw`.
