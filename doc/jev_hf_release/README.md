---
license: apache-2.0
base_model: autotrust/JEV-27B-VL
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - en
  - ja
tags:
  - exl3
  - rocm
  - rocm_exl3
  - jev
  - quantized
  - mixed-precision
  - vision
  - typed-decisions
  - lora
---

# JEV-27B-VL — EXL3 4bpw, ROCm

**Use [jyohukuchan/rocm_exl3](https://github.com/jyohukuchan/rocm_exl3) for
the complete JEV System 1 decision runtime in this release.** These are EXL3
quantized weights, not standard Transformers/vLLM weights. No hosted inference
provider, generic EXL3 runtime compatibility, or NVIDIA execution is claimed.

This is a mixed-precision quantization of
[autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL/tree/f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc),
whose unchanged generation backbone is Qwen3.8-27B. The decision LoRA remains
separate: System 1 selects it per forward; ordinary System 2 generation uses
the base model. This quantization does not add model training or new datasets.
The original model card is preserved in `SOURCE_MODEL_CARD.md`; its benchmark
claims describe the upstream model, not new evaluations of this quantization.

## Precision and contents

| Component | Precision |
|---|---|
| 400 language projections | EXL3 4bit |
| Normal generation head | EXL3 6bit |
| Eight MTP projections | EXL3 6bit; packaged, not used by the JEV server |
| Vision, embeddings, norms and recurrent controls | Unquantized source storage (vision BF16) |
| Decision LoRA | Original adapter; FP32 execution |
| Decision vocabulary readout | 264 exact source BF16 rows + LoRA, computed in FP32 |

This is a **4bpw language trunk**, not a 4bpw whole-package claim. The model
pack is approximately 16.25 GiB including adapter/tokenizer/metadata. Weight
tensors alone occupy 16,966,676,548 bytes (4.886 effective bits per original
parameter); metadata and decision assets add storage. Calibration used 250×2048
tokens and FP32 Hessians. The output contains 17 weight shards / 2,426 indexed
tensors, plus the adapter and exact decision-row sidecar.

`config.json`, `quantization_config.json`, tokenizer/preprocessor files,
`adapter_vllm/`, `calibration.json`, `decision_config.json` and
`decision_rows.safetensors` must stay together. **Do not merge the decision
adapter into the backbone or replace its readout with quantized vocabulary
logits.** `FILE_MANIFEST.json` records the release file sizes and SHA256 values.

## Run

The tested engine revision is
[`7675fa26cdf52fbe5a90060a69c60c752855fd8f`](https://github.com/jyohukuchan/rocm_exl3/tree/7675fa26cdf52fbe5a90060a69c60c752855fd8f).
Follow that revision's
[ROCm build requirements](https://github.com/jyohukuchan/rocm_exl3/blob/7675fa26cdf52fbe5a90060a69c60c752855fd8f/README.md#requirements).
Build the native extension for `gfx1030` (V620) or `gfx1201` (R9700); a standard
PyPI ExLlamaV3 install alone does not provide this fork's JEV server. The
validation stack used Python 3.12, PyTorch `2.12.0+rocm7.2` and a custom ROCm
7.14 SDK. No prebuilt runtime image or clean-machine install verification is
included in this model release.

After installing that engine and `fastapi`, `uvicorn`, `transformers` and
`huggingface_hub` in its environment:

```sh
hf download jyohukuchan/JEV-27B-VL-exl3-4bpw --local-dir ./JEV-27B-VL-exl3-4bpw
git clone https://github.com/jyohukuchan/rocm_exl3
cd rocm_exl3
git checkout 7675fa26cdf52fbe5a90060a69c60c752855fd8f
# Build/install the native extension following the linked ROCm requirements.
ulimit -n 65536
# On V620, use HSA_ENABLE_SDMA=0 as in the verified environment.
python -m rocm_tools.jev_server \
  -m ../JEV-27B-VL-exl3-4bpw --context 16384 --port 3960
```

Or use the included wrapper with `--engine-dir /path/to/rocm_exl3`:
`python /path/to/model/serve_exl3.py --engine-dir /path/to/rocm_exl3`.
The server raises its open-file limit and disables the incompatible MLP
range-balance policy. Default bind is loopback; `--api-key` enables Bearer auth.

```sh
curl http://127.0.0.1:3960/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"kind":"noul","state":"Tokyo is the capital of Japan.","question":"Is this correct?","thinking":"off"}'
```

System 1 returns calibrated option probabilities for `noul`, six-level
`score`, or 2–256-way `choice`, including images. `thinking: auto/on` optionally
invokes the base reasoning model for noul/choice and mixes its result with
System 1. The server also implements llama.cpp-style `/v1/systemone` and
non-streaming text/image `/v1/chat/completions`.
See the pinned [JEV runtime/API guide](https://github.com/jyohukuchan/rocm_exl3/blob/7675fa26cdf52fbe5a90060a69c60c752855fd8f/doc/jev.md)
for supported fields and limitations. Video input, streaming, tools and
constrained output schemas are not implemented by this JEV server.

## Verified results (2026-10-06)

Both single-GPU candidates match the original BF16 reference's top decision
on all 12 small text/vision regression cases. Combined mean KL
`D_KL(reference || quantized)` is 0.00070321 (R9700) / 0.00068287 (V620).
Maximum per-option probability difference is 0.039731 / 0.038927, respectively.
Peak allocated VRAM in the basic screen is 15.60 / 15.52 GiB at context 16,384.
Base→decision→base generation isolation and actual image/decision/adaptive HTTP
execution passed on both GPUs. No serialized-kernel debug setting was used.

Warmed serial HTTP **median / p95** (one GPU, fresh KV/SSM, no prefix/image reuse):

| Workload | R9700 | V620 |
|---|---:|---:|
| Short text, 35 prompt tokens, 30 samples | 299 / 300 ms | 1,195 / 1,199 ms |
| 16 choices, 121 tokens, 30 samples | 313 / 314 ms | 2,710 / 2,735 ms |
| Text, 2,846 tokens, 30 samples | 2,683 / 2,686 ms | 12,107 / 12,130 ms |
| Browser screenshot + element text, 277 decisions | 588 / 626 ms | 2,191 / 2,498 ms |

The browser test uses 60 synthetic mail/shop/settings tasks. Both GPUs succeed
in 57/60 (95%), receive identical request bodies and choose identical actions.
The screenshot is 1100×640 with the current server's 262,144-pixel processing
cap; prompts span 388–532 tokens. Timing includes image preprocessing, vision,
language inference, probability readout and HTTP; model loading, browser
rendering and PNG encoding are excluded. It is not directly comparable to the
upstream README's 260 ms because its serving/image/concurrency conditions differ.
See [full raw latency evidence](https://github.com/jyohukuchan/rocm_exl3/tree/7675fa26cdf52fbe5a90060a69c60c752855fd8f/doc/jev_latency_20261006)
and [BF16 fidelity evidence](https://github.com/jyohukuchan/rocm_exl3/tree/7675fa26cdf52fbe5a90060a69c60c752855fd8f/doc/jev_validation).

These are bounded regression and synthetic-task checks. Broad quality,
recalibration, reasoning quality and maximum context are not established for
this quantization. The source's nominal 256K context is not a validated EXL3
limit. Tensor-parallel LoRA is unsupported; unquantized layer-split loading had
an asynchronous-loading issue. Single-GPU execution is the verified route.

## License and attribution

Model weights and upstream assets retain Apache-2.0; see `LICENSE`, `NOTICE`
and [AutoTrust's source model](https://huggingface.co/autotrust/JEV-27B-VL).
The underlying model is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Quantization/packaging and the ROCm fork are by jyohukuchan. The included launcher
uses the engine repository's MIT terms (`LICENSE-code`), separate from the
model-weight license. No affiliation or endorsement by Qwen or AutoTrust is
implied.
