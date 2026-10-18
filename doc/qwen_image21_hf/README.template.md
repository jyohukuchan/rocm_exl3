---
license: other
license_name: qwen-research
license_link: LICENSE
base_model: @SOURCE_ID@
base_model_relation: quantized
pipeline_tag: text-to-image
language:
- en
- zh
- ja
tags:
- exl3
- quantized
- image-generation
- image-editing
- qwen-image-2.1
---

# @TITLE@

**Built with Qwen.** Community EXL3 DiT quantizations derived from
[@SOURCE_ID@](https://huggingface.co/@SOURCE_ID@/tree/@SOURCE_REV@),
published by [@PUBLISHER@](https://huggingface.co/@PUBLISHER@).
This is a third-party conversion, not an official Qwen release.

**License: Qwen Research License Agreement (2026-09-20). Research/evaluation,
non-commercial use only; commercial use requires a separate upstream license.**
Read [LICENSE](LICENSE) and [Notice](Notice). The inference engine's separate MIT
license does not replace the model license.

## Files and supported runtime

| Folder | Quantized Linear bits | Effective DiT bpw | Safetensors size (GB) | BF16-relative NMSE ↓ |
|---|---:|---:|---:|---:|
@WEIGHT_ROWS@

Each folder contains 224 EXL3 block Linear layers, retained high-precision tensors,
a transformer configuration, and `quantization.json` with checksums. Effective bpw
includes scales and retained high-precision tensors. Sizes are decimal GB and
include the publication metadata. The text encoder and VAE are **not included**;
they remain BF16 and are fetched from the pinned upstream model.

Use [rocm_exl3_forme](https://github.com/jyohukuchan/rocm_exl3_forme/tree/@ENGINE_REV@)
with its Qwen Image CLI. These are **custom EXL3 diffusion checkpoints**, not a
standard Diffusers `from_pretrained()` folder, GGUF, NF4, or Nunchaku checkpoint.
The repository is not configured for Hugging Face hosted inference.

Validated hardware: **one Radeon AI PRO R9700 32GB (gfx1201)**. CUDA, V620,
other GPUs, and multi-GPU image inference were not validated for this release.

## Inference

### 1. Prepare the engine

Use Python 3.12 and an installed ROCm PyTorch/toolchain compatible with your GPU.
The measured stack was PyTorch `2.12.0+rocm7.2`, a custom HIP compiler
`7.14.60850`, Transformers `5.17.0`, Accelerate `1.15.0`, safetensors `0.8.0`,
and Diffusers commit `b800f828f3951c4d206e154411bcbfa693cd322e`.
The engine requires ROCm >= 7.2.4 for its ROCm build.

```bash
git clone https://github.com/jyohukuchan/rocm_exl3_forme.git
cd rocm_exl3_forme
git checkout @ENGINE_REV@

# Install a compatible ROCm torch first; point ROCM_PATH at your installed SDK.
EXL3_BACKEND=rocm PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=8 \
  python -m pip install --no-build-isolation '.[diffusion]'
python -m pip install 'huggingface-hub==1.31.0' 'safetensors==0.8.0'
python -c 'import exllamav3_ext'
```

See the engine [build requirements](https://github.com/jyohukuchan/rocm_exl3_forme/blob/@ENGINE_REV@/README.md#requirements)
and [tested environment](https://github.com/jyohukuchan/rocm_exl3_forme/blob/@ENGINE_REV@/benchmarks/2026-10-19/qwen-image21-environment.json).
The evaluation reused an existing compiled R9700 extension (SHA recorded there);
its original build commit is unknown. A clean rebuild of that exact binary was
not demonstrated. Do not rely on automatic CUDA-style JIT compilation on ROCm.

### 2. Download one bit width and the upstream components

Run from the engine checkout. This example selects **5bit**, which had the lowest
NMSE in the fixed test set. Replace `bits = 5` with `3` or `4` as needed.

```python
from pathlib import Path
import json
from huggingface_hub import snapshot_download

bits = 5
source_id = "@SOURCE_ID@"
source_revision = "@SOURCE_REV@"
components = Path("models/@COMPONENT_DIR@")
quantized = Path("models/@REPO_NAME@")

# No upstream dense transformer weights are needed for EXL3 inference.
snapshot_download(
    repo_id=source_id, revision=source_revision, local_dir=components,
    allow_patterns=["model_index.json", "processor/*", "scheduler/*",
                    "text_encoder/*", "vae/*", "LICENSE", "README.md"],
)
(components / "exl3_source.json").write_text(
    json.dumps({"repo_id": source_id, "revision": source_revision}) + "\n"
)
snapshot_download(
    repo_id="@REPO_ID@", revision="exl3-v1", local_dir=quantized,
    allow_patterns=[f"{bits}bit/*", "README.md", "LICENSE", "Notice"],
)
```

The `exl3-v1` release tag pins the published checkpoint package. Its commit hash
can also be used as `revision=`.

### 3. Generate

```bash
python rocm_tools/qwen_image21.py generate \
  --variant @VARIANT@ --model models/@COMPONENT_DIR@ \
  --exl3 models/@REPO_NAME@/5bit \
  --prompt 'A red sphere on the left and a blue cube on the right, sitting on a pale gray surface, a small yellow pyramid between them, clean studio rendering.' \
  --width 1024 --height 1024 --seed 1703 \
  --output outputs/@VARIANT@-5bit
```

This writes `run-0.png` and `report.json`. The DiT stays on the GPU; the BF16 text
encoder is streamed from CPU RAM. VAE tiling uses 256px tiles / 192px stride.
@SCHEDULE_NOTE@
For editing, append `--images reference.png` (or multiple image paths) and an
editing prompt. The [engine guide](https://github.com/jyohukuchan/rocm_exl3_forme/blob/@ENGINE_REV@/doc/qwen_image_21_exl3.md)
also documents cached prompt embeddings, RGBA, conversion, and NMSE measurement.

## Accuracy evaluation

**NMSE measures fidelity of DiT predictions to this model's BF16 reference, not
perceptual quality or a percentage of preserved quality.**

```
NMSE = sum((candidate_prediction - BF16_prediction)^2)
       / sum(BF16_prediction^2)
```

Both models receive the same BF16-trajectory latent, time, and conditioning.
Each builds its own prefix KV cache. Candidate predictions do not drive the
reference scheduler. Only target-image tokens are scored, at every denoising
step; text/reference prefix outputs are excluded. Differences/squares are FP32,
reductions are FP64. The headline score is the ratio of global sums over the
three cases, not an unweighted mean of case scores.

- Three held-out cases: text, composition, and two-reference composition.
- One fixed seed per case: 1702 / 1703 / 1706, respectively.
- 1024×1024, batch 1, CFG 1, prefix KV cache enabled; @STEPS@ official steps.
- Same three source-specific BF16 calibration captures for 3/4/5bit,
  sampled at steps @CAL_STEPS@. Calibration and evaluation inputs are separate.
- Retained high-precision weights, reference output energy, and BF16 image hashes
  match across the compared bit widths.

| Case | 3bit NMSE ↓ | 4bit NMSE ↓ | 5bit NMSE ↓ |
|---|---:|---:|---:|
@CASE_ROWS@

[Evaluation data](evaluation.json) · [Fixed prompts and seeds](evaluation_cases.json)
· [Image observations](visual_review.json)
· [Full archived evidence](https://github.com/jyohukuchan/rocm_exl3_forme/tree/@ENGINE_REV@/benchmarks/2026-10-19/qwen-image21-multibit-reports)

![BF16 / 3bit / 4bit / 5bit comparison, white alpha composite](comparison.png)

@VISUAL_NOTE@

All 12 newly generated 3bit/5bit images across the two source models had finite
final latents and VAE outputs, with every EXL3 layer used on every step.
This is a small fixed-case screen, not a broad quality guarantee. For 3bit/5bit,
this release's evaluation covers the three 1024px cases above. The earlier 4bit
work additionally screened 2048px, single-image editing and RGBA; those extra
cases are not part of this NMSE table. No new 3bit/5bit speed benchmark is claimed.
Faint VAE tiling seams can also be present in BF16 reference images.

To measure a candidate with the engine, first cache a prompt with `encode`, then
use `generate --encoding ENCODING --nmse-exl3 CHECKPOINT --variant @VARIANT@ --model ORIGINAL_MODEL --output RESULT`.
NMSE evaluation needs the **original BF16 DiT** in `ORIGINAL_MODEL/transformer`,
unlike ordinary EXL3 inference. Use the engine's `download --variant @VARIANT@`
command to fetch the complete pinned upstream model for that purpose.

## Provenance and redistribution

Source: **@SOURCE_ID@@@SOURCE_REV@**.
Engine/evaluation evidence revision: **@ENGINE_REV@**.
This package preserves the source model identity and official schedule; it changes
eligible block Linear weights to EXL3 and splits the DiT into per-layer files.

For redistribution, modification/source/license notices are embedded in every
published safetensors header. This publication changes **metadata only** relative
to the evaluated EXL3 tensor payloads; payload hashes are verified and the runtime
manifest checksums are updated. [FILE_MANIFEST.json](FILE_MANIFEST.json) records
published file hashes, and [PUBLICATION.json](PUBLICATION.json) records original
checkpoint hashes and tensor-payload equivalence. Unmodified transformer configs
are copied from the evaluated checkpoints. Each bit-width folder includes the
license and required `Notice` so it can be downloaded independently.

日本語: 元モデルのQwen Research Licenseを継承した非商用の研究・評価用モデルです。
通常版とTurboは別リポジトリです。NMSEは各モデルのBF16への数値的な近さであり、
画質の絶対評価ではありません。推論には上記の専用EXL3エンジンを使用してください。
