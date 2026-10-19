---
license: other
license_name: flux-kommunity
license_link: LICENSE.md
base_model: @SOURCE_ID@
base_model_relation: quantized
pipeline_tag: text-to-image
language:
- en
tags:
- exl3
- quantized
- image-generation
- image-editing
- flux3
---

# EXL3 quantization of FLUX.3 Image Klein

Independent community **3bit / 4bit / 5bit EXL3** conversions of the distilled
5.6B [FLUX.3 Image Klein model](https://huggingface.co/@SOURCE_ID@/tree/@SOURCE_REV@).
Modified and published by jyohukuchan. This is not an official product of Black
Forest Labs Inc. and has not been endorsed, approved or validated by the company.

**License: [FLUX Kommunity License v1.0](LICENSE.md).** Non-commercial use and
redistribution are permitted subject to its terms. Under §2(a), Qualifying Users
(combined annualized gross revenue with affiliates below US$5 million) may use
Outputs commercially, and operate the model in production solely to generate
those Outputs, subject to the license including §2(e). This exception is not a
blanket permission for commercial model redistribution, training, or all production
uses. Other uses require an appropriate license from Black Forest Labs.
Read the complete license and [NOTICE.md](NOTICE.md), including use restrictions.
Rights to use the model are granted directly by Black Forest Labs Inc. to recipients
under that license. The engine's MIT license does not apply to these weights.

This FLUX Model is licensed by Black Forest Labs Inc. under the FLUX Kommunity License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

## Checkpoints

| Folder | Linear bits | Effective DiT bpw | Safetensors GB | BF16-relative NMSE ↓ |
|---|---:|---:|---:|---:|
@WEIGHT_ROWS@

Each folder includes 258 quantized block Linear layers, one retained high-precision
tensor file, transformer config, checksummed `quantization.json`, license and notice.
Effective bpw includes retained high-precision tensors and quantization scales.
Sizes are decimal GB including publication headers. Text encoder, VAE and prompt
upsampler (PUP) are fetched separately; they are not quantized in this release.
The DiT/text encoder compute in BF16; the VAE uses FP32. EXL3 weights are restored
in bounded workspaces for matrix multiplication, without expanding the whole DiT.

These are custom checkpoints for the [EXL3 engine](https://github.com/jyohukuchan/rocm_exl3_forme/tree/@ENGINE_REV@),
not standard Diffusers `from_pretrained()` folders. Hugging Face hosted inference
is not configured. The source is the distilled Klein model, not a Base model;
the official schedule is **16 flow-map steps, shift 6.93**, with `target_timestep`.
Output is RGB.

## Inference

Use Python 3.12, a compatible ROCm PyTorch installation/toolchain and a built
native EXL3 extension. Validated GPUs are a single Radeon AI PRO R9700 (gfx1201)
and a single Radeon PRO V620 (gfx1030). CUDA and other GPUs have not been validated
with this release. Measured environment: PyTorch 2.12.0+rocm7.2, HIP compiler
7.14.60850, Transformers 5.17.0, Accelerate 1.15.0, safetensors 0.8.0, and Diffusers
commit `b800f828f3951c4d206e154411bcbfa693cd322e`.

```bash
git clone https://github.com/jyohukuchan/rocm_exl3_forme.git
cd rocm_exl3_forme
git checkout @ENGINE_REV@

# Install compatible ROCm torch first and point ROCM_PATH at the SDK.
# Use gfx1030 for V620; gfx1201 for R9700.
EXL3_BACKEND=rocm PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=8 \
  python -m pip install --no-build-isolation '.[diffusion]'
python -m pip install 'huggingface-hub==1.31.0' 'safetensors==0.8.0'
python -c 'import exllamav3_ext'
```

See [build requirements](https://github.com/jyohukuchan/rocm_exl3_forme/blob/@ENGINE_REV@/README.md#requirements)
and the [tested environment](r9700-environment.json). Validation reused existing
compiled GPU extensions, with binary hashes recorded in the environment files;
their original build commits are unknown. A clean rebuild of those exact binaries
was not demonstrated. Do not rely on automatic CUDA-style JIT compilation on ROCm.

Accept the upstream model's access conditions on Hugging Face and authenticate
with `hf auth login` as needed. Download the pinned text encoder/VAE without
downloading the original dense DiT:

```bash
python rocm_tools/flux3.py download --components-only --output models/flux3-components
```

Download one EXL3 bit width (change `bits` to 3 or 5 as desired):

```python
from huggingface_hub import snapshot_download
bits = 4
snapshot_download(
    repo_id="@REPO_ID@", revision="exl3-v1",
    local_dir="models/@REPO_NAME@",
    allow_patterns=[f"{bits}bit/*", "README.md", "LICENSE.md", "NOTICE.md", "example-caption.txt"],
)
```

The CLI accepts a **full scene caption expanded by an LLM**, not a short request,
through `--caption` / `--caption-file`. The included caption reproduces the fixed
three-object composition test:

```bash
python rocm_tools/flux3.py generate \
  --model models/flux3-components --exl3 models/@REPO_NAME@/4bit \
  --caption-file models/@REPO_NAME@/example-caption.txt \
  --seed 1703 --width 1024 --height 1024 --output outputs/flux3-exl3
```

To use a short request, download the official, unquantized **18.8 GB PUP** and let
it expand the request. The CLI frees PUP before loading the DiT:

```bash
python rocm_tools/flux3.py download --component upsampler --output models/flux3-pup
python rocm_tools/flux3.py generate \
  --model models/flux3-components --upsampler models/flux3-pup \
  --exl3 models/@REPO_NAME@/4bit \
  --prompt 'A red sphere on the left and a blue cube on the right.' \
  --seed 1703 --output outputs/flux3-pup-exl3
```

Outputs include `run-0.png` and `report.json`. For editing, add `--images reference.png`
or multiple image paths, and describe the desired result. PUP receives the same
reference images as generation. The text encoder uses CPU module offload, so
sufficient host RAM is also needed. The V620 path automatically uses query-sliced
DiT attention and PyTorch reference functions for unsupported PUP kernels.
See the [full CLI guide](https://github.com/jyohukuchan/rocm_exl3_forme/blob/@ENGINE_REV@/doc/flux3_exl3.md).

## BF16-relative accuracy

```
NMSE = sum((candidate_prediction - BF16_prediction)^2) / sum(BF16_prediction^2)
```

Both models receive the same latent, timestep, target timestep and text/reference
conditioning on the BF16 trajectory. Candidate predictions never drive its scheduler.
Only target-image outputs are scored, at all 16 steps. Differences and squares are
FP32, reductions FP64. The headline score is the ratio of global sums across three
held-out 1024×1024 cases, one seed per case (1702 / 1703 / 1706), batch 1.
All per-step BF16 latent/prediction and conditioning hashes match across bit widths;
BF16 self-comparison is zero. Calibration uses three separate fixed cases with
captures at steps 0 / 8 / 15, including reference-image activations.

| Case | 3bit NMSE ↓ | 4bit NMSE ↓ | 5bit NMSE ↓ |
|---|---:|---:|---:|
@CASE_ROWS@

NMSE is numerical fidelity to BF16 DiT predictions, **not a perceptual quality score**.
All tested bit widths preserved the HELLO text, six stars, basic object colors,
boat and white sail in the fixed cases. Small differences in shapes, layout,
reflections and cloth folds remain. This small test set does not establish a broad
quality guarantee.

![BF16 / 3bit / 4bit / 5bit](comparison.png)

[Raw evaluation](evaluation.json) · [Cases and captions](evaluation_cases.json) ·
[Visual review](visual_review.json) ·
[Full archived reports](https://github.com/jyohukuchan/rocm_exl3_forme/tree/@ENGINE_REV@/benchmarks/2026-10-19/flux3)

R9700: BF16/4bit validated photo, text, composition, single-reference editing,
two references and 2048px photo; 3bit/5bit validated text, composition and two
references at 1024px. V620: all three bit widths validated text generation at
1024px; 4bit also validated single/two-reference editing. Official PUP was tested
on both GPUs. V620 2048px, V620 3bit/5bit image references and multi-GPU image
inference are outside this validation scope.

## Speed and memory

R9700, 1024×1024 composition, batch 1, 16 steps, one warmup plus three measured
runs (medians). DiT resident on GPU, text encoder CPU module offload, FP32 tiled
VAE, 256 MiB EXL3 reconstruction workspace. Includes text encoding, denoising
and VAE decoding; **excludes initial model loading, PUP and PNG saving**.

| Precision | Denoising s | Pipeline s | Peak allocated GiB |
|---|---:|---:|---:|
| BF16 | 10.028 | 11.622 | 13.744 |
| EXL3 3bit | 10.753 | 12.348 | 5.775 |
| EXL3 4bit | 10.751 | 12.345 | 6.444 |
| EXL3 5bit | 10.784 | 12.383 | 7.012 |

Allocated memory is PyTorch allocation, not total GPU usage. PUP can take much
longer than the downstream generation: a single R9700 reference-edit run spent
98.64 s in prompt expansion (excluding PUP loading), with 18.11 GiB peak allocated
for PUP. This is a separate cold diagnostic, not part of the speed table.

## Provenance

Source: **@SOURCE_ID@@@SOURCE_REV@**. Engine/evaluation: **@ENGINE_REV@**.
See [sources.json](sources.json) for pinned model/PUP/Diffusers revisions.
Quantization and per-layer repackaging are the modifications to the original DiT.
No additional license terms are imposed on recipients.

Every published tensor file includes source/license/modification metadata, and
each bit-width folder includes LICENSE.md and NOTICE.md. Publication only adds
headers relative to the evaluated checkpoints: every tensor payload and descriptor
is preserved byte-for-byte, original files remain unchanged, and runtime file
checksums are updated. [PUBLICATION.json](PUBLICATION.json) records evaluated and
published hashes and payload equivalence; [FILE_MANIFEST.json](FILE_MANIFEST.json)
lists published file sizes and hashes. Transformer configs are copied unchanged.
The [publication validation](VALIDATION.json) records a fresh 4bit generation
using the staged package and components without a dense DiT; its output pixel
hash matches the previously evaluated 4bit composition exactly.

日本語: FLUX Kommunity License v1.0に基づく第三者のEXL3量子化です。非商用利用・再配布と、
条件を満たす利用者の生成物の商用利用には、それぞれライセンスの条件が適用されます。
商用用途全般を無条件に許諾するものではありません。NMSEはBF16予測への数値的な近さです。
