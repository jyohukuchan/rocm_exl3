# Index-Translate-9B EXL3 4.05 bpw on R9700

The per-tensor recipe reuses the allocation in
[`Piter254/IndexTeam_Index-Translate-9B-exl3-6.05bpw`](https://huggingface.co/Piter254/IndexTeam_Index-Translate-9B-exl3-6.05bpw/tree/793a1736f984724a9af7bc776c93628d52f948fc).
It subtracts two bits from every quantized decoder tensor: 177 tensors change
from 6 to 4 bits and 23 from 7 to 5 bits. Their weighted mean is
**4.048543689320389 bpw**, reported as **4.05 bpw** by the converter.
The 8-bit output head, unquantized small decoder projections, `mul1` codebook,
automatic output scales, and 250 × 2048 calibration dimensions are retained.
MTP uses the reference configuration's 4-bit setting; additional vision weights
are copied without quantization. Embeddings and auxiliary weights are outside
the decoder bitrate budget.

## Completed conversion (2026-10-14 JST)

- Model: `/home/homelab1/datapool/ai_models/safetensors/Index-Translate-9B-exl3-4.05bpw`
- Weight file: **7,603,684,487 bytes** (7.60 GB / 7.08 GiB).
- SHA-256: `820527195aed7b67a37d288677b12a9c4b22070a93a3067d41da7b4d8296d9c2`.
- Conversion: approximately **70 minutes**; the 8-bit head alone took 805 seconds,
  including its error metrics. The converter's remaining-time estimate based on
  decoder layers substantially underestimated this final head step.
- Validation: all 200 decoder allocations, the 8-bit head, and all eight 4-bit
  MTP tensors passed; English → Japanese and Japanese → English samples passed.
- The model contains its recipe, validation report, run metadata, and an EXL3
  model card; the upstream card is preserved as `README.source.md`.

See [validation.json](validation.json) for translations and hashes, and
[run.json](run.json) for the pinned code/source and conversion command.
The installed model uses regular files hard-linked to the converter output in
`/home/homelab1/datapool/rocm-exl3-rdna2/models/Index-Translate-9B-exl3-4.05bpw`,
so it is also accessible through the container's read-only `/models` mount
without duplicating the weight data.

The model is published at
[`jyohukuchan/Index-Translate-9B-exl3-4.05bpw`](https://huggingface.co/jyohukuchan/Index-Translate-9B-exl3-4.05bpw).
The public [model card](model-card.md) describes capabilities, quantization,
usage, and translation samples. Its code example was tested against the final
weights. The public package contains model/tokenizer files, the allocation
recipe, an Apache-2.0 license, and a validation report containing model checks
and translation samples. Publication details and anonymous-access/hash
verification are recorded in [hf-publication.json](hf-publication.json).

`recipe-4.05bpw.json` is also valid YAML for `convert.py --recipe`. It records
the source/reference revisions and the reference configuration's SHA-256.
The original BF16 weights, rather than the 6.05 bpw weights, are quantized.

## Reproduce

The existing `rocm-exl3-r9700-conv` container exposes only
`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55` (Radeon AI PRO R9700, gfx1201).
Its `/src` bind mount is this repository, and `/work` is
`/home/homelab1/datapool/rocm-exl3-rdna2` on the host. The prebuilt extension
is `/work/lib-r9700-opt`; the two V620 cards are not exposed to the process.

Download the pinned source:

```bash
docker exec -w /src rocm-exl3-r9700-conv python -c '
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="IndexTeam/Index-Translate-9B",
    revision="4e06279e62352dab96bbeb6fa0579bca5e822d75",
    local_dir="/work/models/Index-Translate-9B-BF16",
    allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md"],
    max_workers=4,
)'
```

Convert with explicit single-device selection:

```bash
docker exec -w /src \
  -e ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 \
  rocm-exl3-r9700-conv python convert.py \
  -i /work/models/Index-Translate-9B-BF16 \
  -o /work/models/Index-Translate-9B-exl3-4.05bpw \
  -w /work/runs/index-translate-9b-4.05bpw/checkpoint \
  -rcp /src/doc/index_translate_quantization/recipe-4.05bpw.json \
  -hb 8 -mb 4 -vb 16 --out_scales auto -cb mul1 \
  -cr 250 -cc 2048 -d 0
```

For interruption recovery, use the same container/environment with
`python convert.py -w /work/runs/index-translate-9b-4.05bpw/checkpoint -r
--out_scales auto`. Retain the checkpoint until validation is complete.

Validate actual trellis bitrates, generate English/Japanese translation samples
using the native template with thinking disabled, and hash the final shards:

```bash
docker exec -w /src \
  -e ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 \
  rocm-exl3-r9700-conv python doc/index_translate_quantization/validate.py \
  /work/models/Index-Translate-9B-exl3-4.05bpw \
  --report /work/runs/index-translate-9b-4.05bpw/validation.json
```

The validation is a short functionality check, not a translation-quality
benchmark against BF16 or the reference 6.05 bpw quantization.
