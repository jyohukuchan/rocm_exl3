# Index-Translate-2B EXL3 4.05 bpw

Source: [`IndexTeam/Index-Translate-2B`](https://huggingface.co/IndexTeam/Index-Translate-2B/tree/9bc08445315723eb9b77d1b93505915eb03c3f01).
The pinned BF16 weight SHA-256 is
`2f499211fe352913464be69365e2e5724f913051efb3ae2b6fd0fa617a4c0b60`.

The recipe follows the previous 9B conversion's allocation policy, applied to
the 2B model's own 24-layer architecture. First derive a default 6.05 bpw
strategy, then subtract two bits from each budgeted decoder tensor. This is
identical to the default 4.05 bpw allocation on this model. The same allocator
was verified to reproduce all 200 decoder bitrates of the 9B reference used by
the previous conversion.

The resulting decoder contains **132 tensors at 4 bit and 18 at 5 bit**,
with a weighted mean of **4.048929663608563 bpw**, reported as **4.05 bpw**.
The output head uses 8 bits, MTP uses 4 bits, and additional vision weights
remain unquantized. Embeddings, small unquantized projections and auxiliary
components are outside the quoted decoder bitrate budget.

## Completed conversion (2026-10-15 JST)

- Installed model: `/home/homelab1/datapool/ai_models/safetensors/Index-Translate-2B-exl3-4.05bpw`.
- Weight file: **2,919,237,952 bytes** (2.92 GB / 2.72 GiB).
- SHA-256: `b27ec77c4681a424ad7063368ce0129cdc79f3a424f8e8d4fc55b2bc1a5bea85`.
- Conversion elapsed time: **983.8 seconds**, approximately 16 minutes 24 seconds.
  The 8-bit head took 435 seconds including its error metrics and advancement.
- Validation passed for all 150 decoder tensor allocations, the 8-bit head,
  and all eight 4-bit MTP tensors. English → Japanese and Japanese → English
  samples passed; see [validation.json](validation.json).
- Installed files are hard-linked to the converter output under
  `/home/homelab1/datapool/rocm-exl3-rdna2/models/Index-Translate-2B-exl3-4.05bpw`,
  allowing access through both `/models` and `/work` without duplicating data.

Public model:
[`jyohukuchan/Index-Translate-2B-exl3-4.05bpw`](https://huggingface.co/jyohukuchan/Index-Translate-2B-exl3-4.05bpw).
The [model card](model-card.md) covers capabilities, quantization, download,
inference, and actual translation samples. Its inference example was tested
against the installed model. The public package includes weights, model and
tokenizer configurations, a precision recipe, Apache-2.0 license, and model-only
validation results. See [hf-publication.json](hf-publication.json) for publication
and anonymous-access/hash verification details, and [run.json](run.json) for
the local conversion record.

## Reproduction

The `rocm-exl3-r9700-conv` container exposes only
`ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55` (Radeon AI PRO R9700).
`/src` is this repository, `/models` is the read-only
`/home/homelab1/datapool/ai_models` mount, and `/work` is
`/home/homelab1/datapool/rocm-exl3-rdna2`.

Download the pinned source into
`/home/homelab1/datapool/ai_models/safetensors/Index-Translate-2B-BF16`:

```bash
hf download IndexTeam/Index-Translate-2B \
  --revision 9bc08445315723eb9b77d1b93505915eb03c3f01 \
  --local-dir /home/homelab1/datapool/ai_models/safetensors/Index-Translate-2B-BF16
```

Convert:

```bash
docker exec -w /src \
  -e ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 \
  rocm-exl3-r9700-conv python convert.py \
  -i /models/safetensors/Index-Translate-2B-BF16 \
  -o /work/models/Index-Translate-2B-exl3-4.05bpw \
  -w /work/runs/index-translate-2b-4.05bpw/checkpoint \
  -rcp /src/doc/index_translate_2b_quantization/recipe-4.05bpw.json \
  -hb 8 -mb 4 -vb 16 --out_scales auto -cb mul1 \
  -cr 250 -cc 2048 -d 0
```

Resume with the same container/environment and
`python convert.py -w /work/runs/index-translate-2b-4.05bpw/checkpoint -r --out_scales auto`.

Validate allocation against actual stored tensor shapes, generate two native-
template English/Japanese translation samples, and hash the final weight file:

```bash
docker exec -w /src \
  -e ROCR_VISIBLE_DEVICES=GPU-a8e9ddefa2d60f55 \
  rocm-exl3-r9700-conv python doc/index_translate_quantization/validate.py \
  /work/models/Index-Translate-2B-exl3-4.05bpw \
  --recipe /src/doc/index_translate_2b_quantization/recipe-4.05bpw.json \
  --report /work/runs/index-translate-2b-4.05bpw/validation.json
```

The translation samples are functionality checks, not a quality benchmark
against BF16 or other quantizations.
