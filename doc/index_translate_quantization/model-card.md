---
license: apache-2.0
base_model:
- IndexTeam/Index-Translate-9B
base_model_relation: quantized
pipeline_tag: translation
tags:
- exl3
- translation
- rocm
---

# Index-Translate-9B EXL3 4.05 bpw

EXL3 quantization of [IndexTeam/Index-Translate-9B](https://huggingface.co/IndexTeam/Index-Translate-9B),
created from BF16 weights at revision `4e06279e62352dab96bbeb6fa0579bca5e822d75`
using the `rocm_exl3` project on one Radeon AI PRO R9700.

The decoder allocation is derived from
[Piter254's 6.05 bpw conversion](https://huggingface.co/Piter254/IndexTeam_Index-Translate-9B-exl3-6.05bpw/tree/793a1736f984724a9af7bc776c93628d52f948fc):
177 tensors change from 6 to 4 bits; 23 change from 7 to 5 bits. The weighted
decoder mean is **4.048543689320389 bpw**, rounded to **4.05 bpw**.
The output head remains **8 bit**, MTP uses **4 bit**, and additional vision
weights remain unquantized. Embeddings, small unquantized projections,
head, MTP, vision weights, and quantization overhead are outside the quoted
decoder bitrate budget.

Settings: `mul1` codebook, `out_scales=auto`, bundled mixed-corpus calibration
with 250 rows × 2048 tokens. `recipe-4.05bpw.json` records the complete decoder
allocation and provenance. See `validation.json` for shard hashes and two
English/Japanese translation smoke samples; these are not a translation-quality
benchmark.

Use the model's native chat template with `enable_thinking=False` and greedy
decoding. The upstream model card and translation instructions are preserved
in `README.source.md`.
