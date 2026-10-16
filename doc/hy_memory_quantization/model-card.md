---
license: apache-2.0
base_model: tencent/Hy-Memory-30B-A3B
base_model_relation: quantized
language:
  - en
  - zh
pipeline_tag: text-generation
tags:
  - exl3
  - quantized
  - moe
  - memory
---

# Hy-Memory-30B-A3B — EXL3 4.08 bpw

Mixed-precision EXL3 conversion of [tencent/Hy-Memory-30B-A3B](https://huggingface.co/tencent/Hy-Memory-30B-A3B)
at revision `64a460a94d5498b7316fb02395c72d01c47a7347`. The source is a memory-writer model for extracting,
updating and reconciling persistent information in external memory frameworks,
including Mem0, MemOS and Hy-Memory. Consult the source model card for its intended
use and original evaluation; those scores have not been reproduced for this quant.

## Precision allocation

| Component | Precision |
| --- | --- |
| Routed experts, layers 1–47, all gate/up/down projections | 4 bit |
| Attention, layers 0–47, all Q/K/V/O projections | 6 bit |
| Shared experts, layers 1–47, all gate/up/down projections | 6 bit |
| Dense FFN, layer 0, all gate/up/down projections | 5 bit |
| Output head | 8 bit |
| Input embeddings, routers and norms | Unquantized 16 bit |
| Expert routing correction bias | Original FP32 |

The decoder's parameter-weighted bitrate is **4.07774585 bpw**;
the name rounds this to 4.08. This excludes the head, embeddings, routers, norms,
and quantization auxiliary tensors. Total stored weight files occupy
**15.946 GB / 14.851 GiB**. KV cache and runtime buffers require
additional memory. `recipe-4bit-base.json` records all 18,384 decoder allocations.
This is a 4-bit-base allocation with selected precision increases, not a uniform
4-bit quant or a claim of a sensitivity-optimized allocation.

## Conversion and validation

Converted with [rocm_exl3_forme](https://github.com/jyohukuchan/rocm_exl3_forme),
engine revision `d51528ef795955525c794a1a6935f1599c258efe`, on a Radeon AI PRO R9700. Uses the `mul1`
codebook, output channel scales, FP32 Hessians and 250 × 2048 calibration tokens:
150 rows of the bundled general corpus mix and 100 rows of reproducible fictional
memory-writing examples. The latter cover extraction, fact correction, exact coding
details and Japanese text. They are synthetic examples, not private user data or
a self-sampled domain trace. See `calibration-metadata.json` for the token-file digest.

All stored decoder bitrates and the head were checked against the recipe.
Three held-out functional smoke checks passed: explicit fact extraction,
replacement of a superseded location while preserving a dietary preference,
and abstention when employer/address evidence is absent. Full prompts and responses,
GPU/runtime details and weight SHA-256 digests are in `validation.json`.
These checks do not measure quality loss relative to BF16 or establish benchmark
equivalence with the source model.

**Prompt specificity:** in an initial check with broader instructions, the model
kept a superseded location with an annotation and retained a temporary weather
observation. The final checks explicitly request only currently valid facts,
complete removal of superseded items and exclusion of transient weather.
`validation-initial.json` preserves those original prompts, responses and the
failed correction check. Use explicit memory-bank policies in deployment;
these observations have not been compared against BF16 and cannot be attributed
specifically to quantization.

## Download and inference

```bash
hf download jyohukuchan/Hy-Memory-30B-A3B-exl3-4.08bpw --local-dir ./Hy-Memory-EXL3
```

Use a HYV3-compatible ExLlamaV3 runtime. In the ROCm fork, revision `67cbdd1`
or later fixes the expert-width interpretation required by this checkpoint:
the actual width is `expert_hidden_dim=768`, although the original config also
contains the legacy `moe_intermediate_size=1536`. Keep the published config intact.

```python
from exllamav3 import Cache, Config, Generator, Model, Tokenizer, TopPSampler

config = Config.from_directory("./Hy-Memory-EXL3")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096, max_batch_size=1)
model.load(device="cuda:0")
tokenizer = Tokenizer.from_config(config)
generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
prompt = tokenizer.hf_render_chat_template([
    {"role": "system", "content": 'Extract explicit user facts. Do not infer missing information. Return only JSON with a "memories" list of strings.'},
    {"role": "user", "content": "My name is Nora. I live in Madrid and am vegetarian."},
], reasoning_effort="no_think")
print(generator.generate(
    prompt, max_new_tokens=256, completion_only=True,
    encode_special_tokens=True, stop_conditions=config.eos_token_id_list,
    sampler=TopPSampler(top_p=1.0, temperature=0.0),
))
```

The native `chat_template.jinja` and tokenizer files are included. This EXL3
weight format requires ExLlamaV3; ordinary Transformers weight loading does
not decode the EXL3 trellis tensors. The source model supports a 262,144-token
position limit; long-context behavior and that cache size were not validated here.

## License and attribution

Apache-2.0, matching the source weights with no additional restrictions.
The original Tencent copyright and full license are preserved in `LICENSE`.
`NOTICE` identifies this quantization as a modified distribution;
`SOURCE_MODEL_CARD.md` preserves the original model-card text. All quality
claims and provenance for the original model belong to its authors.
