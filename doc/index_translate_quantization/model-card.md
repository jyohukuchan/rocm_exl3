---
license: apache-2.0
base_model:
- IndexTeam/Index-Translate-9B
base_model_relation: quantized
pipeline_tag: translation
library_name: exllamav3
language:
- multilingual
tags:
- exl3
- translation
- qwen3.5
---

# Index-Translate-9B EXL3 4.05 bpw

**4.05 bpw EXL3 quantization** of
[IndexTeam/Index-Translate-9B](https://huggingface.co/IndexTeam/Index-Translate-9B),
a Qwen3.5-based multilingual translation model. The base model supports text
translation across **150 languages**, including instructions for terminology,
formatting, style, and context. See the [original model card](https://huggingface.co/IndexTeam/Index-Translate-9B)
for the language inventory and base-model evaluations.

This repository contains EXL3 weights for use with **ExLlamaV3 or a compatible
EXL3 runtime**. The weight file is **7.60 GB** (7.08 GiB).

## Quantization

| Component / setting | Value |
| --- | --- |
| Decoder | 4.04854 bpw, reported as 4.05 bpw |
| Decoder tensor allocation | 177 tensors at 4 bit; 23 tensors at 5 bit |
| Output head | 8 bit |
| MTP layers | 4 bit |
| Embeddings and additional vision weights | Unquantized |
| Codebook | `mul1` |
| Output-channel scales | `auto` |
| Calibration | Mixed corpus, 250 rows × 2048 tokens |

The per-tensor allocation is derived from
[Piter254's 6.05 bpw conversion](https://huggingface.co/Piter254/IndexTeam_Index-Translate-9B-exl3-6.05bpw/tree/793a1736f984724a9af7bc776c93628d52f948fc):
each 6-bit decoder tensor becomes 4 bit, and each 7-bit decoder tensor becomes
5 bit, retaining the higher-precision tensor locations. Quantization starts from
the original BF16 checkpoint, not the already quantized reference weights.
The source revision is `4e06279e62352dab96bbeb6fa0579bca5e822d75`;
[recipe-4.05bpw.json](recipe-4.05bpw.json) records the complete allocation.

The quoted bitrate covers the budgeted decoder tensors. Embeddings, small
unquantized projections, the output head, MTP, vision weights, and storage
overhead are outside that budget, so total file size exceeds 9B × 4.05 bits.

## Usage

Download the complete repository:

```bash
hf download jyohukuchan/Index-Translate-9B-exl3-4.05bpw \
  --local-dir ./Index-Translate-9B-exl3-4.05bpw
```

Use the model's native chat template with **`enable_thinking=False`** and
**greedy decoding**. A standard translation request uses one user message and
no system message:

```text
请将以下英语文本翻译为日语，直接输出翻译结果，不要进行任何解释。

The meeting will start at ten tomorrow morning. Please bring the latest report.
```

Minimal ExLlamaV3 example:

```python
from exllamav3 import Cache, Config, Generator, Model, Tokenizer, TopPSampler

config = Config.from_directory("./Index-Translate-9B-exl3-4.05bpw")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=2048, max_batch_size=1)
model.load(progressbar=True)
tokenizer = Tokenizer.from_config(config)
generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

instruction = (
    "请将以下英语文本翻译为日语，直接输出翻译结果，不要进行任何解释。\n\n"
    "The meeting will start at ten tomorrow morning. Please bring the latest report."
)
prompt = tokenizer.hf_render_chat_template(
    [{"role": "user", "content": instruction}], enable_thinking=False,
)
print(generator.generate(
    prompt, max_new_tokens=128, completion_only=True,
    encode_special_tokens=True, stop_conditions=config.eos_token_id_list,
    sampler=TopPSampler(top_p=1.0, temperature=0.0),
))
```

The EXL3 files require an EXL3 runtime; `transformers` alone does not load these
quantized weights. Vision weights are retained, but this conversion was checked
with text translation only.

## Validation

All 200 decoder tensor bitrates, the 8-bit head, and all eight 4-bit MTP tensors
were checked against the stored weights. Two short translation samples produced:

| Direction | Input | Output |
| --- | --- | --- |
| English → Japanese | The meeting will start at ten tomorrow morning. Please bring the latest report. | 会議は明日の朝10時に始まります。最新の報告書を持参してください。 |
| Japanese → English | このモデルは日本語と英語の間で文章を翻訳できます。 | This model can translate text between Japanese and English. |

These are functionality checks, not a comprehensive translation-quality
benchmark. Original model scores have not been re-measured for this quantization.
See [validation.json](validation.json) for the allocation checks, samples, and
weight-file SHA-256.

## License and credits

Apache-2.0, following the base model. Credit to **IndexTeam** for the original
model and **Piter254** for the reference quantization allocation.
