# JEV native EXL3 support (validation in progress)

JEV-27B-VL combines an unchanged Qwen language/vision backbone for generation
(System 2) with a runtime decision LoRA (System 1). Merging the adapter into the
main checkpoint would change ordinary generation. Reading unadapted vocabulary
logits or applying a sampling filter would not implement its trained decisions.

This fork keeps both systems in one model, selects LoRA per forward, and starts
each request with fresh KV and recurrent state. The dedicated server serializes
GPU work, including client cancellation, so two systems cannot share mutable
state. Single-GPU loading is supported by this server; `--gpu-split 28,28`
selects single-process layer split. The early unquantized V620 pair probe needed
`AMD_SERIALIZE_KERNEL=3` during loading; asynchronous source loading still
needs investigation. Tensor-parallel LoRA is
explicitly rejected by the library.

## Conversion and precision

The current JEV-27B-VL conversion uses:

```sh
ulimit -n 65536
EXL3_ROCM_MLP_RANGE_BALANCE=0 python convert.py \
  -i /models/JEV-27B-VL -w /work/jev-convert \
  -o /models/JEV-27B-VL-exl3-4bpw \
  -b 4 -hb 6 -mb 6 -vb 16 -ss 1024
```

The language trunk uses 4bpw and the normal generation head and MTP component
use 6bit. The vision tower is unquantized because visual decisions are part of
JEV's intended use. The decision adapter remains unquantized, evaluated in
FP32. The converter detects `adapter_vllm/decision_head.json`, copies the
adapter/calibration, and saves 264 relevant source BF16 vocabulary rows in
`decision_rows.safetensors`. System 1 computes those exact source rows plus the
head LoRA in FP32. It bypasses the quantized generation head entirely.

This is a 4bpw **trunk** setting, not a claim that the whole pack is exactly
4bpw: embeddings, auxiliary weights, exact decision rows and LoRA are additional.
Default calibration is 250 rows × 2048 tokens, with FP32 Hessians.

## HTTP server

Install `fastapi`, `uvicorn` and `transformers` in the existing ROCm/PyTorch
EXL3 environment. Use the native extension compiled for the GPU being tested.

```sh
python -m rocm_tools.jev_server \
  -m /models/JEV-27B-VL-exl3-4bpw \
  --context 16384 --chunk-size 1024 --port 3960
```

The launcher raises its own open-file soft limit to 65536 when permitted; the
JEV adapter needs more descriptors than the common 1024 default. Library callers
should set `ulimit -n 65536` before starting their process. The default bind is loopback. `--api-key` enables Bearer authentication on POST
routes. This launcher disables the existing MLP metadata range-balance policy,
which rejects runtime LoRA. GPU validation of the original-scale JEV route is
still pending; no general R9700-support claim follows from these changes.

`POST /v1/decide` accepts the official bare-v1 fields:

```json
{"kind":"noul","state":"Tokyo is the capital of Japan.","question":"Is this correct?"}
```

Kinds are `noul` (false/true), `score` (0..5), and `choice` (2..256 string
options). It returns the complete normalized probability vector, chosen index,
usage and timing. No token is generated for System 1. Bias is applied before
its calibrated per-kind temperature. For choices beyond the 16 trained labels,
`single`, `permute` and `tournament` follow the source model's extension
strategies; the default is `single`.

`thinking: "auto"` invokes System 2 when the largest System 1 probability is
below `threshold` (default 0.8). `thinking: "on"` always invokes it; the default
is `off`. `think_budget` defaults to 1024 tokens. The base model reasons over
the state/question/options, then its contextual answer-letter probabilities
are read without generation and mixed 50:50 with System 1, as in the official
server. `return_reasoning`, `debug` and `reasoning_effort` (low/medium/xhigh) are
supported. Scores have no adaptive-thinking route. A bounded reasoning budget
can force readout before reasoning completes; `finished_within_budget` reports
that condition. Quantization and this greedy reasoning implementation still
need task-level validation; source-model calibration is not an EXL3 guarantee.

Images may appear in state as a list of strings and `{"image":"data:image/png;base64,..."}`
or standard `image_url` parts. Native image embeddings and MRoPE are used for
both systems. Up to 16 images are accepted, with a 262144-pixel processing cap;
URLs are currently limited to data URLs.

`POST /v1/systemone` follows llama.cpp's TypeSafe batch request/answer shape:

```json
{"state":"The user needs a billing refund.","questions":{
  "route":{"type":"choice","instructions":"Choose the appropriate team.",
           "criteria":{"billing":"Payments and refunds","technical":"Software problems"}},
  "refund":{"type":"noul","instructions":"Does the user request a refund?"}
}}
```

Answers map back to the original criteria keys. Choice confidence and score
confidence use the formulas in llama.cpp's `server-decision.cpp`. JEV score
criteria must contain its six trained levels; incompatible scales are rejected.
Questions are evaluated sequentially. Shared-prefix caching across questions
is not implemented yet.

`POST /v1/chat/completions` offers ordinary text/vision System 2 generation
with `max_tokens`, `temperature`, and `chat_template_kwargs` containing
`enable_thinking`/`reasoning_effort`. This small JEV server currently returns
non-streaming responses. Tools, response-format constraints and top-p/top-k
controls are explicitly rejected; use the existing general EXL3 server when
those features are required for ordinary generation.

## Validation tools

`rocm_tools/jev_quality.py` creates eight frozen text/vision cases, collects an
unquantized BF16-body/FP32-head HF+PEFT reference, collects an EXL3 candidate,
checks System 1 → System 2 isolation and image generation, and compares full
probability distributions. It reports KL, max probability difference and top-1
agreement per case. These are a small regression screen, not broad model
quality or long-context validation.

The BF16 reference disables Transformers' optional allocator warmup because
a single ~26GiB allocation failed on V620. The collector uses a complete per-module device map, and supports a CPU-staged
`--reference-placement hybrid` alternative. No parent/root placement overrides
are used, because Accelerate can move all child weights to that parent device
during hook setup. All source weights and forward math remain unchanged. The oracle explicitly uses the upstream Torch GDN/conv
functions and SDPA MATH because the installed FLA BF16 dot kernel cannot compile
on gfx1030. Original failed logs must be retained beside successful runs.

Sources: [model and reference server](https://huggingface.co/autotrust/JEV-27B-VL),
[llama.cpp decision API](https://github.com/ggml-org/llama.cpp/blob/7049ff0cbeb1f5ead231de4522af6b75d8d773c0/tools/server/server-decision.cpp).

## Interim hardware evidence (2026-10-06)

The BF16 oracle and native unquantized V620 pair each completed all eight
decisions. Native-vs-oracle mean KL is 1.8139607e-6, maximum probability
difference .00103208, with 8/8 identical top choices. Base generation before
and after a decision is identical (`東京`); image chat returns ` red square`.
The native run used source weights cast to FP16, FP32 LoRA/exact head, the live
V620 native binary and serialized GPU operations. These are **unquantized**
results; final 4bpw V620/R9700 validation is still pending conversion.

Real HTTP tests also exercised all decision/chat endpoints and an 8-token
adaptive-thinking budget. TypeSafe score descriptions are appended to the
question with their numeric mapping, preserving the trained 0..5 option lines.
A correct `2 + 2 = 4` answer then receives its highest probability at grade5.
