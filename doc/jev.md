# JEV native EXL3 support

The verified mixed-precision checkpoint is publicly available at
[jyohukuchan/JEV-27B-VL-exl3-4bpw](https://huggingface.co/jyohukuchan/JEV-27B-VL-exl3-4bpw).
All 44 release files were checked against the upload manifest; see
[publication verification](jev_hf_release/PUBLICATION.md).

JEV-27B-VL combines an unchanged Qwen language/vision backbone for generation
(System 2) with a runtime decision LoRA (System 1). Merging the adapter into the
main checkpoint would change ordinary generation. Reading unadapted vocabulary
logits or applying a sampling filter would not implement its trained decisions.

This fork keeps both systems in one model, selects LoRA per forward, and starts
each request with fresh KV and recurrent state. The dedicated server serializes
GPU work, including client cancellation, so two systems cannot share mutable
state. JEV-27B-VL EXL3 inference is verified on a single V620 and a single
R9700. `--gpu-split 28,28` selects single-process layer split, but the early
unquantized V620 pair probe needed `AMD_SERIALIZE_KERNEL=3` during loading;
asynchronous source layer-split loading still needs investigation. The verified
quantized single-GPU runs need no serialized-kernel setting. Tensor-parallel
LoRA is explicitly rejected by the library.

## Conversion and precision

The completed JEV-27B-VL conversion uses:

```sh
ulimit -n 65536
EXL3_ROCM_MLP_RANGE_BALANCE=0 python convert.py \
  -i /models/safetensors/JEV-27B-VL -w /work/build-jev27-vl-4bpw \
  -o /work/models/JEV-27B-VL-exl3-4bpw \
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
Calibration is 250 rows × 2048 tokens, with FP32 Hessians. The resulting pack
has 17 weight shards, 2,426 indexed tensors, and totals 17,445,036,040 bytes
(16.247 GiB), including the unmerged adapter. Its weight tensors occupy
16,966,676,548 bytes: 4.886 effective bits per original parameter; the entire
pack including metadata/adapter is 5.024. The 400 quantized language projections
are 4bit, the generation head and eight MTP projections are 6bit, and the 333
vision tensors remain BF16. Embeddings, norms and recurrent controls keep their
original storage precision. MTP is packaged but is not used by the JEV server.

The downloaded source is `autotrust/JEV-27B-VL` revision
`f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc`. All 18 source shards and the adapter
match their Hugging Face LFS SHA256 values. The output's 39 files have a saved
SHA256 manifest; the adapter/calibration match the source byte-for-byte and the
264 decision rows match the original BF16 head exactly.

On the validation host, the canonical model directory is
`/home/homelab1/datapool/ai_models/safetensors/JEV-27B-VL-exl3-4bpw`.
It shares weight inodes with the validated `/work/models/...` pack, avoiding a
second 17GB copy. Do not modify shared weight files in place.

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
which rejects runtime LoRA. The original-scale JEV route is verified on both
GPUs below; this does not establish general R9700 coverage for other models.

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
that condition. Adaptive HTTP execution and probability mixing are verified;
reasoning quality has only been screened on small examples. Source-model
calibration is not an EXL3 calibration guarantee.

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

Sources: [model and reference server](https://huggingface.co/autotrust/JEV-27B-VL/tree/f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc),
[llama.cpp decision API](https://github.com/ggml-org/llama.cpp/blob/7049ff0cbeb1f5ead231de4522af6b75d8d773c0/tools/server/server-decision.cpp).

## Hardware evidence (2026-10-06)

Warmed System 1 latency on these GPUs is measured separately with serial HTTP
requests and the official browser demo; see [decision latency report](jev_latency_20261006/README.md).

Frozen regression cases cover Japanese/English binary decisions, six-level
scores, 4/16/256-way choices, two visual decisions, and four less decisive
examples. Both quantized single-GPU candidates match the BF16 oracle's top
choice on all 12 cases.

| Candidate | Basic top-1 / mean KL | Sensitive top-1 / mean KL | Maximum probability difference | Peak allocated VRAM |
|---|---|---|---|---|
| R9700, gfx1201 | 8/8 / 0.00085924 | 4/4 / 0.00039115 | 0.039731 | 15.60 GiB |
| V620, gfx1030 | 8/8 / 0.00083092 | 4/4 / 0.00038679 | 0.038927 | 15.52 GiB |

KL is `D_KL(reference || candidate)`. The largest probability shift is the
basic score example. This bounded screen supports retaining the 4bit trunk
with the higher precision components above; it does not establish broad
accuracy, probability calibration or long-context quality. Both candidates
produce `東京` before and after System 1 with identical token output and
` Red square.` for the image-generation probe.

Real HTTP checks on **both quantized GPUs** returned 200 for health/info,
bare-v1 decisions, TypeSafe noul/score/choice batch evaluation, image chat and
8-token adaptive thinking. They assert normalized probabilities, correct
criteria-key mapping, highest arithmetic grade at 5, and the 50:50 S1/S2 mix.
The 13 targeted CPU regressions cover sliced/strict LoRA, exact head selection,
calibration, TypeSafe mapping, inference mode and adaptive mixing.

Additional R9700 runtime checks select identifier 173 correctly among 256
options using both `permute` (four model requests) and `tournament` (17
requests), with normalized probabilities. High-confidence `thinking: "auto"`
also skips System 2 as intended. These extension checks are saved separately
from the 12-case BF16 comparison and do not expand that comparison's scope.

The tested environments use PyTorch `2.12.0+rocm7.2`, context 16,384,
chunk size 1,024, FP16 KV cache, and one GPU per model. R9700 uses
`PYTHONPATH=/work/lib-r9700-opt:/src`; V620 uses
`PYTHONPATH=/work/lib-expert-placement:/src` and `HSA_ENABLE_SDMA=0`.
`EXL3_ROCM_MLP_RANGE_BALANCE=0` applies to both. Neither quantized run sets
`AMD_SERIALIZE_KERNEL`. Native binary and source hashes are in the candidate
JSON metadata. The container's older default `/work/lib` was not used.

For the existing host containers, reproduce the collector with:

```sh
docker exec -e PYTHONPATH=/work/lib-r9700-opt:/src \
  -e EXL3_ROCM_MLP_RANGE_BALANCE=0 rocm-exl3-r9700-conv \
  python -m rocm_tools.jev_quality --backend exl3 \
  --model /work/models/JEV-27B-VL-exl3-4bpw \
  --cases /work/runs/jev-20261006/cases.json \
  --output /work/runs/jev-20261006/quantized-r9700.json
```

For V620 replace the container with `rocm-exl3-v620-pair` and native library
with `/work/lib-expert-placement:/src`. Use `cases-sensitive.json` for the
independent four-case screen. The server uses the same environment and model
path; HTTP checks ran on loopback inside each bridge container.

Evidence JSON, cases, integrity manifests and the HTTP verification script
are checked in under [jev_validation/](jev_validation/README.md). Large
conversion logs and source diagnostics remain at
`/home/homelab1/datapool/rocm-exl3-rdna2/runs/jev-20261006`.

The converter completed every decoder layer and printed `All done`, with no
excluded/nonfinite calibration rows. Its shell wrapper recorded exit 127
because editing that running wrapper shifted Bash's post-child read offset.
The original error/status is retained, alongside a separate artifact audit;
the conversion was not rerun or relabeled successful based on its exit code.
All shard headers/index entries, integrity checks and GPU runs passed.

### Unquantized baseline

The BF16 oracle and native unquantized V620 pair each completed all eight
decisions. Native-vs-oracle mean KL is 1.8139607e-6, maximum probability
difference .00103208, with 8/8 identical top choices. Base generation before
and after a decision is identical (`東京`); image chat returns ` red square`.
The native run used source weights cast to FP16, FP32 LoRA/exact head, the live
V620 native binary and serialized GPU operations. These are unquantized
baseline results, separate from the quantized single-GPU results above.

Real unquantized HTTP tests also exercised all decision/chat endpoints and an 8-token
adaptive-thinking budget. TypeSafe score descriptions are appended to the
question with their numeric mapping, preserving the trained 0..5 option lines.
A correct `2 + 2 = 4` answer then receives its highest probability at grade5.
