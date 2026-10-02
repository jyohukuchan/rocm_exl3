# exl3_server

`exl3_server` is a local HTTP server for EXL3 models. It exposes OpenAI Chat
Completions and Text Completions, plus the llama.cpp-style completion endpoint.
The default bind address is `127.0.0.1:3953`.

```bash
python rocm_tools/exl3_server/server.py \
  -m /path/to/model \
  -cs 32768 \
  -host 127.0.0.1 \
  -port 3953 \
  -key replace-with-a-local-key
```

Install the normal ROCm requirements first:

```bash
python -m pip install -r requirements_rocm.txt
```

The server uses the model's own Hugging Face chat template for chat requests.
The model directory must contain a compatible tokenizer and chat template for
`/v1/chat/completions`; `/v1/completions` can accept a caller-rendered prompt.

## Connecting OpenCode

OpenCode v2.0.12 uses an OpenAI-compatible provider definition. Copy
[`examples/opencode.jsonc`](../../examples/opencode.jsonc) to a project-level
`opencode.jsonc`, then change only the local key if the server was started with
one. The example points to `http://127.0.0.1:3953/v1`, advertises a
786,432-token context, reserves 753,664 input tokens and 32,768 output tokens, and
uses `enable_thinking: true` with `reasoning_effort: xhigh`. The original
OpenCode coding sample was validated with `low`; future coding tasks use `xhigh`.
Set `enable_thinking: false` for text-only smoke checks.

Match the server allocation to this client configuration with `-gs 28.5,27.5 -cs 786432
--context-limit 786432 --max-output-tokens 32768`, and use
`-ctk '{"enable_thinking":true,"reasoning_effort":"xhigh"}'` for the same server
defaults. The [2026-10-02 full-cache memory probe](../../doc/full_cache_768ki_memory.md)
confirmed 768Ki allocation, tail prefill and MTP at the end of a synthetically
filled cache. It did not evaluate full-context language/retrieval quality.
The earlier dated OpenCode sample in the validation report used a 32Ki cache.

The [QSA request transition fix](../../doc/qsa_request_transition_fix.md) prevents
GPU faults when a shorter request follows a longer conversation. Rebuild the
native extension and restart the model workers when applying that update; an
older separately loaded extension still contains the fault even with new Python
files.

Start OpenCode from the project containing that configuration:

```bash
opencode /path/to/project
```

For native prefill/generation speed in the terminal footer and persistent
`/goal` execution, see [the display and Goal plugin setup](../opencode/README.md).
Chat responses include engine timing averages in `exl3_metrics`; streamed
responses carry this extension in their final choices chunk, independently of
whether usage chunks were requested.

The project configuration does not modify the existing global MCP setup. It
selects the local `rocm-exl3` provider for the primary model and its worker,
title, and compaction agents. Keep the server bound to loopback unless you add
authentication and a deliberate network boundary.

For initial validation, leave compaction disabled. This makes request bodies and
tool history easier to inspect. Once the provider works, set
`compaction.auto` to `true` and size `compaction.buffer` for the amount of
history you want to retain. The model limit in the example is an API budget;
the actual cache still comes from `-cs`.

### Local parallel workers (2026-10-02)

The local OpenCode setup now uses Qwen/xhigh for both the primary agent and the
`worker` subagent. The portable configuration explicitly declares
`agents.worker.mode: "subagent"`; assigning a model alone does not declare that mode.
Use OpenCode V2's native `subagent` tool for delegation, with three independent
calls in one turn. It is outside the Code Mode `execute` catalog.
The optional [worker instructions](../opencode/local-worker-instructions.md) explain
this to the model; add their path to `instructions` when using this repository.

For the validated TP/MTP server, keep the 786,432-token cache and use
`-ambs 3 -ndt 1` without `-dds`. This selects fixed one-token MTP, the faster
batch-three setting in the [earlier screening](../../doc/qwen38_v620_context_batch.md).
The verified runtime now reserves recurrent history for the resolved draft depth,
so `-ndt 1` uses one history state per slot instead of retaining the model's
four-token default. Omitted `-ndt` still uses the model default.

The 768Ki cache is a **shared total pool**. Up to three jobs run concurrently;
additional requests wait for slots and pages. A primary request also uses a slot.
This does not provide three separate 768Ki contexts. Configured batch>1 retains
`profile_peak` for the running service, as previously selected.

HTTP verification completed three concurrent generations and three independent
tool constraints. OpenCode then completed three actual worker sessions, all with
`rocm-exl3/qwen38#xhigh`, while the engine reported three active jobs.
A synthetic near-full 768Ki test with all three recurrent slots allocated also
completed 32 MTP1 tokens; board peaks were 31.549/31.197 GiB. This memory
check used one active full-cache job; three 9K-context workers were exercised
separately. [Dated verification](../../benchmarks/2026-10-02/opencode-batch3.json).

If loading a large MoE block exceeds the startup worker wait, set
`EXL3_TP_LOAD_TIMEOUT=600`. The local launcher uses this load-only allowance;
inference dispatch waits and the RCCL timeout remain unchanged. A startup retry
was required during this setup, followed by successful batch-three verification.

## API surface

The main endpoints are:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat format; server-side chat template, streaming or non-streaming, tools and structured output. |
| `POST /v1/completions` | OpenAI text completion format; prompt is supplied by the client. |
| `POST /completion` and `/completions` | llama.cpp-style request and response fields such as `n_predict`, `repeat_penalty`, `return_tokens`, and native timings. |
| `GET /v1/models` | Served model and advertised limits. |
| `GET /health` | Readiness status. |
| `GET /props` | Context, template, runtime, and model information. |
| `POST /apply-template` | Render the model chat template without generating. |
| `POST /tokenize` and `/detokenize` | Token conversion helpers. |

Chat Completions supports `messages`, `stream`, `n` for non-streaming requests,
`stop`, `seed`, logit bias, the usual temperature/top-p/top-k/min-p and penalty
fields, `chat_template_kwargs`, `reasoning_effort`, `enable_thinking`, and
`continue_final_message`. Unknown OpenAI-style request fields are ignored.

`developer` instructions are accepted. For templates with a single system turn,
all system/developer instructions are combined into the leading system message,
with system instructions first and role labels when multiple instruction turns
are present. Order within each role is preserved. This is a template adaptation,
not an independently enforced OpenAI instruction hierarchy.

`response_format: {"type": "text"}` means ordinary unconstrained text, including
tool use and the optional timing footer. Reasoning levels `minimal` and `high`
map to this model's native `low` and `xhigh`; `none` disables thinking. `low`,
`medium`, and `xhigh` remain native values. These are prompt hints, not fixed
reasoning-token budgets. Aliases work in top-level fields and template kwargs;
the configured default remains xhigh when no override is supplied.

Unsupported content parts (images when Vision is disabled, audio, files, video,
and unknown types) return HTTP 400 instead of being silently discarded. Extracted
file/OCR text can still be supplied as text. This applies to `/apply-template` too.

### Image input

Start with `--vision --vision-device 0` to load the checkpoint's vision component
and enable OpenAI `image_url` parts in user messages. The default remains
text-only. `/v1/models` advertises image input only when the component is loaded.
Images are converted to RGB with EXIF orientation applied and transparency on
white, encoded, and passed as native EXL3 embeddings to both target and MTP.

```json
{
  "model": "qwen38-local",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}},
    {"type": "text", "text": "この画像を説明してください"}
  ]}]
}
```

PNG, JPEG, WebP and static GIF data URLs work by default. Optional
`--vision-remote-urls` allows public HTTP(S) URLs; DNS results and redirects are
checked and connections use the validated address. Private addresses and local
file URLs are rejected. Image bytes are capped at 20 MiB, decoded pixels at
16,777,216, images per conversation at 16. `--vision-max-pixels` defaults to
262,144; `detail: low` uses at most 65,536 (subject to the model's minimum).
Processor, count and cache limits have corresponding `--vision-*` flags.

`--vision-cache-mb` bounds the CPU embedding cache (default 128 MiB). Repeated
image data/budget pairs reuse aliases so returning image history can reuse its
KV prefix; evicted images are encoded again. The TP tensor caches are also
byte-accounted and bounded. Nonfinite image features fail before creating a
generation job. Video/audio and images in system/developer/assistant messages
remain unsupported. `/apply-template` renders image placeholders, not pixels;
raw completion endpoints do not accept image embeddings.

Vision loads before text autosplit so resident weights count against available
VRAM. It does not lower the configured context. See
[the image verification report](../../doc/vision_api_validation.md) for V620 TP2/MTP
checks, R9700's safe Linear fallback, and the distinction between image encoder
checks and full inference.

### Optional timing footer

For `/v1/chat/completions`, set `"include_timings": true` in the request to append
a timing line to the assistant's answer. Both streaming and non-streaming work:

```json
{
  "model": "qwen38-local",
  "messages": [{"role": "user", "content": "こんにちは"}],
  "include_timings": true
}
```

The visible line looks like:

```text
Prefill: 456.12 tok/s | Decode: 53.25 tok/s | Total: 16.09 s | Draft: 75.00%
```

The footer is **off by default**. `--include-timings` enables it as a server
default; an explicit request `"include_timings": false` overrides that default.
Client provider bodies can also send the request field, such as OpenCode's
model `body` or LibreChat's endpoint `addParams`. Neither existing client
configuration is enabled automatically.

For clients whose automatic titles reuse the same model parameters, start with
`--plain-model-name qwen38-local-plain` and select that alias as the title model.
It uses the same weights/cache and retains `exl3_metrics`, but defaults to no
footer even when `--include-timings` is enabled. An explicit request setting
still overrides it. This avoids inserting timing text into saved conversation
titles without guessing a request's purpose from its prompt.

Prefill uses only uncached input tokens and the engine's prefill time. Decode
uses generated tokens, including reasoning, and the engine's generation time.
Draft acceptance is accepted draft tokens divided by accepted plus rejected
draft tokens. It is `N/A` when no draft tokens were proposed (including AR-only
requests); it is not the fraction of output tokens produced by the draft.
`exl3_metrics` also includes both draft counters, their sum, and the unrounded
`draft_acceptance_rate`. Version 2 footers and earlier version 1 footers are both
removed from assistant history.
Unavailable phase rates are displayed as `N/A`. Total is measured from server
ASGI request arrival to answer completion, including body parsing, template and
grammar preparation, queueing, prefill, and generation. It excludes delivery to
the client and previous requests or external tool execution in a multi-request
agent turn. Chat's `exl3_metrics.total_seconds` exposes the same unrounded value.

The footer is ordinary `message.content` (a final `delta.content` chunk for SSE),
so clients store it as assistant text. Hidden versioned HTML comments identify
the exact suffix. When assistant history comes back to this engine, the suffix
is removed before templating/tokenization, even if display is now disabled.
`/apply-template` performs the same removal. This prevents the footer from
entering the model input or KV cache recomputation. User/tool content and tool
arguments are preserved. Keep the marker comments in stored history; a client
that removes or rewrites them cannot be recognized. Other engines may retain
the footer in their prompts.

No footer is appended to tool-call messages or `response_format` JSON answers.
Raw-prompt `/v1/completions` and native `/completion` do not use this chat option.
Usage counts and output budgets remain the model-generated token counts; the
server-added footer consumes no generation tokens.

### Token probabilities

Chat requests can set `logprobs: true` and `top_logprobs: 0..20`. Native sampled
token probabilities are returned in `choices[].logprobs.content`, with token
strings, original UTF-8 bytes, natural-log probabilities and top alternatives.
Thinking and tool markup are excluded from the visible-content trace. The
server-added timing footer and injected generation prefixes have no sampled
probability. Tokens retain their original boundaries, including whitespace that
the assistant parser trims; split UTF-8 tokens retain their original bytes.

For streaming, the completed trace is attached to the final choices chunk;
earlier text/think/tool deltas retain their usual incremental delivery. Optional
`include_raw_logprobs: true` adds `exl3_logprobs` for all emitted native tokens,
including reasoning and tool syntax, with token IDs. This is useful for draft
diagnostics. Both alternatives/raw traces require `logprobs: true`.

These are the engine's target probabilities after its logit processing, not
draft-model probabilities or a calibrated confidence score. Zero probabilities
use the OpenAI-compatible `-9999.0` sentinel. Missing/nonfinite traces or failed
text alignment produce an explicit error rather than invented probabilities.
Trace collection is disabled by default and has sampling/transfer/formatting
overhead when enabled. It does not change the default thinking/xhigh setting.

The server also accepts the EXL3 sampler extensions `banned_strings`,
`decode_special_tokens`, DRY, and XTC. Native `/completion` accepts its
llama.cpp field names and returns native timings; it does not implement every
llama.cpp sampler or grammar option.

The server is an OpenAI-compatible implementation, not a complete OpenAI
platform clone. Responses API, Anthropic Messages, embeddings, image
generation, and audio endpoints are not provided.

## Tools and structured output

The constrained tool adapter uses Qwen XML-compatible tokenizers/templates.
Tool requests use OpenAI function definitions:

```json
{
  "type": "function",
  "function": {
    "name": "read",
    "description": "Read a file",
    "parameters": {
      "type": "object",
      "properties": {"path": {"type": "string"}},
      "required": ["path"],
      "additionalProperties": false
    }
  }
}
```

`tool_choice` supports `auto`, `none`, `required`, and a named function.
`parallel_tool_calls` permits up to eight contiguous tool-call blocks. Tool
arguments are generated through LLGuidance JSON Schema constraints, including
nested property schemas, enums, arrays, object values, and local `$defs`
references. String parameters are emitted as JSON strings inside Qwen's XML
parameter elements and are converted back to OpenAI JSON arguments.

`response_format` supports `json_object` and `json_schema`. With thinking
enabled, the structured-output filter starts after the model's `</think>`
token. With thinking disabled, it applies from the first generated token.
Assistant tool-call history preserves call IDs, `assistant.tool_calls`,
`tool_call_id`, and tool results. Streaming emits reasoning deltas and
`delta.tool_calls` in OpenAI SSE chunks.

The server rejects a request that combines an active tool choice with
`response_format`, because one assistant turn cannot simultaneously be forced
to be a normal JSON document and a Qwen function-call document. Tool schemas
using object-level cross-field operators such as top-level `oneOf`, `anyOf`,
`allOf`, dependencies, or property-count constraints are rejected with a
request error instead of being silently weakened. Such operators inside an
individual parameter value remain available to the JSON Schema compiler.

Optional XML parameters remain optional for tools with any number of settings;
the grammar grows linearly instead of enumerating every possible subset. The
compiler also removes redundant `propertyNames: {type: string}` checks at schema
positions, since JSON keys are already strings. Restrictive name checks and
literal `const`/`enum` data are preserved, as are the original post-validation
schemas.

The Qwen XML adapter renders tool history and the template's parameter example
with JSON-quoted strings, including escaped newlines for multiline source code.
This keeps the prompt representation consistent with the generation grammar.
Parallel calls are constrained separately so native newlines between calls work.

The live verification scripts cover automatic and named calls, history,
parallel calls, incremental arguments, structured reasoning, prefix reuse,
errors and cancellation. See [the dated integration report](../../doc/opencode_api_validation.md) and
[the OpenCode-generated sample](../../examples/opencode_lru/README.md) for actual results.

## Errors and limits

- `401`: the request did not supply the configured `Authorization: Bearer` or
  `x-api-key` value.
- `400`: invalid request fields, unsupported schema controls, a missing chat
  template, or a prompt that cannot fit the cache with at least one output
  token.
- `404`: an unknown requested model.
- `502`: a non-streaming generation finished with malformed tool or structured
  output. A stream already started instead emits an error SSE event.
- `503`: the server or runtime is not ready.

The default server cache is the model's maximum context unless `-cs` is given.
Set `-cs` explicitly for long-context models. `--context-limit` caps the
usable context below the allocated cache, and `--max-output-tokens` caps each
response. A portable OpenCode configuration cannot increase either server-side
limit.

## V620 × 2 TP2 configuration

The following values were used for the V620×2 HTTP validation on 2026-10-01.
Use the Qwen3.8 Flash Next 3.05 bpw checkpoint and an extension built for
`gfx1030`, following [the build/model guide](../../doc/reproduce_v620.md).
Select your own GPU UUIDs and PCI addresses in that guide; these identifiers
are machine-specific.

```text
LD_PRELOAD=libhsa-runtime64.so
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
EXL3_TP_REPLICATE_ROUTER=1
EXL3_NGRAM_MLOCK=1
EXL3_HOST_MEM_RESERVE_MB=0
EXL3_BATCH_RECURRENT_PRUNE=1
EXL3_TP_TIMEOUT_S=180
NCCL_DEBUG=WARN
HSA_DISABLE_COREDUMP_ON_EXCEPTION=1

-m /work/models/qwen38-flash-next-exl3-3.05bpw \
-tp -tpb nccl -mtp -ngr \
-gs 28,28 -cq 5,4 -cs 32768 \
-ambs 1 -chunk_size 2048 \
-ndt 4 -dds -dc 0.6 \
-temp 0 -ctk '{"enable_thinking":false}' \
-host 0.0.0.0 -port 3953 \
-key replace-with-a-local-key \
-smn qwen38-local \
--context-limit 32768 --max-output-tokens 8192 \
--power-socket /work/runs/opencode-api/replace-power.sock \
--audit-log /work/runs/opencode-api/replace-requests.jsonl
```

The measured container used a host loopback bridge. For a direct host launch,
use `-host 127.0.0.1` and replace the container paths. `-cq 5,4` selects K5/V4 KV cache, `-tp -tpb nccl` selects two-way
tensor parallel, `-mtp -ndt 4 -dds -dc 0.6` selects the packed MTP path with a
dynamic window of up to four tokens from the original 3-bit MTP head, and `-ambs 1` is the batch-one setting.

The TP+MTP path accepts independent K/V cache widths from 2 to 8 bits via
`-cq k_bits,v_bits`; K5/V4 remains the reference configuration. The requested
widths are applied to both target and draft, and startup audits reject an
unexpected cache type or bitrate. See the [2026-10-02 K2/V2 and K8/V8 smoke checks](../../doc/kv_cache_width_smoke.md)
for the limited startup and text-response validation. Other widths and long-context
quality are not covered by those checks. Larger bit widths need more VRAM, so
reduce `-cs` and `--context-limit` together when testing them.

QSA now uses a [compact raw tail](../../doc/qsa_compact_cache.md) by default.
The measured K5/V4 target-plus-MTP cache at 512Ki saves 1.5234375 GiB while
retaining the pooled history. Rebuild the native extension to use the compact
graph path; older extensions use the eager path. Set `EXL3_QSA_FULL_RAW=1`
before starting the engine to restore the full raw plane for comparison.

For the V620 pair with vision on GPU0 and MTP on GPU1, the
[2026-10-02 memory-balance check](../../doc/v620_tp_memory_balance.md) selected
`-gs 28.375,27.625` at 512Ki. It reduced the sampled whole-card peak difference
to about 0.20 GiB with comparable coding-request speeds. A 768Ki allocation plus
short inference passed with `-gs 28.5,27.5 -cs 786432 --context-limit 786432`.
The subsequent [near-full-cache memory check](../../doc/full_cache_768ki_memory.md)
made that the deployed default. Historical reproduction commands retain
their recorded reference budgets.

Start the power helper below before executing this server command.
The server itself does not require root. The measured Engram configuration
needs a 34 GiB memlock limit and `nofile=65536` on the inference process. Apply
the limits to a dedicated subshell and execute the unprivileged server from
that subshell. Set `MODEL_DIR`, `EXL3_API_KEY`, GPU visibility, and the
environment values above first:

```bash
export LD_PRELOAD=libhsa-runtime64.so OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export EXL3_TP_REPLICATE_ROUTER=1 EXL3_NGRAM_MLOCK=1
export EXL3_HOST_MEM_RESERVE_MB=0 EXL3_BATCH_RECURRENT_PRUNE=1
export EXL3_TP_TIMEOUT_S=180
(
  set -e
  sudo prlimit --pid "$BASHPID" \
    --memlock=36507222016:36507222016 --nofile=65536:65536
  exec python -m rocm_tools.exl3_server.server \
    -m "$MODEL_DIR" -tp -tpb nccl -mtp -ngr \
    -gs 28,28 -cq 5,4 -cs 32768 -ambs 1 -chunk_size 2048 \
    -ndt 4 -dds -dc 0.6 -temp 0 \
    -ctk '{"enable_thinking":false}' \
    -host 127.0.0.1 -port 3953 \
    -key "$EXL3_API_KEY" -smn qwen38-local \
    --context-limit 32768 --max-output-tokens 8192 \
    --power-socket "$RUN_DIR/power.sock"
)
```

For the measured power policy, start the tracked helper in a separate terminal
before the server. It requires sudo only for the GPU policy files and restores
the original policies when its client disconnects. Create a private artifact
directory, and replace the two PCI addresses with your selected V620s:

```bash
RUN_DIR=$(mktemp -d "$PWD/api-run.XXXXXX")
chmod 700 "$RUN_DIR"
sudo -v
sudo -n python3 rocm_tools/rdna2/power_server.py \
  --devices "$GPU0_PCI" "$GPU1_PCI" \
  --socket "$RUN_DIR/power.sock" \
  --report "$RUN_DIR/power.json"
```

Pass `--power-socket "$RUN_DIR/power.sock"` to the server when using the helper.
Set the same `RUN_DIR` in the server terminal. The runtime policy uses `auto`
for batch-one prefill and idle, `profile_peak` for draft/verification/decode,
and `profile_peak` throughout configured batches greater than one. Do not expose the server's `0.0.0.0` container
bind outside the local bridge.

For the full TP2 benchmark harness, including exact prompt manifests, power
restoration audits, and the 34 GiB memlock bootstrap, see
[`doc/reproduce_v620.md`](../../doc/reproduce_v620.md). That guide also records
the validated `EXL3_HOST_MEM_RESERVE_MB=0` diagnostic override. It is not a
normal default and does not change OS swap or ARC settings.

## Sampling and cancellation

Concurrent requests are dynamically batched. SSE disconnects cancel the owned
generation job. A successful SSE response ends with a finish-reason chunk and `[DONE]`. `usage` reports prompt, cached prompt, and
completion token counts where the runtime has them; `stream_options.include_usage`
adds usage to the final SSE chunk.

`/health` reports active versus queued jobs. An absent, failed, or stopped
generator returns 503 with a safe status code; native exception text is not
exposed. Generation failures return 503 (or an SSE error event if streaming has
already started), never a successful empty completion. Optional
`--request-timeout SECONDS` returns `inference_timeout` and cancels the owned
job, including queued work. The default is zero, with no deadline. Failure or
cancellation of an `n`-choice request releases its other choices, including if
constructing a later choice fails.

For bounded crash/hang recovery, run the optional supervisor in the same PID
namespace as the engine:

```bash
python -m rocm_tools.exl3_server.supervisor \
  --health-url http://127.0.0.1:3953/health \
  --status-file /private/run/engine-status.json \
  --startup-timeout 300 --unhealthy-timeout 90 --max-restarts 3 -- \
  python -m rocm_tools.exl3_server.server -m /path/to/model -port 3953
```

It retries after process exit or sustained failed health, kills only the process
group it created (including orphaned engine workers), and never resets a GPU.
Its private status file records starting/ready/unhealthy/restarting/failed
states and restart count. It exits nonzero when its retry budget is exhausted;
an intentional SIGTERM/SIGINT cleans up and exits zero. A native call can block
the inference event loop, so an in-process deadline alone cannot recover that
case; the supervisor remains outside it. Set the health grace above normal
single-step prefill latency and use a startup grace large enough for model/RAM
loading.

The supervisor must not wrap host-side `docker exec`: run it inside the
container, with the command and health endpoint belonging to that engine.
If using the one-client power helper, supervise a complete launch session which
creates a fresh helper/socket on each attempt; its socket cannot be reused after
engine disconnect. The existing live V620 service is not automatically replaced
or restarted when installing these files.

The server logs a private JSONL audit only when `--audit-log` is supplied. Do
not put API keys in that path or in an OpenCode project file committed to a
repository.
