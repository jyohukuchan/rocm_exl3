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
32,768-token context, reserves 24,576 input tokens and 8,192 output tokens, and
uses `enable_thinking: true` with `reasoning_effort: low`, matching the
OpenCode coding sample. Set `enable_thinking: false` for text-only smoke checks.

Start OpenCode from the project containing that configuration:

```bash
opencode /path/to/project
```

The project configuration does not modify the existing global MCP setup. It
selects the local `rocm-exl3` provider for the primary model and its worker,
title, and compaction agents. Keep the server bound to loopback unless you add
authentication and a deliberate network boundary.

For initial validation, leave compaction disabled. This makes request bodies and
tool history easier to inspect. Once the provider works, set
`compaction.auto` to `true` and size `compaction.buffer` for the amount of
history you want to retain. The model limit in the example is an API budget;
the actual cache still comes from `-cs`.

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

The server logs a private JSONL audit only when `--audit-log` is supplied. Do
not put API keys in that path or in an OpenCode project file committed to a
repository.
