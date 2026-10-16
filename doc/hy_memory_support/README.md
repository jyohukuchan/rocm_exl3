# Hy-Memory-30B-A3B inference support

2026-10-17 JST. The published
[`Hy-Memory-30B-A3B EXL3 4.08bpw`](https://huggingface.co/jyohukuchan/Hy-Memory-30B-A3B-exl3-4.08bpw)
works through the normal engine and HTTP entry points on **RDNA2 (V620)** and
**RDNA4 (R9700)**. This verification covers those GPUs and the paths below.

## Implementation

- HYV3 already has registered GQA, Q/K norms, sigmoid routing with expert bias,
  shared experts and EXL3 matrices. The preceding `expert_hidden_dim` fix correctly
  resolves this model's expert width to 768 despite the legacy 1536 field.
- `HyV3Model.default_chat_prompt()` now uses the checkpoint's actual Jinja
  template when available. It no longer injects the original Hy3 `:opensource`
  tokens into Hy-Memory requests. Template-less legacy checkpoints keep their
  existing fallback. The default helper uses native `no_think`.
- HTTP controls select the HYV3 family: native `no_think`, `low`, `high`, client
  aliases and explicit request overrides are resolved before rendering, constraint
  construction and parsing. Qwen's `high` → `xhigh` policy remains Qwen-specific.
- Native `<tool_calls>` groups and argument tags are constrained with llguidance
  and converted to OpenAI tool calls. Calls are bounded to the requested parallel
  policy. Native SSE buffers arguments until each call closes; quote-aware parsing
  preserves literal control-looking text inside JSON strings. Generated tools are
  never executed by these checks or by the server.

No new GPU kernels or RDNA3/3.5 paths were required for this model.

## Actual checks

| Path | Checks | Evidence |
| --- | --- | --- |
| V620 single GPU | Native helper/template equality; extraction; 3,331-token correction; short abstention after long input | [v620-single.json](v620-single.json) |
| V620 ×2, RCCL TP2, output rank 1 | Same three generation checks, actual active devices `[0,1]` | [v620-tp2.json](v620-tp2.json) |
| R9700 single GPU | Same three generation checks | [r9700-single.json](r9700-single.json) |
| R9700 HTTP | Native/default/high/xhigh template checks; JSON Schema response and named native tool call, each JSON + SSE | [r9700-http.json](r9700-http.json) |
| V620 ×2 HTTP, RCCL TP2, output rank 0 | Same HTTP checks through generic `model_init` and runtime placement/cache audit | [v620-tp2-http.json](v620-tp2-http.json) |

The generation checks use an 8,192-token FP16 cache and 2,048-token chunks.
Memory extraction excludes temporary weather; correction requires deleting
superseded locations, preserving the vegetarian preference; abstention must
return an empty list when employer/address facts are absent. All checks passed.
HTTP native tools used a string fact and integer revision constrained to 2.

Unit/HTTP contract tests exercise the existing Qwen path and the new HYV3 path,
including one-byte UTF-8 chunks, parallel native calls, literal tags inside JSON,
and a split reasoning close token. Unfinished thought text never becomes a tool
call when generation reaches a token limit. The final targeted suite passed **129 tests**,
with **7 optional fixtures skipped**.

These are functional checks, not a BF16 quality comparison, throughput ranking,
full-262,144-token validation, or an RDNA3/3.5 qualification. The unchanged weights
retain the [prior release's observed prompt-policy caveat](../hy_memory_quantization/README.md):
explicitly request current facts and complete removal of superseded/transient data.

## Environment and startup

Existing containers expose only their intended GPUs:

- `rocm-exl3-v620-pair`: the two V620s; extension from `/work/lib-moe-exact`.
- `rocm-exl3-r9700-conv`: R9700; extension from `/work/lib-r9700-opt`.

The direct reports include Torch/HIP versions, the exact extension path and SHA-256.
The R9700 conversion container needed the already-declared HTTP/test dependencies
`jsonschema`, `sse-starlette`, `pytest-asyncio` and `httpx` before these checks.

The container's default **nofile soft limit 1024** caused an initial TP load to
report GPU OOM while approximately 29 GiB of VRAM remained free. Raising the
launcher soft limit to **65536** fixed the same model load. The dedicated check
script raises its own soft limit (within the existing hard limit), following the
existing V620 harness policy. The server/library continues to leave process
limits to the caller. No host-wide limit or power-policy changes were made.

R9700 server:

```bash
docker exec -w /src rocm-exl3-r9700-conv bash -c \
  'ulimit -S -n 65536 && exec python rocm_tools/exl3_server/server.py \
   -m /work/models/Hy-Memory-30B-A3B-exl3-4.08bpw \
   -gs 24 -cs 8192 -ambs 1 -host 127.0.0.1 -port 3983 -smn hy-memory'
```

V620 TP2 server:

```bash
docker exec -w /src -e PYTHONPATH=/work/lib-moe-exact:/src \
  -e EXL3_TP_LOAD_TIMEOUT=600 rocm-exl3-v620-pair bash -c \
  'ulimit -S -n 65536 && exec python rocm_tools/exl3_server/server.py \
   -m /work/models/Hy-Memory-30B-A3B-exl3-4.08bpw \
   -tp -tpb nccl -gs 24,24 -cs 8192 -ambs 1 \
   -host 127.0.0.1 -port 3984 -smn hy-memory'
```

`-tpb nccl` selects the ROCm RCCL implementation; the inherited CUDA-native TP
backend is unavailable on ROCm. Do not pass `-mtp`: this model has no MTP layers.

Run `rocm_tools/hy_memory_check.py -m /work/models/Hy-Memory-30B-A3B-exl3-4.08bpw
--report /work/runs/hy-memory-support/check.json` in the intended container;
add `--tp` for the V620 pair. Run `rocm_tools/hy_memory_http_check.py --url
http://127.0.0.1:3983 --report /work/runs/hy-memory-support/http.json` against a
running server (3984 for the V620 command). Validation servers were stopped
after the checks.
