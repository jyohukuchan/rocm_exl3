# API feature work (2026-10-01)

Requested scope: implement the medium-or-higher priority integration work, add
draft acceptance to the opt-in timing footer, then implement Responses API.
Each completed work item is tested and pushed separately. The active LibreChat
inference process must remain running while source changes are prepared.

| Item | Source state | Validation / remaining work |
|---|---|---|
| Draft acceptance footer | Implemented, pushed as `9f43559` | 112 API tests; denominator is accepted + rejected; old v1 footer cleanup retained |
| Developer, text response format, reasoning aliases, unsupported input errors | Implemented, pushed as `7a458fa` | 133 tests including actual Qwen tokenizer template rendering |
| Failure/cancellation cleanup, readiness, bounded supervisor | Implemented | 149 tests including real CPU process crash and failed-health recovery; no live GPU crash/reset attempted |
| Sanitized client request regression fixtures | Pending | Preserve actual client transport/schema shapes without publishing private conversation bodies |
| Vision API wiring and ROCm/TP/MTP verification | Pending | Preserve 512Ki context; verify available VRAM, preprocessing, embeddings, image history and resource ownership |
| Logprobs | Pending | Wire native probabilities with correct visible-token/UTF-8 alignment and SSE/non-stream parity |
| Responses API | Pending, after preceding items | Text/image input, instructions, tools, reasoning, structured output, streaming events, response history and client verification |

The running service still uses its earlier imports. Pushing Python changes does
not reload an already running engine. New-source tests run in separate CPU test
processes inside the existing container; they do not stop or reload LibreChat's
model. Live deployment and GPU validation remain separate completion gates.
Long-context batch ≥2 and CORS changes were ranked below the requested threshold
for the current batch-one local configuration.
