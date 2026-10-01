# API feature work (2026-10-01)

Requested scope: implement the medium-or-higher priority integration work, add
draft acceptance to the opt-in timing footer, then implement Responses API.
Each completed work item is tested and pushed separately. The active LibreChat
inference process was initially left running while source changes were prepared.
On 2026-10-02 the user authorized stopping/restarting it for deployment and checks.

| Item | Source state | Validation / remaining work |
|---|---|---|
| Draft acceptance footer | Implemented, pushed as `9f43559` | 112 API tests; denominator is accepted + rejected; old v1 footer cleanup retained |
| Developer, text response format, reasoning aliases, unsupported input errors | Implemented, pushed as `7a458fa` | 133 tests including actual Qwen tokenizer template rendering |
| Failure/cancellation cleanup, readiness, bounded supervisor | Implemented, pushed as `d7fa490` | 149 tests including real CPU process crash and failed-health recovery; no live GPU crash/reset attempted |
| Sanitized client request regression fixtures | Implemented | Actual LibreChat/Firecrawl and OpenCode shapes, stripped conversation/schema annotations; real tokenizer/LLGuidance tests and one-server API transition regression |
| Vision API wiring and ROCm/TP/MTP verification | Implemented and deployed | 170 tests; V620/R9700 encoders; V620 TP2/MTP red/blue inference and returning image history; 512Ki allocation retained. Browser upload check remains |
| Logprobs | Implemented and deployed for verification | Native TP2/MTP JSON and SSE traces matched `こんにちは`; 41 raw tokens, one visible token, top 3 alternatives; UTF-8/role separation covered by CPU tests |
| Responses API | Pending, after preceding items | Text/image input, instructions, tools, reasoning, structured output, streaming events, response history and client verification |

The `vision-logprobs-02` service loads the new API, Vision and probability code. Further Python
changes require a restart to deploy. CPU tests run in separate processes.
Live deployment and GPU validation remain separate completion gates.
Long-context batch ≥2 and CORS changes were ranked below the requested threshold
for the current batch-one local configuration.
