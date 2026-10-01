# TP2 + MTP: K2/V2 and K8/V8 startup smoke checks

Date: 2026-10-02 (Asia/Tokyo). Both configurations started and answered one
`hello` request. This is a startup/text-response check, not a speed benchmark
or a quality/long-context evaluation.

## Change

The specialized TP+MTP HTTP runtime now accepts independent K and V widths
from 2 to 8 bits. K5/V4 remains the reference configuration, rather than a
mandatory bitrate. The same requested widths apply to target and MTP caches.
The requested-vs-observed cache audit remains enabled, including rejection of
silent FP16 fallback or different observed bitrates. Explicit quantized caches
are still required on this specialized path.

No attention, quantization, TP, or MTP kernel changes were needed. CPU runtime
tests: 31 passed, including symmetric endpoint widths and asymmetric widths.

## Actual model checks

- Hardware: Radeon Pro V620 x2 (gfx1030), TP2/RCCL.
- Target: Qwen3.8 Flash Next EXL3 3.05bpw; original packed 3-bit MTP weights.
- MTP: dynamic maximum 4 tokens, confidence 0.6; batch limit 1.
- Engram: one locked CPU RAM owner; the existing placement audits passed.
- Both runs: 8,192 cache tokens, prefill chunk 2,048, GPU budgets 28/28 GiB.
  The smaller cache isolates bitrate support from the VRAM requirement of a
  512Ki cache. Vision remained enabled, but only text input was tested.
- Request: `hello`, temperature 0, maximum 64 output tokens, non-streaming,
  thinking and timing footer disabled for this request.

| Cache | Startup | Observed target / draft cache | Reply | Finish / health after |
|---|---|---|---|---|
| K2/V2 | Passed | 12 / 1 QSA quantized layers, actual 2/2 bits | `Hello! How can I help you today?` | `stop` / `ok` |
| K8/V8 | Passed | 12 / 1 QSA quantized layers, actual 8/8 bits | `Hello! How can I help you today?` | `stop` / `ok` |

The [sanitized evidence](../benchmarks/2026-10-02/kv-cache-smoke.json) contains
the requests, replies, usage, and observed cache/TP/MTP audit summaries. The
actual cache bitrates were checked for both target and draft; flags alone were
not treated as proof.

## Reproduction

Use the [server setup](../rocm_tools/exl3_server/README.md) with the same model,
TP+MTP placement, environment, and process limits. For each separate startup,
set `-cs 8192 --context-limit 8192 --max-output-tokens 256` and either `-cq 2,2`
or `-cq 8,8`. Start a fresh power helper for each process, as described in the
setup. Confirm `/health` and inspect `/props` before submitting:

```json
{
  "model": "qwen38-local",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 64,
  "temperature": 0,
  "enable_thinking": false,
  "include_timings": false,
  "stream": false
}
```

This does not validate every supported bitrate combination, sparse long-context
attention, long outputs, concurrency, tools, vision responses, or quality.
K8/V8 at 512Ki was not tested.
