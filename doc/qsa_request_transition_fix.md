# QSA request transition crash fix (2026-10-01)

Qwen3.8 Flash Next could finish a long LibreChat conversation, then fault on the
automatic title request. The pending title and subsequent conversation stopped
receiving responses while the TP workers continued consuming power. This was a
sparse attention memory access error in the engine.

## Cause

`BC_Attention::run_gr` used the entire persistent `qsa_scores` workspace as the
input to top-k selection. The few-query scoring kernel writes only the current
context's live prefix. Finite scores from a previous longer request, or another
slot, could therefore survive in the workspace's tail and win top-k selection.
The resulting pool indices could point beyond the current block table, leading
to an invalid GPU memory read during sparse KV gathering.

The Python workspace initialization to negative infinity was insufficient:
later requests reuse the same workspace. Graph replay already patched the top-k
scan width, but eager execution and graph warmup used the full tensor width.
ROCm graph replay is disabled by default, so this affects ordinary ROCm requests.

In this model QSA starts above 2,051 tokens. The approximately 3,002-token title
request was itself sparse, even though it was much shorter than the preceding
55k-token conversation. Valid sequence lengths and incoming block tables did
not prevent the crash because the selected pool indices were invalid.

## Change

- Native top-k receives a view limited to `t_scan` on every call. The view keeps
  the original address and row stride; the shared workspace remains reusable.
- Pool expansion masks token indices outside each query's visible context before
  they can become sparse KV reads. Legitimate pools and the incomplete tail remain
  unchanged. This protects the shared DSA expansion path too.
- Chat admission logs include request ID, timestamp, prompt length, output budget,
  stream mode, tool count, and message roles before GPU execution. Message text,
  tool arguments, and credentials are excluded.

The implementation is in [attention.cpp](../exllamav3/exllamav3_ext/libtorch/attention.cpp)
and [dsa_triton.py](../exllamav3/modules/attention_fn/dsa_triton.py).

## Validation

The test environment used V620×2, TP2/RCCL, EXL3 3.05bpw target weights, the
original 3-bit MTP head, dynamic MTP up to four tokens, K5/V4 cache, a 524,288-token
cache allocation, batch one, thinking/xhigh, and one mlocked RAM Engram owner.
ROCm was 7.2.1 with graph replay disabled. The rebuilt extension reused unchanged
objects from the existing verified build and recompiled `attention.cpp`.

| Check | Result |
|---|---|
| Unfixed long-conversation → title replay | GPU faults reproduced, including with only 128 generated tokens before the title |
| Poisoned score workspace | Full-width selection picked 1,024/1,024 indices outside the live prefix; narrowed selection stayed within it |
| Native fix alone, 54,990-token prompt → 128-token generation → title | Both completed; title HTTP 200 |
| Native fix alone, same prompt → 4,096-token generation → title | Both completed; title HTTP 200 in 5.68 s |
| API/protocol/schema/runtime suite | 96 passed |
| GPU selection/expansion regression suite | 3 passed |
| Deployed native + Python fix, 3,002-token title after a 9,643-token OpenCode request | HTTP 200, 502 output tokens, natural stop, final language/title answer present |
| LibreChat with deployed fix | Firecrawl MCP roundtrip, new conversation, follow-up to the first conversation, and both automatic titles completed |
| OpenCode 2.0.12 with deployed fix | Standalone local provider, thinking/xhigh; requested Japanese answer received |

The successful model replays loaded the native fix with the original Python
expansion code. They establish that limiting selection fixes the reproduced
failure without relying on the extra expansion mask. The second replay reused
prompt cache; its timings are functional verification, not a cold-prefill
benchmark. This does not establish full 512Ki-input coverage or bit-identical
generation after request/cache history changes.

The final service remained healthy after these checks, with both V620 cards back
at the batch-one idle `auto` power setting. No new GPU page fault, reset, or OOM
entry appeared in the kernel log during the fixed-service checks. The native
library SHA256 was
`bb489ef3dd1600d5f205b97cbfb835e772317f071f8c546b3b513a13472b8842`.

## Reproduction

Rebuild the native extension after updating the source; updating Python alone
does not apply the selection fix. See [the V620 build instructions](reproduce_v620.md).
When loading a separately built extension, put its build directory before the
source directory in `PYTHONPATH` and restart the model workers.

Run the focused checks in the ROCm/Python environment:

```bash
EXL3_SCHEMA_TEST_TOKENIZER=/path/to/model/tokenizer.json \
  python -m pytest rocm_tools/exl3_server/tests -q
EXL3_GPU_TESTS=1 \
  python -m pytest rocm_tools/rdna2/tests/test_qsa_selection_gpu.py -q
```

The GPU tests cover a poisoned tail, the split-top-k path with a non-contiguous
row stride, and per-query visibility during pool expansion. A model-level check
should keep one server process alive, generate from a long tool conversation,
then send a shorter request above the model's QSA threshold and create another
conversation. Restarting between requests removes the workspace reuse trigger.
Original private conversation bodies and tool results are not published.
