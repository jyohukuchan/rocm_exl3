# Image API verification (2026-10-01–02)

The optional `--vision` path accepts OpenAI user `image_url` content parts,
encodes their pixels with the checkpoint's native vision component, and supplies
EXL3 embeddings to target and MTP jobs. Text-only serving remains the default.
The model used was Qwen3.8 Flash Next EXL3 3.05bpw with its existing 5-bit vision
weights and original 3-bit MTP head.

| Check | Result |
|---|---|
| CPU API/processor/template/ownership regressions | 170 tests passed |
| Isolated V620 encoder, red/blue 256×256 images | Finite `[64, 2560]` features; distinct mean absolute difference 0.0079397 |
| Isolated R9700 encoder, same images | Finite `[64, 2560]` features; difference 0.0079563 |
| V620 pair, TP2/RCCL, K5/V4, dynamic MTP up to 4 | Red image answered `赤`; blue image answered `青` |
| Non-streaming / SSE | Both completed with natural stop and timing/draft footer |
| Returning image + assistant history | Follow-up correctly recalled the red image |
| Context allocation | `/props` confirmed 524,288, batch one; not a full-512Ki-input test |

The encoder probes peaked at approximately 446 MiB of Torch reserved memory on
V620 and 486 MiB on R9700. These are allocator measurements, not the complete
driver/context footprint. The live pair retained one mlocked RAM Engram owner,
thinking/xhigh and the batch-one power policy. After the user authorized
restarts, a new process loaded the vision component before target autosplit.

HF templates and native Qwen image embeddings both contain vision delimiters.
The tokenizer now replaces the complete wrapped placeholder where applicable,
avoiding duplicated start/end tokens. Stable cached aliases preserve image
identity in returning history. Producer/consumer TP caches now account bytes
and evict in order, and cached transfers no longer copy payloads unnecessarily.

R9700 initially trapped in `model.visual.merger.linear_fc1` for 64 input rows.
The gfx12 WMMA wrapper is explicitly unimplemented; cooperative GEMM also
reaches it, despite an older comment mentioning only fused MoE. Python EXL3
Linears now use the existing reconstruct + GEMM path for multi-row gfx12 input.
Single-row decode and gfx10/gfx11 paths are preserved. This is a functional
fallback, not a native gfx12 WMMA implementation or a performance claim.

The default image pixel budget is 262,144 (65,536 for `detail: low`), with a
16-image conversation limit and a 128 MiB CPU embedding cache. Processor limits,
EXIF/transparency, invalid payloads, internal URL rejection and DNS pinning,
cache eviction, nonfinite features, API-to-job wiring and delimiter counts have
focused CPU tests. Remote URLs are opt-in; video/audio remain explicit errors.

LibreChat browser attachment verification is a separate client check. The API
checks above establish native inference; they do not establish every frontend
upload mode or image resolution. The short-caption timings are not throughput
benchmarks.
