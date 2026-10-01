# QSA compact raw tail: memory and output validation

2026-10-02 (Asia/Tokyo). The QSA cache now retains pooled keys for the complete
history and a small FP16 raw ring per physical page, instead of every historical
raw indexer key. The pooled values are unchanged in the controlled checks below.
This is the default for new cache instances. Weights and KV quantization are
unchanged.

## Storage and correctness

With the current MTP history, each 256-token page retains 16 raw keys rather
than 256: 93.75% less raw-key storage. The ring grows to cover larger configured
target histories. TP exports the resolved ring width to its workers. A generator
rejects a draft window that cannot fit the allocated ring instead of silently
reading overwritten keys.

New projections feed pooling directly. Pooling runs before the ring append,
including inside native decode graphs, so a long prefill cannot overwrite the
preceding partial pool's keys before reading them. The FP32 accumulation order,
FP16 rounding, normalization, and RoPE calculations are retained.

Page copies and CPU paging move the ring and pooled plane together. Partial
prefix reuse ends at a pool boundary; at most three extra tokens are re-fed for
this model. Recurrent models continue to reuse their page-aligned checkpoints.
Non-recurrent banned-string rewinds beyond the ring replay from a page boundary.
`EXL3_QSA_FULL_RAW=1`, set before cache construction, restores the original full
raw plane for comparison or unusual draft configurations.

## Measured allocation

Actual cache tensor audits on V620 x2, Qwen3.8 Flash Next EXL3 3.05bpw,
TP2/RCCL, K5/V4, original packed 3-bit dynamic MTP max4/confidence0.6,
batch limit 1, cache capacity 524,288:

| Target + MTP cache | Before | Compact | Saved |
|---|---:|---:|---:|
| Bytes | 6,543,114,240 | 4,907,335,680 | 1,635,778,560 |
| GiB | 6.09375 | 4.5703125 | 1.5234375 |

These are cache allocations, not the entire process or driver reservation.
The corresponding 1Mi cache calculation is 12.1875 -> 9.140625 GiB, a saving
of 3.046875 GiB. A 1Mi allocation/inference run was not performed.

## Validation

- Six GPU tests passed on both V620s. Compact pooling was compared bit-for-bit
  with the unchanged full-raw pooling kernel using 2/4-warps, two sequences at
  different positions, page crossings, long appends with many ring wraps, and
  rejected speculative tokens. Cache copying, history sizing, and TP export
  were also exercised.
- ROCm CPU suite: 829 passed, 9 skipped, 148 subtests. Server runtime: 31 passed.
- Nine actual HTTP/model cases: hello, Japanese explanation, factorial code,
  approximately 11.6K Japanese/code prompts, exact prefix reuse, a short request
  after a long request, and an extended prefix. All completed with natural stop.
  Factorial and duplicate-removal functions were executed against expected
  results; arithmetic answers were checked.
- Keeping the original TP planning costs while using compact storage recovered
  **9/9 exact answer strings and exact raw token/probability traces**, covering
  347 generated tokens. The test changed planning costs only, not the compact
  cache representation or pooling implementation.
- Normal production planning uses the smaller actual cache cost. It changed
  MoE slices for 30 components (60 rank/component entries); three answers
  differed in phrasing/spacing. All nine semantic/code checks still passed.
  Restoring the old planning costs removed these differences, consistent with
  finite-precision changes from the different expert split, rather than an
  altered pooled key.

The [sanitized quality evidence](../benchmarks/2026-10-02/qsa-compact-quality.json)
contains answers, trace hashes, cache byte counts, and the changed TP slices.
These limited checks show no quality regression; they are not a general quality
evaluation, a speed benchmark, or a full-512Ki retrieval test. Batch concurrency,
vision answers, and full 1Mi context were not evaluated here.

## Build and reproduce

Rebuild the native extension using [the V620 build instructions](reproduce_v620.md).
The C++ graph path must pool before appending the compact raw tail. An older
extension lacks `qsa_compact_supported`; Python then uses the exact eager path
for compact QSA instead of passing the ring to the old graph implementation.

Run the GPU comparisons in the ROCm environment against the rebuilt extension:

```bash
EXL3_GPU_TESTS=1 LD_PRELOAD=libhsa-runtime64.so \
python -m pytest -q -p no:cacheprovider rocm_tools/rdna2/tests/test_qsa_compact_gpu.py
```

Use the existing [server configuration](../rocm_tools/exl3_server/README.md).
The K5/V4, 512Ki, TP2/MTP, batch1, xhigh, vision and optional timing-footer
settings are retained in the deployed service. The comparison requests disabled
thinking per request; they did not change the server default.
