# Hy-Memory-30B-A3B: EXL3 4-bit-base conversion

Source: [`tencent/Hy-Memory-30B-A3B`](https://huggingface.co/tencent/Hy-Memory-30B-A3B/tree/64a460a94d5498b7316fb02395c72d01c47a7347).
Source revision: `64a460a94d5498b7316fb02395c72d01c47a7347`.
Original weights are Apache-2.0 without additional restrictions. The release
preserves Tencent's complete LICENSE and copyright and identifies the modified
quantization in NOTICE.

## Completed release (2026-10-16 JST)

Public model:
[`jyohukuchan/Hy-Memory-30B-A3B-exl3-4.08bpw`](https://huggingface.co/jyohukuchan/Hy-Memory-30B-A3B-exl3-4.08bpw).

- Four weight shards: **15,946,470,830 bytes / 15.946 GB / 14.851 GiB**.
- Decoder: **4.077745849297574 bpw**; head: **8 bits**.
- All 18,384 decoder allocations and the head match the stored EXL3 trellis shapes.
- All 288 unquantized tensors passed their dtype audit: 194 BF16 embeddings/norms,
  47 FP16 routers, and 47 original FP32 routing-bias tensors.
- All 48 blocks completed without non-finite calibration warnings. The conversion
  used checkpoint-based process recycling to bound host allocator growth.
- Final memory extraction, correction and abstention smoke checks passed on R9700.
  These are functional checks, not a BF16 quality benchmark.
- The original Apache-2.0 LICENSE is preserved byte-for-byte with a modification
  NOTICE, source model-card text, recipe, calibration metadata and validation evidence.

The installed model is
`/home/homelab1/datapool/ai_models/safetensors/Hy-Memory-30B-A3B-exl3-4.08bpw`.
It shares hard-linked files with the converter output under
`/home/homelab1/datapool/rocm-exl3-rdna2/models/Hy-Memory-30B-A3B-exl3-4.08bpw`.
See [run.json](run.json), [validation.json](validation.json),
[file-manifest.json](file-manifest.json), [hf-publication.json](hf-publication.json)
and the [model card](model-card.md). Anonymous access, all file sizes and the four
weight LFS SHA-256 digests were verified. README, LICENSE, NOTICE, release metadata
and manifest bytes also match the public release. The manifest includes the LFS
rules automatically added by the Hub to `.gitattributes`; verification recovery
updated only the manifest, without reuploading or modifying weights.

**Observed prompt behavior:** the initial broader instructions retained a
superseded location with an annotation and included temporary weather.
The stricter checks explicitly require only currently valid facts, complete
deletion of superseded information and exclusion of transient observations.
All three passed with those instructions. The initial prompts and responses
remain in [validation-initial.json](validation-initial.json); this behavior has
not been compared with BF16 and cannot be attributed specifically to quantization.

## Compatibility

HYV3's `expert_hidden_dim` is authoritative when present. This checkpoint has
`expert_hidden_dim=768`, but also retains `moe_intermediate_size=1536`; actual
routed and shared expert gate/up weights are `[768, 2048]` and down weights are
`[2048, 768]`. Revision `67cbdd1` fixes configuration loading for both the trunk
and MTP while retaining the legacy fallback. Three regression tests cover both
fields, the new field alone and the old field alone, including shared experts.
The source config is preserved intact in the quantized distribution.

## Allocation and calibration

This is the allocation produced by `-b 4.0 -hq -hb 8`, frozen into an explicit
recipe: 18,048 routed-expert matrices at 4 bits, three layer-0 dense matrices at
5 bits, 333 attention/shared-expert matrices at 6 bits, and the head at 8 bits.
Decoder average: **4.077745849297574 bpw**, reported as **4.08 bpw**. The head,
embeddings, routers, norms, expert bias and quantization auxiliary tensors sit
outside that average. Embeddings/routers/norms retain 16-bit storage, routing
correction biases retain FP32. MTP is absent in this source model.

`prepare.py` checks every source Linear shape against the architecture and
creates the exact recipe plus 250 × 2048 calibration tokens: 150 rows from the
bundled EXL3 corpus mix and 100 rows of deterministic fictional memory-writing
examples (extraction, correction, coding details and Japanese text). This is
an initial structural allocation with mixed synthetic calibration, not a
sensitivity-optimized recipe or a self-sampled task trace.

## Reproduction

The existing `rocm-exl3-r9700-conv` container mounts this checkout at `/src`,
`/home/homelab1/datapool/ai_models` read-only at `/models`, and
`/home/homelab1/datapool/rocm-exl3-rdna2` read-write at `/work`. It exposes one
Radeon AI PRO R9700. `pipeline.py` records each stage and its log separately;
it never publishes unless allocation auditing and all three memory smoke
checks pass. It creates a new public Hub repository and refuses to overwrite
an existing one.

```bash
hf download tencent/Hy-Memory-30B-A3B \
  --revision 64a460a94d5498b7316fb02395c72d01c47a7347 \
  --local-dir /home/homelab1/datapool/ai_models/safetensors/Hy-Memory-30B-A3B-BF16

docker exec -w /src rocm-exl3-r9700-conv python \
  doc/hy_memory_quantization/prepare.py \
  /models/safetensors/Hy-Memory-30B-A3B-BF16 \
  --output /work/runs/hy-memory-30b-a3b-4bit-base --check-shapes

docker exec -w /src rocm-exl3-r9700-conv python convert.py \
  -i /models/safetensors/Hy-Memory-30B-A3B-BF16 \
  -o /work/models/Hy-Memory-30B-A3B-exl3-4.08bpw \
  -w /work/runs/hy-memory-30b-a3b-4bit-base/checkpoint \
  -rcp /work/runs/hy-memory-30b-a3b-4bit-base/recipe-4bit-base.json \
  -hb 8 -cb mul1 -cr 250 -cc 2048 -d 0 -ss 4096 \
  -cd /work/runs/hy-memory-30b-a3b-4bit-base/calibration.safetensors
```

For interrupted conversion use the same container and
`python convert.py -w /work/runs/hy-memory-30b-a3b-4bit-base/checkpoint -r`.
`validate.py` checks all actual trellis bitrates and runs extraction, correction
and abstention tasks using the native chat template. `publish.py` packages the
license, model card, recipe, metadata and validation evidence, uploads on explicit
`--publish`, and checks anonymous remote file sizes and weight LFS SHA-256 hashes.
These functional checks do not measure quality loss relative to BF16.

## Long-run RAM control

The 48-layer conversion's host RSS increased by roughly 2 GB per MoE layer,
reaching 28.33 GB after eight blocks despite allocator trimming. The datapool's
ZFS ARC also held about 63 GB of file cache. The workflow therefore recycles
the converter process every eight modules through its native checkpoints.
The FP32 calibration state, original token rows, tensor recipe and per-module
random seeds survive the restart; no weights are requantized at a different
precision. `--checkpoint_interval 1` ensures the end of every chunk is saved.
The final process runs the remaining norm/head and compiles all saved tensors.

Resume the workflow (conversion through public upload) with:

```bash
python3 doc/hy_memory_quantization/pipeline.py --resume --modules-per-process 8
```

This run used the user service `hy-memory-exl3-4bit-base.service`.
Progress and logs live under
`/home/homelab1/datapool/rocm-exl3-rdna2/runs/hy-memory-30b-a3b-4bit-base/`.
