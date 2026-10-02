# R9700 conversion optimization — 2026-10-02

Qwen3.5-2B-BF16 → EXL3 4bpw, one R9700 (`gfx1201`), PyTorch
2.12.0+ROCm7.2. The existing dense K4 kernel is retained. Two optional
conversion changes accelerate calibration:

- `--hessian_fp16`: multiply the already-FP16 activation matrices with FP32
  output, then add to the FP32 Hessian. This changes rounding and quantized
  weights. Every calibration row/token remains present. Wide matrices use
  output-row chunks to cap the additional FP32 product tensor at 256 MiB.
- `--fast_head_capture`: collect the terminal Linear head's input Hessian but
  omit logits for rows beyond the first five reference rows. The input Hessian
  and sample count remain bit-identical in the head probe.

Both are **off by default** and persisted in checkpoint arguments. They are
wired into serial and multi-GPU capture; hardware validation here is single
R9700. The live V620 inference service is unaffected.

## Timing

All full conversions retain 250 × 2048 calibration tokens, 4bit text/MTP,
6bit head/vision, and the default output scales/checkpoint interval.

| Conversion | Full wall time |
|---|---:|
| Original plain K4 | 1046.6 s |
| Existing dense K4, bit-identical model | 948.5 s |
| Dense K4 + both new options | **773.044 s** |

The new full run is **18.5% shorter** than the previous dense-K4 run and 26.1%
shorter than the original run. These are individual end-to-end measurements;
the original repeat took 1212.8 s, so wall-time variation must not be mistaken
for a precise universal speedup.

A subsequent default-setting partial conversion covered embedding plus the
first four blocks. Matching those five modules against the new full run gives:

| Matched modules | Default capture | New capture |
|---|---:|---:|
| Capture wall time | 39.914 s | 10.390 s |
| Total module wall time | 96.55 s | 69.10 s |

Across all 24 text blocks, total module wall time was 534.01 → 366.48 s.
The isolated `2048 × 6144` activation Hessian GEMM took 25.392 → 1.252 ms
(20.3×); relative Frobenius difference was `2.63e-6`.

The 6bit head still takes substantial time: capture 2.443 s, quantization
126.111 s, and post-conversion forward/error measurement 31.766 s. The dense
K6 quantizer port and head error measurement are further candidates. No speedup
for either is claimed here.

Calibration batch2 was rejected: the first partial screen slowed from
79.211 to 88.362 s. Moving concatenation onto the GPU did not improve capture
or state advancement. That implementation is not included.

## Quality and adoption status

The model SHA256 changes from
`2b2d57a192539120a0e83580735197e9d6abd8bfc7333a0f1690e6e707ac4d12`
to `2efab08c82331ae38b5dfe305ef066ede955abf75a34267dfd4828df0c286ec7`.
The default partial conversion reproduces the original five module files;
the H16 partial conversion reproduces the corresponding five files from the
new full conversion.

The primary quality probe uses six fixed Japanese/English/code cases, 895
teacher-forced positions, and all 248077 valid vocabulary entries. BF16
Transformers is the reference; both EXL3 models use the same native library.
All inputs and tokenizer files are aligned. KL and next-token NLL are computed
in FP64 on CPU, in nats.

| Primary metric | Original EXL3 | New EXL3 |
|---|---:|---:|
| KL(BF16 ∥ EXL3), position-weighted mean | 0.01663361 | 0.01651122 |
| Next-token NLL | 2.35063103 | 2.34890717 |

**The primary per-case screen did not pass.** `ja-cache` increased from
0.02130518 to 0.02740947 KL (+28.7%), exceeding its 20% per-case allowance.
Three positions account for about 86% of that case's total KL increase. Its
NLL improved from 3.61836 to 3.59731, but that does not cancel the KL result.
The global mean and the other five case screens passed. The screening limits
are max(0.001 nats, 10%) for the global KL increase, and max(0.002 nats, 20%)
per case; the raw failed result is retained.

Six additional independent cases (four Japanese, two coding; 2723 positions)
were then evaluated without retuning the candidate. Their mean KL was
0.02388213 → 0.02586717 (+8.3%), and NLL was 2.73814 → 2.74595 (+0.3%).
All six additional case screens and the additional global screen passed.
This follow-up characterizes the initial observation; it does not replace the
failed primary screen.

Across both sets (12 cases, 3618 positions), mean KL is 0.02208903 → 0.02355275
(+6.6%), and NLL is 2.64228 → 2.64773 (+0.21%). By category:

| Category | Positions | Original KL | New KL |
|---|---:|---:|---:|
| Japanese | 2181 | 0.02848520 | 0.03100001 |
| English | 273 | 0.01875133 | 0.01900627 |
| Code | 1164 | 0.01088727 | 0.01066504 |

H16 is therefore an **opt-in experiment with model/data-dependent numerical
changes**, not a generally quality-equivalent default. Short teacher-forced
checks do not establish long-context, vision, MTP or task-level quality.

## Reproduction

Inside the existing `rocm-exl3-r9700` container, `/src` is the repository and
`/work` is the benchmark/model volume. Use the same dense-K4 native library for
all comparisons:

```bash
export PYTHONPATH=/work/lib-r9700-opt:/src
cd /src
python convert.py -i /work/models/qwen3.5-2b-bf16 \
  -w /work/build-qwen35-h16-head -o /work/models/qwen35-h16-head \
  -b 4 --hessian_fp16 --fast_head_capture
```

Omit the two options for a baseline. Use `--max_module 4` with fresh work/output
directories for a short timing/determinism screen; it does not produce a full
model suitable for the quality collection below.

```bash
python rocm_tools/benchmark_conversion_capture.py \
  --model /work/models/qwen3.5-2b-bf16 --output /work/capture-probe.json

python rocm_tools/conversion_quality.py freeze \
  --model /work/models/qwen3.5-2b-bf16 \
  --prompts benchmarks/2026-10-02/conversion-quality-prompts.json \
  --manifest /work/quality-manifest.json --tokens 192

python rocm_tools/conversion_quality.py collect \
  --manifest /work/quality-manifest.json --backend transformers \
  --model /work/models/qwen3.5-2b-bf16 --output /work/quality-reference
python rocm_tools/conversion_quality.py collect \
  --manifest /work/quality-manifest.json --backend exl3 \
  --model /work/models/qwen3.5-2b-exl3-4bpw-r9700 --output /work/quality-baseline
python rocm_tools/conversion_quality.py collect \
  --manifest /work/quality-manifest.json --backend exl3 \
  --model /work/models/qwen35-h16-head --output /work/quality-candidate
python rocm_tools/conversion_quality.py compare \
  --manifest /work/quality-manifest.json --reference /work/quality-reference \
  --baseline /work/quality-baseline --candidate /work/quality-candidate \
  --output /work/quality.json
```

`compare` writes the result even when a gate fails, then exits 2. GPU jobs must
run serially. Frozen token manifests, prompt texts, quality results, phase
times, capture probes and compressed conversion logs are under
[`benchmarks/2026-10-02`](../benchmarks/2026-10-02/).
For the independent follow-up, freeze `conversion-quality-holdout-prompts.json`
with `--tokens 512`, use separate collection/output directories, and retain the
same baseline and candidate model files.
The primary reference/baseline collection predates runtime-fingerprint fields
in the collector; their tokenizer and native-library provenance was checked
after collection and is labeled as such in the metadata.
