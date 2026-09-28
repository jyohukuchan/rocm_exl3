# rocm_tools/rdna2 — single-V620 measurement harness (Phase 0-2)

Small, reusable toolset for measuring exllamav3 on one RDNA2 card (gfx1030,
"V620"): a frozen token manifest, teacher-forced top-1 collection for the
exl3 and transformers backends, a strict comparison gate, and a smoke/bench
throughput runner. Nothing here builds the extension, downloads models or
touches GPU state until you run a GPU command inside the container.

**Accuracy criterion (per Phase 0-2 decision):** teacher-forced top-1
agreement over fixed manifest positions — *not* exact generated-token
sequences and *not* numeric `allclose`. A low rate between two artifacts means
two different *executions*; whether that is kernel-path error or weight
difference is answered by the identity lines, not by the rate alone. Full-vocab
KLD is the documented fallback criterion if top-1 turns out too coarse; this
harness stores only argmax IDs (small artifacts) and does not emit KLD yet (see
Limitations).

Three intended comparisons:

| question | reference | candidate |
|---|---|---|
| EXL3 vs unquantized source (independent validation) | `collect_top1 --backend transformers` on the Qwen3-8B BF16 dir (`--dtype auto` keeps the native BF16; BF16 GEMM and full BF16 generation are verified working on this gfx1030 stack) | `--backend exl3` on `/work/models/qwen3-8b-exl3-4bpw` |
| conservative vs optimized execution of the *same* EXL3 checkpoint (kernel-only error) | exl3 run with `EXL3_ROCM_FORCE_TORCH=1` (reconstruct + torch mm reference path) | exl3 default optimized kernels |
| bulk prefill vs autoregressive/decode route | `--execution bulk` | `--execution chunked --chunk-size 1` |

In rows 2-3 the model dir and manifest are identical, so a top-1 difference
**is** kernel-path/cache-path difference — that's the Phase-2 gate. In row 1
the weights genuinely differ (quantization), so the rate is a quality signal,
not an error budget.

**Runtime state carried by the orchestrator (2026-09-28):** the extension
builds/imports now, and **plain default EXL3 (BC attention ON) works**: the
isolated warm plain and fresh-cache plain runs both generated correctly and
exited 0. The earlier "default EXL3 BC attention hangs at module load" finding
was a misattribution — the initial hang coincided with a *concurrent* GPU probe
on the same card. `EXL3_BC_ATTN=0` is therefore a **diagnostic-only** switch
(bisecting the attention path), not a requirement in normal commands; both a
sync-before-load wrapper and the unmodified plain load passed, so nothing here
makes sync-before-load necessary either. The harness captures the EXL3*/EXL3_ROCM_*
environment into every artifact, so the switch set used by a run is auditable
afterwards.

## Ownership

These files are the harness's own. `setup.py` / extension sources belong to
the build implementer; top-level `docs`, environment and downloads belong to
the orchestrator; GPU runtime work belongs to the runtime agent. Nothing
outside this directory is modified.

## Commands (container `rocm-exl3-rdna2`, repo mounted at `/src`, artifacts at `/work`)

All GPU-side commands use the container venv. `--help` for every tool also
works on the host (`python3`, CPU torch or none): GPU/torch imports are lazy.

### 1. Manifest (tokenizer only; seconds)

```bash
/opt/venv/bin/python /src/rocm_tools/rdna2/manifest.py \
    -m /work/models/qwen3-8b-exl3-4bpw \
    -o /work/phase0/manifest_qwen3_8b.json \
    --positions 1024
```

Freezes literal token IDs of the 8 fixed examples, the selected positions,
input hashes, `valid_vocab_size` (from the model's own tokenizer) and a
canonical `manifest_sha256`. Example sources and provenance are listed in
`examples.py`: two excerpts of repo-local `eval/eval_texts/` files (provenance
= repo path + observed content kind; no author/licence claims beyond the repo
file), six original embedded texts (EN tech prose, two Japanese prose, three
Python source). Use the tokenizer of the model family you will measure —
Qwen3-8B and the EXL3 conversion of it share it, so either dir works, but one
manifest must feed every downstream run. Position `p` in a case of length `L`
always means *the next-token distribution after consuming ids[0..p]* (row `p`
of teacher-forced logits, `0 <= p <= L-1`), identically for both backends.
Fails if `--positions` exceeds the examples' token capacity (~2x 1024 with
Qwen3 tokenizers; the tool prints the true capacity when it does not fit).

### 2. Reference collection — transformers backend (unquantized source)

```bash
# native BF16 reference (default: --dtype auto = checkpoint's declared dtype)
/opt/venv/bin/python /src/rocm_tools/rdna2/collect_top1.py \
    --manifest /work/phase0/manifest_qwen3_8b.json \
    --backend transformers -m /work/models/qwen3-8b-bf16 \
    --device cuda:0 --attn-implementation eager \
    -o /work/phase0/top1_ref_transformers.json

# host-RAM variant for references that do not fit the V620 (e.g. a 30B BF16
# source dir the orchestrator provides) -- slower, numerically valid:
/opt/venv/bin/python /src/rocm_tools/rdna2/collect_top1.py \
    --manifest /work/phase0/manifest_qwen3_30b.json \
    --backend transformers -m /work/models/qwen3-30b-a3b-bf16 \
    --device cpu \
    -o /work/phase0/top1_ref_30b_cpu.json
```

`--dtype auto` (default) keeps the checkpoint's declared `torch_dtype` (so a
BF16 source gives a native BF16 reference); `fp16|bf16|fp32` force a cast.
Requested, resolved, declared and effective dtypes are all recorded, plus the
attention implementation **actually selected** (read back from
`model.config._attn_implementation`; `from_pretrained` is passed the real
kwargs — `dtype=`, `torch_dtype=` fallback — since signature inspection would
hide `**kwargs` options). Teacher forcing = one forward of the literal IDs per
case with `use_cache=False`; no continuation is generated. Runs under
`torch.inference_mode` (explicit here; the exl3 `Model.forward` is decorated
with it). `--device cpu` is only valid for this backend.

### 3. Candidate collection — exl3 backend

```bash
# bulk prefill route (cache-free whole-sequence forward, eval/ppl.py pattern)
/opt/venv/bin/python /src/rocm_tools/rdna2/collect_top1.py \
    --manifest /work/phase0/manifest_qwen3_8b.json \
    --backend exl3 -m /work/models/qwen3-8b-exl3-4bpw \
    --device cuda:0 --execution bulk --load-max-chunk-size 2048 \
    -o /work/phase0/top1_exl3_bulk.json

# autoregressive / decode route (Model.forward with params/cache, chunk=1)
/opt/venv/bin/python /src/rocm_tools/rdna2/collect_top1.py \
    --manifest /work/phase0/manifest_qwen3_8b.json \
    --backend exl3 -m /work/models/qwen3-8b-exl3-4bpw \
    --device cuda:0 --execution chunked --chunk-size 1 \
    -o /work/phase0/top1_exl3_chunked1.json

# conservative reference execution of the same checkpoint (bisect handle)
EXL3_ROCM_FORCE_TORCH=1 /opt/venv/bin/python /src/rocm_tools/rdna2/collect_top1.py \
    --manifest /work/phase0/manifest_qwen3_8b.json \
    --backend exl3 -m /work/models/qwen3-8b-exl3-4bpw \
    --device cuda:0 --execution bulk \
    -o /work/phase0/top1_exl3_ref_torch.json
```

The switch set actually in effect lands in the artifact's env block, so
"which execution produced which file" survives to `compare`. Chunked C=1 does
`L` single-token cached forwards per case: slow by design. `--limit-cases`
localizes failures, but `total_positions` in the artifact **stays the full
manifest count**, so any subset run is flagged incomplete and comparison
refuses it — even a pair of identically truncated artifacts. Cleanup
(`model.unload()` for exl3; reference release for transformers) runs in a
`finally`; a teardown failure is printed, recorded in the artifact's error
list and exits nonzero — never swallowed, and no `os._exit`.

### 4. Compare (host or container; no torch needed)

```bash
/opt/venv/bin/python /src/rocm_tools/rdna2/compare_top1.py \
    --reference /work/phase0/top1_ref_transformers.json \
    --candidate /work/phase0/top1_exl3_bulk.json \
    --min-agreement 0.995 --json-out /work/phase0/cmp_ref_vs_bulk.json

/opt/venv/bin/python /src/rocm_tools/rdna2/compare_top1.py \
    --reference /work/phase0/top1_exl3_ref_torch.json \
    --candidate /work/phase0/top1_exl3_bulk.json

/opt/venv/bin/python /src/rocm_tools/rdna2/compare_top1.py \
    --reference /work/phase0/top1_exl3_bulk.json \
    --candidate /work/phase0/top1_exl3_chunked1.json
```

Prints both model roots, config fingerprints, execution modes, git commits and
env switches **before** the agreement rate. Exit codes: `0` valid comparison
(≥ threshold if given), `1` rejected (different manifest/positions/vocab,
incomplete or truncated results, error records, non-finite logit flags,
out-of-vocab IDs), `2` agreement below `--min-agreement`. Recommended
thresholds (orchestrator to confirm on first runs): same-checkpoint
conservative-vs-optimized and bulk-vs-chunked rows should be ≈1.0 (any real
divergence is kernel-path error); the transformers-vs-exl3 row is a
quantization-quality expectation, typically 0.97-0.995 at 4 bpw on Qwen3-8B.

### 5. Smoke + benchmark (exl3, gated to the single V620)

```bash
# smoke: one run per job on the visible GPU; early EOS allowed but labeled
/opt/venv/bin/python /src/rocm_tools/rdna2/bench.py \
    -m /work/models/qwen3-8b-exl3-4bpw --mode smoke \
    --json-out /work/phase0/bench_smoke.json

# bench: warmup 1 + 5 timed repeats, prefill+decode at 512/2048/8192, out 256
/opt/venv/bin/python /src/rocm_tools/rdna2/bench.py \
    -m /work/models/qwen3-8b-exl3-4bpw --mode bench \
    --contexts 512 2048 8192 --new-tokens 256 --repeats 5 --warmup 1 \
    --max-chunk-size 2048 --seed 1234 \
    --json-out /work/phase0/bench_qwen3_8b_exl3.json
```

Before loading anything the tool fails (nonzero) unless: `torch.version.hip`
is set, **exactly one** GPU is visible, `gcnArchName == gfx1030`
(`--expect-arch` to override for another RDNA2 card), and `--device cuda:0`.
Speculative decoding is structurally disabled and asserted (no draft model,
`ngram_match_min=0`, `num_draft_tokens==0`). Every job uses a fresh random
prompt of exactly `ctx` in-vocab IDs from the seeded CPU rng; the run fails if
a job reports `prompt_tokens != ctx`, `cached_tokens != 0` (prefix-cache hit),
`new_tokens` short of the request, or a missing completion — bench mode also
requires `eos_reason == "max_new_tokens"` (stop tokens suppressed to the
requested length via `min_new_tokens`, the public Job option that supports the
requested length). Smoke allows EOS and labels the row. `model.unload()` runs
in a `finally` and its failure is recorded and forces nonzero exit. Metrics:
TTFT = job `time_prefill`; TPOT = `time_generate/(new_tokens-1)`; prefill t/s,
decode t/s, e2e t/s (formulas embedded in the JSON); per-run peak
`torch.cuda.max_memory_allocated`; medians + spread with the 5% noise-floor
flag (conventions from `rocm_tools/bench_model.py`). Per-token ITL (decode
rows): `generator.iterate()` completions are timestamped with
`time.perf_counter` — deliberately **no extra per-token
`torch.cuda.synchronize`**, which would itself perturb the latency — and the
increase of `job.new_tokens` (batch=1, spec decode asserted off ⇒ plain AR
advances by +1 per round) gives the raw intervals. The first token is
excluded (its dt carries prefill/TTFT), so N generated tokens ⇒ N-1 positive
samples. Each decode run stores `itl_samples_ms` plus nearest-rank
`itl_p50_ms`/`itl_p95_ms`; the summary merges the raw samples of all timed
runs into `itl_ms` (one empirical distribution — distinct from the per-job
mean `tpot_ms`). A multi-token increment is impossible in plain AR; it is
recorded in `itl_bursts` and fails the row instead of being averaged into
fake per-token samples. All rows/summary/params/
seed/commit/env go to `--json-out`, written and fsynced before **normal**
process exit — no `os._exit`. If native teardown still segfaults on some
builds (this fork documents such crashes), the JSON file and the RESULT line
were already emitted: judge by the artifacts.

## CPU tests

```bash
python3 -m unittest discover -s /src/rocm_tools/rdna2/tests -v
```

Tests in the harness's two CPU suites — `test_harness_cpu.py` (28) and
`test_bench_itl_cpu.py` (14), stdlib only (`unittest`, `tempfile`): compare
accepts a well-formed pair and computes hand-checked agreement (10/12 with
two flips localized to
positions 2 and 6); rejects mismatched manifests/positions/vocab, incomplete,
error-status, non-finite, out-of-vocab, missing-case and **identically
truncated** artifacts (the `--limit-cases` regression: `total_positions` is
full-manifest-relative, and a subset pair is refused with or without
`--manifest`); threshold exit codes 0/1/2; manifest digest tamper detection;
position bounds/semantics validation; deterministic position selection and
allocation; `--dtype auto` resolution (native BF16 kept, explicit wins,
fallbacks labeled); bench's nearest-rank `percentile()` against hand-checked
known samples (including the ceil-boundary ranks), and `token_intervals()` on
synthetic event streams (first-token anchor excluded ⇒ N-1 samples for plain
AR, non-positive dt dropped, multi-token jumps reported as bursts that
contribute no averaged sample).

## Limitations / to validate at runtime (GPU, by orchestrator/runtime agent)

1. **Nothing GPU has been executed by these tools yet.** The exl3 chunked
   params mirror `eval/perf.py` (`attn_mode: flash_attn`, `past_len`,
   `batch_shape`) and the MTP full-chain prefill forward (`job.py`); bulk
   mirrors `eval/ppl.py`. The harness shape-checks logits (`(1, w, >=vocab)`,
   row i = global s+i mapping in chunked mode) and refuses mismatches — but
   first container runs must confirm those invariants on the RDNA2 build.
2. Manifest token capacity (~2x1024) was estimated without running the
   tokenizer; `manifest.py` hard-fails (with the true capacity) if a run's
   tokenizer makes 1024 positions not fit, so it is safe by refusal. Both
   source BF16 CPU references now complete with 1024 valid positions.
3. Top-1 fp16/bf16 ties (possible, rare) resolve by argmax's lowest-index rule
   on both backends; a spurious disagreement would show as a handful of flipped
   positions. `--dtype fp32` on the transformers side, or KLD mode, resolves
   it.
4. KLD / full-logit artifacts are intentionally out of scope (artifact
   volume); extend `collect_top1` to dump fp16 logit rows and reuse
   `exllamav3.util.measures.compute_kl_div` semantics if needed later.
5. Chunked C=1 over a 1024-position manifest is O(total tokens) forwards —
   slow by nature; use `--limit-cases` for localization (comparison then
   correctly refuses the partial file).
6. Recurrent/hybrid models are refused by design (dense-attention Qwen3 only).
7. Bench prompts are random in-vocab IDs (cold-cache throughput proxy, the
   `bench_model.py` convention), not natural text; quality is measured by the
   top-1 tools, not the bench.
8. Troubleshooting (updated 2026-09-28): the default EXL3 BC attention is
   **not** blocked on this stack — isolated warm plain and fresh-cache plain
   runs generate correctly and exit 0. The original "hangs at module load"
   observation coincided with a *concurrent GPU probe* on the same card, so:
   run exactly one GPU workload at a time, and keep cold JIT/module-load time
   out of the numbers (untimed warmup runs exist for this; do not fold
   first-touch JIT into timed medians). `EXL3_BC_ATTN=0` remains available as
   a *diagnostic* for bisecting the attention path under contention — it is
   not part of the normal command set. A sync-before-load wrapper and the
   unmodified plain load both passed, so no command here requires syncing
   before model load.
