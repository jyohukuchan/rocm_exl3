
# <img src="doc/cat.png" width="40"> ExLlamaV3 — ROCm / RDNA fork (experimental V620 branch)

This repository is an **experimental branch of the ROCm fork of
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp. Ancestry:

- Upstream base: [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) (tracking v1.5.0).
- Direct parent: [`CarouselAether/rocm_exl3`](https://github.com/CarouselAether/rocm_exl3)
  at `dd7a670065f37943f09a5eeb53818f38e9751472` — the RDNA2/3/4 port this fork was branched from.
- This repo: [`jyohukuchan/rocm_exl3`](https://github.com/jyohukuchan/rocm_exl3) — a
  **gfx1030 (Radeon Pro V620) first** research fork. Two V620s, tensor-parallel loading, the
  Qwen3.8-Flash-Next architecture (QSA sparse attention + PLE n-gram embeddings + 512-expert
  MoE) and its packed MTP draft head are what has been driven hardest and measured here.

The Python package is still named `exllamav3`, so it is a drop-in for code that imports it.
The MIT licence and all vendor notices from the parent and upstream are preserved.

**Read the scope honestly:** this is a *fork of a fork*. It makes no universal AMD or CUDA
support claims. Sections inherited from the parent/upstream are labelled where they begin;
they describe generic ExLlamaV3 and CUDA behaviour that has **not** been revalidated on this
branch.

### Supported / verified matrix (this branch)

| Capability | Hardware | Status on this fork | Evidence |
|---|---|---|---|
| Single-GPU inference (bench + quality gates) | 1× V620 (gfx1030) | **Validated** — Qwen3-8B EXL3 4bpw, Qwen3-30B-A3B EXL3 3bpw | [doc/rdna2_phase2_results.md](doc/rdna2_phase2_results.md) |
| Layer-split load (`use_per_device`) | 2× V620 | **Validated** (batch 1, placement-audited) | [doc/v620_pair_results.md](doc/v620_pair_results.md) |
| Tensor-parallel load + RCCL collectives | 2× V620 | **Validated** — Qwen3.8-Flash-Next 3.05bpw + packed MTP3, K5/V4 KV, Engram CPU table, batches 1–4 | [doc/qwen38_v620_context_batch.md](doc/qwen38_v620_context_batch.md), [doc/v620_tp_decode_optimization.md](doc/v620_tp_decode_optimization.md) |
| MTP draft windows 1–4 | 2× V620, TP2 | **Supported and exercised**; deeper drafts are *not* universally faster — acceptance is workload-dependent (screening table in the context/batch report) | [doc/qwen38_v620_context_batch.md](doc/qwen38_v620_context_batch.md) |
| Common multi-token MoE route | RDNA2/3 native path; measured on 2× V620 | **Default enabled**, 2–24 rows with bounded chunks; grouped MTP/batch decode, gateless experts, and smaller R1 activation. TP2 batch4 code decode 55.83→73.09 aggregate tok/s (+30.9%). | [doc/v620_moe_common_paths.md](doc/v620_moe_common_paths.md) |
| Engram single-owner CPU-RAM table + optional `EXL3_NGRAM_MLOCK=1` | 2× V620, TP2 | **Validated** — one 32,640,156,672-byte table, one owning rank, mlock/mincore residency audits | [doc/qwen38_v620_tp_config.json](doc/qwen38_v620_tp_config.json) |
| K5/V4 quantized KV cache (QSA layers) | 2× V620, TP2 | **Validated** — default for the TP2 measurements; recurrent states keep their original FP32/BF16 types | [doc/v620_tp_decode_optimization.md](doc/v620_tp_decode_optimization.md) |
| Max generated context | 2× V620, TP2, **batch 1 only** | 261,632 input + 256 output completed with fixed MTP4. **Batch 2–4 maximums are not established** — long-context tests were deferred by the operator | [doc/qwen38_v620_context_batch.md](doc/qwen38_v620_context_batch.md) |
| R9700 (gfx1201 / RDNA4) | 1× R9700 | **Tested only through documented comparison adapters/workarounds** (64 KiB LDS build, MLP range-balance adapter, MoE reconstruct adapter). Not general RDNA4 support | [doc/r9700_vs_v620.md](doc/r9700_vs_v620.md) |
| RDNA3 / RDNA3.5 (`gfx1100`…`gfx1151`) | — | **Inherited from the parent fork's gfx1151 validation; not independently rerun on this branch** after the TP / Qwen3.8 / kernel changes below. The parent's claims stand as the parent's, not ours | [doc/fork_changes.md](doc/fork_changes.md) |
| CUDA path | NVIDIA | Upstream code; untouched by design except one shared-header race fix (see fork changes). Not tested here | — |

### What changed relative to the fork parent

The headline: **the parent's two "not available / not modified" statements no longer hold on
this branch.**

- **Tensor-parallel is available on the ROCm path.** The TP2 executions behind the benchmark
  reports go through `exllamav3/model/model_tp*.py` with an RCCL backend
  (`model_tp_rccl.py`), validated on 2× V620. These are not the upstream CUDA `parallel/`
  kernels; other rank counts, GPUs and models are untested in TP mode.
- **Shared C++/CUDA code is now modified.** One upstream header,
  `exllamav3/exllamav3_ext/reduction.cuh`, carries a small race fix (single writer for the
  shared-memory broadcast slot), and the RDNA2 native siblings under
  `exllamav3/exllamav3_ext/rocm/` changed materially (gfx1030 SIMT/fdot2 fallback in
  `rdna_wmma.hip.h`, padded-row bounds fix in `exl3_gemv_multirow_rdna.hip`).
- On top of that, this branch touches dozens of shared Python modules (TP loading,
  Qwen3.8-Flash-Next architecture pieces — PLE/n-gram, QSA indexer, GDN, block-sparse MLP —
  `rocm_py` steering, vendored FLA dispatch) and adds the `rocm_tools/rdna2` measurement
  harness with CPU test suites.

[doc/fork_changes.md](doc/fork_changes.md) is the rough change list; this README stays a
summary, not a git diary.

*How the port works (inherited architecture):* CUDA kernels that cannot compile for RDNA are
replaced by hand-written HIP siblings under `exllamav3/exllamav3_ext/rocm/`, reached through
a compat shim; Python divergences live in `exllamav3/rocm_py/` and are applied as
monkeypatches at import. On this branch `rocm/` grew the gfx1030 SIMT fallback and the
multirow bounds fix, and the shared engine code changed as listed above.

### Requirements

| | |
|---|---|
| ROCm | **7.2.4 or newer** — `setup.py` hard-fails below this. The *measured* stack used a custom ROCm 7.14 HIP/ROCr SDK (`hipcc` 7.14.60850) under `/opt/rocm/core-7.14` with `torch 2.12.0+rocm7.2`. That exact combination produced the numbers below; it is not a distributable prebuilt image, and passing the `>= 7.2.4` check is not a claim that every newer version was tested. Point `ROCM_PATH`/`PATH`/`LD_LIBRARY_PATH` at **your** install |
| GPU | Primary target: RDNA2 `gfx1030` (V620). See the matrix above. RDNA3/3.5 (`gfx1100`–`gfx1151`): inherited parent coverage, not rerun here. RDNA4 (`gfx1200`/`gfx1201`): R9700 measured only with the documented workarounds — not general support |
| Python | 3.10+ (measured on 3.12) |
| Torch | ROCm build from `download.pytorch.org/whl/rocmX.Y` (measured: 2.12.0+rocm7.2, triton-rocm 3.7.0) |

You do **not** need FlashAttention. Upstream uses Triton paged attention, so the FA2
dependency that earlier ROCm forks required is gone.

### Quick start (build for gfx1030)

```sh
git clone https://github.com/jyohukuchan/rocm_exl3
cd rocm_exl3

python3 -m venv .venv
source .venv/bin/activate

# Set this to your installed ROCm SDK (the measured SDK was /opt/rocm/core-7.14).
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
hipcc --version

python -m pip install --index-url https://download.pytorch.org/whl/rocm7.2 \
  'torch==2.12.0+rocm7.2'
python -m pip install -r requirements_rocm.txt -c benchmarks/2026-09-30/constraints.txt
EXL3_BACKEND=rocm PYTORCH_ROCM_ARCH=gfx1030 MAX_JOBS=12 \
  python -m pip install --no-build-isolation .
```

`--no-build-isolation` is required, not optional: pip otherwise builds in an isolated
environment with no torch in it, and a torch C++ extension has to be compiled against the
same torch it will run against. Set `PYTORCH_ROCM_ARCH` (or `GPU_ARCHS`) explicitly so the
build does not depend on autodetection; `MAX_JOBS` limits the parallel `hipcc` jobs on
low-RAM machines. There is no prebuilt wheel for this fork, and the custom 7.14 SDK the
measurements ran on is not part of any one-command turnkey install.

**Full reproduction** — exact environment, runtime switches, process limits, power helper,
frozen prompts and the TP2 batch-4 benchmark command — is in
**[doc/reproduce_v620.md](doc/reproduce_v620.md)**.

### Dated benchmark overview (2026-09-30)

The public bundle at **[benchmarks/2026-09-30/](benchmarks/2026-09-30/README.md)**
(`results.json`, `environment.json`, frozen prompt files, sanitized per-run reports under
`reports/`) contains the validated V620×2 TP2 measurements on Qwen3.8-Flash-Next EXL3
3.05bpw with the original distributed pack plus original packed 3-bit MTP weights, K5/V4 KV,
Engram as a single CPU-RAM table, input 8192 / output 256 tokens per sequence. Each
language ran warm 1 + timed 2 groups (batch 1–3) and warm 1 + timed 3 groups (batch-4
confirmation). Full-span aggregate decode tok/s (Japanese / code):

| Batch | Draft | JA decode | code decode |
|---|---|---:|---:|
| 1 | dynamic MTP max 4 | 38.88 | 50.52 |
| 2 | fixed MTP1 | 55.85 | 56.68 |
| 3 | fixed MTP1 | 66.25 | 73.95 |
| 4 | fixed MTP1 | 70.56 | 66.51 |

Batch-4 aggregate prefill (conservative: total input tokens over the *latest* first
delivery): **471.80 / 459.68** tok/s (JA / code). Batch-4 *common-window* aggregate decode —
the interval where every job is still generating, excluding later queue drain — was
**76.29 / 85.80** tok/s, while the three timed groups' full-span rates ranged 68.46–73.24
(JA) and 63.98–72.04 (code). MTP acceptance at batch 4: 70.6% (JA) / 92.2% (code);
acceptance is workload-dependent and does not transfer to other prompts.

Reading notes, also recorded in the bundle: the full-span decode metric counts delivered
tokens from the first delivery to the last, including final drain delay — it is ongoing
generation plus drain, and never a sum of independently measured per-job speeds. Batch 1's single long-context run (261,632 + 256, fixed MTP4) completed;
batch 2–4 long-context maximums are not established.

### Known caveats on this branch

- **Residual numerical differences exist across execution paths.** Cold vs prefix-reused
  executions have pre-existing cross-path numerical differences (distinct from the batched
  recurrent-prune change, which was shown to leave same-input outputs identical). Exact
  generated-token equality across all paths is *not* claimed.
- **MTP acceptance is workload-dependent.** Draft windows 1–4 are supported; the screening
  table shows the cases where more draft tokens make things *slower*.
- **Context ceiling: only batch 1 at 261,632 input + 256 output is proven.** Batch 2–4
  long-context tests were deferred; allocation-only probes are not validated limits.
- **The benchmarking engines and harness tools were validated on hardware** (frozen prompts,
  placement / cache / RAM audits, repeated timed groups). The HTTP runtime was also
  validated on 2026-10-01 with TP2/RCCL, the original 3-bit MTP head, K5/V4, and
  one locked Engram RAM table at a 32,768-token API context. See the
  [server guide](rocm_tools/exl3_server/README.md).
- Vision/multimodal remains untested on ROCm (inherited status).
- Full target-model quantization is not validated here. A limited official-source MTP
  3/5-bit quantization comparison was performed on V620; see [the precision study](doc/qwen38_v620_mtp_precision.md). The main inference benchmarks use publicly distributed EXL3 checkpoints. The MIT licence covers the code, **not** the model
  weights.

### Additional inherited limitations

- **Small-row MoE uses the common grouped `exl3_mgemm` route on gfx10/gfx11**,
  enabled by default. `EXL3_ROCM_MOE_MGEMM_MAX_ROWS` accepts 8–24 (default 24);
  row-aligned chunks respect the 128-slot native bound. R1 keeps the compact
  per-token fallback. `EXL3_ROCM_MOE_MULTI_TOKEN=0` disables grouping.
  See [the implementation and representative measurements](doc/v620_moe_common_paths.md).
- **The one-launch sliced Q/K/V bundle is off by default** (`EXL3_ROCM_QKV_SLICE=1` to
  enable): ported but unvalidated on RDNA.
- **RDNA4 (gfx1200/gfx1201) runs MoE through the per-expert path**: the fused kernel's WMMA
  has no gfx12 encoding. The R9700 numbers needed explicit comparison adapters
  ([doc/r9700_vs_v620.md](doc/r9700_vs_v620.md)); never claim general unmodified RDNA4
  support from them.
- **MoE 32/64-row tiles fall back to the 16-row kernel** (same numerics; slower mul1 MoE
  prefill).
- **The batched expert-reconstruct tier is off by default** (`EXL3_ROCM_BATCH_RECON=1`).
  Batched recurrent checkpoint pruning is also opt-in (`EXL3_BATCH_RECURRENT_PRUNE=1`).
- **fp16-accumulate `hgemm` and the sm_120 quantizer specialisations are CUDA-only.**
- **The int8-activation GEMV is not ported** (disabled stub).
- **HIP graph capture is off by default** (`EXL3_ROCM_HIP_GRAPHS=1` re-enables; known
  corruption/hang on ROCm 7.2.x).
- **`EXL3_NGRAM_MLOCK=1` needs a ≥ 34 GiB per-process `RLIMIT_MEMLOCK`** and fails
  explicitly if locking cannot be granted — see [doc/reproduce_v620.md](doc/reproduce_v620.md)
  for how to raise the limit without running inference as root.
- Kernel behaviour can be bisected at runtime with the `EXL3_ROCM_*` environment switches —
  see `exllamav3/rocm_py/__init__.py`, whose module docstring lists each one and why it
  exists.

### Server

The bundled `rocm_tools/exl3_server` serves OpenAI Chat/Text Completions and a
llama.cpp-style completion API. It provides structured function calls, incremental
SSE arguments, preserved tool history, thinking controls, and JSON Schema generation
constraints. Dependencies are in `requirements_rocm.txt`.

```sh
python -m rocm_tools.exl3_server.server -m ~/models/<model>-exl3 -cs 32768
# serves on http://127.0.0.1:3953
```

The V620×2 HTTP path has been validated with TP2/RCCL, K5/V4, the original MTP
head, Engram in one mlocked RAM table, and the batch-one power policy. Context and
output limits are exposed through `/v1/models` and `/props`. The HTTP integration
now allocates a 786,432-token (768Ki) context with a 32,768-token output limit. The original
OpenCode coding sample used a 32K context; the larger cache allocation and later
integration checks are described in the API guide.

See the [API/OpenCode guide](rocm_tools/exl3_server/README.md) for endpoints,
configuration, supported schemas, and the V620 launch commands. A portable
[OpenCode v2 configuration](examples/opencode.jsonc) selects the local provider.
The [2026-10-01 integration report](doc/opencode_api_validation.md) includes
real tool roundtrip verification and an [OpenCode-generated coding sample](examples/opencode_lru/README.md).
The [OpenCode metrics and Goal setup](rocm_tools/opencode/README.md) adds native
prefill/generation averages to the terminal footer and persistent `/goal` execution.
The [2026-10-01 QSA request transition fix](doc/qsa_request_transition_fix.md)
addresses GPU faults after a long conversation followed by a short title request.
Static MoE expert storage permutations can be loaded with `--tp-expert-order`.
The [2026-10-02 expert-placement experiment](doc/expert_placement.md) describes
Japanese coding/chat profiles, correctness checks, and decode comparisons.

#### All flags

Every flag has a short and a long form; the short form is shown. Run `server.py -h` for the
live list.

**Model loading**

| flag | what it does |
|---|---|
| `-m DIR` | model directory (required) |
| `-gs GB[,GB...]` | max VRAM to use per device, in GB |
| `-lm` | print loader metrics |
| `-or FILE` | tensor override spec (YAML) |
| `-tp` | load in tensor-parallel mode (multi-GPU); respects `-gs` where it can |
| `-tpb native\|nccl` | tensor-parallel backend, default `native` |
| `-tp_attn N`, `-tp_mlp N`, `-tp_moe N`, `-tp_linear N`, `-tp_linear_attn N` | (TP) cap the parallelism of that layer class |
| `-tp_moe_ts` | (TP) tensor-split MoE layers instead of expert parallelism |
| `-swa_full` | use a full cache for sliding-window layers instead of recurrent mode with snapshots |
| `-ambs N` | max batch size to account for when autosplitting, default 4 |
| `-chunk_size N` | max prefill chunk size |
| `-lv` | verbose loading |
| `-asnf` | skip the forward pass during autosplit (debug) |
| `-layer_map SPEC` | RYS layer map, e.g. `0..15,11..31` repeats layers 11-15 once |

**MoE on the CPU** (experimental; layer-split mode only; needs mul1-codebook experts)

| flag | what it does |
|---|---|
| `-mcl N` | run the routed experts of the first N block-sparse MoE layers on the CPU, weights in system RAM |
| `-mcs N` | per-layer split: run the tail N routed experts of every eligible MoE layer on the CPU, overlapped with the GPU experts; dynamic hot/cold placement is on (`EXL3_MOE_CPU_SWAP=0` for static). Mutually exclusive with `-mcl` |
| `-mct N` | worker threads for the two above (default `EXL3_MOE_CPU_THREADS`, else half the cores) |
| `-ngr` | load a PLE model's n-gram embedding table fully into RAM (tens of GB) instead of streaming rows from disk per forward, e.g. Qwen3.8-Flash-Next |

**KV cache**

| flag | what it does |
|---|---|
| `-cs N` | total cache size in tokens. **Default = the model's max context**; long-context models advertise 256K-1M, so pass `-cs` (e.g. `-cs 32768`) or `-cq` to keep the cache sane |
| `-cq BITS` or `-cq K,V` | quantized cache, one bit width for both or separate K and V widths |
| `-cca A` | compand `a` value for the simulated cache, default 0 |
| `-ccs GB` | CPU second-tier cache size, GB |
| `-rcs GB` | recurrent-state second-tier cache size, GB |

**Speculative decoding**

| flag | what it does |
|---|---|
| `-dm DIR` | separate draft model, like llama.cpp's `--model-draft`; DFlash / EAGLE-3-style drafters load directly |
| `-mtp` | draft with the model's own MTP head (DeepSeek V4, Qwen3.8-Flash-Next, ...); not with `-dm` |
| `-ndt N` | draft tokens per step (default: the draft model's own default, else 4) |
| `-ngram N` | n-gram drafting from repeats already in the context, minimum match length N; no extra model. `-ngram 2` is the cheap default |
| `-dds` | dynamic draft length: skip or shorten drafting while acceptance is low; `-ndt` becomes the ceiling |
| `-dc X` | confidence target for dynamic draft truncation, default 0.4 |
| `-dmcl N` | like `-mcl` for the draft model or MTP head (experimental) |

Draft acceptance is printed per request in the server log and returned in the native
`timings` as `draft_n` / `draft_n_accepted`.

**Sampling defaults** (per-request values override these)

| flag | what it does |
|---|---|
| `-temp X` | temperature, default 0.8 |
| `-temp_first` | apply temperature before truncation |
| `-repp X` | HF-style repetition penalty, 1 disables |
| `-presp X`, `-freqp X` | presence / frequency penalty, 0 disables |
| `-penr N` | range in tokens the penalties look back over, default 1024 |
| `-minp X` | min-P truncation, default 0.08, 0 disables |
| `-topk N` | top-K truncation, 0 disables |
| `-topp X` | top-P truncation, 1 disables |
| `-adaptive_target X`, `-adaptive_decay X` | Adaptive-P target (1 disables) and decay |
| `-xtcp X`, `-xtct X` | XTC probability (0 disables) and threshold (default 0.1) |
| `-drym X`, `-dryb X`, `-dryal N`, `-dryln N` | DRY multiplier (0 disables), base (1.75), allowed repeat length (2), scan range in tokens (-1 = whole context, 0 disables) |

Keep the penalty range bounded: unbounded frequency/presence penalties over a long context
were the cause of the "coherency cliff" around 8K tokens that was once blamed on the kernels.

**Server**

| flag | what it does |
|---|---|
| `-host ADDR`, `-port N` | bind address and port, default `127.0.0.1:3953` |
| `-key KEY` | require this API key (`Authorization: Bearer` or `x-api-key` header) |
| `-smn NAME` | model name reported by the API, default: the model directory name |
| `-maxr N` | server-side cap on tokens per response, default: fill the remaining context |
| `-ctk JSON` | default chat-template kwargs, e.g. `'{"enable_thinking": false}'` |
| `-lw N`, `-lmr N` | loop detection: stop after a window of N tokens repeats `-lmr` times (default 3); `-lw 0` disables |
| `--context-limit N` | API context cap within the allocated cache |
| `--max-output-tokens N` | maximum output tokens per choice |
| `--power-socket PATH` | selected-card power helper socket |
| `--audit-log PATH` | private request/response verification log |

---

## Inherited from upstream ExLlamaV3 (not revalidated on this branch)

> Everything below this line is the upstream/parent README content, kept for the CUDA path,
> the general API surface and model/architecture reference. It describes what upstream and
> the fork parent document, not what this branch re-measured: the V620 validation status and
> the ROCm build/reproduction specifics are in the sections above and in
> [doc/reproduce_v620.md](doc/reproduce_v620.md) / [doc/fork_changes.md](doc/fork_changes.md).

<p align="center">
  <img src="doc/logo.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

[Installation](#installation) · [Supported models](#architecture-support) · [Examples](#examples) · [Quantization](#exl3-quantization) · [Community](#community)

ExLlamaV3 is an inference library for running local LLMs on modern consumer GPUs, with flexible quantization and parallel inference.

- **Quantization** - [EXL3](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md), based on QTIP, plus 2–8 bit cache quantization.
- **Parallel inference** - Flexible tensor-parallel and expert-parallel inference for consumer hardware setups.
- **CPU offloading** - Allows large MoE models to run with limited GPU resources. AVX2 and AVX512 support.  
- **Generation** - Continuous, dynamic batching, speculative decoding, multimodal support.
- **Integrations** - Broad [HF model support](#architecture-support), a [Transformers plugin](examples/transformers_integration.py), and an OpenAI-compatible API via the [bundled server](#server).

<p align="center">
  <img src="doc/qb_kld.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

## Installation

Start by making sure you have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed (CUDA 12.4 or later) since the Torch dependency is not automatically handled by `pip`. Then pick a method below:

> **ROCm:** see [Quick start](#quick-start-build-for-gfx1030) at the top of this file, and
> [doc/reproduce_v620.md](doc/reproduce_v620.md) for the fully specified V620 environment. The
> methods below are upstream's CUDA instructions, kept for the CUDA path. There is no prebuilt
> ROCm wheel; building from source is the ROCm route.

### Prebuilt wheel · recommended

Pick a wheel from the [releases page](https://github.com/turboderp-org/exllamav3/releases), then e.g.:

```sh
pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.8.0-cp313-cp313-linux_x86_64.whl
```

### Install from PyPI

```sh
pip install exllamav3
```
Note that the PyPI package does not contain a prebuilt extension and requires the CUDA toolkit and build prerequisites (i.e. VS Build Tools on Windows, gcc on Linux, `python-dev` headers etc.).

### Build from source

<details>
<summary>Source installation with uv or pip</summary>


`exllamav3` declares a minimum `torch` version (>= 2.6.0) and CUDA version (>= 12.4), but beyond that the user is free to select a version of `torch` that is compatible with their environment.

`torch` can be installed in three ways (from least to most effort):
1. **with `uv`, setting only `--extra cuXXX`** installs `torch` automatically with the specified CUDA version, `torch` version is selected by `uv` from compatible versions in the specific index associated with the chosen CUDA version (options 1 and 2)
2. **with `uv`, creating a thin project that depends on `exllamav3[cuXXX]` and pins a specific `torch` version** — like (1) but `torch` is pinned in the thin project's `pyproject.toml`, see [pinning a specific PyTorch version (optional)](#pinning-a-specific-pytorch-version-optional) for details
3. Manually with `uv pip` or `pip` (options 3 and 4)

The flavor extras (`--extra`) are `cu124`, `cu126`, `cu128`, `cu129`, `cu130`, and `cu132` — pick the one matching your installed CUDA build. Both `uv sync` and `pip install .` build the package in an isolated environment where your `torch` is not visible, so they install the extension sources and compile them at first import (JIT, a few minutes once per torch version). For a precompiled install run `pip install --no-build-isolation .` in an environment that already has `torch`, or use the release wheels. Selecting a flavor installs the matching CUDA build of `torch`.

**Option 1 — Working in the cloned repo directly (`uv sync`):**

```sh
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3
# (Optional) switch to dev branch for latest in-progress features
git checkout dev

uv venv
uv sync --extra cu130
# add --extra examples and/or --extra eval for those extra dependencies
```

**Option 2 — Using `exllamav3` as a dependency from another project (`uv add`):**

```sh
# `uv add` works inside an existing project (a directory with a pyproject.toml).
# `uv init` creates one if you're starting a new project, if integrating into
# an existing project skip `uv init`.
uv init my-project
cd my-project

# local checkout
uv add 'path/to/exllamav3[cu130]'               # non-editable
uv add 'path/to/exllamav3[cu130]' --editable    # editable

# straight from GitHub
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]'                 # default branch
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]' --branch dev    # specific branch
```

**Option 3 — Bring your own `torch` and let `uv` pick the backend automatically:**

```sh
uv venv            # or: uv venv --python-preference only-managed
source .venv/bin/activate
uv pip install torch --torch-backend=auto
uv pip install .
```

`--torch-backend=auto` inspects your system and installs the matching PyTorch CUDA build; see [Automatic backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection).

**Option 4 — With `pip`:**

On Windows, you also need the `triton-windows` package (declared as a dependency in `pyproject.toml`); the attention, cache and recurrent kernels are Triton and ExLlamaV3 does not import without it.

```sh
# install a CUDA-enabled torch first so it matches your setup, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install .
```

</details>

ROCm-specific build variables:
- `EXL3_BACKEND`: `cuda` or `rocm`, forcing the backend. Otherwise it follows the installed torch.
- `MAX_JOBS`: also honoured by the ROCm builder, which drives `hipcc` directly. It defaults to a value
  bounded by core count, RAM (~2.5 GB budgeted per job) and a cap of 32, so lower it if you still run out of memory.
- `PYTORCH_ROCM_ARCH` / `GPU_ARCHS`: comma- or space-separated `gfx` list to build for (e.g. `gfx1030`
  for a V620, or `gfx1100,gfx1151`; semicolons are *not* separators). An explicit value is used as-is;
  this fork documents building V620 work with an explicit `gfx1030`. If unset, the build uses what
  `rocminfo` reports, filtered against the supported list (`gfx1030` is included on this branch).
  `PYTORCH_ROCM_ARCH` takes precedence.
- LDS budget: not an environment variable. `setup.py` passes `-DEXL3_RDNA_SMEM_MAX` from the target archs
  (64 KB for `gfx1030` (RDNA2), Strix / Strix Halo, `gfx1201` as measured on R9700, and unknown targets;
  90 KB for discrete RDNA3 and `gfx1200`), and at runtime it is further clamped to the device's
  `sharedMemPerBlock`.
- `EXL3_RDNA_MOE_TILESIZE_K`: `32` (default) or `16`. 16 forces the MoE GEMMs onto the single-K path — the
  tile geometry every RDNA shape is validated on — at a cost of roughly 1.4–1.6× MoE throughput. It is the
  first thing to try if fused MoE output ever looks wrong.
- `EXL3_SKIP_ROCM_VERSION_CHECK`: bypass the ROCm >= 7.2.4 requirement. Not advised — older ROCm builds this
  extension successfully and then computes wrong results.

<details>
<summary>Pinning a specific PyTorch version (optional)</summary>

#### Pinning a specific PyTorch version (optional)

The flavor extra picks the *index*, but by default torch resolves to the latest version on that
index that satisfies `>=2.6.0`. To pin a specific torch version while developing on `exllamav3`,
create a **"thin" project** that consumes your local checkout as an editable install and declares
the exact `torch` version itself. This keeps the pin out of the `exllamav3` pyproject, so
you can change the torch version freely without touching the repo.

```
my-exllamav3-dev/          # thin project (uv init)
├── pyproject.toml
└── src/                  # package sources (auto-generated)
```

In `pyproject.toml`:

```toml
[project]
name = "my-exllamav3-dev"
version = "0.1.0"
description = "Dev environment for exllamav3"
requires-python = ">=3.10.11"
dependencies = [
    "exllamav3[cu130]",   # select correct CUDA version
    "torch==2.13.0",      # pin the exact torch version you need
]

[tool.uv.sources]
exllamav3 = { path = "../exllamav3", editable = true }
```

Adjust `../exllamav3` to point at your local checkout, then a plain `uv sync` sets up an
environment with the correct PyTorch index (routed via the `cuXXX` extra),
the pinned version of `torch` from that index (as long as it exists), and an editable install of `exllamav3` so code
changes apply immediately. Switch CUDA flavors by changing the extra (`exllamav3[cu124]`,
`exllamav3[cu128]`, …) and/or the torch pin in the thin project.

Or, if you're installing torch manually with `uv pip install torch` (e.g. as in Option 3 above),
specify the version directly, e.g. `uv pip install "torch==2.11.0" --torch-backend=auto`.

</details>

After installing with one of the options above, you should be able to run the conversion, eval and 
example scripts from the main repo directory, e.g., `uv run python convert.py -i ...` or, for manual
installations once the venv is active, `python convert.py -i ...`

**Build environment variables**

- `MAX_JOBS`: by default ninja may launch too many processes and run out of system memory for 
compilation. Set this to a reasonable value like 4 in that case.
- `EXLLAMA_NOCOMPILE`: set to install the library without compiling the C++/CUDA extension. Torch
will build/load it at runtime instead.

## Examples

A number of example scripts are provided to showcase the features of the backend and generator. 
For instance, a versatile CLI chatbot:

<p align="center">
  <img src="doc/chatpy.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

```sh
python examples/chat.py -m /path/to/model -mode PROMPT_MODE

# Wealth of options
python examples/chat.py -h
```

## Architecture support

| Model family                                     | HF architecture | Multimodal | Notes |
|--------------------------------------------------| --- | :---: | --- |
| **AFM**                                          | `ArceeForCausalLM` |  |  |
| **AfMoE**                                        | `AfmoeForCausalLM` |  |  |
| **Apertus**                                      | `ApertursForCausalLM` |  |  |
| **Command-R** etc.                               | `CohereForCausalLM` |  |  |
| **Command-A**, **Command-R+** etc.               | `Cohere2ForCausalLM` |  |  |
| **DeciLM**, **Nemotron**                         | `DeciLMForCausalLM` |  |  |
| **Deepseek V3**                                  | `DeepseekV3ForCausalLM` |  |  |
| **Deepseek V4**                                  | `DeepseekV4ForCausalLM` | ✓ |  |
| **dots.llm1**                                    | `Dots1ForCausalLM` |  | |
| **ERNIE 4.5**                                    | `Ernie4_5_ForCausalLM`<br>`Ernie4_5_MoeForCausalLM` |  |  |
| **EXAONE 4.0**                                   | `Exaone4ForCausalLM` |  |  |
| **Gemma 2**                                      | `Gemma2ForCausalLM` |  |  |
| **Gemma 3**                                      | `Gemma3ForCausalLM`<br>`Gemma3ForConditionalGeneration` | ✓ |  |
| **Gemma 4**                                      | `Gemma4ForConditionalGeneration`<br>`Gemma4UnifiedForConditionalGeneration` | ✓ | E2B/E4B unsupported |
| **GLM 4**, **GLM 4.6**, etc.                     | `Glm4ForCausalLM`<br>`Glm4MoeForCausalLM` |  |  |
| **GLM 4.1V**, **GLM 4.5V**                       | `Glm4vForConditionalGeneration`<br>`Glm4vMoeForConditionalGeneration` | ✓ |  |
| **GLM 4.7 Flash**                                | `Glm4MoeLiteForCausalLM` |  |  |
| **GLM 5.2**                                      | `GlmMoeDsaForCausalLM` |  |  |
| **GLM 5.3-Flash**                                | `Glm5NextForConditionalGeneration` | ✓ |  |
| **GPT-OSS**                                      | `GptOssForCausalLM` |  |  |
| **HyperCLOVAX**                                  | `HyperCLOVAXForCausalLM`<br>`HCXVisionV2ForCausalLM` | ✓ |  |
| **Hy3**                                          | `HYV3ForCausalLM` |  |  |
| **IQuest-Coder**                                 | `IQuestCoderForCausalLM` |  |  |
| **Laguna 2.1**                                   | `LagunaForCausalLM` |  |  |
| **LFM 2.5**                                      | `Lfm2ForCausalLM`<br>`Lfm2MoeForCausalLM` |  |  |
| **Llama 1/2/3**,**3.1-Nemotron** etc.            | `LlamaForCausalLM` |  |  |
| **MiMo-RL**                                      | `MiMoForCausalLM` |  |  |
| **MiniMax-M2**                                   | `MiniMaxM2ForCausalLM` |  |  |
| **Mistral**, **Ministral 3**, **Mistral-4** etc. | `MistralForCausalLM`<br>`Mistral3ForConditionalGeneration` | ✓ |  |
| **Mixtral**                                      | `MixtralForCausalLM` |  |  |
| **NemotronH, Nemotron-3 Nano/Super**              | `NemotronHForCausalLM` |  |  |
| **Olmo 3.1**                                     | `Olmo3ForCausalLM` |  |  |
| **Olmo-Hybrid**                                  | `OlmoHybridForCausalLM` |  |  |
| **Phi3**, **Phi4**                               | `Phi3ForCausalLM` |  |  |
| **Qwen 2**, **Qwen 2.5**, **Qwen 2.5 VL**        | `Qwen2ForCausalLM`<br>`Qwen2_5_VLForConditionalGeneration` | ✓ |  |
| **Qwen 3**                                       | `Qwen3ForCausalLM`<br>`Qwen3MoeForCausalLM` |  |  |
| **Qwen 3-Next**                                  | `Qwen3NextForCausalLM` |  |  |
| **Qwen 3-VL**                                    | `Qwen3VLForConditionalGeneration` | ✓ |  |
| **Qwen 3-VL MoE**                                | `Qwen3VLMoeForConditionalGeneration` | ✓ |  |
| **Qwen 3.5**                                     | `Qwen3_5ForConditionalGeneration` | ✓ |  |
| **Qwen 3.5 MoE**                                 | `Qwen3_5MoeForConditionalGeneration` | ✓ |  |
| **Qwen 3.8-Flash-Next**                          | `Qwen4ExpForConditionalGeneration` | ✓ |  |
| **Seed-OSS**                                     | `SeedOssForCausalLM` |  |  |
| **SmolLM**                                       | `SmolLM3ForCausalLM` |  |  |
| **SolarOpen**                                    | `SolarOpenForCausalLM` |  |  |
| **Step 3.5 Flash**                               | `Step3p5ForCausalLM` |  |  |
| **Step 3.7 Flash**                               | `Step3p7ForConditionalGeneration` | ✓ |  |

Always adding more, stay tuned.

## Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i /path/to/input -o /path/to/output -w /path/to/work -b 4.0

# Resume an interrupted quant job
python convert.py -w <working_dir> -r

# More options
python convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors 
until the converted model can be compiled. It should have enough free space to store an entire copy 
of the output model.

See the [conversion guide](doc/convert.md) for more information, or the 
[self-calibration guide](doc/optimize.md). 

## EXL3 quantization

EXL3 quantization is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. It aims to make
SOTA quantization available to users on consumer hardware. The conversion process is designed to be
simple and efficient and requires only an input model (in HF format) and a target bitrate. By
computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a 
model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger
ones (70B+) on a single high-end consumer GPU (see the [conversion guide](doc/convert.md)).

For more information, see the [**QTIP**](https://arxiv.org/abs/2406.11235) and [**QuIP#**](https://arxiv.org/abs/2402.04396) papers, as well as this 
[excellent writeup](https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip) on **QTIP** from together.ai.


## Community

You are always welcome to join the [ExLlama discord server](https://discord.gg/NSFwVuCjRq) ←🎮


### 🤗 Models on Hugging Face

Browse the [EXL3 model collection](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a) for quantized models. Also shout out to the following lovely
people:

- [ArtusDev](https://huggingface.co/ArtusDev)
- [MikeRoz](https://huggingface.co/MikeRoz)
- [MetaphoricalCode](https://huggingface.co/MetaphoricalCode)
- [Ready.Art](https://huggingface.co/ReadyArt)
- [isogen](https://huggingface.co/isogen/models)


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous
supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (chunked linear-attention prefill kernels, vendored under `exllamav3/vendor/fla`)

<p align="center">
  <img src="doc/cat.png" width="40" alt="">
</p>

