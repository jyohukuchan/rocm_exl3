# Reproducing the V620 × 2 TP2 benchmarks (2026-09-30 bundle)

This guide reproduces the dated benchmark bundle at
[`benchmarks/2026-09-30/`](../benchmarks/2026-09-30/README.md) — Qwen3.8-Flash-Next on
**two Radeon Pro V620 (gfx1030)** in 2-way tensor parallel, MTP draft, K5/V4 KV cache,
Engram as a single CPU-RAM table. The commands below assume a Linux host shell with access to the GPUs and sudo for the power helper/process-limit setup. It uses only this repository's tracked code plus
dependencies you acquire yourself (ROCm, torch, the EXL3 model directory). There is no
private artifact directory, no local container image requirement, and no bundled model
weights — MIT covers the code, not the weights.

Numbers being reproduced (full-span aggregate decode tok/s, JA / code): B1 38.88/50.52,
B2 55.85/56.68, B3 66.25/73.95, B4 70.56/66.51; B4 conservative aggregate prefill
471.80/459.68; B4 common-window decode 76.29/85.80. See
[doc/qwen38_v620_context_batch.md](qwen38_v620_context_batch.md) for the full analysis and
[doc/v620_tp_decode_optimization.md](v620_tp_decode_optimization.md) for the decode-path
breakdown.

## 1. What was actually measured (validated environment)

| item | value used for the measurements |
|---|---|
| GPUs | 2× Radeon Pro V620, `gfx1030`, ≈32 GiB VRAM each, PCIe Gen4 ×16 each |
| Host RAM | ≈109 GiB. The Engram n-gram table alone is 32,640,156,672 bytes (>30 GiB) in one physical CPU copy; plan real table size plus working headroom |
| Python | 3.12.3 |
| torch | 2.12.0+rocm7.2 |
| triton-rocm | 3.7.0 |
| numpy / transformers / safetensors / tokenizers | 2.4.4 / 5.17.0 / 0.8.0 / 0.23.2 |
| HIP compiler | 7.14.60850-0000000, installed under `/opt/rocm/core-7.14` |
| Loaded ROCr library | SDK `libhsa-runtime64.so.1.21.0`; HIP/RCCL libraries came from the Torch wheel |
| Model | Qwen3.8-Flash-Next, original distributed EXL3 pack at 3.05 bpw + original packed 3-bit MTP component, repository `turboderp/Qwen3.8-Flash-Next-exl3`, branch `3.05bpw_h5_ng5`, revision `69e33439ae950f17bcbe95c98f117d80f759ab6d` (user-supplied directory; provenance recorded in the bundle's `results.json` / [qwen38_v620_tp_config.json](qwen38_v620_tp_config.json)) |

**Toolchain scope.** The measured build combined a *custom* ROCm 7.14 HIP/ROCr SDK
(not a distributable prebuilt image) with the `rocm7.2` torch wheel. `setup.py` enforces
only **ROCm ≥ 7.2.4**; passing that check does not mean a given version was tested. Your
own ROCm install path is fine — use it explicitly (`ROCM_PATH`, `PATH`, library search
paths pointing at *your* locations) and build with an explicit arch. Do not expect a
one-command turnkey install of the exact measured stack.

**Model weights are not part of this repository.** Point `MODEL_DIR` at your own
converted or downloaded EXL3 directory. To match the published numbers you need the
checkpoint revision above (original 3.05 bpw pack with its packed MTP component).

## 2. Build

```bash
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

`--no-build-isolation` is mandatory (the extension must compile against the torch it will
run against). `MAX_JOBS` bounds parallel `hipcc` jobs on low-RAM hosts. The measured runs
preloaded the ROCm runtime explicitly (see §4) — build and runtime should agree on which
ROCm install is in play.

## 3. Fresh private run directory + frozen prompts

The bundle ships gzip-compressed frozen token-prompt files (one per batch setting).
Prompts are exact token IDs; decompress into a fresh private dir, never in place:

```bash
RUN_DIR=$(mktemp -d)
gzip -dc benchmarks/2026-09-30/prompts-8192-b4-r3.json.gz > "$RUN_DIR/prompts.json"
```

(`prompts-8192-b1-r2.json.gz`, `b2-r2`, `b3-r2` are the batch-1/2/3 sets.) The batch-4
file above is what reproduces the 2026-09-30 B4 confirmation. The single verified
long-context batch-1 run (261,632 input + 256 output, fixed MTP4) used a separately frozen
long-prompt file: `long-prompts-cap262144-b1.json.gz` is included for that separate observation. It is not run by the batch4 command below.

## 4. Runtime environment for the measured configuration

Set before starting Python. Every value below is what the published runs used;
`doc/qwen38_v620_tp_config.json` is documentation — it is **not** auto-loaded.

```bash
# ROCM_PATH/PATH were selected before building in section2.
# Locate Torch before enabling LD_PRELOAD; its location depends on your venv.
TORCH_LIB=$(python -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD=libhsa-runtime64.so        # bare name; see the duplicate-runtime note below
export HSA_ENABLE_SDMA=0                    # measured peer-copy workaround

export EXL3_TP_REPLICATE_ROUTER=1            # std MoE router computed on both ranks
export EXL3_ROCM_MOE_MGEMM_MAX_ROWS=20       # opt-in per-token MoE route threshold (8..24)
export EXL3_BATCH_RECURRENT_PRUNE=1          # batched recurrent checkpoint pruning
export EXL3_NGRAM_MLOCK=1                    # lock the Engram CPU table (needs §5 limits)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export EXL3_TP_TIMEOUT_S=45
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
export NCCL_DEBUG=WARN
```

Select your two V620s and keep their order consistent everywhere:

```bash
export ROCR_VISIBLE_DEVICES='GPU-REPLACE_FIRST_UUID,GPU-REPLACE_SECOND_UUID'
GPU0_PCI='REPLACE_FIRST_PCI_BDF'     # canonical lowercase, e.g. 0000:43:00.0
GPU1_PCI='REPLACE_SECOND_PCI_BDF'    # select your second V620, not an unrelated GPU
```

Replace all four device placeholders before running. GPU UUIDs and PCI BDFs are different identifiers; use `rocminfo` and your ROCm/AMD SMI device listing to match them. UUIDs are machine-specific — use **your own**; nothing here assumes a particular pair. The
same order must be used for the power helper's `--devices` list and matches the GPU order
`--use-per-device` budgets apply to.

## 5. Process limits — raise them, but do not run inference as root

`EXL3_NGRAM_MLOCK=1` needs a ≥34 GiB `RLIMIT_MEMLOCK` on the *inference* process; the
harness also wants `nofile` 65536. Apply both to a dedicated subshell whose `exec`ed
Python (and children) inherit them — the limits never touch your login shell or the OS:

```bash
(
  set -e
  sudo prlimit --pid "$BASHPID" \
    --memlock=36507222016:36507222016 --nofile=65536:65536
  exec python -m rocm_tools.rdna2.tp_run ...        # §6
)
```

36507222016 bytes = 34 GiB. If your inference environment already has these limits, omit the `prlimit` step. A container setup additionally needs GPU access, sufficient shared memory and a shared private socket directory for a host-side power helper; this guide does not supply a prebuilt container. No system-wide swap/ARC changes are needed.

## 6. The two long-running pieces

### 6.1 Power helper (started first, one client at a time)

```bash
sudo -v                         # authenticate in the foreground before starting a background helper
sudo -n python3 rocm_tools/rdna2/power_server.py \
  --devices "$GPU0_PCI" "$GPU1_PCI" \
  --socket "$RUN_DIR/power.sock" \
  --report "$RUN_DIR/power.json" &
POWER_HELPER_JOB=$!
# Wait for READY (or inspect the process if it exits); do not start the runner before this socket exists.
for attempt in {1..100}; do
  [ -S "$RUN_DIR/power.sock" ] && break
  kill -0 "$POWER_HELPER_JOB" 2>/dev/null || break
  sleep 0.1
done
test -S "$RUN_DIR/power.sock" || { echo "Power helper did not become ready" >&2; exit 1; }
```

- `--devices` takes the **actual PCI BDFs of the two GPUs you selected** (find them with
  `lspci` / `rocminfo`); there are no hardcoded device numbers, GPU UUIDs or indices.
- The helper is pure-stdlib Python (no venv needed for the `sudo` invocation).
- It serves exactly one client (the runner), sets GPU power profiles on request
  (`profile_peak` during inference; batch-1 stage switching auto↔peak is described in
  `doc/qwen38_v620_tp_config.json`), and **restores the original policies on disconnect or
  SIGTERM**. Check `"$RUN_DIR/power.json"` afterwards — a completed run must show
  restoration. A separate `--validate-finite` run adds finite-value hooks and is marked validation-only; do not mix its timings into these benchmarks.
- The socket is created mode `0600` and owned by `SUDO_UID`, so only your invoking user
  can connect.

### 6.2 Benchmark runner (in the raised-limit subshell from §5)

```bash
MODEL_DIR=/path/to/qwen38-flash-next-exl3-3.05bpw     # your directory

(
  set -e
  sudo prlimit --pid "$BASHPID" \
    --memlock=36507222016:36507222016 --nofile=65536:65536
  exec python -m rocm_tools.rdna2.tp_run \
    --model "$MODEL_DIR" \
    --execution tp \
    --mode mtp \
    --prompts-json "$RUN_DIR/prompts.json" \
    --output "$RUN_DIR/report.json" \
    --power-socket "$RUN_DIR/power.sock" \
    --batch-size 4 \
    --draft-tokens 1 \
    --cache-tokens 34816 \
    --new-tokens 256 \
    --max-chunk-size 2048 \
    --use-per-device 28 28
)
```

Flag meanings (all accepted by the current parser; `tp_run --help` is authoritative):

| flag | value used | why |
|---|---|---|
| `--execution tp` | 2-way tensor parallel over both V620s | the measured mode (`--execution ls` is the layer-split alternative) |
| `--mode mtp` | model's own packed MTP head | no standalone draft override exists; requires a checkpoint with an MTP component |
| `--draft-tokens 1` | fixed draft window 1 | B2–B4 selection from the MTP1–4 screening; B1 used `--draft-tokens 4 --dynamic-draft --draft-confidence 0.6` |
| `--cache-tokens 34816` | total across the batch | 8704 per sequence at batch 4 — 8192 input + 256 output + margin |
| `--new-tokens 256`, `--max-chunk-size 2048` | as measured | |
| `--use-per-device 28 28` | GiB load budget per GPU | board VRAM peaks were 27.47/28.72 GiB |
| (default) K5/V4 KV | `--cache-k-bits 5 --cache-v-bits 4` are the defaults | the `--cache-fp16` flag is a diagnostic baseline, never the default |

After the runner exits, `wait "$POWER_HELPER_JOB"` and inspect the power report. If model loading failed **before** the runner connected, the helper is still waiting for its one client: stop that helper with Ctrl-C in its terminal, or connect to its socket and immediately close it to request a clean empty session. Do not leave an unused helper running.

The runner performs the audits the results depend on: TP placement, actual K5/V4 cache
layers, single-owner Engram RAM residency (before/after, with mlock), and completion of every job (batch 4: 32 jobs total). Success requires the report,
the audit sections, **and** the power report to all be present and passing.

### 6.3 Optional exact historical reproduction: host-reserve override

The published batch-4 confirmation ran with one extra variable:

```bash
export EXL3_HOST_MEM_RESERVE_MB=0
```

**Why this existed:** the loader's host-memory guard compares the table allocation size against kernel `MemAvailable` minus a 2 GiB reserve, but `MemAvailable` omits
reclaimable ZFS ARC — on that host the guard rejected an otherwise-safe load (needed
31128 MiB, reported 24997 MiB). The override was recorded as a *diagnostic* for that run
only; the mlock and residency audits stayed mandatory. **Do not** change swap or ARC OS
settings to make the guard happy, and treat the default reserve as correct for normal use.
Whether the guard succeeds depends on host memory conditions, not just batch size. This override is not a normal default.

## 7. Other published batch settings

Use the matching prompt manifest and a fresh run directory/helper for each run. Keep all other settings above, except:

| Batch | Prompt file under the dated bundle | Draft flags | Total cache tokens |
|---:|---|---|---:|
| 1 | `prompts-8192-b1-r2.json.gz` | `--draft-tokens 4 --dynamic-draft --draft-confidence 0.6` | 8704 |
| 2 | `prompts-8192-b2-r2.json.gz` | `--draft-tokens 1` | 17408 |
| 3 | `prompts-8192-b3-r2.json.gz` | `--draft-tokens 1` | 26112 |
| 4 | `prompts-8192-b4-r3.json.gz` | `--draft-tokens 1` | 34816 |

Also set `--batch-size` accordingly. The historical batch1–3 runs did not set `EXL3_NGRAM_MLOCK`; they passed before/after residency audits. Leaving mlock enabled is useful for stable table residency; unset it only when intentionally matching that historical setting. Keep the same model/tokenizer revision because the prompt files contain frozen token IDs.

The separate recorded batch1 long-context observation uses `long-prompts-cap262144-b1.json.gz`, `--batch-size 1 --draft-tokens 4 --cache-tokens 262144 --use-per-device 30 29 --max-chunk-size 2048`, without dynamic draft. It is not part of the default speed run and does not establish larger-batch context limits.

## 8. Summarize

```bash
python -m rocm_tools.rdna2.summarize_tp "$RUN_DIR/report.json"
```

prints the public metrics (per-language full-span aggregate decode, common-window
aggregate decode, conservative aggregate prefill, acceptance) that
[`benchmarks/2026-09-30/results.json`](../benchmarks/2026-09-30/README.md) reports.

## 9. Interpreting the measurements

- **Metric definitions.** Full-span aggregate decode = delivered tokens from first
  delivery to last, *including* final queue-drain delay, excluding the first burst.
  Common-window aggregate decode = only the interval while every job is still generating.
  Aggregate prefill = total input tokens over the **latest** first delivery (conservative).
  Engine-internal `time_generate` numbers are not interchangeable with these.
- **Protocol.** Per language: warm 1 + timed 2 (B1–B3) / warm 1 + timed 3 (B4); input
  8192, output 256 each. B1 dynamic-MTP4 acceptance differs from B2–B4 fixed-MTP1; compare
  like with like.
- **Observed variation.** Batch4 full-span rates ranged68.46–73.24 tok/s (Japanese) and63.98–72.04 tok/s (code) across the three timed groups.
- Run one V620 benchmark at a time.
- **Known residual issues.** Cross-path numerical differences (cold vs prefix-reuse) were observed in both pruning controls; MTP acceptance is workload-dependent; B2–B4 maximum
  context is **not** established (only B1 at 261,632+256 completed; long-context B2–4 were
  deferred by the operator). Allocation-only probes are not runtime limits.
- Do not silently pool earlier two-repeat runs into the three-repeat confirmation; the
  bundle keeps them distinct.

## 10. Related documents

- Bundle index & sanitized data: [`benchmarks/2026-09-30/README.md`](../benchmarks/2026-09-30/README.md)
  (`results.json`, `environment.json`, prompt files, `reports/*.json.gz`)
- Raw analysis: [doc/qwen38_v620_context_batch.md](qwen38_v620_context_batch.md),
  [doc/v620_tp_decode_optimization.md](v620_tp_decode_optimization.md),
  [doc/qwen38_v620_tp_config.json](qwen38_v620_tp_config.json)
- Harness internals: [../rocm_tools/rdna2/README.md](../rocm_tools/rdna2/README.md)
- Fork delta vs the parent: [fork_changes.md](fork_changes.md)
