# JEV-27B-VL System 1 HTTP latency, 2026-10-06

This measures the EXL3 mixed-precision pack described in [../jev.md](../jev.md).
The image task is the official computer-use demo: numbered browser screenshot
marks plus element text, choosing the next click or task completion. A local
Chromium actually executes the selected clicks in its synthetic mail, shop and
settings applications. No robot hardware or MuJoCo is needed.

## Measurement

- One client request at a time; System 1, `thinking: off`, `single` strategy.
- Client timer uses `perf_counter_ns`, starting before HTTP POST and ending
  after response JSON parsing. The server's `elapsed_seconds` is also saved.
- Model loading, browser rendering, drawing numbered marks and PNG encoding
  are outside this timer. Image decoding, resizing, vision inference, text
  prefill, LoRA, exact decision readout and calibration are inside it.
- Fresh KV/recurrent state per request; no cross-request prefix/image caching.
  Every measured response has one model request and zero generated tokens.
- Three warmups per text workload, then 30 repetitions. Three complete browser
  episodes warm up the vision path, followed by seeds 1–20 for each of the three
  apps (60 episodes). Browser requests count every decision, including the
  final completion decision. Failed tasks remain in the statistics.
- Median uses the usual middle/average-of-two definition; p95 is nearest rank.
  All samples and traces are retained, including outliers.

The official demo's task generation, screenshot marking, options, clicks and
completion checks are unchanged. Only the API destination and instrumentation
are replaced. Source revisions/digests, GPU/native runtime fingerprints and
model precision are in [conditions.json](conditions.json). Input screenshot
size is 1100×640; the current EXL3 server resizes images to its 262,144-pixel
maximum (65,536 minimum). Context is 16,384, prefill chunks 1,024, KV FP16.

## R9700 results

| Workload | Prompt tokens | Samples | HTTP median | HTTP p95 |
|---|---:|---:|---:|---:|
| Short text yes/no | 35 | 30 | 299.25 ms | 299.71 ms |
| Text 16-way choice | 121 | 30 | 313.03 ms | 314.12 ms |
| Longer text yes/no | 2,846 | 30 | 2,683.50 ms | 2,686.41 ms |
| Browser screenshot + element text | 388–532 | 277 | 587.80 ms | 626.48 ms |

The browser tasks succeed in 57/60 episodes (95%). The server-side image
decision median is 585.12 ms, about 2.7 ms below the client HTTP median.
The initial dependency installation overlapped the first two text workloads;
the 16-choice set retains one 588 ms outlier in its mean/max. Browser runtime
dependencies were installed and preflighted before image measurements began.
Raw output: [r9700.json](r9700.json).

## Comparison scope

The [model README](https://huggingface.co/autotrust/JEV-27B-VL#new-3-october-2026-robot-arm-and-computer-use)
reports about 0.26 seconds per computer-use click. Its checked-in
[demo results JSON](https://huggingface.co/autotrust/JEV-27B-VL/blob/f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc/reports/demos/cu_results.json)
instead has a marks+text median of 719.97 ms and 57/60 successes. The upstream
[demo code](https://huggingface.co/autotrust/JEV-9B/blob/b63f651/vl/demos/computer_use.py)
defaults to six browser workers; this measurement is serial. The demo GPU,
image-processing settings and serving conditions behind the README number
are not fully specified, so these numbers do not establish a hardware speedup
ratio. Image pixel caps, prefix caching, batch/concurrency, precision and prompt
length all affect latency. These 60 synthetic tasks are not a broad computer-use
quality benchmark.

## Reproduce

Download `computer_use.py` and `webapp.html` from the above pinned demo revision
into one directory. Install `requests`, `Pillow` and `playwright` in the client
environment, and install/preflight its Chromium runtime. Start the native JEV
server with the documented model/native extension, then run:

```sh
python -m rocm_tools.jev_latency \
  --label R9700 --url http://127.0.0.1:3960 \
  --demo-dir /work/runs/jev-latency-20261006 \
  --output /work/runs/jev-latency-20261006/r9700.json
```

The test servers use private container networking without published host ports.
The existing Qwen API is temporarily stopped for V620 GPU availability and is
restored after measurements. Large server/client logs and downloaded reference
files remain in `/home/homelab1/datapool/rocm-exl3-rdna2/runs/jev-latency-20261006`.
