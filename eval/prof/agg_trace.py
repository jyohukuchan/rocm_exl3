#!/usr/bin/env python3
"""Aggregate a rocprofv3 kernel-trace CSV into stage buckets (per-forward means).
Usage: agg_trace.py TRACE.csv [--forwards N]
Trace region = 1 warmup + N measured forwards (default N=3 -> divide by 4)."""
import csv, sys, re
from collections import defaultdict

path = sys.argv[1]
nfw = 4
for i, a in enumerate(sys.argv):
    if a == "--forwards" and i + 1 < len(sys.argv):
        nfw = 1 + int(sys.argv[i + 1])

with open(path) as f:
    rdr = csv.DictReader(f)
    cols = {c.lower(): c for c in rdr.fieldnames}
    kcol = next(cols[c] for c in cols if "kernel" in c and "name" in c)
    dcol = next((cols[c] for c in cols if c in ("durationns", "duration_ns", "duration", "dur")), None)
    if dcol is None:
        dcol = next(cols[c] for c in cols if "dur" in c)
    rows = [(r[kcol], float(r[dcol])) for r in rdr]

def bucket(name):
    n = name.lower()
    if re.search(r"attn|flash|paged|decode|kv_|qk_|softmax|rope|rotary|cache|kvcache", n):
        return "attention"
    if re.search(r"moe|expert|rout|gate.*top|topk|top_k|sort|scan|bincount|gather_moe|slot|index_add|scatter_add", n):
        return "moe"
    if re.search(r"hgemm|gemm|reconstruct|dequant|had_|hadamard|quant", n):
        # hadamard input transform belongs to the linear it feeds; keep split visible
        if re.search(r"had_", n):
            return "hadamard"
        return "dense_linear"
    if re.search(r"norm|rms|layernorm|silu|gelu|activ|embed|sampl|multinomial|top_p", n):
        return "norms_acts_misc"
    if re.search(r"\bcopy\b|memset|memcpy|fill|zero|set_", n):
        return "mem_overhead"
    return "other"

tot = defaultdict(float)
cnt = defaultdict(int)
per_kernel = defaultdict(float)
for name, dur in rows:
    b = bucket(name)
    tot[b] += dur
    cnt[b] += 1
    per_kernel[name] += dur

grand = sum(tot.values())
print(f"kernels={len(rows)} traced_time={grand/1e9:.3f}s per_forward~{grand/1e9/nfw:.3f}s (/{nfw})")
print(f"{'bucket':20s} {'ms/fwd':>10s} {'share':>7s} {'calls/fwd':>9s}")
for b, t in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"{b:20s} {t/1e6/nfw:10.2f} {t/grand:6.1%} {cnt[b]/nfw:9.1f}")
print("\nTop 25 kernels by time (ms/fwd):")
for name, t in sorted(per_kernel.items(), key=lambda x: -x[1])[:25]:
    print(f"  {t/1e6/nfw:10.2f}ms  [{bucket(name):16s}] {name[:110]}")
others = [(n, t) for n, t in per_kernel.items() if bucket(n) == "other"]
if others:
    print("\nUnclassified kernels (refine rules):")
    for name, t in sorted(others, key=lambda x: -x[1])[:15]:
        print(f"  {t/1e6/nfw:10.2f}ms  {name[:130]}")
