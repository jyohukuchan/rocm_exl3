import os, statistics, sys
sys.path.insert(0, "/src")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch
torch.cuda.set_device(0)
from exllamav3.model import Config, Model
from exllamav3.rocm_py.moe_int8 import quant_w8_slab
from exllamav3.ext import exllamav3_ext as ext

config = Config.from_directory("/work/models/qwen38-27b-exl3-305-int8grid")
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))

hits = []


def _walk(m, seen):
    if type(m).__name__ == "LinearEXL3":
        if getattr(m, "s0", None) is not None and (m.in_features, m.out_features) == (5120, 10240):
            hits.append(m)
        return
    try:
        attrs = vars(m).values()
    except TypeError:
        return
    for a in attrs:
        if a is None or a is m or id(a) in seen:
            continue
        if "exllamav3" in type(a).__module__:
            seen.add(id(a))
            _walk(a, seen)
        elif isinstance(a, (list, tuple)):
            for x in a:
                if "exllamav3" in type(x).__module__ and id(x) not in seen:
                    seen.add(id(x))
                    _walk(x, seen)


_walk(model, set())
m = hits[0]
k, n = m.in_features, m.out_features
dev = m.trellis.device
s0 = m.s0.to(dtype=torch.float32, device=dev)
w = torch.empty((k, n), dtype=torch.half, device=dev)
w8 = torch.empty((k, n), dtype=torch.int8, device=dev)


def med(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return round(statistics.median(ts), 4)


def _rq():
    ext.reconstruct(w, m.trellis, m.K, m.mcg, m.mul1)
    quant_w8_slab(w.unsqueeze(0), s0, w8.unsqueeze(0))


def _q_only():
    quant_w8_slab(w.unsqueeze(0), s0, w8.unsqueeze(0))


with torch.inference_mode():
    # swapped order: int8 first
    print("rq_first:", med(_rq), flush=True)
    print("recon_second:", med(lambda: ext.reconstruct(w, m.trellis, m.K, m.mcg, m.mul1)), flush=True)
    ext.reconstruct(w, m.trellis, m.K, m.mcg, m.mul1)
    torch.cuda.synchronize()
    print("q_only:", med(_q_only), flush=True)
    print("recon_third:", med(lambda: ext.reconstruct(w, m.trellis, m.K, m.mcg, m.mul1)), flush=True)
print("done", flush=True)
