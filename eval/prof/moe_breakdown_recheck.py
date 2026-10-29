import sys
sys.path.insert(0, "/src")
import statistics
import torch
torch.cuda.set_device(0)
from exllamav3.ext import exllamav3_ext as ext


def med(fn, iters=40):
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
    ts.sort()
    return statistics.median(ts), ts[0], ts[-1]


g = torch.Generator(device="cuda:0").manual_seed(11)
with torch.inference_mode():
    for (K, N, R) in [(2560, 640, 1024), (2560, 640, 2048), (2048, 768, 160)]:
        for rep in range(3):
            a8 = torch.randint(-100, 100, (R, K), generator=g, device="cuda:0",
                               dtype=torch.int32).to(torch.int8)
            w8 = torch.randint(-100, 100, (K, N), generator=g, device="cuda:0",
                               dtype=torch.int32).to(torch.int8)
            c32 = torch.empty((R, N), dtype=torch.int32, device="cuda:0")
            xh = torch.randn((R, K), generator=g, device="cuda:0", dtype=torch.float16)
            w = torch.randn((K, N), generator=g, device="cuda:0", dtype=torch.float16)
            y = torch.empty((R, N), dtype=torch.float16, device="cuda:0")
            ti, ta, tb = med(lambda: ext.igemm_recon(a8, w8, c32))
            th, ha, hb = med(lambda: ext.hgemm_recon(xh, w, y))
            print(f"K={K} N={N} R={R} rep{rep}: fp16 med={th:.4f} [{ha:.4f},{hb:.4f}] "
                  f"int8 med={ti:.4f} [{ta:.4f},{tb:.4f}] su={th/ti:.2f}x", flush=True)
    # batched anomaly recheck: q30b gu B=8 cmax=160
    for rep in range(3):
        B, R, K, N = 8, 160, 2048, 768
        A8 = torch.randint(-100, 100, (B, R, K), generator=g, device="cuda:0",
                           dtype=torch.int32).to(torch.int8)
        W8 = torch.randint(-100, 100, (B, K, N), generator=g, device="cuda:0",
                           dtype=torch.int32).to(torch.int8)
        Cb = torch.empty((B, R, N), dtype=torch.int32, device="cuda:0")
        Xh = torch.randn((B, R, K), generator=g, device="cuda:0", dtype=torch.float16)
        W = torch.randn((B, K, N), generator=g, device="cuda:0", dtype=torch.float16)
        Y = torch.empty((B, R, N), dtype=torch.float16, device="cuda:0")
        ti, _, _ = med(lambda: ext.igemm_batched(A8, W8, Cb))
        th, _, _ = med(lambda: ext.hgemm_exact_group(Xh, W, Y, [R] * B))
        print(f"batched gu160 rep{rep}: fp16={th:.4f} int8={ti:.4f} su={th/ti:.2f}x", flush=True)
print("done", flush=True)
