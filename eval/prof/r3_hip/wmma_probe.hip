// Round-3 gfx1201 WMMA probe + rate microbench (host hipcc, no torch needed).
//
// 1) Layout battery: tries hypothesized (A-load, B-load, operand-order,
//    C-store) mappings for fp16/int8/fp8 WMMA w32 gfx12 builtins against a
//    CPU reference with non-symmetric matrices. Prints PASS/FAIL per combo.
// 2) Rate: uniform-operand WMMA loops (operands already in registers) timed
//    with HIP events -> upper-bound VALU issue rate per dtype.
//
// Build: hipcc -O3 -o /tmp/r3/wmma_probe wmma_probe.hip --offload-arch=gfx1201
// Run:   HIP_VISIBLE_DEVICES=1 /tmp/r3/wmma_probe
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

// ---------------------------------------------------------------- fp16
__global__ void probe_f16(const _Float16* A, const _Float16* B, float* C,
                          int ia, int ib, int ord, int ic) {
    int L = threadIdx.x & 31;
    f16x8 a, b;
    // A hypotheses
    if (ia == 0) { int r = L / 2, k0 = 8 * (L % 2);
        for (int s = 0; s < 8; s++) a[s] = A[r * 16 + k0 + s];
    } else { int r = L % 16, k0 = 8 * (L / 16);
        for (int s = 0; s < 8; s++) a[s] = A[r * 16 + k0 + s];
    }
    // B hypotheses
    if (ib == 0) { int c = L / 2, k0 = 8 * (L % 2);
        for (int s = 0; s < 8; s++) b[s] = B[(k0 + s) * 16 + c];
    } else { int c = L % 16, k0 = 8 * (L / 16);
        for (int s = 0; s < 8; s++) b[s] = B[(k0 + s) * 16 + c];
    }
    f32x8 acc; for (int s = 0; s < 8; s++) acc[s] = 0.0f;
    if (ord == 0) acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    else acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b, a, acc);
    // C store hypotheses
    if (ic == 0) { int r = L / 2, c0 = 8 * (L % 2);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s];
    } else { int r = L % 16, c0 = 8 * (L / 16);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s];
    }
}

// ---------------------------------------------------------------- int8
__global__ void probe_i8(const int8_t* A, const int8_t* B, int* C,
                         int ia, int ib, int ord, int ic) {
    int L = threadIdx.x & 31;
    auto load8 = [&](const int8_t* M, int row, int k0, bool rowmajor) {
        i32x2 v;
        // pack 8 int8 at M[row*16+k0 .. +8) (row-major) into 2 dwords
        int32_t w0, w1;
        __builtin_memcpy(&w0, M + row * 16 + k0, 4);
        __builtin_memcpy(&w1, M + row * 16 + k0 + 4, 4);
        v[0] = w0; v[1] = w1;
        return v;
    };
    i32x2 a, b;
    if (ia == 0) { int r = L / 2, k0 = 8 * (L % 2); a = load8(A, r, k0, true); }
    else { int r = L % 16, k0 = 8 * (L / 16); a = load8(A, r, k0, true); }
    // B is KxN row-major: element (k, c) at B[k*16+c]; column c, k-slice:
    // gather 8 bytes (non-contiguous) manually.
    auto loadB = [&](int col, int k0) {
        uint8_t tmp[8];
        for (int s = 0; s < 8; s++) tmp[s] = (uint8_t)B[(k0 + s) * 16 + col];
        i32x2 v; int32_t w0, w1;
        __builtin_memcpy(&w0, tmp, 4); __builtin_memcpy(&w1, tmp + 4, 4);
        v[0] = w0; v[1] = w1;
        return v;
    };
    if (ib == 0) { int c = L / 2, k0 = 8 * (L % 2); b = loadB(c, k0); }
    else { int c = L % 16, k0 = 8 * (L / 16); b = loadB(c, k0); }
    i32x8 acc; for (int s = 0; s < 8; s++) acc[s] = 0;
    if (ord == 0) acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, acc, false);
    else acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, b, false, a, acc, false);
    if (ic == 0) { int r = L / 2, c0 = 8 * (L % 2);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s];
    } else { int r = L % 16, c0 = 8 * (L / 16);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s];
    }
}

// ---------------------------------------------------------------- fp8 (E4M3 bits)
__global__ void probe_fp8(const uint8_t* A, const uint8_t* B, float* C,
                          int ia, int ib, int ord, int ic) {
    int L = threadIdx.x & 31;
    auto loadA = [&](int row, int k0) {
        i32x2 v; int32_t w0, w1;
        __builtin_memcpy(&w0, A + row * 16 + k0, 4);
        __builtin_memcpy(&w1, A + row * 16 + k0 + 4, 4);
        v[0] = w0; v[1] = w1;
        return v;
    };
    auto loadB = [&](int col, int k0) {
        uint8_t tmp[8];
        for (int s = 0; s < 8; s++) tmp[s] = B[(k0 + s) * 16 + col];
        i32x2 v; int32_t w0, w1;
        __builtin_memcpy(&w0, tmp, 4); __builtin_memcpy(&w1, tmp + 4, 4);
        v[0] = w0; v[1] = w1;
        return v;
    };
    i32x2 a, b;
    if (ia == 0) { int r = L / 2, k0 = 8 * (L % 2); a = loadA(r, k0); }
    else { int r = L % 16, k0 = 8 * (L / 16); a = loadA(r, k0); }
    if (ib == 0) { int c = L / 2, k0 = 8 * (L % 2); b = loadB(c, k0); }
    else { int c = L % 16, k0 = 8 * (L / 16); b = loadB(c, k0); }
    f32x8 acc; for (int s = 0; s < 8; s++) acc[s] = 0.0f;
    if (ord == 0) acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
    else acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b, a, acc);
    if (ic == 0) { int r = L / 2, c0 = 8 * (L % 2);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s];
    } else { int r = L % 16, c0 = 8 * (L / 16);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s];
    }
}

// ---------------------------------------------------------------- rate kernels
// Uniform operands in registers; 4 independent accumulators.
__global__ void rate_f16(float* out, int iters) {
    f16x8 a, b;
    for (int s = 0; s < 8; s++) { a[s] = (_Float16)2.0f; b[s] = (_Float16)3.0f; }
    f32x8 c0, c1, c2, c3;
    // Distinct inits defeat cross-accumulator CSE (else clang computes once).
    for (int s = 0; s < 8; s++) { c0[s] = 0.0f; c1[s] = 1.0f; c2[s] = 2.0f; c3[s] = 3.0f; }
    for (int t = 0; t < iters; t++) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c3);
    }
    int L = threadIdx.x & 31;
    int wave = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    float t = 0;
    for (int s = 0; s < 8; s++) t += c0[s] + c1[s] + c2[s] + c3[s];
    out[(size_t)wave * 32 + L] = t;
}

__global__ void rate_i8(int* out, int iters) {
    i32x2 a, b;
    a[0] = a[1] = 0x02020202; b[0] = b[1] = 0x03030303;
    i32x8 c0, c1, c2, c3;
    for (int s = 0; s < 8; s++) { c0[s] = 0; c1[s] = 10; c2[s] = 20; c3[s] = 30; }
    for (int t = 0; t < iters; t++) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c3, false);
    }
    int L = threadIdx.x & 31;
    int wave = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int t = 0;
    for (int s = 0; s < 8; s++) t += c0[s] + c1[s] + c2[s] + c3[s];
    out[(size_t)wave * 32 + L] = t;
}

// E4M3: 2.0 = 0x48, 3.0 = 0x4C
__global__ void rate_fp8(float* out, int iters) {
    i32x2 a, b;
    a[0] = a[1] = 0x48484848; b[0] = b[1] = 0x4C4C4C4C;
    f32x8 c0, c1, c2, c3;
    for (int s = 0; s < 8; s++) { c0[s] = 0.0f; c1[s] = 1.0f; c2[s] = 2.0f; c3[s] = 3.0f; }
    for (int t = 0; t < iters; t++) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c3);
    }
    int L = threadIdx.x & 31;
    int wave = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    float t = 0;
    for (int s = 0; s < 8; s++) t += c0[s] + c1[s] + c2[s] + c3[s];
    out[(size_t)wave * 32 + L] = t;
}

#define CHECK(x) do { hipError_t __he = (x); if (__he != hipSuccess) { \
    printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(__he)); return 1; } } while (0)

int main() {
    int ndev = 0; CHECK(hipGetDeviceCount(&ndev));
    int dev = -1;
    for (int i = 0; i < ndev; i++) {
        hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, i));
        printf("hip device %d: %s %s warpSize=%d\n", i, p.name, p.gcnArchName, p.warpSize);
        if (strstr(p.gcnArchName, "gfx1201")) dev = i;
    }
    if (dev < 0) { printf("no gfx1201 device found\n"); return 1; }
    CHECK(hipSetDevice(dev));

    // ---- battery inputs ----
    _Float16 hA[256], hB[256];
    int8_t iA[256], iB[256];
    uint8_t fA[256], fB[256];
    float ref_f[256]; int ref_i[256]; float ref_fp8[256];
    // E4M3-exact small values
    const float vals[6] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 16; i++) for (int k = 0; k < 16; k++) {
        hA[i * 16 + k] = (_Float16)((i * 16 + k) * 0.25f);
        hB[k * 16 + i] = (_Float16)((k * 16 + i) * 0.125f + 1.0f);
        iA[i * 16 + k] = (int8_t)((i * 16 + k) % 13 - 6);
        iB[k * 16 + i] = (int8_t)((k * 16 + i + 5) % 11 - 5);
        fA[i * 16 + k] = 0; // filled below via float->bits map
        fB[k * 16 + i] = 0;
    }
    float fAf[256], fBf[256];
    for (int i = 0; i < 16; i++) for (int k = 0; k < 16; k++) {
        fAf[i * 16 + k] = vals[(i + k) % 6];
        fBf[k * 16 + i] = vals[(i * 3 + k) % 6];
    }
    auto f2e4m3 = [](float v) -> uint8_t {
        if (v == 0.5f) return 0x38; if (v == 1.0f) return 0x40;
        if (v == 1.5f) return 0x44; if (v == 2.0f) return 0x48;
        if (v == 3.0f) return 0x4C; if (v == 4.0f) return 0x50;
        return 0x00;
    };
    for (int n = 0; n < 256; n++) { fA[n] = f2e4m3(fAf[n]); fB[n] = f2e4m3(fBf[n]); }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) {
        double s = 0, si = 0, sf = 0;
        for (int k = 0; k < 16; k++) {
            s += (double)(float)hA[i * 16 + k] * (double)(float)hB[k * 16 + j];
            si += (double)iA[i * 16 + k] * (double)iB[k * 16 + j];
            sf += (double)fAf[i * 16 + k] * (double)fBf[k * 16 + j];
        }
        ref_f[i * 16 + j] = (float)s; ref_i[i * 16 + j] = (int)si; ref_fp8[i * 16 + j] = (float)sf;
    }

    _Float16 *dA, *dB; float *dC; int8_t *dAi, *dBi; int *dCi; uint8_t *dAf, *dBf;
    CHECK(hipMalloc(&dA, sizeof(hA))); CHECK(hipMalloc(&dB, sizeof(hB)));
    CHECK(hipMalloc(&dC, 256 * 4));
    CHECK(hipMalloc(&dAi, sizeof(iA))); CHECK(hipMalloc(&dBi, sizeof(iB)));
    CHECK(hipMalloc(&dCi, 256 * 4));
    CHECK(hipMalloc(&dAf, sizeof(fA))); CHECK(hipMalloc(&dBf, sizeof(fB)));
    CHECK(hipMemcpy(dA, hA, sizeof(hA), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, hB, sizeof(hB), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dAi, iA, sizeof(iA), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dBi, iB, sizeof(iB), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dAf, fA, sizeof(fA), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dBf, fB, sizeof(fB), hipMemcpyHostToDevice));

    float hC[256]; int hCi[256];
    int pass_f = 0, pass_i = 0, pass_fp8 = 0;
    for (int ia = 0; ia < 2; ia++) for (int ib = 0; ib < 2; ib++)
    for (int ord = 0; ord < 2; ord++) for (int ic = 0; ic < 2; ic++) {
        hipMemset(dC, 0, 256 * 4);
        hipLaunchKernelGGL(probe_f16, dim3(1), dim3(32), 0, 0, dA, dB, dC, ia, ib, ord, ic);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost));
        float mx = 0; for (int n = 0; n < 256; n++) mx = fmaxf(mx, fabsf(hC[n] - ref_f[n]));
        if (mx < 0.5f) { printf("F16  PASS ia=%d ib=%d ord=%d ic=%d maxerr=%.4f\n", ia, ib, ord, ic, mx); pass_f++; }

        hipMemset(dCi, 0, 256 * 4);
        hipLaunchKernelGGL(probe_i8, dim3(1), dim3(32), 0, 0, dAi, dBi, dCi, ia, ib, ord, ic);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hCi, dCi, sizeof(hCi), hipMemcpyDeviceToHost));
        int mxi = 0; for (int n = 0; n < 256; n++) mxi = abs(hCi[n] - ref_i[n]) > mxi ? abs(hCi[n] - ref_i[n]) : mxi;
        if (mxi == 0) { printf("I8   PASS ia=%d ib=%d ord=%d ic=%d\n", ia, ib, ord, ic); pass_i++; }

        hipMemset(dC, 0, 256 * 4);
        hipLaunchKernelGGL(probe_fp8, dim3(1), dim3(32), 0, 0, dAf, dBf, dC, ia, ib, ord, ic);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost));
        float mx8 = 0; for (int n = 0; n < 256; n++) mx8 = fmaxf(mx8, fabsf(hC[n] - ref_fp8[n]));
        if (mx8 < 0.05f) { printf("FP8  PASS ia=%d ib=%d ord=%d ic=%d maxerr=%.4f\n", ia, ib, ord, ic, mx8); pass_fp8++; }
    }
    printf("battery summary: f16=%d i8=%d fp8=%d /16 combos\n", pass_f, pass_i, pass_fp8);

    // ---- rate measurement ----
    const int WAVES = 4096, ITERS = 2048;
    float *dRo; int *dRi;
    CHECK(hipMalloc(&dRo, (size_t)WAVES * 32 * 4));
    CHECK(hipMalloc(&dRi, (size_t)WAVES * 32 * 4));
    hipEvent_t s, e; CHECK(hipEventCreate(&s)); CHECK(hipEventCreate(&e));
    auto bench = [&](const char* name, auto launch, double ops, size_t outb, void* out) {
        CHECK(hipMemset(out, 0, outb));
        // warmup
        for (int w = 0; w < 5; w++) { launch(); }
        CHECK(hipDeviceSynchronize());
        float best = 1e30f;
        for (int r = 0; r < 7; r++) {
            CHECK(hipEventRecord(s));
            launch();
            CHECK(hipEventRecord(e));
            CHECK(hipEventSynchronize(e));
            float ms = 0; CHECK(hipEventElapsedTime(&ms, s, e));
            if (ms < best) best = ms;
        }
        double tops = (double)WAVES * (ITERS * 4) * ops / (best / 1000.0) / 1e12;
        printf("RATE %-6s best=%8.3f ms  -> %7.2f TOPS (ops/wm=%g)\n", name, best, tops, ops);
    };
    dim3 blocks((WAVES * 32 + 127) / 128); dim3 thr(128);
    bench("fp16", [&]{ hipLaunchKernelGGL(rate_f16, blocks, thr, 0, 0, dRo, ITERS); }, 8192.0, (size_t)WAVES * 32 * 4, dRo);
    bench("int8", [&]{ hipLaunchKernelGGL(rate_i8, blocks, thr, 0, 0, dRi, ITERS); }, 8192.0, (size_t)WAVES * 32 * 4, dRi);
    bench("fp8", [&]{ hipLaunchKernelGGL(rate_fp8, blocks, thr, 0, 0, dRo, ITERS); }, 8192.0, (size_t)WAVES * 32 * 4, dRo);
    // correctness of rate kernels (uniform): per lane sum over 4 accs x 8 slots,
    // each slot = ITERS*96 + init; inits {0,1,2,3} (fp) / {0,10,20,30} (int).
    float hRo[32]; int hRi[32];
    CHECK(hipMemcpy(hRo, dRo, sizeof(float) * 32, hipMemcpyDeviceToHost));
    // dRo currently holds fp8 results (last bench); fp8: 6.0*16=96 per slot.
    printf("fp8 uniform sample: %.1f (expect %.1f)\n", hRo[0],
           (float)(8 * (4 * ITERS * 96) + 8 * 6));
    printf("ALL_DONE\n");
    return 0;
}
