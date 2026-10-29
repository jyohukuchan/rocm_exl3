// Round-3 probe2: single-shot uniform WMMA dumps (full 16x16 C) for
// fp16/int8/fp8 gfx12 w32 builtins + chrono-based rate loops.
// Build: hipcc -O3 -o probe2 probe2.hip --offload-arch=gfx1201
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
#define CHECK(x) do { hipError_t __he = (x); if (__he != hipSuccess) { \
    printf("HIP error %s:%d: %s (%d)\n", __FILE__, __LINE__, hipGetErrorString(__he), (int)__he); \
    fflush(stdout); return 1; } } while (0)

// One wave, uniform operands, store full C with the fp16-winning layout
// (row=L%16, col=8*(L/16)+s).
__global__ void one_f16(const _Float16* A, const _Float16* B, float* C) {
    int L = threadIdx.x & 31;
    f16x8 a, b;
    { int r = L % 16, k0 = 8 * (L / 16);
      for (int s = 0; s < 8; s++) a[s] = A[r * 16 + k0 + s]; }
    { int c = L % 16, k0 = 8 * (L / 16);
      for (int s = 0; s < 8; s++) b[s] = B[(k0 + s) * 16 + c]; }
    f32x8 acc; for (int s = 0; s < 8; s++) acc[s] = 0.0f;
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b, a, acc);
    { int r = L % 16, c0 = 8 * (L / 16);
      for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s]; }
}
__global__ void one_i8(const int8_t* A, const int8_t* B, int* C) {
    int L = threadIdx.x & 31;
    i32x2 a, b;
    { int r = L % 16, k0 = 8 * (L / 16);
      int32_t w0, w1;
      __builtin_memcpy(&w0, A + r * 16 + k0, 4);
      __builtin_memcpy(&w1, A + r * 16 + k0 + 4, 4);
      a[0] = w0; a[1] = w1; }
    { int c = L % 16, k0 = 8 * (L / 16);
      uint8_t t[8]; for (int s = 0; s < 8; s++) t[s] = (uint8_t)B[(k0 + s) * 16 + c];
      int32_t w0, w1;
      __builtin_memcpy(&w0, t, 4); __builtin_memcpy(&w1, t + 4, 4);
      b[0] = w0; b[1] = w1; }
    i32x8 acc; for (int s = 0; s < 8; s++) acc[s] = 0;
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, acc, false);
    { int r = L % 16, c0 = 8 * (L / 16);
      for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s]; }
}
__global__ void one_fp8(const uint8_t* A, const uint8_t* B, float* C) {
    int L = threadIdx.x & 31;
    i32x2 a, b;
    { int r = L % 16, k0 = 8 * (L / 16);
      int32_t w0, w1;
      __builtin_memcpy(&w0, A + r * 16 + k0, 4);
      __builtin_memcpy(&w1, A + r * 16 + k0 + 4, 4);
      a[0] = w0; a[1] = w1; }
    { int c = L % 16, k0 = 8 * (L / 16);
      uint8_t t[8]; for (int s = 0; s < 8; s++) t[s] = B[(k0 + s) * 16 + c];
      int32_t w0, w1;
      __builtin_memcpy(&w0, t, 4); __builtin_memcpy(&w1, t + 4, 4);
      b[0] = w0; b[1] = w1; }
    f32x8 acc; for (int s = 0; s < 8; s++) acc[s] = 0.0f;
    acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
    { int r = L % 16, c0 = 8 * (L / 16);
      for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s]; }
}
// Rate loops with distinct accumulator inits (anti-CSE), chrono timing.
__global__ void loop_f16(float* out, int iters) {
    f16x8 a, b;
    for (int s = 0; s < 8; s++) { a[s] = (_Float16)2.0f; b[s] = (_Float16)3.0f; }
    f32x8 c0, c1, c2, c3;
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
__global__ void loop_i8(int* out, int iters) {
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
__global__ void loop_fp8(float* out, int iters) {
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
int main() {
    CHECK(hipSetDevice(0)); // isolated R9700 via HIP_VISIBLE_DEVICES
    hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
    printf("device: %s %s warpSize=%d\n", p.name, p.gcnArchName, p.warpSize);
    // uniform inputs: A=2, B=3 (int8: 2,3; fp8: 0x48=2.0, 0x4C=3.0) -> 96/elem
    _Float16 hA[256], hB[256]; int8_t iA[256], iB[256]; uint8_t fA[256], fB[256];
    for (int n = 0; n < 256; n++) {
        hA[n] = (_Float16)2.0f; hB[n] = (_Float16)3.0f;
        iA[n] = 2; iB[n] = 3; fA[n] = 0x48; fB[n] = 0x4C;
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
    hipMemset(dC, 0, 256 * 4);
    hipLaunchKernelGGL(one_f16, dim3(1), dim3(32), 0, 0, dA, dB, dC);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost));
    printf("one_f16: C[0]=%.1f C[5]=%.1f C[250]=%.1f (expect 96)\n", hC[0], hC[5], hC[250]);
    hipMemset(dCi, 0, 256 * 4);
    hipLaunchKernelGGL(one_i8, dim3(1), dim3(32), 0, 0, dAi, dBi, dCi);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(hCi, dCi, sizeof(hCi), hipMemcpyDeviceToHost));
    printf("one_i8: C[0]=%d C[5]=%d C[250]=%d (expect 96)\n", hCi[0], hCi[5], hCi[250]);
    hipMemset(dC, 0, 256 * 4);
    hipLaunchKernelGGL(one_fp8, dim3(1), dim3(32), 0, 0, dAf, dBf, dC);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost));
    printf("one_fp8: C[0]=%.1f C[5]=%.1f C[250]=%.1f (expect 96)\n", hC[0], hC[5], hC[250]);
    // rate via chrono
    const int WAVES = 4096, ITERS = 2048;
    float *dRo; int *dRi;
    CHECK(hipMalloc(&dRo, (size_t)WAVES * 32 * 4));
    CHECK(hipMalloc(&dRi, (size_t)WAVES * 32 * 4));
    dim3 blocks((WAVES * 32 + 127) / 128); dim3 thr(128);
    auto bench = [&](const char* nm, auto fn) {
        for (int w = 0; w < 3; w++) { fn(); CHECK(hipDeviceSynchronize()); }
        double best = 1e30;
        for (int r = 0; r < 7; r++) {
            auto t0 = std::chrono::steady_clock::now();
            fn();
            CHECK(hipDeviceSynchronize());
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        double tops = (double)WAVES * (ITERS * 4) * 8192.0 / (best / 1000.0) / 1e12;
        printf("RATE %-5s best=%8.3f ms -> %8.2f TOPS\n", nm, best, tops);
        fflush(stdout);
    };
    bench("fp16", [&]{ hipLaunchKernelGGL(loop_f16, blocks, thr, 0, 0, dRo, ITERS); });
    bench("int8", [&]{ hipLaunchKernelGGL(loop_i8, blocks, thr, 0, 0, dRi, ITERS); });
    bench("fp8", [&]{ hipLaunchKernelGGL(loop_fp8, blocks, thr, 0, 0, dRo, ITERS); });
    float s0; int si0;
    CHECK(hipMemcpy(&s0, dRo, 4, hipMemcpyDeviceToHost));
    printf("loop_fp8 lane0 sum=%.1f (expect %.1f)\n", s0, (float)(8 * (4 * ITERS * 96) + 48));
    printf("PROBE2_DONE\n");
    return 0;
}
