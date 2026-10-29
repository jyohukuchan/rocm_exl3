// Round-3 probe3: pin down int8 signed-flag semantics + fp8 layout/linearity.
//  - int8 battery: flags (fa,fb) in {(T,T),(T,F),(F,T),(F,F)} x layout(ia,ib,ord,ic)
//    with SIGNED test data -> expect exactly one PASS family.
//  - fp8 battery: same 16 layouts vs 4x reference (probe2 showed uniform 4x).
//  - fp8 linearity: (A,B) in {(1,1),(0.5,2),(1.5,4),(3,3)} uniform -> C/96?
//  - rates: chrono, longer warmup, 2 ITERS settings to check scaling.
// Build: hipcc -O3 -o probe3 probe3.hip --offload-arch=gfx1201
#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
static void chk(hipError_t e, int line) {
    if (e != hipSuccess) { printf("HIP error line %d: %s (%d)\n", line, hipGetErrorString(e), (int)e); fflush(stdout); exit(1); }
}
#define CHECK(x) chk((x), __LINE__)
template <bool FA, bool FB>
__global__ void probe_i8(const int8_t* A, const int8_t* B, int* C,
                         int ia, int ib, int ord, int ic) {
    int L = threadIdx.x & 31;
    i32x2 a, b;
    if (ia == 0) { int r = L / 2, k0 = 8 * (L % 2);
        int32_t w0, w1; __builtin_memcpy(&w0, A + r * 16 + k0, 4);
        __builtin_memcpy(&w1, A + r * 16 + k0 + 4, 4); a[0] = w0; a[1] = w1; }
    else { int r = L % 16, k0 = 8 * (L / 16);
        int32_t w0, w1; __builtin_memcpy(&w0, A + r * 16 + k0, 4);
        __builtin_memcpy(&w1, A + r * 16 + k0 + 4, 4); a[0] = w0; a[1] = w1; }
    if (ib == 0) { int c = L / 2, k0 = 8 * (L % 2);
        uint8_t t[8]; for (int s = 0; s < 8; s++) t[s] = (uint8_t)B[(k0 + s) * 16 + c];
        int32_t w0, w1; __builtin_memcpy(&w0, t, 4); __builtin_memcpy(&w1, t + 4, 4);
        b[0] = w0; b[1] = w1; }
    else { int c = L % 16, k0 = 8 * (L / 16);
        uint8_t t[8]; for (int s = 0; s < 8; s++) t[s] = (uint8_t)B[(k0 + s) * 16 + c];
        int32_t w0, w1; __builtin_memcpy(&w0, t, 4); __builtin_memcpy(&w1, t + 4, 4);
        b[0] = w0; b[1] = w1; }
    i32x8 acc; for (int s = 0; s < 8; s++) acc[s] = 0;
    if (ord == 0) acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(FA, a, FB, b, acc, false);
    else acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(FB, b, FA, a, acc, false);
    if (ic == 0) { int r = L / 2, c0 = 8 * (L % 2);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s]; }
    else { int r = L % 16, c0 = 8 * (L / 16);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s]; }
}
__global__ void probe_fp8(const uint8_t* A, const uint8_t* B, float* C,
                          int ia, int ib, int ord, int ic) {
    int L = threadIdx.x & 31;
    i32x2 a, b;
    if (ia == 0) { int r = L / 2, k0 = 8 * (L % 2);
        int32_t w0, w1; __builtin_memcpy(&w0, A + r * 16 + k0, 4);
        __builtin_memcpy(&w1, A + r * 16 + k0 + 4, 4); a[0] = w0; a[1] = w1; }
    else { int r = L % 16, k0 = 8 * (L / 16);
        int32_t w0, w1; __builtin_memcpy(&w0, A + r * 16 + k0, 4);
        __builtin_memcpy(&w1, A + r * 16 + k0 + 4, 4); a[0] = w0; a[1] = w1; }
    if (ib == 0) { int c = L / 2, k0 = 8 * (L % 2);
        uint8_t t[8]; for (int s = 0; s < 8; s++) t[s] = B[(k0 + s) * 16 + c];
        int32_t w0, w1; __builtin_memcpy(&w0, t, 4); __builtin_memcpy(&w1, t + 4, 4);
        b[0] = w0; b[1] = w1; }
    else { int c = L % 16, k0 = 8 * (L / 16);
        uint8_t t[8]; for (int s = 0; s < 8; s++) t[s] = B[(k0 + s) * 16 + c];
        int32_t w0, w1; __builtin_memcpy(&w0, t, 4); __builtin_memcpy(&w1, t + 4, 4);
        b[0] = w0; b[1] = w1; }
    f32x8 acc; for (int s = 0; s < 8; s++) acc[s] = 0.0f;
    if (ord == 0) acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
    else acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b, a, acc);
    if (ic == 0) { int r = L / 2, c0 = 8 * (L % 2);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s]; }
    else { int r = L % 16, c0 = 8 * (L / 16);
        for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s]; }
}
__global__ void one_fp8val(const uint8_t* A, const uint8_t* B, float* C) {
    int L = threadIdx.x & 31;
    i32x2 a, b;
    { int32_t w0, w1; __builtin_memcpy(&w0, A + L * 2, 2); w1 = 0;
      // simpler: broadcast A[0],B[0] to all slots
      uint8_t av = A[0], bv = B[0];
      uint8_t ta[8], tb[8];
      for (int s = 0; s < 8; s++) { ta[s] = av; tb[s] = bv; }
      __builtin_memcpy(&w0, ta, 4); __builtin_memcpy(&w1, ta + 4, 4);
      a[0] = w0; a[1] = w1;
      __builtin_memcpy(&w0, tb, 4); __builtin_memcpy(&w1, tb + 4, 4);
      b[0] = w0; b[1] = w1; }
    f32x8 acc; for (int s = 0; s < 8; s++) acc[s] = 0.0f;
    acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc);
    { int r = L % 16, c0 = 8 * (L / 16);
      for (int s = 0; s < 8; s++) C[r * 16 + c0 + s] = acc[s]; }
}
int main() {
    CHECK(hipSetDevice(0));
    int8_t iA[256], iB[256]; uint8_t fA[256], fB[256];
    int ref_i[256]; float ref_fp8[256];
    const float vals[6] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
    float fAf[256], fBf[256];
    auto f2e4m3 = [](float v) -> uint8_t {
        if (v == 0.5f) return 0x38; if (v == 1.0f) return 0x40;
        if (v == 1.5f) return 0x44; if (v == 2.0f) return 0x48;
        if (v == 3.0f) return 0x4C; if (v == 4.0f) return 0x50;
        return 0x00;
    };
    for (int i = 0; i < 16; i++) for (int k = 0; k < 16; k++) {
        iA[i * 16 + k] = (int8_t)((i * 16 + k) % 13 - 6);
        iB[k * 16 + i] = (int8_t)((k * 16 + i + 5) % 11 - 5);
        fAf[i * 16 + k] = vals[(i + k) % 6];
        fBf[k * 16 + i] = vals[(i * 3 + k) % 6];
    }
    for (int n = 0; n < 256; n++) { fA[n] = f2e4m3(fAf[n]); fB[n] = f2e4m3(fBf[n]); }
    for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) {
        double si = 0, sf = 0;
        for (int k = 0; k < 16; k++) {
            si += (double)iA[i * 16 + k] * (double)iB[k * 16 + j];
            sf += (double)fAf[i * 16 + k] * (double)fBf[k * 16 + j];
        }
        ref_i[i * 16 + j] = (int)si; ref_fp8[i * 16 + j] = (float)(4.0 * sf);
    }
    int8_t *dAi, *dBi; int *dCi; uint8_t *dAf, *dBf; float *dC;
    CHECK(hipMalloc(&dAi, sizeof(iA))); CHECK(hipMalloc(&dBi, sizeof(iB)));
    CHECK(hipMalloc(&dCi, 256 * 4));
    CHECK(hipMalloc(&dAf, sizeof(fA))); CHECK(hipMalloc(&dBf, sizeof(fB)));
    CHECK(hipMalloc(&dC, 256 * 4));
    CHECK(hipMemcpy(dAi, iA, sizeof(iA), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dBi, iB, sizeof(iB), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dAf, fA, sizeof(fA), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dBf, fB, sizeof(fB), hipMemcpyHostToDevice));
    int hCi[256]; float hC[256];
    int pass_i = 0;
    for (int ia = 0; ia < 2; ia++) for (int ib = 0; ib < 2; ib++)
    for (int ord = 0; ord < 2; ord++) for (int ic = 0; ic < 2; ic++) {
        // (FA,FB): sign flags pair with the FOLLOWING vector arg; first vector is B-side.
        CHECK(hipMemset(dCi, 0, 256 * 4));
        hipLaunchKernelGGL((probe_i8<true, true>), dim3(1), dim3(32), 0, 0, dAi, dBi, dCi, ia, ib, ord, ic);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hCi, dCi, sizeof(hCi), hipMemcpyDeviceToHost));
        int bad = 0; for (int n = 0; n < 256; n++) if (hCi[n] != ref_i[n]) bad++;
        if (bad == 0) { printf("I8 PASS fa=1 fb=1 ia=%d ib=%d ord=%d ic=%d\n", ia, ib, ord, ic); pass_i++; }
        CHECK(hipMemset(dCi, 0, 256 * 4));
        hipLaunchKernelGGL((probe_i8<true, false>), dim3(1), dim3(32), 0, 0, dAi, dBi, dCi, ia, ib, ord, ic);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hCi, dCi, sizeof(hCi), hipMemcpyDeviceToHost));
        bad = 0; for (int n = 0; n < 256; n++) if (hCi[n] != ref_i[n]) bad++;
        if (bad == 0) { printf("I8 PASS fa=1 fb=0 ia=%d ib=%d ord=%d ic=%d\n", ia, ib, ord, ic); pass_i++; }
        CHECK(hipMemset(dCi, 0, 256 * 4));
        hipLaunchKernelGGL((probe_i8<false, true>), dim3(1), dim3(32), 0, 0, dAi, dBi, dCi, ia, ib, ord, ic);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hCi, dCi, sizeof(hCi), hipMemcpyDeviceToHost));
        bad = 0; for (int n = 0; n < 256; n++) if (hCi[n] != ref_i[n]) bad++;
        if (bad == 0) { printf("I8 PASS fa=0 fb=1 ia=%d ib=%d ord=%d ic=%d\n", ia, ib, ord, ic); pass_i++; }
        CHECK(hipMemset(dCi, 0, 256 * 4));
        hipLaunchKernelGGL((probe_i8<false, false>), dim3(1), dim3(32), 0, 0, dAi, dBi, dCi, ia, ib, ord, ic);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hCi, dCi, sizeof(hCi), hipMemcpyDeviceToHost));
        bad = 0; for (int n = 0; n < 256; n++) if (hCi[n] != ref_i[n]) bad++;
        if (bad == 0) { printf("I8 PASS fa=0 fb=0 ia=%d ib=%d ord=%d ic=%d\n", ia, ib, ord, ic); pass_i++; }
    }
    printf("int8 battery: %d/64\n", pass_i);
    int pass_f = 0;
    for (int ia = 0; ia < 2; ia++) for (int ib = 0; ib < 2; ib++)
    for (int ord = 0; ord < 2; ord++) for (int ic = 0; ic < 2; ic++) {
        CHECK(hipMemset(dC, 0, 256 * 4));
        hipLaunchKernelGGL(probe_fp8, dim3(1), dim3(32), 0, 0, dAf, dBf, dC, ia, ib, ord, ic);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost));
        float mx = 0; for (int n = 0; n < 256; n++) { float d = hC[n] - ref_fp8[n]; if (d < 0) d = -d; if (d > mx) mx = d; }
        if (mx < 0.5f) { printf("FP8 PASS ia=%d ib=%d ord=%d ic=%d maxerr=%.3f\n", ia, ib, ord, ic, mx); pass_f++; }
    }
    printf("fp8 battery (vs 4x ref): %d/16\n", pass_f);
    // linearity: uniform (A,B) pairs
    struct Pair { float a, b; uint8_t ab, bb; };
    Pair pairs[4] = {{1,1,0x40,0x40},{0.5f,2,0x38,0x48},{1.5f,4,0x44,0x50},{3,3,0x4C,0x4C}};
    for (int t = 0; t < 4; t++) {
        uint8_t uA = pairs[t].ab, uB = pairs[t].bb;
        CHECK(hipMemcpy(dAf, &uA, 1, hipMemcpyHostToDevice)); // broadcast via kernel below
        uint8_t h1[256]; for (int n = 0; n < 256; n++) h1[n] = uA;
        uint8_t h2[256]; for (int n = 0; n < 256; n++) h2[n] = uB;
        CHECK(hipMemcpy(dAf, h1, sizeof(h1), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dBf, h2, sizeof(h2), hipMemcpyHostToDevice));
        CHECK(hipMemset(dC, 0, 256 * 4));
        hipLaunchKernelGGL(one_fp8val, dim3(1), dim3(32), 0, 0, dAf, dBf, dC);
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemcpy(hC, dC, sizeof(hC), hipMemcpyDeviceToHost));
        printf("fp8 lin A=%.1f B=%.1f: C=%.1f (plain expect %.1f, 4x expect %.1f)\n",
               pairs[t].a, pairs[t].b, hC[0], 16 * pairs[t].a * pairs[t].b, 64 * pairs[t].a * pairs[t].b);
    }
    printf("PROBE3_DONE\n");
    return 0;
}
