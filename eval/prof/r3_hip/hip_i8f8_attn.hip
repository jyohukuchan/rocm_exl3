// Round-3 HIP prototype: cached-format INT8-K / FP8-V prefill attention.
//
// Reads CacheLayer_i8f8 packed tensors directly (k_q int8 + k_sc fp32 +
// k_mu fp16 + v_q fp8 + v_sc fp16 + fp16 stage + stage_len) through a
// block_table, exactly like the Triton cached kernel. Flash structure with
// online softmax; QK^ T via gfx1201 WMMA i32 16x16x16 iu8 (signed x signed),
// PV via gfx1201 WMMA f32 16x16x16 fp8 fp8 (known 4x hardware scale,
// compensated with FP8_WMMA_RSCALE). Lane mapping (row=L%16, k-slice by
// wave half, operand order (B,A,C)) verified by probe3 on R9700.
//
// Scope (prototype): B=1, head_dim=256, BLOCK_M=16 rows/block, BLOCK_N=32
// (= one quant block, so scales/mu line up), causal or not, last <32 rows
// via a scalar fp32 tail (correct, slow). GQA: one block per q-head; K/V
// reloads are shared through L2 across the group.
//
// Build (in the R9700 container):
//   python3 bench script -> torch.utils.cpp_extension.load -> hipcc gfx1201
#include <hip/hip_runtime.h>
#include <cuda_fp16.h>

typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x8 __attribute__((ext_vector_type(8)));

// E4M3 on this stack (torch + gfx1201 WMMA, verified by probe3): 0x38 = 1.0,
// i.e. effective exponent bias 7 (tiny = 2^-6, max 0x7E = 448). The earlier
// "4x WMMA scale" reading was this bias misread: 0x48 is 4.0, not 2.0.
// No scale correction is needed.
#define FP8_WMMA_RSCALE 1.0f

// Round-half-to-even fp32 -> E4M3 (probe-tested against torch on R9700 via
// the test_e4m3 binding below). Input must be >= 0 (P weights); finite
// inputs >= ~460 saturate like torch only up to its 0x7E region (torch emits
// 0x7F past ~460; P never exceeds 448 here).
__device__ __forceinline__ uint8_t f32_to_e4m3(float x) {
    uint32_t u; __builtin_memcpy(&u, &x, 4);
    uint32_t ax = u & 0x7FFFFFFFu;
    if (ax < 0x00800000u) return 0x00;   // zero / f32-subnormal -> 0
    if (u != ax) return 0x00;            // negative (P never is; guard)
    if (ax >= 0x7F800000u) return 0x7E;  // inf/nan -> max
    int E = (int)(ax >> 23) - 127;
    uint32_t mant = ax & 0x7FFFFFu;
    if (E >= -6) {
        if (E > 8) return 0x7E;
        uint32_t m = 0x800000u | mant;
        uint32_t q = m >> 20;            // 4-bit 1.xxx (8..15)
        uint32_t rem = m & 0xFFFFFu;
        if (rem > 0x80000u || (rem == 0x80000u && (q & 1u))) {
            q++;
            if (q > 15u) { q = 8u; E++; if (E > 8) return 0x7E; }
        }
        uint32_t f = q - 8u;
        int Ef = E + 7;
        if (Ef == 15 && f == 7u) return 0x7E; // NaN pattern -> max
        return (uint8_t)((Ef << 3) | f);
    } else {
        // E4M3 subnormal (unit 2^-9): f = RNE(x / 2^-9), x = M * 2^(E-23)
        int sh = 14 - E; // >= 21
        if (sh > 25) return 0x00;
        uint32_t M = 0x800000u | mant;
        uint32_t q = (sh >= 24) ? 0u : (M >> sh);
        uint32_t half = (M >> (sh - 1)) & 1u;
        uint32_t rmask = (sh - 1 == 0) ? 0u : ((1u << (sh - 1)) - 1u);
        uint32_t rest = M & rmask;
        uint32_t f = q + ((half && (rest || (q & 1u))) ? 1u : 0u);
        if (E == -7 && f >= 8u) return 0x08; // rounds up to smallest normal
        if (f > 7u) f = 7u;
        return (uint8_t)f;
    }
}

// Round-half-away quantize to int8 lane (matches round-2 _quant_i8).
__device__ __forceinline__ int q_i8(float x) {
    float r = x >= 0.0f ? x + 0.5f : x - 0.5f;
    int v = (int)r;
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    return v;
}

#define D_HEAD 256
#define BM 16
#define BN 32
#define D2C 64
#define NDC (D_HEAD / D2C)

// One block = 16 Q rows of one q-head. Single wave (32 threads).
__global__ __launch_bounds__(32) void hip_i8f8_prefill_kernel(
    const half* __restrict__ q,        // (Tq, hq, D) fp16
    const int8_t* __restrict__ kq,       // (P, 256, hkv, D) int8
    const float* __restrict__ ksc,       // (P, 8, hkv) fp32
    const half* __restrict__ kmu,      // (P, 8, hkv, D) fp16
    const uint8_t* __restrict__ vq,      // (P, 256, hkv, D) fp8 bits
    const half* __restrict__ vsc,      // (P, 8, hkv, D) fp16
    const half* __restrict__ kst,      // (P, 32, hkv, D) fp16 stage
    const half* __restrict__ vst,      // (P, 32, hkv, D) fp16 stage
    const int* __restrict__ slen,        // (P,) int32
    const int* __restrict__ block_table, // (nPages,) int32 (B=1)
    half* __restrict__ out,            // (Tq, hq, D) fp16
    int Tq, int Tk, int past,
    int hq, int hkv, float sm_scale, int causal) {
    const int L = threadIdx.x; // 0..31, wave32 assumed (verified warpSize=32)
    const int mTile = blockIdx.x;
    const int h = blockIdx.y;
    const int G = hq / hkv;
    const int kh = h / G;
    const int m0 = mTile * BM;
    if (m0 >= Tq) return;
    const int myRow = L % 16;            // Q row handled by this lane (shared w/ L^16)

    // ---- LDS: Q fp16 (8KB) + Q int8 (4KB) + mu (512B) + O fp32 (16KB)
    //         + K tile int8 (8KB) + V tile fp8 (2KB) ----
    __shared__ half Qf[BM][D_HEAD];
    __shared__ int8_t Qi[BM][D_HEAD];
    __shared__ half Mu[D_HEAD];
    __shared__ float Oacc[BM][D_HEAD];
    __shared__ int8_t Ktile[BN][D_HEAD];
    __shared__ uint8_t Vtile[BN][D2C];

    // Q rows: flat cyclic load (128 fp16 per lane), then per-row amax.
    float prow_max[BM];
    #pragma unroll
    for (int r = 0; r < BM; r++) prow_max[r] = 0.0f;
    for (int t = 0; t < 128; t++) {
        int flat = L + t * 32;
        int r = flat / D_HEAD, d = flat % D_HEAD;
        int grow = m0 + r;
        half v = __float2half(0.0f);
        if (grow < Tq) v = q[((size_t)grow * hq + h) * D_HEAD + d];
        Qf[r][d] = v;
        float a = __half2float(v);
        if (a < 0) a = -a;
        if (a > prow_max[r]) prow_max[r] = a;
    }
    __syncthreads();
    // row max over the wave (butterfly), then Qi quantize (same elements).
    float sq[BM];
    #pragma unroll
    for (int r = 0; r < BM; r++) {
        float v = prow_max[r];
        v = fmaxf(v, __shfl(v, L ^ 16));
        v = fmaxf(v, __shfl(v, L ^ 8));
        v = fmaxf(v, __shfl(v, L ^ 4));
        v = fmaxf(v, __shfl(v, L ^ 2));
        v = fmaxf(v, __shfl(v, L ^ 1));
        if (v < 1e-12f) v = 1e-12f;
        sq[r] = v / 127.0f;
    }
    for (int t = 0; t < 128; t++) {
        int flat = L + t * 32;
        int r = flat / D_HEAD, d = flat % D_HEAD;
        float qv = __half2float(Qf[r][d]);
        Qi[r][d] = (int8_t)q_i8(qv / sq[r]);
    }
    __syncthreads();

    // Online softmax state for my row (O lives in LDS: 128 cols per lane).
    float m = -INFINITY, l = 0.0f;
    for (int t = 0; t < 128; t++) {
        int flat = L + t * 32;
        Oacc[flat / D_HEAD][flat % D_HEAD] = 0.0f;
    }
    __syncthreads();

    const int q_abs = past + m0 + myRow;
    const int r_valid = (m0 + myRow) < Tq ? 1 : 0;

    // Tail bounds: staged fp16 rows [t_full, Tk).
    int phys_last = block_table[(Tk - 1) / 256];
    int r_tail = slen[phys_last];
    int t_full = Tk - r_tail;

    // ---- Full 32-blocks [0, t_full) ----
    for (int n0 = 0; n0 < t_full; n0 += BN) {
        // causal block skip: no row of this block attends here
        if (causal && n0 > past + m0 + BM - 1) break;
        int pg = n0 / 256;
        int phys = block_table[pg];
        int sub = (n0 % 256) / 32;
        float sk = ksc[((size_t)phys * 8 + sub) * hkv + kh];
        // Stage K tile [32 x 256] into LDS, fully coalesced (lane L owns
        // bytes [L*8, L*8+8) of every row: 32 x u64).
        {
            size_t krow = (((size_t)phys * 256 + (n0 % 256)) * hkv + kh) * D_HEAD;
            #pragma unroll
            for (int t = 0; t < 32; t++) {
                int64_t w = 0;
                __builtin_memcpy(&w, (const int8_t*)(kq + krow + (size_t)t * hkv * D_HEAD + L * 8), 8);
                __builtin_memcpy(&Ktile[t][L * 8], &w, 8);
            }
        }
        __syncthreads();
        // mu[D] to LDS (cooperative), then q.mu for my row
        for (int t = 0; t < 8; t++) {
            int d = L * 8 + t;
            Mu[d] = kmu[(((size_t)phys * 8 + sub) * hkv + kh) * D_HEAD + d];
        }
        __syncthreads();
        // q.mu for my row; the two lanes sharing the row split the 256
        // in half and exchange (halves the serial FMA chain).
        float qmu = 0.0f;
        for (int d = 0; d < D_HEAD / 2; d++)
            qmu += __half2float(Qf[myRow][(L / 16) * (D_HEAD / 2) + d]) *
                   __half2float(Mu[(L / 16) * (D_HEAD / 2) + d]);
        qmu += __shfl(qmu, L ^ 16);
        __syncthreads(); // Mu reuse next tile only after all lanes done
        float row_scale = sq[myRow] * sk * sm_scale;
        float bias = qmu * sm_scale;

        // QK: 16 D-chunks x 2 N-groups of WMMA.
        i32x8 dot0, dot1;
        #pragma unroll
        for (int i = 0; i < 8; i++) dot0[i] = dot1[i] = 0;
        for (int kc = 0; kc < 16; kc++) {
            int d0 = kc * 16 + 8 * (L / 16);
            i32x2 qa;
            {
                int64_t av;
                __builtin_memcpy(&av, &Qi[myRow][d0], 8);
                qa[0] = (int)(av & 0xFFFFFFFFll);
                qa[1] = (int)(av >> 32);
            }
            // B frags from the LDS K tile (one u64 per chunk/group).
            size_t doff = (size_t)(kc * 16 + 8 * (L / 16));
            // N-group 0: rows n0..n0+15, col = L%16
            // (operand order is (B,A): K fragment first, verified by probe3)
            {
                int64_t bv = 0;
                __builtin_memcpy(&bv, &Ktile[L % 16][doff], 8);
                i32x2 kb; kb[0] = (int)(bv & 0xFFFFFFFFll); kb[1] = (int)(bv >> 32);
                dot0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, kb, true, qa, dot0, false);
            }
            // N-group 1: rows n0+16..n0+31
            {
                int64_t bv = 0;
                __builtin_memcpy(&bv, &Ktile[16 + (L % 16)][doff], 8);
                i32x2 kb; kb[0] = (int)(bv & 0xFFFFFFFFll); kb[1] = (int)(bv >> 32);
                dot1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, kb, true, qa, dot1, false);
            }
        }

        // Scores -> online softmax. My 16 scores: groups g=0,1, slots s=0..7.
        float s0[8], s1[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col0 = n0 + 8 * (L / 16) + i;
            int col1 = n0 + 16 + 8 * (L / 16) + i;
            float x0 = (float)dot0[i] * row_scale + bias;
            float x1 = (float)dot1[i] * row_scale + bias;
            if (!r_valid) { x0 = -INFINITY; x1 = -INFINITY; }
            else if (causal) {
                if (col0 > q_abs) x0 = -INFINITY;
                if (col1 > q_abs) x1 = -INFINITY;
            }
            s0[i] = x0; s1[i] = x1;
        }
        float mx = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (s0[i] > mx) mx = s0[i];
            if (s1[i] > mx) mx = s1[i];
        }
        mx = fmaxf(mx, __shfl(mx, L ^ 16));
        float m_new = fmaxf(m, mx);
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        // reuse the score buffers for P in place (saves 16 VGPRs)
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            s0[i] = (s0[i] == -INFINITY) ? 0.0f : expf(s0[i] - m_new);
            s1[i] = (s1[i] == -INFINITY) ? 0.0f : expf(s1[i] - m_new);
        }
        float *p0 = s0, *p1 = s1;
        float psum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) psum += p0[i] + p1[i];
        psum += __shfl(psum, L ^ 16);
        float l_new = l * alpha + psum;
        // P row max for the fp8 scale (over the 32).
        float pmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (p0[i] > pmax) pmax = p0[i];
            if (p1[i] > pmax) pmax = p1[i];
        }
        pmax = fmaxf(pmax, __shfl(pmax, L ^ 16));
        float sp = (pmax < 1e-12f ? 1e-12f : pmax) / 448.0f;
        float sp_inv = 1.0f / sp;
        i32x2 pq0, pq1;
        {
            uint8_t b0[8], b1[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b0[i] = f32_to_e4m3(p0[i] * sp_inv);
                b1[i] = f32_to_e4m3(p1[i] * sp_inv);
            }
            int32_t w0, w1, w2, w3;
            __builtin_memcpy(&w0, b0, 4); __builtin_memcpy(&w1, b0 + 4, 4);
            __builtin_memcpy(&w2, b1, 4); __builtin_memcpy(&w3, b1 + 4, 4);
            pq0[0] = w0; pq0[1] = w1; pq1[0] = w2; pq1[1] = w3;
        }

        // PV per D-chunk: P(16x32) x V(32x64). K-chunks kc2=0,1; N-groups ng=0..3.
        size_t vbase = (((size_t)phys * 256 + (n0 % 256)) * hkv + kh) * D_HEAD;
        size_t sbase = (((size_t)phys * 8 + sub) * hkv + kh) * D_HEAD;
        #pragma unroll
        for (int dc = 0; dc < NDC; dc++) {
            // Stage V chunk [32 x 64] into LDS, coalesced (lane L owns 64
            // bytes: 8 x u64 at flat bytes L*8 + t*256).
            for (int t = 0; t < 8; t++) {
                size_t flat = (size_t)L * 8 + (size_t)t * 256;
                int tr = (int)(flat / D2C), dd = (int)(flat % D2C);
                int64_t w = 0;
                __builtin_memcpy(&w, (const uint8_t*)(vq + vbase + (size_t)tr * hkv * D_HEAD + dc * D2C + dd), 8);
                __builtin_memcpy(&Vtile[tr][dd], &w, 8);
            }
            __syncthreads();
            f32x8 pv[4];
            #pragma unroll
            for (int ng = 0; ng < 4; ng++)
                #pragma unroll
                for (int i = 0; i < 8; i++) pv[ng][i] = 0.0f;
            #pragma unroll
            for (int ng = 0; ng < 4; ng++) {
                // V scale is per-channel: one scale per output slot.
                float sv8[8];
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    sv8[i] = __half2float(vsc[sbase + dc * D2C + ng * 16 + 8 * (L / 16) + i]);
                float pv_base = sp * FP8_WMMA_RSCALE;
                #pragma unroll
                for (int kc2 = 0; kc2 < 2; kc2++) {
                    // V rows n0+kc2*16 + 8*(L/16)+s, one column (LDS bytes)
                    uint8_t vb[8];
                    int vr0 = kc2 * 16 + 8 * (L / 16);
                    int vc0 = ng * 16 + (L % 16);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) vb[i] = Vtile[vr0 + i][vc0];
                    int32_t w0, w1;
                    __builtin_memcpy(&w0, vb, 4); __builtin_memcpy(&w1, vb + 4, 4);
                    i32x2 vv; vv[0] = w0; vv[1] = w1;
                    // operand order (B,A): V fragment first (probe3 ord=1)
                    if (kc2 == 0) pv[ng] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vv, pq0, pv[ng]);
                    else pv[ng] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vv, pq1, pv[ng]);
                }
                #pragma unroll
                for (int i = 0; i < 8; i++) pv[ng][i] *= pv_base * sv8[i];
            }
            __syncthreads(); // Vtile reuse guard before next dc overwrite
            #pragma unroll
            for (int ng = 0; ng < 4; ng++) {
                int c0 = dc * D2C + ng * 16 + 8 * (L / 16);
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    Oacc[myRow][c0 + i] = Oacc[myRow][c0 + i] * alpha + pv[ng][i];
            }
        }
        m = m_new; l = l_new;
        __syncthreads(); // Mu LDS reuse guard (all lanes done before next tile)
    } // end N-tiles

    // ---- Staged tail [t_full, Tk): scalar fp32 (correct, rare/slow path) ----
    // kst/vst hold the last r_tail rows (fp16) of phys_last's stage.
    for (int t = t_full; t < Tk; t++) {
        if (causal && t > q_abs) continue;
        int srow = t - t_full; // 0..r_tail-1
        size_t kbase = (((size_t)phys_last * 32 + srow) * hkv + kh) * D_HEAD;
        float s = 0.0f;
        for (int d = 0; d < D_HEAD; d++)
            s += __half2float(Qf[myRow][d]) * __half2float(kst[kbase + d]);
        s *= sm_scale;
        float m_new = fmaxf(m, s);
        float alpha = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        float p = r_valid ? expf(s - m_new) : 0.0f;
        float l_new = l * alpha + p;
        // acc per D-chunk from LDS, update, write back (scalar RMW: the
        // 64-float array form cost 64 VGPRs for a rare path)
        #pragma unroll
        for (int dc = 0; dc < NDC; dc++) {
            for (int i = 0; i < D2C; i++) {
                float a = Oacc[myRow][dc * D2C + i];
                a = a * alpha + p * __half2float(vst[kbase + dc * D2C + i]);
                Oacc[myRow][dc * D2C + i] = a;
            }
        }
        m = m_new; l = l_new;
    }

    // ---- Finalize: /l, fp16, store (lanes 0..15 write the 16 rows) ----
    float inv_l = (l == 0.0f) ? 0.0f : 1.0f / l;
    if (L < BM && r_valid) {
        int grow = m0 + L;
        if (grow < Tq) {
            #pragma unroll
            for (int dc = 0; dc < NDC; dc++) {
                #pragma unroll
                for (int i = 0; i < D2C; i++) {
                    float v = Oacc[L][dc * D2C + i] * inv_l;
                    out[((size_t)grow * hq + h) * D_HEAD + dc * D2C + i] = __float2half(v);
                }
            }
        }
    }
}

// Test binding: RNE E4M3 converter vs torch (called from the bench script).
__global__ void test_e4m3_kernel(const float* __restrict__ x, uint8_t* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f32_to_e4m3(x[i]);
}

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

void hip_i8f8_prefill(
    torch::Tensor q, torch::Tensor kq, torch::Tensor ksc, torch::Tensor kmu,
    torch::Tensor vq, torch::Tensor vsc, torch::Tensor kst, torch::Tensor vst,
    torch::Tensor slen, torch::Tensor block_table, torch::Tensor out,
    int64_t Tk, int64_t past, double sm_scale, bool causal) {
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kHalf, "q fp16 cuda");
    TORCH_CHECK(kq.dtype() == torch::kChar, "kq int8");
    TORCH_CHECK(vq.dtype() == torch::kByte, "vq uint8");
    int Tq = q.size(0), hq = q.size(1), hkv = kq.size(2);
    TORCH_CHECK(q.size(2) == D_HEAD, "D must be 256 in this prototype");
    TORCH_CHECK(out.sizes() == q.sizes(), "out shape");
    dim3 grid((Tq + BM - 1) / BM, hq);
    dim3 block(32);
    hip_i8f8_prefill_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*)q.data_ptr(), (const int8_t*)kq.data_ptr(),
        (const float*)ksc.data_ptr(), (const half*)kmu.data_ptr(),
        (const uint8_t*)vq.data_ptr(), (const half*)vsc.data_ptr(),
        (const half*)kst.data_ptr(), (const half*)vst.data_ptr(),
        (const int*)slen.data_ptr(), (const int*)block_table.data_ptr(),
        (half*)out.data_ptr(), Tq, (int)Tk, (int)past, hq, hkv,
        (float)sm_scale, causal ? 1 : 0);
}

torch::Tensor test_e4m3(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x fp32 cuda");
    auto y = torch::empty({x.numel()}, x.options().dtype(torch::kUInt8));
    int n = (int)x.numel();
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    test_e4m3_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const float*)x.data_ptr(), (uint8_t*)y.data_ptr(), n);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill", &hip_i8f8_prefill, "HIP i8f8 cached prefill (prototype)");
    m.def("test_e4m3", &test_e4m3, "RNE fp32->E4M3 test");
}
