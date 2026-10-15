// SPDX-License-Identifier: MIT
// Host-only bridge: keep the installed EXL3 GPU kernels byte-for-byte intact.
#include <torch/extension.h>
#include <dlfcn.h>
#include <string>
#include <vector>

using Gemm = void (*)(at::Tensor, at::Tensor, at::Tensor);
static Gemm original_gemm = nullptr;
static void* engine_handle = nullptr;

void configure_engine(const std::string& path) {
    if (original_gemm) return;
    engine_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
    TORCH_CHECK(engine_handle, "EXL3 engine library must already be loaded");
    original_gemm = reinterpret_cast<Gemm>(dlsym(engine_handle, "_Z11hgemm_reconN2at6TensorES0_S0_"));
    TORCH_CHECK(original_gemm, "original engine hgemm_recon symbol not found");
}

void exact_group(at::Tensor a, at::Tensor w, at::Tensor c,
                 const std::vector<int64_t>& counts) {
    TORCH_CHECK(original_gemm, "configure_engine must be called first");
    TORCH_CHECK(a.dim()==3 && w.dim()==3 && c.dim()==3 && a.is_contiguous()
                && w.is_contiguous() && c.is_contiguous(), "contiguous rank-3 slabs required");
    TORCH_CHECK(a.size(0)==w.size(0) && a.size(0)==c.size(0)
                && a.size(0)==(int64_t)counts.size(), "batch mismatch");
    TORCH_CHECK(a.size(1)==c.size(1) && a.size(2)==w.size(1)
                && w.size(2)==c.size(2), "matrix dimensions mismatch");
    c.zero_();
    for (int64_t b=0; b<a.size(0); ++b) {
        TORCH_CHECK(counts[b]>0 && counts[b]<=a.size(1), "invalid expert row count");
        original_gemm(a.select(0,b).slice(0,0,counts[b]), w.select(0,b),
                      c.select(0,b).slice(0,0,counts[b]));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("configure_engine", &configure_engine);
    m.def("hgemm_exact_group", &exact_group);
}
