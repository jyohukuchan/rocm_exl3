#pragma once

#include <ATen/Tensor.h>

// int8 -> int32 row-major GEMM for the EXL3 W8A8 prefill path
// (implemented in quant/exl3_igemm.cu, both backends).
void igemm_recon(at::Tensor a, at::Tensor b, at::Tensor c);
