#pragma once

#include <ATen/Tensor.h>
#include <tuple>

std::tuple<bool, int64_t> quantize_tiles_scratch(int device, int K, bool mcg, bool mul1, int L);

// s0: INT8-exact codebook grid step (round 2 of the W8A8 prefill work). When
// > 0 the Viterbi search scores codebook values rounded to the int8 grid
// T = q * s0, so stored states decode to exactly representable int8. Only the
// plain kernel implements it; the optimized specialization rejects s0 > 0.
void quantize_tiles
(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int K,
    bool mcg,
    bool mul1,
    double s0
);

void decode
(
    at::Tensor input_indices,
    at::Tensor output_tiles,
    bool mcg,
    bool mul1
);

void test_distribution
(
    at::Tensor& input,
    at::Tensor& dist_output,
    const c10::optional<at::Tensor>& ref_output,
    float min_value,
    float max_value,
    bool mcg,
    bool mul1
);