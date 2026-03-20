#pragma once

#include <cstddef>

#include "bitnet_weight.h"

bool quantize_ternary_tl2_generated(
    BitNetMatTL2Gen& out,
    const float* W,
    std::size_t rows,
    std::size_t cols,
    const char* scope = nullptr
);

void linear_forward_bitnet_tl2_generated(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMatTL2Gen& W,
    std::size_t T
);
