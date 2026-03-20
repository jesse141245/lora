#pragma once

#include <cstddef>
#include <cstdint>

#include "bitnet_weight.h"

void quantize_ternary_tl2_blockwise(
    BitNetMatTL2& out,
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t bk = 96
);

void linear_forward_bitnet_tl2_accum(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMatTL2& W,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch
);

void linear_fused_dual_forward_bitnet_tl2_accum(
    float* __restrict Ya,
    float* __restrict Yb,
    const float* __restrict X,
    const BitNetMatTL2& Wa,
    const BitNetMatTL2& Wb,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch
);

void linear_fused_qkv_forward_bitnet_tl2_accum(
    float* __restrict Q,
    float* __restrict K,
    float* __restrict V,
    const float* __restrict X,
    const BitNetMatTL2& Wq,
    const BitNetMatTL2& Wk,
    const BitNetMatTL2& Wv,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch
);
