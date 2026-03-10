#pragma once

#include <cstddef>
#include <cstdint>

#include "bitnet_weight.h"

// ------------------------------------------------------------
// linear_THH_bitnet_i2s_accum
// ------------------------------------------------------------
// Performs a forward linear layer computation: Y += X * W
// X is quantized to int8 on the fly into x8Scratch. W is accessed 
// in its 2-bit packed row-major format.
// Results are accumulated additively into Y.
void linear_THH_bitnet_i2s_accum(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMatI2S& W,
    std::size_t T,
    std::size_t H,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch
);

// ------------------------------------------------------------
// linear_THH_fused_qkv_bitnet_i2s_accum
// ------------------------------------------------------------
// Fused forward computation for Q, K, and V projections.
// Reads input X once, quantizes it, and sequentially applies it to 
// Wq, Wk, and Wv while the inputs remain hot in the L1 cache.
void linear_THH_fused_qkv_bitnet_i2s_accum(
    float* __restrict Q,
    float* __restrict K,
    float* __restrict V,
    const float* __restrict X,
    const BitNetMatI2S& Wq,
    const BitNetMatI2S& Wk,
    const BitNetMatI2S& Wv,
    std::size_t T,
    std::size_t H,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch
);

// ------------------------------------------------------------
// linear_HH_T_bitnet_i2s_packed
// ------------------------------------------------------------
// Computes the backward pass input gradient: dX += dY * W^T
// Utilizes the pre-transposed, packed 2-bit representation (q2T) 
// and transposed scales (w_scale_tileT) generated during quantization 
// to maintain dense, sequential memory access and full AVX2 vectorization.
void linear_HH_T_bitnet_i2s_packed(
    float* __restrict dX,
    const float* __restrict dY,
    const BitNetMatI2S& W,
    std::size_t T,
    std::size_t H,
    std::size_t tTile,
    int8_t* __restrict dy8Scratch,
    float* __restrict dyScaleScratch
);

// ------------------------------------------------------------
// qkv_input_backward_bitnet_i2s_packed
// ------------------------------------------------------------
// Computes the fused backward pass input gradient for Q, K, and V:
// dX += dQ * Wq^T + dK * Wk^T + dV * Wv^T
// Leverages transposed packed weights (q2T) for all three matrices 
// to prevent structural memory bottlenecks during backpropagation.
void qkv_input_backward_bitnet_i2s_packed(
    float* __restrict dX,
    const float* __restrict dQ,
    const float* __restrict dK,
    const float* __restrict dV,
    const BitNetMatI2S& Wq,
    const BitNetMatI2S& Wk,
    const BitNetMatI2S& Wv,
    std::size_t T,
    std::size_t H,
    std::size_t tTile,
    int8_t* __restrict dq8Scratch,
    float* __restrict dqScaleScratch,
    int8_t* __restrict dk8Scratch,
    float* __restrict dkScaleScratch,
    int8_t* __restrict dv8Scratch,
    float* __restrict dvScaleScratch
);