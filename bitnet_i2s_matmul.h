#pragma once

#include <cstddef>
#include <cstdint>

#include "bitnet_weight.h"

// ============================================================
// BitNet I2S matmul entry points
//
// Forward:
//   Uses forward-packed q2 + w_scale_tile
//
// Backward (Option B):
//   Uses transposed-packed q2T + w_scale_tileT
// ============================================================

// ------------------------------------------------------------
// linear_forward_bitnet_i2s_accum
// ------------------------------------------------------------
// Performs a forward linear layer computation: Y += X * W
// X is quantized to int8 on the fly into x8Scratch. W is accessed 
// in its 2-bit packed row-major format.
// Results are accumulated additively into Y.
void linear_forward_bitnet_i2s_accum(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMatI2S& W,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch
);

// ------------------------------------------------------------
// linear_fused_qkv_forward_bitnet_i2s_accum
// ------------------------------------------------------------
// Fused forward computation for Q, K, and V projections.
// Reads input X once, quantizes it, and sequentially applies it to 
// Wq, Wk, and Wv while the inputs remain hot in the L1 cache.
void linear_fused_qkv_forward_bitnet_i2s_accum(
    float* __restrict Q,
    float* __restrict K,
    float* __restrict V,
    const float* __restrict X,
    const BitNetMatI2S& Wq,
    const BitNetMatI2S& Wk,
    const BitNetMatI2S& Wv,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch
);

// ------------------------------------------------------------
// linear_fused_dual_forward_bitnet_i2s_accum
// ------------------------------------------------------------
// Fused forward computation for two same-shape projections.
// Reuses the quantized X tile and walks both weight matrices together.
void linear_fused_dual_forward_bitnet_i2s_accum(
    float* __restrict Ya,
    float* __restrict Yb,
    const float* __restrict X,
    const BitNetMatI2S& Wa,
    const BitNetMatI2S& Wb,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch
);

// ------------------------------------------------------------
// linear_backward_bitnet_i2s_packed
// ------------------------------------------------------------
// Computes the backward pass input gradient: dX += dY * W^T
// Utilizes the pre-transposed, packed 2-bit representation (q2T) 
// and transposed scales (w_scale_tileT) generated during quantization 
// to maintain dense, sequential memory access and full AVX2 vectorization.
void linear_backward_bitnet_i2s_packed(
    float* __restrict dX,
    const float* __restrict dY,
    const BitNetMatI2S& W,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict dy8Scratch,
    float* __restrict dyScaleScratch
);

// ------------------------------------------------------------
// linear_fused_qkv_backward_bitnet_i2s_packed
// ------------------------------------------------------------
// Computes the fused backward pass input gradient for Q, K, and V:
// dX += dQ * Wq^T + dK * Wk^T + dV * Wv^T
// Leverages transposed packed weights (q2T) for all three matrices 
// to prevent structural memory bottlenecks during backpropagation.
void linear_fused_qkv_backward_bitnet_i2s_packed(
    float* __restrict dX,
    const float* __restrict dQ,
    const float* __restrict dK,
    const float* __restrict dV,
    const BitNetMatI2S& Wq,
    const BitNetMatI2S& Wk,
    const BitNetMatI2S& Wv,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict dq8Scratch,
    float* __restrict dqScaleScratch,
    int8_t* __restrict dk8Scratch,
    float* __restrict dkScaleScratch,
    int8_t* __restrict dv8Scratch,
    float* __restrict dvScaleScratch
);

// ------------------------------------------------------------
// linear_fused_dual_backward_bitnet_i2s_packed
// ------------------------------------------------------------
// Computes a fused backward pass for two same-shape projections:
// dX += dYa * Wa^T + dYb * Wb^T
void linear_fused_dual_backward_bitnet_i2s_packed(
    float* __restrict dX,
    const float* __restrict dYa,
    const float* __restrict dYb,
    const BitNetMatI2S& Wa,
    const BitNetMatI2S& Wb,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict dyA8Scratch,
    float* __restrict dyAScaleScratch,
    int8_t* __restrict dyB8Scratch,
    float* __restrict dyBScaleScratch
);
