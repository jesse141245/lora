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
// Backward (Option C):
//   Uses forward-packed q2 + w_scale_tile + bwd assist metadata
//   No full q2T / w_scale_tileT are stored.
// ============================================================


// ------------------------------------------------------------
// Forward: Y[T,H] += X[T,H] * W[H,H]
//
// W is BitNet I2S packed row-major.
// Output accumulation semantics are additive.
// ------------------------------------------------------------

void linear_THH_bitnet_i2s_accum(
    float* __restrict Y,              // [T x H], row-major
    const float* __restrict X,        // [T x H], row-major
    const BitNetMatI2S& W,
    std::size_t T,
    std::size_t H,
    std::size_t tTile
);


// ------------------------------------------------------------
// Forward: fused QKV
//
// Q,K,V each receive:
//   += X * W{q,k,v}
// ------------------------------------------------------------

void linear_THH_fused_qkv_bitnet_i2s_accum(
    float* __restrict Q,              // [T x H]
    float* __restrict K,              // [T x H]
    float* __restrict V,              // [T x H]
    const float* __restrict X,        // [T x H]
    const BitNetMatI2S& Wq,
    const BitNetMatI2S& Wk,
    const BitNetMatI2S& Wv,
    std::size_t T,
    std::size_t H,
    std::size_t tTile
);


// ------------------------------------------------------------
// Backward input gradient: dX += dY * W^T
//
// Option C:
//   Reads transpose-style weights directly from forward q2
//   using W.bwd metadata.
// ------------------------------------------------------------

void linear_HH_T_bitnet_i2s_assisted_block32(
    float* __restrict dX,             // [T x H]
    const float* __restrict dY,       // [T x H]
    const BitNetMatI2S& W,
    std::size_t T,
    std::size_t H,
    std::size_t tTile,
    std::size_t iTile
);


// ------------------------------------------------------------
// Backward input gradient for fused QKV:
//
// dX += dQ * Wq^T + dK * Wk^T + dV * Wv^T
//
// Uses assisted transpose access from forward q2 buffers.
// ------------------------------------------------------------

void qkv_input_backward_bitnet_i2s_assisted(
    float* __restrict dX,             // [T x H]
    const float* __restrict dQ,       // [T x H]
    const float* __restrict dK,       // [T x H]
    const float* __restrict dV,       // [T x H]
    const BitNetMatI2S& Wq,
    const BitNetMatI2S& Wk,
    const BitNetMatI2S& Wv,
    std::size_t T,
    std::size_t H,
    std::size_t tTile,
    std::size_t iTile
);