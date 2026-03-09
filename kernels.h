#pragma once
#include <cstddef>
#include "bitnet_weight.h"

// ===========================
// RMSNorm
// ===========================
void rmsnorm(
    float* y,               // [H]
    const float* x,         // [H]
    const float* w,         // [H]
    std::size_t H,
    float eps
);

void rmsnorm_backward(
    float* dx,              // [H]
    const float* g,         // [H]
    const float* x,         // [H]
    const float* gamma,     // [H]
    std::size_t H,
    float eps
);

// ===========================
// Linear layers
// ===========================
void linear_HH(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict W,
    std::size_t H,
    std::size_t hTile
);

void linear_THH(
    float* __restrict Y,          // [T x H]
    const float* __restrict X,    // [T x H]
    const float* __restrict W,    // [H x H] row-major
    std::size_t T,
    std::size_t H,
    std::size_t jTile
);

// ===========================
// LoRA add
// ===========================
void lora_add_H(
    float* y,               // [H] add into this
    const float* x,         // [H]
    const float* A,         // [R x H]
    const float* B,         // [R x H]
    float* tmp,             // [R] cached = A*x
    std::size_t H,
    std::size_t R,
    float scale,
    std::size_t hTile
);

// ===========================
// Misc helper kernel
// ===========================
void linear_HH_T_accum(
    float* y,
    const float* A,
    const float* B,
    std::size_t H,
    std::size_t hTile
);

// ===========================
// Fused QKV (baseline + accum)
// ===========================
void linear_THH_fused_qkv(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile
);

// �accum� variant (your comparison kernel)
void linear_THH_fused_qkv_accum(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile
);

// ===========================
// Strengthened blocked QKV
// ===========================
// Actual implementations provided in kernels.cpp
void linear_THH_fused_qkv_accum_blocked16(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile
);

void linear_THH_fused_qkv_accum_blocked32(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile
);

// Stable public names (map to 16/32-wide AVX2 blocks).
inline void linear_THH_fused_qkv_accum_blocked_j2(
    float* Q, float* K, float* V,
    const float* X,
    const float* Wq, const float* Wk, const float* Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile
) {
    linear_THH_fused_qkv_accum_blocked16(Q, K, V, X, Wq, Wk, Wv, T, H, jTile, tTile);
}

inline void linear_THH_fused_qkv_accum_blocked_j4(
    float* Q, float* K, float* V,
    const float* X,
    const float* Wq, const float* Wk, const float* Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile
) {
    linear_THH_fused_qkv_accum_blocked32(Q, K, V, X, Wq, Wk, Wv, T, H, jTile, tTile);
}

// ===========================
// QKV backward (input gradient)
// ===========================
void qkv_input_backward_THH(
    float* __restrict dX,
    const float* __restrict dQ,
    const float* __restrict dK,
    const float* __restrict dV,
    const float* __restrict Wq,
    const float* __restrict Wk,
    const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t hTile);


// ===========================
// BitNet (ternary) kernels
// ===========================

void linear_THH_bitnet(
    float* __restrict Y,             // [T x H]
    const float* __restrict X,       // [T x H]
    const BitNetMat& Wq,             // [H x H]
    std::size_t T,
    std::size_t H,
    std::size_t jTile
);

void qkv_input_backward_THH_bitnet(
    float* __restrict dX,            // [T x H]
    const float* __restrict dQ,      // [T x H]
    const float* __restrict dK,      // [T x H]
    const float* __restrict dV,      // [T x H]
    const BitNetMat& Wq,
    const BitNetMat& Wk,
    const BitNetMat& Wv,
    std::size_t T,
    std::size_t H,
    std::size_t hTile
);

// Fast fused QKV for BitNet ternary weights (blocked32)
void linear_THH_fused_qkv_bitnet_accum_blocked32(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const BitNetMat& Wq, const BitNetMat& Wk, const BitNetMat& Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile
);

// Wo^T backward: y += A * Wo^T  (Wo stored row-major)
void linear_HH_T_accum_bitnet(
    float* y,
    const float* A,
    const BitNetMat& Wo,
    std::size_t H,
    std::size_t hTile
);


void linear_THH_fused_qkv_bitnet_backward_dX_blocked32(
    float* __restrict dX,                 // [T x H] (accum)
    const float* __restrict dQ,            // [T x H]
    const float* __restrict dK,            // [T x H]
    const float* __restrict dV,            // [T x H]
    const BitNetMat& WqM, const BitNetMat& WkM, const BitNetMat& WvM,
    std::size_t T, std::size_t H,
    const int* __restrict attentionMask,   // [T] may be null
    std::size_t iTile,                    // tile over i
    std::size_t tTile
);