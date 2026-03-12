#pragma once

#include <cstddef>

// Explicit forward declaration prevents circular include loops 
// and fixes the "undefined" error.
struct BitNetMat;

// ===========================================================================
// Helpers
// ===========================================================================

/**
 * @brief Clamps the T-dimension tile size for cache blocking.
 * @param tTile The requested tile size.
 * @return Clamped tile size (max 4).
 */
std::size_t clamp_ttile(std::size_t tTile);

/**
 * @brief Clamps the T-dimension tile size specifically for AVX2 blocked kernels.
 * @param tTile The requested tile size.
 * @return Clamped tile size (max 2).
 */
std::size_t clamp_ttile_avx2(std::size_t tTile);

// ===========================================================================
// RMSNorm
// ===========================================================================

/**
 * @brief Applies Root Mean Square Normalization (RMSNorm) to the input vector.
 * @param y Output normalized vector (size H).
 * @param x Input vector (size H).
 * @param w Learnable scaling weights (gamma) (size H).
 * @param H Hidden dimension size.
 * @param eps Small constant to prevent division by zero.
 */
void rmsnorm(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict w,
    std::size_t H,
    float eps);

/**
 * @brief Computes the gradient of the RMSNorm operation for the backward pass.
 * @param dx Output gradient with respect to input x (size H).
 * @param g Incoming gradient from the next layer (dY) (size H).
 * @param x Original input vector from the forward pass (size H).
 * @param gamma Original learnable scaling weights (size H).
 * @param H Hidden dimension size.
 * @param eps Small constant used in the forward pass.
 */
void rmsnorm_backward(
    float* __restrict dx,
    const float* __restrict g,
    const float* __restrict x,
    const float* __restrict gamma,
    std::size_t H,
    float eps);

// ===========================================================================
// Dense Linear Algebra (Float32)
// ===========================================================================

/**
 * @brief Performs a standard matrix-vector multiplication (y = x * W).
 * Used for single-token fallback paths.
 * @param y Output vector (size H).
 * @param x Input vector (size H).
 * @param W Weight matrix (size H x H).
 * @param H Hidden dimension size.
 * @param hTile Cache blocking size for the H dimension.
 */
void linear_HH(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict W,
    std::size_t H,
    std::size_t hTile);

/**
 * @brief Performs a batched matrix multiplication (Y = X * W) for T tokens.
 * @param Y Output matrix (size T x H).
 * @param X Input matrix (size T x H).
 * @param W Weight matrix (size H x H).
 * @param T Sequence length (number of tokens).
 * @param H Hidden dimension size.
 * @param jTile Cache blocking size for the inner loop.
 */
void linear_THH(
    float* __restrict Y,
    const float* __restrict X,
    const float* __restrict W,
    std::size_t T,
    std::size_t H,
    std::size_t jTile);

/**
 * @brief Accumulates the result of a transposed matrix-vector multiplication (y += A * B^T).
 * @param y Output vector to accumulate into (size H).
 * @param A Input vector (size H).
 * @param B Transposed weight matrix (size H x H).
 * @param H Hidden dimension size.
 * @param hTile Cache blocking size.
 */
void linear_HH_T_accum(
    float* __restrict y,
    const float* __restrict A,
    const float* __restrict B,
    std::size_t H,
    std::size_t hTile);

// ===========================================================================
// LoRA
// ===========================================================================

/**
 * @brief Computes and adds a LoRA adapter update to a hidden state (y += scale * (x * A) * B).
 * @param y Hidden state vector to update (size H).
 * @param x Original normalized hidden state input (size H).
 * @param A LoRA 'A' matrix (size R x H).
 * @param B LoRA 'B' matrix (size R x H, represented as transposed H x R).
 * @param tmp Scratch buffer for intermediate rank calculation (size R).
 * @param H Hidden dimension size.
 * @param R LoRA Rank.
 * @param scale Scaling factor (alpha / rank).
 * @param hTile Cache blocking size.
 */
void lora_add_H(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict tmp,
    std::size_t H,
    std::size_t R,
    float scale,
    std::size_t hTile);

// ===========================================================================
// Fused QKV Forward (Float32)
// ===========================================================================

/**
 * @brief Baseline fused projection for Query, Key, and Value matrices.
 * @param Q Output Query matrix (size T x H).
 * @param K Output Key matrix (size T x kvDim).
 * @param V Output Value matrix (size T x kvDim).
 * @param X Input hidden states (size T x H).
 * @param Wq Query weights.
 * @param Wk Key weights.
 * @param Wv Value weights.
 * @param T Sequence length.
 * @param H Hidden dimension size.
 * @param jTile Cache blocking size (width).
 * @param tTile Cache blocking size (time).
 */
void linear_THH_fused_qkv(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile);

/**
 * @brief Accumulation variant of the fused QKV projection.
 */
void linear_THH_fused_qkv_accum(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile);

/**
 * @brief Highly optimized, 16-element AVX2 blocked fused QKV projection.
 */
void linear_THH_fused_qkv_accum_blocked16(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile);

/**
 * @brief Highly optimized, 32-element AVX2 blocked fused QKV projection.
 */
void linear_THH_fused_qkv_accum_blocked32(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile);

// ===========================================================================
// QKV Input Backward (Float32)
// ===========================================================================

/**
 * @brief Computes the gradient of the input X with respect to the Q, K, and V gradients.
 * Effectively: dX = dQ * Wq^T + dK * Wk^T + dV * Wv^T.
 * @param dX Output gradient for the input (size T x H).
 * @param dQ Incoming gradient for Queries.
 * @param dK Incoming gradient for Keys.
 * @param dV Incoming gradient for Values.
 * @param Wq Query weights.
 * @param Wk Key weights.
 * @param Wv Value weights.
 * @param T Sequence length.
 * @param H Hidden dimension size.
 * @param hTile Cache blocking size.
 */
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

// ===========================================================================
// BitNet (1.58-bit Ternary) Kernels
// ===========================================================================

/**
 * @brief Performs a linear projection using BitNet ternary quantized weights.
 * @param Y Output matrix (size T x H).
 * @param X Input matrix (size T x H).
 * @param Wq BitNet quantized weight matrix containing scales and ternary int8 values.
 * @param T Sequence length.
 * @param H Hidden dimension size.
 * @param jTile Cache blocking size.
 */
void linear_THH_bitnet(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMat& Wq,
    std::size_t T,
    std::size_t H,
    std::size_t jTile);

/**
 * @brief Computes the input gradient (dX) for the QKV projections using BitNet weights.
 * @param dX Output gradient for the input (size T x H).
 * @param dQ Incoming Query gradient.
 * @param dK Incoming Key gradient.
 * @param dV Incoming Value gradient.
 * @param Wq BitNet Query weights.
 * @param Wk BitNet Key weights.
 * @param Wv BitNet Value weights.
 * @param T Sequence length.
 * @param H Hidden dimension size.
 * @param hTile Cache blocking size.
 */
void qkv_input_backward_THH_bitnet(
    float* __restrict dX,
    const float* __restrict dQ,
    const float* __restrict dK,
    const float* __restrict dV,
    const BitNetMat& Wq,
    const BitNetMat& Wk,
    const BitNetMat& Wv,
    std::size_t T,
    std::size_t H,
    std::size_t hTile);

/**
 * @brief Fused QKV forward pass using BitNet ternary weights, blocked for AVX2.
 */
void linear_THH_fused_qkv_bitnet_accum_blocked32(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const BitNetMat& WqM, const BitNetMat& WkM, const BitNetMat& WvM,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile);

/**
 * @brief Accumulates the result of a transposed matrix-vector multiplication using BitNet weights.
 */
void linear_HH_T_accum_bitnet(
    float* __restrict y,
    const float* __restrict A,
    const BitNetMat& WoM,
    std::size_t H,
    std::size_t hTile);

/**
 * @brief Backward pass for BitNet fused QKV, specifically generating the dX gradient, optimized with AVX2 blocking.
 */
void linear_THH_fused_qkv_bitnet_backward_dX_blocked32(
    float* __restrict dX,
    const float* __restrict dQ,
    const float* __restrict dK,
    const float* __restrict dV,
    const BitNetMat& WqM, const BitNetMat& WkM, const BitNetMat& WvM,
    std::size_t T, std::size_t H,
    const int* __restrict attentionMask,
    std::size_t iTile,
    std::size_t tTile);

// ===========================================================================
// Utilities
// ===========================================================================

/**
 * @brief Scales an array of floats in-place (data[i] *= scale).
 * Automatically dispatches to AVX2 if supported, falling back to scalar.
 * @param data Pointer to the float array.
 * @param n Number of elements.
 * @param scale The float multiplier.
 */
void scale_inplace_avx2_or_scalar(float* data, std::size_t n, float scale);