#pragma once

#include <cstddef>
#include <immintrin.h>
#include <stdexcept>

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

void rmsnorm_backward_accum(
    float* __restrict dx_accum,
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
// FFN GEMM Helpers
// ===========================================================================

/**
 * @brief Computes two FFN input projections in one blocked GEMM-style pass.
 * Effectively:
 *   Y0 = X * W0
 *   Y1 = X * W1
 * with X laid out as [T x H] and W0/W1 laid out as [H x F].
 */
void linear_THF_fused_dual_accum_blocked32(
    float* __restrict Y0,
    float* __restrict Y1,
    const float* __restrict X,
    const float* __restrict W0,
    const float* __restrict W1,
    std::size_t T,
    std::size_t H,
    std::size_t F,
    std::size_t fTile,
    std::size_t tTile);

/**
 * @brief Computes a blocked GEMM-style down projection:
 *   Y = X * W
 * with X laid out as [T x F] and W laid out as [F x H].
 */
void linear_TFH_accum_blocked32(
    float* __restrict Y,
    const float* __restrict X,
    const float* __restrict W,
    std::size_t T,
    std::size_t F,
    std::size_t H,
    std::size_t hTile,
    std::size_t tTile);

/**
 * @brief Computes dAct = dOut * W^T for W stored as [F x H].
 */
void linear_THF_from_FH_blocked4(
    float* __restrict Y,
    const float* __restrict X,
    const float* __restrict W,
    std::size_t T,
    std::size_t H,
    std::size_t F,
    std::size_t fTile,
    std::size_t tTile);

/**
 * @brief Computes a fused FFN input backward GEMM:
 *   Y = X0 * W0^T + X1 * W1^T
 * with X0/X1 laid out as [T x F] and W0/W1 laid out as [H x F].
 */
void linear_TFH_fused_dual_from_HF_blocked4(
    float* __restrict Y,
    const float* __restrict X0,
    const float* __restrict X1,
    const float* __restrict W0,
    const float* __restrict W1,
    std::size_t T,
    std::size_t F,
    std::size_t H,
    std::size_t hTile,
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

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/**
 * @brief Computes a dot product (sum_i a[i] * b[i]) with AVX2 fallback.
 * @param a Pointer to first float array.
 * @param b Pointer to second float array.
 * @param n Number of elements.
 * @return Dot product value.
 */
inline float dot_avx2_or_scalar(
    const float* a,
    const float* b,
    std::size_t n)
{
    if (!a || !b) throw std::runtime_error("dot_avx2_or_scalar: null pointer");
    if (n == 0) return 0.0f;

    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }

    float out = hsum256(acc);
    for (; i < n; ++i) out += a[i] * b[i];
    return out;
}

/**
 * @brief Computes two dot products over the same length in one pass.
 * @param a0 Pointer to first lhs array.
 * @param b0 Pointer to first rhs array.
 * @param a1 Pointer to second lhs array.
 * @param b1 Pointer to second rhs array.
 * @param n Number of elements.
 * @param out0 Output for dot(a0, b0).
 * @param out1 Output for dot(a1, b1).
 */
inline void dot2_avx2_or_scalar(
    const float* a0,
    const float* b0,
    const float* a1,
    const float* b1,
    std::size_t n,
    float& out0,
    float& out1)
{
    if (!a0 || !b0 || !a1 || !b1) throw std::runtime_error("dot2_avx2_or_scalar: null pointer");
    if (n == 0) {
        out0 = 0.0f;
        out1 = 0.0f;
        return;
    }

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 a0v = _mm256_loadu_ps(a0 + i);
        const __m256 b0v = _mm256_loadu_ps(b0 + i);
        const __m256 a1v = _mm256_loadu_ps(a1 + i);
        const __m256 b1v = _mm256_loadu_ps(b1 + i);
        acc0 = _mm256_fmadd_ps(a0v, b0v, acc0);
        acc1 = _mm256_fmadd_ps(a1v, b1v, acc1);
    }

    out0 = hsum256(acc0);
    out1 = hsum256(acc1);
    for (; i < n; ++i) {
        out0 += a0[i] * b0[i];
        out1 += a1[i] * b1[i];
    }
}

inline void dot_x4_avx2_or_scalar(
    const float* a0,
    const float* a1,
    const float* a2,
    const float* a3,
    const float* b,
    std::size_t n,
    float& out0,
    float& out1,
    float& out2,
    float& out3)
{
    if (!a0 || !a1 || !a2 || !a3 || !b) throw std::runtime_error("dot_x4_avx2_or_scalar: null pointer");
    if (n == 0) {
        out0 = out1 = out2 = out3 = 0.0f;
        return;
    }

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 bv = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + i), bv, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + i), bv, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + i), bv, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + i), bv, acc3);
    }

    out0 = hsum256(acc0);
    out1 = hsum256(acc1);
    out2 = hsum256(acc2);
    out3 = hsum256(acc3);
    for (; i < n; ++i) {
        const float bv = b[i];
        out0 += a0[i] * bv;
        out1 += a1[i] * bv;
        out2 += a2[i] * bv;
        out3 += a3[i] * bv;
    }
}

/**
 * @brief Performs in-place AXPY: y[i] += alpha * x[i].
 * @param y Output/input array.
 * @param x Input array.
 * @param n Number of elements.
 * @param alpha Scale multiplier for x.
 */
inline void axpy_inplace_avx2_or_scalar(
    float* y,
    const float* x,
    std::size_t n,
    float alpha)
{
    if (!y || !x) throw std::runtime_error("axpy_inplace_avx2_or_scalar: null pointer");
    if (n == 0 || alpha == 0.0f) return;

    const __m256 av = _mm256_set1_ps(alpha);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        __m256 yv = _mm256_loadu_ps(y + i);
        yv = _mm256_fmadd_ps(av, xv, yv);
        _mm256_storeu_ps(y + i, yv);
    }
    for (; i < n; ++i) y[i] += alpha * x[i];
}

inline void axpy_x4_inplace_avx2_or_scalar(
    float* y0,
    float* y1,
    float* y2,
    float* y3,
    const float* x,
    std::size_t n,
    float alpha0,
    float alpha1,
    float alpha2,
    float alpha3)
{
    if (!y0 || !y1 || !y2 || !y3 || !x) throw std::runtime_error("axpy_x4_inplace_avx2_or_scalar: null pointer");
    const __m256 a0 = _mm256_set1_ps(alpha0);
    const __m256 a1 = _mm256_set1_ps(alpha1);
    const __m256 a2 = _mm256_set1_ps(alpha2);
    const __m256 a3 = _mm256_set1_ps(alpha3);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y0 + i, _mm256_fmadd_ps(a0, xv, _mm256_loadu_ps(y0 + i)));
        _mm256_storeu_ps(y1 + i, _mm256_fmadd_ps(a1, xv, _mm256_loadu_ps(y1 + i)));
        _mm256_storeu_ps(y2 + i, _mm256_fmadd_ps(a2, xv, _mm256_loadu_ps(y2 + i)));
        _mm256_storeu_ps(y3 + i, _mm256_fmadd_ps(a3, xv, _mm256_loadu_ps(y3 + i)));
    }
    for (; i < n; ++i) {
        const float xv = x[i];
        y0[i] += alpha0 * xv;
        y1[i] += alpha1 * xv;
        y2[i] += alpha2 * xv;
        y3[i] += alpha3 * xv;
    }
}

inline void axpy_bias_x4_inplace_avx2_or_scalar(
    float* y0,
    float* y1,
    float* y2,
    float* y3,
    float* yBias,
    const float* x,
    std::size_t n,
    float alpha0,
    float alpha1,
    float alpha2,
    float alpha3)
{
    if (!y0 || !y1 || !y2 || !y3 || !yBias || !x) {
        throw std::runtime_error("axpy_bias_x4_inplace_avx2_or_scalar: null pointer");
    }
    const __m256 a0 = _mm256_set1_ps(alpha0);
    const __m256 a1 = _mm256_set1_ps(alpha1);
    const __m256 a2 = _mm256_set1_ps(alpha2);
    const __m256 a3 = _mm256_set1_ps(alpha3);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y0 + i, _mm256_fmadd_ps(a0, xv, _mm256_loadu_ps(y0 + i)));
        _mm256_storeu_ps(y1 + i, _mm256_fmadd_ps(a1, xv, _mm256_loadu_ps(y1 + i)));
        _mm256_storeu_ps(y2 + i, _mm256_fmadd_ps(a2, xv, _mm256_loadu_ps(y2 + i)));
        _mm256_storeu_ps(y3 + i, _mm256_fmadd_ps(a3, xv, _mm256_loadu_ps(y3 + i)));
        _mm256_storeu_ps(yBias + i, _mm256_add_ps(_mm256_loadu_ps(yBias + i), xv));
    }
    for (; i < n; ++i) {
        const float xv = x[i];
        y0[i] += alpha0 * xv;
        y1[i] += alpha1 * xv;
        y2[i] += alpha2 * xv;
        y3[i] += alpha3 * xv;
        yBias[i] += xv;
    }
}

inline void copy_avx2_or_scalar(
    float* dst,
    const float* src,
    std::size_t n)
{
    if (!dst || !src) throw std::runtime_error("copy_avx2_or_scalar: null pointer");
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
    }
    for (; i < n; ++i) dst[i] = src[i];
}

inline void add_inplace_avx2_or_scalar(
    float* dst,
    const float* src,
    std::size_t n)
{
    if (!dst || !src) throw std::runtime_error("add_inplace_avx2_or_scalar: null pointer");
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 dv = _mm256_loadu_ps(dst + i);
        const __m256 sv = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(dv, sv));
    }
    for (; i < n; ++i) dst[i] += src[i];
}

inline void add_to_out_avx2_or_scalar(
    float* out,
    const float* a,
    const float* b,
    std::size_t n)
{
    if (!out || !a || !b) throw std::runtime_error("add_to_out_avx2_or_scalar: null pointer");
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(av, bv));
    }
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

/**
 * @brief Performs two in-place AXPY operations in one pass.
 * @param y0 First output/input array.
 * @param x0 First input array.
 * @param y1 Second output/input array.
 * @param x1 Second input array.
 * @param n Number of elements.
 * @param alpha Shared scale multiplier.
 */
void axpy2_inplace_avx2_or_scalar(
    float* y0,
    const float* x0,
    float* y1,
    const float* x1,
    std::size_t n,
    float alpha);

inline void axpy2_x4_inplace_avx2_or_scalar(
    float* y00,
    float* y01,
    float* y10,
    float* y11,
    float* y20,
    float* y21,
    float* y30,
    float* y31,
    const float* x0,
    const float* x1,
    std::size_t n,
    float alpha0,
    float alpha1,
    float alpha2,
    float alpha3)
{
    if (!y00 || !y01 || !y10 || !y11 || !y20 || !y21 || !y30 || !y31 || !x0 || !x1) {
        throw std::runtime_error("axpy2_x4_inplace_avx2_or_scalar: null pointer");
    }

    const __m256 a0 = _mm256_set1_ps(alpha0);
    const __m256 a1 = _mm256_set1_ps(alpha1);
    const __m256 a2 = _mm256_set1_ps(alpha2);
    const __m256 a3 = _mm256_set1_ps(alpha3);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x0v = _mm256_loadu_ps(x0 + i);
        const __m256 x1v = _mm256_loadu_ps(x1 + i);
        _mm256_storeu_ps(y00 + i, _mm256_fmadd_ps(a0, x0v, _mm256_loadu_ps(y00 + i)));
        _mm256_storeu_ps(y01 + i, _mm256_fmadd_ps(a0, x1v, _mm256_loadu_ps(y01 + i)));
        _mm256_storeu_ps(y10 + i, _mm256_fmadd_ps(a1, x0v, _mm256_loadu_ps(y10 + i)));
        _mm256_storeu_ps(y11 + i, _mm256_fmadd_ps(a1, x1v, _mm256_loadu_ps(y11 + i)));
        _mm256_storeu_ps(y20 + i, _mm256_fmadd_ps(a2, x0v, _mm256_loadu_ps(y20 + i)));
        _mm256_storeu_ps(y21 + i, _mm256_fmadd_ps(a2, x1v, _mm256_loadu_ps(y21 + i)));
        _mm256_storeu_ps(y30 + i, _mm256_fmadd_ps(a3, x0v, _mm256_loadu_ps(y30 + i)));
        _mm256_storeu_ps(y31 + i, _mm256_fmadd_ps(a3, x1v, _mm256_loadu_ps(y31 + i)));
    }
    for (; i < n; ++i) {
        const float x0s = x0[i];
        const float x1s = x1[i];
        y00[i] += alpha0 * x0s;
        y01[i] += alpha0 * x1s;
        y10[i] += alpha1 * x0s;
        y11[i] += alpha1 * x1s;
        y20[i] += alpha2 * x0s;
        y21[i] += alpha2 * x1s;
        y30[i] += alpha3 * x0s;
        y31[i] += alpha3 * x1s;
    }
}

/**
 * @brief Scales an array of floats in-place (data[i] *= scale).
 * Automatically dispatches to AVX2 if supported, falling back to scalar.
 * @param data Pointer to the float array.
 * @param n Number of elements.
 * @param scale The float multiplier.
 */
void scale_inplace_avx2_or_scalar(float* data, std::size_t n, float scale);
