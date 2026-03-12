#include "bitnet_i2s_matmul.h"
#include "bitnet_i2_s.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <immintrin.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

    static inline std::size_t ceil_div(std::size_t a, std::size_t b) {
        return (a + b - 1) / b;
    }

    // ============================================================
    // Forward Helpers
    // ============================================================

    static inline float load_forward_tile_scale_block32(
        const BitNetMatI2S& W, std::size_t j0, std::size_t kb)
    {
        const std::size_t nKBlocks = ceil_div(W.rows, W.qk);
        const std::size_t jb = j0 / W.jq;
        return W.w_scale_tile[jb * nKBlocks + kb];
    }

    static inline void accum_block32_scaled_to_y(
        float* __restrict yRow, const int32_t* __restrict blockAcc,
        const BitNetMatI2S& W, std::size_t j0, std::size_t kb, float xs)
    {
        const __m256 gain8 = _mm256_set1_ps(xs * load_forward_tile_scale_block32(W, j0, kb));
        for (int off = 0; off < 32; off += 8) {
            const __m256 accF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAcc + off)));
            __m256 y = _mm256_loadu_ps(yRow + j0 + off);
            _mm256_storeu_ps(yRow + j0 + off, _mm256_fmadd_ps(accF, gain8, y));
        }
    }

    static inline void accum_block32_scaled_to_y_pair(
        float* __restrict yRowA, float* __restrict yRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        const BitNetMatI2S& W, std::size_t j0, std::size_t kb, float xsA, float xsB)
    {
        const float wsScalar = load_forward_tile_scale_block32(W, j0, kb);
        const __m256 gainA8 = _mm256_set1_ps(xsA * wsScalar);
        const __m256 gainB8 = _mm256_set1_ps(xsB * wsScalar);
        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 yA = _mm256_loadu_ps(yRowA + j0 + off);
            __m256 yB = _mm256_loadu_ps(yRowB + j0 + off);
            _mm256_storeu_ps(yRowA + j0 + off, _mm256_fmadd_ps(accFA, gainA8, yA));
            _mm256_storeu_ps(yRowB + j0 + off, _mm256_fmadd_ps(accFB, gainB8, yB));
        }
    }

    static inline void accum_block32_scaled_to_y3(
        float* __restrict qRow, float* __restrict kRow, float* __restrict vRow,
        const int32_t* __restrict accQ, const int32_t* __restrict accK, const int32_t* __restrict accV,
        const BitNetMatI2S& Wq, const BitNetMatI2S& Wk, const BitNetMatI2S& Wv,
        std::size_t j0, std::size_t kb, float xs)
    {
        const __m256 gainQ = _mm256_set1_ps(xs * load_forward_tile_scale_block32(Wq, j0, kb));
        const __m256 gainK = _mm256_set1_ps(xs * load_forward_tile_scale_block32(Wk, j0, kb));
        const __m256 gainV = _mm256_set1_ps(xs * load_forward_tile_scale_block32(Wv, j0, kb));

        for (int off = 0; off < 32; off += 8) {
            const __m256 qF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQ + off)));
            const __m256 kF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accK + off)));
            const __m256 vF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accV + off)));

            __m256 q = _mm256_loadu_ps(qRow + j0 + off);
            __m256 k = _mm256_loadu_ps(kRow + j0 + off);
            __m256 v = _mm256_loadu_ps(vRow + j0 + off);

            _mm256_storeu_ps(qRow + j0 + off, _mm256_fmadd_ps(qF, gainQ, q));
            _mm256_storeu_ps(kRow + j0 + off, _mm256_fmadd_ps(kF, gainK, k));
            _mm256_storeu_ps(vRow + j0 + off, _mm256_fmadd_ps(vF, gainV, v));
        }
    }

    static inline void accum_block32_scaled_to_y3_pair(
        float* __restrict qRowA, float* __restrict kRowA, float* __restrict vRowA,
        float* __restrict qRowB, float* __restrict kRowB, float* __restrict vRowB,
        const int32_t* __restrict accQA, const int32_t* __restrict accKA, const int32_t* __restrict accVA,
        const int32_t* __restrict accQB, const int32_t* __restrict accKB, const int32_t* __restrict accVB,
        const BitNetMatI2S& Wq, const BitNetMatI2S& Wk, const BitNetMatI2S& Wv,
        std::size_t j0, std::size_t kb, float xsA, float xsB)
    {
        const float wsQ = load_forward_tile_scale_block32(Wq, j0, kb);
        const float wsK = load_forward_tile_scale_block32(Wk, j0, kb);
        const float wsV = load_forward_tile_scale_block32(Wv, j0, kb);

        const __m256 gainQA = _mm256_set1_ps(xsA * wsQ), gainQB = _mm256_set1_ps(xsB * wsQ);
        const __m256 gainKA = _mm256_set1_ps(xsA * wsK), gainKB = _mm256_set1_ps(xsB * wsK);
        const __m256 gainVA = _mm256_set1_ps(xsA * wsV), gainVB = _mm256_set1_ps(xsB * wsV);

        for (int off = 0; off < 32; off += 8) {
            __m256 qA = _mm256_loadu_ps(qRowA + j0 + off), qB = _mm256_loadu_ps(qRowB + j0 + off);
            __m256 kA = _mm256_loadu_ps(kRowA + j0 + off), kB = _mm256_loadu_ps(kRowB + j0 + off);
            __m256 vA = _mm256_loadu_ps(vRowA + j0 + off), vB = _mm256_loadu_ps(vRowB + j0 + off);

            qA = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQA + off))), gainQA, qA);
            qB = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQB + off))), gainQB, qB);
            kA = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKA + off))), gainKA, kA);
            kB = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKB + off))), gainKB, kB);
            vA = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVA + off))), gainVA, vA);
            vB = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVB + off))), gainVB, vB);

            _mm256_storeu_ps(qRowA + j0 + off, qA); _mm256_storeu_ps(qRowB + j0 + off, qB);
            _mm256_storeu_ps(kRowA + j0 + off, kA); _mm256_storeu_ps(kRowB + j0 + off, kB);
            _mm256_storeu_ps(vRowA + j0 + off, vA); _mm256_storeu_ps(vRowB + j0 + off, vB);
        }
    }

    static void validate_forward_common(const BitNetMatI2S& W, std::size_t inDim, std::size_t outDim, const char* who) {
        if (W.rows != inDim || W.cols != outDim || W.qk == 0 || W.jq != 32 || (outDim % 32) != 0) {
            throw std::runtime_error(std::string(who) + ": H/jq setup invalid");
        }
    }

    // ============================================================
    // Backward Helpers (Packed Transpose Option B)
    // ============================================================

    static inline float load_backward_tile_scale_block32(
        const BitNetMatI2S& W, std::size_t i0, std::size_t jb)
    {
        const std::size_t nJBlocks = ceil_div(W.cols, W.jq);
        const std::size_t kb = i0 / W.qk;
        return W.w_scale_tileT[kb * nJBlocks + jb];
    }

    static inline void accum_bwd_block32_scaled_to_x(
        float* __restrict dxRow, const int32_t* __restrict blockAcc,
        const BitNetMatI2S& W, std::size_t i0, std::size_t jb, float dy_scale)
    {
        const __m256 gain = _mm256_set1_ps(dy_scale * load_backward_tile_scale_block32(W, i0, jb));
        for (int off = 0; off < 32; off += 8) {
            const __m256 accF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAcc + off)));
            __m256 dx = _mm256_loadu_ps(dxRow + i0 + off);
            _mm256_storeu_ps(dxRow + i0 + off, _mm256_fmadd_ps(accF, gain, dx));
        }
    }

    static inline void accum_bwd_block32_scaled_to_x_pair(
        float* __restrict dxRowA, float* __restrict dxRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        const BitNetMatI2S& W, std::size_t i0, std::size_t jb, float dy_scaleA, float dy_scaleB)
    {
        const float ws = load_backward_tile_scale_block32(W, i0, jb);
        const __m256 gainA = _mm256_set1_ps(dy_scaleA * ws), gainB = _mm256_set1_ps(dy_scaleB * ws);

        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 dxA = _mm256_loadu_ps(dxRowA + i0 + off);
            __m256 dxB = _mm256_loadu_ps(dxRowB + i0 + off);
            _mm256_storeu_ps(dxRowA + i0 + off, _mm256_fmadd_ps(accFA, gainA, dxA));
            _mm256_storeu_ps(dxRowB + i0 + off, _mm256_fmadd_ps(accFB, gainB, dxB));
        }
    }

    static void validate_backward_packed(const BitNetMatI2S& W, std::size_t inDim, std::size_t outDim, const char* who) {
        if (W.rows != inDim || W.cols != outDim || W.qk == 0 || W.jq == 0 || (inDim % 32) != 0 || (W.qk % 32) != 0) {
            throw std::runtime_error(std::string(who) + ": dims/qk/jq bounds setup invalid for bwd layout");
        }
    }

} // namespace

// ============================================================
// Forward Kernels
// ============================================================

void linear_forward_bitnet_i2s_accum(
    float* __restrict Y, const float* __restrict X, const BitNetMatI2S& W,
    std::size_t T, std::size_t tTile,
    int8_t* __restrict x8Scratch, float* __restrict xScaleScratch)
{
    const std::size_t inDim = W.rows;
    const std::size_t outDim = W.cols;
    validate_forward_common(W, inDim, outDim, "linear_forward_bitnet_i2s_accum");
    const std::size_t hBlock = W.qk;
    int8_t* const x8 = x8Scratch; float* const xScale = xScaleScratch;

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        quantize_x_tile_i8_tiled(x8, xScale, X, T, inDim, t0, tCount, tTile, hBlock);

        const int outDim_int = static_cast<int>(outDim);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int j0_i = 0; j0_i < outDim_int; j0_i += 32) {
            std::size_t j0 = static_cast<std::size_t>(j0_i);

            for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                const std::size_t hCount = std::min(hBlock, inDim - h0);
                const std::size_t kb = h0 / hBlock;
                const float xs_block = xScale[kb];

                std::size_t tt = 0;
                for (; tt + 3 < tCount; tt += 4) {
                    const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                    const int8_t* xC = x8 + (tt + 2) * inDim + h0; const int8_t* xD = x8 + (tt + 3) * inDim + h0;

                    float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                    float* yC = Y + (t0 + tt + 2) * outDim; float* yD = Y + (t0 + tt + 3) * outDim;

                    int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);

                    accum_block32_scaled_to_y_pair(yA, yB, aA, aB, W, j0, kb, xs_block, xs_block);
                    accum_block32_scaled_to_y_pair(yC, yD, aC, aD, W, j0, kb, xs_block, xs_block);
                }

                for (; tt + 1 < tCount; tt += 2) {
                    const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                    float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                    int32_t aA[32] = { 0 }, aB[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                    accum_block32_scaled_to_y_pair(yA, yB, aA, aB, W, j0, kb, xs_block, xs_block);
                }

                for (; tt < tCount; ++tt) {
                    float* yRow = Y + (t0 + tt) * outDim;
                    int32_t blockAcc[32] = { 0 };
                    dot_i2s_i8_accum_i32_block32(blockAcc, W.q2.data(), outDim, h0, hCount, x8 + tt * inDim + h0, j0);
                    accum_block32_scaled_to_y(yRow, blockAcc, W, j0, kb, xs_block);
                }
            }
        }
    }
}

void linear_fused_qkv_forward_bitnet_i2s_accum(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const BitNetMatI2S& Wq, const BitNetMatI2S& Wk, const BitNetMatI2S& Wv,
    std::size_t T, std::size_t tTile,
    int8_t* __restrict x8Scratch, float* __restrict xScaleScratch)
{
    const std::size_t inDim = Wq.rows;
    const std::size_t outDim = Wq.cols;
    validate_forward_common(Wq, inDim, outDim, "linear_fused_qkv_forward_bitnet_i2s_accum");
    const std::size_t hBlock = Wq.qk;
    int8_t* const x8 = x8Scratch; float* const xScale = xScaleScratch;

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        quantize_x_tile_i8_tiled(x8, xScale, X, T, inDim, t0, tCount, tTile, hBlock);

        const int outDim_int = static_cast<int>(outDim);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int j0_i = 0; j0_i < outDim_int; j0_i += 32) {
            std::size_t j0 = static_cast<std::size_t>(j0_i);

            for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                const std::size_t hCount = std::min(hBlock, inDim - h0);
                const std::size_t kb = h0 / hBlock;
                const float xs_block = xScale[kb];

                std::size_t tt = 0;
                for (; tt + 3 < tCount; tt += 4) {
                    const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                    const int8_t* xC = x8 + (tt + 2) * inDim + h0; const int8_t* xD = x8 + (tt + 3) * inDim + h0;

                    float* qA = Q + (t0 + tt + 0) * outDim; float* kA = K + (t0 + tt + 0) * outDim; float* vA = V + (t0 + tt + 0) * outDim;
                    float* qB = Q + (t0 + tt + 1) * outDim; float* kB = K + (t0 + tt + 1) * outDim; float* vB = V + (t0 + tt + 1) * outDim;
                    float* qC = Q + (t0 + tt + 2) * outDim; float* kC = K + (t0 + tt + 2) * outDim; float* vC = V + (t0 + tt + 2) * outDim;
                    float* qD = Q + (t0 + tt + 3) * outDim; float* kD = K + (t0 + tt + 3) * outDim; float* vD = V + (t0 + tt + 3) * outDim;

                    int32_t qQA[32] = { 0 }, qQB[32] = { 0 }, qQC[32] = { 0 }, qQD[32] = { 0 };
                    int32_t qKA[32] = { 0 }, qKB[32] = { 0 }, qKC[32] = { 0 }, qKD[32] = { 0 };
                    int32_t qVA[32] = { 0 }, qVB[32] = { 0 }, qVC[32] = { 0 }, qVD[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32_x4(qQA, qQB, qQC, qQD, Wq.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                    dot_i2s_i8_accum_i32_block32_x4(qKA, qKB, qKC, qKD, Wk.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                    dot_i2s_i8_accum_i32_block32_x4(qVA, qVB, qVC, qVD, Wv.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);

                    accum_block32_scaled_to_y3_pair(qA, kA, vA, qB, kB, vB, qQA, qKA, qVA, qQB, qKB, qVB, Wq, Wk, Wv, j0, kb, xs_block, xs_block);
                    accum_block32_scaled_to_y3_pair(qC, kC, vC, qD, kD, vD, qQC, qKC, qVC, qQD, qKD, qVD, Wq, Wk, Wv, j0, kb, xs_block, xs_block);
                }

                for (; tt + 1 < tCount; tt += 2) {
                    const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                    float* qA = Q + (t0 + tt + 0) * outDim; float* kA = K + (t0 + tt + 0) * outDim; float* vA = V + (t0 + tt + 0) * outDim;
                    float* qB = Q + (t0 + tt + 1) * outDim; float* kB = K + (t0 + tt + 1) * outDim; float* vB = V + (t0 + tt + 1) * outDim;
                    int32_t qQA[32] = { 0 }, qKA[32] = { 0 }, qVA[32] = { 0 }, qQB[32] = { 0 }, qKB[32] = { 0 }, qVB[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32_x2(qQA, qQB, Wq.q2.data(), outDim, h0, hCount, xA, xB, j0);
                    dot_i2s_i8_accum_i32_block32_x2(qKA, qKB, Wk.q2.data(), outDim, h0, hCount, xA, xB, j0);
                    dot_i2s_i8_accum_i32_block32_x2(qVA, qVB, Wv.q2.data(), outDim, h0, hCount, xA, xB, j0);

                    accum_block32_scaled_to_y3_pair(qA, kA, vA, qB, kB, vB, qQA, qKA, qVA, qQB, qKB, qVB, Wq, Wk, Wv, j0, kb, xs_block, xs_block);
                }

                for (; tt < tCount; ++tt) {
                    const int8_t* xRow8 = x8 + tt * inDim + h0;
                    float* qRow = Q + (t0 + tt) * outDim; float* kRow = K + (t0 + tt) * outDim; float* vRow = V + (t0 + tt) * outDim;
                    int32_t accQ[32] = { 0 }, accK[32] = { 0 }, accV[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32(accQ, Wq.q2.data(), outDim, h0, hCount, xRow8, j0);
                    dot_i2s_i8_accum_i32_block32(accK, Wk.q2.data(), outDim, h0, hCount, xRow8, j0);
                    dot_i2s_i8_accum_i32_block32(accV, Wv.q2.data(), outDim, h0, hCount, xRow8, j0);

                    accum_block32_scaled_to_y3(qRow, kRow, vRow, accQ, accK, accV, Wq, Wk, Wv, j0, kb, xs_block);
                }
            }
        }
    }
}

// ============================================================
// Backward Kernels (Packed Transpose Option B)
// ============================================================

void linear_backward_bitnet_i2s_packed(
    float* __restrict dX, const float* __restrict dY, const BitNetMatI2S& W,
    std::size_t T, std::size_t tTile,
    int8_t* __restrict dy8Scratch, float* __restrict dyScaleScratch)
{
    const std::size_t inDim = W.rows;
    const std::size_t outDim = W.cols;
    validate_backward_packed(W, inDim, outDim, "linear_backward_bitnet_i2s_packed");
    const std::size_t hBlock = W.jq;

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        quantize_x_tile_i8_tiled(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, tTile, hBlock);

        const int inDim_int = static_cast<int>(inDim);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i0_i = 0; i0_i < inDim_int; i0_i += 32) {
            std::size_t i0 = static_cast<std::size_t>(i0_i);

            for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                const std::size_t jCount = std::min(hBlock, outDim - j0);
                const std::size_t jb = j0 / hBlock;
                const float dy_scale = dyScaleScratch[jb];

                std::size_t tt = 0;
                for (; tt + 3 < tCount; tt += 4) {
                    const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dyC = dy8Scratch + (tt + 2) * outDim + j0; const int8_t* dyD = dy8Scratch + (tt + 3) * outDim + j0;
                    float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                    float* dxC = dX + (t0 + tt + 2) * inDim; float* dxD = dX + (t0 + tt + 3) * inDim;

                    int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2T.data(), inDim, j0, jCount, dyA, dyB, dyC, dyD, i0);

                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aA, aB, W, i0, jb, dy_scale, dy_scale);
                    accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aC, aD, W, i0, jb, dy_scale, dy_scale);
                }

                for (; tt + 1 < tCount; tt += 2) {
                    const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                    float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                    int32_t aA[32] = { 0 }, aB[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2T.data(), inDim, j0, jCount, dyA, dyB, i0);
                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aA, aB, W, i0, jb, dy_scale, dy_scale);
                }

                for (; tt < tCount; ++tt) {
                    float* dxRow = dX + (t0 + tt) * inDim;
                    int32_t acc[32] = { 0 };
                    dot_i2s_i8_accum_i32_block32(acc, W.q2T.data(), inDim, j0, jCount, dy8Scratch + tt * outDim + j0, i0);
                    accum_bwd_block32_scaled_to_x(dxRow, acc, W, i0, jb, dy_scale);
                }
            }
        }
    }
}

void linear_fused_qkv_backward_bitnet_i2s_packed(
    float* __restrict dX, const float* __restrict dQ, const float* __restrict dK, const float* __restrict dV,
    const BitNetMatI2S& Wq, const BitNetMatI2S& Wk, const BitNetMatI2S& Wv,
    std::size_t T, std::size_t tTile,
    int8_t* __restrict dq8Scratch, float* __restrict dqScaleScratch,
    int8_t* __restrict dk8Scratch, float* __restrict dkScaleScratch,
    int8_t* __restrict dv8Scratch, float* __restrict dvScaleScratch)
{
    const std::size_t inDim = Wq.rows;
    const std::size_t outDim = Wq.cols;
    validate_backward_packed(Wq, inDim, outDim, "linear_fused_qkv_backward_bitnet_i2s_packed");
    const std::size_t hBlock = Wq.jq;

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);

        quantize_x_tile_i8_tiled(dq8Scratch, dqScaleScratch, dQ, T, outDim, t0, tCount, tTile, hBlock);
        quantize_x_tile_i8_tiled(dk8Scratch, dkScaleScratch, dK, T, outDim, t0, tCount, tTile, hBlock);
        quantize_x_tile_i8_tiled(dv8Scratch, dvScaleScratch, dV, T, outDim, t0, tCount, tTile, hBlock);

        const int inDim_int = static_cast<int>(inDim);

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i0_i = 0; i0_i < inDim_int; i0_i += 32) {
            std::size_t i0 = static_cast<std::size_t>(i0_i);

            for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                const std::size_t jCount = std::min(hBlock, outDim - j0);
                const std::size_t jb = j0 / hBlock;

                const float sq = dqScaleScratch[jb], sk = dkScaleScratch[jb], sv = dvScaleScratch[jb];

                std::size_t tt = 0;
                for (; tt + 3 < tCount; tt += 4) {
                    const int8_t* dqA = dq8Scratch + (tt + 0) * outDim + j0; const int8_t* dqB = dq8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dqC = dq8Scratch + (tt + 2) * outDim + j0; const int8_t* dqD = dq8Scratch + (tt + 3) * outDim + j0;

                    const int8_t* dkA = dk8Scratch + (tt + 0) * outDim + j0; const int8_t* dkB = dk8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dkC = dk8Scratch + (tt + 2) * outDim + j0; const int8_t* dkD = dk8Scratch + (tt + 3) * outDim + j0;

                    const int8_t* dvA = dv8Scratch + (tt + 0) * outDim + j0; const int8_t* dvB = dv8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dvC = dv8Scratch + (tt + 2) * outDim + j0; const int8_t* dvD = dv8Scratch + (tt + 3) * outDim + j0;

                    float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                    float* dxC = dX + (t0 + tt + 2) * inDim; float* dxD = dX + (t0 + tt + 3) * inDim;

                    int32_t aQA[32] = { 0 }, aQB[32] = { 0 }, aQC[32] = { 0 }, aQD[32] = { 0 };
                    int32_t aKA[32] = { 0 }, aKB[32] = { 0 }, aKC[32] = { 0 }, aKD[32] = { 0 };
                    int32_t aVA[32] = { 0 }, aVB[32] = { 0 }, aVC[32] = { 0 }, aVD[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32_x4(aQA, aQB, aQC, aQD, Wq.q2T.data(), inDim, j0, jCount, dqA, dqB, dqC, dqD, i0);
                    dot_i2s_i8_accum_i32_block32_x4(aKA, aKB, aKC, aKD, Wk.q2T.data(), inDim, j0, jCount, dkA, dkB, dkC, dkD, i0);
                    dot_i2s_i8_accum_i32_block32_x4(aVA, aVB, aVC, aVD, Wv.q2T.data(), inDim, j0, jCount, dvA, dvB, dvC, dvD, i0);

                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aQA, aQB, Wq, i0, jb, sq, sq);
                    accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aQC, aQD, Wq, i0, jb, sq, sq);
                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aKA, aKB, Wk, i0, jb, sk, sk);
                    accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aKC, aKD, Wk, i0, jb, sk, sk);
                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aVA, aVB, Wv, i0, jb, sv, sv);
                    accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aVC, aVD, Wv, i0, jb, sv, sv);
                }

                for (; tt + 1 < tCount; tt += 2) {
                    const int8_t* dqA = dq8Scratch + (tt + 0) * outDim + j0; const int8_t* dqB = dq8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dkA = dk8Scratch + (tt + 0) * outDim + j0; const int8_t* dkB = dk8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dvA = dv8Scratch + (tt + 0) * outDim + j0; const int8_t* dvB = dv8Scratch + (tt + 1) * outDim + j0;
                    float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;

                    int32_t aQA[32] = { 0 }, aQB[32] = { 0 }, aKA[32] = { 0 }, aKB[32] = { 0 }, aVA[32] = { 0 }, aVB[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32_x2(aQA, aQB, Wq.q2T.data(), inDim, j0, jCount, dqA, dqB, i0);
                    dot_i2s_i8_accum_i32_block32_x2(aKA, aKB, Wk.q2T.data(), inDim, j0, jCount, dkA, dkB, i0);
                    dot_i2s_i8_accum_i32_block32_x2(aVA, aVB, Wv.q2T.data(), inDim, j0, jCount, dvA, dvB, i0);

                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aQA, aQB, Wq, i0, jb, sq, sq);
                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aKA, aKB, Wk, i0, jb, sk, sk);
                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aVA, aVB, Wv, i0, jb, sv, sv);
                }

                for (; tt < tCount; ++tt) {
                    float* dxRow = dX + (t0 + tt) * inDim;
                    int32_t aQ[32] = { 0 }, aK[32] = { 0 }, aV[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32(aQ, Wq.q2T.data(), inDim, j0, jCount, dq8Scratch + tt * outDim + j0, i0);
                    dot_i2s_i8_accum_i32_block32(aK, Wk.q2T.data(), inDim, j0, jCount, dk8Scratch + tt * outDim + j0, i0);
                    dot_i2s_i8_accum_i32_block32(aV, Wv.q2T.data(), inDim, j0, jCount, dv8Scratch + tt * outDim + j0, i0);

                    accum_bwd_block32_scaled_to_x(dxRow, aQ, Wq, i0, jb, sq);
                    accum_bwd_block32_scaled_to_x(dxRow, aK, Wk, i0, jb, sk);
                    accum_bwd_block32_scaled_to_x(dxRow, aV, Wv, i0, jb, sv);
                }
            }
        }
    }
}