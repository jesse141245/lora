#include "bitnet_i2s_matmul.h"
#include "bitnet_i2_s.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <immintrin.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

    static constexpr std::size_t kRowsPerQuantScale = 1;

    static inline std::size_t ceil_div(std::size_t a, std::size_t b) {
        return (a + b - 1) / b;
    }

    static inline float load_tile_row_scale(
        const float* scaleScratch, std::size_t nBlocks, std::size_t tileRow, std::size_t block)
    {
        return scaleScratch[tileRow * nBlocks + block];
    }

    // ============================================================
    // Forward Helpers
    // ============================================================

    static inline const float* forward_scale_row(
        const BitNetMatI2S& W, std::size_t j0, std::size_t nKBlocks)
    {
        return W.w_scale_tile.data() + (j0 / W.jq) * nKBlocks;
    }

    static inline bool fused_x4_forward_enabled() {
        static const bool cached = []() {
            const char* value = std::getenv("QLORA_TEST_BITNET_FUSED_X4_FORWARD");
            if (!value || *value == '\0') return false;
            std::string s(value);
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return s == "1" || s == "true" || s == "yes" || s == "on";
        }();
        return cached;
    }

    static inline void accum_block32_scaled_to_y(
        float* __restrict yRow, const int32_t* __restrict blockAcc,
        const float* __restrict scaleRow, std::size_t j0, std::size_t kb, float xs)
    {
        const __m256 gain8 = _mm256_set1_ps(xs * scaleRow[kb]);
        for (int off = 0; off < 32; off += 8) {
            const __m256 accF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAcc + off)));
            __m256 y = _mm256_loadu_ps(yRow + j0 + off);
            _mm256_storeu_ps(yRow + j0 + off, _mm256_fmadd_ps(accF, gain8, y));
        }
    }

    static inline void accum_block32_scaled_to_y_pair(
        float* __restrict yRowA, float* __restrict yRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        const float* __restrict scaleRow, std::size_t j0, std::size_t kb, float xsA, float xsB)
    {
        const float wsScalar = scaleRow[kb];
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

    static inline void accum_block32_scaled_to_y_dual(
        float* __restrict yRowA, float* __restrict yRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        const float* __restrict scaleRowA, const float* __restrict scaleRowB,
        std::size_t j0, std::size_t kb, float xsA, float xsB)
    {
        const __m256 gainA8 = _mm256_set1_ps(xsA * scaleRowA[kb]);
        const __m256 gainB8 = _mm256_set1_ps(xsB * scaleRowB[kb]);
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
        const float* __restrict scaleRowQ, const float* __restrict scaleRowK, const float* __restrict scaleRowV,
        std::size_t j0, std::size_t kb, float xs)
    {
        const __m256 gainQ = _mm256_set1_ps(xs * scaleRowQ[kb]);
        const __m256 gainK = _mm256_set1_ps(xs * scaleRowK[kb]);
        const __m256 gainV = _mm256_set1_ps(xs * scaleRowV[kb]);

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
        const float* __restrict scaleRowQ, const float* __restrict scaleRowK, const float* __restrict scaleRowV,
        std::size_t j0, std::size_t kb, float xsA, float xsB)
    {
        const float wsQ = scaleRowQ[kb];
        const float wsK = scaleRowK[kb];
        const float wsV = scaleRowV[kb];

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

    static inline const float* backward_scale_row(
        const BitNetMatI2S& W, std::size_t i0, std::size_t nJBlocks)
    {
        return W.w_scale_tileT.data() + (i0 / W.qk) * nJBlocks;
    }

    static inline void accum_bwd_block32_scaled_to_x(
        float* __restrict dxRow, const int32_t* __restrict blockAcc,
        const float* __restrict scaleRow, std::size_t i0, std::size_t jb, float dy_scale)
    {
        const __m256 gain = _mm256_set1_ps(dy_scale * scaleRow[jb]);
        for (int off = 0; off < 32; off += 8) {
            const __m256 accF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAcc + off)));
            __m256 dx = _mm256_loadu_ps(dxRow + i0 + off);
            _mm256_storeu_ps(dxRow + i0 + off, _mm256_fmadd_ps(accF, gain, dx));
        }
    }

    static inline void accum_bwd_block32_scaled_to_x_pair(
        float* __restrict dxRowA, float* __restrict dxRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        const float* __restrict scaleRow, std::size_t i0, std::size_t jb, float dy_scaleA, float dy_scaleB)
    {
        const float ws = scaleRow[jb];
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

    static inline void accum_bwd_block32_scaled_to_x_dual(
        float* __restrict dxRow,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        const float* __restrict scaleRowA, const float* __restrict scaleRowB,
        std::size_t i0, std::size_t jb, float dyScaleA, float dyScaleB)
    {
        const __m256 gainA = _mm256_set1_ps(dyScaleA * scaleRowA[jb]);
        const __m256 gainB = _mm256_set1_ps(dyScaleB * scaleRowB[jb]);
        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 dx = _mm256_loadu_ps(dxRow + i0 + off);
            dx = _mm256_fmadd_ps(accFA, gainA, dx);
            dx = _mm256_fmadd_ps(accFB, gainB, dx);
            _mm256_storeu_ps(dxRow + i0 + off, dx);
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
    const std::size_t nKBlocks = ceil_div(inDim, hBlock);
    int8_t* const x8 = x8Scratch; float* const xScale = xScaleScratch;
    const int outDim_int = static_cast<int>(outDim);
    const bool useFusedX4 = fused_x4_forward_enabled();

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
            const std::size_t tCount = std::min(tTile, T - t0);
#if defined(_OPENMP)
 #pragma omp single
            quantize_x_tile_i8_tiled(x8, xScale, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
#else
            quantize_x_tile_i8_tiled(x8, xScale, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
#endif

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int j0_i = 0; j0_i < outDim_int; j0_i += 32) {
                std::size_t j0 = static_cast<std::size_t>(j0_i);
                const float* const scaleRow = forward_scale_row(W, j0, nKBlocks);

                for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                    const std::size_t hCount = std::min(hBlock, inDim - h0);
                    const std::size_t kb = h0 / hBlock;

                    std::size_t tt = 0;
                    for (; tt + 3 < tCount; tt += 4) {
                        const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                        const int8_t* xC = x8 + (tt + 2) * inDim + h0; const int8_t* xD = x8 + (tt + 3) * inDim + h0;

                        float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                        float* yC = Y + (t0 + tt + 2) * outDim; float* yD = Y + (t0 + tt + 3) * outDim;

                        if (useFusedX4) {
                            const float ws = scaleRow[kb];
                            const float gainA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb) * ws;
                            const float gainB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb) * ws;
                            const float gainC = load_tile_row_scale(xScale, nKBlocks, tt + 2, kb) * ws;
                            const float gainD = load_tile_row_scale(xScale, nKBlocks, tt + 3, kb) * ws;

                            dot_i2s_i8_scale_store_block16_x4_exact(
                                yA, yB, yC, yD,
                                W.q2.data(), outDim, h0, hCount,
                                xA, xB, xC, xD,
                                j0,
                                gainA, gainB, gainC, gainD);
                            dot_i2s_i8_scale_store_block16_x4_exact(
                                yA, yB, yC, yD,
                                W.q2.data(), outDim, h0, hCount,
                                xA, xB, xC, xD,
                                j0 + 16,
                                gainA, gainB, gainC, gainD);
                        }
                        else {
                            int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };
                            const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                            const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);
                            const float xsC = load_tile_row_scale(xScale, nKBlocks, tt + 2, kb);
                            const float xsD = load_tile_row_scale(xScale, nKBlocks, tt + 3, kb);

                            dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                            accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB);
                            accum_block32_scaled_to_y_pair(yC, yD, aC, aD, scaleRow, j0, kb, xsC, xsD);
                        }
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                        float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                        int32_t aA[32] = { 0 }, aB[32] = { 0 };
                        const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                        const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);

                        dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                        accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB);
                    }

                    for (; tt < tCount; ++tt) {
                        float* yRow = Y + (t0 + tt) * outDim;
                        int32_t blockAcc[32] = { 0 };
                        const float xs = load_tile_row_scale(xScale, nKBlocks, tt, kb);
                        dot_i2s_i8_accum_i32_block32(blockAcc, W.q2.data(), outDim, h0, hCount, x8 + tt * inDim + h0, j0);
                        accum_block32_scaled_to_y(yRow, blockAcc, scaleRow, j0, kb, xs);
                    }
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
    const std::size_t qDim = Wq.cols;
    const std::size_t kDim = Wk.cols;
    const std::size_t vDim = Wv.cols;
    validate_forward_common(Wq, inDim, qDim, "linear_fused_qkv_forward_bitnet_i2s_accum/Wq");
    validate_forward_common(Wk, inDim, kDim, "linear_fused_qkv_forward_bitnet_i2s_accum/Wk");
    validate_forward_common(Wv, inDim, vDim, "linear_fused_qkv_forward_bitnet_i2s_accum/Wv");
    if (Wk.rows != inDim || Wv.rows != inDim || Wq.qk != Wk.qk || Wq.qk != Wv.qk || Wq.hadamard32 != Wk.hadamard32 || Wq.hadamard32 != Wv.hadamard32) {
        throw std::runtime_error("linear_fused_qkv_forward_bitnet_i2s_accum: incompatible Q/K/V shapes");
    }
    const std::size_t hBlock = Wq.qk;
    const std::size_t nKBlocks = ceil_div(inDim, hBlock);
    int8_t* const x8 = x8Scratch; float* const xScale = xScaleScratch;
    const std::size_t fusedCols = std::min(qDim, std::min(kDim, vDim));
    const bool useFusedX4 = fused_x4_forward_enabled();

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        quantize_x_tile_i8_tiled(x8, xScale, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);

#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
            auto run_single = [&](float* __restrict Y, const BitNetMatI2S& W, std::size_t outDim, std::size_t jBegin) {
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
                for (int j0_i = static_cast<int>(jBegin); j0_i < static_cast<int>(outDim); j0_i += 32) {
                    const std::size_t j0 = static_cast<std::size_t>(j0_i);
                    const float* const scaleRow = forward_scale_row(W, j0, nKBlocks);
                    for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                        const std::size_t hCount = std::min(hBlock, inDim - h0);
                        const std::size_t kb = h0 / hBlock;

                        std::size_t tt = 0;
                        for (; tt + 3 < tCount; tt += 4) {
                            const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                            const int8_t* xC = x8 + (tt + 2) * inDim + h0; const int8_t* xD = x8 + (tt + 3) * inDim + h0;
                            float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                            float* yC = Y + (t0 + tt + 2) * outDim; float* yD = Y + (t0 + tt + 3) * outDim;
                            if (useFusedX4) {
                                const float ws = scaleRow[kb];
                                const float gainA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb) * ws;
                                const float gainB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb) * ws;
                                const float gainC = load_tile_row_scale(xScale, nKBlocks, tt + 2, kb) * ws;
                                const float gainD = load_tile_row_scale(xScale, nKBlocks, tt + 3, kb) * ws;

                                dot_i2s_i8_scale_store_block16_x4_exact(
                                    yA, yB, yC, yD,
                                    W.q2.data(), outDim, h0, hCount,
                                    xA, xB, xC, xD,
                                    j0,
                                    gainA, gainB, gainC, gainD);
                                dot_i2s_i8_scale_store_block16_x4_exact(
                                    yA, yB, yC, yD,
                                    W.q2.data(), outDim, h0, hCount,
                                    xA, xB, xC, xD,
                                    j0 + 16,
                                    gainA, gainB, gainC, gainD);
                            }
                            else {
                                int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };
                                const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                                const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);
                                const float xsC = load_tile_row_scale(xScale, nKBlocks, tt + 2, kb);
                                const float xsD = load_tile_row_scale(xScale, nKBlocks, tt + 3, kb);

                                dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                                accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB);
                                accum_block32_scaled_to_y_pair(yC, yD, aC, aD, scaleRow, j0, kb, xsC, xsD);
                            }
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                            float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                            const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);

                            dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB);
                        }

                        for (; tt < tCount; ++tt) {
                            float* yRow = Y + (t0 + tt) * outDim;
                            int32_t acc[32] = { 0 };
                            const float xs = load_tile_row_scale(xScale, nKBlocks, tt, kb);
                            dot_i2s_i8_accum_i32_block32(acc, W.q2.data(), outDim, h0, hCount, x8 + tt * inDim + h0, j0);
                            accum_block32_scaled_to_y(yRow, acc, scaleRow, j0, kb, xs);
                        }
                    }
                }
            };

#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
            for (int j0_i = 0; j0_i < static_cast<int>(fusedCols); j0_i += 32) {
                const std::size_t j0 = static_cast<std::size_t>(j0_i);
                const float* const scaleRowQ = forward_scale_row(Wq, j0, nKBlocks);
                const float* const scaleRowK = forward_scale_row(Wk, j0, nKBlocks);
                const float* const scaleRowV = forward_scale_row(Wv, j0, nKBlocks);
                for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                    const std::size_t hCount = std::min(hBlock, inDim - h0);
                    const std::size_t kb = h0 / hBlock;

                    std::size_t tt = 0;
                    for (; tt + 3 < tCount; tt += 4) {
                        const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                        const int8_t* xC = x8 + (tt + 2) * inDim + h0; const int8_t* xD = x8 + (tt + 3) * inDim + h0;

                        float* qA = Q + (t0 + tt + 0) * qDim; float* kA = K + (t0 + tt + 0) * kDim; float* vA = V + (t0 + tt + 0) * vDim;
                        float* qB = Q + (t0 + tt + 1) * qDim; float* kB = K + (t0 + tt + 1) * kDim; float* vB = V + (t0 + tt + 1) * vDim;
                        float* qC = Q + (t0 + tt + 2) * qDim; float* kC = K + (t0 + tt + 2) * kDim; float* vC = V + (t0 + tt + 2) * vDim;
                        float* qD = Q + (t0 + tt + 3) * qDim; float* kD = K + (t0 + tt + 3) * kDim; float* vD = V + (t0 + tt + 3) * vDim;

                        int32_t qQA[32] = { 0 }, qQB[32] = { 0 }, qQC[32] = { 0 }, qQD[32] = { 0 };
                        int32_t qKA[32] = { 0 }, qKB[32] = { 0 }, qKC[32] = { 0 }, qKD[32] = { 0 };
                        int32_t qVA[32] = { 0 }, qVB[32] = { 0 }, qVC[32] = { 0 }, qVD[32] = { 0 };
                        const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                        const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);
                        const float xsC = load_tile_row_scale(xScale, nKBlocks, tt + 2, kb);
                        const float xsD = load_tile_row_scale(xScale, nKBlocks, tt + 3, kb);

                        dot_i2s_i8_accum_i32_block32_x4(qQA, qQB, qQC, qQD, Wq.q2.data(), qDim, h0, hCount, xA, xB, xC, xD, j0);
                        dot_i2s_i8_accum_i32_block32_x4(qKA, qKB, qKC, qKD, Wk.q2.data(), kDim, h0, hCount, xA, xB, xC, xD, j0);
                        dot_i2s_i8_accum_i32_block32_x4(qVA, qVB, qVC, qVD, Wv.q2.data(), vDim, h0, hCount, xA, xB, xC, xD, j0);

                        accum_block32_scaled_to_y3_pair(qA, kA, vA, qB, kB, vB, qQA, qKA, qVA, qQB, qKB, qVB, scaleRowQ, scaleRowK, scaleRowV, j0, kb, xsA, xsB);
                        accum_block32_scaled_to_y3_pair(qC, kC, vC, qD, kD, vD, qQC, qKC, qVC, qQD, qKD, qVD, scaleRowQ, scaleRowK, scaleRowV, j0, kb, xsC, xsD);
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                        float* qA = Q + (t0 + tt + 0) * qDim; float* kA = K + (t0 + tt + 0) * kDim; float* vA = V + (t0 + tt + 0) * vDim;
                        float* qB = Q + (t0 + tt + 1) * qDim; float* kB = K + (t0 + tt + 1) * kDim; float* vB = V + (t0 + tt + 1) * vDim;
                        int32_t qQA[32] = { 0 }, qKA[32] = { 0 }, qVA[32] = { 0 }, qQB[32] = { 0 }, qKB[32] = { 0 }, qVB[32] = { 0 };
                        const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                        const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);

                        dot_i2s_i8_accum_i32_block32_x2(qQA, qQB, Wq.q2.data(), qDim, h0, hCount, xA, xB, j0);
                        dot_i2s_i8_accum_i32_block32_x2(qKA, qKB, Wk.q2.data(), kDim, h0, hCount, xA, xB, j0);
                        dot_i2s_i8_accum_i32_block32_x2(qVA, qVB, Wv.q2.data(), vDim, h0, hCount, xA, xB, j0);

                        accum_block32_scaled_to_y3_pair(qA, kA, vA, qB, kB, vB, qQA, qKA, qVA, qQB, qKB, qVB, scaleRowQ, scaleRowK, scaleRowV, j0, kb, xsA, xsB);
                    }

                    for (; tt < tCount; ++tt) {
                        const int8_t* xRow8 = x8 + tt * inDim + h0;
                        float* qRow = Q + (t0 + tt) * qDim; float* kRow = K + (t0 + tt) * kDim; float* vRow = V + (t0 + tt) * vDim;
                        int32_t accQ[32] = { 0 }, accK[32] = { 0 }, accV[32] = { 0 };
                        const float xs = load_tile_row_scale(xScale, nKBlocks, tt, kb);

                        dot_i2s_i8_accum_i32_block32(accQ, Wq.q2.data(), qDim, h0, hCount, xRow8, j0);
                        dot_i2s_i8_accum_i32_block32(accK, Wk.q2.data(), kDim, h0, hCount, xRow8, j0);
                        dot_i2s_i8_accum_i32_block32(accV, Wv.q2.data(), vDim, h0, hCount, xRow8, j0);

                        accum_block32_scaled_to_y3(qRow, kRow, vRow, accQ, accK, accV, scaleRowQ, scaleRowK, scaleRowV, j0, kb, xs);
                    }
                }
            }

            if (qDim > fusedCols) run_single(Q, Wq, qDim, fusedCols);
            if (kDim > fusedCols) run_single(K, Wk, kDim, fusedCols);
            if (vDim > fusedCols) run_single(V, Wv, vDim, fusedCols);
        }
    }
}

void linear_fused_dual_forward_bitnet_i2s_accum(
    float* __restrict Ya, float* __restrict Yb,
    const float* __restrict X,
    const BitNetMatI2S& Wa, const BitNetMatI2S& Wb,
    std::size_t T, std::size_t tTile,
    int8_t* __restrict x8Scratch, float* __restrict xScaleScratch)
{
    const std::size_t inDim = Wa.rows;
    const std::size_t outDim = Wa.cols;
    validate_forward_common(Wa, inDim, outDim, "linear_fused_dual_forward_bitnet_i2s_accum/Wa");
    validate_forward_common(Wb, Wb.rows, Wb.cols, "linear_fused_dual_forward_bitnet_i2s_accum/Wb");
    if (Wb.rows != inDim || Wb.cols != outDim || Wa.qk != Wb.qk || Wa.jq != Wb.jq || Wa.hadamard32 != Wb.hadamard32) {
        throw std::runtime_error("linear_fused_dual_forward_bitnet_i2s_accum: incompatible matrix shapes");
    }
    const std::size_t hBlock = Wa.qk;
    const std::size_t nKBlocks = ceil_div(inDim, hBlock);
    const int outDimInt = static_cast<int>(outDim);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
            const std::size_t tCount = std::min(tTile, T - t0);
#if defined(_OPENMP)
#pragma omp single
#endif
            {
                quantize_x_tile_i8_tiled(x8Scratch, xScaleScratch, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
            }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int j0_i = 0; j0_i < outDimInt; j0_i += 32) {
                const std::size_t j0 = static_cast<std::size_t>(j0_i);
                const float* const scaleRowA = forward_scale_row(Wa, j0, nKBlocks);
                const float* const scaleRowB = forward_scale_row(Wb, j0, nKBlocks);

                for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                    const std::size_t hCount = std::min(hBlock, inDim - h0);
                    const std::size_t kb = h0 / hBlock;

                    std::size_t tt = 0;
                    for (; tt + 3 < tCount; tt += 4) {
                        const int8_t* x0 = x8Scratch + (tt + 0) * inDim + h0;
                        const int8_t* x1 = x8Scratch + (tt + 1) * inDim + h0;
                        const int8_t* x2 = x8Scratch + (tt + 2) * inDim + h0;
                        const int8_t* x3 = x8Scratch + (tt + 3) * inDim + h0;

                        float* yA0 = Ya + (t0 + tt + 0) * outDim;
                        float* yA1 = Ya + (t0 + tt + 1) * outDim;
                        float* yA2 = Ya + (t0 + tt + 2) * outDim;
                        float* yA3 = Ya + (t0 + tt + 3) * outDim;
                        float* yB0 = Yb + (t0 + tt + 0) * outDim;
                        float* yB1 = Yb + (t0 + tt + 1) * outDim;
                        float* yB2 = Yb + (t0 + tt + 2) * outDim;
                        float* yB3 = Yb + (t0 + tt + 3) * outDim;

                        int32_t accA0[32] = { 0 }, accA1[32] = { 0 }, accA2[32] = { 0 }, accA3[32] = { 0 };
                        int32_t accB0[32] = { 0 }, accB1[32] = { 0 }, accB2[32] = { 0 }, accB3[32] = { 0 };
                        const float xs0 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb);
                        const float xs1 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb);
                        const float xs2 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 2, kb);
                        const float xs3 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 3, kb);

                        dot_i2s_i8_accum_i32_block32_x4(accA0, accA1, accA2, accA3, Wa.q2.data(), outDim, h0, hCount, x0, x1, x2, x3, j0);
                        dot_i2s_i8_accum_i32_block32_x4(accB0, accB1, accB2, accB3, Wb.q2.data(), outDim, h0, hCount, x0, x1, x2, x3, j0);

                        accum_block32_scaled_to_y_dual(yA0, yB0, accA0, accB0, scaleRowA, scaleRowB, j0, kb, xs0, xs0);
                        accum_block32_scaled_to_y_dual(yA1, yB1, accA1, accB1, scaleRowA, scaleRowB, j0, kb, xs1, xs1);
                        accum_block32_scaled_to_y_dual(yA2, yB2, accA2, accB2, scaleRowA, scaleRowB, j0, kb, xs2, xs2);
                        accum_block32_scaled_to_y_dual(yA3, yB3, accA3, accB3, scaleRowA, scaleRowB, j0, kb, xs3, xs3);
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* x0 = x8Scratch + (tt + 0) * inDim + h0;
                        const int8_t* x1 = x8Scratch + (tt + 1) * inDim + h0;
                        float* yA0 = Ya + (t0 + tt + 0) * outDim;
                        float* yA1 = Ya + (t0 + tt + 1) * outDim;
                        float* yB0 = Yb + (t0 + tt + 0) * outDim;
                        float* yB1 = Yb + (t0 + tt + 1) * outDim;

                        int32_t accA0[32] = { 0 }, accA1[32] = { 0 };
                        int32_t accB0[32] = { 0 }, accB1[32] = { 0 };
                        const float xs0 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb);
                        const float xs1 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb);

                        dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, Wa.q2.data(), outDim, h0, hCount, x0, x1, j0);
                        dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, Wb.q2.data(), outDim, h0, hCount, x0, x1, j0);

                        accum_block32_scaled_to_y_dual(yA0, yB0, accA0, accB0, scaleRowA, scaleRowB, j0, kb, xs0, xs0);
                        accum_block32_scaled_to_y_dual(yA1, yB1, accA1, accB1, scaleRowA, scaleRowB, j0, kb, xs1, xs1);
                    }

                    for (; tt < tCount; ++tt) {
                        const int8_t* xRow = x8Scratch + tt * inDim + h0;
                        float* yA = Ya + (t0 + tt) * outDim;
                        float* yB = Yb + (t0 + tt) * outDim;
                        int32_t accA[32] = { 0 };
                        int32_t accB[32] = { 0 };
                        const float xs = load_tile_row_scale(xScaleScratch, nKBlocks, tt, kb);

                        dot_i2s_i8_accum_i32_block32(accA, Wa.q2.data(), outDim, h0, hCount, xRow, j0);
                        dot_i2s_i8_accum_i32_block32(accB, Wb.q2.data(), outDim, h0, hCount, xRow, j0);

                        accum_block32_scaled_to_y_dual(yA, yB, accA, accB, scaleRowA, scaleRowB, j0, kb, xs, xs);
                    }
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
    const std::size_t nJBlocks = ceil_div(outDim, hBlock);

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        const int inDim_int = static_cast<int>(inDim);

#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
#if defined(_OPENMP)
 #pragma omp single
            quantize_x_tile_i8_tiled(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
#else
            quantize_x_tile_i8_tiled(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
#endif

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int i0_i = 0; i0_i < inDim_int; i0_i += 32) {
            std::size_t i0 = static_cast<std::size_t>(i0_i);
            const float* const scaleRow = backward_scale_row(W, i0, nJBlocks);

            for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                const std::size_t jCount = std::min(hBlock, outDim - j0);
                const std::size_t jb = j0 / hBlock;

                std::size_t tt = 0;
                for (; tt + 3 < tCount; tt += 4) {
                    const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dyC = dy8Scratch + (tt + 2) * outDim + j0; const int8_t* dyD = dy8Scratch + (tt + 3) * outDim + j0;
                    float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                    float* dxC = dX + (t0 + tt + 2) * inDim; float* dxD = dX + (t0 + tt + 3) * inDim;

                    int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };
                    const float dyScaleA = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 0, jb);
                    const float dyScaleB = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 1, jb);
                    const float dyScaleC = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 2, jb);
                    const float dyScaleD = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 3, jb);

                    dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2T.data(), inDim, j0, jCount, dyA, dyB, dyC, dyD, i0);

                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aA, aB, scaleRow, i0, jb, dyScaleA, dyScaleB);
                    accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aC, aD, scaleRow, i0, jb, dyScaleC, dyScaleD);
                }

                for (; tt + 1 < tCount; tt += 2) {
                    const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                    float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                    int32_t aA[32] = { 0 }, aB[32] = { 0 };
                    const float dyScaleA = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 0, jb);
                    const float dyScaleB = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 1, jb);

                    dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2T.data(), inDim, j0, jCount, dyA, dyB, i0);
                    accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aA, aB, scaleRow, i0, jb, dyScaleA, dyScaleB);
                }

                for (; tt < tCount; ++tt) {
                    float* dxRow = dX + (t0 + tt) * inDim;
                    int32_t acc[32] = { 0 };
                    const float dyScale = load_tile_row_scale(dyScaleScratch, nJBlocks, tt, jb);
                    dot_i2s_i8_accum_i32_block32(acc, W.q2T.data(), inDim, j0, jCount, dy8Scratch + tt * outDim + j0, i0);
                    accum_bwd_block32_scaled_to_x(dxRow, acc, scaleRow, i0, jb, dyScale);
                }
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
    const std::size_t qDim = Wq.cols;
    const std::size_t kDim = Wk.cols;
    const std::size_t vDim = Wv.cols;
    validate_backward_packed(Wq, inDim, qDim, "linear_fused_qkv_backward_bitnet_i2s_packed/Wq");
    validate_backward_packed(Wk, inDim, kDim, "linear_fused_qkv_backward_bitnet_i2s_packed/Wk");
    validate_backward_packed(Wv, inDim, vDim, "linear_fused_qkv_backward_bitnet_i2s_packed/Wv");
    if (Wk.rows != inDim || Wv.rows != inDim || Wq.jq != Wk.jq || Wq.jq != Wv.jq || Wq.hadamard32 != Wk.hadamard32 || Wq.hadamard32 != Wv.hadamard32) {
        throw std::runtime_error("linear_fused_qkv_backward_bitnet_i2s_packed: incompatible Q/K/V shapes");
    }
    const std::size_t hBlock = Wq.jq;
    const std::size_t qJBlocks = ceil_div(qDim, hBlock);
    const std::size_t kJBlocks = ceil_div(kDim, hBlock);
    const std::size_t vJBlocks = ceil_div(vDim, hBlock);
    const std::size_t fusedCols = std::min(qDim, std::min(kDim, vDim));

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);

#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
#if defined(_OPENMP)
 #pragma omp single
            {
                quantize_x_tile_i8_tiled(dq8Scratch, dqScaleScratch, dQ, T, qDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);
                quantize_x_tile_i8_tiled(dk8Scratch, dkScaleScratch, dK, T, kDim, t0, tCount, kRowsPerQuantScale, hBlock, Wk.hadamard32);
                quantize_x_tile_i8_tiled(dv8Scratch, dvScaleScratch, dV, T, vDim, t0, tCount, kRowsPerQuantScale, hBlock, Wv.hadamard32);
            }
#else
            quantize_x_tile_i8_tiled(dq8Scratch, dqScaleScratch, dQ, T, qDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);
            quantize_x_tile_i8_tiled(dk8Scratch, dkScaleScratch, dK, T, kDim, t0, tCount, kRowsPerQuantScale, hBlock, Wk.hadamard32);
            quantize_x_tile_i8_tiled(dv8Scratch, dvScaleScratch, dV, T, vDim, t0, tCount, kRowsPerQuantScale, hBlock, Wv.hadamard32);
#endif

            auto run_single = [&](const float* __restrict dY, const BitNetMatI2S& W,
                                  std::size_t outDim, int8_t* __restrict dy8, float* __restrict dyScale,
                                  std::size_t jBegin) {
                (void)dY;
                const std::size_t dyJBlocks = ceil_div(outDim, hBlock);
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
                for (int i0_i = 0; i0_i < static_cast<int>(inDim); i0_i += 32) {
                    const std::size_t i0 = static_cast<std::size_t>(i0_i);
                    const float* const scaleRow = backward_scale_row(W, i0, dyJBlocks);
                    for (std::size_t j0 = jBegin; j0 < outDim; j0 += hBlock) {
                        const std::size_t jCount = std::min(hBlock, outDim - j0);
                        const std::size_t jb = j0 / hBlock;

                        std::size_t tt = 0;
                        for (; tt + 3 < tCount; tt += 4) {
                            const int8_t* dyA = dy8 + (tt + 0) * outDim + j0; const int8_t* dyB = dy8 + (tt + 1) * outDim + j0;
                            const int8_t* dyC = dy8 + (tt + 2) * outDim + j0; const int8_t* dyD = dy8 + (tt + 3) * outDim + j0;
                            float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                            float* dxC = dX + (t0 + tt + 2) * inDim; float* dxD = dX + (t0 + tt + 3) * inDim;
                            int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };
                            const float sA = load_tile_row_scale(dyScale, dyJBlocks, tt + 0, jb);
                            const float sB = load_tile_row_scale(dyScale, dyJBlocks, tt + 1, jb);
                            const float sC = load_tile_row_scale(dyScale, dyJBlocks, tt + 2, jb);
                            const float sD = load_tile_row_scale(dyScale, dyJBlocks, tt + 3, jb);

                            dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2T.data(), inDim, j0, jCount, dyA, dyB, dyC, dyD, i0);
                            accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aA, aB, scaleRow, i0, jb, sA, sB);
                            accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aC, aD, scaleRow, i0, jb, sC, sD);
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* dyA = dy8 + (tt + 0) * outDim + j0; const int8_t* dyB = dy8 + (tt + 1) * outDim + j0;
                            float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            const float sA = load_tile_row_scale(dyScale, dyJBlocks, tt + 0, jb);
                            const float sB = load_tile_row_scale(dyScale, dyJBlocks, tt + 1, jb);

                            dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2T.data(), inDim, j0, jCount, dyA, dyB, i0);
                            accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aA, aB, scaleRow, i0, jb, sA, sB);
                        }

                        for (; tt < tCount; ++tt) {
                            float* dxRow = dX + (t0 + tt) * inDim;
                            int32_t acc[32] = { 0 };
                            const float s = load_tile_row_scale(dyScale, dyJBlocks, tt, jb);
                            dot_i2s_i8_accum_i32_block32(acc, W.q2T.data(), inDim, j0, jCount, dy8 + tt * outDim + j0, i0);
                            accum_bwd_block32_scaled_to_x(dxRow, acc, scaleRow, i0, jb, s);
                        }
                    }
                }
            };

#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
            for (int i0_i = 0; i0_i < static_cast<int>(inDim); i0_i += 32) {
                const std::size_t i0 = static_cast<std::size_t>(i0_i);
                const float* const scaleRowQ = backward_scale_row(Wq, i0, qJBlocks);
                const float* const scaleRowK = backward_scale_row(Wk, i0, kJBlocks);
                const float* const scaleRowV = backward_scale_row(Wv, i0, vJBlocks);
                for (std::size_t j0 = 0; j0 < fusedCols; j0 += hBlock) {
                    const std::size_t jCount = std::min(hBlock, fusedCols - j0);
                    const std::size_t jbQ = j0 / hBlock;
                    const std::size_t jbK = j0 / hBlock;
                    const std::size_t jbV = j0 / hBlock;

                    std::size_t tt = 0;
                    for (; tt + 3 < tCount; tt += 4) {
                        const int8_t* dqA = dq8Scratch + (tt + 0) * qDim + j0; const int8_t* dqB = dq8Scratch + (tt + 1) * qDim + j0;
                        const int8_t* dqC = dq8Scratch + (tt + 2) * qDim + j0; const int8_t* dqD = dq8Scratch + (tt + 3) * qDim + j0;
                        const int8_t* dkA = dk8Scratch + (tt + 0) * kDim + j0; const int8_t* dkB = dk8Scratch + (tt + 1) * kDim + j0;
                        const int8_t* dkC = dk8Scratch + (tt + 2) * kDim + j0; const int8_t* dkD = dk8Scratch + (tt + 3) * kDim + j0;
                        const int8_t* dvA = dv8Scratch + (tt + 0) * vDim + j0; const int8_t* dvB = dv8Scratch + (tt + 1) * vDim + j0;
                        const int8_t* dvC = dv8Scratch + (tt + 2) * vDim + j0; const int8_t* dvD = dv8Scratch + (tt + 3) * vDim + j0;
                        float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                        float* dxC = dX + (t0 + tt + 2) * inDim; float* dxD = dX + (t0 + tt + 3) * inDim;

                        int32_t aQA[32] = { 0 }, aQB[32] = { 0 }, aQC[32] = { 0 }, aQD[32] = { 0 };
                        int32_t aKA[32] = { 0 }, aKB[32] = { 0 }, aKC[32] = { 0 }, aKD[32] = { 0 };
                        int32_t aVA[32] = { 0 }, aVB[32] = { 0 }, aVC[32] = { 0 }, aVD[32] = { 0 };
                        const float sqA = load_tile_row_scale(dqScaleScratch, qJBlocks, tt + 0, jbQ);
                        const float sqB = load_tile_row_scale(dqScaleScratch, qJBlocks, tt + 1, jbQ);
                        const float sqC = load_tile_row_scale(dqScaleScratch, qJBlocks, tt + 2, jbQ);
                        const float sqD = load_tile_row_scale(dqScaleScratch, qJBlocks, tt + 3, jbQ);
                        const float skA = load_tile_row_scale(dkScaleScratch, kJBlocks, tt + 0, jbK);
                        const float skB = load_tile_row_scale(dkScaleScratch, kJBlocks, tt + 1, jbK);
                        const float skC = load_tile_row_scale(dkScaleScratch, kJBlocks, tt + 2, jbK);
                        const float skD = load_tile_row_scale(dkScaleScratch, kJBlocks, tt + 3, jbK);
                        const float svA = load_tile_row_scale(dvScaleScratch, vJBlocks, tt + 0, jbV);
                        const float svB = load_tile_row_scale(dvScaleScratch, vJBlocks, tt + 1, jbV);
                        const float svC = load_tile_row_scale(dvScaleScratch, vJBlocks, tt + 2, jbV);
                        const float svD = load_tile_row_scale(dvScaleScratch, vJBlocks, tt + 3, jbV);

                        dot_i2s_i8_accum_i32_block32_x4(aQA, aQB, aQC, aQD, Wq.q2T.data(), inDim, j0, jCount, dqA, dqB, dqC, dqD, i0);
                        dot_i2s_i8_accum_i32_block32_x4(aKA, aKB, aKC, aKD, Wk.q2T.data(), inDim, j0, jCount, dkA, dkB, dkC, dkD, i0);
                        dot_i2s_i8_accum_i32_block32_x4(aVA, aVB, aVC, aVD, Wv.q2T.data(), inDim, j0, jCount, dvA, dvB, dvC, dvD, i0);

                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aQA, aQB, scaleRowQ, i0, jbQ, sqA, sqB);
                        accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aQC, aQD, scaleRowQ, i0, jbQ, sqC, sqD);
                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aKA, aKB, scaleRowK, i0, jbK, skA, skB);
                        accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aKC, aKD, scaleRowK, i0, jbK, skC, skD);
                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aVA, aVB, scaleRowV, i0, jbV, svA, svB);
                        accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aVC, aVD, scaleRowV, i0, jbV, svC, svD);
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* dqA = dq8Scratch + (tt + 0) * qDim + j0; const int8_t* dqB = dq8Scratch + (tt + 1) * qDim + j0;
                        const int8_t* dkA = dk8Scratch + (tt + 0) * kDim + j0; const int8_t* dkB = dk8Scratch + (tt + 1) * kDim + j0;
                        const int8_t* dvA = dv8Scratch + (tt + 0) * vDim + j0; const int8_t* dvB = dv8Scratch + (tt + 1) * vDim + j0;
                        float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                        int32_t aQA[32] = { 0 }, aQB[32] = { 0 }, aKA[32] = { 0 }, aKB[32] = { 0 }, aVA[32] = { 0 }, aVB[32] = { 0 };
                        const float sqA = load_tile_row_scale(dqScaleScratch, qJBlocks, tt + 0, jbQ);
                        const float sqB = load_tile_row_scale(dqScaleScratch, qJBlocks, tt + 1, jbQ);
                        const float skA = load_tile_row_scale(dkScaleScratch, kJBlocks, tt + 0, jbK);
                        const float skB = load_tile_row_scale(dkScaleScratch, kJBlocks, tt + 1, jbK);
                        const float svA = load_tile_row_scale(dvScaleScratch, vJBlocks, tt + 0, jbV);
                        const float svB = load_tile_row_scale(dvScaleScratch, vJBlocks, tt + 1, jbV);

                        dot_i2s_i8_accum_i32_block32_x2(aQA, aQB, Wq.q2T.data(), inDim, j0, jCount, dqA, dqB, i0);
                        dot_i2s_i8_accum_i32_block32_x2(aKA, aKB, Wk.q2T.data(), inDim, j0, jCount, dkA, dkB, i0);
                        dot_i2s_i8_accum_i32_block32_x2(aVA, aVB, Wv.q2T.data(), inDim, j0, jCount, dvA, dvB, i0);

                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aQA, aQB, scaleRowQ, i0, jbQ, sqA, sqB);
                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aKA, aKB, scaleRowK, i0, jbK, skA, skB);
                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aVA, aVB, scaleRowV, i0, jbV, svA, svB);
                    }

                    for (; tt < tCount; ++tt) {
                        float* dxRow = dX + (t0 + tt) * inDim;
                        int32_t aQ[32] = { 0 }, aK[32] = { 0 }, aV[32] = { 0 };
                        const float sq = load_tile_row_scale(dqScaleScratch, qJBlocks, tt, jbQ);
                        const float sk = load_tile_row_scale(dkScaleScratch, kJBlocks, tt, jbK);
                        const float sv = load_tile_row_scale(dvScaleScratch, vJBlocks, tt, jbV);

                        dot_i2s_i8_accum_i32_block32(aQ, Wq.q2T.data(), inDim, j0, jCount, dq8Scratch + tt * qDim + j0, i0);
                        dot_i2s_i8_accum_i32_block32(aK, Wk.q2T.data(), inDim, j0, jCount, dk8Scratch + tt * kDim + j0, i0);
                        dot_i2s_i8_accum_i32_block32(aV, Wv.q2T.data(), inDim, j0, jCount, dv8Scratch + tt * vDim + j0, i0);

                        accum_bwd_block32_scaled_to_x(dxRow, aQ, scaleRowQ, i0, jbQ, sq);
                        accum_bwd_block32_scaled_to_x(dxRow, aK, scaleRowK, i0, jbK, sk);
                        accum_bwd_block32_scaled_to_x(dxRow, aV, scaleRowV, i0, jbV, sv);
                    }
                }
            }

            if (qDim > fusedCols) run_single(dQ, Wq, qDim, dq8Scratch, dqScaleScratch, fusedCols);
            if (kDim > fusedCols) run_single(dK, Wk, kDim, dk8Scratch, dkScaleScratch, fusedCols);
            if (vDim > fusedCols) run_single(dV, Wv, vDim, dv8Scratch, dvScaleScratch, fusedCols);
        }
    }
}

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
    float* __restrict dyBScaleScratch)
{
    const std::size_t inDim = Wa.rows;
    const std::size_t outDim = Wa.cols;
    validate_backward_packed(Wa, inDim, outDim, "linear_fused_dual_backward_bitnet_i2s_packed/Wa");
    validate_backward_packed(Wb, Wb.rows, Wb.cols, "linear_fused_dual_backward_bitnet_i2s_packed/Wb");
    if (Wb.rows != inDim || Wb.cols != outDim || Wa.qk != Wb.qk || Wa.jq != Wb.jq || Wa.hadamard32 != Wb.hadamard32) {
        throw std::runtime_error("linear_fused_dual_backward_bitnet_i2s_packed: incompatible matrix shapes");
    }
    const std::size_t hBlock = Wa.jq;
    const std::size_t nJBlocks = ceil_div(outDim, hBlock);
    
    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        quantize_x_tile_i8_tiled(dyA8Scratch, dyAScaleScratch, dYa, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
        quantize_x_tile_i8_tiled(dyB8Scratch, dyBScaleScratch, dYb, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, Wb.hadamard32);

        const int inDimInt = static_cast<int>(inDim);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i0_i = 0; i0_i < inDimInt; i0_i += 32) {
            const std::size_t i0 = static_cast<std::size_t>(i0_i);
            const float* const scaleRowA = backward_scale_row(Wa, i0, nJBlocks);
            const float* const scaleRowB = backward_scale_row(Wb, i0, nJBlocks);

            for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                const std::size_t jCount = std::min(hBlock, outDim - j0);
                const std::size_t jb = j0 / hBlock;

                std::size_t tt = 0;
                for (; tt + 3 < tCount; tt += 4) {
                    const int8_t* dyA0 = dyA8Scratch + (tt + 0) * outDim + j0;
                    const int8_t* dyA1 = dyA8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dyA2 = dyA8Scratch + (tt + 2) * outDim + j0;
                    const int8_t* dyA3 = dyA8Scratch + (tt + 3) * outDim + j0;
                    const int8_t* dyB0 = dyB8Scratch + (tt + 0) * outDim + j0;
                    const int8_t* dyB1 = dyB8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dyB2 = dyB8Scratch + (tt + 2) * outDim + j0;
                    const int8_t* dyB3 = dyB8Scratch + (tt + 3) * outDim + j0;

                    float* dx0 = dX + (t0 + tt + 0) * inDim;
                    float* dx1 = dX + (t0 + tt + 1) * inDim;
                    float* dx2 = dX + (t0 + tt + 2) * inDim;
                    float* dx3 = dX + (t0 + tt + 3) * inDim;

                    int32_t accA0[32] = { 0 }, accA1[32] = { 0 }, accA2[32] = { 0 }, accA3[32] = { 0 };
                    int32_t accB0[32] = { 0 }, accB1[32] = { 0 }, accB2[32] = { 0 }, accB3[32] = { 0 };
                    const float scaleA0 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 0, jb);
                    const float scaleA1 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 1, jb);
                    const float scaleA2 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 2, jb);
                    const float scaleA3 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 3, jb);
                    const float scaleB0 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 0, jb);
                    const float scaleB1 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 1, jb);
                    const float scaleB2 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 2, jb);
                    const float scaleB3 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 3, jb);

                    dot_i2s_i8_accum_i32_block32_x4(accA0, accA1, accA2, accA3, Wa.q2T.data(), inDim, j0, jCount, dyA0, dyA1, dyA2, dyA3, i0);
                    dot_i2s_i8_accum_i32_block32_x4(accB0, accB1, accB2, accB3, Wb.q2T.data(), inDim, j0, jCount, dyB0, dyB1, dyB2, dyB3, i0);

                    accum_bwd_block32_scaled_to_x_dual(dx0, accA0, accB0, scaleRowA, scaleRowB, i0, jb, scaleA0, scaleB0);
                    accum_bwd_block32_scaled_to_x_dual(dx1, accA1, accB1, scaleRowA, scaleRowB, i0, jb, scaleA1, scaleB1);
                    accum_bwd_block32_scaled_to_x_dual(dx2, accA2, accB2, scaleRowA, scaleRowB, i0, jb, scaleA2, scaleB2);
                    accum_bwd_block32_scaled_to_x_dual(dx3, accA3, accB3, scaleRowA, scaleRowB, i0, jb, scaleA3, scaleB3);
                }

                for (; tt + 1 < tCount; tt += 2) {
                    const int8_t* dyA0 = dyA8Scratch + (tt + 0) * outDim + j0;
                    const int8_t* dyA1 = dyA8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dyB0 = dyB8Scratch + (tt + 0) * outDim + j0;
                    const int8_t* dyB1 = dyB8Scratch + (tt + 1) * outDim + j0;

                    float* dx0 = dX + (t0 + tt + 0) * inDim;
                    float* dx1 = dX + (t0 + tt + 1) * inDim;

                    int32_t accA0[32] = { 0 }, accA1[32] = { 0 };
                    int32_t accB0[32] = { 0 }, accB1[32] = { 0 };
                    const float scaleA0 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 0, jb);
                    const float scaleA1 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 1, jb);
                    const float scaleB0 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 0, jb);
                    const float scaleB1 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 1, jb);

                    dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, Wa.q2T.data(), inDim, j0, jCount, dyA0, dyA1, i0);
                    dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, Wb.q2T.data(), inDim, j0, jCount, dyB0, dyB1, i0);

                    accum_bwd_block32_scaled_to_x_dual(dx0, accA0, accB0, scaleRowA, scaleRowB, i0, jb, scaleA0, scaleB0);
                    accum_bwd_block32_scaled_to_x_dual(dx1, accA1, accB1, scaleRowA, scaleRowB, i0, jb, scaleA1, scaleB1);
                }

                for (; tt < tCount; ++tt) {
                    float* dx = dX + (t0 + tt) * inDim;
                    int32_t accA[32] = { 0 };
                    int32_t accB[32] = { 0 };
                    const float scaleA = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt, jb);
                    const float scaleB = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt, jb);

                    dot_i2s_i8_accum_i32_block32(accA, Wa.q2T.data(), inDim, j0, jCount, dyA8Scratch + tt * outDim + j0, i0);
                    dot_i2s_i8_accum_i32_block32(accB, Wb.q2T.data(), inDim, j0, jCount, dyB8Scratch + tt * outDim + j0, i0);

                    accum_bwd_block32_scaled_to_x_dual(dx, accA, accB, scaleRowA, scaleRowB, i0, jb, scaleA, scaleB);
                }
            }
        }
    }
}
