#include "bitnet_i2s_matmul.h"
#include "bitnet_i2_s.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
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

    static inline std::optional<std::string> try_get_env_string(const char* name) {
#if defined(_WIN32)
        char* value = nullptr;
        size_t len = 0;
        if (_dupenv_s(&value, &len, name) != 0 || !value || len == 0 || value[0] == '\0') {
            if (value) {
                std::free(value);
            }
            return std::nullopt;
        }
        std::string out(value);
        std::free(value);
        return out;
#else
        const char* value = std::getenv(name);
        if (!value || *value == '\0') {
            return std::nullopt;
        }
        return std::string(value);
#endif
    }

    static inline int env_toggle_override(const char* name) {
        const auto value = try_get_env_string(name);
        if (!value) return -1;
        std::string s(*value);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (s == "1" || s == "true" || s == "yes" || s == "on") return 1;
        if (s == "0" || s == "false" || s == "no" || s == "off") return 0;
        return -1;
    }

    static inline std::size_t env_size_override(const char* name, std::size_t fallback) {
        const auto value = try_get_env_string(name);
        if (!value) return fallback;

        char* end = nullptr;
        const char* raw = value->c_str();
        const unsigned long parsed = std::strtoul(raw, &end, 10);
        if (end == raw || (end && *end != '\0') || parsed == 0) return fallback;
        return static_cast<std::size_t>(parsed);
    }

    static inline bool fused_x4_forward_preferred(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_FORWARD");
        if (override >= 0) return override != 0;

        // On the long-sequence tTile=4 branch, the exact fused x4 forward
        // family is materially faster than the older accumulator path.
        // Keep the auto-promotion narrow so shorter-sequence regimes preserve
        // the old defaults unless explicitly opted in.
        return tTile >= 4 && T >= 128;
    }

    static inline bool fused_x2_single_forward_enabled() {
        static const bool cached = env_toggle_override("QLORA_TEST_BITNET_FUSED_X2_FORWARD_SINGLE") > 0;
        return cached;
    }

    static inline bool dense_x2_forward_accum_preferred(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_DENSE_X2_FORWARD_ACCUM");
        if (override >= 0) return override != 0;

        // Keep this confined to the long-sequence tTile=2 branch, where the
        // pair-accumulator path is still the default exact forward route.
        return tTile <= 2 && T >= 128;
    }

    static inline bool fused_x4_backward_preferred(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_BACKWARD");
        if (override >= 0) return override != 0;

        // The dual backward x4 path pays off on the long-sequence tTile=4
        // regime, but was not consistently helpful on shorter tiles.
        return tTile >= 4 && T >= 128;
    }

    static inline bool fused_x4_single_backward_preferred(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_BACKWARD_SINGLE");
        if (override >= 0) return override != 0;

        // The single backward x4 path only paid off consistently on the same
        // long-sequence tTile=4 branch as the dual x4 path.
        return tTile >= 4 && T >= 128;
    }

    static inline bool fused_x4_single_backward_overwrite_enabled() {
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_BACKWARD_SINGLE_OVERWRITE");
        if (override >= 0) return override != 0;
        return true;
    }

    static inline bool fused_x4_single_backward_block32_exact_enabled(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_BACKWARD_SINGLE_BLOCK32");
        if (override >= 0) return override != 0;
        return tTile >= 4 && T >= 128;
    }

    static inline bool fused_x2_single_backward_preferred(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X2_BACKWARD_SINGLE");
        if (override >= 0) return override != 0;

        // The tTile=2 long-sequence branch is materially faster than the
        // accumulator fallback, but it needs the exact x2 backward family to
        // preserve the good loss curve. Keep the auto-promotion narrow.
        return tTile <= 2 && T >= 128;
    }

    static inline bool fused_x2_dual_backward_preferred(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X2_BACKWARD_DUAL");
        if (override >= 0) return override != 0;

        return tTile <= 2 && T >= 128;
    }

    static inline bool parallel_single_backward_quant_enabled(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_PARALLEL_BWD_QUANT");
        if (override >= 0) return override != 0;

        return tTile >= 4 && T >= 128;
    }

    static inline bool single_team_backward_preferred(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_SINGLE_TEAM_BWD_SINGLE");
        if (override >= 0) return override != 0;

        // The long-sequence exact tTile<=2 branch spends a lot of time re-entering
        // OpenMP teams in the backward tile loop. Auto-promote the shared
        // single-team path there, but keep the old default elsewhere.
        return tTile <= 2 && T >= 128;
    }

    static inline bool parallel_multi_backward_quant_enabled() {
        static const bool cached = env_toggle_override("QLORA_TEST_BITNET_PARALLEL_BWD_QUANT_MULTI") > 0;
        return cached;
    }

    static inline bool master_dual_chunk_quant_preferred() {
        const int override = env_toggle_override("QLORA_TEST_BITNET_DUAL_BWD_MASTER_QUANT");
        if (override >= 0) return override != 0;
        return true;
    }

    static inline bool parallel_multi_forward_quant_enabled() {
        static const bool cached = env_toggle_override("QLORA_TEST_BITNET_PARALLEL_FWD_QUANT_MULTI") != 0;
        return cached;
    }

    static inline bool parallel_single_forward_quant_enabled() {
        static const bool cached = env_toggle_override("QLORA_TEST_BITNET_PARALLEL_FWD_QUANT_SINGLE") > 0;
        return cached;
    }

    static inline bool fused_qkv_forward_overwrite_enabled(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_QKV_FORWARD_OVERWRITE");
        if (override >= 0) return override != 0;

        // Keep this auto-promotion narrow to the long-sequence tutor branch,
        // where fused exact QKV forward runs with large enough tiles that the
        // first k-block always overwrites fresh rows.
        return tTile >= 4 && T >= 128;
    }

    static inline bool fused_qkv_forward_x4_triple_accum_enabled(std::size_t T, std::size_t tTile) {
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_QKV_FORWARD_X4_TRIPLE_ACCUM");
        if (override >= 0) return override != 0;

        // Restrict this to the same long-sequence fused exact branch that
        // already uses the x4 path, so same-binary A/B validation stays clean.
        return tTile >= 4 && T >= 128;
    }

    static inline bool tile_all_masked(
        const int* attentionMask,
        std::size_t activePrefixT,
        std::size_t t0,
        std::size_t tCount)
    {
        if (activePrefixT != static_cast<std::size_t>(-1)) {
            return t0 >= activePrefixT;
        }
        if (!attentionMask) return false;
        for (std::size_t tt = 0; tt < tCount; ++tt) {
            if (attentionMask[t0 + tt] != 0) return false;
        }
        return true;
    }

    static inline void accum_block32_scaled_to_y(
        float* __restrict yRow, const int32_t* __restrict blockAcc,
        const float* __restrict scaleRow, std::size_t j0, std::size_t kb, float xs,
        bool addToDest)
    {
        const __m256 gain8 = _mm256_set1_ps(xs * scaleRow[kb]);
        for (int off = 0; off < 32; off += 8) {
            const __m256 accF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAcc + off)));
            __m256 y = _mm256_mul_ps(accF, gain8);
            if (addToDest) {
                y = _mm256_add_ps(y, _mm256_loadu_ps(yRow + j0 + off));
            }
            _mm256_storeu_ps(yRow + j0 + off, y);
        }
    }

    static inline void accum_block32_scaled_to_y_pair(
        float* __restrict yRowA, float* __restrict yRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        const float* __restrict scaleRow, std::size_t j0, std::size_t kb, float xsA, float xsB,
        bool addToDest)
    {
        const float wsScalar = scaleRow[kb];
        const __m256 gainA8 = _mm256_set1_ps(xsA * wsScalar);
        const __m256 gainB8 = _mm256_set1_ps(xsB * wsScalar);
        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 yA = _mm256_mul_ps(accFA, gainA8);
            __m256 yB = _mm256_mul_ps(accFB, gainB8);
            if (addToDest) {
                yA = _mm256_add_ps(yA, _mm256_loadu_ps(yRowA + j0 + off));
                yB = _mm256_add_ps(yB, _mm256_loadu_ps(yRowB + j0 + off));
            }
            _mm256_storeu_ps(yRowA + j0 + off, yA);
            _mm256_storeu_ps(yRowB + j0 + off, yB);
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
        std::size_t j0, std::size_t kb, float xs, bool addToDest)
    {
        const __m256 gainQ = _mm256_set1_ps(xs * scaleRowQ[kb]);
        const __m256 gainK = _mm256_set1_ps(xs * scaleRowK[kb]);
        const __m256 gainV = _mm256_set1_ps(xs * scaleRowV[kb]);

        for (int off = 0; off < 32; off += 8) {
            const __m256 qF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQ + off)));
            const __m256 kF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accK + off)));
            const __m256 vF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accV + off)));

            __m256 q = _mm256_mul_ps(qF, gainQ);
            __m256 k = _mm256_mul_ps(kF, gainK);
            __m256 v = _mm256_mul_ps(vF, gainV);
            if (addToDest) {
                q = _mm256_add_ps(q, _mm256_loadu_ps(qRow + j0 + off));
                k = _mm256_add_ps(k, _mm256_loadu_ps(kRow + j0 + off));
                v = _mm256_add_ps(v, _mm256_loadu_ps(vRow + j0 + off));
            }

            _mm256_storeu_ps(qRow + j0 + off, q);
            _mm256_storeu_ps(kRow + j0 + off, k);
            _mm256_storeu_ps(vRow + j0 + off, v);
        }
    }

    static inline void accum_block32_scaled_to_y3_pair(
        float* __restrict qRowA, float* __restrict kRowA, float* __restrict vRowA,
        float* __restrict qRowB, float* __restrict kRowB, float* __restrict vRowB,
        const int32_t* __restrict accQA, const int32_t* __restrict accKA, const int32_t* __restrict accVA,
        const int32_t* __restrict accQB, const int32_t* __restrict accKB, const int32_t* __restrict accVB,
        const float* __restrict scaleRowQ, const float* __restrict scaleRowK, const float* __restrict scaleRowV,
        std::size_t j0, std::size_t kb, float xsA, float xsB, bool addToDest)
    {
        const float wsQ = scaleRowQ[kb];
        const float wsK = scaleRowK[kb];
        const float wsV = scaleRowV[kb];

        const __m256 gainQA = _mm256_set1_ps(xsA * wsQ), gainQB = _mm256_set1_ps(xsB * wsQ);
        const __m256 gainKA = _mm256_set1_ps(xsA * wsK), gainKB = _mm256_set1_ps(xsB * wsK);
        const __m256 gainVA = _mm256_set1_ps(xsA * wsV), gainVB = _mm256_set1_ps(xsB * wsV);

        for (int off = 0; off < 32; off += 8) {
            __m256 qA = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQA + off))), gainQA);
            __m256 qB = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQB + off))), gainQB);
            __m256 kA = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKA + off))), gainKA);
            __m256 kB = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKB + off))), gainKB);
            __m256 vA = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVA + off))), gainVA);
            __m256 vB = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVB + off))), gainVB);
            if (addToDest) {
                qA = _mm256_add_ps(qA, _mm256_loadu_ps(qRowA + j0 + off));
                qB = _mm256_add_ps(qB, _mm256_loadu_ps(qRowB + j0 + off));
                kA = _mm256_add_ps(kA, _mm256_loadu_ps(kRowA + j0 + off));
                kB = _mm256_add_ps(kB, _mm256_loadu_ps(kRowB + j0 + off));
                vA = _mm256_add_ps(vA, _mm256_loadu_ps(vRowA + j0 + off));
                vB = _mm256_add_ps(vB, _mm256_loadu_ps(vRowB + j0 + off));
            }

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

    static inline void accum_bwd_block32_gain_to_x(
        float* __restrict dxRow, const int32_t* __restrict blockAcc,
        std::size_t i0, float gain,
        bool addToDest)
    {
        const __m256 gain8 = _mm256_set1_ps(gain);
        for (int off = 0; off < 32; off += 8) {
            const __m256 accF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAcc + off)));
            __m256 dx = _mm256_mul_ps(accF, gain8);
            if (addToDest) dx = _mm256_add_ps(dx, _mm256_loadu_ps(dxRow + i0 + off));
            _mm256_storeu_ps(dxRow + i0 + off, dx);
        }
    }

    static inline void accum_bwd_block32_gain_to_x_pair(
        float* __restrict dxRowA, float* __restrict dxRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        std::size_t i0, float gainA, float gainB,
        bool addToDest)
    {
        const __m256 gainA8 = _mm256_set1_ps(gainA);
        const __m256 gainB8 = _mm256_set1_ps(gainB);

        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 dxA = _mm256_mul_ps(accFA, gainA8);
            __m256 dxB = _mm256_mul_ps(accFB, gainB8);
            if (addToDest) {
                dxA = _mm256_add_ps(dxA, _mm256_loadu_ps(dxRowA + i0 + off));
                dxB = _mm256_add_ps(dxB, _mm256_loadu_ps(dxRowB + i0 + off));
            }
            _mm256_storeu_ps(dxRowA + i0 + off, dxA);
            _mm256_storeu_ps(dxRowB + i0 + off, dxB);
        }
    }

    static inline void accum_bwd_block32_gain_to_x_dual(
        float* __restrict dxRow,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        std::size_t i0, float gainA, float gainB,
        bool addToDest)
    {
        const __m256 gainA8 = _mm256_set1_ps(gainA);
        const __m256 gainB8 = _mm256_set1_ps(gainB);
        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 dx = _mm256_mul_ps(accFA, gainA8);
            dx = _mm256_fmadd_ps(accFB, gainB8, dx);
            if (addToDest) dx = _mm256_add_ps(dx, _mm256_loadu_ps(dxRow + i0 + off));
            _mm256_storeu_ps(dxRow + i0 + off, dx);
        }
    }

    static inline void accum_bwd_block32_gain_to_x_dual_x4(
        float* __restrict dx0, float* __restrict dx1,
        float* __restrict dx2, float* __restrict dx3,
        const int32_t* __restrict blockAccA0, const int32_t* __restrict blockAccA1,
        const int32_t* __restrict blockAccA2, const int32_t* __restrict blockAccA3,
        const int32_t* __restrict blockAccB0, const int32_t* __restrict blockAccB1,
        const int32_t* __restrict blockAccB2, const int32_t* __restrict blockAccB3,
        std::size_t i0,
        float gainA0, float gainA1, float gainA2, float gainA3,
        float gainB0, float gainB1, float gainB2, float gainB3,
        bool addToDest)
    {
        const __m256 gainA0v = _mm256_set1_ps(gainA0);
        const __m256 gainA1v = _mm256_set1_ps(gainA1);
        const __m256 gainA2v = _mm256_set1_ps(gainA2);
        const __m256 gainA3v = _mm256_set1_ps(gainA3);
        const __m256 gainB0v = _mm256_set1_ps(gainB0);
        const __m256 gainB1v = _mm256_set1_ps(gainB1);
        const __m256 gainB2v = _mm256_set1_ps(gainB2);
        const __m256 gainB3v = _mm256_set1_ps(gainB3);

        if (addToDest) {
            for (int off = 0; off < 32; off += 8) {
                __m256 dxv0 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA0 + off))),
                    gainA0v);
                dxv0 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB0 + off))),
                    gainB0v,
                    dxv0);
                dxv0 = _mm256_add_ps(dxv0, _mm256_loadu_ps(dx0 + i0 + off));
                _mm256_storeu_ps(dx0 + i0 + off, dxv0);

                __m256 dxv1 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA1 + off))),
                    gainA1v);
                dxv1 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB1 + off))),
                    gainB1v,
                    dxv1);
                dxv1 = _mm256_add_ps(dxv1, _mm256_loadu_ps(dx1 + i0 + off));
                _mm256_storeu_ps(dx1 + i0 + off, dxv1);

                __m256 dxv2 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA2 + off))),
                    gainA2v);
                dxv2 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB2 + off))),
                    gainB2v,
                    dxv2);
                dxv2 = _mm256_add_ps(dxv2, _mm256_loadu_ps(dx2 + i0 + off));
                _mm256_storeu_ps(dx2 + i0 + off, dxv2);

                __m256 dxv3 = _mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA3 + off))),
                    gainA3v);
                dxv3 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB3 + off))),
                    gainB3v,
                    dxv3);
                dxv3 = _mm256_add_ps(dxv3, _mm256_loadu_ps(dx3 + i0 + off));
                _mm256_storeu_ps(dx3 + i0 + off, dxv3);
            }
            return;
        }

        for (int off = 0; off < 32; off += 8) {
            __m256 dxv0 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA0 + off))),
                gainA0v);
            dxv0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB0 + off))),
                gainB0v,
                dxv0);
            _mm256_storeu_ps(dx0 + i0 + off, dxv0);

            __m256 dxv1 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA1 + off))),
                gainA1v);
            dxv1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB1 + off))),
                gainB1v,
                dxv1);
            _mm256_storeu_ps(dx1 + i0 + off, dxv1);

            __m256 dxv2 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA2 + off))),
                gainA2v);
            dxv2 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB2 + off))),
                gainB2v,
                dxv2);
            _mm256_storeu_ps(dx2 + i0 + off, dxv2);

            __m256 dxv3 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA3 + off))),
                gainA3v);
            dxv3 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB3 + off))),
                gainB3v,
                dxv3);
            _mm256_storeu_ps(dx3 + i0 + off, dxv3);
        }
    }

    static inline void accum_bwd_block32_scaled_to_x(
        float* __restrict dxRow, const int32_t* __restrict blockAcc,
        const float* __restrict scaleRow, std::size_t i0, std::size_t jb, float dy_scale,
        bool addToDest)
    {
        accum_bwd_block32_gain_to_x(dxRow, blockAcc, i0, dy_scale * scaleRow[jb], addToDest);
    }

    static inline void accum_bwd_block32_scaled_to_x_pair(
        float* __restrict dxRowA, float* __restrict dxRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        const float* __restrict scaleRow, std::size_t i0, std::size_t jb, float dy_scaleA, float dy_scaleB,
        bool addToDest)
    {
        const float ws = scaleRow[jb];
        accum_bwd_block32_gain_to_x_pair(dxRowA, dxRowB, blockAccA, blockAccB, i0, dy_scaleA * ws, dy_scaleB * ws, addToDest);
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
    int8_t* __restrict x8Scratch, float* __restrict xScaleScratch,
    const int* attentionMask, std::size_t activePrefixT)
{
    const std::size_t inDim = W.rows;
    const std::size_t outDim = W.cols;
    validate_forward_common(W, inDim, outDim, "linear_forward_bitnet_i2s_accum");
    const std::size_t hBlock = W.qk;
    const std::size_t nKBlocks = ceil_div(inDim, hBlock);
    int8_t* const x8 = x8Scratch; float* const xScale = xScaleScratch;
    const int outDim_int = static_cast<int>(outDim);
    const bool useFusedX4 = fused_x4_forward_preferred(T, tTile);
    const bool useFusedX2 = fused_x2_single_forward_enabled();
    const bool useDenseX2Accum = dense_x2_forward_accum_preferred(T, tTile);
    const bool useParallelQuant = parallel_single_forward_quant_enabled();

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
            const std::size_t tCount = std::min(tTile, T - t0);
            if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;
#if defined(_OPENMP)
            if (useParallelQuant) {
                quantize_x_tile_i8_tiled_omp_for(x8, xScale, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
            }
            else {
 #pragma omp single
                quantize_x_tile_i8_tiled(x8, xScale, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
            }
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
                            const bool addToDest = (kb != 0);
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
                                addToDest,
                                gainA, gainB, gainC, gainD);
                            dot_i2s_i8_scale_store_block16_x4_exact(
                                yA, yB, yC, yD,
                                W.q2.data(), outDim, h0, hCount,
                                xA, xB, xC, xD,
                                j0 + 16,
                                addToDest,
                                gainA, gainB, gainC, gainD);
                        }
                        else {
                            int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };
                            const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                            const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);
                            const float xsC = load_tile_row_scale(xScale, nKBlocks, tt + 2, kb);
                            const float xsD = load_tile_row_scale(xScale, nKBlocks, tt + 3, kb);

                            dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                            accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB, true);
                            accum_block32_scaled_to_y_pair(yC, yD, aC, aD, scaleRow, j0, kb, xsC, xsD, true);
                        }
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                        float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                        const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                        const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);

                        if (useFusedX2) {
                            const float ws = scaleRow[kb];
                            const float gainA = xsA * ws;
                            const float gainB = xsB * ws;
                            dot_i2s_i8_scale_store_block16_x2_exact(
                                yA, yB,
                                W.q2.data(), outDim, h0, hCount,
                                xA, xB,
                                j0,
                                gainA, gainB);
                            dot_i2s_i8_scale_store_block16_x2_exact(
                                yA, yB,
                                W.q2.data(), outDim, h0, hCount,
                                xA, xB,
                                j0 + 16,
                                gainA, gainB);
                        }
                        else {
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            if (useDenseX2Accum) {
                                dot_i2s_i8_accum_i32_block32_x2_dense(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            } else {
                                dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            }
                            accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB, true);
                        }
                    }

                    for (; tt < tCount; ++tt) {
                        float* yRow = Y + (t0 + tt) * outDim;
                        int32_t blockAcc[32] = { 0 };
                        const float xs = load_tile_row_scale(xScale, nKBlocks, tt, kb);
                        dot_i2s_i8_accum_i32_block32(blockAcc, W.q2.data(), outDim, h0, hCount, x8 + tt * inDim + h0, j0);
                        accum_block32_scaled_to_y(yRow, blockAcc, scaleRow, j0, kb, xs, true);
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
    const bool useFusedX4 = fused_x4_forward_preferred(T, tTile);
    const bool useDenseX2Accum = dense_x2_forward_accum_preferred(T, tTile);
    const bool useParallelQuant = parallel_multi_forward_quant_enabled();
    const bool useFirstBlockOverwrite = fused_qkv_forward_overwrite_enabled(T, tTile);
    const bool useTripleX4Accum = fused_qkv_forward_x4_triple_accum_enabled(T, tTile);

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
#if defined(_OPENMP)
        if (useParallelQuant) {
            quantize_x_tile_i8_tiled_omp_for(x8, xScale, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);
        }
        else {
            quantize_x_tile_i8_tiled(x8, xScale, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);
        }
#else
        quantize_x_tile_i8_tiled(x8, xScale, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);
#endif

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
                                const bool addToDest = !useFirstBlockOverwrite || (kb != 0);
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
                                    addToDest,
                                    gainA, gainB, gainC, gainD);
                                dot_i2s_i8_scale_store_block16_x4_exact(
                                    yA, yB, yC, yD,
                                    W.q2.data(), outDim, h0, hCount,
                                    xA, xB, xC, xD,
                                    j0 + 16,
                                    addToDest,
                                    gainA, gainB, gainC, gainD);
                            }
                            else {
                                int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };
                                const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                                const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);
                                const float xsC = load_tile_row_scale(xScale, nKBlocks, tt + 2, kb);
                                const float xsD = load_tile_row_scale(xScale, nKBlocks, tt + 3, kb);

                                dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                                accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB, true);
                                accum_block32_scaled_to_y_pair(yC, yD, aC, aD, scaleRow, j0, kb, xsC, xsD, true);
                            }
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                            float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                            const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);
                            const bool addToDest = !useFirstBlockOverwrite || (kb != 0);

                            if (useDenseX2Accum) {
                                dot_i2s_i8_accum_i32_block32_x2_dense(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            } else {
                                dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            }
                            accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB, addToDest);
                        }

                        for (; tt < tCount; ++tt) {
                            float* yRow = Y + (t0 + tt) * outDim;
                            int32_t acc[32] = { 0 };
                            const float xs = load_tile_row_scale(xScale, nKBlocks, tt, kb);
                            const bool addToDest = !useFirstBlockOverwrite || (kb != 0);
                            dot_i2s_i8_accum_i32_block32(acc, W.q2.data(), outDim, h0, hCount, x8 + tt * inDim + h0, j0);
                            accum_block32_scaled_to_y(yRow, acc, scaleRow, j0, kb, xs, addToDest);
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
                    const bool addToDest = !useFirstBlockOverwrite || (kb != 0);

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

                        if (useTripleX4Accum) {
                            dot_i2s_i8_accum_i32_block32_x4_triple(
                                qQA, qQB, qQC, qQD,
                                qKA, qKB, qKC, qKD,
                                qVA, qVB, qVC, qVD,
                                Wq.q2.data(), Wk.q2.data(), Wv.q2.data(),
                                qDim, kDim, vDim,
                                h0, hCount, xA, xB, xC, xD, j0);
                        }
                        else {
                            dot_i2s_i8_accum_i32_block32_x4(qQA, qQB, qQC, qQD, Wq.q2.data(), qDim, h0, hCount, xA, xB, xC, xD, j0);
                            dot_i2s_i8_accum_i32_block32_x4(qKA, qKB, qKC, qKD, Wk.q2.data(), kDim, h0, hCount, xA, xB, xC, xD, j0);
                            dot_i2s_i8_accum_i32_block32_x4(qVA, qVB, qVC, qVD, Wv.q2.data(), vDim, h0, hCount, xA, xB, xC, xD, j0);
                        }

                        accum_block32_scaled_to_y3_pair(qA, kA, vA, qB, kB, vB, qQA, qKA, qVA, qQB, qKB, qVB, scaleRowQ, scaleRowK, scaleRowV, j0, kb, xsA, xsB, addToDest);
                        accum_block32_scaled_to_y3_pair(qC, kC, vC, qD, kD, vD, qQC, qKC, qVC, qQD, qKD, qVD, scaleRowQ, scaleRowK, scaleRowV, j0, kb, xsC, xsD, addToDest);
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                        float* qA = Q + (t0 + tt + 0) * qDim; float* kA = K + (t0 + tt + 0) * kDim; float* vA = V + (t0 + tt + 0) * vDim;
                        float* qB = Q + (t0 + tt + 1) * qDim; float* kB = K + (t0 + tt + 1) * kDim; float* vB = V + (t0 + tt + 1) * vDim;
                        int32_t qQA[32] = { 0 }, qKA[32] = { 0 }, qVA[32] = { 0 }, qQB[32] = { 0 }, qKB[32] = { 0 }, qVB[32] = { 0 };
                        const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                        const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);

                        if (useDenseX2Accum) {
                            dot_i2s_i8_accum_i32_block32_x2_dense(qQA, qQB, Wq.q2.data(), qDim, h0, hCount, xA, xB, j0);
                            dot_i2s_i8_accum_i32_block32_x2_dense(qKA, qKB, Wk.q2.data(), kDim, h0, hCount, xA, xB, j0);
                            dot_i2s_i8_accum_i32_block32_x2_dense(qVA, qVB, Wv.q2.data(), vDim, h0, hCount, xA, xB, j0);
                        } else {
                            dot_i2s_i8_accum_i32_block32_x2(qQA, qQB, Wq.q2.data(), qDim, h0, hCount, xA, xB, j0);
                            dot_i2s_i8_accum_i32_block32_x2(qKA, qKB, Wk.q2.data(), kDim, h0, hCount, xA, xB, j0);
                            dot_i2s_i8_accum_i32_block32_x2(qVA, qVB, Wv.q2.data(), vDim, h0, hCount, xA, xB, j0);
                        }

                        accum_block32_scaled_to_y3_pair(qA, kA, vA, qB, kB, vB, qQA, qKA, qVA, qQB, qKB, qVB, scaleRowQ, scaleRowK, scaleRowV, j0, kb, xsA, xsB, addToDest);
                    }

                    for (; tt < tCount; ++tt) {
                        const int8_t* xRow8 = x8 + tt * inDim + h0;
                        float* qRow = Q + (t0 + tt) * qDim; float* kRow = K + (t0 + tt) * kDim; float* vRow = V + (t0 + tt) * vDim;
                        int32_t accQ[32] = { 0 }, accK[32] = { 0 }, accV[32] = { 0 };
                        const float xs = load_tile_row_scale(xScale, nKBlocks, tt, kb);

                        dot_i2s_i8_accum_i32_block32(accQ, Wq.q2.data(), qDim, h0, hCount, xRow8, j0);
                        dot_i2s_i8_accum_i32_block32(accK, Wk.q2.data(), kDim, h0, hCount, xRow8, j0);
                        dot_i2s_i8_accum_i32_block32(accV, Wv.q2.data(), vDim, h0, hCount, xRow8, j0);

                        accum_block32_scaled_to_y3(qRow, kRow, vRow, accQ, accK, accV, scaleRowQ, scaleRowK, scaleRowV, j0, kb, xs, addToDest);
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
    int8_t* __restrict x8Scratch, float* __restrict xScaleScratch,
    const int* attentionMask, std::size_t activePrefixT)
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
    const bool useFusedX4 = fused_x4_forward_preferred(T, tTile);
    const bool useDenseX2Accum = dense_x2_forward_accum_preferred(T, tTile);
    const bool useParallelQuant = parallel_multi_forward_quant_enabled();

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
            const std::size_t tCount = std::min(tTile, T - t0);
            if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;
#if defined(_OPENMP)
            if (useParallelQuant) {
                quantize_x_tile_i8_tiled_omp_for(x8Scratch, xScaleScratch, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
            }
            else {
#pragma omp single
                {
                    quantize_x_tile_i8_tiled(x8Scratch, xScaleScratch, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
                }
            }
#else
            quantize_x_tile_i8_tiled(x8Scratch, xScaleScratch, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
#endif

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

                        const float xs0 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb);
                        const float xs1 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb);
                        const float xs2 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 2, kb);
                        const float xs3 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 3, kb);
                        if (useFusedX4) {
                            const bool addToDest = (kb != 0);
                            const float blockScaleA = scaleRowA[kb];
                            const float blockScaleB = scaleRowB[kb];
                            const float gainA0 = xs0 * blockScaleA;
                            const float gainA1 = xs1 * blockScaleA;
                            const float gainA2 = xs2 * blockScaleA;
                            const float gainA3 = xs3 * blockScaleA;
                            const float gainB0 = xs0 * blockScaleB;
                            const float gainB1 = xs1 * blockScaleB;
                            const float gainB2 = xs2 * blockScaleB;
                            const float gainB3 = xs3 * blockScaleB;
                            dot_i2s_i8_scale_store_block16_x4_dual_exact(
                                yA0, yA1, yA2, yA3,
                                yB0, yB1, yB2, yB3,
                                Wa.q2.data(), Wb.q2.data(),
                                outDim, h0, hCount,
                                x0, x1, x2, x3,
                                j0,
                                addToDest,
                                gainA0, gainA1, gainA2, gainA3,
                                gainB0, gainB1, gainB2, gainB3);
                            dot_i2s_i8_scale_store_block16_x4_dual_exact(
                                yA0, yA1, yA2, yA3,
                                yB0, yB1, yB2, yB3,
                                Wa.q2.data(), Wb.q2.data(),
                                outDim, h0, hCount,
                                x0, x1, x2, x3,
                                j0 + 16,
                                addToDest,
                                gainA0, gainA1, gainA2, gainA3,
                                gainB0, gainB1, gainB2, gainB3);
                        }
                        else {
                            int32_t accA0[32] = { 0 }, accA1[32] = { 0 }, accA2[32] = { 0 }, accA3[32] = { 0 };
                            int32_t accB0[32] = { 0 }, accB1[32] = { 0 }, accB2[32] = { 0 }, accB3[32] = { 0 };

                            dot_i2s_i8_accum_i32_block32_x4(accA0, accA1, accA2, accA3, Wa.q2.data(), outDim, h0, hCount, x0, x1, x2, x3, j0);
                            dot_i2s_i8_accum_i32_block32_x4(accB0, accB1, accB2, accB3, Wb.q2.data(), outDim, h0, hCount, x0, x1, x2, x3, j0);

                            accum_block32_scaled_to_y_dual(yA0, yB0, accA0, accB0, scaleRowA, scaleRowB, j0, kb, xs0, xs0);
                            accum_block32_scaled_to_y_dual(yA1, yB1, accA1, accB1, scaleRowA, scaleRowB, j0, kb, xs1, xs1);
                            accum_block32_scaled_to_y_dual(yA2, yB2, accA2, accB2, scaleRowA, scaleRowB, j0, kb, xs2, xs2);
                            accum_block32_scaled_to_y_dual(yA3, yB3, accA3, accB3, scaleRowA, scaleRowB, j0, kb, xs3, xs3);
                        }
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* x0 = x8Scratch + (tt + 0) * inDim + h0;
                        const int8_t* x1 = x8Scratch + (tt + 1) * inDim + h0;
                        float* yA0 = Ya + (t0 + tt + 0) * outDim;
                        float* yA1 = Ya + (t0 + tt + 1) * outDim;
                        float* yB0 = Yb + (t0 + tt + 0) * outDim;
                        float* yB1 = Yb + (t0 + tt + 1) * outDim;

                        const float xs0 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb);
                        const float xs1 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb);
                        int32_t accA0[32] = { 0 }, accA1[32] = { 0 };
                        int32_t accB0[32] = { 0 }, accB1[32] = { 0 };
                        if (useDenseX2Accum) {
                            dot_i2s_i8_accum_i32_block32_x2_dense(accA0, accA1, Wa.q2.data(), outDim, h0, hCount, x0, x1, j0);
                            dot_i2s_i8_accum_i32_block32_x2_dense(accB0, accB1, Wb.q2.data(), outDim, h0, hCount, x0, x1, j0);
                        } else {
                            dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, Wa.q2.data(), outDim, h0, hCount, x0, x1, j0);
                            dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, Wb.q2.data(), outDim, h0, hCount, x0, x1, j0);
                        }

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

void linear_fused_mixed_dual_forward_bitnet_i2s_accum(
    float* __restrict Ya, float* __restrict Yb,
    const float* __restrict X,
    const BitNetMatI2S& Wa, const BitNetMatI2S& Wb,
    std::size_t T, std::size_t tTile,
    int8_t* __restrict x8Scratch, float* __restrict xScaleScratch,
    const int* attentionMask, std::size_t activePrefixT)
{
    const std::size_t inDim = Wa.rows;
    const std::size_t outDimA = Wa.cols;
    const std::size_t outDimB = Wb.cols;
    validate_forward_common(Wa, inDim, outDimA, "linear_fused_mixed_dual_forward_bitnet_i2s_accum/Wa");
    validate_forward_common(Wb, Wb.rows, outDimB, "linear_fused_mixed_dual_forward_bitnet_i2s_accum/Wb");
    if (Wb.rows != inDim || Wa.qk != Wb.qk || Wa.jq != Wb.jq || Wa.hadamard32 != Wb.hadamard32) {
        throw std::runtime_error("linear_fused_mixed_dual_forward_bitnet_i2s_accum: incompatible matrix shapes");
    }

    const std::size_t fusedCols = std::min(outDimA, outDimB);
    const std::size_t hBlock = Wa.qk;
    const std::size_t nKBlocks = ceil_div(inDim, hBlock);
    const bool useFusedX4 = fused_x4_forward_preferred(T, tTile);
    const bool useDenseX2Accum = dense_x2_forward_accum_preferred(T, tTile);
    const bool useParallelQuant = parallel_multi_forward_quant_enabled();
    const int fusedColsInt = static_cast<int>(fusedCols);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
            const std::size_t tCount = std::min(tTile, T - t0);
            if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;
#if defined(_OPENMP)
            if (useParallelQuant) {
                quantize_x_tile_i8_tiled_omp_for(x8Scratch, xScaleScratch, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
            }
            else {
#pragma omp single
                {
                    quantize_x_tile_i8_tiled(x8Scratch, xScaleScratch, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
                }
            }
#else
            quantize_x_tile_i8_tiled(x8Scratch, xScaleScratch, X, T, inDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
#endif

            auto run_single = [&](float* __restrict Y, const BitNetMatI2S& W, std::size_t outDim, std::size_t jBegin) {
                const int outDimInt = static_cast<int>(outDim);
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
                for (int j0_i = static_cast<int>(jBegin); j0_i < outDimInt; j0_i += 32) {
                    const std::size_t j0 = static_cast<std::size_t>(j0_i);
                    const float* const scaleRow = forward_scale_row(W, j0, nKBlocks);
                    for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                        const std::size_t hCount = std::min(hBlock, inDim - h0);
                        const std::size_t kb = h0 / hBlock;
                        std::size_t tt = 0;

                        for (; tt + 3 < tCount; tt += 4) {
                            const int8_t* xA = x8Scratch + (tt + 0) * inDim + h0; const int8_t* xB = x8Scratch + (tt + 1) * inDim + h0;
                            const int8_t* xC = x8Scratch + (tt + 2) * inDim + h0; const int8_t* xD = x8Scratch + (tt + 3) * inDim + h0;
                            float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                            float* yC = Y + (t0 + tt + 2) * outDim; float* yD = Y + (t0 + tt + 3) * outDim;
                            if (useFusedX4) {
                                const bool addToDest = (kb != 0);
                                const float ws = scaleRow[kb];
                                const float gainA = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb) * ws;
                                const float gainB = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb) * ws;
                                const float gainC = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 2, kb) * ws;
                                const float gainD = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 3, kb) * ws;

                                dot_i2s_i8_scale_store_block16_x4_exact(yA, yB, yC, yD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0, addToDest, gainA, gainB, gainC, gainD);
                                dot_i2s_i8_scale_store_block16_x4_exact(yA, yB, yC, yD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0 + 16, addToDest, gainA, gainB, gainC, gainD);
                            }
                            else {
                                int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };
                                const float xsA = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb);
                                const float xsB = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb);
                                const float xsC = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 2, kb);
                                const float xsD = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 3, kb);

                                dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                                accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB, true);
                                accum_block32_scaled_to_y_pair(yC, yD, aC, aD, scaleRow, j0, kb, xsC, xsD, true);
                            }
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* xA = x8Scratch + (tt + 0) * inDim + h0; const int8_t* xB = x8Scratch + (tt + 1) * inDim + h0;
                            float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                            const float xsA = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb);
                            const float xsB = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb);
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            if (useDenseX2Accum) {
                                dot_i2s_i8_accum_i32_block32_x2_dense(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            }
                            else {
                                dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            }
                            accum_block32_scaled_to_y_pair(yA, yB, aA, aB, scaleRow, j0, kb, xsA, xsB, true);
                        }

                        for (; tt < tCount; ++tt) {
                            float* yRow = Y + (t0 + tt) * outDim;
                            int32_t blockAcc[32] = { 0 };
                            const float xs = load_tile_row_scale(xScaleScratch, nKBlocks, tt, kb);
                            dot_i2s_i8_accum_i32_block32(blockAcc, W.q2.data(), outDim, h0, hCount, x8Scratch + tt * inDim + h0, j0);
                        accum_block32_scaled_to_y(yRow, blockAcc, scaleRow, j0, kb, xs, true);
                        }
                    }
                }
            };

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int j0_i = 0; j0_i < fusedColsInt; j0_i += 32) {
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

                        float* yA0 = Ya + (t0 + tt + 0) * outDimA;
                        float* yA1 = Ya + (t0 + tt + 1) * outDimA;
                        float* yA2 = Ya + (t0 + tt + 2) * outDimA;
                        float* yA3 = Ya + (t0 + tt + 3) * outDimA;
                        float* yB0 = Yb + (t0 + tt + 0) * outDimB;
                        float* yB1 = Yb + (t0 + tt + 1) * outDimB;
                        float* yB2 = Yb + (t0 + tt + 2) * outDimB;
                        float* yB3 = Yb + (t0 + tt + 3) * outDimB;

                        const float xs0 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb);
                        const float xs1 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb);
                        const float xs2 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 2, kb);
                        const float xs3 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 3, kb);

                        if (useFusedX4) {
                            const bool addToDest = (kb != 0);
                            const float blockScaleA = scaleRowA[kb];
                            const float blockScaleB = scaleRowB[kb];
                            const float gainA0 = xs0 * blockScaleA;
                            const float gainA1 = xs1 * blockScaleA;
                            const float gainA2 = xs2 * blockScaleA;
                            const float gainA3 = xs3 * blockScaleA;
                            const float gainB0 = xs0 * blockScaleB;
                            const float gainB1 = xs1 * blockScaleB;
                            const float gainB2 = xs2 * blockScaleB;
                            const float gainB3 = xs3 * blockScaleB;

                            dot_i2s_i8_scale_store_block16_x4_dual_exact(yA0, yA1, yA2, yA3, yB0, yB1, yB2, yB3, Wa.q2.data(), Wb.q2.data(), fusedCols, h0, hCount, x0, x1, x2, x3, j0, addToDest, gainA0, gainA1, gainA2, gainA3, gainB0, gainB1, gainB2, gainB3);
                            dot_i2s_i8_scale_store_block16_x4_dual_exact(yA0, yA1, yA2, yA3, yB0, yB1, yB2, yB3, Wa.q2.data(), Wb.q2.data(), fusedCols, h0, hCount, x0, x1, x2, x3, j0 + 16, addToDest, gainA0, gainA1, gainA2, gainA3, gainB0, gainB1, gainB2, gainB3);
                        }
                        else {
                            int32_t accA0[32] = { 0 }, accA1[32] = { 0 }, accA2[32] = { 0 }, accA3[32] = { 0 };
                            int32_t accB0[32] = { 0 }, accB1[32] = { 0 }, accB2[32] = { 0 }, accB3[32] = { 0 };

                            dot_i2s_i8_accum_i32_block32_x4(accA0, accA1, accA2, accA3, Wa.q2.data(), fusedCols, h0, hCount, x0, x1, x2, x3, j0);
                            dot_i2s_i8_accum_i32_block32_x4(accB0, accB1, accB2, accB3, Wb.q2.data(), fusedCols, h0, hCount, x0, x1, x2, x3, j0);

                            accum_block32_scaled_to_y_dual(yA0, yB0, accA0, accB0, scaleRowA, scaleRowB, j0, kb, xs0, xs0);
                            accum_block32_scaled_to_y_dual(yA1, yB1, accA1, accB1, scaleRowA, scaleRowB, j0, kb, xs1, xs1);
                            accum_block32_scaled_to_y_dual(yA2, yB2, accA2, accB2, scaleRowA, scaleRowB, j0, kb, xs2, xs2);
                            accum_block32_scaled_to_y_dual(yA3, yB3, accA3, accB3, scaleRowA, scaleRowB, j0, kb, xs3, xs3);
                        }
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* x0 = x8Scratch + (tt + 0) * inDim + h0;
                        const int8_t* x1 = x8Scratch + (tt + 1) * inDim + h0;
                        float* yA0 = Ya + (t0 + tt + 0) * outDimA;
                        float* yA1 = Ya + (t0 + tt + 1) * outDimA;
                        float* yB0 = Yb + (t0 + tt + 0) * outDimB;
                        float* yB1 = Yb + (t0 + tt + 1) * outDimB;

                        const float xs0 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb);
                        const float xs1 = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb);
                        int32_t accA0[32] = { 0 }, accA1[32] = { 0 };
                        int32_t accB0[32] = { 0 }, accB1[32] = { 0 };
                        if (useDenseX2Accum) {
                            dot_i2s_i8_accum_i32_block32_x2_dense(accA0, accA1, Wa.q2.data(), fusedCols, h0, hCount, x0, x1, j0);
                            dot_i2s_i8_accum_i32_block32_x2_dense(accB0, accB1, Wb.q2.data(), fusedCols, h0, hCount, x0, x1, j0);
                        }
                        else {
                            dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, Wa.q2.data(), fusedCols, h0, hCount, x0, x1, j0);
                            dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, Wb.q2.data(), fusedCols, h0, hCount, x0, x1, j0);
                        }

                        accum_block32_scaled_to_y_dual(yA0, yB0, accA0, accB0, scaleRowA, scaleRowB, j0, kb, xs0, xs0);
                        accum_block32_scaled_to_y_dual(yA1, yB1, accA1, accB1, scaleRowA, scaleRowB, j0, kb, xs1, xs1);
                    }

                    for (; tt < tCount; ++tt) {
                        const int8_t* xRow = x8Scratch + tt * inDim + h0;
                        float* yA = Ya + (t0 + tt) * outDimA;
                        float* yB = Yb + (t0 + tt) * outDimB;
                        int32_t accA[32] = { 0 };
                        int32_t accB[32] = { 0 };
                        const float xs = load_tile_row_scale(xScaleScratch, nKBlocks, tt, kb);

                        dot_i2s_i8_accum_i32_block32(accA, Wa.q2.data(), fusedCols, h0, hCount, xRow, j0);
                        dot_i2s_i8_accum_i32_block32(accB, Wb.q2.data(), fusedCols, h0, hCount, xRow, j0);

                        accum_block32_scaled_to_y_dual(yA, yB, accA, accB, scaleRowA, scaleRowB, j0, kb, xs, xs);
                    }
                }
            }

            if (outDimA > fusedCols) run_single(Ya, Wa, outDimA, fusedCols);
            if (outDimB > fusedCols) run_single(Yb, Wb, outDimB, fusedCols);
        }
    }
}

// ============================================================
// Backward Kernels (Packed Transpose Option B)
// ============================================================

void linear_backward_bitnet_i2s_packed(
    float* __restrict dX, const float* __restrict dY, const BitNetMatI2S& W,
    std::size_t T, std::size_t tTile,
    int8_t* __restrict dy8Scratch, float* __restrict dyScaleScratch,
    const int* attentionMask, std::size_t activePrefixT, bool destStartsZero)
{
    const std::size_t inDim = W.rows;
    const std::size_t outDim = W.cols;
    validate_backward_packed(W, inDim, outDim, "linear_backward_bitnet_i2s_packed");
    const std::size_t hBlock = W.jq;
    const std::size_t nJBlocks = ceil_div(outDim, hBlock);
    const bool useFusedX4 = fused_x4_single_backward_preferred(T, tTile);
    const bool useFusedX2 = fused_x2_single_backward_preferred(T, tTile);
    const bool useParallelQuant = parallel_single_backward_quant_enabled(T, tTile);
    const bool useSingleTeam = single_team_backward_preferred(T, tTile);
    const bool useFusedX4Overwrite = fused_x4_single_backward_overwrite_enabled();
    const bool useFusedX4Block32 = useFusedX4 && fused_x4_single_backward_block32_exact_enabled(T, tTile);

#if defined(_OPENMP)
    if (useSingleTeam) {
#pragma omp parallel
        {
            for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
                const std::size_t tCount = std::min(tTile, T - t0);
                if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;
                const int inDim_int = static_cast<int>(inDim);

                if (useParallelQuant) {
                    quantize_x_tile_i8_tiled_omp_for(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
                }
                else {
#pragma omp single
                    quantize_x_tile_i8_tiled(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
                }

#pragma omp for schedule(static)
                for (int i0_i = 0; i0_i < inDim_int; i0_i += 32) {
                    std::size_t i0 = static_cast<std::size_t>(i0_i);
                    const float* const scaleRow = backward_scale_row(W, i0, nJBlocks);
                    const uint8_t* const wq2T = W.q2T.data();

                    for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                        const std::size_t jCount = std::min(hBlock, outDim - j0);
                        const std::size_t jb = j0 / hBlock;
                        const bool addToDest = (jb != 0);
                        const bool overwriteDest = destStartsZero && !addToDest;
                        const float blockScale = scaleRow[jb];

                        std::size_t tt = 0;
                        for (; tt + 3 < tCount; tt += 4) {
                            const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                            const int8_t* dyC = dy8Scratch + (tt + 2) * outDim + j0; const int8_t* dyD = dy8Scratch + (tt + 3) * outDim + j0;
                            float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                            float* dxC = dX + (t0 + tt + 2) * inDim; float* dxD = dX + (t0 + tt + 3) * inDim;

                            const float dyScaleA = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 0, jb);
                            const float dyScaleB = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 1, jb);
                            const float dyScaleC = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 2, jb);
                            const float dyScaleD = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 3, jb);
                            const float gainA = dyScaleA * blockScale;
                            const float gainB = dyScaleB * blockScale;
                            const float gainC = dyScaleC * blockScale;
                            const float gainD = dyScaleD * blockScale;
                            if (useFusedX4) {
                                if (useFusedX4Block32) {
                                    if (overwriteDest && useFusedX4Overwrite) {
                                        dot_i2s_i8_scale_store_block32_x4_bwd_exact_overwrite(
                                            dxA, dxB, dxC, dxD,
                                            wq2T,
                                            inDim, j0, jCount,
                                            dyA, dyB, dyC, dyD,
                                            i0,
                                            gainA, gainB, gainC, gainD);
                                    }
                                    else {
                                        dot_i2s_i8_scale_store_block32_x4_bwd_exact(
                                            dxA, dxB, dxC, dxD,
                                            wq2T,
                                            inDim, j0, jCount,
                                            dyA, dyB, dyC, dyD,
                                            i0,
                                            addToDest,
                                            gainA, gainB, gainC, gainD);
                                    }
                                }
                                else if (overwriteDest && useFusedX4Overwrite) {
                                    dot_i2s_i8_scale_store_block16_x4_bwd_exact_overwrite(
                                        dxA, dxB, dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB, dyC, dyD,
                                        i0,
                                        gainA, gainB, gainC, gainD);
                                    dot_i2s_i8_scale_store_block16_x4_bwd_exact_overwrite(
                                        dxA, dxB, dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB, dyC, dyD,
                                        i0 + 16,
                                        gainA, gainB, gainC, gainD);
                                }
                                else {
                                    dot_i2s_i8_scale_store_block16_x4_bwd_exact(
                                        dxA, dxB, dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB, dyC, dyD,
                                        i0,
                                        addToDest,
                                        gainA, gainB, gainC, gainD);
                                    dot_i2s_i8_scale_store_block16_x4_bwd_exact(
                                        dxA, dxB, dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB, dyC, dyD,
                                        i0 + 16,
                                        addToDest,
                                        gainA, gainB, gainC, gainD);
                                }
                            }
                            else if (useFusedX2) {
                                if (overwriteDest) {
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                        dxA, dxB,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB,
                                        i0,
                                        gainA, gainB);
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                        dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyC, dyD,
                                        i0,
                                        gainC, gainD);
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                        dxA, dxB,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB,
                                        i0 + 16,
                                        gainA, gainB);
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                        dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyC, dyD,
                                        i0 + 16,
                                        gainC, gainD);
                                }
                                else {
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                        dxA, dxB,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB,
                                        i0,
                                        addToDest,
                                        gainA, gainB);
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                        dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyC, dyD,
                                        i0,
                                        addToDest,
                                        gainC, gainD);
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                        dxA, dxB,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB,
                                        i0 + 16,
                                        addToDest,
                                        gainA, gainB);
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                        dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyC, dyD,
                                        i0 + 16,
                                        addToDest,
                                        gainC, gainD);
                                }
                            }
                            else {
                                int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };
                                dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, wq2T, inDim, j0, jCount, dyA, dyB, dyC, dyD, i0);
                                accum_bwd_block32_gain_to_x_pair(dxA, dxB, aA, aB, i0, gainA, gainB, addToDest);
                                accum_bwd_block32_gain_to_x_pair(dxC, dxD, aC, aD, i0, gainC, gainD, addToDest);
                            }
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                            float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                            const float dyScaleA = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 0, jb);
                            const float dyScaleB = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 1, jb);
                            const float gainA = dyScaleA * blockScale;
                            const float gainB = dyScaleB * blockScale;
                            if (useFusedX2) {
                                if (overwriteDest) {
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                        dxA, dxB,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB,
                                        i0,
                                        gainA, gainB);
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                        dxA, dxB,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB,
                                        i0 + 16,
                                        gainA, gainB);
                                }
                                else {
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                        dxA, dxB,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB,
                                        i0,
                                        addToDest,
                                        gainA, gainB);
                                    dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                        dxA, dxB,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB,
                                        i0 + 16,
                                        addToDest,
                                        gainA, gainB);
                                }
                            }
                            else {
                                int32_t aA[32] = { 0 }, aB[32] = { 0 };
                                dot_i2s_i8_accum_i32_block32_x2(aA, aB, wq2T, inDim, j0, jCount, dyA, dyB, i0);
                                accum_bwd_block32_gain_to_x_pair(dxA, dxB, aA, aB, i0, gainA, gainB, addToDest);
                            }
                        }

                        for (; tt < tCount; ++tt) {
                            float* dxRow = dX + (t0 + tt) * inDim;
                            int32_t acc[32] = { 0 };
                            const float dyScale = load_tile_row_scale(dyScaleScratch, nJBlocks, tt, jb);
                            dot_i2s_i8_accum_i32_block32(acc, wq2T, inDim, j0, jCount, dy8Scratch + tt * outDim + j0, i0);
                            accum_bwd_block32_gain_to_x(dxRow, acc, i0, dyScale * blockScale, addToDest);
                        }
                    }
                }
            }
        }
        return;
    }
#endif

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;
        const int inDim_int = static_cast<int>(inDim);

#if defined(_OPENMP)
#pragma omp parallel
        {
            if (useParallelQuant) {
                quantize_x_tile_i8_tiled_omp_for(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
            }
            else {
#pragma omp single
                quantize_x_tile_i8_tiled(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
            }
#else
        quantize_x_tile_i8_tiled(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
#endif

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int i0_i = 0; i0_i < inDim_int; i0_i += 32) {
                std::size_t i0 = static_cast<std::size_t>(i0_i);
                const float* const scaleRow = backward_scale_row(W, i0, nJBlocks);
                const uint8_t* const wq2T = W.q2T.data();

                for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                    const std::size_t jCount = std::min(hBlock, outDim - j0);
                    const std::size_t jb = j0 / hBlock;
                    const bool addToDest = (jb != 0);
                    const bool overwriteDest = destStartsZero && !addToDest;
                    const float blockScale = scaleRow[jb];
                    std::size_t tt = 0;
                    for (; tt + 3 < tCount; tt += 4) {
                        const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                        const int8_t* dyC = dy8Scratch + (tt + 2) * outDim + j0; const int8_t* dyD = dy8Scratch + (tt + 3) * outDim + j0;
                        float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                        float* dxC = dX + (t0 + tt + 2) * inDim; float* dxD = dX + (t0 + tt + 3) * inDim;

                        const float dyScaleA = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 0, jb);
                        const float dyScaleB = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 1, jb);
                        const float dyScaleC = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 2, jb);
                        const float dyScaleD = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 3, jb);
                        const float gainA = dyScaleA * blockScale;
                        const float gainB = dyScaleB * blockScale;
                        const float gainC = dyScaleC * blockScale;
                        const float gainD = dyScaleD * blockScale;
                        if (useFusedX4) {
                            if (useFusedX4Block32) {
                                if (overwriteDest && useFusedX4Overwrite) {
                                    dot_i2s_i8_scale_store_block32_x4_bwd_exact_overwrite(
                                        dxA, dxB, dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB, dyC, dyD,
                                        i0,
                                        gainA, gainB, gainC, gainD);
                                }
                                else {
                                    dot_i2s_i8_scale_store_block32_x4_bwd_exact(
                                        dxA, dxB, dxC, dxD,
                                        wq2T,
                                        inDim, j0, jCount,
                                        dyA, dyB, dyC, dyD,
                                        i0,
                                        addToDest,
                                        gainA, gainB, gainC, gainD);
                                }
                            }
                            else if (overwriteDest && useFusedX4Overwrite) {
                                dot_i2s_i8_scale_store_block16_x4_bwd_exact_overwrite(
                                    dxA, dxB, dxC, dxD,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB, dyC, dyD,
                                    i0,
                                    gainA, gainB, gainC, gainD);
                                dot_i2s_i8_scale_store_block16_x4_bwd_exact_overwrite(
                                    dxA, dxB, dxC, dxD,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB, dyC, dyD,
                                    i0 + 16,
                                    gainA, gainB, gainC, gainD);
                            }
                            else {
                                dot_i2s_i8_scale_store_block16_x4_bwd_exact(
                                    dxA, dxB, dxC, dxD,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB, dyC, dyD,
                                    i0,
                                    addToDest,
                                    gainA, gainB, gainC, gainD);
                                dot_i2s_i8_scale_store_block16_x4_bwd_exact(
                                    dxA, dxB, dxC, dxD,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB, dyC, dyD,
                                    i0 + 16,
                                    addToDest,
                                    gainA, gainB, gainC, gainD);
                            }
                        }
                        else if (useFusedX2) {
                            if (overwriteDest) {
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                    dxA, dxB,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB,
                                    i0,
                                    gainA, gainB);
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                    dxC, dxD,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyC, dyD,
                                    i0,
                                    gainC, gainD);
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                    dxA, dxB,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB,
                                    i0 + 16,
                                    gainA, gainB);
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                    dxC, dxD,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyC, dyD,
                                    i0 + 16,
                                    gainC, gainD);
                            }
                            else {
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                    dxA, dxB,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB,
                                    i0,
                                    addToDest,
                                    gainA, gainB);
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                    dxC, dxD,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyC, dyD,
                                    i0,
                                    addToDest,
                                    gainC, gainD);
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                    dxA, dxB,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB,
                                    i0 + 16,
                                    addToDest,
                                    gainA, gainB);
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                    dxC, dxD,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyC, dyD,
                                    i0 + 16,
                                    addToDest,
                                    gainC, gainD);
                            }
                        }
                        else {
                            int32_t aA[32] = { 0 }, aB[32] = { 0 }, aC[32] = { 0 }, aD[32] = { 0 };
                            dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, wq2T, inDim, j0, jCount, dyA, dyB, dyC, dyD, i0);
                            accum_bwd_block32_gain_to_x_pair(dxA, dxB, aA, aB, i0, gainA, gainB, addToDest);
                            accum_bwd_block32_gain_to_x_pair(dxC, dxD, aC, aD, i0, gainC, gainD, addToDest);
                        }
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                        float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                        const float dyScaleA = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 0, jb);
                        const float dyScaleB = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 1, jb);
                        const float gainA = dyScaleA * blockScale;
                        const float gainB = dyScaleB * blockScale;
                        if (useFusedX2) {
                            if (overwriteDest) {
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                    dxA, dxB,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB,
                                    i0,
                                    gainA, gainB);
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
                                    dxA, dxB,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB,
                                    i0 + 16,
                                    gainA, gainB);
                            }
                            else {
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                    dxA, dxB,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB,
                                    i0,
                                    addToDest,
                                    gainA, gainB);
                                dot_i2s_i8_scale_store_block16_x2_bwd_exact(
                                    dxA, dxB,
                                    wq2T,
                                    inDim, j0, jCount,
                                    dyA, dyB,
                                    i0 + 16,
                                    addToDest,
                                    gainA, gainB);
                            }
                        }
                        else {
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            dot_i2s_i8_accum_i32_block32_x2(aA, aB, wq2T, inDim, j0, jCount, dyA, dyB, i0);
                            accum_bwd_block32_gain_to_x_pair(dxA, dxB, aA, aB, i0, gainA, gainB, addToDest);
                        }
                    }

                    for (; tt < tCount; ++tt) {
                        float* dxRow = dX + (t0 + tt) * inDim;
                        int32_t acc[32] = { 0 };
                        const float dyScale = load_tile_row_scale(dyScaleScratch, nJBlocks, tt, jb);
                        dot_i2s_i8_accum_i32_block32(acc, wq2T, inDim, j0, jCount, dy8Scratch + tt * outDim + j0, i0);
                        accum_bwd_block32_gain_to_x(dxRow, acc, i0, dyScale * blockScale, addToDest);
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
    const bool useParallelQuant = parallel_multi_backward_quant_enabled();
    const bool useDualKvQuant = (kDim == vDim) && (Wk.hadamard32 == Wv.hadamard32);
    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);

#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
#if defined(_OPENMP)
            if (useParallelQuant) {
                quantize_x_tile_i8_tiled_omp_for(dq8Scratch, dqScaleScratch, dQ, T, qDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);
                if (useDualKvQuant) {
                    quantize_x_tile_i8_tiled_dual_omp_for(
                        dk8Scratch, dkScaleScratch, dK,
                        dv8Scratch, dvScaleScratch, dV,
                        T, kDim, t0, tCount, kRowsPerQuantScale, hBlock, Wk.hadamard32);
                }
                else {
                    quantize_x_tile_i8_tiled_omp_for(dk8Scratch, dkScaleScratch, dK, T, kDim, t0, tCount, kRowsPerQuantScale, hBlock, Wk.hadamard32);
                    quantize_x_tile_i8_tiled_omp_for(dv8Scratch, dvScaleScratch, dV, T, vDim, t0, tCount, kRowsPerQuantScale, hBlock, Wv.hadamard32);
                }
            }
            else {
 #pragma omp single
                {
                    quantize_x_tile_i8_tiled(dq8Scratch, dqScaleScratch, dQ, T, qDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);
                    if (useDualKvQuant) {
                        quantize_x_tile_i8_tiled_dual(
                            dk8Scratch, dkScaleScratch, dK,
                            dv8Scratch, dvScaleScratch, dV,
                            T, kDim, t0, tCount, kRowsPerQuantScale, hBlock, Wk.hadamard32);
                    }
                    else {
                        quantize_x_tile_i8_tiled(dk8Scratch, dkScaleScratch, dK, T, kDim, t0, tCount, kRowsPerQuantScale, hBlock, Wk.hadamard32);
                        quantize_x_tile_i8_tiled(dv8Scratch, dvScaleScratch, dV, T, vDim, t0, tCount, kRowsPerQuantScale, hBlock, Wv.hadamard32);
                    }
                }
            }
#else
            quantize_x_tile_i8_tiled(dq8Scratch, dqScaleScratch, dQ, T, qDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);
            if (useDualKvQuant) {
                quantize_x_tile_i8_tiled_dual(
                    dk8Scratch, dkScaleScratch, dK,
                    dv8Scratch, dvScaleScratch, dV,
                    T, kDim, t0, tCount, kRowsPerQuantScale, hBlock, Wk.hadamard32);
            }
            else {
                quantize_x_tile_i8_tiled(dk8Scratch, dkScaleScratch, dK, T, kDim, t0, tCount, kRowsPerQuantScale, hBlock, Wk.hadamard32);
                quantize_x_tile_i8_tiled(dv8Scratch, dvScaleScratch, dV, T, vDim, t0, tCount, kRowsPerQuantScale, hBlock, Wv.hadamard32);
            }
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
                        const bool addToDest = (j0 != 0);

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
                            accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aA, aB, scaleRow, i0, jb, sA, sB, addToDest);
                            accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aC, aD, scaleRow, i0, jb, sC, sD, addToDest);
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* dyA = dy8 + (tt + 0) * outDim + j0; const int8_t* dyB = dy8 + (tt + 1) * outDim + j0;
                            float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            const float sA = load_tile_row_scale(dyScale, dyJBlocks, tt + 0, jb);
                            const float sB = load_tile_row_scale(dyScale, dyJBlocks, tt + 1, jb);

                            dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2T.data(), inDim, j0, jCount, dyA, dyB, i0);
                            accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aA, aB, scaleRow, i0, jb, sA, sB, addToDest);
                        }

                        for (; tt < tCount; ++tt) {
                            float* dxRow = dX + (t0 + tt) * inDim;
                            int32_t acc[32] = { 0 };
                            const float s = load_tile_row_scale(dyScale, dyJBlocks, tt, jb);
                            dot_i2s_i8_accum_i32_block32(acc, W.q2T.data(), inDim, j0, jCount, dy8 + tt * outDim + j0, i0);
                            accum_bwd_block32_scaled_to_x(dxRow, acc, scaleRow, i0, jb, s, addToDest);
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
                    const bool addToDestQ = (j0 != 0);
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

                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aQA, aQB, scaleRowQ, i0, jbQ, sqA, sqB, addToDestQ);
                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aKA, aKB, scaleRowK, i0, jbK, skA, skB, true);
                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aVA, aVB, scaleRowV, i0, jbV, svA, svB, true);
                        accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aQC, aQD, scaleRowQ, i0, jbQ, sqC, sqD, addToDestQ);
                        accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aKC, aKD, scaleRowK, i0, jbK, skC, skD, true);
                        accum_bwd_block32_scaled_to_x_pair(dxC, dxD, aVC, aVD, scaleRowV, i0, jbV, svC, svD, true);
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

                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aQA, aQB, scaleRowQ, i0, jbQ, sqA, sqB, addToDestQ);
                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aKA, aKB, scaleRowK, i0, jbK, skA, skB, true);
                        accum_bwd_block32_scaled_to_x_pair(dxA, dxB, aVA, aVB, scaleRowV, i0, jbV, svA, svB, true);
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

                        accum_bwd_block32_scaled_to_x(dxRow, aQ, scaleRowQ, i0, jbQ, sq, addToDestQ);
                        accum_bwd_block32_scaled_to_x(dxRow, aK, scaleRowK, i0, jbK, sk, true);
                        accum_bwd_block32_scaled_to_x(dxRow, aV, scaleRowV, i0, jbV, sv, true);
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
    float* __restrict dyBScaleScratch,
    const int* attentionMask, std::size_t activePrefixT, bool destStartsZero)
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
    const bool useFusedX4 = fused_x4_backward_preferred(T, tTile);
    const bool useFusedX2 = fused_x2_dual_backward_preferred(T, tTile);
    const bool useParallelQuant = parallel_multi_backward_quant_enabled();
    const bool useSingleTeam = single_team_backward_preferred(T, tTile);
    const bool useDenseX2Dual = useFusedX2 && tTile <= 2 && T >= 128;

#if defined(_OPENMP)
    if (useSingleTeam) {
#pragma omp parallel
        {
            const bool useChunkQuant =
                !useParallelQuant &&
                tTile <= 2 &&
                T >= 128 &&
                activePrefixT == T;
            const bool useMasterQuant =
                useChunkQuant &&
                master_dual_chunk_quant_preferred();
            const std::size_t quantChunkRows = useChunkQuant
                ? env_size_override("QLORA_TEST_BITNET_DUAL_BWD_QUANT_CHUNK_ROWS", static_cast<std::size_t>(4))
                : tTile;

            for (std::size_t q0 = 0; q0 < T; q0 += quantChunkRows) {
                const std::size_t qCount = std::min(quantChunkRows, T - q0);
                const int inDimInt = static_cast<int>(inDim);

                if (useParallelQuant) {
                    quantize_x_tile_i8_tiled_dual_omp_for(
                        dyA8Scratch, dyAScaleScratch, dYa,
                        dyB8Scratch, dyBScaleScratch, dYb,
                        T, outDim, q0, qCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
                }
                else if (useMasterQuant) {
#pragma omp master
                    {
                        quantize_x_tile_i8_tiled_dual(
                            dyA8Scratch, dyAScaleScratch, dYa,
                            dyB8Scratch, dyBScaleScratch, dYb,
                            T, outDim, q0, qCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
                    }
#pragma omp barrier
                }
                else {
#pragma omp single
                    {
                        quantize_x_tile_i8_tiled_dual(
                            dyA8Scratch, dyAScaleScratch, dYa,
                            dyB8Scratch, dyBScaleScratch, dYb,
                            T, outDim, q0, qCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
                    }
                }

                const std::size_t defaultWorkTileRows =
                    (useChunkQuant && tTile == 2 && qCount >= 4)
                    ? std::min<std::size_t>(qCount, static_cast<std::size_t>(4))
                    : tTile;
                const std::size_t workTileRows = useChunkQuant
                    ? std::min(
                        qCount,
                        std::max<std::size_t>(
                            1,
                            env_size_override(
                                "QLORA_TEST_BITNET_DUAL_BWD_WORK_TILE_ROWS",
                                defaultWorkTileRows)))
                    : defaultWorkTileRows;

                for (std::size_t t0 = q0; t0 < q0 + qCount; t0 += workTileRows) {
                    const std::size_t tCount = std::min(workTileRows, (q0 + qCount) - t0);
                    if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;

                    const std::size_t tileRowBase = t0 - q0;
                    const int8_t* const dyA8Tile = dyA8Scratch + tileRowBase * outDim;
                    const int8_t* const dyB8Tile = dyB8Scratch + tileRowBase * outDim;
                    const float* const dyAScaleTile = dyAScaleScratch + tileRowBase * nJBlocks;
                    const float* const dyBScaleTile = dyBScaleScratch + tileRowBase * nJBlocks;

#pragma omp for schedule(static)
                    for (int i0_i = 0; i0_i < inDimInt; i0_i += 32) {
                    const std::size_t i0 = static_cast<std::size_t>(i0_i);
                    const float* const scaleRowA = backward_scale_row(Wa, i0, nJBlocks);
                    const float* const scaleRowB = backward_scale_row(Wb, i0, nJBlocks);
                    const uint8_t* const waq2T = Wa.q2T.data();
                    const uint8_t* const wbq2T = Wb.q2T.data();
                    for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                        const std::size_t jCount = std::min(hBlock, outDim - j0);
                        const std::size_t jb = j0 / hBlock;
                        const bool addToDest = (jb != 0);
                        const bool overwriteDest = destStartsZero && !addToDest;
                        const float blockScaleA = scaleRowA[jb];
                        const float blockScaleB = scaleRowB[jb];

                        std::size_t tt = 0;
                        for (; tt + 3 < tCount; tt += 4) {
                            const int8_t* dyA0 = dyA8Tile + (tt + 0) * outDim + j0;
                            const int8_t* dyA1 = dyA8Tile + (tt + 1) * outDim + j0;
                            const int8_t* dyA2 = dyA8Tile + (tt + 2) * outDim + j0;
                            const int8_t* dyA3 = dyA8Tile + (tt + 3) * outDim + j0;
                            const int8_t* dyB0 = dyB8Tile + (tt + 0) * outDim + j0;
                            const int8_t* dyB1 = dyB8Tile + (tt + 1) * outDim + j0;
                            const int8_t* dyB2 = dyB8Tile + (tt + 2) * outDim + j0;
                            const int8_t* dyB3 = dyB8Tile + (tt + 3) * outDim + j0;

                            float* dx0 = dX + (t0 + tt + 0) * inDim;
                            float* dx1 = dX + (t0 + tt + 1) * inDim;
                            float* dx2 = dX + (t0 + tt + 2) * inDim;
                            float* dx3 = dX + (t0 + tt + 3) * inDim;

                            const float scaleA0 = load_tile_row_scale(dyAScaleTile, nJBlocks, tt + 0, jb);
                            const float scaleA1 = load_tile_row_scale(dyAScaleTile, nJBlocks, tt + 1, jb);
                            const float scaleA2 = load_tile_row_scale(dyAScaleTile, nJBlocks, tt + 2, jb);
                            const float scaleA3 = load_tile_row_scale(dyAScaleTile, nJBlocks, tt + 3, jb);
                            const float scaleB0 = load_tile_row_scale(dyBScaleTile, nJBlocks, tt + 0, jb);
                            const float scaleB1 = load_tile_row_scale(dyBScaleTile, nJBlocks, tt + 1, jb);
                            const float scaleB2 = load_tile_row_scale(dyBScaleTile, nJBlocks, tt + 2, jb);
                            const float scaleB3 = load_tile_row_scale(dyBScaleTile, nJBlocks, tt + 3, jb);
                            const float gainA0 = scaleA0 * blockScaleA;
                            const float gainA1 = scaleA1 * blockScaleA;
                            const float gainA2 = scaleA2 * blockScaleA;
                            const float gainA3 = scaleA3 * blockScaleA;
                            const float gainB0 = scaleB0 * blockScaleB;
                            const float gainB1 = scaleB1 * blockScaleB;
                            const float gainB2 = scaleB2 * blockScaleB;
                            const float gainB3 = scaleB3 * blockScaleB;
                            if (useFusedX4) {
                                dot_i2s_i8_scale_store_block16_x4_dual_bwd_exact(
                                    dx0, dx1, dx2, dx3,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1, dyA2, dyA3,
                                    dyB0, dyB1, dyB2, dyB3,
                                    i0,
                                    addToDest,
                                    gainA0, gainA1, gainA2, gainA3,
                                    gainB0, gainB1, gainB2, gainB3);
                                dot_i2s_i8_scale_store_block16_x4_dual_bwd_exact(
                                    dx0, dx1, dx2, dx3,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1, dyA2, dyA3,
                                    dyB0, dyB1, dyB2, dyB3,
                                    i0 + 16,
                                    addToDest,
                                    gainA0, gainA1, gainA2, gainA3,
                                    gainB0, gainB1, gainB2, gainB3);
                            }
                            else if (useFusedX2) {
                                if (overwriteDest) {
                                    const auto storeOverwrite = useDenseX2Dual
                                        ? dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite_dense
                                        : dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite;
                                    storeOverwrite(
                                        dx0, dx1,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA0, dyA1,
                                        dyB0, dyB1,
                                        i0,
                                        gainA0, gainA1,
                                        gainB0, gainB1);
                                    storeOverwrite(
                                        dx0, dx1,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA0, dyA1,
                                        dyB0, dyB1,
                                        i0 + 16,
                                        gainA0, gainA1,
                                        gainB0, gainB1);
                                    storeOverwrite(
                                        dx2, dx3,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA2, dyA3,
                                        dyB2, dyB3,
                                        i0,
                                        gainA2, gainA3,
                                        gainB2, gainB3);
                                    storeOverwrite(
                                        dx2, dx3,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA2, dyA3,
                                        dyB2, dyB3,
                                        i0 + 16,
                                        gainA2, gainA3,
                                        gainB2, gainB3);
                                }
                                else {
                                    const auto storeAccum = useDenseX2Dual
                                        ? dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_dense
                                        : dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact;
                                    storeAccum(
                                        dx0, dx1,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA0, dyA1,
                                        dyB0, dyB1,
                                        i0,
                                        addToDest,
                                        gainA0, gainA1,
                                        gainB0, gainB1);
                                    storeAccum(
                                        dx0, dx1,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA0, dyA1,
                                        dyB0, dyB1,
                                        i0 + 16,
                                        addToDest,
                                        gainA0, gainA1,
                                        gainB0, gainB1);
                                    storeAccum(
                                        dx2, dx3,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA2, dyA3,
                                        dyB2, dyB3,
                                        i0,
                                        addToDest,
                                        gainA2, gainA3,
                                        gainB2, gainB3);
                                    storeAccum(
                                        dx2, dx3,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA2, dyA3,
                                        dyB2, dyB3,
                                        i0 + 16,
                                        addToDest,
                                        gainA2, gainA3,
                                        gainB2, gainB3);
                                }
                            }
                            else {
                                int32_t aA0[32] = { 0 }, aA1[32] = { 0 }, aA2[32] = { 0 }, aA3[32] = { 0 };
                                int32_t aB0[32] = { 0 }, aB1[32] = { 0 }, aB2[32] = { 0 }, aB3[32] = { 0 };
                                dot_i2s_i8_accum_i32_block32_x4_dual(
                                    aA0, aA1, aA2, aA3,
                                    aB0, aB1, aB2, aB3,
                                    waq2T, wbq2T, inDim, j0, jCount,
                                    dyA0, dyA1, dyA2, dyA3,
                                    dyB0, dyB1, dyB2, dyB3,
                                    i0);
                                accum_bwd_block32_gain_to_x_dual_x4(
                                    dx0, dx1, dx2, dx3,
                                    aA0, aA1, aA2, aA3,
                                    aB0, aB1, aB2, aB3,
                                    i0,
                                    gainA0, gainA1, gainA2, gainA3,
                                    gainB0, gainB1, gainB2, gainB3,
                                    addToDest);
                            }
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* dyA0 = dyA8Tile + (tt + 0) * outDim + j0;
                            const int8_t* dyA1 = dyA8Tile + (tt + 1) * outDim + j0;
                            const int8_t* dyB0 = dyB8Tile + (tt + 0) * outDim + j0;
                            const int8_t* dyB1 = dyB8Tile + (tt + 1) * outDim + j0;
                            float* dx0 = dX + (t0 + tt + 0) * inDim;
                            float* dx1 = dX + (t0 + tt + 1) * inDim;
                            const float scaleA0 = load_tile_row_scale(dyAScaleTile, nJBlocks, tt + 0, jb);
                            const float scaleA1 = load_tile_row_scale(dyAScaleTile, nJBlocks, tt + 1, jb);
                            const float scaleB0 = load_tile_row_scale(dyBScaleTile, nJBlocks, tt + 0, jb);
                            const float scaleB1 = load_tile_row_scale(dyBScaleTile, nJBlocks, tt + 1, jb);
                            const float gainA0 = scaleA0 * blockScaleA;
                            const float gainA1 = scaleA1 * blockScaleA;
                            const float gainB0 = scaleB0 * blockScaleB;
                            const float gainB1 = scaleB1 * blockScaleB;
                            if (useFusedX2) {
                                if (overwriteDest) {
                                    const auto storeOverwrite = useDenseX2Dual
                                        ? dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite_dense
                                        : dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite;
                                    storeOverwrite(
                                        dx0, dx1,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA0, dyA1,
                                        dyB0, dyB1,
                                        i0,
                                        gainA0, gainA1,
                                        gainB0, gainB1);
                                    storeOverwrite(
                                        dx0, dx1,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA0, dyA1,
                                        dyB0, dyB1,
                                        i0 + 16,
                                        gainA0, gainA1,
                                        gainB0, gainB1);
                                }
                                else {
                                    const auto storeAccum = useDenseX2Dual
                                        ? dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_dense
                                        : dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact;
                                    storeAccum(
                                        dx0, dx1,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA0, dyA1,
                                        dyB0, dyB1,
                                        i0,
                                        addToDest,
                                        gainA0, gainA1,
                                        gainB0, gainB1);
                                    storeAccum(
                                        dx0, dx1,
                                        waq2T, wbq2T,
                                        inDim, j0, jCount,
                                        dyA0, dyA1,
                                        dyB0, dyB1,
                                        i0 + 16,
                                        addToDest,
                                        gainA0, gainA1,
                                        gainB0, gainB1);
                                }
                            }
                            else {
                                int32_t aA0[32] = { 0 }, aA1[32] = { 0 };
                                int32_t aB0[32] = { 0 }, aB1[32] = { 0 };
                                dot_i2s_i8_accum_i32_block32_x2(aA0, aA1, waq2T, inDim, j0, jCount, dyA0, dyA1, i0);
                                dot_i2s_i8_accum_i32_block32_x2(aB0, aB1, wbq2T, inDim, j0, jCount, dyB0, dyB1, i0);
                                accum_bwd_block32_gain_to_x_dual(dx0, aA0, aB0, i0, gainA0, gainB0, addToDest);
                                accum_bwd_block32_gain_to_x_dual(dx1, aA1, aB1, i0, gainA1, gainB1, addToDest);
                            }
                        }

                        for (; tt < tCount; ++tt) {
                            float* dxRow = dX + (t0 + tt) * inDim;
                            int32_t accA[32] = { 0 };
                            int32_t accB[32] = { 0 };
                            const float scaleA = load_tile_row_scale(dyAScaleTile, nJBlocks, tt, jb);
                            const float scaleB = load_tile_row_scale(dyBScaleTile, nJBlocks, tt, jb);
                            dot_i2s_i8_accum_i32_block32(accA, waq2T, inDim, j0, jCount, dyA8Tile + tt * outDim + j0, i0);
                            dot_i2s_i8_accum_i32_block32(accB, wbq2T, inDim, j0, jCount, dyB8Tile + tt * outDim + j0, i0);
                            accum_bwd_block32_gain_to_x_dual(dxRow, accA, accB, i0, scaleA * blockScaleA, scaleB * blockScaleB, addToDest);
                            }
                        }

                    }
                }
            }
        }
    }
        return;
#endif
    
    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;

        const int inDimInt = static_cast<int>(inDim);

#if defined(_OPENMP)
#pragma omp parallel
        {
            if (useParallelQuant) {
                quantize_x_tile_i8_tiled_dual_omp_for(
                    dyA8Scratch, dyAScaleScratch, dYa,
                    dyB8Scratch, dyBScaleScratch, dYb,
                    T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
            }
            else {
#pragma omp single
                {
                    quantize_x_tile_i8_tiled_dual(
                        dyA8Scratch, dyAScaleScratch, dYa,
                        dyB8Scratch, dyBScaleScratch, dYb,
                        T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
                }
            }
#pragma omp for schedule(static)
            for (int i0_i = 0; i0_i < inDimInt; i0_i += 32) {
                const std::size_t i0 = static_cast<std::size_t>(i0_i);
                const float* const scaleRowA = backward_scale_row(Wa, i0, nJBlocks);
                const float* const scaleRowB = backward_scale_row(Wb, i0, nJBlocks);
                const uint8_t* const waq2T = Wa.q2T.data();
                const uint8_t* const wbq2T = Wb.q2T.data();

                for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                    const std::size_t jCount = std::min(hBlock, outDim - j0);
                    const std::size_t jb = j0 / hBlock;
                    const bool addToDest = (jb != 0);
                    const bool overwriteDest = destStartsZero && !addToDest;
                    const float blockScaleA = scaleRowA[jb];
                    const float blockScaleB = scaleRowB[jb];

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

                        const float scaleA0 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 0, jb);
                        const float scaleA1 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 1, jb);
                        const float scaleA2 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 2, jb);
                        const float scaleA3 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 3, jb);
                        const float scaleB0 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 0, jb);
                        const float scaleB1 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 1, jb);
                        const float scaleB2 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 2, jb);
                        const float scaleB3 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 3, jb);
                        const float gainA0 = scaleA0 * blockScaleA;
                        const float gainA1 = scaleA1 * blockScaleA;
                        const float gainA2 = scaleA2 * blockScaleA;
                        const float gainA3 = scaleA3 * blockScaleA;
                        const float gainB0 = scaleB0 * blockScaleB;
                        const float gainB1 = scaleB1 * blockScaleB;
                        const float gainB2 = scaleB2 * blockScaleB;
                        const float gainB3 = scaleB3 * blockScaleB;
                        if (useFusedX4) {
                            dot_i2s_i8_scale_store_block16_x4_dual_bwd_exact(
                                dx0, dx1, dx2, dx3,
                                waq2T, wbq2T,
                                inDim, j0, jCount,
                                dyA0, dyA1, dyA2, dyA3,
                                dyB0, dyB1, dyB2, dyB3,
                                i0,
                                addToDest,
                                gainA0, gainA1, gainA2, gainA3,
                                gainB0, gainB1, gainB2, gainB3);
                            dot_i2s_i8_scale_store_block16_x4_dual_bwd_exact(
                                dx0, dx1, dx2, dx3,
                                waq2T, wbq2T,
                                inDim, j0, jCount,
                                dyA0, dyA1, dyA2, dyA3,
                                dyB0, dyB1, dyB2, dyB3,
                                i0 + 16,
                                addToDest,
                                gainA0, gainA1, gainA2, gainA3,
                                gainB0, gainB1, gainB2, gainB3);
                        }
                        else if (useFusedX2) {
                            if (overwriteDest) {
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite(
                                    dx0, dx1,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1,
                                    dyB0, dyB1,
                                    i0,
                                    gainA0, gainA1,
                                    gainB0, gainB1);
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite(
                                    dx0, dx1,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1,
                                    dyB0, dyB1,
                                    i0 + 16,
                                    gainA0, gainA1,
                                    gainB0, gainB1);
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite(
                                    dx2, dx3,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA2, dyA3,
                                    dyB2, dyB3,
                                    i0,
                                    gainA2, gainA3,
                                    gainB2, gainB3);
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite(
                                    dx2, dx3,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA2, dyA3,
                                    dyB2, dyB3,
                                    i0 + 16,
                                    gainA2, gainA3,
                                    gainB2, gainB3);
                            }
                            else {
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact(
                                    dx0, dx1,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1,
                                    dyB0, dyB1,
                                    i0,
                                    addToDest,
                                    gainA0, gainA1,
                                    gainB0, gainB1);
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact(
                                    dx0, dx1,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1,
                                    dyB0, dyB1,
                                    i0 + 16,
                                    addToDest,
                                    gainA0, gainA1,
                                    gainB0, gainB1);
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact(
                                    dx2, dx3,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA2, dyA3,
                                    dyB2, dyB3,
                                    i0,
                                    addToDest,
                                    gainA2, gainA3,
                                    gainB2, gainB3);
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact(
                                    dx2, dx3,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA2, dyA3,
                                    dyB2, dyB3,
                                    i0 + 16,
                                    addToDest,
                                    gainA2, gainA3,
                                    gainB2, gainB3);
                            }
                        }
                        else {
                            int32_t accA0[32] = { 0 }, accA1[32] = { 0 }, accA2[32] = { 0 }, accA3[32] = { 0 };
                            int32_t accB0[32] = { 0 }, accB1[32] = { 0 }, accB2[32] = { 0 }, accB3[32] = { 0 };

                            dot_i2s_i8_accum_i32_block32_x4_dual(
                                accA0, accA1, accA2, accA3,
                                accB0, accB1, accB2, accB3,
                                waq2T, wbq2T, inDim, j0, jCount,
                                dyA0, dyA1, dyA2, dyA3,
                                dyB0, dyB1, dyB2, dyB3,
                                i0);
                            accum_bwd_block32_gain_to_x_dual_x4(
                                dx0, dx1, dx2, dx3,
                                accA0, accA1, accA2, accA3,
                                accB0, accB1, accB2, accB3,
                                i0,
                                gainA0, gainA1, gainA2, gainA3,
                                gainB0, gainB1, gainB2, gainB3,
                                addToDest);
                        }
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* dyA0 = dyA8Scratch + (tt + 0) * outDim + j0;
                        const int8_t* dyA1 = dyA8Scratch + (tt + 1) * outDim + j0;
                        const int8_t* dyB0 = dyB8Scratch + (tt + 0) * outDim + j0;
                        const int8_t* dyB1 = dyB8Scratch + (tt + 1) * outDim + j0;

                        float* dx0 = dX + (t0 + tt + 0) * inDim;
                        float* dx1 = dX + (t0 + tt + 1) * inDim;

                        const float scaleA0 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 0, jb);
                        const float scaleA1 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 1, jb);
                        const float scaleB0 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 0, jb);
                        const float scaleB1 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 1, jb);
                        const float gainA0 = scaleA0 * blockScaleA;
                        const float gainA1 = scaleA1 * blockScaleA;
                        const float gainB0 = scaleB0 * blockScaleB;
                        const float gainB1 = scaleB1 * blockScaleB;
                        if (useFusedX2) {
                            if (overwriteDest) {
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite(
                                    dx0, dx1,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1,
                                    dyB0, dyB1,
                                    i0,
                                    gainA0, gainA1,
                                    gainB0, gainB1);
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite(
                                    dx0, dx1,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1,
                                    dyB0, dyB1,
                                    i0 + 16,
                                    gainA0, gainA1,
                                    gainB0, gainB1);
                            }
                            else {
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact(
                                    dx0, dx1,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1,
                                    dyB0, dyB1,
                                    i0,
                                    addToDest,
                                    gainA0, gainA1,
                                    gainB0, gainB1);
                                dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact(
                                    dx0, dx1,
                                    waq2T, wbq2T,
                                    inDim, j0, jCount,
                                    dyA0, dyA1,
                                    dyB0, dyB1,
                                    i0 + 16,
                                    addToDest,
                                    gainA0, gainA1,
                                    gainB0, gainB1);
                            }
                        }
                        else {
                            int32_t accA0[32] = { 0 }, accA1[32] = { 0 };
                            int32_t accB0[32] = { 0 }, accB1[32] = { 0 };

                            dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, waq2T, inDim, j0, jCount, dyA0, dyA1, i0);
                            dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, wbq2T, inDim, j0, jCount, dyB0, dyB1, i0);

                            accum_bwd_block32_gain_to_x_dual(dx0, accA0, accB0, i0, gainA0, gainB0, addToDest);
                            accum_bwd_block32_gain_to_x_dual(dx1, accA1, accB1, i0, gainA1, gainB1, addToDest);
                        }
                    }

                    for (; tt < tCount; ++tt) {
                        float* dx = dX + (t0 + tt) * inDim;
                        int32_t accA[32] = { 0 };
                        int32_t accB[32] = { 0 };
                        const float scaleA = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt, jb);
                        const float scaleB = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt, jb);

                        dot_i2s_i8_accum_i32_block32(accA, waq2T, inDim, j0, jCount, dyA8Scratch + tt * outDim + j0, i0);
                        dot_i2s_i8_accum_i32_block32(accB, wbq2T, inDim, j0, jCount, dyB8Scratch + tt * outDim + j0, i0);

                        accum_bwd_block32_gain_to_x_dual(dx, accA, accB, i0, scaleA * blockScaleA, scaleB * blockScaleB, addToDest);
                    }
                }
            }
        }
#else
        quantize_x_tile_i8_tiled_dual(
            dyA8Scratch, dyAScaleScratch, dYa,
            dyB8Scratch, dyBScaleScratch, dYb,
            T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
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

                    const float scaleA0 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 0, jb);
                    const float scaleA1 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 1, jb);
                    const float scaleA2 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 2, jb);
                    const float scaleA3 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 3, jb);
                    const float scaleB0 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 0, jb);
                    const float scaleB1 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 1, jb);
                    const float scaleB2 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 2, jb);
                    const float scaleB3 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 3, jb);

                    if (useFusedX4) {
                        dot_i2s_i8_scale_store_block16_x4_dual_bwd_exact(
                            dx0, dx1, dx2, dx3,
                            Wa.q2T.data(), Wb.q2T.data(),
                            inDim, j0, jCount,
                            dyA0, dyA1, dyA2, dyA3,
                            dyB0, dyB1, dyB2, dyB3,
                            i0,
                            addToDest,
                            scaleA0 * scaleRowA[jb], scaleA1 * scaleRowA[jb], scaleA2 * scaleRowA[jb], scaleA3 * scaleRowA[jb],
                            scaleB0 * scaleRowB[jb], scaleB1 * scaleRowB[jb], scaleB2 * scaleRowB[jb], scaleB3 * scaleRowB[jb]);
                        dot_i2s_i8_scale_store_block16_x4_dual_bwd_exact(
                            dx0, dx1, dx2, dx3,
                            Wa.q2T.data(), Wb.q2T.data(),
                            inDim, j0, jCount,
                            dyA0, dyA1, dyA2, dyA3,
                            dyB0, dyB1, dyB2, dyB3,
                            i0 + 16,
                            addToDest,
                            scaleA0 * scaleRowA[jb], scaleA1 * scaleRowA[jb], scaleA2 * scaleRowA[jb], scaleA3 * scaleRowA[jb],
                            scaleB0 * scaleRowB[jb], scaleB1 * scaleRowB[jb], scaleB2 * scaleRowB[jb], scaleB3 * scaleRowB[jb]);
                    }
                    else {
                        int32_t accA0[32] = { 0 }, accA1[32] = { 0 }, accA2[32] = { 0 }, accA3[32] = { 0 };
                        int32_t accB0[32] = { 0 }, accB1[32] = { 0 }, accB2[32] = { 0 }, accB3[32] = { 0 };

                        dot_i2s_i8_accum_i32_block32_x4_dual(
                            accA0, accA1, accA2, accA3,
                            accB0, accB1, accB2, accB3,
                            Wa.q2T.data(), Wb.q2T.data(), inDim, j0, jCount,
                            dyA0, dyA1, dyA2, dyA3,
                            dyB0, dyB1, dyB2, dyB3,
                            i0);
                        accum_bwd_block32_gain_to_x_dual_x4(
                            dx0, dx1, dx2, dx3,
                            accA0, accA1, accA2, accA3,
                            accB0, accB1, accB2, accB3,
                            i0,
                            scaleA0 * scaleRowA[jb], scaleA1 * scaleRowA[jb], scaleA2 * scaleRowA[jb], scaleA3 * scaleRowA[jb],
                            scaleB0 * scaleRowB[jb], scaleB1 * scaleRowB[jb], scaleB2 * scaleRowB[jb], scaleB3 * scaleRowB[jb],
                            addToDest);
                    }
                }

                for (; tt + 1 < tCount; tt += 2) {
                    const int8_t* dyA0 = dyA8Scratch + (tt + 0) * outDim + j0;
                    const int8_t* dyA1 = dyA8Scratch + (tt + 1) * outDim + j0;
                    const int8_t* dyB0 = dyB8Scratch + (tt + 0) * outDim + j0;
                    const int8_t* dyB1 = dyB8Scratch + (tt + 1) * outDim + j0;

                    float* dx0 = dX + (t0 + tt + 0) * inDim;
                    float* dx1 = dX + (t0 + tt + 1) * inDim;
                    const float scaleA0 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 0, jb);
                    const float scaleA1 = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt + 1, jb);
                    const float scaleB0 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 0, jb);
                    const float scaleB1 = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt + 1, jb);

                    int32_t accA0[32] = { 0 }, accA1[32] = { 0 };
                    int32_t accB0[32] = { 0 }, accB1[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, Wa.q2T.data(), inDim, j0, jCount, dyA0, dyA1, i0);
                    dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, Wb.q2T.data(), inDim, j0, jCount, dyB0, dyB1, i0);

                    accum_bwd_block32_gain_to_x_dual(
                        dx0, accA0, accB0, i0,
                        scaleA0 * scaleRowA[jb], scaleB0 * scaleRowB[jb], addToDest);
                    accum_bwd_block32_gain_to_x_dual(
                        dx1, accA1, accB1, i0,
                        scaleA1 * scaleRowA[jb], scaleB1 * scaleRowB[jb], addToDest);
                }

                for (; tt < tCount; ++tt) {
                    float* dx = dX + (t0 + tt) * inDim;
                    int32_t accA[32] = { 0 };
                    int32_t accB[32] = { 0 };
                    const float scaleA = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt, jb);
                    const float scaleB = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt, jb);

                    dot_i2s_i8_accum_i32_block32(accA, Wa.q2T.data(), inDim, j0, jCount, dyA8Scratch + tt * outDim + j0, i0);
                    dot_i2s_i8_accum_i32_block32(accB, Wb.q2T.data(), inDim, j0, jCount, dyB8Scratch + tt * outDim + j0, i0);

                    accum_bwd_block32_gain_to_x_dual(
                        dx, accA, accB, i0,
                        scaleA * scaleRowA[jb], scaleB * scaleRowB[jb], addToDest);
                }
            }
        }
#endif
    }
}
