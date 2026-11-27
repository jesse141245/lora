#include "bitnet_i2s_matmul.h"
#include "bitnet_i2_s.h"
#include "backend_runtime_policy.h"
#include "env_utils.h"

#include <algorithm>
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

    static inline __m256 make_gain8(float xs, const float* ws8) {
        return _mm256_mul_ps(_mm256_set1_ps(xs), _mm256_loadu_ps(ws8));
    }

    static inline float make_gain1(float xs, const float* ws8) {
        return xs * ws8[0];
    }

    static inline float broadcast_scale_value(const float* ws8) {
        return ws8[0];
    }

    static inline float load_tile_row_scale(
        const float* scaleScratch, std::size_t nBlocks, std::size_t tileRow, std::size_t block)
    {
        return scaleScratch[tileRow * nBlocks + block];
    }

    // ============================================================
    // Forward Helpers
    // ============================================================

    static inline const float* forward_scale_broadcast_row(
        const BitNetMatI2S& W, std::size_t j0, std::size_t nKBlocks)
    {
        if (!W.hasSupportedScaleLayout()) {
            throw std::runtime_error("forward_scale_broadcast_row: unsupported BitNet I2S scale layout");
        }
        (void)nKBlocks;
        return W.forwardScaleBroadcastRow(j0 / W.jq);
    }

    static inline int env_toggle_override(const char* name) {
        const auto value = envutil::tryGetEnvString(name);
        if (!value) return -1;
        const std::string s = envutil::lowerCopy(*value);
        if (envutil::envStringTruthy(s)) return 1;
        if (envutil::envStringFalsy(s)) return 0;
        return -1;
    }

    static inline std::size_t env_size_override(const char* name, std::size_t fallback) {
        const auto value = envutil::tryGetEnvString(name);
        if (!value) return fallback;

        char* end = nullptr;
        const char* raw = value->c_str();
        const unsigned long parsed = std::strtoul(raw, &end, 10);
        if (end == raw || (end && *end != '\0') || parsed == 0) return fallback;
        return static_cast<std::size_t>(parsed);
    }

    static inline std::optional<bool> runtime_policy_bool(
        std::optional<bool> BitNetMatmulRuntimePolicy::*member)
    {
        const BackendRuntimePolicy* policy = currentBackendRuntimePolicy();
        if (!policy) return std::nullopt;
        return policy->bitnetMatmul.*member;
    }

    static inline std::optional<std::size_t> runtime_policy_size(
        std::optional<std::size_t> BitNetMatmulRuntimePolicy::*member)
    {
        const BackendRuntimePolicy* policy = currentBackendRuntimePolicy();
        if (!policy) return std::nullopt;
        return policy->bitnetMatmul.*member;
    }

    static inline bool fused_x4_forward_preferred(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedX4Forward)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_FORWARD");
        if (override >= 0) return override != 0;

        // On the long-sequence tTile=4 branch, the exact fused x4 forward
        // family is materially faster than the older accumulator path.
        // Keep the auto-promotion narrow so shorter-sequence regimes preserve
        // the old defaults unless explicitly opted in.
        return tTile >= 4 && T >= 128;
    }

    static inline bool fused_x2_single_forward_enabled() {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedX2SingleForward)) {
            return *forced;
        }
        static const bool cached = env_toggle_override("QLORA_TEST_BITNET_FUSED_X2_FORWARD_SINGLE") > 0;
        return cached;
    }

    static inline bool dense_x2_forward_accum_preferred(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::denseX2ForwardAccum)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_DENSE_X2_FORWARD_ACCUM");
        if (override >= 0) return override != 0;

        // Keep this confined to the long-sequence tTile=2 branch, where the
        // pair-accumulator path is still the default exact forward route.
        return tTile <= 2 && T >= 128;
    }

    static inline bool fused_x4_backward_preferred(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedX4Backward)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_BACKWARD");
        if (override >= 0) return override != 0;

        // The dual backward x4 path pays off on the long-sequence tTile=4
        // regime, but was not consistently helpful on shorter tiles.
        return tTile >= 4 && T >= 128;
    }

    static inline bool fused_x4_single_backward_preferred(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedX4SingleBackward)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_BACKWARD_SINGLE");
        if (override >= 0) return override != 0;

        // The single backward x4 path only paid off consistently on the same
        // long-sequence tTile=4 branch as the dual x4 path.
        return tTile >= 4 && T >= 128;
    }

    static inline bool fused_x4_single_backward_overwrite_enabled() {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedX4SingleBackwardOverwrite)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_BACKWARD_SINGLE_OVERWRITE");
        if (override >= 0) return override != 0;
        return true;
    }

    static inline bool fused_x4_single_backward_block32_exact_enabled(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedX4SingleBackwardBlock32)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X4_BACKWARD_SINGLE_BLOCK32");
        if (override >= 0) return override != 0;
        return tTile >= 4 && T >= 128;
    }

    static inline bool fused_x2_single_backward_preferred(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedX2SingleBackward)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X2_BACKWARD_SINGLE");
        if (override >= 0) return override != 0;

        // The tTile=2 long-sequence branch is materially faster than the
        // accumulator fallback, but it needs the exact x2 backward family to
        // preserve the good loss curve. Keep the auto-promotion narrow.
        return tTile <= 2 && T >= 128;
    }

    static inline bool fused_x2_dual_backward_preferred(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedX2DualBackward)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_X2_BACKWARD_DUAL");
        if (override >= 0) return override != 0;

        return tTile <= 2 && T >= 128;
    }

    static inline bool parallel_single_backward_quant_enabled(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::parallelSingleBackwardQuant)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_PARALLEL_BWD_QUANT");
        if (override >= 0) return override != 0;

        return tTile >= 4 && T >= 128;
    }

    static inline bool single_team_backward_preferred(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::singleTeamBackward)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_SINGLE_TEAM_BWD_SINGLE");
        if (override >= 0) return override != 0;

        // The long-sequence exact tTile<=2 branch spends a lot of time re-entering
        // OpenMP teams in the backward tile loop. Auto-promote the shared
        // single-team path there, but keep the old default elsewhere.
        return tTile <= 2 && T >= 128;
    }

    static inline bool parallel_multi_backward_quant_enabled() {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::parallelMultiBackwardQuant)) {
            return *forced;
        }
        static const bool cached = env_toggle_override("QLORA_TEST_BITNET_PARALLEL_BWD_QUANT_MULTI") > 0;
        return cached;
    }

    static inline bool master_dual_chunk_quant_preferred() {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::masterDualChunkQuant)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_DUAL_BWD_MASTER_QUANT");
        if (override >= 0) return override != 0;
        return true;
    }

    static inline bool parallel_multi_forward_quant_enabled() {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::parallelMultiForwardQuant)) {
            return *forced;
        }
        static const bool cached = env_toggle_override("QLORA_TEST_BITNET_PARALLEL_FWD_QUANT_MULTI") != 0;
        return cached;
    }

    static inline bool parallel_single_forward_quant_enabled() {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::parallelSingleForwardQuant)) {
            return *forced;
        }
        static const bool cached = env_toggle_override("QLORA_TEST_BITNET_PARALLEL_FWD_QUANT_SINGLE") > 0;
        return cached;
    }

    static inline bool fused_qkv_forward_overwrite_enabled(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedQkvForwardOverwrite)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_QKV_FORWARD_OVERWRITE");
        if (override >= 0) return override != 0;

        // Keep this auto-promotion narrow to the long-sequence tutor branch,
        // where fused exact QKV forward runs with large enough tiles that the
        // first k-block always overwrites fresh rows.
        return tTile >= 4 && T >= 128;
    }

    static inline bool decoded_qkv_backward_triple_enabled() {
        const int override = env_toggle_override("QLORA_TEST_BITNET_DECODED_QKV_BWD_TRIPLE");
        if (override >= 0) return override != 0;
        return false;
    }

    static inline bool fused_qkv_forward_x4_triple_accum_enabled(std::size_t T, std::size_t tTile) {
        if (const std::optional<bool> forced = runtime_policy_bool(&BitNetMatmulRuntimePolicy::fusedQkvForwardTripleX4Accum)) {
            return *forced;
        }
        const int override = env_toggle_override("QLORA_TEST_BITNET_FUSED_QKV_FORWARD_X4_TRIPLE_ACCUM");
        if (override >= 0) return override != 0;

        // Restrict this to the same long-sequence fused exact branch that
        // already uses the x4 path, so same-binary A/B validation stays clean.
        return tTile >= 4 && T >= 128;
    }

    struct SingleForwardKernelPolicy {
        bool useFusedX4 = false;
        bool useFusedX2 = false;
        bool useDenseX2Accum = false;
        bool useParallelQuant = false;

        static SingleForwardKernelPolicy resolve(std::size_t T, std::size_t tTile) {
            SingleForwardKernelPolicy policy{};
            policy.useFusedX4 = fused_x4_forward_preferred(T, tTile);
            policy.useFusedX2 = fused_x2_single_forward_enabled();
            policy.useDenseX2Accum = dense_x2_forward_accum_preferred(T, tTile);
            policy.useParallelQuant = parallel_single_forward_quant_enabled();
            return policy;
        }
    };

    struct MultiForwardKernelPolicy {
        bool useFusedX4 = false;
        bool useDenseX2Accum = false;
        bool useParallelQuant = false;

        static MultiForwardKernelPolicy resolve(std::size_t T, std::size_t tTile) {
            MultiForwardKernelPolicy policy{};
            policy.useFusedX4 = fused_x4_forward_preferred(T, tTile);
            policy.useDenseX2Accum = dense_x2_forward_accum_preferred(T, tTile);
            policy.useParallelQuant = parallel_multi_forward_quant_enabled();
            return policy;
        }
    };

    struct FusedQkvForwardKernelPolicy {
        bool useFusedX4 = false;
        bool useDenseX2Accum = false;
        bool useParallelQuant = false;
        bool useFirstBlockOverwrite = false;
        bool useTripleX4Accum = false;

        static FusedQkvForwardKernelPolicy resolve(std::size_t T, std::size_t tTile) {
            FusedQkvForwardKernelPolicy policy{};
            policy.useFusedX4 = fused_x4_forward_preferred(T, tTile);
            policy.useDenseX2Accum = dense_x2_forward_accum_preferred(T, tTile);
            policy.useParallelQuant = parallel_multi_forward_quant_enabled();
            policy.useFirstBlockOverwrite = fused_qkv_forward_overwrite_enabled(T, tTile);
            policy.useTripleX4Accum = fused_qkv_forward_x4_triple_accum_enabled(T, tTile);
            return policy;
        }
    };

    struct SingleBackwardKernelPolicy {
        bool useFusedX4 = false;
        bool useFusedX2 = false;
        bool useParallelQuant = false;
        bool useSingleTeam = false;
        bool useFusedX4Overwrite = false;
        bool useFusedX4Block32 = false;

        static SingleBackwardKernelPolicy resolve(std::size_t T, std::size_t tTile) {
            SingleBackwardKernelPolicy policy{};
            policy.useFusedX4 = fused_x4_single_backward_preferred(T, tTile);
            policy.useFusedX2 = fused_x2_single_backward_preferred(T, tTile);
            policy.useParallelQuant = parallel_single_backward_quant_enabled(T, tTile);
            policy.useSingleTeam = single_team_backward_preferred(T, tTile);
            policy.useFusedX4Overwrite = fused_x4_single_backward_overwrite_enabled();
            policy.useFusedX4Block32 =
                policy.useFusedX4 &&
                fused_x4_single_backward_block32_exact_enabled(T, tTile);
            return policy;
        }
    };

    struct DualBackwardQuantPolicy {
        bool useChunkQuant = false;
        bool useMasterQuant = false;
        std::size_t quantChunkRows = 0;
        std::size_t workTileRows = 0;

        static DualBackwardQuantPolicy resolve(
            const bool useParallelQuant,
            std::size_t T,
            std::size_t tTile,
            std::size_t activePrefixT,
            std::size_t qCount)
        {
            DualBackwardQuantPolicy policy{};
            policy.useChunkQuant =
                !useParallelQuant &&
                tTile <= 2 &&
                T >= 128 &&
                activePrefixT == T;
            policy.useMasterQuant =
                policy.useChunkQuant &&
                master_dual_chunk_quant_preferred();
            const std::size_t quantChunkOverride =
                runtime_policy_size(&BitNetMatmulRuntimePolicy::dualBackwardQuantChunkRows)
                .value_or(env_size_override("QLORA_TEST_BITNET_DUAL_BWD_QUANT_CHUNK_ROWS", static_cast<std::size_t>(4)));
            policy.quantChunkRows = policy.useChunkQuant ? quantChunkOverride : tTile;
            const std::size_t defaultWorkTileRows =
                (policy.useChunkQuant && tTile == 2 && qCount >= 4)
                ? std::min<std::size_t>(qCount, static_cast<std::size_t>(4))
                : tTile;
            policy.workTileRows = policy.useChunkQuant
                ? std::min(
                    qCount,
                    std::max<std::size_t>(
                        1,
                        runtime_policy_size(&BitNetMatmulRuntimePolicy::dualBackwardWorkTileRows)
                            .value_or(env_size_override(
                                "QLORA_TEST_BITNET_DUAL_BWD_WORK_TILE_ROWS",
                                defaultWorkTileRows))))
                : defaultWorkTileRows;
            return policy;
        }
    };

    struct DualBackwardKernelPolicy {
        bool useFusedX4 = false;
        bool useFusedX2 = false;
        bool useParallelQuant = false;
        bool useSingleTeam = false;
        bool useDenseX2Dual = false;

        static DualBackwardKernelPolicy resolve(std::size_t T, std::size_t tTile) {
            DualBackwardKernelPolicy policy{};
            policy.useFusedX4 = fused_x4_backward_preferred(T, tTile);
            policy.useFusedX2 = fused_x2_dual_backward_preferred(T, tTile);
            policy.useParallelQuant = parallel_multi_backward_quant_enabled();
            policy.useSingleTeam = single_team_backward_preferred(T, tTile);
            policy.useDenseX2Dual = policy.useFusedX2 && tTile <= 2 && T >= 128;
            return policy;
        }
    };

    struct FusedQkvBackwardKernelPolicy {
        bool useParallelQuant = false;
        bool useDualKvQuant = false;

        static FusedQkvBackwardKernelPolicy resolve(
            std::size_t kDim,
            std::size_t vDim,
            bool sharedKvHadamard32)
        {
            FusedQkvBackwardKernelPolicy policy{};
            policy.useParallelQuant = parallel_multi_backward_quant_enabled();
            policy.useDualKvQuant = (kDim == vDim) && sharedKvHadamard32;
            return policy;
        }
    };

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

    static inline void accum_block32_gain_to_y(
        float* __restrict yRow, const int32_t* __restrict blockAcc,
        std::size_t j0, const __m256 gain8,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            const __m256 accF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAcc + off)));
            __m256 y = addToDest
                ? _mm256_fmadd_ps(accF, gain8, _mm256_loadu_ps(yRow + j0 + off))
                : _mm256_mul_ps(accF, gain8);
            _mm256_storeu_ps(yRow + j0 + off, y);
        }
    }

    static inline void accum_block32_gain_to_y_pair(
        float* __restrict yRowA, float* __restrict yRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        std::size_t j0, const __m256 gainA8, const __m256 gainB8,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 yA = addToDest
                ? _mm256_fmadd_ps(accFA, gainA8, _mm256_loadu_ps(yRowA + j0 + off))
                : _mm256_mul_ps(accFA, gainA8);
            __m256 yB = addToDest
                ? _mm256_fmadd_ps(accFB, gainB8, _mm256_loadu_ps(yRowB + j0 + off))
                : _mm256_mul_ps(accFB, gainB8);
            _mm256_storeu_ps(yRowA + j0 + off, yA);
            _mm256_storeu_ps(yRowB + j0 + off, yB);
        }
    }

    static inline void accum_block32_gain_to_y_dual(
        float* __restrict yRowA, float* __restrict yRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        std::size_t j0, const __m256 gainA8, const __m256 gainB8)
    {
        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 yA = _mm256_loadu_ps(yRowA + j0 + off);
            __m256 yB = _mm256_loadu_ps(yRowB + j0 + off);
            _mm256_storeu_ps(yRowA + j0 + off, _mm256_fmadd_ps(accFA, gainA8, yA));
            _mm256_storeu_ps(yRowB + j0 + off, _mm256_fmadd_ps(accFB, gainB8, yB));
        }
    }

    static inline void accum_block32_gain_to_y3(
        float* __restrict qRow, float* __restrict kRow, float* __restrict vRow,
        const int32_t* __restrict accQ, const int32_t* __restrict accK, const int32_t* __restrict accV,
        std::size_t j0,
        const __m256 gainQ, const __m256 gainK, const __m256 gainV,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            const __m256 qF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQ + off)));
            const __m256 kF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accK + off)));
            const __m256 vF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accV + off)));

            __m256 q = addToDest
                ? _mm256_fmadd_ps(qF, gainQ, _mm256_loadu_ps(qRow + j0 + off))
                : _mm256_mul_ps(qF, gainQ);
            __m256 k = addToDest
                ? _mm256_fmadd_ps(kF, gainK, _mm256_loadu_ps(kRow + j0 + off))
                : _mm256_mul_ps(kF, gainK);
            __m256 v = addToDest
                ? _mm256_fmadd_ps(vF, gainV, _mm256_loadu_ps(vRow + j0 + off))
                : _mm256_mul_ps(vF, gainV);

            _mm256_storeu_ps(qRow + j0 + off, q);
            _mm256_storeu_ps(kRow + j0 + off, k);
            _mm256_storeu_ps(vRow + j0 + off, v);
        }
    }

    static inline void accum_block32_gain_to_y3_pair(
        float* __restrict qRowA, float* __restrict kRowA, float* __restrict vRowA,
        float* __restrict qRowB, float* __restrict kRowB, float* __restrict vRowB,
        const int32_t* __restrict accQA, const int32_t* __restrict accKA, const int32_t* __restrict accVA,
        const int32_t* __restrict accQB, const int32_t* __restrict accKB, const int32_t* __restrict accVB,
        std::size_t j0,
        const __m256 gainQA, const __m256 gainKA, const __m256 gainVA,
        const __m256 gainQB, const __m256 gainKB, const __m256 gainVB,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            const __m256 qFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQA + off)));
            const __m256 qFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQB + off)));
            const __m256 kFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKA + off)));
            const __m256 kFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKB + off)));
            const __m256 vFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVA + off)));
            const __m256 vFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVB + off)));
            __m256 qA = addToDest
                ? _mm256_fmadd_ps(qFA, gainQA, _mm256_loadu_ps(qRowA + j0 + off))
                : _mm256_mul_ps(qFA, gainQA);
            __m256 qB = addToDest
                ? _mm256_fmadd_ps(qFB, gainQB, _mm256_loadu_ps(qRowB + j0 + off))
                : _mm256_mul_ps(qFB, gainQB);
            __m256 kA = addToDest
                ? _mm256_fmadd_ps(kFA, gainKA, _mm256_loadu_ps(kRowA + j0 + off))
                : _mm256_mul_ps(kFA, gainKA);
            __m256 kB = addToDest
                ? _mm256_fmadd_ps(kFB, gainKB, _mm256_loadu_ps(kRowB + j0 + off))
                : _mm256_mul_ps(kFB, gainKB);
            __m256 vA = addToDest
                ? _mm256_fmadd_ps(vFA, gainVA, _mm256_loadu_ps(vRowA + j0 + off))
                : _mm256_mul_ps(vFA, gainVA);
            __m256 vB = addToDest
                ? _mm256_fmadd_ps(vFB, gainVB, _mm256_loadu_ps(vRowB + j0 + off))
                : _mm256_mul_ps(vFB, gainVB);

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

    static inline const float* backward_scale_broadcast_row(
        const BitNetMatI2S& W, std::size_t i0, std::size_t nJBlocks)
    {
        if (!W.hasSupportedScaleLayout()) {
            throw std::runtime_error("backward_scale_broadcast_row: unsupported BitNet I2S scale layout");
        }
        (void)nJBlocks;
        return W.backwardScaleBroadcastRow(i0 / W.qk);
    }

    static inline void accum_bwd_block32_gain8_to_x(
        float* __restrict dxRow, const int32_t* __restrict blockAcc,
        std::size_t i0, const __m256 gain8,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            const __m256 accF = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAcc + off)));
            __m256 dx = addToDest
                ? _mm256_fmadd_ps(accF, gain8, _mm256_loadu_ps(dxRow + i0 + off))
                : _mm256_mul_ps(accF, gain8);
            _mm256_storeu_ps(dxRow + i0 + off, dx);
        }
    }

    static inline void accum_bwd_block32_gain8_to_x_pair(
        float* __restrict dxRowA, float* __restrict dxRowB,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        std::size_t i0, const __m256 gainA8, const __m256 gainB8,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 dxA = addToDest
                ? _mm256_fmadd_ps(accFA, gainA8, _mm256_loadu_ps(dxRowA + i0 + off))
                : _mm256_mul_ps(accFA, gainA8);
            __m256 dxB = addToDest
                ? _mm256_fmadd_ps(accFB, gainB8, _mm256_loadu_ps(dxRowB + i0 + off))
                : _mm256_mul_ps(accFB, gainB8);
            _mm256_storeu_ps(dxRowA + i0 + off, dxA);
            _mm256_storeu_ps(dxRowB + i0 + off, dxB);
        }
    }

    static inline void accum_bwd_block32_gain8_to_x_dual(
        float* __restrict dxRow,
        const int32_t* __restrict blockAccA, const int32_t* __restrict blockAccB,
        std::size_t i0, const __m256 gainA8, const __m256 gainB8,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            __m256 dx = _mm256_mul_ps(accFA, gainA8);
            dx = _mm256_fmadd_ps(accFB, gainB8, dx);
            if (addToDest) dx = _mm256_add_ps(dx, _mm256_loadu_ps(dxRow + i0 + off));
            _mm256_storeu_ps(dxRow + i0 + off, dx);
        }
    }

    static inline void accum_bwd_block32_gain8_to_x_triple(
        float* __restrict dxRow,
        const int32_t* __restrict blockAccA,
        const int32_t* __restrict blockAccB,
        const int32_t* __restrict blockAccC,
        std::size_t i0,
        const __m256 gainA8,
        const __m256 gainB8,
        const __m256 gainC8,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            const __m256 accFA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA + off)));
            const __m256 accFB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB + off)));
            const __m256 accFC = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccC + off)));
            __m256 dx = _mm256_mul_ps(accFA, gainA8);
            dx = _mm256_fmadd_ps(accFB, gainB8, dx);
            dx = _mm256_fmadd_ps(accFC, gainC8, dx);
            if (addToDest) dx = _mm256_add_ps(dx, _mm256_loadu_ps(dxRow + i0 + off));
            _mm256_storeu_ps(dxRow + i0 + off, dx);
        }
    }

    static inline void accum_bwd_block32_gain8_to_x_dual_x4(
        float* __restrict dx0, float* __restrict dx1,
        float* __restrict dx2, float* __restrict dx3,
        const int32_t* __restrict blockAccA0, const int32_t* __restrict blockAccA1,
        const int32_t* __restrict blockAccA2, const int32_t* __restrict blockAccA3,
        const int32_t* __restrict blockAccB0, const int32_t* __restrict blockAccB1,
        const int32_t* __restrict blockAccB2, const int32_t* __restrict blockAccB3,
        std::size_t i0,
        const __m256 gainA0v, const __m256 gainA1v, const __m256 gainA2v, const __m256 gainA3v,
        const __m256 gainB0v, const __m256 gainB1v, const __m256 gainB2v, const __m256 gainB3v,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            __m256 dxv0 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA0 + off))),
                gainA0v);
            dxv0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB0 + off))),
                gainB0v,
                dxv0);
            if (addToDest) dxv0 = _mm256_add_ps(dxv0, _mm256_loadu_ps(dx0 + i0 + off));
            _mm256_storeu_ps(dx0 + i0 + off, dxv0);

            __m256 dxv1 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA1 + off))),
                gainA1v);
            dxv1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB1 + off))),
                gainB1v,
                dxv1);
            if (addToDest) dxv1 = _mm256_add_ps(dxv1, _mm256_loadu_ps(dx1 + i0 + off));
            _mm256_storeu_ps(dx1 + i0 + off, dxv1);

            __m256 dxv2 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA2 + off))),
                gainA2v);
            dxv2 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB2 + off))),
                gainB2v,
                dxv2);
            if (addToDest) dxv2 = _mm256_add_ps(dxv2, _mm256_loadu_ps(dx2 + i0 + off));
            _mm256_storeu_ps(dx2 + i0 + off, dxv2);

            __m256 dxv3 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA3 + off))),
                gainA3v);
            dxv3 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB3 + off))),
                gainB3v,
                dxv3);
            if (addToDest) dxv3 = _mm256_add_ps(dxv3, _mm256_loadu_ps(dx3 + i0 + off));
            _mm256_storeu_ps(dx3 + i0 + off, dxv3);
        }
    }

    static inline void accum_bwd_block32_gain8_to_x_triple_x4(
        float* __restrict dx0, float* __restrict dx1,
        float* __restrict dx2, float* __restrict dx3,
        const int32_t* __restrict blockAccA0, const int32_t* __restrict blockAccA1,
        const int32_t* __restrict blockAccA2, const int32_t* __restrict blockAccA3,
        const int32_t* __restrict blockAccB0, const int32_t* __restrict blockAccB1,
        const int32_t* __restrict blockAccB2, const int32_t* __restrict blockAccB3,
        const int32_t* __restrict blockAccC0, const int32_t* __restrict blockAccC1,
        const int32_t* __restrict blockAccC2, const int32_t* __restrict blockAccC3,
        std::size_t i0,
        const __m256 gainA0v, const __m256 gainA1v, const __m256 gainA2v, const __m256 gainA3v,
        const __m256 gainB0v, const __m256 gainB1v, const __m256 gainB2v, const __m256 gainB3v,
        const __m256 gainC0v, const __m256 gainC1v, const __m256 gainC2v, const __m256 gainC3v,
        bool addToDest)
    {
        for (int off = 0; off < 32; off += 8) {
            __m256 dxv0 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA0 + off))),
                gainA0v);
            dxv0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB0 + off))),
                gainB0v,
                dxv0);
            dxv0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccC0 + off))),
                gainC0v,
                dxv0);
            if (addToDest) dxv0 = _mm256_add_ps(dxv0, _mm256_loadu_ps(dx0 + i0 + off));
            _mm256_storeu_ps(dx0 + i0 + off, dxv0);

            __m256 dxv1 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA1 + off))),
                gainA1v);
            dxv1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB1 + off))),
                gainB1v,
                dxv1);
            dxv1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccC1 + off))),
                gainC1v,
                dxv1);
            if (addToDest) dxv1 = _mm256_add_ps(dxv1, _mm256_loadu_ps(dx1 + i0 + off));
            _mm256_storeu_ps(dx1 + i0 + off, dxv1);

            __m256 dxv2 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA2 + off))),
                gainA2v);
            dxv2 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB2 + off))),
                gainB2v,
                dxv2);
            dxv2 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccC2 + off))),
                gainC2v,
                dxv2);
            if (addToDest) dxv2 = _mm256_add_ps(dxv2, _mm256_loadu_ps(dx2 + i0 + off));
            _mm256_storeu_ps(dx2 + i0 + off, dxv2);

            __m256 dxv3 = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccA3 + off))),
                gainA3v);
            dxv3 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccB3 + off))),
                gainB3v,
                dxv3);
            dxv3 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blockAccC3 + off))),
                gainC3v,
                dxv3);
            if (addToDest) dxv3 = _mm256_add_ps(dxv3, _mm256_loadu_ps(dx3 + i0 + off));
            _mm256_storeu_ps(dx3 + i0 + off, dxv3);
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
    const SingleForwardKernelPolicy kernelPolicy = SingleForwardKernelPolicy::resolve(T, tTile);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
            const std::size_t tCount = std::min(tTile, T - t0);
            if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;
#if defined(_OPENMP)
            if (kernelPolicy.useParallelQuant) {
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
                const float* const scaleRow8 = forward_scale_broadcast_row(W, j0, nKBlocks);

                for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                    const std::size_t hCount = std::min(hBlock, inDim - h0);
                    const std::size_t kb = h0 / hBlock;
                    const float* const ws8 = scaleRow8 + kb * BitNetMatI2S::scaleBroadcastWidth;

                    std::size_t tt = 0;
                    for (; tt + 3 < tCount; tt += 4) {
                        const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                        const int8_t* xC = x8 + (tt + 2) * inDim + h0; const int8_t* xD = x8 + (tt + 3) * inDim + h0;

                        float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                        float* yC = Y + (t0 + tt + 2) * outDim; float* yD = Y + (t0 + tt + 3) * outDim;

                        if (kernelPolicy.useFusedX4) {
                            const bool addToDest = (kb != 0);
                            const float gainA = make_gain1(load_tile_row_scale(xScale, nKBlocks, tt + 0, kb), ws8);
                            const float gainB = make_gain1(load_tile_row_scale(xScale, nKBlocks, tt + 1, kb), ws8);
                            const float gainC = make_gain1(load_tile_row_scale(xScale, nKBlocks, tt + 2, kb), ws8);
                            const float gainD = make_gain1(load_tile_row_scale(xScale, nKBlocks, tt + 3, kb), ws8);

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
                            const __m256 gainA8 = make_gain8(xsA, ws8);
                            const __m256 gainB8 = make_gain8(xsB, ws8);
                            const __m256 gainC8 = make_gain8(xsC, ws8);
                            const __m256 gainD8 = make_gain8(xsD, ws8);

                            dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                            accum_block32_gain_to_y_pair(yA, yB, aA, aB, j0, gainA8, gainB8, true);
                            accum_block32_gain_to_y_pair(yC, yD, aC, aD, j0, gainC8, gainD8, true);
                        }
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                        float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                        const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                        const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);

                        if (kernelPolicy.useFusedX2) {
                            const float gainA = make_gain1(xsA, ws8);
                            const float gainB = make_gain1(xsB, ws8);
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
                            const __m256 gainA8 = make_gain8(xsA, ws8);
                            const __m256 gainB8 = make_gain8(xsB, ws8);
                            if (kernelPolicy.useDenseX2Accum) {
                                dot_i2s_i8_accum_i32_block32_x2_dense(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            } else {
                                dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            }
                            accum_block32_gain_to_y_pair(yA, yB, aA, aB, j0, gainA8, gainB8, true);
                        }
                    }

                    for (; tt < tCount; ++tt) {
                        float* yRow = Y + (t0 + tt) * outDim;
                        int32_t blockAcc[32] = { 0 };
                        const float xs = load_tile_row_scale(xScale, nKBlocks, tt, kb);
                        const __m256 gain8 = make_gain8(xs, ws8);
                        dot_i2s_i8_accum_i32_block32(blockAcc, W.q2.data(), outDim, h0, hCount, x8 + tt * inDim + h0, j0);
                        accum_block32_gain_to_y(yRow, blockAcc, j0, gain8, true);
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
    const bool useDecodedFused =
        !Wq.q8.empty() && !Wk.q8.empty() && !Wv.q8.empty();
    const FusedQkvForwardKernelPolicy kernelPolicy = FusedQkvForwardKernelPolicy::resolve(T, tTile);

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
#if defined(_OPENMP)
        if (kernelPolicy.useParallelQuant) {
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
                    const float* const scaleRow8 = forward_scale_broadcast_row(W, j0, nKBlocks);
                    for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                        const std::size_t hCount = std::min(hBlock, inDim - h0);
                        const std::size_t kb = h0 / hBlock;
                        const float* const ws8 = scaleRow8 + kb * BitNetMatI2S::scaleBroadcastWidth;

                        std::size_t tt = 0;
                        for (; tt + 3 < tCount; tt += 4) {
                            const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                            const int8_t* xC = x8 + (tt + 2) * inDim + h0; const int8_t* xD = x8 + (tt + 3) * inDim + h0;
                            float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                            float* yC = Y + (t0 + tt + 2) * outDim; float* yD = Y + (t0 + tt + 3) * outDim;
                            if (kernelPolicy.useFusedX4 && W.q8.empty()) {
                                const bool addToDest = !kernelPolicy.useFirstBlockOverwrite || (kb != 0);
                                const float gainA = make_gain1(load_tile_row_scale(xScale, nKBlocks, tt + 0, kb), ws8);
                                const float gainB = make_gain1(load_tile_row_scale(xScale, nKBlocks, tt + 1, kb), ws8);
                                const float gainC = make_gain1(load_tile_row_scale(xScale, nKBlocks, tt + 2, kb), ws8);
                                const float gainD = make_gain1(load_tile_row_scale(xScale, nKBlocks, tt + 3, kb), ws8);

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
                                const __m256 gainA8 = make_gain8(xsA, ws8);
                                const __m256 gainB8 = make_gain8(xsB, ws8);
                                const __m256 gainC8 = make_gain8(xsC, ws8);
                                const __m256 gainD8 = make_gain8(xsD, ws8);

                                if (!W.q8.empty()) {
                                    dot_ternary_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q8.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                                }
                                else {
                                    dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                                }
                                accum_block32_gain_to_y_pair(yA, yB, aA, aB, j0, gainA8, gainB8, true);
                                accum_block32_gain_to_y_pair(yC, yD, aC, aD, j0, gainC8, gainD8, true);
                            }
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                            float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                            const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);
                            const bool addToDest = !kernelPolicy.useFirstBlockOverwrite || (kb != 0);
                            const __m256 gainA8 = make_gain8(xsA, ws8);
                            const __m256 gainB8 = make_gain8(xsB, ws8);

                            if (!W.q8.empty()) {
                                dot_ternary_i8_accum_i32_block32_x2(aA, aB, W.q8.data(), outDim, h0, hCount, xA, xB, j0);
                            }
                            else if (kernelPolicy.useDenseX2Accum) {
                                dot_i2s_i8_accum_i32_block32_x2_dense(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            } else {
                                dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            }
                            accum_block32_gain_to_y_pair(yA, yB, aA, aB, j0, gainA8, gainB8, addToDest);
                        }

                        for (; tt < tCount; ++tt) {
                            float* yRow = Y + (t0 + tt) * outDim;
                            int32_t acc[32] = { 0 };
                            const float xs = load_tile_row_scale(xScale, nKBlocks, tt, kb);
                            const bool addToDest = !kernelPolicy.useFirstBlockOverwrite || (kb != 0);
                            const __m256 gain8 = make_gain8(xs, ws8);
                            if (!W.q8.empty()) {
                                dot_ternary_i8_accum_i32_block32(acc, W.q8.data(), outDim, h0, hCount, x8 + tt * inDim + h0, j0);
                            }
                            else {
                                dot_i2s_i8_accum_i32_block32(acc, W.q2.data(), outDim, h0, hCount, x8 + tt * inDim + h0, j0);
                            }
                            accum_block32_gain_to_y(yRow, acc, j0, gain8, addToDest);
                        }
                    }
                }
            };

#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
            for (int j0_i = 0; j0_i < static_cast<int>(fusedCols); j0_i += 32) {
                const std::size_t j0 = static_cast<std::size_t>(j0_i);
                const float* const scaleRowQ8 = forward_scale_broadcast_row(Wq, j0, nKBlocks);
                const float* const scaleRowK8 = forward_scale_broadcast_row(Wk, j0, nKBlocks);
                const float* const scaleRowV8 = forward_scale_broadcast_row(Wv, j0, nKBlocks);
                for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                    const std::size_t hCount = std::min(hBlock, inDim - h0);
                    const std::size_t kb = h0 / hBlock;
                    const bool addToDest = !kernelPolicy.useFirstBlockOverwrite || (kb != 0);
                    const float* const wsQ8 = scaleRowQ8 + kb * BitNetMatI2S::scaleBroadcastWidth;
                    const float* const wsK8 = scaleRowK8 + kb * BitNetMatI2S::scaleBroadcastWidth;
                    const float* const wsV8 = scaleRowV8 + kb * BitNetMatI2S::scaleBroadcastWidth;

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

                        if (useDecodedFused) {
                            dot_ternary_i8_accum_i32_block32_x4(qQA, qQB, qQC, qQD, Wq.q8.data(), qDim, h0, hCount, xA, xB, xC, xD, j0);
                            dot_ternary_i8_accum_i32_block32_x4(qKA, qKB, qKC, qKD, Wk.q8.data(), kDim, h0, hCount, xA, xB, xC, xD, j0);
                            dot_ternary_i8_accum_i32_block32_x4(qVA, qVB, qVC, qVD, Wv.q8.data(), vDim, h0, hCount, xA, xB, xC, xD, j0);
                        }
                        else if (kernelPolicy.useTripleX4Accum) {
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

                        const __m256 gainQA = make_gain8(xsA, wsQ8), gainQB = make_gain8(xsB, wsQ8);
                        const __m256 gainKA = make_gain8(xsA, wsK8), gainKB = make_gain8(xsB, wsK8);
                        const __m256 gainVA = make_gain8(xsA, wsV8), gainVB = make_gain8(xsB, wsV8);
                        const __m256 gainQC = make_gain8(xsC, wsQ8), gainQD = make_gain8(xsD, wsQ8);
                        const __m256 gainKC = make_gain8(xsC, wsK8), gainKD = make_gain8(xsD, wsK8);
                        const __m256 gainVC = make_gain8(xsC, wsV8), gainVD = make_gain8(xsD, wsV8);

                        accum_block32_gain_to_y3_pair(qA, kA, vA, qB, kB, vB, qQA, qKA, qVA, qQB, qKB, qVB, j0, gainQA, gainKA, gainVA, gainQB, gainKB, gainVB, addToDest);
                        accum_block32_gain_to_y3_pair(qC, kC, vC, qD, kD, vD, qQC, qKC, qVC, qQD, qKD, qVD, j0, gainQC, gainKC, gainVC, gainQD, gainKD, gainVD, addToDest);
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* xA = x8 + (tt + 0) * inDim + h0; const int8_t* xB = x8 + (tt + 1) * inDim + h0;
                        float* qA = Q + (t0 + tt + 0) * qDim; float* kA = K + (t0 + tt + 0) * kDim; float* vA = V + (t0 + tt + 0) * vDim;
                        float* qB = Q + (t0 + tt + 1) * qDim; float* kB = K + (t0 + tt + 1) * kDim; float* vB = V + (t0 + tt + 1) * vDim;
                        int32_t qQA[32] = { 0 }, qKA[32] = { 0 }, qVA[32] = { 0 }, qQB[32] = { 0 }, qKB[32] = { 0 }, qVB[32] = { 0 };
                        const float xsA = load_tile_row_scale(xScale, nKBlocks, tt + 0, kb);
                        const float xsB = load_tile_row_scale(xScale, nKBlocks, tt + 1, kb);

                        if (useDecodedFused) {
                            dot_ternary_i8_accum_i32_block32_x2(qQA, qQB, Wq.q8.data(), qDim, h0, hCount, xA, xB, j0);
                            dot_ternary_i8_accum_i32_block32_x2(qKA, qKB, Wk.q8.data(), kDim, h0, hCount, xA, xB, j0);
                            dot_ternary_i8_accum_i32_block32_x2(qVA, qVB, Wv.q8.data(), vDim, h0, hCount, xA, xB, j0);
                        }
                        else if (kernelPolicy.useDenseX2Accum) {
                            dot_i2s_i8_accum_i32_block32_x2_dense(qQA, qQB, Wq.q2.data(), qDim, h0, hCount, xA, xB, j0);
                            dot_i2s_i8_accum_i32_block32_x2_dense(qKA, qKB, Wk.q2.data(), kDim, h0, hCount, xA, xB, j0);
                            dot_i2s_i8_accum_i32_block32_x2_dense(qVA, qVB, Wv.q2.data(), vDim, h0, hCount, xA, xB, j0);
                        } else {
                            dot_i2s_i8_accum_i32_block32_x2(qQA, qQB, Wq.q2.data(), qDim, h0, hCount, xA, xB, j0);
                            dot_i2s_i8_accum_i32_block32_x2(qKA, qKB, Wk.q2.data(), kDim, h0, hCount, xA, xB, j0);
                            dot_i2s_i8_accum_i32_block32_x2(qVA, qVB, Wv.q2.data(), vDim, h0, hCount, xA, xB, j0);
                        }

                        const __m256 gainQA = make_gain8(xsA, wsQ8), gainQB = make_gain8(xsB, wsQ8);
                        const __m256 gainKA = make_gain8(xsA, wsK8), gainKB = make_gain8(xsB, wsK8);
                        const __m256 gainVA = make_gain8(xsA, wsV8), gainVB = make_gain8(xsB, wsV8);

                        accum_block32_gain_to_y3_pair(qA, kA, vA, qB, kB, vB, qQA, qKA, qVA, qQB, qKB, qVB, j0, gainQA, gainKA, gainVA, gainQB, gainKB, gainVB, addToDest);
                    }

                    for (; tt < tCount; ++tt) {
                        const int8_t* xRow8 = x8 + tt * inDim + h0;
                        float* qRow = Q + (t0 + tt) * qDim; float* kRow = K + (t0 + tt) * kDim; float* vRow = V + (t0 + tt) * vDim;
                        int32_t accQ[32] = { 0 }, accK[32] = { 0 }, accV[32] = { 0 };
                        const float xs = load_tile_row_scale(xScale, nKBlocks, tt, kb);

                        if (useDecodedFused) {
                            dot_ternary_i8_accum_i32_block32(accQ, Wq.q8.data(), qDim, h0, hCount, xRow8, j0);
                            dot_ternary_i8_accum_i32_block32(accK, Wk.q8.data(), kDim, h0, hCount, xRow8, j0);
                            dot_ternary_i8_accum_i32_block32(accV, Wv.q8.data(), vDim, h0, hCount, xRow8, j0);
                        }
                        else {
                            dot_i2s_i8_accum_i32_block32(accQ, Wq.q2.data(), qDim, h0, hCount, xRow8, j0);
                            dot_i2s_i8_accum_i32_block32(accK, Wk.q2.data(), kDim, h0, hCount, xRow8, j0);
                            dot_i2s_i8_accum_i32_block32(accV, Wv.q2.data(), vDim, h0, hCount, xRow8, j0);
                        }

                        const __m256 gainQ = make_gain8(xs, wsQ8);
                        const __m256 gainK = make_gain8(xs, wsK8);
                        const __m256 gainV = make_gain8(xs, wsV8);
                        accum_block32_gain_to_y3(qRow, kRow, vRow, accQ, accK, accV, j0, gainQ, gainK, gainV, addToDest);
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
    const MultiForwardKernelPolicy kernelPolicy = MultiForwardKernelPolicy::resolve(T, tTile);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
            const std::size_t tCount = std::min(tTile, T - t0);
            if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;
#if defined(_OPENMP)
            if (kernelPolicy.useParallelQuant) {
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
                const float* const scaleRowA8 = forward_scale_broadcast_row(Wa, j0, nKBlocks);
                const float* const scaleRowB8 = forward_scale_broadcast_row(Wb, j0, nKBlocks);

                for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                    const std::size_t hCount = std::min(hBlock, inDim - h0);
                    const std::size_t kb = h0 / hBlock;
                    const float* const wsA8 = scaleRowA8 + kb * BitNetMatI2S::scaleBroadcastWidth;
                    const float* const wsB8 = scaleRowB8 + kb * BitNetMatI2S::scaleBroadcastWidth;

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
                        if (kernelPolicy.useFusedX4) {
                            const bool addToDest = (kb != 0);
                            const float blockScaleA = broadcast_scale_value(wsA8);
                            const float blockScaleB = broadcast_scale_value(wsB8);
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

                            accum_block32_gain_to_y_dual(yA0, yB0, accA0, accB0, j0, make_gain8(xs0, wsA8), make_gain8(xs0, wsB8));
                            accum_block32_gain_to_y_dual(yA1, yB1, accA1, accB1, j0, make_gain8(xs1, wsA8), make_gain8(xs1, wsB8));
                            accum_block32_gain_to_y_dual(yA2, yB2, accA2, accB2, j0, make_gain8(xs2, wsA8), make_gain8(xs2, wsB8));
                            accum_block32_gain_to_y_dual(yA3, yB3, accA3, accB3, j0, make_gain8(xs3, wsA8), make_gain8(xs3, wsB8));
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
                        if (kernelPolicy.useDenseX2Accum) {
                            dot_i2s_i8_accum_i32_block32_x2_dense(accA0, accA1, Wa.q2.data(), outDim, h0, hCount, x0, x1, j0);
                            dot_i2s_i8_accum_i32_block32_x2_dense(accB0, accB1, Wb.q2.data(), outDim, h0, hCount, x0, x1, j0);
                        } else {
                            dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, Wa.q2.data(), outDim, h0, hCount, x0, x1, j0);
                            dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, Wb.q2.data(), outDim, h0, hCount, x0, x1, j0);
                        }

                        accum_block32_gain_to_y_dual(yA0, yB0, accA0, accB0, j0, make_gain8(xs0, wsA8), make_gain8(xs0, wsB8));
                        accum_block32_gain_to_y_dual(yA1, yB1, accA1, accB1, j0, make_gain8(xs1, wsA8), make_gain8(xs1, wsB8));
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

                        accum_block32_gain_to_y_dual(yA, yB, accA, accB, j0, make_gain8(xs, wsA8), make_gain8(xs, wsB8));
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
    const MultiForwardKernelPolicy kernelPolicy = MultiForwardKernelPolicy::resolve(T, tTile);
    const int fusedColsInt = static_cast<int>(fusedCols);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
            const std::size_t tCount = std::min(tTile, T - t0);
            if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;
#if defined(_OPENMP)
            if (kernelPolicy.useParallelQuant) {
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
                    const float* const scaleRow8 = forward_scale_broadcast_row(W, j0, nKBlocks);
                    for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                        const std::size_t hCount = std::min(hBlock, inDim - h0);
                        const std::size_t kb = h0 / hBlock;
                        const float* const ws8 = scaleRow8 + kb * BitNetMatI2S::scaleBroadcastWidth;
                        std::size_t tt = 0;

                        for (; tt + 3 < tCount; tt += 4) {
                            const int8_t* xA = x8Scratch + (tt + 0) * inDim + h0; const int8_t* xB = x8Scratch + (tt + 1) * inDim + h0;
                            const int8_t* xC = x8Scratch + (tt + 2) * inDim + h0; const int8_t* xD = x8Scratch + (tt + 3) * inDim + h0;
                            float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                            float* yC = Y + (t0 + tt + 2) * outDim; float* yD = Y + (t0 + tt + 3) * outDim;
                            if (kernelPolicy.useFusedX4) {
                                const bool addToDest = (kb != 0);
                                const float ws = broadcast_scale_value(ws8);
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
                                const __m256 gainA8 = make_gain8(xsA, ws8);
                                const __m256 gainB8 = make_gain8(xsB, ws8);
                                const __m256 gainC8 = make_gain8(xsC, ws8);
                                const __m256 gainD8 = make_gain8(xsD, ws8);

                                dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2.data(), outDim, h0, hCount, xA, xB, xC, xD, j0);
                                accum_block32_gain_to_y_pair(yA, yB, aA, aB, j0, gainA8, gainB8, true);
                                accum_block32_gain_to_y_pair(yC, yD, aC, aD, j0, gainC8, gainD8, true);
                            }
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* xA = x8Scratch + (tt + 0) * inDim + h0; const int8_t* xB = x8Scratch + (tt + 1) * inDim + h0;
                            float* yA = Y + (t0 + tt + 0) * outDim; float* yB = Y + (t0 + tt + 1) * outDim;
                            const float xsA = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 0, kb);
                            const float xsB = load_tile_row_scale(xScaleScratch, nKBlocks, tt + 1, kb);
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            const __m256 gainA8 = make_gain8(xsA, ws8);
                            const __m256 gainB8 = make_gain8(xsB, ws8);
                            if (kernelPolicy.useDenseX2Accum) {
                                dot_i2s_i8_accum_i32_block32_x2_dense(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            }
                            else {
                                dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2.data(), outDim, h0, hCount, xA, xB, j0);
                            }
                            accum_block32_gain_to_y_pair(yA, yB, aA, aB, j0, gainA8, gainB8, true);
                        }

                        for (; tt < tCount; ++tt) {
                            float* yRow = Y + (t0 + tt) * outDim;
                            int32_t blockAcc[32] = { 0 };
                            const float xs = load_tile_row_scale(xScaleScratch, nKBlocks, tt, kb);
                            const __m256 gain8 = make_gain8(xs, ws8);
                            dot_i2s_i8_accum_i32_block32(blockAcc, W.q2.data(), outDim, h0, hCount, x8Scratch + tt * inDim + h0, j0);
                            accum_block32_gain_to_y(yRow, blockAcc, j0, gain8, true);
                        }
                    }
                }
            };

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int j0_i = 0; j0_i < fusedColsInt; j0_i += 32) {
                const std::size_t j0 = static_cast<std::size_t>(j0_i);
                const float* const scaleRowA8 = forward_scale_broadcast_row(Wa, j0, nKBlocks);
                const float* const scaleRowB8 = forward_scale_broadcast_row(Wb, j0, nKBlocks);

                for (std::size_t h0 = 0; h0 < inDim; h0 += hBlock) {
                    const std::size_t hCount = std::min(hBlock, inDim - h0);
                    const std::size_t kb = h0 / hBlock;
                    const float* const wsA8 = scaleRowA8 + kb * BitNetMatI2S::scaleBroadcastWidth;
                    const float* const wsB8 = scaleRowB8 + kb * BitNetMatI2S::scaleBroadcastWidth;
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

                        if (kernelPolicy.useFusedX4) {
                            const bool addToDest = (kb != 0);
                            const float blockScaleA = broadcast_scale_value(wsA8);
                            const float blockScaleB = broadcast_scale_value(wsB8);
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
                            accum_block32_gain_to_y_dual(yA0, yB0, accA0, accB0, j0, make_gain8(xs0, wsA8), make_gain8(xs0, wsB8));
                            accum_block32_gain_to_y_dual(yA1, yB1, accA1, accB1, j0, make_gain8(xs1, wsA8), make_gain8(xs1, wsB8));
                            accum_block32_gain_to_y_dual(yA2, yB2, accA2, accB2, j0, make_gain8(xs2, wsA8), make_gain8(xs2, wsB8));
                            accum_block32_gain_to_y_dual(yA3, yB3, accA3, accB3, j0, make_gain8(xs3, wsA8), make_gain8(xs3, wsB8));
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
                        if (kernelPolicy.useDenseX2Accum) {
                            dot_i2s_i8_accum_i32_block32_x2_dense(accA0, accA1, Wa.q2.data(), fusedCols, h0, hCount, x0, x1, j0);
                            dot_i2s_i8_accum_i32_block32_x2_dense(accB0, accB1, Wb.q2.data(), fusedCols, h0, hCount, x0, x1, j0);
                        }
                        else {
                            dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, Wa.q2.data(), fusedCols, h0, hCount, x0, x1, j0);
                            dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, Wb.q2.data(), fusedCols, h0, hCount, x0, x1, j0);
                        }

                        accum_block32_gain_to_y_dual(yA0, yB0, accA0, accB0, j0, make_gain8(xs0, wsA8), make_gain8(xs0, wsB8));
                        accum_block32_gain_to_y_dual(yA1, yB1, accA1, accB1, j0, make_gain8(xs1, wsA8), make_gain8(xs1, wsB8));
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

                        accum_block32_gain_to_y_dual(yA, yB, accA, accB, j0, make_gain8(xs, wsA8), make_gain8(xs, wsB8));
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
    const SingleBackwardKernelPolicy kernelPolicy =
        SingleBackwardKernelPolicy::resolve(T, tTile);

#if defined(_OPENMP)
    if (kernelPolicy.useSingleTeam) {
#pragma omp parallel
        {
            for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
                const std::size_t tCount = std::min(tTile, T - t0);
                if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;
                const int inDim_int = static_cast<int>(inDim);

                if (kernelPolicy.useParallelQuant) {
                    quantize_x_tile_i8_tiled_omp_for(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
                }
                else {
#pragma omp single
                    quantize_x_tile_i8_tiled(dy8Scratch, dyScaleScratch, dY, T, outDim, t0, tCount, kRowsPerQuantScale, hBlock, W.hadamard32);
                }

#pragma omp for schedule(static)
                for (int i0_i = 0; i0_i < inDim_int; i0_i += 32) {
                    std::size_t i0 = static_cast<std::size_t>(i0_i);
                    const float* const scaleRow8 = backward_scale_broadcast_row(W, i0, nJBlocks);
                    const uint8_t* const wq2T = W.q2T.data();
                    const int8_t* const wt8 = W.qT8.empty() ? nullptr : W.qT8.data();

                    for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                        const std::size_t jCount = std::min(hBlock, outDim - j0);
                        const std::size_t jb = j0 / hBlock;
                        const bool addToDest = (jb != 0);
                        const bool overwriteDest = destStartsZero && !addToDest;
                        const float* const ws8 = scaleRow8 + jb * BitNetMatI2S::scaleBroadcastWidth;
                        const float blockScale = broadcast_scale_value(ws8);

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
                            if (kernelPolicy.useFusedX4) {
                                if (kernelPolicy.useFusedX4Block32) {
                                    if (wt8 && overwriteDest && kernelPolicy.useFusedX4Overwrite) {
                                        dot_ternary_i8_scale_store_block32_x4_bwd_exact_overwrite(
                                            dxA, dxB, dxC, dxD,
                                            wt8,
                                            inDim, j0, jCount,
                                            dyA, dyB, dyC, dyD,
                                            i0,
                                            gainA, gainB, gainC, gainD);
                                    }
                                    else if (wt8) {
                                        dot_ternary_i8_scale_store_block32_x4_bwd_exact(
                                            dxA, dxB, dxC, dxD,
                                            wt8,
                                            inDim, j0, jCount,
                                            dyA, dyB, dyC, dyD,
                                            i0,
                                            gainA, gainB, gainC, gainD);
                                    }
                                    else if (overwriteDest && kernelPolicy.useFusedX4Overwrite) {
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
                                else if (overwriteDest && kernelPolicy.useFusedX4Overwrite) {
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
                            else if (kernelPolicy.useFusedX2) {
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
                                accum_bwd_block32_gain8_to_x_pair(dxA, dxB, aA, aB, i0, make_gain8(dyScaleA, ws8), make_gain8(dyScaleB, ws8), addToDest);
                                accum_bwd_block32_gain8_to_x_pair(dxC, dxD, aC, aD, i0, make_gain8(dyScaleC, ws8), make_gain8(dyScaleD, ws8), addToDest);
                            }
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                            float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                            const float dyScaleA = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 0, jb);
                            const float dyScaleB = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 1, jb);
                            const float gainA = dyScaleA * blockScale;
                            const float gainB = dyScaleB * blockScale;
                            if (kernelPolicy.useFusedX2) {
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
                                accum_bwd_block32_gain8_to_x_pair(dxA, dxB, aA, aB, i0, make_gain8(dyScaleA, ws8), make_gain8(dyScaleB, ws8), addToDest);
                            }
                        }

                        for (; tt < tCount; ++tt) {
                            float* dxRow = dX + (t0 + tt) * inDim;
                            int32_t acc[32] = { 0 };
                            const float dyScale = load_tile_row_scale(dyScaleScratch, nJBlocks, tt, jb);
                            dot_i2s_i8_accum_i32_block32(acc, wq2T, inDim, j0, jCount, dy8Scratch + tt * outDim + j0, i0);
                            accum_bwd_block32_gain8_to_x(dxRow, acc, i0, make_gain8(dyScale, ws8), addToDest);
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
            if (kernelPolicy.useParallelQuant) {
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
                const float* const scaleRow8 = backward_scale_broadcast_row(W, i0, nJBlocks);
                const uint8_t* const wq2T = W.q2T.data();
                const int8_t* const wt8 = W.qT8.empty() ? nullptr : W.qT8.data();

                for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                    const std::size_t jCount = std::min(hBlock, outDim - j0);
                    const std::size_t jb = j0 / hBlock;
                    const bool addToDest = (jb != 0);
                    const bool overwriteDest = destStartsZero && !addToDest;
                    const float* const ws8 = scaleRow8 + jb * BitNetMatI2S::scaleBroadcastWidth;
                    const float blockScale = broadcast_scale_value(ws8);
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
                        if (kernelPolicy.useFusedX4) {
                            if (kernelPolicy.useFusedX4Block32) {
                                if (wt8 && overwriteDest && kernelPolicy.useFusedX4Overwrite) {
                                    dot_ternary_i8_scale_store_block32_x4_bwd_exact_overwrite(
                                        dxA, dxB, dxC, dxD,
                                        wt8,
                                        inDim, j0, jCount,
                                        dyA, dyB, dyC, dyD,
                                        i0,
                                        gainA, gainB, gainC, gainD);
                                }
                                else if (wt8) {
                                    dot_ternary_i8_scale_store_block32_x4_bwd_exact(
                                        dxA, dxB, dxC, dxD,
                                        wt8,
                                        inDim, j0, jCount,
                                        dyA, dyB, dyC, dyD,
                                        i0,
                                        gainA, gainB, gainC, gainD);
                                }
                                else if (overwriteDest && kernelPolicy.useFusedX4Overwrite) {
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
                            else if (overwriteDest && kernelPolicy.useFusedX4Overwrite) {
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
                        else if (kernelPolicy.useFusedX2) {
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
                            accum_bwd_block32_gain8_to_x_pair(dxA, dxB, aA, aB, i0, make_gain8(dyScaleA, ws8), make_gain8(dyScaleB, ws8), addToDest);
                            accum_bwd_block32_gain8_to_x_pair(dxC, dxD, aC, aD, i0, make_gain8(dyScaleC, ws8), make_gain8(dyScaleD, ws8), addToDest);
                        }
                    }

                    for (; tt + 1 < tCount; tt += 2) {
                        const int8_t* dyA = dy8Scratch + (tt + 0) * outDim + j0; const int8_t* dyB = dy8Scratch + (tt + 1) * outDim + j0;
                        float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                        const float dyScaleA = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 0, jb);
                        const float dyScaleB = load_tile_row_scale(dyScaleScratch, nJBlocks, tt + 1, jb);
                        const float gainA = dyScaleA * blockScale;
                        const float gainB = dyScaleB * blockScale;
                        if (kernelPolicy.useFusedX2) {
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
                            accum_bwd_block32_gain8_to_x_pair(dxA, dxB, aA, aB, i0, make_gain8(dyScaleA, ws8), make_gain8(dyScaleB, ws8), addToDest);
                        }
                    }

                    for (; tt < tCount; ++tt) {
                        float* dxRow = dX + (t0 + tt) * inDim;
                        int32_t acc[32] = { 0 };
                        const float dyScale = load_tile_row_scale(dyScaleScratch, nJBlocks, tt, jb);
                        dot_i2s_i8_accum_i32_block32(acc, wq2T, inDim, j0, jCount, dy8Scratch + tt * outDim + j0, i0);
                        accum_bwd_block32_gain8_to_x(dxRow, acc, i0, make_gain8(dyScale, ws8), addToDest);
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
    int8_t* __restrict dv8Scratch, float* __restrict dvScaleScratch,
    bool destStartsZero)
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
    const bool useDecodedQkvCache =
        !Wq.qT8.empty() && !Wk.qT8.empty() && !Wv.qT8.empty();
    const FusedQkvBackwardKernelPolicy kernelPolicy =
        FusedQkvBackwardKernelPolicy::resolve(kDim, vDim, Wk.hadamard32 == Wv.hadamard32);
    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);

#if defined(_OPENMP)
#pragma omp parallel
#endif
        {
#if defined(_OPENMP)
            if (kernelPolicy.useParallelQuant) {
                quantize_x_tile_i8_tiled_omp_for(dq8Scratch, dqScaleScratch, dQ, T, qDim, t0, tCount, kRowsPerQuantScale, hBlock, Wq.hadamard32);
                if (kernelPolicy.useDualKvQuant) {
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
                    if (kernelPolicy.useDualKvQuant) {
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
            if (kernelPolicy.useDualKvQuant) {
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
                    const float* const scaleRow8 = backward_scale_broadcast_row(W, i0, dyJBlocks);
                    for (std::size_t j0 = jBegin; j0 < outDim; j0 += hBlock) {
                        const std::size_t jCount = std::min(hBlock, outDim - j0);
                        const std::size_t jb = j0 / hBlock;
                        const bool addToDest = (j0 != 0);
                        const float* const ws8 = scaleRow8 + jb * BitNetMatI2S::scaleBroadcastWidth;

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

                            if (!W.qT8.empty()) {
                                dot_ternary_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.qT8.data(), inDim, j0, jCount, dyA, dyB, dyC, dyD, i0);
                            }
                            else {
                                dot_i2s_i8_accum_i32_block32_x4(aA, aB, aC, aD, W.q2T.data(), inDim, j0, jCount, dyA, dyB, dyC, dyD, i0);
                            }
                            accum_bwd_block32_gain8_to_x_pair(dxA, dxB, aA, aB, i0, make_gain8(sA, ws8), make_gain8(sB, ws8), addToDest);
                            accum_bwd_block32_gain8_to_x_pair(dxC, dxD, aC, aD, i0, make_gain8(sC, ws8), make_gain8(sD, ws8), addToDest);
                        }

                        for (; tt + 1 < tCount; tt += 2) {
                            const int8_t* dyA = dy8 + (tt + 0) * outDim + j0; const int8_t* dyB = dy8 + (tt + 1) * outDim + j0;
                            float* dxA = dX + (t0 + tt + 0) * inDim; float* dxB = dX + (t0 + tt + 1) * inDim;
                            int32_t aA[32] = { 0 }, aB[32] = { 0 };
                            const float sA = load_tile_row_scale(dyScale, dyJBlocks, tt + 0, jb);
                            const float sB = load_tile_row_scale(dyScale, dyJBlocks, tt + 1, jb);

                            if (!W.qT8.empty()) {
                                dot_ternary_i8_accum_i32_block32_x2(aA, aB, W.qT8.data(), inDim, j0, jCount, dyA, dyB, i0);
                            }
                            else {
                                dot_i2s_i8_accum_i32_block32_x2(aA, aB, W.q2T.data(), inDim, j0, jCount, dyA, dyB, i0);
                            }
                            accum_bwd_block32_gain8_to_x_pair(dxA, dxB, aA, aB, i0, make_gain8(sA, ws8), make_gain8(sB, ws8), addToDest);
                        }

                        for (; tt < tCount; ++tt) {
                            float* dxRow = dX + (t0 + tt) * inDim;
                            int32_t acc[32] = { 0 };
                            const float s = load_tile_row_scale(dyScale, dyJBlocks, tt, jb);
                            if (!W.qT8.empty()) {
                                dot_ternary_i8_accum_i32_block32(acc, W.qT8.data(), inDim, j0, jCount, dy8 + tt * outDim + j0, i0);
                            }
                            else {
                                dot_i2s_i8_accum_i32_block32(acc, W.q2T.data(), inDim, j0, jCount, dy8 + tt * outDim + j0, i0);
                            }
                            accum_bwd_block32_gain8_to_x(dxRow, acc, i0, make_gain8(s, ws8), addToDest);
                        }
                    }
                }
            };

#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
                for (int i0_i = 0; i0_i < static_cast<int>(inDim); i0_i += 32) {
                    const std::size_t i0 = static_cast<std::size_t>(i0_i);
                    const float* const scaleRowQ8 = backward_scale_broadcast_row(Wq, i0, qJBlocks);
                    const float* const scaleRowK8 = backward_scale_broadcast_row(Wk, i0, kJBlocks);
                    const float* const scaleRowV8 = backward_scale_broadcast_row(Wv, i0, vJBlocks);
                    for (std::size_t j0 = 0; j0 < fusedCols; j0 += hBlock) {
                        const std::size_t jCount = std::min(hBlock, fusedCols - j0);
                        const std::size_t jbQ = j0 / hBlock;
                        const std::size_t jbK = j0 / hBlock;
                        const std::size_t jbV = j0 / hBlock;
                        const bool addToDestQ = !destStartsZero || (j0 != 0);
                        const float* const wsQ8 = scaleRowQ8 + jbQ * BitNetMatI2S::scaleBroadcastWidth;
                        const float* const wsK8 = scaleRowK8 + jbK * BitNetMatI2S::scaleBroadcastWidth;
                        const float* const wsV8 = scaleRowV8 + jbV * BitNetMatI2S::scaleBroadcastWidth;
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

                            if (useDecodedQkvCache && decoded_qkv_backward_triple_enabled()) {
                                dot_ternary_i8_accum_i32_block32_x4_triple(
                                    aQA, aQB, aQC, aQD,
                                    aKA, aKB, aKC, aKD,
                                    aVA, aVB, aVC, aVD,
                                    Wq.qT8.data(), Wk.qT8.data(), Wv.qT8.data(),
                                    inDim, j0, jCount,
                                    dqA, dqB, dqC, dqD,
                                    dkA, dkB, dkC, dkD,
                                    dvA, dvB, dvC, dvD,
                                    i0);
                            }
                            else if (useDecodedQkvCache) {
                                dot_ternary_i8_accum_i32_block32_x4(aQA, aQB, aQC, aQD, Wq.qT8.data(), inDim, j0, jCount, dqA, dqB, dqC, dqD, i0);
                                dot_ternary_i8_accum_i32_block32_x4(aKA, aKB, aKC, aKD, Wk.qT8.data(), inDim, j0, jCount, dkA, dkB, dkC, dkD, i0);
                                dot_ternary_i8_accum_i32_block32_x4(aVA, aVB, aVC, aVD, Wv.qT8.data(), inDim, j0, jCount, dvA, dvB, dvC, dvD, i0);
                            }
                            else {
                                dot_i2s_i8_accum_i32_block32_x4(aQA, aQB, aQC, aQD, Wq.q2T.data(), inDim, j0, jCount, dqA, dqB, dqC, dqD, i0);
                                dot_i2s_i8_accum_i32_block32_x4(aKA, aKB, aKC, aKD, Wk.q2T.data(), inDim, j0, jCount, dkA, dkB, dkC, dkD, i0);
                                dot_i2s_i8_accum_i32_block32_x4(aVA, aVB, aVC, aVD, Wv.q2T.data(), inDim, j0, jCount, dvA, dvB, dvC, dvD, i0);
                            }

                            accum_bwd_block32_gain8_to_x_triple_x4(
                                dxA, dxB, dxC, dxD,
                                aQA, aQB, aQC, aQD,
                                aKA, aKB, aKC, aKD,
                                aVA, aVB, aVC, aVD,
                                i0,
                                make_gain8(sqA, wsQ8), make_gain8(sqB, wsQ8),
                                make_gain8(sqC, wsQ8), make_gain8(sqD, wsQ8),
                                make_gain8(skA, wsK8), make_gain8(skB, wsK8),
                                make_gain8(skC, wsK8), make_gain8(skD, wsK8),
                                make_gain8(svA, wsV8), make_gain8(svB, wsV8),
                                make_gain8(svC, wsV8), make_gain8(svD, wsV8),
                                addToDestQ);
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

                            if (useDecodedQkvCache) {
                                dot_ternary_i8_accum_i32_block32_x2(aQA, aQB, Wq.qT8.data(), inDim, j0, jCount, dqA, dqB, i0);
                                dot_ternary_i8_accum_i32_block32_x2(aKA, aKB, Wk.qT8.data(), inDim, j0, jCount, dkA, dkB, i0);
                                dot_ternary_i8_accum_i32_block32_x2(aVA, aVB, Wv.qT8.data(), inDim, j0, jCount, dvA, dvB, i0);
                            }
                            else {
                                dot_i2s_i8_accum_i32_block32_x2(aQA, aQB, Wq.q2T.data(), inDim, j0, jCount, dqA, dqB, i0);
                                dot_i2s_i8_accum_i32_block32_x2(aKA, aKB, Wk.q2T.data(), inDim, j0, jCount, dkA, dkB, i0);
                                dot_i2s_i8_accum_i32_block32_x2(aVA, aVB, Wv.q2T.data(), inDim, j0, jCount, dvA, dvB, i0);
                            }

                            accum_bwd_block32_gain8_to_x_triple(
                                dxA, aQA, aKA, aVA, i0,
                                make_gain8(sqA, wsQ8),
                                make_gain8(skA, wsK8),
                                make_gain8(svA, wsV8),
                                addToDestQ);
                            accum_bwd_block32_gain8_to_x_triple(
                                dxB, aQB, aKB, aVB, i0,
                                make_gain8(sqB, wsQ8),
                                make_gain8(skB, wsK8),
                                make_gain8(svB, wsV8),
                                addToDestQ);
                        }

                        for (; tt < tCount; ++tt) {
                            float* dxRow = dX + (t0 + tt) * inDim;
                            int32_t aQ[32] = { 0 }, aK[32] = { 0 }, aV[32] = { 0 };
                            const float sq = load_tile_row_scale(dqScaleScratch, qJBlocks, tt, jbQ);
                            const float sk = load_tile_row_scale(dkScaleScratch, kJBlocks, tt, jbK);
                            const float sv = load_tile_row_scale(dvScaleScratch, vJBlocks, tt, jbV);

                            if (useDecodedQkvCache) {
                                dot_ternary_i8_accum_i32_block32(aQ, Wq.qT8.data(), inDim, j0, jCount, dq8Scratch + tt * qDim + j0, i0);
                                dot_ternary_i8_accum_i32_block32(aK, Wk.qT8.data(), inDim, j0, jCount, dk8Scratch + tt * kDim + j0, i0);
                                dot_ternary_i8_accum_i32_block32(aV, Wv.qT8.data(), inDim, j0, jCount, dv8Scratch + tt * vDim + j0, i0);
                            }
                            else {
                                dot_i2s_i8_accum_i32_block32(aQ, Wq.q2T.data(), inDim, j0, jCount, dq8Scratch + tt * qDim + j0, i0);
                                dot_i2s_i8_accum_i32_block32(aK, Wk.q2T.data(), inDim, j0, jCount, dk8Scratch + tt * kDim + j0, i0);
                                dot_i2s_i8_accum_i32_block32(aV, Wv.q2T.data(), inDim, j0, jCount, dv8Scratch + tt * vDim + j0, i0);
                            }

                            accum_bwd_block32_gain8_to_x_triple(
                                dxRow, aQ, aK, aV, i0,
                                make_gain8(sq, wsQ8),
                                make_gain8(sk, wsK8),
                                make_gain8(sv, wsV8),
                                addToDestQ);
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
    const DualBackwardKernelPolicy kernelPolicy = DualBackwardKernelPolicy::resolve(T, tTile);

#if defined(_OPENMP)
    if (kernelPolicy.useSingleTeam) {
#pragma omp parallel
        {
            const DualBackwardQuantPolicy quantPolicy =
                DualBackwardQuantPolicy::resolve(
                    kernelPolicy.useParallelQuant,
                    T,
                    tTile,
                    activePrefixT,
                    tTile);

            for (std::size_t q0 = 0; q0 < T; q0 += quantPolicy.quantChunkRows) {
                const std::size_t qCount = std::min(quantPolicy.quantChunkRows, T - q0);
                const int inDimInt = static_cast<int>(inDim);
                const DualBackwardQuantPolicy tileQuantPolicy =
                    DualBackwardQuantPolicy::resolve(
                        kernelPolicy.useParallelQuant,
                        T,
                        tTile,
                        activePrefixT,
                        qCount);

                if (kernelPolicy.useParallelQuant) {
                    quantize_x_tile_i8_tiled_dual_omp_for(
                        dyA8Scratch, dyAScaleScratch, dYa,
                        dyB8Scratch, dyBScaleScratch, dYb,
                        T, outDim, q0, qCount, kRowsPerQuantScale, hBlock, Wa.hadamard32);
                }
                else if (tileQuantPolicy.useMasterQuant) {
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

                for (std::size_t t0 = q0; t0 < q0 + qCount; t0 += tileQuantPolicy.workTileRows) {
                    const std::size_t tCount = std::min(tileQuantPolicy.workTileRows, (q0 + qCount) - t0);
                    if (tile_all_masked(attentionMask, activePrefixT, t0, tCount)) continue;

                    const std::size_t tileRowBase = t0 - q0;
                    const int8_t* const dyA8Tile = dyA8Scratch + tileRowBase * outDim;
                    const int8_t* const dyB8Tile = dyB8Scratch + tileRowBase * outDim;
                    const float* const dyAScaleTile = dyAScaleScratch + tileRowBase * nJBlocks;
                    const float* const dyBScaleTile = dyBScaleScratch + tileRowBase * nJBlocks;

#pragma omp for schedule(static)
                    for (int i0_i = 0; i0_i < inDimInt; i0_i += 32) {
                    const std::size_t i0 = static_cast<std::size_t>(i0_i);
                    const float* const scaleRowA8 = backward_scale_broadcast_row(Wa, i0, nJBlocks);
                    const float* const scaleRowB8 = backward_scale_broadcast_row(Wb, i0, nJBlocks);
                    const uint8_t* const waq2T = Wa.q2T.data();
                    const uint8_t* const wbq2T = Wb.q2T.data();
                    for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                        const std::size_t jCount = std::min(hBlock, outDim - j0);
                        const std::size_t jb = j0 / hBlock;
                        const bool addToDest = (jb != 0);
                        const bool overwriteDest = destStartsZero && !addToDest;
                        const float* const wsA8 = scaleRowA8 + jb * BitNetMatI2S::scaleBroadcastWidth;
                        const float* const wsB8 = scaleRowB8 + jb * BitNetMatI2S::scaleBroadcastWidth;
                        const float blockScaleA = broadcast_scale_value(wsA8);
                        const float blockScaleB = broadcast_scale_value(wsB8);

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
                            if (kernelPolicy.useFusedX4) {
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
                            else if (kernelPolicy.useFusedX2) {
                                if (overwriteDest) {
                                    const auto storeOverwrite = kernelPolicy.useDenseX2Dual
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
                                    const auto storeAccum = kernelPolicy.useDenseX2Dual
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
                                accum_bwd_block32_gain8_to_x_dual_x4(
                                    dx0, dx1, dx2, dx3,
                                    aA0, aA1, aA2, aA3,
                                    aB0, aB1, aB2, aB3,
                                    i0,
                                    make_gain8(scaleA0, wsA8), make_gain8(scaleA1, wsA8),
                                    make_gain8(scaleA2, wsA8), make_gain8(scaleA3, wsA8),
                                    make_gain8(scaleB0, wsB8), make_gain8(scaleB1, wsB8),
                                    make_gain8(scaleB2, wsB8), make_gain8(scaleB3, wsB8),
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
                            if (kernelPolicy.useFusedX2) {
                                if (overwriteDest) {
                                    const auto storeOverwrite = kernelPolicy.useDenseX2Dual
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
                                    const auto storeAccum = kernelPolicy.useDenseX2Dual
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
                                accum_bwd_block32_gain8_to_x_dual(dx0, aA0, aB0, i0, make_gain8(scaleA0, wsA8), make_gain8(scaleB0, wsB8), addToDest);
                                accum_bwd_block32_gain8_to_x_dual(dx1, aA1, aB1, i0, make_gain8(scaleA1, wsA8), make_gain8(scaleB1, wsB8), addToDest);
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
                            accum_bwd_block32_gain8_to_x_dual(dxRow, accA, accB, i0, make_gain8(scaleA, wsA8), make_gain8(scaleB, wsB8), addToDest);
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
            if (kernelPolicy.useParallelQuant) {
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
                const float* const scaleRowA8 = backward_scale_broadcast_row(Wa, i0, nJBlocks);
                const float* const scaleRowB8 = backward_scale_broadcast_row(Wb, i0, nJBlocks);
                const uint8_t* const waq2T = Wa.q2T.data();
                const uint8_t* const wbq2T = Wb.q2T.data();

                for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                    const std::size_t jCount = std::min(hBlock, outDim - j0);
                    const std::size_t jb = j0 / hBlock;
                    const bool addToDest = (jb != 0);
                    const bool overwriteDest = destStartsZero && !addToDest;
                    const float* const wsA8 = scaleRowA8 + jb * BitNetMatI2S::scaleBroadcastWidth;
                    const float* const wsB8 = scaleRowB8 + jb * BitNetMatI2S::scaleBroadcastWidth;
                    const float blockScaleA = broadcast_scale_value(wsA8);
                    const float blockScaleB = broadcast_scale_value(wsB8);

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
                        if (kernelPolicy.useFusedX4) {
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
                        else if (kernelPolicy.useFusedX2) {
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
                            accum_bwd_block32_gain8_to_x_dual_x4(
                                dx0, dx1, dx2, dx3,
                                accA0, accA1, accA2, accA3,
                                accB0, accB1, accB2, accB3,
                                i0,
                                make_gain8(scaleA0, wsA8), make_gain8(scaleA1, wsA8),
                                make_gain8(scaleA2, wsA8), make_gain8(scaleA3, wsA8),
                                make_gain8(scaleB0, wsB8), make_gain8(scaleB1, wsB8),
                                make_gain8(scaleB2, wsB8), make_gain8(scaleB3, wsB8),
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
                        if (kernelPolicy.useFusedX2) {
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

                            accum_bwd_block32_gain8_to_x_dual(
                                dx0, accA0, accB0, i0,
                                make_gain8(scaleA0, wsA8), make_gain8(scaleB0, wsB8), addToDest);
                            accum_bwd_block32_gain8_to_x_dual(
                                dx1, accA1, accB1, i0,
                                make_gain8(scaleA1, wsA8), make_gain8(scaleB1, wsB8), addToDest);
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

                        accum_bwd_block32_gain8_to_x_dual(
                            dx, accA, accB, i0,
                            make_gain8(scaleA, wsA8), make_gain8(scaleB, wsB8), addToDest);
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
            const float* const scaleRowA8 = backward_scale_broadcast_row(Wa, i0, nJBlocks);
            const float* const scaleRowB8 = backward_scale_broadcast_row(Wb, i0, nJBlocks);

            for (std::size_t j0 = 0; j0 < outDim; j0 += hBlock) {
                const std::size_t jCount = std::min(hBlock, outDim - j0);
                const std::size_t jb = j0 / hBlock;
                const float* const wsA8 = scaleRowA8 + jb * BitNetMatI2S::scaleBroadcastWidth;
                const float* const wsB8 = scaleRowB8 + jb * BitNetMatI2S::scaleBroadcastWidth;

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

                    if (kernelPolicy.useFusedX4) {
                        dot_i2s_i8_scale_store_block16_x4_dual_bwd_exact(
                            dx0, dx1, dx2, dx3,
                            Wa.q2T.data(), Wb.q2T.data(),
                            inDim, j0, jCount,
                            dyA0, dyA1, dyA2, dyA3,
                            dyB0, dyB1, dyB2, dyB3,
                            i0,
                            addToDest,
                            scaleA0 * broadcast_scale_value(wsA8), scaleA1 * broadcast_scale_value(wsA8),
                            scaleA2 * broadcast_scale_value(wsA8), scaleA3 * broadcast_scale_value(wsA8),
                            scaleB0 * broadcast_scale_value(wsB8), scaleB1 * broadcast_scale_value(wsB8),
                            scaleB2 * broadcast_scale_value(wsB8), scaleB3 * broadcast_scale_value(wsB8));
                        dot_i2s_i8_scale_store_block16_x4_dual_bwd_exact(
                            dx0, dx1, dx2, dx3,
                            Wa.q2T.data(), Wb.q2T.data(),
                            inDim, j0, jCount,
                            dyA0, dyA1, dyA2, dyA3,
                            dyB0, dyB1, dyB2, dyB3,
                            i0 + 16,
                            addToDest,
                            scaleA0 * broadcast_scale_value(wsA8), scaleA1 * broadcast_scale_value(wsA8),
                            scaleA2 * broadcast_scale_value(wsA8), scaleA3 * broadcast_scale_value(wsA8),
                            scaleB0 * broadcast_scale_value(wsB8), scaleB1 * broadcast_scale_value(wsB8),
                            scaleB2 * broadcast_scale_value(wsB8), scaleB3 * broadcast_scale_value(wsB8));
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
                            accum_bwd_block32_gain8_to_x_dual_x4(
                                dx0, dx1, dx2, dx3,
                                accA0, accA1, accA2, accA3,
                                accB0, accB1, accB2, accB3,
                                i0,
                                make_gain8(scaleA0, wsA8), make_gain8(scaleA1, wsA8), make_gain8(scaleA2, wsA8), make_gain8(scaleA3, wsA8),
                                make_gain8(scaleB0, wsB8), make_gain8(scaleB1, wsB8), make_gain8(scaleB2, wsB8), make_gain8(scaleB3, wsB8),
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

                    accum_bwd_block32_gain8_to_x_dual(
                        dx0, accA0, accB0, i0,
                        make_gain8(scaleA0, wsA8), make_gain8(scaleB0, wsB8), addToDest);
                    accum_bwd_block32_gain8_to_x_dual(
                        dx1, accA1, accB1, i0,
                        make_gain8(scaleA1, wsA8), make_gain8(scaleB1, wsB8), addToDest);
                }

                for (; tt < tCount; ++tt) {
                    float* dx = dX + (t0 + tt) * inDim;
                    int32_t accA[32] = { 0 };
                    int32_t accB[32] = { 0 };
                    const float scaleA = load_tile_row_scale(dyAScaleScratch, nJBlocks, tt, jb);
                    const float scaleB = load_tile_row_scale(dyBScaleScratch, nJBlocks, tt, jb);

                    dot_i2s_i8_accum_i32_block32(accA, Wa.q2T.data(), inDim, j0, jCount, dyA8Scratch + tt * outDim + j0, i0);
                    dot_i2s_i8_accum_i32_block32(accB, Wb.q2T.data(), inDim, j0, jCount, dyB8Scratch + tt * outDim + j0, i0);

                    accum_bwd_block32_gain8_to_x_dual(
                        dx, accA, accB, i0,
                        make_gain8(scaleA, wsA8), make_gain8(scaleB, wsB8), addToDest);
                }
            }
        }
#endif
    }
}
