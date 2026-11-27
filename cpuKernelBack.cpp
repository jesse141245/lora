#include "cpuKernelBack.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <chrono>
#include <utility>
#include <cstring>
#include <filesystem>
#include <cmath>
#include <optional>
#include <string>
#include <limits>

#include <nlohmann/json.hpp>

#include "attention.h"
#include "backend_runtime_policy.h"
#include "backend_runtime_policy_builder.h"
#include "backend_runtime_policy_json.h"
#include "bitnet_i2_s.h"
#include "bitnet_i2s_matmul.h"
#include "bitnet_tl2_generated.h"
#include "bitnet_tl2_matmul.h"
#include "bitnet_quant.h"
#include "bitnet_weight.h"
#include "env_utils.h"
#include "kernels.h"
#include "paramRef.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

using ProfileClock = std::chrono::steady_clock;
using envutil::envBoolOr;
using envutil::envSizeTOr;

static inline void lora_add_out(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict tmp,
    std::size_t inDim,
    std::size_t outDim,
    std::size_t R,
    float scale)
{
    for (std::size_t r = 0; r < R; ++r) {
        tmp[r] = dot_avx2_or_scalar(x, A + r * inDim, inDim);
    }
    for (std::size_t r = 0; r < R; ++r) {
        axpy_inplace_avx2_or_scalar(y, B + r * outDim, outDim, scale * tmp[r]);
    }
}

constexpr uint32_t kLoraFlagQ = 1u;
constexpr uint32_t kLoraFlagK = 2u;
constexpr uint32_t kLoraFlagV = 4u;
constexpr uint32_t kLoraFlagO = 8u;
constexpr uint32_t kLoraFlagFfnDown = 16u;
constexpr uint32_t kLoraFlagLmHead = 32u;
constexpr uint32_t kLoraFlagFfnGateUp = 64u;
constexpr uint32_t kLoraFlagLmHeadBias = 128u;
constexpr uint32_t kLoraFlagFinalNormAffine = 256u;
constexpr uint32_t kLoraFlagLayerRmsNorm = 512u;
constexpr uint32_t kLoraFlagFinalRmsNorm = 1024u;
constexpr uint32_t kLoraFlagDoRA = 2048u;
constexpr uint32_t kLoraFlagPiSSA = 4096u;

struct DoraStats {
    double mSum = 0.0;
    double cSum = 0.0;
    double ratioSum = 0.0;
    double deltaSq = 0.0;
    float mMin = std::numeric_limits<float>::max();
    float mMax = 0.0f;
    float cMin = std::numeric_limits<float>::max();
    float cMax = 0.0f;
    float ratioMin = std::numeric_limits<float>::max();
    float ratioMax = 0.0f;
    std::size_t n = 0;

    void add(float m, float c, float ratio, double deltaColSq) {
        mSum += m;
        cSum += c;
        ratioSum += ratio;
        deltaSq += deltaColSq;
        mMin = std::min(mMin, m);
        mMax = std::max(mMax, m);
        cMin = std::min(cMin, c);
        cMax = std::max(cMax, c);
        ratioMin = std::min(ratioMin, ratio);
        ratioMax = std::max(ratioMax, ratio);
        ++n;
    }

    void merge(const DoraStats& other) {
        if (other.n == 0) return;
        mSum += other.mSum;
        cSum += other.cSum;
        ratioSum += other.ratioSum;
        deltaSq += other.deltaSq;
        mMin = std::min(mMin, other.mMin);
        mMax = std::max(mMax, other.mMax);
        cMin = std::min(cMin, other.cMin);
        cMax = std::max(cMax, other.cMax);
        ratioMin = std::min(ratioMin, other.ratioMin);
        ratioMax = std::max(ratioMax, other.ratioMax);
        n += other.n;
    }
};

static inline void dora_init_column_magnitude(
    std::vector<float>& m,
    std::vector<float>& c,
    std::vector<float>& ratio,
    const std::vector<float>& W,
    std::size_t inDim,
    std::size_t outDim)
{
    m.assign(outDim, 0.0f);
    c.assign(outDim, 1.0f);
    ratio.assign(outDim, 1.0f);
    constexpr float eps = 1e-12f;
    for (std::size_t j = 0; j < outDim; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < inDim; ++i) {
            const double v = W[i * outDim + j];
            sum += v * v;
        }
        m[j] = static_cast<float>(std::sqrt(std::max(sum, 0.0)));
        c[j] = std::max(m[j], eps);
        ratio[j] = m[j] / c[j];
    }
}

static inline DoraStats dora_cached_stats(
    const std::vector<float>& m,
    const std::vector<float>& c,
    std::vector<float>& ratio)
{
    DoraStats stats;
    constexpr float eps = 1e-12f;
    ratio.resize(std::min(m.size(), c.size()));
    const std::size_t n = ratio.size();
    for (std::size_t j = 0; j < n; ++j) {
        ratio[j] = m[j] / std::max(c[j], eps);
        stats.add(m[j], c[j], ratio[j], 0.0);
    }
    return stats;
}

static inline DoraStats dora_update_detached_norm_ratio(
    const std::vector<float>& W,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& m,
    std::vector<float>& c,
    std::vector<float>& ratio,
    std::size_t inDim,
    std::size_t outDim,
    std::size_t rank,
    float scale)
{
    if (W.size() != inDim * outDim || A.size() != rank * inDim ||
        B.size() != rank * outDim || m.size() != outDim) {
        throw std::runtime_error("DoRA norm update: tensor size mismatch");
    }
    c.resize(outDim);
    ratio.resize(outDim);
    DoraStats stats;
    constexpr float eps = 1e-12f;
    for (std::size_t j = 0; j < outDim; ++j) {
        double colSq = 0.0;
        double deltaColSq = 0.0;
        for (std::size_t i = 0; i < inDim; ++i) {
            double delta = 0.0;
            for (std::size_t r = 0; r < rank; ++r) {
                delta += static_cast<double>(B[r * outDim + j]) *
                    static_cast<double>(A[r * inDim + i]);
            }
            delta *= scale;
            const double v = static_cast<double>(W[i * outDim + j]) + delta;
            colSq += v * v;
            deltaColSq += delta * delta;
        }
        const float cj = static_cast<float>(std::sqrt(std::max(colSq, static_cast<double>(eps))));
        const float rj = m[j] / std::max(cj, eps);
        c[j] = cj;
        ratio[j] = rj;
        stats.add(m[j], cj, rj, deltaColSq);
    }
    return stats;
}

static inline float bitnet_i2s_weight_value(
    const BitNetMatI2S& W,
    std::size_t row,
    std::size_t col)
{
    const std::size_t idx = row * W.cols + col;
    const uint8_t code = static_cast<uint8_t>((W.q2[idx >> 2] >> ((idx & 3u) * 2u)) & 0x3u);
    const float q = static_cast<float>(i2_code_to_ternary(code));
    if (q == 0.0f) return 0.0f;
    const std::size_t kb = row / W.qk;
    const std::size_t jb = col / W.jq;
    return q * W.w_scale_tile[W.forwardScaleIndex(jb, kb)];
}

static inline void dora_transform_a_for_i2s(
    std::vector<float>& scratch,
    const std::vector<float>& A,
    std::size_t inDim,
    std::size_t rank,
    bool hadamard32)
{
    if (!hadamard32) {
        return;
    }
    scratch = A;
    for (std::size_t r = 0; r < rank; ++r) {
        float* row = scratch.data() + r * inDim;
        for (std::size_t i = 0; i < inDim; i += 32) {
            hadamard32_inplace(row + i);
        }
    }
}

static inline void dora_init_column_magnitude_i2s(
    std::vector<float>& m,
    std::vector<float>& c,
    std::vector<float>& ratio,
    const BitNetMatI2S& W)
{
    if (W.empty()) return;
    m.assign(W.cols, 0.0f);
    c.assign(W.cols, 1.0f);
    ratio.assign(W.cols, 1.0f);
    constexpr float eps = 1e-12f;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int jI = 0; jI < static_cast<int>(W.cols); ++jI) {
        const std::size_t j = static_cast<std::size_t>(jI);
        double sum = 0.0;
        for (std::size_t i = 0; i < W.rows; ++i) {
            const double v = bitnet_i2s_weight_value(W, i, j);
            sum += v * v;
        }
        m[j] = static_cast<float>(std::sqrt(std::max(sum, 0.0)));
        c[j] = std::max(m[j], eps);
        ratio[j] = m[j] / c[j];
    }
}

static inline DoraStats dora_update_detached_norm_ratio_i2s(
    const BitNetMatI2S& W,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& m,
    std::vector<float>& c,
    std::vector<float>& ratio,
    std::size_t inDim,
    std::size_t outDim,
    std::size_t rank,
    float scale)
{
    if (W.empty()) {
        throw std::runtime_error("DoRA I2S norm update: empty quantized weight");
    }
    if (W.rows != inDim || W.cols != outDim || A.size() != rank * inDim ||
        B.size() != rank * outDim || m.size() != outDim) {
        throw std::runtime_error("DoRA I2S norm update: tensor size mismatch");
    }
    c.resize(outDim);
    ratio.resize(outDim);

    std::vector<float> aScratch;
    dora_transform_a_for_i2s(aScratch, A, inDim, rank, W.hadamard32);
    const float* Aeff = W.hadamard32 ? aScratch.data() : A.data();

    std::vector<float> gram(rank * rank, 0.0f);
    for (std::size_t r0 = 0; r0 < rank; ++r0) {
        for (std::size_t r1 = r0; r1 < rank; ++r1) {
            const float g = dot_avx2_or_scalar(Aeff + r0 * inDim, Aeff + r1 * inDim, inDim);
            gram[r0 * rank + r1] = g;
            gram[r1 * rank + r0] = g;
        }
    }

    constexpr float eps = 1e-12f;
    DoraStats merged;
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        std::vector<float> cross(rank, 0.0f);
        DoraStats local;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int jI = 0; jI < static_cast<int>(outDim); ++jI) {
            const std::size_t j = static_cast<std::size_t>(jI);
            std::fill(cross.begin(), cross.end(), 0.0f);

            double baseSq = 0.0;
            for (std::size_t i = 0; i < inDim; ++i) {
                const float w = bitnet_i2s_weight_value(W, i, j);
                if (w == 0.0f) continue;
                baseSq += static_cast<double>(w) * static_cast<double>(w);
                for (std::size_t r = 0; r < rank; ++r) {
                    cross[r] += w * Aeff[r * inDim + i];
                }
            }

            double crossTerm = 0.0;
            for (std::size_t r = 0; r < rank; ++r) {
                crossTerm += static_cast<double>(B[r * outDim + j]) *
                    static_cast<double>(cross[r]);
            }

            double deltaColSq = 0.0;
            for (std::size_t r0 = 0; r0 < rank; ++r0) {
                const double b0 = static_cast<double>(B[r0 * outDim + j]);
                if (b0 == 0.0) continue;
                for (std::size_t r1 = 0; r1 < rank; ++r1) {
                    deltaColSq += b0 *
                        static_cast<double>(B[r1 * outDim + j]) *
                        static_cast<double>(gram[r0 * rank + r1]);
                }
            }
            deltaColSq *= static_cast<double>(scale) * static_cast<double>(scale);

            const double colSq = baseSq + 2.0 * static_cast<double>(scale) * crossTerm + deltaColSq;
            const float cj = static_cast<float>(std::sqrt(std::max(colSq, static_cast<double>(eps))));
            const float rj = m[j] / std::max(cj, eps);
            c[j] = cj;
            ratio[j] = rj;
            local.add(m[j], cj, rj, deltaColSq);
        }
#if defined(_OPENMP)
#pragma omp critical
#endif
        {
            merged.merge(local);
        }
    }
    return merged;
}

static inline void dora_apply_ratio(float* y, const std::vector<float>& ratio, std::size_t outDim)
{
    if (ratio.empty()) return;
    for (std::size_t j = 0; j < outDim; ++j) {
        y[j] *= ratio[j];
    }
}

static inline void dora_backward_magnitude_and_scale(
    float* dy,
    const float* postDora,
    const std::vector<float>& c,
    const std::vector<float>& ratio,
    std::vector<float>& gradM,
    std::size_t outDim)
{
    if (ratio.empty()) return;
    constexpr float eps = 1e-12f;
    for (std::size_t j = 0; j < outDim; ++j) {
        const float d = dy[j];
        const float r = ratio[j];
        const float pre = (std::abs(r) > eps) ? (postDora[j] / r) : 0.0f;
        gradM[j] += d * pre / std::max(c[j], eps);
        dy[j] = d * r;
    }
}

std::optional<bool> envBoolOverride(const char* name) {
    const auto value = envutil::tryGetEnvString(name);
    if (!value) {
        return std::nullopt;
    }
    return envutil::envStringTruthy(envutil::lowerCopy(*value));
}

bool lmHeadLoraBackwardRank4Enabled(std::size_t rank, std::size_t vocab, std::size_t rows) {
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_LM_HEAD_LORA_BWD_R4");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    // LM-head LoRA backward is dominated by repeated full-vocab dLogits passes.
    // Processing four ranks per pass improves dLogits reuse while keeping each
    // gradB row private to one OpenMP worker.
    return rank >= 4 && vocab >= 1024 && rows > 0;
}

std::optional<bool> runtimeCpuPolicyBool(
    std::optional<bool> CpuBackendRuntimePolicy::*member);
std::optional<std::size_t> runtimeCpuPolicySize(
    std::optional<std::size_t> CpuBackendRuntimePolicy::*member);
bool lmHeadBackwardLossMaskSkipEnabled();
bool ffnBackwardFuseResidualAddEnabled(std::size_t activeComputeT);
bool exactChunkedFfnBackwardEnabled(std::size_t activeComputeT);
std::size_t exactChunkedFfnBackwardRows(std::size_t activeComputeT, std::size_t fallback);

struct BackendProfileState {
    double fwdEmbedMs = 0.0;
    double fwdAttnNormMs = 0.0;
    double fwdAttnPrepMs = 0.0;
    double fwdQkvMs = 0.0;
    double fwdLoraMs = 0.0;
    double fwdRopeMs = 0.0;
    double fwdAttnMs = 0.0;
    double fwdWoMs = 0.0;
    double fwdResidualMs = 0.0;
    double fwdFfnNormMs = 0.0;
    double fwdFfnMs = 0.0;
    double fwdCacheMs = 0.0;
    double fwdFinalNormMs = 0.0;
    double fwdLmHeadMs = 0.0;

    double bwdLmHeadMs = 0.0;
    double bwdFinalNormMs = 0.0;
    double bwdClearMs = 0.0;
    double bwdRestoreMs = 0.0;
    double bwdFfnMs = 0.0;
    double bwdFfnWdownMs = 0.0;
    double bwdFfnSiluMs = 0.0;
    double bwdFfnDualMs = 0.0;
    double bwdFfnNormMs = 0.0;
    double bwdWoMs = 0.0;
    double bwdAttnMs = 0.0;
    double bwdRopeMs = 0.0;
    double bwdLoraMs = 0.0;
    double bwdQkvMs = 0.0;
    double bwdResidualMs = 0.0;

    int profiledBackwardCalls = 0;
};

bool backendProfileEnabled() {
    static const bool enabled = envBoolOr("QLORA_PROFILE_BACKEND", false);
    return enabled;
}

bool fastSiluForwardEnabled() {
    static const bool enabled = envBoolOr("QLORA_TEST_FAST_SILU_FWD", false);
    return enabled;
}

bool fastSiluBackwardEnabled() {
    static const bool enabled = envBoolOr("QLORA_TEST_FAST_SILU_BWD", false);
    return enabled;
}

bool fp32FfnFastSiluForwardEnabled() {
    static const bool enabled = envBoolOr("QLORA_TEST_FP32_FFN_FAST_SILU_FWD", false);
    return enabled;
}

bool fp32FfnFastSiluBackwardEnabled() {
    static const bool enabled = envBoolOr("QLORA_TEST_FP32_FFN_FAST_SILU_BWD", false);
    return enabled;
}

bool exactSiluBackwardChunkedEnabled() {
    static const bool cached = envBoolOr("QLORA_TEST_EXACT_SILU_BWD_CHUNKED", false);
    return cached;
}

bool exactMaskedTileSkipEnabled() {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::exactMaskedTileSkip)) {
        return *forced;
    }
    static const bool cached = envBoolOr("QLORA_TEST_I2S_MASK_SKIP", false);
    return cached;
}

bool prefixMaskFastpathEnabled() {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::prefixMaskFastpath)) {
        return *forced;
    }
    static const bool cached = envBoolOr("QLORA_TEST_PREFIX_MASK_FASTPATH", false);
    return cached;
}

bool activePrefixComputeEnabled() {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::activePrefixCompute)) {
        return *forced;
    }
    static const bool cached = envBoolOr("QLORA_TEST_ACTIVE_PREFIX_COMPUTE", false);
    return cached;
}

bool rmsnormBackwardAccumEnabled() {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::rmsnormBackwardAccum)) {
        return *forced;
    }
    static const bool cached = envBoolOr("QLORA_TEST_RMSNORM_BWD_ACCUM", false);
    return cached;
}

bool parallelRmsnormForwardEnabled() {
    static const bool cached = envBoolOr("QLORA_TEST_PARALLEL_RMSNORM_FWD", false);
    return cached;
}

bool tl2GeneratedCallerFillsEnabled() {
    static const bool cached = envBoolOr("QLORA_TEST_TL2_GENERATED_CALLER_FILLS", false);
    return cached;
}

bool fusedTl2GeneratedQVEnabled() {
    static const bool cached = envBoolOr("QLORA_TEST_TL2_FUSED_QV_GEN", false);
    return cached;
}

bool fusedTl2GeneratedHybridQkvEnabled() {
    static const bool cached = envBoolOr("QLORA_TEST_TL2_FUSED_HYBRID_QKV", false);
    return cached;
}

bool attentionStoreScoresEnabled() {
    // Backward consumes the probability cache, not the raw attention scores.
    // Keeping scores is useful for diagnostics, but it adds a full attention
    // matrix write in the hot forward path.
    static const bool cached = envBoolOr("QLORA_TEST_ATTN_STORE_SCORES", false);
    return cached;
}

bool attentionProbCacheEnabled(std::size_t maxSeqLen, bool configuredEnabled) {
    if (const std::optional<bool> forced = envBoolOverride("QLORA_TEST_CACHE_ATTN_PROBS")) {
        return *forced;
    }

    if (!configuredEnabled) {
        return false;
    }

    // The legacy cache is useful at short sequence lengths, but at long context
    // it grows as O(layers * heads * T^2). Default long-context runs to the
    // recompute path so seq2048 remains a practical training target.
    return maxSeqLen < 1024;
}

bool lmHeadBackwardRowCompactEnabled(
    std::size_t activeComputeT,
    std::size_t supervisedRows)
{
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::lmHeadBackwardRowCompactOverride)) {
        return *forced;
    }
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_LM_HEAD_BWD_ROW_COMPACT");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    if (supervisedRows == 0 || supervisedRows >= activeComputeT) {
        return false;
    }

    // The LM-head backward is vocab dominated. Even modest row reduction usually
    // beats the copy/scatter overhead on the seq256 tutor batches.
    return supervisedRows < activeComputeT;
}

bool parallelAttentionPrepEnabled(std::size_t activeComputeT) {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::parallelAttentionPrep)) {
        return *forced;
    }
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_PARALLEL_ATTN_PREP");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    // Long-sequence tutor runs spend enough time in LoRA prep + RoPE to
    // justify the extra OpenMP region. Shorter runs still keep the old default.
    return activeComputeT >= 128;
}

bool fusedQkvForwardOverwriteEnabled(std::size_t activeComputeT, std::size_t tTile) {
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_BITNET_FUSED_QKV_FORWARD_OVERWRITE");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return tTile >= 4 && activeComputeT >= 128;
}

bool splitFusedQkvBackwardEnabled() {
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_BITNET_SPLIT_FUSED_QKV_BACKWARD");
    return cached.value_or(false);
}

bool parallelAttentionLoraBackwardEnabled(std::size_t activeComputeT) {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::parallelAttentionLoraBackward)) {
        return *forced;
    }
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_PARALLEL_ATTN_LORA_BWD");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return activeComputeT >= 128;
}

std::size_t contiguousActivePrefixLength(std::span<const int> attentionMask) {
    bool sawZero = false;
    std::size_t activePrefixT = 0;
    for (std::size_t t = 0; t < attentionMask.size(); ++t) {
        if (attentionMask[t] != 0) {
            if (sawZero) {
                return static_cast<std::size_t>(-1);
            }
            activePrefixT = t + 1;
        }
        else {
            sawZero = true;
        }
    }
    return activePrefixT;
}

std::vector<std::size_t> packedSegmentStarts(
    std::span<const int> attentionMask,
    bool enabled)
{
    if (!enabled || attentionMask.empty()) {
        return {};
    }

    bool hasPackedBoundary = false;
    for (const int value : attentionMask) {
        if (value > 1) {
            hasPackedBoundary = true;
            break;
        }
    }
    if (!hasPackedBoundary) {
        return {};
    }

    std::vector<std::size_t> starts(attentionMask.size(), 0);
    std::size_t currentStart = 0;
    for (std::size_t t = 0; t < attentionMask.size(); ++t) {
        if (attentionMask[t] > 1) {
            currentStart = t;
        }
        starts[t] = currentStart;
    }
    return starts;
}

bool ffnGemmEnabled() {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::ffnGemm)) {
        return *forced;
    }
    static const bool enabled = envBoolOr("QLORA_TEST_FFN_GEMM", false);
    return enabled;
}

bool ffnBitnetEnabled() {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::ffnBitnet)) {
        return *forced;
    }
    static const bool enabled = envBoolOr("QLORA_TEST_BITNET_FFN", true);
    return enabled;
}

std::optional<bool> runtimeCpuPolicyBool(
    std::optional<bool> CpuBackendRuntimePolicy::*member)
{
    const BackendRuntimePolicy* policy = currentBackendRuntimePolicy();
    if (!policy) return std::nullopt;
    return policy->cpuBackend.*member;
}

std::optional<std::size_t> runtimeCpuPolicySize(
    std::optional<std::size_t> CpuBackendRuntimePolicy::*member)
{
    const BackendRuntimePolicy* policy = currentBackendRuntimePolicy();
    if (!policy) return std::nullopt;
    return policy->cpuBackend.*member;
}

bool bitnetFfnStreamedPrefixBackwardEnabled() {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::bitnetFfnStreamedPrefixBackward)) {
        return *forced;
    }
    static const bool enabled =
        envBoolOr("QLORA_TEST_BITNET_FFN_BWD_STREAM_PREFIX", false);
    return enabled;
}

bool fusedAttentionNormCacheWriteEnabled(std::size_t activeComputeT) {
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_FUSED_ATTN_NORM_CACHE_WRITE");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return activeComputeT >= 128;
}

bool fusedAttentionQkvCacheWriteEnabled(std::size_t activeComputeT) {
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_FUSED_ATTN_QKV_CACHE_WRITE");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return activeComputeT >= 128;
}

bool directAttentionBackwardCacheEnabled(std::size_t activeComputeT) {
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_DIRECT_ATTN_BWD_CACHE");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return activeComputeT >= 128 && fusedAttentionQkvCacheWriteEnabled(activeComputeT);
}

bool directFfnBackwardCacheEnabled(std::size_t activeComputeT) {
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_DIRECT_FFN_BWD_CACHE");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return activeComputeT >= 128;
}

bool woBackwardOverwriteEnabled(std::size_t activeComputeT) {
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_WO_BWD_OVERWRITE");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return activeComputeT >= 128;
}

bool attentionBaseBackwardFirstEnabled(std::size_t activeComputeT) {
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_ATTN_BASE_BWD_FIRST");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return false;
}

bool groupedAttentionBackwardKvPreclearEnabled(
    std::size_t activeComputeT,
    std::size_t nHeads,
    std::size_t nKvHeads)
{
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_ATTN_GROUP_LOCAL_SKIP_KV_PRECLEAR");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return activeComputeT >= 128 &&
        nHeads > nKvHeads &&
        envBoolOr("QLORA_TEST_ATTN_BWD_GROUP_LOCAL_ACCUM", true);
}

std::size_t bitnetFfnStreamedPrefixBackwardChunkT(std::size_t fallback) {
    if (const std::optional<std::size_t> forced = runtimeCpuPolicySize(&CpuBackendRuntimePolicy::bitnetFfnStreamedPrefixChunkT)) {
        return *forced;
    }
    return envSizeTOr("QLORA_TEST_BITNET_FFN_BWD_STREAM_CHUNK_T", fallback);
}

std::size_t bitnetDualBackwardQuantChunkRows(std::size_t fallback) {
    return envSizeTOr("QLORA_TEST_BITNET_DUAL_BWD_QUANT_CHUNK_ROWS", fallback);
}

std::size_t exactFusedQkvForwardTileRows(std::size_t activeComputeT, std::size_t fallback) {
    if (const std::optional<std::size_t> forced = runtimeCpuPolicySize(&CpuBackendRuntimePolicy::exactFusedQkvForwardTileRows)) {
        return *forced;
    }
    const std::size_t promoted =
        (activeComputeT >= 128 && fallback >= 4)
        ? std::max<std::size_t>(fallback, 16)
        : fallback;
    return envSizeTOr("QLORA_TEST_BITNET_FUSED_QKV_FORWARD_TILE_ROWS", promoted);
}

std::size_t exactWdownBackwardTileRows(std::size_t activeComputeT, std::size_t fallback) {
    if (const std::optional<std::size_t> forced = runtimeCpuPolicySize(&CpuBackendRuntimePolicy::exactWdownBackwardTileRows)) {
        return *forced;
    }
    const std::size_t promoted =
        (activeComputeT >= 128 && fallback >= 4)
        ? std::max<std::size_t>(fallback, 16)
        : fallback;
    return envSizeTOr("QLORA_TEST_BITNET_WDOWN_BWD_TILE_ROWS", promoted);
}

bool decodedWdownBackwardCacheEnabled() {
    return envBoolOr("QLORA_TEST_BITNET_WDOWN_BWD_DECODED_TERNARY_CACHE", false);
}

bool decodedQkvBackwardCacheEnabled() {
    return envBoolOr("QLORA_TEST_BITNET_QKV_BWD_DECODED_TERNARY_CACHE", false);
}

bool decodedQkvForwardCacheEnabled() {
    return envBoolOr("QLORA_TEST_BITNET_QKV_FWD_DECODED_TERNARY_CACHE", false);
}

bool decodedLmHeadForwardCacheEnabled() {
    return envBoolOr("QLORA_TEST_BITNET_LM_HEAD_FWD_DECODED_TERNARY_CACHE", false);
}

bool decodedLmHeadBackwardCacheEnabled() {
    return envBoolOr("QLORA_TEST_BITNET_LM_HEAD_BWD_DECODED_TERNARY_CACHE", false);
}

std::size_t exactLmHeadBackwardTileRows(std::size_t activeComputeT, std::size_t fallback) {
    if (const std::optional<std::size_t> forced = runtimeCpuPolicySize(&CpuBackendRuntimePolicy::exactLmHeadBackwardTileRows)) {
        return *forced;
    }
    const std::size_t promoted =
        (activeComputeT >= 128 && fallback >= 4)
        ? std::max<std::size_t>(fallback, 256)
        : fallback;
    return envSizeTOr("QLORA_TEST_BITNET_LM_HEAD_BWD_TILE_ROWS", promoted);
}

std::size_t exactFusedQkvBackwardTileRows(std::size_t activeComputeT, std::size_t fallback) {
    if (const std::optional<std::size_t> forced = runtimeCpuPolicySize(&CpuBackendRuntimePolicy::exactFusedQkvBackwardTileRows)) {
        return *forced;
    }
    const std::size_t promoted =
        (activeComputeT >= 128 && fallback >= 4)
        ? std::max<std::size_t>(fallback, 32)
        : fallback;
    return envSizeTOr("QLORA_TEST_BITNET_FUSED_QKV_BACKWARD_TILE_ROWS", promoted);
}

struct ExactBackwardTilePolicy {
    std::size_t qkvForwardRows = 0;
    std::size_t qkvBackwardRows = 0;
    std::size_t wdownBackwardRows = 0;
    std::size_t lmHeadBackwardRows = 0;
    std::size_t lmHeadBackwardHChunk = 0;

    static ExactBackwardTilePolicy resolve(std::size_t activeComputeT, std::size_t fallbackT) {
        ExactBackwardTilePolicy policy{};
        policy.qkvForwardRows = exactFusedQkvForwardTileRows(activeComputeT, fallbackT);
        policy.qkvBackwardRows = exactFusedQkvBackwardTileRows(activeComputeT, fallbackT);
        policy.wdownBackwardRows = exactWdownBackwardTileRows(activeComputeT, fallbackT);
        policy.lmHeadBackwardRows = exactLmHeadBackwardTileRows(activeComputeT, fallbackT);
        policy.lmHeadBackwardHChunk = runtimeCpuPolicySize(&CpuBackendRuntimePolicy::lmHeadBackwardHChunk)
            .value_or(envSizeTOr("QLORA_TEST_LMHEAD_BWD_HCHUNK", 64));
        return policy;
    }
};

struct SequenceBackwardPolicy {
    bool usePrefixMaskFastpath = false;
    bool bitnetFfnOverwritesSplitGradBuffers = false;

    static SequenceBackwardPolicy resolve(
        bool useBitNetFfn,
        std::size_t activePrefixT,
        std::size_t activeComputeT)
    {
        SequenceBackwardPolicy policy{};
        policy.usePrefixMaskFastpath =
            activePrefixT != static_cast<std::size_t>(-1) &&
            (prefixMaskFastpathEnabled() || activeComputeT == activePrefixT);
        policy.bitnetFfnOverwritesSplitGradBuffers =
            useBitNetFfn &&
            policy.usePrefixMaskFastpath &&
            activeComputeT == activePrefixT;
        return policy;
    }
};

struct LmHeadBackwardPolicy {
    ExactBackwardTilePolicy exactTiles{};
    bool useQuantBackward = false;
    bool useLossMaskSkip = false;
    bool useRowCompact = false;

    static LmHeadBackwardPolicy resolve(
        const TrainerConfig& cfg,
        bool hasQuantWeights,
        std::size_t activeComputeT,
        std::size_t tTile,
        bool hasLossMask,
        std::size_t supervisedRows)
    {
        LmHeadBackwardPolicy policy{};
        policy.exactTiles = ExactBackwardTilePolicy::resolve(activeComputeT, tTile);
        policy.useQuantBackward =
            cfg.useBitNetQuant &&
            hasQuantWeights &&
            !cfg.bitnetLmHeadFloatBackward &&
            (cfg.bitnetQuantizeLmHead || cfg.bitnetLmHeadQuantBackwardOnly);
        policy.useLossMaskSkip =
            hasLossMask &&
            lmHeadBackwardLossMaskSkipEnabled();
        policy.useRowCompact =
            hasLossMask &&
            !policy.useLossMaskSkip &&
            lmHeadBackwardRowCompactEnabled(activeComputeT, supervisedRows);
        return policy;
    }
};

struct FfnBackwardPolicy {
    ExactBackwardTilePolicy exactTiles{};
    bool useStreamedPrefix = false;
    bool useChunked = false;
    std::size_t chunkRows = 0;
    bool fuseResidualAdd = false;

    static FfnBackwardPolicy resolve(
        bool useBitNetFfn,
        bool bitnetFfnOverwritesSplitGradBuffers,
        bool useRmsnormBwdAccum,
        std::size_t activeComputeT,
        std::size_t tTile)
    {
        FfnBackwardPolicy policy{};
        policy.exactTiles = ExactBackwardTilePolicy::resolve(activeComputeT, tTile);
        policy.useStreamedPrefix =
            bitnetFfnOverwritesSplitGradBuffers &&
            bitnetFfnStreamedPrefixBackwardEnabled();
        policy.useChunked =
            useBitNetFfn &&
            !policy.useStreamedPrefix &&
            exactChunkedFfnBackwardEnabled(activeComputeT);
        policy.chunkRows = exactChunkedFfnBackwardRows(activeComputeT, tTile);
        policy.fuseResidualAdd =
            useBitNetFfn &&
            !useRmsnormBwdAccum &&
            ffnBackwardFuseResidualAddEnabled(activeComputeT);
        return policy;
    }
};

struct AttentionBackwardPolicy {
    ExactBackwardTilePolicy exactTiles{};
    bool runBaseFirst = false;
    bool parallelLora = false;

    static AttentionBackwardPolicy resolve(std::size_t activeComputeT, std::size_t tTile) {
        AttentionBackwardPolicy policy{};
        policy.exactTiles = ExactBackwardTilePolicy::resolve(activeComputeT, tTile);
        policy.runBaseFirst = attentionBaseBackwardFirstEnabled(activeComputeT);
        policy.parallelLora = parallelAttentionLoraBackwardEnabled(activeComputeT);
        return policy;
    }
};

bool ffnBackwardFuseResidualAddEnabled(std::size_t activeComputeT) {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::ffnBackwardFuseResidualAdd)) {
        return *forced;
    }
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_FFN_BWD_FUSE_RESIDUAL_ADD");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return activeComputeT >= 128;
}

bool exactChunkedFfnBackwardEnabled(std::size_t activeComputeT) {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::exactChunkedFfnBackward)) {
        return *forced;
    }
    static const std::optional<bool> cached = envBoolOverride("QLORA_TEST_BITNET_FFN_BWD_CHUNKED");
    if (cached) {
        const std::optional<bool> forced = cached;
        return *forced;
    }

    return false;
}

bool lmHeadForwardLossMaskSkipEnabled() {
    static const bool cached = envBoolOr("QLORA_TEST_LM_HEAD_FWD_LOSS_MASK_SKIP", true);
    return cached;
}

bool lmHeadForwardRowCompactEnabled(std::size_t activeComputeT, std::size_t supervisedRows) {
    static const bool cached = envBoolOr("QLORA_TEST_LM_HEAD_FWD_ROW_COMPACT", false);
    return cached && supervisedRows > 0 && supervisedRows < activeComputeT;
}

bool lmHeadForwardMaskedRowClearEnabled() {
    static const bool cached = envBoolOr("QLORA_TEST_LM_HEAD_FWD_MASKED_ROW_CLEAR", false);
    return cached;
}

bool lmHeadBackwardLossMaskSkipEnabled() {
    if (const std::optional<bool> forced = runtimeCpuPolicyBool(&CpuBackendRuntimePolicy::lmHeadBackwardLossMaskSkip)) {
        return *forced;
    }
    static const bool cached = envBoolOr("QLORA_TEST_LM_HEAD_BWD_LOSS_MASK_SKIP", false);
    return cached;
}

std::size_t exactChunkedFfnBackwardRows(std::size_t activeComputeT, std::size_t fallback) {
    if (const std::optional<std::size_t> forced = runtimeCpuPolicySize(&CpuBackendRuntimePolicy::exactChunkedFfnBackwardRows)) {
        return *forced;
    }
    const std::size_t promoted =
        (exactChunkedFfnBackwardEnabled(activeComputeT) && fallback >= 4)
        ? std::max<std::size_t>(fallback, 32)
        : fallback;
    return envSizeTOr("QLORA_TEST_BITNET_FFN_BWD_CHUNK_ROWS", promoted);
}

std::size_t backwardScratchRows(std::size_t activeComputeT, std::size_t tTile) {
    const ExactBackwardTilePolicy tilePolicy =
        ExactBackwardTilePolicy::resolve(activeComputeT, tTile);
    std::size_t rows = tTile;

    if (bitnetFfnStreamedPrefixBackwardEnabled()) {
        rows = std::max(rows, bitnetFfnStreamedPrefixBackwardChunkT(32));
    }

    // The long-sequence exact FFN dual-backward path can quantize more rows
    // than cfg_.tTile when single-team master quant is active.
    rows = std::max(rows, bitnetDualBackwardQuantChunkRows(4));
    rows = std::max(rows, tilePolicy.lmHeadBackwardRows);
    rows = std::max(rows, tilePolicy.wdownBackwardRows);
    rows = std::max(rows, tilePolicy.qkvBackwardRows);
    rows = std::max(rows, exactChunkedFfnBackwardRows(activeComputeT, tTile));

    return rows;
}

bool tl2QEnabled(const TrainerConfig& cfg) {
    return envBoolOr("QLORA_TEST_BITNET_TL2_Q", cfg.bitnetTl2Qkv);
}

bool tl2KEnabled(const TrainerConfig& cfg) {
    return envBoolOr("QLORA_TEST_BITNET_TL2_K", cfg.bitnetTl2Qkv);
}

bool tl2VEnabled(const TrainerConfig& cfg) {
    return envBoolOr("QLORA_TEST_BITNET_TL2_V", cfg.bitnetTl2Qkv);
}

bool tl2WoEnabled(const TrainerConfig& cfg) {
    return envBoolOr("QLORA_TEST_BITNET_TL2_WO_ONLY", cfg.bitnetTl2Wo);
}

bool tl2GateEnabled(const TrainerConfig& cfg) {
    return envBoolOr("QLORA_TEST_BITNET_TL2_GATE", cfg.bitnetTl2Ffn);
}

bool tl2UpEnabled(const TrainerConfig& cfg) {
    return envBoolOr("QLORA_TEST_BITNET_TL2_UP", cfg.bitnetTl2Ffn);
}

bool tl2DownEnabled(const TrainerConfig& cfg) {
    return envBoolOr("QLORA_TEST_BITNET_TL2_DOWN", cfg.bitnetTl2Ffn);
}

bool tl2Supported(const TrainerConfig& cfg, std::size_t rows, std::size_t cols) {
    return cfg.useBitNetQuant &&
        !cfg.bitnetHadamard32 &&
        cfg.bitnetTl2Bk == 96 &&
        (rows % 32) == 0 &&
        (cols % 32) == 0;
}

bool tl2GeneratedSupported(const TrainerConfig& cfg, std::size_t rows, std::size_t cols) {
    if (!tl2Supported(cfg, rows, cols)) return false;
    return (rows == 6912 && cols == 2560) ||
        (rows == 2560 && cols == 6912) ||
        (rows == 2560 && cols == 2560) ||
        (rows == 2560 && cols == 640);
}

double elapsedMs(ProfileClock::time_point start) {
    return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}

BackendProfileState& backendProfileState() {
    static BackendProfileState state{};
    return state;
}

void printBackendProfile() {
    auto& p = backendProfileState();
    ++p.profiledBackwardCalls;
    std::cout
        << "[PROFILE backend " << p.profiledBackwardCalls << "] "
        << "fwd(ms): embed=" << p.fwdEmbedMs
        << " attn_norm=" << p.fwdAttnNormMs
        << " attn_prep=" << p.fwdAttnPrepMs
        << " qkv=" << p.fwdQkvMs
        << " lora=" << p.fwdLoraMs
        << " rope=" << p.fwdRopeMs
        << " attn=" << p.fwdAttnMs
        << " wo=" << p.fwdWoMs
        << " residual=" << p.fwdResidualMs
        << " ffn_norm=" << p.fwdFfnNormMs
        << " ffn=" << p.fwdFfnMs
        << " cache=" << p.fwdCacheMs
        << " final_norm=" << p.fwdFinalNormMs
        << " lm_head=" << p.fwdLmHeadMs
        << " | bwd(ms): lm_head=" << p.bwdLmHeadMs
        << " final_norm=" << p.bwdFinalNormMs
        << " clear=" << p.bwdClearMs
        << " restore=" << p.bwdRestoreMs
        << " ffn=" << p.bwdFfnMs
        << " ffn_wdown=" << p.bwdFfnWdownMs
        << " ffn_silu=" << p.bwdFfnSiluMs
        << " ffn_dual=" << p.bwdFfnDualMs
        << " ffn_norm=" << p.bwdFfnNormMs
        << " wo=" << p.bwdWoMs
        << " attn=" << p.bwdAttnMs
        << " rope=" << p.bwdRopeMs
        << " lora=" << p.bwdLoraMs
        << " qkv=" << p.bwdQkvMs
        << " residual=" << p.bwdResidualMs
        << '\n';

    p = BackendProfileState{};
}

} // namespace

static void load_raw_bin(const std::string& filepath, std::vector<float>& buffer) {
    std::ifstream in(filepath, std::ios::binary | std::ios::ate);
    if (!in.is_open()) throw std::runtime_error("Cannot open weight file: " + filepath);

    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (size != buffer.size() * sizeof(float)) {
        throw std::runtime_error("Size mismatch in " + filepath + " expected: " + std::to_string(buffer.size() * sizeof(float)) + " got: " + std::to_string(size));
    }

    if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read " + filepath);
    }
}

ModelShape CpuKernelBackend::loadShapeFromConfig(const std::string& modelPath)
{
    const std::string configFile = modelPath + "/config.json";

    std::ifstream in(configFile);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open config.json at " + configFile);
    }

    nlohmann::json j;
    in >> j;

    ModelShape shape{};
    shape.vocabSize = j.at("vocab_size").get<std::size_t>();
    shape.hiddenSize = j.at("hidden_size").get<std::size_t>();
    shape.nLayers = j.at("num_hidden_layers").get<std::size_t>();
    shape.nHeads = j.at("num_attention_heads").get<std::size_t>();
    shape.nKvHeads = j.value("num_key_value_heads", shape.nHeads);
    shape.ffnSize = j.value("intermediate_size", shape.hiddenSize * 4);
    shape.ropeTheta = j.value("rope_theta", 10000.0f);
    shape.ropeDim = j.value("rope_dim", shape.hiddenSize / shape.nHeads);
    return shape;
}

static bool save_i2s_cache(const std::string& path, const BitNetMatI2S& W) {
    try {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) return false;

        const uint32_t magic = 0x31533249u; // "I2S1" little-endian
        const uint32_t version = 3u;
        const uint64_t rows = static_cast<uint64_t>(W.rows);
        const uint64_t cols = static_cast<uint64_t>(W.cols);
        const uint64_t qk = static_cast<uint64_t>(W.qk);
        const uint64_t jq = static_cast<uint64_t>(W.jq);
        const uint32_t scaleLayout = static_cast<uint32_t>(W.scaleLayout);
        const uint64_t q2N = static_cast<uint64_t>(W.q2.size());
        const uint64_t q2TN = static_cast<uint64_t>(W.q2T.size());
        const uint64_t sN = static_cast<uint64_t>(W.w_scale_tile.size());
        const uint64_t sTN = static_cast<uint64_t>(W.w_scale_tileT.size());
        const uint64_t sB8N = static_cast<uint64_t>(W.w_scale_tile_broadcast8.size());
        const uint64_t sTB8N = static_cast<uint64_t>(W.w_scale_tileT_broadcast8.size());

        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        out.write(reinterpret_cast<const char*>(&qk), sizeof(qk));
        out.write(reinterpret_cast<const char*>(&jq), sizeof(jq));
        out.write(reinterpret_cast<const char*>(&scaleLayout), sizeof(scaleLayout));
        out.write(reinterpret_cast<const char*>(&q2N), sizeof(q2N));
        out.write(reinterpret_cast<const char*>(&q2TN), sizeof(q2TN));
        out.write(reinterpret_cast<const char*>(&sN), sizeof(sN));
        out.write(reinterpret_cast<const char*>(&sTN), sizeof(sTN));
        out.write(reinterpret_cast<const char*>(&sB8N), sizeof(sB8N));
        out.write(reinterpret_cast<const char*>(&sTB8N), sizeof(sTB8N));

        if (q2N) out.write(reinterpret_cast<const char*>(W.q2.data()), static_cast<std::streamsize>(q2N));
        if (q2TN) out.write(reinterpret_cast<const char*>(W.q2T.data()), static_cast<std::streamsize>(q2TN));
        if (sN) out.write(reinterpret_cast<const char*>(W.w_scale_tile.data()), static_cast<std::streamsize>(sN * sizeof(float)));
        if (sTN) out.write(reinterpret_cast<const char*>(W.w_scale_tileT.data()), static_cast<std::streamsize>(sTN * sizeof(float)));
        if (sB8N) out.write(reinterpret_cast<const char*>(W.w_scale_tile_broadcast8.data()), static_cast<std::streamsize>(sB8N * sizeof(float)));
        if (sTB8N) out.write(reinterpret_cast<const char*>(W.w_scale_tileT_broadcast8.data()), static_cast<std::streamsize>(sTB8N * sizeof(float)));

        return static_cast<bool>(out);
    }
    catch (...) {
        return false;
    }
}

static bool load_i2s_cache(
    const std::string& path,
    BitNetMatI2S& W,
    std::size_t expectedRows,
    std::size_t expectedCols,
    std::size_t expectedQk,
    std::size_t expectedJq,
    bool expectedHadamard32)
{
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;

        uint32_t magic = 0, version = 0;
        uint64_t rows = 0, cols = 0, qk = 0, jq = 0;
        uint32_t scaleLayout = static_cast<uint32_t>(BitNetI2SScaleLayout::JMajorContiguousRows);
        uint64_t q2N = 0, q2TN = 0, sN = 0, sTN = 0, sB8N = 0, sTB8N = 0;

        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        in.read(reinterpret_cast<char*>(&qk), sizeof(qk));
        in.read(reinterpret_cast<char*>(&jq), sizeof(jq));
        if (version >= 2u) {
            in.read(reinterpret_cast<char*>(&scaleLayout), sizeof(scaleLayout));
        }
        in.read(reinterpret_cast<char*>(&q2N), sizeof(q2N));
        in.read(reinterpret_cast<char*>(&q2TN), sizeof(q2TN));
        in.read(reinterpret_cast<char*>(&sN), sizeof(sN));
        in.read(reinterpret_cast<char*>(&sTN), sizeof(sTN));
        if (version >= 3u) {
            in.read(reinterpret_cast<char*>(&sB8N), sizeof(sB8N));
            in.read(reinterpret_cast<char*>(&sTB8N), sizeof(sTB8N));
        }
        if (!in) return false;

        if (magic != 0x31533249u || (version != 1u && version != 2u && version != 3u)) return false;
        if (rows != expectedRows || cols != expectedCols || qk != expectedQk || jq != expectedJq) return false;
        if (scaleLayout != static_cast<uint32_t>(BitNetI2SScaleLayout::JMajorContiguousRows)
            && scaleLayout != static_cast<uint32_t>(BitNetI2SScaleLayout::JMajorBroadcastRows)) return false;

        W.rows = static_cast<std::size_t>(rows);
        W.cols = static_cast<std::size_t>(cols);
        W.hadamard32 = expectedHadamard32;
        W.qk = static_cast<std::size_t>(qk);
        W.jq = static_cast<std::size_t>(jq);
        W.scaleLayout = static_cast<BitNetI2SScaleLayout>(scaleLayout);
        W.q2.resize(static_cast<std::size_t>(q2N));
        W.q2T.resize(static_cast<std::size_t>(q2TN));
        W.w_scale_tile.resize(static_cast<std::size_t>(sN));
        W.w_scale_tileT.resize(static_cast<std::size_t>(sTN));
        W.w_scale_tile_broadcast8.resize(static_cast<std::size_t>(sB8N));
        W.w_scale_tileT_broadcast8.resize(static_cast<std::size_t>(sTB8N));

        if (q2N) in.read(reinterpret_cast<char*>(W.q2.data()), static_cast<std::streamsize>(q2N));
        if (q2TN) in.read(reinterpret_cast<char*>(W.q2T.data()), static_cast<std::streamsize>(q2TN));
        if (sN) in.read(reinterpret_cast<char*>(W.w_scale_tile.data()), static_cast<std::streamsize>(sN * sizeof(float)));
        if (sTN) in.read(reinterpret_cast<char*>(W.w_scale_tileT.data()), static_cast<std::streamsize>(sTN * sizeof(float)));
        if (sB8N) in.read(reinterpret_cast<char*>(W.w_scale_tile_broadcast8.data()), static_cast<std::streamsize>(sB8N * sizeof(float)));
        if (sTB8N) in.read(reinterpret_cast<char*>(W.w_scale_tileT_broadcast8.data()), static_cast<std::streamsize>(sTB8N * sizeof(float)));
        if (!in) return false;

        if (version < 3u
            || W.w_scale_tile_broadcast8.size() < W.expectedBroadcastScaleCount()
            || W.w_scale_tileT_broadcast8.size() < W.expectedBroadcastScaleCount()) {
            rebuild_bitnet_i2s_scale_broadcast_views(W);
        }
        if (!W.validForShape()) return false;
        return true;
    }
    catch (...) {
        return false;
    }
}

static constexpr const char* kI2SCacheRevisionTag = ".rev4";

static inline float sigmoid_scalar(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

static inline float silu_scalar(float x) {
    return x * sigmoid_scalar(x);
}

static inline float silu_derivative_scalar(float x) {
    const float s = sigmoid_scalar(x);
    return s * (1.0f + x * (1.0f - s));
}

static inline void silu_value_and_derivative_scalar(float x, float& value, float& derivative) {
    const float s = sigmoid_scalar(x);
    value = x * s;
    derivative = s * (1.0f + x * (1.0f - s));
}

static inline void accumulate_rmsnorm_weight_grad(
    float* __restrict grad,
    const float* __restrict dy,
    const float* __restrict x,
    std::size_t H,
    float eps)
{
    float ss = 0.0f;
    for (std::size_t h = 0; h < H; ++h) ss += x[h] * x[h];
    const float inv = 1.0f / std::sqrt(ss / float(H) + eps);
    for (std::size_t h = 0; h < H; ++h) {
        grad[h] += dy[h] * x[h] * inv;
    }
}

#if defined(__AVX2__)
static inline __m256 exp256_ps(__m256 x) {
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    const __m256 fx = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(x, log2e), _mm256_set1_ps(0.5f)));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, ln2_hi));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, ln2_lo));

    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));

    const __m256 x2 = _mm256_mul_ps(x, x);
    y = _mm256_fmadd_ps(y, x2, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));

    __m256i emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
}
#endif

static inline void silu_mul_avx2_or_scalar(
    float* __restrict act,
    const float* __restrict gate,
    const float* __restrict up,
    std::size_t n)
{
#if defined(__AVX2__)
    std::size_t i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        const __m256 gatev = _mm256_loadu_ps(gate + i);
        const __m256 upv = _mm256_loadu_ps(up + i);
        const __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), gatev))));
        _mm256_storeu_ps(act + i, _mm256_mul_ps(_mm256_mul_ps(gatev, sig), upv));
    }
    for (; i < n; ++i) act[i] = silu_scalar(gate[i]) * up[i];
#else
    for (std::size_t i = 0; i < n; ++i) act[i] = silu_scalar(gate[i]) * up[i];
#endif
}

static inline void silu_backward_avx2_or_scalar(
    float* __restrict dGate,
    float* __restrict dUp,
    const float* __restrict dAct,
    const float* __restrict gate,
    const float* __restrict up,
    std::size_t n)
{
#if defined(__AVX2__)
    std::size_t i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        const __m256 dActv = _mm256_loadu_ps(dAct + i);
        const __m256 gatev = _mm256_loadu_ps(gate + i);
        const __m256 upv = _mm256_loadu_ps(up + i);
        const __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), gatev))));
        const __m256 siluDeriv = _mm256_mul_ps(sig, _mm256_add_ps(one, _mm256_mul_ps(gatev, _mm256_sub_ps(one, sig))));
        _mm256_storeu_ps(dUp + i, _mm256_mul_ps(dActv, _mm256_mul_ps(gatev, sig)));
        _mm256_storeu_ps(dGate + i, _mm256_mul_ps(_mm256_mul_ps(dActv, upv), siluDeriv));
    }
    for (; i < n; ++i) {
        const float siluG = silu_scalar(gate[i]);
        dUp[i] = dAct[i] * siluG;
        dGate[i] = dAct[i] * up[i] * silu_derivative_scalar(gate[i]);
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        const float siluG = silu_scalar(gate[i]);
        dUp[i] = dAct[i] * siluG;
        dGate[i] = dAct[i] * up[i] * silu_derivative_scalar(gate[i]);
    }
#endif
}

static inline void silu_mul_exact_scalar(
    float* __restrict act,
    const float* __restrict gate,
    const float* __restrict up,
    std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) act[i] = silu_scalar(gate[i]) * up[i];
}

static inline void silu_backward_exact_scalar(
    float* __restrict dGate,
    float* __restrict dUp,
    const float* __restrict dAct,
    const float* __restrict gate,
    const float* __restrict up,
    std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i) {
        float siluG = 0.0f;
        float siluDeriv = 0.0f;
        silu_value_and_derivative_scalar(gate[i], siluG, siluDeriv);
        dUp[i] = dAct[i] * siluG;
        dGate[i] = dAct[i] * up[i] * siluDeriv;
    }
}

static inline void silu_backward_exact_chunked(
    float* __restrict dGate,
    float* __restrict dUp,
    const float* __restrict dAct,
    const float* __restrict gate,
    const float* __restrict up,
    std::size_t T,
    std::size_t F,
    const int* __restrict attentionMask)
{
    static constexpr std::size_t kChunk = 256;
    const std::size_t nChunks = (F + kChunk - 1) / kChunk;

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
        for (int c_i = 0; c_i < static_cast<int>(nChunks); ++c_i) {
            const std::size_t t = static_cast<std::size_t>(t_i);
            const std::size_t chunk = static_cast<std::size_t>(c_i);
            const std::size_t f0 = chunk * kChunk;
            const std::size_t fCount = std::min(kChunk, F - f0);
            float* dGate_t = dGate + t * F + f0;
            float* dUp_t = dUp + t * F + f0;

            if (attentionMask && attentionMask[t] == 0) {
                std::fill(dGate_t, dGate_t + fCount, 0.0f);
                std::fill(dUp_t, dUp_t + fCount, 0.0f);
                continue;
            }

            silu_backward_exact_scalar(
                dGate_t,
                dUp_t,
                dAct + t * F + f0,
                gate + t * F + f0,
                up + t * F + f0,
                fCount);
        }
    }
}

static inline void apply_rope_forward_inplace_cached(
    float* vec,
    std::size_t headDim,
    std::size_t ropeDim,
    std::size_t pos,
    const float* cosTable,
    const float* sinTable)
{
    if (!vec || !cosTable || !sinTable || headDim == 0 || ropeDim < 2) return;
    const std::size_t rd = std::min(headDim, ropeDim) & ~std::size_t(1);
    if (rd == 0) return;

    const std::size_t pairs = ropeDim / 2;
    const float* const cosRow = cosTable + pos * pairs;
    const float* const sinRow = sinTable + pos * pairs;
    for (std::size_t pair = 0; pair < (rd / 2); ++pair) {
        const std::size_t i = pair * 2;
        const float c = cosRow[pair];
        const float s = sinRow[pair];
        const float x0 = vec[i + 0];
        const float x1 = vec[i + 1];
        vec[i + 0] = x0 * c - x1 * s;
        vec[i + 1] = x0 * s + x1 * c;
    }
}

static inline void apply_rope_backward_inplace_cached(
    float* gradVec,
    std::size_t headDim,
    std::size_t ropeDim,
    std::size_t pos,
    const float* cosTable,
    const float* sinTable)
{
    if (!gradVec || !cosTable || !sinTable || headDim == 0 || ropeDim < 2) return;
    const std::size_t rd = std::min(headDim, ropeDim) & ~std::size_t(1);
    if (rd == 0) return;

    const std::size_t pairs = ropeDim / 2;
    const float* const cosRow = cosTable + pos * pairs;
    const float* const sinRow = sinTable + pos * pairs;
    for (std::size_t pair = 0; pair < (rd / 2); ++pair) {
        const std::size_t i = pair * 2;
        const float c = cosRow[pair];
        const float s = sinRow[pair];
        const float g0 = gradVec[i + 0];
        const float g1 = gradVec[i + 1];
        gradVec[i + 0] = g0 * c + g1 * s;
        gradVec[i + 1] = -g0 * s + g1 * c;
    }
}

CpuKernelBackend::CpuKernelBackend(TrainerConfig cfg)
    : cfg_{ cfg }
{
    if (cfg_.modelPath.empty()) throw std::runtime_error("modelPath is empty");
    if (cfg_.loraRank == 0)     throw std::runtime_error("loraRank must be > 0");
    if (cfg_.vTile == 0)        throw std::runtime_error("vTile must be > 0");
    if (cfg_.maxSeqLen == 0)    throw std::runtime_error("maxSeqLen must be > 0");

    shape_ = loadShapeFromConfig(cfg_.modelPath);

    if (shape_.hiddenSize % shape_.nHeads != 0) {
        throw std::runtime_error("hiddenSize must be divisible by nHeads");
    }
    if (shape_.nKvHeads == 0) {
        throw std::runtime_error("num_key_value_heads must be > 0");
    }
    if (shape_.nKvHeads > shape_.nHeads) {
        throw std::runtime_error("num_key_value_heads cannot exceed num_attention_heads");
    }
    if (shape_.nHeads % shape_.nKvHeads != 0) {
        throw std::runtime_error("num_attention_heads must be divisible by num_key_value_heads (GQA)");
    }
    if (shape_.ffnSize == 0) {
        throw std::runtime_error("intermediate_size must be > 0");
    }
    if (shape_.ropeDim == 0 || shape_.ropeDim > shape_.headDim()) {
        throw std::runtime_error("rope_dim must be in [1, head_dim]");
    }
    if ((shape_.ropeDim % 2) != 0) {
        throw std::runtime_error("rope_dim must be even");
    }
    if (!(shape_.ropeTheta > 0.0f)) {
        throw std::runtime_error("rope_theta must be > 0");
    }

    if (cfg_.forcedLayers > 0) {
        if (cfg_.forcedLayers > shape_.nLayers) {
            throw std::runtime_error("forcedLayers exceeds model num_hidden_layers");
        }
        shape_.nLayers = cfg_.forcedLayers;
    }

    const std::size_t H = shape_.hiddenSize;
    const std::size_t V = shape_.vocabSize;
    const std::size_t F = shape_.ffnSize;
    const std::size_t D = shape_.headDim();
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);
    const std::size_t Tm = static_cast<std::size_t>(cfg_.maxSeqLen);
    const std::size_t kvDim = shape_.nKvHeads * D;
    const std::size_t nLoRALayers = shape_.nLayers;
    const std::size_t nHeads = shape_.nHeads;
    const bool canUseI2S = (H % 32 == 0);
    const bool streamQuantizedDenseWeights =
        cfg_.releaseQuantizedDenseWeights &&
        cfg_.useBitNetQuant &&
        canUseI2S;

    embeddings_.resize(V * H);
    W_out_.resize(H * V);

    const std::size_t ropePairs = shape_.ropeDim / 2;
    ropeCosTable_.resize(Tm * ropePairs);
    ropeSinTable_.resize(Tm * ropePairs);
    for (std::size_t pos = 0; pos < Tm; ++pos) {
        for (std::size_t pair = 0; pair < ropePairs; ++pair) {
            const std::size_t i = pair * 2;
            const float invFreq = std::pow(shape_.ropeTheta, -static_cast<float>(i) / static_cast<float>(shape_.ropeDim));
            const float angle = static_cast<float>(pos) * invFreq;
            ropeCosTable_[pos * ropePairs + pair] = std::cos(angle);
            ropeSinTable_[pos * ropePairs + pair] = std::sin(angle);
        }
    }

    layers_.resize(nLoRALayers);
    loras_.resize(nLoRALayers);
    loraGrads_.resize(nLoRALayers);

    auto init_mat_HH = [&](std::vector<float>& W) { W.resize(H * H); };
    auto init_mat_HKV = [&](std::vector<float>& W) { W.resize(H * kvDim); };
    auto init_mat_HF = [&](std::vector<float>& W) { W.resize(H * F); };
    auto init_mat_FH = [&](std::vector<float>& W) { W.resize(F * H); };
    auto init_vec_H = [&](std::vector<float>& w) { w.resize(H); };

    for (std::size_t L = 0; L < nLoRALayers; ++L) {
        if (!streamQuantizedDenseWeights) {
            init_mat_HH(layers_[L].Wq);
            init_mat_HKV(layers_[L].Wk);
            init_mat_HKV(layers_[L].Wv);
            init_mat_HH(layers_[L].Wo);
            init_mat_HF(layers_[L].Wgate);
            init_mat_HF(layers_[L].Wup);
            init_mat_FH(layers_[L].Wdown);
        }
        init_vec_H(layers_[L].rms_attn);
        init_vec_H(layers_[L].rms_ffn);

        loras_[L].Aq.resize(R * H);
        loras_[L].Bq.resize(R * H);
        loras_[L].Ak.resize(R * H);
        loras_[L].Bk.resize(R * kvDim);
        loras_[L].Av.resize(R * H);
        loras_[L].Bv.resize(R * kvDim);
        loras_[L].Ao.resize(R * H);
        loras_[L].Bo.resize(R * H);
        if (cfg_.loraTargetFfnDown) {
            loras_[L].Afd.resize(R * F);
            loras_[L].Bfd.resize(R * H);
        }
        if (cfg_.loraTargetFfnGateUp) {
            loras_[L].Afg.resize(R * H);
            loras_[L].Bfg.resize(R * F);
            loras_[L].Afu.resize(R * H);
            loras_[L].Bfu.resize(R * F);
        }
        if (cfg_.useDoRA) {
            if (cfg_.loraTargetQ) {
                loras_[L].Mq.resize(H); loras_[L].Cq.resize(H); loras_[L].Rq.resize(H);
                loraGrads_[L].gradMq.assign(H, 0.0f);
            }
            if (cfg_.loraTargetK) {
                loras_[L].Mk.resize(kvDim); loras_[L].Ck.resize(kvDim); loras_[L].Rk.resize(kvDim);
                loraGrads_[L].gradMk.assign(kvDim, 0.0f);
            }
            if (cfg_.loraTargetV) {
                loras_[L].Mv.resize(kvDim); loras_[L].Cv.resize(kvDim); loras_[L].Rv.resize(kvDim);
                loraGrads_[L].gradMv.assign(kvDim, 0.0f);
            }
            if (cfg_.loraTargetO) {
                loras_[L].Mo.resize(H); loras_[L].Co.resize(H); loras_[L].Ro.resize(H);
                loraGrads_[L].gradMo.assign(H, 0.0f);
            }
            if (cfg_.loraTargetFfnDown) {
                loras_[L].Mfd.resize(H); loras_[L].Cfd.resize(H); loras_[L].Rfd.resize(H);
                loraGrads_[L].gradMfd.assign(H, 0.0f);
            }
            if (cfg_.loraTargetFfnGateUp) {
                loras_[L].Mfg.resize(F); loras_[L].Cfg.resize(F); loras_[L].Rfg.resize(F);
                loras_[L].Mfu.resize(F); loras_[L].Cfu.resize(F); loras_[L].Rfu.resize(F);
                loraGrads_[L].gradMfg.assign(F, 0.0f);
                loraGrads_[L].gradMfu.assign(F, 0.0f);
            }
        }

        loraGrads_[L].gradAq.assign(R * H, 0.0f);
        loraGrads_[L].gradBq.assign(R * H, 0.0f);
        loraGrads_[L].gradAk.assign(R * H, 0.0f);
        loraGrads_[L].gradBk.assign(R * kvDim, 0.0f);
        loraGrads_[L].gradAv.assign(R * H, 0.0f);
        loraGrads_[L].gradBv.assign(R * kvDim, 0.0f);
        loraGrads_[L].gradAo.assign(R * H, 0.0f);
        loraGrads_[L].gradBo.assign(R * H, 0.0f);
        if (cfg_.loraTargetFfnDown) {
            loraGrads_[L].gradAfd.assign(R * F, 0.0f);
            loraGrads_[L].gradBfd.assign(R * H, 0.0f);
        }
        if (cfg_.loraTargetFfnGateUp) {
            loraGrads_[L].gradAfg.assign(R * H, 0.0f);
            loraGrads_[L].gradBfg.assign(R * F, 0.0f);
            loraGrads_[L].gradAfu.assign(R * H, 0.0f);
            loraGrads_[L].gradBfu.assign(R * F, 0.0f);
        }
        if (cfg_.trainLayerRmsNorm) {
            loraGrads_[L].gradRmsAttn.assign(H, 0.0f);
            loraGrads_[L].gradRmsFfn.assign(H, 0.0f);
        }
    }
    if (cfg_.loraTargetLmHead) {
        lmHeadLora_.A.resize(R * H);
        lmHeadLora_.B.resize(R * V);
        lmHeadLoraGrads_.gradA.assign(R * H, 0.0f);
        lmHeadLoraGrads_.gradB.assign(R * V, 0.0f);
        if (cfg_.useDoRA) {
            lmHeadLora_.M.resize(V);
            lmHeadLora_.C.resize(V);
            lmHeadLora_.R.resize(V);
            lmHeadLoraGrads_.gradM.assign(V, 0.0f);
        }
    }
    if (cfg_.trainLmHeadBias) {
        lmHeadBias_.assign(V, 0.0f);
        lmHeadBiasGrad_.assign(V, 0.0f);
    }
    if (cfg_.trainFinalNormAffine) {
        finalNormScale_.assign(H, 1.0f);
        finalNormBias_.assign(H, 0.0f);
        finalNormScaleGrad_.assign(H, 0.0f);
        finalNormBiasGrad_.assign(H, 0.0f);
        xnorm_final_base_cache_.resize(Tm * H);
    }

    tmpQ_cache_.resize(nLoRALayers * Tm * R);
    tmpK_cache_.resize(nLoRALayers * Tm * R);
    tmpV_cache_.resize(nLoRALayers * Tm * R);
    tmpO_cache_.resize(nLoRALayers * Tm * R);
    tmpFfnDown_cache_.resize(nLoRALayers * Tm * R);
    if (cfg_.loraTargetFfnGateUp) {
        tmpFfnGate_cache_.resize(nLoRALayers * Tm * R);
        tmpFfnUp_cache_.resize(nLoRALayers * Tm * R);
        xnorm_ffn_cache_.resize(nLoRALayers * Tm * H);
    }
    if (cfg_.loraTargetLmHead) {
        tmpLmHead_cache_.resize(Tm * R);
        dTmpLmHead_cache_.resize(Tm * R);
    }
    xnorm_cache_.resize(nLoRALayers * Tm * H);
    q_cache_.resize(nLoRALayers * Tm * H);
    k_cache_.resize(nLoRALayers * Tm * kvDim);
    v_cache_.resize(nLoRALayers * Tm * kvDim);
    if (cfg_.loraTargetO) {
        ctx_cache_.resize(nLoRALayers * Tm * H);
    }
    if (attentionProbCacheEnabled(Tm, cfg_.cacheAttentionProbs)) {
        probs_cache_.resize(nLoRALayers * nHeads * Tm * Tm);
    }
    x_mid_cache_.resize(nLoRALayers * Tm * H);
    gate_cache_.resize(nLoRALayers * Tm * F);
    up_cache_.resize(nLoRALayers * Tm * F);
    if (cfg_.loraTargetFfnDown) {
        ffn_act_cache_.resize(nLoRALayers * Tm * F);
    }
    if (cfg_.useDoRA && cfg_.loraTargetQ && !cfg_.doraRecomputeQKPreRope) {
        q_dora_prerope_cache_.resize(nLoRALayers * Tm * H);
    }
    if (cfg_.useDoRA && cfg_.loraTargetK && !cfg_.doraRecomputeQKPreRope) {
        k_dora_prerope_cache_.resize(nLoRALayers * Tm * kvDim);
    }
    if (cfg_.useDoRA && cfg_.doraRecomputeQKPreRope && cfg_.loraTargetQ) {
        q_dora_prerope_recompute_.resize(Tm * H);
    }
    if (cfg_.useDoRA && cfg_.doraRecomputeQKPreRope && cfg_.loraTargetK) {
        k_dora_prerope_recompute_.resize(Tm * kvDim);
    }

    x_.resize(Tm * H);
    xnorm_.resize(Tm * H);
    q_.resize(Tm * H);
    k_.resize(Tm * kvDim);
    v_.resize(Tm * kvDim);
    ctx_.resize(Tm * H);
    attn_.resize(Tm * H);

    scores_.resize(nHeads * Tm * Tm);
    probs_.resize(nHeads * Tm * Tm);

    dTmpRank_.resize(R);
#if defined(_OPENMP)
    const std::size_t attnLoraScratchThreads =
        static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#else
    const std::size_t attnLoraScratchThreads = 1;
#endif
    if (cfg_.loraTargetQ && cfg_.loraTargetV && !cfg_.loraTargetK && !cfg_.loraTargetO) {
        attnLoraGradAqLocal_.resize(attnLoraScratchThreads * R * H);
        attnLoraGradBqLocal_.resize(attnLoraScratchThreads * R * H);
        attnLoraGradAvLocal_.resize(attnLoraScratchThreads * R * H);
        attnLoraGradBvLocal_.resize(attnLoraScratchThreads * R * kvDim);
        attnLoraTmpRankLocal_.resize(attnLoraScratchThreads * R);
    }

    dX_.resize(Tm * H);
    dAttn_.resize(Tm * H);
    dCtx_.resize(Tm * H);
    dQ_.resize(Tm * H);
    dK_.resize(Tm * kvDim);
    dV_.resize(Tm * kvDim);
    dXnorm_.resize(Tm * H);
    dXin_.resize(Tm * H);
    dXmid_.resize(Tm * H);

    x_mid_.resize(Tm * H);
    xnorm_ffn_.resize(Tm * H);
    gate_.resize(Tm * F);
    up_.resize(Tm * F);
    ffn_act_.resize(Tm * F);
    ffn_out_.resize(Tm * H);

    dFfnOut_.resize(Tm * H);
    dAct_.resize(Tm * F);
    dGate_.resize(Tm * F);
    dUp_.resize(Tm * F);
    dXnormFfn_.resize(Tm * H);

    rms_final_.resize(H);
    if (cfg_.trainFinalRmsNorm) {
        rmsFinalGrad_.assign(H, 0.0f);
    }
    xnorm_final_.resize(Tm * H);
    dXnorm_final_.resize(Tm * H);

    x_ckpt_.resize((nLoRALayers + 1) * Tm * H);
    tmpdP_.resize(Tm);
    lmHeadRowsScratch_.reserve(Tm);

    // ── Pre-size persistent scratch buffers ──────────────────────────────────
    // forward x8/xScale: quantises input rows of width H, tTile rows at a time.
    //   x8 needs tTile * H bytes.
    //   xScale needs tTile * nKBlocks where nKBlocks = ceil(H / qk).
    // backward dY8/dYScale: quantises gradient rows whose width is the *output*
    //   dimension of the weight matrix. The widest output dim across all matrices
    //   is V (LM head). The scale count per row is nJBlocks = ceil(out_dim / jq).
    //   Largest: ceil(V / jq) for the LM head.  For layer matrices (out_dim = H)
    //   ceil(H / jq) is also needed (Wo backward).  Both are covered by ceil(V/jq).
    const bool useBitNetFfn = ffnBitnetEnabled();
    const std::size_t ffnQk = (cfg_.bitnetFfnQK == 0) ? cfg_.bitnetQK : cfg_.bitnetFfnQK;
    const std::size_t maxFwdInputDim = useBitNetFfn ? std::max(H, F) : H;
    const std::size_t attnKBlocks = (H + cfg_.bitnetQK - 1) / cfg_.bitnetQK;
    const std::size_t ffnInputKBlocks = useBitNetFfn ? (H + ffnQk - 1) / ffnQk : 0;
    const std::size_t ffnActKBlocks = useBitNetFfn ? (F + ffnQk - 1) / ffnQk : 0;
    const std::size_t maxKBlocks = std::max(attnKBlocks, std::max(ffnInputKBlocks, ffnActKBlocks));
    const std::size_t maxJBlocks = (V + cfg_.bitnetJQ - 1) / cfg_.bitnetJQ; // for dY scale bwd (V >= H)
    const std::size_t maxAuxBwdDim = useBitNetFfn ? std::max(kvDim, F) : kvDim;
    const std::size_t auxJBlocks = (maxAuxBwdDim + cfg_.bitnetJQ - 1) / cfg_.bitnetJQ;
    const std::size_t tTileConst = static_cast<std::size_t>(cfg_.tTile);
    const std::size_t fwdScratchRows =
        std::max(tTileConst, exactFusedQkvForwardTileRows(cfg_.maxSeqLen, tTileConst));
    const std::size_t bwdScratchRows = backwardScratchRows(cfg_.maxSeqLen, tTileConst);

    fwdX8Scratch_.resize(fwdScratchRows * maxFwdInputDim);
    fwdXScaleScratch_.resize(fwdScratchRows * maxKBlocks);
    // Backward buffers must cover the widest output dimension and the largest
    // runtime row chunk used by the exact BitNet backward paths.
    bwdDY8Scratch_.resize(bwdScratchRows * V);
    bwdDYScaleScratch_.resize(bwdScratchRows * maxJBlocks);
    bwdDK8Scratch_.resize(bwdScratchRows * maxAuxBwdDim);
    bwdDKScaleScratch_.resize(bwdScratchRows * auxJBlocks);
    bwdDV8Scratch_.resize(bwdScratchRows * maxAuxBwdDim);
    bwdDVScaleScratch_.resize(bwdScratchRows * auxJBlocks);

    std::mt19937 rng(42);
    std::normal_distribution<float> wdist(0.0f, 0.02f);

    std::string weight_dir = cfg_.modelPath + "/weights";
    const std::string pissa_dir = cfg_.pissaPath;
    if (cfg_.usePiSSA) {
        if (cfg_.useDoRA) {
            throw std::runtime_error("PiSSA and DoRA are not supported together in this reference path");
        }
        if (pissa_dir.empty()) {
            throw std::runtime_error("usePiSSA requires QLORA_TEST_PISSA_PATH");
        }
        if (!std::filesystem::exists(pissa_dir)) {
            throw std::runtime_error("PiSSA path does not exist: " + pissa_dir);
        }
        std::cout << "PiSSA: enabled; loading residual weights and A/B initializers from "
            << pissa_dir << "\n";
    }
    auto pissaLayerStem = [&](std::size_t L, const std::string& name) {
        return pissa_dir + "/layer_" + std::to_string(L) + "_" + name;
    };
    auto pissaRootStem = [&](const std::string& name) {
        return pissa_dir + "/" + name;
    };
    auto pissaTargetActive = [&](const std::string& name) {
        if (name == "Wq") return cfg_.loraTargetQ;
        if (name == "Wk") return cfg_.loraTargetK;
        if (name == "Wv") return cfg_.loraTargetV;
        if (name == "Wo") return cfg_.loraTargetO;
        if (name == "Wdown") return cfg_.loraTargetFfnDown;
        if (name == "Wgate" || name == "Wup") return cfg_.loraTargetFfnGateUp;
        if (name == "W_out") return cfg_.loraTargetLmHead;
        return false;
    };
    auto pissaHasResidual = [&](const std::string& stem) {
        return cfg_.usePiSSA && std::filesystem::exists(stem + ".residual.bin");
    };
    auto loadLayerWeight = [&](std::size_t L, const std::string& name, std::vector<float>& dst) {
        const std::string stem = pissaLayerStem(L, name);
        if (pissaTargetActive(name) && pissaHasResidual(stem)) {
            load_raw_bin(stem + ".residual.bin", dst);
        }
        else {
            load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_" + name + ".bin", dst);
        }
    };
    auto layerCacheStem = [&](std::size_t L, const std::string& name) {
        const std::string stem = pissaLayerStem(L, name);
        return (pissaTargetActive(name) && pissaHasResidual(stem))
            ? stem
            : (weight_dir + "/layer_" + std::to_string(L) + "_" + name);
    };

    std::cout << "Loading real FP32 weights from disk...\n";
    std::cout << "Architecture: " << nLoRALayers << " Layers | H=" << H
        << " | kvDim=" << kvDim << " | ffnDim=" << F
        << " | ropeDim=" << shape_.ropeDim << "\n";

    load_raw_bin(weight_dir + "/embeddings.bin", embeddings_);
    const std::string pissaLmHeadStem = pissaRootStem("W_out");
    if (pissaTargetActive("W_out") && pissaHasResidual(pissaLmHeadStem)) {
        load_raw_bin(pissaLmHeadStem + ".residual.bin", W_out_);
    }
    else {
        load_raw_bin(weight_dir + "/W_out.bin", W_out_);
    }
    load_raw_bin(weight_dir + "/rms_final.bin", rms_final_);

    // ── Load weights and quantize ─────────────────────────────────────────────
    // On a cold start (no cache files) all nLoRALayers × {Wq,Wk,Wv,Wo}
    // quantization jobs are parallelised with OpenMP so the one-time
    // startup cost scales with hardware thread count rather than being
    // purely serial.  Cache reads/writes inside the parallel region are
    // per-file and non-overlapping, so no additional synchronisation is needed.
    const std::size_t streamQk = cfg_.bitnetQK;
    const std::size_t streamFfnQk = (cfg_.bitnetFfnQK == 0) ? streamQk : cfg_.bitnetFfnQK;
    const std::size_t streamJq = cfg_.bitnetJQ;
    const std::string streamRotTag = cfg_.bitnetHadamard32 ? ".h32" : ".h0";
    const std::string streamCacheTag =
        ".i2s.qk" + std::to_string(streamQk) +
        ".jq" + std::to_string(streamJq) +
        streamRotTag +
        kI2SCacheRevisionTag + ".bin";
    const std::string streamFfnCacheTag =
        ".i2s.qk" + std::to_string(streamFfnQk) +
        ".jq" + std::to_string(streamJq) +
        streamRotTag +
        kI2SCacheRevisionTag + ".bin";

    std::size_t streamLayersCached = 0;
    std::size_t streamLayersQuantized = 0;
    std::size_t streamReleasedBytes = 0;
    auto streamReleaseDense = [&](std::vector<float>& w) {
        streamReleasedBytes += w.size() * sizeof(float);
        std::vector<float>().swap(w);
    };
    auto quantizeLoadedLayer = [&](std::size_t L) {
        bool anyCacheMiss = false;
        auto& W = layers_[L];
        const bool enableTl2Q = tl2QEnabled(cfg_);
        const bool enableTl2K = tl2KEnabled(cfg_);
        const bool enableTl2V = tl2VEnabled(cfg_);
        const bool enableTl2Wo = tl2WoEnabled(cfg_);
        const bool enableTl2Gate = tl2GateEnabled(cfg_);
        const bool enableTl2Up = tl2UpEnabled(cfg_);
        const bool enableTl2Down = tl2DownEnabled(cfg_);

        if (cfg_.bitnetQuantizeQKV) {
            const std::string qPath = layerCacheStem(L, "Wq") + streamCacheTag;
            const std::string kPath = layerCacheStem(L, "Wk") + streamCacheTag;
            const std::string vPath = layerCacheStem(L, "Wv") + streamCacheTag;
            if (!load_i2s_cache(qPath, W.Wq_q, H, H, streamQk, streamJq, cfg_.bitnetHadamard32)) {
                anyCacheMiss = true;
                std::cout << "  [quant] layer " << (L + 1) << " Wq...\n"; std::cout.flush();
                quantize_ternary_i2s_blockwise(W.Wq_q, W.Wq.data(), H, H, streamQk, streamJq, cfg_.bitnetHadamard32);
                save_i2s_cache(qPath, W.Wq_q);
            }
            if (!load_i2s_cache(kPath, W.Wk_q, H, kvDim, streamQk, streamJq, cfg_.bitnetHadamard32)) {
                anyCacheMiss = true;
                std::cout << "  [quant] layer " << (L + 1) << " Wk...\n"; std::cout.flush();
                quantize_ternary_i2s_blockwise(W.Wk_q, W.Wk.data(), H, kvDim, streamQk, streamJq, cfg_.bitnetHadamard32);
                save_i2s_cache(kPath, W.Wk_q);
            }
            if (!load_i2s_cache(vPath, W.Wv_q, H, kvDim, streamQk, streamJq, cfg_.bitnetHadamard32)) {
                anyCacheMiss = true;
                std::cout << "  [quant] layer " << (L + 1) << " Wv...\n"; std::cout.flush();
                quantize_ternary_i2s_blockwise(W.Wv_q, W.Wv.data(), H, kvDim, streamQk, streamJq, cfg_.bitnetHadamard32);
                save_i2s_cache(vPath, W.Wv_q);
            }
            if (decodedQkvForwardCacheEnabled()) {
                build_bitnet_i2s_decoded_forward_cache(W.Wq_q);
                build_bitnet_i2s_decoded_forward_cache(W.Wk_q);
                build_bitnet_i2s_decoded_forward_cache(W.Wv_q);
            }
            if (decodedQkvBackwardCacheEnabled()) {
                build_bitnet_i2s_decoded_transposed_cache(W.Wq_q);
                build_bitnet_i2s_decoded_transposed_cache(W.Wk_q);
                build_bitnet_i2s_decoded_transposed_cache(W.Wv_q);
            }
            if (enableTl2Q && tl2GeneratedSupported(cfg_, H, H)) {
                quantize_ternary_tl2_generated(W.Wq_tl2g, W.Wq.data(), H, H, "Q");
            }
            else if (enableTl2Q && tl2Supported(cfg_, H, H)) {
                quantize_ternary_tl2_blockwise(W.Wq_tl2, W.Wq.data(), H, H, cfg_.bitnetTl2Bk);
            }
            if (enableTl2K && tl2GeneratedSupported(cfg_, H, kvDim)) {
                quantize_ternary_tl2_generated(W.Wk_tl2g, W.Wk.data(), H, kvDim, "K");
            }
            else if (enableTl2K && tl2Supported(cfg_, H, kvDim)) {
                quantize_ternary_tl2_blockwise(W.Wk_tl2, W.Wk.data(), H, kvDim, cfg_.bitnetTl2Bk);
            }
            if (enableTl2V && tl2GeneratedSupported(cfg_, H, kvDim)) {
                quantize_ternary_tl2_generated(W.Wv_tl2g, W.Wv.data(), H, kvDim, "V");
            }
            else if (enableTl2V && tl2Supported(cfg_, H, kvDim)) {
                quantize_ternary_tl2_blockwise(W.Wv_tl2, W.Wv.data(), H, kvDim, cfg_.bitnetTl2Bk);
            }
            streamReleaseDense(W.Wq);
            streamReleaseDense(W.Wk);
            streamReleaseDense(W.Wv);
        }

        if (cfg_.bitnetQuantizeWo) {
            const std::string oPath = layerCacheStem(L, "Wo") + streamCacheTag;
            if (!load_i2s_cache(oPath, W.Wo_q, H, H, streamQk, streamJq, cfg_.bitnetHadamard32)) {
                anyCacheMiss = true;
                std::cout << "  [quant] layer " << (L + 1) << " Wo...\n"; std::cout.flush();
                quantize_ternary_i2s_blockwise(W.Wo_q, W.Wo.data(), H, H, streamQk, streamJq, cfg_.bitnetHadamard32);
                save_i2s_cache(oPath, W.Wo_q);
            }
            if (enableTl2Wo && tl2GeneratedSupported(cfg_, H, H)) {
                quantize_ternary_tl2_generated(W.Wo_tl2g, W.Wo.data(), H, H, "WO");
            }
            else if (enableTl2Wo && tl2Supported(cfg_, H, H)) {
                quantize_ternary_tl2_blockwise(W.Wo_tl2, W.Wo.data(), H, H, cfg_.bitnetTl2Bk);
            }
            streamReleaseDense(W.Wo);
        }

        if (useBitNetFfn) {
            const std::string gatePath = layerCacheStem(L, "Wgate") + streamFfnCacheTag;
            const std::string upPath = layerCacheStem(L, "Wup") + streamFfnCacheTag;
            const std::string downPath = layerCacheStem(L, "Wdown") + streamFfnCacheTag;
            if (!load_i2s_cache(gatePath, W.Wgate_q, H, F, streamFfnQk, streamJq, cfg_.bitnetHadamard32)) {
                anyCacheMiss = true;
                std::cout << "  [quant] layer " << (L + 1) << " Wgate...\n"; std::cout.flush();
                quantize_ternary_i2s_blockwise(W.Wgate_q, W.Wgate.data(), H, F, streamFfnQk, streamJq, cfg_.bitnetHadamard32);
                save_i2s_cache(gatePath, W.Wgate_q);
            }
            if (!load_i2s_cache(upPath, W.Wup_q, H, F, streamFfnQk, streamJq, cfg_.bitnetHadamard32)) {
                anyCacheMiss = true;
                std::cout << "  [quant] layer " << (L + 1) << " Wup...\n"; std::cout.flush();
                quantize_ternary_i2s_blockwise(W.Wup_q, W.Wup.data(), H, F, streamFfnQk, streamJq, cfg_.bitnetHadamard32);
                save_i2s_cache(upPath, W.Wup_q);
            }
            if (!load_i2s_cache(downPath, W.Wdown_q, F, H, streamFfnQk, streamJq, cfg_.bitnetHadamard32)) {
                anyCacheMiss = true;
                std::cout << "  [quant] layer " << (L + 1) << " Wdown...\n"; std::cout.flush();
                quantize_ternary_i2s_blockwise(W.Wdown_q, W.Wdown.data(), F, H, streamFfnQk, streamJq, cfg_.bitnetHadamard32);
                save_i2s_cache(downPath, W.Wdown_q);
            }
            if (decodedWdownBackwardCacheEnabled()) {
                build_bitnet_i2s_decoded_transposed_cache(W.Wdown_q);
            }
            if (enableTl2Gate && tl2GeneratedSupported(cfg_, H, F)) {
                quantize_ternary_tl2_generated(W.Wgate_tl2g, W.Wgate.data(), H, F, "GATE");
            }
            else if (enableTl2Gate && tl2Supported(cfg_, H, F)) {
                quantize_ternary_tl2_blockwise(W.Wgate_tl2, W.Wgate.data(), H, F, cfg_.bitnetTl2Bk);
            }
            if (enableTl2Up && tl2GeneratedSupported(cfg_, H, F)) {
                quantize_ternary_tl2_generated(W.Wup_tl2g, W.Wup.data(), H, F, "UP");
            }
            else if (enableTl2Up && tl2Supported(cfg_, H, F)) {
                quantize_ternary_tl2_blockwise(W.Wup_tl2, W.Wup.data(), H, F, cfg_.bitnetTl2Bk);
            }
            if (enableTl2Down && tl2GeneratedSupported(cfg_, F, H)) {
                quantize_ternary_tl2_generated(W.Wdown_tl2g, W.Wdown.data(), F, H, "DOWN");
            }
            else if (enableTl2Down && tl2Supported(cfg_, F, H)) {
                quantize_ternary_tl2_blockwise(W.Wdown_tl2, W.Wdown.data(), F, H, cfg_.bitnetTl2Bk);
            }
            streamReleaseDense(W.Wgate);
            streamReleaseDense(W.Wup);
            streamReleaseDense(W.Wdown);
        }

        if (anyCacheMiss) { ++streamLayersQuantized; }
        else { ++streamLayersCached; }
    };

    // ── Serial weight loading (disk I/O, must stay serial) ───────────────────
    for (std::size_t L = 0; L < nLoRALayers; ++L)
    {
        std::cout << "Loading layer " << (L + 1) << "/" << nLoRALayers << "...\n";
        if (streamQuantizedDenseWeights) {
            init_mat_HH(layers_[L].Wq);
            init_mat_HKV(layers_[L].Wk);
            init_mat_HKV(layers_[L].Wv);
            init_mat_HH(layers_[L].Wo);
            init_mat_HF(layers_[L].Wgate);
            init_mat_HF(layers_[L].Wup);
            init_mat_FH(layers_[L].Wdown);
        }
        loadLayerWeight(L, "Wq", layers_[L].Wq);
        loadLayerWeight(L, "Wk", layers_[L].Wk);
        loadLayerWeight(L, "Wv", layers_[L].Wv);
        loadLayerWeight(L, "Wo", layers_[L].Wo);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_rms_attn.bin", layers_[L].rms_attn);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_rms_ffn.bin", layers_[L].rms_ffn);
        loadLayerWeight(L, "Wgate", layers_[L].Wgate);
        loadLayerWeight(L, "Wup", layers_[L].Wup);
        loadLayerWeight(L, "Wdown", layers_[L].Wdown);
        if (streamQuantizedDenseWeights) {
            quantizeLoadedLayer(L);
        }
    }

    if (streamQuantizedDenseWeights) {
        std::cout << "Stream-quantized " << nLoRALayers
            << " layer(s) during load to avoid retaining dense FP32 layer weights.\n";
        if (streamLayersQuantized == 0) {
            std::cout << "  All " << nLoRALayers << " stream-quantized layers loaded from cache.\n";
        }
        else {
            std::cout << "  " << streamLayersQuantized << " stream-quantized layer(s), "
                << streamLayersCached << " loaded from cache.\n";
        }
        if (streamReleasedBytes > 0) {
            const double releasedGiB = static_cast<double>(streamReleasedBytes) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "Stream-released dense FP32 layer weights during load: "
                << releasedGiB << " GiB\n";
        }
        std::cout.flush();
    }

    if (cfg_.useDoRA) {
        std::cout << "DoRA: enabled; initializing trainable magnitudes from base column norms.\n";
        if (!streamQuantizedDenseWeights) {
            for (std::size_t L = 0; L < nLoRALayers; ++L) {
                auto& P = loras_[L];
                if (cfg_.loraTargetQ) dora_init_column_magnitude(P.Mq, P.Cq, P.Rq, layers_[L].Wq, H, H);
                if (cfg_.loraTargetK) dora_init_column_magnitude(P.Mk, P.Ck, P.Rk, layers_[L].Wk, H, kvDim);
                if (cfg_.loraTargetV) dora_init_column_magnitude(P.Mv, P.Cv, P.Rv, layers_[L].Wv, H, kvDim);
                if (cfg_.loraTargetO) dora_init_column_magnitude(P.Mo, P.Co, P.Ro, layers_[L].Wo, H, H);
                if (cfg_.loraTargetFfnDown) dora_init_column_magnitude(P.Mfd, P.Cfd, P.Rfd, layers_[L].Wdown, F, H);
                if (cfg_.loraTargetFfnGateUp) {
                    dora_init_column_magnitude(P.Mfg, P.Cfg, P.Rfg, layers_[L].Wgate, H, F);
                    dora_init_column_magnitude(P.Mfu, P.Cfu, P.Rfu, layers_[L].Wup, H, F);
                }
            }
        }
        else {
            std::cout << "DoRA: deferring layer magnitudes to quantized base weights.\n";
        }
        const bool deferLmHeadDoraMagnitude =
            cfg_.useBitNetQuant &&
            canUseI2S &&
            cfg_.bitnetQuantizeLmHead &&
            !cfg_.bitnetLmHeadFloatBackward;
        if (cfg_.loraTargetLmHead) {
            if (deferLmHeadDoraMagnitude) {
                std::cout << "DoRA: deferring LM-head magnitude to quantized base weights.\n";
            }
            else {
                dora_init_column_magnitude(lmHeadLora_.M, lmHeadLora_.C, lmHeadLora_.R, W_out_, H, V);
            }
        }
    }

    // ── Quantization (serial outer loop, parallel inner) ─────────────────────
    // quantize_ternary_i2s_blockwise already contains
    //   #pragma omp parallel for collapse(2)
    // internally, so it saturates all available threads on its own.
    // Adding an outer #pragma omp parallel for creates nested parallelism:
    //   N_outer_threads × N_inner_threads compete for N_cores cores.
    // On MSVC's OpenMP runtime, nested parallelism is disabled by default,
    // so the inner pragma silently serialises while all outer threads still
    // block each other for the single thread pool — this causes the hang.
    // Fix: outer loop is serial; inner OMP parallelism does the real work.
    if (cfg_.useBitNetQuant && canUseI2S && !streamQuantizedDenseWeights) {
        const std::size_t qk = cfg_.bitnetQK;
        const std::size_t ffnQk = (cfg_.bitnetFfnQK == 0) ? qk : cfg_.bitnetFfnQK;
        const std::size_t jq = cfg_.bitnetJQ;
        const std::string rotTag = cfg_.bitnetHadamard32 ? ".h32" : ".h0";
        const std::string cacheTag =
            ".i2s.qk" + std::to_string(qk) +
            ".jq" + std::to_string(jq) +
            rotTag +
            kI2SCacheRevisionTag + ".bin";
        const std::string ffnCacheTag =
            ".i2s.qk" + std::to_string(ffnQk) +
            ".jq" + std::to_string(jq) +
            rotTag +
            kI2SCacheRevisionTag + ".bin";

        std::cout << "Quantizing " << nLoRALayers << " layers...\n";
        std::cout.flush();

        // Track whether any layer needed actual quantization work (cache miss)
        // so the final summary is accurate for troubleshooting.
        std::size_t nLayersCached = 0;
        std::size_t nLayersQuantized = 0;

        for (std::size_t L = 0; L < nLoRALayers; ++L) {
            bool anyCacheMiss = false;
            const bool enableTl2Q = tl2QEnabled(cfg_);
            const bool enableTl2K = tl2KEnabled(cfg_);
            const bool enableTl2V = tl2VEnabled(cfg_);
            const bool enableTl2Wo = tl2WoEnabled(cfg_);
            const bool enableTl2Gate = tl2GateEnabled(cfg_);
            const bool enableTl2Up = tl2UpEnabled(cfg_);
            const bool enableTl2Down = tl2DownEnabled(cfg_);

            if (cfg_.bitnetQuantizeQKV) {
                const std::string qPath = layerCacheStem(L, "Wq") + cacheTag;
                const std::string kPath = layerCacheStem(L, "Wk") + cacheTag;
                const std::string vPath = layerCacheStem(L, "Wv") + cacheTag;

                if (!load_i2s_cache(qPath, layers_[L].Wq_q, H, H, qk, jq, cfg_.bitnetHadamard32)) {
                    anyCacheMiss = true;
                    std::cout << "  [quant] layer " << (L + 1) << " Wq...\n"; std::cout.flush();
                    quantize_ternary_i2s_blockwise(layers_[L].Wq_q, layers_[L].Wq.data(), H, H, qk, jq, cfg_.bitnetHadamard32);
                    save_i2s_cache(qPath, layers_[L].Wq_q);
                }
                if (!load_i2s_cache(kPath, layers_[L].Wk_q, H, kvDim, qk, jq, cfg_.bitnetHadamard32)) {
                    anyCacheMiss = true;
                    std::cout << "  [quant] layer " << (L + 1) << " Wk...\n"; std::cout.flush();
                    quantize_ternary_i2s_blockwise(layers_[L].Wk_q, layers_[L].Wk.data(), H, kvDim, qk, jq, cfg_.bitnetHadamard32);
                    save_i2s_cache(kPath, layers_[L].Wk_q);
                }
                if (!load_i2s_cache(vPath, layers_[L].Wv_q, H, kvDim, qk, jq, cfg_.bitnetHadamard32)) {
                    anyCacheMiss = true;
                    std::cout << "  [quant] layer " << (L + 1) << " Wv...\n"; std::cout.flush();
                    quantize_ternary_i2s_blockwise(layers_[L].Wv_q, layers_[L].Wv.data(), H, kvDim, qk, jq, cfg_.bitnetHadamard32);
                    save_i2s_cache(vPath, layers_[L].Wv_q);
                }
                if (decodedQkvForwardCacheEnabled()) {
                    build_bitnet_i2s_decoded_forward_cache(layers_[L].Wq_q);
                    build_bitnet_i2s_decoded_forward_cache(layers_[L].Wk_q);
                    build_bitnet_i2s_decoded_forward_cache(layers_[L].Wv_q);
                }
                if (decodedQkvBackwardCacheEnabled()) {
                    build_bitnet_i2s_decoded_transposed_cache(layers_[L].Wq_q);
                    build_bitnet_i2s_decoded_transposed_cache(layers_[L].Wk_q);
                    build_bitnet_i2s_decoded_transposed_cache(layers_[L].Wv_q);
                }
                if (enableTl2Q && tl2GeneratedSupported(cfg_, H, H)) {
                    quantize_ternary_tl2_generated(layers_[L].Wq_tl2g, layers_[L].Wq.data(), H, H, "Q");
                }
                else if (enableTl2Q && tl2Supported(cfg_, H, H)) {
                    quantize_ternary_tl2_blockwise(layers_[L].Wq_tl2, layers_[L].Wq.data(), H, H, cfg_.bitnetTl2Bk);
                }
                if (enableTl2K && tl2GeneratedSupported(cfg_, H, kvDim)) {
                    quantize_ternary_tl2_generated(layers_[L].Wk_tl2g, layers_[L].Wk.data(), H, kvDim, "K");
                }
                else if (enableTl2K && tl2Supported(cfg_, H, kvDim)) {
                    quantize_ternary_tl2_blockwise(layers_[L].Wk_tl2, layers_[L].Wk.data(), H, kvDim, cfg_.bitnetTl2Bk);
                }
                if (enableTl2V && tl2GeneratedSupported(cfg_, H, kvDim)) {
                    quantize_ternary_tl2_generated(layers_[L].Wv_tl2g, layers_[L].Wv.data(), H, kvDim, "V");
                }
                else if (enableTl2V && tl2Supported(cfg_, H, kvDim)) {
                    quantize_ternary_tl2_blockwise(layers_[L].Wv_tl2, layers_[L].Wv.data(), H, kvDim, cfg_.bitnetTl2Bk);
                }
                if (!cfg_.useDoRA) {
                    std::vector<float>().swap(layers_[L].Wq);
                    std::vector<float>().swap(layers_[L].Wk);
                    std::vector<float>().swap(layers_[L].Wv);
                }
            }

            if (cfg_.bitnetQuantizeWo) {
                const std::string oPath = layerCacheStem(L, "Wo") + cacheTag;
                if (!load_i2s_cache(oPath, layers_[L].Wo_q, H, H, qk, jq, cfg_.bitnetHadamard32)) {
                    anyCacheMiss = true;
                    std::cout << "  [quant] layer " << (L + 1) << " Wo...\n"; std::cout.flush();
                    quantize_ternary_i2s_blockwise(layers_[L].Wo_q, layers_[L].Wo.data(), H, H, qk, jq, cfg_.bitnetHadamard32);
                    save_i2s_cache(oPath, layers_[L].Wo_q);
                }
                if (enableTl2Wo && tl2GeneratedSupported(cfg_, H, H)) {
                    quantize_ternary_tl2_generated(layers_[L].Wo_tl2g, layers_[L].Wo.data(), H, H, "WO");
                }
                else if (enableTl2Wo && tl2Supported(cfg_, H, H)) {
                    quantize_ternary_tl2_blockwise(layers_[L].Wo_tl2, layers_[L].Wo.data(), H, H, cfg_.bitnetTl2Bk);
                }
                if (!cfg_.useDoRA) {
                    std::vector<float>().swap(layers_[L].Wo);
                }
            }

            if (useBitNetFfn) {
                const std::string gatePath = layerCacheStem(L, "Wgate") + ffnCacheTag;
                const std::string upPath = layerCacheStem(L, "Wup") + ffnCacheTag;
                const std::string downPath = layerCacheStem(L, "Wdown") + ffnCacheTag;

                if (!load_i2s_cache(gatePath, layers_[L].Wgate_q, H, F, ffnQk, jq, cfg_.bitnetHadamard32)) {
                    anyCacheMiss = true;
                    std::cout << "  [quant] layer " << (L + 1) << " Wgate...\n"; std::cout.flush();
                    quantize_ternary_i2s_blockwise(layers_[L].Wgate_q, layers_[L].Wgate.data(), H, F, ffnQk, jq, cfg_.bitnetHadamard32);
                    save_i2s_cache(gatePath, layers_[L].Wgate_q);
                }
                if (!load_i2s_cache(upPath, layers_[L].Wup_q, H, F, ffnQk, jq, cfg_.bitnetHadamard32)) {
                    anyCacheMiss = true;
                    std::cout << "  [quant] layer " << (L + 1) << " Wup...\n"; std::cout.flush();
                    quantize_ternary_i2s_blockwise(layers_[L].Wup_q, layers_[L].Wup.data(), H, F, ffnQk, jq, cfg_.bitnetHadamard32);
                    save_i2s_cache(upPath, layers_[L].Wup_q);
                }
                if (!load_i2s_cache(downPath, layers_[L].Wdown_q, F, H, ffnQk, jq, cfg_.bitnetHadamard32)) {
                    anyCacheMiss = true;
                    std::cout << "  [quant] layer " << (L + 1) << " Wdown...\n"; std::cout.flush();
                    quantize_ternary_i2s_blockwise(layers_[L].Wdown_q, layers_[L].Wdown.data(), F, H, ffnQk, jq, cfg_.bitnetHadamard32);
                    save_i2s_cache(downPath, layers_[L].Wdown_q);
                }
                if (decodedWdownBackwardCacheEnabled()) {
                    build_bitnet_i2s_decoded_transposed_cache(layers_[L].Wdown_q);
                }
                if (enableTl2Gate && tl2GeneratedSupported(cfg_, H, F)) {
                    quantize_ternary_tl2_generated(layers_[L].Wgate_tl2g, layers_[L].Wgate.data(), H, F, "GATE");
                }
                else if (enableTl2Gate && tl2Supported(cfg_, H, F)) {
                    quantize_ternary_tl2_blockwise(layers_[L].Wgate_tl2, layers_[L].Wgate.data(), H, F, cfg_.bitnetTl2Bk);
                }
                if (enableTl2Up && tl2GeneratedSupported(cfg_, H, F)) {
                    quantize_ternary_tl2_generated(layers_[L].Wup_tl2g, layers_[L].Wup.data(), H, F, "UP");
                }
                else if (enableTl2Up && tl2Supported(cfg_, H, F)) {
                    quantize_ternary_tl2_blockwise(layers_[L].Wup_tl2, layers_[L].Wup.data(), H, F, cfg_.bitnetTl2Bk);
                }
                if (enableTl2Down && tl2GeneratedSupported(cfg_, F, H)) {
                    quantize_ternary_tl2_generated(layers_[L].Wdown_tl2g, layers_[L].Wdown.data(), F, H, "DOWN");
                }
                else if (enableTl2Down && tl2Supported(cfg_, F, H)) {
                    quantize_ternary_tl2_blockwise(layers_[L].Wdown_tl2, layers_[L].Wdown.data(), F, H, cfg_.bitnetTl2Bk);
                }
                if (!cfg_.useDoRA) {
                    std::vector<float>().swap(layers_[L].Wgate);
                    std::vector<float>().swap(layers_[L].Wup);
                    std::vector<float>().swap(layers_[L].Wdown);
                }
            }

            if (anyCacheMiss) { ++nLayersQuantized; }
            else { ++nLayersCached; }
        }

        if (nLayersQuantized == 0) {
            std::cout << "  All " << nLoRALayers << " layers loaded from cache.\n";
        }
        else {
            std::cout << "  " << nLayersQuantized << " layer(s) quantized, "
                << nLayersCached << " loaded from cache.\n";
        }
        std::cout.flush();
    }

    for (std::size_t L = 0; L < nLoRALayers; ++L) {
        // Standard LoRA init: B=0 so delta_W = scale*B@A = 0 at t=0.
        // A sigma=0.0005 (wdist*0.025): with lr=1e-5, B reaches A-parity
        // at step ~63 — stable growth without explosion.
        for (auto& x : loras_[L].Aq) x = wdist(rng) * 0.025f;
        // Keep Q/V-only runs reproducible with the original two-target init.
        for (auto& x : loras_[L].Av) x = wdist(rng) * 0.025f;
        if (cfg_.loraTargetK) {
            for (auto& x : loras_[L].Ak) x = wdist(rng) * 0.025f;
        }
        if (cfg_.loraTargetO) {
            for (auto& x : loras_[L].Ao) x = wdist(rng) * 0.025f;
        }
        if (cfg_.loraTargetFfnDown) {
            for (auto& x : loras_[L].Afd) x = wdist(rng) * 0.025f;
        }
        if (cfg_.loraTargetFfnGateUp) {
            for (auto& x : loras_[L].Afg) x = wdist(rng) * 0.025f;
            for (auto& x : loras_[L].Afu) x = wdist(rng) * 0.025f;
        }
        std::fill(loras_[L].Bq.begin(), loras_[L].Bq.end(), 0.0f);
        std::fill(loras_[L].Bk.begin(), loras_[L].Bk.end(), 0.0f);
        std::fill(loras_[L].Bv.begin(), loras_[L].Bv.end(), 0.0f);
        std::fill(loras_[L].Bo.begin(), loras_[L].Bo.end(), 0.0f);
        std::fill(loras_[L].Bfd.begin(), loras_[L].Bfd.end(), 0.0f);
        std::fill(loras_[L].Bfg.begin(), loras_[L].Bfg.end(), 0.0f);
        std::fill(loras_[L].Bfu.begin(), loras_[L].Bfu.end(), 0.0f);
    }
    if (cfg_.loraTargetLmHead) {
        for (auto& x : lmHeadLora_.A) x = wdist(rng) * 0.025f;
        std::fill(lmHeadLora_.B.begin(), lmHeadLora_.B.end(), 0.0f);
    }
    std::fill(lmHeadBias_.begin(), lmHeadBias_.end(), 0.0f);
    std::fill(finalNormScale_.begin(), finalNormScale_.end(), 1.0f);
    std::fill(finalNormBias_.begin(), finalNormBias_.end(), 0.0f);

    if (cfg_.usePiSSA) {
        std::size_t pissaFactorPairsLoaded = 0;
        auto loadPissaFactors = [&](const std::string& stem,
                                    const std::string& displayName,
                                    std::vector<float>& A,
                                    std::vector<float>& B) {
            const std::string residualPath = stem + ".residual.bin";
            const std::string aPath = stem + ".A.bin";
            const std::string bPath = stem + ".B.bin";
            const bool hasResidual = std::filesystem::exists(residualPath);
            const bool hasA = std::filesystem::exists(aPath);
            const bool hasB = std::filesystem::exists(bPath);
            if (!hasResidual && !hasA && !hasB) {
                return;
            }
            if (!(hasResidual && hasA && hasB)) {
                throw std::runtime_error(
                    "Incomplete PiSSA files for " + displayName +
                    "; expected .residual.bin, .A.bin, and .B.bin");
            }
            load_raw_bin(aPath, A);
            load_raw_bin(bPath, B);
            ++pissaFactorPairsLoaded;
        };

        for (std::size_t L = 0; L < nLoRALayers; ++L) {
            auto& P = loras_[L];
            if (cfg_.loraTargetQ) loadPissaFactors(pissaLayerStem(L, "Wq"), "layer " + std::to_string(L) + " Wq", P.Aq, P.Bq);
            if (cfg_.loraTargetK) loadPissaFactors(pissaLayerStem(L, "Wk"), "layer " + std::to_string(L) + " Wk", P.Ak, P.Bk);
            if (cfg_.loraTargetV) loadPissaFactors(pissaLayerStem(L, "Wv"), "layer " + std::to_string(L) + " Wv", P.Av, P.Bv);
            if (cfg_.loraTargetO) loadPissaFactors(pissaLayerStem(L, "Wo"), "layer " + std::to_string(L) + " Wo", P.Ao, P.Bo);
            if (cfg_.loraTargetFfnDown) loadPissaFactors(pissaLayerStem(L, "Wdown"), "layer " + std::to_string(L) + " Wdown", P.Afd, P.Bfd);
            if (cfg_.loraTargetFfnGateUp) {
                loadPissaFactors(pissaLayerStem(L, "Wgate"), "layer " + std::to_string(L) + " Wgate", P.Afg, P.Bfg);
                loadPissaFactors(pissaLayerStem(L, "Wup"), "layer " + std::to_string(L) + " Wup", P.Afu, P.Bfu);
            }
        }
        if (cfg_.loraTargetLmHead) {
            loadPissaFactors(pissaLmHeadStem, "LM head", lmHeadLora_.A, lmHeadLora_.B);
        }
        std::cout << "PiSSA: loaded " << pissaFactorPairsLoaded
            << " residual/factor target(s). Missing targets keep standard LoRA initialization.\n";
    }

    const bool useQuantLmHeadCache =
        cfg_.useBitNetQuant &&
        canUseI2S &&
        (cfg_.bitnetQuantizeLmHead || cfg_.bitnetLmHeadQuantBackwardOnly);
    if (useQuantLmHeadCache) {
        const std::size_t qk = cfg_.bitnetQK;
        const std::size_t jq = cfg_.bitnetJQ;
        const std::string rotTag = cfg_.bitnetHadamard32 ? ".h32" : ".h0";
        const std::string lmHeadCacheStem =
            (pissaTargetActive("W_out") && pissaHasResidual(pissaLmHeadStem))
                ? pissaLmHeadStem
                : (weight_dir + "/W_out");
        const std::string lmHeadCache =
            lmHeadCacheStem + ".i2s.qk" + std::to_string(qk) +
            ".jq" + std::to_string(jq) +
            rotTag +
            kI2SCacheRevisionTag + ".bin";

        if (load_i2s_cache(lmHeadCache, W_out_q, H, V, qk, jq, cfg_.bitnetHadamard32)) {
            std::cout << "  [cache] LM head loaded from cache.\n";
        }
        else {
            std::cout << "  [quant] LM head...\n"; std::cout.flush();
            quantize_ternary_i2s_blockwise(W_out_q, W_out_.data(), H, V, qk, jq, cfg_.bitnetHadamard32);
            if (save_i2s_cache(lmHeadCache, W_out_q)) {
                std::cout << "  [cache] LM head saved to cache.\n";
            }
        }
        if (decodedLmHeadForwardCacheEnabled()) {
            build_bitnet_i2s_decoded_forward_cache(W_out_q);
        }
        if (decodedLmHeadBackwardCacheEnabled()) {
            build_bitnet_i2s_decoded_transposed_cache(W_out_q);
        }
        if (cfg_.bitnetQuantizeLmHead && !cfg_.bitnetLmHeadFloatBackward) {
            std::cout << "  [quant] LM head uses quantized forward/backward.\n";
            if (!cfg_.useDoRA) {
                // DoRA initializes magnitudes after this block, then releases
                // the dense copy through the shared quantized-weight cleanup.
                std::vector<float>().swap(W_out_);
            }
        }
        else if (cfg_.bitnetLmHeadQuantBackwardOnly) {
            std::cout << "  [hybrid] LM head uses dense FP32 forward with quantized backward.\n";
        }
        else {
            std::cout << "  [hybrid] LM head uses quantized forward with FP32 backward.\n";
        }
    }
    else {
        std::cout << "LM head quantization disabled for faster startup.\n";
    }

    if (cfg_.useDoRA) {
        bool anyQuantizedDoraBase = false;
        for (std::size_t L = 0; L < nLoRALayers; ++L) {
            auto& P = loras_[L];
            auto& W = layers_[L];
            if (cfg_.loraTargetQ && !W.Wq_q.empty()) {
                dora_init_column_magnitude_i2s(P.Mq, P.Cq, P.Rq, W.Wq_q);
                std::vector<float>().swap(W.Wq);
                anyQuantizedDoraBase = true;
            }
            if (cfg_.loraTargetK && !W.Wk_q.empty()) {
                dora_init_column_magnitude_i2s(P.Mk, P.Ck, P.Rk, W.Wk_q);
                std::vector<float>().swap(W.Wk);
                anyQuantizedDoraBase = true;
            }
            if (cfg_.loraTargetV && !W.Wv_q.empty()) {
                dora_init_column_magnitude_i2s(P.Mv, P.Cv, P.Rv, W.Wv_q);
                std::vector<float>().swap(W.Wv);
                anyQuantizedDoraBase = true;
            }
            if (cfg_.loraTargetO && !W.Wo_q.empty()) {
                dora_init_column_magnitude_i2s(P.Mo, P.Co, P.Ro, W.Wo_q);
                std::vector<float>().swap(W.Wo);
                anyQuantizedDoraBase = true;
            }
            if (cfg_.loraTargetFfnDown && !W.Wdown_q.empty()) {
                dora_init_column_magnitude_i2s(P.Mfd, P.Cfd, P.Rfd, W.Wdown_q);
                std::vector<float>().swap(W.Wdown);
                anyQuantizedDoraBase = true;
            }
            if (cfg_.loraTargetFfnGateUp) {
                if (!W.Wgate_q.empty()) {
                    dora_init_column_magnitude_i2s(P.Mfg, P.Cfg, P.Rfg, W.Wgate_q);
                    std::vector<float>().swap(W.Wgate);
                    anyQuantizedDoraBase = true;
                }
                if (!W.Wup_q.empty()) {
                    dora_init_column_magnitude_i2s(P.Mfu, P.Cfu, P.Rfu, W.Wup_q);
                    std::vector<float>().swap(W.Wup);
                    anyQuantizedDoraBase = true;
                }
            }
        }
        if (cfg_.loraTargetLmHead && !W_out_q.empty()) {
            dora_init_column_magnitude_i2s(lmHeadLora_.M, lmHeadLora_.C, lmHeadLora_.R, W_out_q);
            if (cfg_.bitnetQuantizeLmHead && !cfg_.bitnetLmHeadFloatBackward) {
                std::vector<float>().swap(W_out_);
            }
            anyQuantizedDoraBase = true;
        }
        if (anyQuantizedDoraBase) {
            std::cout << "DoRA: using ternary/I2S base magnitudes for quantized adapted weights.\n";
        }
    }

    if (cfg_.releaseQuantizedDenseWeights) {
        std::size_t releasedBytes = 0;
        auto releaseDense = [&](std::vector<float>& w) {
            releasedBytes += w.size() * sizeof(float);
            std::vector<float>().swap(w);
        };

        for (std::size_t L = 0; L < nLoRALayers; ++L) {
            auto& W = layers_[L];
            if (!W.Wq_q.empty()) releaseDense(W.Wq);
            if (!W.Wk_q.empty()) releaseDense(W.Wk);
            if (!W.Wv_q.empty()) releaseDense(W.Wv);
            if (!W.Wo_q.empty()) releaseDense(W.Wo);
            if (!W.Wgate_q.empty()) releaseDense(W.Wgate);
            if (!W.Wup_q.empty()) releaseDense(W.Wup);
            if (!W.Wdown_q.empty()) releaseDense(W.Wdown);
        }

        if (!W_out_q.empty() &&
            cfg_.bitnetQuantizeLmHead &&
            !cfg_.bitnetLmHeadFloatBackward &&
            !cfg_.bitnetLmHeadQuantBackwardOnly) {
            releaseDense(W_out_);
        }

        if (releasedBytes > 0) {
            const double releasedGiB = static_cast<double>(releasedBytes) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "Released dense FP32 copies for quantized weights: "
                << releasedGiB << " GiB\n";
        }
    }
}

// ==============================================================================
// FORWARD
// ==============================================================================
bool CpuKernelBackend::ffnLoraLayerEnabled(std::size_t layer) const
{
    if (!cfg_.loraTargetFfnDown && !cfg_.loraTargetFfnGateUp) {
        return false;
    }

    const std::size_t nLayers = shape_.nLayers;
    const std::size_t first = (cfg_.loraFfnFirstLayer < 0)
        ? 0u
        : std::min<std::size_t>(static_cast<std::size_t>(cfg_.loraFfnFirstLayer), nLayers);
    const std::size_t last = (cfg_.loraFfnLastLayer < 0)
        ? nLayers
        : std::min<std::size_t>(static_cast<std::size_t>(cfg_.loraFfnLastLayer), nLayers);

    return first < last && layer >= first && layer < last;
}

void CpuKernelBackend::forward(
    std::span<const int>   inputIds,
    std::span<const int>   attentionMask,
    std::span<float>       logitsOut,
    std::span<const int>   lossMask)
{
    const std::size_t T = inputIds.size();
    if (attentionMask.size() != T) throw std::runtime_error("forward: attentionMask size mismatch");
    if (T > cfg_.maxSeqLen)        throw std::runtime_error("forward: seqLen exceeds maxSeqLen");
    const bool hasLossMask = !lossMask.empty();
    if (hasLossMask && lossMask.size() != T) {
        throw std::runtime_error("forward: lossMask size mismatch");
    }

    const std::size_t H = shape_.hiddenSize;
    const std::size_t V = shape_.vocabSize;
    const std::size_t F = shape_.ffnSize;
    const std::size_t D = shape_.headDim();
    const std::size_t ropeDim = shape_.ropeDim;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);
    const std::size_t kvDim = shape_.kvDim(); // from ModelShape, not fragile Wk.size()/H
    const std::size_t nKvHeads = shape_.nKvHeads;
    const std::size_t nHeads = shape_.nHeads;
    const std::size_t tTile = static_cast<std::size_t>(cfg_.tTile);
    const std::size_t hTile = static_cast<std::size_t>(cfg_.hTile);
    const float       scale = cfg_.loraAlpha / float(R);
    const std::size_t CKPT_STRIDE = cfg_.maxSeqLen * H;
    const std::size_t activePrefixT = contiguousActivePrefixLength(attentionMask);
    const std::size_t activeComputeT =
        (activePrefixComputeEnabled() && activePrefixT != static_cast<std::size_t>(-1))
        ? activePrefixT : T;
    const BackendRuntimePolicy runtimePolicy =
        buildBackendRuntimePolicy(activeComputeT, tTile);
    recordBackendRuntimePolicySnapshot(runtimePolicy);
    const ScopedBackendRuntimePolicy runtimePolicyScope(runtimePolicy);
    const int* const activeAttentionMask =
        (activeComputeT < T) ? nullptr : attentionMask.data();
    const std::size_t i2sActivePrefixT = exactMaskedTileSkipEnabled()
        ? activePrefixT : static_cast<std::size_t>(-1);
    const int* const  i2sAttentionMask = (exactMaskedTileSkipEnabled() && i2sActivePrefixT == static_cast<std::size_t>(-1))
        ? attentionMask.data() : nullptr;
    const bool profile = backendProfileEnabled();
    const std::vector<std::size_t> segmentStarts =
        packedSegmentStarts(attentionMask, cfg_.enablePackedSegments);
    const bool useFfnGemm = ffnGemmEnabled();
    const bool useBitNetFfn = cfg_.useBitNetQuant && ffnBitnetEnabled();
    auto& prof = backendProfileState();
    DoraStats doraStats;
    const int doraNormInterval = std::max(0, cfg_.doraNormUpdateInterval);
    const bool doraRefreshNorms =
        cfg_.useDoRA &&
        doraNormInterval > 0 &&
        doraParamVersion_ > doraNormVersion_ &&
        (doraParamVersion_ - doraNormVersion_) >= static_cast<std::uint64_t>(doraNormInterval);
    const int doraLmHeadNormInterval = std::max(0, cfg_.doraLmHeadNormUpdateInterval);
    const bool doraRefreshLmHeadNorms =
        cfg_.useDoRA &&
        doraLmHeadNormInterval > 0 &&
        doraParamVersion_ > doraLmHeadNormVersion_ &&
        (doraParamVersion_ - doraLmHeadNormVersion_) >= static_cast<std::uint64_t>(doraLmHeadNormInterval);
    auto doraUpdateDense = [&](const std::vector<float>& W,
                               const std::vector<float>& A,
                               const std::vector<float>& B,
                               const std::vector<float>& M,
                               std::vector<float>& C,
                               std::vector<float>& Ratio,
                               std::size_t inDim,
                               std::size_t outDim) -> DoraStats {
        if (!doraRefreshNorms) {
            return dora_cached_stats(M, C, Ratio);
        }
        return dora_update_detached_norm_ratio(W, A, B, M, C, Ratio, inDim, outDim, R, scale);
    };
    auto doraUpdateI2S = [&](const BitNetMatI2S& W,
                             const std::vector<float>& A,
                             const std::vector<float>& B,
                             const std::vector<float>& M,
                             std::vector<float>& C,
                             std::vector<float>& Ratio,
                             std::size_t inDim,
                             std::size_t outDim) -> DoraStats {
        if (!doraRefreshNorms) {
            return dora_cached_stats(M, C, Ratio);
        }
        return dora_update_detached_norm_ratio_i2s(W, A, B, M, C, Ratio, inDim, outDim, R, scale);
    };
    auto doraUpdateLmHeadDense = [&](const std::vector<float>& W,
                                     const std::vector<float>& A,
                                     const std::vector<float>& B,
                                     const std::vector<float>& M,
                                     std::vector<float>& C,
                                     std::vector<float>& Ratio,
                                     std::size_t inDim,
                                     std::size_t outDim) -> DoraStats {
        if (!doraRefreshLmHeadNorms) {
            return dora_cached_stats(M, C, Ratio);
        }
        return dora_update_detached_norm_ratio(W, A, B, M, C, Ratio, inDim, outDim, R, scale);
    };
    auto doraUpdateLmHeadI2S = [&](const BitNetMatI2S& W,
                                   const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   const std::vector<float>& M,
                                   std::vector<float>& C,
                                   std::vector<float>& Ratio,
                                   std::size_t inDim,
                                   std::size_t outDim) -> DoraStats {
        if (!doraRefreshLmHeadNorms) {
            return dora_cached_stats(M, C, Ratio);
        }
        return dora_update_detached_norm_ratio_i2s(W, A, B, M, C, Ratio, inDim, outDim, R, scale);
    };

    {
        const auto start = profile ? ProfileClock::now() : ProfileClock::time_point{};
        for (std::size_t t = 0; t < T; ++t) {
            float* xt = &x_[t * H];
            if (attentionMask[t] == 0) { std::fill(xt, xt + H, 0.0f); continue; }

            const int tok = inputIds[t];
            if (tok < 0 || static_cast<std::size_t>(tok) >= V) { std::fill(xt, xt + H, 0.0f); continue; }

            const float* emb = &embeddings_[static_cast<std::size_t>(tok) * H];
            std::copy(emb, emb + H, xt);
        }
        if (profile) prof.fwdEmbedMs += elapsedMs(start);
    }

    std::copy(x_.begin(), x_.begin() + (T * H), &x_ckpt_[0]);

    const std::size_t nLayers = shape_.nLayers;
    const float eps = 1e-5f;

    for (std::size_t L = 0; L < nLayers; ++L) {
        const std::size_t layerTHOffset = L * cfg_.maxSeqLen * H;
        const std::size_t layerTFOffset = L * cfg_.maxSeqLen * F;
        const std::size_t layerTKVOffset = L * cfg_.maxSeqLen * kvDim;
        const std::size_t layerProbsOffset = L * nHeads * cfg_.maxSeqLen * cfg_.maxSeqLen;
        float* const layerXnormCache = xnorm_cache_.data() + layerTHOffset;
        float* const layerQCache = q_cache_.data() + layerTHOffset;
        float* const layerKCache = k_cache_.data() + layerTKVOffset;
        float* const layerVCache = v_cache_.data() + layerTKVOffset;
        float* const layerCtxCache = cfg_.loraTargetO ? (ctx_cache_.data() + layerTHOffset) : nullptr;
        const bool ffnLoraEnabled = ffnLoraLayerEnabled(L);
        const bool ffnDownLoraEnabled = cfg_.loraTargetFfnDown && ffnLoraEnabled;
        const bool ffnGateUpLoraEnabled = cfg_.loraTargetFfnGateUp && ffnLoraEnabled;
        float* const layerFfnActCache = ffnDownLoraEnabled ? (ffn_act_cache_.data() + layerTFOffset) : nullptr;
        const bool fusedAttentionNormCacheWrite = fusedAttentionNormCacheWriteEnabled(activeComputeT);
        const bool fusedAttentionQkvCacheWrite = fusedAttentionQkvCacheWriteEnabled(activeComputeT);
        float* const attnXnormForwardWrite = fusedAttentionNormCacheWrite ? layerXnormCache : xnorm_.data();
        const float* const attnXnormForward = attnXnormForwardWrite;
        float* const qProj = fusedAttentionQkvCacheWrite ? layerQCache : q_.data();
        float* const kProj = fusedAttentionQkvCacheWrite ? layerKCache : k_.data();
        float* const vProj = fusedAttentionQkvCacheWrite ? layerVCache : v_.data();

        const auto attnNormStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (parallelRmsnormForwardEnabled() && T > 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t t = 0; t < static_cast<ptrdiff_t>(T); ++t) {
                const std::size_t tu = static_cast<std::size_t>(t);
                float* xn = attnXnormForwardWrite + tu * H;
                const float* x = &x_[tu * H];
                if (attentionMask[tu] == 0) {
                    std::fill(xn, xn + H, 0.0f);
                    continue;
                }
                rmsnorm(xn, x, layers_[L].rms_attn.data(), H, eps);
            }
        }
        else {
            for (std::size_t t = 0; t < T; ++t) {
                float* xn = attnXnormForwardWrite + t * H;
                const float* x = &x_[t * H];
                if (attentionMask[t] == 0) {
                    std::fill(xn, xn + H, 0.0f);
                    continue;
                }
                rmsnorm(xn, x, layers_[L].rms_attn.data(), H, eps);
            }
        }
        if (!fusedAttentionNormCacheWrite) {
            std::copy_n(xnorm_.data(), T * H, layerXnormCache);
        }
        if (profile) prof.fwdAttnNormMs += elapsedMs(attnNormStart);

        const bool enableTl2Q = tl2QEnabled(cfg_);
        const bool enableTl2K = tl2KEnabled(cfg_);
        const bool enableTl2V = tl2VEnabled(cfg_);
        const bool enableTl2Wo = tl2WoEnabled(cfg_);
        const bool useTl2GenQ =
            enableTl2Q &&
            tl2GeneratedSupported(cfg_, H, H) &&
            !layers_[L].Wq_tl2g.empty();
        const bool useTl2Q =
            enableTl2Q &&
            tl2Supported(cfg_, H, H) &&
            !layers_[L].Wq_tl2.empty() &&
            !useTl2GenQ;
        const bool useTl2GenK =
            enableTl2K &&
            tl2GeneratedSupported(cfg_, H, kvDim) &&
            !layers_[L].Wk_tl2g.empty();
        const bool useTl2K =
            enableTl2K &&
            tl2Supported(cfg_, H, kvDim) &&
            !layers_[L].Wk_tl2.empty() &&
            !useTl2GenK;
        const bool useTl2GenV =
            enableTl2V &&
            tl2GeneratedSupported(cfg_, H, kvDim) &&
            !layers_[L].Wv_tl2g.empty();
        const bool useTl2V =
            enableTl2V &&
            tl2Supported(cfg_, H, kvDim) &&
            !layers_[L].Wv_tl2.empty() &&
            !useTl2GenV;
        const bool useI2SQ = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV && !useTl2GenQ && !useTl2Q;
        const bool useI2SK = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV && !useTl2GenK && !useTl2K;
        const bool useI2SV = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV && !useTl2GenV && !useTl2V;
        const bool useFusedI2SQKV = useI2SQ && useI2SK && useI2SV && cfg_.bitnetFuseQKV;
        const bool useSplitFusedQkvBackward = useFusedI2SQKV && splitFusedQkvBackwardEnabled();
        const bool useFusedI2SQKVBackward = useFusedI2SQKV && !useSplitFusedQkvBackward;
        const bool useFusedI2SKV = useI2SK && useI2SV && !useFusedI2SQKVBackward;
        const bool useTl2GenWo =
            enableTl2Wo &&
            tl2GeneratedSupported(cfg_, H, H) &&
            !layers_[L].Wo_tl2g.empty();
        const bool useTl2Wo =
            enableTl2Wo &&
            tl2Supported(cfg_, H, H) &&
            !layers_[L].Wo_tl2.empty() &&
            !useTl2GenWo;
        const bool useI2SWo = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeWo && !useTl2GenWo && !useTl2Wo;
        const auto preAttnStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        auto clearInactiveTail = [&](std::vector<float>& buffer, std::size_t rowWidth) {
            if (activeComputeT < T) {
                std::fill(
                    buffer.begin() + static_cast<ptrdiff_t>(activeComputeT * rowWidth),
                    buffer.begin() + static_cast<ptrdiff_t>(T * rowWidth),
                    0.0f);
            }
        };
        auto clearInactiveTailRaw = [&](float* buffer, std::size_t rowWidth) {
            if (activeComputeT < T) {
                std::fill(
                    buffer + activeComputeT * rowWidth,
                    buffer + T * rowWidth,
                    0.0f);
            }
        };
        auto prepareGeneratedOutput = [&](std::vector<float>& buffer, std::size_t rowWidth) {
            if (tl2GeneratedCallerFillsEnabled()) {
                std::fill(buffer.begin(), buffer.begin() + static_cast<ptrdiff_t>(T * rowWidth), 0.0f);
            }
            else {
                clearInactiveTail(buffer, rowWidth);
            }
        };
        auto prepareGeneratedOutputRaw = [&](float* buffer, std::size_t rowWidth) {
            if (tl2GeneratedCallerFillsEnabled()) {
                std::fill(buffer, buffer + T * rowWidth, 0.0f);
            }
            else {
                clearInactiveTailRaw(buffer, rowWidth);
            }
        };
        const bool useFusedTl2GenQV =
            useTl2GenQ &&
            !useTl2GenK &&
            useTl2GenV &&
            fusedTl2GeneratedQVEnabled() &&
            !layers_[L].Wq_tl2g.i2sMain.empty() &&
            !layers_[L].Wv_tl2g.i2sMain.empty();
        const bool useFusedHybridQkv =
            useTl2GenQ &&
            !useTl2GenK &&
            useTl2GenV &&
            useI2SK &&
            fusedTl2GeneratedHybridQkvEnabled() &&
            !layers_[L].Wq_tl2g.i2sMain.empty() &&
            !layers_[L].Wv_tl2g.i2sMain.empty();
        const bool useFusedQkvOverwrite = fusedQkvForwardOverwriteEnabled(activeComputeT, tTile);
        const std::size_t fusedQkvTileRows = exactFusedQkvForwardTileRows(activeComputeT, tTile);

        // ── QKV Projection ──
        const auto qkvStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (useTl2GenQ && useTl2GenK && useTl2GenV) {
            prepareGeneratedOutputRaw(qProj, H);
            prepareGeneratedOutputRaw(kProj, kvDim);
            prepareGeneratedOutputRaw(vProj, kvDim);
            linear_forward_bitnet_tl2_generated(qProj, attnXnormForward, layers_[L].Wq_tl2g, activeComputeT);
            linear_forward_bitnet_tl2_generated(kProj, attnXnormForward, layers_[L].Wk_tl2g, activeComputeT);
            linear_forward_bitnet_tl2_generated(vProj, attnXnormForward, layers_[L].Wv_tl2g, activeComputeT);
        }
        else if (useFusedHybridQkv) {
            if (useFusedQkvOverwrite) {
                clearInactiveTailRaw(qProj, H);
                clearInactiveTailRaw(kProj, kvDim);
                clearInactiveTailRaw(vProj, kvDim);
            }
            else {
                prepareGeneratedOutputRaw(qProj, H);
                std::fill_n(kProj, T * kvDim, 0.0f);
                prepareGeneratedOutputRaw(vProj, kvDim);
            }
            linear_fused_qkv_forward_bitnet_i2s_accum(
                qProj, kProj, vProj,
                attnXnormForward,
                layers_[L].Wq_tl2g.i2sMain, layers_[L].Wk_q, layers_[L].Wv_tl2g.i2sMain,
                activeComputeT, tTile,
                fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            linear_forward_bitnet_tl2_generated_residual_only(qProj, attnXnormForward, layers_[L].Wq_tl2g, activeComputeT);
            linear_forward_bitnet_tl2_generated_residual_only(vProj, attnXnormForward, layers_[L].Wv_tl2g, activeComputeT);
        }
        else if (useFusedTl2GenQV) {
            prepareGeneratedOutputRaw(qProj, H);
            prepareGeneratedOutputRaw(vProj, kvDim);
            linear_fused_dual_forward_bitnet_tl2_generated(
                qProj,
                vProj,
                attnXnormForward,
                layers_[L].Wq_tl2g,
                layers_[L].Wv_tl2g,
                activeComputeT);

            std::fill_n(kProj, T * kvDim, 0.0f);
            if (useTl2K) {
                linear_forward_bitnet_tl2_accum(
                    kProj, attnXnormForward, layers_[L].Wk_tl2, activeComputeT, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            }
            else if (useI2SK) {
                linear_forward_bitnet_i2s_accum(
                    kProj, attnXnormForward, layers_[L].Wk_q, activeComputeT, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
            }
            else {
                for (std::size_t t = 0; t < T; ++t) {
                    if (attentionMask[t] == 0) continue;
                    const float* xn = attnXnormForward + t * H;
                    float* kt = kProj + t * kvDim;
                    for (std::size_t i = 0; i < kvDim; ++i) {
                        float sum_k = 0; for (std::size_t j = 0; j < H; ++j) sum_k += xn[j] * layers_[L].Wk[j * kvDim + i]; kt[i] = sum_k;
                    }
                }
            }
        }
        else if (useTl2Q && useTl2K && useTl2V) {
            std::fill_n(qProj, T * H, 0.0f);
            std::fill_n(kProj, T * kvDim, 0.0f);
            std::fill_n(vProj, T * kvDim, 0.0f);

            linear_fused_qkv_forward_bitnet_tl2_accum(
                qProj, kProj, vProj,
                attnXnormForward,
                layers_[L].Wq_tl2, layers_[L].Wk_tl2, layers_[L].Wv_tl2,
                activeComputeT, tTile,
                fwdX8Scratch_.data(), fwdXScaleScratch_.data());
        }
        else if (useFusedI2SQKV) {
            if (useFusedQkvOverwrite) {
                clearInactiveTailRaw(qProj, H);
                clearInactiveTailRaw(kProj, kvDim);
                clearInactiveTailRaw(vProj, kvDim);
            }
            else {
                std::fill_n(qProj, T * H, 0.0f);
                std::fill_n(kProj, T * kvDim, 0.0f);
                std::fill_n(vProj, T * kvDim, 0.0f);
            }

            linear_fused_qkv_forward_bitnet_i2s_accum(
                qProj, kProj, vProj,
                attnXnormForward,
                layers_[L].Wq_q, layers_[L].Wk_q, layers_[L].Wv_q,
                activeComputeT, fusedQkvTileRows,
                fwdX8Scratch_.data(), fwdXScaleScratch_.data());
        }
        else if (useI2SQ && useI2SK && useI2SV) {
            std::fill_n(qProj, T * H, 0.0f);
            std::fill_n(kProj, T * kvDim, 0.0f);
            std::fill_n(vProj, T * kvDim, 0.0f);

            linear_forward_bitnet_i2s_accum(qProj, attnXnormForward, layers_[L].Wq_q, activeComputeT, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
            linear_forward_bitnet_i2s_accum(kProj, attnXnormForward, layers_[L].Wk_q, activeComputeT, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
            linear_forward_bitnet_i2s_accum(vProj, attnXnormForward, layers_[L].Wv_q, activeComputeT, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
        }
        else {
            if (useTl2GenQ && !useFusedTl2GenQV) {
                prepareGeneratedOutputRaw(qProj, H);
                linear_forward_bitnet_tl2_generated(qProj, attnXnormForward, layers_[L].Wq_tl2g, activeComputeT);
            }
            else {
                std::fill_n(qProj, T * H, 0.0f);
                if (useTl2Q) {
                    linear_forward_bitnet_tl2_accum(
                        qProj, attnXnormForward, layers_[L].Wq_tl2, activeComputeT, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
                else if (useI2SQ) {
                    linear_forward_bitnet_i2s_accum(
                        qProj, attnXnormForward, layers_[L].Wq_q, activeComputeT, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                }
                else {
                    for (std::size_t t = 0; t < T; ++t) {
                        if (attentionMask[t] == 0) continue;
                        const float* xn = attnXnormForward + t * H;
                        float* qt = qProj + t * H;
                        for (std::size_t i = 0; i < H; ++i) {
                            float sum_q = 0; for (std::size_t j = 0; j < H; ++j) sum_q += xn[j] * layers_[L].Wq[j * H + i]; qt[i] = sum_q;
                        }
                    }
                }
            }

            if (useFusedI2SKV) {
                std::fill_n(kProj, T * kvDim, 0.0f);
                std::fill_n(vProj, T * kvDim, 0.0f);
                linear_fused_dual_forward_bitnet_i2s_accum(
                    kProj, vProj,
                    attnXnormForward,
                    layers_[L].Wk_q, layers_[L].Wv_q,
                    activeComputeT, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
            }
            else {
                if (useTl2GenK) {
                    prepareGeneratedOutputRaw(kProj, kvDim);
                    linear_forward_bitnet_tl2_generated(kProj, attnXnormForward, layers_[L].Wk_tl2g, activeComputeT);
                }
                else {
                    std::fill_n(kProj, T * kvDim, 0.0f);
                    if (useTl2K) {
                        linear_forward_bitnet_tl2_accum(
                            kProj, attnXnormForward, layers_[L].Wk_tl2, activeComputeT, tTile,
                            fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                    }
                    else if (useI2SK) {
                        linear_forward_bitnet_i2s_accum(
                            kProj, attnXnormForward, layers_[L].Wk_q, activeComputeT, tTile,
                            fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                    }
                    else {
                        for (std::size_t t = 0; t < T; ++t) {
                            if (attentionMask[t] == 0) continue;
                            const float* xn = attnXnormForward + t * H;
                            float* kt = kProj + t * kvDim;
                            for (std::size_t i = 0; i < kvDim; ++i) {
                                float sum_k = 0; for (std::size_t j = 0; j < H; ++j) sum_k += xn[j] * layers_[L].Wk[j * kvDim + i]; kt[i] = sum_k;
                            }
                        }
                    }
                }

                if (useTl2GenV && !useFusedTl2GenQV) {
                    prepareGeneratedOutputRaw(vProj, kvDim);
                    linear_forward_bitnet_tl2_generated(vProj, attnXnormForward, layers_[L].Wv_tl2g, activeComputeT);
                }
                else {
                    std::fill_n(vProj, T * kvDim, 0.0f);
                    if (useTl2V) {
                        linear_forward_bitnet_tl2_accum(
                            vProj, attnXnormForward, layers_[L].Wv_tl2, activeComputeT, tTile,
                            fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                    }
                    else if (useI2SV) {
                        linear_forward_bitnet_i2s_accum(
                            vProj, attnXnormForward, layers_[L].Wv_q, activeComputeT, tTile,
                            fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                    }
                    else {
                        for (std::size_t t = 0; t < T; ++t) {
                            if (attentionMask[t] == 0) continue;
                            const float* xn = attnXnormForward + t * H;
                            float* vt = vProj + t * kvDim;
                            for (std::size_t i = 0; i < kvDim; ++i) {
                                float sum_v = 0; for (std::size_t j = 0; j < H; ++j) sum_v += xn[j] * layers_[L].Wv[j * kvDim + i]; vt[i] = sum_v;
                            }
                        }
                    }
                }
            }
        }
        if (profile) prof.fwdQkvMs += elapsedMs(qkvStart);

        // ── LoRA Injection ──
        const bool parallelAttentionPrep = parallelAttentionPrepEnabled(activeComputeT) && activeComputeT > 1;
        const auto loraStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (cfg_.useDoRA) {
            auto& P = loras_[L];
            if (cfg_.loraTargetQ) {
                doraStats.merge(!layers_[L].Wq_q.empty()
                    ? doraUpdateI2S(layers_[L].Wq_q, P.Aq, P.Bq, P.Mq, P.Cq, P.Rq, H, H)
                    : doraUpdateDense(layers_[L].Wq, P.Aq, P.Bq, P.Mq, P.Cq, P.Rq, H, H));
            }
            if (cfg_.loraTargetK) {
                doraStats.merge(!layers_[L].Wk_q.empty()
                    ? doraUpdateI2S(layers_[L].Wk_q, P.Ak, P.Bk, P.Mk, P.Ck, P.Rk, H, kvDim)
                    : doraUpdateDense(layers_[L].Wk, P.Ak, P.Bk, P.Mk, P.Ck, P.Rk, H, kvDim));
            }
            if (cfg_.loraTargetV) {
                doraStats.merge(!layers_[L].Wv_q.empty()
                    ? doraUpdateI2S(layers_[L].Wv_q, P.Av, P.Bv, P.Mv, P.Cv, P.Rv, H, kvDim)
                    : doraUpdateDense(layers_[L].Wv, P.Av, P.Bv, P.Mv, P.Cv, P.Rv, H, kvDim));
            }
        }
        if (parallelAttentionPrep) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t t = 0; t < static_cast<ptrdiff_t>(activeComputeT); ++t) {
                const std::size_t tu = static_cast<std::size_t>(t);
                float* tmpQ = &tmpQ_cache_[(L * cfg_.maxSeqLen + tu) * R];
                float* tmpK = &tmpK_cache_[(L * cfg_.maxSeqLen + tu) * R];
                float* tmpV = &tmpV_cache_[(L * cfg_.maxSeqLen + tu) * R];

                if (activeAttentionMask && activeAttentionMask[tu] == 0) {
                    std::fill(tmpQ, tmpQ + R, 0.0f);
                    std::fill(tmpK, tmpK + R, 0.0f);
                    std::fill(tmpV, tmpV + R, 0.0f);
                    continue;
                }

                const float* xn = attnXnormForward + tu * H;
                if (cfg_.loraTargetQ) {
                    lora_add_H(qProj + tu * H, xn, loras_[L].Aq.data(), loras_[L].Bq.data(), tmpQ, H, R, scale, hTile);
                    if (cfg_.useDoRA) dora_apply_ratio(qProj + tu * H, loras_[L].Rq, H);
                }
                else {
                    std::fill(tmpQ, tmpQ + R, 0.0f);
                }
                if (cfg_.loraTargetK) {
                    lora_add_out(kProj + tu * kvDim, xn, loras_[L].Ak.data(), loras_[L].Bk.data(), tmpK, H, kvDim, R, scale);
                    if (cfg_.useDoRA) dora_apply_ratio(kProj + tu * kvDim, loras_[L].Rk, kvDim);
                }
                else {
                    std::fill(tmpK, tmpK + R, 0.0f);
                }
                if (cfg_.loraTargetV) {
                    lora_add_out(vProj + tu * kvDim, xn, loras_[L].Av.data(), loras_[L].Bv.data(), tmpV, H, kvDim, R, scale);
                    if (cfg_.useDoRA) dora_apply_ratio(vProj + tu * kvDim, loras_[L].Rv, kvDim);
                }
                else {
                    std::fill(tmpV, tmpV + R, 0.0f);
                }
            }
        }
        else {
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                float* tmpQ = &tmpQ_cache_[(L * cfg_.maxSeqLen + t) * R];
                float* tmpK = &tmpK_cache_[(L * cfg_.maxSeqLen + t) * R];
                float* tmpV = &tmpV_cache_[(L * cfg_.maxSeqLen + t) * R];

                if (activeAttentionMask && activeAttentionMask[t] == 0) {
                    std::fill(tmpQ, tmpQ + R, 0.0f);
                    std::fill(tmpK, tmpK + R, 0.0f);
                    std::fill(tmpV, tmpV + R, 0.0f);
                    continue;
                }

                const float* xn = attnXnormForward + t * H;
                if (cfg_.loraTargetQ) {
                    lora_add_H(qProj + t * H, xn, loras_[L].Aq.data(), loras_[L].Bq.data(), tmpQ, H, R, scale, hTile);
                    if (cfg_.useDoRA) dora_apply_ratio(qProj + t * H, loras_[L].Rq, H);
                }
                else {
                    std::fill(tmpQ, tmpQ + R, 0.0f);
                }
                if (cfg_.loraTargetK) {
                    lora_add_out(kProj + t * kvDim, xn, loras_[L].Ak.data(), loras_[L].Bk.data(), tmpK, H, kvDim, R, scale);
                    if (cfg_.useDoRA) dora_apply_ratio(kProj + t * kvDim, loras_[L].Rk, kvDim);
                }
                else {
                    std::fill(tmpK, tmpK + R, 0.0f);
                }
                if (cfg_.loraTargetV) {
                    lora_add_out(vProj + t * kvDim, xn, loras_[L].Av.data(), loras_[L].Bv.data(), tmpV, H, kvDim, R, scale);
                    if (cfg_.useDoRA) dora_apply_ratio(vProj + t * kvDim, loras_[L].Rv, kvDim);
                }
                else {
                    std::fill(tmpV, tmpV + R, 0.0f);
                }
            }
        }
        for (std::size_t t = activeComputeT; t < T; ++t) {
            float* tmpQ = &tmpQ_cache_[(L * cfg_.maxSeqLen + t) * R];
            float* tmpK = &tmpK_cache_[(L * cfg_.maxSeqLen + t) * R];
            float* tmpV = &tmpV_cache_[(L * cfg_.maxSeqLen + t) * R];
            std::fill(tmpQ, tmpQ + R, 0.0f);
            std::fill(tmpK, tmpK + R, 0.0f);
            std::fill(tmpV, tmpV + R, 0.0f);
        }
        if (profile) prof.fwdLoraMs += elapsedMs(loraStart);

        if (cfg_.useDoRA && !cfg_.doraRecomputeQKPreRope) {
            if (cfg_.loraTargetQ) {
                std::copy_n(qProj, T * H, q_dora_prerope_cache_.data() + layerTHOffset);
            }
            if (cfg_.loraTargetK) {
                std::copy_n(kProj, T * kvDim, k_dora_prerope_cache_.data() + layerTKVOffset);
            }
        }

        // ── RoPE on Q/K (pre-attention) ──
        const auto ropeStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (parallelAttentionPrep) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t t = 0; t < static_cast<ptrdiff_t>(activeComputeT); ++t) {
                const std::size_t tu = static_cast<std::size_t>(t);
                if (activeAttentionMask && activeAttentionMask[tu] == 0) continue;
                const std::size_t ropePos = segmentStarts.empty() ? tu : (tu - segmentStarts[tu]);

                for (std::size_t h = 0; h < nHeads; ++h) {
                    apply_rope_forward_inplace_cached(
                        qProj + tu * H + h * D,
                        D,
                        ropeDim,
                        ropePos,
                        ropeCosTable_.data(),
                        ropeSinTable_.data());
                }
                for (std::size_t kvh = 0; kvh < nKvHeads; ++kvh) {
                    apply_rope_forward_inplace_cached(
                        kProj + tu * kvDim + kvh * D,
                        D,
                        ropeDim,
                        ropePos,
                        ropeCosTable_.data(),
                        ropeSinTable_.data());
                }
            }
        }
        else {
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                const std::size_t ropePos = segmentStarts.empty() ? t : (t - segmentStarts[t]);

                for (std::size_t h = 0; h < nHeads; ++h) {
                    apply_rope_forward_inplace_cached(
                        qProj + t * H + h * D,
                        D,
                        ropeDim,
                        ropePos,
                        ropeCosTable_.data(),
                        ropeSinTable_.data());
                }
                for (std::size_t kvh = 0; kvh < nKvHeads; ++kvh) {
                    apply_rope_forward_inplace_cached(
                        kProj + t * kvDim + kvh * D,
                        D,
                        ropeDim,
                        ropePos,
                        ropeCosTable_.data(),
                        ropeSinTable_.data());
                }
            }
        }
        if (profile) prof.fwdRopeMs += elapsedMs(ropeStart);

        const auto cacheStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (!fusedAttentionQkvCacheWrite) {
            std::copy_n(qProj, T * H, layerQCache);
            std::copy_n(kProj, T * kvDim, layerKCache);
            std::copy_n(vProj, T * kvDim, layerVCache);
        }
        if (profile) prof.fwdCacheMs += elapsedMs(cacheStart);
        if (profile) prof.fwdAttnPrepMs += elapsedMs(preAttnStart);

        // ── Attention ──
        const auto attnCoreStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (activeComputeT < T) {
            std::fill(ctx_.begin() + (activeComputeT * H), ctx_.begin() + (T * H), 0.0f);
        }
        float* attnScores = attentionStoreScoresEnabled() ? scores_.data() : nullptr;
        attn_causal(
            ctx_.data(), qProj, kProj, vProj,
            activeComputeT, H, shape_.nHeads, nKvHeads, attnScores, probs_.data(),
            activeAttentionMask
        );
        if (layerCtxCache) {
            std::copy_n(ctx_.data(), T * H, layerCtxCache);
        }
        if (!probs_cache_.empty()) {
            std::copy_n(probs_.data(), nHeads * T * T, probs_cache_.data() + layerProbsOffset);
        }

        // ── Output Projection (Wo) ──
        if (useTl2GenWo) {
            prepareGeneratedOutput(attn_, H);
            linear_forward_bitnet_tl2_generated(attn_.data(), ctx_.data(), layers_[L].Wo_tl2g, activeComputeT);
        }
        else if (useTl2Wo) {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);
            linear_forward_bitnet_tl2_accum(
                attn_.data(), ctx_.data(), layers_[L].Wo_tl2, activeComputeT, tTile,
                fwdX8Scratch_.data(), fwdXScaleScratch_.data());
        }
        else if (useI2SWo) {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);
            const std::size_t nXBlocks = layers_[L].Wo_q.nKBlocks();
            (void)nXBlocks;
            linear_forward_bitnet_i2s_accum(attn_.data(), ctx_.data(), layers_[L].Wo_q, activeComputeT, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
        }
        else {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                const float* ct = &ctx_[t * H];
                float* at = &attn_[t * H];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum = 0; for (std::size_t j = 0; j < H; ++j) sum += ct[j] * layers_[L].Wo[j * H + i];
                    at[i] = sum;
                }
            }
        }

        if (cfg_.loraTargetO) {
            if (cfg_.useDoRA) {
                doraStats.merge(!layers_[L].Wo_q.empty()
                    ? doraUpdateI2S(
                        layers_[L].Wo_q, loras_[L].Ao, loras_[L].Bo,
                        loras_[L].Mo, loras_[L].Co, loras_[L].Ro, H, H)
                    : doraUpdateDense(
                        layers_[L].Wo, loras_[L].Ao, loras_[L].Bo,
                        loras_[L].Mo, loras_[L].Co, loras_[L].Ro, H, H));
            }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t t_i = 0; t_i < static_cast<ptrdiff_t>(activeComputeT); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                float* tmpO = &tmpO_cache_[(L * cfg_.maxSeqLen + t) * R];
                if (activeAttentionMask && activeAttentionMask[t] == 0) {
                    std::fill(tmpO, tmpO + R, 0.0f);
                    continue;
                }
                lora_add_H(&attn_[t * H], &ctx_[t * H], loras_[L].Ao.data(), loras_[L].Bo.data(), tmpO, H, R, scale, hTile);
                if (cfg_.useDoRA) dora_apply_ratio(&attn_[t * H], loras_[L].Ro, H);
            }
        }
        else {
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                float* tmpO = &tmpO_cache_[(L * cfg_.maxSeqLen + t) * R];
                std::fill(tmpO, tmpO + R, 0.0f);
            }
        }
        for (std::size_t t = activeComputeT; t < T; ++t) {
            float* tmpO = &tmpO_cache_[(L * cfg_.maxSeqLen + t) * R];
            std::fill(tmpO, tmpO + R, 0.0f);
        }

        // First residual: x_mid = x + attn
        for (std::size_t t = 0; t < activeComputeT; ++t) {
            float* xMid = &x_mid_[t * H];
            const float* xIn = &x_[t * H];
            const float* at = &attn_[t * H];
            if (activeAttentionMask && activeAttentionMask[t] == 0) {
                std::fill(xMid, xMid + H, 0.0f);
                continue;
            }
            for (std::size_t h = 0; h < H; ++h) xMid[h] = xIn[h] + at[h];
        }
        for (std::size_t t = activeComputeT; t < T; ++t) {
            std::fill(&x_mid_[t * H], &x_mid_[t * H] + H, 0.0f);
        }
        std::copy_n(x_mid_.data(), T * H, x_mid_cache_.data() + layerTHOffset);
        if (profile) prof.fwdAttnMs += elapsedMs(attnCoreStart);

        // FFN pre-norm.
        const auto ffnStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        const auto ffnNormStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (parallelRmsnormForwardEnabled() && activeComputeT > 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t t = 0; t < static_cast<ptrdiff_t>(activeComputeT); ++t) {
                float* xn = &xnorm_ffn_[static_cast<std::size_t>(t) * H];
                if (activeAttentionMask && activeAttentionMask[static_cast<std::size_t>(t)] == 0) {
                    std::fill(xn, xn + H, 0.0f);
                    continue;
                }
                rmsnorm(xn, &x_mid_[static_cast<std::size_t>(t) * H], layers_[L].rms_ffn.data(), H, eps);
            }
        }
        else {
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                float* xn = &xnorm_ffn_[t * H];
                if (activeAttentionMask && activeAttentionMask[t] == 0) {
                    std::fill(xn, xn + H, 0.0f);
                    continue;
                }
                rmsnorm(xn, &x_mid_[t * H], layers_[L].rms_ffn.data(), H, eps);
            }
        }
        for (std::size_t t = activeComputeT; t < T; ++t) {
            std::fill(&xnorm_ffn_[t * H], &xnorm_ffn_[t * H] + H, 0.0f);
        }
        if (ffnGateUpLoraEnabled) {
            std::copy_n(xnorm_ffn_.data(), T * H, xnorm_ffn_cache_.data() + layerTHOffset);
        }
        if (profile) prof.fwdFfnNormMs += elapsedMs(ffnNormStart);

        if (cfg_.useDoRA && ffnGateUpLoraEnabled) {
            doraStats.merge(!layers_[L].Wgate_q.empty()
                ? doraUpdateI2S(
                    layers_[L].Wgate_q, loras_[L].Afg, loras_[L].Bfg,
                    loras_[L].Mfg, loras_[L].Cfg, loras_[L].Rfg, H, F)
                : doraUpdateDense(
                    layers_[L].Wgate, loras_[L].Afg, loras_[L].Bfg,
                    loras_[L].Mfg, loras_[L].Cfg, loras_[L].Rfg, H, F));
            doraStats.merge(!layers_[L].Wup_q.empty()
                ? doraUpdateI2S(
                    layers_[L].Wup_q, loras_[L].Afu, loras_[L].Bfu,
                    loras_[L].Mfu, loras_[L].Cfu, loras_[L].Rfu, H, F)
                : doraUpdateDense(
                    layers_[L].Wup, loras_[L].Afu, loras_[L].Bfu,
                    loras_[L].Mfu, loras_[L].Cfu, loras_[L].Rfu, H, F));
        }
        auto addFfnGateUpLoraForward = [&]() {
            if (!ffnGateUpLoraEnabled) return;
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                float* tmpFg = &tmpFfnGate_cache_[((L * cfg_.maxSeqLen) + t) * R];
                float* tmpFu = &tmpFfnUp_cache_[((L * cfg_.maxSeqLen) + t) * R];
                if (activeAttentionMask && activeAttentionMask[t] == 0) {
                    std::fill(tmpFg, tmpFg + R, 0.0f);
                    std::fill(tmpFu, tmpFu + R, 0.0f);
                    continue;
                }
                const float* xn = &xnorm_ffn_[t * H];
                lora_add_out(&gate_[t * F], xn, loras_[L].Afg.data(), loras_[L].Bfg.data(), tmpFg, H, F, R, scale);
                lora_add_out(&up_[t * F], xn, loras_[L].Afu.data(), loras_[L].Bfu.data(), tmpFu, H, F, R, scale);
                if (cfg_.useDoRA) {
                    dora_apply_ratio(&gate_[t * F], loras_[L].Rfg, F);
                    dora_apply_ratio(&up_[t * F], loras_[L].Rfu, F);
                }
            }
            for (std::size_t t = activeComputeT; t < T; ++t) {
                float* tmpFg = &tmpFfnGate_cache_[((L * cfg_.maxSeqLen) + t) * R];
                float* tmpFu = &tmpFfnUp_cache_[((L * cfg_.maxSeqLen) + t) * R];
                std::fill(tmpFg, tmpFg + R, 0.0f);
                std::fill(tmpFu, tmpFu + R, 0.0f);
            }
        };

        const bool enableTl2Gate = tl2GateEnabled(cfg_);
        const bool enableTl2Up = tl2UpEnabled(cfg_);
        const bool enableTl2Down = tl2DownEnabled(cfg_);
        const bool useTl2GenGate =
            enableTl2Gate &&
            tl2GeneratedSupported(cfg_, H, F) &&
            !layers_[L].Wgate_tl2g.empty();
        const bool useTl2Gate =
            enableTl2Gate &&
            tl2Supported(cfg_, H, F) &&
            !layers_[L].Wgate_tl2.empty() &&
            !useTl2GenGate;
        const bool useTl2GenUp =
            enableTl2Up &&
            tl2GeneratedSupported(cfg_, H, F) &&
            !layers_[L].Wup_tl2g.empty();
        const bool useTl2Up =
            enableTl2Up &&
            tl2Supported(cfg_, H, F) &&
            !layers_[L].Wup_tl2.empty() &&
            !useTl2GenUp;
        const bool useTl2GenDown =
            enableTl2Down &&
            tl2GeneratedSupported(cfg_, F, H) &&
            !layers_[L].Wdown_tl2g.empty();
        const bool useTl2Down =
            enableTl2Down &&
            tl2Supported(cfg_, F, H) &&
            !layers_[L].Wdown_tl2.empty() &&
            !useTl2GenDown;
        const bool useI2SGate = useBitNetFfn && !useTl2GenGate && !useTl2Gate;
        const bool useI2SUp = useBitNetFfn && !useTl2GenUp && !useTl2Up;
        const bool useI2SDown = useBitNetFfn && !useTl2GenDown && !useTl2Down;
        const bool useFusedI2SGateUp = useI2SGate && useI2SUp;
        if (useBitNetFfn) {
            if (useTl2GenGate) {
                prepareGeneratedOutput(gate_, F);
            }
            else {
                std::fill(gate_.begin(), gate_.end(), 0.0f);
            }
            if (useTl2GenUp) {
                prepareGeneratedOutput(up_, F);
            }
            else {
                std::fill(up_.begin(), up_.end(), 0.0f);
            }
            if (useTl2GenDown) {
                prepareGeneratedOutput(ffn_out_, H);
            }
            else {
                std::fill(ffn_out_.begin(), ffn_out_.end(), 0.0f);
            }

            if (useFusedI2SGateUp) {
                linear_fused_dual_forward_bitnet_i2s_accum(
                    gate_.data(), up_.data(),
                    xnorm_ffn_.data(),
                    layers_[L].Wgate_q, layers_[L].Wup_q,
                    activeComputeT, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
            }
            else {
                if (useTl2GenGate) {
                    linear_forward_bitnet_tl2_generated(gate_.data(), xnorm_ffn_.data(), layers_[L].Wgate_tl2g, activeComputeT);
                }
                else if (useTl2Gate) {
                    linear_forward_bitnet_tl2_accum(
                        gate_.data(), xnorm_ffn_.data(), layers_[L].Wgate_tl2, activeComputeT, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
                else if (useI2SGate) {
                    linear_forward_bitnet_i2s_accum(
                        gate_.data(), xnorm_ffn_.data(), layers_[L].Wgate_q, activeComputeT, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                }

                if (useTl2GenUp) {
                    linear_forward_bitnet_tl2_generated(up_.data(), xnorm_ffn_.data(), layers_[L].Wup_tl2g, activeComputeT);
                }
                else if (useTl2Up) {
                    linear_forward_bitnet_tl2_accum(
                        up_.data(), xnorm_ffn_.data(), layers_[L].Wup_tl2, activeComputeT, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
                else if (useI2SUp) {
                    linear_forward_bitnet_i2s_accum(
                        up_.data(), xnorm_ffn_.data(), layers_[L].Wup_q, activeComputeT, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                }
            }
            addFfnGateUpLoraForward();
            if (((activePrefixT == T) || prefixMaskFastpathEnabled()) &&
                activePrefixT != static_cast<std::size_t>(-1)) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                for (int t_i = 0; t_i < static_cast<int>(activePrefixT); ++t_i) {
                    const std::size_t t = static_cast<std::size_t>(t_i);
                    float* gate_t = &gate_[t * F];
                    float* up_t = &up_[t * F];
                    float* act_t = &ffn_act_[t * F];
                    if (fastSiluForwardEnabled()) {
                        silu_mul_avx2_or_scalar(act_t, gate_t, up_t, F);
                    }
                    else {
                        silu_mul_exact_scalar(act_t, gate_t, up_t, F);
                    }
                }
                if (activePrefixT < T) {
                    const std::size_t tailOffset = activePrefixT * F;
                    const std::size_t tailCount = (T - activePrefixT) * F;
                    std::fill_n(ffn_act_.data() + tailOffset, tailCount, 0.0f);
                }
            }
            else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                    const std::size_t t = static_cast<std::size_t>(t_i);
                    float* gate_t = &gate_[t * F];
                    float* up_t = &up_[t * F];
                    float* act_t = &ffn_act_[t * F];
                    if (attentionMask[t] == 0) {
                        std::fill(gate_t, gate_t + F, 0.0f);
                        std::fill(up_t, up_t + F, 0.0f);
                        std::fill(act_t, act_t + F, 0.0f);
                        continue;
                    }
                    if (fastSiluForwardEnabled()) {
                        silu_mul_avx2_or_scalar(act_t, gate_t, up_t, F);
                    }
                    else {
                        silu_mul_exact_scalar(act_t, gate_t, up_t, F);
                    }
                }
            }
            if (useTl2GenDown) {
                linear_forward_bitnet_tl2_generated(ffn_out_.data(), ffn_act_.data(), layers_[L].Wdown_tl2g, activeComputeT);
            }
            else if (useTl2Down) {
                linear_forward_bitnet_tl2_accum(
                    ffn_out_.data(), ffn_act_.data(), layers_[L].Wdown_tl2, activeComputeT, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            }
            else if (useI2SDown) {
                linear_forward_bitnet_i2s_accum(ffn_out_.data(), ffn_act_.data(), layers_[L].Wdown_q, activeComputeT, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
            }
        }
        else if (useFfnGemm) {
            linear_THF_fused_dual_accum_blocked32(
                gate_.data(),
                up_.data(),
                xnorm_ffn_.data(),
                layers_[L].Wgate.data(),
                layers_[L].Wup.data(),
                T, H, F, cfg_.hTile, tTile);
            addFfnGateUpLoraForward();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                float* gate_t = &gate_[t * F];
                float* up_t = &up_[t * F];
                float* act_t = &ffn_act_[t * F];
                if (attentionMask[t] == 0) {
                    std::fill(gate_t, gate_t + F, 0.0f);
                    std::fill(up_t, up_t + F, 0.0f);
                    std::fill(act_t, act_t + F, 0.0f);
                    continue;
                }
                if (fp32FfnFastSiluForwardEnabled()) {
                    silu_mul_avx2_or_scalar(act_t, gate_t, up_t, F);
                }
                else {
                    silu_mul_exact_scalar(act_t, gate_t, up_t, F);
                }
            }

            linear_TFH_accum_blocked32(
                ffn_out_.data(),
                ffn_act_.data(),
                layers_[L].Wdown.data(),
                T, F, H, cfg_.hTile, tTile);
        }
        else {
            if (ffnGateUpLoraEnabled) {
                throw std::runtime_error("FFN gate/up LoRA requires BitNet/TL2 FFN or QLORA_TEST_FFN_GEMM=1");
            }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int t0_i = 0; t0_i < static_cast<int>(T); t0_i += 4) {
                const std::size_t t0 = static_cast<std::size_t>(t0_i);
                const std::size_t tt = std::min<std::size_t>(4, T - t0);
                if (tt < 4) {
                    for (std::size_t r = 0; r < tt; ++r) {
                        const std::size_t t = t0 + r;
                        float* gate_t = &gate_[t * F];
                        float* up_t = &up_[t * F];
                        float* act_t = &ffn_act_[t * F];
                        float* out_t = &ffn_out_[t * H];

                        if (attentionMask[t] == 0) {
                            std::fill(gate_t, gate_t + F, 0.0f);
                            std::fill(up_t, up_t + F, 0.0f);
                            std::fill(act_t, act_t + F, 0.0f);
                            std::fill(out_t, out_t + H, 0.0f);
                            continue;
                        }

                        std::fill(gate_t, gate_t + F, 0.0f);
                        std::fill(up_t, up_t + F, 0.0f);
                        for (std::size_t h = 0; h < H; ++h) {
                            const float xh = xnorm_ffn_[t * H + h];
                            const float* wGateRow = &layers_[L].Wgate[h * F];
                            const float* wUpRow = &layers_[L].Wup[h * F];
                            axpy2_inplace_avx2_or_scalar(gate_t, wGateRow, up_t, wUpRow, F, xh);
                        }

                        if (fp32FfnFastSiluForwardEnabled()) {
                            silu_mul_avx2_or_scalar(act_t, gate_t, up_t, F);
                        }
                        else {
                            silu_mul_exact_scalar(act_t, gate_t, up_t, F);
                        }

                        std::fill(out_t, out_t + H, 0.0f);
                        for (std::size_t f = 0; f < F; ++f) {
                            axpy_inplace_avx2_or_scalar(out_t, &layers_[L].Wdown[f * H], H, act_t[f]);
                        }
                    }
                    continue;
                }

                float* gate0 = &gate_[(t0 + 0) * F];
                float* gate1 = &gate_[(t0 + 1) * F];
                float* gate2 = &gate_[(t0 + 2) * F];
                float* gate3 = &gate_[(t0 + 3) * F];
                float* up0 = &up_[(t0 + 0) * F];
                float* up1 = &up_[(t0 + 1) * F];
                float* up2 = &up_[(t0 + 2) * F];
                float* up3 = &up_[(t0 + 3) * F];
                float* act0 = &ffn_act_[(t0 + 0) * F];
                float* act1 = &ffn_act_[(t0 + 1) * F];
                float* act2 = &ffn_act_[(t0 + 2) * F];
                float* act3 = &ffn_act_[(t0 + 3) * F];
                float* out0 = &ffn_out_[(t0 + 0) * H];
                float* out1 = &ffn_out_[(t0 + 1) * H];
                float* out2 = &ffn_out_[(t0 + 2) * H];
                float* out3 = &ffn_out_[(t0 + 3) * H];

                std::fill(gate0, gate0 + F, 0.0f); std::fill(up0, up0 + F, 0.0f); std::fill(out0, out0 + H, 0.0f);
                std::fill(gate1, gate1 + F, 0.0f); std::fill(up1, up1 + F, 0.0f); std::fill(out1, out1 + H, 0.0f);
                std::fill(gate2, gate2 + F, 0.0f); std::fill(up2, up2 + F, 0.0f); std::fill(out2, out2 + H, 0.0f);
                std::fill(gate3, gate3 + F, 0.0f); std::fill(up3, up3 + F, 0.0f); std::fill(out3, out3 + H, 0.0f);

                for (std::size_t h = 0; h < H; ++h) {
                    const float* wGateRow = &layers_[L].Wgate[h * F];
                    const float* wUpRow = &layers_[L].Wup[h * F];
                    const float x0 = (attentionMask[t0 + 0] == 0) ? 0.0f : xnorm_ffn_[(t0 + 0) * H + h];
                    const float x1 = (attentionMask[t0 + 1] == 0) ? 0.0f : xnorm_ffn_[(t0 + 1) * H + h];
                    const float x2 = (attentionMask[t0 + 2] == 0) ? 0.0f : xnorm_ffn_[(t0 + 2) * H + h];
                    const float x3 = (attentionMask[t0 + 3] == 0) ? 0.0f : xnorm_ffn_[(t0 + 3) * H + h];
                    axpy2_x4_inplace_avx2_or_scalar(
                        gate0, up0, gate1, up1, gate2, up2, gate3, up3,
                        wGateRow, wUpRow, F, x0, x1, x2, x3);
                }

                if (fp32FfnFastSiluForwardEnabled()) {
                    silu_mul_avx2_or_scalar(act0, gate0, up0, F);
                    silu_mul_avx2_or_scalar(act1, gate1, up1, F);
                    silu_mul_avx2_or_scalar(act2, gate2, up2, F);
                    silu_mul_avx2_or_scalar(act3, gate3, up3, F);
                    for (std::size_t f = 0; f < F; ++f) {
                        axpy_x4_inplace_avx2_or_scalar(out0, out1, out2, out3, &layers_[L].Wdown[f * H], H, act0[f], act1[f], act2[f], act3[f]);
                    }
                }
                else {
                    for (std::size_t f = 0; f < F; ++f) {
                        const float a0 = (attentionMask[t0 + 0] == 0) ? 0.0f : (silu_scalar(gate0[f]) * up0[f]);
                        const float a1 = (attentionMask[t0 + 1] == 0) ? 0.0f : (silu_scalar(gate1[f]) * up1[f]);
                        const float a2 = (attentionMask[t0 + 2] == 0) ? 0.0f : (silu_scalar(gate2[f]) * up2[f]);
                        const float a3 = (attentionMask[t0 + 3] == 0) ? 0.0f : (silu_scalar(gate3[f]) * up3[f]);
                        act0[f] = a0; act1[f] = a1; act2[f] = a2; act3[f] = a3;
                        axpy_x4_inplace_avx2_or_scalar(out0, out1, out2, out3, &layers_[L].Wdown[f * H], H, a0, a1, a2, a3);
                    }
                }
            }
        }
        std::copy_n(gate_.data(), T * F, gate_cache_.data() + layerTFOffset);
        std::copy_n(up_.data(), T * F, up_cache_.data() + layerTFOffset);
        if (layerFfnActCache) {
            std::copy_n(ffn_act_.data(), T * F, layerFfnActCache);
            if (cfg_.useDoRA && ffnDownLoraEnabled) {
                doraStats.merge(!layers_[L].Wdown_q.empty()
                    ? doraUpdateI2S(
                        layers_[L].Wdown_q, loras_[L].Afd, loras_[L].Bfd,
                        loras_[L].Mfd, loras_[L].Cfd, loras_[L].Rfd, F, H)
                    : doraUpdateDense(
                        layers_[L].Wdown, loras_[L].Afd, loras_[L].Bfd,
                        loras_[L].Mfd, loras_[L].Cfd, loras_[L].Rfd, F, H));
            }
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                float* tmpFd = &tmpFfnDown_cache_[(L * cfg_.maxSeqLen + t) * R];
                if (activeAttentionMask && activeAttentionMask[t] == 0) {
                    std::fill(tmpFd, tmpFd + R, 0.0f);
                    continue;
                }
                lora_add_out(&ffn_out_[t * H], &ffn_act_[t * F], loras_[L].Afd.data(), loras_[L].Bfd.data(), tmpFd, F, H, R, scale);
                if (cfg_.useDoRA) dora_apply_ratio(&ffn_out_[t * H], loras_[L].Rfd, H);
            }
            for (std::size_t t = activeComputeT; t < T; ++t) {
                float* tmpFd = &tmpFfnDown_cache_[(L * cfg_.maxSeqLen + t) * R];
                std::fill(tmpFd, tmpFd + R, 0.0f);
            }
        }

        // Second residual: x = x_mid + ffn_out
        for (std::size_t t = 0; t < activeComputeT; ++t) {
            float* xOut = &x_[t * H];
            const float* xMid = &x_mid_[t * H];
            const float* ffn = &ffn_out_[t * H];
            if (activeAttentionMask && activeAttentionMask[t] == 0) {
                std::fill(xOut, xOut + H, 0.0f);
                continue;
            }
            for (std::size_t h = 0; h < H; ++h) xOut[h] = xMid[h] + ffn[h];
        }
        for (std::size_t t = activeComputeT; t < T; ++t) {
            std::fill(&x_[t * H], &x_[t * H] + H, 0.0f);
        }

        std::copy(x_.begin(), x_.begin() + (T * H), &x_ckpt_[(L + 1) * CKPT_STRIDE]);
        if (profile) prof.fwdFfnMs += elapsedMs(ffnStart);
    }

    const std::size_t expected = T * V;
    if (logitsOut.size() != expected) throw std::runtime_error("forward: logitsOut size mismatch");

    {
        const auto start = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (parallelRmsnormForwardEnabled() && activeComputeT > 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t t = 0; t < static_cast<ptrdiff_t>(activeComputeT); ++t) {
                if (activeAttentionMask && activeAttentionMask[static_cast<std::size_t>(t)] == 0) {
                    std::fill(&xnorm_final_[static_cast<std::size_t>(t) * H], &xnorm_final_[static_cast<std::size_t>(t) * H] + H, 0.0f);
                    continue;
                }
                rmsnorm(&xnorm_final_[static_cast<std::size_t>(t) * H], &x_[static_cast<std::size_t>(t) * H], rms_final_.data(), H, eps);
            }
        }
        else {
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                if (activeAttentionMask && activeAttentionMask[t] == 0) { std::fill(&xnorm_final_[t * H], &xnorm_final_[t * H] + H, 0.0f); continue; }
                rmsnorm(&xnorm_final_[t * H], &x_[t * H], rms_final_.data(), H, eps);
            }
        }
        for (std::size_t t = activeComputeT; t < T; ++t) {
            std::fill(&xnorm_final_[t * H], &xnorm_final_[t * H] + H, 0.0f);
        }
        if (cfg_.trainFinalNormAffine) {
            std::copy_n(xnorm_final_.data(), T * H, xnorm_final_base_cache_.data());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (ptrdiff_t t_i = 0; t_i < static_cast<ptrdiff_t>(activeComputeT); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                float* row = &xnorm_final_[t * H];
                for (std::size_t h = 0; h < H; ++h) {
                    row[h] = row[h] * finalNormScale_[h] + finalNormBias_[h];
                }
            }
        }
        if (profile) prof.fwdFinalNormMs += elapsedMs(start);
    }

    const bool useQuantLmForward = cfg_.useBitNetQuant && cfg_.bitnetQuantizeLmHead && !W_out_q.empty();
    const auto lmHeadStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
    if (useQuantLmForward) {
        const bool useLossMaskSkip = hasLossMask && lmHeadForwardLossMaskSkipEnabled();
        lmHeadRowsScratch_.clear();
        if (useLossMaskSkip) {
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                if (lossMask[t] != 0) {
                    lmHeadRowsScratch_.push_back(t);
                }
            }
        }
        const bool useRowCompact =
            useLossMaskSkip &&
            lmHeadForwardRowCompactEnabled(activeComputeT, lmHeadRowsScratch_.size());
        if (useRowCompact) {
            const std::size_t compactRows = lmHeadRowsScratch_.size();
            lmHeadDxCompactScratch_.resize(compactRows * H);
            lmHeadDyCompactScratch_.resize(compactRows * V);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int row_i = 0; row_i < static_cast<int>(compactRows); ++row_i) {
                const std::size_t compactRow = static_cast<std::size_t>(row_i);
                const std::size_t srcRow = lmHeadRowsScratch_[compactRow];
                std::memcpy(
                    lmHeadDxCompactScratch_.data() + compactRow * H,
                    xnorm_final_.data() + srcRow * H,
                    H * sizeof(float));
                std::fill_n(
                    lmHeadDyCompactScratch_.data() + compactRow * V,
                    V,
                    0.0f);
            }
            linear_forward_bitnet_i2s_accum(
                lmHeadDyCompactScratch_.data(),
                lmHeadDxCompactScratch_.data(),
                W_out_q,
                compactRows,
                tTile,
                fwdX8Scratch_.data(),
                fwdXScaleScratch_.data(),
                nullptr,
                compactRows);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int row_i = 0; row_i < static_cast<int>(compactRows); ++row_i) {
                const std::size_t compactRow = static_cast<std::size_t>(row_i);
                const std::size_t dstRow = lmHeadRowsScratch_[compactRow];
                std::memcpy(
                    logitsOut.data() + dstRow * V,
                    lmHeadDyCompactScratch_.data() + compactRow * V,
                    V * sizeof(float));
            }
        }
        else {
            if (useLossMaskSkip && lmHeadForwardMaskedRowClearEnabled()) {
                for (std::size_t t = 0; t < activeComputeT; ++t) {
                    if (lossMask[t] == 0) continue;
                    std::fill_n(logitsOut.data() + t * V, V, 0.0f);
                }
            }
            else {
                std::fill(logitsOut.begin(), logitsOut.end(), 0.0f);
            }
            const int* const lmHeadMask = useLossMaskSkip ? lossMask.data() : i2sAttentionMask;
            const std::size_t lmHeadActivePrefixT = useLossMaskSkip
                ? contiguousActivePrefixLength(lossMask)
                : i2sActivePrefixT;
            linear_forward_bitnet_i2s_accum(
                logitsOut.data(),
                xnorm_final_.data(),
                W_out_q,
                activeComputeT,
                tTile,
                fwdX8Scratch_.data(),
                fwdXScaleScratch_.data(),
                lmHeadMask,
                lmHeadActivePrefixT);
        }
    }
    else {
        const std::size_t vTile = static_cast<std::size_t>(cfg_.vTile);
        lmHeadRowsScratch_.clear();
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;
            if (hasLossMask && lossMask[t] == 0) continue;
            lmHeadRowsScratch_.push_back(t);
        }
        for (std::size_t row_i = 0; row_i < lmHeadRowsScratch_.size(); ++row_i) {
            std::fill_n(&logitsOut[lmHeadRowsScratch_[row_i] * V], V, 0.0f);
        }
        const std::size_t rowBlockCount = (lmHeadRowsScratch_.size() + 3) / 4;
        const std::size_t vTileCount = (V + vTile - 1) / vTile;
 #if defined(_OPENMP)
#pragma warning(suppress : 6993)
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int rowBlock_i = 0; rowBlock_i < static_cast<int>(rowBlockCount); ++rowBlock_i) {
            for (int vTile_i = 0; vTile_i < static_cast<int>(vTileCount); ++vTile_i) {
                const std::size_t rowBase = static_cast<std::size_t>(rowBlock_i) * 4;
                const std::size_t rowCount = std::min<std::size_t>(4, lmHeadRowsScratch_.size() - rowBase);
                const std::size_t v0 = static_cast<std::size_t>(vTile_i) * vTile;
                const std::size_t v1 = std::min(V, v0 + vTile);
                const std::size_t len = v1 - v0;

                std::size_t rows[4] = { 0, 0, 0, 0 };
                const float* hidden[4] = { nullptr, nullptr, nullptr, nullptr };
                float* logits[4] = { nullptr, nullptr, nullptr, nullptr };
                for (std::size_t r = 0; r < rowCount; ++r) {
                    rows[r] = lmHeadRowsScratch_[rowBase + r];
                    hidden[r] = &xnorm_final_[rows[r] * H];
                    logits[r] = &logitsOut[rows[r] * V];
                }

                if (rowCount == 4) {
                    float* out0 = logits[0] + v0;
                    float* out1 = logits[1] + v0;
                    float* out2 = logits[2] + v0;
                    float* out3 = logits[3] + v0;

                    for (std::size_t h0 = 0; h0 < H; h0 += hTile) {
                        const std::size_t h1 = std::min(H, h0 + hTile);
                        for (std::size_t hh = h0; hh < h1; ++hh) {
                            const float* w = &W_out_[hh * V + v0];
                            axpy_x4_inplace_avx2_or_scalar(
                                out0, out1, out2, out3,
                                w,
                                len,
                                hidden[0][hh],
                                hidden[1][hh],
                                hidden[2][hh],
                                hidden[3][hh]);
                        }
                    }
                    continue;
                }

                for (std::size_t r = 0; r < rowCount; ++r) {
                    float* out = logits[r] + v0;
                    const float* hiddenRow = hidden[r];
                    for (std::size_t h0 = 0; h0 < H; h0 += hTile) {
                        const std::size_t h1 = std::min(H, h0 + hTile);
                        for (std::size_t hh = h0; hh < h1; ++hh) {
                            const float* w = &W_out_[hh * V + v0];
                            axpy_inplace_avx2_or_scalar(out, w, len, hiddenRow[hh]);
                        }
                    }
                }
            }
        }
    }
    if (cfg_.loraTargetLmHead) {
        if (cfg_.useDoRA) {
            doraStats.merge(!W_out_q.empty()
                ? doraUpdateLmHeadI2S(
                    W_out_q, lmHeadLora_.A, lmHeadLora_.B,
                    lmHeadLora_.M, lmHeadLora_.C, lmHeadLora_.R, H, V)
                : doraUpdateLmHeadDense(
                    W_out_, lmHeadLora_.A, lmHeadLora_.B,
                    lmHeadLora_.M, lmHeadLora_.C, lmHeadLora_.R, H, V));
        }
        lmHeadRowsScratch_.clear();
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;
            if (hasLossMask && lossMask[t] == 0) continue;
            lmHeadRowsScratch_.push_back(t);
        }
        if (cfg_.useDoRA) {
            tmpLmHeadDora_cache_.resize(lmHeadRowsScratch_.size() * V);
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int row_i = 0; row_i < static_cast<int>(lmHeadRowsScratch_.size()); ++row_i) {
            const std::size_t t = lmHeadRowsScratch_[static_cast<std::size_t>(row_i)];
            const float* hiddenRow = &xnorm_final_[t * H];
            float* logitsRow = &logitsOut[t * V];
            float* tmp = &tmpLmHead_cache_[t * R];

            for (std::size_t r = 0; r < R; ++r) {
                const float* Arow = &lmHeadLora_.A[r * H];
                tmp[r] = dot_avx2_or_scalar(Arow, hiddenRow, H);
            }
            for (std::size_t r = 0; r < R; ++r) {
                const float tr = scale * tmp[r];
                const float* Brow = &lmHeadLora_.B[r * V];
                axpy_inplace_avx2_or_scalar(logitsRow, Brow, V, tr);
            }
            if (cfg_.useDoRA) {
                dora_apply_ratio(logitsRow, lmHeadLora_.R, V);
                std::copy_n(
                    logitsRow,
                    V,
                    tmpLmHeadDora_cache_.data() + static_cast<std::size_t>(row_i) * V);
            }
        }
    }
    if (cfg_.trainLmHeadBias) {
        lmHeadRowsScratch_.clear();
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;
            if (hasLossMask && lossMask[t] == 0) continue;
            lmHeadRowsScratch_.push_back(t);
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int row_i = 0; row_i < static_cast<int>(lmHeadRowsScratch_.size()); ++row_i) {
            const std::size_t t = lmHeadRowsScratch_[static_cast<std::size_t>(row_i)];
            axpy_inplace_avx2_or_scalar(&logitsOut[t * V], lmHeadBias_.data(), V, 1.0f);
        }
    }
    if (cfg_.useDoRA && doraStats.n > 0) {
        static bool loggedOnce = false;
        if (!loggedOnce || envBoolOr("QLORA_TEST_DORA_LOG_EVERY_FORWARD", false)) {
            loggedOnce = true;
            const double invN = 1.0 / static_cast<double>(doraStats.n);
            std::cout << "[DoRA] useDoRA=on"
                << " m(mean/min/max)=" << (doraStats.mSum * invN) << "/" << doraStats.mMin << "/" << doraStats.mMax
                << " C(mean/min/max)=" << (doraStats.cSum * invN) << "/" << doraStats.cMin << "/" << doraStats.cMax
                << " ratio(mean/min/max)=" << (doraStats.ratioSum * invN) << "/" << doraStats.ratioMin << "/" << doraStats.ratioMax
                << " deltaVNorm=" << std::sqrt(doraStats.deltaSq)
                << "\n";
        }
    }
    if (doraRefreshNorms) {
        doraNormVersion_ = doraParamVersion_;
    }
    if (doraRefreshLmHeadNorms) {
        doraLmHeadNormVersion_ = doraParamVersion_;
    }
    if (profile) prof.fwdLmHeadMs += elapsedMs(lmHeadStart);
}

// ==============================================================================
// BACKWARD
// ==============================================================================
void CpuKernelBackend::backward(
    std::span<const int>   inputIds,
    std::span<const int>   attentionMask,
    std::span<const float> dLogits,
    std::span<const int>   lossMask)
{
    const std::size_t T = inputIds.size();
    if (attentionMask.size() != T) throw std::runtime_error("backward: attentionMask size mismatch");
    if (T > cfg_.maxSeqLen)        throw std::runtime_error("backward: seqLen exceeds maxSeqLen");
    const bool hasLossMask = !lossMask.empty();
    if (hasLossMask && lossMask.size() != T) {
        throw std::runtime_error("backward: lossMask size mismatch");
    }

    (void)inputIds;

    const std::size_t H = shape_.hiddenSize;
    const std::size_t V = shape_.vocabSize;
    const std::size_t F = shape_.ffnSize;
    const std::size_t D = shape_.headDim();
    const std::size_t ropeDim = shape_.ropeDim;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);
    const std::size_t kvDim = shape_.kvDim(); // from ModelShape, not fragile Wk.size()/H
    const std::size_t nKvHeads = shape_.nKvHeads;
    const std::size_t nHeads = shape_.nHeads;
    const std::size_t tTile = static_cast<std::size_t>(cfg_.tTile);
    const std::size_t hTile = static_cast<std::size_t>(cfg_.hTile);
    const float       scale = cfg_.loraAlpha / float(R);
    const std::size_t nLayers = shape_.nLayers;
    const float       eps = 1e-5f;
    const std::size_t CKPT_STRIDE = cfg_.maxSeqLen * H;
    const std::size_t activePrefixT = contiguousActivePrefixLength(attentionMask);
    const std::size_t activeComputeT =
        (activePrefixComputeEnabled() && activePrefixT != static_cast<std::size_t>(-1))
        ? activePrefixT
        : T;
    const BackendRuntimePolicy runtimePolicy =
        buildBackendRuntimePolicy(activeComputeT, tTile);
    recordBackendRuntimePolicySnapshot(runtimePolicy);
    const ScopedBackendRuntimePolicy runtimePolicyScope(runtimePolicy);
    const int* const activeAttentionMask =
        (activeComputeT < T) ? nullptr : attentionMask.data();
    const std::size_t i2sActivePrefixT = exactMaskedTileSkipEnabled()
        ? activePrefixT
        : static_cast<std::size_t>(-1);
    const int* const  i2sAttentionMask = (exactMaskedTileSkipEnabled() && i2sActivePrefixT == static_cast<std::size_t>(-1))
        ? attentionMask.data()
        : nullptr;
    const bool profile = backendProfileEnabled();
    const std::vector<std::size_t> segmentStarts =
        packedSegmentStarts(attentionMask, cfg_.enablePackedSegments);
    const bool useFfnGemm = ffnGemmEnabled();
    const bool useBitNetFfn = cfg_.useBitNetQuant && ffnBitnetEnabled();
    const SequenceBackwardPolicy sequencePolicy =
        SequenceBackwardPolicy::resolve(useBitNetFfn, activePrefixT, activeComputeT);
    bool lmHeadRowsCollected = false;
    auto ensureLmHeadRows = [&]() {
        if (lmHeadRowsCollected) {
            return;
        }
        lmHeadRowsScratch_.clear();
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;
            if (hasLossMask && lossMask[t] == 0) continue;
            lmHeadRowsScratch_.push_back(t);
        }
        lmHeadRowsCollected = true;
    };
    auto& prof = backendProfileState();

    // ── Backprop through LM head ──────────────────────────────────────────────
    const auto lmHeadStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
    std::fill_n(dXnorm_final_.data(), T * H, 0.0f);
    const float* dLogitsForLmHead = dLogits.data();

    // dLogits rows for non-loss tokens are guaranteed zero by the trainer
    // (dLogitsBuf_ is zero-initialised and cross_entropy only writes loss rows),
    // so the i2s backward kernel is always safe to use regardless of lossMask.
    ensureLmHeadRows();
    const LmHeadBackwardPolicy lmHeadPolicy =
        LmHeadBackwardPolicy::resolve(
            cfg_,
            !W_out_q.empty(),
            activeComputeT,
            tTile,
            hasLossMask,
            lmHeadRowsScratch_.size());
    const bool useCompactDoraLmHead =
        cfg_.useDoRA &&
        cfg_.loraTargetLmHead &&
        lmHeadPolicy.useQuantBackward &&
        lmHeadPolicy.useRowCompact;
    const float* dLogitsLmHeadCompact = nullptr;
    if (cfg_.useDoRA && cfg_.loraTargetLmHead) {
        if (tmpLmHeadDora_cache_.size() < lmHeadRowsScratch_.size() * V) {
            throw std::runtime_error("DoRA LM-head cache is smaller than active loss rows");
        }
        if (useCompactDoraLmHead) {
            const std::size_t compactRows = lmHeadRowsScratch_.size();
            lmHeadDyCompactScratch_.resize(compactRows * V);
            for (int row_i = 0; row_i < static_cast<int>(compactRows); ++row_i) {
                const std::size_t compactRow = static_cast<std::size_t>(row_i);
                const std::size_t srcRow = lmHeadRowsScratch_[compactRow];
                float* dyCompact = lmHeadDyCompactScratch_.data() + compactRow * V;
                std::memcpy(
                    dyCompact,
                    dLogits.data() + srcRow * V,
                    V * sizeof(float));
                dora_backward_magnitude_and_scale(
                    dyCompact,
                    tmpLmHeadDora_cache_.data() + compactRow * V,
                    lmHeadLora_.C,
                    lmHeadLora_.R,
                    lmHeadLoraGrads_.gradM,
                    V);
            }
            dLogitsLmHeadCompact = lmHeadDyCompactScratch_.data();
        }
        else {
            dLogitsDoraScratch_.assign(T * V, 0.0f);
            for (std::size_t row_i = 0; row_i < lmHeadRowsScratch_.size(); ++row_i) {
                const std::size_t t = lmHeadRowsScratch_[row_i];
                std::memcpy(dLogitsDoraScratch_.data() + t * V, dLogits.data() + t * V, V * sizeof(float));
                dora_backward_magnitude_and_scale(
                    dLogitsDoraScratch_.data() + t * V,
                    tmpLmHeadDora_cache_.data() + row_i * V,
                    lmHeadLora_.C,
                    lmHeadLora_.R,
                    lmHeadLoraGrads_.gradM,
                    V);
            }
            dLogitsForLmHead = dLogitsDoraScratch_.data();
        }
    }
    if (lmHeadPolicy.useQuantBackward) {
        if (lmHeadPolicy.useRowCompact) {
            const std::size_t compactRows = lmHeadRowsScratch_.size();
            if (!useCompactDoraLmHead) {
                lmHeadDyCompactScratch_.resize(compactRows * V);
            }
            lmHeadDxCompactScratch_.resize(compactRows * H);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int row_i = 0; row_i < static_cast<int>(compactRows); ++row_i) {
                const std::size_t compactRow = static_cast<std::size_t>(row_i);
                const std::size_t srcRow = lmHeadRowsScratch_[compactRow];
                if (!useCompactDoraLmHead) {
                    std::memcpy(
                        lmHeadDyCompactScratch_.data() + compactRow * V,
                        dLogitsForLmHead + srcRow * V,
                        V * sizeof(float));
                }
                std::fill_n(
                    lmHeadDxCompactScratch_.data() + compactRow * H,
                    H,
                    0.0f);
            }

            linear_backward_bitnet_i2s_packed(
                lmHeadDxCompactScratch_.data(),
                useCompactDoraLmHead ? dLogitsLmHeadCompact : lmHeadDyCompactScratch_.data(),
                W_out_q,
                compactRows,
                std::min(lmHeadPolicy.exactTiles.lmHeadBackwardRows, compactRows),
                bwdDY8Scratch_.data(),
                bwdDYScaleScratch_.data(),
                nullptr,
                compactRows,
                true);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int row_i = 0; row_i < static_cast<int>(compactRows); ++row_i) {
                const std::size_t compactRow = static_cast<std::size_t>(row_i);
                const std::size_t dstRow = lmHeadRowsScratch_[compactRow];
                std::memcpy(
                    dXnorm_final_.data() + dstRow * H,
                    lmHeadDxCompactScratch_.data() + compactRow * H,
                    H * sizeof(float));
            }
        }
        else {
            const int* const lmHeadBwdMask =
                lmHeadPolicy.useLossMaskSkip ? lossMask.data() : i2sAttentionMask;
            const std::size_t lmHeadBwdActivePrefixT =
                lmHeadPolicy.useLossMaskSkip ? static_cast<std::size_t>(-1) : i2sActivePrefixT;
            linear_backward_bitnet_i2s_packed(
                dXnorm_final_.data(),
                dLogitsForLmHead,
                W_out_q,
                activeComputeT,
                lmHeadPolicy.exactTiles.lmHeadBackwardRows,
                bwdDY8Scratch_.data(),
                bwdDYScaleScratch_.data(),
                lmHeadBwdMask,
                lmHeadBwdActivePrefixT);
        }
    }
    else {
        const std::size_t hChunk = lmHeadPolicy.exactTiles.lmHeadBackwardHChunk;
        const bool useDenseLmBackward =
            activeComputeT >= 128 &&
            lmHeadRowsScratch_.size() * 2 >= activeComputeT;
        if (useDenseLmBackward) {
            linear_THF_from_FH_blocked4(
                dXnorm_final_.data(),
                dLogitsForLmHead,
                W_out_.data(),
                activeComputeT,
                V,
                H,
                hChunk,
                tTile);
        }
        else {
            const std::size_t hChunkCount = (H + hChunk - 1) / hChunk;
 #if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
            for (int row_i = 0; row_i < static_cast<int>(lmHeadRowsScratch_.size()); ++row_i) {
                for (int hChunk_i = 0; hChunk_i < static_cast<int>(hChunkCount); ++hChunk_i) {
                    const std::size_t t = lmHeadRowsScratch_[static_cast<std::size_t>(row_i)];
                    const float* dlog_t = &dLogitsForLmHead[t * V];
                    float* dx_t = &dXnorm_final_[t * H];
                    const std::size_t hStart = static_cast<std::size_t>(hChunk_i) * hChunk;
                    const std::size_t hEnd = std::min(H, hStart + hChunk);

                    std::size_t h = hStart;
                    for (; h + 3 < hEnd; h += 4) {
                        float dx0 = 0.0f;
                        float dx1 = 0.0f;
                        float dx2 = 0.0f;
                        float dx3 = 0.0f;
                        dot_x4_avx2_or_scalar(
                            &W_out_[(h + 0) * V],
                            &W_out_[(h + 1) * V],
                            &W_out_[(h + 2) * V],
                            &W_out_[(h + 3) * V],
                            dlog_t,
                            V,
                            dx0,
                            dx1,
                            dx2,
                            dx3);
                        dx_t[h + 0] = dx0;
                        dx_t[h + 1] = dx1;
                        dx_t[h + 2] = dx2;
                        dx_t[h + 3] = dx3;
                    }
                    if (h + 1 < hEnd) {
                        float dx0 = 0.0f;
                        float dx1 = 0.0f;
                        dot2_avx2_or_scalar(
                            &W_out_[h * V],
                            dlog_t,
                            &W_out_[(h + 1) * V],
                            dlog_t,
                            V,
                            dx0,
                            dx1);
                        dx_t[h + 0] = dx0;
                        dx_t[h + 1] = dx1;
                        h += 2;
                    }
                    for (; h < hEnd; ++h) {
                        dx_t[h] = dot_avx2_or_scalar(dlog_t, &W_out_[h * V], V);
                    }
                }
            }
        }
    }
    if (cfg_.loraTargetLmHead) {
        ensureLmHeadRows();

        const bool useRank4LmHeadLoraBwd =
            lmHeadLoraBackwardRank4Enabled(R, V, lmHeadRowsScratch_.size());
        const std::size_t r4Limit = useRank4LmHeadLoraBwd ? (R & ~std::size_t{3}) : 0;
        const bool fuseLmHeadBiasGradWithRank4 =
            cfg_.trainLmHeadBias &&
            !cfg_.useDoRA &&
            r4Limit > 0;
        if (r4Limit > 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int rb_i = 0; rb_i < static_cast<int>(r4Limit / 4); ++rb_i) {
                const std::size_t r = static_cast<std::size_t>(rb_i) * 4;
                const float* B0 = &lmHeadLora_.B[(r + 0) * V];
                const float* B1 = &lmHeadLora_.B[(r + 1) * V];
                const float* B2 = &lmHeadLora_.B[(r + 2) * V];
                const float* B3 = &lmHeadLora_.B[(r + 3) * V];
                float* gB0 = &lmHeadLoraGrads_.gradB[(r + 0) * V];
                float* gB1 = &lmHeadLoraGrads_.gradB[(r + 1) * V];
                float* gB2 = &lmHeadLoraGrads_.gradB[(r + 2) * V];
                float* gB3 = &lmHeadLoraGrads_.gradB[(r + 3) * V];
                float* gA0 = &lmHeadLoraGrads_.gradA[(r + 0) * H];
                float* gA1 = &lmHeadLoraGrads_.gradA[(r + 1) * H];
                float* gA2 = &lmHeadLoraGrads_.gradA[(r + 2) * H];
                float* gA3 = &lmHeadLoraGrads_.gradA[(r + 3) * H];

                for (std::size_t row_i = 0; row_i < lmHeadRowsScratch_.size(); ++row_i) {
                    const std::size_t t = lmHeadRowsScratch_[row_i];
                    const float* dlog_t = useCompactDoraLmHead
                        ? (dLogitsLmHeadCompact + row_i * V)
                        : &dLogitsForLmHead[t * V];
                    const float* hiddenRow = &xnorm_final_[t * H];
                    const float* tmp = &tmpLmHead_cache_[t * R + r];

                    float d0 = 0.0f;
                    float d1 = 0.0f;
                    float d2 = 0.0f;
                    float d3 = 0.0f;
                    dot_x4_avx2_or_scalar(B0, B1, B2, B3, dlog_t, V, d0, d1, d2, d3);
                    d0 *= scale;
                    d1 *= scale;
                    d2 *= scale;
                    d3 *= scale;

                    float* dRank = &dTmpLmHead_cache_[t * R + r];
                    dRank[0] = d0;
                    dRank[1] = d1;
                    dRank[2] = d2;
                    dRank[3] = d3;

                    if (fuseLmHeadBiasGradWithRank4 && r == 0) {
                        axpy_bias_x4_inplace_avx2_or_scalar(
                            gB0, gB1, gB2, gB3,
                            lmHeadBiasGrad_.data(),
                            dlog_t,
                            V,
                            scale * tmp[0],
                            scale * tmp[1],
                            scale * tmp[2],
                            scale * tmp[3]);
                    }
                    else {
                        axpy_x4_inplace_avx2_or_scalar(
                            gB0, gB1, gB2, gB3,
                            dlog_t,
                            V,
                            scale * tmp[0],
                            scale * tmp[1],
                            scale * tmp[2],
                            scale * tmp[3]);
                    }
                    axpy_inplace_avx2_or_scalar(gA0, hiddenRow, H, d0);
                    axpy_inplace_avx2_or_scalar(gA1, hiddenRow, H, d1);
                    axpy_inplace_avx2_or_scalar(gA2, hiddenRow, H, d2);
                    axpy_inplace_avx2_or_scalar(gA3, hiddenRow, H, d3);
                }
            }
        }

        if (r4Limit < R) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int r_i = static_cast<int>(r4Limit); r_i < static_cast<int>(R); ++r_i) {
                const std::size_t r = static_cast<std::size_t>(r_i);
                const float* Brow = &lmHeadLora_.B[r * V];
                float* gBrow = &lmHeadLoraGrads_.gradB[r * V];
                float* gArow = &lmHeadLoraGrads_.gradA[r * H];

                for (std::size_t row_i = 0; row_i < lmHeadRowsScratch_.size(); ++row_i) {
                    const std::size_t t = lmHeadRowsScratch_[row_i];
                    const float* dlog_t = useCompactDoraLmHead
                        ? (dLogitsLmHeadCompact + row_i * V)
                        : &dLogitsForLmHead[t * V];
                    const float* hiddenRow = &xnorm_final_[t * H];
                    const float tr = scale * tmpLmHead_cache_[t * R + r];
                    const float dtr = scale * dot_avx2_or_scalar(Brow, dlog_t, V);

                    dTmpLmHead_cache_[t * R + r] = dtr;
                    axpy_inplace_avx2_or_scalar(gBrow, dlog_t, V, tr);
                    axpy_inplace_avx2_or_scalar(gArow, hiddenRow, H, dtr);
                }
            }
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int row_i = 0; row_i < static_cast<int>(lmHeadRowsScratch_.size()); ++row_i) {
            const std::size_t t = lmHeadRowsScratch_[static_cast<std::size_t>(row_i)];
            float* dx_t = &dXnorm_final_[t * H];
            const float* dRank = &dTmpLmHead_cache_[t * R];
            for (std::size_t r = 0; r < R; ++r) {
                const float dtr = dRank[r];
                const float* Arow = &lmHeadLora_.A[r * H];
                axpy_inplace_avx2_or_scalar(dx_t, Arow, H, dtr);
            }
        }
    }
    if (cfg_.trainLmHeadBias &&
        !(cfg_.loraTargetLmHead && !cfg_.useDoRA && lmHeadLoraBackwardRank4Enabled(R, V, lmHeadRowsScratch_.size()))) {
        ensureLmHeadRows();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int v_i = 0; v_i < static_cast<int>(V); ++v_i) {
            const std::size_t v = static_cast<std::size_t>(v_i);
            float sum = 0.0f;
            for (const std::size_t t : lmHeadRowsScratch_) {
                sum += dLogits[t * V + v];
            }
            lmHeadBiasGrad_[v] += sum;
        }
    }
    if (profile) prof.bwdLmHeadMs += elapsedMs(lmHeadStart);

    const bool hasLayerLoraTargets =
        cfg_.loraTargetQ ||
        cfg_.loraTargetK ||
        cfg_.loraTargetV ||
        cfg_.loraTargetO ||
        cfg_.loraTargetFfnDown ||
        cfg_.loraTargetFfnGateUp ||
        cfg_.trainLayerRmsNorm ||
        cfg_.trainFinalRmsNorm;
    if (!hasLayerLoraTargets) {
        return;
    }

    // ── Backprop through final RMSNorm ────────────────────────────────────────
    const auto finalNormStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
    if (cfg_.trainFinalNormAffine) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int h_i = 0; h_i < static_cast<int>(H); ++h_i) {
            const std::size_t h = static_cast<std::size_t>(h_i);
            float gScale = 0.0f;
            float gBias = 0.0f;
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                const float dy = dXnorm_final_[t * H + h];
                gScale += dy * xnorm_final_base_cache_[t * H + h];
                gBias += dy;
                dXnorm_final_[t * H + h] = dy * finalNormScale_[h];
            }
            finalNormScaleGrad_[h] += gScale;
            finalNormBiasGrad_[h] += gBias;
        }
    }
    if (cfg_.trainFinalRmsNorm) {
        for (std::size_t t = 0; t < activeComputeT; ++t) {
            if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
            accumulate_rmsnorm_weight_grad(
                rmsFinalGrad_.data(),
                &dXnorm_final_[t * H],
                &x_[t * H],
                H,
                eps);
        }
    }
    std::fill_n(dX_.data(), activeComputeT * H, 0.0f);
    for (std::size_t t = 0; t < activeComputeT; ++t) {
        if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
        rmsnorm_backward(&dX_[t * H], &dXnorm_final_[t * H], &x_[t * H], rms_final_.data(), H, eps);
    }
    if (profile) prof.bwdFinalNormMs += elapsedMs(finalNormStart);

    // ── Layer loop (reverse) ─────────────────────────────────────────────────
    for (std::size_t Lrev = 0; Lrev < nLayers; ++Lrev) {
        const std::size_t L = nLayers - 1 - Lrev;
        const std::size_t activeThSize = activeComputeT * H;
        const std::size_t activeTkvSize = activeComputeT * kvDim;
        const std::size_t activeTfSize = activeComputeT * F;
        const std::size_t thSize = T * H;
        const std::size_t tkvSize = T * kvDim;
        const std::size_t tfSize = T * F;
        const bool useI2SWo = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeWo;
        const bool useWoBackwardOverwrite =
            useI2SWo &&
            woBackwardOverwriteEnabled(activeComputeT);
        const bool skipGroupedAttentionKvPreclear =
            groupedAttentionBackwardKvPreclearEnabled(activeComputeT, nHeads, nKvHeads);

        // Clear all per-layer temporaries.
        const auto layerClearStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (!useWoBackwardOverwrite) {
            std::fill_n(dCtx_.data(), activeThSize, 0.0f);
        }
        if (cfg_.loraTargetO) {
            std::fill_n(dAttn_.data(), activeThSize, 0.0f);
        }
        std::fill_n(dQ_.data(), activeThSize, 0.0f);
        if (!skipGroupedAttentionKvPreclear) {
            std::fill_n(dK_.data(), activeTkvSize, 0.0f);
            std::fill_n(dV_.data(), activeTkvSize, 0.0f);
        }
        std::fill_n(dXnorm_.data(), activeThSize, 0.0f);
        std::fill_n(dXin_.data(), activeThSize, 0.0f);
        std::fill_n(dAct_.data(), activeTfSize, 0.0f);
        std::fill_n(dGate_.data(), activeTfSize, 0.0f);
        std::fill_n(dUp_.data(), activeTfSize, 0.0f);
        std::fill_n(dXnormFfn_.data(), activeThSize, 0.0f);
        std::fill_n(tmpdP_.data(), T, 0.0f);
        if (profile) prof.bwdClearMs += elapsedMs(layerClearStart);

        // Restore the pre-layer activation from the checkpoint.
        const auto layerRestoreStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        std::copy(&x_ckpt_[L * CKPT_STRIDE], &x_ckpt_[L * CKPT_STRIDE] + thSize, x_.begin());

        const bool enableTl2Q = tl2QEnabled(cfg_);
        const bool enableTl2K = tl2KEnabled(cfg_);
        const bool recomputeUseTl2GenQ =
            enableTl2Q &&
            tl2GeneratedSupported(cfg_, H, H) &&
            !layers_[L].Wq_tl2g.empty();
        const bool recomputeUseTl2Q =
            enableTl2Q &&
            tl2Supported(cfg_, H, H) &&
            !layers_[L].Wq_tl2.empty() &&
            !recomputeUseTl2GenQ;
        const bool recomputeUseTl2GenK =
            enableTl2K &&
            tl2GeneratedSupported(cfg_, H, kvDim) &&
            !layers_[L].Wk_tl2g.empty();
        const bool recomputeUseTl2K =
            enableTl2K &&
            tl2Supported(cfg_, H, kvDim) &&
            !layers_[L].Wk_tl2.empty() &&
            !recomputeUseTl2GenK;
        const bool recomputeUseI2SQ = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV && !recomputeUseTl2GenQ && !recomputeUseTl2Q;
        const bool recomputeUseI2SK = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV && !recomputeUseTl2GenK && !recomputeUseTl2K;
        const bool useI2SQ = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV;
        const bool useI2SK = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV;
        const bool useI2SV = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV;
        const bool useFusedI2SQKV = useI2SQ && useI2SK && useI2SV && cfg_.bitnetFuseQKV;
        const bool useSplitFusedQkvBackward = useFusedI2SQKV && splitFusedQkvBackwardEnabled();
        const bool useFusedI2SQKVBackward = useFusedI2SQKV && !useSplitFusedQkvBackward;
        const bool useFusedI2SKV = useI2SK && useI2SV && !useFusedI2SQKVBackward;
        const bool useDirectAttentionBackwardCache =
            directAttentionBackwardCacheEnabled(activeComputeT);
        const bool useDirectFfnBackwardCache =
            directFfnBackwardCacheEnabled(activeComputeT);
        const bool useRmsnormBwdAccum = rmsnormBackwardAccumEnabled();
        const FfnBackwardPolicy ffnPolicy =
            FfnBackwardPolicy::resolve(
                useBitNetFfn,
                sequencePolicy.bitnetFfnOverwritesSplitGradBuffers,
                useRmsnormBwdAccum,
                activeComputeT,
                tTile);
        const AttentionBackwardPolicy attentionPolicy =
            AttentionBackwardPolicy::resolve(activeComputeT, tTile);
        (void)useI2SWo;

        const std::size_t layerTHOffset = L * cfg_.maxSeqLen * H;
        const std::size_t layerTFOffset = L * cfg_.maxSeqLen * F;
        const std::size_t layerTKVOffset = L * cfg_.maxSeqLen * kvDim;
        const std::size_t layerProbsOffset = L * nHeads * cfg_.maxSeqLen * cfg_.maxSeqLen;

        const float* const layerXnormCache = xnorm_cache_.data() + layerTHOffset;
        const float* const layerQCache = q_cache_.data() + layerTHOffset;
        const float* const layerKCache = k_cache_.data() + layerTKVOffset;
        const float* const layerVCache = v_cache_.data() + layerTKVOffset;
        const float* const layerCtxCache = cfg_.loraTargetO ? (ctx_cache_.data() + layerTHOffset) : nullptr;
        const float* const layerProbsCache =
            probs_cache_.empty() ? nullptr : (probs_cache_.data() + layerProbsOffset);
        const float* const layerXmidCache = x_mid_cache_.data() + layerTHOffset;
        const float* const layerGateCache = gate_cache_.data() + layerTFOffset;
        const float* const layerUpCache = up_cache_.data() + layerTFOffset;
        const bool ffnLoraEnabled = ffnLoraLayerEnabled(L);
        const bool ffnDownLoraEnabled = cfg_.loraTargetFfnDown && ffnLoraEnabled;
        const bool ffnGateUpLoraEnabled = cfg_.loraTargetFfnGateUp && ffnLoraEnabled;
        const float* const layerFfnActCache = ffnDownLoraEnabled ? (ffn_act_cache_.data() + layerTFOffset) : nullptr;
        const float* const layerFfnXnormCache = ffnGateUpLoraEnabled ? (xnorm_ffn_cache_.data() + layerTHOffset) : nullptr;
        const float* const attnXnorm = useDirectAttentionBackwardCache ? layerXnormCache : xnorm_.data();
        const float* const attnQ = useDirectAttentionBackwardCache ? layerQCache : q_.data();
        const float* const attnK = useDirectAttentionBackwardCache ? layerKCache : k_.data();
        const float* const attnV = useDirectAttentionBackwardCache ? layerVCache : v_.data();
        const float* const attnProbs =
            (useDirectAttentionBackwardCache && layerProbsCache) ? layerProbsCache : probs_.data();
        const float* const ffnXmid = useDirectFfnBackwardCache ? layerXmidCache : x_mid_.data();
        const float* const ffnGate = useDirectFfnBackwardCache ? layerGateCache : gate_.data();
        const float* const ffnUp = useDirectFfnBackwardCache ? layerUpCache : up_.data();

        auto accumulateFfnDownLoraBackward = [&](std::size_t tBase, std::size_t rows) {
            if (!ffnDownLoraEnabled) return;
            if (!layerFfnActCache) {
                throw std::runtime_error("FFN-down LoRA backward missing activation cache");
            }
            auto& gL = loraGrads_[L];
            for (std::size_t tr = 0; tr < rows; ++tr) {
                const std::size_t t = tBase + tr;
                if (t >= activeComputeT) break;
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;

                const float* dy = &dX_[t * H];
                const float* act = layerFfnActCache + t * F;
                const float* tmp = &tmpFfnDown_cache_[((L * cfg_.maxSeqLen) + t) * R];
                const float* B = loras_[L].Bfd.data();
                const float* A = loras_[L].Afd.data();

                for (std::size_t r = 0; r < R; ++r) {
                    const float trank = scale * tmp[r];
                    float* gBrow = &gL.gradBfd[r * H];
                    const float* Brow = &B[r * H];
                    axpy_inplace_avx2_or_scalar(gBrow, dy, H, trank);
                    dTmpRank_[r] = scale * dot_avx2_or_scalar(Brow, dy, H);
                }

                for (std::size_t r = 0; r < R; ++r) {
                    float* gArow = &gL.gradAfd[r * F];
                    axpy_inplace_avx2_or_scalar(gArow, act, F, dTmpRank_[r]);
                }

                float* dAct_t = &dAct_[t * F];
                for (std::size_t r = 0; r < R; ++r) {
                    const float* Arow = &A[r * F];
                    axpy_inplace_avx2_or_scalar(dAct_t, Arow, F, dTmpRank_[r]);
                }
            }
        };

        auto accumulateFfnGateUpLoraBackward = [&](std::size_t tBase, std::size_t rows) {
            if (!ffnGateUpLoraEnabled) return;
            if (!layerFfnXnormCache) {
                throw std::runtime_error("FFN gate/up LoRA backward missing xnorm cache");
            }
            auto& gL = loraGrads_[L];
            for (std::size_t tr = 0; tr < rows; ++tr) {
                const std::size_t t = tBase + tr;
                if (t >= activeComputeT) break;
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;

                const float* xnorm_t = layerFfnXnormCache + t * H;
                float* dXnorm_t = &dXnormFfn_[t * H];

                const float* dGate_t = &dGate_[t * F];
                const float* tmpFg = &tmpFfnGate_cache_[((L * cfg_.maxSeqLen) + t) * R];
                for (std::size_t r = 0; r < R; ++r) {
                    const float trank = scale * tmpFg[r];
                    float* gBrow = &gL.gradBfg[r * F];
                    const float* Brow = &loras_[L].Bfg[r * F];
                    axpy_inplace_avx2_or_scalar(gBrow, dGate_t, F, trank);
                    dTmpRank_[r] = scale * dot_avx2_or_scalar(Brow, dGate_t, F);
                }
                for (std::size_t r = 0; r < R; ++r) {
                    float* gArow = &gL.gradAfg[r * H];
                    const float* Arow = &loras_[L].Afg[r * H];
                    axpy_inplace_avx2_or_scalar(gArow, xnorm_t, H, dTmpRank_[r]);
                    axpy_inplace_avx2_or_scalar(dXnorm_t, Arow, H, dTmpRank_[r]);
                }

                const float* dUp_t = &dUp_[t * F];
                const float* tmpFu = &tmpFfnUp_cache_[((L * cfg_.maxSeqLen) + t) * R];
                for (std::size_t r = 0; r < R; ++r) {
                    const float trank = scale * tmpFu[r];
                    float* gBrow = &gL.gradBfu[r * F];
                    const float* Brow = &loras_[L].Bfu[r * F];
                    axpy_inplace_avx2_or_scalar(gBrow, dUp_t, F, trank);
                    dTmpRank_[r] = scale * dot_avx2_or_scalar(Brow, dUp_t, F);
                }
                for (std::size_t r = 0; r < R; ++r) {
                    float* gArow = &gL.gradAfu[r * H];
                    const float* Arow = &loras_[L].Afu[r * H];
                    axpy_inplace_avx2_or_scalar(gArow, xnorm_t, H, dTmpRank_[r]);
                    axpy_inplace_avx2_or_scalar(dXnorm_t, Arow, H, dTmpRank_[r]);
                }
            }
        };

        auto accumulateFfnRmsNormGrad = [&](std::size_t tBase, std::size_t rows) {
            if (!cfg_.trainLayerRmsNorm) return;
            auto& gL = loraGrads_[L];
            for (std::size_t tr = 0; tr < rows; ++tr) {
                const std::size_t t = tBase + tr;
                if (t >= activeComputeT) break;
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                accumulate_rmsnorm_weight_grad(
                    gL.gradRmsFfn.data(),
                    &dXnormFfn_[t * H],
                    ffnXmid + t * H,
                    H,
                    eps);
            }
        };

        if (!useDirectAttentionBackwardCache) {
            std::copy_n(layerXnormCache, thSize, xnorm_.data());
            std::copy_n(layerQCache, thSize, q_.data());
            std::copy_n(layerKCache, tkvSize, k_.data());
            std::copy_n(layerVCache, tkvSize, v_.data());
            if (layerProbsCache) {
                std::copy_n(layerProbsCache, nHeads * T * T, probs_.data());
            }
        }
        if (!useDirectFfnBackwardCache) {
            std::copy_n(layerXmidCache, thSize, x_mid_.data());
        }
        if (!ffnPolicy.useStreamedPrefix && !useDirectFfnBackwardCache) {
            std::copy_n(layerGateCache, tfSize, gate_.data());
            std::copy_n(layerUpCache, tfSize, up_.data());
        }
        if (profile) prof.bwdRestoreMs += elapsedMs(layerRestoreStart);

        // Backprop through FFN second residual:
        // x_out = x_mid + down(silu(gate(xnorm_ffn)) * up(xnorm_ffn)).
        const auto ffnStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (!ffnPolicy.fuseResidualAdd) {
            for (std::size_t i = 0; i < activeThSize; ++i) {
                dXmid_[i] = dX_[i];
            }
        }
        if (ffnDownLoraEnabled && !useBitNetFfn && !useFfnGemm) {
            throw std::runtime_error("FFN-down LoRA requires BitNet/TL2 FFN or QLORA_TEST_FFN_GEMM=1");
        }
        if (ffnGateUpLoraEnabled && !useBitNetFfn && !useFfnGemm) {
            throw std::runtime_error("FFN gate/up LoRA requires BitNet/TL2 FFN or QLORA_TEST_FFN_GEMM=1");
        }
        if (cfg_.trainLayerRmsNorm && !useBitNetFfn && !useFfnGemm) {
            throw std::runtime_error("Layer RMSNorm tuning requires BitNet/TL2 FFN or QLORA_TEST_FFN_GEMM=1");
        }
        if (cfg_.useDoRA && ffnDownLoraEnabled) {
            auto& P = loras_[L];
            auto& G = loraGrads_[L];
            const float* layerOut = x_ckpt_.data() + (L + 1) * CKPT_STRIDE;
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                for (std::size_t h = 0; h < H; ++h) {
                    ffn_out_[t * H + h] = layerOut[t * H + h] - ffnXmid[t * H + h];
                }
                dora_backward_magnitude_and_scale(&dX_[t * H], &ffn_out_[t * H], P.Cfd, P.Rfd, G.gradMfd, H);
            }
        }

        auto applyFfnGateUpDoraBackward = [&](std::size_t tBase, std::size_t rows) {
            if (!cfg_.useDoRA || !ffnGateUpLoraEnabled) return;
            auto& P = loras_[L];
            auto& G = loraGrads_[L];
            for (std::size_t tr = 0; tr < rows; ++tr) {
                const std::size_t t = tBase + tr;
                if (t >= activeComputeT) break;
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                dora_backward_magnitude_and_scale(&dGate_[t * F], ffnGate + t * F, P.Cfg, P.Rfg, G.gradMfg, F);
                dora_backward_magnitude_and_scale(&dUp_[t * F], ffnUp + t * F, P.Cfu, P.Rfu, G.gradMfu, F);
            }
        };

        if (useBitNetFfn) {
            if (ffnPolicy.useStreamedPrefix) {
                const std::size_t requestedChunkT =
                    bitnetFfnStreamedPrefixBackwardChunkT(32);
                const std::size_t chunkT =
                    std::max<std::size_t>(tTile, requestedChunkT);
                const float* gateCacheLayer = layerGateCache;
                const float* upCacheLayer = layerUpCache;

                for (std::size_t tBase = 0; tBase < activeComputeT; tBase += chunkT) {
                    const std::size_t rows = std::min(chunkT, activeComputeT - tBase);
                    float* dActChunk = dAct_.data() + tBase * F;
                    float* dGateChunk = dGate_.data() + tBase * F;
                    float* dUpChunk = dUp_.data() + tBase * F;
                    float* dXChunk = dX_.data() + tBase * H;
                    float* dXnormFfnChunk = dXnormFfn_.data() + tBase * H;
                    float* dXinChunk = dXin_.data() + tBase * H;
                    float* dXmidChunk = dXmid_.data() + tBase * H;
                    const float* gateChunk = gateCacheLayer + tBase * F;
                    const float* upChunk = upCacheLayer + tBase * F;
                    const float* xMidChunk = ffnXmid + tBase * H;

                    const auto wdownStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
                    linear_backward_bitnet_i2s_packed(
                        dActChunk,
                        dXChunk,
                        layers_[L].Wdown_q,
                        rows,
                        tTile,
                        bwdDY8Scratch_.data(),
                        bwdDYScaleScratch_.data(),
                        nullptr,
                        rows,
                        true);
                    if (profile) prof.bwdFfnWdownMs += elapsedMs(wdownStart);
                    accumulateFfnDownLoraBackward(tBase, rows);

                    const auto siluStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
                    if (fastSiluBackwardEnabled()) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                        for (int t_i = 0; t_i < static_cast<int>(rows); ++t_i) {
                            const std::size_t t = static_cast<std::size_t>(t_i);
                            silu_backward_avx2_or_scalar(
                                dGateChunk + t * F,
                                dUpChunk + t * F,
                                dActChunk + t * F,
                                gateChunk + t * F,
                                upChunk + t * F,
                                F);
                        }
                    }
                    else if (exactSiluBackwardChunkedEnabled()) {
                        silu_backward_exact_chunked(
                            dGateChunk,
                            dUpChunk,
                            dActChunk,
                            gateChunk,
                            upChunk,
                            rows,
                            F,
                            nullptr);
                    }
                    else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                        for (int t_i = 0; t_i < static_cast<int>(rows); ++t_i) {
                            const std::size_t t = static_cast<std::size_t>(t_i);
                            silu_backward_exact_scalar(
                                dGateChunk + t * F,
                                dUpChunk + t * F,
                                dActChunk + t * F,
                                gateChunk + t * F,
                                upChunk + t * F,
                                F);
                        }
                    }
                    if (profile) prof.bwdFfnSiluMs += elapsedMs(siluStart);

                    applyFfnGateUpDoraBackward(tBase, rows);
                    const auto dualStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
                    linear_fused_dual_backward_bitnet_i2s_packed(
                        dXnormFfnChunk,
                        dUpChunk,
                        dGateChunk,
                        layers_[L].Wup_q,
                        layers_[L].Wgate_q,
                        rows,
                        tTile,
                        bwdDY8Scratch_.data(),
                        bwdDYScaleScratch_.data(),
                        bwdDK8Scratch_.data(),
                        bwdDKScaleScratch_.data(),
                        nullptr,
                        rows,
                        false);
                    if (profile) prof.bwdFfnDualMs += elapsedMs(dualStart);
                    accumulateFfnGateUpLoraBackward(tBase, rows);
                    accumulateFfnRmsNormGrad(tBase, rows);

                    const auto normStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                    for (int t_i = 0; t_i < static_cast<int>(rows); ++t_i) {
                        const std::size_t t = static_cast<std::size_t>(t_i);
                        if (useRmsnormBwdAccum) {
                            rmsnorm_backward_accum(
                                dXmidChunk + t * H,
                                dXnormFfnChunk + t * H,
                                xMidChunk + t * H,
                                layers_[L].rms_ffn.data(),
                                H,
                                eps);
                        }
                        else {
                            rmsnorm_backward(
                                dXinChunk + t * H,
                                dXnormFfnChunk + t * H,
                                xMidChunk + t * H,
                                layers_[L].rms_ffn.data(),
                                H,
                                eps);

                            float* dXmid_t = dXmidChunk + t * H;
                            const float* dResidual_t = dXChunk + t * H;
                            const float* dXmidFromNorm_t = dXinChunk + t * H;
                            for (std::size_t h = 0; h < H; ++h) {
                                dXmid_t[h] = ffnPolicy.fuseResidualAdd
                                    ? (dResidual_t[h] + dXmidFromNorm_t[h])
                                    : (dXmid_t[h] + dXmidFromNorm_t[h]);
                            }
                        }
                    }
                    if (profile) prof.bwdFfnNormMs += elapsedMs(normStart);
                }
            }
            else {
                if (ffnPolicy.useChunked && ffnPolicy.chunkRows < activeComputeT) {
                    const std::size_t wdownBwdTileRows = ffnPolicy.exactTiles.wdownBackwardRows;
                    for (std::size_t tBase = 0; tBase < activeComputeT; tBase += ffnPolicy.chunkRows) {
                        const std::size_t rows = std::min(ffnPolicy.chunkRows, activeComputeT - tBase);
                        float* dActChunk = dAct_.data() + tBase * F;
                        float* dGateChunk = dGate_.data() + tBase * F;
                        float* dUpChunk = dUp_.data() + tBase * F;
                        float* dXChunk = dX_.data() + tBase * H;
                        float* dXnormFfnChunk = dXnormFfn_.data() + tBase * H;
                        float* dXinChunk = dXin_.data() + tBase * H;
                        float* dXmidChunk = dXmid_.data() + tBase * H;
                        const float* gateChunk = ffnGate + tBase * F;
                        const float* upChunk = ffnUp + tBase * F;
                        const float* xMidChunk = ffnXmid + tBase * H;
                        const int* const chunkActiveMask =
                            activeAttentionMask ? (activeAttentionMask + tBase) : nullptr;
                        const int* const chunkI2sMask =
                            i2sAttentionMask ? (i2sAttentionMask + tBase) : nullptr;
                        const std::size_t chunkI2sActivePrefixT =
                            (i2sActivePrefixT == static_cast<std::size_t>(-1))
                            ? static_cast<std::size_t>(-1)
                            : ((i2sActivePrefixT > tBase)
                                ? std::min(rows, i2sActivePrefixT - tBase)
                                : static_cast<std::size_t>(0));

                        const auto wdownStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
                        linear_backward_bitnet_i2s_packed(
                            dActChunk,
                            dXChunk,
                            layers_[L].Wdown_q,
                            rows,
                            wdownBwdTileRows,
                            bwdDY8Scratch_.data(),
                            bwdDYScaleScratch_.data(),
                            chunkI2sMask,
                            chunkI2sActivePrefixT,
                            true);
                        if (profile) prof.bwdFfnWdownMs += elapsedMs(wdownStart);
                        accumulateFfnDownLoraBackward(tBase, rows);

                        const auto siluStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
                        if (fastSiluBackwardEnabled()) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                            for (int t_i = 0; t_i < static_cast<int>(rows); ++t_i) {
                                const std::size_t t = static_cast<std::size_t>(t_i);
                                float* dAct_t = dActChunk + t * F;
                                float* dGate_t = dGateChunk + t * F;
                                float* dUp_t = dUpChunk + t * F;
                                const float* gate_t = gateChunk + t * F;
                                const float* up_t = upChunk + t * F;

                                if (chunkActiveMask && chunkActiveMask[t] == 0) {
                                    std::fill(dAct_t, dAct_t + F, 0.0f);
                                    std::fill(dGate_t, dGate_t + F, 0.0f);
                                    std::fill(dUp_t, dUp_t + F, 0.0f);
                                    continue;
                                }
                                silu_backward_avx2_or_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                            }
                        }
                        else if (exactSiluBackwardChunkedEnabled()) {
                            silu_backward_exact_chunked(
                                dGateChunk,
                                dUpChunk,
                                dActChunk,
                                gateChunk,
                                upChunk,
                                rows,
                                F,
                                chunkActiveMask);
                        }
                        else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                            for (int t_i = 0; t_i < static_cast<int>(rows); ++t_i) {
                                const std::size_t t = static_cast<std::size_t>(t_i);
                                float* dAct_t = dActChunk + t * F;
                                float* dGate_t = dGateChunk + t * F;
                                float* dUp_t = dUpChunk + t * F;
                                const float* gate_t = gateChunk + t * F;
                                const float* up_t = upChunk + t * F;
                                if (chunkActiveMask && chunkActiveMask[t] == 0) {
                                    std::fill(dAct_t, dAct_t + F, 0.0f);
                                    std::fill(dGate_t, dGate_t + F, 0.0f);
                                    std::fill(dUp_t, dUp_t + F, 0.0f);
                                    continue;
                                }
                                silu_backward_exact_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                            }
                        }
                        if (profile) prof.bwdFfnSiluMs += elapsedMs(siluStart);

                        applyFfnGateUpDoraBackward(tBase, rows);
                        const auto dualStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
                        linear_fused_dual_backward_bitnet_i2s_packed(
                            dXnormFfnChunk,
                            dUpChunk, dGateChunk,
                            layers_[L].Wup_q, layers_[L].Wgate_q,
                            rows, tTile,
                            bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                            bwdDK8Scratch_.data(), bwdDKScaleScratch_.data(),
                            chunkI2sMask, chunkI2sActivePrefixT,
                            false);
                        if (profile) prof.bwdFfnDualMs += elapsedMs(dualStart);
                        accumulateFfnGateUpLoraBackward(tBase, rows);
                        accumulateFfnRmsNormGrad(tBase, rows);

                        const auto normStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                        for (int t_i = 0; t_i < static_cast<int>(rows); ++t_i) {
                            const std::size_t t = static_cast<std::size_t>(t_i);
                            if (chunkActiveMask && chunkActiveMask[t] == 0) continue;
                            if (useRmsnormBwdAccum) {
                                rmsnorm_backward_accum(
                                    dXmidChunk + t * H,
                                    dXnormFfnChunk + t * H,
                                    xMidChunk + t * H,
                                    layers_[L].rms_ffn.data(),
                                    H,
                                    eps);
                            }
                            else {
                                rmsnorm_backward(
                                    dXinChunk + t * H,
                                    dXnormFfnChunk + t * H,
                                    xMidChunk + t * H,
                                    layers_[L].rms_ffn.data(),
                                    H,
                                    eps);

                                float* dXmid_t = dXmidChunk + t * H;
                                const float* dResidual_t = dXChunk + t * H;
                                const float* dXmidFromNorm_t = dXinChunk + t * H;
                                for (std::size_t h = 0; h < H; ++h) {
                                    dXmid_t[h] = ffnPolicy.fuseResidualAdd
                                        ? (dResidual_t[h] + dXmidFromNorm_t[h])
                                        : (dXmid_t[h] + dXmidFromNorm_t[h]);
                                }
                            }
                        }
                        if (profile) prof.bwdFfnNormMs += elapsedMs(normStart);
                    }
                }
                else {
                    const std::size_t wdownBwdTileRows = ffnPolicy.exactTiles.wdownBackwardRows;
                    const auto wdownStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
                    linear_backward_bitnet_i2s_packed(
                        dAct_.data(),
                        dX_.data(),
                        layers_[L].Wdown_q,
                        activeComputeT,
                        wdownBwdTileRows,
                        bwdDY8Scratch_.data(),
                        bwdDYScaleScratch_.data(),
                        i2sAttentionMask,
                        i2sActivePrefixT,
                        true);
                    if (profile) prof.bwdFfnWdownMs += elapsedMs(wdownStart);
                    accumulateFfnDownLoraBackward(0, activeComputeT);

                    const auto siluStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
                    if (fastSiluBackwardEnabled()) {
                        if (sequencePolicy.usePrefixMaskFastpath) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                            for (int t_i = 0; t_i < static_cast<int>(activePrefixT); ++t_i) {
                                const std::size_t t = static_cast<std::size_t>(t_i);
                                float* dAct_t = &dAct_[t * F];
                                float* dGate_t = &dGate_[t * F];
                                float* dUp_t = &dUp_[t * F];
                                const float* gate_t = ffnGate + t * F;
                                const float* up_t = ffnUp + t * F;
                                silu_backward_avx2_or_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                            }
                            if (activePrefixT < activeComputeT) {
                                const std::size_t tailOffset = activePrefixT * F;
                                const std::size_t tailCount = (activeComputeT - activePrefixT) * F;
                                std::fill_n(dAct_.data() + tailOffset, tailCount, 0.0f);
                                std::fill_n(dGate_.data() + tailOffset, tailCount, 0.0f);
                                std::fill_n(dUp_.data() + tailOffset, tailCount, 0.0f);
                            }
                        }
                        else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                            for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                                const std::size_t t = static_cast<std::size_t>(t_i);
                                float* dAct_t = &dAct_[t * F];
                                float* dGate_t = &dGate_[t * F];
                                float* dUp_t = &dUp_[t * F];
                                const float* gate_t = ffnGate + t * F;
                                const float* up_t = ffnUp + t * F;

                                if (attentionMask[t] == 0) {
                                    std::fill(dAct_t, dAct_t + F, 0.0f);
                                    std::fill(dGate_t, dGate_t + F, 0.0f);
                                    std::fill(dUp_t, dUp_t + F, 0.0f);
                                    continue;
                                }
                                silu_backward_avx2_or_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                            }
                        }
                    }
                    else {
                        if (exactSiluBackwardChunkedEnabled()) {
                            silu_backward_exact_chunked(
                                dGate_.data(),
                                dUp_.data(),
                                dAct_.data(),
                                ffnGate,
                                ffnUp,
                                T,
                                F,
                                attentionMask.data());
                        }
                        else {
                            if (sequencePolicy.usePrefixMaskFastpath) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                                for (int t_i = 0; t_i < static_cast<int>(activePrefixT); ++t_i) {
                                    const std::size_t t = static_cast<std::size_t>(t_i);
                                    float* dAct_t = &dAct_[t * F];
                                    float* dGate_t = &dGate_[t * F];
                                    float* dUp_t = &dUp_[t * F];
                                    const float* gate_t = ffnGate + t * F;
                                    const float* up_t = ffnUp + t * F;
                                    silu_backward_exact_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                                }
                                if (activePrefixT < activeComputeT) {
                                    const std::size_t tailOffset = activePrefixT * F;
                                    const std::size_t tailCount = (activeComputeT - activePrefixT) * F;
                                    std::fill_n(dAct_.data() + tailOffset, tailCount, 0.0f);
                                    std::fill_n(dGate_.data() + tailOffset, tailCount, 0.0f);
                                    std::fill_n(dUp_.data() + tailOffset, tailCount, 0.0f);
                                }
                            }
                            else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                                for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                                    const std::size_t t = static_cast<std::size_t>(t_i);
                                    float* dAct_t = &dAct_[t * F];
                                    float* dGate_t = &dGate_[t * F];
                                    float* dUp_t = &dUp_[t * F];
                                    const float* gate_t = ffnGate + t * F;
                                    const float* up_t = ffnUp + t * F;
                                    if (attentionMask[t] == 0) {
                                        std::fill(dAct_t, dAct_t + F, 0.0f);
                                        std::fill(dGate_t, dGate_t + F, 0.0f);
                                        std::fill(dUp_t, dUp_t + F, 0.0f);
                                        continue;
                                    }
                                    silu_backward_exact_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                                }
                            }
                        }
                    }
                    if (profile) prof.bwdFfnSiluMs += elapsedMs(siluStart);

                    applyFfnGateUpDoraBackward(0, activeComputeT);
                    const auto dualStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
                    linear_fused_dual_backward_bitnet_i2s_packed(
                        dXnormFfn_.data(),
                        dUp_.data(), dGate_.data(),
                        layers_[L].Wup_q, layers_[L].Wgate_q,
                        activeComputeT, tTile,
                        bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                        bwdDK8Scratch_.data(), bwdDKScaleScratch_.data(),
                        i2sAttentionMask, i2sActivePrefixT,
                        false);
                    if (profile) prof.bwdFfnDualMs += elapsedMs(dualStart);
                    accumulateFfnGateUpLoraBackward(0, activeComputeT);
                    accumulateFfnRmsNormGrad(0, activeComputeT);

                    const auto normStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                    for (int t_i = 0; t_i < static_cast<int>(activeComputeT); ++t_i) {
                        const std::size_t t = static_cast<std::size_t>(t_i);
                        if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                        if (useRmsnormBwdAccum) {
                            rmsnorm_backward_accum(
                                &dXmid_[t * H],
                                &dXnormFfn_[t * H],
                                ffnXmid + t * H,
                                layers_[L].rms_ffn.data(),
                                H, eps
                            );
                        }
                        else {
                            rmsnorm_backward(
                                &dXin_[t * H],
                                &dXnormFfn_[t * H],
                                ffnXmid + t * H,
                                layers_[L].rms_ffn.data(),
                                H, eps
                            );

                            float* dXmid_t = &dXmid_[t * H];
                            const float* dResidual_t = &dX_[t * H];
                            const float* dXmidFromNorm_t = &dXin_[t * H];
                            for (std::size_t h = 0; h < H; ++h) {
                                dXmid_t[h] = ffnPolicy.fuseResidualAdd
                                    ? (dResidual_t[h] + dXmidFromNorm_t[h])
                                    : (dXmid_t[h] + dXmidFromNorm_t[h]);
                            }
                        }
                    }
                    if (profile) prof.bwdFfnNormMs += elapsedMs(normStart);
                }
            }
        }
        else if (useFfnGemm) {
            linear_THF_from_FH_blocked4(
                dAct_.data(),
                dX_.data(),
                layers_[L].Wdown.data(),
                T, H, F, cfg_.hTile, tTile);
            accumulateFfnDownLoraBackward(0, activeComputeT);

            if (fp32FfnFastSiluBackwardEnabled()) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                    const std::size_t t = static_cast<std::size_t>(t_i);
                    float* dAct_t = &dAct_[t * F];
                    float* dGate_t = &dGate_[t * F];
                    float* dUp_t = &dUp_[t * F];
                    const float* gate_t = ffnGate + t * F;
                    const float* up_t = ffnUp + t * F;

                    if (attentionMask[t] == 0) {
                        std::fill(dAct_t, dAct_t + F, 0.0f);
                        std::fill(dGate_t, dGate_t + F, 0.0f);
                        std::fill(dUp_t, dUp_t + F, 0.0f);
                        continue;
                    }
                    silu_backward_avx2_or_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                }
            }
            else {
                if (exactSiluBackwardChunkedEnabled()) {
                    silu_backward_exact_chunked(
                        dGate_.data(),
                        dUp_.data(),
                        dAct_.data(),
                        ffnGate,
                        ffnUp,
                        T,
                        F,
                        attentionMask.data());
                }
                else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
                    for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                        const std::size_t t = static_cast<std::size_t>(t_i);
                        float* dAct_t = &dAct_[t * F];
                        float* dGate_t = &dGate_[t * F];
                        float* dUp_t = &dUp_[t * F];
                        const float* gate_t = ffnGate + t * F;
                        const float* up_t = ffnUp + t * F;
                        if (attentionMask[t] == 0) {
                            std::fill(dAct_t, dAct_t + F, 0.0f);
                            std::fill(dGate_t, dGate_t + F, 0.0f);
                            std::fill(dUp_t, dUp_t + F, 0.0f);
                            continue;
                        }
                        silu_backward_exact_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                    }
                }
            }

            applyFfnGateUpDoraBackward(0, activeComputeT);
            linear_TFH_fused_dual_from_HF_blocked4(
                dXnormFfn_.data(),
                dUp_.data(),
                dGate_.data(),
                layers_[L].Wup.data(),
                layers_[L].Wgate.data(),
                T, F, H, cfg_.hTile, tTile);
            accumulateFfnGateUpLoraBackward(0, activeComputeT);
            accumulateFfnRmsNormGrad(0, activeComputeT);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                if (attentionMask[t] == 0) continue;
                if (useRmsnormBwdAccum) {
                    rmsnorm_backward_accum(
                        &dXmid_[t * H],
                        &dXnormFfn_[t * H],
                        ffnXmid + t * H,
                        layers_[L].rms_ffn.data(),
                        H, eps
                    );
                }
                else {
                    rmsnorm_backward(
                        &dXin_[t * H],
                        &dXnormFfn_[t * H],
                        ffnXmid + t * H,
                        layers_[L].rms_ffn.data(),
                        H, eps
                    );

                    float* dXmid_t = &dXmid_[t * H];
                    const float* dXmidFromNorm_t = &dXin_[t * H];
                    for (std::size_t h = 0; h < H; ++h) dXmid_t[h] += dXmidFromNorm_t[h];
                }
            }
        }
        else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int t0_i = 0; t0_i < static_cast<int>(T); t0_i += 4) {
                const std::size_t t0 = static_cast<std::size_t>(t0_i);
                const std::size_t tt = std::min<std::size_t>(4, T - t0);
                if (tt < 4) {
                    for (std::size_t r = 0; r < tt; ++r) {
                        const std::size_t t = t0 + r;
                        if (attentionMask[t] == 0) continue;

                        const float* dOut = &dX_[t * H];
                        float* dAct_t = &dAct_[t * F];
                        float* dGate_t = &dGate_[t * F];
                        float* dUp_t = &dUp_[t * F];
                        float* dXnFfn_t = &dXnormFfn_[t * H];
                        const float* gate_t = ffnGate + t * F;
                        const float* up_t = ffnUp + t * F;

                        for (std::size_t f = 0; f < F; ++f) {
                            const float* wDownRow = &layers_[L].Wdown[f * H];
                            dAct_t[f] = dot_avx2_or_scalar(dOut, wDownRow, H);
                        }

                        for (std::size_t f = 0; f < F; ++f) {
                            float siluG = 0.0f;
                            float siluDeriv = 0.0f;
                            silu_value_and_derivative_scalar(gate_t[f], siluG, siluDeriv);
                            dUp_t[f] = dAct_t[f] * siluG;
                            dGate_t[f] = dAct_t[f] * up_t[f] * siluDeriv;
                        }

                        std::fill(dXnFfn_t, dXnFfn_t + H, 0.0f);
                        for (std::size_t h = 0; h < H; ++h) {
                            const float* wUpRow = &layers_[L].Wup[h * F];
                            const float* wGateRow = &layers_[L].Wgate[h * F];
                            float upDot = 0.0f;
                            float gateDot = 0.0f;
                            dot2_avx2_or_scalar(dUp_t, wUpRow, dGate_t, wGateRow, F, upDot, gateDot);
                            dXnFfn_t[h] = upDot + gateDot;
                        }

                        if (useRmsnormBwdAccum) {
                            rmsnorm_backward_accum(
                                &dXmid_[t * H],
                                dXnFfn_t,
                                ffnXmid + t * H,
                                layers_[L].rms_ffn.data(),
                                H, eps
                            );
                        }
                        else {
                            rmsnorm_backward(
                                &dXin_[t * H],
                                dXnFfn_t,
                                ffnXmid + t * H,
                                layers_[L].rms_ffn.data(),
                                H, eps
                            );

                            float* dXmid_t = &dXmid_[t * H];
                            const float* dXmidFromNorm_t = &dXin_[t * H];
                            for (std::size_t h = 0; h < H; ++h) dXmid_t[h] += dXmidFromNorm_t[h];
                        }
                    }
                    continue;
                }

                const float* dOut0 = &dX_[(t0 + 0) * H];
                const float* dOut1 = &dX_[(t0 + 1) * H];
                const float* dOut2 = &dX_[(t0 + 2) * H];
                const float* dOut3 = &dX_[(t0 + 3) * H];
                float* dAct0 = &dAct_[(t0 + 0) * F];
                float* dAct1 = &dAct_[(t0 + 1) * F];
                float* dAct2 = &dAct_[(t0 + 2) * F];
                float* dAct3 = &dAct_[(t0 + 3) * F];
                float* dGate0 = &dGate_[(t0 + 0) * F];
                float* dGate1 = &dGate_[(t0 + 1) * F];
                float* dGate2 = &dGate_[(t0 + 2) * F];
                float* dGate3 = &dGate_[(t0 + 3) * F];
                float* dUp0 = &dUp_[(t0 + 0) * F];
                float* dUp1 = &dUp_[(t0 + 1) * F];
                float* dUp2 = &dUp_[(t0 + 2) * F];
                float* dUp3 = &dUp_[(t0 + 3) * F];
                float* dXn0 = &dXnormFfn_[(t0 + 0) * H];
                float* dXn1 = &dXnormFfn_[(t0 + 1) * H];
                float* dXn2 = &dXnormFfn_[(t0 + 2) * H];
                float* dXn3 = &dXnormFfn_[(t0 + 3) * H];
                const float* gate0 = ffnGate + (t0 + 0) * F;
                const float* gate1 = ffnGate + (t0 + 1) * F;
                const float* gate2 = ffnGate + (t0 + 2) * F;
                const float* gate3 = ffnGate + (t0 + 3) * F;
                const float* up0 = ffnUp + (t0 + 0) * F;
                const float* up1 = ffnUp + (t0 + 1) * F;
                const float* up2 = ffnUp + (t0 + 2) * F;
                const float* up3 = ffnUp + (t0 + 3) * F;

                for (std::size_t f = 0; f < F; ++f) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    dot_x4_avx2_or_scalar(dOut0, dOut1, dOut2, dOut3, &layers_[L].Wdown[f * H], H, a0, a1, a2, a3);
                    dAct0[f] = (attentionMask[t0 + 0] == 0) ? 0.0f : a0;
                    dAct1[f] = (attentionMask[t0 + 1] == 0) ? 0.0f : a1;
                    dAct2[f] = (attentionMask[t0 + 2] == 0) ? 0.0f : a2;
                    dAct3[f] = (attentionMask[t0 + 3] == 0) ? 0.0f : a3;
                }

                if (fp32FfnFastSiluBackwardEnabled()) {
                    silu_backward_avx2_or_scalar(dGate0, dUp0, dAct0, gate0, up0, F);
                    silu_backward_avx2_or_scalar(dGate1, dUp1, dAct1, gate1, up1, F);
                    silu_backward_avx2_or_scalar(dGate2, dUp2, dAct2, gate2, up2, F);
                    silu_backward_avx2_or_scalar(dGate3, dUp3, dAct3, gate3, up3, F);
                }
                else {
                    for (std::size_t f = 0; f < F; ++f) {
                        float silu0 = 0.0f, silu1 = 0.0f, silu2 = 0.0f, silu3 = 0.0f;
                        float deriv0 = 0.0f, deriv1 = 0.0f, deriv2 = 0.0f, deriv3 = 0.0f;
                        silu_value_and_derivative_scalar(gate0[f], silu0, deriv0);
                        silu_value_and_derivative_scalar(gate1[f], silu1, deriv1);
                        silu_value_and_derivative_scalar(gate2[f], silu2, deriv2);
                        silu_value_and_derivative_scalar(gate3[f], silu3, deriv3);
                        dUp0[f] = dAct0[f] * silu0;
                        dUp1[f] = dAct1[f] * silu1;
                        dUp2[f] = dAct2[f] * silu2;
                        dUp3[f] = dAct3[f] * silu3;
                        dGate0[f] = dAct0[f] * up0[f] * deriv0;
                        dGate1[f] = dAct1[f] * up1[f] * deriv1;
                        dGate2[f] = dAct2[f] * up2[f] * deriv2;
                        dGate3[f] = dAct3[f] * up3[f] * deriv3;
                    }
                }

                for (std::size_t h = 0; h < H; ++h) {
                    float upDot0 = 0.0f, upDot1 = 0.0f, upDot2 = 0.0f, upDot3 = 0.0f;
                    float gateDot0 = 0.0f, gateDot1 = 0.0f, gateDot2 = 0.0f, gateDot3 = 0.0f;
                    const float* wUpRow = &layers_[L].Wup[h * F];
                    const float* wGateRow = &layers_[L].Wgate[h * F];
                    dot_x4_avx2_or_scalar(dUp0, dUp1, dUp2, dUp3, wUpRow, F, upDot0, upDot1, upDot2, upDot3);
                    dot_x4_avx2_or_scalar(dGate0, dGate1, dGate2, dGate3, wGateRow, F, gateDot0, gateDot1, gateDot2, gateDot3);
                    dXn0[h] = (attentionMask[t0 + 0] == 0) ? 0.0f : (upDot0 + gateDot0);
                    dXn1[h] = (attentionMask[t0 + 1] == 0) ? 0.0f : (upDot1 + gateDot1);
                    dXn2[h] = (attentionMask[t0 + 2] == 0) ? 0.0f : (upDot2 + gateDot2);
                    dXn3[h] = (attentionMask[t0 + 3] == 0) ? 0.0f : (upDot3 + gateDot3);
                }

                if (useRmsnormBwdAccum) {
                    if (attentionMask[t0 + 0] != 0) rmsnorm_backward_accum(&dXmid_[(t0 + 0) * H], dXn0, ffnXmid + (t0 + 0) * H, layers_[L].rms_ffn.data(), H, eps);
                    if (attentionMask[t0 + 1] != 0) rmsnorm_backward_accum(&dXmid_[(t0 + 1) * H], dXn1, ffnXmid + (t0 + 1) * H, layers_[L].rms_ffn.data(), H, eps);
                    if (attentionMask[t0 + 2] != 0) rmsnorm_backward_accum(&dXmid_[(t0 + 2) * H], dXn2, ffnXmid + (t0 + 2) * H, layers_[L].rms_ffn.data(), H, eps);
                    if (attentionMask[t0 + 3] != 0) rmsnorm_backward_accum(&dXmid_[(t0 + 3) * H], dXn3, ffnXmid + (t0 + 3) * H, layers_[L].rms_ffn.data(), H, eps);
                }
                else {
                    if (attentionMask[t0 + 0] != 0) rmsnorm_backward(&dXin_[(t0 + 0) * H], dXn0, ffnXmid + (t0 + 0) * H, layers_[L].rms_ffn.data(), H, eps);
                    if (attentionMask[t0 + 1] != 0) rmsnorm_backward(&dXin_[(t0 + 1) * H], dXn1, ffnXmid + (t0 + 1) * H, layers_[L].rms_ffn.data(), H, eps);
                    if (attentionMask[t0 + 2] != 0) rmsnorm_backward(&dXin_[(t0 + 2) * H], dXn2, ffnXmid + (t0 + 2) * H, layers_[L].rms_ffn.data(), H, eps);
                    if (attentionMask[t0 + 3] != 0) rmsnorm_backward(&dXin_[(t0 + 3) * H], dXn3, ffnXmid + (t0 + 3) * H, layers_[L].rms_ffn.data(), H, eps);

                    if (attentionMask[t0 + 0] != 0) for (std::size_t h = 0; h < H; ++h) dXmid_[(t0 + 0) * H + h] += dXin_[(t0 + 0) * H + h];
                    if (attentionMask[t0 + 1] != 0) for (std::size_t h = 0; h < H; ++h) dXmid_[(t0 + 1) * H + h] += dXin_[(t0 + 1) * H + h];
                    if (attentionMask[t0 + 2] != 0) for (std::size_t h = 0; h < H; ++h) dXmid_[(t0 + 2) * H + h] += dXin_[(t0 + 2) * H + h];
                    if (attentionMask[t0 + 3] != 0) for (std::size_t h = 0; h < H; ++h) dXmid_[(t0 + 3) * H + h] += dXin_[(t0 + 3) * H + h];
                }
            }
        }
        if (profile) prof.bwdFfnMs += elapsedMs(ffnStart);

        // x_mid = x_in + attn_out, so attn branch receives dXmid.
        const auto attnTailStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        const float* woGradSrc = dXmid_.data();
        if (cfg_.loraTargetO) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (std::size_t i = 0; i < activeThSize; ++i) {
                dAttn_[i] = dXmid_[i];
            }
            if (cfg_.useDoRA) {
                auto& P = loras_[L];
                auto& G = loraGrads_[L];
                const float* xIn = x_ckpt_.data() + L * CKPT_STRIDE;
                for (std::size_t t = 0; t < activeComputeT; ++t) {
                    if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                    for (std::size_t h = 0; h < H; ++h) {
                        attn_[t * H + h] = layerXmidCache[t * H + h] - xIn[t * H + h];
                    }
                    dora_backward_magnitude_and_scale(&dAttn_[t * H], &attn_[t * H], P.Co, P.Ro, G.gradMo, H);
                }
            }
            woGradSrc = dAttn_.data();
        }

        double woBwdMs = 0.0;
        double attnCoreBwdMs = 0.0;
        double ropeBwdMs = 0.0;
        double loraBwdMs = 0.0;
        double qkvBwdMs = 0.0;
        double residualBwdMs = 0.0;

        const auto woBwdStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        if (useI2SWo) {
            const std::size_t nYBlocks = layers_[L].Wo_q.nJBlocks();
            (void)nYBlocks;
            // dAttn rows are H-wide; nJBlocks = ceil(H/jq). bwdDY* are sized to
            // ceil(V/jq) which is always >= ceil(H/jq) since V >= H.
            linear_backward_bitnet_i2s_packed(
                dCtx_.data(),
                woGradSrc,
                layers_[L].Wo_q,
                activeComputeT,
                tTile,
                bwdDY8Scratch_.data(),
                bwdDYScaleScratch_.data(),
                i2sAttentionMask,
                i2sActivePrefixT,
                useWoBackwardOverwrite);
        }
        else {
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                const float* dat = woGradSrc + t * H; float* dct = &dCtx_[t * H];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum = 0; for (std::size_t j = 0; j < H; ++j) sum += dat[j] * layers_[L].Wo[i * H + j]; dct[i] += sum;
                }
            }
        }
        if (profile) {
            woBwdMs = elapsedMs(woBwdStart);
            prof.bwdWoMs += woBwdMs;
        }

        if (cfg_.loraTargetO) {
            auto& gL = loraGrads_[L];
            const float* const attnCtx = layerCtxCache ? layerCtxCache : ctx_.data();
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;

                const float* dy = &dAttn_[t * H];
                const float* xCtx = attnCtx + t * H;
                const float* tmp = &tmpO_cache_[((L * cfg_.maxSeqLen) + t) * R];
                const float* B = loras_[L].Bo.data();
                const float* A = loras_[L].Ao.data();

                for (std::size_t r = 0; r < R; ++r) {
                    const float tr = scale * tmp[r];
                    float* gBrow = &gL.gradBo[r * H];
                    const float* Brow = &B[r * H];
                    axpy_inplace_avx2_or_scalar(gBrow, dy, H, tr);
                    dTmpRank_[r] = scale * dot_avx2_or_scalar(Brow, dy, H);
                }

                for (std::size_t r = 0; r < R; ++r) {
                    float* gArow = &gL.gradAo[r * H];
                    axpy_inplace_avx2_or_scalar(gArow, xCtx, H, dTmpRank_[r]);
                }

                float* dct = &dCtx_[t * H];
                for (std::size_t r = 0; r < R; ++r) {
                    const float* Arow = &A[r * H];
                    axpy_inplace_avx2_or_scalar(dct, Arow, H, dTmpRank_[r]);
                }
            }
        }

        if (!layerProbsCache) {
            // Recompute detached attention probabilities for long-context runs
            // instead of keeping one T x T matrix per layer resident.
            attn_causal(
                ctx_.data(),
                attnQ,
                attnK,
                attnV,
                activeComputeT,
                H,
                shape_.nHeads,
                nKvHeads,
                nullptr,
                probs_.data(),
                activeAttentionMask);
        }

        const auto attnCoreBwdStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        attn_causal_backward(
            dQ_.data(), dK_.data(), dV_.data(),
            dCtx_.data(),
            attnQ, attnK, attnV,
            activeComputeT, H, shape_.nHeads, nKvHeads, attnProbs,
            activeAttentionMask, tmpdP_.data()
        );
        if (profile) {
            attnCoreBwdMs = elapsedMs(attnCoreBwdStart);
        }

        // Backprop through RoPE: d(pre-rope Q/K) = R(theta)^T * d(post-rope Q/K).
        const auto ropeBwdStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::size_t t = 0; t < activeComputeT; ++t) {
            if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
            const std::size_t ropePos = segmentStarts.empty() ? t : (t - segmentStarts[t]);
            for (std::size_t h = 0; h < nHeads; ++h) {
                apply_rope_backward_inplace_cached(
                    &dQ_[t * H + h * D],
                    D,
                    ropeDim,
                    ropePos,
                    ropeCosTable_.data(),
                    ropeSinTable_.data());
            }
            for (std::size_t kvh = 0; kvh < nKvHeads; ++kvh) {
                apply_rope_backward_inplace_cached(
                    &dK_[t * kvDim + kvh * D],
                    D,
                    ropeDim,
                    ropePos,
                    ropeCosTable_.data(),
                    ropeSinTable_.data());
            }
        }
        if (profile) {
            ropeBwdMs = elapsedMs(ropeBwdStart);
            prof.bwdRopeMs += ropeBwdMs;
        }

        const bool recomputeDoRAPreRope =
            cfg_.useDoRA &&
            cfg_.doraRecomputeQKPreRope &&
            (cfg_.loraTargetQ || cfg_.loraTargetK);
        const float* qDoraPreRope = nullptr;
        const float* kDoraPreRope = nullptr;
        if (recomputeDoRAPreRope) {
            auto recomputeProjection = [&](float* out, std::size_t outDim, char target) {
                std::fill_n(out, T * outDim, 0.0f);
                if (target == 'q') {
                    if (recomputeUseTl2GenQ) {
                        linear_forward_bitnet_tl2_generated(out, layerXnormCache, layers_[L].Wq_tl2g, activeComputeT);
                    }
                    else if (recomputeUseTl2Q) {
                        linear_forward_bitnet_tl2_accum(
                            out, layerXnormCache, layers_[L].Wq_tl2, activeComputeT, tTile,
                            fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                    }
                    else if (recomputeUseI2SQ) {
                        linear_forward_bitnet_i2s_accum(
                            out, layerXnormCache, layers_[L].Wq_q, activeComputeT, tTile,
                            fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                    }
                    else {
                        for (std::size_t t = 0; t < activeComputeT; ++t) {
                            if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                            const float* xn = layerXnormCache + t * H;
                            float* yt = out + t * outDim;
                            for (std::size_t i = 0; i < outDim; ++i) {
                                float sum = 0.0f;
                                for (std::size_t j = 0; j < H; ++j) sum += xn[j] * layers_[L].Wq[j * outDim + i];
                                yt[i] = sum;
                            }
                        }
                    }
                }
                else {
                    if (recomputeUseTl2GenK) {
                        linear_forward_bitnet_tl2_generated(out, layerXnormCache, layers_[L].Wk_tl2g, activeComputeT);
                    }
                    else if (recomputeUseTl2K) {
                        linear_forward_bitnet_tl2_accum(
                            out, layerXnormCache, layers_[L].Wk_tl2, activeComputeT, tTile,
                            fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                    }
                    else if (recomputeUseI2SK) {
                        linear_forward_bitnet_i2s_accum(
                            out, layerXnormCache, layers_[L].Wk_q, activeComputeT, tTile,
                            fwdX8Scratch_.data(), fwdXScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                    }
                    else {
                        for (std::size_t t = 0; t < activeComputeT; ++t) {
                            if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                            const float* xn = layerXnormCache + t * H;
                            float* yt = out + t * outDim;
                            for (std::size_t i = 0; i < outDim; ++i) {
                                float sum = 0.0f;
                                for (std::size_t j = 0; j < H; ++j) sum += xn[j] * layers_[L].Wk[j * outDim + i];
                                yt[i] = sum;
                            }
                        }
                    }
                }
            };

            auto& P = loras_[L];
            if (cfg_.loraTargetQ) {
                float* qPost = q_dora_prerope_recompute_.data();
                recomputeProjection(qPost, H, 'q');
                for (std::size_t t = 0; t < activeComputeT; ++t) {
                    if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                    float* tmpQ = &tmpQ_cache_[((L * cfg_.maxSeqLen) + t) * R];
                    lora_add_H(qPost + t * H, layerXnormCache + t * H, P.Aq.data(), P.Bq.data(), tmpQ, H, R, scale, hTile);
                    dora_apply_ratio(qPost + t * H, P.Rq, H);
                }
                qDoraPreRope = qPost;
            }
            if (cfg_.loraTargetK) {
                float* kPost = k_dora_prerope_recompute_.data();
                recomputeProjection(kPost, kvDim, 'k');
                for (std::size_t t = 0; t < activeComputeT; ++t) {
                    if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                    float* tmpK = &tmpK_cache_[((L * cfg_.maxSeqLen) + t) * R];
                    lora_add_out(kPost + t * kvDim, layerXnormCache + t * H, P.Ak.data(), P.Bk.data(), tmpK, H, kvDim, R, scale);
                    dora_apply_ratio(kPost + t * kvDim, P.Rk, kvDim);
                }
                kDoraPreRope = kPost;
            }
        }

        if (cfg_.useDoRA) {
            auto& P = loras_[L];
            auto& G = loraGrads_[L];
            const float* qPost = cfg_.loraTargetQ
                ? (cfg_.doraRecomputeQKPreRope ? qDoraPreRope : (q_dora_prerope_cache_.data() + layerTHOffset))
                : nullptr;
            const float* kPost = cfg_.loraTargetK
                ? (cfg_.doraRecomputeQKPreRope ? kDoraPreRope : (k_dora_prerope_cache_.data() + layerTKVOffset))
                : nullptr;
            const float* vPost = cfg_.loraTargetV ? attnV : nullptr;
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                if (cfg_.loraTargetQ) {
                    dora_backward_magnitude_and_scale(&dQ_[t * H], qPost + t * H, P.Cq, P.Rq, G.gradMq, H);
                }
                if (cfg_.loraTargetK) {
                    dora_backward_magnitude_and_scale(&dK_[t * kvDim], kPost + t * kvDim, P.Ck, P.Rk, G.gradMk, kvDim);
                }
                if (cfg_.loraTargetV) {
                    dora_backward_magnitude_and_scale(&dV_[t * kvDim], vPost + t * kvDim, P.Cv, P.Rv, G.gradMv, kvDim);
                }
            }
        }

        if (attentionPolicy.runBaseFirst) {
            const auto qkvBwdStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
            if (useFusedI2SQKVBackward) {
                linear_fused_qkv_backward_bitnet_i2s_packed(
                    dXnorm_.data(),
                    dQ_.data(), dK_.data(), dV_.data(),
                    layers_[L].Wq_q, layers_[L].Wk_q, layers_[L].Wv_q,
                    activeComputeT, attentionPolicy.exactTiles.qkvBackwardRows,
                    bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                    bwdDK8Scratch_.data(), bwdDKScaleScratch_.data(),
                    bwdDV8Scratch_.data(), bwdDVScaleScratch_.data(),
                    true);
            }
            else if (useI2SQ || useI2SK || useI2SV) {
                bool baseDxInitialized = false;
                if (useI2SQ) {
                    linear_backward_bitnet_i2s_packed(
                        dXnorm_.data(), dQ_.data(), layers_[L].Wq_q,
                        activeComputeT, tTile,
                        bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                        i2sAttentionMask, i2sActivePrefixT,
                        !baseDxInitialized);
                    baseDxInitialized = true;
                }
                if (useFusedI2SKV) {
                    linear_fused_dual_backward_bitnet_i2s_packed(
                        dXnorm_.data(),
                        dK_.data(), dV_.data(),
                        layers_[L].Wk_q, layers_[L].Wv_q,
                        activeComputeT, tTile,
                        bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                        bwdDK8Scratch_.data(), bwdDKScaleScratch_.data(),
                        i2sAttentionMask, i2sActivePrefixT,
                        !baseDxInitialized);
                    baseDxInitialized = true;
                }
                else {
                    if (useI2SK) {
                        linear_backward_bitnet_i2s_packed(
                            dXnorm_.data(), dK_.data(), layers_[L].Wk_q,
                            activeComputeT, tTile,
                            bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                            i2sAttentionMask, i2sActivePrefixT,
                            !baseDxInitialized);
                        baseDxInitialized = true;
                    }
                    if (useI2SV) {
                        linear_backward_bitnet_i2s_packed(
                            dXnorm_.data(), dV_.data(), layers_[L].Wv_q,
                            activeComputeT, tTile,
                            bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                            i2sAttentionMask, i2sActivePrefixT,
                            !baseDxInitialized);
                        baseDxInitialized = true;
                    }
                }
            }
            else {
                for (std::size_t t = 0; t < T; ++t) {
                    if (attentionMask[t] == 0) continue;
                    float* dxn = &dXnorm_[t * H];
                    const float* dqt = &dQ_[t * H];
                    const float* dkt = &dK_[t * kvDim];
                    const float* dvt = &dV_[t * kvDim];
                    for (std::size_t i = 0; i < H; ++i) {
                        const float* wqRow = &layers_[L].Wq[i * H];
                        const float* wkRow = &layers_[L].Wk[i * kvDim];
                        const float* wvRow = &layers_[L].Wv[i * kvDim];
                        dxn[i] = dot_avx2_or_scalar(dqt, wqRow, H)
                            + dot_avx2_or_scalar(dkt, wkRow, kvDim)
                            + dot_avx2_or_scalar(dvt, wvRow, kvDim);
                    }
                }
            }
            if (profile) {
                const double baseQkvMs = elapsedMs(qkvBwdStart);
                qkvBwdMs += baseQkvMs;
                prof.bwdQkvMs += baseQkvMs;
            }
        }

        // ── LoRA gradient accumulation ────────────────────────────────────────
        auto& gL = loraGrads_[L];
        const auto loraBwdStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
#if defined(_OPENMP)
        if (attentionPolicy.parallelLora &&
            activeComputeT > 1 &&
            cfg_.loraTargetQ &&
            cfg_.loraTargetV &&
            !cfg_.loraTargetK &&
            !cfg_.loraTargetO) {
            const std::size_t tmpRankSize = 2 * activeComputeT * R;
            if (attnLoraTmpRankLocal_.size() < tmpRankSize) {
                attnLoraTmpRankLocal_.resize(tmpRankSize);
            }
            float* dTmpQ = attnLoraTmpRankLocal_.data();
            float* dTmpV = dTmpQ + activeComputeT * R;

            const float* Bq = loras_[L].Bq.data();
            const float* Aq = loras_[L].Aq.data();
            const float* Bv = loras_[L].Bv.data();
            const float* Av = loras_[L].Av.data();

#pragma omp parallel for schedule(static)
            for (int t_i = 0; t_i < static_cast<int>(activeComputeT); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                float* dTmpQRow = dTmpQ + t * R;
                float* dTmpVRow = dTmpV + t * R;
                if (activeAttentionMask && activeAttentionMask[t] == 0) {
                    std::fill_n(dTmpQRow, R, 0.0f);
                    std::fill_n(dTmpVRow, R, 0.0f);
                    continue;
                }

                const float* dyQ = &dQ_[t * H];
                const float* dyV = &dV_[t * kvDim];
                for (std::size_t r = 0; r < R; ++r) {
                    dTmpQRow[r] = scale * dot_avx2_or_scalar(&Bq[r * H], dyQ, H);
                    dTmpVRow[r] = scale * dot_avx2_or_scalar(&Bv[r * kvDim], dyV, kvDim);
                }
            }

#pragma omp parallel for schedule(static)
            for (int r_i = 0; r_i < static_cast<int>(R); ++r_i) {
                const std::size_t r = static_cast<std::size_t>(r_i);
                float* gradAqRow = &gL.gradAq[r * H];
                float* gradBqRow = &gL.gradBq[r * H];
                float* gradAvRow = &gL.gradAv[r * H];
                float* gradBvRow = &gL.gradBv[r * kvDim];

                for (std::size_t t = 0; t < activeComputeT; ++t) {
                    if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                    const float* xnorm_t = attnXnorm + t * H;
                    const float* dyQ = &dQ_[t * H];
                    const float* dyV = &dV_[t * kvDim];
                    const float* tmpQ = &tmpQ_cache_[((L * cfg_.maxSeqLen) + t) * R];
                    const float* tmpV = &tmpV_cache_[((L * cfg_.maxSeqLen) + t) * R];

                    axpy_inplace_avx2_or_scalar(gradBqRow, dyQ, H, scale * tmpQ[r]);
                    axpy_inplace_avx2_or_scalar(gradAqRow, xnorm_t, H, dTmpQ[t * R + r]);
                    axpy_inplace_avx2_or_scalar(gradBvRow, dyV, kvDim, scale * tmpV[r]);
                    axpy_inplace_avx2_or_scalar(gradAvRow, xnorm_t, H, dTmpV[t * R + r]);
                }
            }

#pragma omp parallel for schedule(static)
            for (int t_i = 0; t_i < static_cast<int>(activeComputeT); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
                float* dxn = &dXnorm_[t * H];
                const float* dTmpQRow = dTmpQ + t * R;
                const float* dTmpVRow = dTmpV + t * R;
                for (std::size_t r = 0; r < R; ++r) {
                    axpy_inplace_avx2_or_scalar(dxn, &Aq[r * H], H, dTmpQRow[r]);
                    axpy_inplace_avx2_or_scalar(dxn, &Av[r * H], H, dTmpVRow[r]);
                }
            }
        }
        else
#endif
        {
            for (std::size_t t = 0; t < activeComputeT; ++t) {
                if (activeAttentionMask && activeAttentionMask[t] == 0) continue;

                const float* xnorm_t = attnXnorm + t * H;

                // Q LoRA grads
                if (cfg_.loraTargetQ) {
                    const float* dy = &dQ_[t * H];
                    const float* tmp = &tmpQ_cache_[((L * cfg_.maxSeqLen) + t) * R];
                    const float* B = loras_[L].Bq.data();
                    const float* A = loras_[L].Aq.data();

                    for (std::size_t r = 0; r < R; ++r) {
                        const float  tr = scale * tmp[r];
                        float* gBrow = &gL.gradBq[r * H];
                        const float* Brow = &B[r * H];
                        axpy_inplace_avx2_or_scalar(gBrow, dy, H, tr);
                        dTmpRank_[r] = scale * dot_avx2_or_scalar(Brow, dy, H);
                    }

                    for (std::size_t r = 0; r < R; ++r) {
                        const float  dtr = dTmpRank_[r];
                        float* gArow = &gL.gradAq[r * H];
                        axpy_inplace_avx2_or_scalar(gArow, xnorm_t, H, dtr);
                    }

                    float* dxn = &dXnorm_[t * H];
                    for (std::size_t r = 0; r < R; ++r) {
                        const float  dr = dTmpRank_[r];
                        const float* Arow = &A[r * H];
                        axpy_inplace_avx2_or_scalar(dxn, Arow, H, dr);
                    }
                }

                // K LoRA grads
                if (cfg_.loraTargetK) {
                    const float* dyK = &dK_[t * kvDim];
                    const float* tmpK = &tmpK_cache_[((L * cfg_.maxSeqLen) + t) * R];
                    const float* Bk = loras_[L].Bk.data();
                    const float* Ak = loras_[L].Ak.data();

                    for (std::size_t r = 0; r < R; ++r) {
                        const float tr = scale * tmpK[r];
                        float* gBrow = &gL.gradBk[r * kvDim];
                        const float* Brow = &Bk[r * kvDim];
                        axpy_inplace_avx2_or_scalar(gBrow, dyK, kvDim, tr);
                        dTmpRank_[r] = scale * dot_avx2_or_scalar(Brow, dyK, kvDim);
                    }

                    for (std::size_t r = 0; r < R; ++r) {
                        float* gArow = &gL.gradAk[r * H];
                        axpy_inplace_avx2_or_scalar(gArow, xnorm_t, H, dTmpRank_[r]);
                    }

                    float* dxn = &dXnorm_[t * H];
                    for (std::size_t r = 0; r < R; ++r) {
                        const float* Arow = &Ak[r * H];
                        axpy_inplace_avx2_or_scalar(dxn, Arow, H, dTmpRank_[r]);
                    }
                }

                // V LoRA grads
                if (cfg_.loraTargetV) {
                    const float* dyV = &dV_[t * kvDim];
                    const float* tmpV = &tmpV_cache_[((L * cfg_.maxSeqLen) + t) * R];
                    const float* Bv = loras_[L].Bv.data();
                    const float* Av = loras_[L].Av.data();

                    for (std::size_t r = 0; r < R; ++r) {
                        const float  tr = scale * tmpV[r];
                        float* gBrow = &gL.gradBv[r * kvDim];
                        const float* Brow = &Bv[r * kvDim];
                        axpy_inplace_avx2_or_scalar(gBrow, dyV, kvDim, tr);
                        dTmpRank_[r] = scale * dot_avx2_or_scalar(Brow, dyV, kvDim);
                    }

                    for (std::size_t r = 0; r < R; ++r) {
                        const float  dtr = dTmpRank_[r];
                        float* gArow = &gL.gradAv[r * H];
                        axpy_inplace_avx2_or_scalar(gArow, xnorm_t, H, dtr);
                    }

                    float* dxn = &dXnorm_[t * H];
                    for (std::size_t r = 0; r < R; ++r) {
                        const float  dr = dTmpRank_[r];
                        const float* Arow = &Av[r * H];
                        axpy_inplace_avx2_or_scalar(dxn, Arow, H, dr);
                    }
                }
            }
        }
        if (profile) {
            loraBwdMs = elapsedMs(loraBwdStart);
            prof.bwdLoraMs += loraBwdMs;
        }

        // ── Base weight backward into dXnorm_ ────────────────────────────────
        if (!attentionPolicy.runBaseFirst) {
            const auto qkvBwdStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
            if (useFusedI2SQKVBackward) {
                linear_fused_qkv_backward_bitnet_i2s_packed(
                    dXnorm_.data(),
                    dQ_.data(), dK_.data(), dV_.data(),
                    layers_[L].Wq_q, layers_[L].Wk_q, layers_[L].Wv_q,
                    activeComputeT, attentionPolicy.exactTiles.qkvBackwardRows,
                    bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                    bwdDK8Scratch_.data(), bwdDKScaleScratch_.data(),
                    bwdDV8Scratch_.data(), bwdDVScaleScratch_.data(),
                    false);
            }
            else if (useI2SQ || useI2SK || useI2SV) {
                if (useI2SQ) {
                    linear_backward_bitnet_i2s_packed(dXnorm_.data(), dQ_.data(), layers_[L].Wq_q, activeComputeT, tTile, bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                }
                if (useFusedI2SKV) {
                    linear_fused_dual_backward_bitnet_i2s_packed(
                        dXnorm_.data(),
                        dK_.data(), dV_.data(),
                        layers_[L].Wk_q, layers_[L].Wv_q,
                        activeComputeT, tTile,
                        bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                        bwdDK8Scratch_.data(), bwdDKScaleScratch_.data(),
                        i2sAttentionMask, i2sActivePrefixT);
                }
                else {
                    if (useI2SK) {
                        linear_backward_bitnet_i2s_packed(dXnorm_.data(), dK_.data(), layers_[L].Wk_q, activeComputeT, tTile, bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                    }
                    if (useI2SV) {
                        linear_backward_bitnet_i2s_packed(dXnorm_.data(), dV_.data(), layers_[L].Wv_q, activeComputeT, tTile, bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(), i2sAttentionMask, i2sActivePrefixT);
                    }
                }
            }
            else {
                for (std::size_t t = 0; t < T; ++t) {
                    if (attentionMask[t] == 0) continue;
                    float* dxn = &dXnorm_[t * H];
                    const float* dqt = &dQ_[t * H]; const float* dkt = &dK_[t * kvDim]; const float* dvt = &dV_[t * kvDim];
                    for (std::size_t i = 0; i < H; ++i) {
                        const float* wqRow = &layers_[L].Wq[i * H];
                        const float* wkRow = &layers_[L].Wk[i * kvDim];
                        const float* wvRow = &layers_[L].Wv[i * kvDim];
                        dxn[i] += dot_avx2_or_scalar(dqt, wqRow, H)
                            + dot_avx2_or_scalar(dkt, wkRow, kvDim)
                            + dot_avx2_or_scalar(dvt, wvRow, kvDim);
                    }
                }
            }
            if (profile) {
                qkvBwdMs = elapsedMs(qkvBwdStart);
                prof.bwdQkvMs += qkvBwdMs;
            }
        }

        // ── FIX: Correct residual gradient accumulation ───────────────────────
        const auto residualBwdStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::size_t t = 0; t < activeComputeT; ++t) {
            if (activeAttentionMask && activeAttentionMask[t] == 0) continue;
            if (cfg_.trainLayerRmsNorm) {
                accumulate_rmsnorm_weight_grad(
                    gL.gradRmsAttn.data(),
                    &dXnorm_[t * H],
                    &x_[t * H],
                    H,
                    eps);
            }
            // Step 1: compute the RMSNorm gradient into the zero-initialised dXin_
            rmsnorm_backward(
                &dXin_[t * H],
                &dXnorm_[t * H],
                &x_[t * H],
                layers_[L].rms_attn.data(),
                H, eps
            );
        }
 
        // Step 2: x_mid = x_in + attn_out, so dX_in = dX_mid + d(attn_path)/dx_in.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::size_t i = 0; i < activeThSize; ++i) {
            dX_[i] = dXmid_[i] + dXin_[i];
        }
        if (profile) {
            residualBwdMs = elapsedMs(residualBwdStart);
            prof.bwdResidualMs += residualBwdMs;
        }
        if (profile) {
            const double attnTailMs = elapsedMs(attnTailStart);
            const double splitMs =
                woBwdMs +
                attnCoreBwdMs +
                ropeBwdMs +
                loraBwdMs +
                qkvBwdMs +
                residualBwdMs;
            // Keep bwdAttnMs as the whole attention-tail bucket while the
            // substage fields capture the newer breakdown.
            prof.bwdAttnMs += std::max(attnTailMs, splitMs);
        }
    }

    if (profile) printBackendProfile();
}

// ==============================================================================
// UTILITIES
// ==============================================================================
void CpuKernelBackend::zeroGrad()
{
    for (auto& g : loraGrads_) {
        std::fill(g.gradAq.begin(), g.gradAq.end(), 0.0f);
        std::fill(g.gradBq.begin(), g.gradBq.end(), 0.0f);
        std::fill(g.gradAk.begin(), g.gradAk.end(), 0.0f);
        std::fill(g.gradBk.begin(), g.gradBk.end(), 0.0f);
        std::fill(g.gradAv.begin(), g.gradAv.end(), 0.0f);
        std::fill(g.gradBv.begin(), g.gradBv.end(), 0.0f);
        std::fill(g.gradAo.begin(), g.gradAo.end(), 0.0f);
        std::fill(g.gradBo.begin(), g.gradBo.end(), 0.0f);
        std::fill(g.gradAfd.begin(), g.gradAfd.end(), 0.0f);
        std::fill(g.gradBfd.begin(), g.gradBfd.end(), 0.0f);
        std::fill(g.gradAfg.begin(), g.gradAfg.end(), 0.0f);
        std::fill(g.gradBfg.begin(), g.gradBfg.end(), 0.0f);
        std::fill(g.gradAfu.begin(), g.gradAfu.end(), 0.0f);
        std::fill(g.gradBfu.begin(), g.gradBfu.end(), 0.0f);
        std::fill(g.gradMq.begin(), g.gradMq.end(), 0.0f);
        std::fill(g.gradMk.begin(), g.gradMk.end(), 0.0f);
        std::fill(g.gradMv.begin(), g.gradMv.end(), 0.0f);
        std::fill(g.gradMo.begin(), g.gradMo.end(), 0.0f);
        std::fill(g.gradMfd.begin(), g.gradMfd.end(), 0.0f);
        std::fill(g.gradMfg.begin(), g.gradMfg.end(), 0.0f);
        std::fill(g.gradMfu.begin(), g.gradMfu.end(), 0.0f);
        std::fill(g.gradRmsAttn.begin(), g.gradRmsAttn.end(), 0.0f);
        std::fill(g.gradRmsFfn.begin(), g.gradRmsFfn.end(), 0.0f);
    }
    std::fill(lmHeadLoraGrads_.gradA.begin(), lmHeadLoraGrads_.gradA.end(), 0.0f);
    std::fill(lmHeadLoraGrads_.gradB.begin(), lmHeadLoraGrads_.gradB.end(), 0.0f);
    std::fill(lmHeadLoraGrads_.gradM.begin(), lmHeadLoraGrads_.gradM.end(), 0.0f);
    std::fill(lmHeadBiasGrad_.begin(), lmHeadBiasGrad_.end(), 0.0f);
    std::fill(finalNormScaleGrad_.begin(), finalNormScaleGrad_.end(), 0.0f);
    std::fill(finalNormBiasGrad_.begin(), finalNormBiasGrad_.end(), 0.0f);
    std::fill(rmsFinalGrad_.begin(), rmsFinalGrad_.end(), 0.0f);
}

void CpuKernelBackend::markTrainableParamsUpdated()
{
    if (cfg_.useDoRA) {
        ++doraParamVersion_;
    }
}

void CpuKernelBackend::saveLoRA(const std::string& path) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write to " + path);

    const uint32_t H = static_cast<uint32_t>(shape_.hiddenSize);
    const uint32_t R = static_cast<uint32_t>(cfg_.loraRank);
    uint32_t flags = 0;
    if (cfg_.loraTargetQ) flags |= kLoraFlagQ;
    if (cfg_.loraTargetK) flags |= kLoraFlagK;
    if (cfg_.loraTargetV) flags |= kLoraFlagV;
    if (cfg_.loraTargetO) flags |= kLoraFlagO;
    if (cfg_.loraTargetFfnDown) flags |= kLoraFlagFfnDown;
    if (cfg_.loraTargetLmHead) flags |= kLoraFlagLmHead;
    if (cfg_.loraTargetFfnGateUp) flags |= kLoraFlagFfnGateUp;
    if (cfg_.trainLmHeadBias) flags |= kLoraFlagLmHeadBias;
    if (cfg_.trainFinalNormAffine) flags |= kLoraFlagFinalNormAffine;
    if (cfg_.trainLayerRmsNorm) flags |= kLoraFlagLayerRmsNorm;
    if (cfg_.trainFinalRmsNorm) flags |= kLoraFlagFinalRmsNorm;
    if (cfg_.useDoRA) flags |= kLoraFlagDoRA;
    if (cfg_.usePiSSA) flags |= kLoraFlagPiSSA;
    const uint32_t nL = static_cast<uint32_t>(loras_.size());
    const float    alpha = cfg_.loraAlpha;

    out.write("LORA", 4);
    uint32_t version = cfg_.usePiSSA ? 3u : (cfg_.useDoRA ? 2u : 1u), dtype = 0;
    out.write((char*)&version, 4);
    out.write((char*)&dtype, 4);
    out.write((char*)&flags, 4);
    out.write((char*)&nL, 4);
    out.write((char*)&H, 4);
    out.write((char*)&R, 4);
    out.write((char*)&alpha, 4);

    const std::size_t kvDim = shape_.kvDim(); // from ModelShape, not fragile Wk.size()/H
    const uint32_t    kvD = static_cast<uint32_t>(kvDim);

    for (std::size_t L = 0; L < nL; ++L) {
        if (flags & kLoraFlagQ) {
            out.write((const char*)loras_[L].Aq.data(), R * H * sizeof(float));
            out.write((const char*)loras_[L].Bq.data(), H * R * sizeof(float));
            if (cfg_.useDoRA) out.write((const char*)loras_[L].Mq.data(), H * sizeof(float));
        }
        if (flags & kLoraFlagK) {
            out.write((const char*)loras_[L].Ak.data(), R * H * sizeof(float));
            out.write((const char*)loras_[L].Bk.data(), kvD * R * sizeof(float));
            if (cfg_.useDoRA) out.write((const char*)loras_[L].Mk.data(), kvD * sizeof(float));
        }
        if (flags & kLoraFlagV) {
            out.write((const char*)loras_[L].Av.data(), R * H * sizeof(float));
            out.write((const char*)loras_[L].Bv.data(), kvD * R * sizeof(float));
            if (cfg_.useDoRA) out.write((const char*)loras_[L].Mv.data(), kvD * sizeof(float));
        }
        if (flags & kLoraFlagO) {
            out.write((const char*)loras_[L].Ao.data(), R * H * sizeof(float));
            out.write((const char*)loras_[L].Bo.data(), H * R * sizeof(float));
            if (cfg_.useDoRA) out.write((const char*)loras_[L].Mo.data(), H * sizeof(float));
        }
        if (flags & kLoraFlagFfnDown) {
            out.write((const char*)loras_[L].Afd.data(), R * shape_.ffnSize * sizeof(float));
            out.write((const char*)loras_[L].Bfd.data(), H * R * sizeof(float));
            if (cfg_.useDoRA) out.write((const char*)loras_[L].Mfd.data(), H * sizeof(float));
        }
        if (flags & kLoraFlagFfnGateUp) {
            out.write((const char*)loras_[L].Afg.data(), R * H * sizeof(float));
            out.write((const char*)loras_[L].Bfg.data(), R * shape_.ffnSize * sizeof(float));
            if (cfg_.useDoRA) out.write((const char*)loras_[L].Mfg.data(), shape_.ffnSize * sizeof(float));
            out.write((const char*)loras_[L].Afu.data(), R * H * sizeof(float));
            out.write((const char*)loras_[L].Bfu.data(), R * shape_.ffnSize * sizeof(float));
            if (cfg_.useDoRA) out.write((const char*)loras_[L].Mfu.data(), shape_.ffnSize * sizeof(float));
        }
        if (flags & kLoraFlagLayerRmsNorm) {
            out.write((const char*)layers_[L].rms_attn.data(), H * sizeof(float));
            out.write((const char*)layers_[L].rms_ffn.data(), H * sizeof(float));
        }
    }
    if (flags & kLoraFlagLmHead) {
        out.write((const char*)lmHeadLora_.A.data(), R * H * sizeof(float));
        out.write((const char*)lmHeadLora_.B.data(), R * shape_.vocabSize * sizeof(float));
        if (cfg_.useDoRA) out.write((const char*)lmHeadLora_.M.data(), shape_.vocabSize * sizeof(float));
    }
    if (flags & kLoraFlagLmHeadBias) {
        out.write((const char*)lmHeadBias_.data(), shape_.vocabSize * sizeof(float));
    }
    if (flags & kLoraFlagFinalNormAffine) {
        out.write((const char*)finalNormScale_.data(), H * sizeof(float));
        out.write((const char*)finalNormBias_.data(), H * sizeof(float));
    }
    if (flags & kLoraFlagFinalRmsNorm) {
        out.write((const char*)rms_final_.data(), H * sizeof(float));
    }
}

void CpuKernelBackend::loadLoRA(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return;

    char magic[4]; in.read(magic, 4);
    if (std::strncmp(magic, "LORA", 4) != 0) return;

    uint32_t version, dtype, flags, nL, H, R; float alpha;
    in.read((char*)&version, 4); in.read((char*)&dtype, 4);
    in.read((char*)&flags, 4); in.read((char*)&nL, 4);
    in.read((char*)&H, 4); in.read((char*)&R, 4);
    in.read((char*)&alpha, 4);

    const std::size_t kvDim = shape_.kvDim(); // from ModelShape, not fragile Wk.size()/H
    if (H != shape_.hiddenSize || R != cfg_.loraRank || nL > loras_.size()) {
        throw std::runtime_error("loadLoRA: adapter shape does not match this run");
    }
    const bool fileHasDoRA = (flags & kLoraFlagDoRA) != 0;
    if (fileHasDoRA && !cfg_.useDoRA) {
        throw std::runtime_error("loadLoRA: adapter has DoRA magnitudes but this run disabled useDoRA");
    }
    const bool fileHasPiSSA = (flags & kLoraFlagPiSSA) != 0;
    if (fileHasPiSSA && !cfg_.usePiSSA) {
        throw std::runtime_error("loadLoRA: adapter was trained against a PiSSA residual base; enable usePiSSA or export a normal-base delta first");
    }

    for (std::size_t L = 0; L < nL; ++L) {
        if (flags & kLoraFlagQ) {
            in.read((char*)loras_[L].Aq.data(), R * H * sizeof(float));
            in.read((char*)loras_[L].Bq.data(), H * R * sizeof(float));
            if (fileHasDoRA) in.read((char*)loras_[L].Mq.data(), H * sizeof(float));
        }
        if (flags & kLoraFlagK) {
            in.read((char*)loras_[L].Ak.data(), R * H * sizeof(float));
            in.read((char*)loras_[L].Bk.data(), kvDim * R * sizeof(float));
            if (fileHasDoRA) in.read((char*)loras_[L].Mk.data(), kvDim * sizeof(float));
        }
        if (flags & kLoraFlagV) {
            in.read((char*)loras_[L].Av.data(), R * H * sizeof(float));
            in.read((char*)loras_[L].Bv.data(), kvDim * R * sizeof(float));
            if (fileHasDoRA) in.read((char*)loras_[L].Mv.data(), kvDim * sizeof(float));
        }
        if (flags & kLoraFlagO) {
            in.read((char*)loras_[L].Ao.data(), R * H * sizeof(float));
            in.read((char*)loras_[L].Bo.data(), H * R * sizeof(float));
            if (fileHasDoRA) in.read((char*)loras_[L].Mo.data(), H * sizeof(float));
        }
        if (flags & kLoraFlagFfnDown) {
            in.read((char*)loras_[L].Afd.data(), R * shape_.ffnSize * sizeof(float));
            in.read((char*)loras_[L].Bfd.data(), H * R * sizeof(float));
            if (fileHasDoRA) in.read((char*)loras_[L].Mfd.data(), H * sizeof(float));
        }
        if (flags & kLoraFlagFfnGateUp) {
            if (!cfg_.loraTargetFfnGateUp) {
                throw std::runtime_error("loadLoRA: adapter has FFN gate/up target but this run disabled it");
            }
            in.read((char*)loras_[L].Afg.data(), R * H * sizeof(float));
            in.read((char*)loras_[L].Bfg.data(), R * shape_.ffnSize * sizeof(float));
            if (fileHasDoRA) in.read((char*)loras_[L].Mfg.data(), shape_.ffnSize * sizeof(float));
            in.read((char*)loras_[L].Afu.data(), R * H * sizeof(float));
            in.read((char*)loras_[L].Bfu.data(), R * shape_.ffnSize * sizeof(float));
            if (fileHasDoRA) in.read((char*)loras_[L].Mfu.data(), shape_.ffnSize * sizeof(float));
        }
        if (flags & kLoraFlagLayerRmsNorm) {
            if (!cfg_.trainLayerRmsNorm) {
                throw std::runtime_error("loadLoRA: adapter has layer RMSNorm tuning but this run disabled it");
            }
            in.read((char*)layers_[L].rms_attn.data(), H * sizeof(float));
            in.read((char*)layers_[L].rms_ffn.data(), H * sizeof(float));
        }
    }
    if (flags & kLoraFlagLmHead) {
        if (!cfg_.loraTargetLmHead) {
            throw std::runtime_error("loadLoRA: adapter has LM-head target but this run disabled it");
        }
        in.read((char*)lmHeadLora_.A.data(), R * H * sizeof(float));
        in.read((char*)lmHeadLora_.B.data(), R * shape_.vocabSize * sizeof(float));
        if (fileHasDoRA) in.read((char*)lmHeadLora_.M.data(), shape_.vocabSize * sizeof(float));
    }
    if (flags & kLoraFlagLmHeadBias) {
        if (!cfg_.trainLmHeadBias) {
            throw std::runtime_error("loadLoRA: adapter has LM-head bias but this run disabled it");
        }
        in.read((char*)lmHeadBias_.data(), shape_.vocabSize * sizeof(float));
    }
    if (flags & kLoraFlagFinalNormAffine) {
        if (!cfg_.trainFinalNormAffine) {
            throw std::runtime_error("loadLoRA: adapter has final norm affine but this run disabled it");
        }
        in.read((char*)finalNormScale_.data(), H * sizeof(float));
        in.read((char*)finalNormBias_.data(), H * sizeof(float));
    }
    if (flags & kLoraFlagFinalRmsNorm) {
        if (!cfg_.trainFinalRmsNorm) {
            throw std::runtime_error("loadLoRA: adapter has final RMSNorm tuning but this run disabled it");
        }
        in.read((char*)rms_final_.data(), H * sizeof(float));
    }
    if (cfg_.useDoRA) {
        ++doraParamVersion_;
    }
}

std::vector<ParamRef> CpuKernelBackend::collectTrainableParams()
{
    if (loras_.size() != loraGrads_.size()) {
        throw std::runtime_error("collectTrainableParams: loras_ and loraGrads_ size mismatch");
    }

    std::vector<ParamRef> out;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);
    const std::size_t H = shape_.hiddenSize;
    const std::size_t F = shape_.ffnSize;
    const std::size_t kvDim = shape_.kvDim();
    const std::size_t activeTargets =
        (cfg_.loraTargetQ ? 1u : 0u) +
        (cfg_.loraTargetK ? 1u : 0u) +
        (cfg_.loraTargetV ? 1u : 0u) +
        (cfg_.loraTargetO ? 1u : 0u) +
        (cfg_.loraTargetFfnDown ? 1u : 0u) +
        (cfg_.loraTargetFfnGateUp ? 2u : 0u);
    out.reserve(
        loras_.size() * activeTargets * (cfg_.useDoRA ? 3u : 2u) +
        loras_.size() * (cfg_.trainLayerRmsNorm ? 2u : 0u) +
        (cfg_.loraTargetLmHead ? (cfg_.useDoRA ? 3u : 2u) : 0u) +
        (cfg_.trainLmHeadBias ? 1u : 0u) +
        (cfg_.trainFinalNormAffine ? 2u : 0u) +
        (cfg_.trainFinalRmsNorm ? 1u : 0u));

    for (std::size_t l = 0; l < loras_.size(); ++l) {
        auto& P = loras_[l];
        auto& G = loraGrads_[l];

        if (P.Aq.size() != G.gradAq.size()) throw std::runtime_error("Aq grad size mismatch");
        if (P.Bq.size() != G.gradBq.size()) throw std::runtime_error("Bq grad size mismatch");
        if (P.Ak.size() != G.gradAk.size()) throw std::runtime_error("Ak grad size mismatch");
        if (P.Bk.size() != G.gradBk.size()) throw std::runtime_error("Bk grad size mismatch");
        if (P.Av.size() != G.gradAv.size()) throw std::runtime_error("Av grad size mismatch");
        if (P.Bv.size() != G.gradBv.size()) throw std::runtime_error("Bv grad size mismatch");
        if (P.Ao.size() != G.gradAo.size()) throw std::runtime_error("Ao grad size mismatch");
        if (P.Bo.size() != G.gradBo.size()) throw std::runtime_error("Bo grad size mismatch");
        if (P.Afd.size() != G.gradAfd.size()) throw std::runtime_error("Afd grad size mismatch");
        if (P.Bfd.size() != G.gradBfd.size()) throw std::runtime_error("Bfd grad size mismatch");
        if (P.Afg.size() != G.gradAfg.size()) throw std::runtime_error("Afg grad size mismatch");
        if (P.Bfg.size() != G.gradBfg.size()) throw std::runtime_error("Bfg grad size mismatch");
        if (P.Afu.size() != G.gradAfu.size()) throw std::runtime_error("Afu grad size mismatch");
        if (P.Bfu.size() != G.gradBfu.size()) throw std::runtime_error("Bfu grad size mismatch");
        if (cfg_.useDoRA) {
            if (cfg_.loraTargetQ && P.Mq.size() != G.gradMq.size()) throw std::runtime_error("Mq grad size mismatch");
            if (cfg_.loraTargetK && P.Mk.size() != G.gradMk.size()) throw std::runtime_error("Mk grad size mismatch");
            if (cfg_.loraTargetV && P.Mv.size() != G.gradMv.size()) throw std::runtime_error("Mv grad size mismatch");
            if (cfg_.loraTargetO && P.Mo.size() != G.gradMo.size()) throw std::runtime_error("Mo grad size mismatch");
            if (cfg_.loraTargetFfnDown && P.Mfd.size() != G.gradMfd.size()) throw std::runtime_error("Mfd grad size mismatch");
            if (cfg_.loraTargetFfnGateUp && P.Mfg.size() != G.gradMfg.size()) throw std::runtime_error("Mfg grad size mismatch");
            if (cfg_.loraTargetFfnGateUp && P.Mfu.size() != G.gradMfu.size()) throw std::runtime_error("Mfu grad size mismatch");
        }
        if (cfg_.trainLayerRmsNorm) {
            if (G.gradRmsAttn.size() != layers_[l].rms_attn.size()) throw std::runtime_error("rms attn grad size mismatch");
            if (G.gradRmsFfn.size() != layers_[l].rms_ffn.size()) throw std::runtime_error("rms ffn grad size mismatch");
        }

        if (cfg_.loraTargetQ) {
            out.push_back({ P.Aq.data(), G.gradAq.data(), P.Aq.size(), "Aq", cfg_.loraAttnALrScale, R, H });
            out.push_back({ P.Bq.data(), G.gradBq.data(), P.Bq.size(), "Bq", cfg_.loraAttnBLrScale, R, H });
            if (cfg_.useDoRA) out.push_back({ P.Mq.data(), G.gradMq.data(), P.Mq.size(), "Mq", cfg_.loraAttnBLrScale });
        }
        if (cfg_.loraTargetK) {
            out.push_back({ P.Ak.data(), G.gradAk.data(), P.Ak.size(), "Ak", cfg_.loraKLrScale * cfg_.loraAttnALrScale, R, H });
            out.push_back({ P.Bk.data(), G.gradBk.data(), P.Bk.size(), "Bk", cfg_.loraKLrScale * cfg_.loraAttnBLrScale, R, kvDim });
            if (cfg_.useDoRA) out.push_back({ P.Mk.data(), G.gradMk.data(), P.Mk.size(), "Mk", cfg_.loraKLrScale * cfg_.loraAttnBLrScale });
        }
        if (cfg_.loraTargetV) {
            out.push_back({ P.Av.data(), G.gradAv.data(), P.Av.size(), "Av", cfg_.loraAttnALrScale, R, H });
            out.push_back({ P.Bv.data(), G.gradBv.data(), P.Bv.size(), "Bv", cfg_.loraAttnBLrScale, R, kvDim });
            if (cfg_.useDoRA) out.push_back({ P.Mv.data(), G.gradMv.data(), P.Mv.size(), "Mv", cfg_.loraAttnBLrScale });
        }
        if (cfg_.loraTargetO) {
            out.push_back({ P.Ao.data(), G.gradAo.data(), P.Ao.size(), "Ao", cfg_.loraOLrScale * cfg_.loraAttnALrScale, R, H });
            out.push_back({ P.Bo.data(), G.gradBo.data(), P.Bo.size(), "Bo", cfg_.loraOLrScale * cfg_.loraAttnBLrScale, R, H });
            if (cfg_.useDoRA) out.push_back({ P.Mo.data(), G.gradMo.data(), P.Mo.size(), "Mo", cfg_.loraOLrScale * cfg_.loraAttnBLrScale });
        }
        const bool ffnLoraEnabled = ffnLoraLayerEnabled(l);
        if (cfg_.loraTargetFfnDown && ffnLoraEnabled) {
            out.push_back({ P.Afd.data(), G.gradAfd.data(), P.Afd.size(), "Afd", cfg_.loraFfnDownLrScale, R, F });
            out.push_back({ P.Bfd.data(), G.gradBfd.data(), P.Bfd.size(), "Bfd", cfg_.loraFfnDownLrScale, R, H });
            if (cfg_.useDoRA) out.push_back({ P.Mfd.data(), G.gradMfd.data(), P.Mfd.size(), "Mfd", cfg_.loraFfnDownLrScale });
        }
        if (cfg_.loraTargetFfnGateUp && ffnLoraEnabled) {
            out.push_back({ P.Afg.data(), G.gradAfg.data(), P.Afg.size(), "Afg", cfg_.loraFfnGateUpLrScale, R, H });
            out.push_back({ P.Bfg.data(), G.gradBfg.data(), P.Bfg.size(), "Bfg", cfg_.loraFfnGateUpLrScale, R, F });
            if (cfg_.useDoRA) out.push_back({ P.Mfg.data(), G.gradMfg.data(), P.Mfg.size(), "Mfg", cfg_.loraFfnGateUpLrScale });
            out.push_back({ P.Afu.data(), G.gradAfu.data(), P.Afu.size(), "Afu", cfg_.loraFfnGateUpLrScale, R, H });
            out.push_back({ P.Bfu.data(), G.gradBfu.data(), P.Bfu.size(), "Bfu", cfg_.loraFfnGateUpLrScale, R, F });
            if (cfg_.useDoRA) out.push_back({ P.Mfu.data(), G.gradMfu.data(), P.Mfu.size(), "Mfu", cfg_.loraFfnGateUpLrScale });
        }
        if (cfg_.trainLayerRmsNorm) {
            out.push_back({ layers_[l].rms_attn.data(), G.gradRmsAttn.data(), layers_[l].rms_attn.size(), "RmsAttn", cfg_.layerRmsNormLrScale });
            out.push_back({ layers_[l].rms_ffn.data(), G.gradRmsFfn.data(), layers_[l].rms_ffn.size(), "RmsFfn", cfg_.layerRmsNormLrScale });
        }
    }
    if (cfg_.loraTargetLmHead) {
        if (lmHeadLora_.A.size() != lmHeadLoraGrads_.gradA.size()) throw std::runtime_error("lmHead A grad size mismatch");
        if (lmHeadLora_.B.size() != lmHeadLoraGrads_.gradB.size()) throw std::runtime_error("lmHead B grad size mismatch");
        if (cfg_.useDoRA && lmHeadLora_.M.size() != lmHeadLoraGrads_.gradM.size()) throw std::runtime_error("lmHead M grad size mismatch");
        out.push_back({ lmHeadLora_.A.data(), lmHeadLoraGrads_.gradA.data(), lmHeadLora_.A.size(), "Alm", cfg_.loraLmHeadALrScale, R, H });
        out.push_back({ lmHeadLora_.B.data(), lmHeadLoraGrads_.gradB.data(), lmHeadLora_.B.size(), "Blm", cfg_.loraLmHeadBLrScale, R, shape_.vocabSize });
        if (cfg_.useDoRA) out.push_back({ lmHeadLora_.M.data(), lmHeadLoraGrads_.gradM.data(), lmHeadLora_.M.size(), "Mlm", cfg_.loraLmHeadBLrScale });
    }
    if (cfg_.trainLmHeadBias) {
        if (lmHeadBias_.size() != lmHeadBiasGrad_.size()) throw std::runtime_error("lmHead bias grad size mismatch");
        out.push_back({ lmHeadBias_.data(), lmHeadBiasGrad_.data(), lmHeadBias_.size(), "BiasLm", cfg_.lmHeadBiasLrScale });
    }
    if (cfg_.trainFinalNormAffine) {
        if (finalNormScale_.size() != finalNormScaleGrad_.size()) throw std::runtime_error("final norm scale grad size mismatch");
        if (finalNormBias_.size() != finalNormBiasGrad_.size()) throw std::runtime_error("final norm bias grad size mismatch");
        out.push_back({ finalNormScale_.data(), finalNormScaleGrad_.data(), finalNormScale_.size(), "FinalNormScale", cfg_.finalNormAffineLrScale });
        out.push_back({ finalNormBias_.data(), finalNormBiasGrad_.data(), finalNormBias_.size(), "FinalNormBias", cfg_.finalNormAffineLrScale });
    }
    if (cfg_.trainFinalRmsNorm) {
        if (rms_final_.size() != rmsFinalGrad_.size()) throw std::runtime_error("final rms grad size mismatch");
        out.push_back({ rms_final_.data(), rmsFinalGrad_.data(), rms_final_.size(), "RmsFinal", cfg_.finalRmsNormLrScale });
    }

    return out;
}
