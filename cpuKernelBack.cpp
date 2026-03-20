#include "cpuKernelBack.h"

#include <algorithm>
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

#include <nlohmann/json.hpp>

#include "attention.h"
#include "bitnet_i2s_matmul.h"
#include "bitnet_tl2_generated.h"
#include "bitnet_tl2_matmul.h"
#include "bitnet_quant.h"
#include "bitnet_weight.h"
#include "kernels.h"
#include "paramRef.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

using ProfileClock = std::chrono::steady_clock;

struct BackendProfileState {
    double fwdEmbedMs = 0.0;
    double fwdAttnNormMs = 0.0;
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
    double bwdWoMs = 0.0;
    double bwdAttnMs = 0.0;
    double bwdRopeMs = 0.0;
    double bwdLoraMs = 0.0;
    double bwdQkvMs = 0.0;
    double bwdResidualMs = 0.0;

    int profiledBackwardCalls = 0;
};

bool backendProfileEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("QLORA_PROFILE_BACKEND");
        if (!value || *value == '\0') return false;
        std::string s(value);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s == "1" || s == "true" || s == "yes" || s == "on";
    }();
    return enabled;
}

bool fastSiluForwardEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("QLORA_TEST_FAST_SILU_FWD");
        if (!value || *value == '\0') return false;
        std::string s(value);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s == "1" || s == "true" || s == "yes" || s == "on";
    }();
    return enabled;
}

bool fastSiluBackwardEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("QLORA_TEST_FAST_SILU_BWD");
        if (!value || *value == '\0') return false;
        std::string s(value);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s == "1" || s == "true" || s == "yes" || s == "on";
    }();
    return enabled;
}

bool ffnGemmEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("QLORA_TEST_FFN_GEMM");
        if (!value || *value == '\0') return false;
        std::string s(value);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s == "1" || s == "true" || s == "yes" || s == "on";
    }();
    return enabled;
}

bool ffnBitnetEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("QLORA_TEST_BITNET_FFN");
        if (!value || *value == '\0') return true;
        std::string s(value);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s == "1" || s == "true" || s == "yes" || s == "on";
    }();
    return enabled;
}

bool envBoolOr(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (!value || *value == '\0') return fallback;
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    return fallback;
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
        const uint32_t version = 1u;
        const uint64_t rows = static_cast<uint64_t>(W.rows);
        const uint64_t cols = static_cast<uint64_t>(W.cols);
        const uint64_t qk = static_cast<uint64_t>(W.qk);
        const uint64_t jq = static_cast<uint64_t>(W.jq);
        const uint64_t q2N = static_cast<uint64_t>(W.q2.size());
        const uint64_t q2TN = static_cast<uint64_t>(W.q2T.size());
        const uint64_t sN = static_cast<uint64_t>(W.w_scale_tile.size());
        const uint64_t sTN = static_cast<uint64_t>(W.w_scale_tileT.size());

        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        out.write(reinterpret_cast<const char*>(&qk), sizeof(qk));
        out.write(reinterpret_cast<const char*>(&jq), sizeof(jq));
        out.write(reinterpret_cast<const char*>(&q2N), sizeof(q2N));
        out.write(reinterpret_cast<const char*>(&q2TN), sizeof(q2TN));
        out.write(reinterpret_cast<const char*>(&sN), sizeof(sN));
        out.write(reinterpret_cast<const char*>(&sTN), sizeof(sTN));

        if (q2N) out.write(reinterpret_cast<const char*>(W.q2.data()), static_cast<std::streamsize>(q2N));
        if (q2TN) out.write(reinterpret_cast<const char*>(W.q2T.data()), static_cast<std::streamsize>(q2TN));
        if (sN) out.write(reinterpret_cast<const char*>(W.w_scale_tile.data()), static_cast<std::streamsize>(sN * sizeof(float)));
        if (sTN) out.write(reinterpret_cast<const char*>(W.w_scale_tileT.data()), static_cast<std::streamsize>(sTN * sizeof(float)));

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
        uint64_t q2N = 0, q2TN = 0, sN = 0, sTN = 0;

        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        in.read(reinterpret_cast<char*>(&qk), sizeof(qk));
        in.read(reinterpret_cast<char*>(&jq), sizeof(jq));
        in.read(reinterpret_cast<char*>(&q2N), sizeof(q2N));
        in.read(reinterpret_cast<char*>(&q2TN), sizeof(q2TN));
        in.read(reinterpret_cast<char*>(&sN), sizeof(sN));
        in.read(reinterpret_cast<char*>(&sTN), sizeof(sTN));
        if (!in) return false;

        if (magic != 0x31533249u || version != 1u) return false;
        if (rows != expectedRows || cols != expectedCols || qk != expectedQk || jq != expectedJq) return false;

        W.rows = static_cast<std::size_t>(rows);
        W.cols = static_cast<std::size_t>(cols);
        W.hadamard32 = expectedHadamard32;
        W.qk = static_cast<std::size_t>(qk);
        W.jq = static_cast<std::size_t>(jq);
        W.q2.resize(static_cast<std::size_t>(q2N));
        W.q2T.resize(static_cast<std::size_t>(q2TN));
        W.w_scale_tile.resize(static_cast<std::size_t>(sN));
        W.w_scale_tileT.resize(static_cast<std::size_t>(sTN));

        if (q2N) in.read(reinterpret_cast<char*>(W.q2.data()), static_cast<std::streamsize>(q2N));
        if (q2TN) in.read(reinterpret_cast<char*>(W.q2T.data()), static_cast<std::streamsize>(q2TN));
        if (sN) in.read(reinterpret_cast<char*>(W.w_scale_tile.data()), static_cast<std::streamsize>(sN * sizeof(float)));
        if (sTN) in.read(reinterpret_cast<char*>(W.w_scale_tileT.data()), static_cast<std::streamsize>(sTN * sizeof(float)));
        if (!in) return false;

        return W.validForShape();
    }
    catch (...) {
        return false;
    }
}

static constexpr const char* kI2SCacheRevisionTag = ".rev2";

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
        const float siluG = silu_scalar(gate[i]);
        dUp[i] = dAct[i] * siluG;
        dGate[i] = dAct[i] * up[i] * silu_derivative_scalar(gate[i]);
    }
}

static inline void apply_rope_forward_inplace(
    float* vec,
    std::size_t headDim,
    std::size_t ropeDim,
    std::size_t pos,
    float ropeTheta)
{
    if (!vec || headDim == 0 || ropeDim < 2) return;
    const std::size_t rd = std::min(headDim, ropeDim) & ~std::size_t(1);
    if (rd == 0) return;

    for (std::size_t i = 0; i < rd; i += 2) {
        const float invFreq = std::pow(ropeTheta, -static_cast<float>(i) / static_cast<float>(rd));
        const float angle = static_cast<float>(pos) * invFreq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);

        const float x0 = vec[i + 0];
        const float x1 = vec[i + 1];
        vec[i + 0] = x0 * c - x1 * s;
        vec[i + 1] = x0 * s + x1 * c;
    }
}

static inline void apply_rope_backward_inplace(
    float* gradVec,
    std::size_t headDim,
    std::size_t ropeDim,
    std::size_t pos,
    float ropeTheta)
{
    if (!gradVec || headDim == 0 || ropeDim < 2) return;
    const std::size_t rd = std::min(headDim, ropeDim) & ~std::size_t(1);
    if (rd == 0) return;

    for (std::size_t i = 0; i < rd; i += 2) {
        const float invFreq = std::pow(ropeTheta, -static_cast<float>(i) / static_cast<float>(rd));
        const float angle = static_cast<float>(pos) * invFreq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);

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

    embeddings_.resize(V * H);
    W_out_.resize(H * V);

    layers_.resize(nLoRALayers);
    loras_.resize(nLoRALayers);
    loraGrads_.resize(nLoRALayers);

    auto init_mat_HH = [&](std::vector<float>& W) { W.resize(H * H); };
    auto init_mat_HKV = [&](std::vector<float>& W) { W.resize(H * kvDim); };
    auto init_mat_HF = [&](std::vector<float>& W) { W.resize(H * F); };
    auto init_mat_FH = [&](std::vector<float>& W) { W.resize(F * H); };
    auto init_vec_H = [&](std::vector<float>& w) { w.resize(H); };

    for (std::size_t L = 0; L < nLoRALayers; ++L) {
        init_mat_HH(layers_[L].Wq);
        init_mat_HKV(layers_[L].Wk);
        init_mat_HKV(layers_[L].Wv);
        init_mat_HH(layers_[L].Wo);
        init_vec_H(layers_[L].rms_attn);
        init_vec_H(layers_[L].rms_ffn);
        init_mat_HF(layers_[L].Wgate);
        init_mat_HF(layers_[L].Wup);
        init_mat_FH(layers_[L].Wdown);

        loras_[L].Aq.resize(R * H);
        loras_[L].Bq.resize(R * H);
        loras_[L].Av.resize(R * H);
        loras_[L].Bv.resize(R * kvDim);

        loraGrads_[L].gradAq.assign(R * H, 0.0f);
        loraGrads_[L].gradBq.assign(R * H, 0.0f);
        loraGrads_[L].gradAv.assign(R * H, 0.0f);
        loraGrads_[L].gradBv.assign(R * kvDim, 0.0f);
    }

    tmpQ_cache_.resize(nLoRALayers * Tm * R);
    tmpV_cache_.resize(nLoRALayers * Tm * R);
    xnorm_cache_.resize(nLoRALayers * Tm * H);
    q_cache_.resize(nLoRALayers * Tm * H);
    k_cache_.resize(nLoRALayers * Tm * kvDim);
    v_cache_.resize(nLoRALayers * Tm * kvDim);
    probs_cache_.resize(nLoRALayers * nHeads * Tm * Tm);
    x_mid_cache_.resize(nLoRALayers * Tm * H);
    gate_cache_.resize(nLoRALayers * Tm * F);
    up_cache_.resize(nLoRALayers * Tm * F);

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

    fwdX8Scratch_.resize(tTileConst * maxFwdInputDim);
    fwdXScaleScratch_.resize(tTileConst * maxKBlocks);
    // bwd buffers must cover the widest output dimension (V) and its nJBlocks.
    bwdDY8Scratch_.resize(tTileConst * V);
    bwdDYScaleScratch_.resize(tTileConst * maxJBlocks);
    bwdDK8Scratch_.resize(tTileConst * maxAuxBwdDim);
    bwdDKScaleScratch_.resize(tTileConst * auxJBlocks);
    bwdDV8Scratch_.resize(tTileConst * maxAuxBwdDim);
    bwdDVScaleScratch_.resize(tTileConst * auxJBlocks);

    std::mt19937 rng(42);
    std::normal_distribution<float> wdist(0.0f, 0.02f);

    std::string weight_dir = cfg_.modelPath + "/weights";

    std::cout << "Loading real FP32 weights from disk...\n";
    std::cout << "Architecture: " << nLoRALayers << " Layers | H=" << H
        << " | kvDim=" << kvDim << " | ffnDim=" << F
        << " | ropeDim=" << shape_.ropeDim << "\n";

    load_raw_bin(weight_dir + "/embeddings.bin", embeddings_);
    load_raw_bin(weight_dir + "/W_out.bin", W_out_);
    load_raw_bin(weight_dir + "/rms_final.bin", rms_final_);

    // ── Load weights and quantize ─────────────────────────────────────────────
    // On a cold start (no cache files) all nLoRALayers × {Wq,Wk,Wv,Wo}
    // quantization jobs are parallelised with OpenMP so the one-time
    // startup cost scales with hardware thread count rather than being
    // purely serial.  Cache reads/writes inside the parallel region are
    // per-file and non-overlapping, so no additional synchronisation is needed.
    const bool canUseI2S = (H % 32 == 0);

    // ── Serial weight loading (disk I/O, must stay serial) ───────────────────
    for (std::size_t L = 0; L < nLoRALayers; ++L)
    {
        std::cout << "Loading layer " << (L + 1) << "/" << nLoRALayers << "...\n";
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wq.bin", layers_[L].Wq);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wk.bin", layers_[L].Wk);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wv.bin", layers_[L].Wv);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wo.bin", layers_[L].Wo);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_rms_attn.bin", layers_[L].rms_attn);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_rms_ffn.bin", layers_[L].rms_ffn);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wgate.bin", layers_[L].Wgate);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wup.bin", layers_[L].Wup);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wdown.bin", layers_[L].Wdown);
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
    if (cfg_.useBitNetQuant && canUseI2S) {
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
                const std::string qPath = weight_dir + "/layer_" + std::to_string(L) + "_Wq" + cacheTag;
                const std::string kPath = weight_dir + "/layer_" + std::to_string(L) + "_Wk" + cacheTag;
                const std::string vPath = weight_dir + "/layer_" + std::to_string(L) + "_Wv" + cacheTag;

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
                std::vector<float>().swap(layers_[L].Wq);
                std::vector<float>().swap(layers_[L].Wk);
                std::vector<float>().swap(layers_[L].Wv);
            }

            if (cfg_.bitnetQuantizeWo) {
                const std::string oPath = weight_dir + "/layer_" + std::to_string(L) + "_Wo" + cacheTag;
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
                std::vector<float>().swap(layers_[L].Wo);
            }

            if (useBitNetFfn) {
                const std::string gatePath = weight_dir + "/layer_" + std::to_string(L) + "_Wgate" + ffnCacheTag;
                const std::string upPath = weight_dir + "/layer_" + std::to_string(L) + "_Wup" + ffnCacheTag;
                const std::string downPath = weight_dir + "/layer_" + std::to_string(L) + "_Wdown" + ffnCacheTag;

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
                std::vector<float>().swap(layers_[L].Wgate);
                std::vector<float>().swap(layers_[L].Wup);
                std::vector<float>().swap(layers_[L].Wdown);
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
        for (auto& x : loras_[L].Av) x = wdist(rng) * 0.025f;
        std::fill(loras_[L].Bq.begin(), loras_[L].Bq.end(), 0.0f);
        std::fill(loras_[L].Bv.begin(), loras_[L].Bv.end(), 0.0f);
    }

    const bool useQuantLmHeadCache =
        cfg_.useBitNetQuant &&
        canUseI2S &&
        (cfg_.bitnetQuantizeLmHead || cfg_.bitnetLmHeadQuantBackwardOnly);
    if (useQuantLmHeadCache) {
        const std::size_t qk = cfg_.bitnetQK;
        const std::size_t jq = cfg_.bitnetJQ;
        const std::string rotTag = cfg_.bitnetHadamard32 ? ".h32" : ".h0";
        const std::string lmHeadCache =
            weight_dir + "/W_out.i2s.qk" + std::to_string(qk) +
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
        if (cfg_.bitnetQuantizeLmHead && !cfg_.bitnetLmHeadFloatBackward) {
            // Quantized LM-head forward/backward can release the FP32 weights.
            std::vector<float>().swap(W_out_);
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
}

// ==============================================================================
// FORWARD
// ==============================================================================
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
    const float ropeTheta = shape_.ropeTheta;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);
    const std::size_t kvDim = shape_.kvDim(); // from ModelShape, not fragile Wk.size()/H
    const std::size_t nKvHeads = shape_.nKvHeads;
    const std::size_t nHeads = shape_.nHeads;
    const std::size_t tTile = static_cast<std::size_t>(cfg_.tTile);
    const std::size_t hTile = static_cast<std::size_t>(cfg_.hTile);
    const float       scale = cfg_.loraAlpha / float(R);
    const std::size_t CKPT_STRIDE = cfg_.maxSeqLen * H;
    const bool profile = backendProfileEnabled();
    const bool useFfnGemm = ffnGemmEnabled();
    const bool useBitNetFfn = cfg_.useBitNetQuant && ffnBitnetEnabled();
    auto& prof = backendProfileState();

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

        for (std::size_t t = 0; t < T; ++t) {
            float* xn = &xnorm_[t * H];
            const float* x = &x_[t * H];
            if (attentionMask[t] == 0) { std::fill(xn, xn + H, 0.0f); continue; }
            rmsnorm(xn, x, layers_[L].rms_attn.data(), H, eps);
        }
        std::copy_n(xnorm_.data(), T * H, xnorm_cache_.data() + layerTHOffset);

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
        const bool useFusedI2SKV = useI2SK && useI2SV && !useFusedI2SQKV;
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

        // ── QKV Projection ──
        if (useTl2GenQ && useTl2GenK && useTl2GenV) {
            std::fill(q_.begin(), q_.begin() + (T * H), 0.0f);
            std::fill(k_.begin(), k_.begin() + (T * kvDim), 0.0f);
            std::fill(v_.begin(), v_.begin() + (T * kvDim), 0.0f);

            linear_forward_bitnet_tl2_generated(q_.data(), xnorm_.data(), layers_[L].Wq_tl2g, T);
            linear_forward_bitnet_tl2_generated(k_.data(), xnorm_.data(), layers_[L].Wk_tl2g, T);
            linear_forward_bitnet_tl2_generated(v_.data(), xnorm_.data(), layers_[L].Wv_tl2g, T);
        }
        else if (useTl2Q && useTl2K && useTl2V) {
            std::fill(q_.begin(), q_.begin() + (T * H), 0.0f);
            std::fill(k_.begin(), k_.begin() + (T * kvDim), 0.0f);
            std::fill(v_.begin(), v_.begin() + (T * kvDim), 0.0f);

            linear_fused_qkv_forward_bitnet_tl2_accum(
                q_.data(), k_.data(), v_.data(),
                xnorm_.data(),
                layers_[L].Wq_tl2, layers_[L].Wk_tl2, layers_[L].Wv_tl2,
                T, tTile,
                fwdX8Scratch_.data(), fwdXScaleScratch_.data());
        }
        else if (useFusedI2SQKV) {
            std::fill(q_.begin(), q_.begin() + (T * H), 0.0f);
            std::fill(k_.begin(), k_.begin() + (T * kvDim), 0.0f);
            std::fill(v_.begin(), v_.begin() + (T * kvDim), 0.0f);

            linear_fused_qkv_forward_bitnet_i2s_accum(
                q_.data(), k_.data(), v_.data(),
                xnorm_.data(),
                layers_[L].Wq_q, layers_[L].Wk_q, layers_[L].Wv_q,
                T, tTile,
                fwdX8Scratch_.data(), fwdXScaleScratch_.data());
        }
        else if (useI2SQ && useI2SK && useI2SV) {
            std::fill(q_.begin(), q_.begin() + (T * H), 0.0f);
            std::fill(k_.begin(), k_.begin() + (T * kvDim), 0.0f);
            std::fill(v_.begin(), v_.begin() + (T * kvDim), 0.0f);

            linear_forward_bitnet_i2s_accum(q_.data(), xnorm_.data(), layers_[L].Wq_q, T, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            linear_forward_bitnet_i2s_accum(k_.data(), xnorm_.data(), layers_[L].Wk_q, T, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            linear_forward_bitnet_i2s_accum(v_.data(), xnorm_.data(), layers_[L].Wv_q, T, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data());
        }
        else {
            std::fill(q_.begin(), q_.begin() + (T * H), 0.0f);
            std::fill(k_.begin(), k_.begin() + (T * kvDim), 0.0f);
            std::fill(v_.begin(), v_.begin() + (T * kvDim), 0.0f);

            if (useTl2GenQ) {
                linear_forward_bitnet_tl2_generated(q_.data(), xnorm_.data(), layers_[L].Wq_tl2g, T);
            }
            else if (useTl2Q) {
                linear_forward_bitnet_tl2_accum(
                    q_.data(), xnorm_.data(), layers_[L].Wq_tl2, T, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            }
            else if (useI2SQ) {
                linear_forward_bitnet_i2s_accum(
                    q_.data(), xnorm_.data(), layers_[L].Wq_q, T, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            }
            else {
                for (std::size_t t = 0; t < T; ++t) {
                    if (attentionMask[t] == 0) continue;
                    const float* xn = &xnorm_[t * H];
                    float* qt = &q_[t * H];
                    for (std::size_t i = 0; i < H; ++i) {
                        float sum_q = 0; for (std::size_t j = 0; j < H; ++j) sum_q += xn[j] * layers_[L].Wq[j * H + i]; qt[i] = sum_q;
                    }
                }
            }

            if (useFusedI2SKV) {
                linear_fused_dual_forward_bitnet_i2s_accum(
                    k_.data(), v_.data(),
                    xnorm_.data(),
                    layers_[L].Wk_q, layers_[L].Wv_q,
                    T, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            }
            else {
                if (useTl2GenK) {
                    linear_forward_bitnet_tl2_generated(k_.data(), xnorm_.data(), layers_[L].Wk_tl2g, T);
                }
                else if (useTl2K) {
                    linear_forward_bitnet_tl2_accum(
                        k_.data(), xnorm_.data(), layers_[L].Wk_tl2, T, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
                else if (useI2SK) {
                    linear_forward_bitnet_i2s_accum(
                        k_.data(), xnorm_.data(), layers_[L].Wk_q, T, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
                else {
                    for (std::size_t t = 0; t < T; ++t) {
                        if (attentionMask[t] == 0) continue;
                        const float* xn = &xnorm_[t * H];
                        float* kt = &k_[t * kvDim];
                        for (std::size_t i = 0; i < kvDim; ++i) {
                            float sum_k = 0; for (std::size_t j = 0; j < H; ++j) sum_k += xn[j] * layers_[L].Wk[j * kvDim + i]; kt[i] = sum_k;
                        }
                    }
                }

                if (useTl2GenV) {
                    linear_forward_bitnet_tl2_generated(v_.data(), xnorm_.data(), layers_[L].Wv_tl2g, T);
                }
                else if (useTl2V) {
                    linear_forward_bitnet_tl2_accum(
                        v_.data(), xnorm_.data(), layers_[L].Wv_tl2, T, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
                else if (useI2SV) {
                    linear_forward_bitnet_i2s_accum(
                        v_.data(), xnorm_.data(), layers_[L].Wv_q, T, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
                else {
                    for (std::size_t t = 0; t < T; ++t) {
                        if (attentionMask[t] == 0) continue;
                        const float* xn = &xnorm_[t * H];
                        float* vt = &v_[t * kvDim];
                        for (std::size_t i = 0; i < kvDim; ++i) {
                            float sum_v = 0; for (std::size_t j = 0; j < H; ++j) sum_v += xn[j] * layers_[L].Wv[j * kvDim + i]; vt[i] = sum_v;
                        }
                    }
                }
            }
        }

        // ── LoRA Injection ──
        for (std::size_t t = 0; t < T; ++t) {
            float* tmpQ = &tmpQ_cache_[(L * cfg_.maxSeqLen + t) * R];
            float* tmpV = &tmpV_cache_[(L * cfg_.maxSeqLen + t) * R];

            if (attentionMask[t] == 0) {
                std::fill(tmpQ, tmpQ + R, 0.0f);
                std::fill(tmpV, tmpV + R, 0.0f);
                continue;
            }

            lora_add_H(&q_[t * H], &xnorm_[t * H], loras_[L].Aq.data(), loras_[L].Bq.data(), tmpQ, H, R, scale, hTile);

            float* v_t = &v_[t * kvDim];
            const float* xn = &xnorm_[t * H];

            for (std::size_t r = 0; r < R; ++r) {
                const float* aRow = &loras_[L].Av[r * H];
                tmpV[r] = dot_avx2_or_scalar(xn, aRow, H);
            }
            for (std::size_t r = 0; r < R; ++r) {
                const float* bRow = &loras_[L].Bv[r * kvDim];
                axpy_inplace_avx2_or_scalar(v_t, bRow, kvDim, scale * tmpV[r]);
            }
        }

        // ── RoPE on Q/K (pre-attention) ──
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;

            for (std::size_t h = 0; h < nHeads; ++h) {
                apply_rope_forward_inplace(&q_[t * H + h * D], D, ropeDim, t, ropeTheta);
            }
            for (std::size_t kvh = 0; kvh < nKvHeads; ++kvh) {
                apply_rope_forward_inplace(&k_[t * kvDim + kvh * D], D, ropeDim, t, ropeTheta);
            }
        }
        std::copy_n(q_.data(), T * H, q_cache_.data() + layerTHOffset);
        std::copy_n(k_.data(), T * kvDim, k_cache_.data() + layerTKVOffset);
        std::copy_n(v_.data(), T * kvDim, v_cache_.data() + layerTKVOffset);
        if (profile) prof.fwdAttnNormMs += elapsedMs(preAttnStart);

        // ── Attention ──
        const auto attnCoreStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        attn_causal(
            ctx_.data(), q_.data(), k_.data(), v_.data(),
            T, H, shape_.nHeads, nKvHeads, scores_.data(), probs_.data(),
            attentionMask.data()
        );
        std::copy_n(probs_.data(), nHeads * T * T, probs_cache_.data() + layerProbsOffset);

        // ── Output Projection (Wo) ──
        if (useTl2GenWo) {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);
            linear_forward_bitnet_tl2_generated(attn_.data(), ctx_.data(), layers_[L].Wo_tl2g, T);
        }
        else if (useTl2Wo) {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);
            linear_forward_bitnet_tl2_accum(
                attn_.data(), ctx_.data(), layers_[L].Wo_tl2, T, tTile,
                fwdX8Scratch_.data(), fwdXScaleScratch_.data());
        }
        else if (useI2SWo) {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);
            const std::size_t nXBlocks = layers_[L].Wo_q.nKBlocks();
            (void)nXBlocks;
            linear_forward_bitnet_i2s_accum(attn_.data(), ctx_.data(), layers_[L].Wo_q, T, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data());
        }
        else {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                const float* ct = &ctx_[t * H]; float* at = &attn_[t * H];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum = 0; for (std::size_t j = 0; j < H; ++j) sum += ct[j] * layers_[L].Wo[j * H + i]; at[i] = sum;
                }
            }
        }

        // First residual: x_mid = x + attn
        for (std::size_t t = 0; t < T; ++t) {
            float* xMid = &x_mid_[t * H];
            const float* xIn = &x_[t * H];
            const float* at = &attn_[t * H];
            if (attentionMask[t] == 0) {
                std::fill(xMid, xMid + H, 0.0f);
                continue;
            }
            for (std::size_t h = 0; h < H; ++h) xMid[h] = xIn[h] + at[h];
        }
        std::copy_n(x_mid_.data(), T * H, x_mid_cache_.data() + layerTHOffset);
        if (profile) prof.fwdAttnMs += elapsedMs(attnCoreStart);

        // FFN pre-norm.
        const auto ffnStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        for (std::size_t t = 0; t < T; ++t) {
            float* xn = &xnorm_ffn_[t * H];
            if (attentionMask[t] == 0) {
                std::fill(xn, xn + H, 0.0f);
                continue;
            }
            rmsnorm(xn, &x_mid_[t * H], layers_[L].rms_ffn.data(), H, eps);
        }

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
            std::fill(gate_.begin(), gate_.begin() + (T * F), 0.0f);
            std::fill(up_.begin(), up_.begin() + (T * F), 0.0f);
            std::fill(ffn_out_.begin(), ffn_out_.begin() + (T * H), 0.0f);

            if (useFusedI2SGateUp) {
                linear_fused_dual_forward_bitnet_i2s_accum(
                    gate_.data(), up_.data(),
                    xnorm_ffn_.data(),
                    layers_[L].Wgate_q, layers_[L].Wup_q,
                    T, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            }
            else {
                if (useTl2GenGate) {
                    linear_forward_bitnet_tl2_generated(gate_.data(), xnorm_ffn_.data(), layers_[L].Wgate_tl2g, T);
                }
                else if (useTl2Gate) {
                    linear_forward_bitnet_tl2_accum(
                        gate_.data(), xnorm_ffn_.data(), layers_[L].Wgate_tl2, T, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
                else if (useI2SGate) {
                    linear_forward_bitnet_i2s_accum(
                        gate_.data(), xnorm_ffn_.data(), layers_[L].Wgate_q, T, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }

                if (useTl2GenUp) {
                    linear_forward_bitnet_tl2_generated(up_.data(), xnorm_ffn_.data(), layers_[L].Wup_tl2g, T);
                }
                else if (useTl2Up) {
                    linear_forward_bitnet_tl2_accum(
                        up_.data(), xnorm_ffn_.data(), layers_[L].Wup_tl2, T, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
                else if (useI2SUp) {
                    linear_forward_bitnet_i2s_accum(
                        up_.data(), xnorm_ffn_.data(), layers_[L].Wup_q, T, tTile,
                        fwdX8Scratch_.data(), fwdXScaleScratch_.data());
                }
            }
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
            if (useTl2GenDown) {
                linear_forward_bitnet_tl2_generated(ffn_out_.data(), ffn_act_.data(), layers_[L].Wdown_tl2g, T);
            }
            else if (useTl2Down) {
                linear_forward_bitnet_tl2_accum(
                    ffn_out_.data(), ffn_act_.data(), layers_[L].Wdown_tl2, T, tTile,
                    fwdX8Scratch_.data(), fwdXScaleScratch_.data());
            }
            else if (useI2SDown) {
                linear_forward_bitnet_i2s_accum(ffn_out_.data(), ffn_act_.data(), layers_[L].Wdown_q, T, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data());
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

            linear_TFH_accum_blocked32(
                ffn_out_.data(),
                ffn_act_.data(),
                layers_[L].Wdown.data(),
                T, F, H, cfg_.hTile, tTile);
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

                        if (fastSiluForwardEnabled()) {
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
        std::copy_n(gate_.data(), T * F, gate_cache_.data() + layerTFOffset);
        std::copy_n(up_.data(), T * F, up_cache_.data() + layerTFOffset);

        // Second residual: x = x_mid + ffn_out
        for (std::size_t t = 0; t < T; ++t) {
            float* xOut = &x_[t * H];
            const float* xMid = &x_mid_[t * H];
            const float* ffn = &ffn_out_[t * H];
            if (attentionMask[t] == 0) {
                std::fill(xOut, xOut + H, 0.0f);
                continue;
            }
            for (std::size_t h = 0; h < H; ++h) xOut[h] = xMid[h] + ffn[h];
        }

        std::copy(x_.begin(), x_.begin() + (T * H), &x_ckpt_[(L + 1) * CKPT_STRIDE]);
        if (profile) prof.fwdFfnMs += elapsedMs(ffnStart);
    }

    const std::size_t expected = T * V;
    if (logitsOut.size() != expected) throw std::runtime_error("forward: logitsOut size mismatch");

    {
        const auto start = profile ? ProfileClock::now() : ProfileClock::time_point{};
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) { std::fill(&xnorm_final_[t * H], &xnorm_final_[t * H] + H, 0.0f); continue; }
            rmsnorm(&xnorm_final_[t * H], &x_[t * H], rms_final_.data(), H, eps);
        }
        if (profile) prof.fwdFinalNormMs += elapsedMs(start);
    }

    // The i2s kernel has no per-row mask — it always processes all T rows.
    // When a lossMask is active, zero out non-loss rows of xnorm_final_ so
    // the kernel produces zero logits for those positions, then restore those
    // rows after the call so cached activations remain intact.
    const bool useQuantLmForward = cfg_.useBitNetQuant && cfg_.bitnetQuantizeLmHead && !W_out_q.empty();
    const auto lmHeadStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
    if (useQuantLmForward) {
        std::fill(logitsOut.begin(), logitsOut.end(), 0.0f);

        // Zero non-contributing rows before the kernel.
        if (hasLossMask) {
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0 || lossMask[t] == 0) {
                    std::fill(&xnorm_final_[t * H], &xnorm_final_[t * H] + H, 0.0f);
                }
            }
        }
        linear_forward_bitnet_i2s_accum(logitsOut.data(), xnorm_final_.data(), W_out_q, T, tTile, fwdX8Scratch_.data(), fwdXScaleScratch_.data());

        // Restore masked supervised-excluded rows so downstream users of
        // xnorm_final_ (current or future) observe consistent activations.
        if (hasLossMask) {
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0 || lossMask[t] != 0) continue;
                rmsnorm(&xnorm_final_[t * H], &x_[t * H], rms_final_.data(), H, eps);
            }
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
#if defined(_OPENMP)
#pragma warning(suppress : 6993)
#pragma omp parallel for schedule(static)
#endif
        for (int rowBase_i = 0; rowBase_i < static_cast<int>(lmHeadRowsScratch_.size()); rowBase_i += 4) {
            const std::size_t rowBase = static_cast<std::size_t>(rowBase_i);
            const std::size_t rowCount = std::min<std::size_t>(4, lmHeadRowsScratch_.size() - rowBase);

            std::size_t rows[4] = { 0, 0, 0, 0 };
            const float* hidden[4] = { nullptr, nullptr, nullptr, nullptr };
            float* logits[4] = { nullptr, nullptr, nullptr, nullptr };
            for (std::size_t r = 0; r < rowCount; ++r) {
                rows[r] = lmHeadRowsScratch_[rowBase + r];
                hidden[r] = &xnorm_final_[rows[r] * H];
                logits[r] = &logitsOut[rows[r] * V];
                std::fill_n(logits[r], V, 0.0f);
            }

            for (std::size_t v0 = 0; v0 < V; v0 += vTile) {
                const std::size_t v1 = std::min(V, v0 + vTile);
                const std::size_t len = v1 - v0;

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
    const float ropeTheta = shape_.ropeTheta;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);
    const std::size_t kvDim = shape_.kvDim(); // from ModelShape, not fragile Wk.size()/H
    const std::size_t nKvHeads = shape_.nKvHeads;
    const std::size_t nHeads = shape_.nHeads;
    const std::size_t tTile = static_cast<std::size_t>(cfg_.tTile);
    const float       scale = cfg_.loraAlpha / float(R);
    const std::size_t nLayers = shape_.nLayers;
    const float       eps = 1e-5f;
    const std::size_t CKPT_STRIDE = cfg_.maxSeqLen * H;
    const bool profile = backendProfileEnabled();
    const bool useFfnGemm = ffnGemmEnabled();
    const bool useBitNetFfn = cfg_.useBitNetQuant && ffnBitnetEnabled();
    auto& prof = backendProfileState();

    // ── Backprop through LM head ──────────────────────────────────────────────
    const auto lmHeadStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
    std::fill_n(dXnorm_final_.data(), T * H, 0.0f);

    // dLogits rows for non-loss tokens are guaranteed zero by the trainer
    // (dLogitsBuf_ is zero-initialised and cross_entropy only writes loss rows),
    // so the i2s backward kernel is always safe to use regardless of lossMask.
    const bool useQuantLmBackward =
        cfg_.useBitNetQuant &&
        !W_out_q.empty() &&
        !cfg_.bitnetLmHeadFloatBackward &&
        (cfg_.bitnetQuantizeLmHead || cfg_.bitnetLmHeadQuantBackwardOnly);
    if (useQuantLmBackward) {
        linear_backward_bitnet_i2s_packed(dXnorm_final_.data(), dLogits.data(), W_out_q, T, tTile, bwdDY8Scratch_.data(), bwdDYScaleScratch_.data());
    }
    else {
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
            const float* dlog_t = &dLogits[t * V];
            float* dx_t = &dXnorm_final_[t * H];

            std::size_t h = 0;
            for (; h + 3 < H; h += 4) {
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
            if (h + 1 < H) {
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
            for (; h < H; ++h) {
                dx_t[h] = dot_avx2_or_scalar(dlog_t, &W_out_[h * V], V);
            }
        }
    }
    if (profile) prof.bwdLmHeadMs += elapsedMs(lmHeadStart);

    // ── Backprop through final RMSNorm ────────────────────────────────────────
    const auto finalNormStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
    std::fill_n(dX_.data(), T * H, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        if (attentionMask[t] == 0) continue;
        rmsnorm_backward(&dX_[t * H], &dXnorm_final_[t * H], &x_[t * H], rms_final_.data(), H, eps);
    }
    if (profile) prof.bwdFinalNormMs += elapsedMs(finalNormStart);

    // ── Layer loop (reverse) ─────────────────────────────────────────────────
    for (std::size_t Lrev = 0; Lrev < nLayers; ++Lrev) {
        const std::size_t L = nLayers - 1 - Lrev;
        const std::size_t thSize = T * H;
        const std::size_t tkvSize = T * kvDim;
        const std::size_t tfSize = T * F;

        // Clear all per-layer temporaries.
        const auto layerClearStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        std::fill_n(dAttn_.data(), thSize, 0.0f);
        std::fill_n(dCtx_.data(), thSize, 0.0f);
        std::fill_n(dQ_.data(), thSize, 0.0f);
        std::fill_n(dK_.data(), tkvSize, 0.0f);
        std::fill_n(dV_.data(), tkvSize, 0.0f);
        std::fill_n(dXnorm_.data(), thSize, 0.0f);
        std::fill_n(dXin_.data(), thSize, 0.0f);
        std::fill_n(dXmid_.data(), thSize, 0.0f);
        std::fill_n(dFfnOut_.data(), thSize, 0.0f);
        std::fill_n(dAct_.data(), tfSize, 0.0f);
        std::fill_n(dGate_.data(), tfSize, 0.0f);
        std::fill_n(dUp_.data(), tfSize, 0.0f);
        std::fill_n(dXnormFfn_.data(), thSize, 0.0f);
        std::fill_n(tmpdP_.data(), T, 0.0f);
        if (profile) prof.bwdClearMs += elapsedMs(layerClearStart);

        // Restore the pre-layer activation from the checkpoint.
        const auto layerRestoreStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        std::copy(&x_ckpt_[L * CKPT_STRIDE], &x_ckpt_[L * CKPT_STRIDE] + (T * H), x_.begin());

        const bool useI2SQ = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV;
        const bool useI2SK = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV;
        const bool useI2SV = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV;
        const bool useFusedI2SQKV = useI2SQ && useI2SK && useI2SV && cfg_.bitnetFuseQKV;
        const bool useFusedI2SKV = useI2SK && useI2SV && !useFusedI2SQKV;
        const bool useI2SWo = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeWo;
        (void)useI2SWo;

        const std::size_t layerTHOffset = L * cfg_.maxSeqLen * H;
        const std::size_t layerTFOffset = L * cfg_.maxSeqLen * F;
        const std::size_t layerTKVOffset = L * cfg_.maxSeqLen * kvDim;
        const std::size_t layerProbsOffset = L * nHeads * cfg_.maxSeqLen * cfg_.maxSeqLen;

        std::copy_n(xnorm_cache_.data() + layerTHOffset, T * H, xnorm_.data());
        std::copy_n(q_cache_.data() + layerTHOffset, T * H, q_.data());
        std::copy_n(k_cache_.data() + layerTKVOffset, T * kvDim, k_.data());
        std::copy_n(v_cache_.data() + layerTKVOffset, T * kvDim, v_.data());
        std::copy_n(probs_cache_.data() + layerProbsOffset, nHeads * T * T, probs_.data());
        std::copy_n(x_mid_cache_.data() + layerTHOffset, T * H, x_mid_.data());
        std::copy_n(gate_cache_.data() + layerTFOffset, T * F, gate_.data());
        std::copy_n(up_cache_.data() + layerTFOffset, T * F, up_.data());
        if (profile) prof.bwdRestoreMs += elapsedMs(layerRestoreStart);

        // Backprop through FFN second residual:
        // x_out = x_mid + down(silu(gate(xnorm_ffn)) * up(xnorm_ffn)).
        const auto ffnStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        for (std::size_t i = 0; i < T * H; ++i) {
            dXmid_[i] = dX_[i];
        }

        if (useBitNetFfn) {
            linear_backward_bitnet_i2s_packed(dAct_.data(), dX_.data(), layers_[L].Wdown_q, T, tTile, bwdDY8Scratch_.data(), bwdDYScaleScratch_.data());

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                float* dAct_t = &dAct_[t * F];
                float* dGate_t = &dGate_[t * F];
                float* dUp_t = &dUp_[t * F];
                const float* gate_t = &gate_[t * F];
                const float* up_t = &up_[t * F];

                if (attentionMask[t] == 0) {
                    std::fill(dAct_t, dAct_t + F, 0.0f);
                    std::fill(dGate_t, dGate_t + F, 0.0f);
                    std::fill(dUp_t, dUp_t + F, 0.0f);
                    continue;
                }
                if (fastSiluBackwardEnabled()) {
                    silu_backward_avx2_or_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                }
                else {
                    silu_backward_exact_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                }
            }

            std::fill(dXnormFfn_.begin(), dXnormFfn_.begin() + (T * H), 0.0f);
            linear_fused_dual_backward_bitnet_i2s_packed(
                dXnormFfn_.data(),
                dUp_.data(), dGate_.data(),
                layers_[L].Wup_q, layers_[L].Wgate_q,
                T, tTile,
                bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                bwdDK8Scratch_.data(), bwdDKScaleScratch_.data());

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                if (attentionMask[t] == 0) continue;
                rmsnorm_backward(
                    &dXin_[t * H],
                    &dXnormFfn_[t * H],
                    &x_mid_[t * H],
                    layers_[L].rms_ffn.data(),
                    H, eps
                );

                float* dXmid_t = &dXmid_[t * H];
                const float* dXmidFromNorm_t = &dXin_[t * H];
                for (std::size_t h = 0; h < H; ++h) dXmid_t[h] += dXmidFromNorm_t[h];
            }
        }
        else if (useFfnGemm) {
            linear_THF_from_FH_blocked4(
                dAct_.data(),
                dX_.data(),
                layers_[L].Wdown.data(),
                T, H, F, cfg_.hTile, tTile);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                float* dAct_t = &dAct_[t * F];
                float* dGate_t = &dGate_[t * F];
                float* dUp_t = &dUp_[t * F];
                const float* gate_t = &gate_[t * F];
                const float* up_t = &up_[t * F];

                if (attentionMask[t] == 0) {
                    std::fill(dAct_t, dAct_t + F, 0.0f);
                    std::fill(dGate_t, dGate_t + F, 0.0f);
                    std::fill(dUp_t, dUp_t + F, 0.0f);
                    continue;
                }
                if (fastSiluBackwardEnabled()) {
                    silu_backward_avx2_or_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                }
                else {
                    silu_backward_exact_scalar(dGate_t, dUp_t, dAct_t, gate_t, up_t, F);
                }
            }

            linear_TFH_fused_dual_from_HF_blocked4(
                dXnormFfn_.data(),
                dUp_.data(),
                dGate_.data(),
                layers_[L].Wup.data(),
                layers_[L].Wgate.data(),
                T, F, H, cfg_.hTile, tTile);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
                const std::size_t t = static_cast<std::size_t>(t_i);
                if (attentionMask[t] == 0) continue;
                rmsnorm_backward(
                    &dXin_[t * H],
                    &dXnormFfn_[t * H],
                    &x_mid_[t * H],
                    layers_[L].rms_ffn.data(),
                    H, eps
                );

                float* dXmid_t = &dXmid_[t * H];
                const float* dXmidFromNorm_t = &dXin_[t * H];
                for (std::size_t h = 0; h < H; ++h) dXmid_t[h] += dXmidFromNorm_t[h];
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
                        const float* gate_t = &gate_[t * F];
                        const float* up_t = &up_[t * F];

                        for (std::size_t f = 0; f < F; ++f) {
                            const float* wDownRow = &layers_[L].Wdown[f * H];
                            dAct_t[f] = dot_avx2_or_scalar(dOut, wDownRow, H);
                        }

                        for (std::size_t f = 0; f < F; ++f) {
                            const float siluG = silu_scalar(gate_t[f]);
                            dUp_t[f] = dAct_t[f] * siluG;
                            dGate_t[f] = dAct_t[f] * up_t[f] * silu_derivative_scalar(gate_t[f]);
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

                        rmsnorm_backward(
                            &dXin_[t * H],
                            dXnFfn_t,
                            &x_mid_[t * H],
                            layers_[L].rms_ffn.data(),
                            H, eps
                        );

                        float* dXmid_t = &dXmid_[t * H];
                        const float* dXmidFromNorm_t = &dXin_[t * H];
                        for (std::size_t h = 0; h < H; ++h) dXmid_t[h] += dXmidFromNorm_t[h];
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
                const float* gate0 = &gate_[(t0 + 0) * F];
                const float* gate1 = &gate_[(t0 + 1) * F];
                const float* gate2 = &gate_[(t0 + 2) * F];
                const float* gate3 = &gate_[(t0 + 3) * F];
                const float* up0 = &up_[(t0 + 0) * F];
                const float* up1 = &up_[(t0 + 1) * F];
                const float* up2 = &up_[(t0 + 2) * F];
                const float* up3 = &up_[(t0 + 3) * F];

                for (std::size_t f = 0; f < F; ++f) {
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    dot_x4_avx2_or_scalar(dOut0, dOut1, dOut2, dOut3, &layers_[L].Wdown[f * H], H, a0, a1, a2, a3);
                    dAct0[f] = (attentionMask[t0 + 0] == 0) ? 0.0f : a0;
                    dAct1[f] = (attentionMask[t0 + 1] == 0) ? 0.0f : a1;
                    dAct2[f] = (attentionMask[t0 + 2] == 0) ? 0.0f : a2;
                    dAct3[f] = (attentionMask[t0 + 3] == 0) ? 0.0f : a3;

                    dUp0[f] = dAct0[f] * silu_scalar(gate0[f]);
                    dUp1[f] = dAct1[f] * silu_scalar(gate1[f]);
                    dUp2[f] = dAct2[f] * silu_scalar(gate2[f]);
                    dUp3[f] = dAct3[f] * silu_scalar(gate3[f]);
                    dGate0[f] = dAct0[f] * up0[f] * silu_derivative_scalar(gate0[f]);
                    dGate1[f] = dAct1[f] * up1[f] * silu_derivative_scalar(gate1[f]);
                    dGate2[f] = dAct2[f] * up2[f] * silu_derivative_scalar(gate2[f]);
                    dGate3[f] = dAct3[f] * up3[f] * silu_derivative_scalar(gate3[f]);
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

                if (attentionMask[t0 + 0] != 0) rmsnorm_backward(&dXin_[(t0 + 0) * H], dXn0, &x_mid_[(t0 + 0) * H], layers_[L].rms_ffn.data(), H, eps);
                if (attentionMask[t0 + 1] != 0) rmsnorm_backward(&dXin_[(t0 + 1) * H], dXn1, &x_mid_[(t0 + 1) * H], layers_[L].rms_ffn.data(), H, eps);
                if (attentionMask[t0 + 2] != 0) rmsnorm_backward(&dXin_[(t0 + 2) * H], dXn2, &x_mid_[(t0 + 2) * H], layers_[L].rms_ffn.data(), H, eps);
                if (attentionMask[t0 + 3] != 0) rmsnorm_backward(&dXin_[(t0 + 3) * H], dXn3, &x_mid_[(t0 + 3) * H], layers_[L].rms_ffn.data(), H, eps);

                if (attentionMask[t0 + 0] != 0) for (std::size_t h = 0; h < H; ++h) dXmid_[(t0 + 0) * H + h] += dXin_[(t0 + 0) * H + h];
                if (attentionMask[t0 + 1] != 0) for (std::size_t h = 0; h < H; ++h) dXmid_[(t0 + 1) * H + h] += dXin_[(t0 + 1) * H + h];
                if (attentionMask[t0 + 2] != 0) for (std::size_t h = 0; h < H; ++h) dXmid_[(t0 + 2) * H + h] += dXin_[(t0 + 2) * H + h];
                if (attentionMask[t0 + 3] != 0) for (std::size_t h = 0; h < H; ++h) dXmid_[(t0 + 3) * H + h] += dXin_[(t0 + 3) * H + h];
            }
        }
        if (profile) prof.bwdFfnMs += elapsedMs(ffnStart);

        // x_mid = x_in + attn_out, so attn branch receives dXmid.
        const auto attnTailStart = profile ? ProfileClock::now() : ProfileClock::time_point{};
        for (std::size_t i = 0; i < T * H; ++i) {
            dAttn_[i] = dXmid_[i];
        }

        if (useI2SWo) {
            const std::size_t nYBlocks = layers_[L].Wo_q.nJBlocks();
            (void)nYBlocks;
            // dAttn rows are H-wide; nJBlocks = ceil(H/jq). bwdDY* are sized to
            // ceil(V/jq) which is always >= ceil(H/jq) since V >= H.
            linear_backward_bitnet_i2s_packed(dCtx_.data(), dAttn_.data(), layers_[L].Wo_q, T, tTile, bwdDY8Scratch_.data(), bwdDYScaleScratch_.data());
        }
        else {
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                const float* dat = &dAttn_[t * H]; float* dct = &dCtx_[t * H];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum = 0; for (std::size_t j = 0; j < H; ++j) sum += dat[j] * layers_[L].Wo[i * H + j]; dct[i] += sum;
                }
            }
        }

        attn_causal_backward(
            dQ_.data(), dK_.data(), dV_.data(),
            dCtx_.data(),
            q_.data(), k_.data(), v_.data(),
            T, H, shape_.nHeads, nKvHeads, probs_.data(),
            attentionMask.data(), tmpdP_.data()
        );

        // Backprop through RoPE: d(pre-rope Q/K) = R(theta)^T * d(post-rope Q/K).
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;
            for (std::size_t h = 0; h < nHeads; ++h) {
                apply_rope_backward_inplace(&dQ_[t * H + h * D], D, ropeDim, t, ropeTheta);
            }
            for (std::size_t kvh = 0; kvh < nKvHeads; ++kvh) {
                apply_rope_backward_inplace(&dK_[t * kvDim + kvh * D], D, ropeDim, t, ropeTheta);
            }
        }

        // ── LoRA gradient accumulation ────────────────────────────────────────
        auto& gL = loraGrads_[L];

        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;

            const float* xnorm_t = &xnorm_[t * H];

            // Q LoRA grads
            {
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

            // V LoRA grads
            {
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

        // ── Base weight backward into dXnorm_ ────────────────────────────────
        if (useFusedI2SQKV) {
            linear_fused_qkv_backward_bitnet_i2s_packed(
                dXnorm_.data(),
                dQ_.data(), dK_.data(), dV_.data(),
                layers_[L].Wq_q, layers_[L].Wk_q, layers_[L].Wv_q,
                T, tTile,
                bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                bwdDK8Scratch_.data(), bwdDKScaleScratch_.data(),
                bwdDV8Scratch_.data(), bwdDVScaleScratch_.data());
        }
        else if (useI2SQ || useI2SK || useI2SV) {
            if (useI2SQ) {
                linear_backward_bitnet_i2s_packed(dXnorm_.data(), dQ_.data(), layers_[L].Wq_q, T, tTile, bwdDY8Scratch_.data(), bwdDYScaleScratch_.data());
            }
            if (useFusedI2SKV) {
                linear_fused_dual_backward_bitnet_i2s_packed(
                    dXnorm_.data(),
                    dK_.data(), dV_.data(),
                    layers_[L].Wk_q, layers_[L].Wv_q,
                    T, tTile,
                    bwdDY8Scratch_.data(), bwdDYScaleScratch_.data(),
                    bwdDK8Scratch_.data(), bwdDKScaleScratch_.data());
            }
            else {
                if (useI2SK) {
                    linear_backward_bitnet_i2s_packed(dXnorm_.data(), dK_.data(), layers_[L].Wk_q, T, tTile, bwdDY8Scratch_.data(), bwdDYScaleScratch_.data());
                }
                if (useI2SV) {
                    linear_backward_bitnet_i2s_packed(dXnorm_.data(), dV_.data(), layers_[L].Wv_q, T, tTile, bwdDY8Scratch_.data(), bwdDYScaleScratch_.data());
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

        // ── FIX: Correct residual gradient accumulation ───────────────────────
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;
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
        for (std::size_t i = 0; i < T * H; ++i) {
            dX_[i] = dXmid_[i] + dXin_[i];
        }
        if (profile) prof.bwdAttnMs += elapsedMs(attnTailStart);
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
        std::fill(g.gradAv.begin(), g.gradAv.end(), 0.0f);
        std::fill(g.gradBv.begin(), g.gradBv.end(), 0.0f);
    }
}

void CpuKernelBackend::saveLoRA(const std::string& path) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write to " + path);

    const uint32_t H = static_cast<uint32_t>(shape_.hiddenSize);
    const uint32_t R = static_cast<uint32_t>(cfg_.loraRank);
    const uint32_t flags = 5;
    const uint32_t nL = static_cast<uint32_t>(loras_.size());
    const float    alpha = cfg_.loraAlpha;

    out.write("LORA", 4);
    uint32_t version = 1, dtype = 0;
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
        out.write((const char*)loras_[L].Aq.data(), R * H * sizeof(float));
        out.write((const char*)loras_[L].Bq.data(), H * R * sizeof(float));
        out.write((const char*)loras_[L].Av.data(), R * H * sizeof(float));
        out.write((const char*)loras_[L].Bv.data(), kvD * R * sizeof(float));
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

    for (std::size_t L = 0; L < nL; ++L) {
        in.read((char*)loras_[L].Aq.data(), R * H * sizeof(float));
        in.read((char*)loras_[L].Bq.data(), H * R * sizeof(float));
        in.read((char*)loras_[L].Av.data(), R * H * sizeof(float));
        in.read((char*)loras_[L].Bv.data(), kvDim * R * sizeof(float));
    }
}

std::vector<ParamRef> CpuKernelBackend::collectTrainableParams()
{
    if (loras_.size() != loraGrads_.size()) {
        throw std::runtime_error("collectTrainableParams: loras_ and loraGrads_ size mismatch");
    }

    std::vector<ParamRef> out;
    out.reserve(loras_.size() * 4);

    for (std::size_t l = 0; l < loras_.size(); ++l) {
        auto& P = loras_[l];
        auto& G = loraGrads_[l];

        if (P.Aq.size() != G.gradAq.size()) throw std::runtime_error("Aq grad size mismatch");
        if (P.Bq.size() != G.gradBq.size()) throw std::runtime_error("Bq grad size mismatch");
        if (P.Av.size() != G.gradAv.size()) throw std::runtime_error("Av grad size mismatch");
        if (P.Bv.size() != G.gradBv.size()) throw std::runtime_error("Bv grad size mismatch");

        out.push_back({ P.Aq.data(), G.gradAq.data(), P.Aq.size(), "Aq" });
        out.push_back({ P.Bq.data(), G.gradBq.data(), P.Bq.size(), "Bq" });
        out.push_back({ P.Av.data(), G.gradAv.data(), P.Av.size(), "Av" });
        out.push_back({ P.Bv.data(), G.gradBv.data(), P.Bv.size(), "Bv" });
    }

    return out;
}
