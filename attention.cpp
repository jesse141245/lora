#include <cstddef>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "attention.h"

// ===========================================================================
// Internal helpers
// ===========================================================================

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

static inline std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static inline bool env_string_truthy(const std::string& s) {
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

static inline bool env_string_falsy(const std::string& s) {
    return s == "0" || s == "false" || s == "no" || s == "off";
}

static inline bool env_bool_or(const char* name, bool fallback) {
    const auto value = try_get_env_string(name);
    if (!value) {
        return fallback;
    }
    const std::string lowered = lower_copy(*value);
    if (env_string_truthy(lowered)) {
        return true;
    }
    if (env_string_falsy(lowered)) {
        return false;
    }
    return fallback;
}

static inline std::optional<bool> env_bool_override(const char* name) {
    const auto value = try_get_env_string(name);
    if (!value) {
        return std::nullopt;
    }
    const std::string lowered = lower_copy(*value);
    if (env_string_truthy(lowered)) {
        return true;
    }
    if (env_string_falsy(lowered)) {
        return false;
    }
    return std::nullopt;
}

static inline bool attn_d80_specialized_enabled() {
    static const bool cached = env_bool_or("QLORA_TEST_ATTN_D80_SPECIALIZED", false);
    return cached;
}

static inline bool attn_fast_softmax_enabled() {
    static const bool cached = env_bool_or("QLORA_TEST_FAST_ATTN_SOFTMAX", false);
    return cached;
}

static inline bool attn_fill_tails_enabled() {
    static const bool cached = env_bool_or("QLORA_TEST_ATTN_FILL_TAILS", false);
    return cached;
}

static inline bool attn_d128_weighted_sum_enabled() {
    static const bool cached = env_bool_or("QLORA_TEST_ATTN_D128_WEIGHTED_SUM", false);
    return cached;
}

struct BoolEnvKnob {
    const char* name;
    bool defaultValue;
};

static constexpr BoolEnvKnob kAttnDqOverwriteD128Knob{
    "QLORA_TEST_ATTN_DQ_OVERWRITE_D128",
    true
};
static constexpr BoolEnvKnob kAttnBwdGroupLocalAccumKnob{
    "QLORA_TEST_ATTN_BWD_GROUP_LOCAL_ACCUM",
    false
};
static constexpr BoolEnvKnob kAttnBwdGroupLocalRowOverwriteKnob{
    "QLORA_TEST_ATTN_BWD_GROUP_LOCAL_ROW_OVERWRITE",
    false
};
static constexpr BoolEnvKnob kAttnBwdGroup4FusedD128Knob{
    "QLORA_TEST_ATTN_BWD_GROUP4_FUSED_D128",
    true
};
static constexpr BoolEnvKnob kAttnBwdGroup4RowOverwriteKnob{
    "QLORA_TEST_ATTN_BWD_GROUP4_ROW_OVERWRITE",
    false
};
static constexpr BoolEnvKnob kAttnBwdPrecomputeDsD128Knob{
    "QLORA_TEST_ATTN_BWD_PRECOMPUTE_DS_D128",
    false
};
static constexpr BoolEnvKnob kAttnBwdGroup4SplitDqDkvD128Knob{
    "QLORA_TEST_ATTN_BWD_GROUP4_SPLIT_DQ_DKV_D128",
    false
};
static constexpr BoolEnvKnob kAttnBwdHeadParallelKnob{
    "QLORA_TEST_ATTN_BWD_HEAD_PARALLEL",
    false
};

static inline bool env_bool_knob(const BoolEnvKnob& knob) {
    return env_bool_or(knob.name, knob.defaultValue);
}

static inline bool attn_backward_head_parallel_enabled() {
    static const bool cached = env_bool_knob(kAttnBwdHeadParallelKnob);
    return cached;
}

static inline bool attn_dq_overwrite_d128_enabled() {
    static const bool cached = env_bool_knob(kAttnDqOverwriteD128Knob);
    return cached;
}

static inline bool attn_backward_group_local_accum_enabled() {
    static const bool cached = env_bool_knob(kAttnBwdGroupLocalAccumKnob);
    return cached;
}

static inline bool attn_backward_group_local_row_overwrite_enabled() {
    static const bool cached = env_bool_knob(kAttnBwdGroupLocalRowOverwriteKnob);
    return cached;
}

static inline bool attn_backward_group4_fused_d128_enabled() {
    static const bool cached = env_bool_knob(kAttnBwdGroup4FusedD128Knob);
    return cached;
}

static inline bool attn_backward_group4_row_overwrite_enabled() {
    static const bool cached = env_bool_knob(kAttnBwdGroup4RowOverwriteKnob);
    return cached;
}

static inline bool attn_backward_precompute_ds_d128_enabled() {
    static const bool cached = env_bool_knob(kAttnBwdPrecomputeDsD128Knob);
    return cached;
}

static inline bool attn_backward_group4_split_dq_dkv_d128_enabled() {
    static const bool cached = env_bool_knob(kAttnBwdGroup4SplitDqDkvD128Knob);
    return cached;
}

static inline bool attn_backward_debug_compare_enabled() {
    static const bool cached = env_bool_or("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE", false);
    return cached;
}

static inline bool attn_backward_debug_compare_once_enabled() {
    static const bool cached = env_bool_or("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE_ONCE", true);
    return cached;
}

static inline bool attn_backward_group_local_group_compare_enabled() {
    static const bool cached = env_bool_or("QLORA_TEST_ATTN_BWD_GROUP_LOCAL_GROUP_COMPARE", false);
    return cached;
}

static inline bool attn_backward_group_local_head_compare_enabled() {
    static const bool cached = env_bool_or("QLORA_TEST_ATTN_BWD_GROUP_LOCAL_HEAD_COMPARE", false);
    return cached;
}

static inline std::size_t attn_backward_group_local_group_compare_call_index() {
    const auto value = try_get_env_string("QLORA_TEST_ATTN_BWD_GROUP_LOCAL_GROUP_COMPARE_CALL");
    if (!value) {
        return 0;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value->c_str(), &end, 10);
    if (end == value->c_str()) {
        return 0;
    }
    return static_cast<std::size_t>(parsed);
}

struct AttnBackwardOptions {
    bool enableDqOverwriteD128;
    bool enableGroupedLocalAccum;
    bool enableGroupedLocalRowOverwrite;
    bool enableGrouped4FusedD128;
    bool enableGrouped4RowOverwrite;
    bool enablePrecomputeDsD128;
    bool enableGrouped4SplitDqDkvD128;
    bool enableHeadParallel;
};

static inline void normalize_attn_backward_options(AttnBackwardOptions& options) {
    if (!options.enableGroupedLocalAccum) {
        options.enableGroupedLocalRowOverwrite = false;
        options.enableGrouped4FusedD128 = false;
        options.enableGrouped4RowOverwrite = false;
    }
    if (!options.enableGrouped4FusedD128) {
        options.enableGrouped4RowOverwrite = false;
        options.enableGrouped4SplitDqDkvD128 = false;
    }
}

static inline AttnBackwardOptions default_attn_backward_options() {
    AttnBackwardOptions options{
        attn_dq_overwrite_d128_enabled(),
        attn_backward_group_local_accum_enabled(),
        attn_backward_group_local_row_overwrite_enabled(),
        attn_backward_group4_fused_d128_enabled(),
        attn_backward_group4_row_overwrite_enabled(),
        attn_backward_precompute_ds_d128_enabled(),
        attn_backward_group4_split_dq_dkv_d128_enabled(),
        attn_backward_head_parallel_enabled(),
    };
    normalize_attn_backward_options(options);
    return options;
}

static inline AttnBackwardOptions baseline_attn_backward_options() {
    return AttnBackwardOptions{
        attn_dq_overwrite_d128_enabled(),
        false,
        false,
        false,
        false,
        false,
        false,
        false,
    };
}

static inline void apply_debug_candidate_overrides(AttnBackwardOptions& options) {
    if (const auto override = env_bool_override("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE_GROUP_LOCAL")) {
        options.enableGroupedLocalAccum = *override;
    }
    if (const auto override = env_bool_override("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE_GROUP_LOCAL_ROW_OVERWRITE")) {
        options.enableGroupedLocalRowOverwrite = *override;
    }
    if (const auto override = env_bool_override("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE_GROUP4_FUSED_D128")) {
        options.enableGrouped4FusedD128 = *override;
    }
    if (const auto override = env_bool_override("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE_GROUP4_ROW_OVERWRITE")) {
        options.enableGrouped4RowOverwrite = *override;
    }
    if (const auto override = env_bool_override("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE_PRECOMPUTE_DS_D128")) {
        options.enablePrecomputeDsD128 = *override;
    }
    if (const auto override = env_bool_override("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE_GROUP4_SPLIT_DQ_DKV_D128")) {
        options.enableGrouped4SplitDqDkvD128 = *override;
    }
    if (const auto override = env_bool_override("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE_DQ_OVERWRITE_D128")) {
        options.enableDqOverwriteD128 = *override;
    }
    if (const auto override = env_bool_override("QLORA_TEST_ATTN_BWD_DEBUG_COMPARE_HEAD_PARALLEL")) {
        options.enableHeadParallel = *override;
    }
    normalize_attn_backward_options(options);
}

static inline void report_backward_diff(
    const char* name,
    const float* ref,
    const float* test,
    std::size_t count)
{
    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    std::size_t maxAbsIdx = 0;
    std::size_t maxRelIdx = 0;

    for (std::size_t i = 0; i < count; ++i) {
        const float diff = std::fabs(ref[i] - test[i]);
        if (diff > maxAbs) {
            maxAbs = diff;
            maxAbsIdx = i;
        }

        const float denom = std::fabs(ref[i]) + 1.0e-6f;
        const float rel = diff / denom;
        if (rel > maxRel) {
            maxRel = rel;
            maxRelIdx = i;
        }
    }

    std::fprintf(
        stderr,
        "[ATTN_BWD_COMPARE] %s max_abs=%.9g idx=%zu ref=%.9g test=%.9g | max_rel=%.9g idx=%zu ref=%.9g test=%.9g\n",
        name,
        static_cast<double>(maxAbs),
        maxAbsIdx,
        static_cast<double>(ref[maxAbsIdx]),
        static_cast<double>(test[maxAbsIdx]),
        static_cast<double>(maxRel),
        maxRelIdx,
        static_cast<double>(ref[maxRelIdx]),
        static_cast<double>(test[maxRelIdx]));
}

static inline float hsum128(__m128 v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hsum128(_mm_add_ps(lo, hi));
}

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

// ===========================================================================
// AVX2 softmax (in-place)
// Replaces the fully scalar version. With n=32 (T) this runs in ~4 AVX passes
// instead of 32 scalar iterations.
// ===========================================================================

static void softmax_inplace(float* __restrict p, std::size_t n)
{
    // ── find max ────────────────────────────────────────────────────────────
    __m256 vmx = _mm256_set1_ps(-1e38f);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(p + i));
    float mx = -1e38f;
    // horizontal max from the 8 lanes
    __m128 lo = _mm256_castps256_ps128(vmx);
    __m128 hi = _mm256_extractf128_ps(vmx, 1);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ps(m4, _mm_movehdup_ps(m4));
    mx = _mm_cvtss_f32(m4);
    for (; i < n; ++i) mx = std::max(mx, p[i]);

    // ── exp(x - max) and sum ─────────────────────────────────────────────────
    const __m256 vmxv = _mm256_set1_ps(mx);
    __m256 vsum = _mm256_setzero_ps();
    const bool fastExp = attn_fast_softmax_enabled();
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(p + i), vmxv);
        if (fastExp) {
            v = exp256_ps(v);
        }
        else {
            float tmp[8];
            _mm256_storeu_ps(tmp, v);
            for (int k = 0; k < 8; ++k) tmp[k] = std::exp(tmp[k]);
            v = _mm256_loadu_ps(tmp);
        }
        _mm256_storeu_ps(p + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    float s = hsum256(vsum);
    for (; i < n; ++i) { p[i] = std::exp(p[i] - mx); s += p[i]; }

    // ── normalise ────────────────────────────────────────────────────────────
    const float inv = 1.0f / (s + 1e-9f);
    const __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(p + i, _mm256_mul_ps(_mm256_loadu_ps(p + i), vinv));
    for (; i < n; ++i) p[i] *= inv;
}

// ===========================================================================
// AVX2 dot product over D dimensions.
// Keep the common head widths fully unrolled so attention stays off the
// generic strip-mined loop on the 2B shapes used here.
// ===========================================================================

static inline float dot_D(const float* __restrict a,
    const float* __restrict b,
    std::size_t D)
{
    // -----------------------------
    // D = 80 (2 accumulators)
    // -----------------------------
    if (D == 80 && attn_d80_specialized_enabled()) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 0), _mm256_loadu_ps(b + 0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), acc1);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), acc1);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), acc1);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), acc1);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 64), _mm256_loadu_ps(b + 64), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 72), _mm256_loadu_ps(b + 72), acc1);

        __m256 acc = _mm256_add_ps(acc0, acc1);
        return hsum256(acc);
    }

    // -----------------------------
    // D = 128 (4 accumulators)
    // -----------------------------
    if (D == 128) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 0), _mm256_loadu_ps(b + 0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 16), _mm256_loadu_ps(b + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 24), _mm256_loadu_ps(b + 24), acc3);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 32), _mm256_loadu_ps(b + 32), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 40), _mm256_loadu_ps(b + 40), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 48), _mm256_loadu_ps(b + 48), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 56), _mm256_loadu_ps(b + 56), acc3);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 64), _mm256_loadu_ps(b + 64), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 72), _mm256_loadu_ps(b + 72), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 80), _mm256_loadu_ps(b + 80), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 88), _mm256_loadu_ps(b + 88), acc3);

        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 96), _mm256_loadu_ps(b + 96), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 104), _mm256_loadu_ps(b + 104), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 112), _mm256_loadu_ps(b + 112), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + 120), _mm256_loadu_ps(b + 120), acc3);

        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3));
        return hsum256(acc);
    }

    // -----------------------------
    // Generic fallback
    // -----------------------------
    __m256 acc = _mm256_setzero_ps();
    std::size_t d = 0;

    for (; d + 8 <= D; d += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + d),
            _mm256_loadu_ps(b + d),
            acc);
    }

    float s = hsum256(acc);
    for (; d < D; ++d) {
        s += a[d] * b[d];
    }

    return s;
}

static inline void dot4_128_against_one(
    const float* __restrict a0,
    const float* __restrict a1,
    const float* __restrict a2,
    const float* __restrict a3,
    const float* __restrict b,
    float& out0,
    float& out1,
    float& out2,
    float& out3)
{
    __m256 acc00 = _mm256_setzero_ps();
    __m256 acc01 = _mm256_setzero_ps();
    __m256 acc10 = _mm256_setzero_ps();
    __m256 acc11 = _mm256_setzero_ps();
    __m256 acc20 = _mm256_setzero_ps();
    __m256 acc21 = _mm256_setzero_ps();
    __m256 acc30 = _mm256_setzero_ps();
    __m256 acc31 = _mm256_setzero_ps();

    for (std::size_t d = 0; d < 128; d += 16) {
        const __m256 b0 = _mm256_loadu_ps(b + d + 0);
        const __m256 b1 = _mm256_loadu_ps(b + d + 8);

        acc00 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + d + 0), b0, acc00);
        acc01 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + d + 8), b1, acc01);

        acc10 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + d + 0), b0, acc10);
        acc11 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + d + 8), b1, acc11);

        acc20 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + d + 0), b0, acc20);
        acc21 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + d + 8), b1, acc21);

        acc30 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + d + 0), b0, acc30);
        acc31 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + d + 8), b1, acc31);
    }

    out0 = hsum256(_mm256_add_ps(acc00, acc01));
    out1 = hsum256(_mm256_add_ps(acc10, acc11));
    out2 = hsum256(_mm256_add_ps(acc20, acc21));
    out3 = hsum256(_mm256_add_ps(acc30, acc31));
}
// AVX2 AXPY: dst[d] += scale * src[d]  over D dims
static inline void axpy_D(float* __restrict dst,
    const float* __restrict src,
    float scale,
    std::size_t D)
{
    if (D == 80 && attn_d80_specialized_enabled()) {
        const __m256 sv = _mm256_set1_ps(scale);
        _mm256_storeu_ps(dst +  0, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  0), _mm256_loadu_ps(dst +  0)));
        _mm256_storeu_ps(dst +  8, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  8), _mm256_loadu_ps(dst +  8)));
        _mm256_storeu_ps(dst + 16, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 16), _mm256_loadu_ps(dst + 16)));
        _mm256_storeu_ps(dst + 24, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 24), _mm256_loadu_ps(dst + 24)));
        _mm256_storeu_ps(dst + 32, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 32), _mm256_loadu_ps(dst + 32)));
        _mm256_storeu_ps(dst + 40, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 40), _mm256_loadu_ps(dst + 40)));
        _mm256_storeu_ps(dst + 48, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 48), _mm256_loadu_ps(dst + 48)));
        _mm256_storeu_ps(dst + 56, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 56), _mm256_loadu_ps(dst + 56)));
        _mm256_storeu_ps(dst + 64, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 64), _mm256_loadu_ps(dst + 64)));
        _mm256_storeu_ps(dst + 72, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 72), _mm256_loadu_ps(dst + 72)));
        return;
    }

    if (D == 128) {
        const __m256 sv = _mm256_set1_ps(scale);
        _mm256_storeu_ps(dst +   0, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +   0), _mm256_loadu_ps(dst +   0)));
        _mm256_storeu_ps(dst +   8, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +   8), _mm256_loadu_ps(dst +   8)));
        _mm256_storeu_ps(dst +  16, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  16), _mm256_loadu_ps(dst +  16)));
        _mm256_storeu_ps(dst +  24, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  24), _mm256_loadu_ps(dst +  24)));
        _mm256_storeu_ps(dst +  32, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  32), _mm256_loadu_ps(dst +  32)));
        _mm256_storeu_ps(dst +  40, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  40), _mm256_loadu_ps(dst +  40)));
        _mm256_storeu_ps(dst +  48, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  48), _mm256_loadu_ps(dst +  48)));
        _mm256_storeu_ps(dst +  56, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  56), _mm256_loadu_ps(dst +  56)));
        _mm256_storeu_ps(dst +  64, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  64), _mm256_loadu_ps(dst +  64)));
        _mm256_storeu_ps(dst +  72, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  72), _mm256_loadu_ps(dst +  72)));
        _mm256_storeu_ps(dst +  80, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  80), _mm256_loadu_ps(dst +  80)));
        _mm256_storeu_ps(dst +  88, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  88), _mm256_loadu_ps(dst +  88)));
        _mm256_storeu_ps(dst +  96, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src +  96), _mm256_loadu_ps(dst +  96)));
        _mm256_storeu_ps(dst + 104, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 104), _mm256_loadu_ps(dst + 104)));
        _mm256_storeu_ps(dst + 112, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 112), _mm256_loadu_ps(dst + 112)));
        _mm256_storeu_ps(dst + 120, _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + 120), _mm256_loadu_ps(dst + 120)));
        return;
    }

    const __m256 sv = _mm256_set1_ps(scale);
    std::size_t d = 0;
    for (; d + 8 <= D; d += 8)
        _mm256_storeu_ps(dst + d,
            _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + d),
                _mm256_loadu_ps(dst + d)));
    for (; d < D; ++d) dst[d] += scale * src[d];
}

static inline void weighted_sum_D128_store(
    float* __restrict dst,
    const float* __restrict vBase,
    const float* __restrict probs,
    std::size_t prefix,
    std::size_t kvDim)
{
    __m256 acc[16];
    for (int blk = 0; blk < 16; ++blk) {
        acc[blk] = _mm256_setzero_ps();
    }

    for (std::size_t j = 0; j < prefix; ++j) {
        const __m256 pj = _mm256_set1_ps(probs[j]);
        const float* vj = vBase + j * kvDim;
        for (int blk = 0; blk < 16; ++blk) {
            acc[blk] = _mm256_fmadd_ps(pj, _mm256_loadu_ps(vj + blk * 8), acc[blk]);
        }
    }

    for (int blk = 0; blk < 16; ++blk) {
        _mm256_storeu_ps(dst + blk * 8, acc[blk]);
    }
}

// ===========================================================================
// attn_causal  (forward)
// ===========================================================================

void attn_causal(
    float* __restrict ctx,
    const float* __restrict q,
    const float* __restrict k,
    const float* __restrict v,
    std::size_t T,
    std::size_t H,
    std::size_t nHeads,
    std::size_t nKvHeads,
    float* __restrict scores,
    float* __restrict probs,
    const int* attentionMask)
{
    if (nHeads == 0 || nKvHeads == 0) {
        throw std::runtime_error("attn_causal: nHeads and nKvHeads must be > 0");
    }
    if (H % nHeads != 0) {
        throw std::runtime_error("attn_causal: H must be divisible by nHeads");
    }
    if (nHeads % nKvHeads != 0) {
        throw std::runtime_error("attn_causal: nHeads must be divisible by nKvHeads (GQA)");
    }

    const std::size_t D = H / nHeads;
    const std::size_t kvDim = nKvHeads * D;
    const float       inv_sqrt_d = 1.0f / std::sqrt(float(D));
    const std::size_t heads_per_group = nHeads / nKvHeads;
    const bool segmented = attentionMask &&
        std::any_of(attentionMask, attentionMask + T, [](int v) { return v > 1; });
    std::vector<std::size_t> segmentStarts;
    if (segmented) {
        segmentStarts.resize(T, 0);
        std::size_t currentStart = 0;
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] > 1) {
                currentStart = t;
            }
            segmentStarts[t] = currentStart;
        }
    }
    const bool allActive = !segmented && (!attentionMask ||
        std::all_of(attentionMask, attentionMask + T, [](int v) { return v != 0; }));
    const bool useD128WeightedSum = allActive && D == 128 && attn_d128_weighted_sum_enabled();
    const bool fillTails = attn_fill_tails_enabled();

    if (!useD128WeightedSum) {
        std::fill(ctx, ctx + T * H, 0.0f);
    }

    auto forward_one = [&](std::size_t h, std::size_t t) {
        const std::size_t kv_h = h / heads_per_group;
        if (attentionMask && attentionMask[t] == 0) return;

        const float* qt = q + t * H + h * D;
        float* pr = probs + (h * T * T) + (t * T);
        float* sc = scores ? (scores + (h * T * T) + (t * T)) : pr;
        const std::size_t begin = segmented ? segmentStarts[t] : 0;
        const std::size_t prefix = t + 1;
        const std::size_t count = prefix - begin;

        // ── Q·K scores ───────────────────────────────────────────────────
        if (allActive) {
            for (std::size_t j = begin; j < prefix; ++j) {
                sc[j] = dot_D(qt, k + j * kvDim + kv_h * D, D) * inv_sqrt_d;
            }
        }
        else {
            for (std::size_t j = begin; j < prefix; ++j) {
                if (attentionMask[j] == 0) {
                    sc[j] = -1e9f;
                    continue;
                }
                sc[j] = dot_D(qt, k + j * kvDim + kv_h * D, D) * inv_sqrt_d;
            }
        }
        if (fillTails) {
            if (begin > 0) {
                std::fill(sc, sc + begin, -1e9f);
            }
            std::fill(sc + prefix, sc + T, -1e9f);
        }

        // ── softmax ───────────────────────────────────────────────────────
        if (pr != sc) {
            std::memcpy(pr + begin, sc + begin, count * sizeof(float));
        }
        softmax_inplace(pr + begin, count);
        if (fillTails) {
            if (begin > 0) {
                std::fill(pr, pr + begin, 0.0f);
            }
            std::fill(pr + prefix, pr + T, 0.0f);
        }

        // ── weighted sum of V ─────────────────────────────────────────────
        float* out = ctx + t * H + h * D;
        if (allActive) {
            if (useD128WeightedSum) {
                weighted_sum_D128_store(out, v + begin * kvDim + kv_h * D, pr + begin, count, kvDim);
            }
            else {
                for (std::size_t j = begin; j < prefix; ++j) {
                    axpy_D(out, v + j * kvDim + kv_h * D, pr[j], D);
                }
            }
        }
        else {
            for (std::size_t j = begin; j < prefix; ++j) {
                if (attentionMask[j] == 0) continue;
                axpy_D(out, v + j * kvDim + kv_h * D, pr[j], D);
            }
        }
    };

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int h_i = 0; h_i < static_cast<int>(nHeads); ++h_i) {
        const std::size_t h = static_cast<std::size_t>(h_i);
        for (std::size_t t = 0; t < T; ++t) {
            forward_one(h, t);
        }
    }
}

// ===========================================================================
// attn_causal_backward
//
// BUG FIX: tmp_dP was never zeroed between heads. For masked positions
// (attentionMask[j]==0) tmp_dP[j] was left with the value from the previous
// head, causing a stale dP contribution in the softmax backward of every
// subsequent head. Fixed by zeroing the full T-length buffer at the start of
// each (head, t) pair before the dP computation loop.
// ===========================================================================

static void attn_causal_backward_impl(
    float* __restrict dQ,
    float* __restrict dK,
    float* __restrict dV,
    const float* __restrict dCtx,
    const float* __restrict q,
    const float* __restrict k,
    const float* __restrict v,
    std::size_t T,
    std::size_t H,
    std::size_t nHeads,
    std::size_t nKvHeads,
    const float* __restrict probs,
    const int* attentionMask,
    float* __restrict tmp_dP,
    const AttnBackwardOptions& options,
    std::size_t debugCallIndex)
{
    if (nHeads == 0 || nKvHeads == 0) {
        throw std::runtime_error("attn_causal_backward: nHeads and nKvHeads must be > 0");
    }
    if (H % nHeads != 0) {
        throw std::runtime_error("attn_causal_backward: H must be divisible by nHeads");
    }
    if (nHeads % nKvHeads != 0) {
        throw std::runtime_error("attn_causal_backward: nHeads must be divisible by nKvHeads (GQA)");
    }
    constexpr std::size_t kGroupedTileMinT = 96;
    constexpr std::size_t kTileWorkMin = 8192;


    const std::size_t D = H / nHeads;
    const std::size_t kvDim = nKvHeads * D;
    const float       inv_sqrt_d = 1.0f / std::sqrt(float(D));
    const std::size_t heads_per_group = nHeads / nKvHeads;
    const std::size_t dkvSize = T * kvDim;
    const bool segmented = attentionMask &&
        std::any_of(attentionMask, attentionMask + T, [](int v) { return v > 1; });
    std::vector<std::size_t> segmentStarts;
    if (segmented) {
        segmentStarts.resize(T, 0);
        std::size_t currentStart = 0;
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] > 1) {
                currentStart = t;
            }
            segmentStarts[t] = currentStart;
        }
    }
    const bool allActive = !segmented && (!attentionMask ||
        std::all_of(attentionMask, attentionMask + T, [](int v) { return v != 0; }));
    const bool useDqOverwriteD128 = allActive && D == 128 && options.enableDqOverwriteD128;
    const bool useGroupedLocalAccum =
        options.enableGroupedLocalAccum &&
        heads_per_group > 1 &&
        (T * D >= kTileWorkMin);
    const bool useGrouped4FusedD128 =
        useGroupedLocalAccum &&
        allActive &&
        D == 128 &&
        heads_per_group == 4 &&
        options.enableGrouped4FusedD128 &&
        T >= kGroupedTileMinT;
    const bool useGrouped4RowOverwrite =
        useGrouped4FusedD128 &&
        options.enableGrouped4RowOverwrite;
    const bool useGrouped4SplitDqDkvD128 =
        useGrouped4FusedD128 &&
        options.enableGrouped4SplitDqDkvD128;
    const bool usePrecomputeDsD128 =
        allActive &&
        D == 128 &&
        options.enablePrecomputeDsD128;
    const bool useGroupLocalGroupCompare =
        useGroupedLocalAccum &&
        attn_backward_group_local_group_compare_enabled() &&
        debugCallIndex == attn_backward_group_local_group_compare_call_index();
    const bool useGroupLocalHeadCompare =
        useGroupedLocalAccum &&
        !useGrouped4FusedD128 &&
        attn_backward_group_local_head_compare_enabled() &&
        debugCallIndex == attn_backward_group_local_group_compare_call_index();
    (void)tmp_dP;

    auto backward_one_head = [&](std::size_t head,
                                 float* __restrict dKDst,
                                 float* __restrict dVDst,
                                 std::size_t kvRowStride,
                                 bool kvSliceAlreadyOffset,
                                 std::vector<float>& tmpLocal,
                                 bool kvRowStartsZero) {
        const std::size_t kv_h = head / heads_per_group;
        const std::size_t kvOffset = kv_h * D;
        float* const dKSlice = kvSliceAlreadyOffset ? dKDst : (dKDst + kvOffset);
        float* const dVSlice = kvSliceAlreadyOffset ? dVDst : (dVDst + kvOffset);
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask && attentionMask[t] == 0) continue;

            const float* pr = probs + (head * T * T) + (t * T);
            const float* dCt = dCtx + t * H + head * D;
            const float* Qt = q + t * H + head * D;
            float* dQt = dQ + t * H + head * D;

            constexpr std::size_t JB = 16;
            float sum = 0.0f;

            const std::size_t begin = segmented ? segmentStarts[t] : 0;
            const std::size_t prefix = t + 1;
            if (allActive) {
                for (std::size_t j0 = begin; j0 < prefix; j0 += JB) {
                    const std::size_t j1 = std::min(j0 + JB, prefix);

                    for (std::size_t j = j0; j < j1; ++j) {
                        const float dp = dot_D(dCt, v + j * kvDim + kvOffset, D);
                        tmpLocal[j] = dp;
                        sum += pr[j] * dp;
                    }
                }
            }
            else {
                for (std::size_t j = begin; j < prefix; ++j) {
                    if (attentionMask[j] == 0) continue;
                    const float dp = dot_D(dCt, v + j * kvDim + kv_h * D, D);
                    tmpLocal[j] = dp;
                    sum += pr[j] * dp;
                }
            }

            constexpr float kDSClamp = 10.0f;
            if (usePrecomputeDsD128) {
                for (std::size_t j = begin; j < prefix; ++j) {
                    tmpLocal[j] =
                        std::max(-kDSClamp, std::min(kDSClamp, pr[j] * (tmpLocal[j] - sum))) * inv_sqrt_d;
                }
            }
            if (allActive) {
                if (D == 128) {
                    for (std::size_t j0 = begin; j0 < prefix; j0 += JB) {
                        const std::size_t j1 = std::min(j0 + JB, prefix);
                        const std::size_t tileRows = j1 - j0;

                        for (std::size_t base = 0; base < D; base += 32) {
                            alignas(32) float dKTile[JB][32];
                            alignas(32) float dVTile[JB][32];

                            // zero local tile scratch
                            for (std::size_t r = 0; r < tileRows; ++r) {
                                for (std::size_t x = 0; x < 32; ++x) {
                                    dKTile[r][x] = 0.0f;
                                    dVTile[r][x] = 0.0f;
                                }
                            }

                            const float* qBase = Qt + base;
                            const float* dCtBase = dCt + base;
                            float* dQBase = dQt + base;

                            const __m256 q0 = _mm256_loadu_ps(qBase + 0);
                            const __m256 q1 = _mm256_loadu_ps(qBase + 8);
                            const __m256 q2 = _mm256_loadu_ps(qBase + 16);
                            const __m256 q3 = _mm256_loadu_ps(qBase + 24);

                            const __m256 dc0 = _mm256_loadu_ps(dCtBase + 0);
                            const __m256 dc1 = _mm256_loadu_ps(dCtBase + 8);
                            const __m256 dc2 = _mm256_loadu_ps(dCtBase + 16);
                            const __m256 dc3 = _mm256_loadu_ps(dCtBase + 24);

                            __m256 dq0 = _mm256_loadu_ps(dQBase + 0);
                            __m256 dq1 = _mm256_loadu_ps(dQBase + 8);
                            __m256 dq2 = _mm256_loadu_ps(dQBase + 16);
                            __m256 dq3 = _mm256_loadu_ps(dQBase + 24);

                            // accumulate into local tile
                            for (std::size_t j = j0; j < j1; ++j) {
                                const std::size_t r = j - j0;

                                const float pj = pr[j];
                                const float dS = usePrecomputeDsD128
                                    ? tmpLocal[j]
                                    : std::max(-kDSClamp, std::min(kDSClamp, pj * (tmpLocal[j] - sum))) * inv_sqrt_d;

                                const __m256 pjv = _mm256_set1_ps(pj);
                                const __m256 aqk = _mm256_set1_ps(dS);

                                const float* kBase = k + j * kvDim + kvOffset + base;

                                const __m256 k0 = _mm256_loadu_ps(kBase + 0);
                                const __m256 k1 = _mm256_loadu_ps(kBase + 8);
                                const __m256 k2 = _mm256_loadu_ps(kBase + 16);
                                const __m256 k3 = _mm256_loadu_ps(kBase + 24);

                                // dQ still accumulates directly in registers
                                dq0 = _mm256_fmadd_ps(aqk, k0, dq0);
                                dq1 = _mm256_fmadd_ps(aqk, k1, dq1);
                                dq2 = _mm256_fmadd_ps(aqk, k2, dq2);
                                dq3 = _mm256_fmadd_ps(aqk, k3, dq3);

                                // dVTile[r] += pj * dCtBase
                                __m256 tv0 = _mm256_load_ps(&dVTile[r][0]);
                                __m256 tv1 = _mm256_load_ps(&dVTile[r][8]);
                                __m256 tv2 = _mm256_load_ps(&dVTile[r][16]);
                                __m256 tv3 = _mm256_load_ps(&dVTile[r][24]);
                                tv0 = _mm256_fmadd_ps(pjv, dc0, tv0);
                                tv1 = _mm256_fmadd_ps(pjv, dc1, tv1);
                                tv2 = _mm256_fmadd_ps(pjv, dc2, tv2);
                                tv3 = _mm256_fmadd_ps(pjv, dc3, tv3);
                                _mm256_store_ps(&dVTile[r][0], tv0);
                                _mm256_store_ps(&dVTile[r][8], tv1);
                                _mm256_store_ps(&dVTile[r][16], tv2);
                                _mm256_store_ps(&dVTile[r][24], tv3);

                                // dKTile[r] += dS * qBase
                                __m256 tk0 = _mm256_load_ps(&dKTile[r][0]);
                                __m256 tk1 = _mm256_load_ps(&dKTile[r][8]);
                                __m256 tk2 = _mm256_load_ps(&dKTile[r][16]);
                                __m256 tk3 = _mm256_load_ps(&dKTile[r][24]);
                                tk0 = _mm256_fmadd_ps(aqk, q0, tk0);
                                tk1 = _mm256_fmadd_ps(aqk, q1, tk1);
                                tk2 = _mm256_fmadd_ps(aqk, q2, tk2);
                                tk3 = _mm256_fmadd_ps(aqk, q3, tk3);
                                _mm256_store_ps(&dKTile[r][0], tk0);
                                _mm256_store_ps(&dKTile[r][8], tk1);
                                _mm256_store_ps(&dKTile[r][16], tk2);
                                _mm256_store_ps(&dKTile[r][24], tk3);
                            }

                            // write dQ once for this base block
                            _mm256_storeu_ps(dQBase + 0, dq0);
                            _mm256_storeu_ps(dQBase + 8, dq1);
                            _mm256_storeu_ps(dQBase + 16, dq2);
                            _mm256_storeu_ps(dQBase + 24, dq3);

                            // flush local tile scratch back to global dK/dV once
                            for (std::size_t j = j0; j < j1; ++j) {
                                const std::size_t r = j - j0;

                                float* dKBase = dKSlice + j * kvRowStride + base;
                                float* dVBase = dVSlice + j * kvRowStride + base;

                                __m256 dk0 = _mm256_load_ps(&dKTile[r][0]);
                                __m256 dk1 = _mm256_load_ps(&dKTile[r][8]);
                                __m256 dk2 = _mm256_load_ps(&dKTile[r][16]);
                                __m256 dk3 = _mm256_load_ps(&dKTile[r][24]);

                                __m256 dv0 = _mm256_load_ps(&dVTile[r][0]);
                                __m256 dv1 = _mm256_load_ps(&dVTile[r][8]);
                                __m256 dv2 = _mm256_load_ps(&dVTile[r][16]);
                                __m256 dv3 = _mm256_load_ps(&dVTile[r][24]);

                                if (!(kvRowStartsZero && t == j)) {
                                    dk0 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 0), dk0);
                                    dk1 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 8), dk1);
                                    dk2 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 16), dk2);
                                    dk3 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 24), dk3);

                                    dv0 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 0), dv0);
                                    dv1 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 8), dv1);
                                    dv2 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 16), dv2);
                                    dv3 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 24), dv3);
                                }

                                _mm256_storeu_ps(dKBase + 0, dk0);
                                _mm256_storeu_ps(dKBase + 8, dk1);
                                _mm256_storeu_ps(dKBase + 16, dk2);
                                _mm256_storeu_ps(dKBase + 24, dk3);

                                _mm256_storeu_ps(dVBase + 0, dv0);
                                _mm256_storeu_ps(dVBase + 8, dv1);
                                _mm256_storeu_ps(dVBase + 16, dv2);
                                _mm256_storeu_ps(dVBase + 24, dv3);
                            }
                        }
                    }
                }
                else {
                    for (std::size_t j = begin; j < prefix; ++j) {
                        const float pj = pr[j];
                        const float dS_raw = pj * (tmpLocal[j] - sum);
                        const float dS = std::max(-kDSClamp, std::min(kDSClamp, dS_raw));
                        const float aQK = dS * inv_sqrt_d;
                        axpy_D(dVSlice + j * kvRowStride, dCt, pj, D);
                        axpy_D(dQt, k + j * kvDim + kvOffset, aQK, D);
                        axpy_D(dKSlice + j * kvRowStride, Qt, aQK, D);
                    }
                }
            }
            else {
                for (std::size_t j = begin; j < prefix; ++j) {
                    if (attentionMask[j] == 0) continue;

                    const float pj = pr[j];
                    const float dS_raw = pj * (tmpLocal[j] - sum);
                    const float dS = std::max(-kDSClamp, std::min(kDSClamp, dS_raw));
                    const float aQK = dS * inv_sqrt_d;
                    axpy_D(dVSlice + j * kvRowStride, dCt, pj, D);
                    axpy_D(dQt, k + j * kvDim + kvOffset, aQK, D);
                    axpy_D(dKSlice + j * kvRowStride, Qt, aQK, D);
                }
            }
        }
    };

    auto backward_group4_fused_d128 = [&](
        std::size_t kv_h,
        std::size_t headBegin,
        float* __restrict dKLocal,
        float* __restrict dVLocal,
        float* __restrict tmp0,
        float* __restrict tmp1,
        float* __restrict tmp2,
        float* __restrict tmp3) {
            constexpr float kDSClamp = 10.0f;
            constexpr std::size_t JB = 16;

            const std::size_t head0 = headBegin + 0;
            const std::size_t head1 = headBegin + 1;
            const std::size_t head2 = headBegin + 2;
            const std::size_t head3 = headBegin + 3;
            const std::size_t kvOffset = kv_h * D;

            for (std::size_t t = 0; t < T; ++t) {
                const std::size_t prefix = t + 1;

                const float* pr0 = probs + (head0 * T * T) + (t * T);
                const float* pr1 = probs + (head1 * T * T) + (t * T);
                const float* pr2 = probs + (head2 * T * T) + (t * T);
                const float* pr3 = probs + (head3 * T * T) + (t * T);

                const float* dCt0 = dCtx + t * H + head0 * D;
                const float* dCt1 = dCtx + t * H + head1 * D;
                const float* dCt2 = dCtx + t * H + head2 * D;
                const float* dCt3 = dCtx + t * H + head3 * D;

                const float* Qt0 = q + t * H + head0 * D;
                const float* Qt1 = q + t * H + head1 * D;
                const float* Qt2 = q + t * H + head2 * D;
                const float* Qt3 = q + t * H + head3 * D;

                float* dQt0 = dQ + t * H + head0 * D;
                float* dQt1 = dQ + t * H + head1 * D;
                float* dQt2 = dQ + t * H + head2 * D;
                float* dQt3 = dQ + t * H + head3 * D;

                // --------------------------------------------------
                // Pass 1: compute dp and store tmp0..tmp3, plus sums
                // --------------------------------------------------
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                float sum3 = 0.0f;

                for (std::size_t j = 0; j < prefix; ++j) {
                    const float* vRow = v + j * kvDim + kvOffset;

                    float dp0, dp1, dp2, dp3;
                    dot4_128_against_one(dCt0, dCt1, dCt2, dCt3, vRow, dp0, dp1, dp2, dp3);

                    tmp0[j] = dp0;
                    tmp1[j] = dp1;
                    tmp2[j] = dp2;
                    tmp3[j] = dp3;

                    sum0 += pr0[j] * dp0;
                    sum1 += pr1[j] * dp1;
                    sum2 += pr2[j] * dp2;
                    sum3 += pr3[j] * dp3;
                }

                if (usePrecomputeDsD128) {
                    for (std::size_t j = 0; j < prefix; ++j) {
                        tmp0[j] = std::max(-kDSClamp, std::min(kDSClamp, pr0[j] * (tmp0[j] - sum0))) * inv_sqrt_d;
                        tmp1[j] = std::max(-kDSClamp, std::min(kDSClamp, pr1[j] * (tmp1[j] - sum1))) * inv_sqrt_d;
                        tmp2[j] = std::max(-kDSClamp, std::min(kDSClamp, pr2[j] * (tmp2[j] - sum2))) * inv_sqrt_d;
                        tmp3[j] = std::max(-kDSClamp, std::min(kDSClamp, pr3[j] * (tmp3[j] - sum3))) * inv_sqrt_d;
                    }
                }

                if (useGrouped4SplitDqDkvD128) {
                    if (!usePrecomputeDsD128) {
                        for (std::size_t j = 0; j < prefix; ++j) {
                            tmp0[j] = std::max(-kDSClamp, std::min(kDSClamp, pr0[j] * (tmp0[j] - sum0))) * inv_sqrt_d;
                            tmp1[j] = std::max(-kDSClamp, std::min(kDSClamp, pr1[j] * (tmp1[j] - sum1))) * inv_sqrt_d;
                            tmp2[j] = std::max(-kDSClamp, std::min(kDSClamp, pr2[j] * (tmp2[j] - sum2))) * inv_sqrt_d;
                            tmp3[j] = std::max(-kDSClamp, std::min(kDSClamp, pr3[j] * (tmp3[j] - sum3))) * inv_sqrt_d;
                        }
                    }

                    // Pass 2a: dQ only. Keeping the four dQ rows resident avoids
                    // carrying q/dCtx state at the same time and removes per-tile
                    // dQ reload/store traffic from the fused path.
                    for (std::size_t base = 0; base < D; base += 32) {
                        float* dQBase0 = dQt0 + base;
                        float* dQBase1 = dQt1 + base;
                        float* dQBase2 = dQt2 + base;
                        float* dQBase3 = dQt3 + base;

                        __m256 dq00 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase0 + 0);
                        __m256 dq01 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase0 + 8);
                        __m256 dq02 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase0 + 16);
                        __m256 dq03 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase0 + 24);

                        __m256 dq10 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase1 + 0);
                        __m256 dq11 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase1 + 8);
                        __m256 dq12 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase1 + 16);
                        __m256 dq13 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase1 + 24);

                        __m256 dq20 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase2 + 0);
                        __m256 dq21 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase2 + 8);
                        __m256 dq22 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase2 + 16);
                        __m256 dq23 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase2 + 24);

                        __m256 dq30 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase3 + 0);
                        __m256 dq31 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase3 + 8);
                        __m256 dq32 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase3 + 16);
                        __m256 dq33 = useDqOverwriteD128 ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase3 + 24);

                        for (std::size_t j = 0; j < prefix; ++j) {
                            const __m256 aqk0 = _mm256_set1_ps(tmp0[j]);
                            const __m256 aqk1 = _mm256_set1_ps(tmp1[j]);
                            const __m256 aqk2 = _mm256_set1_ps(tmp2[j]);
                            const __m256 aqk3 = _mm256_set1_ps(tmp3[j]);

                            const float* kBase = k + j * kvDim + kvOffset + base;
                            const __m256 k0 = _mm256_loadu_ps(kBase + 0);
                            const __m256 k1 = _mm256_loadu_ps(kBase + 8);
                            const __m256 k2 = _mm256_loadu_ps(kBase + 16);
                            const __m256 k3 = _mm256_loadu_ps(kBase + 24);

                            dq00 = _mm256_fmadd_ps(aqk0, k0, dq00);
                            dq01 = _mm256_fmadd_ps(aqk0, k1, dq01);
                            dq02 = _mm256_fmadd_ps(aqk0, k2, dq02);
                            dq03 = _mm256_fmadd_ps(aqk0, k3, dq03);

                            dq10 = _mm256_fmadd_ps(aqk1, k0, dq10);
                            dq11 = _mm256_fmadd_ps(aqk1, k1, dq11);
                            dq12 = _mm256_fmadd_ps(aqk1, k2, dq12);
                            dq13 = _mm256_fmadd_ps(aqk1, k3, dq13);

                            dq20 = _mm256_fmadd_ps(aqk2, k0, dq20);
                            dq21 = _mm256_fmadd_ps(aqk2, k1, dq21);
                            dq22 = _mm256_fmadd_ps(aqk2, k2, dq22);
                            dq23 = _mm256_fmadd_ps(aqk2, k3, dq23);

                            dq30 = _mm256_fmadd_ps(aqk3, k0, dq30);
                            dq31 = _mm256_fmadd_ps(aqk3, k1, dq31);
                            dq32 = _mm256_fmadd_ps(aqk3, k2, dq32);
                            dq33 = _mm256_fmadd_ps(aqk3, k3, dq33);
                        }

                        _mm256_storeu_ps(dQBase0 + 0, dq00);
                        _mm256_storeu_ps(dQBase0 + 8, dq01);
                        _mm256_storeu_ps(dQBase0 + 16, dq02);
                        _mm256_storeu_ps(dQBase0 + 24, dq03);
                        _mm256_storeu_ps(dQBase1 + 0, dq10);
                        _mm256_storeu_ps(dQBase1 + 8, dq11);
                        _mm256_storeu_ps(dQBase1 + 16, dq12);
                        _mm256_storeu_ps(dQBase1 + 24, dq13);
                        _mm256_storeu_ps(dQBase2 + 0, dq20);
                        _mm256_storeu_ps(dQBase2 + 8, dq21);
                        _mm256_storeu_ps(dQBase2 + 16, dq22);
                        _mm256_storeu_ps(dQBase2 + 24, dq23);
                        _mm256_storeu_ps(dQBase3 + 0, dq30);
                        _mm256_storeu_ps(dQBase3 + 8, dq31);
                        _mm256_storeu_ps(dQBase3 + 16, dq32);
                        _mm256_storeu_ps(dQBase3 + 24, dq33);
                    }

                    // Pass 2b: dK/dV only. This keeps only q and dCtx live.
                    for (std::size_t j0 = 0; j0 < prefix; j0 += JB) {
                        const std::size_t j1 = std::min(j0 + JB, prefix);
                        const std::size_t tileRows = j1 - j0;

                        for (std::size_t base = 0; base < D; base += 32) {
                            alignas(32) float dKTile[JB][32];
                            alignas(32) float dVTile[JB][32];
                            for (std::size_t r = 0; r < tileRows; ++r) {
                                for (std::size_t x = 0; x < 32; ++x) {
                                    dKTile[r][x] = 0.0f;
                                    dVTile[r][x] = 0.0f;
                                }
                            }

                            const __m256 q00 = _mm256_loadu_ps(Qt0 + base + 0);
                            const __m256 q01 = _mm256_loadu_ps(Qt0 + base + 8);
                            const __m256 q02 = _mm256_loadu_ps(Qt0 + base + 16);
                            const __m256 q03 = _mm256_loadu_ps(Qt0 + base + 24);
                            const __m256 q10 = _mm256_loadu_ps(Qt1 + base + 0);
                            const __m256 q11 = _mm256_loadu_ps(Qt1 + base + 8);
                            const __m256 q12 = _mm256_loadu_ps(Qt1 + base + 16);
                            const __m256 q13 = _mm256_loadu_ps(Qt1 + base + 24);
                            const __m256 q20 = _mm256_loadu_ps(Qt2 + base + 0);
                            const __m256 q21 = _mm256_loadu_ps(Qt2 + base + 8);
                            const __m256 q22 = _mm256_loadu_ps(Qt2 + base + 16);
                            const __m256 q23 = _mm256_loadu_ps(Qt2 + base + 24);
                            const __m256 q30 = _mm256_loadu_ps(Qt3 + base + 0);
                            const __m256 q31 = _mm256_loadu_ps(Qt3 + base + 8);
                            const __m256 q32 = _mm256_loadu_ps(Qt3 + base + 16);
                            const __m256 q33 = _mm256_loadu_ps(Qt3 + base + 24);

                            const __m256 dc00 = _mm256_loadu_ps(dCt0 + base + 0);
                            const __m256 dc01 = _mm256_loadu_ps(dCt0 + base + 8);
                            const __m256 dc02 = _mm256_loadu_ps(dCt0 + base + 16);
                            const __m256 dc03 = _mm256_loadu_ps(dCt0 + base + 24);
                            const __m256 dc10 = _mm256_loadu_ps(dCt1 + base + 0);
                            const __m256 dc11 = _mm256_loadu_ps(dCt1 + base + 8);
                            const __m256 dc12 = _mm256_loadu_ps(dCt1 + base + 16);
                            const __m256 dc13 = _mm256_loadu_ps(dCt1 + base + 24);
                            const __m256 dc20 = _mm256_loadu_ps(dCt2 + base + 0);
                            const __m256 dc21 = _mm256_loadu_ps(dCt2 + base + 8);
                            const __m256 dc22 = _mm256_loadu_ps(dCt2 + base + 16);
                            const __m256 dc23 = _mm256_loadu_ps(dCt2 + base + 24);
                            const __m256 dc30 = _mm256_loadu_ps(dCt3 + base + 0);
                            const __m256 dc31 = _mm256_loadu_ps(dCt3 + base + 8);
                            const __m256 dc32 = _mm256_loadu_ps(dCt3 + base + 16);
                            const __m256 dc33 = _mm256_loadu_ps(dCt3 + base + 24);

                            for (std::size_t j = j0; j < j1; ++j) {
                                const std::size_t r = j - j0;
                                const __m256 pjv0 = _mm256_set1_ps(pr0[j]);
                                const __m256 pjv1 = _mm256_set1_ps(pr1[j]);
                                const __m256 pjv2 = _mm256_set1_ps(pr2[j]);
                                const __m256 pjv3 = _mm256_set1_ps(pr3[j]);
                                const __m256 aqk0 = _mm256_set1_ps(tmp0[j]);
                                const __m256 aqk1 = _mm256_set1_ps(tmp1[j]);
                                const __m256 aqk2 = _mm256_set1_ps(tmp2[j]);
                                const __m256 aqk3 = _mm256_set1_ps(tmp3[j]);

                                __m256 tv0 = _mm256_load_ps(&dVTile[r][0]);
                                __m256 tv1 = _mm256_load_ps(&dVTile[r][8]);
                                __m256 tv2 = _mm256_load_ps(&dVTile[r][16]);
                                __m256 tv3 = _mm256_load_ps(&dVTile[r][24]);
                                tv0 = _mm256_fmadd_ps(pjv0, dc00, tv0);
                                tv1 = _mm256_fmadd_ps(pjv0, dc01, tv1);
                                tv2 = _mm256_fmadd_ps(pjv0, dc02, tv2);
                                tv3 = _mm256_fmadd_ps(pjv0, dc03, tv3);
                                tv0 = _mm256_fmadd_ps(pjv1, dc10, tv0);
                                tv1 = _mm256_fmadd_ps(pjv1, dc11, tv1);
                                tv2 = _mm256_fmadd_ps(pjv1, dc12, tv2);
                                tv3 = _mm256_fmadd_ps(pjv1, dc13, tv3);
                                tv0 = _mm256_fmadd_ps(pjv2, dc20, tv0);
                                tv1 = _mm256_fmadd_ps(pjv2, dc21, tv1);
                                tv2 = _mm256_fmadd_ps(pjv2, dc22, tv2);
                                tv3 = _mm256_fmadd_ps(pjv2, dc23, tv3);
                                tv0 = _mm256_fmadd_ps(pjv3, dc30, tv0);
                                tv1 = _mm256_fmadd_ps(pjv3, dc31, tv1);
                                tv2 = _mm256_fmadd_ps(pjv3, dc32, tv2);
                                tv3 = _mm256_fmadd_ps(pjv3, dc33, tv3);
                                _mm256_store_ps(&dVTile[r][0], tv0);
                                _mm256_store_ps(&dVTile[r][8], tv1);
                                _mm256_store_ps(&dVTile[r][16], tv2);
                                _mm256_store_ps(&dVTile[r][24], tv3);

                                __m256 tk0 = _mm256_load_ps(&dKTile[r][0]);
                                __m256 tk1 = _mm256_load_ps(&dKTile[r][8]);
                                __m256 tk2 = _mm256_load_ps(&dKTile[r][16]);
                                __m256 tk3 = _mm256_load_ps(&dKTile[r][24]);
                                tk0 = _mm256_fmadd_ps(aqk0, q00, tk0);
                                tk1 = _mm256_fmadd_ps(aqk0, q01, tk1);
                                tk2 = _mm256_fmadd_ps(aqk0, q02, tk2);
                                tk3 = _mm256_fmadd_ps(aqk0, q03, tk3);
                                tk0 = _mm256_fmadd_ps(aqk1, q10, tk0);
                                tk1 = _mm256_fmadd_ps(aqk1, q11, tk1);
                                tk2 = _mm256_fmadd_ps(aqk1, q12, tk2);
                                tk3 = _mm256_fmadd_ps(aqk1, q13, tk3);
                                tk0 = _mm256_fmadd_ps(aqk2, q20, tk0);
                                tk1 = _mm256_fmadd_ps(aqk2, q21, tk1);
                                tk2 = _mm256_fmadd_ps(aqk2, q22, tk2);
                                tk3 = _mm256_fmadd_ps(aqk2, q23, tk3);
                                tk0 = _mm256_fmadd_ps(aqk3, q30, tk0);
                                tk1 = _mm256_fmadd_ps(aqk3, q31, tk1);
                                tk2 = _mm256_fmadd_ps(aqk3, q32, tk2);
                                tk3 = _mm256_fmadd_ps(aqk3, q33, tk3);
                                _mm256_store_ps(&dKTile[r][0], tk0);
                                _mm256_store_ps(&dKTile[r][8], tk1);
                                _mm256_store_ps(&dKTile[r][16], tk2);
                                _mm256_store_ps(&dKTile[r][24], tk3);
                            }

                            for (std::size_t j = j0; j < j1; ++j) {
                                const std::size_t r = j - j0;
                                const std::size_t rowOffset = j * D + base;
                                float* dKBase = dKLocal + rowOffset;
                                float* dVBase = dVLocal + rowOffset;
                                __m256 dk0 = _mm256_load_ps(&dKTile[r][0]);
                                __m256 dk1 = _mm256_load_ps(&dKTile[r][8]);
                                __m256 dk2 = _mm256_load_ps(&dKTile[r][16]);
                                __m256 dk3 = _mm256_load_ps(&dKTile[r][24]);
                                __m256 dv0 = _mm256_load_ps(&dVTile[r][0]);
                                __m256 dv1 = _mm256_load_ps(&dVTile[r][8]);
                                __m256 dv2 = _mm256_load_ps(&dVTile[r][16]);
                                __m256 dv3 = _mm256_load_ps(&dVTile[r][24]);
                                if (!(useGrouped4RowOverwrite && t == j)) {
                                    dk0 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 0), dk0);
                                    dk1 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 8), dk1);
                                    dk2 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 16), dk2);
                                    dk3 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 24), dk3);
                                    dv0 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 0), dv0);
                                    dv1 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 8), dv1);
                                    dv2 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 16), dv2);
                                    dv3 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 24), dv3);
                                }
                                _mm256_storeu_ps(dKBase + 0, dk0);
                                _mm256_storeu_ps(dKBase + 8, dk1);
                                _mm256_storeu_ps(dKBase + 16, dk2);
                                _mm256_storeu_ps(dKBase + 24, dk3);
                                _mm256_storeu_ps(dVBase + 0, dv0);
                                _mm256_storeu_ps(dVBase + 8, dv1);
                                _mm256_storeu_ps(dVBase + 16, dv2);
                                _mm256_storeu_ps(dVBase + 24, dv3);
                            }
                        }
                    }
                    continue;
                }

                // --------------------------------------------------
                // Pass 2: tiled consume using tmp0..tmp3
                // --------------------------------------------------
                for (std::size_t j0 = 0; j0 < prefix; j0 += JB) {
                    const std::size_t j1 = std::min(j0 + JB, prefix);
                    const std::size_t tileRows = j1 - j0;

                    for (std::size_t base = 0; base < D; base += 32) {
                        alignas(32) float dKTile[JB][32];
                        alignas(32) float dVTile[JB][32];

                        for (std::size_t r = 0; r < tileRows; ++r) {
                            for (std::size_t x = 0; x < 32; ++x) {
                                dKTile[r][x] = 0.0f;
                                dVTile[r][x] = 0.0f;
                            }
                        }

                        const float* qBase0 = Qt0 + base;
                        const float* qBase1 = Qt1 + base;
                        const float* qBase2 = Qt2 + base;
                        const float* qBase3 = Qt3 + base;

                        const float* dCtBase0 = dCt0 + base;
                        const float* dCtBase1 = dCt1 + base;
                        const float* dCtBase2 = dCt2 + base;
                        const float* dCtBase3 = dCt3 + base;

                        float* dQBase0 = dQt0 + base;
                        float* dQBase1 = dQt1 + base;
                        float* dQBase2 = dQt2 + base;
                        float* dQBase3 = dQt3 + base;

                        const __m256 q00 = _mm256_loadu_ps(qBase0 + 0);
                        const __m256 q01 = _mm256_loadu_ps(qBase0 + 8);
                        const __m256 q02 = _mm256_loadu_ps(qBase0 + 16);
                        const __m256 q03 = _mm256_loadu_ps(qBase0 + 24);

                        const __m256 q10 = _mm256_loadu_ps(qBase1 + 0);
                        const __m256 q11 = _mm256_loadu_ps(qBase1 + 8);
                        const __m256 q12 = _mm256_loadu_ps(qBase1 + 16);
                        const __m256 q13 = _mm256_loadu_ps(qBase1 + 24);

                        const __m256 q20 = _mm256_loadu_ps(qBase2 + 0);
                        const __m256 q21 = _mm256_loadu_ps(qBase2 + 8);
                        const __m256 q22 = _mm256_loadu_ps(qBase2 + 16);
                        const __m256 q23 = _mm256_loadu_ps(qBase2 + 24);

                        const __m256 q30 = _mm256_loadu_ps(qBase3 + 0);
                        const __m256 q31 = _mm256_loadu_ps(qBase3 + 8);
                        const __m256 q32 = _mm256_loadu_ps(qBase3 + 16);
                        const __m256 q33 = _mm256_loadu_ps(qBase3 + 24);

                        const __m256 dc00 = _mm256_loadu_ps(dCtBase0 + 0);
                        const __m256 dc01 = _mm256_loadu_ps(dCtBase0 + 8);
                        const __m256 dc02 = _mm256_loadu_ps(dCtBase0 + 16);
                        const __m256 dc03 = _mm256_loadu_ps(dCtBase0 + 24);

                        const __m256 dc10 = _mm256_loadu_ps(dCtBase1 + 0);
                        const __m256 dc11 = _mm256_loadu_ps(dCtBase1 + 8);
                        const __m256 dc12 = _mm256_loadu_ps(dCtBase1 + 16);
                        const __m256 dc13 = _mm256_loadu_ps(dCtBase1 + 24);

                        const __m256 dc20 = _mm256_loadu_ps(dCtBase2 + 0);
                        const __m256 dc21 = _mm256_loadu_ps(dCtBase2 + 8);
                        const __m256 dc22 = _mm256_loadu_ps(dCtBase2 + 16);
                        const __m256 dc23 = _mm256_loadu_ps(dCtBase2 + 24);

                        const __m256 dc30 = _mm256_loadu_ps(dCtBase3 + 0);
                        const __m256 dc31 = _mm256_loadu_ps(dCtBase3 + 8);
                        const __m256 dc32 = _mm256_loadu_ps(dCtBase3 + 16);
                        const __m256 dc33 = _mm256_loadu_ps(dCtBase3 + 24);

                        // dQ is accumulated across j-tiles for the same query row. The
                        // overwrite fast path is valid only for the first tile; later
                        // tiles must reload the partial dQ produced by earlier tiles.
                        const bool overwriteDqTile = useDqOverwriteD128 && j0 == 0;
                        __m256 dq00 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase0 + 0);
                        __m256 dq01 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase0 + 8);
                        __m256 dq02 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase0 + 16);
                        __m256 dq03 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase0 + 24);

                        __m256 dq10 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase1 + 0);
                        __m256 dq11 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase1 + 8);
                        __m256 dq12 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase1 + 16);
                        __m256 dq13 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase1 + 24);

                        __m256 dq20 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase2 + 0);
                        __m256 dq21 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase2 + 8);
                        __m256 dq22 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase2 + 16);
                        __m256 dq23 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase2 + 24);

                        __m256 dq30 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase3 + 0);
                        __m256 dq31 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase3 + 8);
                        __m256 dq32 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase3 + 16);
                        __m256 dq33 = overwriteDqTile ? _mm256_setzero_ps() : _mm256_loadu_ps(dQBase3 + 24);

                        for (std::size_t j = j0; j < j1; ++j) {
                            const std::size_t r = j - j0;

                            const float pj0 = pr0[j];
                            const float pj1 = pr1[j];
                            const float pj2 = pr2[j];
                            const float pj3 = pr3[j];

                            const float dS0 = usePrecomputeDsD128
                                ? tmp0[j]
                                : std::max(-kDSClamp, std::min(kDSClamp, pj0 * (tmp0[j] - sum0))) * inv_sqrt_d;
                            const float dS1 = usePrecomputeDsD128
                                ? tmp1[j]
                                : std::max(-kDSClamp, std::min(kDSClamp, pj1 * (tmp1[j] - sum1))) * inv_sqrt_d;
                            const float dS2 = usePrecomputeDsD128
                                ? tmp2[j]
                                : std::max(-kDSClamp, std::min(kDSClamp, pj2 * (tmp2[j] - sum2))) * inv_sqrt_d;
                            const float dS3 = usePrecomputeDsD128
                                ? tmp3[j]
                                : std::max(-kDSClamp, std::min(kDSClamp, pj3 * (tmp3[j] - sum3))) * inv_sqrt_d;

                            const __m256 pjv0 = _mm256_set1_ps(pj0);
                            const __m256 pjv1 = _mm256_set1_ps(pj1);
                            const __m256 pjv2 = _mm256_set1_ps(pj2);
                            const __m256 pjv3 = _mm256_set1_ps(pj3);

                            const __m256 aqk0 = _mm256_set1_ps(dS0);
                            const __m256 aqk1 = _mm256_set1_ps(dS1);
                            const __m256 aqk2 = _mm256_set1_ps(dS2);
                            const __m256 aqk3 = _mm256_set1_ps(dS3);

                            const float* kBase = k + j * kvDim + kvOffset + base;

                            const __m256 k0 = _mm256_loadu_ps(kBase + 0);
                            const __m256 k1 = _mm256_loadu_ps(kBase + 8);
                            const __m256 k2 = _mm256_loadu_ps(kBase + 16);
                            const __m256 k3 = _mm256_loadu_ps(kBase + 24);

                            dq00 = _mm256_fmadd_ps(aqk0, k0, dq00);
                            dq01 = _mm256_fmadd_ps(aqk0, k1, dq01);
                            dq02 = _mm256_fmadd_ps(aqk0, k2, dq02);
                            dq03 = _mm256_fmadd_ps(aqk0, k3, dq03);

                            dq10 = _mm256_fmadd_ps(aqk1, k0, dq10);
                            dq11 = _mm256_fmadd_ps(aqk1, k1, dq11);
                            dq12 = _mm256_fmadd_ps(aqk1, k2, dq12);
                            dq13 = _mm256_fmadd_ps(aqk1, k3, dq13);

                            dq20 = _mm256_fmadd_ps(aqk2, k0, dq20);
                            dq21 = _mm256_fmadd_ps(aqk2, k1, dq21);
                            dq22 = _mm256_fmadd_ps(aqk2, k2, dq22);
                            dq23 = _mm256_fmadd_ps(aqk2, k3, dq23);

                            dq30 = _mm256_fmadd_ps(aqk3, k0, dq30);
                            dq31 = _mm256_fmadd_ps(aqk3, k1, dq31);
                            dq32 = _mm256_fmadd_ps(aqk3, k2, dq32);
                            dq33 = _mm256_fmadd_ps(aqk3, k3, dq33);

                            __m256 tv0 = _mm256_load_ps(&dVTile[r][0]);
                            __m256 tv1 = _mm256_load_ps(&dVTile[r][8]);
                            __m256 tv2 = _mm256_load_ps(&dVTile[r][16]);
                            __m256 tv3 = _mm256_load_ps(&dVTile[r][24]);

                            tv0 = _mm256_fmadd_ps(pjv0, dc00, tv0);
                            tv1 = _mm256_fmadd_ps(pjv0, dc01, tv1);
                            tv2 = _mm256_fmadd_ps(pjv0, dc02, tv2);
                            tv3 = _mm256_fmadd_ps(pjv0, dc03, tv3);

                            tv0 = _mm256_fmadd_ps(pjv1, dc10, tv0);
                            tv1 = _mm256_fmadd_ps(pjv1, dc11, tv1);
                            tv2 = _mm256_fmadd_ps(pjv1, dc12, tv2);
                            tv3 = _mm256_fmadd_ps(pjv1, dc13, tv3);

                            tv0 = _mm256_fmadd_ps(pjv2, dc20, tv0);
                            tv1 = _mm256_fmadd_ps(pjv2, dc21, tv1);
                            tv2 = _mm256_fmadd_ps(pjv2, dc22, tv2);
                            tv3 = _mm256_fmadd_ps(pjv2, dc23, tv3);

                            tv0 = _mm256_fmadd_ps(pjv3, dc30, tv0);
                            tv1 = _mm256_fmadd_ps(pjv3, dc31, tv1);
                            tv2 = _mm256_fmadd_ps(pjv3, dc32, tv2);
                            tv3 = _mm256_fmadd_ps(pjv3, dc33, tv3);

                            _mm256_store_ps(&dVTile[r][0], tv0);
                            _mm256_store_ps(&dVTile[r][8], tv1);
                            _mm256_store_ps(&dVTile[r][16], tv2);
                            _mm256_store_ps(&dVTile[r][24], tv3);

                            __m256 tk0 = _mm256_load_ps(&dKTile[r][0]);
                            __m256 tk1 = _mm256_load_ps(&dKTile[r][8]);
                            __m256 tk2 = _mm256_load_ps(&dKTile[r][16]);
                            __m256 tk3 = _mm256_load_ps(&dKTile[r][24]);

                            tk0 = _mm256_fmadd_ps(aqk0, q00, tk0);
                            tk1 = _mm256_fmadd_ps(aqk0, q01, tk1);
                            tk2 = _mm256_fmadd_ps(aqk0, q02, tk2);
                            tk3 = _mm256_fmadd_ps(aqk0, q03, tk3);

                            tk0 = _mm256_fmadd_ps(aqk1, q10, tk0);
                            tk1 = _mm256_fmadd_ps(aqk1, q11, tk1);
                            tk2 = _mm256_fmadd_ps(aqk1, q12, tk2);
                            tk3 = _mm256_fmadd_ps(aqk1, q13, tk3);

                            tk0 = _mm256_fmadd_ps(aqk2, q20, tk0);
                            tk1 = _mm256_fmadd_ps(aqk2, q21, tk1);
                            tk2 = _mm256_fmadd_ps(aqk2, q22, tk2);
                            tk3 = _mm256_fmadd_ps(aqk2, q23, tk3);

                            tk0 = _mm256_fmadd_ps(aqk3, q30, tk0);
                            tk1 = _mm256_fmadd_ps(aqk3, q31, tk1);
                            tk2 = _mm256_fmadd_ps(aqk3, q32, tk2);
                            tk3 = _mm256_fmadd_ps(aqk3, q33, tk3);

                            _mm256_store_ps(&dKTile[r][0], tk0);
                            _mm256_store_ps(&dKTile[r][8], tk1);
                            _mm256_store_ps(&dKTile[r][16], tk2);
                            _mm256_store_ps(&dKTile[r][24], tk3);
                        }

                        _mm256_storeu_ps(dQBase0 + 0, dq00);
                        _mm256_storeu_ps(dQBase0 + 8, dq01);
                        _mm256_storeu_ps(dQBase0 + 16, dq02);
                        _mm256_storeu_ps(dQBase0 + 24, dq03);

                        _mm256_storeu_ps(dQBase1 + 0, dq10);
                        _mm256_storeu_ps(dQBase1 + 8, dq11);
                        _mm256_storeu_ps(dQBase1 + 16, dq12);
                        _mm256_storeu_ps(dQBase1 + 24, dq13);

                        _mm256_storeu_ps(dQBase2 + 0, dq20);
                        _mm256_storeu_ps(dQBase2 + 8, dq21);
                        _mm256_storeu_ps(dQBase2 + 16, dq22);
                        _mm256_storeu_ps(dQBase2 + 24, dq23);

                        _mm256_storeu_ps(dQBase3 + 0, dq30);
                        _mm256_storeu_ps(dQBase3 + 8, dq31);
                        _mm256_storeu_ps(dQBase3 + 16, dq32);
                        _mm256_storeu_ps(dQBase3 + 24, dq33);

                        for (std::size_t j = j0; j < j1; ++j) {
                            const std::size_t r = j - j0;
                            const std::size_t rowOffset = j * D + base;

                            float* dKBase = dKLocal + rowOffset;
                            float* dVBase = dVLocal + rowOffset;

                            __m256 dk0 = _mm256_load_ps(&dKTile[r][0]);
                            __m256 dk1 = _mm256_load_ps(&dKTile[r][8]);
                            __m256 dk2 = _mm256_load_ps(&dKTile[r][16]);
                            __m256 dk3 = _mm256_load_ps(&dKTile[r][24]);

                            __m256 dv0 = _mm256_load_ps(&dVTile[r][0]);
                            __m256 dv1 = _mm256_load_ps(&dVTile[r][8]);
                            __m256 dv2 = _mm256_load_ps(&dVTile[r][16]);
                            __m256 dv3 = _mm256_load_ps(&dVTile[r][24]);

                            // Row j is first touched at query t == j in causal
                            // attention, so the first contribution can overwrite
                            // stale local scratch instead of paying a full clear.
                            if (!(useGrouped4RowOverwrite && t == j)) {
                                dk0 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 0), dk0);
                                dk1 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 8), dk1);
                                dk2 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 16), dk2);
                                dk3 = _mm256_add_ps(_mm256_loadu_ps(dKBase + 24), dk3);

                                dv0 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 0), dv0);
                                dv1 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 8), dv1);
                                dv2 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 16), dv2);
                                dv3 = _mm256_add_ps(_mm256_loadu_ps(dVBase + 24), dv3);
                            }

                            _mm256_storeu_ps(dKBase + 0, dk0);
                            _mm256_storeu_ps(dKBase + 8, dk1);
                            _mm256_storeu_ps(dKBase + 16, dk2);
                            _mm256_storeu_ps(dKBase + 24, dk3);

                            _mm256_storeu_ps(dVBase + 0, dv0);
                            _mm256_storeu_ps(dVBase + 8, dv1);
                            _mm256_storeu_ps(dVBase + 16, dv2);
                            _mm256_storeu_ps(dVBase + 24, dv3);
                        }
                    }
                }
            }
    };

#if defined(_OPENMP)
    if (options.enableHeadParallel && nHeads > nKvHeads) {
        const int threadCount = omp_get_max_threads();
        std::vector<float> dKLocal(static_cast<std::size_t>(threadCount) * dkvSize, 0.0f);
        std::vector<float> dVLocal(static_cast<std::size_t>(threadCount) * dkvSize, 0.0f);

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            float* dKThread = dKLocal.data() + static_cast<std::size_t>(tid) * dkvSize;
            float* dVThread = dVLocal.data() + static_cast<std::size_t>(tid) * dkvSize;
            static thread_local std::vector<float> tmpLocal;
            tmpLocal.resize(T);

#pragma omp for schedule(static)
                for (int head_i = 0; head_i < static_cast<int>(nHeads); ++head_i) {
                backward_one_head(static_cast<std::size_t>(head_i), dKThread, dVThread, kvDim, false, tmpLocal, false);
                }
            }

#pragma omp parallel for schedule(static)
        for (int idx_i = 0; idx_i < static_cast<int>(dkvSize); ++idx_i) {
            const std::size_t idx = static_cast<std::size_t>(idx_i);
            float sumK = 0.0f;
            float sumV = 0.0f;
            for (int tid = 0; tid < threadCount; ++tid) {
                const std::size_t base = static_cast<std::size_t>(tid) * dkvSize + idx;
                sumK += dKLocal[base];
                sumV += dVLocal[base];
            }
            dK[idx] += sumK;
            dV[idx] += sumV;
        }
        return;
    }
#endif

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        static thread_local std::vector<float> tmpLocal;
        tmpLocal.resize(T);
        static thread_local std::vector<float> tmpLocalGroup4;
        if (useGrouped4FusedD128) {
            tmpLocalGroup4.resize(T * 4);
        }
        static thread_local std::vector<float> dKGroupLocal;
        static thread_local std::vector<float> dVGroupLocal;
        if (useGroupedLocalAccum) {
            dKGroupLocal.resize(T * D);
            dVGroupLocal.resize(T * D);
        }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int kv_i = 0; kv_i < static_cast<int>(nKvHeads); ++kv_i) {
            const std::size_t kv_h = static_cast<std::size_t>(kv_i);
            const std::size_t headBegin = kv_h * heads_per_group;
            const std::size_t headEnd = headBegin + heads_per_group;
            if (useGroupedLocalAccum) {
                const bool useRowOverwriteInit =
                    allActive &&
                    D == 128 &&
                    !useGrouped4FusedD128 &&
                    options.enableGroupedLocalRowOverwrite;
                const bool useGroup4RowOverwriteInit =
                    useGrouped4RowOverwrite &&
                    !useGroupLocalHeadCompare;
                std::vector<float> dKGroupRef;
                std::vector<float> dVGroupRef;
                std::vector<float> dQGroupSnapshot;
                std::vector<float> dKGroupHeadRef;
                std::vector<float> dVGroupHeadRef;
                std::vector<float> dQHeadSnapshot;
                std::vector<float> dQHeadRepeatSnapshot;
                std::vector<float> dKHeadLocalBefore;
                std::vector<float> dVHeadLocalBefore;
                std::vector<float> dKHeadLocalDelta;
                std::vector<float> dVHeadLocalDelta;
                std::vector<float> tmpLocalRef;
                if (useGroupLocalGroupCompare) {
                    dKGroupRef.assign(T * kvDim, 0.0f);
                    dVGroupRef.assign(T * kvDim, 0.0f);
                    dQGroupSnapshot.resize(T * heads_per_group * D);
                    tmpLocalRef.resize(T);

                    for (std::size_t t = 0; t < T; ++t) {
                        std::memcpy(
                            dQGroupSnapshot.data() + t * heads_per_group * D,
                            dQ + t * H + headBegin * D,
                            heads_per_group * D * sizeof(float));
                    }

                    for (std::size_t head = headBegin; head < headEnd; ++head) {
                        backward_one_head(
                            head,
                            dKGroupRef.data(),
                            dVGroupRef.data(),
                            kvDim,
                            false,
                            tmpLocalRef,
                            false);
                    }

                    for (std::size_t t = 0; t < T; ++t) {
                        std::memcpy(
                            dQ + t * H + headBegin * D,
                            dQGroupSnapshot.data() + t * heads_per_group * D,
                            heads_per_group * D * sizeof(float));
                    }
                }
                if (useGroupLocalHeadCompare) {
                    dKGroupHeadRef.assign(T * D, 0.0f);
                    dVGroupHeadRef.assign(T * D, 0.0f);
                    dQHeadSnapshot.resize(T * D);
                    dQHeadRepeatSnapshot.resize(T * D);
                    dKHeadLocalBefore.resize(T * D);
                    dVHeadLocalBefore.resize(T * D);
                    dKHeadLocalDelta.resize(T * D);
                    dVHeadLocalDelta.resize(T * D);
                    tmpLocalRef.resize(T);
                }
                if (!useRowOverwriteInit && !useGroup4RowOverwriteInit) {
                    dKGroupLocal.assign(T * D, 0.0f);
                    dVGroupLocal.assign(T * D, 0.0f);
                }
                if (useGroupLocalHeadCompare) {
                    std::vector<float> zeroSlice(T * D, 0.0f);
                    std::fprintf(
                        stderr,
                        "[ATTN_BWD_HEAD_COMPARE] call=%zu kv=%zu prezero\n",
                        debugCallIndex,
                        kv_h);
                    report_backward_diff("pre_dK", zeroSlice.data(), dKGroupLocal.data(), T * D);
                    report_backward_diff("pre_dV", zeroSlice.data(), dVGroupLocal.data(), T * D);
                }

                if (useGrouped4FusedD128) {
                    backward_group4_fused_d128(
                        kv_h,
                        headBegin,
                        dKGroupLocal.data(),
                        dVGroupLocal.data(),
                        tmpLocalGroup4.data() + 0 * T,
                        tmpLocalGroup4.data() + 1 * T,
                        tmpLocalGroup4.data() + 2 * T,
                        tmpLocalGroup4.data() + 3 * T);
                }
                else {
                    bool firstHead = true;
                    for (std::size_t head = headBegin; head < headEnd; ++head) {
                        if (useGroupLocalHeadCompare) {
                            std::fill(dKGroupHeadRef.begin(), dKGroupHeadRef.end(), 0.0f);
                            std::fill(dVGroupHeadRef.begin(), dVGroupHeadRef.end(), 0.0f);
                            for (std::size_t t = 0; t < T; ++t) {
                                std::memcpy(
                                    dQHeadSnapshot.data() + t * D,
                                    dQ + t * H + head * D,
                                    D * sizeof(float));
                            }

                            backward_one_head(
                                head,
                                dKGroupHeadRef.data(),
                                dVGroupHeadRef.data(),
                                D,
                                true,
                                tmpLocalRef,
                                false);

                            for (std::size_t t = 0; t < T; ++t) {
                                std::memcpy(
                                    dQ + t * H + head * D,
                                    dQHeadSnapshot.data() + t * D,
                                    D * sizeof(float));
                            }

                            if (firstHead) {
                                std::vector<float> dKRepeatA(T * D, 0.0f);
                                std::vector<float> dVRepeatA(T * D, 0.0f);
                                std::vector<float> dKRepeatB(T * D, 0.0f);
                                std::vector<float> dVRepeatB(T * D, 0.0f);

                                for (std::size_t t = 0; t < T; ++t) {
                                    std::memcpy(
                                        dQHeadRepeatSnapshot.data() + t * D,
                                        dQ + t * H + head * D,
                                        D * sizeof(float));
                                }
                                backward_one_head(
                                    head,
                                    dKRepeatA.data(),
                                    dVRepeatA.data(),
                                    D,
                                    true,
                                    tmpLocalRef,
                                    false);
                                for (std::size_t t = 0; t < T; ++t) {
                                    std::memcpy(
                                        dQ + t * H + head * D,
                                        dQHeadRepeatSnapshot.data() + t * D,
                                        D * sizeof(float));
                                }

                                backward_one_head(
                                    head,
                                    dKRepeatB.data(),
                                    dVRepeatB.data(),
                                    D,
                                    true,
                                    tmpLocalRef,
                                    false);
                                for (std::size_t t = 0; t < T; ++t) {
                                    std::memcpy(
                                        dQ + t * H + head * D,
                                        dQHeadRepeatSnapshot.data() + t * D,
                                        D * sizeof(float));
                                }

                                std::fprintf(
                                    stderr,
                                    "[ATTN_BWD_HEAD_COMPARE] call=%zu kv=%zu head=%zu self_repeat\n",
                                    debugCallIndex,
                                    kv_h,
                                    head);
                                report_backward_diff("self_dK", dKRepeatA.data(), dKRepeatB.data(), T * D);
                                report_backward_diff("self_dV", dVRepeatA.data(), dVRepeatB.data(), T * D);
                            }
                        }

                        if (useGroupLocalHeadCompare) {
                            std::copy(dKGroupLocal.begin(), dKGroupLocal.end(), dKHeadLocalBefore.begin());
                            std::copy(dVGroupLocal.begin(), dVGroupLocal.end(), dVHeadLocalBefore.begin());
                        }

                        backward_one_head(
                            head,
                            dKGroupLocal.data(),
                            dVGroupLocal.data(),
                            D,
                            true,
                            tmpLocal,
                            useRowOverwriteInit && firstHead);

                        if (useGroupLocalHeadCompare) {
                            std::fprintf(
                                stderr,
                                "[ATTN_BWD_HEAD_COMPARE] call=%zu kv=%zu head=%zu local_head=%zu\n",
                                debugCallIndex,
                                kv_h,
                                head,
                                head - headBegin);
                            const bool compareWholeLocal = useRowOverwriteInit && firstHead;
                            if (compareWholeLocal) {
                                report_backward_diff("head_dK", dKGroupHeadRef.data(), dKGroupLocal.data(), T * D);
                                report_backward_diff("head_dV", dVGroupHeadRef.data(), dVGroupLocal.data(), T * D);
                            }
                            else {
                                for (std::size_t i = 0; i < T * D; ++i) {
                                    dKHeadLocalDelta[i] = dKGroupLocal[i] - dKHeadLocalBefore[i];
                                    dVHeadLocalDelta[i] = dVGroupLocal[i] - dVHeadLocalBefore[i];
                                }
                                report_backward_diff("head_dK", dKGroupHeadRef.data(), dKHeadLocalDelta.data(), T * D);
                                report_backward_diff("head_dV", dVGroupHeadRef.data(), dVHeadLocalDelta.data(), T * D);
                            }
                        }
                        firstHead = false;
                    }
                }

                if (useGroupLocalGroupCompare) {
                    std::vector<float> dKRefSlice(T * D, 0.0f);
                    std::vector<float> dVRefSlice(T * D, 0.0f);
                    for (std::size_t t = 0; t < T; ++t) {
                        std::copy_n(
                            dKGroupRef.data() + t * kvDim + kv_h * D,
                            D,
                            dKRefSlice.data() + t * D);
                        std::copy_n(
                            dVGroupRef.data() + t * kvDim + kv_h * D,
                            D,
                            dVRefSlice.data() + t * D);
                    }

                    std::fprintf(
                        stderr,
                        "[ATTN_BWD_GROUP_COMPARE] call=%zu kv=%zu heads=[%zu,%zu)\n",
                        debugCallIndex,
                        kv_h,
                        headBegin,
                        headEnd);
                    report_backward_diff("group_dK", dKRefSlice.data(), dKGroupLocal.data(), T * D);
                    report_backward_diff("group_dV", dVRefSlice.data(), dVGroupLocal.data(), T * D);
                }

                for (std::size_t t = 0; t < T; ++t) {
                    float* dKDst = dK + t * kvDim + kv_h * D;
                    float* dVDst = dV + t * kvDim + kv_h * D;
                    const float* dKSrc = dKGroupLocal.data() + t * D;
                    const float* dVSrc = dVGroupLocal.data() + t * D;
                    std::copy_n(dKSrc, D, dKDst);
                    std::copy_n(dVSrc, D, dVDst);
                }
            }
            else {
                for (std::size_t head = headBegin; head < headEnd; ++head) {
                    backward_one_head(head, dK, dV, kvDim, false, tmpLocal, false);
                }
            }
        }
    }
}

void attn_causal_backward(
    float* __restrict dQ,
    float* __restrict dK,
    float* __restrict dV,
    const float* __restrict dCtx,
    const float* __restrict q,
    const float* __restrict k,
    const float* __restrict v,
    std::size_t T,
    std::size_t H,
    std::size_t nHeads,
    std::size_t nKvHeads,
    const float* __restrict probs,
    const int* attentionMask,
    float* __restrict tmp_dP)
{
    const AttnBackwardOptions defaultOptions = default_attn_backward_options();
    static std::size_t debugCallCounter = 0;
    const std::size_t debugCallIndex = debugCallCounter++;
    if (!attn_backward_debug_compare_enabled()) {
        attn_causal_backward_impl(
            dQ, dK, dV,
            dCtx, q, k, v,
            T, H, nHeads, nKvHeads,
            probs, attentionMask, tmp_dP,
            defaultOptions,
            debugCallIndex);
        return;
    }

    static bool compareConsumed = false;
    if (attn_backward_debug_compare_once_enabled() && compareConsumed) {
        attn_causal_backward_impl(
            dQ, dK, dV,
            dCtx, q, k, v,
            T, H, nHeads, nKvHeads,
            probs, attentionMask, tmp_dP,
            defaultOptions,
            debugCallIndex);
        return;
    }
    compareConsumed = true;

    AttnBackwardOptions baselineOptions = baseline_attn_backward_options();

    AttnBackwardOptions candidateOptions = defaultOptions;
    apply_debug_candidate_overrides(candidateOptions);

    const std::size_t kvDim = (nKvHeads == 0 || H == 0 || nHeads == 0) ? 0 : (H / nHeads) * nKvHeads;
    std::vector<float> dQRef(T * H, 0.0f);
    std::vector<float> dKRef(T * kvDim, 0.0f);
    std::vector<float> dVRef(T * kvDim, 0.0f);
    std::vector<float> tmpRef(T, 0.0f);

    attn_causal_backward_impl(
        dQRef.data(), dKRef.data(), dVRef.data(),
        dCtx, q, k, v,
        T, H, nHeads, nKvHeads,
        probs, attentionMask, tmpRef.data(),
        baselineOptions,
        debugCallIndex);

    std::fill_n(dQ, T * H, 0.0f);
    std::fill_n(dK, T * kvDim, 0.0f);
    std::fill_n(dV, T * kvDim, 0.0f);
    if (tmp_dP) {
        std::fill_n(tmp_dP, T, 0.0f);
    }

    attn_causal_backward_impl(
        dQ, dK, dV,
        dCtx, q, k, v,
        T, H, nHeads, nKvHeads,
        probs, attentionMask, tmp_dP,
        candidateOptions,
        debugCallIndex);

    std::fprintf(
        stderr,
        "[ATTN_BWD_COMPARE] candidate={group_local=%d,row_overwrite=%d,group4_fused=%d,group4_row_overwrite=%d,precompute_ds=%d,group4_split=%d,dq_overwrite=%d,head_parallel=%d} baseline={group_local=%d}\n",
        candidateOptions.enableGroupedLocalAccum ? 1 : 0,
        candidateOptions.enableGroupedLocalRowOverwrite ? 1 : 0,
        candidateOptions.enableGrouped4FusedD128 ? 1 : 0,
        candidateOptions.enableGrouped4RowOverwrite ? 1 : 0,
        candidateOptions.enablePrecomputeDsD128 ? 1 : 0,
        candidateOptions.enableGrouped4SplitDqDkvD128 ? 1 : 0,
        candidateOptions.enableDqOverwriteD128 ? 1 : 0,
        candidateOptions.enableHeadParallel ? 1 : 0,
        baselineOptions.enableGroupedLocalAccum ? 1 : 0);
    report_backward_diff("dQ", dQRef.data(), dQ, T * H);
    report_backward_diff("dK", dKRef.data(), dK, T * kvDim);
    report_backward_diff("dV", dVRef.data(), dV, T * kvDim);
}
