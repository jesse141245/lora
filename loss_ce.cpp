#include "loss_ce.h"
#include "kernels.h" // Pulls in scale_inplace_avx2_or_scalar safely

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

#if defined(__AVX2__)
    inline float hmax256_ps(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 m = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        // FIX: Use _mm_movehdup_ps instead of _mm_shuffle_ps to prevent IntelliSense __builtin_ia32 error
        m = _mm_max_ps(m, _mm_movehdup_ps(m));
        return _mm_cvtss_f32(m);
    }

    inline float hsum256_ps(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }
#endif

    inline float max_avx2_or_scalar(const float* x, std::size_t n) {
        if (!x) throw std::runtime_error("max_avx2_or_scalar: x is null");
        if (n == 0) throw std::runtime_error("max_avx2_or_scalar: n must be > 0");

#if defined(__AVX2__)
        std::size_t i = 0;
        // FIX: Use -1e38f instead of std::numeric_limits to bypass IntelliSense template glitch
        __m256 vmax = _mm256_set1_ps(-1e38f);

        for (; i + 8 <= n; i += 8) {
            const __m256 xv = _mm256_loadu_ps(x + i);
            vmax = _mm256_max_ps(vmax, xv);
        }

        float m = hmax256_ps(vmax);
        for (; i < n; ++i) {
            m = std::max(m, x[i]);
        }
        return m;
#else
        float m = -1e38f;
        for (std::size_t i = 0; i < n; ++i) {
            m = std::max(m, x[i]);
        }
        return m;
#endif
    }

    inline float sum_avx2_or_scalar(const float* x, std::size_t n) {
        if (!x) throw std::runtime_error("sum_avx2_or_scalar: x is null");

#if defined(__AVX2__)
        std::size_t i = 0;
        __m256 acc = _mm256_setzero_ps();

        for (; i + 8 <= n; i += 8) {
            const __m256 xv = _mm256_loadu_ps(x + i);
            acc = _mm256_add_ps(acc, xv);
        }

        float s = hsum256_ps(acc);
        for (; i < n; ++i) {
            s += x[i];
        }
        return s;
#else
        float s = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            s += x[i];
        }
        return s;
#endif
    }

} // namespace

// ============================================================
// Scaling & Normalization
// ============================================================

void normalize_gradients(float* grads, std::size_t n, float batch_size) {
    if (batch_size <= 1e-6f) return;
    scale_inplace_avx2_or_scalar(grads, n, 1.0f / batch_size);
}

// ============================================================
// Gradient Clipping
// ============================================================

void clip_gradients(float* grads, std::size_t n, float max_norm) {
    if (n == 0 || max_norm <= 0.0f) return;

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_sq += (double)grads[i] * (double)grads[i];
    }

    float norm = std::sqrt((float)sum_sq);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-9f);
        scale_inplace_avx2_or_scalar(grads, n, scale);
    }
}

// ============================================================
// Cross Entropy
// ============================================================

float cross_entropy_dlogits_row(
    const float* logits,
    float* dlog,
    std::size_t V,
    int y)
{
    if (!logits || !dlog) throw std::runtime_error("CE: null pointers");
    if (V == 0) return 0.0f;
    if (y < 0 || static_cast<std::size_t>(y) >= V) return 0.0f;

    // Numerical stability
    const float m = max_avx2_or_scalar(logits, V);

    for (std::size_t v = 0; v < V; ++v) {
        dlog[v] = std::exp(logits[v] - m);
    }

    const float sumExp = sum_avx2_or_scalar(dlog, V) + 1e-9f;
    const float invSum = 1.0f / sumExp;
    scale_inplace_avx2_or_scalar(dlog, V, invSum);

    // Gradient of CE w.r.t Logits: (Prob_i - Target_i)
    dlog[static_cast<std::size_t>(y)] -= 1.0f;

    return std::log(sumExp) + m - logits[static_cast<std::size_t>(y)];
}