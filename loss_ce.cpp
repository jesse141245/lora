#include "loss_ce.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

#if defined(__AVX2__)
    static inline float hmax256_ps(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 m = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0x55));
        return _mm_cvtss_f32(m);
    }

    static inline float hsum256_ps(__m256 v) {
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

    static inline float max_avx2_or_scalar(const float* x, std::size_t n) {
        if (!x) {
            throw std::runtime_error("max_avx2_or_scalar: x is null");
        }
        if (n == 0) {
            throw std::runtime_error("max_avx2_or_scalar: n must be > 0");
        }

#if defined(__AVX2__)
        std::size_t i = 0;
        __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

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
        float m = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            m = std::max(m, x[i]);
        }
        return m;
#endif
    }

    static inline float sum_avx2_or_scalar(const float* x, std::size_t n) {
        if (!x) {
            throw std::runtime_error("sum_avx2_or_scalar: x is null");
        }

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

void scale_inplace_avx2_or_scalar(float* x, std::size_t n, float scale) {
    if (!x && n != 0) {
        throw std::runtime_error("scale_inplace_avx2_or_scalar: x is null");
    }

#if defined(__AVX2__)
    std::size_t i = 0;
    const __m256 sv = _mm256_set1_ps(scale);

    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        xv = _mm256_mul_ps(xv, sv);
        _mm256_storeu_ps(x + i, xv);
    }

    for (; i < n; ++i) {
        x[i] *= scale;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        x[i] *= scale;
    }
#endif
}

float cross_entropy_dlogits_row(
    const float* logits,
    float* dlog,
    std::size_t V,
    int y)
{
    if (!logits) {
        throw std::runtime_error("cross_entropy_dlogits_row: logits is null");
    }
    if (!dlog) {
        throw std::runtime_error("cross_entropy_dlogits_row: dlog is null");
    }
    if (V == 0) {
        throw std::runtime_error("cross_entropy_dlogits_row: V must be > 0");
    }
    if (y < 0 || static_cast<std::size_t>(y) >= V) {
        throw std::runtime_error("cross_entropy_dlogits_row: target index out of range");
    }

    const float m = max_avx2_or_scalar(logits, V);

    // Scalar exp for exact/simple behavior.
    for (std::size_t v = 0; v < V; ++v) {
        dlog[v] = std::exp(logits[v] - m);
    }

    const float sumExp = sum_avx2_or_scalar(dlog, V);
    const float invSum = 1.0f / sumExp;

    scale_inplace_avx2_or_scalar(dlog, V, invSum);
    dlog[static_cast<std::size_t>(y)] -= 1.0f;

    return std::log(sumExp) + m - logits[static_cast<std::size_t>(y)];
}