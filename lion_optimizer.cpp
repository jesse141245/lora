#include "lion_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

#if defined(__AVX2__)
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

} // namespace

void LionOptimizer::init(const std::vector<ParamRef>& params)
{
    // SAFEGUARD: If already initialized to the exact right shape (e.g. via checkpoint), do not wipe.
    if (m_.size() == params.size()) {
        bool match = true;
        for (std::size_t i = 0; i < params.size(); ++i) {
            if (m_[i].size() != params[i].n) match = false;
        }
        if (match) return;
    }

    m_.clear();
    m_.reserve(params.size());

    for (const auto& p : params) {
        if (!p.w || !p.g) throw std::runtime_error("LionOptimizer::init: null w/g");
        if (p.n == 0)     throw std::runtime_error("LionOptimizer::init: zero-sized param");
        m_.emplace_back(p.n, 0.0f);
    }
}

void LionOptimizer::clipGradNorm(const std::vector<ParamRef>& params, float maxNorm)
{
    if (maxNorm <= 0.0f) return;
    if (params.size() != m_.size()) {
        throw std::runtime_error("LionOptimizer::clipGradNorm: params size changed; call init() again");
    }

    double ss = 0.0;

    for (const auto& p : params) {
        const float* g = p.g;
        const std::size_t n = p.n;

#if defined(__AVX2__)
        std::size_t i = 0;
        __m256 acc = _mm256_setzero_ps();

        for (; i + 8 <= n; i += 8) {
            __m256 gv = _mm256_loadu_ps(g + i);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(gv, gv));
        }

        ss += static_cast<double>(hsum256_ps(acc));

        for (; i < n; ++i) {
            const double v = static_cast<double>(g[i]);
            ss += v * v;
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(g[i]);
            ss += v * v;
        }
#endif
    }

    const double norm = std::sqrt(ss);
    if (norm == 0.0 || norm <= static_cast<double>(maxNorm)) return;

    const float scale = static_cast<float>(static_cast<double>(maxNorm) / norm);

    for (const auto& p : params) {
        float* g = p.g;
        const std::size_t n = p.n;

#if defined(__AVX2__)
        std::size_t i = 0;
        const __m256 sv = _mm256_set1_ps(scale);

        for (; i + 8 <= n; i += 8) {
            __m256 gv = _mm256_loadu_ps(g + i);
            gv = _mm256_mul_ps(gv, sv);
            _mm256_storeu_ps(g + i, gv);
        }

        for (; i < n; ++i) {
            g[i] *= scale;
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            g[i] *= scale;
        }
#endif
    }
}

void LionOptimizer::step(const std::vector<ParamRef>& params, bool zeroGrads)
{
    if (params.size() != m_.size()) {
        throw std::runtime_error("LionOptimizer::step: params size changed; call init() again");
    }

    // FIX: Use |lr| so the sign-OR trick in AVX2 is always correct,
    // even if a caller accidentally passes a negative learning rate.
    const float lr_abs = std::abs(lr_);

    for (std::size_t k = 0; k < params.size(); ++k) {
        float* w = params[k].w;
        float* g = params[k].g;
        auto& m = m_[k];

        const std::size_t n = params[k].n;
        if (m.size() != n) {
            throw std::runtime_error("LionOptimizer::step: momentum size mismatch");
        }

#if defined(__AVX2__)
        const __m256 beta1v = _mm256_set1_ps(beta1_);
        const __m256 one_mb1v = _mm256_set1_ps(1.0f - beta1_);
        const __m256 beta2v = _mm256_set1_ps(beta2_);
        const __m256 one_mb2v = _mm256_set1_ps(1.0f - beta2_);
        const __m256 wdv = _mm256_set1_ps(weightDecay_);
        const __m256 lrv = _mm256_set1_ps(lr_abs);  // always positive magnitude

        // Mask that isolates the IEEE-754 sign bit of each float lane.
        const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 wv = _mm256_loadu_ps(w + i);
            __m256 gv = _mm256_loadu_ps(g + i);
            __m256 mv = _mm256_loadu_ps(m.data() + i);  // m_old

            // ── 1. Weight-decayed gradient ───────────────────────────────────
            // grad = g + weightDecay * w
#if defined(__FMA__)
            __m256 grad = _mm256_fmadd_ps(wdv, wv, gv);
#else
            __m256 grad = _mm256_add_ps(gv, _mm256_mul_ps(wdv, wv));
#endif

            // ── 2. FIX: compute u from m_OLD, THEN update m ──────────────────
            //
            // Lion paper Algorithm 1 (Chen et al., 2023):
            //   Line 6:  u = β2·m + (1−β2)·grad       ← uses m BEFORE update
            //   Line 7:  m ← β1·m + (1−β1)·grad       ← update happens AFTER
            //   Line 8:  w ← w − lr·sign(u)
            //
            // The old code updated mv first then computed uv from the new mv,
            // introducing a one-step lag that made the optimizer less stable.

            // u = β2·m_old + (1−β2)·grad
            __m256 uv = _mm256_add_ps(
                _mm256_mul_ps(beta2v, mv),
                _mm256_mul_ps(one_mb2v, grad)
            );

            // m ← β1·m_old + (1−β1)·grad
            mv = _mm256_add_ps(
                _mm256_mul_ps(beta1v, mv),
                _mm256_mul_ps(one_mb1v, grad)
            );
            _mm256_storeu_ps(m.data() + i, mv);

            // ── 3. Weight update: w -= copysign(|lr|, u) ────────────────────
            // Extract sign bit from u and OR it onto the positive lr magnitude.
            __m256 signbits = _mm256_and_ps(uv, signmask);
            __m256 stepv = _mm256_or_ps(lrv, signbits);

            wv = _mm256_sub_ps(wv, stepv);
            _mm256_storeu_ps(w + i, wv);

            if (zeroGrads) {
                _mm256_storeu_ps(g + i, _mm256_setzero_ps());
            }
        }

        // ── Scalar tail (handles n % 8 remainder) ───────────────────────────
        for (; i < n; ++i) {
            const float grad = g[i] + weightDecay_ * w[i];
            // u uses m_old — same order as AVX2 path above
            const float u = beta2_ * m[i] + (1.0f - beta2_) * grad;
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * grad;
            w[i] -= std::copysign(lr_abs, u);
            if (zeroGrads) g[i] = 0.0f;
        }

#else  // ── Pure scalar fallback (no AVX2) ───────────────────────────────────

        for (std::size_t i = 0; i < n; ++i) {
            const float grad = g[i] + weightDecay_ * w[i];
            // u uses m_old
            const float u = beta2_ * m[i] + (1.0f - beta2_) * grad;
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * grad;
            w[i] -= std::copysign(lr_abs, u);
            if (zeroGrads) g[i] = 0.0f;
        }
#endif
    }
}