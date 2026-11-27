#include "lion_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
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

    inline __m256 fmadd_ps(__m256 a, __m256 b, __m256 c) {
#if defined(__FMA__)
        return _mm256_fmadd_ps(a, b, c);
#else
        return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
    }

    inline __m256 copy_sign_ps(__m256 magnitude, __m256 sign_source, __m256 signmask) {
        return _mm256_or_ps(magnitude, _mm256_and_ps(sign_source, signmask));
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

    const ptrdiff_t paramCount = static_cast<ptrdiff_t>(params.size());
#if defined(_OPENMP)
#pragma omp parallel for reduction(+:ss) schedule(static) if(paramCount > 1)
#endif
    for (ptrdiff_t pIdx = 0; pIdx < paramCount; ++pIdx) {
        const auto& p = params[static_cast<std::size_t>(pIdx)];
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

    const ptrdiff_t scaleParamCount = static_cast<ptrdiff_t>(params.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(scaleParamCount > 1)
#endif
    for (ptrdiff_t pIdx = 0; pIdx < scaleParamCount; ++pIdx) {
        const auto& p = params[static_cast<std::size_t>(pIdx)];
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

    const float lr_abs = std::abs(lr_);

    const ptrdiff_t paramCount = static_cast<ptrdiff_t>(params.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(paramCount > 1)
#endif
    for (ptrdiff_t k_i = 0; k_i < paramCount; ++k_i) {
        const std::size_t k = static_cast<std::size_t>(k_i);
        float* w = params[k].w;
        float* g = params[k].g;
        auto& m = m_[k];
        const float param_lr_abs = lr_abs * params[k].lrScale;

        const std::size_t n = params[k].n;
        if (m.size() != n) {
            throw std::runtime_error("LionOptimizer::step: momentum size mismatch");
        }

#if defined(__AVX2__)
        float* const mPtr = m.data();
        const __m256 beta1v = _mm256_set1_ps(beta1_);
        const __m256 one_mb1v = _mm256_set1_ps(1.0f - beta1_);
        const __m256 beta2v = _mm256_set1_ps(beta2_);
        const __m256 one_mb2v = _mm256_set1_ps(1.0f - beta2_);
        const __m256 lrv = _mm256_set1_ps(param_lr_abs);
        const __m256 zerov = _mm256_setzero_ps();
        const bool useWeightDecay = weightDecay_ != 0.0f;
        const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
        const __m256 decayv = _mm256_set1_ps(1.0f - (param_lr_abs * weightDecay_));

        std::size_t i = 0;
        if (zeroGrads) {
            for (; i + 8 <= n; i += 8) {
                __m256 wv = _mm256_loadu_ps(w + i);
                const __m256 gv = _mm256_loadu_ps(g + i);
                const __m256 mv = _mm256_loadu_ps(mPtr + i);
                const __m256 updatev = fmadd_ps(one_mb1v, gv, _mm256_mul_ps(beta1v, mv));
                const __m256 newMv = fmadd_ps(one_mb2v, gv, _mm256_mul_ps(beta2v, mv));
                if (useWeightDecay) {
                    wv = _mm256_mul_ps(wv, decayv);
                }
                const __m256 stepv = copy_sign_ps(lrv, updatev, signmask);
                wv = _mm256_sub_ps(wv, stepv);
                _mm256_storeu_ps(w + i, wv);
                _mm256_storeu_ps(mPtr + i, newMv);
                _mm256_storeu_ps(g + i, zerov);
            }
        }
        else {
            for (; i + 8 <= n; i += 8) {
                __m256 wv = _mm256_loadu_ps(w + i);
                const __m256 gv = _mm256_loadu_ps(g + i);
                const __m256 mv = _mm256_loadu_ps(mPtr + i);
                const __m256 updatev = fmadd_ps(one_mb1v, gv, _mm256_mul_ps(beta1v, mv));
                const __m256 newMv = fmadd_ps(one_mb2v, gv, _mm256_mul_ps(beta2v, mv));
                if (useWeightDecay) {
                    wv = _mm256_mul_ps(wv, decayv);
                }
                const __m256 stepv = copy_sign_ps(lrv, updatev, signmask);
                wv = _mm256_sub_ps(wv, stepv);
                _mm256_storeu_ps(w + i, wv);
                _mm256_storeu_ps(mPtr + i, newMv);
            }
        }

        if (useWeightDecay) {
            const float decay = 1.0f - (param_lr_abs * weightDecay_);
            for (; i < n; ++i) {
                const float grad = g[i];
                const float update = beta1_ * m[i] + (1.0f - beta1_) * grad;
                w[i] *= decay;
                w[i] -= std::copysign(param_lr_abs, update);
                m[i] = beta2_ * m[i] + (1.0f - beta2_) * grad;
                if (zeroGrads) g[i] = 0.0f;
            }
        }
        else {
            for (; i < n; ++i) {
                const float grad = g[i];
                const float update = beta1_ * m[i] + (1.0f - beta1_) * grad;
                w[i] -= std::copysign(param_lr_abs, update);
                m[i] = beta2_ * m[i] + (1.0f - beta2_) * grad;
                if (zeroGrads) g[i] = 0.0f;
            }
        }

#else
        if (weightDecay_ != 0.0f) {
            const float decay = 1.0f - (param_lr_abs * weightDecay_);
            for (std::size_t i = 0; i < n; ++i) {
                const float grad = g[i];
                const float update = beta1_ * m[i] + (1.0f - beta1_) * grad;
                w[i] *= decay;
                w[i] -= std::copysign(param_lr_abs, update);
                m[i] = beta2_ * m[i] + (1.0f - beta2_) * grad;
                if (zeroGrads) g[i] = 0.0f;
            }
        }
        else {
            for (std::size_t i = 0; i < n; ++i) {
                const float grad = g[i];
                const float update = beta1_ * m[i] + (1.0f - beta1_) * grad;
                w[i] -= std::copysign(param_lr_abs, update);
                m[i] = beta2_ * m[i] + (1.0f - beta2_) * grad;
                if (zeroGrads) g[i] = 0.0f;
            }
        }
#endif
    }
}

void LionOptimizer::zeroState()
{
    for (auto& buf : m_) {
        std::fill(buf.begin(), buf.end(), 0.0f);
    }
    std::fill(mFlat_.begin(), mFlat_.end(), 0.0f);
}

void LionOptimizer::step(std::span<float> params, std::span<float> grads)
{
    if (params.size() != grads.size()) {
        throw std::runtime_error("LionOptimizer::step: params/grads size mismatch");
    }
    if (params.empty()) return;

    if (mFlat_.size() != params.size()) {
        mFlat_.assign(params.size(), 0.0f);
    }

    const float lr_abs = std::abs(lr_);
    const std::size_t n = params.size();
    float* const w = params.data();
    float* const g = grads.data();
    float* const m = mFlat_.data();

#if defined(__AVX2__)
    const __m256 beta1v = _mm256_set1_ps(beta1_);
    const __m256 one_mb1v = _mm256_set1_ps(1.0f - beta1_);
    const __m256 beta2v = _mm256_set1_ps(beta2_);
    const __m256 one_mb2v = _mm256_set1_ps(1.0f - beta2_);
    const __m256 lrv = _mm256_set1_ps(lr_abs);
    const __m256 zerov = _mm256_setzero_ps();
    const bool useWeightDecay = weightDecay_ != 0.0f;
    const __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    const __m256 decayv = _mm256_set1_ps(1.0f - (lr_abs * weightDecay_));

    std::size_t i = 0;
    if (useWeightDecay) {
        for (; i + 8 <= n; i += 8) {
            __m256 wv = _mm256_loadu_ps(w + i);
            const __m256 gv = _mm256_loadu_ps(g + i);
            const __m256 mv = _mm256_loadu_ps(m + i);
            const __m256 updatev = fmadd_ps(one_mb1v, gv, _mm256_mul_ps(beta1v, mv));
            const __m256 newMv = fmadd_ps(one_mb2v, gv, _mm256_mul_ps(beta2v, mv));
            wv = _mm256_mul_ps(wv, decayv);
            const __m256 stepv = copy_sign_ps(lrv, updatev, signmask);
            wv = _mm256_sub_ps(wv, stepv);
            _mm256_storeu_ps(w + i, wv);
            _mm256_storeu_ps(m + i, newMv);
            _mm256_storeu_ps(g + i, zerov);
        }
        const float decay = 1.0f - (lr_abs * weightDecay_);
        for (; i < n; ++i) {
            const float grad = g[i];
            const float update = beta1_ * m[i] + (1.0f - beta1_) * grad;
            w[i] *= decay;
            w[i] -= std::copysign(lr_abs, update);
            m[i] = beta2_ * m[i] + (1.0f - beta2_) * grad;
            g[i] = 0.0f;
        }
    }
    else {
        for (; i + 8 <= n; i += 8) {
            __m256 wv = _mm256_loadu_ps(w + i);
            const __m256 gv = _mm256_loadu_ps(g + i);
            const __m256 mv = _mm256_loadu_ps(m + i);
            const __m256 updatev = fmadd_ps(one_mb1v, gv, _mm256_mul_ps(beta1v, mv));
            const __m256 newMv = fmadd_ps(one_mb2v, gv, _mm256_mul_ps(beta2v, mv));
            const __m256 stepv = copy_sign_ps(lrv, updatev, signmask);
            wv = _mm256_sub_ps(wv, stepv);
            _mm256_storeu_ps(w + i, wv);
            _mm256_storeu_ps(m + i, newMv);
            _mm256_storeu_ps(g + i, zerov);
        }
        for (; i < n; ++i) {
            const float grad = g[i];
            const float update = beta1_ * m[i] + (1.0f - beta1_) * grad;
            w[i] -= std::copysign(lr_abs, update);
            m[i] = beta2_ * m[i] + (1.0f - beta2_) * grad;
            g[i] = 0.0f;
        }
    }
#else
    if (weightDecay_ != 0.0f) {
        const float decay = 1.0f - (lr_abs * weightDecay_);
        for (std::size_t i = 0; i < n; ++i) {
            const float grad = g[i];
            const float update = beta1_ * m[i] + (1.0f - beta1_) * grad;
            w[i] *= decay;
            w[i] -= std::copysign(lr_abs, update);
            m[i] = beta2_ * m[i] + (1.0f - beta2_) * grad;
            g[i] = 0.0f;
        }
    }
    else {
        for (std::size_t i = 0; i < n; ++i) {
            const float grad = g[i];
            const float update = beta1_ * m[i] + (1.0f - beta1_) * grad;
            w[i] -= std::copysign(lr_abs, update);
            m[i] = beta2_ * m[i] + (1.0f - beta2_) * grad;
            g[i] = 0.0f;
        }
    }
#endif
}
