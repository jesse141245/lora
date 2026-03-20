#include "adamw_optimizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

void AdamWOptimizer::init(const std::vector<ParamRef>& params)
{
    if (mByParam_.size() == params.size() && vByParam_.size() == params.size()) {
        bool match = true;
        for (std::size_t i = 0; i < params.size(); ++i) {
            if (mByParam_[i].size() != params[i].n || vByParam_[i].size() != params[i].n) {
                match = false;
                break;
            }
        }
        if (match) return;
    }

    mByParam_.clear();
    vByParam_.clear();
    mByParam_.reserve(params.size());
    vByParam_.reserve(params.size());

    for (const auto& p : params) {
        if (!p.w || !p.g) throw std::runtime_error("AdamWOptimizer::init: null w/g");
        if (p.n == 0) throw std::runtime_error("AdamWOptimizer::init: zero-sized param");
        mByParam_.emplace_back(p.n, 0.0f);
        vByParam_.emplace_back(p.n, 0.0f);
    }
    step_ = 0;
}

void AdamWOptimizer::clipGradNorm(const std::vector<ParamRef>& params, float maxNorm)
{
    if (maxNorm <= 0.0f) return;

    double ss = 0.0;
    for (const auto& p : params) {
        const float* g = p.g;
        for (std::size_t i = 0; i < p.n; ++i) {
            const double v = static_cast<double>(g[i]);
            ss += v * v;
        }
    }

    const double norm = std::sqrt(ss);
    if (norm == 0.0 || norm <= static_cast<double>(maxNorm)) return;

    const float scale = static_cast<float>(static_cast<double>(maxNorm) / norm);
    for (const auto& p : params) {
        float* g = p.g;
#if defined(__AVX2__)
        const __m256 sv = _mm256_set1_ps(scale);
        std::size_t i = 0;
        for (; i + 8 <= p.n; i += 8) {
            __m256 gv = _mm256_loadu_ps(g + i);
            gv = _mm256_mul_ps(gv, sv);
            _mm256_storeu_ps(g + i, gv);
        }
        for (; i < p.n; ++i) g[i] *= scale;
#else
        for (std::size_t i = 0; i < p.n; ++i) g[i] *= scale;
#endif
    }
}

void AdamWOptimizer::updateBlock(
    float* w,
    float* g,
    float* m,
    float* v,
    std::size_t n,
    bool zeroGradAfterStep,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weightDecay,
    float invBc1,
    float invBc2)
{
    const float oneMinusLrWd = 1.0f - lr * weightDecay;

#if defined(__AVX2__)
    const __m256 beta1v = _mm256_set1_ps(beta1);
    const __m256 beta2v = _mm256_set1_ps(beta2);
    const __m256 oneMb1v = _mm256_set1_ps(1.0f - beta1);
    const __m256 oneMb2v = _mm256_set1_ps(1.0f - beta2);
    const __m256 invBc1v = _mm256_set1_ps(invBc1);
    const __m256 invBc2v = _mm256_set1_ps(invBc2);
    const __m256 lrv = _mm256_set1_ps(lr);
    const __m256 epsv = _mm256_set1_ps(eps);
    const __m256 oneMinusLrWdv = _mm256_set1_ps(oneMinusLrWd);
    const __m256 zerov = _mm256_setzero_ps();

    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 wv = _mm256_loadu_ps(w + i);
        __m256 gv = _mm256_loadu_ps(g + i);
        __m256 mv = _mm256_loadu_ps(m + i);
        __m256 vv = _mm256_loadu_ps(v + i);

        mv = _mm256_add_ps(_mm256_mul_ps(beta1v, mv), _mm256_mul_ps(oneMb1v, gv));
        const __m256 g2 = _mm256_mul_ps(gv, gv);
        vv = _mm256_add_ps(_mm256_mul_ps(beta2v, vv), _mm256_mul_ps(oneMb2v, g2));

        const __m256 mHat = _mm256_mul_ps(mv, invBc1v);
        const __m256 vHat = _mm256_mul_ps(vv, invBc2v);
        const __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(vHat), epsv);
        const __m256 update = _mm256_mul_ps(lrv, _mm256_div_ps(mHat, denom));

        wv = _mm256_mul_ps(wv, oneMinusLrWdv);
        wv = _mm256_sub_ps(wv, update);

        _mm256_storeu_ps(w + i, wv);
        _mm256_storeu_ps(m + i, mv);
        _mm256_storeu_ps(v + i, vv);

        if (zeroGradAfterStep) {
            _mm256_storeu_ps(g + i, zerov);
        }
    }

    for (; i < n; ++i) {
        const float gi = g[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;

        const float mHat = m[i] * invBc1;
        const float vHat = v[i] * invBc2;
        w[i] = w[i] * oneMinusLrWd - lr * (mHat / (std::sqrt(vHat) + eps));
        if (zeroGradAfterStep) g[i] = 0.0f;
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        const float gi = g[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;

        const float mHat = m[i] * invBc1;
        const float vHat = v[i] * invBc2;
        w[i] = w[i] * oneMinusLrWd - lr * (mHat / (std::sqrt(vHat) + eps));
        if (zeroGradAfterStep) g[i] = 0.0f;
    }
#endif
}

void AdamWOptimizer::step(const std::vector<ParamRef>& params, bool zeroGrads)
{
    if (params.size() != mByParam_.size() || params.size() != vByParam_.size()) {
        throw std::runtime_error("AdamWOptimizer::step(params): call init() first");
    }

    ++step_;
    const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_));
    const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_));
    const float invBc1 = 1.0f / std::max(1e-12f, bc1);
    const float invBc2 = 1.0f / std::max(1e-12f, bc2);

    for (std::size_t k = 0; k < params.size(); ++k) {
        if (params[k].n != mByParam_[k].size() || params[k].n != vByParam_[k].size()) {
            throw std::runtime_error("AdamWOptimizer::step(params): state size mismatch");
        }
        updateBlock(
            params[k].w,
            params[k].g,
            mByParam_[k].data(),
            vByParam_[k].data(),
            params[k].n,
            zeroGrads,
            lr_,
            beta1_,
            beta2_,
            eps_,
            weightDecay_,
            invBc1,
            invBc2
        );
    }
}

void AdamWOptimizer::ensureStateSize(std::size_t n)
{
    if (m_.size() == n && v_.size() == n) return;
    m_.assign(n, 0.0f);
    v_.assign(n, 0.0f);
    step_ = 0;
}

void AdamWOptimizer::zeroState()
{
    step_ = 0;
    std::fill(m_.begin(), m_.end(), 0.0f);
    std::fill(v_.begin(), v_.end(), 0.0f);
}

void AdamWOptimizer::step(std::span<float> params, std::span<float> grads)
{
    if (params.size() != grads.size()) {
        throw std::runtime_error("AdamWOptimizer::step: params/grads size mismatch");
    }
    if (params.empty()) return;

    ensureStateSize(params.size());
    ++step_;

    const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_));
    const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_));
    const float invBc1 = 1.0f / std::max(1e-12f, bc1);
    const float invBc2 = 1.0f / std::max(1e-12f, bc2);

    updateBlock(
        params.data(),
        grads.data(),
        m_.data(),
        v_.data(),
        params.size(),
        zeroGradAfterStep_,
        lr_,
        beta1_,
        beta2_,
        eps_,
        weightDecay_,
        invBc1,
        invBc2
    );
}
