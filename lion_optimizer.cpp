#include "lion_optimizer.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

void LionOptimizer::init(const std::vector<ParamRef>& params)
{
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
        throw std::runtime_error("clipGradNorm: params size changed; call init() again");
    }

    // Compute global L2 norm
    double ss = 0.0;
    for (const auto& p : params) {
        const float* g = p.g;
        for (std::size_t i = 0; i < p.n; ++i) {
            const double v = (double)g[i];
            ss += v * v;
        }
    }

    const double norm = std::sqrt(ss);
    if (norm <= (double)maxNorm || norm == 0.0) return;

    const float scale = (float)((double)maxNorm / norm);

    for (auto& p : params) {
        float* g = p.g;
        for (std::size_t i = 0; i < p.n; ++i) g[i] *= scale;
    }
}

void LionOptimizer::step(const std::vector<ParamRef>& params, bool zeroGrads)
{
    if (params.size() != m_.size()) {
        throw std::runtime_error("LionOptimizer::step: params size changed; call init() again");
    }

    for (std::size_t k = 0; k < params.size(); ++k) {
        float* w = params[k].w;
        float* g = params[k].g;
        auto& m = m_[k];

        const std::size_t n = params[k].n;
        if (m.size() != n) throw std::runtime_error("LionOptimizer::step: momentum size mismatch");

        for (std::size_t i = 0; i < n; ++i) {
            const float grad = g[i] + weightDecay_ * w[i];

            // momentum update
            m[i] = beta1_ * m[i] + (1.0f - beta1_) * grad;

            // lion update direction (beta2 mixing)
            const float u = beta2_ * m[i] + (1.0f - beta2_) * grad;

            // sign step
            w[i] -= lr_ * std::copysign(1.0f, u);

            if (zeroGrads) g[i] = 0.0f;
        }
    }
}