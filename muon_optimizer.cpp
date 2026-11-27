#include "muon_optimizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>

void MuonOptimizer::setHyperParams(
    float lr,
    float momentum,
    float adamBeta1,
    float adamBeta2,
    float eps,
    float weightDecay)
{
    lr_ = lr;
    momentum_ = momentum;
    adamBeta1_ = adamBeta1;
    adamBeta2_ = adamBeta2;
    eps_ = eps;
    weightDecay_ = weightDecay;
}

void MuonOptimizer::init(const std::vector<ParamRef>& params)
{
    momentumByParam_.clear();
    adamMByParam_.clear();
    adamVByParam_.clear();
    momentumByParam_.reserve(params.size());
    adamMByParam_.reserve(params.size());
    adamVByParam_.reserve(params.size());
    for (const auto& p : params) {
        if (!p.w || !p.g || p.n == 0) {
            throw std::runtime_error("MuonOptimizer::init: invalid param");
        }
        momentumByParam_.emplace_back(p.n, 0.0f);
        adamMByParam_.emplace_back(p.n, 0.0f);
        adamVByParam_.emplace_back(p.n, 0.0f);
    }
    step_ = 0;
}

bool MuonOptimizer::isMuonParam(const ParamRef& param)
{
    if (param.rows == 0 || param.cols == 0 || param.rows * param.cols != param.n) {
        return false;
    }
    if (!param.name) return false;
    const std::string_view name{ param.name };
    return name == "Bq" ||
        name == "Bk" ||
        name == "Bv" ||
        name == "Bo";
}

double MuonOptimizer::rmsOf(const float* x, std::size_t n)
{
    if (!x || n == 0) return 0.0;
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = x[i];
        ss += v * v;
    }
    return std::sqrt(ss / static_cast<double>(n));
}

void MuonOptimizer::orthogonalizeNewtonSchulz(
    const float* in,
    float* out,
    std::size_t rows,
    std::size_t cols,
    int steps)
{
    const std::size_t n = rows * cols;
    const double rms = rmsOf(in, n);
    if (rms <= 0.0 || !std::isfinite(rms)) {
        std::fill(out, out + n, 0.0f);
        return;
    }

    const double inv = 1.0 / (rms * std::sqrt(static_cast<double>(n)));
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(static_cast<double>(in[i]) * inv);
    }

    std::vector<float> gram(rows * rows, 0.0f);
    std::vector<float> gram2(rows * rows, 0.0f);
    std::vector<float> mixed(n, 0.0f);
    constexpr float a = 3.4445f;
    constexpr float b = -4.7750f;
    constexpr float c = 2.0315f;

    for (int iter = 0; iter < steps; ++iter) {
        std::fill(gram.begin(), gram.end(), 0.0f);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k < cols; ++k) {
                    sum += static_cast<double>(out[i * cols + k]) *
                        static_cast<double>(out[j * cols + k]);
                }
                gram[i * rows + j] = static_cast<float>(sum);
                gram[j * rows + i] = static_cast<float>(sum);
            }
        }

        std::fill(gram2.begin(), gram2.end(), 0.0f);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < rows; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k < rows; ++k) {
                    sum += static_cast<double>(gram[i * rows + k]) *
                        static_cast<double>(gram[k * rows + j]);
                }
                gram2[i * rows + j] = static_cast<float>(sum);
            }
        }

        std::fill(mixed.begin(), mixed.end(), 0.0f);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t k = 0; k < cols; ++k) {
                double sum = static_cast<double>(a) * out[i * cols + k];
                for (std::size_t j = 0; j < rows; ++j) {
                    const float coeff = b * gram[i * rows + j] + c * gram2[i * rows + j];
                    sum += static_cast<double>(coeff) * out[j * cols + k];
                }
                mixed[i * cols + k] = static_cast<float>(sum);
            }
        }
        std::copy(mixed.begin(), mixed.end(), out);
    }
}

void MuonOptimizer::clipGradNorm(const std::vector<ParamRef>& params, float maxNorm)
{
    if (maxNorm <= 0.0f) return;
    double ss = 0.0;
    for (const auto& p : params) {
        for (std::size_t i = 0; i < p.n; ++i) {
            const double v = p.g[i];
            ss += v * v;
        }
    }
    const double norm = std::sqrt(ss);
    if (norm <= static_cast<double>(maxNorm) || norm <= 0.0) return;
    const float scale = static_cast<float>(static_cast<double>(maxNorm) / norm);
    for (const auto& p : params) {
        for (std::size_t i = 0; i < p.n; ++i) p.g[i] *= scale;
    }
}

void MuonOptimizer::step(const std::vector<ParamRef>& params, bool zeroGrads)
{
    if (params.size() != momentumByParam_.size()) {
        throw std::runtime_error("MuonOptimizer::step: call init() first");
    }
    ++step_;
    const float adamBc1 = 1.0f / std::max(1e-12f, 1.0f - std::pow(adamBeta1_, static_cast<float>(step_)));
    const float adamBc2 = 1.0f / std::max(1e-12f, 1.0f - std::pow(adamBeta2_, static_cast<float>(step_)));

    for (std::size_t pidx = 0; pidx < params.size(); ++pidx) {
        const auto& p = params[pidx];
        if (p.n != momentumByParam_[pidx].size()) {
            throw std::runtime_error("MuonOptimizer::step: state size mismatch");
        }
        const float lr = lr_ * p.lrScale;
        if (lr == 0.0f) {
            if (zeroGrads) std::fill(p.g, p.g + p.n, 0.0f);
            continue;
        }

        if (isMuonParam(p)) {
            auto& m = momentumByParam_[pidx];
            scratch_.resize(p.n);
            for (std::size_t i = 0; i < p.n; ++i) {
                m[i] = momentum_ * m[i] + p.g[i];
                scratch_[i] = p.g[i] + momentum_ * m[i];
            }
            orthogonalizeNewtonSchulz(scratch_.data(), scratch_.data(), p.rows, p.cols, 5);
            const float shapeScale = std::sqrt(
                std::max(1.0f, static_cast<float>(p.cols) / static_cast<float>(std::max<std::size_t>(1, p.rows))));
            const float oneMinusLrWd = 1.0f - lr * weightDecay_;
            for (std::size_t i = 0; i < p.n; ++i) {
                p.w[i] = p.w[i] * oneMinusLrWd - lr * shapeScale * scratch_[i];
                if (zeroGrads) p.g[i] = 0.0f;
            }
        }
        else {
            auto& m = adamMByParam_[pidx];
            auto& v = adamVByParam_[pidx];
            const float oneMinusLrWd = 1.0f - lr * weightDecay_;
            for (std::size_t i = 0; i < p.n; ++i) {
                const float gi = p.g[i];
                m[i] = adamBeta1_ * m[i] + (1.0f - adamBeta1_) * gi;
                v[i] = adamBeta2_ * v[i] + (1.0f - adamBeta2_) * gi * gi;
                const float mh = m[i] * adamBc1;
                const float vh = v[i] * adamBc2;
                p.w[i] = p.w[i] * oneMinusLrWd - lr * (mh / (std::sqrt(vh) + eps_));
                if (zeroGrads) p.g[i] = 0.0f;
            }
        }
    }
}

void MuonOptimizer::step(std::span<float> params, std::span<float> grads)
{
    if (params.size() != grads.size()) throw std::runtime_error("MuonOptimizer::step span size mismatch");
    ParamRef p{ params.data(), grads.data(), params.size(), "flat", 1.0f, 0, 0 };
    std::vector<ParamRef> refs{ p };
    if (momentumByParam_.size() != 1 || momentumByParam_[0].size() != params.size()) init(refs);
    step(refs, zeroGradAfterStep_);
}

void MuonOptimizer::zeroState()
{
    step_ = 0;
    for (auto& m : momentumByParam_) std::fill(m.begin(), m.end(), 0.0f);
    for (auto& m : adamMByParam_) std::fill(m.begin(), m.end(), 0.0f);
    for (auto& v : adamVByParam_) std::fill(v.begin(), v.end(), 0.0f);
}
