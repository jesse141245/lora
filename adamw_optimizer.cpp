#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "adamw_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

bool adamUpdateDebugEnabled()
{
    const char* v = std::getenv("QLORA_DEBUG_ADAM_UPDATE");
    if (!v) return false;
    const std::string_view s{ v };
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}

int adamUpdateDebugTopK()
{
    const char* v = std::getenv("QLORA_DEBUG_ADAM_UPDATE_TOPK");
    if (!v) return 8;
    return std::max(1, std::atoi(v));
}

struct AdamUpdateDebugRow {
    std::string_view name;
    float updateRms = 0.0f;
    float updateToWeight = 0.0f;
    float weightRms = 0.0f;
    float updateScale = 1.0f;
    float lrScale = 1.0f;
    std::size_t n = 0;
};

struct AdamUpdateDebugGroup {
    std::string_view name;
    double weightedUpdateRms = 0.0;
    double weightedUpdateToWeight = 0.0;
    double weightedWeightRms = 0.0;
    float maxUpdateToWeight = 0.0f;
    float minUpdateScale = 1.0f;
    std::size_t n = 0;
    int count = 0;
};

} // namespace

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

void AdamWOptimizer::setLoraPairRmsBalance(
    bool enabled,
    float targetRatio,
    float maxScale,
    float minRms)
{
    loraPairRmsBalance_ = enabled;
    loraPairRmsBalanceTargetRatio_ = std::max(1e-6f, targetRatio);
    loraPairRmsBalanceMaxScale_ = std::max(1.0f, maxScale);
    loraPairRmsBalanceMinRms_ = std::max(0.0f, minRms);
}

void AdamWOptimizer::clipGradNorm(const std::vector<ParamRef>& params, float maxNorm)
{
    if (maxNorm <= 0.0f) return;

    constexpr int kClipGroups = 4;
    double ssByGroup[kClipGroups] = {};
    for (const auto& p : params) {
        const float* g = p.g;
        const double normScale = clipNormScaleForParam(p);
        const int group = clipGradNormByParamGroup_
            ? clipGroupForParam(p)
            : 0;
        for (std::size_t i = 0; i < p.n; ++i) {
            const double v = static_cast<double>(g[i]) * normScale;
            ssByGroup[group] += v * v;
        }
    }

    float scaleByGroup[kClipGroups] = {};
    bool anyClipped = false;
    for (int group = 0; group < kClipGroups; ++group) {
        const double norm = std::sqrt(ssByGroup[group]);
        if (norm == 0.0 || norm <= static_cast<double>(maxNorm)) {
            scaleByGroup[group] = 1.0f;
            continue;
        }

        scaleByGroup[group] = static_cast<float>(static_cast<double>(maxNorm) / norm);
        anyClipped = true;
    }
    if (!anyClipped) return;

    for (const auto& p : params) {
        const int group = clipGradNormByParamGroup_
            ? clipGroupForParam(p)
            : 0;
        scaleParamGrad(p, scaleByGroup[group]);
    }
}

void AdamWOptimizer::scaleParamGrad(ParamRef param, float scale)
{
    if (scale == 1.0f) return;

    float* g = param.g;
#if defined(__AVX2__)
    const __m256 sv = _mm256_set1_ps(scale);
    std::size_t i = 0;
    for (; i + 8 <= param.n; i += 8) {
        __m256 gv = _mm256_loadu_ps(g + i);
        gv = _mm256_mul_ps(gv, sv);
        _mm256_storeu_ps(g + i, gv);
    }
    for (; i < param.n; ++i) g[i] *= scale;
#else
    for (std::size_t i = 0; i < param.n; ++i) g[i] *= scale;
#endif
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

void AdamWOptimizer::updateBlockWithUpdateRmsCap(
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
    float invBc2,
    float maxUpdateRms,
    float* actualUpdateRmsOut,
    float* updateScaleOut)
{
    if (maxUpdateRms <= 0.0f || n == 0) {
        updateBlock(
            w,
            g,
            m,
            v,
            n,
            zeroGradAfterStep,
            lr,
            beta1,
            beta2,
            eps,
            weightDecay,
            invBc1,
            invBc2);
        if (actualUpdateRmsOut) *actualUpdateRmsOut = 0.0f;
        if (updateScaleOut) *updateScaleOut = 1.0f;
        return;
    }

    double updateSq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float gi = g[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;

        const float mHat = m[i] * invBc1;
        const float vHat = v[i] * invBc2;
        const float update = lr * (mHat / (std::sqrt(vHat) + eps));
        updateSq += static_cast<double>(update) * static_cast<double>(update);
    }

    const double updateRms = std::sqrt(updateSq / static_cast<double>(n));
    const float updateScale = (updateRms > static_cast<double>(maxUpdateRms))
        ? static_cast<float>(static_cast<double>(maxUpdateRms) / updateRms)
        : 1.0f;
    if (actualUpdateRmsOut) {
        *actualUpdateRmsOut = static_cast<float>(updateRms * static_cast<double>(updateScale));
    }
    if (updateScaleOut) {
        *updateScaleOut = updateScale;
    }
    const float oneMinusLrWd = 1.0f - lr * weightDecay;

    for (std::size_t i = 0; i < n; ++i) {
        const float mHat = m[i] * invBc1;
        const float vHat = v[i] * invBc2;
        const float update = lr * updateScale * (mHat / (std::sqrt(vHat) + eps));
        w[i] = w[i] * oneMinusLrWd - update;
        if (zeroGradAfterStep) g[i] = 0.0f;
    }
}

bool AdamWOptimizer::isAttentionAParam(const char* name)
{
    if (!name) return false;
    const std::string_view n{name};
    return n == "Aq" || n == "Ak" || n == "Av" || n == "Ao";
}

bool AdamWOptimizer::isAttentionBParam(const char* name)
{
    if (!name) return false;
    const std::string_view n{name};
    return n == "Bq" || n == "Bk" || n == "Bv" || n == "Bo";
}

bool AdamWOptimizer::isMatchingAttentionBParam(const char* aName, const char* bName)
{
    if (!aName || !bName) return false;
    const std::string_view a{ aName };
    const std::string_view b{ bName };
    return (a == "Aq" && b == "Bq") ||
        (a == "Ak" && b == "Bk") ||
        (a == "Av" && b == "Bv") ||
        (a == "Ao" && b == "Bo");
}

bool AdamWOptimizer::isLmHeadParam(const char* name)
{
    if (!name) return false;
    const std::string_view n{name};
    return n == "Alm" || n == "Blm" || n == "BiasLm";
}

bool AdamWOptimizer::isLmHeadMatrixParam(const char* name)
{
    if (!name) return false;
    const std::string_view n{name};
    return n == "Alm" || n == "Blm";
}

int AdamWOptimizer::clipGroupForParam(const ParamRef& param)
{
    if (isLmHeadParam(param.name)) return 1;
    if (isAttentionBParam(param.name)) return 2;
    if (isAttentionAParam(param.name)) return 3;
    return 0;
}

double AdamWOptimizer::clipNormScaleForParam(const ParamRef& param) const
{
    return clipGradNormByLrScale_
        ? static_cast<double>(param.lrScale)
        : 1.0;
}

double AdamWOptimizer::rmsOf(const float* w, std::size_t n)
{
    if (!w || n == 0) return 0.0;
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(w[i]);
        ss += v * v;
    }
    return std::sqrt(ss / static_cast<double>(n));
}

void AdamWOptimizer::scaleBlock(float* w, std::size_t n, float scale)
{
    if (!w || n == 0 || scale == 1.0f) return;

#if defined(__AVX2__)
    const __m256 sv = _mm256_set1_ps(scale);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 wv = _mm256_loadu_ps(w + i);
        wv = _mm256_mul_ps(wv, sv);
        _mm256_storeu_ps(w + i, wv);
    }
    for (; i < n; ++i) w[i] *= scale;
#else
    for (std::size_t i = 0; i < n; ++i) w[i] *= scale;
#endif
}

void AdamWOptimizer::maybeBalanceLoraPairs(const std::vector<ParamRef>& params)
{
    if (!loraPairRmsBalance_) return;
    if (params.size() != mByParam_.size() || params.size() != vByParam_.size()) return;

    for (std::size_t i = 0; i + 1 < params.size(); ++i) {
        const ParamRef& a = params[i];
        const ParamRef& b = params[i + 1];
        if (!isMatchingAttentionBParam(a.name, b.name)) continue;

        const double aRms = rmsOf(a.w, a.n);
        const double bRms = rmsOf(b.w, b.n);
        if (aRms <= static_cast<double>(loraPairRmsBalanceMinRms_) ||
            bRms <= static_cast<double>(loraPairRmsBalanceMinRms_)) {
            continue;
        }

        const double ratio = aRms / bRms;
        if (!std::isfinite(ratio) || ratio <= 0.0) continue;

        double aScale = std::sqrt(
            static_cast<double>(loraPairRmsBalanceTargetRatio_) / ratio);
        const double maxScale = static_cast<double>(loraPairRmsBalanceMaxScale_);
        aScale = std::clamp(aScale, 1.0 / maxScale, maxScale);
        if (!std::isfinite(aScale) || aScale <= 0.0 || aScale == 1.0) continue;

        const float aScaleF = static_cast<float>(aScale);
        const float bScaleF = static_cast<float>(1.0 / aScale);
        scaleBlock(a.w, a.n, aScaleF);
        scaleBlock(b.w, b.n, bScaleF);
        scaleBlock(mByParam_[i].data(), mByParam_[i].size(), aScaleF);
        scaleBlock(mByParam_[i + 1].data(), mByParam_[i + 1].size(), bScaleF);
        scaleBlock(vByParam_[i].data(), vByParam_[i].size(), aScaleF * aScaleF);
        scaleBlock(vByParam_[i + 1].data(), vByParam_[i + 1].size(), bScaleF * bScaleF);
    }
}

float AdamWOptimizer::beta2ForParam(const ParamRef& param) const
{
    if (attentionABeta2_ >= 0.0f && isAttentionAParam(param.name)) {
        return attentionABeta2_;
    }
    if (attentionBBeta2_ >= 0.0f && isAttentionBParam(param.name)) {
        return attentionBBeta2_;
    }
    return beta2_;
}

float AdamWOptimizer::beta1ForParam(const ParamRef& param) const
{
    if (attentionABeta1_ >= 0.0f && isAttentionAParam(param.name)) {
        return attentionABeta1_;
    }
    if (attentionBBeta1_ >= 0.0f && isAttentionBParam(param.name)) {
        return attentionBBeta1_;
    }
    if (lmHeadMatrixBeta1_ >= 0.0f && isLmHeadMatrixParam(param.name)) {
        return lmHeadMatrixBeta1_;
    }
    return beta1_;
}

void AdamWOptimizer::step(const std::vector<ParamRef>& params, bool zeroGrads)
{
    if (params.size() != mByParam_.size() || params.size() != vByParam_.size()) {
        throw std::runtime_error("AdamWOptimizer::step(params): call init() first");
    }

    ++step_;
    const bool debugAdamUpdate = adamUpdateDebugEnabled();
    std::vector<AdamUpdateDebugRow> debugRows;
    if (debugAdamUpdate) {
        debugRows.reserve(params.size());
    }

    for (std::size_t k = 0; k < params.size(); ++k) {
        if (params[k].n != mByParam_[k].size() || params[k].n != vByParam_[k].size()) {
            throw std::runtime_error("AdamWOptimizer::step(params): state size mismatch");
        }
        const float paramBeta1 = beta1ForParam(params[k]);
        const float bc1 = 1.0f - std::pow(paramBeta1, static_cast<float>(step_));
        const float invBc1 = 1.0f / std::max(1e-12f, bc1);
        const float paramBeta2 = beta2ForParam(params[k]);
        const float bc2 = 1.0f - std::pow(paramBeta2, static_cast<float>(step_));
        const float invBc2 = 1.0f / std::max(1e-12f, bc2);
        float updateRmsMaxRatio = 0.0f;
        if (isAttentionAParam(params[k].name)) {
            updateRmsMaxRatio = attentionAUpdateRmsMaxRatio_;
        }
        else if (isAttentionBParam(params[k].name)) {
            updateRmsMaxRatio = attentionBUpdateRmsMaxRatio_;
        }
        else if (isLmHeadMatrixParam(params[k].name)) {
            updateRmsMaxRatio = lmHeadMatrixUpdateRmsMaxRatio_;
        }
        const bool capWindowActive =
            updateRmsCapStartStep_ == 0 ||
            step_ >= updateRmsCapStartStep_;
        const bool capUpdate =
            capWindowActive &&
            updateRmsMaxRatio > 0.0f &&
            updateRmsCapMinWeightRms_ > 0.0f;
        const double weightRms = (capUpdate || debugAdamUpdate)
            ? rmsOf(params[k].w, params[k].n)
            : 0.0;
        float actualUpdateRms = 0.0f;
        float updateScale = 1.0f;
        if (capUpdate || debugAdamUpdate) {
            const bool applyCap = capUpdate &&
                weightRms >= static_cast<double>(updateRmsCapMinWeightRms_);
            const float maxUpdateRms = static_cast<float>(
                applyCap
                    ? std::max(0.0, weightRms * static_cast<double>(updateRmsMaxRatio))
                    : static_cast<double>(std::numeric_limits<float>::max()));
            updateBlockWithUpdateRmsCap(
                params[k].w,
                params[k].g,
                mByParam_[k].data(),
                vByParam_[k].data(),
                params[k].n,
                zeroGrads,
                lr_ * params[k].lrScale,
                paramBeta1,
                paramBeta2,
                eps_,
                weightDecay_,
                invBc1,
                invBc2,
                maxUpdateRms,
                &actualUpdateRms,
                &updateScale
            );
        }
        else {
            updateBlock(
                params[k].w,
                params[k].g,
                mByParam_[k].data(),
                vByParam_[k].data(),
                params[k].n,
                zeroGrads,
                lr_ * params[k].lrScale,
                paramBeta1,
                paramBeta2,
                eps_,
                weightDecay_,
                invBc1,
                invBc2
            );
        }

        if (debugAdamUpdate) {
            const float updateToWeight = (weightRms > 0.0)
                ? static_cast<float>(static_cast<double>(actualUpdateRms) / weightRms)
                : actualUpdateRms;
            debugRows.push_back({
                params[k].name ? std::string_view{ params[k].name } : std::string_view{},
                actualUpdateRms,
                updateToWeight,
                static_cast<float>(weightRms),
                updateScale,
                params[k].lrScale,
                params[k].n
            });
        }
    }

    maybeBalanceLoraPairs(params);

    if (debugAdamUpdate && !debugRows.empty()) {
        std::vector<AdamUpdateDebugGroup> debugGroups;
        debugGroups.reserve(debugRows.size());
        for (const auto& row : debugRows) {
            auto it = std::find_if(debugGroups.begin(), debugGroups.end(), [&](const auto& group) {
                return group.name == row.name;
            });
            if (it == debugGroups.end()) {
                debugGroups.push_back({ row.name });
                it = debugGroups.end() - 1;
            }
            const double n = static_cast<double>(row.n);
            it->weightedUpdateRms += static_cast<double>(row.updateRms) * n;
            it->weightedUpdateToWeight += static_cast<double>(row.updateToWeight) * n;
            it->weightedWeightRms += static_cast<double>(row.weightRms) * n;
            it->maxUpdateToWeight = std::max(it->maxUpdateToWeight, row.updateToWeight);
            it->minUpdateScale = std::min(it->minUpdateScale, row.updateScale);
            it->n += row.n;
            ++it->count;
        }
        const int groupTopK = std::min<int>(
            adamUpdateDebugTopK(),
            static_cast<int>(debugGroups.size()));
        std::sort(debugGroups.begin(), debugGroups.end(), [](const auto& a, const auto& b) {
            if (a.maxUpdateToWeight != b.maxUpdateToWeight) {
                return a.maxUpdateToWeight > b.maxUpdateToWeight;
            }
            return a.weightedUpdateToWeight > b.weightedUpdateToWeight;
        });
        std::cout << "[AdamUpdateGroup step " << step_ << "] topK=" << groupTopK;
        for (int i = 0; i < groupTopK; ++i) {
            const auto& group = debugGroups[static_cast<std::size_t>(i)];
            const double invN = group.n > 0
                ? 1.0 / static_cast<double>(group.n)
                : 0.0;
            std::cout << " | " << group.name
                << ":cnt=" << group.count
                << ",maxUpdW=" << group.maxUpdateToWeight
                << ",meanUpdW=" << static_cast<float>(group.weightedUpdateToWeight * invN)
                << ",meanUpdRms=" << static_cast<float>(group.weightedUpdateRms * invN)
                << ",meanWRms=" << static_cast<float>(group.weightedWeightRms * invN)
                << ",minScale=" << group.minUpdateScale
                << ",n=" << group.n;
        }
        std::cout << '\n';

        const int topK = std::min<int>(
            adamUpdateDebugTopK(),
            static_cast<int>(debugRows.size()));
        std::sort(debugRows.begin(), debugRows.end(), [](const auto& a, const auto& b) {
            if (a.updateToWeight != b.updateToWeight) {
                return a.updateToWeight > b.updateToWeight;
            }
            return a.updateRms > b.updateRms;
        });
        std::cout << "[AdamUpdate step " << step_ << "] topK=" << topK;
        for (int i = 0; i < topK; ++i) {
            const auto& row = debugRows[static_cast<std::size_t>(i)];
            std::cout << " | " << row.name
                << ":updRms=" << row.updateRms
                << ",updW=" << row.updateToWeight
                << ",wRms=" << row.weightRms
                << ",scale=" << row.updateScale
                << ",lrScale=" << row.lrScale
                << ",n=" << row.n;
        }
        std::cout << '\n';
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
