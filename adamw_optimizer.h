#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "optimizer.h"
#include "paramRef.h"

// AdamW optimizer over a flat parameter vector.
// Implements decoupled weight decay and optional gradient zeroing.
class AdamWOptimizer : public Optimizer {
public:
    void setHyperParams(float lr, float beta1, float beta2, float eps, float weightDecay) {
        lr_ = lr;
        beta1_ = beta1;
        beta2_ = beta2;
        eps_ = eps;
        weightDecay_ = weightDecay;
    }
    void setHyperParams(float lr, float beta1, float beta2, float weightDecay) {
        setHyperParams(lr, beta1, beta2, eps_, weightDecay);
    }
    void setParamBeta2Overrides(float attentionA, float attentionB) {
        attentionABeta2_ = attentionA;
        attentionBBeta2_ = attentionB;
    }
    void setParamBeta1Overrides(float attentionA, float attentionB, float lmHeadMatrix) {
        attentionABeta1_ = attentionA;
        attentionBBeta1_ = attentionB;
        lmHeadMatrixBeta1_ = lmHeadMatrix;
    }

    void setZeroGradAfterStep(bool enabled) { zeroGradAfterStep_ = enabled; }
    void setClipGradNormByLrScale(bool enabled) { clipGradNormByLrScale_ = enabled; }
    void setClipGradNormByParamGroup(bool enabled) { clipGradNormByParamGroup_ = enabled; }
    void setLoraPairRmsBalance(
        bool enabled,
        float targetRatio,
        float maxScale,
        float minRms);
    void setAttentionAUpdateRmsMaxRatio(float ratio) {
        attentionAUpdateRmsMaxRatio_ = ratio;
    }
    void setUpdateRmsMaxRatios(float attentionA, float attentionB, float lmHeadMatrix) {
        attentionAUpdateRmsMaxRatio_ = attentionA;
        attentionBUpdateRmsMaxRatio_ = attentionB;
        lmHeadMatrixUpdateRmsMaxRatio_ = lmHeadMatrix;
    }
    void setUpdateRmsCapMinWeightRms(float minWeightRms) {
        updateRmsCapMinWeightRms_ = minWeightRms;
    }
    void setUpdateRmsCapStartStep(std::uint64_t startStep) {
        updateRmsCapStartStep_ = startStep;
    }

    // ParamRef-based API used by the trainer.
    void init(const std::vector<ParamRef>& params);
    void clipGradNorm(const std::vector<ParamRef>& params, float maxNorm);
    void step(const std::vector<ParamRef>& params, bool zeroGrads = true);

    std::vector<std::vector<float>>& getFirstMoment() { return mByParam_; }
    const std::vector<std::vector<float>>& getFirstMoment() const { return mByParam_; }
    std::vector<std::vector<float>>& getSecondMoment() { return vByParam_; }
    const std::vector<std::vector<float>>& getSecondMoment() const { return vByParam_; }
    void setStep(std::uint64_t step) { step_ = step; }
    std::uint64_t stepCount() const { return step_; }

    void step(std::span<float> params, std::span<float> grads) override;
    void zeroState() override;

private:
    void ensureStateSize(std::size_t n);
    static void updateBlock(
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
        float invBc2);
    static void updateBlockWithUpdateRmsCap(
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
        float* actualUpdateRmsOut = nullptr,
        float* updateScaleOut = nullptr);
    static bool isAttentionAParam(const char* name);
    static bool isAttentionBParam(const char* name);
    static bool isMatchingAttentionBParam(const char* aName, const char* bName);
    static bool isLmHeadParam(const char* name);
    static bool isLmHeadMatrixParam(const char* name);
    static int clipGroupForParam(const ParamRef& param);
    double clipNormScaleForParam(const ParamRef& param) const;
    static void scaleParamGrad(ParamRef param, float scale);
    static double rmsOf(const float* w, std::size_t n);
    static void scaleBlock(float* w, std::size_t n, float scale);
    void maybeBalanceLoraPairs(const std::vector<ParamRef>& params);
    float beta2ForParam(const ParamRef& param) const;
    float beta1ForParam(const ParamRef& param) const;

private:
    float lr_ = 1e-4f;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float eps_ = 1e-8f;
    float weightDecay_ = 0.0f;
    float attentionABeta1_ = -1.0f;
    float attentionBBeta1_ = -1.0f;
    float lmHeadMatrixBeta1_ = -1.0f;
    float attentionABeta2_ = -1.0f;
    float attentionBBeta2_ = -1.0f;
    bool zeroGradAfterStep_ = true;
    bool clipGradNormByLrScale_ = false;
    bool clipGradNormByParamGroup_ = false;
    bool loraPairRmsBalance_ = false;
    float loraPairRmsBalanceTargetRatio_ = 1.0f;
    float loraPairRmsBalanceMaxScale_ = 2.0f;
    float loraPairRmsBalanceMinRms_ = 1e-6f;
    float attentionAUpdateRmsMaxRatio_ = 0.0f;
    float attentionBUpdateRmsMaxRatio_ = 0.0f;
    float lmHeadMatrixUpdateRmsMaxRatio_ = 0.0f;
    float updateRmsCapMinWeightRms_ = 1e-6f;
    std::uint64_t updateRmsCapStartStep_ = 0;

    std::uint64_t step_ = 0;
    std::vector<float> m_;
    std::vector<float> v_;
    std::vector<std::vector<float>> mByParam_;
    std::vector<std::vector<float>> vByParam_;
};
