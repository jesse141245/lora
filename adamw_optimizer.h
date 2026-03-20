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

    void setZeroGradAfterStep(bool enabled) { zeroGradAfterStep_ = enabled; }

    // ParamRef-based API used by the trainer.
    void init(const std::vector<ParamRef>& params);
    void clipGradNorm(const std::vector<ParamRef>& params, float maxNorm);
    void step(const std::vector<ParamRef>& params, bool zeroGrads = true);

    std::vector<std::vector<float>>& getFirstMoment() { return mByParam_; }
    const std::vector<std::vector<float>>& getFirstMoment() const { return mByParam_; }
    std::vector<std::vector<float>>& getSecondMoment() { return vByParam_; }
    const std::vector<std::vector<float>>& getSecondMoment() const { return vByParam_; }

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

private:
    float lr_ = 1e-4f;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float eps_ = 1e-8f;
    float weightDecay_ = 0.0f;
    bool zeroGradAfterStep_ = true;

    std::uint64_t step_ = 0;
    std::vector<float> m_;
    std::vector<float> v_;
    std::vector<std::vector<float>> mByParam_;
    std::vector<std::vector<float>> vByParam_;
};
