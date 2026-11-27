#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "optimizer.h"
#include "paramRef.h" 

class LionOptimizer : public Optimizer {
public:

    void setHyperParams(float lr, float beta1, float beta2, float weightDecay) {
        lr_ = lr;
        beta1_ = beta1;
        beta2_ = beta2;
        weightDecay_ = weightDecay;
    }

    // Core Functions
    void init(const std::vector<ParamRef>& params);
    void clipGradNorm(const std::vector<ParamRef>& params, float maxNorm);
    void step(const std::vector<ParamRef>& params, bool zeroGrads = true);
    void step(std::span<float> params, std::span<float> grads) override;
    void zeroState() override;

    std::vector<std::vector<float>>& getMomentum() { return m_; }
    const std::vector<std::vector<float>>& getMomentum() const { return m_; }
    void setStep(std::uint64_t) {}

private:
    // Hyperparameters
    float lr_ = 1e-4f;
    float beta1_ = 0.9f;
    float beta2_ = 0.99f;
    float weightDecay_ = 0.0f;
    std::vector<std::vector<float>> m_; // Momentum tracker
    std::vector<float> mFlat_;
};
