#pragma once
#include <vector>
#include <cstddef>
#include "paramRef.h"

class LionOptimizer {
public:
    LionOptimizer() = default;

    void setHyperParams(float lr, float beta1, float beta2, float weightDecay) {
        lr_ = lr; beta1_ = beta1; beta2_ = beta2; weightDecay_ = weightDecay;
    }

    void init(const std::vector<ParamRef>& params);

    // Clips global L2 norm of grads across all params to maxNorm
    void clipGradNorm(const std::vector<ParamRef>& params, float maxNorm);

    // Applies Lion update to all params in-place
    void step(const std::vector<ParamRef>& params, bool zeroGrads = true);

private:
    float lr_ = 1e-4f;
    float beta1_ = 0.9f;
    float beta2_ = 0.99f;
    float weightDecay_ = 0.0f;

    std::vector<std::vector<float>> m_;
};