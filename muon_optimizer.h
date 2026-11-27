#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "optimizer.h"
#include "paramRef.h"

class MuonOptimizer : public Optimizer {
public:
    void setHyperParams(
        float lr,
        float momentum,
        float adamBeta1,
        float adamBeta2,
        float eps,
        float weightDecay);
    void setZeroGradAfterStep(bool enabled) { zeroGradAfterStep_ = enabled; }

    void init(const std::vector<ParamRef>& params);
    void clipGradNorm(const std::vector<ParamRef>& params, float maxNorm);
    void step(const std::vector<ParamRef>& params, bool zeroGrads = true);

    void step(std::span<float> params, std::span<float> grads) override;
    void zeroState() override;

private:
    static bool isMuonParam(const ParamRef& param);
    static void orthogonalizeNewtonSchulz(
        const float* in,
        float* out,
        std::size_t rows,
        std::size_t cols,
        int steps);
    static double rmsOf(const float* x, std::size_t n);

private:
    float lr_ = 1e-4f;
    float momentum_ = 0.95f;
    float adamBeta1_ = 0.9f;
    float adamBeta2_ = 0.999f;
    float eps_ = 1e-8f;
    float weightDecay_ = 0.0f;
    bool zeroGradAfterStep_ = true;
    std::uint64_t step_ = 0;
    std::vector<std::vector<float>> momentumByParam_;
    std::vector<std::vector<float>> adamMByParam_;
    std::vector<std::vector<float>> adamVByParam_;
    std::vector<float> scratch_;
};
