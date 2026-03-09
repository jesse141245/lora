#pragma once
#include <span>

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void step(std::span<float> params,
        std::span<float> grads) = 0;

    virtual void zeroState() = 0;
};

