#pragma once
#include <cstddef>

struct ParamRef {
    float* w = nullptr;      // weights
    float* g = nullptr;      // gradients (mutable so we can zero)
    std::size_t n = 0;       // number of floats
    const char* name = nullptr; // optional for debugging
};