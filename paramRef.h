#pragma once
#include <cstddef>

struct ParamRef {
    float* w = nullptr;      // weights
    float* g = nullptr;      // gradients (mutable so we can zero)
    std::size_t n = 0;       // number of floats
    const char* name = nullptr; // optional for debugging
    float lrScale = 1.0f;    // optional per-parameter LR multiplier
    std::size_t rows = 0;    // optional matrix rows for matrix-aware optimizers
    std::size_t cols = 0;    // optional matrix cols for matrix-aware optimizers
};
