#pragma once

#include <cstddef>

static inline void hadamard32_inplace(float* values) {
    for (std::size_t span = 1; span < 32; span <<= 1) {
        const std::size_t step = span << 1;
        for (std::size_t base = 0; base < 32; base += step) {
            for (std::size_t i = 0; i < span; ++i) {
                const float a = values[base + i];
                const float b = values[base + span + i];
                values[base + i] = a + b;
                values[base + span + i] = a - b;
            }
        }
    }

    static constexpr float kInvSqrt32 = 0.1767766952966369f;
    for (std::size_t i = 0; i < 32; ++i) {
        values[i] *= kInvSqrt32;
    }
}
