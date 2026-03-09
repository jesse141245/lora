#pragma once
#include <vector>
#include <cstdint>

struct LayerLoRA {
    std::vector<float> Aq, Bq;
    std::vector<float> Av, Bv;
};

struct LayerLoRAGrad {
    std::vector<float> gradAq, gradBq;
    std::vector<float> gradAv, gradBv;
};

enum class LoraDType : uint32_t {
    F32 = 0,
};