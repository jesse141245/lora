#pragma once
#include <cstddef>

struct ModelShape {
    std::size_t vocabSize;
    std::size_t hiddenSize;
    std::size_t nLayers;
    std::size_t nHeads;

    std::size_t headDim() const { return hiddenSize / nHeads; }
};
