#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// ------------------------------------------------------------
// Legacy stage-1 BitNet container
// ------------------------------------------------------------
struct BitNetMat {
    std::size_t H = 0;
    std::size_t block = 64;
    std::vector<int8_t> q;     // {-1,0,+1}
    std::vector<float> s;      // per-block scales
};

// ------------------------------------------------------------
// Packed 2-bit BitNet I2S container (Option B / x4 Arch)
//
// Weight matrix W is [rows x cols] row-major.
// ------------------------------------------------------------
struct BitNetMatI2S {
    std::size_t rows = 0;
    std::size_t cols = 0;

    // Forward quantization / kernel layout
    std::size_t qk = 128;  // block size along rows (input dim)
    std::size_t jq = 32;   // block size along cols (output dim)

    // Packed 2-bit weights
    std::vector<uint8_t> q2;           // W packed row-major
    std::vector<uint8_t> q2T;          // W^T packed row-major

    // Tiled scales
    std::vector<float> w_scale_tile;   // size = nKBlocks * nJBlocks
    std::vector<float> w_scale_tileT;  // transposed for backward

    static constexpr std::size_t weightsPerByte = 4;

    bool empty() const noexcept {
        return rows == 0 || cols == 0 || q2.empty() || w_scale_tile.empty();
    }

    std::size_t nKBlocks() const noexcept {
        return (rows == 0 || qk == 0) ? 0 : ((rows + qk - 1) / qk);
    }

    std::size_t nJBlocks() const noexcept {
        return (cols == 0 || jq == 0) ? 0 : ((cols + jq - 1) / jq);
    }

    std::size_t expectedPackedBytes() const noexcept {
        if (rows == 0 || cols == 0) return 0;
        const std::size_t N = rows * cols;
        return (N + weightsPerByte - 1) / weightsPerByte;
    }

    std::size_t expectedScaleCount() const noexcept {
        return nKBlocks() * nJBlocks();
    }

    bool validForShape() const noexcept {
        return rows != 0 && cols != 0 && qk != 0 && jq != 0
            && q2.size() >= expectedPackedBytes()
            && q2T.size() >= expectedPackedBytes()
            && w_scale_tile.size() >= expectedScaleCount()
            && w_scale_tileT.size() >= expectedScaleCount();
    }

    void clear() noexcept {
        rows = 0; cols = 0; qk = 128; jq = 32;
        q2.clear(); q2T.clear(); w_scale_tile.clear(); w_scale_tileT.clear();
    }
};

void build_bitnet_i2s_transposed_view(BitNetMatI2S& W);