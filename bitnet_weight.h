#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// ------------------------------------------------------------
// Legacy stage-1 BitNet container
// Keep only while old kernels still exist.
// ------------------------------------------------------------
struct BitNetMat {
    std::size_t H = 0;
    std::size_t block = 64;
    std::vector<int8_t> q;     // {-1,0,+1}
    std::vector<float> s;      // per-block scales
};

// ------------------------------------------------------------
// Packed 2-bit BitNet I2S container (Option B)
//
// Weight matrix W is [H x H] row-major:
//
//   Y[t, j] = sum_i X[t, i] * W[i, j]
//
// Packed storage:
//   q2  : packed row-major W
//   q2T : packed col-major W (W^T row-major)
//
// Tiled scale storage:
//   qk = K/input tile height   (recommended 128)
//   jq = output-column tile    (recommended 32)
//
// Forward scale layout:
//   w_scale_tile[jb * nKBlocks + kb]
//
// Backward scale layout (transposed):
//   w_scale_tileT[kb * nJBlocks + jb]
// ------------------------------------------------------------
struct BitNetMatI2S {
    std::size_t H = 0;

    // Forward quantization / kernel layout
    std::size_t qk = 128;  // K/input tile height for scales
    std::size_t jq = 32;   // output-column tile width for scales

    // Packed 2-bit weights
    std::vector<uint8_t> q2;
    std::vector<uint8_t> q2T;

    // Tiled scales
    std::vector<float> w_scale_tile;
    std::vector<float> w_scale_tileT;

    static constexpr std::size_t weightsPerByte = 4;

    bool empty() const noexcept {
        return H == 0 || q2.empty() || w_scale_tile.empty();
    }

    std::size_t nKBlocks() const noexcept {
        return (H == 0 || qk == 0) ? 0 : ((H + qk - 1) / qk);
    }

    std::size_t nJBlocks() const noexcept {
        return (H == 0 || jq == 0) ? 0 : ((H + jq - 1) / jq);
    }

    std::size_t expectedPackedBytes() const noexcept {
        if (H == 0) return 0;
        const std::size_t N = H * H;
        return (N + weightsPerByte - 1) / weightsPerByte;
    }

    std::size_t expectedScaleCount() const noexcept {
        if (H == 0 || qk == 0 || jq == 0) return 0;
        return nKBlocks() * nJBlocks();
    }

    bool validForShape() const noexcept {
        return H != 0
            && qk != 0
            && jq != 0
            && q2.size() >= expectedPackedBytes()
            && q2T.size() >= expectedPackedBytes()
            && w_scale_tile.size() >= expectedScaleCount()
            && w_scale_tileT.size() >= expectedScaleCount();
    }

    void clear() noexcept {
        H = 0;
        qk = 128;
        jq = 32;
        q2.clear();
        q2T.clear();
        w_scale_tile.clear();
        w_scale_tileT.clear();
    }
};

// Build the explicit transposed view for the backward pass
void build_bitnet_i2s_transposed_view(BitNetMatI2S& W);