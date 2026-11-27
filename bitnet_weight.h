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

enum class BitNetI2SScaleLayout : std::uint32_t {
    JMajorContiguousRows = 1, // [jBlock][kBlock]
    JMajorBroadcastRows = 2,  // [jBlock][kBlock][8]
};

// ------------------------------------------------------------
// Packed 2-bit BitNet I2S container (Option B / x4 Arch)
//
// Weight matrix W is [rows x cols] row-major.
// ------------------------------------------------------------
struct BitNetMatI2S {
    std::size_t rows = 0;
    std::size_t cols = 0;
    bool hadamard32 = false;

    // Forward quantization / kernel layout
    std::size_t qk = 128;  // block size along rows (input dim)
    std::size_t jq = 32;   // block size along cols (output dim)
    BitNetI2SScaleLayout scaleLayout = BitNetI2SScaleLayout::JMajorBroadcastRows;

    // Packed 2-bit weights
    std::vector<uint8_t> q2;           // W packed row-major
    std::vector<uint8_t> q2T;          // W^T packed row-major
    std::vector<int8_t> q8;            // Optional decoded W cache for hot forward kernels
    std::vector<int8_t> qT8;           // Optional decoded W^T cache for hot backward kernels

    // Tiled scales
    std::vector<float> w_scale_tile;   // size = nKBlocks * nJBlocks
    std::vector<float> w_scale_tileT;  // transposed for backward
    std::vector<float> w_scale_tile_broadcast8;   // size = nKBlocks * nJBlocks * 8
    std::vector<float> w_scale_tileT_broadcast8;  // transposed for backward, broadcast-friendly

    static constexpr std::size_t weightsPerByte = 4;
    static constexpr std::size_t scaleBroadcastWidth = 8;

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

    std::size_t expectedBroadcastScaleCount() const noexcept {
        return expectedScaleCount() * scaleBroadcastWidth;
    }

    bool hasSupportedScaleLayout() const noexcept {
        return scaleLayout == BitNetI2SScaleLayout::JMajorContiguousRows
            || scaleLayout == BitNetI2SScaleLayout::JMajorBroadcastRows;
    }

    std::size_t forwardScaleIndex(std::size_t jBlock, std::size_t kBlock) const noexcept {
        return jBlock * nKBlocks() + kBlock;
    }

    std::size_t backwardScaleIndex(std::size_t kBlock, std::size_t jBlock) const noexcept {
        return kBlock * nJBlocks() + jBlock;
    }

    const float* forwardScaleBroadcastRow(std::size_t jBlock) const noexcept {
        return w_scale_tile_broadcast8.data()
            + forwardScaleIndex(jBlock, 0) * scaleBroadcastWidth;
    }

    float* forwardScaleBroadcastRow(std::size_t jBlock) noexcept {
        return w_scale_tile_broadcast8.data()
            + forwardScaleIndex(jBlock, 0) * scaleBroadcastWidth;
    }

    const float* backwardScaleBroadcastRow(std::size_t kBlock) const noexcept {
        return w_scale_tileT_broadcast8.data()
            + backwardScaleIndex(kBlock, 0) * scaleBroadcastWidth;
    }

    float* backwardScaleBroadcastRow(std::size_t kBlock) noexcept {
        return w_scale_tileT_broadcast8.data()
            + backwardScaleIndex(kBlock, 0) * scaleBroadcastWidth;
    }

    bool validForShape() const noexcept {
        return rows != 0 && cols != 0 && qk != 0 && jq != 0
            && hasSupportedScaleLayout()
            && q2.size() >= expectedPackedBytes()
            && q2T.size() >= expectedPackedBytes()
            && w_scale_tile.size() >= expectedScaleCount()
            && w_scale_tileT.size() >= expectedScaleCount()
            && w_scale_tile_broadcast8.size() >= expectedBroadcastScaleCount()
            && w_scale_tileT_broadcast8.size() >= expectedBroadcastScaleCount();
    }

    void clear() noexcept {
        rows = 0; cols = 0; hadamard32 = false; qk = 128; jq = 32;
        scaleLayout = BitNetI2SScaleLayout::JMajorBroadcastRows;
        q2.clear(); q2T.clear(); q8.clear(); qT8.clear(); w_scale_tile.clear(); w_scale_tileT.clear();
        w_scale_tile_broadcast8.clear(); w_scale_tileT_broadcast8.clear();
    }
};

// ------------------------------------------------------------
// Experimental BitNet TL2 forward container
//
// Generic AVX2 lookup-table path:
// - use TL2-style 3-weight lookup groups over the largest prefix that is
//   divisible by bk (default 96)
// - use a smaller 2-weight lookup tail over the remaining 32/64 rows
//
// This container is forward-only for now. Backward still uses I2S.
// ------------------------------------------------------------
struct BitNetMatTL2 {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t bk = 96;
    std::size_t tl2Rows = 0;
    std::size_t tailRows = 0;

    std::vector<uint8_t> tripleCodes; // [jBlock][kBlock][bk/3 groups][32 cols], low nibble = abs TL2 index, bit4 = sign
    std::vector<uint8_t> pairCodes;   // [jBlock][tailRows/2 groups][32 cols]
    std::vector<float> tripleScales;  // [jBlock][kBlock]
    std::vector<float> tailScales;    // [jBlock]

    bool empty() const noexcept {
        return rows == 0 || cols == 0 || tripleCodes.empty();
    }

    std::size_t nJBlocks() const noexcept {
        return cols / 32;
    }

    std::size_t nTl2Blocks() const noexcept {
        return (bk == 0) ? 0 : (tl2Rows / bk);
    }

    std::size_t tripleGroupsPerBlock() const noexcept {
        return bk / 3;
    }

    std::size_t tailPairGroups() const noexcept {
        return tailRows / 2;
    }

    std::size_t tripleBlockBytes() const noexcept {
        return tripleGroupsPerBlock() * 32;
    }

    void clear() noexcept {
        rows = 0;
        cols = 0;
        bk = 96;
        tl2Rows = 0;
        tailRows = 0;
        tripleCodes.clear();
        pairCodes.clear();
        tripleScales.clear();
        tailScales.clear();
    }
};

// ------------------------------------------------------------
// Microsoft-style shape-specialized TL2 container
//
// Layout matches the generated x86 TL2 kernels:
// - triple-prefix absolute-code bytes
// - triple-prefix sign bytes
// - optional two-weight tail bytes
// - one per-tensor scale
// ------------------------------------------------------------
struct BitNetMatTL2Gen {
    std::size_t rows = 0;      // input dim (K)
    std::size_t cols = 0;      // output dim (M)
    std::size_t bm = 0;        // output tile
    std::size_t bk = 96;       // TL2 three-weight block
    std::size_t threeK = 0;    // largest prefix handled by TL2
    std::size_t twoK = 0;      // 32/64-weight tail
    std::size_t nTileNum = 0;  // cols / bm
    float scale = 1.0f;        // fallback/global scale
    std::vector<float> tileScales; // optional per-output-tile scales
    std::vector<uint8_t> qweights;
    BitNetMatI2S i2sMain;
    float residualScale = 1.0f; // optional second-stage fallback/global scale
    std::vector<float> residualTileScales;
    std::vector<uint8_t> residualQweights;
    BitNetMatI2S i2sResidualStage;
    BitNetMatI2S i2sResidual;
    std::size_t denseResidualRowsPerTile = 0;
    std::vector<uint16_t> denseResidualRowIdx;
    std::vector<float> denseResidualWeights;

    bool empty() const noexcept {
        return rows == 0 || cols == 0 || bm == 0 || qweights.empty();
    }

    bool hasResidual() const noexcept {
        return !residualQweights.empty();
    }

    void clear() noexcept {
        rows = 0;
        cols = 0;
        bm = 0;
        bk = 96;
        threeK = 0;
        twoK = 0;
        nTileNum = 0;
        scale = 1.0f;
        tileScales.clear();
        qweights.clear();
        i2sMain.clear();
        residualScale = 1.0f;
        residualTileScales.clear();
        residualQweights.clear();
        i2sResidualStage.clear();
        i2sResidual.clear();
        denseResidualRowsPerTile = 0;
        denseResidualRowIdx.clear();
        denseResidualWeights.clear();
    }
};

void build_bitnet_i2s_transposed_view(BitNetMatI2S& W);
void build_bitnet_i2s_decoded_forward_cache(BitNetMatI2S& W);
void build_bitnet_i2s_decoded_transposed_cache(BitNetMatI2S& W);
void rebuild_bitnet_i2s_scale_broadcast_views(BitNetMatI2S& W);
