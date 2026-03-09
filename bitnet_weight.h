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
// Backward-assist metadata for Option C
//
// This does NOT store a full packed W^T.
// Instead, it stores a compact description of how a backward
// kernel can read transpose-style weights directly from the
// forward-packed q2 buffer.
//
// For a logical backward access:
//   W^T[j, i] == W[i, j]
//
// we precompute:
//
//   byteIndex[t] : which byte in q2 contains the 2-bit code
//   bitShift[t]  : which 2-bit lane inside that byte (0,2,4,6)
//   scaleIndex[t]: which forward scale entry in w_scale_tile to use
//
// The metadata usually describes one canonical backward tile
// pattern (for example 32x32), which kernels can reuse.
// ------------------------------------------------------------
struct BitNetBwdAssistI2S {
    // Backward kernel tile shape
    std::size_t ib = 32;   // input-feature/output-dX tile
    std::size_t jb = 32;   // reduction/output-channel tile

    // Usually ib * jb
    std::size_t tileElems = 0;

    std::vector<uint32_t> byteIndex;
    std::vector<uint8_t>  bitShift;
    std::vector<uint32_t> scaleIndex;

    bool valid() const noexcept {
        return tileElems != 0
            && byteIndex.size() == tileElems
            && bitShift.size() == tileElems
            && scaleIndex.size() == tileElems;
    }

    void clear() {
        tileElems = 0;
        byteIndex.clear();
        bitShift.clear();
        scaleIndex.clear();
    }
};

// ------------------------------------------------------------
// Packed 2-bit BitNet I2S container (Option C)
//
// Weight matrix W is [H x H] row-major:
//
//   Y[t, j] = sum_i X[t, i] * W[i, j]
//
// Packed storage:
//   q2 : packed row-major W only
//
// Tiled scale storage:
//   qk = K/input tile height   (recommended 128)
//   jq = output-column tile    (recommended 32)
//
// Forward scale layout:
//   w_scale_tile[jb * nKBlocks + kb]
//
// where:
//   kb = i / qk
//   jb = j / jq
//
// So each scale covers one tile:
//   rows [kb*qk, (kb+1)*qk)
//   cols [jb*jq, jb*jq+jq)
//
// Backward path:
//   No full q2T / w_scale_tileT are stored.
//   Instead, compact metadata in `bwd` helps kernels access
//   transpose-style weights directly from q2.
// ------------------------------------------------------------
struct BitNetMatI2S {
    std::size_t H = 0;

    // Forward quantization / kernel layout
    std::size_t qk = 128;  // K/input tile
    std::size_t jq = 32;   // output-column tile

    // Packed 2-bit weights for W only
    std::vector<uint8_t> q2;

    // Forward tiled scales only
    std::vector<float> w_scale_tile;

    // Compact backward assist metadata (Option C)
    BitNetBwdAssistI2S bwd;

    bool empty() const noexcept {
        return H == 0 || q2.empty() || w_scale_tile.empty();
    }

    void clear() {
        H = 0;
        qk = 128;
        jq = 32;
        q2.clear();
        w_scale_tile.clear();
        bwd.clear();
    }
};

// Build compact backward-assist metadata from forward q2 + w_scale_tile.
void build_bitnet_i2s_backward_assist(BitNetMatI2S& W);