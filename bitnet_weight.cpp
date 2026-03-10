#include "bitnet_weight.h"
#include "bitnet_i2_s.h" // Required for pack/unpack functions

#include <stdexcept>
#include <vector>

void build_bitnet_i2s_transposed_view(BitNetMatI2S& W) {
    if (W.H == 0) {
        throw std::runtime_error("build_bitnet_i2s_transposed_view: W.H == 0");
    }
    if (W.qk == 0 || W.jq == 0) {
        throw std::runtime_error("build_bitnet_i2s_transposed_view: W.qk or W.jq == 0");
    }
    if (W.q2.empty() || W.w_scale_tile.empty()) {
        throw std::runtime_error("build_bitnet_i2s_transposed_view: q2 or w_scale_tile is empty");
    }

    const std::size_t H = W.H;
    const std::size_t N = H * H;
    const std::size_t nKBlocks = W.nKBlocks();
    const std::size_t nJBlocks = W.nJBlocks();

    // 1. Unpack forward weights to ternary
    std::vector<int8_t> q;
    unpack_2bit_to_ternary(q, W.q2.data(), N);

    // 2. Transpose the ternary weights
    // W is [H x H], so W^T[j, i] = W[i, j]
    std::vector<int8_t> qT(N);
    for (std::size_t i = 0; i < H; ++i) {
        for (std::size_t j = 0; j < H; ++j) {
            qT[j * H + i] = q[i * H + j];
        }
    }

    // 3. Pack transposed weights into 2-bit storage
    pack_ternary_to_2bit(W.q2T, qT.data(), N);

    // 4. Transpose the tile scales
    // Forward scale layout: w_scale_tile[jb * nKBlocks + kb]
    // Transposed scale layout: w_scale_tileT[kb * nJBlocks + jb]
    W.w_scale_tileT.resize(nKBlocks * nJBlocks);
    for (std::size_t jb = 0; jb < nJBlocks; ++jb) {
        for (std::size_t kb = 0; kb < nKBlocks; ++kb) {
            W.w_scale_tileT[kb * nJBlocks + jb] = W.w_scale_tile[jb * nKBlocks + kb];
        }
    }
}