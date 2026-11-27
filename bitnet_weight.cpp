#include "bitnet_weight.h"
#include "bitnet_i2_s.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

void rebuild_bitnet_i2s_scale_broadcast_views(BitNetMatI2S& W) {
    if (!W.hasSupportedScaleLayout()) {
        throw std::runtime_error("rebuild_bitnet_i2s_scale_broadcast_views: unsupported scale layout");
    }
    const std::size_t scaleCount = W.expectedScaleCount();
    if (W.w_scale_tile.size() < scaleCount || W.w_scale_tileT.size() < scaleCount) {
        throw std::runtime_error("rebuild_bitnet_i2s_scale_broadcast_views: scale buffers are incomplete");
    }

    W.w_scale_tile_broadcast8.resize(W.expectedBroadcastScaleCount());
    W.w_scale_tileT_broadcast8.resize(W.expectedBroadcastScaleCount());

    for (std::size_t idx = 0; idx < scaleCount; ++idx) {
        const float s = W.w_scale_tile[idx];
        std::fill_n(
            W.w_scale_tile_broadcast8.data() + idx * BitNetMatI2S::scaleBroadcastWidth,
            BitNetMatI2S::scaleBroadcastWidth,
            s);
    }

    for (std::size_t idx = 0; idx < scaleCount; ++idx) {
        const float s = W.w_scale_tileT[idx];
        std::fill_n(
            W.w_scale_tileT_broadcast8.data() + idx * BitNetMatI2S::scaleBroadcastWidth,
            BitNetMatI2S::scaleBroadcastWidth,
            s);
    }
}

void build_bitnet_i2s_transposed_view(BitNetMatI2S& W) {
    if (W.rows == 0 || W.cols == 0) throw std::runtime_error("build_bitnet_i2s_transposed_view: Empty matrix");
    if (W.qk == 0 || W.jq == 0) throw std::runtime_error("build_bitnet_i2s_transposed_view: Block size 0");
    if (W.q2.empty() || W.w_scale_tile.empty()) throw std::runtime_error("build_bitnet_i2s_transposed_view: W is empty");
    if (!W.hasSupportedScaleLayout()) {
        throw std::runtime_error("build_bitnet_i2s_transposed_view: unsupported scale layout");
    }

    const std::size_t rows = W.rows;
    const std::size_t cols = W.cols;
    const std::size_t N = rows * cols;

    std::vector<int8_t> q;
    unpack_2bit_to_ternary(q, W.q2.data(), N);

    std::vector<int8_t> qT(N);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            qT[j * rows + i] = q[i * cols + j];
        }
    }
    pack_ternary_to_2bit(W.q2T, qT.data(), N);

    W.w_scale_tileT.resize(W.nKBlocks() * W.nJBlocks());
    for (std::size_t jb = 0; jb < W.nJBlocks(); ++jb) {
        for (std::size_t kb = 0; kb < W.nKBlocks(); ++kb) {
            W.w_scale_tileT[W.backwardScaleIndex(kb, jb)] =
                W.w_scale_tile[W.forwardScaleIndex(jb, kb)];
        }
    }

    rebuild_bitnet_i2s_scale_broadcast_views(W);
}

void build_bitnet_i2s_decoded_transposed_cache(BitNetMatI2S& W) {
    if (W.rows == 0 || W.cols == 0) {
        throw std::runtime_error("build_bitnet_i2s_decoded_transposed_cache: Empty matrix");
    }
    if (W.q2T.size() < W.expectedPackedBytes()) {
        throw std::runtime_error("build_bitnet_i2s_decoded_transposed_cache: q2T is incomplete");
    }

    unpack_2bit_to_ternary(W.qT8, W.q2T.data(), W.rows * W.cols);
}

void build_bitnet_i2s_decoded_forward_cache(BitNetMatI2S& W) {
    if (W.rows == 0 || W.cols == 0) {
        throw std::runtime_error("build_bitnet_i2s_decoded_forward_cache: Empty matrix");
    }
    if (W.q2.size() < W.expectedPackedBytes()) {
        throw std::runtime_error("build_bitnet_i2s_decoded_forward_cache: q2 is incomplete");
    }

    unpack_2bit_to_ternary(W.q8, W.q2.data(), W.rows * W.cols);
}
