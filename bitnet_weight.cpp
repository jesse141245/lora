#include "bitnet_weight.h"
#include "bitnet_i2_s.h"

#include <stdexcept>
#include <vector>

void build_bitnet_i2s_transposed_view(BitNetMatI2S& W) {
    if (W.rows == 0 || W.cols == 0) throw std::runtime_error("build_bitnet_i2s_transposed_view: Empty matrix");
    if (W.qk == 0 || W.jq == 0) throw std::runtime_error("build_bitnet_i2s_transposed_view: Block size 0");
    if (W.q2.empty() || W.w_scale_tile.empty()) throw std::runtime_error("build_bitnet_i2s_transposed_view: W is empty");

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
            W.w_scale_tileT[kb * W.nJBlocks() + jb] = W.w_scale_tile[jb * W.nKBlocks() + kb];
        }
    }
}