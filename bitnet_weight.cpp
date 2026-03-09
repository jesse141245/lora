#include "bitnet_weight.h"

#include <limits>
#include <stdexcept>

namespace {

    static inline std::size_t ceil_div(std::size_t a, std::size_t b) {
        return (a + b - 1) / b;
    }

} // namespace

void build_bitnet_i2s_backward_assist(BitNetMatI2S& W) {
    if (W.H == 0) {
        throw std::runtime_error("build_bitnet_i2s_backward_assist: W.H == 0");
    }
    if (W.qk == 0) {
        throw std::runtime_error("build_bitnet_i2s_backward_assist: W.qk == 0");
    }
    if (W.jq == 0) {
        throw std::runtime_error("build_bitnet_i2s_backward_assist: W.jq == 0");
    }
    if ((W.H % 32) != 0) {
        throw std::runtime_error("build_bitnet_i2s_backward_assist: H must be multiple of 32");
    }

    const std::size_t H = W.H;
    const std::size_t N = H * H;
    const std::size_t packedBytes = (N + 3) / 4;
    const std::size_t nKBlocks = ceil_div(H, W.qk);
    const std::size_t nJBlocks = ceil_div(H, W.jq);

    if (W.q2.size() < packedBytes) {
        throw std::runtime_error("build_bitnet_i2s_backward_assist: W.q2 too small");
    }
    if (W.w_scale_tile.size() < nJBlocks * nKBlocks) {
        throw std::runtime_error("build_bitnet_i2s_backward_assist: W.w_scale_tile too small");
    }

    BitNetBwdAssistI2S& B = W.bwd;
    if (B.ib == 0 || B.jb == 0) {
        throw std::runtime_error("build_bitnet_i2s_backward_assist: bwd tile dims must be nonzero");
    }
    if ((B.ib % 4) != 0) {
        throw std::runtime_error("build_bitnet_i2s_backward_assist: bwd.ib must be multiple of 4");
    }

    B.tileElems = B.ib * B.jb;
    B.byteIndex.resize(B.tileElems);
    B.bitShift.resize(B.tileElems);
    B.scaleIndex.resize(B.tileElems);

    for (std::size_t li = 0; li < B.ib; ++li) {
        for (std::size_t lj = 0; lj < B.jb; ++lj) {
            const std::size_t t = li * B.jb + lj;

            const std::size_t i = li;
            const std::size_t j = lj;

            const std::size_t flat = i * H + j;
            const std::size_t byteIdx = flat >> 2;
            const std::size_t shift = (flat & 3u) * 2u;

            const std::size_t kb = i / W.qk;
            const std::size_t jbBlock = j / W.jq;
            const std::size_t sidx = jbBlock * nKBlocks + kb;

            if (byteIdx >= packedBytes) {
                throw std::runtime_error("build_bitnet_i2s_backward_assist: byteIdx OOB");
            }
            if (shift > 6) {
                throw std::runtime_error("build_bitnet_i2s_backward_assist: invalid bit shift");
            }
            if (sidx >= W.w_scale_tile.size()) {
                throw std::runtime_error("build_bitnet_i2s_backward_assist: scaleIndex OOB");
            }
            if (byteIdx > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
                throw std::runtime_error("build_bitnet_i2s_backward_assist: byteIdx exceeds uint32_t");
            }
            if (sidx > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
                throw std::runtime_error("build_bitnet_i2s_backward_assist: scaleIndex exceeds uint32_t");
            }

            B.byteIndex[t] = static_cast<uint32_t>(byteIdx);
            B.bitShift[t] = static_cast<uint8_t>(shift);
            B.scaleIndex[t] = static_cast<uint32_t>(sidx);
        }
    }

    if (!B.valid()) {
        throw std::runtime_error("build_bitnet_i2s_backward_assist: invalid assist metadata");
    }
}