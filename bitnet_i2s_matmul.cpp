#include "bitnet_i2s_matmul.h"
#include "bitnet_i2_s.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

    static inline std::size_t ceil_div(std::size_t a, std::size_t b) {
        return (a + b - 1) / b;
    }

    static inline float load_forward_tile_scale(
        const BitNetMatI2S& W,
        std::size_t j,
        std::size_t kb)
    {
        const std::size_t nKBlocks = ceil_div(W.H, W.qk);
        const std::size_t jb = j / W.jq;
        const std::size_t idx = jb * nKBlocks + kb;
        return W.w_scale_tile[idx];
    }

    static inline int8_t load_weight_forward_q2(
        const uint8_t* q2,
        std::size_t H,
        std::size_t i,
        std::size_t j) noexcept
    {
        const std::size_t flat = i * H + j;
        const std::size_t byteIdx = flat >> 2;
        const uint8_t shift = static_cast<uint8_t>((flat & 3u) * 2u);
        return bitnet_i2s_load_value(q2, byteIdx, shift);
    }

    static void validate_forward_common(
        const BitNetMatI2S& W,
        std::size_t H,
        const char* who)
    {
        if (W.H != H) {
            throw std::runtime_error(std::string(who) + ": W.H mismatch");
        }
        if (W.qk == 0) {
            throw std::runtime_error(std::string(who) + ": W.qk == 0");
        }
        if (W.jq == 0) {
            throw std::runtime_error(std::string(who) + ": W.jq == 0");
        }
        if ((H % 32) != 0) {
            throw std::runtime_error(std::string(who) + ": H must be multiple of 32");
        }

        const std::size_t packedBytes = ((H * H) + 3) / 4;
        const std::size_t nKBlocks = ceil_div(H, W.qk);
        const std::size_t nJBlocks = ceil_div(H, W.jq);

        if (W.q2.size() < packedBytes) {
            throw std::runtime_error(std::string(who) + ": W.q2 too small");
        }
        if (W.w_scale_tile.size() < nJBlocks * nKBlocks) {
            throw std::runtime_error(std::string(who) + ": W.w_scale_tile too small");
        }
    }

    static void validate_backward_common(
        const BitNetMatI2S& W,
        std::size_t H,
        const char* who)
    {
        validate_forward_common(W, H, who);

        if (!W.bwd.valid()) {
            throw std::runtime_error(std::string(who) + ": W.bwd metadata invalid/not built");
        }
        if (W.bwd.ib == 0 || W.bwd.jb == 0) {
            throw std::runtime_error(std::string(who) + ": invalid bwd tile sizes");
        }
    }

} // namespace

// ============================================================
// Forward: Y += X * W
// ============================================================

void linear_THH_bitnet_i2s_accum(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMatI2S& W,
    std::size_t T,
    std::size_t H,
    std::size_t tTile)
{
    if (!Y) {
        throw std::runtime_error("linear_THH_bitnet_i2s_accum: Y null");
    }
    if (!X) {
        throw std::runtime_error("linear_THH_bitnet_i2s_accum: X null");
    }
    if (tTile == 0) {
        throw std::runtime_error("linear_THH_bitnet_i2s_accum: tTile == 0");
    }

    validate_forward_common(W, H, "linear_THH_bitnet_i2s_accum");

    const std::size_t hBlock = W.qk;
    const std::size_t nKBlocks = ceil_div(H, hBlock);
    const std::size_t nXBlocks = ceil_div(H, hBlock);

    (void)nKBlocks;

    std::vector<int8_t> x8(tTile * H, 0);
    std::vector<float> xScale(tTile * nXBlocks, 1.0f);

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);

        quantize_x_tile_i8_tiled(
            x8.data(),
            xScale.data(),
            X,
            T,
            H,
            t0,
            tCount,
            tTile,
            hBlock
        );

        for (std::size_t tt = 0; tt < tCount; ++tt) {
            const int8_t* xRow8 = x8.data() + tt * H;
            const float* xScaleRow = xScale.data();
            float* yRow = Y + (t0 + tt) * H;

            for (std::size_t j0 = 0; j0 < H; j0 += 32) {
                for (std::size_t h0 = 0; h0 < H; h0 += hBlock) {
                    const std::size_t hCount = std::min(hBlock, H - h0);
                    const std::size_t kb = h0 / hBlock;
                    const float xs = xScaleRow[kb];

                    int32_t blockAcc[32] = { 0 };
                    dot_i2s_i8_accum_i32_block32(
                        blockAcc,
                        W.q2.data(),
                        H,
                        h0,
                        hCount,
                        xRow8 + h0,
                        j0
                    );

                    for (std::size_t jj = 0; jj < 32; ++jj) {
                        const std::size_t j = j0 + jj;
                        const float ws = load_forward_tile_scale(W, j, kb);
                        yRow[j] += static_cast<float>(blockAcc[jj]) * xs * ws;
                    }
                }
            }
        }
    }
}

// ============================================================
// Forward fused QKV
// ============================================================

void linear_THH_fused_qkv_bitnet_i2s_accum(
    float* __restrict Q,
    float* __restrict K,
    float* __restrict V,
    const float* __restrict X,
    const BitNetMatI2S& Wq,
    const BitNetMatI2S& Wk,
    const BitNetMatI2S& Wv,
    std::size_t T,
    std::size_t H,
    std::size_t tTile)
{
    if (!Q) {
        throw std::runtime_error("linear_THH_fused_qkv_bitnet_i2s_accum: Q null");
    }
    if (!K) {
        throw std::runtime_error("linear_THH_fused_qkv_bitnet_i2s_accum: K null");
    }
    if (!V) {
        throw std::runtime_error("linear_THH_fused_qkv_bitnet_i2s_accum: V null");
    }
    if (!X) {
        throw std::runtime_error("linear_THH_fused_qkv_bitnet_i2s_accum: X null");
    }
    if (tTile == 0) {
        throw std::runtime_error("linear_THH_fused_qkv_bitnet_i2s_accum: tTile == 0");
    }

    validate_forward_common(Wq, H, "linear_THH_fused_qkv_bitnet_i2s_accum/Wq");
    validate_forward_common(Wk, H, "linear_THH_fused_qkv_bitnet_i2s_accum/Wk");
    validate_forward_common(Wv, H, "linear_THH_fused_qkv_bitnet_i2s_accum/Wv");

    if (Wq.qk != Wk.qk || Wq.qk != Wv.qk) {
        throw std::runtime_error("linear_THH_fused_qkv_bitnet_i2s_accum: qk mismatch");
    }
    if (Wq.jq != Wk.jq || Wq.jq != Wv.jq) {
        throw std::runtime_error("linear_THH_fused_qkv_bitnet_i2s_accum: jq mismatch");
    }

    const std::size_t hBlock = Wq.qk;
    const std::size_t nKBlocks = ceil_div(H, hBlock);
    const std::size_t nXBlocks = ceil_div(H, hBlock);

    (void)nKBlocks;

    std::vector<int8_t> x8(tTile * H, 0);
    std::vector<float> xScale(tTile * nXBlocks, 1.0f);

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);

        quantize_x_tile_i8_tiled(
            x8.data(),
            xScale.data(),
            X,
            T,
            H,
            t0,
            tCount,
            tTile,
            hBlock
        );

        for (std::size_t tt = 0; tt < tCount; ++tt) {
            const int8_t* xRow8 = x8.data() + tt * H;
            const float* xScaleRow = xScale.data();

            float* qRow = Q + (t0 + tt) * H;
            float* kRow = K + (t0 + tt) * H;
            float* vRow = V + (t0 + tt) * H;

            for (std::size_t j0 = 0; j0 < H; j0 += 32) {
                for (std::size_t h0 = 0; h0 < H; h0 += hBlock) {
                    const std::size_t hCount = std::min(hBlock, H - h0);
                    const std::size_t kb = h0 / hBlock;
                    const float xs = xScaleRow[kb];

                    int32_t accQ[32] = { 0 };
                    int32_t accK[32] = { 0 };
                    int32_t accV[32] = { 0 };

                    dot_i2s_i8_accum_i32_block32(
                        accQ, Wq.q2.data(), H, h0, hCount, xRow8 + h0, j0
                    );
                    dot_i2s_i8_accum_i32_block32(
                        accK, Wk.q2.data(), H, h0, hCount, xRow8 + h0, j0
                    );
                    dot_i2s_i8_accum_i32_block32(
                        accV, Wv.q2.data(), H, h0, hCount, xRow8 + h0, j0
                    );

                    for (std::size_t jj = 0; jj < 32; ++jj) {
                        const std::size_t j = j0 + jj;

                        const float wsQ = load_forward_tile_scale(Wq, j, kb);
                        const float wsK = load_forward_tile_scale(Wk, j, kb);
                        const float wsV = load_forward_tile_scale(Wv, j, kb);

                        qRow[j] += static_cast<float>(accQ[jj]) * xs * wsQ;
                        kRow[j] += static_cast<float>(accK[jj]) * xs * wsK;
                        vRow[j] += static_cast<float>(accV[jj]) * xs * wsV;
                    }
                }
            }
        }
    }
}

// ============================================================
// Backward: dX += dY * W^T
// ============================================================

void linear_HH_T_bitnet_i2s_assisted_block32(
    float* __restrict dX,
    const float* __restrict dY,
    const BitNetMatI2S& W,
    std::size_t T,
    std::size_t H,
    std::size_t tTile,
    std::size_t iTile)
{
    if (!dX) {
        throw std::runtime_error("linear_HH_T_bitnet_i2s_assisted_block32: dX null");
    }
    if (!dY) {
        throw std::runtime_error("linear_HH_T_bitnet_i2s_assisted_block32: dY null");
    }
    if (tTile == 0) {
        throw std::runtime_error("linear_HH_T_bitnet_i2s_assisted_block32: tTile == 0");
    }
    if (iTile == 0) {
        throw std::runtime_error("linear_HH_T_bitnet_i2s_assisted_block32: iTile == 0");
    }
    if ((iTile % 32) != 0) {
        throw std::runtime_error("linear_HH_T_bitnet_i2s_assisted_block32: iTile must be multiple of 32");
    }

    validate_backward_common(W, H, "linear_HH_T_bitnet_i2s_assisted_block32");

    const std::size_t jBlock = W.bwd.jb;
    if ((jBlock % 32) != 0) {
        throw std::runtime_error("linear_HH_T_bitnet_i2s_assisted_block32: bwd.jb must be multiple of 32");
    }

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);

        for (std::size_t i0 = 0; i0 < H; i0 += iTile) {
            const std::size_t iCount = std::min(iTile, H - i0);

            for (std::size_t j0 = 0; j0 < H; j0 += jBlock) {
                const std::size_t jCount = std::min(jBlock, H - j0);
                if (jCount != 32) {
                    throw std::runtime_error("linear_HH_T_bitnet_i2s_assisted_block32: H must tile by 32 in j");
                }

                for (std::size_t tt = 0; tt < tCount; ++tt) {
                    const float* dyRow = dY + (t0 + tt) * H;
                    float* dxRow = dX + (t0 + tt) * H;

                    for (std::size_t ii = 0; ii < iCount; ++ii) {
                        const std::size_t i = i0 + ii;
                        float accum = 0.0f;

                        for (std::size_t jj = 0; jj < jCount; ++jj) {
                            const std::size_t j = j0 + jj;
                            const float dy = dyRow[j];
                            if (dy == 0.0f) continue;

                            const std::size_t kb = j / W.qk;
                            const float ws = load_forward_tile_scale(W, j, kb);
                            const int8_t w = load_weight_forward_q2(W.q2.data(), H, i, j);

                            accum += dy * (static_cast<float>(w) * ws);
                        }

                        dxRow[i] += accum;
                    }
                }
            }
        }
    }
}

// ============================================================
// Backward fused QKV input gradient:
// dX += dQ * Wq^T + dK * Wk^T + dV * Wv^T
// ============================================================

void qkv_input_backward_bitnet_i2s_assisted(
    float* __restrict dX,
    const float* __restrict dQ,
    const float* __restrict dK,
    const float* __restrict dV,
    const BitNetMatI2S& Wq,
    const BitNetMatI2S& Wk,
    const BitNetMatI2S& Wv,
    std::size_t T,
    std::size_t H,
    std::size_t tTile,
    std::size_t iTile)
{
    if (!dX) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: dX null");
    }
    if (!dQ) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: dQ null");
    }
    if (!dK) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: dK null");
    }
    if (!dV) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: dV null");
    }
    if (tTile == 0) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: tTile == 0");
    }
    if (iTile == 0) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: iTile == 0");
    }
    if ((iTile % 32) != 0) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: iTile must be multiple of 32");
    }

    validate_backward_common(Wq, H, "qkv_input_backward_bitnet_i2s_assisted/Wq");
    validate_backward_common(Wk, H, "qkv_input_backward_bitnet_i2s_assisted/Wk");
    validate_backward_common(Wv, H, "qkv_input_backward_bitnet_i2s_assisted/Wv");

    if (Wq.qk != Wk.qk || Wq.qk != Wv.qk) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: qk mismatch");
    }
    if (Wq.jq != Wk.jq || Wq.jq != Wv.jq) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: jq mismatch");
    }
    if (Wq.bwd.jb != Wk.bwd.jb || Wq.bwd.jb != Wv.bwd.jb) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: bwd.jb mismatch");
    }

    const std::size_t jBlock = Wq.bwd.jb;
    if ((jBlock % 32) != 0) {
        throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: bwd.jb must be multiple of 32");
    }

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);

        for (std::size_t i0 = 0; i0 < H; i0 += iTile) {
            const std::size_t iCount = std::min(iTile, H - i0);

            for (std::size_t j0 = 0; j0 < H; j0 += jBlock) {
                const std::size_t jCount = std::min(jBlock, H - j0);
                if (jCount != 32) {
                    throw std::runtime_error("qkv_input_backward_bitnet_i2s_assisted: H must tile by 32 in j");
                }

                for (std::size_t tt = 0; tt < tCount; ++tt) {
                    const float* dqRow = dQ + (t0 + tt) * H;
                    const float* dkRow = dK + (t0 + tt) * H;
                    const float* dvRow = dV + (t0 + tt) * H;
                    float* dxRow = dX + (t0 + tt) * H;

                    for (std::size_t ii = 0; ii < iCount; ++ii) {
                        const std::size_t i = i0 + ii;
                        float accum = 0.0f;

                        for (std::size_t jj = 0; jj < jCount; ++jj) {
                            const std::size_t j = j0 + jj;

                            const float gq = dqRow[j];
                            const float gk = dkRow[j];
                            const float gv = dvRow[j];
                            if (gq == 0.0f && gk == 0.0f && gv == 0.0f) continue;

                            const std::size_t kb = j / Wq.qk;

                            const float wsQ = load_forward_tile_scale(Wq, j, kb);
                            const float wsK = load_forward_tile_scale(Wk, j, kb);
                            const float wsV = load_forward_tile_scale(Wv, j, kb);

                            const int8_t wq = load_weight_forward_q2(Wq.q2.data(), H, i, j);
                            const int8_t wk = load_weight_forward_q2(Wk.q2.data(), H, i, j);
                            const int8_t wv = load_weight_forward_q2(Wv.q2.data(), H, i, j);

                            accum += gq * (static_cast<float>(wq) * wsQ)
                                + gk * (static_cast<float>(wk) * wsK)
                                + gv * (static_cast<float>(wv) * wsV);
                        }

                        dxRow[i] += accum;
                    }
                }
            }
        }
    }
}