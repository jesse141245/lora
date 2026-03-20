#include "bitnet_tl2_matmul.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "bitnet_i2_s.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr std::size_t kTl2JBlock = 32;
constexpr std::size_t kRowsPerQuantScale = 1;
constexpr std::size_t kTripleLutSize = 16;
constexpr uint8_t kTripleIdxMask = 0x0f;
constexpr uint8_t kTripleSignBit = 0x10;

inline std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

inline int8_t ternary_quant(float scaled) {
    if (scaled > 0.5f) return 1;
    if (scaled < -0.5f) return -1;
    return 0;
}

inline std::size_t load_tile_row_block_scale_index(
    std::size_t nBlocks,
    std::size_t tileRow,
    std::size_t block)
{
    return tileRow * nBlocks + block;
}

inline void validate_tl2_common(const BitNetMatTL2& W, const char* who) {
    if (W.rows == 0 || W.cols == 0 || W.bk == 0 || (W.cols % kTl2JBlock) != 0) {
        throw std::runtime_error(std::string(who) + ": invalid TL2 matrix shape");
    }
    if (W.bk != 96 && W.bk != 192) {
        throw std::runtime_error(std::string(who) + ": only bk=96 and bk=192 are supported in this experimental path");
    }
    if ((W.rows % 32) != 0) {
        throw std::runtime_error(std::string(who) + ": rows must be a multiple of 32");
    }
}

inline uint8_t pack_triple_weight_code(int8_t q0, int8_t q1, int8_t q2) {
    const int combo = 9 * static_cast<int>(q0) + 3 * static_cast<int>(q1) + static_cast<int>(q2);
    return static_cast<uint8_t>(std::abs(combo) | ((combo < 0) ? kTripleSignBit : 0));
}

inline uint8_t pack_pair_weight_code(int8_t q0, int8_t q1) {
    return static_cast<uint8_t>(3 * static_cast<int>(q0) + static_cast<int>(q1) + 4);
}

inline void build_triple_luts(
    int32_t* __restrict out,
    const int8_t* __restrict xBlock,
    std::size_t groups)
{
    for (std::size_t g = 0; g < groups; ++g) {
        const int x0 = static_cast<int>(xBlock[g * 3 + 0]);
        const int x1 = static_cast<int>(xBlock[g * 3 + 1]);
        const int x2 = static_cast<int>(xBlock[g * 3 + 2]);
        int32_t* lut = out + g * kTripleLutSize;
        std::fill(lut, lut + kTripleLutSize, 0);
        for (int w0 = -1; w0 <= 1; ++w0) {
            for (int w1 = -1; w1 <= 1; ++w1) {
                for (int w2 = -1; w2 <= 1; ++w2) {
                    const int combo = 9 * w0 + 3 * w1 + w2;
                    if (combo >= 0) {
                        lut[combo] = w0 * x0 + w1 * x1 + w2 * x2;
                    }
                }
            }
        }
    }
}

inline void build_pair_luts(
    int16_t* __restrict out,
    const int8_t* __restrict xBlock,
    std::size_t groups)
{
    for (std::size_t g = 0; g < groups; ++g) {
        const int x0 = static_cast<int>(xBlock[g * 2 + 0]);
        const int x1 = static_cast<int>(xBlock[g * 2 + 1]);
        int16_t* lut = out + g * 9;
        for (int w0 = -1; w0 <= 1; ++w0) {
            for (int w1 = -1; w1 <= 1; ++w1) {
                lut[3 * w0 + w1 + 4] = static_cast<int16_t>(w0 * x0 + w1 * x1);
            }
        }
    }
}

inline void add_scaled_acc32_to_y(
    float* __restrict yRow,
    const int32_t* __restrict acc,
    float gain)
{
#if defined(__AVX2__)
    const __m256 gain8 = _mm256_set1_ps(gain);
    for (int off = 0; off < 32; off += 8) {
        const __m256 accF = _mm256_cvtepi32_ps(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + off)));
        __m256 y = _mm256_loadu_ps(yRow + off);
        y = _mm256_fmadd_ps(accF, gain8, y);
        _mm256_storeu_ps(yRow + off, y);
    }
#else
    for (int off = 0; off < 32; ++off) {
        yRow[off] += gain * static_cast<float>(acc[off]);
    }
#endif
}

inline void accum_triple_codes_block(
    int32_t* __restrict acc,
    const BitNetMatTL2& W,
    std::size_t jb,
    std::size_t kb,
    const int32_t* __restrict tripleLuts)
{
    const std::size_t groups = W.tripleGroupsPerBlock();
    const std::size_t base =
        (jb * W.nTl2Blocks() + kb) * W.tripleBlockBytes();
    const uint8_t* codes = W.tripleCodes.data() + base;
#if defined(__AVX2__)
    const __m256i idxMask = _mm256_set1_epi32(kTripleIdxMask);
    const __m256i signMask = _mm256_set1_epi32(kTripleSignBit);
    const __m256i zero = _mm256_setzero_si256();
    const auto lookupChunk = [&](const uint8_t* idxPtr, const int32_t* lut) -> __m256i {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(idxPtr));
        const __m256i code = _mm256_cvtepu8_epi32(raw);
        const __m256i idx = _mm256_and_si256(code, idxMask);
        __m256i vals = _mm256_i32gather_epi32(lut, idx, 4);
        const __m256i neg = _mm256_cmpgt_epi32(_mm256_and_si256(code, signMask), zero);
        return _mm256_sub_epi32(_mm256_xor_si256(vals, neg), neg);
    };
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    for (std::size_t g = 0; g < groups; ++g) {
        const uint8_t* idx = codes + g * 32;
        const int32_t* lut = tripleLuts + g * kTripleLutSize;
        acc0 = _mm256_add_epi32(acc0, lookupChunk(idx + 0, lut));
        acc1 = _mm256_add_epi32(acc1, lookupChunk(idx + 8, lut));
        acc2 = _mm256_add_epi32(acc2, lookupChunk(idx + 16, lut));
        acc3 = _mm256_add_epi32(acc3, lookupChunk(idx + 24, lut));
    }
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + 0), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + 8), acc1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + 16), acc2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + 24), acc3);
#else
    for (std::size_t g = 0; g < groups; ++g) {
        const uint8_t* idx = codes + g * 32;
        const int32_t* lut = tripleLuts + g * kTripleLutSize;
        for (int c = 0; c < 32; ++c) {
            const uint8_t code = idx[c];
            const int32_t v = lut[code & kTripleIdxMask];
            acc[c] += (code & kTripleSignBit) ? -v : v;
        }
    }
#endif
}

inline void accum_pair_codes_tail(
    int32_t* __restrict acc,
    const BitNetMatTL2& W,
    std::size_t jb,
    const int16_t* __restrict pairLuts)
{
    const std::size_t groups = W.tailPairGroups();
    const std::size_t base = jb * groups * 32;
    const uint8_t* codes = W.pairCodes.data() + base;
    for (std::size_t g = 0; g < groups; ++g) {
        const uint8_t* idx = codes + g * 32;
        const int16_t* lut = pairLuts + g * 9;
        for (int c = 0; c < 32; ++c) {
            acc[c] += lut[idx[c]];
        }
    }
}

} // namespace

void quantize_ternary_tl2_blockwise(
    BitNetMatTL2& out,
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t bk)
{
    if (!W) throw std::runtime_error("quantize_ternary_tl2_blockwise: W is null");
    if (rows == 0 || cols == 0) throw std::runtime_error("quantize_ternary_tl2_blockwise: zero dim");
    if ((cols % kTl2JBlock) != 0) throw std::runtime_error("quantize_ternary_tl2_blockwise: cols must be multiple of 32");
    if ((rows % 32) != 0) throw std::runtime_error("quantize_ternary_tl2_blockwise: rows must be multiple of 32");
    if (bk != 96 && bk != 192) {
        throw std::runtime_error("quantize_ternary_tl2_blockwise: only bk=96 and bk=192 are supported in this experimental path");
    }

    out.clear();
    out.rows = rows;
    out.cols = cols;
    out.bk = bk;
    out.tl2Rows = (rows / bk) * bk;
    out.tailRows = rows - out.tl2Rows;

    const std::size_t nJBlocks = out.nJBlocks();
    const std::size_t nTl2Blocks = out.nTl2Blocks();
    const std::size_t tripleGroups = out.tripleGroupsPerBlock();
    const std::size_t tailPairs = out.tailPairGroups();

    out.tripleCodes.resize(nJBlocks * nTl2Blocks * tripleGroups * 32);
    out.tripleScales.assign(nJBlocks * nTl2Blocks, 1.0f);

    if (out.tailRows > 0) {
        if ((out.tailRows % 2) != 0) {
            throw std::runtime_error("quantize_ternary_tl2_blockwise: tailRows must be even");
        }
        out.pairCodes.resize(nJBlocks * tailPairs * 32);
        out.tailScales.assign(nJBlocks, 1.0f);
    }

    const int nJBlocksI = static_cast<int>(nJBlocks);
    const int nTl2BlocksI = static_cast<int>(nTl2Blocks);

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jbI = 0; jbI < nJBlocksI; ++jbI) {
        for (int kbI = 0; kbI < nTl2BlocksI; ++kbI) {
            const std::size_t jb = static_cast<std::size_t>(jbI);
            const std::size_t kbIdx = static_cast<std::size_t>(kbI);
            const std::size_t j0 = jb * 32;
            const std::size_t i0 = kbIdx * bk;

            float amax = 0.0f;
            for (std::size_t i = i0; i < i0 + bk; ++i) {
                const float* row = W + i * cols + j0;
                for (std::size_t c = 0; c < 32; ++c) {
                    amax = std::max(amax, std::fabs(row[c]));
                }
            }

            const float scale = (amax > 0.0f) ? amax : 1.0f;
            const float invScale = 1.0f / scale;
            out.tripleScales[jb * nTl2Blocks + kbIdx] = scale;

            const std::size_t base = (jb * nTl2Blocks + kbIdx) * tripleGroups * 32;
            for (std::size_t g = 0; g < tripleGroups; ++g) {
                const std::size_t rowBase = i0 + g * 3;
                uint8_t* codeRow = out.tripleCodes.data() + base + g * 32;
                for (std::size_t c = 0; c < 32; ++c) {
                    const int8_t q0 = ternary_quant(W[(rowBase + 0) * cols + j0 + c] * invScale);
                    const int8_t q1 = ternary_quant(W[(rowBase + 1) * cols + j0 + c] * invScale);
                    const int8_t q2 = ternary_quant(W[(rowBase + 2) * cols + j0 + c] * invScale);
                    codeRow[c] = pack_triple_weight_code(q0, q1, q2);
                }
            }
        }
    }

    if (out.tailRows > 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int jbI = 0; jbI < nJBlocksI; ++jbI) {
            const std::size_t jb = static_cast<std::size_t>(jbI);
            const std::size_t j0 = jb * 32;
            const std::size_t i0 = out.tl2Rows;

            float amax = 0.0f;
            for (std::size_t i = i0; i < rows; ++i) {
                const float* row = W + i * cols + j0;
                for (std::size_t c = 0; c < 32; ++c) {
                    amax = std::max(amax, std::fabs(row[c]));
                }
            }

            const float scale = (amax > 0.0f) ? amax : 1.0f;
            const float invScale = 1.0f / scale;
            out.tailScales[jb] = scale;

            const std::size_t base = jb * tailPairs * 32;
            for (std::size_t g = 0; g < tailPairs; ++g) {
                const std::size_t rowBase = i0 + g * 2;
                uint8_t* codeRow = out.pairCodes.data() + base + g * 32;
                for (std::size_t c = 0; c < 32; ++c) {
                    const int8_t q0 = ternary_quant(W[(rowBase + 0) * cols + j0 + c] * invScale);
                    const int8_t q1 = ternary_quant(W[(rowBase + 1) * cols + j0 + c] * invScale);
                    codeRow[c] = pack_pair_weight_code(q0, q1);
                }
            }
        }
    }
}

void linear_forward_bitnet_tl2_accum(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMatTL2& W,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch)
{
    validate_tl2_common(W, "linear_forward_bitnet_tl2_accum");
    if (!Y || !X || !x8Scratch || !xScaleScratch) {
        throw std::runtime_error("linear_forward_bitnet_tl2_accum: null buffer");
    }

    const std::size_t inDim = W.rows;
    const std::size_t outDim = W.cols;
    const std::size_t nBlocks = ceil_div(inDim, W.bk);
    const std::size_t nTl2Blocks = W.nTl2Blocks();
    const std::size_t tripleGroups = W.tripleGroupsPerBlock();
    const std::size_t tailPairGroups = W.tailPairGroups();
    const int outBlocksI = static_cast<int>(W.nJBlocks());

    std::vector<int32_t> tripleLutsAll(nTl2Blocks * tripleGroups * kTripleLutSize);
    std::vector<int16_t> pairLutsAll(tailPairGroups * 9);

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        quantize_x_tile_i8_tiled(
            x8Scratch, xScaleScratch,
            X, T, inDim,
            t0, tCount,
            kRowsPerQuantScale, W.bk, false);

        for (std::size_t tt = 0; tt < tCount; ++tt) {
            const int8_t* xRow = x8Scratch + tt * inDim;
            for (std::size_t kb = 0; kb < nTl2Blocks; ++kb) {
                build_triple_luts(
                    tripleLutsAll.data() + kb * tripleGroups * kTripleLutSize,
                    xRow + kb * W.bk,
                    tripleGroups);
            }
            if (W.tailRows > 0) {
                build_pair_luts(pairLutsAll.data(), xRow + W.tl2Rows, tailPairGroups);
            }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int jbI = 0; jbI < outBlocksI; ++jbI) {
                const std::size_t jb = static_cast<std::size_t>(jbI);
                float* yBlock = Y + (t0 + tt) * outDim + jb * 32;
                for (std::size_t kb = 0; kb < nTl2Blocks; ++kb) {
                    const float xScale = xScaleScratch[load_tile_row_block_scale_index(nBlocks, tt, kb)];
                    alignas(32) int32_t acc[32] = {};
                    accum_triple_codes_block(
                        acc, W, jb, kb,
                        tripleLutsAll.data() + kb * tripleGroups * kTripleLutSize);
                    const float gain = xScale * W.tripleScales[jb * nTl2Blocks + kb];
                    add_scaled_acc32_to_y(yBlock, acc, gain);
                }
                if (W.tailRows > 0) {
                    alignas(32) int32_t acc[32] = {};
                    const float xScale = xScaleScratch[load_tile_row_block_scale_index(nBlocks, tt, nTl2Blocks)];
                    accum_pair_codes_tail(acc, W, jb, pairLutsAll.data());
                    const float gain = xScale * W.tailScales[jb];
                    add_scaled_acc32_to_y(yBlock, acc, gain);
                }
            }
        }
    }
}

void linear_fused_dual_forward_bitnet_tl2_accum(
    float* __restrict Ya,
    float* __restrict Yb,
    const float* __restrict X,
    const BitNetMatTL2& Wa,
    const BitNetMatTL2& Wb,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch)
{
    validate_tl2_common(Wa, "linear_fused_dual_forward_bitnet_tl2_accum/Wa");
    validate_tl2_common(Wb, "linear_fused_dual_forward_bitnet_tl2_accum/Wb");
    if (Wa.rows != Wb.rows || Wa.cols != Wb.cols || Wa.bk != Wb.bk ||
        Wa.tl2Rows != Wb.tl2Rows || Wa.tailRows != Wb.tailRows) {
        throw std::runtime_error("linear_fused_dual_forward_bitnet_tl2_accum: incompatible matrix shapes");
    }

    const std::size_t inDim = Wa.rows;
    const std::size_t outDim = Wa.cols;
    const std::size_t nBlocks = ceil_div(inDim, Wa.bk);
    const std::size_t nTl2Blocks = Wa.nTl2Blocks();
    const std::size_t tripleGroups = Wa.tripleGroupsPerBlock();
    const std::size_t tailPairGroups = Wa.tailPairGroups();
    const int outBlocksI = static_cast<int>(Wa.nJBlocks());

    std::vector<int32_t> tripleLutsAll(nTl2Blocks * tripleGroups * kTripleLutSize);
    std::vector<int16_t> pairLutsAll(tailPairGroups * 9);

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        quantize_x_tile_i8_tiled(
            x8Scratch, xScaleScratch,
            X, T, inDim,
            t0, tCount,
            kRowsPerQuantScale, Wa.bk, false);

        for (std::size_t tt = 0; tt < tCount; ++tt) {
            const int8_t* xRow = x8Scratch + tt * inDim;
            for (std::size_t kb = 0; kb < nTl2Blocks; ++kb) {
                build_triple_luts(
                    tripleLutsAll.data() + kb * tripleGroups * kTripleLutSize,
                    xRow + kb * Wa.bk,
                    tripleGroups);
            }
            if (Wa.tailRows > 0) {
                build_pair_luts(pairLutsAll.data(), xRow + Wa.tl2Rows, tailPairGroups);
            }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int jbI = 0; jbI < outBlocksI; ++jbI) {
                const std::size_t jb = static_cast<std::size_t>(jbI);
                float* yBlockA = Ya + (t0 + tt) * outDim + jb * 32;
                float* yBlockB = Yb + (t0 + tt) * outDim + jb * 32;
                for (std::size_t kb = 0; kb < nTl2Blocks; ++kb) {
                    const float xScale = xScaleScratch[load_tile_row_block_scale_index(nBlocks, tt, kb)];
                    alignas(32) int32_t accA[32] = {};
                    alignas(32) int32_t accB[32] = {};
                    accum_triple_codes_block(accA, Wa, jb, kb, tripleLutsAll.data() + kb * tripleGroups * kTripleLutSize);
                    accum_triple_codes_block(accB, Wb, jb, kb, tripleLutsAll.data() + kb * tripleGroups * kTripleLutSize);
                    add_scaled_acc32_to_y(yBlockA, accA, xScale * Wa.tripleScales[jb * nTl2Blocks + kb]);
                    add_scaled_acc32_to_y(yBlockB, accB, xScale * Wb.tripleScales[jb * nTl2Blocks + kb]);
                }
                if (Wa.tailRows > 0) {
                    const float xScale = xScaleScratch[load_tile_row_block_scale_index(nBlocks, tt, nTl2Blocks)];
                    alignas(32) int32_t accA[32] = {};
                    alignas(32) int32_t accB[32] = {};
                    accum_pair_codes_tail(accA, Wa, jb, pairLutsAll.data());
                    accum_pair_codes_tail(accB, Wb, jb, pairLutsAll.data());
                    add_scaled_acc32_to_y(yBlockA, accA, xScale * Wa.tailScales[jb]);
                    add_scaled_acc32_to_y(yBlockB, accB, xScale * Wb.tailScales[jb]);
                }
            }
        }
    }
}

void linear_fused_qkv_forward_bitnet_tl2_accum(
    float* __restrict Q,
    float* __restrict K,
    float* __restrict V,
    const float* __restrict X,
    const BitNetMatTL2& Wq,
    const BitNetMatTL2& Wk,
    const BitNetMatTL2& Wv,
    std::size_t T,
    std::size_t tTile,
    int8_t* __restrict x8Scratch,
    float* __restrict xScaleScratch)
{
    validate_tl2_common(Wq, "linear_fused_qkv_forward_bitnet_tl2_accum/Wq");
    validate_tl2_common(Wk, "linear_fused_qkv_forward_bitnet_tl2_accum/Wk");
    validate_tl2_common(Wv, "linear_fused_qkv_forward_bitnet_tl2_accum/Wv");
    if (Wq.rows != Wk.rows || Wq.rows != Wv.rows ||
        Wq.bk != Wk.bk || Wq.bk != Wv.bk ||
        Wq.tl2Rows != Wk.tl2Rows || Wq.tl2Rows != Wv.tl2Rows ||
        Wq.tailRows != Wk.tailRows || Wq.tailRows != Wv.tailRows) {
        throw std::runtime_error("linear_fused_qkv_forward_bitnet_tl2_accum: incompatible Q/K/V shapes");
    }

    const std::size_t inDim = Wq.rows;
    const std::size_t nBlocks = ceil_div(inDim, Wq.bk);
    const std::size_t nTl2Blocks = Wq.nTl2Blocks();
    const std::size_t tripleGroups = Wq.tripleGroupsPerBlock();
    const std::size_t tailPairGroups = Wq.tailPairGroups();
    const std::size_t qBlocks = Wq.nJBlocks();
    const std::size_t kBlocks = Wk.nJBlocks();
    const std::size_t vBlocks = Wv.nJBlocks();
    const int maxBlocksI = static_cast<int>(std::max(qBlocks, std::max(kBlocks, vBlocks)));

    std::vector<int32_t> tripleLutsAll(nTl2Blocks * tripleGroups * kTripleLutSize);
    std::vector<int16_t> pairLutsAll(tailPairGroups * 9);

    for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
        const std::size_t tCount = std::min(tTile, T - t0);
        quantize_x_tile_i8_tiled(
            x8Scratch, xScaleScratch,
            X, T, inDim,
            t0, tCount,
            kRowsPerQuantScale, Wq.bk, false);

        for (std::size_t tt = 0; tt < tCount; ++tt) {
            const int8_t* xRow = x8Scratch + tt * inDim;
            for (std::size_t kb = 0; kb < nTl2Blocks; ++kb) {
                build_triple_luts(
                    tripleLutsAll.data() + kb * tripleGroups * kTripleLutSize,
                    xRow + kb * Wq.bk,
                    tripleGroups);
            }
            if (Wq.tailRows > 0) {
                build_pair_luts(pairLutsAll.data(), xRow + Wq.tl2Rows, tailPairGroups);
            }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int jbI = 0; jbI < maxBlocksI; ++jbI) {
                const std::size_t jb = static_cast<std::size_t>(jbI);
                for (std::size_t kb = 0; kb < nTl2Blocks; ++kb) {
                    const float xScale = xScaleScratch[load_tile_row_block_scale_index(nBlocks, tt, kb)];
                    if (jb < qBlocks) {
                        alignas(32) int32_t accQ[32] = {};
                        accum_triple_codes_block(accQ, Wq, jb, kb, tripleLutsAll.data() + kb * tripleGroups * kTripleLutSize);
                        add_scaled_acc32_to_y(
                            Q + (t0 + tt) * Wq.cols + jb * 32,
                            accQ,
                            xScale * Wq.tripleScales[jb * nTl2Blocks + kb]);
                    }
                    if (jb < kBlocks) {
                        alignas(32) int32_t accK[32] = {};
                        accum_triple_codes_block(accK, Wk, jb, kb, tripleLutsAll.data() + kb * tripleGroups * kTripleLutSize);
                        add_scaled_acc32_to_y(
                            K + (t0 + tt) * Wk.cols + jb * 32,
                            accK,
                            xScale * Wk.tripleScales[jb * nTl2Blocks + kb]);
                    }
                    if (jb < vBlocks) {
                        alignas(32) int32_t accV[32] = {};
                        accum_triple_codes_block(accV, Wv, jb, kb, tripleLutsAll.data() + kb * tripleGroups * kTripleLutSize);
                        add_scaled_acc32_to_y(
                            V + (t0 + tt) * Wv.cols + jb * 32,
                            accV,
                            xScale * Wv.tripleScales[jb * nTl2Blocks + kb]);
                    }
                }
                if (Wq.tailRows > 0) {
                    const float xScale = xScaleScratch[load_tile_row_block_scale_index(nBlocks, tt, nTl2Blocks)];
                    if (jb < qBlocks) {
                        alignas(32) int32_t accQ[32] = {};
                        accum_pair_codes_tail(accQ, Wq, jb, pairLutsAll.data());
                        add_scaled_acc32_to_y(
                            Q + (t0 + tt) * Wq.cols + jb * 32,
                            accQ,
                            xScale * Wq.tailScales[jb]);
                    }
                    if (jb < kBlocks) {
                        alignas(32) int32_t accK[32] = {};
                        accum_pair_codes_tail(accK, Wk, jb, pairLutsAll.data());
                        add_scaled_acc32_to_y(
                            K + (t0 + tt) * Wk.cols + jb * 32,
                            accK,
                            xScale * Wk.tailScales[jb]);
                    }
                    if (jb < vBlocks) {
                        alignas(32) int32_t accV[32] = {};
                        accum_pair_codes_tail(accV, Wv, jb, pairLutsAll.data());
                        add_scaled_acc32_to_y(
                            V + (t0 + tt) * Wv.cols + jb * 32,
                            accV,
                            xScale * Wv.tailScales[jb]);
                    }
                }
            }
        }
    }
}
