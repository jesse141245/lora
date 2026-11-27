#include "bitnet_tl2_generated.h"
#include "env_utils.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#define GGML_BITNET_X86_TL2
#include "qlora_tl2_kernels.h"
#include "bitnet_i2_s.h"
#include "bitnet_i2s_matmul.h"
#include "bitnet_quant.h"
#include "kernels.h"

namespace {

constexpr std::size_t kBk = 96;
constexpr std::size_t kBmInner = 32;
constexpr std::array<int, 32> kRowPerm = {
    0, 1, 2, 3, 4, 5, 6, 7,
    16, 17, 18, 19, 20, 21, 22, 23,
    8, 9, 10, 11, 12, 13, 14, 15,
    24, 25, 26, 27, 28, 29, 30, 31
};

inline int8_t ternary_quant(float scaled) {
    const int q = static_cast<int>(std::lrint(scaled));
    if (q > 1) return 1;
    if (q < -1) return -1;
    return static_cast<int8_t>(q);
}

enum class Tl2QuantMode {
    MeanAbs,
    AbsMax,
    Hybrid,
    Refit,
};

enum class Tl2ScaleMode {
    Global,
    PerTile,
};

std::optional<std::string> scopedTl2Env(const char* scope, const char* suffix) {
    if (scope && *scope) {
        std::string scoped = "QLORA_TEST_BITNET_TL2_";
        scoped += scope;
        scoped += "_";
        scoped += suffix;
        if (auto value = envutil::tryGetEnvString(scoped.c_str())) {
            return value;
        }
    }
    std::string global = "QLORA_TEST_BITNET_TL2_";
    global += suffix;
    return envutil::tryGetEnvString(global.c_str());
}

Tl2QuantMode parseTl2QuantMode(const std::optional<std::string>& value) {
    if (!value) return Tl2QuantMode::MeanAbs;
    const std::string s = envutil::lowerCopy(*value);
    if (s == "absmax" || s == "max") return Tl2QuantMode::AbsMax;
    if (s == "hybrid" || s == "best" || s == "auto") return Tl2QuantMode::Hybrid;
    if (s == "refit" || s == "lsq") return Tl2QuantMode::Refit;
    return Tl2QuantMode::MeanAbs;
}

Tl2ScaleMode parseTl2ScaleMode(const std::optional<std::string>& value, Tl2ScaleMode fallback) {
    if (!value) return fallback;
    const std::string s = envutil::lowerCopy(*value);
    if (s == "per_tile" || s == "tile" || s == "block") return Tl2ScaleMode::PerTile;
    return Tl2ScaleMode::Global;
}

bool parseBoolString(const std::optional<std::string>& value, bool fallback) {
    if (!value) return fallback;
    const std::string s = envutil::lowerCopy(*value);
    if (envutil::envStringTruthy(s)) return true;
    if (envutil::envStringFalsy(s)) return false;
    return fallback;
}

int parseIntString(const std::optional<std::string>& value, int fallback, int lo, int hi) {
    if (!value) return fallback;
    try {
        return std::clamp(std::stoi(*value), lo, hi);
    }
    catch (...) {
        return fallback;
    }
}

std::size_t parseQkString(const std::optional<std::string>& value, std::size_t fallback) {
    if (!value) return fallback;
    try {
        const int parsed = std::stoi(*value);
        const int clamped = std::max(32, parsed);
        return static_cast<std::size_t>((clamped + 31) / 32 * 32);
    }
    catch (...) {
        return fallback;
    }
}

struct Tl2RuntimeOptions {
    Tl2QuantMode quantMode = Tl2QuantMode::MeanAbs;
    Tl2ScaleMode scaleMode = Tl2ScaleMode::Global;
    Tl2ScaleMode residualScaleMode = Tl2ScaleMode::Global;
    int stageCount = 1;
    int denseResidualRows = 0;
    bool i2sResidual = false;
    bool execI2S = false;
    std::size_t i2sQk = 64;
};

Tl2RuntimeOptions tl2OptionsForScope(const char* scope) {
    Tl2RuntimeOptions opt;
    opt.quantMode = parseTl2QuantMode(scopedTl2Env(scope, "QUANT"));
    opt.scaleMode = parseTl2ScaleMode(scopedTl2Env(scope, "SCALE"), Tl2ScaleMode::Global);
    opt.residualScaleMode = parseTl2ScaleMode(scopedTl2Env(scope, "RESID_SCALE"), opt.scaleMode);
    opt.stageCount = parseIntString(scopedTl2Env(scope, "STAGES"), 1, 1, 2);
    opt.denseResidualRows = parseIntString(scopedTl2Env(scope, "DENSE_RESID_ROWS"), 0, 0, 16);
    opt.i2sResidual = parseBoolString(scopedTl2Env(scope, "I2S_RESIDUAL"), false);
    opt.execI2S = parseBoolString(scopedTl2Env(scope, "EXEC_I2S"), false);
    opt.i2sQk = parseQkString(scopedTl2Env(scope, "I2S_QK"), std::size_t{64});
    return opt;
}

void build_tl2_stage_i2s_exact(
    BitNetMatI2S& out,
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t bm,
    std::size_t nTileNum,
    const std::vector<float>& tileScales,
    float globalScale,
    std::size_t qk)
{
    const std::size_t jq = 32;
    const std::size_t nKBlocks = (rows + qk - 1) / qk;
    const std::size_t nJBlocks = cols / jq;

    out.clear();
    out.rows = rows;
    out.cols = cols;
    out.hadamard32 = false;
    out.qk = qk;
    out.jq = jq;
    out.w_scale_tile.assign(nJBlocks * nKBlocks, 1.0f);

    std::vector<int8_t> q(rows * cols, 0);
    for (std::size_t jb = 0; jb < nJBlocks; ++jb) {
        const std::size_t tile = (jb * jq) / bm;
        const float tileScale = tileScales.empty() ? globalScale : tileScales[tile];
        for (std::size_t kb = 0; kb < nKBlocks; ++kb) {
            out.w_scale_tile[jb * nKBlocks + kb] = tileScale;
        }
    }

    for (std::size_t tile = 0; tile < nTileNum; ++tile) {
        const std::size_t m0 = tile * bm;
        const float tileScale = tileScales.empty() ? globalScale : tileScales[tile];
        const float invScale = 1.0f / ((tileScale < 1e-5f) ? 1e-5f : tileScale);
        for (std::size_t r = 0; r < rows; ++r) {
            const float* src = W + r * cols + m0;
            int8_t* dst = q.data() + r * cols + m0;
            for (std::size_t c = 0; c < bm; ++c) {
                dst[c] = ternary_quant(src[c] * invScale);
            }
        }
    }

    pack_ternary_to_2bit(out.q2, q.data(), rows * cols);
    build_bitnet_i2s_transposed_view(out);
}

inline bool select_codegen_shape(
    std::size_t rows,
    std::size_t cols,
    std::size_t& bm)
{
    if (rows == 6912 && cols == 2560) {
        bm = 256;
        return true;
    }
    if (rows == 2560 && cols == 6912) {
        bm = 256;
        return true;
    }
    if (rows == 2560 && cols == 2560) {
        bm = 128;
        return true;
    }
    if (rows == 2560 && cols == 640) {
        bm = 128;
        return true;
    }
    return false;
}

inline uint8_t triple_abs_code(
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t row,
    std::size_t kBase,
    float invScale)
{
    const int8_t q0 = ternary_quant(W[(kBase + 0) * cols + row] * invScale);
    const int8_t q1 = ternary_quant(W[(kBase + 1) * cols + row] * invScale);
    const int8_t q2 = ternary_quant(W[(kBase + 2) * cols + row] * invScale);
    const int combo = 9 * static_cast<int>(q0) + 3 * static_cast<int>(q1) + static_cast<int>(q2);
    return static_cast<uint8_t>(std::abs(combo));
}

inline bool triple_sign_bit(
    const float* W,
    std::size_t cols,
    std::size_t row,
    std::size_t kBase,
    float invScale)
{
    const int8_t q0 = ternary_quant(W[(kBase + 0) * cols + row] * invScale);
    const int8_t q1 = ternary_quant(W[(kBase + 1) * cols + row] * invScale);
    const int8_t q2 = ternary_quant(W[(kBase + 2) * cols + row] * invScale);
    const int combo = 9 * static_cast<int>(q0) + 3 * static_cast<int>(q1) + static_cast<int>(q2);
    return combo < 0;
}

inline uint8_t pair_abs_code(
    const float* W,
    std::size_t cols,
    std::size_t row,
    std::size_t kBase,
    float invScale)
{
    const int8_t q0 = ternary_quant(W[(kBase + 0) * cols + row] * invScale);
    const int8_t q1 = ternary_quant(W[(kBase + 1) * cols + row] * invScale);
    return static_cast<uint8_t>(3 * static_cast<int>(q0) + static_cast<int>(q1) + 4);
}

inline std::size_t three_abs_bytes(std::size_t m, std::size_t threeK) {
    return threeK * m / 6;
}

inline std::size_t three_sign_bytes(std::size_t m, std::size_t threeK) {
    return threeK * m / 24;
}

inline std::size_t two_abs_bytes(std::size_t m, std::size_t twoK) {
    return twoK * m / 4;
}

inline float clamp_scale(float scale) {
    return (scale < 1e-5f) ? 1e-5f : scale;
}

float compute_tile_scale(
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t col0,
    std::size_t colCount,
    Tl2QuantMode mode)
{
    float scale = 0.0f;
    if (mode == Tl2QuantMode::AbsMax) {
        for (std::size_t r = 0; r < rows; ++r) {
            const float* rowPtr = W + r * cols + col0;
            for (std::size_t c = 0; c < colCount; ++c) {
                scale = std::max(scale, std::fabs(rowPtr[c]));
            }
        }
        return clamp_scale(scale);
    }

    for (std::size_t r = 0; r < rows; ++r) {
        const float* rowPtr = W + r * cols + col0;
        for (std::size_t c = 0; c < colCount; ++c) {
            scale += std::fabs(rowPtr[c]);
        }
    }
    scale /= static_cast<float>(rows * colCount);
    return clamp_scale(scale);
}

float compute_tile_quant_error(
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t col0,
    std::size_t colCount,
    float scale)
{
    const float invScale = 1.0f / clamp_scale(scale);
    float err = 0.0f;
    for (std::size_t r = 0; r < rows; ++r) {
        const float* rowPtr = W + r * cols + col0;
        for (std::size_t c = 0; c < colCount; ++c) {
            const float w = rowPtr[c];
            const float q = static_cast<float>(ternary_quant(w * invScale)) * scale;
            const float diff = w - q;
            err += diff * diff;
        }
    }
    return err;
}

float select_tile_scale(
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t col0,
    std::size_t colCount,
    Tl2QuantMode mode)
{
    if (mode != Tl2QuantMode::Hybrid) {
        return compute_tile_scale(W, rows, cols, col0, colCount, mode);
    }

    const float meanAbs = compute_tile_scale(W, rows, cols, col0, colCount, Tl2QuantMode::MeanAbs);
    const float absMax = compute_tile_scale(W, rows, cols, col0, colCount, Tl2QuantMode::AbsMax);
    const float meanErr = compute_tile_quant_error(W, rows, cols, col0, colCount, meanAbs);
    const float maxErr = compute_tile_quant_error(W, rows, cols, col0, colCount, absMax);
    return (maxErr < meanErr) ? absMax : meanAbs;
}

float refit_tile_scale(
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t col0,
    std::size_t colCount)
{
    float scale = select_tile_scale(W, rows, cols, col0, colCount, Tl2QuantMode::Hybrid);
    for (int iter = 0; iter < 3; ++iter) {
        const float invScale = 1.0f / clamp_scale(scale);
        float numer = 0.0f;
        float denom = 0.0f;
        for (std::size_t r = 0; r < rows; ++r) {
            const float* rowPtr = W + r * cols + col0;
            for (std::size_t c = 0; c < colCount; ++c) {
                const float w = rowPtr[c];
                const float q = static_cast<float>(ternary_quant(w * invScale));
                numer += w * q;
                denom += q * q;
            }
        }
        if (denom <= 0.0f) {
            break;
        }
        const float nextScale = clamp_scale(numer / denom);
        if (std::fabs(nextScale - scale) <= (1e-4f * scale)) {
            scale = nextScale;
            break;
        }
        scale = nextScale;
    }
    return clamp_scale(scale);
}

float choose_tile_scale(
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t col0,
    std::size_t colCount,
    Tl2QuantMode mode)
{
    if (mode == Tl2QuantMode::Refit) {
        return refit_tile_scale(W, rows, cols, col0, colCount);
    }
    return select_tile_scale(W, rows, cols, col0, colCount, mode);
}

bool pack_tl2_generated_stage(
    std::vector<uint8_t>& qweights,
    std::vector<float>& tileScales,
    float& globalScale,
    std::size_t rows,
    std::size_t cols,
    std::size_t bm,
    std::size_t threeK,
    std::size_t twoK,
    std::size_t nTileNum,
    const float* W,
    Tl2QuantMode quantMode,
    Tl2ScaleMode scaleMode)
{
    const float scale = choose_tile_scale(W, rows, cols, 0, cols, quantMode);
    globalScale = scale;

    const std::size_t absBytes = three_abs_bytes(cols, threeK);
    const std::size_t signBytes = three_sign_bytes(cols, threeK);
    const std::size_t twoBytes = two_abs_bytes(cols, twoK);
    qweights.assign(absBytes + signBytes + twoBytes, uint8_t{0});
    tileScales.assign(nTileNum, scale);

    const std::size_t rowBlocksPerTile = bm / kBmInner;
    const std::size_t tripleTiles = threeK / kBk;
    const std::size_t absTileBytes = three_abs_bytes(bm, kBk);
    const std::size_t signTileBytes = three_sign_bytes(bm, kBk);
    const std::size_t by = 192 / kBmInner;

    for (std::size_t tile = 0; tile < nTileNum; ++tile) {
        const std::size_t m0 = tile * bm;
        float tileScale = scale;
        if (scaleMode == Tl2ScaleMode::PerTile) {
            tileScale = choose_tile_scale(W, rows, cols, m0, bm, quantMode);
        }
        tileScales[tile] = tileScale;
        const float tileInvScale = 1.0f / tileScale;
        const std::size_t tileAbsBase = tile * three_abs_bytes(bm, threeK);
        const std::size_t tileSignBase = absBytes + tile * three_sign_bytes(bm, threeK);

        for (std::size_t kt = 0; kt < tripleTiles; ++kt) {
            const std::size_t k0 = kt * kBk;
            const std::size_t absBase = tileAbsBase + kt * absTileBytes;
            const std::size_t signBase = tileSignBase + kt * signTileBytes;

            for (std::size_t rb = 0; rb < rowBlocksPerTile; ++rb) {
                const std::size_t rowBase = m0 + rb * kBmInner;
                const std::size_t absRowBase = absBase + rb * 16 * 32;
                const std::size_t signRowBase = signBase + rb * 4 * 32;

                for (std::size_t pg = 0; pg < 16; ++pg) {
                    const std::size_t g0 = k0 + (2 * pg + 0) * 3;
                    const std::size_t g1 = k0 + (2 * pg + 1) * 3;
                    uint8_t* dst = qweights.data() + absRowBase + pg * 32;
                    for (int b = 0; b < 32; ++b) {
                        const std::size_t row = rowBase + static_cast<std::size_t>(kRowPerm[b]);
                        const uint8_t c0 = triple_abs_code(W, rows, cols, row, g0, tileInvScale);
                        const uint8_t c1 = triple_abs_code(W, rows, cols, row, g1, tileInvScale);
                        dst[b] = static_cast<uint8_t>((c0 << 4) | c1);
                    }
                }

                for (std::size_t sg = 0; sg < (kBk / (by * 4)); ++sg) {
                    uint8_t* dst = qweights.data() + signRowBase + sg * 32;
                    const std::size_t groupBase = sg * (by / 3 * 4);
                    for (std::size_t pair = 0; pair < 16; ++pair) {
                        const std::size_t rowLo = rowBase + pair;
                        const std::size_t rowHi = rowBase + 16 + pair;
                        uint16_t word = 0;
                        for (std::size_t g = 0; g < (by / 3 * 4); ++g) {
                            const std::size_t tripleGroup = groupBase + g;
                            const std::size_t kTriple = k0 + tripleGroup * 3;
                            if (triple_sign_bit(W, cols, rowLo, kTriple, tileInvScale)) {
                                word |= static_cast<uint16_t>(1u << (15 - (2 * g + 0)));
                            }
                            if (triple_sign_bit(W, cols, rowHi, kTriple, tileInvScale)) {
                                word |= static_cast<uint16_t>(1u << (15 - (2 * g + 1)));
                            }
                        }
                        dst[2 * pair + 0] = static_cast<uint8_t>(word & 0xff);
                        dst[2 * pair + 1] = static_cast<uint8_t>((word >> 8) & 0xff);
                    }
                }
            }
        }
    }

    if (twoK > 0) {
        const std::size_t twoBase = absBytes + signBytes;
        const std::size_t twoTileBytes = two_abs_bytes(bm, twoK);
        const std::size_t tailChunks = twoK / 32;
        const std::size_t rowBlocksPerTile = bm / kBmInner;

        for (std::size_t tile = 0; tile < nTileNum; ++tile) {
            const std::size_t m0 = tile * bm;
            const std::size_t tileTwoBase = twoBase + tile * twoTileBytes;
            float tileScale = globalScale;
            if (!tileScales.empty() && tile < tileScales.size()) {
                tileScale = tileScales[tile];
            }
            const float tileInvScale = 1.0f / tileScale;
            for (std::size_t tc = 0; tc < tailChunks; ++tc) {
                const std::size_t k0 = threeK + tc * 32;
                const std::size_t chunkBase = tileTwoBase + tc * (bm * 32 / 4);
                for (std::size_t rb = 0; rb < rowBlocksPerTile; ++rb) {
                    const std::size_t rowBase = m0 + rb * kBmInner;
                    const std::size_t rowChunkBase = chunkBase + rb * 8 * 32;
                    for (std::size_t pg = 0; pg < 8; ++pg) {
                        const std::size_t p0 = k0 + (2 * pg + 0) * 2;
                        const std::size_t p1 = k0 + (2 * pg + 1) * 2;
                        uint8_t* dst = qweights.data() + rowChunkBase + pg * 32;
                        for (int b = 0; b < 32; ++b) {
                            const std::size_t row = rowBase + static_cast<std::size_t>(kRowPerm[b]);
                            const uint8_t c0 = pair_abs_code(W, cols, row, p0, tileInvScale);
                            const uint8_t c1 = pair_abs_code(W, cols, row, p1, tileInvScale);
                            dst[b] = static_cast<uint8_t>((c0 << 4) | c1);
                        }
                    }
                }
            }
        }
    }

    return true;
}

void subtract_quantized_stage_approx(
    std::vector<float>& residual,
    std::size_t rows,
    std::size_t cols,
    std::size_t bm,
    std::size_t nTileNum,
    const std::vector<float>& tileScales,
    float globalScale)
{
    for (std::size_t tile = 0; tile < nTileNum; ++tile) {
        const std::size_t m0 = tile * bm;
        const float tileScale = tileScales.empty() ? globalScale : tileScales[tile];
        const float invScale = 1.0f / tileScale;
        for (std::size_t r = 0; r < rows; ++r) {
            float* dst = residual.data() + r * cols + m0;
            for (std::size_t c = 0; c < bm; ++c) {
                const float q = static_cast<float>(ternary_quant(dst[c] * invScale)) * tileScale;
                dst[c] -= q;
            }
        }
    }
}

void build_dense_tile_residual(
    BitNetMatTL2Gen& out,
    const std::vector<float>& residual,
    int rowsPerTile)
{
    if (rowsPerTile <= 0) {
        return;
    }

    const std::size_t rows = out.rows;
    const std::size_t cols = out.cols;
    const std::size_t bm = out.bm;
    const std::size_t nTileNum = out.nTileNum;
    const std::size_t slotsPerTile = static_cast<std::size_t>(rowsPerTile);

    out.denseResidualRowsPerTile = slotsPerTile;
    out.denseResidualRowIdx.assign(nTileNum * slotsPerTile, static_cast<uint16_t>(0xffffu));
    out.denseResidualWeights.assign(nTileNum * slotsPerTile * bm, 0.0f);

    std::vector<std::pair<float, std::size_t>> norms(rows);
    for (std::size_t tile = 0; tile < nTileNum; ++tile) {
        const std::size_t m0 = tile * bm;
        for (std::size_t r = 0; r < rows; ++r) {
            const float* src = residual.data() + r * cols + m0;
            float sumSq = 0.0f;
            for (std::size_t c = 0; c < bm; ++c) {
                sumSq += src[c] * src[c];
            }
            norms[r] = { sumSq, r };
        }

        const std::size_t keep = std::min<std::size_t>(slotsPerTile, rows);
        std::partial_sort(
            norms.begin(),
            norms.begin() + keep,
            norms.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            });

        for (std::size_t s = 0; s < keep; ++s) {
            if (norms[s].first <= 0.0f) {
                break;
            }
            const std::size_t row = norms[s].second;
            out.denseResidualRowIdx[tile * slotsPerTile + s] = static_cast<uint16_t>(row);
            float* dst = out.denseResidualWeights.data() + (tile * slotsPerTile + s) * bm;
            const float* src = residual.data() + row * cols + m0;
            std::copy_n(src, bm, dst);
        }
    }
}

void tl2_generated_stage_forward(
    float* __restrict Y,
    const BitNetMatTL2Gen& W,
    const uint8_t* qweights,
    const float* tileScales,
    float globalScale,
    const std::vector<int8_t>& threeQLut,
    const std::vector<int8_t>& twoQLut,
    const std::vector<float>& lutScales,
    std::size_t T)
{
    const int m = static_cast<int>(W.cols);
    const int k = static_cast<int>(W.rows);
    const int threeK = static_cast<int>(W.threeK);
    const int twoK = static_cast<int>(W.twoK);
    const std::size_t threeAbsBytes = three_abs_bytes(W.cols, W.threeK);
    const std::size_t signBytes = three_sign_bytes(W.cols, W.threeK);
    const std::size_t twoBase = threeAbsBytes + signBytes;

    const uint8_t* sign = qweights + threeAbsBytes;
    const uint8_t* twoA = qweights + twoBase;
    constexpr int batchSizes[] = { 512, 256, 128, 32, 8, 1 };
    std::size_t start = 0;

    for (int bs : batchSizes) {
        while (start + static_cast<std::size_t>(bs) <= T) {
            void* scaleArg = tileScales
                ? const_cast<float*>(tileScales)
                : const_cast<float*>(&globalScale);
            ggml_qgemm_lut(
                bs,
                m,
                k,
                threeK,
                const_cast<uint8_t*>(qweights),
                const_cast<uint8_t*>(sign),
                const_cast<int8_t*>(threeQLut.data()) + start * (W.threeK / 3) * 32,
                scaleArg,
                const_cast<float*>(lutScales.data()) + start,
                Y + start * W.cols);

            ggml_qgemm_lut(
                bs,
                m,
                k,
                twoK,
                const_cast<uint8_t*>(twoA),
                nullptr,
                const_cast<int8_t*>(twoQLut.data()) + start * (W.twoK / 2) * 32,
                scaleArg,
                const_cast<float*>(lutScales.data()) + start,
                Y + start * W.cols);
            start += static_cast<std::size_t>(bs);
        }
    }
}

} // namespace

static bool quantize_ternary_tl2_generated_impl(
    BitNetMatTL2Gen& out,
    const float* W,
    std::size_t rows,
    std::size_t cols,
    const Tl2RuntimeOptions& opt,
    bool allowResidual,
    Tl2QuantMode quantMode,
    Tl2ScaleMode scaleMode,
    Tl2ScaleMode residualScaleMode)
{
    if (!W || rows == 0 || cols == 0 || (cols % 32) != 0) {
        return false;
    }

    std::size_t bm = 0;
    if (!select_codegen_shape(rows, cols, bm)) {
        return false;
    }

    out.clear();
    out.rows = rows;
    out.cols = cols;
    out.bm = bm;
    out.bk = kBk;
    out.threeK = (rows / kBk) * kBk;
    out.twoK = rows - out.threeK;
    out.nTileNum = cols / bm;
    if (!pack_tl2_generated_stage(
        out.qweights,
        out.tileScales,
        out.scale,
        rows,
        cols,
        bm,
        out.threeK,
        out.twoK,
        out.nTileNum,
        W,
        quantMode,
        scaleMode)) {
        out.clear();
        return false;
    }

    if (opt.execI2S) {
        build_tl2_stage_i2s_exact(
            out.i2sMain,
            W,
            rows,
            cols,
            bm,
            out.nTileNum,
            out.tileScales,
            out.scale,
            opt.i2sQk);
    }

    std::vector<float> residual(rows * cols, 0.0f);
    std::copy_n(W, rows * cols, residual.data());
    subtract_quantized_stage_approx(residual, rows, cols, bm, out.nTileNum, out.tileScales, out.scale);

    if (allowResidual && opt.stageCount > 1) {
        BitNetMatTL2Gen residualStage;
        if (quantize_ternary_tl2_generated_impl(
            residualStage,
            residual.data(),
            rows,
            cols,
            opt,
            false,
            quantMode,
            residualScaleMode,
            residualScaleMode)) {
            out.residualScale = residualStage.scale;
            out.residualTileScales = std::move(residualStage.tileScales);
            out.residualQweights = std::move(residualStage.qweights);
            out.i2sResidualStage = std::move(residualStage.i2sMain);
            subtract_quantized_stage_approx(residual, rows, cols, bm, out.nTileNum, out.residualTileScales, out.residualScale);
        }
    }

    if (opt.i2sResidual) {
        quantize_ternary_i2s_blockwise(
            out.i2sResidual,
            residual.data(),
            rows,
            cols,
            opt.i2sQk,
            32,
            false);
    }
    else {
        if (opt.denseResidualRows > 0) {
            build_dense_tile_residual(out, residual, opt.denseResidualRows);
        }
    }

    return true;
}

bool quantize_ternary_tl2_generated(
    BitNetMatTL2Gen& out,
    const float* W,
    std::size_t rows,
    std::size_t cols,
    const char* scope)
{
    const Tl2RuntimeOptions opt = tl2OptionsForScope(scope);
    return quantize_ternary_tl2_generated_impl(
        out,
        W,
        rows,
        cols,
        opt,
        true,
        opt.quantMode,
        opt.scaleMode,
        opt.residualScaleMode);
}

void linear_forward_bitnet_tl2_generated(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMatTL2Gen& W,
    std::size_t T)
{
    if (!Y || !X || W.empty() || T == 0) {
        throw std::runtime_error("linear_forward_bitnet_tl2_generated: invalid input");
    }

    if (!W.i2sMain.empty()) {
        std::fill(Y, Y + T * W.cols, 0.0f);

        const std::size_t tTile = 2;
        const std::size_t mainKBlocks = W.i2sMain.nKBlocks();
        thread_local std::vector<int8_t> x8Scratch;
        thread_local std::vector<float> xScaleScratch;
        x8Scratch.resize(tTile * W.rows);
        xScaleScratch.resize(tTile * mainKBlocks);
        linear_forward_bitnet_i2s_accum(
            Y,
            X,
            W.i2sMain,
            T,
            tTile,
            x8Scratch.data(),
            xScaleScratch.data());

        if (!W.i2sResidualStage.empty()) {
            const std::size_t residualStageKBlocks = W.i2sResidualStage.nKBlocks();
            xScaleScratch.resize(tTile * residualStageKBlocks);
            linear_forward_bitnet_i2s_accum(
                Y,
                X,
                W.i2sResidualStage,
                T,
                tTile,
                x8Scratch.data(),
                xScaleScratch.data());
        }
    }
    else {
    const int m = static_cast<int>(W.cols);
    const int threeK = static_cast<int>(W.threeK);
    const int twoK = static_cast<int>(W.twoK);

    std::fill(Y, Y + T * W.cols, 0.0f);

    std::vector<int8_t> threeQLut(std::max<std::size_t>(1, T * (W.threeK / 3) * 32));
    std::vector<int8_t> twoQLut(std::max<std::size_t>(1, T * (W.twoK / 2) * 32));
    std::vector<float> lutScales(T, 0.0f);

    ggml_preprocessor(
        static_cast<int>(T),
        m,
        threeK,
        twoK,
        const_cast<float*>(X),
        lutScales.data(),
        threeQLut.data(),
        twoQLut.data());
    tl2_generated_stage_forward(
        Y,
        W,
        W.qweights.data(),
        W.tileScales.empty() ? nullptr : W.tileScales.data(),
        W.scale,
        threeQLut,
        twoQLut,
        lutScales,
        T);
    if (W.hasResidual()) {
        tl2_generated_stage_forward(
            Y,
            W,
            W.residualQweights.data(),
            W.residualTileScales.empty() ? nullptr : W.residualTileScales.data(),
            W.residualScale,
            threeQLut,
            twoQLut,
            lutScales,
            T);
    }
    }

    if (W.denseResidualRowsPerTile != 0 && !W.denseResidualWeights.empty() && !W.denseResidualRowIdx.empty()) {
        const std::size_t rowsPerTile = W.denseResidualRowsPerTile;
        const std::size_t bm = W.bm;
        for (std::size_t t = 0; t < T; ++t) {
            const float* xRow = X + t * W.rows;
            float* yRow = Y + t * W.cols;
            for (std::size_t tile = 0; tile < W.nTileNum; ++tile) {
                float* yTile = yRow + tile * bm;
                for (std::size_t s = 0; s < rowsPerTile; ++s) {
                    const uint16_t rowIdx = W.denseResidualRowIdx[tile * rowsPerTile + s];
                    if (rowIdx == 0xffffu || rowIdx >= W.rows) {
                        continue;
                    }
                    const float xv = xRow[rowIdx];
                    if (xv == 0.0f) {
                        continue;
                    }
                    const float* wRow = W.denseResidualWeights.data() + (tile * rowsPerTile + s) * bm;
                    axpy_inplace_avx2_or_scalar(yTile, wRow, bm, xv);
                }
            }
        }
    }

    if (!W.i2sResidual.empty()) {
        const std::size_t tTile = 2;
        const std::size_t nKBlocks = W.i2sResidual.nKBlocks();
        thread_local std::vector<int8_t> x8Scratch;
        thread_local std::vector<float> xScaleScratch;
        x8Scratch.resize(tTile * W.rows);
        xScaleScratch.resize(tTile * nKBlocks);
        linear_forward_bitnet_i2s_accum(
            Y,
            X,
            W.i2sResidual,
            T,
            tTile,
            x8Scratch.data(),
            xScaleScratch.data());
    }
}

void linear_forward_bitnet_tl2_generated_residual_only(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMatTL2Gen& W,
    std::size_t T)
{
    if (!Y || !X || W.empty() || T == 0) {
        throw std::runtime_error("linear_forward_bitnet_tl2_generated_residual_only: invalid input");
    }

    if (!W.i2sResidualStage.empty()) {
        const std::size_t tTile = 2;
        const std::size_t nKBlocks = W.i2sResidualStage.nKBlocks();
        thread_local std::vector<int8_t> x8Scratch;
        thread_local std::vector<float> xScaleScratch;
        x8Scratch.resize(tTile * W.rows);
        xScaleScratch.resize(tTile * nKBlocks);
        linear_forward_bitnet_i2s_accum(
            Y,
            X,
            W.i2sResidualStage,
            T,
            tTile,
            x8Scratch.data(),
            xScaleScratch.data());
    }
    else if (W.hasResidual()) {
        const int m = static_cast<int>(W.cols);
        const int threeK = static_cast<int>(W.threeK);
        const int twoK = static_cast<int>(W.twoK);

        std::vector<int8_t> threeQLut(std::max<std::size_t>(1, T * (W.threeK / 3) * 32));
        std::vector<int8_t> twoQLut(std::max<std::size_t>(1, T * (W.twoK / 2) * 32));
        std::vector<float> lutScales(T, 0.0f);

        ggml_preprocessor(
            static_cast<int>(T),
            m,
            threeK,
            twoK,
            const_cast<float*>(X),
            lutScales.data(),
            threeQLut.data(),
            twoQLut.data());
        tl2_generated_stage_forward(
            Y,
            W,
            W.residualQweights.data(),
            W.residualTileScales.empty() ? nullptr : W.residualTileScales.data(),
            W.residualScale,
            threeQLut,
            twoQLut,
            lutScales,
            T);
    }

    if (W.denseResidualRowsPerTile != 0 && !W.denseResidualWeights.empty() && !W.denseResidualRowIdx.empty()) {
        const std::size_t rowsPerTile = W.denseResidualRowsPerTile;
        const std::size_t bm = W.bm;
        for (std::size_t t = 0; t < T; ++t) {
            const float* xRow = X + t * W.rows;
            float* yRow = Y + t * W.cols;
            for (std::size_t tile = 0; tile < W.nTileNum; ++tile) {
                float* yTile = yRow + tile * bm;
                for (std::size_t s = 0; s < rowsPerTile; ++s) {
                    const uint16_t rowIdx = W.denseResidualRowIdx[tile * rowsPerTile + s];
                    if (rowIdx == 0xffffu || rowIdx >= W.rows) {
                        continue;
                    }
                    const float xv = xRow[rowIdx];
                    if (xv == 0.0f) {
                        continue;
                    }
                    const float* wRow = W.denseResidualWeights.data() + (tile * rowsPerTile + s) * bm;
                    axpy_inplace_avx2_or_scalar(yTile, wRow, bm, xv);
                }
            }
        }
    }

    if (!W.i2sResidual.empty()) {
        const std::size_t tTile = 2;
        const std::size_t nKBlocks = W.i2sResidual.nKBlocks();
        thread_local std::vector<int8_t> x8Scratch;
        thread_local std::vector<float> xScaleScratch;
        x8Scratch.resize(tTile * W.rows);
        xScaleScratch.resize(tTile * nKBlocks);
        linear_forward_bitnet_i2s_accum(
            Y,
            X,
            W.i2sResidual,
            T,
            tTile,
            x8Scratch.data(),
            xScaleScratch.data());
    }
}

void linear_fused_dual_forward_bitnet_tl2_generated(
    float* __restrict Ya,
    float* __restrict Yb,
    const float* __restrict X,
    const BitNetMatTL2Gen& Wa,
    const BitNetMatTL2Gen& Wb,
    std::size_t T)
{
    linear_forward_bitnet_tl2_generated(Ya, X, Wa, T);
    linear_forward_bitnet_tl2_generated(Yb, X, Wb, T);
}
