#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "bitnet_transform.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

inline uint8_t ternary_to_i2_code(int8_t q) noexcept {
    return (q == -1) ? 0u : (q == 0) ? 1u : 2u;
}

inline int8_t i2_code_to_ternary(uint8_t code) noexcept {
    code &= 0x3u;
    if (code == 0u) return -1;
    if (code == 1u) return 0;
    if (code == 2u) return +1;
    return 0;
}

// Cold helpers keep a single out-of-line copy.
void pack_ternary_to_2bit(std::vector<uint8_t>& out_q2, const int8_t* q, std::size_t N);
void unpack_2bit_to_ternary(std::vector<int8_t>& out_q, const uint8_t* q2, std::size_t N);

namespace bitnet_i2s_detail {

inline bool try_get_env_float(const char* name, float& out) {
#if defined(_WIN32)
    char* buffer = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buffer, &len, name) != 0 || !buffer) {
        return false;
    }
    const char* value = buffer;
#else
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return false;
    }
#endif

    char* end = nullptr;
    const float parsed = std::strtof(value, &end);

#if defined(_WIN32)
    std::free(buffer);
#endif

    if (end == value || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

inline bool activation_avx2_quant_enabled() {
    static const bool cached = []() {
#if defined(_WIN32)
        char* buffer = nullptr;
        std::size_t len = 0;
        if (_dupenv_s(&buffer, &len, "QLORA_TEST_BITNET_AVX2_ACT_QUANT") != 0 || !buffer || len == 0 || buffer[0] == '\0') {
            if (buffer) std::free(buffer);
            return false;
        }
        const std::string value(buffer);
        std::free(buffer);
        return !(value == "0" ||
            value == "false" ||
            value == "False" ||
            value == "off" ||
            value == "Off");
#else
        const char* value = std::getenv("QLORA_TEST_BITNET_AVX2_ACT_QUANT");
        if (!value || *value == '\0') {
            return false;
        }
        return !(std::strcmp(value, "0") == 0 ||
            std::strcmp(value, "false") == 0 ||
            std::strcmp(value, "False") == 0 ||
            std::strcmp(value, "off") == 0 ||
            std::strcmp(value, "Off") == 0);
#endif
    }();
    return cached;
}

inline float max_abs_block(
    const float* X, std::size_t H, std::size_t tStart, std::size_t tEnd, std::size_t hStart, std::size_t hEnd)
{
    float m = 0.0f;
#if defined(__AVX2__)
    if (activation_avx2_quant_enabled()) {
        const __m256 signMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        __m256 vmax = _mm256_setzero_ps();
        for (std::size_t t = tStart; t < tEnd; ++t) {
            const float* row = X + t * H;
            std::size_t h = hStart;
            for (; h + 8 <= hEnd; h += 8) {
                const __m256 v = _mm256_and_ps(_mm256_loadu_ps(row + h), signMask);
                vmax = _mm256_max_ps(vmax, v);
            }
            for (; h < hEnd; ++h) {
                const float a = std::fabs(row[h]);
                if (a > m) m = a;
            }
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, vmax);
        for (float v : tmp) {
            if (v > m) m = v;
        }
        return m;
    }
#endif
    for (std::size_t t = tStart; t < tEnd; ++t) {
        const float* row = X + t * H;
        for (std::size_t h = hStart; h < hEnd; ++h) {
            const float a = std::fabs(row[h]);
            if (a > m) m = a;
        }
    }
    return m;
}

struct BlockStats {
    float maxAbs = 0.0f;
    float sumSq = 0.0f;
};

inline BlockStats max_abs_sum_sq_block(
    const float* X, std::size_t H, std::size_t tStart, std::size_t tEnd, std::size_t hStart, std::size_t hEnd)
{
    BlockStats stats;
    for (std::size_t t = tStart; t < tEnd; ++t) {
        const float* row = X + t * H;
        for (std::size_t h = hStart; h < hEnd; ++h) {
            const float v = row[h];
            const float a = std::fabs(v);
            if (a > stats.maxAbs) stats.maxAbs = a;
            stats.sumSq += v * v;
        }
    }
    return stats;
}

inline float activation_clip_rms_factor() {
    static const float cached = []() {
        float parsed = 0.0f;
        if (!try_get_env_float("QLORA_TEST_BITNET_ACT_CLIP_RMS", parsed) || parsed <= 0.0f) {
            return 0.0f;
        }
        return parsed;
    }();
    return cached;
}

#if defined(__AVX2__)
inline __m256 decode_2bytes_to_ps(const uint8_t* ptr, __m256i shift_vec, __m256i mask_vec, __m256i one_vec) {
    uint16_t val16;
    std::memcpy(&val16, ptr, sizeof(uint16_t));
    __m256i v = _mm256_set1_epi32(val16);
    v = _mm256_srlv_epi32(v, shift_vec);
    v = _mm256_and_si256(v, mask_vec);
    v = _mm256_sub_epi32(v, one_vec);
    return _mm256_cvtepi32_ps(v);
}

inline void quantize_row_i8_avx2_no_clip(
    int8_t* __restrict out,
    const float* __restrict row,
    std::size_t hStart,
    std::size_t hEnd,
    float inv)
{
    const __m256 inv8 = _mm256_set1_ps(inv);
    std::size_t h = hStart;
    for (; h + 8 <= hEnd; h += 8) {
        const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(row + h), inv8);
        const __m256i i32 = _mm256_cvtps_epi32(scaled);
        const __m128i lo = _mm256_castsi256_si128(i32);
        const __m128i hi = _mm256_extracti128_si256(i32, 1);
        const __m128i i16 = _mm_packs_epi32(lo, hi);
        const __m128i i8 = _mm_packs_epi16(i16, _mm_setzero_si128());
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + h), i8);
    }
    for (; h < hEnd; ++h) {
        const int qi = static_cast<int>(std::lrint(row[h] * inv));
        out[h] = static_cast<int8_t>(std::max(-127, std::min(127, qi)));
    }
}
#endif

} // namespace bitnet_i2s_detail

inline void quantize_x_tile_i8_tiled(
    int8_t* __restrict Xi8, float* __restrict xScale, const float* __restrict X,
    std::size_t T, std::size_t H, std::size_t t0, std::size_t tCount,
    std::size_t tBlock, std::size_t hBlock, bool hadamard32 = false)
{
    const std::size_t nTBlocks = (tCount + tBlock - 1) / tBlock;
    const std::size_t nHBlocks = (H + hBlock - 1) / hBlock;
    const float clipRmsFactor = bitnet_i2s_detail::activation_clip_rms_factor();
    if (hadamard32 && ((H % 32) != 0 || (hBlock % 32) != 0)) {
        throw std::runtime_error("quantize_x_tile_i8_tiled: hadamard32 requires H and hBlock to be multiples of 32");
    }

    for (std::size_t bt = 0; bt < nTBlocks; ++bt) {
        const std::size_t tStart = t0 + bt * tBlock;
        const std::size_t tEnd = std::min(t0 + tCount, tStart + tBlock);
        for (std::size_t bh = 0; bh < nHBlocks; ++bh) {
            const std::size_t hStart = bh * hBlock;
            const std::size_t hEnd = std::min(H, hStart + hBlock);

            float clip = 0.0f;
            if (hadamard32) {
                bitnet_i2s_detail::BlockStats stats;
                for (std::size_t t = tStart; t < tEnd; ++t) {
                    const float* row = X + t * H;
                    for (std::size_t h = hStart; h < hEnd; h += 32) {
                        float tmp[32];
                        for (std::size_t u = 0; u < 32; ++u) tmp[u] = row[h + u];
                        hadamard32_inplace(tmp);
                        for (std::size_t u = 0; u < 32; ++u) {
                            const float v = tmp[u];
                            const float a = std::fabs(v);
                            if (a > stats.maxAbs) stats.maxAbs = a;
                            stats.sumSq += v * v;
                        }
                    }
                }
                clip = stats.maxAbs;
                if (clipRmsFactor > 0.0f && clip > 0.0f) {
                    const float count = static_cast<float>((tEnd - tStart) * (hEnd - hStart));
                    const float rms = std::sqrt(stats.sumSq / std::max(1.0f, count));
                    if (rms > 0.0f) clip = std::min(clip, clipRmsFactor * rms);
                }
            }
            else if (clipRmsFactor > 0.0f) {
                const bitnet_i2s_detail::BlockStats stats =
                    bitnet_i2s_detail::max_abs_sum_sq_block(X, H, tStart, tEnd, hStart, hEnd);
                clip = stats.maxAbs;
                if (clip > 0.0f) {
                    const float count = static_cast<float>((tEnd - tStart) * (hEnd - hStart));
                    const float rms = std::sqrt(stats.sumSq / std::max(1.0f, count));
                    if (rms > 0.0f) clip = std::min(clip, clipRmsFactor * rms);
                }
            }
            else {
                clip = bitnet_i2s_detail::max_abs_block(X, H, tStart, tEnd, hStart, hEnd);
            }

            const float scale = (clip > 0.0f) ? (clip / 127.0f) : 1.0f;
            const float inv = 1.0f / scale;
            xScale[bt * nHBlocks + bh] = scale;

            for (std::size_t t = tStart; t < tEnd; ++t) {
                const float* row = X + t * H;
                int8_t* outRow = Xi8 + (t - t0) * H;
                if (clip == 0.0f) {
                    std::memset(outRow + hStart, 0, hEnd - hStart);
                }
                else if (hadamard32) {
                    for (std::size_t h = hStart; h < hEnd; h += 32) {
                        float tmp[32];
                        for (std::size_t u = 0; u < 32; ++u) tmp[u] = row[h + u];
                        hadamard32_inplace(tmp);
                        for (std::size_t u = 0; u < 32; ++u) {
                            const float v = (clipRmsFactor > 0.0f)
                                ? std::max(-clip, std::min(clip, tmp[u]))
                                : tmp[u];
                            const int qi = static_cast<int>(std::lrint(v * inv));
                            outRow[h + u] = static_cast<int8_t>(std::max(-127, std::min(127, qi)));
                        }
                    }
                }
#if defined(__AVX2__)
                else if (clipRmsFactor <= 0.0f && bitnet_i2s_detail::activation_avx2_quant_enabled()) {
                    bitnet_i2s_detail::quantize_row_i8_avx2_no_clip(outRow, row, hStart, hEnd, inv);
                }
#endif
                else {
                    for (std::size_t h = hStart; h < hEnd; ++h) {
                        const float v = (clipRmsFactor > 0.0f)
                            ? std::max(-clip, std::min(clip, row[h]))
                            : row[h];
                        const int qi = static_cast<int>(std::lrint(v * inv));
                        outRow[h] = static_cast<int8_t>(std::max(-127, std::min(127, qi)));
                    }
                }
            }
        }
    }
}

inline void quantize_x_tile_i8_tiled_dual(
    int8_t* __restrict Xi8A, float* __restrict xScaleA, const float* __restrict XA,
    int8_t* __restrict Xi8B, float* __restrict xScaleB, const float* __restrict XB,
    std::size_t T, std::size_t H, std::size_t t0, std::size_t tCount,
    std::size_t tBlock, std::size_t hBlock, bool hadamard32 = false)
{
    const std::size_t nTBlocks = (tCount + tBlock - 1) / tBlock;
    const std::size_t nHBlocks = (H + hBlock - 1) / hBlock;
    const float clipRmsFactor = bitnet_i2s_detail::activation_clip_rms_factor();
    if (hadamard32 && ((H % 32) != 0 || (hBlock % 32) != 0)) {
        throw std::runtime_error("quantize_x_tile_i8_tiled_dual: hadamard32 requires H and hBlock to be multiples of 32");
    }

    for (std::size_t bt = 0; bt < nTBlocks; ++bt) {
        const std::size_t tStart = t0 + bt * tBlock;
        const std::size_t tEnd = std::min(t0 + tCount, tStart + tBlock);
        for (std::size_t bh = 0; bh < nHBlocks; ++bh) {
            const std::size_t hStart = bh * hBlock;
            const std::size_t hEnd = std::min(H, hStart + hBlock);

            float clipA = 0.0f;
            float clipB = 0.0f;
            if (hadamard32) {
                bitnet_i2s_detail::BlockStats statsA;
                bitnet_i2s_detail::BlockStats statsB;
                for (std::size_t t = tStart; t < tEnd; ++t) {
                    const float* rowA = XA + t * H;
                    const float* rowB = XB + t * H;
                    for (std::size_t h = hStart; h < hEnd; h += 32) {
                        float tmpA[32];
                        float tmpB[32];
                        for (std::size_t u = 0; u < 32; ++u) {
                            tmpA[u] = rowA[h + u];
                            tmpB[u] = rowB[h + u];
                        }
                        hadamard32_inplace(tmpA);
                        hadamard32_inplace(tmpB);
                        for (std::size_t u = 0; u < 32; ++u) {
                            const float vA = tmpA[u];
                            const float vB = tmpB[u];
                            const float aA = std::fabs(vA);
                            const float aB = std::fabs(vB);
                            if (aA > statsA.maxAbs) statsA.maxAbs = aA;
                            if (aB > statsB.maxAbs) statsB.maxAbs = aB;
                            statsA.sumSq += vA * vA;
                            statsB.sumSq += vB * vB;
                        }
                    }
                }
                clipA = statsA.maxAbs;
                clipB = statsB.maxAbs;
                if (clipRmsFactor > 0.0f) {
                    const float count = static_cast<float>((tEnd - tStart) * (hEnd - hStart));
                    if (clipA > 0.0f) {
                        const float rmsA = std::sqrt(statsA.sumSq / std::max(1.0f, count));
                        if (rmsA > 0.0f) clipA = std::min(clipA, clipRmsFactor * rmsA);
                    }
                    if (clipB > 0.0f) {
                        const float rmsB = std::sqrt(statsB.sumSq / std::max(1.0f, count));
                        if (rmsB > 0.0f) clipB = std::min(clipB, clipRmsFactor * rmsB);
                    }
                }
            }
            else if (clipRmsFactor > 0.0f) {
                const bitnet_i2s_detail::BlockStats statsA =
                    bitnet_i2s_detail::max_abs_sum_sq_block(XA, H, tStart, tEnd, hStart, hEnd);
                const bitnet_i2s_detail::BlockStats statsB =
                    bitnet_i2s_detail::max_abs_sum_sq_block(XB, H, tStart, tEnd, hStart, hEnd);
                clipA = statsA.maxAbs;
                clipB = statsB.maxAbs;
                const float count = static_cast<float>((tEnd - tStart) * (hEnd - hStart));
                if (clipA > 0.0f) {
                    const float rmsA = std::sqrt(statsA.sumSq / std::max(1.0f, count));
                    if (rmsA > 0.0f) clipA = std::min(clipA, clipRmsFactor * rmsA);
                }
                if (clipB > 0.0f) {
                    const float rmsB = std::sqrt(statsB.sumSq / std::max(1.0f, count));
                    if (rmsB > 0.0f) clipB = std::min(clipB, clipRmsFactor * rmsB);
                }
            }
            else {
                clipA = bitnet_i2s_detail::max_abs_block(XA, H, tStart, tEnd, hStart, hEnd);
                clipB = bitnet_i2s_detail::max_abs_block(XB, H, tStart, tEnd, hStart, hEnd);
            }

            const float scaleA = (clipA > 0.0f) ? (clipA / 127.0f) : 1.0f;
            const float scaleB = (clipB > 0.0f) ? (clipB / 127.0f) : 1.0f;
            const float invA = 1.0f / scaleA;
            const float invB = 1.0f / scaleB;
            xScaleA[bt * nHBlocks + bh] = scaleA;
            xScaleB[bt * nHBlocks + bh] = scaleB;

            for (std::size_t t = tStart; t < tEnd; ++t) {
                const float* rowA = XA + t * H;
                const float* rowB = XB + t * H;
                int8_t* outRowA = Xi8A + (t - t0) * H;
                int8_t* outRowB = Xi8B + (t - t0) * H;
                if (clipA == 0.0f && clipB == 0.0f) {
                    std::memset(outRowA + hStart, 0, hEnd - hStart);
                    std::memset(outRowB + hStart, 0, hEnd - hStart);
                }
                else if (hadamard32) {
                    for (std::size_t h = hStart; h < hEnd; h += 32) {
                        float tmpA[32];
                        float tmpB[32];
                        for (std::size_t u = 0; u < 32; ++u) {
                            tmpA[u] = rowA[h + u];
                            tmpB[u] = rowB[h + u];
                        }
                        hadamard32_inplace(tmpA);
                        hadamard32_inplace(tmpB);
                        for (std::size_t u = 0; u < 32; ++u) {
                            if (clipA == 0.0f) {
                                outRowA[h + u] = 0;
                            }
                            else {
                                const float vA = (clipRmsFactor > 0.0f)
                                    ? std::max(-clipA, std::min(clipA, tmpA[u]))
                                    : tmpA[u];
                                const int qiA = static_cast<int>(std::lrint(vA * invA));
                                outRowA[h + u] = static_cast<int8_t>(std::max(-127, std::min(127, qiA)));
                            }
                            if (clipB == 0.0f) {
                                outRowB[h + u] = 0;
                            }
                            else {
                                const float vB = (clipRmsFactor > 0.0f)
                                    ? std::max(-clipB, std::min(clipB, tmpB[u]))
                                    : tmpB[u];
                                const int qiB = static_cast<int>(std::lrint(vB * invB));
                                outRowB[h + u] = static_cast<int8_t>(std::max(-127, std::min(127, qiB)));
                            }
                        }
                    }
                }
#if defined(__AVX2__)
                else if (clipRmsFactor <= 0.0f && clipA != 0.0f && clipB != 0.0f && bitnet_i2s_detail::activation_avx2_quant_enabled()) {
                    bitnet_i2s_detail::quantize_row_i8_avx2_no_clip(outRowA, rowA, hStart, hEnd, invA);
                    bitnet_i2s_detail::quantize_row_i8_avx2_no_clip(outRowB, rowB, hStart, hEnd, invB);
                }
#endif
                else {
                    for (std::size_t h = hStart; h < hEnd; ++h) {
                        if (clipA == 0.0f) {
                            outRowA[h] = 0;
                        }
                        else {
                            const float vA = (clipRmsFactor > 0.0f)
                                ? std::max(-clipA, std::min(clipA, rowA[h]))
                                : rowA[h];
                            const int qiA = static_cast<int>(std::lrint(vA * invA));
                            outRowA[h] = static_cast<int8_t>(std::max(-127, std::min(127, qiA)));
                        }
                        if (clipB == 0.0f) {
                            outRowB[h] = 0;
                        }
                        else {
                            const float vB = (clipRmsFactor > 0.0f)
                                ? std::max(-clipB, std::min(clipB, rowB[h]))
                                : rowB[h];
                            const int qiB = static_cast<int>(std::lrint(vB * invB));
                            outRowB[h] = static_cast<int8_t>(std::max(-127, std::min(127, qiB)));
                        }
                    }
                }
            }
        }
    }
}

inline void quantize_x_tile_i8_tiled_omp_for(
    int8_t* __restrict Xi8, float* __restrict xScale, const float* __restrict X,
    std::size_t T, std::size_t H, std::size_t t0, std::size_t tCount,
    std::size_t tBlock, std::size_t hBlock, bool hadamard32 = false)
{
#if defined(_OPENMP)
    const std::size_t nTBlocks = (tCount + tBlock - 1) / tBlock;
    const std::size_t nHBlocks = (H + hBlock - 1) / hBlock;
    const float clipRmsFactor = bitnet_i2s_detail::activation_clip_rms_factor();
    if (hadamard32 && ((H % 32) != 0 || (hBlock % 32) != 0)) {
        throw std::runtime_error("quantize_x_tile_i8_tiled_omp_for: hadamard32 requires H and hBlock to be multiples of 32");
    }

#pragma omp for collapse(2) schedule(static)
    for (int bt_i = 0; bt_i < static_cast<int>(nTBlocks); ++bt_i) {
        for (int bh_i = 0; bh_i < static_cast<int>(nHBlocks); ++bh_i) {
            const std::size_t bt = static_cast<std::size_t>(bt_i);
            const std::size_t bh = static_cast<std::size_t>(bh_i);
            const std::size_t tStart = t0 + bt * tBlock;
            const std::size_t tEnd = std::min(t0 + tCount, tStart + tBlock);
            const std::size_t hStart = bh * hBlock;
            const std::size_t hEnd = std::min(H, hStart + hBlock);

            float clip = 0.0f;
            if (hadamard32) {
                bitnet_i2s_detail::BlockStats stats;
                for (std::size_t t = tStart; t < tEnd; ++t) {
                    const float* row = X + t * H;
                    for (std::size_t h = hStart; h < hEnd; h += 32) {
                        float tmp[32];
                        for (std::size_t u = 0; u < 32; ++u) tmp[u] = row[h + u];
                        hadamard32_inplace(tmp);
                        for (std::size_t u = 0; u < 32; ++u) {
                            const float v = tmp[u];
                            const float a = std::fabs(v);
                            if (a > stats.maxAbs) stats.maxAbs = a;
                            stats.sumSq += v * v;
                        }
                    }
                }
                clip = stats.maxAbs;
                if (clipRmsFactor > 0.0f && clip > 0.0f) {
                    const float count = static_cast<float>((tEnd - tStart) * (hEnd - hStart));
                    const float rms = std::sqrt(stats.sumSq / std::max(1.0f, count));
                    if (rms > 0.0f) clip = std::min(clip, clipRmsFactor * rms);
                }
            }
            else if (clipRmsFactor > 0.0f) {
                const bitnet_i2s_detail::BlockStats stats =
                    bitnet_i2s_detail::max_abs_sum_sq_block(X, H, tStart, tEnd, hStart, hEnd);
                clip = stats.maxAbs;
                if (clip > 0.0f) {
                    const float count = static_cast<float>((tEnd - tStart) * (hEnd - hStart));
                    const float rms = std::sqrt(stats.sumSq / std::max(1.0f, count));
                    if (rms > 0.0f) clip = std::min(clip, clipRmsFactor * rms);
                }
            }
            else {
                clip = bitnet_i2s_detail::max_abs_block(X, H, tStart, tEnd, hStart, hEnd);
            }

            const float scale = (clip > 0.0f) ? (clip / 127.0f) : 1.0f;
            const float inv = 1.0f / scale;
            xScale[bt * nHBlocks + bh] = scale;

            for (std::size_t t = tStart; t < tEnd; ++t) {
                const float* row = X + t * H;
                int8_t* outRow = Xi8 + (t - t0) * H;
                if (clip == 0.0f) {
                    std::memset(outRow + hStart, 0, hEnd - hStart);
                }
                else if (hadamard32) {
                    for (std::size_t h = hStart; h < hEnd; h += 32) {
                        float tmp[32];
                        for (std::size_t u = 0; u < 32; ++u) tmp[u] = row[h + u];
                        hadamard32_inplace(tmp);
                        for (std::size_t u = 0; u < 32; ++u) {
                            const float v = (clipRmsFactor > 0.0f)
                                ? std::max(-clip, std::min(clip, tmp[u]))
                                : tmp[u];
                            const int qi = static_cast<int>(std::lrint(v * inv));
                            outRow[h + u] = static_cast<int8_t>(std::max(-127, std::min(127, qi)));
                        }
                    }
                }
#if defined(__AVX2__)
                else if (clipRmsFactor <= 0.0f && bitnet_i2s_detail::activation_avx2_quant_enabled()) {
                    bitnet_i2s_detail::quantize_row_i8_avx2_no_clip(outRow, row, hStart, hEnd, inv);
                }
#endif
                else {
                    for (std::size_t h = hStart; h < hEnd; ++h) {
                        const float v = (clipRmsFactor > 0.0f)
                            ? std::max(-clip, std::min(clip, row[h]))
                            : row[h];
                        const int qi = static_cast<int>(std::lrint(v * inv));
                        outRow[h] = static_cast<int8_t>(std::max(-127, std::min(127, qi)));
                    }
                }
            }
        }
    }
#else
    quantize_x_tile_i8_tiled(Xi8, xScale, X, T, H, t0, tCount, tBlock, hBlock, hadamard32);
#endif
}

inline void quantize_x_tile_i8_tiled_dual_omp_for(
    int8_t* __restrict Xi8A, float* __restrict xScaleA, const float* __restrict XA,
    int8_t* __restrict Xi8B, float* __restrict xScaleB, const float* __restrict XB,
    std::size_t T, std::size_t H, std::size_t t0, std::size_t tCount,
    std::size_t tBlock, std::size_t hBlock, bool hadamard32 = false)
{
#if defined(_OPENMP)
    const std::size_t nTBlocks = (tCount + tBlock - 1) / tBlock;
    const std::size_t nHBlocks = (H + hBlock - 1) / hBlock;
    const float clipRmsFactor = bitnet_i2s_detail::activation_clip_rms_factor();
    if (hadamard32 && ((H % 32) != 0 || (hBlock % 32) != 0)) {
        throw std::runtime_error("quantize_x_tile_i8_tiled_dual_omp_for: hadamard32 requires H and hBlock to be multiples of 32");
    }

#pragma omp for collapse(2) schedule(static)
    for (int bt_i = 0; bt_i < static_cast<int>(nTBlocks); ++bt_i) {
        for (int bh_i = 0; bh_i < static_cast<int>(nHBlocks); ++bh_i) {
            const std::size_t bt = static_cast<std::size_t>(bt_i);
            const std::size_t bh = static_cast<std::size_t>(bh_i);
            const std::size_t tStart = t0 + bt * tBlock;
            const std::size_t tEnd = std::min(t0 + tCount, tStart + tBlock);
            const std::size_t hStart = bh * hBlock;
            const std::size_t hEnd = std::min(H, hStart + hBlock);

            float clipA = 0.0f;
            float clipB = 0.0f;
            if (hadamard32) {
                bitnet_i2s_detail::BlockStats statsA;
                bitnet_i2s_detail::BlockStats statsB;
                for (std::size_t t = tStart; t < tEnd; ++t) {
                    const float* rowA = XA + t * H;
                    const float* rowB = XB + t * H;
                    for (std::size_t h = hStart; h < hEnd; h += 32) {
                        float tmpA[32];
                        float tmpB[32];
                        for (std::size_t u = 0; u < 32; ++u) {
                            tmpA[u] = rowA[h + u];
                            tmpB[u] = rowB[h + u];
                        }
                        hadamard32_inplace(tmpA);
                        hadamard32_inplace(tmpB);
                        for (std::size_t u = 0; u < 32; ++u) {
                            const float vA = tmpA[u];
                            const float vB = tmpB[u];
                            const float aA = std::fabs(vA);
                            const float aB = std::fabs(vB);
                            if (aA > statsA.maxAbs) statsA.maxAbs = aA;
                            if (aB > statsB.maxAbs) statsB.maxAbs = aB;
                            statsA.sumSq += vA * vA;
                            statsB.sumSq += vB * vB;
                        }
                    }
                }
                clipA = statsA.maxAbs;
                clipB = statsB.maxAbs;
                if (clipRmsFactor > 0.0f) {
                    const float count = static_cast<float>((tEnd - tStart) * (hEnd - hStart));
                    if (clipA > 0.0f) {
                        const float rmsA = std::sqrt(statsA.sumSq / std::max(1.0f, count));
                        if (rmsA > 0.0f) clipA = std::min(clipA, clipRmsFactor * rmsA);
                    }
                    if (clipB > 0.0f) {
                        const float rmsB = std::sqrt(statsB.sumSq / std::max(1.0f, count));
                        if (rmsB > 0.0f) clipB = std::min(clipB, clipRmsFactor * rmsB);
                    }
                }
            }
            else if (clipRmsFactor > 0.0f) {
                const bitnet_i2s_detail::BlockStats statsA =
                    bitnet_i2s_detail::max_abs_sum_sq_block(XA, H, tStart, tEnd, hStart, hEnd);
                const bitnet_i2s_detail::BlockStats statsB =
                    bitnet_i2s_detail::max_abs_sum_sq_block(XB, H, tStart, tEnd, hStart, hEnd);
                clipA = statsA.maxAbs;
                clipB = statsB.maxAbs;
                const float count = static_cast<float>((tEnd - tStart) * (hEnd - hStart));
                if (clipA > 0.0f) {
                    const float rmsA = std::sqrt(statsA.sumSq / std::max(1.0f, count));
                    if (rmsA > 0.0f) clipA = std::min(clipA, clipRmsFactor * rmsA);
                }
                if (clipB > 0.0f) {
                    const float rmsB = std::sqrt(statsB.sumSq / std::max(1.0f, count));
                    if (rmsB > 0.0f) clipB = std::min(clipB, clipRmsFactor * rmsB);
                }
            }
            else {
                clipA = bitnet_i2s_detail::max_abs_block(XA, H, tStart, tEnd, hStart, hEnd);
                clipB = bitnet_i2s_detail::max_abs_block(XB, H, tStart, tEnd, hStart, hEnd);
            }

            const float scaleA = (clipA > 0.0f) ? (clipA / 127.0f) : 1.0f;
            const float scaleB = (clipB > 0.0f) ? (clipB / 127.0f) : 1.0f;
            const float invA = 1.0f / scaleA;
            const float invB = 1.0f / scaleB;
            xScaleA[bt * nHBlocks + bh] = scaleA;
            xScaleB[bt * nHBlocks + bh] = scaleB;

            for (std::size_t t = tStart; t < tEnd; ++t) {
                const float* rowA = XA + t * H;
                const float* rowB = XB + t * H;
                int8_t* outRowA = Xi8A + (t - t0) * H;
                int8_t* outRowB = Xi8B + (t - t0) * H;
                if (clipA == 0.0f && clipB == 0.0f) {
                    std::memset(outRowA + hStart, 0, hEnd - hStart);
                    std::memset(outRowB + hStart, 0, hEnd - hStart);
                }
                else if (hadamard32) {
                    for (std::size_t h = hStart; h < hEnd; h += 32) {
                        float tmpA[32];
                        float tmpB[32];
                        for (std::size_t u = 0; u < 32; ++u) {
                            tmpA[u] = rowA[h + u];
                            tmpB[u] = rowB[h + u];
                        }
                        hadamard32_inplace(tmpA);
                        hadamard32_inplace(tmpB);
                        for (std::size_t u = 0; u < 32; ++u) {
                            if (clipA == 0.0f) {
                                outRowA[h + u] = 0;
                            }
                            else {
                                const float vA = (clipRmsFactor > 0.0f)
                                    ? std::max(-clipA, std::min(clipA, tmpA[u]))
                                    : tmpA[u];
                                const int qiA = static_cast<int>(std::lrint(vA * invA));
                                outRowA[h + u] = static_cast<int8_t>(std::max(-127, std::min(127, qiA)));
                            }
                            if (clipB == 0.0f) {
                                outRowB[h + u] = 0;
                            }
                            else {
                                const float vB = (clipRmsFactor > 0.0f)
                                    ? std::max(-clipB, std::min(clipB, tmpB[u]))
                                    : tmpB[u];
                                const int qiB = static_cast<int>(std::lrint(vB * invB));
                                outRowB[h + u] = static_cast<int8_t>(std::max(-127, std::min(127, qiB)));
                            }
                        }
                    }
                }
#if defined(__AVX2__)
                else if (clipRmsFactor <= 0.0f && clipA != 0.0f && clipB != 0.0f && bitnet_i2s_detail::activation_avx2_quant_enabled()) {
                    bitnet_i2s_detail::quantize_row_i8_avx2_no_clip(outRowA, rowA, hStart, hEnd, invA);
                    bitnet_i2s_detail::quantize_row_i8_avx2_no_clip(outRowB, rowB, hStart, hEnd, invB);
                }
#endif
                else {
                    for (std::size_t h = hStart; h < hEnd; ++h) {
                        if (clipA == 0.0f) {
                            outRowA[h] = 0;
                        }
                        else {
                            const float vA = (clipRmsFactor > 0.0f)
                                ? std::max(-clipA, std::min(clipA, rowA[h]))
                                : rowA[h];
                            const int qiA = static_cast<int>(std::lrint(vA * invA));
                            outRowA[h] = static_cast<int8_t>(std::max(-127, std::min(127, qiA)));
                        }
                        if (clipB == 0.0f) {
                            outRowB[h] = 0;
                        }
                        else {
                            const float vB = (clipRmsFactor > 0.0f)
                                ? std::max(-clipB, std::min(clipB, rowB[h]))
                                : rowB[h];
                            const int qiB = static_cast<int>(std::lrint(vB * invB));
                            outRowB[h] = static_cast<int8_t>(std::max(-127, std::min(127, qiB)));
                        }
                    }
                }
            }
        }
    }
#else
    quantize_x_tile_i8_tiled_dual(Xi8A, xScaleA, XA, Xi8B, xScaleB, XB, T, H, t0, tCount, tBlock, hBlock, hadamard32);
#endif
}

inline void dot_i2s_i8_accum_i32_block32(
    int32_t* __restrict acc32, const uint8_t* __restrict Wq2, std::size_t stride_cols,
    std::size_t i0, std::size_t hCount, const int8_t* __restrict x8, std::size_t j0)
{
#if defined(__AVX2__)
    __m256 a0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 0)));
    __m256 a1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 8)));
    __m256 a2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 16)));
    __m256 a3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 24)));

    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);

    for (std::size_t di = 0; di < hCount; ++di) {
        if (x8[di] == 0) continue;
        const __m256 x_ps = _mm256_set1_ps(static_cast<float>(x8[di]));
        const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wbytes + stride_cols), _MM_HINT_T0);

        a0 = _mm256_fmadd_ps(x_ps, bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec), a0);
        a1 = _mm256_fmadd_ps(x_ps, bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec), a1);
        a2 = _mm256_fmadd_ps(x_ps, bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 4, shift_vec, mask_vec, one_vec), a2);
        a3 = _mm256_fmadd_ps(x_ps, bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 6, shift_vec, mask_vec, one_vec), a3);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 0), _mm256_cvtps_epi32(a0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 8), _mm256_cvtps_epi32(a1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 16), _mm256_cvtps_epi32(a2));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 24), _mm256_cvtps_epi32(a3));
#else
    for (std::size_t di = 0; di < hCount; ++di) {
        const int xi = static_cast<int>(x8[di]);
        if (xi == 0) continue;
        const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
        for (std::size_t jb = 0; jb < 8; ++jb) {
            const uint8_t packed = wbytes[jb];
            acc32[jb * 4 + 0] += xi * static_cast<int>(i2_code_to_ternary((packed >> 0) & 3u));
            acc32[jb * 4 + 1] += xi * static_cast<int>(i2_code_to_ternary((packed >> 2) & 3u));
            acc32[jb * 4 + 2] += xi * static_cast<int>(i2_code_to_ternary((packed >> 4) & 3u));
            acc32[jb * 4 + 3] += xi * static_cast<int>(i2_code_to_ternary((packed >> 6) & 3u));
        }
    }
#endif
}

inline void dot_i2s_i8_accum_i32_block32_x2(
    int32_t* __restrict accA, int32_t* __restrict accB, const uint8_t* __restrict Wq2,
    std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B, std::size_t j0)
{
#if defined(__AVX2__)
    __m256 aA0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 0)));
    __m256 aA1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 8)));
    __m256 aA2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 16)));
    __m256 aA3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 24)));

    __m256 aB0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 0)));
    __m256 aB1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 8)));
    __m256 aB2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 16)));
    __m256 aB3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 24)));

    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(x8A[di] | x8B[di])) continue;

        const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wbytes + stride_cols), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec);
        const __m256 w2 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 4, shift_vec, mask_vec, one_vec);
        const __m256 w3 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 6, shift_vec, mask_vec, one_vec);

        if (x8A[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di]));
            aA0 = _mm256_fmadd_ps(vx, w0, aA0); aA1 = _mm256_fmadd_ps(vx, w1, aA1);
            aA2 = _mm256_fmadd_ps(vx, w2, aA2); aA3 = _mm256_fmadd_ps(vx, w3, aA3);
        }
        if (x8B[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di]));
            aB0 = _mm256_fmadd_ps(vx, w0, aB0); aB1 = _mm256_fmadd_ps(vx, w1, aB1);
            aB2 = _mm256_fmadd_ps(vx, w2, aB2); aB3 = _mm256_fmadd_ps(vx, w3, aB3);
        }
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 0), _mm256_cvtps_epi32(aA0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 8), _mm256_cvtps_epi32(aA1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 16), _mm256_cvtps_epi32(aA2));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 24), _mm256_cvtps_epi32(aA3));

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 0), _mm256_cvtps_epi32(aB0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 8), _mm256_cvtps_epi32(aB1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 16), _mm256_cvtps_epi32(aB2));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 24), _mm256_cvtps_epi32(aB3));
#else
    for (std::size_t di = 0; di < hCount; ++di) {
        const int xa = static_cast<int>(x8A[di]);
        const int xb = static_cast<int>(x8B[di]);
        if (xa == 0 && xb == 0) continue;
        const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
        for (std::size_t jb = 0; jb < 8; ++jb) {
            const uint8_t packed = wbytes[jb];
            const int w0 = static_cast<int>(i2_code_to_ternary((packed >> 0) & 3u));
            const int w1 = static_cast<int>(i2_code_to_ternary((packed >> 2) & 3u));
            const int w2 = static_cast<int>(i2_code_to_ternary((packed >> 4) & 3u));
            const int w3 = static_cast<int>(i2_code_to_ternary((packed >> 6) & 3u));
            if (xa) { accA[jb * 4 + 0] += xa * w0; accA[jb * 4 + 1] += xa * w1; accA[jb * 4 + 2] += xa * w2; accA[jb * 4 + 3] += xa * w3; }
            if (xb) { accB[jb * 4 + 0] += xb * w0; accB[jb * 4 + 1] += xb * w1; accB[jb * 4 + 2] += xb * w2; accB[jb * 4 + 3] += xb * w3; }
        }
    }
#endif
}

inline void dot_i2s_i8_accum_i32_block32_x2_dense(
    int32_t* __restrict accA, int32_t* __restrict accB, const uint8_t* __restrict Wq2,
    std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B, std::size_t j0)
{
#if defined(__AVX2__)
    __m256 aA0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 0)));
    __m256 aA1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 8)));
    __m256 aA2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 16)));
    __m256 aA3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 24)));

    __m256 aB0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 0)));
    __m256 aB1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 8)));
    __m256 aB2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 16)));
    __m256 aB3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 24)));

    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(x8A[di] | x8B[di])) continue;

        const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wbytes + stride_cols), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec);
        const __m256 w2 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 4, shift_vec, mask_vec, one_vec);
        const __m256 w3 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 6, shift_vec, mask_vec, one_vec);

        const __m256 vxA = _mm256_set1_ps(static_cast<float>(x8A[di]));
        const __m256 vxB = _mm256_set1_ps(static_cast<float>(x8B[di]));

        aA0 = _mm256_fmadd_ps(vxA, w0, aA0); aA1 = _mm256_fmadd_ps(vxA, w1, aA1);
        aA2 = _mm256_fmadd_ps(vxA, w2, aA2); aA3 = _mm256_fmadd_ps(vxA, w3, aA3);
        aB0 = _mm256_fmadd_ps(vxB, w0, aB0); aB1 = _mm256_fmadd_ps(vxB, w1, aB1);
        aB2 = _mm256_fmadd_ps(vxB, w2, aB2); aB3 = _mm256_fmadd_ps(vxB, w3, aB3);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 0), _mm256_cvtps_epi32(aA0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 8), _mm256_cvtps_epi32(aA1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 16), _mm256_cvtps_epi32(aA2));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 24), _mm256_cvtps_epi32(aA3));

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 0), _mm256_cvtps_epi32(aB0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 8), _mm256_cvtps_epi32(aB1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 16), _mm256_cvtps_epi32(aB2));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 24), _mm256_cvtps_epi32(aB3));
#else
    dot_i2s_i8_accum_i32_block32_x2(accA, accB, Wq2, stride_cols, i0, hCount, x8A, x8B, j0);
#endif
}

inline void dot_i2s_i8_accum_i32_block32_x4(
    int32_t* __restrict accA, int32_t* __restrict accB,
    int32_t* __restrict accC, int32_t* __restrict accD,
    const uint8_t* __restrict Wq2, std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B,
    const int8_t* __restrict x8C, const int8_t* __restrict x8D,
    std::size_t j0)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    {
        __m256 aA0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 0)));
        __m256 aA1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 8)));
        __m256 aB0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 0)));
        __m256 aB1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 8)));
        __m256 aC0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accC + 0)));
        __m256 aC1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accC + 8)));
        __m256 aD0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accD + 0)));
        __m256 aD1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accD + 8)));

        for (std::size_t di = 0; di < hCount; ++di) {
            if (!(x8A[di] | x8B[di] | x8C[di] | x8D[di])) continue;
            const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
            _mm_prefetch(reinterpret_cast<const char*>(wbytes + prefetch_stride), _MM_HINT_T0);

            const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec);
            const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec);

            if (x8A[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di])); aA0 = _mm256_fmadd_ps(vx, w0, aA0); aA1 = _mm256_fmadd_ps(vx, w1, aA1); }
            if (x8B[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di])); aB0 = _mm256_fmadd_ps(vx, w0, aB0); aB1 = _mm256_fmadd_ps(vx, w1, aB1); }
            if (x8C[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(x8C[di])); aC0 = _mm256_fmadd_ps(vx, w0, aC0); aC1 = _mm256_fmadd_ps(vx, w1, aC1); }
            if (x8D[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(x8D[di])); aD0 = _mm256_fmadd_ps(vx, w0, aD0); aD1 = _mm256_fmadd_ps(vx, w1, aD1); }
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 0), _mm256_cvtps_epi32(aA0)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 8), _mm256_cvtps_epi32(aA1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 0), _mm256_cvtps_epi32(aB0)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 8), _mm256_cvtps_epi32(aB1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accC + 0), _mm256_cvtps_epi32(aC0)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accC + 8), _mm256_cvtps_epi32(aC1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accD + 0), _mm256_cvtps_epi32(aD0)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accD + 8), _mm256_cvtps_epi32(aD1));
    }

    {
        __m256 aA2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 16)));
        __m256 aA3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 24)));
        __m256 aB2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 16)));
        __m256 aB3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 24)));
        __m256 aC2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accC + 16)));
        __m256 aC3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accC + 24)));
        __m256 aD2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accD + 16)));
        __m256 aD3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accD + 24)));

        for (std::size_t di = 0; di < hCount; ++di) {
            if (!(x8A[di] | x8B[di] | x8C[di] | x8D[di])) continue;
            const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
            _mm_prefetch(reinterpret_cast<const char*>(wbytes + prefetch_stride), _MM_HINT_T0);

            const __m256 w2 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 4, shift_vec, mask_vec, one_vec);
            const __m256 w3 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 6, shift_vec, mask_vec, one_vec);

            if (x8A[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di])); aA2 = _mm256_fmadd_ps(vx, w2, aA2); aA3 = _mm256_fmadd_ps(vx, w3, aA3); }
            if (x8B[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di])); aB2 = _mm256_fmadd_ps(vx, w2, aB2); aB3 = _mm256_fmadd_ps(vx, w3, aB3); }
            if (x8C[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(x8C[di])); aC2 = _mm256_fmadd_ps(vx, w2, aC2); aC3 = _mm256_fmadd_ps(vx, w3, aC3); }
            if (x8D[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(x8D[di])); aD2 = _mm256_fmadd_ps(vx, w2, aD2); aD3 = _mm256_fmadd_ps(vx, w3, aD3); }
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 16), _mm256_cvtps_epi32(aA2)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 24), _mm256_cvtps_epi32(aA3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 16), _mm256_cvtps_epi32(aB2)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 24), _mm256_cvtps_epi32(aB3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accC + 16), _mm256_cvtps_epi32(aC2)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accC + 24), _mm256_cvtps_epi32(aC3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accD + 16), _mm256_cvtps_epi32(aD2)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accD + 24), _mm256_cvtps_epi32(aD3));
    }
#else
    dot_i2s_i8_accum_i32_block32_x2(accA, accB, Wq2, stride_cols, i0, hCount, x8A, x8B, j0);
    dot_i2s_i8_accum_i32_block32_x2(accC, accD, Wq2, stride_cols, i0, hCount, x8C, x8D, j0);
#endif
}

inline void dot_ternary_i8_accum_i32_block32(
    int32_t* __restrict acc32, const int8_t* __restrict Wt8, std::size_t stride_cols,
    std::size_t j0, std::size_t hCount, const int8_t* __restrict dy8, std::size_t i0)
{
#if defined(__AVX2__)
    __m256 a0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 0)));
    __m256 a1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 8)));
    __m256 a2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 16)));
    __m256 a3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 24)));

    for (std::size_t di = 0; di < hCount; ++di) {
        if (dy8[di] == 0) continue;
        const int8_t* const w = Wt8 + (j0 + di) * stride_cols + i0;
        _mm_prefetch(reinterpret_cast<const char*>(w + stride_cols), _MM_HINT_T0);
        const __m256 x = _mm256_set1_ps(static_cast<float>(dy8[di]));
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 0))));
        const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 8))));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 16))));
        const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 24))));
        a0 = _mm256_fmadd_ps(x, w0, a0);
        a1 = _mm256_fmadd_ps(x, w1, a1);
        a2 = _mm256_fmadd_ps(x, w2, a2);
        a3 = _mm256_fmadd_ps(x, w3, a3);
    }

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 0), _mm256_cvtps_epi32(a0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 8), _mm256_cvtps_epi32(a1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 16), _mm256_cvtps_epi32(a2));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 24), _mm256_cvtps_epi32(a3));
#else
    for (std::size_t di = 0; di < hCount; ++di) {
        const int x = static_cast<int>(dy8[di]);
        if (x == 0) continue;
        const int8_t* const w = Wt8 + (j0 + di) * stride_cols + i0;
        for (std::size_t u = 0; u < 32; ++u) {
            acc32[u] += x * static_cast<int>(w[u]);
        }
    }
#endif
}

inline void dot_ternary_i8_accum_i32_block32_x2(
    int32_t* __restrict accA, int32_t* __restrict accB, const int8_t* __restrict Wt8,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dyA, const int8_t* __restrict dyB, std::size_t i0)
{
#if defined(__AVX2__)
    __m256 aA0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 0)));
    __m256 aA1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 8)));
    __m256 aA2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 16)));
    __m256 aA3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA + 24)));
    __m256 aB0 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 0)));
    __m256 aB1 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 8)));
    __m256 aB2 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 16)));
    __m256 aB3 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB + 24)));

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dyA[di] | dyB[di])) continue;
        const int8_t* const w = Wt8 + (j0 + di) * stride_cols + i0;
        _mm_prefetch(reinterpret_cast<const char*>(w + stride_cols), _MM_HINT_T0);
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 0))));
        const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 8))));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 16))));
        const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 24))));
        if (dyA[di]) {
            const __m256 x = _mm256_set1_ps(static_cast<float>(dyA[di]));
            aA0 = _mm256_fmadd_ps(x, w0, aA0); aA1 = _mm256_fmadd_ps(x, w1, aA1);
            aA2 = _mm256_fmadd_ps(x, w2, aA2); aA3 = _mm256_fmadd_ps(x, w3, aA3);
        }
        if (dyB[di]) {
            const __m256 x = _mm256_set1_ps(static_cast<float>(dyB[di]));
            aB0 = _mm256_fmadd_ps(x, w0, aB0); aB1 = _mm256_fmadd_ps(x, w1, aB1);
            aB2 = _mm256_fmadd_ps(x, w2, aB2); aB3 = _mm256_fmadd_ps(x, w3, aB3);
        }
    }

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 0), _mm256_cvtps_epi32(aA0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 8), _mm256_cvtps_epi32(aA1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 16), _mm256_cvtps_epi32(aA2));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 24), _mm256_cvtps_epi32(aA3));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 0), _mm256_cvtps_epi32(aB0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 8), _mm256_cvtps_epi32(aB1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 16), _mm256_cvtps_epi32(aB2));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 24), _mm256_cvtps_epi32(aB3));
#else
    for (std::size_t di = 0; di < hCount; ++di) {
        const int xa = static_cast<int>(dyA[di]);
        const int xb = static_cast<int>(dyB[di]);
        if (xa == 0 && xb == 0) continue;
        const int8_t* const w = Wt8 + (j0 + di) * stride_cols + i0;
        for (std::size_t u = 0; u < 32; ++u) {
            const int wi = static_cast<int>(w[u]);
            if (xa) accA[u] += xa * wi;
            if (xb) accB[u] += xb * wi;
        }
    }
#endif
}

inline void dot_ternary_i8_accum_i32_block32_x4(
    int32_t* __restrict accA, int32_t* __restrict accB,
    int32_t* __restrict accC, int32_t* __restrict accD,
    const int8_t* __restrict Wt8, std::size_t stride_cols,
    std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dyA, const int8_t* __restrict dyB,
    const int8_t* __restrict dyC, const int8_t* __restrict dyD,
    std::size_t i0)
{
    dot_ternary_i8_accum_i32_block32_x2(accA, accB, Wt8, stride_cols, j0, hCount, dyA, dyB, i0);
    dot_ternary_i8_accum_i32_block32_x2(accC, accD, Wt8, stride_cols, j0, hCount, dyC, dyD, i0);
}

inline void dot_ternary_i8_accum_i32_block32_x4_triple(
    int32_t* __restrict accQA, int32_t* __restrict accQB,
    int32_t* __restrict accQC, int32_t* __restrict accQD,
    int32_t* __restrict accKA, int32_t* __restrict accKB,
    int32_t* __restrict accKC, int32_t* __restrict accKD,
    int32_t* __restrict accVA, int32_t* __restrict accVB,
    int32_t* __restrict accVC, int32_t* __restrict accVD,
    const int8_t* __restrict WqT8,
    const int8_t* __restrict WkT8,
    const int8_t* __restrict WvT8,
    std::size_t stride_cols,
    std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dqA, const int8_t* __restrict dqB,
    const int8_t* __restrict dqC, const int8_t* __restrict dqD,
    const int8_t* __restrict dkA, const int8_t* __restrict dkB,
    const int8_t* __restrict dkC, const int8_t* __restrict dkD,
    const int8_t* __restrict dvA, const int8_t* __restrict dvB,
    const int8_t* __restrict dvC, const int8_t* __restrict dvD,
    std::size_t i0)
{
#if defined(__AVX2__)
    auto run_half = [&](int accOffset) {
        __m256 qA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQA + accOffset)));
        __m256 qB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQB + accOffset)));
        __m256 qC = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQC + accOffset)));
        __m256 qD = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQD + accOffset)));
        __m256 kA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKA + accOffset)));
        __m256 kB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKB + accOffset)));
        __m256 kC = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKC + accOffset)));
        __m256 kD = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKD + accOffset)));
        __m256 vA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVA + accOffset)));
        __m256 vB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVB + accOffset)));
        __m256 vC = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVC + accOffset)));
        __m256 vD = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVD + accOffset)));

        for (std::size_t di = 0; di < hCount; ++di) {
            const int8_t q0 = dqA[di], q1 = dqB[di], q2 = dqC[di], q3 = dqD[di];
            const int8_t k0 = dkA[di], k1 = dkB[di], k2 = dkC[di], k3 = dkD[di];
            const int8_t v0 = dvA[di], v1 = dvB[di], v2 = dvC[di], v3 = dvD[di];
            if (!(q0 | q1 | q2 | q3 | k0 | k1 | k2 | k3 | v0 | v1 | v2 | v3)) continue;

            const int8_t* const wq = WqT8 + (j0 + di) * stride_cols + i0 + accOffset;
            const int8_t* const wk = WkT8 + (j0 + di) * stride_cols + i0 + accOffset;
            const int8_t* const wv = WvT8 + (j0 + di) * stride_cols + i0 + accOffset;
            _mm_prefetch(reinterpret_cast<const char*>(wq + stride_cols), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wk + stride_cols), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wv + stride_cols), _MM_HINT_T0);

            const __m256 wq8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(wq))));
            const __m256 wk8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(wk))));
            const __m256 wv8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(wv))));

            if (q0) qA = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(q0)), wq8, qA);
            if (q1) qB = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(q1)), wq8, qB);
            if (q2) qC = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(q2)), wq8, qC);
            if (q3) qD = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(q3)), wq8, qD);
            if (k0) kA = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(k0)), wk8, kA);
            if (k1) kB = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(k1)), wk8, kB);
            if (k2) kC = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(k2)), wk8, kC);
            if (k3) kD = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(k3)), wk8, kD);
            if (v0) vA = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(v0)), wv8, vA);
            if (v1) vB = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(v1)), wv8, vB);
            if (v2) vC = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(v2)), wv8, vC);
            if (v3) vD = _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(v3)), wv8, vD);
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accQA + accOffset), _mm256_cvtps_epi32(qA));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accQB + accOffset), _mm256_cvtps_epi32(qB));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accQC + accOffset), _mm256_cvtps_epi32(qC));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accQD + accOffset), _mm256_cvtps_epi32(qD));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accKA + accOffset), _mm256_cvtps_epi32(kA));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accKB + accOffset), _mm256_cvtps_epi32(kB));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accKC + accOffset), _mm256_cvtps_epi32(kC));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accKD + accOffset), _mm256_cvtps_epi32(kD));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accVA + accOffset), _mm256_cvtps_epi32(vA));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accVB + accOffset), _mm256_cvtps_epi32(vB));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accVC + accOffset), _mm256_cvtps_epi32(vC));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accVD + accOffset), _mm256_cvtps_epi32(vD));
    };

    run_half(0);
    run_half(8);
    run_half(16);
    run_half(24);
#else
    dot_ternary_i8_accum_i32_block32_x4(accQA, accQB, accQC, accQD, WqT8, stride_cols, j0, hCount, dqA, dqB, dqC, dqD, i0);
    dot_ternary_i8_accum_i32_block32_x4(accKA, accKB, accKC, accKD, WkT8, stride_cols, j0, hCount, dkA, dkB, dkC, dkD, i0);
    dot_ternary_i8_accum_i32_block32_x4(accVA, accVB, accVC, accVD, WvT8, stride_cols, j0, hCount, dvA, dvB, dvC, dvD, i0);
#endif
}

inline void dot_i2s_i8_accum_i32_block32_x4_triple(
    int32_t* __restrict accQA, int32_t* __restrict accQB,
    int32_t* __restrict accQC, int32_t* __restrict accQD,
    int32_t* __restrict accKA, int32_t* __restrict accKB,
    int32_t* __restrict accKC, int32_t* __restrict accKD,
    int32_t* __restrict accVA, int32_t* __restrict accVB,
    int32_t* __restrict accVC, int32_t* __restrict accVD,
    const uint8_t* __restrict Wqq2,
    const uint8_t* __restrict Wkq2,
    const uint8_t* __restrict Wvq2,
    std::size_t qStrideCols, std::size_t kStrideCols, std::size_t vStrideCols,
    std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B,
    const int8_t* __restrict x8C, const int8_t* __restrict x8D,
    std::size_t j0)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);

    auto run_half = [&](int accOffset, int byteOffset) {
        __m256 qA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQA + accOffset)));
        __m256 qB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQB + accOffset)));
        __m256 qC = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQC + accOffset)));
        __m256 qD = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accQD + accOffset)));
        __m256 kA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKA + accOffset)));
        __m256 kB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKB + accOffset)));
        __m256 kC = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKC + accOffset)));
        __m256 kD = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accKD + accOffset)));
        __m256 vA = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVA + accOffset)));
        __m256 vB = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVB + accOffset)));
        __m256 vC = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVC + accOffset)));
        __m256 vD = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accVD + accOffset)));

        for (std::size_t di = 0; di < hCount; ++di) {
            const int8_t xa = x8A[di];
            const int8_t xb = x8B[di];
            const int8_t xc = x8C[di];
            const int8_t xd = x8D[di];
            if (!(xa | xb | xc | xd)) continue;

            const uint8_t* wqBytes = Wqq2 + (((i0 + di) * qStrideCols + j0) >> 2);
            const uint8_t* wkBytes = Wkq2 + (((i0 + di) * kStrideCols + j0) >> 2);
            const uint8_t* wvBytes = Wvq2 + (((i0 + di) * vStrideCols + j0) >> 2);
            _mm_prefetch(reinterpret_cast<const char*>(wqBytes + qStrideCols), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wkBytes + kStrideCols), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wvBytes + vStrideCols), _MM_HINT_T0);

            const __m256 vxA = xa ? _mm256_set1_ps(static_cast<float>(xa)) : _mm256_setzero_ps();
            const __m256 vxB = xb ? _mm256_set1_ps(static_cast<float>(xb)) : _mm256_setzero_ps();
            const __m256 vxC = xc ? _mm256_set1_ps(static_cast<float>(xc)) : _mm256_setzero_ps();
            const __m256 vxD = xd ? _mm256_set1_ps(static_cast<float>(xd)) : _mm256_setzero_ps();

            const __m256 wq = bitnet_i2s_detail::decode_2bytes_to_ps(wqBytes + byteOffset, shift_vec, mask_vec, one_vec);
            if (xa) qA = _mm256_fmadd_ps(vxA, wq, qA);
            if (xb) qB = _mm256_fmadd_ps(vxB, wq, qB);
            if (xc) qC = _mm256_fmadd_ps(vxC, wq, qC);
            if (xd) qD = _mm256_fmadd_ps(vxD, wq, qD);

            const __m256 wk = bitnet_i2s_detail::decode_2bytes_to_ps(wkBytes + byteOffset, shift_vec, mask_vec, one_vec);
            if (xa) kA = _mm256_fmadd_ps(vxA, wk, kA);
            if (xb) kB = _mm256_fmadd_ps(vxB, wk, kB);
            if (xc) kC = _mm256_fmadd_ps(vxC, wk, kC);
            if (xd) kD = _mm256_fmadd_ps(vxD, wk, kD);

            const __m256 wv = bitnet_i2s_detail::decode_2bytes_to_ps(wvBytes + byteOffset, shift_vec, mask_vec, one_vec);
            if (xa) vA = _mm256_fmadd_ps(vxA, wv, vA);
            if (xb) vB = _mm256_fmadd_ps(vxB, wv, vB);
            if (xc) vC = _mm256_fmadd_ps(vxC, wv, vC);
            if (xd) vD = _mm256_fmadd_ps(vxD, wv, vD);
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accQA + accOffset), _mm256_cvtps_epi32(qA));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accQB + accOffset), _mm256_cvtps_epi32(qB));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accQC + accOffset), _mm256_cvtps_epi32(qC));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accQD + accOffset), _mm256_cvtps_epi32(qD));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accKA + accOffset), _mm256_cvtps_epi32(kA));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accKB + accOffset), _mm256_cvtps_epi32(kB));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accKC + accOffset), _mm256_cvtps_epi32(kC));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accKD + accOffset), _mm256_cvtps_epi32(kD));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accVA + accOffset), _mm256_cvtps_epi32(vA));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accVB + accOffset), _mm256_cvtps_epi32(vB));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accVC + accOffset), _mm256_cvtps_epi32(vC));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accVD + accOffset), _mm256_cvtps_epi32(vD));
    };

    run_half(0, 0);
    run_half(8, 2);
    run_half(16, 4);
    run_half(24, 6);
#else
    dot_i2s_i8_accum_i32_block32_x4(accQA, accQB, accQC, accQD, Wqq2, qStrideCols, i0, hCount, x8A, x8B, x8C, x8D, j0);
    dot_i2s_i8_accum_i32_block32_x4(accKA, accKB, accKC, accKD, Wkq2, kStrideCols, i0, hCount, x8A, x8B, x8C, x8D, j0);
    dot_i2s_i8_accum_i32_block32_x4(accVA, accVB, accVC, accVD, Wvq2, vStrideCols, i0, hCount, x8A, x8B, x8C, x8D, j0);
#endif
}

inline void dot_i2s_i8_accum_i32_block32_x4_dual(
    int32_t* __restrict accA0, int32_t* __restrict accA1,
    int32_t* __restrict accA2, int32_t* __restrict accA3,
    int32_t* __restrict accB0, int32_t* __restrict accB1,
    int32_t* __restrict accB2, int32_t* __restrict accB3,
    const uint8_t* __restrict Waq2, const uint8_t* __restrict Wbq2,
    std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict xA0, const int8_t* __restrict xA1,
    const int8_t* __restrict xA2, const int8_t* __restrict xA3,
    const int8_t* __restrict xB0, const int8_t* __restrict xB1,
    const int8_t* __restrict xB2, const int8_t* __restrict xB3,
    std::size_t j0)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    {
        __m256 a00 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA0 + 0)));
        __m256 a01 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA0 + 8)));
        __m256 a10 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA1 + 0)));
        __m256 a11 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA1 + 8)));
        __m256 a20 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA2 + 0)));
        __m256 a21 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA2 + 8)));
        __m256 a30 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA3 + 0)));
        __m256 a31 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA3 + 8)));
        __m256 b00 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB0 + 0)));
        __m256 b01 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB0 + 8)));
        __m256 b10 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB1 + 0)));
        __m256 b11 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB1 + 8)));
        __m256 b20 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB2 + 0)));
        __m256 b21 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB2 + 8)));
        __m256 b30 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB3 + 0)));
        __m256 b31 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB3 + 8)));

        for (std::size_t di = 0; di < hCount; ++di) {
            if (!(xA0[di] | xA1[di] | xA2[di] | xA3[di] | xB0[di] | xB1[di] | xB2[di] | xB3[di])) continue;
            const uint8_t* wa = Waq2 + (((i0 + di) * stride_cols + j0) >> 2);
            const uint8_t* wb = Wbq2 + (((i0 + di) * stride_cols + j0) >> 2);
            _mm_prefetch(reinterpret_cast<const char*>(wa + prefetch_stride), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wb + prefetch_stride), _MM_HINT_T0);

            const __m256 wa0 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 0, shift_vec, mask_vec, one_vec);
            const __m256 wa1 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 2, shift_vec, mask_vec, one_vec);
            const __m256 wb0 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 0, shift_vec, mask_vec, one_vec);
            const __m256 wb1 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 2, shift_vec, mask_vec, one_vec);

            if (xA0[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xA0[di])); a00 = _mm256_fmadd_ps(vx, wa0, a00); a01 = _mm256_fmadd_ps(vx, wa1, a01); }
            if (xA1[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xA1[di])); a10 = _mm256_fmadd_ps(vx, wa0, a10); a11 = _mm256_fmadd_ps(vx, wa1, a11); }
            if (xA2[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xA2[di])); a20 = _mm256_fmadd_ps(vx, wa0, a20); a21 = _mm256_fmadd_ps(vx, wa1, a21); }
            if (xA3[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xA3[di])); a30 = _mm256_fmadd_ps(vx, wa0, a30); a31 = _mm256_fmadd_ps(vx, wa1, a31); }
            if (xB0[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xB0[di])); b00 = _mm256_fmadd_ps(vx, wb0, b00); b01 = _mm256_fmadd_ps(vx, wb1, b01); }
            if (xB1[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xB1[di])); b10 = _mm256_fmadd_ps(vx, wb0, b10); b11 = _mm256_fmadd_ps(vx, wb1, b11); }
            if (xB2[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xB2[di])); b20 = _mm256_fmadd_ps(vx, wb0, b20); b21 = _mm256_fmadd_ps(vx, wb1, b21); }
            if (xB3[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xB3[di])); b30 = _mm256_fmadd_ps(vx, wb0, b30); b31 = _mm256_fmadd_ps(vx, wb1, b31); }
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA0 + 0), _mm256_cvtps_epi32(a00));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA0 + 8), _mm256_cvtps_epi32(a01));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA1 + 0), _mm256_cvtps_epi32(a10));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA1 + 8), _mm256_cvtps_epi32(a11));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA2 + 0), _mm256_cvtps_epi32(a20));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA2 + 8), _mm256_cvtps_epi32(a21));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA3 + 0), _mm256_cvtps_epi32(a30));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA3 + 8), _mm256_cvtps_epi32(a31));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB0 + 0), _mm256_cvtps_epi32(b00));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB0 + 8), _mm256_cvtps_epi32(b01));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB1 + 0), _mm256_cvtps_epi32(b10));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB1 + 8), _mm256_cvtps_epi32(b11));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB2 + 0), _mm256_cvtps_epi32(b20));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB2 + 8), _mm256_cvtps_epi32(b21));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB3 + 0), _mm256_cvtps_epi32(b30));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB3 + 8), _mm256_cvtps_epi32(b31));
    }

    {
        __m256 a00 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA0 + 16)));
        __m256 a01 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA0 + 24)));
        __m256 a10 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA1 + 16)));
        __m256 a11 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA1 + 24)));
        __m256 a20 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA2 + 16)));
        __m256 a21 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA2 + 24)));
        __m256 a30 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA3 + 16)));
        __m256 a31 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accA3 + 24)));
        __m256 b00 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB0 + 16)));
        __m256 b01 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB0 + 24)));
        __m256 b10 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB1 + 16)));
        __m256 b11 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB1 + 24)));
        __m256 b20 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB2 + 16)));
        __m256 b21 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB2 + 24)));
        __m256 b30 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB3 + 16)));
        __m256 b31 = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(accB3 + 24)));

        for (std::size_t di = 0; di < hCount; ++di) {
            if (!(xA0[di] | xA1[di] | xA2[di] | xA3[di] | xB0[di] | xB1[di] | xB2[di] | xB3[di])) continue;
            const uint8_t* wa = Waq2 + (((i0 + di) * stride_cols + j0) >> 2);
            const uint8_t* wb = Wbq2 + (((i0 + di) * stride_cols + j0) >> 2);
            _mm_prefetch(reinterpret_cast<const char*>(wa + prefetch_stride), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(wb + prefetch_stride), _MM_HINT_T0);

            const __m256 wa0 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 4, shift_vec, mask_vec, one_vec);
            const __m256 wa1 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 6, shift_vec, mask_vec, one_vec);
            const __m256 wb0 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 4, shift_vec, mask_vec, one_vec);
            const __m256 wb1 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 6, shift_vec, mask_vec, one_vec);

            if (xA0[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xA0[di])); a00 = _mm256_fmadd_ps(vx, wa0, a00); a01 = _mm256_fmadd_ps(vx, wa1, a01); }
            if (xA1[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xA1[di])); a10 = _mm256_fmadd_ps(vx, wa0, a10); a11 = _mm256_fmadd_ps(vx, wa1, a11); }
            if (xA2[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xA2[di])); a20 = _mm256_fmadd_ps(vx, wa0, a20); a21 = _mm256_fmadd_ps(vx, wa1, a21); }
            if (xA3[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xA3[di])); a30 = _mm256_fmadd_ps(vx, wa0, a30); a31 = _mm256_fmadd_ps(vx, wa1, a31); }
            if (xB0[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xB0[di])); b00 = _mm256_fmadd_ps(vx, wb0, b00); b01 = _mm256_fmadd_ps(vx, wb1, b01); }
            if (xB1[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xB1[di])); b10 = _mm256_fmadd_ps(vx, wb0, b10); b11 = _mm256_fmadd_ps(vx, wb1, b11); }
            if (xB2[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xB2[di])); b20 = _mm256_fmadd_ps(vx, wb0, b20); b21 = _mm256_fmadd_ps(vx, wb1, b21); }
            if (xB3[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(xB3[di])); b30 = _mm256_fmadd_ps(vx, wb0, b30); b31 = _mm256_fmadd_ps(vx, wb1, b31); }
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA0 + 16), _mm256_cvtps_epi32(a00));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA0 + 24), _mm256_cvtps_epi32(a01));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA1 + 16), _mm256_cvtps_epi32(a10));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA1 + 24), _mm256_cvtps_epi32(a11));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA2 + 16), _mm256_cvtps_epi32(a20));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA2 + 24), _mm256_cvtps_epi32(a21));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA3 + 16), _mm256_cvtps_epi32(a30));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA3 + 24), _mm256_cvtps_epi32(a31));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB0 + 16), _mm256_cvtps_epi32(b00));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB0 + 24), _mm256_cvtps_epi32(b01));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB1 + 16), _mm256_cvtps_epi32(b10));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB1 + 24), _mm256_cvtps_epi32(b11));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB2 + 16), _mm256_cvtps_epi32(b20));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB2 + 24), _mm256_cvtps_epi32(b21));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB3 + 16), _mm256_cvtps_epi32(b30));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB3 + 24), _mm256_cvtps_epi32(b31));
    }
#else
    dot_i2s_i8_accum_i32_block32_x4(accA0, accA1, accA2, accA3, Waq2, stride_cols, i0, hCount, xA0, xA1, xA2, xA3, j0);
    dot_i2s_i8_accum_i32_block32_x4(accB0, accB1, accB2, accB3, Wbq2, stride_cols, i0, hCount, xB0, xB1, xB2, xB3, j0);
#endif
}

inline void dot_i2s_i8_scale_store_block32_x4(
    float* __restrict yA, float* __restrict yB,
    float* __restrict yC, float* __restrict yD,
    const uint8_t* __restrict Wq2, std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B,
    const int8_t* __restrict x8C, const int8_t* __restrict x8D,
    std::size_t j0,
    float gainA, float gainB, float gainC, float gainD)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    {
        __m256 aA0 = _mm256_setzero_ps();
        __m256 aA1 = _mm256_setzero_ps();
        __m256 aB0 = _mm256_setzero_ps();
        __m256 aB1 = _mm256_setzero_ps();
        __m256 aC0 = _mm256_setzero_ps();
        __m256 aC1 = _mm256_setzero_ps();
        __m256 aD0 = _mm256_setzero_ps();
        __m256 aD1 = _mm256_setzero_ps();

        for (std::size_t di = 0; di < hCount; ++di) {
            if (!(x8A[di] | x8B[di] | x8C[di] | x8D[di])) continue;
            const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
            _mm_prefetch(reinterpret_cast<const char*>(wbytes + prefetch_stride), _MM_HINT_T0);

            const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec);
            const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec);

            if (x8A[di]) {
                const __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di]));
                aA0 = _mm256_fmadd_ps(vx, w0, aA0);
                aA1 = _mm256_fmadd_ps(vx, w1, aA1);
            }
            if (x8B[di]) {
                const __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di]));
                aB0 = _mm256_fmadd_ps(vx, w0, aB0);
                aB1 = _mm256_fmadd_ps(vx, w1, aB1);
            }
            if (x8C[di]) {
                const __m256 vx = _mm256_set1_ps(static_cast<float>(x8C[di]));
                aC0 = _mm256_fmadd_ps(vx, w0, aC0);
                aC1 = _mm256_fmadd_ps(vx, w1, aC1);
            }
            if (x8D[di]) {
                const __m256 vx = _mm256_set1_ps(static_cast<float>(x8D[di]));
                aD0 = _mm256_fmadd_ps(vx, w0, aD0);
                aD1 = _mm256_fmadd_ps(vx, w1, aD1);
            }
        }

        const __m256 gainA8 = _mm256_set1_ps(gainA);
        const __m256 gainB8 = _mm256_set1_ps(gainB);
        const __m256 gainC8 = _mm256_set1_ps(gainC);
        const __m256 gainD8 = _mm256_set1_ps(gainD);

        __m256 yv = _mm256_loadu_ps(yA + j0 + 0);
        yv = _mm256_fmadd_ps(aA0, gainA8, yv);
        _mm256_storeu_ps(yA + j0 + 0, yv);
        yv = _mm256_loadu_ps(yA + j0 + 8);
        yv = _mm256_fmadd_ps(aA1, gainA8, yv);
        _mm256_storeu_ps(yA + j0 + 8, yv);

        yv = _mm256_loadu_ps(yB + j0 + 0);
        yv = _mm256_fmadd_ps(aB0, gainB8, yv);
        _mm256_storeu_ps(yB + j0 + 0, yv);
        yv = _mm256_loadu_ps(yB + j0 + 8);
        yv = _mm256_fmadd_ps(aB1, gainB8, yv);
        _mm256_storeu_ps(yB + j0 + 8, yv);

        yv = _mm256_loadu_ps(yC + j0 + 0);
        yv = _mm256_fmadd_ps(aC0, gainC8, yv);
        _mm256_storeu_ps(yC + j0 + 0, yv);
        yv = _mm256_loadu_ps(yC + j0 + 8);
        yv = _mm256_fmadd_ps(aC1, gainC8, yv);
        _mm256_storeu_ps(yC + j0 + 8, yv);

        yv = _mm256_loadu_ps(yD + j0 + 0);
        yv = _mm256_fmadd_ps(aD0, gainD8, yv);
        _mm256_storeu_ps(yD + j0 + 0, yv);
        yv = _mm256_loadu_ps(yD + j0 + 8);
        yv = _mm256_fmadd_ps(aD1, gainD8, yv);
        _mm256_storeu_ps(yD + j0 + 8, yv);
    }

    {
        __m256 aA2 = _mm256_setzero_ps();
        __m256 aA3 = _mm256_setzero_ps();
        __m256 aB2 = _mm256_setzero_ps();
        __m256 aB3 = _mm256_setzero_ps();
        __m256 aC2 = _mm256_setzero_ps();
        __m256 aC3 = _mm256_setzero_ps();
        __m256 aD2 = _mm256_setzero_ps();
        __m256 aD3 = _mm256_setzero_ps();

        for (std::size_t di = 0; di < hCount; ++di) {
            if (!(x8A[di] | x8B[di] | x8C[di] | x8D[di])) continue;
            const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
            _mm_prefetch(reinterpret_cast<const char*>(wbytes + prefetch_stride), _MM_HINT_T0);

            const __m256 w2 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 4, shift_vec, mask_vec, one_vec);
            const __m256 w3 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 6, shift_vec, mask_vec, one_vec);

            if (x8A[di]) {
                const __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di]));
                aA2 = _mm256_fmadd_ps(vx, w2, aA2);
                aA3 = _mm256_fmadd_ps(vx, w3, aA3);
            }
            if (x8B[di]) {
                const __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di]));
                aB2 = _mm256_fmadd_ps(vx, w2, aB2);
                aB3 = _mm256_fmadd_ps(vx, w3, aB3);
            }
            if (x8C[di]) {
                const __m256 vx = _mm256_set1_ps(static_cast<float>(x8C[di]));
                aC2 = _mm256_fmadd_ps(vx, w2, aC2);
                aC3 = _mm256_fmadd_ps(vx, w3, aC3);
            }
            if (x8D[di]) {
                const __m256 vx = _mm256_set1_ps(static_cast<float>(x8D[di]));
                aD2 = _mm256_fmadd_ps(vx, w2, aD2);
                aD3 = _mm256_fmadd_ps(vx, w3, aD3);
            }
        }

        const __m256 gainA8 = _mm256_set1_ps(gainA);
        const __m256 gainB8 = _mm256_set1_ps(gainB);
        const __m256 gainC8 = _mm256_set1_ps(gainC);
        const __m256 gainD8 = _mm256_set1_ps(gainD);

        __m256 yv = _mm256_loadu_ps(yA + j0 + 16);
        yv = _mm256_fmadd_ps(aA2, gainA8, yv);
        _mm256_storeu_ps(yA + j0 + 16, yv);
        yv = _mm256_loadu_ps(yA + j0 + 24);
        yv = _mm256_fmadd_ps(aA3, gainA8, yv);
        _mm256_storeu_ps(yA + j0 + 24, yv);

        yv = _mm256_loadu_ps(yB + j0 + 16);
        yv = _mm256_fmadd_ps(aB2, gainB8, yv);
        _mm256_storeu_ps(yB + j0 + 16, yv);
        yv = _mm256_loadu_ps(yB + j0 + 24);
        yv = _mm256_fmadd_ps(aB3, gainB8, yv);
        _mm256_storeu_ps(yB + j0 + 24, yv);

        yv = _mm256_loadu_ps(yC + j0 + 16);
        yv = _mm256_fmadd_ps(aC2, gainC8, yv);
        _mm256_storeu_ps(yC + j0 + 16, yv);
        yv = _mm256_loadu_ps(yC + j0 + 24);
        yv = _mm256_fmadd_ps(aC3, gainC8, yv);
        _mm256_storeu_ps(yC + j0 + 24, yv);

        yv = _mm256_loadu_ps(yD + j0 + 16);
        yv = _mm256_fmadd_ps(aD2, gainD8, yv);
        _mm256_storeu_ps(yD + j0 + 16, yv);
        yv = _mm256_loadu_ps(yD + j0 + 24);
        yv = _mm256_fmadd_ps(aD3, gainD8, yv);
        _mm256_storeu_ps(yD + j0 + 24, yv);
    }
#else
    int32_t accA[32] = { 0 }, accB[32] = { 0 }, accC[32] = { 0 }, accD[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x4(
        accA, accB, accC, accD,
        Wq2, stride_cols, i0, hCount,
        x8A, x8B, x8C, x8D,
        j0);
    for (std::size_t off = 0; off < 32; ++off) {
        yA[j0 + off] += gainA * static_cast<float>(accA[off]);
        yB[j0 + off] += gainB * static_cast<float>(accB[off]);
        yC[j0 + off] += gainC * static_cast<float>(accC[off]);
        yD[j0 + off] += gainD * static_cast<float>(accD[off]);
    }
#endif
}

inline void dot_i2s_i8_scale_store_block16_x4_exact(
    float* __restrict yA, float* __restrict yB,
    float* __restrict yC, float* __restrict yD,
    const uint8_t* __restrict Wq2, std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B,
    const int8_t* __restrict x8C, const int8_t* __restrict x8D,
    std::size_t j0,
    bool addToDest,
    float gainA, float gainB, float gainC, float gainD)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 aA0 = _mm256_setzero_ps();
    __m256 aA1 = _mm256_setzero_ps();
    __m256 aB0 = _mm256_setzero_ps();
    __m256 aB1 = _mm256_setzero_ps();
    __m256 aC0 = _mm256_setzero_ps();
    __m256 aC1 = _mm256_setzero_ps();
    __m256 aD0 = _mm256_setzero_ps();
    __m256 aD1 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(x8A[di] | x8B[di] | x8C[di] | x8D[di])) continue;
        const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wbytes + prefetch_stride), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec);

        if (x8A[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di]));
            aA0 = _mm256_fmadd_ps(vx, w0, aA0);
            aA1 = _mm256_fmadd_ps(vx, w1, aA1);
        }
        if (x8B[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di]));
            aB0 = _mm256_fmadd_ps(vx, w0, aB0);
            aB1 = _mm256_fmadd_ps(vx, w1, aB1);
        }
        if (x8C[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x8C[di]));
            aC0 = _mm256_fmadd_ps(vx, w0, aC0);
            aC1 = _mm256_fmadd_ps(vx, w1, aC1);
        }
        if (x8D[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x8D[di]));
            aD0 = _mm256_fmadd_ps(vx, w0, aD0);
            aD1 = _mm256_fmadd_ps(vx, w1, aD1);
        }
    }

    const __m256 gainA8 = _mm256_set1_ps(gainA);
    const __m256 gainB8 = _mm256_set1_ps(gainB);
    const __m256 gainC8 = _mm256_set1_ps(gainC);
    const __m256 gainD8 = _mm256_set1_ps(gainD);

    const __m256 outA0 = _mm256_mul_ps(aA0, gainA8);
    const __m256 outA1 = _mm256_mul_ps(aA1, gainA8);
    const __m256 outB0 = _mm256_mul_ps(aB0, gainB8);
    const __m256 outB1 = _mm256_mul_ps(aB1, gainB8);
    const __m256 outC0 = _mm256_mul_ps(aC0, gainC8);
    const __m256 outC1 = _mm256_mul_ps(aC1, gainC8);
    const __m256 outD0 = _mm256_mul_ps(aD0, gainD8);
    const __m256 outD1 = _mm256_mul_ps(aD1, gainD8);

    if (addToDest) {
        __m256 yv = _mm256_loadu_ps(yA + j0 + 0);
        yv = _mm256_add_ps(outA0, yv);
        _mm256_storeu_ps(yA + j0 + 0, yv);
        yv = _mm256_loadu_ps(yA + j0 + 8);
        yv = _mm256_add_ps(outA1, yv);
        _mm256_storeu_ps(yA + j0 + 8, yv);

        yv = _mm256_loadu_ps(yB + j0 + 0);
        yv = _mm256_add_ps(outB0, yv);
        _mm256_storeu_ps(yB + j0 + 0, yv);
        yv = _mm256_loadu_ps(yB + j0 + 8);
        yv = _mm256_add_ps(outB1, yv);
        _mm256_storeu_ps(yB + j0 + 8, yv);

        yv = _mm256_loadu_ps(yC + j0 + 0);
        yv = _mm256_add_ps(outC0, yv);
        _mm256_storeu_ps(yC + j0 + 0, yv);
        yv = _mm256_loadu_ps(yC + j0 + 8);
        yv = _mm256_add_ps(outC1, yv);
        _mm256_storeu_ps(yC + j0 + 8, yv);

        yv = _mm256_loadu_ps(yD + j0 + 0);
        yv = _mm256_add_ps(outD0, yv);
        _mm256_storeu_ps(yD + j0 + 0, yv);
        yv = _mm256_loadu_ps(yD + j0 + 8);
        yv = _mm256_add_ps(outD1, yv);
        _mm256_storeu_ps(yD + j0 + 8, yv);
    }
    else {
        _mm256_storeu_ps(yA + j0 + 0, outA0);
        _mm256_storeu_ps(yA + j0 + 8, outA1);
        _mm256_storeu_ps(yB + j0 + 0, outB0);
        _mm256_storeu_ps(yB + j0 + 8, outB1);
        _mm256_storeu_ps(yC + j0 + 0, outC0);
        _mm256_storeu_ps(yC + j0 + 8, outC1);
        _mm256_storeu_ps(yD + j0 + 0, outD0);
        _mm256_storeu_ps(yD + j0 + 8, outD1);
    }
#else
    int32_t accA[32] = { 0 }, accB[32] = { 0 }, accC[32] = { 0 }, accD[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x4(
        accA, accB, accC, accD,
        Wq2, stride_cols, i0, hCount,
        x8A, x8B, x8C, x8D,
        (j0 & ~std::size_t(31)));
    const std::size_t base = j0 & std::size_t(16);
    for (std::size_t off = 0; off < 16; ++off) {
        const float outA = gainA * static_cast<float>(accA[base + off]);
        const float outB = gainB * static_cast<float>(accB[base + off]);
        const float outC = gainC * static_cast<float>(accC[base + off]);
        const float outD = gainD * static_cast<float>(accD[base + off]);
        if (addToDest) {
            yA[j0 + off] += outA;
            yB[j0 + off] += outB;
            yC[j0 + off] += outC;
            yD[j0 + off] += outD;
        }
        else {
            yA[j0 + off] = outA;
            yB[j0 + off] = outB;
            yC[j0 + off] = outC;
            yD[j0 + off] = outD;
        }
    }
#endif
}

inline void dot_i2s_i8_scale_store_block16_x2_exact(
    float* __restrict yA, float* __restrict yB,
    const uint8_t* __restrict Wq2, std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B,
    std::size_t j0,
    float gainA, float gainB)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 aA0 = _mm256_setzero_ps();
    __m256 aA1 = _mm256_setzero_ps();
    __m256 aB0 = _mm256_setzero_ps();
    __m256 aB1 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(x8A[di] | x8B[di])) continue;
        const uint8_t* wbytes = Wq2 + (((i0 + di) * stride_cols + j0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wbytes + prefetch_stride), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec);
        if (x8A[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di]));
            aA0 = _mm256_fmadd_ps(vx, w0, aA0);
            aA1 = _mm256_fmadd_ps(vx, w1, aA1);
        }
        if (x8B[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di]));
            aB0 = _mm256_fmadd_ps(vx, w0, aB0);
            aB1 = _mm256_fmadd_ps(vx, w1, aB1);
        }
    }

    const __m256 gainA8 = _mm256_set1_ps(gainA);
    const __m256 gainB8 = _mm256_set1_ps(gainB);

    __m256 yv = _mm256_loadu_ps(yA + j0 + 0);
    yv = _mm256_fmadd_ps(aA0, gainA8, yv);
    _mm256_storeu_ps(yA + j0 + 0, yv);
    yv = _mm256_loadu_ps(yA + j0 + 8);
    yv = _mm256_fmadd_ps(aA1, gainA8, yv);
    _mm256_storeu_ps(yA + j0 + 8, yv);

    yv = _mm256_loadu_ps(yB + j0 + 0);
    yv = _mm256_fmadd_ps(aB0, gainB8, yv);
    _mm256_storeu_ps(yB + j0 + 0, yv);
    yv = _mm256_loadu_ps(yB + j0 + 8);
    yv = _mm256_fmadd_ps(aB1, gainB8, yv);
    _mm256_storeu_ps(yB + j0 + 8, yv);
#else
    int32_t accA[32] = { 0 }, accB[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x2(accA, accB, Wq2, stride_cols, i0, hCount, x8A, x8B, (j0 & ~std::size_t(31)));
    const std::size_t base = j0 & std::size_t(16);
    for (std::size_t off = 0; off < 16; ++off) {
        yA[j0 + off] += gainA * static_cast<float>(accA[base + off]);
        yB[j0 + off] += gainB * static_cast<float>(accB[base + off]);
    }
#endif
}

inline void dot_i2s_i8_scale_store_block16_x4_dual_exact(
    float* __restrict yA0, float* __restrict yA1,
    float* __restrict yA2, float* __restrict yA3,
    float* __restrict yB0, float* __restrict yB1,
    float* __restrict yB2, float* __restrict yB3,
    const uint8_t* __restrict Waq2, const uint8_t* __restrict Wbq2,
    std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x0, const int8_t* __restrict x1,
    const int8_t* __restrict x2, const int8_t* __restrict x3,
    std::size_t j0,
    bool addToDest,
    float gainA0, float gainA1, float gainA2, float gainA3,
    float gainB0, float gainB1, float gainB2, float gainB3)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
    __m256 b00 = _mm256_setzero_ps(), b01 = _mm256_setzero_ps();
    __m256 b10 = _mm256_setzero_ps(), b11 = _mm256_setzero_ps();
    __m256 b20 = _mm256_setzero_ps(), b21 = _mm256_setzero_ps();
    __m256 b30 = _mm256_setzero_ps(), b31 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(x0[di] | x1[di] | x2[di] | x3[di])) continue;
        const uint8_t* wa = Waq2 + (((i0 + di) * stride_cols + j0) >> 2);
        const uint8_t* wb = Wbq2 + (((i0 + di) * stride_cols + j0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wa + prefetch_stride), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(wb + prefetch_stride), _MM_HINT_T0);

        const __m256 wa0 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 0, shift_vec, mask_vec, one_vec);
        const __m256 wa1 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 2, shift_vec, mask_vec, one_vec);
        const __m256 wb0 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 0, shift_vec, mask_vec, one_vec);
        const __m256 wb1 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 2, shift_vec, mask_vec, one_vec);

        if (x0[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x0[di]));
            a00 = _mm256_fmadd_ps(vx, wa0, a00); a01 = _mm256_fmadd_ps(vx, wa1, a01);
            b00 = _mm256_fmadd_ps(vx, wb0, b00); b01 = _mm256_fmadd_ps(vx, wb1, b01);
        }
        if (x1[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x1[di]));
            a10 = _mm256_fmadd_ps(vx, wa0, a10); a11 = _mm256_fmadd_ps(vx, wa1, a11);
            b10 = _mm256_fmadd_ps(vx, wb0, b10); b11 = _mm256_fmadd_ps(vx, wb1, b11);
        }
        if (x2[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x2[di]));
            a20 = _mm256_fmadd_ps(vx, wa0, a20); a21 = _mm256_fmadd_ps(vx, wa1, a21);
            b20 = _mm256_fmadd_ps(vx, wb0, b20); b21 = _mm256_fmadd_ps(vx, wb1, b21);
        }
        if (x3[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(x3[di]));
            a30 = _mm256_fmadd_ps(vx, wa0, a30); a31 = _mm256_fmadd_ps(vx, wa1, a31);
            b30 = _mm256_fmadd_ps(vx, wb0, b30); b31 = _mm256_fmadd_ps(vx, wb1, b31);
        }
    }

    const __m256 ga0 = _mm256_set1_ps(gainA0), ga1 = _mm256_set1_ps(gainA1);
    const __m256 ga2 = _mm256_set1_ps(gainA2), ga3 = _mm256_set1_ps(gainA3);
    const __m256 gb0 = _mm256_set1_ps(gainB0), gb1 = _mm256_set1_ps(gainB1);
    const __m256 gb2 = _mm256_set1_ps(gainB2), gb3 = _mm256_set1_ps(gainB3);

    const __m256 outA00 = _mm256_mul_ps(a00, ga0);
    const __m256 outA01 = _mm256_mul_ps(a01, ga0);
    const __m256 outA10 = _mm256_mul_ps(a10, ga1);
    const __m256 outA11 = _mm256_mul_ps(a11, ga1);
    const __m256 outA20 = _mm256_mul_ps(a20, ga2);
    const __m256 outA21 = _mm256_mul_ps(a21, ga2);
    const __m256 outA30 = _mm256_mul_ps(a30, ga3);
    const __m256 outA31 = _mm256_mul_ps(a31, ga3);
    const __m256 outB00 = _mm256_mul_ps(b00, gb0);
    const __m256 outB01 = _mm256_mul_ps(b01, gb0);
    const __m256 outB10 = _mm256_mul_ps(b10, gb1);
    const __m256 outB11 = _mm256_mul_ps(b11, gb1);
    const __m256 outB20 = _mm256_mul_ps(b20, gb2);
    const __m256 outB21 = _mm256_mul_ps(b21, gb2);
    const __m256 outB30 = _mm256_mul_ps(b30, gb3);
    const __m256 outB31 = _mm256_mul_ps(b31, gb3);

    if (addToDest) {
        __m256 yv = _mm256_loadu_ps(yA0 + j0 + 0); yv = _mm256_add_ps(outA00, yv); _mm256_storeu_ps(yA0 + j0 + 0, yv);
        yv = _mm256_loadu_ps(yA0 + j0 + 8); yv = _mm256_add_ps(outA01, yv); _mm256_storeu_ps(yA0 + j0 + 8, yv);
        yv = _mm256_loadu_ps(yA1 + j0 + 0); yv = _mm256_add_ps(outA10, yv); _mm256_storeu_ps(yA1 + j0 + 0, yv);
        yv = _mm256_loadu_ps(yA1 + j0 + 8); yv = _mm256_add_ps(outA11, yv); _mm256_storeu_ps(yA1 + j0 + 8, yv);
        yv = _mm256_loadu_ps(yA2 + j0 + 0); yv = _mm256_add_ps(outA20, yv); _mm256_storeu_ps(yA2 + j0 + 0, yv);
        yv = _mm256_loadu_ps(yA2 + j0 + 8); yv = _mm256_add_ps(outA21, yv); _mm256_storeu_ps(yA2 + j0 + 8, yv);
        yv = _mm256_loadu_ps(yA3 + j0 + 0); yv = _mm256_add_ps(outA30, yv); _mm256_storeu_ps(yA3 + j0 + 0, yv);
        yv = _mm256_loadu_ps(yA3 + j0 + 8); yv = _mm256_add_ps(outA31, yv); _mm256_storeu_ps(yA3 + j0 + 8, yv);

        yv = _mm256_loadu_ps(yB0 + j0 + 0); yv = _mm256_add_ps(outB00, yv); _mm256_storeu_ps(yB0 + j0 + 0, yv);
        yv = _mm256_loadu_ps(yB0 + j0 + 8); yv = _mm256_add_ps(outB01, yv); _mm256_storeu_ps(yB0 + j0 + 8, yv);
        yv = _mm256_loadu_ps(yB1 + j0 + 0); yv = _mm256_add_ps(outB10, yv); _mm256_storeu_ps(yB1 + j0 + 0, yv);
        yv = _mm256_loadu_ps(yB1 + j0 + 8); yv = _mm256_add_ps(outB11, yv); _mm256_storeu_ps(yB1 + j0 + 8, yv);
        yv = _mm256_loadu_ps(yB2 + j0 + 0); yv = _mm256_add_ps(outB20, yv); _mm256_storeu_ps(yB2 + j0 + 0, yv);
        yv = _mm256_loadu_ps(yB2 + j0 + 8); yv = _mm256_add_ps(outB21, yv); _mm256_storeu_ps(yB2 + j0 + 8, yv);
        yv = _mm256_loadu_ps(yB3 + j0 + 0); yv = _mm256_add_ps(outB30, yv); _mm256_storeu_ps(yB3 + j0 + 0, yv);
        yv = _mm256_loadu_ps(yB3 + j0 + 8); yv = _mm256_add_ps(outB31, yv); _mm256_storeu_ps(yB3 + j0 + 8, yv);
    }
    else {
        _mm256_storeu_ps(yA0 + j0 + 0, outA00);
        _mm256_storeu_ps(yA0 + j0 + 8, outA01);
        _mm256_storeu_ps(yA1 + j0 + 0, outA10);
        _mm256_storeu_ps(yA1 + j0 + 8, outA11);
        _mm256_storeu_ps(yA2 + j0 + 0, outA20);
        _mm256_storeu_ps(yA2 + j0 + 8, outA21);
        _mm256_storeu_ps(yA3 + j0 + 0, outA30);
        _mm256_storeu_ps(yA3 + j0 + 8, outA31);
        _mm256_storeu_ps(yB0 + j0 + 0, outB00);
        _mm256_storeu_ps(yB0 + j0 + 8, outB01);
        _mm256_storeu_ps(yB1 + j0 + 0, outB10);
        _mm256_storeu_ps(yB1 + j0 + 8, outB11);
        _mm256_storeu_ps(yB2 + j0 + 0, outB20);
        _mm256_storeu_ps(yB2 + j0 + 8, outB21);
        _mm256_storeu_ps(yB3 + j0 + 0, outB30);
        _mm256_storeu_ps(yB3 + j0 + 8, outB31);
    }
#else
    int32_t accA0[32] = { 0 }, accA1[32] = { 0 }, accA2[32] = { 0 }, accA3[32] = { 0 };
    int32_t accB0[32] = { 0 }, accB1[32] = { 0 }, accB2[32] = { 0 }, accB3[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x4(accA0, accA1, accA2, accA3, Waq2, stride_cols, i0, hCount, x0, x1, x2, x3, j0);
    dot_i2s_i8_accum_i32_block32_x4(accB0, accB1, accB2, accB3, Wbq2, stride_cols, i0, hCount, x0, x1, x2, x3, j0);
    for (std::size_t off = 0; off < 16; ++off) {
        const float outA0 = gainA0 * static_cast<float>(accA0[off]);
        const float outA1 = gainA1 * static_cast<float>(accA1[off]);
        const float outA2 = gainA2 * static_cast<float>(accA2[off]);
        const float outA3 = gainA3 * static_cast<float>(accA3[off]);
        const float outB0 = gainB0 * static_cast<float>(accB0[off]);
        const float outB1 = gainB1 * static_cast<float>(accB1[off]);
        const float outB2 = gainB2 * static_cast<float>(accB2[off]);
        const float outB3 = gainB3 * static_cast<float>(accB3[off]);
        if (addToDest) {
            yA0[j0 + off] += outA0;
            yA1[j0 + off] += outA1;
            yA2[j0 + off] += outA2;
            yA3[j0 + off] += outA3;
            yB0[j0 + off] += outB0;
            yB1[j0 + off] += outB1;
            yB2[j0 + off] += outB2;
            yB3[j0 + off] += outB3;
        }
        else {
            yA0[j0 + off] = outA0;
            yA1[j0 + off] = outA1;
            yA2[j0 + off] = outA2;
            yA3[j0 + off] = outA3;
            yB0[j0 + off] = outB0;
            yB1[j0 + off] = outB1;
            yB2[j0 + off] = outB2;
            yB3[j0 + off] = outB3;
        }
    }
#endif
}

inline void dot_i2s_i8_scale_store_block16_x4_dual_bwd_exact(
    float* __restrict x0, float* __restrict x1,
    float* __restrict x2, float* __restrict x3,
    const uint8_t* __restrict Waq2, const uint8_t* __restrict Wbq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dyA0, const int8_t* __restrict dyA1,
    const int8_t* __restrict dyA2, const int8_t* __restrict dyA3,
    const int8_t* __restrict dyB0, const int8_t* __restrict dyB1,
    const int8_t* __restrict dyB2, const int8_t* __restrict dyB3,
    std::size_t i0,
    bool addToDest,
    float gainA0, float gainA1, float gainA2, float gainA3,
    float gainB0, float gainB1, float gainB2, float gainB3)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
    __m256 b00 = _mm256_setzero_ps(), b01 = _mm256_setzero_ps();
    __m256 b10 = _mm256_setzero_ps(), b11 = _mm256_setzero_ps();
    __m256 b20 = _mm256_setzero_ps(), b21 = _mm256_setzero_ps();
    __m256 b30 = _mm256_setzero_ps(), b31 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dyA0[di] | dyA1[di] | dyA2[di] | dyA3[di] | dyB0[di] | dyB1[di] | dyB2[di] | dyB3[di])) continue;
        const uint8_t* wa = Waq2 + (((j0 + di) * stride_cols + i0) >> 2);
        const uint8_t* wb = Wbq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wa + prefetch_stride), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(wb + prefetch_stride), _MM_HINT_T0);

        const __m256 wa0 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 0, shift_vec, mask_vec, one_vec);
        const __m256 wa1 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 2, shift_vec, mask_vec, one_vec);
        const __m256 wb0 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 0, shift_vec, mask_vec, one_vec);
        const __m256 wb1 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 2, shift_vec, mask_vec, one_vec);
        if (dyA0[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dyA0[di])); a00 = _mm256_fmadd_ps(vx, wa0, a00); a01 = _mm256_fmadd_ps(vx, wa1, a01); }
        if (dyB0[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dyB0[di])); b00 = _mm256_fmadd_ps(vx, wb0, b00); b01 = _mm256_fmadd_ps(vx, wb1, b01); }
        if (dyA1[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dyA1[di])); a10 = _mm256_fmadd_ps(vx, wa0, a10); a11 = _mm256_fmadd_ps(vx, wa1, a11); }
        if (dyB1[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dyB1[di])); b10 = _mm256_fmadd_ps(vx, wb0, b10); b11 = _mm256_fmadd_ps(vx, wb1, b11); }
        if (dyA2[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dyA2[di])); a20 = _mm256_fmadd_ps(vx, wa0, a20); a21 = _mm256_fmadd_ps(vx, wa1, a21); }
        if (dyB2[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dyB2[di])); b20 = _mm256_fmadd_ps(vx, wb0, b20); b21 = _mm256_fmadd_ps(vx, wb1, b21); }
        if (dyA3[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dyA3[di])); a30 = _mm256_fmadd_ps(vx, wa0, a30); a31 = _mm256_fmadd_ps(vx, wa1, a31); }
        if (dyB3[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dyB3[di])); b30 = _mm256_fmadd_ps(vx, wb0, b30); b31 = _mm256_fmadd_ps(vx, wb1, b31); }
    }

    const __m256 ga0 = _mm256_set1_ps(gainA0), ga1 = _mm256_set1_ps(gainA1);
    const __m256 ga2 = _mm256_set1_ps(gainA2), ga3 = _mm256_set1_ps(gainA3);
    const __m256 gb0 = _mm256_set1_ps(gainB0), gb1 = _mm256_set1_ps(gainB1);
    const __m256 gb2 = _mm256_set1_ps(gainB2), gb3 = _mm256_set1_ps(gainB3);

    __m256 xv = _mm256_loadu_ps(x0 + i0 + 0);
    xv = _mm256_fmadd_ps(a00, ga0, xv);
    xv = _mm256_fmadd_ps(b00, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 8);
    xv = _mm256_fmadd_ps(a01, ga0, xv);
    xv = _mm256_fmadd_ps(b01, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 8, xv);

    xv = _mm256_loadu_ps(x1 + i0 + 0);
    xv = _mm256_fmadd_ps(a10, ga1, xv);
    xv = _mm256_fmadd_ps(b10, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 8);
    xv = _mm256_fmadd_ps(a11, ga1, xv);
    xv = _mm256_fmadd_ps(b11, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 8, xv);

    xv = _mm256_loadu_ps(x2 + i0 + 0);
    xv = _mm256_fmadd_ps(a20, ga2, xv);
    xv = _mm256_fmadd_ps(b20, gb2, xv);
    _mm256_storeu_ps(x2 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x2 + i0 + 8);
    xv = _mm256_fmadd_ps(a21, ga2, xv);
    xv = _mm256_fmadd_ps(b21, gb2, xv);
    _mm256_storeu_ps(x2 + i0 + 8, xv);

    xv = _mm256_loadu_ps(x3 + i0 + 0);
    xv = _mm256_fmadd_ps(a30, ga3, xv);
    xv = _mm256_fmadd_ps(b30, gb3, xv);
    _mm256_storeu_ps(x3 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x3 + i0 + 8);
    xv = _mm256_fmadd_ps(a31, ga3, xv);
    xv = _mm256_fmadd_ps(b31, gb3, xv);
    _mm256_storeu_ps(x3 + i0 + 8, xv);
#else
    int32_t accA0[32] = { 0 }, accA1[32] = { 0 }, accA2[32] = { 0 }, accA3[32] = { 0 };
    int32_t accB0[32] = { 0 }, accB1[32] = { 0 }, accB2[32] = { 0 }, accB3[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x4(accA0, accA1, accA2, accA3, Waq2, stride_cols, j0, hCount, dyA0, dyA1, dyA2, dyA3, i0);
    dot_i2s_i8_accum_i32_block32_x4(accB0, accB1, accB2, accB3, Wbq2, stride_cols, j0, hCount, dyB0, dyB1, dyB2, dyB3, i0);
    for (std::size_t off = 0; off < 16; ++off) {
        const float v0 = gainA0 * static_cast<float>(accA0[off]) + gainB0 * static_cast<float>(accB0[off]);
        const float v1 = gainA1 * static_cast<float>(accA1[off]) + gainB1 * static_cast<float>(accB1[off]);
        const float v2 = gainA2 * static_cast<float>(accA2[off]) + gainB2 * static_cast<float>(accB2[off]);
        const float v3 = gainA3 * static_cast<float>(accA3[off]) + gainB3 * static_cast<float>(accB3[off]);
        x0[i0 + off] += v0;
        x1[i0 + off] += v1;
        x2[i0 + off] += v2;
        x3[i0 + off] += v3;
    }
#endif
}

inline void dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact(
    float* __restrict x0, float* __restrict x1,
    const uint8_t* __restrict Waq2, const uint8_t* __restrict Wbq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dyA0, const int8_t* __restrict dyA1,
    const int8_t* __restrict dyB0, const int8_t* __restrict dyB1,
    std::size_t i0,
    bool addToDest,
    float gainA0, float gainA1,
    float gainB0, float gainB1)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 b00 = _mm256_setzero_ps(), b01 = _mm256_setzero_ps();
    __m256 b10 = _mm256_setzero_ps(), b11 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dyA0[di] | dyA1[di] | dyB0[di] | dyB1[di])) continue;
        const uint8_t* wa = Waq2 + (((j0 + di) * stride_cols + i0) >> 2);
        const uint8_t* wb = Wbq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wa + prefetch_stride), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(wb + prefetch_stride), _MM_HINT_T0);

        const __m256 wa0 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 0, shift_vec, mask_vec, one_vec);
        const __m256 wa1 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 2, shift_vec, mask_vec, one_vec);
        const __m256 wb0 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 0, shift_vec, mask_vec, one_vec);
        const __m256 wb1 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 2, shift_vec, mask_vec, one_vec);
        const __m256 vxA0 = _mm256_set1_ps(static_cast<float>(dyA0[di]));
        const __m256 vxB0 = _mm256_set1_ps(static_cast<float>(dyB0[di]));
        const __m256 vxA1 = _mm256_set1_ps(static_cast<float>(dyA1[di]));
        const __m256 vxB1 = _mm256_set1_ps(static_cast<float>(dyB1[di]));
        a00 = _mm256_fmadd_ps(vxA0, wa0, a00);
        a01 = _mm256_fmadd_ps(vxA0, wa1, a01);
        b00 = _mm256_fmadd_ps(vxB0, wb0, b00);
        b01 = _mm256_fmadd_ps(vxB0, wb1, b01);
        a10 = _mm256_fmadd_ps(vxA1, wa0, a10);
        a11 = _mm256_fmadd_ps(vxA1, wa1, a11);
        b10 = _mm256_fmadd_ps(vxB1, wb0, b10);
        b11 = _mm256_fmadd_ps(vxB1, wb1, b11);
    }

    const __m256 ga0 = _mm256_set1_ps(gainA0), ga1 = _mm256_set1_ps(gainA1);
    const __m256 gb0 = _mm256_set1_ps(gainB0), gb1 = _mm256_set1_ps(gainB1);
    __m256 xv = _mm256_loadu_ps(x0 + i0 + 0);
    xv = _mm256_fmadd_ps(a00, ga0, xv);
    xv = _mm256_fmadd_ps(b00, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 8);
    xv = _mm256_fmadd_ps(a01, ga0, xv);
    xv = _mm256_fmadd_ps(b01, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 8, xv);

    xv = _mm256_loadu_ps(x1 + i0 + 0);
    xv = _mm256_fmadd_ps(a10, ga1, xv);
    xv = _mm256_fmadd_ps(b10, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 8);
    xv = _mm256_fmadd_ps(a11, ga1, xv);
    xv = _mm256_fmadd_ps(b11, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
#else
    int32_t accA0[32] = { 0 }, accA1[32] = { 0 };
    int32_t accB0[32] = { 0 }, accB1[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, Waq2, stride_cols, j0, hCount, dyA0, dyA1, i0);
    dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, Wbq2, stride_cols, j0, hCount, dyB0, dyB1, i0);
    for (std::size_t off = 0; off < 16; ++off) {
        const float v0 = gainA0 * static_cast<float>(accA0[off]) + gainB0 * static_cast<float>(accB0[off]);
        const float v1 = gainA1 * static_cast<float>(accA1[off]) + gainB1 * static_cast<float>(accB1[off]);
        x0[i0 + off] += v0;
        x1[i0 + off] += v1;
    }
#endif
}

inline void dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_dense(
    float* __restrict x0, float* __restrict x1,
    const uint8_t* __restrict Waq2, const uint8_t* __restrict Wbq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dyA0, const int8_t* __restrict dyA1,
    const int8_t* __restrict dyB0, const int8_t* __restrict dyB1,
    std::size_t i0,
    bool addToDest,
    float gainA0, float gainA1,
    float gainB0, float gainB1)
{
    (void)addToDest;
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 b00 = _mm256_setzero_ps(), b01 = _mm256_setzero_ps();
    __m256 b10 = _mm256_setzero_ps(), b11 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        const uint8_t* wa = Waq2 + (((j0 + di) * stride_cols + i0) >> 2);
        const uint8_t* wb = Wbq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wa + prefetch_stride), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(wb + prefetch_stride), _MM_HINT_T0);

        const __m256 wa0 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 0, shift_vec, mask_vec, one_vec);
        const __m256 wa1 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 2, shift_vec, mask_vec, one_vec);
        const __m256 wb0 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 0, shift_vec, mask_vec, one_vec);
        const __m256 wb1 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 2, shift_vec, mask_vec, one_vec);
        const __m256 vxA0 = _mm256_set1_ps(static_cast<float>(dyA0[di]));
        const __m256 vxB0 = _mm256_set1_ps(static_cast<float>(dyB0[di]));
        const __m256 vxA1 = _mm256_set1_ps(static_cast<float>(dyA1[di]));
        const __m256 vxB1 = _mm256_set1_ps(static_cast<float>(dyB1[di]));
        a00 = _mm256_fmadd_ps(vxA0, wa0, a00);
        a01 = _mm256_fmadd_ps(vxA0, wa1, a01);
        b00 = _mm256_fmadd_ps(vxB0, wb0, b00);
        b01 = _mm256_fmadd_ps(vxB0, wb1, b01);
        a10 = _mm256_fmadd_ps(vxA1, wa0, a10);
        a11 = _mm256_fmadd_ps(vxA1, wa1, a11);
        b10 = _mm256_fmadd_ps(vxB1, wb0, b10);
        b11 = _mm256_fmadd_ps(vxB1, wb1, b11);
    }

    const __m256 ga0 = _mm256_set1_ps(gainA0), ga1 = _mm256_set1_ps(gainA1);
    const __m256 gb0 = _mm256_set1_ps(gainB0), gb1 = _mm256_set1_ps(gainB1);
    __m256 xv = _mm256_loadu_ps(x0 + i0 + 0);
    xv = _mm256_fmadd_ps(a00, ga0, xv);
    xv = _mm256_fmadd_ps(b00, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 8);
    xv = _mm256_fmadd_ps(a01, ga0, xv);
    xv = _mm256_fmadd_ps(b01, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 8, xv);

    xv = _mm256_loadu_ps(x1 + i0 + 0);
    xv = _mm256_fmadd_ps(a10, ga1, xv);
    xv = _mm256_fmadd_ps(b10, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 8);
    xv = _mm256_fmadd_ps(a11, ga1, xv);
    xv = _mm256_fmadd_ps(b11, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
#else
    dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact(
        x0, x1, Waq2, Wbq2, stride_cols, j0, hCount,
        dyA0, dyA1, dyB0, dyB1, i0, addToDest,
        gainA0, gainA1, gainB0, gainB1);
#endif
}

inline void dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite(
    float* __restrict x0, float* __restrict x1,
    const uint8_t* __restrict Waq2, const uint8_t* __restrict Wbq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dyA0, const int8_t* __restrict dyA1,
    const int8_t* __restrict dyB0, const int8_t* __restrict dyB1,
    std::size_t i0,
    float gainA0, float gainA1,
    float gainB0, float gainB1)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 b00 = _mm256_setzero_ps(), b01 = _mm256_setzero_ps();
    __m256 b10 = _mm256_setzero_ps(), b11 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dyA0[di] | dyA1[di] | dyB0[di] | dyB1[di])) continue;
        const uint8_t* wa = Waq2 + (((j0 + di) * stride_cols + i0) >> 2);
        const uint8_t* wb = Wbq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wa + prefetch_stride), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(wb + prefetch_stride), _MM_HINT_T0);

        const __m256 wa0 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 0, shift_vec, mask_vec, one_vec);
        const __m256 wa1 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 2, shift_vec, mask_vec, one_vec);
        const __m256 wb0 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 0, shift_vec, mask_vec, one_vec);
        const __m256 wb1 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 2, shift_vec, mask_vec, one_vec);
        const __m256 vxA0 = _mm256_set1_ps(static_cast<float>(dyA0[di]));
        const __m256 vxB0 = _mm256_set1_ps(static_cast<float>(dyB0[di]));
        const __m256 vxA1 = _mm256_set1_ps(static_cast<float>(dyA1[di]));
        const __m256 vxB1 = _mm256_set1_ps(static_cast<float>(dyB1[di]));
        a00 = _mm256_fmadd_ps(vxA0, wa0, a00);
        a01 = _mm256_fmadd_ps(vxA0, wa1, a01);
        b00 = _mm256_fmadd_ps(vxB0, wb0, b00);
        b01 = _mm256_fmadd_ps(vxB0, wb1, b01);
        a10 = _mm256_fmadd_ps(vxA1, wa0, a10);
        a11 = _mm256_fmadd_ps(vxA1, wa1, a11);
        b10 = _mm256_fmadd_ps(vxB1, wb0, b10);
        b11 = _mm256_fmadd_ps(vxB1, wb1, b11);
    }

    const __m256 ga0 = _mm256_set1_ps(gainA0), ga1 = _mm256_set1_ps(gainA1);
    const __m256 gb0 = _mm256_set1_ps(gainB0), gb1 = _mm256_set1_ps(gainB1);
    __m256 xv = _mm256_mul_ps(a00, ga0);
    xv = _mm256_fmadd_ps(b00, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_mul_ps(a01, ga0);
    xv = _mm256_fmadd_ps(b01, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 8, xv);

    xv = _mm256_mul_ps(a10, ga1);
    xv = _mm256_fmadd_ps(b10, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_mul_ps(a11, ga1);
    xv = _mm256_fmadd_ps(b11, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
#else
    int32_t accA0[32] = { 0 }, accA1[32] = { 0 };
    int32_t accB0[32] = { 0 }, accB1[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x2(accA0, accA1, Waq2, stride_cols, j0, hCount, dyA0, dyA1, i0);
    dot_i2s_i8_accum_i32_block32_x2(accB0, accB1, Wbq2, stride_cols, j0, hCount, dyB0, dyB1, i0);
    for (std::size_t off = 0; off < 16; ++off) {
        x0[i0 + off] = gainA0 * static_cast<float>(accA0[off]) + gainB0 * static_cast<float>(accB0[off]);
        x1[i0 + off] = gainA1 * static_cast<float>(accA1[off]) + gainB1 * static_cast<float>(accB1[off]);
    }
#endif
}

inline void dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite_dense(
    float* __restrict x0, float* __restrict x1,
    const uint8_t* __restrict Waq2, const uint8_t* __restrict Wbq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dyA0, const int8_t* __restrict dyA1,
    const int8_t* __restrict dyB0, const int8_t* __restrict dyB1,
    std::size_t i0,
    float gainA0, float gainA1,
    float gainB0, float gainB1)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 b00 = _mm256_setzero_ps(), b01 = _mm256_setzero_ps();
    __m256 b10 = _mm256_setzero_ps(), b11 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        const uint8_t* wa = Waq2 + (((j0 + di) * stride_cols + i0) >> 2);
        const uint8_t* wb = Wbq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(wa + prefetch_stride), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(wb + prefetch_stride), _MM_HINT_T0);

        const __m256 wa0 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 0, shift_vec, mask_vec, one_vec);
        const __m256 wa1 = bitnet_i2s_detail::decode_2bytes_to_ps(wa + 2, shift_vec, mask_vec, one_vec);
        const __m256 wb0 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 0, shift_vec, mask_vec, one_vec);
        const __m256 wb1 = bitnet_i2s_detail::decode_2bytes_to_ps(wb + 2, shift_vec, mask_vec, one_vec);
        const __m256 vxA0 = _mm256_set1_ps(static_cast<float>(dyA0[di]));
        const __m256 vxB0 = _mm256_set1_ps(static_cast<float>(dyB0[di]));
        const __m256 vxA1 = _mm256_set1_ps(static_cast<float>(dyA1[di]));
        const __m256 vxB1 = _mm256_set1_ps(static_cast<float>(dyB1[di]));
        a00 = _mm256_fmadd_ps(vxA0, wa0, a00);
        a01 = _mm256_fmadd_ps(vxA0, wa1, a01);
        b00 = _mm256_fmadd_ps(vxB0, wb0, b00);
        b01 = _mm256_fmadd_ps(vxB0, wb1, b01);
        a10 = _mm256_fmadd_ps(vxA1, wa0, a10);
        a11 = _mm256_fmadd_ps(vxA1, wa1, a11);
        b10 = _mm256_fmadd_ps(vxB1, wb0, b10);
        b11 = _mm256_fmadd_ps(vxB1, wb1, b11);
    }

    const __m256 ga0 = _mm256_set1_ps(gainA0), ga1 = _mm256_set1_ps(gainA1);
    const __m256 gb0 = _mm256_set1_ps(gainB0), gb1 = _mm256_set1_ps(gainB1);
    __m256 xv = _mm256_mul_ps(a00, ga0);
    xv = _mm256_fmadd_ps(b00, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_mul_ps(a01, ga0);
    xv = _mm256_fmadd_ps(b01, gb0, xv);
    _mm256_storeu_ps(x0 + i0 + 8, xv);

    xv = _mm256_mul_ps(a10, ga1);
    xv = _mm256_fmadd_ps(b10, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_mul_ps(a11, ga1);
    xv = _mm256_fmadd_ps(b11, gb1, xv);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
#else
    dot_i2s_i8_scale_store_block16_x2_dual_bwd_exact_overwrite(
        x0, x1, Waq2, Wbq2, stride_cols, j0, hCount,
        dyA0, dyA1, dyB0, dyB1, i0,
        gainA0, gainA1, gainB0, gainB1);
#endif
}

inline void dot_i2s_i8_scale_store_block16_x4_bwd_exact(
    float* __restrict x0, float* __restrict x1,
    float* __restrict x2, float* __restrict x3,
    const uint8_t* __restrict Wq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dy0, const int8_t* __restrict dy1,
    const int8_t* __restrict dy2, const int8_t* __restrict dy3,
    std::size_t i0,
    bool addToDest,
    float gain0, float gain1, float gain2, float gain3)
{
    (void)addToDest;
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dy0[di] | dy1[di] | dy2[di] | dy3[di])) continue;
        const uint8_t* w = Wq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(w + prefetch_stride), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 2, shift_vec, mask_vec, one_vec);
        if (dy0[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dy0[di])); a00 = _mm256_fmadd_ps(vx, w0, a00); a01 = _mm256_fmadd_ps(vx, w1, a01); }
        if (dy1[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dy1[di])); a10 = _mm256_fmadd_ps(vx, w0, a10); a11 = _mm256_fmadd_ps(vx, w1, a11); }
        if (dy2[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dy2[di])); a20 = _mm256_fmadd_ps(vx, w0, a20); a21 = _mm256_fmadd_ps(vx, w1, a21); }
        if (dy3[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dy3[di])); a30 = _mm256_fmadd_ps(vx, w0, a30); a31 = _mm256_fmadd_ps(vx, w1, a31); }
    }

    const __m256 g0 = _mm256_set1_ps(gain0), g1 = _mm256_set1_ps(gain1);
    const __m256 g2 = _mm256_set1_ps(gain2), g3 = _mm256_set1_ps(gain3);

    __m256 xv = _mm256_loadu_ps(x0 + i0 + 0);
    xv = _mm256_fmadd_ps(a00, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 8);
    xv = _mm256_fmadd_ps(a01, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 8, xv);

    xv = _mm256_loadu_ps(x1 + i0 + 0);
    xv = _mm256_fmadd_ps(a10, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 8);
    xv = _mm256_fmadd_ps(a11, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 8, xv);

    xv = _mm256_loadu_ps(x2 + i0 + 0);
    xv = _mm256_fmadd_ps(a20, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x2 + i0 + 8);
    xv = _mm256_fmadd_ps(a21, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 8, xv);

    xv = _mm256_loadu_ps(x3 + i0 + 0);
    xv = _mm256_fmadd_ps(a30, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x3 + i0 + 8);
    xv = _mm256_fmadd_ps(a31, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 8, xv);
#else
    int32_t acc0[32] = { 0 }, acc1[32] = { 0 }, acc2[32] = { 0 }, acc3[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x4(acc0, acc1, acc2, acc3, Wq2, stride_cols, j0, hCount, dy0, dy1, dy2, dy3, i0);
    for (std::size_t off = 0; off < 16; ++off) {
        const float v0 = gain0 * static_cast<float>(acc0[off]);
        const float v1 = gain1 * static_cast<float>(acc1[off]);
        const float v2 = gain2 * static_cast<float>(acc2[off]);
        const float v3 = gain3 * static_cast<float>(acc3[off]);
        x0[i0 + off] += v0;
        x1[i0 + off] += v1;
        x2[i0 + off] += v2;
        x3[i0 + off] += v3;
    }
#endif
}

inline void dot_i2s_i8_scale_store_block16_x4_bwd_exact_overwrite(
    float* __restrict x0, float* __restrict x1,
    float* __restrict x2, float* __restrict x3,
    const uint8_t* __restrict Wq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dy0, const int8_t* __restrict dy1,
    const int8_t* __restrict dy2, const int8_t* __restrict dy3,
    std::size_t i0,
    float gain0, float gain1, float gain2, float gain3)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dy0[di] | dy1[di] | dy2[di] | dy3[di])) continue;
        const uint8_t* w = Wq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(w + prefetch_stride), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 2, shift_vec, mask_vec, one_vec);
        if (dy0[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dy0[di])); a00 = _mm256_fmadd_ps(vx, w0, a00); a01 = _mm256_fmadd_ps(vx, w1, a01); }
        if (dy1[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dy1[di])); a10 = _mm256_fmadd_ps(vx, w0, a10); a11 = _mm256_fmadd_ps(vx, w1, a11); }
        if (dy2[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dy2[di])); a20 = _mm256_fmadd_ps(vx, w0, a20); a21 = _mm256_fmadd_ps(vx, w1, a21); }
        if (dy3[di]) { const __m256 vx = _mm256_set1_ps(static_cast<float>(dy3[di])); a30 = _mm256_fmadd_ps(vx, w0, a30); a31 = _mm256_fmadd_ps(vx, w1, a31); }
    }

    const __m256 g0 = _mm256_set1_ps(gain0), g1 = _mm256_set1_ps(gain1);
    const __m256 g2 = _mm256_set1_ps(gain2), g3 = _mm256_set1_ps(gain3);

    __m256 xv = _mm256_mul_ps(a00, g0);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_mul_ps(a01, g0);
    _mm256_storeu_ps(x0 + i0 + 8, xv);

    xv = _mm256_mul_ps(a10, g1);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_mul_ps(a11, g1);
    _mm256_storeu_ps(x1 + i0 + 8, xv);

    xv = _mm256_mul_ps(a20, g2);
    _mm256_storeu_ps(x2 + i0 + 0, xv);
    xv = _mm256_mul_ps(a21, g2);
    _mm256_storeu_ps(x2 + i0 + 8, xv);

    xv = _mm256_mul_ps(a30, g3);
    _mm256_storeu_ps(x3 + i0 + 0, xv);
    xv = _mm256_mul_ps(a31, g3);
    _mm256_storeu_ps(x3 + i0 + 8, xv);
#else
    int32_t acc0[32] = { 0 }, acc1[32] = { 0 }, acc2[32] = { 0 }, acc3[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x4(acc0, acc1, acc2, acc3, Wq2, stride_cols, j0, hCount, dy0, dy1, dy2, dy3, i0);
    for (std::size_t off = 0; off < 16; ++off) {
        x0[i0 + off] = gain0 * static_cast<float>(acc0[off]);
        x1[i0 + off] = gain1 * static_cast<float>(acc1[off]);
        x2[i0 + off] = gain2 * static_cast<float>(acc2[off]);
        x3[i0 + off] = gain3 * static_cast<float>(acc3[off]);
    }
#endif
}

inline void dot_i2s_i8_scale_store_block32_x4_bwd_exact(
    float* __restrict x0, float* __restrict x1,
    float* __restrict x2, float* __restrict x3,
    const uint8_t* __restrict Wq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dy0, const int8_t* __restrict dy1,
    const int8_t* __restrict dy2, const int8_t* __restrict dy3,
    std::size_t i0,
    bool addToDest,
    float gain0, float gain1, float gain2, float gain3)
{
    (void)addToDest;
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps(), a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps(), a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps(), a22 = _mm256_setzero_ps(), a23 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps(), a32 = _mm256_setzero_ps(), a33 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dy0[di] | dy1[di] | dy2[di] | dy3[di])) continue;
        const uint8_t* w = Wq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(w + prefetch_stride), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 2, shift_vec, mask_vec, one_vec);
        const __m256 w2 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 4, shift_vec, mask_vec, one_vec);
        const __m256 w3 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 6, shift_vec, mask_vec, one_vec);

        if (dy0[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy0[di]));
            a00 = _mm256_fmadd_ps(vx, w0, a00); a01 = _mm256_fmadd_ps(vx, w1, a01);
            a02 = _mm256_fmadd_ps(vx, w2, a02); a03 = _mm256_fmadd_ps(vx, w3, a03);
        }
        if (dy1[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy1[di]));
            a10 = _mm256_fmadd_ps(vx, w0, a10); a11 = _mm256_fmadd_ps(vx, w1, a11);
            a12 = _mm256_fmadd_ps(vx, w2, a12); a13 = _mm256_fmadd_ps(vx, w3, a13);
        }
        if (dy2[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy2[di]));
            a20 = _mm256_fmadd_ps(vx, w0, a20); a21 = _mm256_fmadd_ps(vx, w1, a21);
            a22 = _mm256_fmadd_ps(vx, w2, a22); a23 = _mm256_fmadd_ps(vx, w3, a23);
        }
        if (dy3[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy3[di]));
            a30 = _mm256_fmadd_ps(vx, w0, a30); a31 = _mm256_fmadd_ps(vx, w1, a31);
            a32 = _mm256_fmadd_ps(vx, w2, a32); a33 = _mm256_fmadd_ps(vx, w3, a33);
        }
    }

    const __m256 g0 = _mm256_set1_ps(gain0), g1 = _mm256_set1_ps(gain1);
    const __m256 g2 = _mm256_set1_ps(gain2), g3 = _mm256_set1_ps(gain3);

    __m256 xv = _mm256_loadu_ps(x0 + i0 + 0);
    xv = _mm256_fmadd_ps(a00, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 8);
    xv = _mm256_fmadd_ps(a01, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 8, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 16);
    xv = _mm256_fmadd_ps(a02, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 16, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 24);
    xv = _mm256_fmadd_ps(a03, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 24, xv);

    xv = _mm256_loadu_ps(x1 + i0 + 0);
    xv = _mm256_fmadd_ps(a10, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 8);
    xv = _mm256_fmadd_ps(a11, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 16);
    xv = _mm256_fmadd_ps(a12, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 16, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 24);
    xv = _mm256_fmadd_ps(a13, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 24, xv);

    xv = _mm256_loadu_ps(x2 + i0 + 0);
    xv = _mm256_fmadd_ps(a20, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x2 + i0 + 8);
    xv = _mm256_fmadd_ps(a21, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 8, xv);
    xv = _mm256_loadu_ps(x2 + i0 + 16);
    xv = _mm256_fmadd_ps(a22, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 16, xv);
    xv = _mm256_loadu_ps(x2 + i0 + 24);
    xv = _mm256_fmadd_ps(a23, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 24, xv);

    xv = _mm256_loadu_ps(x3 + i0 + 0);
    xv = _mm256_fmadd_ps(a30, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x3 + i0 + 8);
    xv = _mm256_fmadd_ps(a31, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 8, xv);
    xv = _mm256_loadu_ps(x3 + i0 + 16);
    xv = _mm256_fmadd_ps(a32, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 16, xv);
    xv = _mm256_loadu_ps(x3 + i0 + 24);
    xv = _mm256_fmadd_ps(a33, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 24, xv);
#else
    int32_t acc0[32] = { 0 }, acc1[32] = { 0 }, acc2[32] = { 0 }, acc3[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x4(acc0, acc1, acc2, acc3, Wq2, stride_cols, j0, hCount, dy0, dy1, dy2, dy3, i0);
    accum_bwd_block32_gain_to_x_pair(x0, x1, acc0, acc1, i0, gain0, gain1, addToDest);
    accum_bwd_block32_gain_to_x_pair(x2, x3, acc2, acc3, i0, gain2, gain3, addToDest);
#endif
}

inline void dot_ternary_i8_scale_store_block32_x4_bwd_exact(
    float* __restrict x0, float* __restrict x1,
    float* __restrict x2, float* __restrict x3,
    const int8_t* __restrict Wt8,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dy0, const int8_t* __restrict dy1,
    const int8_t* __restrict dy2, const int8_t* __restrict dy3,
    std::size_t i0,
    float gain0, float gain1, float gain2, float gain3)
{
#if defined(__AVX2__)
    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps(), a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps(), a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps(), a22 = _mm256_setzero_ps(), a23 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps(), a32 = _mm256_setzero_ps(), a33 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dy0[di] | dy1[di] | dy2[di] | dy3[di])) continue;
        const int8_t* w = Wt8 + (j0 + di) * stride_cols + i0;
        _mm_prefetch(reinterpret_cast<const char*>(w + stride_cols), _MM_HINT_T0);

        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 0))));
        const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 8))));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 16))));
        const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 24))));

        if (dy0[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy0[di]));
            a00 = _mm256_fmadd_ps(vx, w0, a00); a01 = _mm256_fmadd_ps(vx, w1, a01);
            a02 = _mm256_fmadd_ps(vx, w2, a02); a03 = _mm256_fmadd_ps(vx, w3, a03);
        }
        if (dy1[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy1[di]));
            a10 = _mm256_fmadd_ps(vx, w0, a10); a11 = _mm256_fmadd_ps(vx, w1, a11);
            a12 = _mm256_fmadd_ps(vx, w2, a12); a13 = _mm256_fmadd_ps(vx, w3, a13);
        }
        if (dy2[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy2[di]));
            a20 = _mm256_fmadd_ps(vx, w0, a20); a21 = _mm256_fmadd_ps(vx, w1, a21);
            a22 = _mm256_fmadd_ps(vx, w2, a22); a23 = _mm256_fmadd_ps(vx, w3, a23);
        }
        if (dy3[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy3[di]));
            a30 = _mm256_fmadd_ps(vx, w0, a30); a31 = _mm256_fmadd_ps(vx, w1, a31);
            a32 = _mm256_fmadd_ps(vx, w2, a32); a33 = _mm256_fmadd_ps(vx, w3, a33);
        }
    }

    const __m256 g0 = _mm256_set1_ps(gain0), g1 = _mm256_set1_ps(gain1);
    const __m256 g2 = _mm256_set1_ps(gain2), g3 = _mm256_set1_ps(gain3);

    __m256 xv = _mm256_loadu_ps(x0 + i0 + 0);
    xv = _mm256_fmadd_ps(a00, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 8);
    xv = _mm256_fmadd_ps(a01, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 8, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 16);
    xv = _mm256_fmadd_ps(a02, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 16, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 24);
    xv = _mm256_fmadd_ps(a03, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 24, xv);

    xv = _mm256_loadu_ps(x1 + i0 + 0);
    xv = _mm256_fmadd_ps(a10, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 8);
    xv = _mm256_fmadd_ps(a11, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 16);
    xv = _mm256_fmadd_ps(a12, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 16, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 24);
    xv = _mm256_fmadd_ps(a13, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 24, xv);

    xv = _mm256_loadu_ps(x2 + i0 + 0);
    xv = _mm256_fmadd_ps(a20, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x2 + i0 + 8);
    xv = _mm256_fmadd_ps(a21, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 8, xv);
    xv = _mm256_loadu_ps(x2 + i0 + 16);
    xv = _mm256_fmadd_ps(a22, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 16, xv);
    xv = _mm256_loadu_ps(x2 + i0 + 24);
    xv = _mm256_fmadd_ps(a23, g2, xv);
    _mm256_storeu_ps(x2 + i0 + 24, xv);

    xv = _mm256_loadu_ps(x3 + i0 + 0);
    xv = _mm256_fmadd_ps(a30, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x3 + i0 + 8);
    xv = _mm256_fmadd_ps(a31, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 8, xv);
    xv = _mm256_loadu_ps(x3 + i0 + 16);
    xv = _mm256_fmadd_ps(a32, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 16, xv);
    xv = _mm256_loadu_ps(x3 + i0 + 24);
    xv = _mm256_fmadd_ps(a33, g3, xv);
    _mm256_storeu_ps(x3 + i0 + 24, xv);
#else
    (void)Wt8;
    (void)stride_cols;
    (void)j0;
    (void)hCount;
    (void)dy0;
    (void)dy1;
    (void)dy2;
    (void)dy3;
    (void)i0;
    (void)gain0;
    (void)gain1;
    (void)gain2;
    (void)gain3;
#endif
}

inline void dot_ternary_i8_scale_store_block32_x4_bwd_exact_overwrite(
    float* __restrict x0, float* __restrict x1,
    float* __restrict x2, float* __restrict x3,
    const int8_t* __restrict Wt8,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dy0, const int8_t* __restrict dy1,
    const int8_t* __restrict dy2, const int8_t* __restrict dy3,
    std::size_t i0,
    float gain0, float gain1, float gain2, float gain3)
{
#if defined(__AVX2__)
    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps(), a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps(), a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps(), a22 = _mm256_setzero_ps(), a23 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps(), a32 = _mm256_setzero_ps(), a33 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dy0[di] | dy1[di] | dy2[di] | dy3[di])) continue;
        const int8_t* w = Wt8 + (j0 + di) * stride_cols + i0;
        _mm_prefetch(reinterpret_cast<const char*>(w + stride_cols), _MM_HINT_T0);

        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 0))));
        const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 8))));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 16))));
        const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 24))));

        if (dy0[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy0[di]));
            a00 = _mm256_fmadd_ps(vx, w0, a00); a01 = _mm256_fmadd_ps(vx, w1, a01);
            a02 = _mm256_fmadd_ps(vx, w2, a02); a03 = _mm256_fmadd_ps(vx, w3, a03);
        }
        if (dy1[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy1[di]));
            a10 = _mm256_fmadd_ps(vx, w0, a10); a11 = _mm256_fmadd_ps(vx, w1, a11);
            a12 = _mm256_fmadd_ps(vx, w2, a12); a13 = _mm256_fmadd_ps(vx, w3, a13);
        }
        if (dy2[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy2[di]));
            a20 = _mm256_fmadd_ps(vx, w0, a20); a21 = _mm256_fmadd_ps(vx, w1, a21);
            a22 = _mm256_fmadd_ps(vx, w2, a22); a23 = _mm256_fmadd_ps(vx, w3, a23);
        }
        if (dy3[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy3[di]));
            a30 = _mm256_fmadd_ps(vx, w0, a30); a31 = _mm256_fmadd_ps(vx, w1, a31);
            a32 = _mm256_fmadd_ps(vx, w2, a32); a33 = _mm256_fmadd_ps(vx, w3, a33);
        }
    }

    const __m256 g0 = _mm256_set1_ps(gain0), g1 = _mm256_set1_ps(gain1);
    const __m256 g2 = _mm256_set1_ps(gain2), g3 = _mm256_set1_ps(gain3);

    __m256 xv = _mm256_mul_ps(a00, g0);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_mul_ps(a01, g0);
    _mm256_storeu_ps(x0 + i0 + 8, xv);
    xv = _mm256_mul_ps(a02, g0);
    _mm256_storeu_ps(x0 + i0 + 16, xv);
    xv = _mm256_mul_ps(a03, g0);
    _mm256_storeu_ps(x0 + i0 + 24, xv);

    xv = _mm256_mul_ps(a10, g1);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_mul_ps(a11, g1);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
    xv = _mm256_mul_ps(a12, g1);
    _mm256_storeu_ps(x1 + i0 + 16, xv);
    xv = _mm256_mul_ps(a13, g1);
    _mm256_storeu_ps(x1 + i0 + 24, xv);

    xv = _mm256_mul_ps(a20, g2);
    _mm256_storeu_ps(x2 + i0 + 0, xv);
    xv = _mm256_mul_ps(a21, g2);
    _mm256_storeu_ps(x2 + i0 + 8, xv);
    xv = _mm256_mul_ps(a22, g2);
    _mm256_storeu_ps(x2 + i0 + 16, xv);
    xv = _mm256_mul_ps(a23, g2);
    _mm256_storeu_ps(x2 + i0 + 24, xv);

    xv = _mm256_mul_ps(a30, g3);
    _mm256_storeu_ps(x3 + i0 + 0, xv);
    xv = _mm256_mul_ps(a31, g3);
    _mm256_storeu_ps(x3 + i0 + 8, xv);
    xv = _mm256_mul_ps(a32, g3);
    _mm256_storeu_ps(x3 + i0 + 16, xv);
    xv = _mm256_mul_ps(a33, g3);
    _mm256_storeu_ps(x3 + i0 + 24, xv);
#else
    (void)Wt8;
    (void)stride_cols;
    (void)j0;
    (void)hCount;
    (void)dy0;
    (void)dy1;
    (void)dy2;
    (void)dy3;
    (void)i0;
    (void)gain0;
    (void)gain1;
    (void)gain2;
    (void)gain3;
#endif
}

inline void dot_i2s_i8_scale_store_block32_x4_bwd_exact_overwrite(
    float* __restrict x0, float* __restrict x1,
    float* __restrict x2, float* __restrict x3,
    const uint8_t* __restrict Wq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dy0, const int8_t* __restrict dy1,
    const int8_t* __restrict dy2, const int8_t* __restrict dy3,
    std::size_t i0,
    float gain0, float gain1, float gain2, float gain3)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps(), a02 = _mm256_setzero_ps(), a03 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps(), a12 = _mm256_setzero_ps(), a13 = _mm256_setzero_ps();
    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps(), a22 = _mm256_setzero_ps(), a23 = _mm256_setzero_ps();
    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps(), a32 = _mm256_setzero_ps(), a33 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dy0[di] | dy1[di] | dy2[di] | dy3[di])) continue;
        const uint8_t* w = Wq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(w + prefetch_stride), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 2, shift_vec, mask_vec, one_vec);
        const __m256 w2 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 4, shift_vec, mask_vec, one_vec);
        const __m256 w3 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 6, shift_vec, mask_vec, one_vec);

        if (dy0[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy0[di]));
            a00 = _mm256_fmadd_ps(vx, w0, a00); a01 = _mm256_fmadd_ps(vx, w1, a01);
            a02 = _mm256_fmadd_ps(vx, w2, a02); a03 = _mm256_fmadd_ps(vx, w3, a03);
        }
        if (dy1[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy1[di]));
            a10 = _mm256_fmadd_ps(vx, w0, a10); a11 = _mm256_fmadd_ps(vx, w1, a11);
            a12 = _mm256_fmadd_ps(vx, w2, a12); a13 = _mm256_fmadd_ps(vx, w3, a13);
        }
        if (dy2[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy2[di]));
            a20 = _mm256_fmadd_ps(vx, w0, a20); a21 = _mm256_fmadd_ps(vx, w1, a21);
            a22 = _mm256_fmadd_ps(vx, w2, a22); a23 = _mm256_fmadd_ps(vx, w3, a23);
        }
        if (dy3[di]) {
            const __m256 vx = _mm256_set1_ps(static_cast<float>(dy3[di]));
            a30 = _mm256_fmadd_ps(vx, w0, a30); a31 = _mm256_fmadd_ps(vx, w1, a31);
            a32 = _mm256_fmadd_ps(vx, w2, a32); a33 = _mm256_fmadd_ps(vx, w3, a33);
        }
    }

    const __m256 g0 = _mm256_set1_ps(gain0), g1 = _mm256_set1_ps(gain1);
    const __m256 g2 = _mm256_set1_ps(gain2), g3 = _mm256_set1_ps(gain3);

    __m256 xv = _mm256_mul_ps(a00, g0);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_mul_ps(a01, g0);
    _mm256_storeu_ps(x0 + i0 + 8, xv);
    xv = _mm256_mul_ps(a02, g0);
    _mm256_storeu_ps(x0 + i0 + 16, xv);
    xv = _mm256_mul_ps(a03, g0);
    _mm256_storeu_ps(x0 + i0 + 24, xv);

    xv = _mm256_mul_ps(a10, g1);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_mul_ps(a11, g1);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
    xv = _mm256_mul_ps(a12, g1);
    _mm256_storeu_ps(x1 + i0 + 16, xv);
    xv = _mm256_mul_ps(a13, g1);
    _mm256_storeu_ps(x1 + i0 + 24, xv);

    xv = _mm256_mul_ps(a20, g2);
    _mm256_storeu_ps(x2 + i0 + 0, xv);
    xv = _mm256_mul_ps(a21, g2);
    _mm256_storeu_ps(x2 + i0 + 8, xv);
    xv = _mm256_mul_ps(a22, g2);
    _mm256_storeu_ps(x2 + i0 + 16, xv);
    xv = _mm256_mul_ps(a23, g2);
    _mm256_storeu_ps(x2 + i0 + 24, xv);

    xv = _mm256_mul_ps(a30, g3);
    _mm256_storeu_ps(x3 + i0 + 0, xv);
    xv = _mm256_mul_ps(a31, g3);
    _mm256_storeu_ps(x3 + i0 + 8, xv);
    xv = _mm256_mul_ps(a32, g3);
    _mm256_storeu_ps(x3 + i0 + 16, xv);
    xv = _mm256_mul_ps(a33, g3);
    _mm256_storeu_ps(x3 + i0 + 24, xv);
#else
    int32_t acc0[32] = { 0 }, acc1[32] = { 0 }, acc2[32] = { 0 }, acc3[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x4(acc0, acc1, acc2, acc3, Wq2, stride_cols, j0, hCount, dy0, dy1, dy2, dy3, i0);
    accum_bwd_block32_gain_to_x_pair(x0, x1, acc0, acc1, i0, gain0, gain1, false);
    accum_bwd_block32_gain_to_x_pair(x2, x3, acc2, acc3, i0, gain2, gain3, false);
#endif
}

inline void dot_i2s_i8_scale_store_block16_x2_bwd_exact(
    float* __restrict x0, float* __restrict x1,
    const uint8_t* __restrict Wq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dy0, const int8_t* __restrict dy1,
    std::size_t i0,
    bool addToDest,
    float gain0, float gain1)
{
    (void)addToDest;
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dy0[di] | dy1[di])) continue;
        const uint8_t* w = Wq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(w + prefetch_stride), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 2, shift_vec, mask_vec, one_vec);
        const __m256 vx0 = _mm256_set1_ps(static_cast<float>(dy0[di]));
        const __m256 vx1 = _mm256_set1_ps(static_cast<float>(dy1[di]));
        a00 = _mm256_fmadd_ps(vx0, w0, a00);
        a01 = _mm256_fmadd_ps(vx0, w1, a01);
        a10 = _mm256_fmadd_ps(vx1, w0, a10);
        a11 = _mm256_fmadd_ps(vx1, w1, a11);
    }

    const __m256 g0 = _mm256_set1_ps(gain0), g1 = _mm256_set1_ps(gain1);
    __m256 xv = _mm256_loadu_ps(x0 + i0 + 0);
    xv = _mm256_fmadd_ps(a00, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x0 + i0 + 8);
    xv = _mm256_fmadd_ps(a01, g0, xv);
    _mm256_storeu_ps(x0 + i0 + 8, xv);

    xv = _mm256_loadu_ps(x1 + i0 + 0);
    xv = _mm256_fmadd_ps(a10, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_loadu_ps(x1 + i0 + 8);
    xv = _mm256_fmadd_ps(a11, g1, xv);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
#else
    int32_t acc0[32] = { 0 }, acc1[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x2(acc0, acc1, Wq2, stride_cols, j0, hCount, dy0, dy1, i0);
    for (std::size_t off = 0; off < 16; ++off) {
        const float v0 = gain0 * static_cast<float>(acc0[off]);
        const float v1 = gain1 * static_cast<float>(acc1[off]);
        x0[i0 + off] += v0;
        x1[i0 + off] += v1;
    }
#endif
}

inline void dot_i2s_i8_scale_store_block16_x2_bwd_exact_overwrite(
    float* __restrict x0, float* __restrict x1,
    const uint8_t* __restrict Wq2,
    std::size_t stride_cols, std::size_t j0, std::size_t hCount,
    const int8_t* __restrict dy0, const int8_t* __restrict dy1,
    std::size_t i0,
    float gain0, float gain1)
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);
    const std::size_t prefetch_stride = stride_cols;

    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();

    for (std::size_t di = 0; di < hCount; ++di) {
        if (!(dy0[di] | dy1[di])) continue;
        const uint8_t* w = Wq2 + (((j0 + di) * stride_cols + i0) >> 2);
        _mm_prefetch(reinterpret_cast<const char*>(w + prefetch_stride), _MM_HINT_T0);

        const __m256 w0 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 0, shift_vec, mask_vec, one_vec);
        const __m256 w1 = bitnet_i2s_detail::decode_2bytes_to_ps(w + 2, shift_vec, mask_vec, one_vec);
        const __m256 vx0 = _mm256_set1_ps(static_cast<float>(dy0[di]));
        const __m256 vx1 = _mm256_set1_ps(static_cast<float>(dy1[di]));
        a00 = _mm256_fmadd_ps(vx0, w0, a00);
        a01 = _mm256_fmadd_ps(vx0, w1, a01);
        a10 = _mm256_fmadd_ps(vx1, w0, a10);
        a11 = _mm256_fmadd_ps(vx1, w1, a11);
    }

    const __m256 g0 = _mm256_set1_ps(gain0), g1 = _mm256_set1_ps(gain1);
    __m256 xv = _mm256_mul_ps(a00, g0);
    _mm256_storeu_ps(x0 + i0 + 0, xv);
    xv = _mm256_mul_ps(a01, g0);
    _mm256_storeu_ps(x0 + i0 + 8, xv);

    xv = _mm256_mul_ps(a10, g1);
    _mm256_storeu_ps(x1 + i0 + 0, xv);
    xv = _mm256_mul_ps(a11, g1);
    _mm256_storeu_ps(x1 + i0 + 8, xv);
#else
    int32_t acc0[32] = { 0 }, acc1[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x2(acc0, acc1, Wq2, stride_cols, j0, hCount, dy0, dy1, i0);
    for (std::size_t off = 0; off < 16; ++off) {
        x0[i0 + off] = gain0 * static_cast<float>(acc0[off]);
        x1[i0 + off] = gain1 * static_cast<float>(acc1[off]);
    }
#endif
}
