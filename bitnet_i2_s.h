#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
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

inline float max_abs_block(
    const float* X, std::size_t H, std::size_t tStart, std::size_t tEnd, std::size_t hStart, std::size_t hEnd)
{
    float m = 0.0f;
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
                    for (std::size_t h = hStart; h < hEnd; ++h) outRow[h] = 0;
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
                    for (std::size_t h = hStart; h < hEnd; ++h) outRow[h] = 0;
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
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aA0)), gainA8, yv);
        _mm256_storeu_ps(yA + j0 + 0, yv);
        yv = _mm256_loadu_ps(yA + j0 + 8);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aA1)), gainA8, yv);
        _mm256_storeu_ps(yA + j0 + 8, yv);

        yv = _mm256_loadu_ps(yB + j0 + 0);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aB0)), gainB8, yv);
        _mm256_storeu_ps(yB + j0 + 0, yv);
        yv = _mm256_loadu_ps(yB + j0 + 8);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aB1)), gainB8, yv);
        _mm256_storeu_ps(yB + j0 + 8, yv);

        yv = _mm256_loadu_ps(yC + j0 + 0);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aC0)), gainC8, yv);
        _mm256_storeu_ps(yC + j0 + 0, yv);
        yv = _mm256_loadu_ps(yC + j0 + 8);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aC1)), gainC8, yv);
        _mm256_storeu_ps(yC + j0 + 8, yv);

        yv = _mm256_loadu_ps(yD + j0 + 0);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aD0)), gainD8, yv);
        _mm256_storeu_ps(yD + j0 + 0, yv);
        yv = _mm256_loadu_ps(yD + j0 + 8);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aD1)), gainD8, yv);
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
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aA2)), gainA8, yv);
        _mm256_storeu_ps(yA + j0 + 16, yv);
        yv = _mm256_loadu_ps(yA + j0 + 24);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aA3)), gainA8, yv);
        _mm256_storeu_ps(yA + j0 + 24, yv);

        yv = _mm256_loadu_ps(yB + j0 + 16);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aB2)), gainB8, yv);
        _mm256_storeu_ps(yB + j0 + 16, yv);
        yv = _mm256_loadu_ps(yB + j0 + 24);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aB3)), gainB8, yv);
        _mm256_storeu_ps(yB + j0 + 24, yv);

        yv = _mm256_loadu_ps(yC + j0 + 16);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aC2)), gainC8, yv);
        _mm256_storeu_ps(yC + j0 + 16, yv);
        yv = _mm256_loadu_ps(yC + j0 + 24);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aC3)), gainC8, yv);
        _mm256_storeu_ps(yC + j0 + 24, yv);

        yv = _mm256_loadu_ps(yD + j0 + 16);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aD2)), gainD8, yv);
        _mm256_storeu_ps(yD + j0 + 16, yv);
        yv = _mm256_loadu_ps(yD + j0 + 24);
        yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aD3)), gainD8, yv);
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

    __m256 yv = _mm256_loadu_ps(yA + j0 + 0);
    yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aA0)), gainA8, yv);
    _mm256_storeu_ps(yA + j0 + 0, yv);
    yv = _mm256_loadu_ps(yA + j0 + 8);
    yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aA1)), gainA8, yv);
    _mm256_storeu_ps(yA + j0 + 8, yv);

    yv = _mm256_loadu_ps(yB + j0 + 0);
    yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aB0)), gainB8, yv);
    _mm256_storeu_ps(yB + j0 + 0, yv);
    yv = _mm256_loadu_ps(yB + j0 + 8);
    yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aB1)), gainB8, yv);
    _mm256_storeu_ps(yB + j0 + 8, yv);

    yv = _mm256_loadu_ps(yC + j0 + 0);
    yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aC0)), gainC8, yv);
    _mm256_storeu_ps(yC + j0 + 0, yv);
    yv = _mm256_loadu_ps(yC + j0 + 8);
    yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aC1)), gainC8, yv);
    _mm256_storeu_ps(yC + j0 + 8, yv);

    yv = _mm256_loadu_ps(yD + j0 + 0);
    yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aD0)), gainD8, yv);
    _mm256_storeu_ps(yD + j0 + 0, yv);
    yv = _mm256_loadu_ps(yD + j0 + 8);
    yv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtps_epi32(aD1)), gainD8, yv);
    _mm256_storeu_ps(yD + j0 + 8, yv);
#else
    int32_t accA[32] = { 0 }, accB[32] = { 0 }, accC[32] = { 0 }, accD[32] = { 0 };
    dot_i2s_i8_accum_i32_block32_x4(
        accA, accB, accC, accD,
        Wq2, stride_cols, i0, hCount,
        x8A, x8B, x8C, x8D,
        (j0 & ~std::size_t(31)));
    const std::size_t base = j0 & std::size_t(16);
    for (std::size_t off = 0; off < 16; ++off) {
        yA[j0 + off] += gainA * static_cast<float>(accA[base + off]);
        yB[j0 + off] += gainB * static_cast<float>(accB[base + off]);
        yC[j0 + off] += gainC * static_cast<float>(accC[base + off]);
        yD[j0 + off] += gainD * static_cast<float>(accD[base + off]);
    }
#endif
}
