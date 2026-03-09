#include "bitnet_i2_s.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// ============================================================
// Helpers
// ============================================================

static inline float max_abs_block(
    const float* X,
    std::size_t H,
    std::size_t tStart,
    std::size_t tEnd,
    std::size_t hStart,
    std::size_t hEnd)
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

#if defined(__AVX2__)
// LUT: packed byte -> 4 decoded int8 ternary weights
static alignas(32) uint32_t g_i2s_byte_to_4i8[256];
static bool g_i2s_lut_init = false;

static void init_i2s_lut()
{
    if (g_i2s_lut_init) return;

    for (int b = 0; b < 256; ++b) {
        const uint8_t pb = static_cast<uint8_t>(b);
        const int8_t w0 = i2_code_to_ternary((pb >> 0) & 3u);
        const int8_t w1 = i2_code_to_ternary((pb >> 2) & 3u);
        const int8_t w2 = i2_code_to_ternary((pb >> 4) & 3u);
        const int8_t w3 = i2_code_to_ternary((pb >> 6) & 3u);

        uint32_t u = 0;
        std::memcpy(reinterpret_cast<uint8_t*>(&u) + 0, &w0, 1);
        std::memcpy(reinterpret_cast<uint8_t*>(&u) + 1, &w1, 1);
        std::memcpy(reinterpret_cast<uint8_t*>(&u) + 2, &w2, 1);
        std::memcpy(reinterpret_cast<uint8_t*>(&u) + 3, &w3, 1);
        g_i2s_byte_to_4i8[b] = u;
    }

    g_i2s_lut_init = true;
}
#endif

// ============================================================
// Packing / unpacking
// ============================================================

void pack_ternary_to_2bit(
    std::vector<uint8_t>& out_q2,
    const int8_t* q,
    std::size_t N)
{
    if (!q) {
        throw std::runtime_error("pack_ternary_to_2bit: q is null");
    }

    const std::size_t nBytes = (N + 3) / 4;
    out_q2.assign(nBytes, 0);

    for (std::size_t i = 0; i < N; ++i) {
        const uint8_t code = ternary_to_i2_code(q[i]);
        const std::size_t byteIdx = i >> 2;
        const std::size_t shift = (i & 3u) * 2u;
        out_q2[byteIdx] |= static_cast<uint8_t>((code & 0x3u) << shift);
    }
}

void unpack_2bit_to_ternary(
    std::vector<int8_t>& out_q,
    const uint8_t* q2,
    std::size_t N)
{
    if (!q2) {
        throw std::runtime_error("unpack_2bit_to_ternary: q2 is null");
    }

    out_q.resize(N);

    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t byteIdx = i >> 2;
        const std::size_t shift = (i & 3u) * 2u;
        const uint8_t code = static_cast<uint8_t>((q2[byteIdx] >> shift) & 0x3u);
        out_q[i] = i2_code_to_ternary(code);
    }
}

// ============================================================
// Activation quantization
// ============================================================

void quantize_x_tile_i8_tiled(
    int8_t* __restrict Xi8,
    float* __restrict xScale,
    const float* __restrict X,
    std::size_t T,
    std::size_t H,
    std::size_t t0,
    std::size_t tCount,
    std::size_t tBlock,
    std::size_t hBlock)
{
    if (!Xi8) {
        throw std::runtime_error("quantize_x_tile_i8_tiled: Xi8 null");
    }
    if (!xScale) {
        throw std::runtime_error("quantize_x_tile_i8_tiled: xScale null");
    }
    if (!X) {
        throw std::runtime_error("quantize_x_tile_i8_tiled: X null");
    }
    if (t0 + tCount > T) {
        throw std::runtime_error("quantize_x_tile_i8_tiled: tile OOB");
    }
    if (tBlock == 0 || hBlock == 0) {
        throw std::runtime_error("quantize_x_tile_i8_tiled: zero block");
    }

    const std::size_t nTBlocks = (tCount + tBlock - 1) / tBlock;
    const std::size_t nHBlocks = (H + hBlock - 1) / hBlock;

    for (std::size_t bt = 0; bt < nTBlocks; ++bt) {
        const std::size_t tStart = t0 + bt * tBlock;
        const std::size_t tEnd = std::min(t0 + tCount, tStart + tBlock);

        for (std::size_t bh = 0; bh < nHBlocks; ++bh) {
            const std::size_t hStart = bh * hBlock;
            const std::size_t hEnd = std::min(H, hStart + hBlock);

            const float m = max_abs_block(X, H, tStart, tEnd, hStart, hEnd);
            const float scale = (m > 0.0f) ? (m / 127.0f) : 1.0f;
            const float inv = 1.0f / scale;

            xScale[bt * nHBlocks + bh] = scale;

            for (std::size_t t = tStart; t < tEnd; ++t) {
                const float* row = X + t * H;
                int8_t* outRow = Xi8 + (t - t0) * H;

                if (m == 0.0f) {
                    for (std::size_t h = hStart; h < hEnd; ++h) {
                        outRow[h] = 0;
                    }
                }
                else {
                    for (std::size_t h = hStart; h < hEnd; ++h) {
                        int qi = static_cast<int>(std::lrint(row[h] * inv));
                        qi = std::max(-127, std::min(127, qi));
                        outRow[h] = static_cast<int8_t>(qi);
                    }
                }
            }
        }
    }
}

// ============================================================
// Dot micro-kernel: scalar
//
// acc32[0..31] += x8[hCount] dot Wq2[(i0:i0+hCount), (j0:j0+31)]
// Wq2 is packed row-major, 4 weights/byte
// ============================================================

static void dot_i2s_i8_accum_i32_block32_scalar(
    int32_t* __restrict acc32,
    const uint8_t* __restrict Wq2,
    std::size_t H,
    std::size_t i0,
    std::size_t hCount,
    const int8_t* __restrict x8,
    std::size_t j0)
{
    constexpr std::size_t JBLOCK = 32;
    constexpr std::size_t BYTES_PER_JBLOCK = (JBLOCK * 2) / 8; // 8 bytes

    for (std::size_t di = 0; di < hCount; ++di) {
        const int xi = static_cast<int>(x8[di]);
        if (xi == 0) continue;

        const std::size_t i = i0 + di;
        const std::size_t idx0 = i * H + j0;
        const std::size_t b0 = idx0 >> 2;

        const uint8_t* wbytes = Wq2 + b0;

        for (std::size_t jb = 0; jb < BYTES_PER_JBLOCK; ++jb) {
            const uint8_t packed = wbytes[jb];

            const int8_t w0 = i2_code_to_ternary((packed >> 0) & 3u);
            const int8_t w1 = i2_code_to_ternary((packed >> 2) & 3u);
            const int8_t w2 = i2_code_to_ternary((packed >> 4) & 3u);
            const int8_t w3 = i2_code_to_ternary((packed >> 6) & 3u);

            const std::size_t jbase = jb * 4;
            acc32[jbase + 0] += xi * static_cast<int>(w0);
            acc32[jbase + 1] += xi * static_cast<int>(w1);
            acc32[jbase + 2] += xi * static_cast<int>(w2);
            acc32[jbase + 3] += xi * static_cast<int>(w3);
        }
    }
}

#if defined(__AVX2__)
// ============================================================
// Dot micro-kernel: AVX2
// ============================================================

static void dot_i2s_i8_accum_i32_block32_avx2(
    int32_t* __restrict acc32,
    const uint8_t* __restrict Wq2,
    std::size_t H,
    std::size_t i0,
    std::size_t hCount,
    const int8_t* __restrict x8,
    std::size_t j0)
{
    init_i2s_lut();

    __m256i acc0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 0));
    __m256i acc1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 8));
    __m256i acc2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 16));
    __m256i acc3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc32 + 24));

    for (std::size_t di = 0; di < hCount; ++di) {
        const int xi = static_cast<int>(x8[di]);
        if (xi == 0) continue;

        const std::size_t i = i0 + di;
        const std::size_t idx0 = i * H + j0;
        const std::size_t b0 = idx0 >> 2;

        const uint8_t* wbytes = Wq2 + b0;

        // 32 columns = 8 packed bytes
        const __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(wbytes));
        const __m256i idx = _mm256_cvtepu8_epi32(raw8);

        // each gathered u32 contains 4 decoded int8 ternary weights
        const __m256i w32x8 = _mm256_i32gather_epi32(
            reinterpret_cast<const int*>(g_i2s_byte_to_4i8), idx, 4);

        const __m128i wLo128 = _mm256_castsi256_si128(w32x8);
        const __m128i wHi128 = _mm256_extracti128_si256(w32x8, 1);

        const __m128i w0_8 = wLo128;                    // bytes  0.. 7
        const __m128i w1_8 = _mm_srli_si128(wLo128, 8); // bytes  8..15
        const __m128i w2_8 = wHi128;                    // bytes 16..23
        const __m128i w3_8 = _mm_srli_si128(wHi128, 8); // bytes 24..31

        const __m256i xi32 = _mm256_set1_epi32(xi);

        const __m256i w0_32 = _mm256_cvtepi8_epi32(w0_8);
        const __m256i w1_32 = _mm256_cvtepi8_epi32(w1_8);
        const __m256i w2_32 = _mm256_cvtepi8_epi32(w2_8);
        const __m256i w3_32 = _mm256_cvtepi8_epi32(w3_8);

        acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(w0_32, xi32));
        acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(w1_32, xi32));
        acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(w2_32, xi32));
        acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(w3_32, xi32));
    }

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 0), acc0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 8), acc1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 16), acc2);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 24), acc3);
}
#endif

// ============================================================
// Public dispatcher
// ============================================================

void dot_i2s_i8_accum_i32_block32(
    int32_t* __restrict acc32,
    const uint8_t* __restrict Wq2,
    std::size_t H,
    std::size_t i0,
    std::size_t hCount,
    const int8_t* __restrict x8,
    std::size_t j0)
{
    if (!acc32) {
        throw std::runtime_error("dot_i2s_i8_accum_i32_block32: acc32 null");
    }
    if (!Wq2) {
        throw std::runtime_error("dot_i2s_i8_accum_i32_block32: Wq2 null");
    }
    if (!x8) {
        throw std::runtime_error("dot_i2s_i8_accum_i32_block32: x8 null");
    }
    if ((j0 % 32) != 0) {
        throw std::runtime_error("dot_i2s_i8_accum_i32_block32: j0 must be multiple of 32");
    }
    if (i0 + hCount > H) {
        throw std::runtime_error("dot_i2s_i8_accum_i32_block32: i0+hCount OOB");
    }

#if defined(__AVX2__)
    dot_i2s_i8_accum_i32_block32_avx2(acc32, Wq2, H, i0, hCount, x8, j0);
#else
    dot_i2s_i8_accum_i32_block32_scalar(acc32, Wq2, H, i0, hCount, x8, j0);
#endif
}