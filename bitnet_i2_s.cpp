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
// Quantization Helpers
// ============================================================

static inline float max_abs_block(
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

void pack_ternary_to_2bit(std::vector<uint8_t>& out_q2, const int8_t* q, std::size_t N) {
    if (!q) throw std::runtime_error("pack_ternary_to_2bit: q is null");
    out_q2.assign((N + 3) / 4, 0);
    for (std::size_t i = 0; i < N; ++i) {
        out_q2[i >> 2] |= static_cast<uint8_t>((ternary_to_i2_code(q[i]) & 0x3u) << ((i & 3u) * 2u));
    }
}

void unpack_2bit_to_ternary(std::vector<int8_t>& out_q, const uint8_t* q2, std::size_t N) {
    if (!q2) throw std::runtime_error("unpack_2bit_to_ternary: q2 is null");
    out_q.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        out_q[i] = i2_code_to_ternary(static_cast<uint8_t>((q2[i >> 2] >> ((i & 3u) * 2u)) & 0x3u));
    }
}

void quantize_x_tile_i8_tiled(
    int8_t* __restrict Xi8, float* __restrict xScale, const float* __restrict X,
    std::size_t T, std::size_t H, std::size_t t0, std::size_t tCount, std::size_t tBlock, std::size_t hBlock)
{
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
                    for (std::size_t h = hStart; h < hEnd; ++h) outRow[h] = 0;
                }
                else {
                    for (std::size_t h = hStart; h < hEnd; ++h) {
                        int qi = static_cast<int>(std::lrint(row[h] * inv));
                        outRow[h] = static_cast<int8_t>(std::max(-127, std::min(127, qi)));
                    }
                }
            }
        }
    }
}

// ============================================================
// Pure Bitwise AVX2 Unpacking Helper
// ============================================================
#if defined(__AVX2__)
static inline __m256 decode_2bytes_to_ps(const uint8_t* ptr, __m256i shift_vec, __m256i mask_vec, __m256i one_vec) {
    uint16_t val16;
    std::memcpy(&val16, ptr, sizeof(uint16_t));
    __m256i v = _mm256_set1_epi32(val16);
    v = _mm256_srlv_epi32(v, shift_vec);
    v = _mm256_and_si256(v, mask_vec);
    v = _mm256_sub_epi32(v, one_vec);
    return _mm256_cvtepi32_ps(v);
}
#endif

// ============================================================
// Block32 X1 Kernel
// ============================================================
void dot_i2s_i8_accum_i32_block32(
    int32_t* __restrict acc32, const uint8_t* __restrict Wq2, std::size_t H,
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
        const uint8_t* wbytes = Wq2 + (((i0 + di) * H + j0) >> 2);

        // Prefetch weights 4 rows ahead
        _mm_prefetch(reinterpret_cast<const char*>(wbytes + H), _MM_HINT_T0);

        a0 = _mm256_fmadd_ps(x_ps, decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec), a0);
        a1 = _mm256_fmadd_ps(x_ps, decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec), a1);
        a2 = _mm256_fmadd_ps(x_ps, decode_2bytes_to_ps(wbytes + 4, shift_vec, mask_vec, one_vec), a2);
        a3 = _mm256_fmadd_ps(x_ps, decode_2bytes_to_ps(wbytes + 6, shift_vec, mask_vec, one_vec), a3);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 0), _mm256_cvtps_epi32(a0));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 8), _mm256_cvtps_epi32(a1));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 16), _mm256_cvtps_epi32(a2));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc32 + 24), _mm256_cvtps_epi32(a3));
#else
    for (std::size_t di = 0; di < hCount; ++di) {
        const int xi = static_cast<int>(x8[di]);
        if (xi == 0) continue;
        const uint8_t* wbytes = Wq2 + (((i0 + di) * H + j0) >> 2);
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

// ============================================================
// Block32 X2 Kernel
// ============================================================
void dot_i2s_i8_accum_i32_block32_x2(
    int32_t* __restrict accA, int32_t* __restrict accB, const uint8_t* __restrict Wq2,
    std::size_t H, std::size_t i0, std::size_t hCount,
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

        const uint8_t* wbytes = Wq2 + (((i0 + di) * H + j0) >> 2);

        // Prefetch weights 4 rows ahead
        _mm_prefetch(reinterpret_cast<const char*>(wbytes + H), _MM_HINT_T0);

        __m256 w0 = decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec);
        __m256 w1 = decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec);
        __m256 w2 = decode_2bytes_to_ps(wbytes + 4, shift_vec, mask_vec, one_vec);
        __m256 w3 = decode_2bytes_to_ps(wbytes + 6, shift_vec, mask_vec, one_vec);

        if (x8A[di]) {
            __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di]));
            aA0 = _mm256_fmadd_ps(vx, w0, aA0); aA1 = _mm256_fmadd_ps(vx, w1, aA1);
            aA2 = _mm256_fmadd_ps(vx, w2, aA2); aA3 = _mm256_fmadd_ps(vx, w3, aA3);
        }
        if (x8B[di]) {
            __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di]));
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
        const int xa = static_cast<int>(x8A[di]), xb = static_cast<int>(x8B[di]);
        if (xa == 0 && xb == 0) continue;
        const uint8_t* wbytes = Wq2 + (((i0 + di) * H + j0) >> 2);
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

// ============================================================
// Block32 X4 Kernel (Zero Register Spilling Architecture)
// ============================================================
void dot_i2s_i8_accum_i32_block32_x4(
    int32_t* __restrict accA, int32_t* __restrict accB,
    int32_t* __restrict accC, int32_t* __restrict accD,
    const uint8_t* __restrict Wq2, std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B,
    const int8_t* __restrict x8C, const int8_t* __restrict x8D, std::size_t j0) 
{
#if defined(__AVX2__)
    const __m256i shift_vec = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i mask_vec = _mm256_set1_epi32(3);
    const __m256i one_vec = _mm256_set1_epi32(1);

    // stride_cols elements = (stride_cols / 4) bytes. 
    // Jumping 4 rows ahead is EXACTLY +stride_cols in bytes!
    const std::size_t prefetch_stride = stride_cols;

    // --- HALF 1: Columns 0 to 15 ---
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

            __m256 w0 = decode_2bytes_to_ps(wbytes + 0, shift_vec, mask_vec, one_vec);
            __m256 w1 = decode_2bytes_to_ps(wbytes + 2, shift_vec, mask_vec, one_vec);

            if (x8A[di]) { __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di])); aA0 = _mm256_fmadd_ps(vx, w0, aA0); aA1 = _mm256_fmadd_ps(vx, w1, aA1); }
            if (x8B[di]) { __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di])); aB0 = _mm256_fmadd_ps(vx, w0, aB0); aB1 = _mm256_fmadd_ps(vx, w1, aB1); }
            if (x8C[di]) { __m256 vx = _mm256_set1_ps(static_cast<float>(x8C[di])); aC0 = _mm256_fmadd_ps(vx, w0, aC0); aC1 = _mm256_fmadd_ps(vx, w1, aC1); }
            if (x8D[di]) { __m256 vx = _mm256_set1_ps(static_cast<float>(x8D[di])); aD0 = _mm256_fmadd_ps(vx, w0, aD0); aD1 = _mm256_fmadd_ps(vx, w1, aD1); }
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 0), _mm256_cvtps_epi32(aA0)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 8), _mm256_cvtps_epi32(aA1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 0), _mm256_cvtps_epi32(aB0)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 8), _mm256_cvtps_epi32(aB1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accC + 0), _mm256_cvtps_epi32(aC0)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accC + 8), _mm256_cvtps_epi32(aC1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accD + 0), _mm256_cvtps_epi32(aD0)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accD + 8), _mm256_cvtps_epi32(aD1));
    }

    // --- HALF 2: Columns 16 to 31 ---
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

            __m256 w2 = decode_2bytes_to_ps(wbytes + 4, shift_vec, mask_vec, one_vec);
            __m256 w3 = decode_2bytes_to_ps(wbytes + 6, shift_vec, mask_vec, one_vec);

            if (x8A[di]) { __m256 vx = _mm256_set1_ps(static_cast<float>(x8A[di])); aA2 = _mm256_fmadd_ps(vx, w2, aA2); aA3 = _mm256_fmadd_ps(vx, w3, aA3); }
            if (x8B[di]) { __m256 vx = _mm256_set1_ps(static_cast<float>(x8B[di])); aB2 = _mm256_fmadd_ps(vx, w2, aB2); aB3 = _mm256_fmadd_ps(vx, w3, aB3); }
            if (x8C[di]) { __m256 vx = _mm256_set1_ps(static_cast<float>(x8C[di])); aC2 = _mm256_fmadd_ps(vx, w2, aC2); aC3 = _mm256_fmadd_ps(vx, w3, aC3); }
            if (x8D[di]) { __m256 vx = _mm256_set1_ps(static_cast<float>(x8D[di])); aD2 = _mm256_fmadd_ps(vx, w2, aD2); aD3 = _mm256_fmadd_ps(vx, w3, aD3); }
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 16), _mm256_cvtps_epi32(aA2)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accA + 24), _mm256_cvtps_epi32(aA3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 16), _mm256_cvtps_epi32(aB2)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accB + 24), _mm256_cvtps_epi32(aB3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accC + 16), _mm256_cvtps_epi32(aC2)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accC + 24), _mm256_cvtps_epi32(aC3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(accD + 16), _mm256_cvtps_epi32(aD2)); _mm256_storeu_si256(reinterpret_cast<__m256i*>(accD + 24), _mm256_cvtps_epi32(aD3));
    }
#else
    dot_i2s_i8_accum_i32_block32_x2(accA, accB, Wq2, H, i0, hCount, x8A, x8B, j0);
    dot_i2s_i8_accum_i32_block32_x2(accC, accD, Wq2, H, i0, hCount, x8C, x8D, j0);
#endif
}