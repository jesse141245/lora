#include "bitnet_quant.h"

#include "bitnet_i2_s.h"
#include "bitnet_weight.h"

#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <stdexcept>
#include <vector>

// ============================================================
// Legacy ternary blockwise quantization
// ============================================================

void quantize_ternary_blockwise(
    BitNetMat& out,
    const float* W,
    std::size_t H,
    std::size_t block)
{
    if (!W) {
        throw std::runtime_error("quantize_ternary_blockwise: W is null");
    }
    if (H == 0) {
        throw std::runtime_error("quantize_ternary_blockwise: H == 0");
    }
    if (block == 0) {
        throw std::runtime_error("quantize_ternary_blockwise: block == 0");
    }

    const std::size_t N = H * H;

    out.H = H;
    out.block = block;
    out.q.resize(N);

    const std::size_t nBlocks = (N + block - 1) / block;
    out.s.resize(nBlocks);

    for (std::size_t b = 0; b < nBlocks; ++b) {
        const std::size_t i0 = b * block;
        const std::size_t i1 = std::min(N, i0 + block);

        float amax = 0.0f;
        for (std::size_t i = i0; i < i1; ++i) {
            amax = std::max(amax, std::abs(W[i]));
        }

        const float s = (amax > 0.0f) ? amax : 1.0f;
        out.s[b] = s;

        const float inv_s = 1.0f / s;

        for (std::size_t i = i0; i < i1; ++i) {
            const float v = W[i] * inv_s;

            int qi;
            if (v > 0.5f) {
                qi = 1;
            }
            else if (v < -0.5f) {
                qi = -1;
            }
            else {
                qi = 0;
            }

            out.q[i] = static_cast<int8_t>(qi);
        }
    }
}

// ============================================================
// I2S tiled quantization (Option C)
//
// W is [H x H] row-major.
//
// Scale layout:
//   w_scale_tile[jb * nKBlocks + kb]
//
// where:
//   kb = input-row tile index   (tile height = qk)
//   jb = output-col tile index  (tile width  = jq)
//
// This builds only forward-packed storage:
//   - q2
//   - w_scale_tile
//
// No q2T / w_scale_tileT are stored.
//
// Backward support is provided through compact assist metadata:
//   build_bitnet_i2s_backward_assist(out)
// ============================================================

void quantize_ternary_i2s_blockwise(
    BitNetMatI2S& out,
    const float* W,
    std::size_t rows,
    std::size_t cols,
    std::size_t qk,
    std::size_t jq)
{
    if (!W) throw std::runtime_error("quantize_ternary_i2s_blockwise: W null");
    if (rows == 0 || cols == 0) throw std::runtime_error("quantize_ternary_i2s_blockwise: Zero dim");
    if ((cols % 32) != 0) throw std::runtime_error("quantize_ternary_i2s_blockwise: cols must be multiple of 32");

    const std::size_t N = rows * cols;
    const std::size_t nKBlocks = (rows + qk - 1) / qk;
    const std::size_t nJBlocks = (cols + jq - 1) / jq;

    out.rows = rows;
    out.cols = cols;
    out.qk = qk;
    out.jq = jq;
    out.q2.clear();
    out.w_scale_tile.assign(nJBlocks * nKBlocks, 1.0f);

    std::vector<int8_t> q(N, 0);

    for (std::size_t jb = 0; jb < nJBlocks; ++jb) {
        const std::size_t j0 = jb * jq;
        const std::size_t j1 = std::min(cols, j0 + jq);

        for (std::size_t kb = 0; kb < nKBlocks; ++kb) {
            const std::size_t i0 = kb * qk;
            const std::size_t i1 = std::min(rows, i0 + qk);

            float amax = 0.0f;
            for (std::size_t i = i0; i < i1; ++i) {
                for (std::size_t j = j0; j < j1; ++j) {
                    amax = std::max(amax, std::abs(W[i * cols + j]));
                }
            }

            const float s = (amax > 0.0f) ? amax : 1.0f;
            out.w_scale_tile[jb * nKBlocks + kb] = s;
            const float inv_s = 1.0f / s;

            for (std::size_t i = i0; i < i1; ++i) {
                for (std::size_t j = j0; j < j1; ++j) {
                    const float v = W[i * cols + j] * inv_s;
                    int qi = (v > 0.5f) ? 1 : (v < -0.5f) ? -1 : 0;
                    q[i * cols + j] = static_cast<int8_t>(qi);
                }
            }
        }
    }

    pack_ternary_to_2bit(out.q2, q.data(), N);
    build_bitnet_i2s_transposed_view(out);
}
// ============================================================
// Legacy dequantization
// ============================================================

void dequant_to_float(
    float* Wout,
    const BitNetMat& Wq)
{
    if (!Wout) {
        throw std::runtime_error("dequant_to_float: Wout is null");
    }

    const std::size_t H = Wq.H;
    const std::size_t N = H * H;
    const std::size_t block = Wq.block;

    if (H == 0 || block == 0) {
        throw std::runtime_error("dequant_to_float: invalid BitNetMat");
    }

    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t b = i / block;

        if (b >= Wq.s.size()) {
            throw std::runtime_error("dequant_to_float: scale index OOB");
        }

        Wout[i] = Wq.s[b] * static_cast<float>(Wq.q[i]);
    }
}

// ============================================================
// Legacy AVX helpers
// ============================================================

__m256 bitnet_dequant8_ps(
    const BitNetMat& M,
    std::size_t idx0)
{
    const std::size_t block = M.block;

    const int8_t* __restrict q = M.q.data();
    const float* __restrict s = M.s.data();

    const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + idx0));
    const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));

    const std::size_t off = idx0 % block;

    if (off <= block - 8) {
        const __m256 sc = _mm256_set1_ps(s[idx0 / block]);
        return _mm256_mul_ps(sc, qf);
    }

    alignas(32) float sca[8];
    sca[0] = s[(idx0 + 0) / block];
    sca[1] = s[(idx0 + 1) / block];
    sca[2] = s[(idx0 + 2) / block];
    sca[3] = s[(idx0 + 3) / block];
    sca[4] = s[(idx0 + 4) / block];
    sca[5] = s[(idx0 + 5) / block];
    sca[6] = s[(idx0 + 6) / block];
    sca[7] = s[(idx0 + 7) / block];

    const __m256 scv = _mm256_load_ps(sca);
    return _mm256_mul_ps(scv, qf);
}

__m256 bitnet_dequant8_ps_twoScale(
    const BitNetMat& M,
    std::size_t idx0)
{
    const std::size_t block = M.block;
    const int8_t* __restrict q = M.q.data();
    const float* __restrict s = M.s.data();

    const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + idx0));
    const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));

    const std::size_t off = idx0 % block;
    const std::size_t b0 = idx0 / block;

    if (off <= block - 8) {
        return _mm256_mul_ps(_mm256_set1_ps(s[b0]), qf);
    }

    const int n0 = static_cast<int>(block - off);
    const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i nn0 = _mm256_set1_epi32(n0);
    const __m256 m0 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(nn0, idx));

    const __m256 sc0 = _mm256_set1_ps(s[b0]);
    const __m256 sc1 = _mm256_set1_ps(s[b0 + 1]);

    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 sc = _mm256_add_ps(
        _mm256_mul_ps(m0, sc0),
        _mm256_mul_ps(_mm256_sub_ps(one, m0), sc1)
    );

    return _mm256_mul_ps(sc, qf);
}

float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}