#include <cstddef>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include "attention.h"

// ===========================================================================
// Internal helpers
// ===========================================================================

static inline float hsum128(__m128 v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hsum128(_mm_add_ps(lo, hi));
}

// ===========================================================================
// AVX2 softmax (in-place)
// Replaces the fully scalar version. With n=32 (T) this runs in ~4 AVX passes
// instead of 32 scalar iterations.
// ===========================================================================

static void softmax_inplace(float* __restrict p, std::size_t n)
{
    // ── find max ────────────────────────────────────────────────────────────
    __m256 vmx = _mm256_set1_ps(-1e38f);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        vmx = _mm256_max_ps(vmx, _mm256_loadu_ps(p + i));
    float mx = hsum256(_mm256_max_ps(vmx, _mm256_permute2f128_ps(vmx, vmx, 1)));
    // horizontal max from the 8 lanes
    __m128 lo = _mm256_castps256_ps128(vmx);
    __m128 hi = _mm256_extractf128_ps(vmx, 1);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ps(m4, _mm_movehdup_ps(m4));
    mx = _mm_cvtss_f32(m4);
    for (; i < n; ++i) mx = std::max(mx, p[i]);

    // ── exp(x - max) and sum ─────────────────────────────────────────────────
    const __m256 vmxv = _mm256_set1_ps(mx);
    __m256 vsum = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(p + i), vmxv);
        // Use __builtin_ia32_* is compiler-specific; just use std::exp via scalar
        // for correctness — Clang-CL with -ffast-math will vectorise this loop.
        // For guaranteed vectorisation we do it scalar then store back.
        float tmp[8];
        _mm256_storeu_ps(tmp, v);
        for (int k = 0; k < 8; ++k) tmp[k] = std::exp(tmp[k]);
        v = _mm256_loadu_ps(tmp);
        _mm256_storeu_ps(p + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    float s = hsum256(vsum);
    for (; i < n; ++i) { p[i] = std::exp(p[i] - mx); s += p[i]; }

    // ── normalise ────────────────────────────────────────────────────────────
    const float inv = 1.0f / (s + 1e-9f);
    const __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(p + i, _mm256_mul_ps(_mm256_loadu_ps(p + i), vinv));
    for (; i < n; ++i) p[i] *= inv;
}

// ===========================================================================
// AVX2 dot product over D dimensions
// D = H/nHeads = 2560/32 = 80 for the 2B model.
// ===========================================================================

static inline float dot_D(const float* __restrict a,
    const float* __restrict b,
    std::size_t D)
{
    __m256 acc = _mm256_setzero_ps();
    std::size_t d = 0;
    for (; d + 8 <= D; d += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc);
    float s = hsum256(acc);
    for (; d < D; ++d) s += a[d] * b[d];
    return s;
}

// AVX2 AXPY: dst[d] += scale * src[d]  over D dims
static inline void axpy_D(float* __restrict dst,
    const float* __restrict src,
    float scale,
    std::size_t D)
{
    const __m256 sv = _mm256_set1_ps(scale);
    std::size_t d = 0;
    for (; d + 8 <= D; d += 8)
        _mm256_storeu_ps(dst + d,
            _mm256_fmadd_ps(sv, _mm256_loadu_ps(src + d),
                _mm256_loadu_ps(dst + d)));
    for (; d < D; ++d) dst[d] += scale * src[d];
}

// ===========================================================================
// attn_causal  (forward)
// ===========================================================================

void attn_causal(
    float* __restrict ctx,
    const float* __restrict q,
    const float* __restrict k,
    const float* __restrict v,
    std::size_t T,
    std::size_t H,
    std::size_t nHeads,
    std::size_t nKvHeads,
    float* __restrict scores,
    float* __restrict probs,
    const int* attentionMask)
{
    const std::size_t D = H / nHeads;
    const std::size_t kvDim = nKvHeads * D;
    const float       inv_sqrt_d = 1.0f / std::sqrt(float(D));
    const std::size_t heads_per_group = nHeads / nKvHeads;

    std::fill(ctx, ctx + T * H, 0.0f);

    for (std::size_t h = 0; h < nHeads; ++h) {
        const std::size_t kv_h = h / heads_per_group;

        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask && attentionMask[t] == 0) continue;

            const float* qt = q + t * H + h * D;
            float* sc = scores + t * T;

            // ── Q·K scores ───────────────────────────────────────────────────
            for (std::size_t j = 0; j <= t; ++j) {
                if (attentionMask && attentionMask[j] == 0) {
                    sc[j] = -1e9f;
                    continue;
                }
                sc[j] = dot_D(qt, k + j * kvDim + kv_h * D, D) * inv_sqrt_d;
            }
            for (std::size_t j = t + 1; j < T; ++j) sc[j] = -1e9f;

            // ── softmax ───────────────────────────────────────────────────────
            float* pr = probs + t * T;
            std::memcpy(pr, sc, T * sizeof(float));
            softmax_inplace(pr, T);

            // ── weighted sum of V ─────────────────────────────────────────────
            float* out = ctx + t * H + h * D;
            for (std::size_t j = 0; j <= t; ++j) {
                if (attentionMask && attentionMask[j] == 0) continue;
                axpy_D(out, v + j * kvDim + kv_h * D, pr[j], D);
            }
        }
    }
}

// ===========================================================================
// attn_causal_backward
//
// BUG FIX: tmp_dP was never zeroed between heads. For masked positions
// (attentionMask[j]==0) tmp_dP[j] was left with the value from the previous
// head, causing a stale dP contribution in the softmax backward of every
// subsequent head. Fixed by zeroing the full T-length buffer at the start of
// each (head, t) pair before the dP computation loop.
// ===========================================================================

void attn_causal_backward(
    float* __restrict dQ,
    float* __restrict dK,
    float* __restrict dV,
    const float* __restrict dCtx,
    const float* __restrict q,
    const float* __restrict k,
    const float* __restrict v,
    std::size_t T,
    std::size_t H,
    std::size_t nHeads,
    std::size_t nKvHeads,
    const float* __restrict probs,
    const int* attentionMask,
    float* __restrict tmp_dP)
{
    const std::size_t D = H / nHeads;
    const std::size_t kvDim = nKvHeads * D;
    const float       inv_sqrt_d = 1.0f / std::sqrt(float(D));
    const std::size_t heads_per_group = nHeads / nKvHeads;

    for (std::size_t head = 0; head < nHeads; ++head) {
        const std::size_t kv_h = head / heads_per_group;

        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask && attentionMask[t] == 0) continue;

            const float* pr = probs + t * T;
            const float* dCt = dCtx + t * H + head * D;
            const float* Qt = q + t * H + head * D;
            float* dQt = dQ + t * H + head * D;

            // FIX: zero the full T-length scratch buffer before use.
            // Previously masked positions carried stale values from the
            // previous head, corrupting the softmax backward sum.
            std::memset(tmp_dP, 0, T * sizeof(float));

            // ── dP[j] = dot(dCtx[t,head], V[j,kv_h])  ───────────────────────
            float sum = 0.0f;
            for (std::size_t j = 0; j <= t; ++j) {
                if (attentionMask && attentionMask[j] == 0) continue;
                const float dp = dot_D(dCt, v + j * kvDim + kv_h * D, D);
                tmp_dP[j] = dp;
                sum += pr[j] * dp;
            }

            // ── accumulate dV, dQ, dK ────────────────────────────────────────
            // dS = pj*(dP[j] - sum) is the softmax score gradient.
            // When late tokens attend many positions, (dP[j] - sum) can be
            // large while pj≈0, making dS numerically unstable.
            // Clamp dS to [-10, 10] before multiplying by inv_sqrt_d.
            // This prevents the attention backward from generating the
            // gradient spikes (~14x normal) seen in early LoRA training.
            constexpr float kDSClamp = 10.0f;
            for (std::size_t j = 0; j <= t; ++j) {
                if (attentionMask && attentionMask[j] == 0) continue;

                const float pj = pr[j];
                const float dS_raw = pj * (tmp_dP[j] - sum);
                const float dS = std::max(-kDSClamp, std::min(kDSClamp, dS_raw));
                const float aQ = dS * inv_sqrt_d;
                const float aK = dS * inv_sqrt_d;

                // dV[j,kv_h] += pr[j] * dCtx[t,head]
                axpy_D(dV + j * kvDim + kv_h * D, dCt, pj, D);

                // dQ[t,head] += dS/sqrt(D) * K[j,kv_h]
                axpy_D(dQt, k + j * kvDim + kv_h * D, aQ, D);

                // dK[j,kv_h] += dS/sqrt(D) * Q[t,head]
                axpy_D(dK + j * kvDim + kv_h * D, Qt, aK, D);
            }
        }
    }
}