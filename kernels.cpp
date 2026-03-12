// =======================
// kernels.cpp  (OPTIMIZED)
// =======================
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <cstdint>
#include <stdexcept>
#include "kernels.h"
#include "bitnet_weight.h"
#include "bitnet_quant.h"

// ===========================================================================
// Internal AVX2 helpers
// ===========================================================================

// Horizontal sum of 8 floats in a __m256 register.
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// ===========================================================================
// Helpers
// ===========================================================================

std::size_t clamp_ttile(std::size_t tTile) {
    if (tTile == 0) return 1;
    return (tTile > 4) ? 4 : tTile;
}

std::size_t clamp_ttile_avx2(std::size_t tTile) {
    if (tTile == 0) return 1;
    return (tTile > 2) ? 2 : tTile;
}

// ===========================================================================
// RMSNorm — AVX2 forward + backward
// ===========================================================================

void rmsnorm(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict w,
    std::size_t H,
    float eps)
{
    // ── sum of squares ──────────────────────────────────────────────────────
    __m256 vacc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= H; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        vacc = _mm256_fmadd_ps(xv, xv, vacc);
    }
    float ss = hsum256(vacc);
    for (; i < H; ++i) ss += x[i] * x[i];

    const float inv = 1.0f / std::sqrt(ss / float(H) + eps);
    const __m256 invv = _mm256_set1_ps(inv);

    // ── scale ───────────────────────────────────────────────────────────────
    i = 0;
    for (; i + 8 <= H; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 wv = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(xv, invv), wv));
    }
    for (; i < H; ++i) y[i] = (x[i] * inv) * w[i];
}

void rmsnorm_backward(
    float* __restrict dx,
    const float* __restrict g,
    const float* __restrict x,
    const float* __restrict gamma,
    std::size_t H,
    float eps)
{
    // ── inv_r = 1/sqrt(mean(x^2)+eps) ──────────────────────────────────────
    __m256 vacc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= H; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        vacc = _mm256_fmadd_ps(xv, xv, vacc);
    }
    float ss = hsum256(vacc);
    for (; i < H; ++i) ss += x[i] * x[i];
    const float inv_r = 1.0f / std::sqrt(ss / float(H) + eps);

    // ── s = sum_h g[h]*gamma[h]*x[h] ───────────────────────────────────────
    __m256 sacc = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= H; i += 8) {
        __m256 gv = _mm256_loadu_ps(g + i);
        __m256 gam = _mm256_loadu_ps(gamma + i);
        __m256 xv = _mm256_loadu_ps(x + i);
        sacc = _mm256_fmadd_ps(_mm256_mul_ps(gv, gam), xv, sacc);
    }
    float s = hsum256(sacc);
    for (; i < H; ++i) s += g[i] * gamma[i] * x[i];

    const float inv_r3 = inv_r * inv_r * inv_r;
    const float coeff = s / float(H) * inv_r3;

    const __m256 inv_rv = _mm256_set1_ps(inv_r);
    const __m256 coeffv = _mm256_set1_ps(coeff);

    // ── dx[h] = inv_r*(g[h]*gamma[h]) - coeff*x[h] ─────────────────────────
    i = 0;
    for (; i + 8 <= H; i += 8) {
        __m256 gv = _mm256_loadu_ps(g + i);
        __m256 gam = _mm256_loadu_ps(gamma + i);
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 res = _mm256_fmsub_ps(inv_rv, _mm256_mul_ps(gv, gam),
            _mm256_mul_ps(coeffv, xv));
        _mm256_storeu_ps(dx + i, res);
    }
    for (; i < H; ++i)
        dx[i] = inv_r * (g[i] * gamma[i]) - coeff * x[i];
}

// ===========================================================================
// Linear HH  (single token, used in non-batched fallback paths)
// ===========================================================================

void linear_HH(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict W,
    std::size_t H,
    std::size_t hTile)
{
    std::fill(y, y + H, 0.0f);

    for (std::size_t h0 = 0; h0 < H; h0 += hTile) {
        const std::size_t h1 = std::min(H, h0 + hTile);

        for (std::size_t i = h0; i < h1; ++i) {
            const __m256    xi = _mm256_set1_ps(x[i]);
            const float* wrow = W + i * H;
            std::size_t j = 0;
            for (; j + 8 <= H; j += 8) {
                __m256 yv = _mm256_loadu_ps(y + j);
                __m256 wv = _mm256_loadu_ps(wrow + j);
                _mm256_storeu_ps(y + j, _mm256_fmadd_ps(xi, wv, yv));
            }
            for (; j < H; ++j) y[j] += x[i] * wrow[j];
        }
    }
}

// ===========================================================================
// Linear THH  (batched T tokens)
// ===========================================================================

void linear_THH(
    float* __restrict Y,
    const float* __restrict X,
    const float* __restrict W,
    std::size_t T,
    std::size_t H,
    std::size_t jTile)
{
    std::fill(Y, Y + T * H, 0.0f);

    for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
        const std::size_t j1 = std::min(H, j0 + jTile);
        const std::size_t width = j1 - j0;

        for (std::size_t i = 0; i < H; ++i) {
            const float* wrow = W + i * H + j0;

            for (std::size_t t = 0; t < T; ++t) {
                const __m256 x8 = _mm256_set1_ps(X[t * H + i]);
                float* yr = Y + t * H + j0;
                std::size_t  j = 0;
                for (; j + 8 <= width; j += 8)
                    _mm256_storeu_ps(yr + j,
                        _mm256_fmadd_ps(x8, _mm256_loadu_ps(wrow + j),
                            _mm256_loadu_ps(yr + j)));
                for (; j < width; ++j) yr[j] += X[t * H + i] * wrow[j];
            }
        }
    }
}

// ===========================================================================
// Linear HH_T accumulate  (y += A * B^T)
// ===========================================================================

void linear_HH_T_accum(
    float* __restrict y,
    const float* __restrict A,
    const float* __restrict B,
    std::size_t H,
    std::size_t /*hTile*/)
{
    // Each output element y[j] = dot(A, B[j,:])
    // Vectorise the inner dot over i with AVX2.
    for (std::size_t j = 0; j < H; ++j) {
        const float* Brow = B + j * H;
        __m256 vacc = _mm256_setzero_ps();
        std::size_t i = 0;
        for (; i + 8 <= H; i += 8)
            vacc = _mm256_fmadd_ps(_mm256_loadu_ps(A + i),
                _mm256_loadu_ps(Brow + i), vacc);
        float acc = hsum256(vacc);
        for (; i < H; ++i) acc += A[i] * Brow[i];
        y[j] += acc;
    }
}

// ===========================================================================
// LoRA  (AVX2 dot products)
// ===========================================================================

void lora_add_H(
    float* __restrict y,
    const float* __restrict x,
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict tmp,
    std::size_t H,
    std::size_t R,
    float scale,
    std::size_t /*hTile*/)
{
    // ── tmp[r] = dot(A[r,:], x)  — AVX2 dot ────────────────────────────────
    for (std::size_t r = 0; r < R; ++r) {
        const float* a = A + r * H;
        __m256 vacc = _mm256_setzero_ps();
        std::size_t h = 0;
        for (; h + 8 <= H; h += 8)
            vacc = _mm256_fmadd_ps(_mm256_loadu_ps(a + h),
                _mm256_loadu_ps(x + h), vacc);
        float acc = hsum256(vacc);
        for (; h < H; ++h) acc += a[h] * x[h];
        tmp[r] = acc;
    }

    // ── y[h] += scale * sum_r(tmp[r] * B[r,h])  — AVX2 axpy ───────────────
    // Iterate over r (small, ≤16) and do an AVX2 AXPY across H=2560.
    for (std::size_t r = 0; r < R; ++r) {
        const float  xr = scale * tmp[r];
        const __m256 xrv = _mm256_set1_ps(xr);
        const float* b = B + r * H;
        std::size_t h = 0;
        for (; h + 8 <= H; h += 8)
            _mm256_storeu_ps(y + h,
                _mm256_fmadd_ps(xrv, _mm256_loadu_ps(b + h),
                    _mm256_loadu_ps(y + h)));
        for (; h < H; ++h) y[h] += xr * b[h];
    }
}

// ===========================================================================
// Fused QKV forward — baseline (kept for compatibility)
// ===========================================================================

void linear_THH_fused_qkv(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile)
{
    std::fill(Q, Q + T * H, 0.0f);
    std::fill(K, K + T * H, 0.0f);
    std::fill(V, V + T * H, 0.0f);

    for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
        const std::size_t j1 = std::min(H, j0 + jTile);
        const std::size_t width = j1 - j0;

        for (std::size_t i = 0; i < H; ++i) {
            const float* wqrow = Wq + i * H + j0;
            const float* wkrow = Wk + i * H + j0;
            const float* wvrow = Wv + i * H + j0;

            for (std::size_t t0 = 0; t0 < T; t0 += tTile) {
                const std::size_t t1 = std::min(T, t0 + tTile);

                for (std::size_t t = t0; t < t1; ++t) {
                    const __m256 x8 = _mm256_set1_ps(X[t * H + i]);
                    float* qr = Q + t * H + j0;
                    float* kr = K + t * H + j0;
                    float* vr = V + t * H + j0;

                    std::size_t j = 0;
                    for (; j + 8 <= width; j += 8) {
                        _mm256_storeu_ps(qr + j, _mm256_fmadd_ps(x8, _mm256_loadu_ps(wqrow + j), _mm256_loadu_ps(qr + j)));
                        _mm256_storeu_ps(kr + j, _mm256_fmadd_ps(x8, _mm256_loadu_ps(wkrow + j), _mm256_loadu_ps(kr + j)));
                        _mm256_storeu_ps(vr + j, _mm256_fmadd_ps(x8, _mm256_loadu_ps(wvrow + j), _mm256_loadu_ps(vr + j)));
                    }
                    for (; j < width; ++j) {
                        qr[j] += X[t * H + i] * wqrow[j];
                        kr[j] += X[t * H + i] * wkrow[j];
                        vr[j] += X[t * H + i] * wvrow[j];
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Fused QKV forward — accum variant
// ===========================================================================

void linear_THH_fused_qkv_accum(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile)
{
    const std::size_t tTileEff = clamp_ttile(tTile);

    for (std::size_t t0 = 0; t0 < T; t0 += tTileEff) {
        const std::size_t tt = std::min(tTileEff, T - t0);

        for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
            const std::size_t j1 = std::min(H, j0 + jTile);
            const std::size_t width = j1 - j0;
            std::size_t j = 0;

            for (; j + 8 <= width; j += 8) {
                const std::size_t col = j0 + j;
                __m256 qacc[4], kacc[4], vacc[4];
                for (std::size_t r = 0; r < tt; ++r) {
                    qacc[r] = _mm256_setzero_ps();
                    kacc[r] = _mm256_setzero_ps();
                    vacc[r] = _mm256_setzero_ps();
                }
                for (std::size_t i = 0; i < H; ++i) {
                    const __m256 wqv = _mm256_loadu_ps(Wq + i * H + col);
                    const __m256 wkv = _mm256_loadu_ps(Wk + i * H + col);
                    const __m256 wvv = _mm256_loadu_ps(Wv + i * H + col);
                    for (std::size_t r = 0; r < tt; ++r) {
                        const __m256 x8 = _mm256_set1_ps(X[(t0 + r) * H + i]);
                        qacc[r] = _mm256_fmadd_ps(x8, wqv, qacc[r]);
                        kacc[r] = _mm256_fmadd_ps(x8, wkv, kacc[r]);
                        vacc[r] = _mm256_fmadd_ps(x8, wvv, vacc[r]);
                    }
                }
                for (std::size_t r = 0; r < tt; ++r) {
                    _mm256_storeu_ps(Q + (t0 + r) * H + col, qacc[r]);
                    _mm256_storeu_ps(K + (t0 + r) * H + col, kacc[r]);
                    _mm256_storeu_ps(V + (t0 + r) * H + col, vacc[r]);
                }
            }

            for (; j < width; ++j) {
                const std::size_t col = j0 + j;
                float qs[4] = { 0,0,0,0 }, ks[4] = { 0,0,0,0 }, vs[4] = { 0,0,0,0 };
                for (std::size_t i = 0; i < H; ++i) {
                    const float wq = Wq[i * H + col], wk = Wk[i * H + col], wv = Wv[i * H + col];
                    for (std::size_t r = 0; r < tt; ++r) {
                        const float xi = X[(t0 + r) * H + i];
                        qs[r] += xi * wq; ks[r] += xi * wk; vs[r] += xi * wv;
                    }
                }
                for (std::size_t r = 0; r < tt; ++r) {
                    Q[(t0 + r) * H + col] = qs[r];
                    K[(t0 + r) * H + col] = ks[r];
                    V[(t0 + r) * H + col] = vs[r];
                }
            }
        }
    }
}

// ===========================================================================
// Fused QKV forward — accum blocked (template)
// ===========================================================================

template <std::size_t jBlockVec>
static void linear_THH_fused_qkv_accum_blocked_impl(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile, std::size_t tTile)
{
    static_assert(jBlockVec == 2 || jBlockVec == 4, "jBlockVec must be 2 or 4");
    const std::size_t ttMax = clamp_ttile_avx2(tTile);

    for (std::size_t t0 = 0; t0 < T; t0 += ttMax) {
        const std::size_t tt = std::min(ttMax, T - t0);

        for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
            const std::size_t j1 = std::min(H, j0 + jTile);
            const std::size_t width = j1 - j0;
            constexpr std::size_t stride = 8 * jBlockVec;
            std::size_t j = 0;

            for (; j + stride <= width; j += stride) {
                const std::size_t col0 = j0 + j;
                __m256 qacc[2][jBlockVec], kacc[2][jBlockVec], vacc[2][jBlockVec];
                for (std::size_t r = 0; r < tt; ++r)
                    for (std::size_t b = 0; b < jBlockVec; ++b) {
                        qacc[r][b] = _mm256_setzero_ps();
                        kacc[r][b] = _mm256_setzero_ps();
                        vacc[r][b] = _mm256_setzero_ps();
                    }

                for (std::size_t i = 0; i < H; ++i) {
                    const float* wq = Wq + i * H + col0;
                    const float* wk = Wk + i * H + col0;
                    const float* wv = Wv + i * H + col0;
                    __m256 wqv[jBlockVec], wkv[jBlockVec], wvv[jBlockVec];
                    for (std::size_t b = 0; b < jBlockVec; ++b) {
                        wqv[b] = _mm256_loadu_ps(wq + 8 * b);
                        wkv[b] = _mm256_loadu_ps(wk + 8 * b);
                        wvv[b] = _mm256_loadu_ps(wv + 8 * b);
                    }
                    for (std::size_t r = 0; r < tt; ++r) {
                        const __m256 x8 = _mm256_set1_ps(X[(t0 + r) * H + i]);
                        for (std::size_t b = 0; b < jBlockVec; ++b) {
                            qacc[r][b] = _mm256_fmadd_ps(x8, wqv[b], qacc[r][b]);
                            kacc[r][b] = _mm256_fmadd_ps(x8, wkv[b], kacc[r][b]);
                            vacc[r][b] = _mm256_fmadd_ps(x8, wvv[b], vacc[r][b]);
                        }
                    }
                }

                for (std::size_t r = 0; r < tt; ++r) {
                    float* qt = Q + (t0 + r) * H + col0;
                    float* kt = K + (t0 + r) * H + col0;
                    float* vt = V + (t0 + r) * H + col0;
                    for (std::size_t b = 0; b < jBlockVec; ++b) {
                        _mm256_storeu_ps(qt + 8 * b, qacc[r][b]);
                        _mm256_storeu_ps(kt + 8 * b, kacc[r][b]);
                        _mm256_storeu_ps(vt + 8 * b, vacc[r][b]);
                    }
                }
            }

            for (; j < width; ++j) {
                const std::size_t col = j0 + j;
                for (std::size_t r = 0; r < tt; ++r) {
                    float qs = 0, ks = 0, vs = 0;
                    const float* xt = X + (t0 + r) * H;
                    for (std::size_t i = 0; i < H; ++i) {
                        qs += xt[i] * Wq[i * H + col];
                        ks += xt[i] * Wk[i * H + col];
                        vs += xt[i] * Wv[i * H + col];
                    }
                    Q[(t0 + r) * H + col] = qs;
                    K[(t0 + r) * H + col] = ks;
                    V[(t0 + r) * H + col] = vs;
                }
            }
        }
    }
}

void linear_THH_fused_qkv_accum_blocked16(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H, std::size_t jTile, std::size_t tTile)
{
    linear_THH_fused_qkv_accum_blocked_impl<2>(Q, K, V, X, Wq, Wk, Wv, T, H, jTile, tTile);
}

void linear_THH_fused_qkv_accum_blocked32(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H, std::size_t jTile, std::size_t tTile)
{
    linear_THH_fused_qkv_accum_blocked_impl<4>(Q, K, V, X, Wq, Wk, Wv, T, H, jTile, tTile);
}

// ===========================================================================
// QKV input backward (float weights)
// ===========================================================================

void qkv_input_backward_THH(
    float* __restrict dX,
    const float* __restrict dQ,
    const float* __restrict dK,
    const float* __restrict dV,
    const float* __restrict Wq,
    const float* __restrict Wk,
    const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t hTile)
{
    for (std::size_t t = 0; t < T; ++t) {
        float* dx = dX + t * H;
        const float* dq = dQ + t * H;
        const float* dk = dK + t * H;
        const float* dv = dV + t * H;

        for (std::size_t i0 = 0; i0 < H; i0 += hTile) {
            const std::size_t i1 = std::min(H, i0 + hTile);

            for (std::size_t j = 0; j < H; ++j) {
                const __m256 dq8 = _mm256_set1_ps(dq[j]);
                const __m256 dk8 = _mm256_set1_ps(dk[j]);
                const __m256 dv8 = _mm256_set1_ps(dv[j]);

                const float* wqr = Wq + j * H + i0;
                const float* wkr = Wk + j * H + i0;
                const float* wvr = Wv + j * H + i0;

                std::size_t i = i0;
                for (; i + 8 <= i1; i += 8) {
                    const std::size_t off = i - i0;
                    __m256 dxv = _mm256_loadu_ps(dx + i);
                    dxv = _mm256_fmadd_ps(dq8, _mm256_loadu_ps(wqr + off), dxv);
                    dxv = _mm256_fmadd_ps(dk8, _mm256_loadu_ps(wkr + off), dxv);
                    dxv = _mm256_fmadd_ps(dv8, _mm256_loadu_ps(wvr + off), dxv);
                    _mm256_storeu_ps(dx + i, dxv);
                }
                for (; i < i1; ++i)
                    dx[i] += dq[j] * Wq[j * H + i] + dk[j] * Wk[j * H + i] + dv[j] * Wv[j * H + i];
            }
        }
    }
}

// ===========================================================================
// BitNet ternary kernels
// ===========================================================================

void linear_THH_bitnet(
    float* __restrict Y,
    const float* __restrict X,
    const BitNetMat& Wq,
    std::size_t T,
    std::size_t H,
    std::size_t jTile)
{
    std::fill(Y, Y + T * H, 0.0f);

    const int8_t* __restrict q = Wq.q.data();
    const float* __restrict s = Wq.s.data();
    const std::size_t        block = Wq.block;

    for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
        const std::size_t j1 = std::min(H, j0 + jTile);

        for (std::size_t i = 0; i < H; ++i) {
            const std::size_t rowBase = i * H;

            for (std::size_t t = 0; t < T; ++t) {
                const float xi = X[t * H + i];
                if (xi == 0.0f) continue;

                float* yrow = Y + t * H + j0;
                for (std::size_t j = j0; j < j1; ++j) {
                    const std::size_t idx = rowBase + j;
                    const int8_t qq = q[idx];
                    if (qq == 0) continue;
                    yrow[j - j0] += xi * (s[idx / block] * float(qq));
                }
            }
        }
    }
}

void qkv_input_backward_THH_bitnet(
    float* __restrict dX,
    const float* __restrict dQ,
    const float* __restrict dK,
    const float* __restrict dV,
    const BitNetMat& Wq,
    const BitNetMat& Wk,
    const BitNetMat& Wv,
    std::size_t T,
    std::size_t H,
    std::size_t /*hTile*/)
{
    // FIX: avoid heap allocation on every call — use a stack-local buffer
    // sized to maxSeqLen*H. Since T<=32 and H=2560 that's 327KB, fine for
    // a worker thread stack on Windows (default 1MB).
    // If T*H ever exceeds ~80K floats (~320KB) switch to a member buffer.
    std::fill(dX, dX + T * H, 0.0f);

    // Reuse a single temporary and accumulate all three projections.
    // Using a fixed-size stack VLA would require C99; use a small static
    // thread_local buffer instead to avoid repeated allocation.
    static thread_local std::vector<float> tmp;
    tmp.assign(T * H, 0.0f);

    linear_THH_bitnet(tmp.data(), dQ, Wq, T, H, 64);
    for (std::size_t i = 0; i < T * H; ++i) dX[i] += tmp[i];

    std::fill(tmp.begin(), tmp.end(), 0.0f);
    linear_THH_bitnet(tmp.data(), dK, Wk, T, H, 64);
    for (std::size_t i = 0; i < T * H; ++i) dX[i] += tmp[i];

    std::fill(tmp.begin(), tmp.end(), 0.0f);
    linear_THH_bitnet(tmp.data(), dV, Wv, T, H, 64);
    for (std::size_t i = 0; i < T * H; ++i) dX[i] += tmp[i];
}

void linear_THH_fused_qkv_bitnet_accum_blocked32(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const BitNetMat& WqM, const BitNetMat& WkM, const BitNetMat& WvM,
    std::size_t T, std::size_t H,
    std::size_t jTile, std::size_t tTile)
{
    const std::size_t block = WqM.block;

    if (block == 0 || (block % 8) != 0 ||
        WkM.block != block || WvM.block != block ||
        WqM.q.size() != H * H || WkM.q.size() != H * H || WvM.q.size() != H * H)
    {
        linear_THH_bitnet(Q, X, WqM, T, H, jTile);
        linear_THH_bitnet(K, X, WkM, T, H, jTile);
        linear_THH_bitnet(V, X, WvM, T, H, jTile);
        return;
    }

    const std::size_t tt = (tTile == 0) ? 1 : std::min<std::size_t>(4, tTile);

    for (std::size_t t0 = 0; t0 < T; t0 += tt) {
        const std::size_t tCount = std::min(tt, T - t0);

        for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
            const std::size_t j1 = std::min(H, j0 + jTile);
            const std::size_t width = j1 - j0;
            std::size_t j = 0;

            for (; j + 8 <= width; j += 8) {
                const std::size_t col = j0 + j;
                __m256 qacc[4], kacc[4], vacc[4];
                for (std::size_t r = 0; r < tCount; ++r) {
                    qacc[r] = _mm256_setzero_ps();
                    kacc[r] = _mm256_setzero_ps();
                    vacc[r] = _mm256_setzero_ps();
                }
                for (std::size_t i = 0; i < H; ++i) {
                    const std::size_t idx0 = i * H + col;
                    const __m256 wQ = bitnet_dequant8_ps_twoScale(WqM, idx0);
                    const __m256 wK = bitnet_dequant8_ps_twoScale(WkM, idx0);
                    const __m256 wV = bitnet_dequant8_ps_twoScale(WvM, idx0);
                    for (std::size_t r = 0; r < tCount; ++r) {
                        const __m256 x8 = _mm256_set1_ps(X[(t0 + r) * H + i]);
                        qacc[r] = _mm256_fmadd_ps(x8, wQ, qacc[r]);
                        kacc[r] = _mm256_fmadd_ps(x8, wK, kacc[r]);
                        vacc[r] = _mm256_fmadd_ps(x8, wV, vacc[r]);
                    }
                }
                for (std::size_t r = 0; r < tCount; ++r) {
                    _mm256_storeu_ps(Q + (t0 + r) * H + col, qacc[r]);
                    _mm256_storeu_ps(K + (t0 + r) * H + col, kacc[r]);
                    _mm256_storeu_ps(V + (t0 + r) * H + col, vacc[r]);
                }
            }

            for (; j < width; ++j) {
                const std::size_t col = j0 + j;
                float qs[4] = { 0,0,0,0 }, ks[4] = { 0,0,0,0 }, vs[4] = { 0,0,0,0 };
                for (std::size_t i = 0; i < H; ++i) {
                    const std::size_t idx = i * H + col;
                    const float wq = WqM.s[idx / block] * (float)WqM.q[idx];
                    const float wk = WkM.s[idx / block] * (float)WkM.q[idx];
                    const float wv = WvM.s[idx / block] * (float)WvM.q[idx];
                    for (std::size_t r = 0; r < tCount; ++r) {
                        const float xi = X[(t0 + r) * H + i];
                        qs[r] += xi * wq; ks[r] += xi * wk; vs[r] += xi * wv;
                    }
                }
                for (std::size_t r = 0; r < tCount; ++r) {
                    Q[(t0 + r) * H + col] = qs[r]; K[(t0 + r) * H + col] = ks[r]; V[(t0 + r) * H + col] = vs[r];
                }
            }
        }
    }
}

void linear_HH_T_accum_bitnet(
    float* __restrict y,
    const float* __restrict A,
    const BitNetMat& WoM,
    std::size_t H,
    std::size_t /*hTile*/)
{
    const int8_t* __restrict q = WoM.q.data();
    const float* __restrict s = WoM.s.data();
    const std::size_t        block = WoM.block;
    const bool vecOk = (block != 0) && ((block % 8) == 0) && ((H % block) == 0);

    for (std::size_t j = 0; j < H; ++j) {
        const std::size_t rowBase = j * H;
        __m256 vacc = _mm256_setzero_ps();
        float  tailAcc = 0.0f;
        std::size_t i = 0;

        if (vecOk) {
            for (; i + 8 <= H; i += 8) {
                const std::size_t idx0 = rowBase + i;
                const __m256 sc = _mm256_set1_ps(s[idx0 / block]);
                const __m128i q8 = _mm_loadl_epi64((const __m128i*)(q + idx0));
                const __m256 wf = _mm256_mul_ps(sc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)));
                vacc = _mm256_fmadd_ps(_mm256_loadu_ps(A + i), wf, vacc);
            }
        }
        for (; i < H; ++i) {
            const std::size_t idx = rowBase + i;
            tailAcc += A[i] * (s[idx / block] * float(q[idx]));
        }

        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        y[j] += _mm_cvtss_f32(sum) + tailAcc;
    }
}

void linear_THH_fused_qkv_bitnet_backward_dX_blocked32(
    float* __restrict dX,
    const float* __restrict dQ,
    const float* __restrict dK,
    const float* __restrict dV,
    const BitNetMat& WqM, const BitNetMat& WkM, const BitNetMat& WvM,
    std::size_t T, std::size_t H,
    const int* __restrict attentionMask,
    std::size_t iTile, std::size_t tTile)
{
    const std::size_t block = WqM.block;

    if (block == 0 || (block % 8) != 0 ||
        WkM.block != block || WvM.block != block ||
        WqM.q.size() != H * H || WkM.q.size() != H * H || WvM.q.size() != H * H ||
        WqM.s.empty() || WkM.s.empty() || WvM.s.empty())
        throw std::runtime_error("linear_THH_fused_qkv_bitnet_backward_dX_blocked32: invalid BitNetMat");

    const std::size_t tt = (tTile == 0) ? 1 : std::min<std::size_t>(4, tTile);
    const std::size_t ii = (iTile == 0) ? 64 : iTile;

    for (std::size_t t0 = 0; t0 < T; t0 += tt) {
        const std::size_t tCount = std::min(tt, T - t0);

        for (std::size_t i0 = 0; i0 < H; i0 += ii) {
            const std::size_t i1 = std::min(H, i0 + ii);

            for (std::size_t i = i0; i < i1; ++i) {
                float acc[4] = { 0,0,0,0 };
                std::size_t j = 0;

                for (; j + 8 <= H; j += 8) {
                    const std::size_t idx0 = i * H + j;
                    const __m256 wQ = bitnet_dequant8_ps_twoScale(WqM, idx0);
                    const __m256 wK = bitnet_dequant8_ps_twoScale(WkM, idx0);
                    const __m256 wV = bitnet_dequant8_ps_twoScale(WvM, idx0);

                    for (std::size_t r = 0; r < tCount; ++r) {
                        const std::size_t t = t0 + r;
                        if (attentionMask && attentionMask[t] == 0) continue;
                        const __m256 dq8 = _mm256_loadu_ps(dQ + t * H + j);
                        const __m256 dk8 = _mm256_loadu_ps(dK + t * H + j);
                        const __m256 dv8 = _mm256_loadu_ps(dV + t * H + j);
                        __m256 sv = _mm256_mul_ps(dq8, wQ);
                        sv = _mm256_fmadd_ps(dk8, wK, sv);
                        sv = _mm256_fmadd_ps(dv8, wV, sv);
                        acc[r] += hsum256(sv);
                    }
                }
                for (; j < H; ++j) {
                    const std::size_t idx = i * H + j;
                    const float wq = WqM.s[idx / block] * (float)WqM.q[idx];
                    const float wk = WkM.s[idx / block] * (float)WkM.q[idx];
                    const float wv = WvM.s[idx / block] * (float)WvM.q[idx];
                    for (std::size_t r = 0; r < tCount; ++r) {
                        const std::size_t t = t0 + r;
                        if (attentionMask && attentionMask[t] == 0) continue;
                        acc[r] += dQ[t * H + j] * wq + dK[t * H + j] * wk + dV[t * H + j] * wv;
                    }
                }

                for (std::size_t r = 0; r < tCount; ++r) {
                    const std::size_t t = t0 + r;
                    if (attentionMask && attentionMask[t] == 0) continue;
                    dX[t * H + i] += acc[r];
                }
            }
        }
    }
}

// ===========================================================================
// scale_inplace  (used by trainer to normalise dLogits)
// ===========================================================================

void scale_inplace_avx2_or_scalar(float* data, std::size_t n, float scale)
{
    const __m256 sv = _mm256_set1_ps(scale);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), sv));
    for (; i < n; ++i) data[i] *= scale;
}