// =======================
// kernels.cpp  (FULL)
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

// ---------- Helpers ----------
std::size_t clamp_ttile(std::size_t tTile) {
    // AVX2 register pressure: 4 is usually safe; 2 often best.
    if (tTile == 0) return 1;
    return (tTile > 4) ? 4 : tTile;
}

std::size_t clamp_ttile_avx2(std::size_t tTile) {
    // Keep small on AVX2 to avoid spills; 1 or 2 usually best here.
    if (tTile == 0) return 1;
    return (tTile > 2) ? 2 : tTile;
}

// ---------- RMSNorm ----------
void rmsnorm(
    float* y,
    const float* x,
    const float* w,
    std::size_t H,
    float eps)
{
    float ss = 0.0f;
    for (std::size_t i = 0; i < H; ++i) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / float(H) + eps);
    for (std::size_t i = 0; i < H; ++i) y[i] = (x[i] * inv) * w[i];
}

void rmsnorm_backward(
    float* dx,
    const float* g,
    const float* x,
    const float* gamma,
    std::size_t H,
    float eps)
{
    // inv_r = 1/sqrt(mean(x^2) + eps)
    float ss = 0.0f;
    for (std::size_t h = 0; h < H; ++h) ss += x[h] * x[h];
    ss = std::sqrt((ss / static_cast<float>(H)) + eps);
    const float inv_r = 1.0f / ss;

    // s = sum_h g[h]*gamma[h]*x[h]
    float s = 0.0f;
    for (std::size_t h = 0; h < H; ++h) s += g[h] * gamma[h] * x[h];

    const float inv_r3 = inv_r * inv_r * inv_r;
    const float coeff = s / static_cast<float>(H) * inv_r3;

    for (std::size_t h = 0; h < H; ++h) {
        dx[h] = inv_r * (g[h] * gamma[h]) - coeff * x[h];
    }
}

// ---------- Linear HH ----------
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
            const __m256 xi = _mm256_set1_ps(x[i]);
            const float* wrow = &W[i * H];

            std::size_t j = 0;
            for (; j + 8 <= H; j += 8) {
                __m256 yv = _mm256_loadu_ps(y + j);
                __m256 wv = _mm256_loadu_ps(wrow + j);
                yv = _mm256_fmadd_ps(xi, wv, yv);
                _mm256_storeu_ps(y + j, yv);
            }
            for (; j < H; ++j) y[j] += x[i] * wrow[j];
        }
    }
}

// ---------- Linear THH ----------
void linear_THH(
    float* __restrict Y,          // [T x H]
    const float* __restrict X,    // [T x H]
    const float* __restrict W,    // [H x H] row-major
    std::size_t T,
    std::size_t H,
    std::size_t jTile)
{
    std::fill(Y, Y + T * H, 0.0f);

    for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
        const std::size_t j1 = std::min(H, j0 + jTile);

        for (std::size_t i = 0; i < H; ++i) {
            const float* wrow = W + i * H + j0;

            for (std::size_t t = 0; t < T; ++t) {
                const __m256 x8 = _mm256_set1_ps(X[t * H + i]);
                float* yrow = Y + t * H + j0;

                const std::size_t width = (j1 - j0);
                std::size_t j = 0;
                for (; j + 8 <= width; j += 8) {
                    __m256 yv = _mm256_loadu_ps(yrow + j);
                    __m256 wv = _mm256_loadu_ps(wrow + j);
                    yv = _mm256_fmadd_ps(x8, wv, yv);
                    _mm256_storeu_ps(yrow + j, yv);
                }
                for (; j < width; ++j) {
                    yrow[j] += X[t * H + i] * wrow[j];
                }
            }
        }
    }
}

// ---------- Linear HH_T accumulate (y += A * B^T) ----------
void linear_HH_T_accum(
    float* y,
    const float* A,
    const float* B,
    std::size_t H,
    std::size_t hTile)
{
    for (std::size_t j = 0; j < H; ++j) {
        const float* Brow = &B[j * H];
        float acc = 0.0f;

        for (std::size_t i0 = 0; i0 < H; i0 += hTile) {
            const std::size_t i1 = std::min(H, i0 + hTile);
            for (std::size_t i = i0; i < i1; ++i) {
                acc += A[i] * Brow[i];
            }
        }
        y[j] += acc;
    }
}

// ---------- LoRA ----------
void lora_add_H(
    float* y,
    const float* x,
    const float* A,
    const float* B,
    float* tmp,
    std::size_t H,
    std::size_t R,
    float scale,
    std::size_t hTile)
{
    // tmp[r] = dot(A[r,*], x)
    for (std::size_t r = 0; r < R; ++r) {
        const float* a = &A[r * H];
        float acc = 0.0f;
        for (std::size_t h = 0; h < H; ++h) acc += a[h] * x[h];
        tmp[r] = acc;
    }

    // y[h] += scale * sum_r tmp[r] * B[r,h]
    for (std::size_t h0 = 0; h0 < H; h0 += hTile) {
        const std::size_t h1 = std::min(H, h0 + hTile);
        for (std::size_t r = 0; r < R; ++r) {
            const float xr = scale * tmp[r];
            const float* b = &B[r * H + h0];
            for (std::size_t h = h0; h < h1; ++h) {
                y[h] += xr * b[h - h0];
            }
        }
    }
}

// ---------- Fused QKV forward (baseline) ----------
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
                    const float xi = X[t * H + i];
                    const __m256 x8 = _mm256_set1_ps(xi);

                    float* qrow = Q + t * H + j0;
                    float* krow = K + t * H + j0;
                    float* vrow = V + t * H + j0;

                    std::size_t j = 0;
                    for (; j + 8 <= width; j += 8) {
                        __m256 qv = _mm256_loadu_ps(qrow + j);
                        __m256 kv = _mm256_loadu_ps(krow + j);
                        __m256 vv = _mm256_loadu_ps(vrow + j);

                        __m256 wqv = _mm256_loadu_ps(wqrow + j);
                        __m256 wkv = _mm256_loadu_ps(wkrow + j);
                        __m256 wvv = _mm256_loadu_ps(wvrow + j);

                        qv = _mm256_fmadd_ps(x8, wqv, qv);
                        kv = _mm256_fmadd_ps(x8, wkv, kv);
                        vv = _mm256_fmadd_ps(x8, wvv, vv);

                        _mm256_storeu_ps(qrow + j, qv);
                        _mm256_storeu_ps(krow + j, kv);
                        _mm256_storeu_ps(vrow + j, vv);
                    }

                    for (; j < width; ++j) {
                        qrow[j] += xi * wqrow[j];
                        krow[j] += xi * wkrow[j];
                        vrow[j] += xi * wvrow[j];
                    }
                }
            }
        }
    }
}

// ---------- Fused QKV forward (accum variant) ----------
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
        const std::size_t tt = std::min(tTileEff, T - t0); // 1..4

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
                    const float* __restrict wqrow = Wq + i * H + col;
                    const float* __restrict wkrow = Wk + i * H + col;
                    const float* __restrict wvrow = Wv + i * H + col;

                    const __m256 wqv = _mm256_loadu_ps(wqrow);
                    const __m256 wkv = _mm256_loadu_ps(wkrow);
                    const __m256 wvv = _mm256_loadu_ps(wvrow);

                    for (std::size_t r = 0; r < tt; ++r) {
                        const float xi = X[(t0 + r) * H + i];
                        const __m256 x8 = _mm256_set1_ps(xi);
                        qacc[r] = _mm256_fmadd_ps(x8, wqv, qacc[r]);
                        kacc[r] = _mm256_fmadd_ps(x8, wkv, kacc[r]);
                        vacc[r] = _mm256_fmadd_ps(x8, wvv, vacc[r]);
                    }
                }

                for (std::size_t r = 0; r < tt; ++r) {
                    float* __restrict qt = Q + (t0 + r) * H;
                    float* __restrict kt = K + (t0 + r) * H;
                    float* __restrict vt = V + (t0 + r) * H;
                    _mm256_storeu_ps(qt + col, qacc[r]);
                    _mm256_storeu_ps(kt + col, kacc[r]);
                    _mm256_storeu_ps(vt + col, vacc[r]);
                }
            }

            for (; j < width; ++j) {
                const std::size_t col = j0 + j;

                float qs[4] = { 0,0,0,0 };
                float ks[4] = { 0,0,0,0 };
                float vs[4] = { 0,0,0,0 };

                for (std::size_t i = 0; i < H; ++i) {
                    const float wq = Wq[i * H + col];
                    const float wk = Wk[i * H + col];
                    const float wv = Wv[i * H + col];
                    for (std::size_t r = 0; r < tt; ++r) {
                        const float xi = X[(t0 + r) * H + i];
                        qs[r] += xi * wq;
                        ks[r] += xi * wk;
                        vs[r] += xi * wv;
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

// ---------- Fused QKV forward (accum blocked) ----------
template <std::size_t jBlockVec>
void linear_THH_fused_qkv_accum_blocked(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile)
{
    static_assert(jBlockVec == 2 || jBlockVec == 4, "Use jBlockVec = 2 or 4 for AVX2");
    const std::size_t ttMax = clamp_ttile_avx2(tTile);

    for (std::size_t t0 = 0; t0 < T; t0 += ttMax) {
        const std::size_t tt = std::min(ttMax, T - t0); // 1..2

        for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
            const std::size_t j1 = std::min(H, j0 + jTile);
            const std::size_t width = j1 - j0;

            std::size_t j = 0;
            constexpr std::size_t stride = 8 * jBlockVec;

            for (; j + stride <= width; j += stride) {
                const std::size_t col0 = j0 + j;

                __m256 qacc[2][jBlockVec];
                __m256 kacc[2][jBlockVec];
                __m256 vacc[2][jBlockVec];

                for (std::size_t r = 0; r < tt; ++r) {
                    for (std::size_t b = 0; b < jBlockVec; ++b) {
                        qacc[r][b] = _mm256_setzero_ps();
                        kacc[r][b] = _mm256_setzero_ps();
                        vacc[r][b] = _mm256_setzero_ps();
                    }
                }

                for (std::size_t i = 0; i < H; ++i) {
                    const float* __restrict wq = Wq + i * H + col0;
                    const float* __restrict wk = Wk + i * H + col0;
                    const float* __restrict wv = Wv + i * H + col0;

                    __m256 wqv[jBlockVec], wkv[jBlockVec], wvv[jBlockVec];
                    for (std::size_t b = 0; b < jBlockVec; ++b) {
                        wqv[b] = _mm256_loadu_ps(wq + 8 * b);
                        wkv[b] = _mm256_loadu_ps(wk + 8 * b);
                        wvv[b] = _mm256_loadu_ps(wv + 8 * b);
                    }

                    for (std::size_t r = 0; r < tt; ++r) {
                        const float xi = X[(t0 + r) * H + i];
                        const __m256 x8 = _mm256_set1_ps(xi);

                        for (std::size_t b = 0; b < jBlockVec; ++b) {
                            qacc[r][b] = _mm256_fmadd_ps(x8, wqv[b], qacc[r][b]);
                            kacc[r][b] = _mm256_fmadd_ps(x8, wkv[b], kacc[r][b]);
                            vacc[r][b] = _mm256_fmadd_ps(x8, wvv[b], vacc[r][b]);
                        }
                    }
                }

                for (std::size_t r = 0; r < tt; ++r) {
                    float* __restrict qt = Q + (t0 + r) * H + col0;
                    float* __restrict kt = K + (t0 + r) * H + col0;
                    float* __restrict vt = V + (t0 + r) * H + col0;

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
                    float qs = 0.0f, ks = 0.0f, vs = 0.0f;
                    const float* __restrict xt = X + (t0 + r) * H;
                    for (std::size_t i = 0; i < H; ++i) {
                        const float xi = xt[i];
                        qs += xi * Wq[i * H + col];
                        ks += xi * Wk[i * H + col];
                        vs += xi * Wv[i * H + col];
                    }
                    Q[(t0 + r) * H + col] = qs;
                    K[(t0 + r) * H + col] = ks;
                    V[(t0 + r) * H + col] = vs;
                }
            }
        }
    }
}

// Explicit instantiations (so other .cpp files can link without seeing template body)
void linear_THH_fused_qkv_accum_blocked16(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile)
{
    linear_THH_fused_qkv_accum_blocked<2>(
        Q, K, V, X, Wq, Wk, Wv, T, H, jTile, tTile
    );
}

void linear_THH_fused_qkv_accum_blocked32(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const float* __restrict Wq, const float* __restrict Wk, const float* __restrict Wv,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile)
{
    linear_THH_fused_qkv_accum_blocked<4>(
        Q, K, V, X, Wq, Wk, Wv, T, H, jTile, tTile
    );
}
// ---------- NEW: QKV input backward (transpose multiply) ----------
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
    // dX[t,i] += sum_j dQ[t,j]*Wq[j,i] + dK[t,j]*Wk[j,i] + dV[t,j]*Wv[j,i]
    for (std::size_t t = 0; t < T; ++t) {
        float* __restrict dx = dX + t * H;
        const float* __restrict dq = dQ + t * H;
        const float* __restrict dk = dK + t * H;
        const float* __restrict dv = dV + t * H;

        for (std::size_t i0 = 0; i0 < H; i0 += hTile) {
            const std::size_t i1 = std::min(H, i0 + hTile);

            for (std::size_t j = 0; j < H; ++j) {
                const float dqj = dq[j];
                const float dkj = dk[j];
                const float dvj = dv[j];

                const float* __restrict wqrow = Wq + j * H + i0;
                const float* __restrict wkrow = Wk + j * H + i0;
                const float* __restrict wvrow = Wv + j * H + i0;

                const __m256 dq8 = _mm256_set1_ps(dqj);
                const __m256 dk8 = _mm256_set1_ps(dkj);
                const __m256 dv8 = _mm256_set1_ps(dvj);

                std::size_t i = i0;
                for (; i + 8 <= i1; i += 8) {
                    __m256 dx8 = _mm256_loadu_ps(dx + i);

                    __m256 wq8 = _mm256_loadu_ps(wqrow + (i - i0));
                    __m256 wk8 = _mm256_loadu_ps(wkrow + (i - i0));
                    __m256 wv8 = _mm256_loadu_ps(wvrow + (i - i0));

                    dx8 = _mm256_fmadd_ps(dq8, wq8, dx8);
                    dx8 = _mm256_fmadd_ps(dk8, wk8, dx8);
                    dx8 = _mm256_fmadd_ps(dv8, wv8, dx8);

                    _mm256_storeu_ps(dx + i, dx8);
                }

                for (; i < i1; ++i) {
                    dx[i] += dqj * Wq[j * H + i]
                        + dkj * Wk[j * H + i]
                        + dvj * Wv[j * H + i];
                }
            }
        }
    }
}
// ===========================
// BitNet kernels (ternary + scale)
// ===========================

// X [T x H], W [H x H] row-major, Y [T x H]
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
    const std::size_t block = Wq.block;

    for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
        const std::size_t j1 = std::min(H, j0 + jTile);

        for (std::size_t i = 0; i < H; ++i) {
            const std::size_t rowBase = i * H;

            for (std::size_t t = 0; t < T; ++t) {
                const float xi = X[t * H + i];
                if (xi == 0.0f) continue;

                float* __restrict yrow = &Y[t * H + j0];

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

// dX = dQ*Wq + dK*Wk + dV*Wv  (same math as float kernels, but ternary weights)
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
    std::fill(dX, dX + T * H, 0.0f);

    std::vector<float> tmp(T * H);

    linear_THH_bitnet(tmp.data(), dQ, Wq, T, H, /*jTile*/64);
    for (std::size_t i = 0; i < T * H; ++i) dX[i] += tmp[i];

    linear_THH_bitnet(tmp.data(), dK, Wk, T, H, /*jTile*/64);
    for (std::size_t i = 0; i < T * H; ++i) dX[i] += tmp[i];

    linear_THH_bitnet(tmp.data(), dV, Wv, T, H, /*jTile*/64);
    for (std::size_t i = 0; i < T * H; ++i) dX[i] += tmp[i];
}

// Fast fused QKV forward for BitNet ternary weights
// Requirements for the fast vector path to be safe:
//  - block % 8 == 0
//  - H % block == 0
//  - j0 and jTile aligned so our 8-wide chunks never cross a block boundary
void linear_THH_fused_qkv_bitnet_accum_blocked32(
    float* __restrict Q, float* __restrict K, float* __restrict V,
    const float* __restrict X,
    const BitNetMat& WqM, const BitNetMat& WkM, const BitNetMat& WvM,
    std::size_t T, std::size_t H,
    std::size_t jTile,
    std::size_t tTile)
{
    const std::size_t block = WqM.block;

    // basic compatibility checks
    if (block == 0 || (block % 8) != 0 ||
        WkM.block != block || WvM.block != block ||
        WqM.q.size() != H * H || WkM.q.size() != H * H || WvM.q.size() != H * H)
    {
        linear_THH_bitnet(Q, X, WqM, T, H, jTile);
        linear_THH_bitnet(K, X, WkM, T, H, jTile);
        linear_THH_bitnet(V, X, WvM, T, H, jTile);
        return;
    }

    // Clamp tTile to 1..4 like your float fused kernel
    const std::size_t tt = (tTile == 0) ? 1 : std::min<std::size_t>(4, tTile);

    for (std::size_t t0 = 0; t0 < T; t0 += tt) {
        const std::size_t tCount = std::min(tt, T - t0);

        for (std::size_t j0 = 0; j0 < H; j0 += jTile) {
            const std::size_t j1 = std::min(H, j0 + jTile);
            const std::size_t width = j1 - j0;

            std::size_t j = 0;

            // 8-wide vector path
            for (; j + 8 <= width; j += 8) {
                const std::size_t col = j0 + j;

                __m256 qacc[4], kacc[4], vacc[4];
                for (std::size_t r = 0; r < tCount; ++r) {
                    qacc[r] = _mm256_setzero_ps();
                    kacc[r] = _mm256_setzero_ps();
                    vacc[r] = _mm256_setzero_ps();
                }

                // i loop (input channel)
                for (std::size_t i = 0; i < H; ++i) {
                    const std::size_t idx0 = i * H + col;

                    const __m256 wQ = bitnet_dequant8_ps_twoScale(WqM, idx0);
                    const __m256 wK = bitnet_dequant8_ps_twoScale(WkM, idx0);
                    const __m256 wV = bitnet_dequant8_ps_twoScale(WvM, idx0);

                    // broadcast x for each row in the tTile
                    for (std::size_t r = 0; r < tCount; ++r) {
                        const float xi = X[(t0 + r) * H + i];
                        const __m256 x8 = _mm256_set1_ps(xi);
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

            // scalar tail
            for (; j < width; ++j) {
                const std::size_t col = j0 + j;

                float qs[4] = { 0,0,0,0 };
                float ks[4] = { 0,0,0,0 };
                float vs[4] = { 0,0,0,0 };

                for (std::size_t i = 0; i < H; ++i) {
                    const std::size_t idx = i * H + col;
                    const float wq = WqM.s[idx / block] * (float)WqM.q[idx];
                    const float wk = WkM.s[idx / block] * (float)WkM.q[idx];
                    const float wv = WvM.s[idx / block] * (float)WvM.q[idx];

                    for (std::size_t r = 0; r < tCount; ++r) {
                        const float xi = X[(t0 + r) * H + i];
                        qs[r] += xi * wq;
                        ks[r] += xi * wk;
                        vs[r] += xi * wv;
                    }
                }

                for (std::size_t r = 0; r < tCount; ++r) {
                    Q[(t0 + r) * H + col] = qs[r];
                    K[(t0 + r) * H + col] = ks[r];
                    V[(t0 + r) * H + col] = vs[r];
                }
            }
        }
    }
}
// BitNet Wo^T backward: y += A * Wo^T
void linear_HH_T_accum_bitnet(
    float* y,
    const float* A,
    const BitNetMat& WoM,
    std::size_t H,
    std::size_t hTile)
{
    const int8_t* __restrict q = WoM.q.data();
    const float* __restrict s = WoM.s.data();
    const std::size_t block = WoM.block;

    const bool vecOk = (block != 0) && ((block % 8) == 0) && ((H % block) == 0);

    for (std::size_t j = 0; j < H; ++j) {
        const std::size_t rowBase = j * H;

        __m256 vacc = _mm256_setzero_ps();
        float tailAcc = 0.0f;

        for (std::size_t i0 = 0; i0 < H; i0 += hTile) {
            const std::size_t i1 = std::min(H, i0 + hTile);

            std::size_t i = i0;
            if (vecOk) {
                for (; i + 8 <= i1; i += 8) {
                    const std::size_t idx0 = rowBase + i;
                    const __m256 sc = _mm256_set1_ps(s[idx0 / block]);
                    const __m128i q8 = _mm_loadl_epi64((const __m128i*)(q + idx0));
                    const __m256 wf = _mm256_mul_ps(sc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)));
                    const __m256 av = _mm256_loadu_ps(A + i);
                    vacc = _mm256_fmadd_ps(av, wf, vacc);
                }
            }
            for (; i < i1; ++i) {
                const std::size_t idx = rowBase + i;
                tailAcc += A[i] * (s[idx / block] * float(q[idx]));
            }
        }

        // horizontal sum of vacc
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        y[j] += _mm_cvtss_f32(sum) + tailAcc;
    }
}


// dX += dQ*Wq^T + dK*Wk^T + dV*Wv^T   (W stored row-major [i,j])
void linear_THH_fused_qkv_bitnet_backward_dX_blocked32(
    float* __restrict dX,                 // [T x H] (accum)
    const float* __restrict dQ,            // [T x H]
    const float* __restrict dK,            // [T x H]
    const float* __restrict dV,            // [T x H]
    const BitNetMat& WqM, const BitNetMat& WkM, const BitNetMat& WvM,
    std::size_t T, std::size_t H,
    const int* __restrict attentionMask,   // [T] may be null
    std::size_t iTile,                    // tile over i
    std::size_t tTile)                    // tile over t
{
    const std::size_t block = WqM.block;

    // Validate layout
    if (block == 0 || (block % 8) != 0 ||
        WkM.block != block || WvM.block != block ||
        WqM.q.size() != H * H || WkM.q.size() != H * H || WvM.q.size() != H * H ||
        WqM.s.empty() || WkM.s.empty() || WvM.s.empty())
    {
        throw std::runtime_error("linear_THH_fused_qkv_bitnet_backward_dX_blocked32: invalid BitNetMat");
    }

    const std::size_t tt = (tTile == 0) ? 1 : std::min<std::size_t>(4, tTile);
    const std::size_t ii = (iTile == 0) ? 64 : iTile;

    for (std::size_t t0 = 0; t0 < T; t0 += tt) {
        const std::size_t tCount = std::min(tt, T - t0);

        for (std::size_t i0 = 0; i0 < H; i0 += ii) {
            const std::size_t i1 = std::min(H, i0 + ii);

            for (std::size_t i = i0; i < i1; ++i) {

                // Keep per-(t within tile) scalar accumulators
                float acc[4] = { 0,0,0,0 };

                // dot over j
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

                        __m256 s = _mm256_mul_ps(dq8, wQ);
                        s = _mm256_fmadd_ps(dk8, wK, s);
                        s = _mm256_fmadd_ps(dv8, wV, s);

                        acc[r] += hsum256_ps(s);
                    }
                }

                // tail
                for (; j < H; ++j) {
                    const std::size_t idx = i * H + j;
                    const float wq = WqM.s[idx / block] * (float)WqM.q[idx];
                    const float wk = WkM.s[idx / block] * (float)WkM.q[idx];
                    const float wv = WvM.s[idx / block] * (float)WvM.q[idx];

                    for (std::size_t r = 0; r < tCount; ++r) {
                        const std::size_t t = t0 + r;
                        if (attentionMask && attentionMask[t] == 0) continue;

                        acc[r] += dQ[t * H + j] * wq
                            + dK[t * H + j] * wk
                            + dV[t * H + j] * wv;
                    }
                }

                // accumulate into dX
                for (std::size_t r = 0; r < tCount; ++r) {
                    const std::size_t t = t0 + r;
                    if (attentionMask && attentionMask[t] == 0) continue;
                    dX[t * H + i] += acc[r];
                }
            }
        }
    }
}