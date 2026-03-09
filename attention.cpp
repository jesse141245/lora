#include <cstddef>
#include <algorithm>
#include <cmath>
#include "attention.h"
#include <cstring>

void softmax_inplace(float* p, std::size_t n) {
    float m = p[0];
    for (std::size_t i = 1; i < n; ++i) m = std::max(m, p[i]);
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) { p[i] = std::exp(p[i] - m); s += p[i]; }
    float inv = 1.0f / (s + 1e-9f);
    for (std::size_t i = 0; i < n; ++i) p[i] *= inv;
}

void attn_causal(
    float* ctx,                 // [T x H] output
    const float* q,             // [T x H]
    const float* k,             // [T x H]
    const float* v,             // [T x H]
    std::size_t T,
    std::size_t H,
    std::size_t nHeads,
    float* scores,              // [maxT x maxT] scratch
    float* probs,               // [maxT x maxT] scratch
    const int* attentionMask     // [T] (0 = ignore token)
) {
    const std::size_t D = H / nHeads;
    const float inv_sqrt_d = 1.0f / std::sqrt(float(D));

    std::fill(ctx, ctx + T * H, 0.0f);

    for (std::size_t h = 0; h < nHeads; ++h) {
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask && attentionMask[t] == 0) continue;

            // compute scores for j<=t
            float* sc = &scores[t * T];
            for (std::size_t j = 0; j <= t; ++j) {
                if (attentionMask && attentionMask[j] == 0) { sc[j] = -1e9f; continue; }
                const float* qt = &q[t * H + h * D];
                const float* kj = &k[j * H + h * D];
                float dot = 0.0f;
                for (std::size_t d = 0; d < D; ++d) dot += qt[d] * kj[d];
                sc[j] = dot * inv_sqrt_d;
            }
            // fill rest as -inf
            for (std::size_t j = t + 1; j < T; ++j) sc[j] = -1e9f;

            // probs = softmax(scores)
            float* pr = &probs[t * T];
            std::memcpy(pr, sc, T * sizeof(float));
            softmax_inplace(pr, T);

            // ctx[t] += sum_j pr[j] * v[j]
            float* out = &ctx[t * H + h * D];
            for (std::size_t j = 0; j <= t; ++j) {
                const float pj = pr[j];
                const float* vj = &v[j * H + h * D];
                for (std::size_t d = 0; d < D; ++d) out[d] += pj * vj[d];
            }
        }
    }
}

void attn_causal_backward(
    float* dQ,
    float* dK,
    float* dV,
    const float* dCtx,
    const float* q,
    const float* k,
    const float* v,
    std::size_t T,
    std::size_t H,
    std::size_t nHeads,
    const float* probs,
    const int* attentionMask,
    float* tmp_dP
) 
{
    const std::size_t D = H / nHeads;
    const float inv_sqrt_d = 1.0f / std::sqrt(float(D));

    for (std::size_t head = 0; head < nHeads; ++head) {
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask && attentionMask[t] == 0) continue;

            const float* pr = &probs[t * T];
            const float* dCt = &dCtx[t * H + head * D];
            const float* Qt = &q[t * H + head * D];
            float* dQt = &dQ[t * H + head * D];

            // ---- 1) dP[t,j] = dot(dCtx[t], V[j]) and sum = ?_j P[t,j]*dP[t,j]
            float sum = 0.0f;
            for (std::size_t j = 0; j <= t; ++j) {
                if (attentionMask && attentionMask[j] == 0) { tmp_dP[j] = 0.0f; continue; }

                const float* Vj = &v[j * H + head * D];
                float dp = 0.0f;
                for (std::size_t d = 0; d < D; ++d) dp += dCt[d] * Vj[d];

                tmp_dP[j] = dp;
                sum += pr[j] * dp;
            }
            // ensure tmp_dP for j>t is 0 (not strictly needed if we never read it)
            // for (j=t+1..T-1) tmp_dP[j]=0;

            // ---- 2/3/4) for each j: dS, then accumulate into dQ, dK, and dV
            for (std::size_t j = 0; j <= t; ++j) {
                if (attentionMask && attentionMask[j] == 0) continue;

                const float pj = pr[j];
                const float dS = pj * (tmp_dP[j] - sum); // softmax backward

                // dV[j] += P[t,j] * dCtx[t]
                float* dVj = &dV[j * H + head * D];
                for (std::size_t d = 0; d < D; ++d) dVj[d] += pj * dCt[d];

                // dQ[t] += dS * K[j] * inv_sqrt_d
                const float* Kj = &k[j * H + head * D];
                const float alphaQ = dS * inv_sqrt_d;
                for (std::size_t d = 0; d < D; ++d) dQt[d] += alphaQ * Kj[d];

                // dK[j] += dS * Q[t] * inv_sqrt_d
                float* dKj = &dK[j * H + head * D];
                const float alphaK = dS * inv_sqrt_d;
                for (std::size_t d = 0; d < D; ++d) dKj[d] += alphaK * Qt[d];
            }
        }
    }
}