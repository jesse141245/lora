#pragma once
#include <cstddef>
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
);

void softmax_inplace(
    float* p, 
    std::size_t n
);

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
);
