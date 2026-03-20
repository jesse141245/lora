#pragma once

#include <cstddef>

/**
 * @brief Computes Grouped-Query/Multi-Head Causal Attention (Forward Pass).
 * Calculates Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d)) * V.
 * Includes masking to prevent future tokens from influencing past tokens.
 * * @param ctx Output context matrix (size T x H).
 * @param q Query matrix (size T x H).
 * @param k Key matrix (size T x kvDim).
 * @param v Value matrix (size T x kvDim).
 * @param T Sequence length (number of tokens).
 * @param H Hidden dimension size.
 * @param nHeads Number of Attention Heads for Queries.
 * @param nKvHeads Number of Attention Heads for Keys/Values (enables GQA/MQA).
 * @param scores Output buffer for raw Q*K^T logits (size nHeads x T x T).
 * @param probs Output buffer for post-softmax probabilities (size nHeads x T x T).
 * @param attentionMask Pointer to the padding mask array (0 to ignore, 1 to keep) (size T).
 */
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
    const int* attentionMask);

/**
 * @brief Computes gradients for Causal Attention (Backward Pass).
 * Backpropagates the gradient from the context output (dCtx) to the
 * Query (dQ), Key (dK), and Value (dV) inputs.
 * * @param dQ Output gradient for Queries (size T x H).
 * @param dK Output gradient for Keys (size T x kvDim).
 * @param dV Output gradient for Values (size T x kvDim).
 * @param dCtx Incoming gradient from the next layer (size T x H).
 * @param q Original Query matrix from forward pass.
 * @param k Original Key matrix from forward pass.
 * @param v Original Value matrix from forward pass.
 * @param T Sequence length.
 * @param H Hidden dimension size.
 * @param nHeads Number of Attention Heads for Queries.
 * @param nKvHeads Number of Attention Heads for Keys/Values.
 * @param probs Original attention probabilities from forward pass (size nHeads x T x T).
 * @param attentionMask Pointer to the padding mask array (size T).
 * @param tmp_dP Scratch buffer for intermediate softmax gradient calculations (size T).
 */
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
    float* __restrict tmp_dP);
