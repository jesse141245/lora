#pragma once

#include <cstddef>

// Computes one token row of:
//   loss = logsumexp(logits) - logits[y]
//   dlog = softmax(logits) - onehot(y)
//
// Arguments:
//   logits : pointer to logits row of length V
//   dlog   : output buffer of length V
//   V      : vocab size
//   y      : target class index
//
// Returns:
//   cross-entropy loss for this row
float cross_entropy_dlogits_row(
    const float* logits,
    float* dlog,
    std::size_t V,
    int y);

// In-place scale helper used by trainer for token-count normalization.
void scale_inplace_avx2_or_scalar(
    float* x,
    std::size_t n,
    float scale);