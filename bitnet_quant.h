#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include "bitnet_weight.h"

// ============================================================
// BitNet Quantization Utilities
// ============================================================
//
// Legacy path:
//   - BitNetMat
//   - explicit ternary {-1,0,1}
//   - per-block scale
//
// New I2S tiled path (Option C):
//   - BitNetMatI2S
//   - packed 2-bit forward storage
//   - per-(jq x qk) forward weight tile scale
//   - compact backward assist metadata
//
// No full transpose-packed q2T / w_scale_tileT are stored.
// ============================================================


// ------------------------------------------------------------
// Legacy quantization: float matrix -> ternary blockwise BitNetMat
// ------------------------------------------------------------
void quantize_ternary_blockwise(
    BitNetMat& out,
    const float* W,
    std::size_t H,
    std::size_t block
);


// ------------------------------------------------------------
// I2S tiled quantization: float matrix -> packed BitNetMatI2S
//
// W is [H x H] row-major.
//
// Tile sizes:
//   qk = input/K tile height   (recommended 128)
//   jq = output-column tile    (recommended 32)
//
// Forward tiled scale layout:
//   out.w_scale_tile[jb * nKBlocks + kb]
//
// where:
//   kb = row/input tile index over i
//   jb = output-column tile index over j
//
// Each scale covers a tile:
//   rows [kb*qk, min(H, (kb+1)*qk))
//   cols [jb*jq, min(H, (jb+1)*jq))
//
// This builder populates:
//   out.q2
//   out.w_scale_tile
//
// It may also initialize compact backward assist metadata in out.bwd
// for Option C kernels.
// ------------------------------------------------------------
void quantize_ternary_i2s_blockwise(
    BitNetMatI2S& out,
    const float* W,
    std::size_t H,
    std::size_t qk,
    std::size_t jq
);


// ------------------------------------------------------------
// Legacy dequantization helper: BitNetMat -> float matrix
// ------------------------------------------------------------
void dequant_to_float(
    float* Wout,
    const BitNetMat& Wq
);


// ------------------------------------------------------------
// Legacy AVX helpers for old BitNetMat path
// ------------------------------------------------------------
__m256 bitnet_dequant8_ps(
    const BitNetMat& M,
    std::size_t idx0
);

__m256 bitnet_dequant8_ps_twoScale(
    const BitNetMat& M,
    std::size_t idx0
);

float hsum256_ps(__m256 v);