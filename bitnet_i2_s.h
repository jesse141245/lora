#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

inline uint8_t ternary_to_i2_code(int8_t q) noexcept {
    return (q == -1) ? 0u : (q == 0) ? 1u : 2u;
}

inline int8_t i2_code_to_ternary(uint8_t code) noexcept {
    code &= 0x3u;
    if (code == 0u) return -1;
    if (code == 1u) return 0;
    if (code == 2u) return +1;
    return 0;
}

// ------------------------------------------------------------
// pack_ternary_to_2bit
// ------------------------------------------------------------
// Packs an array of ternary values (-1, 0, 1) into a dense 2-bit 
// byte array. 4 weights are stored per byte to reduce memory footprint.
void pack_ternary_to_2bit(std::vector<uint8_t>& out_q2, const int8_t* q, std::size_t N);

// ------------------------------------------------------------
// unpack_2bit_to_ternary
// ------------------------------------------------------------
// Decodes a dense 2-bit byte array back into standard int8_t 
// ternary values (-1, 0, 1).
void unpack_2bit_to_ternary(std::vector<int8_t>& out_q, const uint8_t* q2, std::size_t N);

// ------------------------------------------------------------
// quantize_x_tile_i8_tiled
// ------------------------------------------------------------
// Dynamically quantizes a floating-point input activation matrix 
// tile into int8_t. Computes and stores the absolute maximum scale 
// for each block to allow symmetric integer dot products.
void quantize_x_tile_i8_tiled(
    int8_t* __restrict Xi8, float* __restrict xScale, const float* __restrict X,
    std::size_t T, std::size_t H, std::size_t t0, std::size_t tCount,
    std::size_t tBlock, std::size_t hBlock
);

// ------------------------------------------------------------
// dot_i2s_i8_accum_i32_block32
// ------------------------------------------------------------
// Computes the dot product of 1 row of int8 activations 
// against a 32-column block of 2-bit packed weights.
void dot_i2s_i8_accum_i32_block32(
    int32_t* __restrict acc32, const uint8_t* __restrict Wq2, std::size_t stride_cols,
    std::size_t i0, std::size_t hCount, const int8_t* __restrict x8, std::size_t j0
);

// ------------------------------------------------------------
// dot_i2s_i8_accum_i32_block32_x2
// ------------------------------------------------------------
// Computes the dot product of 2 rows of int8 activations 
// against a 32-column block of 2-bit packed weights simultaneously,
// maximizing weight-decode reuse.
void dot_i2s_i8_accum_i32_block32_x2(
    int32_t* __restrict accA, int32_t* __restrict accB, const uint8_t* __restrict Wq2,
    std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B, std::size_t j0
);

// ------------------------------------------------------------
// dot_i2s_i8_accum_i32_block32_x4
// ------------------------------------------------------------
// High-throughput micro-kernel. Computes the dot product of 4 rows 
// of int8 activations against a 32-column block of 2-bit packed weights. 
// Splits the 32 columns into two 16-column passes internally to perfectly 
// utilize the 16 AVX2 vector registers and prevent memory spilling.
void dot_i2s_i8_accum_i32_block32_x4(
    int32_t* __restrict accA, int32_t* __restrict accB,
    int32_t* __restrict accC, int32_t* __restrict accD,
    const uint8_t* __restrict Wq2, std::size_t stride_cols, std::size_t i0, std::size_t hCount,
    const int8_t* __restrict x8A, const int8_t* __restrict x8B,
    const int8_t* __restrict x8C, const int8_t* __restrict x8D,
    std::size_t j0
);