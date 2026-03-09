#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// ============================================================
// BitNet I2S utilities
//
// I2S = int2 weights + int8 activations
//
// Weight encoding:
//   00 = -1
//   01 =  0
//   10 = +1
//   11 = unused
// ============================================================


// ------------------------------------------------------------
// Ternary encoding helpers
// ------------------------------------------------------------

inline uint8_t ternary_to_i2_code(int8_t q) noexcept
{
    return (q == -1) ? 0u :
        (q == 0) ? 1u :
        2u;
}

inline int8_t i2_code_to_ternary(uint8_t code) noexcept
{
    code &= 0x3u;

    if (code == 0u) return -1;
    if (code == 1u) return  0;
    if (code == 2u) return +1;
    return 0;
}


// ------------------------------------------------------------
// Pack / unpack
// ------------------------------------------------------------

void pack_ternary_to_2bit(
    std::vector<uint8_t>& out_q2,
    const int8_t* q,
    std::size_t N
);

void unpack_2bit_to_ternary(
    std::vector<int8_t>& out_q,
    const uint8_t* q2,
    std::size_t N
);


// ------------------------------------------------------------
// Activation quantization
//
// Quantizes X[t,h] into int8 tiles.
//
// xScale layout:
//   xScale[tb * nHBlocks + kb]
//
// where:
//   tb = tile over token rows
//   kb = tile over hidden/K dimension
// ------------------------------------------------------------

void quantize_x_tile_i8_tiled(
    int8_t* __restrict Xi8,
    float* __restrict xScale,
    const float* __restrict X,
    std::size_t T,
    std::size_t H,
    std::size_t t0,
    std::size_t tCount,
    std::size_t tBlock,
    std::size_t hBlock
);


// ------------------------------------------------------------
// Core I2S dot kernel
//
// Computes one 32-column output tile:
//
//   acc32[jj] += sum_di W[i0+di, j0+jj] * x8[di]
//
// Requirements:
//   - j0 must be multiple of 32
//   - hCount is the active K span
// ------------------------------------------------------------

void dot_i2s_i8_accum_i32_block32(
    int32_t* __restrict acc32,
    const uint8_t* __restrict Wq2,
    std::size_t H,
    std::size_t i0,
    std::size_t hCount,
    const int8_t* __restrict x8,
    std::size_t j0
);


// ------------------------------------------------------------
// Packed-load helpers for Option C backward assist
//
// These helpers allow a backward kernel to read transpose-style
// weights directly from the forward-packed q2 buffer using
// precomputed metadata.
//
// q2 packs 4 weights per byte:
//
//   bits 0-1 : element 0
//   bits 2-3 : element 1
//   bits 4-5 : element 2
//   bits 6-7 : element 3
// ------------------------------------------------------------

inline uint8_t bitnet_i2s_load_code(
    const uint8_t* q2,
    std::size_t byteIdx,
    uint8_t shift) noexcept
{
    return static_cast<uint8_t>((q2[byteIdx] >> shift) & 0x3u);
}

inline int8_t bitnet_i2s_load_value(
    const uint8_t* q2,
    std::size_t byteIdx,
    uint8_t shift) noexcept
{
    return i2_code_to_ternary(bitnet_i2s_load_code(q2, byteIdx, shift));
}

inline int8_t bitnet_i2s_load_from_assist(
    const uint8_t* q2,
    const uint32_t* byteIndex,
    const uint8_t* bitShift,
    std::size_t t) noexcept
{
    return i2_code_to_ternary(
        static_cast<uint8_t>((q2[byteIndex[t]] >> bitShift[t]) & 0x3u)
    );
}