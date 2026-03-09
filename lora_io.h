#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "lora_type.h"   // contains LayerLoRA, LoraDType, etc.

// File dtypes (expand later when you add fp16/int8/packed bits)
enum class LoraFileDType : uint32_t {
    F32 = 0,
};

// Which tensors are present in the file
enum LoraFileFlags : uint32_t {
    HAS_AQ = 1u << 0,
    HAS_BQ = 1u << 1,
    HAS_AV = 1u << 2,
    HAS_BV = 1u << 3,
};

#pragma pack(push, 1)
struct LoraFileHeaderV1 {
    char     magic[4];     // "LORA"
    uint32_t version;      // 1
    uint32_t dtype;        // LoraFileDType
    uint32_t flags;        // LoraFileFlags
    uint32_t nLayers;
    uint32_t hiddenSize;   // H
    uint32_t rank;         // R
    float    loraAlpha;    // store alpha (or store effective scaling if you prefer)
    uint32_t reserved[8];  // future: layoutId, checksum, etc.
};
#pragma pack(pop)

// Saves LoRA adapter to a single binary file.
// Expects each tensor present in `flags` to be exactly size (R*H).
void saveLoRA_binary_v1(
    const std::string& path,
    const std::vector<LayerLoRA>& loras,
    uint32_t hiddenSizeH,
    uint32_t rankR,
    float loraAlpha,
    uint32_t flags = HAS_AQ | HAS_BQ | HAS_AV | HAS_BV);

// Loads LoRA adapter from binary file.
// Validates that file shape matches expected (layers/H/R).
// Resizes loras_out and the contained vectors.
// Optionally returns alpha from file.
void loadLoRA_binary_v1(
    const std::string& path,
    std::vector<LayerLoRA>& loras_out,
    uint32_t expectedLayers,
    uint32_t expectedHiddenH,
    uint32_t expectedRankR,
    float* loraAlpha_out = nullptr);