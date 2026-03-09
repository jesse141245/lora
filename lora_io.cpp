#include "lora_io.h"

#include <fstream>
#include <stdexcept>
#include <cstring>   // memcpy, memcmp
#include <algorithm> // min

static void write_exact(std::ofstream& out, const void* data, size_t bytes) {
    out.write(reinterpret_cast<const char*>(data), std::streamsize(bytes));
    if (!out) throw std::runtime_error("LoRA write failed");
}

static void read_exact(std::ifstream& in, void* data, size_t bytes) {
    in.read(reinterpret_cast<char*>(data), std::streamsize(bytes));
    if (!in) throw std::runtime_error("LoRA read failed (EOF/corrupt file)");
}

static void require(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

static void check_tensor_size(const std::vector<float>& v, size_t elems, const char* name) {
    if (v.size() != elems) {
        throw std::runtime_error(std::string("LoRA tensor size mismatch for ") + name +
            " (got " + std::to_string(v.size()) +
            ", expected " + std::to_string(elems) + ")");
    }
}

void saveLoRA_binary_v1(
    const std::string& path,
    const std::vector<LayerLoRA>& loras,
    uint32_t H,
    uint32_t R,
    float loraAlpha,
    uint32_t flags)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Could not open LoRA file for write: " + path);

    LoraFileHeaderV1 hdr{};
    std::memcpy(hdr.magic, "LORA", 4);
    hdr.version = 1;
    hdr.dtype = uint32_t(LoraFileDType::F32);
    hdr.flags = flags;
    hdr.nLayers = uint32_t(loras.size());
    hdr.hiddenSize = H;
    hdr.rank = R;
    hdr.loraAlpha = loraAlpha;

    write_exact(out, &hdr, sizeof(hdr));

    const size_t elems = size_t(R) * size_t(H);
    const size_t bytes = elems * sizeof(float);

    auto dump = [&](const std::vector<float>& v, const char* name) {
        check_tensor_size(v, elems, name);
        write_exact(out, v.data(), bytes);
        };

    for (size_t L = 0; L < loras.size(); ++L) {
        const LayerLoRA& ll = loras[L];

        if (flags & HAS_AQ) dump(ll.Aq, "Aq");
        if (flags & HAS_BQ) dump(ll.Bq, "Bq");
        if (flags & HAS_AV) dump(ll.Av, "Av");
        if (flags & HAS_BV) dump(ll.Bv, "Bv");
    }
}

void loadLoRA_binary_v1(
    const std::string& path,
    std::vector<LayerLoRA>& loras_out,
    uint32_t expectedLayers,
    uint32_t expectedH,
    uint32_t expectedR,
    float* loraAlpha_out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Could not open LoRA file for read: " + path);

    LoraFileHeaderV1 hdr{};
    read_exact(in, &hdr, sizeof(hdr));

    require(std::memcmp(hdr.magic, "LORA", 4) == 0, "Bad LoRA magic");
    require(hdr.version == 1, "Unsupported LoRA version");
    require(hdr.dtype == uint32_t(LoraFileDType::F32), "Unsupported LoRA dtype (expected F32)");

    // Strict shape checks (recommended)
    require(hdr.nLayers == expectedLayers, "LoRA nLayers mismatch");
    require(hdr.hiddenSize == expectedH, "LoRA hiddenSize mismatch");
    require(hdr.rank == expectedR, "LoRA rank mismatch");

    if (loraAlpha_out) *loraAlpha_out = hdr.loraAlpha;

    const size_t elems = size_t(hdr.rank) * size_t(hdr.hiddenSize);
    const size_t bytes = elems * sizeof(float);

    loras_out.clear();
    loras_out.resize(hdr.nLayers);

    auto load = [&](std::vector<float>& v) {
        v.resize(elems);
        read_exact(in, v.data(), bytes);
        };

    for (size_t L = 0; L < loras_out.size(); ++L) {
        LayerLoRA& ll = loras_out[L];

        if (hdr.flags & HAS_AQ) load(ll.Aq); else ll.Aq.clear();
        if (hdr.flags & HAS_BQ) load(ll.Bq); else ll.Bq.clear();
        if (hdr.flags & HAS_AV) load(ll.Av); else ll.Av.clear();
        if (hdr.flags & HAS_BV) load(ll.Bv); else ll.Bv.clear();
    }

    // Optional: ensure we consumed the file exactly (helps catch trailing junk)
    // If you want strictness:
    char extra;
    require(!in.read(&extra, 1), "Trailing bytes found in LoRA file");
}