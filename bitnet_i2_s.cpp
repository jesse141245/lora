#include "bitnet_i2_s.h"

#include <stdexcept>

void pack_ternary_to_2bit(std::vector<uint8_t>& out_q2, const int8_t* q, std::size_t N) {
    if (!q) throw std::runtime_error("pack_ternary_to_2bit: q is null");
    out_q2.assign((N + 3) / 4, 0);
    for (std::size_t i = 0; i < N; ++i) {
        out_q2[i >> 2] |= static_cast<uint8_t>((ternary_to_i2_code(q[i]) & 0x3u) << ((i & 3u) * 2u));
    }
}

void unpack_2bit_to_ternary(std::vector<int8_t>& out_q, const uint8_t* q2, std::size_t N) {
    if (!q2) throw std::runtime_error("unpack_2bit_to_ternary: q2 is null");
    out_q.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        out_q[i] = i2_code_to_ternary(static_cast<uint8_t>((q2[i >> 2] >> ((i & 3u) * 2u)) & 0x3u));
    }
}
