#include <iostream>
#include "bench.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// MSVC aligned alloc helpers
static void* aligned_malloc64(std::size_t bytes) {
#ifdef _MSC_VER
    return _aligned_malloc(bytes, 64);
#else
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
    return p;
#endif
}
static void aligned_free64(void* p) {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
}

// Touch memory to ensure pages are committed (avoid page-fault timing noise)
static void touch_pages(uint8_t* p, std::size_t bytes) {
    constexpr std::size_t stride = 4096;
    for (std::size_t i = 0; i < bytes; i += stride) p[i] = uint8_t(i);
    if (bytes) p[bytes - 1] = 123;
}

static double seconds_since(const std::chrono::high_resolution_clock::time_point& t0,
    const std::chrono::high_resolution_clock::time_point& t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

void benchmark_memcpy_bandwidth(std::size_t bytes, int iters) {
    auto* src = static_cast<uint8_t*>(aligned_malloc64(bytes));
    auto* dst = static_cast<uint8_t*>(aligned_malloc64(bytes));
    if (!src || !dst) {
        std::cerr << "alloc failed\n";
        aligned_free64(src);
        aligned_free64(dst);
        return;
    }

    // Initialize and commit pages
    std::memset(src, 0xA5, bytes);
    std::memset(dst, 0x3C, bytes);
    touch_pages(src, bytes);
    touch_pages(dst, bytes);

    // Warmup (also helps CPU boost stabilize)
    for (int i = 0; i < 10; ++i) {
        std::memcpy(dst, src, bytes);
    }

    // Timed loop
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        std::memcpy(dst, src, bytes);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // Prevent dead-code elimination / overly clever optimizations
    volatile uint64_t sink = 0;
    sink += dst[0];
    sink += dst[bytes / 2];
    sink += dst[bytes - 1];

    const double sec = seconds_since(t0, t1);

    // Each memcpy reads bytes from src and writes bytes to dst => ~2*bytes traffic
    const double total_bytes_moved = double(iters) * double(bytes) * 2.0;
    const double gbps = (total_bytes_moved / sec) / 1e9;

    std::cout << "memcpy size=" << (bytes / (1024.0 * 1024.0)) << " MiB"
        << " iters=" << iters
        << " time=" << sec << " s"
        << " BW=" << gbps << " GB/s"
        << " (sink=" << sink << ")\n";

    aligned_free64(src);
    aligned_free64(dst);
}

int main() {
    // Try a few sizes. Use hundreds of MiB to force DRAM, not cache.
    benchmark_linear(512);
    benchmark_linear(1024);
    benchmark_linear(2048);
    benchmark_linear(4096);
    return 0;
}