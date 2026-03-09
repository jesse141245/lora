#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include "kernels.h"
inline void benchmark_linear(int H) {
    std::vector<float> x(H), y(H), W(H * H);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : x) v = dist(rng);
    for (auto& v : W) v = dist(rng);

    int iters = 200;

    // Warmup
    for (int i = 0; i < 50; ++i)
        linear_HH(y.data(), x.data(), W.data(), H, 128);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iters; ++i)
        linear_HH(y.data(), x.data(), W.data(), H, 128);

    auto end = std::chrono::high_resolution_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    double flops = 2.0 * H * H * iters;
    double gflops = flops / seconds / 1e9;

    std::cout << "H=" << H
        << " time=" << seconds
        << " GFLOPs=" << gflops
        << "\n";
}