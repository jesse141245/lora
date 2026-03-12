#include "checkpoint.h"

#include "cpuKernelBack.h"
#include "lion_optimizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

Checkpointer::Checkpointer(const std::string& dir)
    : dir_{ dir } {
    fs::create_directories(dir_);
}

std::optional<CheckpointState>
Checkpointer::loadLatest(CpuKernelBackend& backend, LionOptimizer& opt) {

    int latest_step = -1;
    std::string latest_file = "";

    // 1. Scan directory for the highest checkpoint step
    if (fs::exists(dir_)) {
        for (const auto& entry : fs::directory_iterator(dir_)) {
            if (entry.is_regular_file()) {
                std::string name = entry.path().filename().string();
                // Expect format: "checkpoint_1234.bin"
                if (name.starts_with("checkpoint_") && name.ends_with(".bin")) {
                    std::string num_str = name.substr(11, name.size() - 15);
                    try {
                        int step = std::stoi(num_str);
                        if (step > latest_step) {
                            latest_step = step;
                            latest_file = entry.path().string();
                        }
                    }
                    catch (...) {
                        // Ignore malformed files
                    }
                }
            }
        }
    }

    if (latest_step == -1 || latest_file.empty()) {
        return std::nullopt; // No checkpoints found
    }

    std::ifstream in(latest_file, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[Warning] Found checkpoint " << latest_file << " but could not open it.\n";
        return std::nullopt;
    }

    // 2. Load Metadata
    CheckpointState state;
    in.read(reinterpret_cast<char*>(&state.step), sizeof(state.step));
    in.read(reinterpret_cast<char*>(&state.epoch), sizeof(state.epoch));
    in.read(reinterpret_cast<char*>(&state.samplerOffset), sizeof(state.samplerOffset));

    // 3. Initialize arrays to receive data
    auto params = backend.collectTrainableParams();
    opt.init(params);
    auto& m = opt.getMomentum();

    // 4. Validate Parameter Topology
    std::size_t num_params = 0;
    in.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
    if (num_params != params.size()) {
        throw std::runtime_error("Checkpoint loading failed: Architecture mismatch (number of trainable params changed).");
    }

    // 5. Load binary arrays
    for (std::size_t k = 0; k < num_params; ++k) {
        std::size_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));

        if (n != params[k].n || n != m[k].size()) {
            throw std::runtime_error("Checkpoint loading failed: Parameter shape mismatch.");
        }

        in.read(reinterpret_cast<char*>(params[k].w), n * sizeof(float));
        in.read(reinterpret_cast<char*>(m[k].data()), n * sizeof(float));
    }

    return state;
}

void Checkpointer::save(const CpuKernelBackend& backend,
    const LionOptimizer& opt,
    const CheckpointState& state) {

    std::string path = dir_ + "/checkpoint_" + std::to_string(state.step) + ".bin";
    std::ofstream out(path, std::ios::binary);

    if (!out.is_open()) {
        throw std::runtime_error("Failed to create checkpoint file at " + path);
    }

    // 1. Write Metadata
    out.write(reinterpret_cast<const char*>(&state.step), sizeof(state.step));
    out.write(reinterpret_cast<const char*>(&state.epoch), sizeof(state.epoch));
    out.write(reinterpret_cast<const char*>(&state.samplerOffset), sizeof(state.samplerOffset));

    // 2. Fetch Trainable parameters
    // We cast away const to access the float* pointers, but we only READ from them.
    auto* mutable_backend = const_cast<CpuKernelBackend*>(&backend);
    auto* mutable_opt = const_cast<LionOptimizer*>(&opt);

    auto params = mutable_backend->collectTrainableParams();
    auto& m = mutable_opt->getMomentum();

    // 3. Write Parameter Topology
    std::size_t num_params = params.size();
    out.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

    // 4. Dump raw arrays
    for (std::size_t k = 0; k < num_params; ++k) {
        std::size_t n = params[k].n;
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        out.write(reinterpret_cast<const char*>(params[k].w), n * sizeof(float));
        out.write(reinterpret_cast<const char*>(m[k].data()), n * sizeof(float));
    }

    std::cout << "[Checkpoint] Saved state to " << path << '\n';
}