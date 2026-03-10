#pragma once

#include <cstddef>
#include <optional>
#include <string>

class CpuKernelBackend;
class LionOptimizer;

struct CheckpointState {
    int step = 0;
    int epoch = 0;
    std::size_t samplerOffset = 0;
};

class Checkpointer {
public:
    explicit Checkpointer(const std::string& dir);

    std::optional<CheckpointState> loadLatest(
        CpuKernelBackend& backend,
        LionOptimizer& opt
    );

    void save(
        const CpuKernelBackend& backend,
        const LionOptimizer& opt,
        const CheckpointState& state
    );

private:
    std::string dir_;
};