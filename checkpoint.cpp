#include "checkpoint.h"

#include "cpuKernelBack.h"
#include "lion_optimizer.h"

#include <filesystem>

Checkpointer::Checkpointer(const std::string& dir)
    : dir_{ dir } {
    std::filesystem::create_directories(dir_);
}

std::optional<CheckpointState>
Checkpointer::loadLatest(CpuKernelBackend& backend,
    LionOptimizer& opt) {
    (void)backend;
    (void)opt;

    // TODO:
    //  - enumerate checkpoint files in dir_
    //  - select latest
    //  - load backend LoRA state
    //  - load optimizer state
    //  - load metadata into CheckpointState
    return std::nullopt;
}

void Checkpointer::save(const CpuKernelBackend& backend,
    const LionOptimizer& opt,
    const CheckpointState& state) {
    (void)backend;
    (void)opt;
    (void)state;

    // TODO:
    //  - save backend LoRA state
    //  - save optimizer state
    //  - save metadata file
}