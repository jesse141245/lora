#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "checkpoint.h"
#include "config.h"
#include "cpuKernelBack.h"
#include "dataset.h"
#include "adamw_optimizer.h"
#include "lion_optimizer.h"

class ProgressReporter {
public:
    void onStep(
        int step,
        float loss,
        float learningRate,
        std::int64_t supervisedTokens,
        int maxSteps,
        int logInterval,
        const std::chrono::steady_clock::time_point& start,
        int epoch,
        const std::chrono::steady_clock::time_point& epochStart
    );

private:
    std::deque<float> lossWindow8_;
    std::deque<float> lossWindow32_;
    float lossSum8_ = 0.0f;
    float lossSum32_ = 0.0f;
};

class CpuQLoRATrainer {
public:
    CpuQLoRATrainer(
        const TrainerConfig& cfg,
        Dataset& dataset,
        CpuKernelBackend& backend,
        AdamWOptimizer& optimizer
    );
    CpuQLoRATrainer(
        const TrainerConfig& cfg,
        Dataset& dataset,
        CpuKernelBackend& backend,
        LionOptimizer& optimizer
    );

    void run();

private:
    std::vector<std::size_t> makeEpochOrder() const;
    std::vector<const TrainingSample*> sampleBatch(int batchSize);

    void trainStep(
        const std::vector<const TrainingSample*>& batch,
        float& accumLoss,
        std::int64_t& accumTokens
    );

    void maybeEval(int step);
    void maybeCheckpoint(int step, const CheckpointState& state);

private:
    bool usingAdamW() const noexcept { return adamOptimizer_ != nullptr; }

private:
    TrainerConfig cfg_;
    Dataset& dataset_;
    CpuKernelBackend& backend_;
    AdamWOptimizer* adamOptimizer_ = nullptr;
    LionOptimizer* lionOptimizer_ = nullptr;
    Checkpointer checkpointer_;
    ProgressReporter reporter_;

    std::vector<float> logitsBuf_;
    std::vector<float> dLogitsBuf_;

    std::vector<std::size_t> epochOrder_;
    std::size_t samplerOffset_ = 0;
    int epoch_ = 0;
    std::chrono::steady_clock::time_point epochStartTime_{};
};

int runCpuQLoRATrainer(
    const TrainerConfig& cfg,
    const std::string& datasetPath
);
