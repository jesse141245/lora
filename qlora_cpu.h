#pragma once
// cpu_qlora_trainer.h
// C++20 outline for CPU-only BitNet QLoRA training


#include <chrono>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include "config.h"
#include "modelShape.h"
#include "cpuKernelBack.h"
#include "lion_optimizer.h"
struct TrainingSample {
    std::vector<int> inputIds;
    std::vector<int> targetIds;
    std::vector<int> attentionMask; // 1 = real token, 0 = pad
};

class Dataset {
public:
    explicit Dataset(const std::string& path);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const TrainingSample& get(std::size_t idx) const;

private:
    std::vector<TrainingSample> samples_;
};



struct CheckpointState {
    int step = 0;
    int epoch = 0;
    std::size_t samplerOffset = 0;  // where we are in dataset / permutation
    // Extend with optimizer moments checksum, config hash, etc.
};


class Checkpointer {
public:
    explicit Checkpointer(const std::string& dir);

    std::optional<CheckpointState> loadLatest(CpuKernelBackend& backend,
        LionOptimizer& opt);
    void save(const CpuKernelBackend& backend,
        const LionOptimizer& opt,
        const CheckpointState& state);

private:
    std::string dir_;
};

class ProgressReporter {
public:
    void onStep(int step,
        float loss,
        int maxSteps,
        int logInterval,
        const std::chrono::steady_clock::time_point& start);

};

class CpuQLoRATrainer {
public:
    CpuQLoRATrainer(const TrainerConfig& cfg,
        Dataset& dataset,
        CpuKernelBackend& backend,
        LionOptimizer& optimizer);

    void run();

private:
    std::vector<std::size_t> makeEpochOrder() const;
    std::vector<const TrainingSample*> sampleBatch(int batchSize);

    void trainStep(const std::vector<const TrainingSample*>& batch,
        float& accumLoss,
        std::int64_t& accumTokens);

    void maybeEval(int step);
    void maybeCheckpoint(int step, const CheckpointState& state);

    TrainerConfig cfg_;
    Dataset& dataset_;
    CpuKernelBackend& backend_;
    LionOptimizer& optimizer_;
    Checkpointer checkpointer_;
    ProgressReporter reporter_;

    std::vector<float> logitsBuf_;
    std::vector<float> dLogitsBuf_;
    std::vector<float> softmaxTmp_; // optional

    std::vector<std::size_t> epochOrder_;
    std::size_t samplerOffset_ = 0;
    int epoch_ = 0;
};

int runCpuQLoRATrainer(const TrainerConfig& cfg,
    const std::string& datasetPath);

    