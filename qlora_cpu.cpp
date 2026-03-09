// cpu_qlora_trainer.cpp
#include "qlora_cpu.h"
#include "lion_optimizer.h"
#include "cpuKernelBack.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <numeric>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>

// Helper

static inline float logsumexp(const float* x, std::size_t n) {
    float m = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, x[i]);
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) s += std::exp(x[i] - m);
    return m + std::log(s);
}

// Dataset

Dataset::Dataset(const std::string& path) {
    // TODO: load tokenized dataset from disk
    (void)path;
}

std::size_t Dataset::size() const noexcept {
    return samples_.size();
}

const TrainingSample& Dataset::get(std::size_t idx) const {
    return samples_.at(idx);
}



// Checkpointer

Checkpointer::Checkpointer(const std::string& dir)
    : dir_{ dir } {
    std::filesystem::create_directories(dir_);
}

std::optional<CheckpointState>
Checkpointer::loadLatest(CpuKernelBackend& backend,
    LionOptimizer& opt) {
    // TODO:
    //  - list files in dir_
    //  - pick latest by step/mtime
    //  - load LoRA + optimizer + metadata (step, epoch, samplerOffset)
    (void)backend;
    (void)opt;
    return std::nullopt;
}

void Checkpointer::save(const CpuKernelBackend& backend,
    const LionOptimizer& opt,
    const CheckpointState& state) {
    // TODO:
    //  - backend.saveLoRA(...)
    //  - opt.save(...)
    //  - write small metadata (state)
    (void)backend;
    (void)opt;
    (void)state;
}

// ProgressReporter

void ProgressReporter::onStep(
    int step,
    float loss,
    int maxSteps,
    int logInterval,
    const std::chrono::steady_clock::time_point& start) {

    if (step <= 0 || maxSteps <= 0) return;

    if (logInterval <= 0) return;              // 0 disables logging
    if (step % logInterval != 0) return;

    const auto now = std::chrono::steady_clock::now();
    const auto elapsedSec =
        std::chrono::duration_cast<std::chrono::seconds>(now - start).count();

    const float stepsPerSec =
        (elapsedSec > 0)
        ? static_cast<float>(step) / static_cast<float>(elapsedSec)
        : 0.0f;

    const float remainingSteps = static_cast<float>(maxSteps - step);
    const float etaSec = (stepsPerSec > 0.0f)
        ? remainingSteps / stepsPerSec
        : -1.0f;

    std::cout << "[Step " << step << "/" << maxSteps << "] "
        << "loss=" << loss
        << " | steps/s=" << stepsPerSec
        << " | ETA(s)=" << etaSec << '\n';
}


// CpuQLoRATrainer

CpuQLoRATrainer::CpuQLoRATrainer(const TrainerConfig& cfg,
    Dataset& dataset,
    CpuKernelBackend& backend,
    LionOptimizer& optimizer)
    : cfg_{ cfg }
    , dataset_{ dataset }
    , backend_{ backend }
    , optimizer_{ optimizer }
    , checkpointer_{ cfg.checkpointDir } {
}

std::vector<std::size_t> CpuQLoRATrainer::makeEpochOrder() const {
    std::vector<std::size_t> order(dataset_.size());
    std::iota(order.begin(), order.end(), 0);

    if (cfg_.seed != 0) {
        std::mt19937 rng{ cfg_.seed + static_cast<unsigned int>(epoch_) };
        std::shuffle(order.begin(), order.end(), rng);
    }

    return order;
}

std::vector<const TrainingSample*>
CpuQLoRATrainer::sampleBatch(int batchSize) {
    std::vector<const TrainingSample*> batch;
    if (dataset_.size() == 0 || batchSize <= 0) return batch;

    if (epochOrder_.empty()) {
        epochOrder_ = makeEpochOrder();
        samplerOffset_ = 0;
        ++epoch_;
    }

    batch.reserve(static_cast<std::size_t>(batchSize));

    for (int i = 0; i < batchSize; ++i) {
        if (samplerOffset_ >= epochOrder_.size()) {
            epochOrder_ = makeEpochOrder();
            samplerOffset_ = 0;
            ++epoch_;
        }
        const auto idx = epochOrder_[samplerOffset_++];
        batch.push_back(&dataset_.get(idx));
    }

    return batch;
}

void CpuQLoRATrainer::trainStep(
    const std::vector<const TrainingSample*>& batch,
    float& accumLoss,
    std::int64_t& accumTokens)
{
    const std::size_t V = backend_.shape().vocabSize;

    for (const auto* sample : batch) {
        const std::size_t T = sample->inputIds.size();
        if (sample->attentionMask.size() != T || sample->targetIds.size() != T) {
            throw std::runtime_error("trainStep: sample sizes mismatch");
        }

        const std::size_t needed = backend_.logitsCount(T);
        logitsBuf_.assign(needed, 0.0f);
        dLogitsBuf_.assign(needed, 0.0f);

        // 1) Forward
        backend_.forward(
            sample->inputIds,
            sample->attentionMask,
            std::span<float>(logitsBuf_.data(), logitsBuf_.size())
        );

        // 2) Loss + dLogits (tokenwise CE)
        float sampleLossSum = 0.0f;
        std::int64_t sampleTokenCount = 0;

        for (std::size_t t = 0; t < T; ++t) {
            if (sample->attentionMask[t] == 0) continue;

            const int y = sample->targetIds[t];
            if (y == cfg_.ignoreIndex) continue;
            if (y < 0 || static_cast<std::size_t>(y) >= V) continue;

            const float* logits_t = &logitsBuf_[t * V];
            float* dlog_t = &dLogitsBuf_[t * V];

            // stable log softmax normalizer
            const float lse = logsumexp(logits_t, V);
            sampleLossSum += (lse - logits_t[static_cast<std::size_t>(y)]);
            ++sampleTokenCount;

            // dlogits = softmax - onehot(y)
            for (std::size_t v = 0; v < V; ++v) {
                const float p = std::exp(logits_t[v] - lse);
                dlog_t[v] = p;
            }
            dlog_t[static_cast<std::size_t>(y)] -= 1.0f;
        }

        accumLoss += sampleLossSum;
        accumTokens += sampleTokenCount;

        // normalize per-token so LR is less seqLen-dependent
        if (sampleTokenCount > 0) {
            const float inv = 1.0f / static_cast<float>(sampleTokenCount);
            for (float& g : dLogitsBuf_) g *= inv;
        }

        // 3) Backward (accumulates into backend LoRA grads)
        backend_.backward(sample->inputIds, sample->attentionMask,
            std::span<const float>(dLogitsBuf_.data(), dLogitsBuf_.size()));
    }
}


void CpuQLoRATrainer::maybeEval(int step) {
    if (cfg_.evalInterval <= 0 || cfg_.numEvalBatches <= 0) return;
    if (step % cfg_.evalInterval != 0) return;
    // TODO: run small eval loop, compute perplexity / loss on held-out set
}

void CpuQLoRATrainer::maybeCheckpoint(int step,
    const CheckpointState& state) {
    if (cfg_.checkpointInterval <= 0) return;
    if (step % cfg_.checkpointInterval != 0 && step != cfg_.maxSteps) return;
    checkpointer_.save(backend_, optimizer_, state);
}

void CpuQLoRATrainer::run() {
    const auto startTime = std::chrono::steady_clock::now();

    CheckpointState state{};
    state.step = 0;
    state.epoch = 0;
    state.samplerOffset = 0;

    if (cfg_.resumeFromCheckpoint) {
        if (auto loaded = checkpointer_.loadLatest(backend_, optimizer_)) {
            state = *loaded;
            epoch_ = state.epoch;
            samplerOffset_ = state.samplerOffset;
            epochOrder_ = makeEpochOrder();
            if (samplerOffset_ > epochOrder_.size()) samplerOffset_ = epochOrder_.size();
            std::cout << "Resuming from step " << state.step
                << ", epoch " << state.epoch << '\n';
        }
    }

    int globalStep = state.step;

    backend_.zeroGrad();
    float accumLoss = 0.0f;
    std::int64_t accumTokens = 0;
    int accumCount = 0;

    auto params = backend_.collectTrainableParams();
    optimizer_.init(params);

    while (globalStep < cfg_.maxSteps) {
        const auto batch = sampleBatch(cfg_.batchSize);
        if (batch.empty()) break;

        // IMPORTANT: trainStep MUST run forward+loss+backward and accumulate into backend grads
        trainStep(batch, accumLoss, accumTokens);
        ++accumCount;

        if (accumCount < cfg_.gradAccumSteps) continue;

        if (cfg_.maxGradNorm > 0.0f) {
            optimizer_.clipGradNorm(params, cfg_.maxGradNorm);
        }

        // Optional: if you want true grad-accum semantics:
        // optimizer_.scaleGrads(params, 1.0f / cfg_.gradAccumSteps); (not shown)

        optimizer_.step(params, /*zeroGrads=*/true);

        ++globalStep;

        const float avgLoss = (accumTokens > 0)
            ? (accumLoss / static_cast<float>(accumTokens))
            : 0.0f;

        reporter_.onStep(globalStep, avgLoss, cfg_.maxSteps, cfg_.logInterval, startTime);

        accumLoss = 0.0f;
        accumTokens = 0;
        accumCount = 0;

        state.step = globalStep;
        state.epoch = epoch_;
        state.samplerOffset = samplerOffset_;

        maybeEval(globalStep);
        maybeCheckpoint(globalStep, state);
    }

    backend_.saveLoRA(cfg_.loraOutputPath);

    state.step = globalStep;
    state.epoch = epoch_;
    state.samplerOffset = samplerOffset_;

    checkpointer_.save(backend_, optimizer_, state);

    std::cout << "Training complete. LoRA adapter saved to " << cfg_.loraOutputPath << '\n';
}

// -----------------------------
// Helper entrypoint
// -----------------------------

int runCpuQLoRATrainer(const TrainerConfig& cfg,
    const std::string& datasetPath) {
    try {
        Dataset dataset{ datasetPath };
        CpuKernelBackend backend{ cfg };
        LionOptimizer optimizer{  };
        CpuQLoRATrainer trainer{ cfg, dataset, backend, optimizer };
        trainer.run();
        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Fatal error in runCpuQLoRATrainer: " << ex.what() << '\n';
        return 1;
    }
}

