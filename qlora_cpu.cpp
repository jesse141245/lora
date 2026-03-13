#include "qlora_cpu.h"
#include "loss_ce.h"
#include "kernels.h" // For scaling function

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <filesystem>
#include <cmath> // For std::sqrt

void ProgressReporter::onStep(
    int step,
    float loss,
    int maxSteps,
    int logInterval,
    const std::chrono::steady_clock::time_point& start) {

    if (step <= 0 || maxSteps <= 0) return;
    if (logInterval <= 0) return;
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

std::vector<const TrainingSample*> CpuQLoRATrainer::sampleBatch(int batchSize) {
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

        backend_.forward(
            sample->inputIds,
            sample->attentionMask,
            std::span<float>(logitsBuf_.data(), logitsBuf_.size())
        );

        float sampleLossSum = 0.0f;
        std::int64_t sampleTokenCount = 0;

        // dataset.cpp buildDefaultTargets already sets targetIds[t] = inputIds[t+1].
        // logitsOut[t*V] is the prediction from token t, so it pairs with targetIds[t].
        // Using targetIds[t+1] was a double-shift: it compared logits[t] against
        // inputIds[t+2], which is causally unreachable and keeps loss above ln(vocab).
        for (std::size_t t = 0; t < T - 1; ++t) {
            if (sample->attentionMask[t] == 0) continue;

            const int y = sample->targetIds[t];
            if (y == cfg_.ignoreIndex) continue;
            if (y < 0 || static_cast<std::size_t>(y) >= V) continue;

            const float* logits_t = &logitsBuf_[t * V];
            float* dlog_t = &dLogitsBuf_[t * V];

            sampleLossSum += cross_entropy_dlogits_row(
                logits_t,
                dlog_t,
                V,
                y
            );
            ++sampleTokenCount;
        }

        accumLoss += sampleLossSum;
        accumTokens += sampleTokenCount;

        if (sampleTokenCount > 0) {
            const float inv = 1.0f / static_cast<float>(sampleTokenCount);
            scale_inplace_avx2_or_scalar(
                dLogitsBuf_.data(),
                dLogitsBuf_.size(),
                inv
            );
        }

        backend_.backward(
            sample->inputIds,
            sample->attentionMask,
            std::span<const float>(dLogitsBuf_.data(), dLogitsBuf_.size())
        );
    }
}

void CpuQLoRATrainer::maybeEval(int step) {
    if (cfg_.evalInterval <= 0 || cfg_.numEvalBatches <= 0) return;
    if (step % cfg_.evalInterval != 0) return;

    // TODO: evaluation loop
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
            if (samplerOffset_ > epochOrder_.size()) {
                samplerOffset_ = epochOrder_.size();
            }

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

        trainStep(batch, accumLoss, accumTokens);
        ++accumCount;

        if (accumCount < cfg_.gradAccumSteps) continue;

        // --- DEBUG: Check LoRA Gradient Flow ---
        double loraA_norm = 0.0, loraB_norm = 0.0, loraA_grad_norm = 0.0;
        for (const auto& p : params) {
            std::string_view pname(p.name);
            if (pname == "Aq" || pname == "Av") {
                for (std::size_t i = 0; i < p.n; ++i) {
                    loraA_norm += (double)p.w[i] * p.w[i];
                    if (p.g) loraA_grad_norm += (double)p.g[i] * p.g[i];
                }
            }
            else if (pname == "Bq" || pname == "Bv") {
                for (std::size_t i = 0; i < p.n; ++i) {
                    loraB_norm += (double)p.w[i] * p.w[i];
                }
            }
        }
        if (globalStep % cfg_.logInterval == 0) {
            printf("[DEBUG] LoRA A norm: %.6f | LoRA B norm: %.6f | dLora A norm: %.6f\n",
                std::sqrt(loraA_norm), std::sqrt(loraB_norm), std::sqrt(loraA_grad_norm));
        }
        // ----------------------------------------

        if (cfg_.maxGradNorm > 0.0f) {
            optimizer_.clipGradNorm(params, cfg_.maxGradNorm);
        }

        optimizer_.step(params, cfg_.zeroGradAfterStep);

        // REQUIRED: zero the backend's LoRA gradient buffers after every step.
        // optimizer_.step() only zeros the gradient copies inside the optimizer;
        // the backend's loraGrads_ (gradAq/gradBq/gradAv/gradBv) are separate
        // memory. Without this call they accumulate across all steps, growing
        // ~N× too large by step N; clipGradNorm then squashes them to near-zero,
        // killing all learning signal.
        backend_.zeroGrad();

        ++globalStep;

        const float avgLoss = (accumTokens > 0)
            ? (accumLoss / static_cast<float>(accumTokens))
            : 0.0f;

        reporter_.onStep(
            globalStep,
            avgLoss,
            cfg_.maxSteps,
            cfg_.logInterval,
            startTime
        );

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

    if (!std::filesystem::exists(cfg_.loraOutputPath)) {
        throw std::runtime_error(
            "saveLoRA returned, but output file was not created: " + cfg_.loraOutputPath
        );
    }

    std::cout << "LoRA file exists at: "
        << std::filesystem::absolute(cfg_.loraOutputPath).string()
        << "\n";
    std::cout << "LoRA file size: "
        << std::filesystem::file_size(cfg_.loraOutputPath)
        << " bytes\n";

    state.step = globalStep;
    state.epoch = epoch_;
    state.samplerOffset = samplerOffset_;

    checkpointer_.save(backend_, optimizer_, state);

    std::cout << "Training complete. LoRA adapter saved to "
        << cfg_.loraOutputPath << '\n';
}

int runCpuQLoRATrainer(const TrainerConfig& cfg,
    const std::string& datasetPath) {
    try {
        Dataset dataset{ datasetPath, cfg };
        CpuKernelBackend backend{ cfg };

        LionOptimizer optimizer{};
        optimizer.setHyperParams(
            cfg.learningRate,
            cfg.lionBeta1,
            cfg.lionBeta2,
            cfg.weightDecay
        );

        CpuQLoRATrainer trainer{ cfg, dataset, backend, optimizer };
        trainer.run();
        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Fatal error in runCpuQLoRATrainer: " << ex.what() << '\n';
        return 1;
    }
}