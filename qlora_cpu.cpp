#include "qlora_cpu.h"
#include "loss_ce.h"
#include "lr_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <filesystem>
#include <cmath> // For std::sqrt
#include <optional>
#include <string>

namespace {

struct LogitStats {
    double mean = 0.0;
    double stddev = 0.0;
    float minLogit = 0.0f;
    float maxLogit = 0.0f;
    float targetLogit = 0.0f;
    bool hasNonFinite = false;
};

std::optional<std::string> tryGetEnvString(const char* name) {
#if defined(_WIN32)
    char* value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) != 0 || !value || len == 0 || value[0] == '\0') {
        if (value) {
            std::free(value);
        }
        return std::nullopt;
    }
    std::string out(value);
    std::free(value);
    return out;
#else
    const char* value = std::getenv(name);
    if (!value || *value == '\0') {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool envBoolOr(const char* name, bool fallback) {
    const auto value = tryGetEnvString(name);
    if (!value) {
        return fallback;
    }
    const std::string s = lowerCopy(*value);
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    return fallback;
}

bool logitsDebugEnabled() {
    return envBoolOr("QLORA_DEBUG_LOGITS", false);
}

bool logitsValidationEnabled() {
    return envBoolOr("QLORA_VALIDATE_LOGITS", logitsDebugEnabled());
}

bool loraNormDebugEnabled() {
    return envBoolOr("QLORA_DEBUG_LORA_NORMS", false);
}

bool quantizedLmHeadBackwardEnabled(const TrainerConfig& cfg) {
    return cfg.useBitNetQuant &&
        !cfg.bitnetLmHeadFloatBackward &&
        (cfg.bitnetQuantizeLmHead || cfg.bitnetLmHeadQuantBackwardOnly);
}

float logitAbsThreshold() {
    const auto value = tryGetEnvString("QLORA_LOGIT_ABS_LIMIT");
    if (!value) return 1.0e6f;
    try {
        return std::stof(*value);
    }
    catch (...) {
        return 1.0e6f;
    }
}

LogitStats inspectLogits(
    const float* logits,
    std::size_t vocabSize,
    int targetToken,
    std::size_t seqIndex)
{
    (void)seqIndex;
    LogitStats stats{};
    if (!logits || vocabSize == 0) return stats;

    double sum = 0.0;
    double sumSq = 0.0;
    float minLogit = std::numeric_limits<float>::max();
    float maxLogit = -std::numeric_limits<float>::max();

    for (std::size_t i = 0; i < vocabSize; ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) {
            stats.hasNonFinite = true;
        }
        sum += static_cast<double>(v);
        sumSq += static_cast<double>(v) * static_cast<double>(v);
        minLogit = std::min(minLogit, v);
        maxLogit = std::max(maxLogit, v);
    }

    stats.mean = sum / static_cast<double>(vocabSize);
    const double var = std::max(0.0, (sumSq / static_cast<double>(vocabSize)) - (stats.mean * stats.mean));
    stats.stddev = std::sqrt(var);
    stats.minLogit = minLogit;
    stats.maxLogit = maxLogit;
    stats.targetLogit =
        (targetToken >= 0 && static_cast<std::size_t>(targetToken) < vocabSize)
        ? logits[targetToken]
        : 0.0f;

    return stats;
}

void maybePrintLogitStats(
    const LogitStats& stats,
    int targetToken,
    std::size_t seqIndex)
{
    static bool printed = false;
    if (printed || !logitsDebugEnabled()) return;
    printed = true;

    std::cout << "[LOGIT_STATS] seq_t=" << seqIndex
        << " | mean=" << stats.mean
        << " | std=" << stats.stddev
        << " | min=" << stats.minLogit
        << " | max=" << stats.maxLogit
        << " | target=" << targetToken
        << " | target_logit=" << stats.targetLogit
        << '\n';
}

void validateLogitsOrThrow(
    const LogitStats& stats,
    std::size_t seqIndex,
    int targetToken)
{
    const float absLimit = logitAbsThreshold();
    const float maxAbs = std::max(std::fabs(stats.minLogit), std::fabs(stats.maxLogit));
    if (stats.hasNonFinite || !std::isfinite(stats.targetLogit) || maxAbs > absLimit) {
        throw std::runtime_error(
            "trainStep: invalid logits at supervised token t=" + std::to_string(seqIndex) +
            " target=" + std::to_string(targetToken) +
            " maxAbs=" + std::to_string(maxAbs) +
            " limit=" + std::to_string(absLimit) +
            (stats.hasNonFinite ? " non_finite=1" : " non_finite=0")
        );
    }
}

float computeStepsPerSec(
    int step,
    const std::chrono::steady_clock::time_point& start)
{
    if (step <= 0) return 0.0f;
    const auto now = std::chrono::steady_clock::now();
    const double elapsedSec =
        std::chrono::duration<double>(now - start).count();
    if (elapsedSec <= 0.0) return 0.0f;
    return static_cast<float>(static_cast<double>(step) / elapsedSec);
}

double elapsedSeconds(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start
    ).count();
}

bool canShortcutRepeatedOverfitAccum(const TrainerConfig& cfg) {
    return cfg.overfitSampleIndex >= 0 &&
        cfg.batchSize == 1 &&
        cfg.gradAccumSteps > 1 &&
        cfg.loraDropout == 0.0f;
}

void scaleGradients(const std::vector<ParamRef>& params, float scale) {
    if (scale == 1.0f) return;
    for (const auto& p : params) {
        if (!p.g || p.n == 0) continue;
        for (std::size_t i = 0; i < p.n; ++i) {
            p.g[i] *= scale;
        }
    }
}

} // namespace

void ProgressReporter::onStep(
    int step,
    float loss,
    float learningRate,
    std::int64_t supervisedTokens,
    int maxSteps,
    int logInterval,
    const std::chrono::steady_clock::time_point& start,
    int epoch,
    const std::chrono::steady_clock::time_point& epochStart) {

    if (step <= 0 || maxSteps <= 0) return;
    if (logInterval <= 0) return;
    if (step % logInterval != 0) return;

    const float stepsPerSec = computeStepsPerSec(step, start);

    const float remainingSteps = static_cast<float>(maxSteps - step);
    const float etaSec = (stepsPerSec > 0.0f)
        ? remainingSteps / stepsPerSec
        : -1.0f;
    const double epochElapsedSec = elapsedSeconds(epochStart);

    lossWindow8_.push_back(loss);
    lossSum8_ += loss;
    if (lossWindow8_.size() > 8) {
        lossSum8_ -= lossWindow8_.front();
        lossWindow8_.pop_front();
    }

    lossWindow32_.push_back(loss);
    lossSum32_ += loss;
    if (lossWindow32_.size() > 32) {
        lossSum32_ -= lossWindow32_.front();
        lossWindow32_.pop_front();
    }

    const float ma8 = lossWindow8_.empty() ? loss : (lossSum8_ / static_cast<float>(lossWindow8_.size()));
    const float ma32 = lossWindow32_.empty() ? loss : (lossSum32_ / static_cast<float>(lossWindow32_.size()));

    std::cout << "[Step " << step << "/" << maxSteps << "] "
        << "loss=" << loss
        << " | ma8=" << ma8
        << " | ma32=" << ma32
        << " | lr=" << learningRate
        << " | supTokens=" << supervisedTokens
        << " | epoch=" << epoch
        << " | epochTime(s)=" << epochElapsedSec
        << " | steps/s=" << stepsPerSec
        << " | ETA(s)=" << etaSec << '\n';
}

CpuQLoRATrainer::CpuQLoRATrainer(const TrainerConfig& cfg,
    Dataset& dataset,
    CpuKernelBackend& backend,
    AdamWOptimizer& optimizer)
    : cfg_{ cfg }
    , dataset_{ dataset }
    , backend_{ backend }
    , adamOptimizer_{ &optimizer }
    , lionOptimizer_{ nullptr }
    , checkpointer_{ cfg.checkpointDir } {
}

CpuQLoRATrainer::CpuQLoRATrainer(const TrainerConfig& cfg,
    Dataset& dataset,
    CpuKernelBackend& backend,
    LionOptimizer& optimizer)
    : cfg_{ cfg }
    , dataset_{ dataset }
    , backend_{ backend }
    , adamOptimizer_{ nullptr }
    , lionOptimizer_{ &optimizer }
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

    if (cfg_.overfitSampleIndex >= 0) {
        if (epoch_ == 0) {
            epoch_ = 1;
            epochStartTime_ = std::chrono::steady_clock::now();
        }
        const std::size_t fixedIdx = std::min<std::size_t>(
            static_cast<std::size_t>(cfg_.overfitSampleIndex),
            dataset_.size() - 1
        );
        batch.reserve(static_cast<std::size_t>(batchSize));
        for (int i = 0; i < batchSize; ++i) {
            batch.push_back(&dataset_.get(fixedIdx));
        }
        return batch;
    }

    if (epochOrder_.empty()) {
        epochOrder_ = makeEpochOrder();
        samplerOffset_ = 0;
        ++epoch_;
        epochStartTime_ = std::chrono::steady_clock::now();
    }

    batch.reserve(static_cast<std::size_t>(batchSize));

    for (int i = 0; i < batchSize; ++i) {
        if (samplerOffset_ >= epochOrder_.size()) {
            const double epochTimeSec = elapsedSeconds(epochStartTime_);
            std::cout << "[Epoch " << epoch_ << " complete] "
                << "epochTime(s)=" << epochTimeSec << '\n';
            epochOrder_ = makeEpochOrder();
            samplerOffset_ = 0;
            ++epoch_;
            epochStartTime_ = std::chrono::steady_clock::now();
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
        const std::size_t paddedT = sample->inputIds.size();
        if (sample->attentionMask.size() != paddedT || sample->targetIds.size() != paddedT) {
            throw std::runtime_error("trainStep: sample sizes mismatch");
        }

        std::size_t T = 0;
        for (std::size_t t = 0; t < paddedT; ++t) {
            if (sample->attentionMask[t] != 0) {
                T = t + 1;
            }
        }
        if (T == 0) continue;

        std::vector<int> lossMask(T, 0);
        std::int64_t sampleTokenCount = 0;
        for (std::size_t t = 0; t < T; ++t) {
            if (sample->attentionMask[t] == 0) continue;
            const int y = sample->targetIds[t];
            if (y == cfg_.ignoreIndex) continue;
            if (y < 0 || static_cast<std::size_t>(y) >= V) continue;
            lossMask[t] = 1;
            ++sampleTokenCount;
        }

        if (sampleTokenCount == 0) continue;

        const std::size_t needed = backend_.logitsCount(T);
        if (logitsBuf_.size() != needed) {
            logitsBuf_.resize(needed);
        }
        if (dLogitsBuf_.size() != needed) {
            dLogitsBuf_.resize(needed);
        }
        if (quantizedLmHeadBackwardEnabled(cfg_)) {
            std::fill(dLogitsBuf_.begin(), dLogitsBuf_.end(), 0.0f);
        }

        backend_.forward(
            std::span<const int>(sample->inputIds.data(), T),
            std::span<const int>(sample->attentionMask.data(), T),
            std::span<float>(logitsBuf_.data(), logitsBuf_.size()),
            std::span<const int>(lossMask.data(), lossMask.size())
        );

        float sampleLossSum = 0.0f;
        const bool validateLogits = logitsValidationEnabled();

        // target_ids are already next-token labels for each timestep t.
        for (std::size_t t = 0; t < T; ++t) {
            if (lossMask[t] == 0) continue;

            const int y = sample->targetIds[t];

            const float* logits_t = &logitsBuf_[t * V];
            float* dlog_t = &dLogitsBuf_[t * V];

            if (validateLogits) {
                const LogitStats stats = inspectLogits(logits_t, V, y, t);
                maybePrintLogitStats(stats, y, t);
                validateLogitsOrThrow(stats, t, y);
            }

            sampleLossSum += cross_entropy_dlogits_row(
                logits_t,
                dlog_t,
                V,
                y
            );
            // NOTE: do NOT increment sampleTokenCount here.
            // It was already counted in the lossMask-building loop above;
            // a second increment would halve the reported loss.
        }

        accumLoss += sampleLossSum;
        accumTokens += sampleTokenCount;

        // Normalize gradients by the number of supervised tokens so that
        // gradient magnitude is independent of how many response tokens are
        // in this sample. Without this, a 30-token response produces 30x
        // the gradient norm of a 1-token response, causing unstable training
        // even with grad-norm clipping (which clips the total norm, not per-
        // token scale). This makes effective LR consistent across samples.
        if (sampleTokenCount > 1) {
            const float inv_n = 1.0f / static_cast<float>(sampleTokenCount);
            const std::size_t totalDLogits = T * V;
            scale_inplace_avx2_or_scalar(dLogitsBuf_.data(), totalDLogits, inv_n);
        }

        backend_.backward(
            std::span<const int>(sample->inputIds.data(), T),
            std::span<const int>(sample->attentionMask.data(), T),
            std::span<const float>(dLogitsBuf_.data(), dLogitsBuf_.size()),
            std::span<const int>(lossMask.data(), lossMask.size())
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

    if (usingAdamW()) {
        checkpointer_.save(backend_, *adamOptimizer_, state);
    }
    else {
        checkpointer_.save(backend_, *lionOptimizer_, state);
    }
}

void CpuQLoRATrainer::run() {
    const auto startTime = std::chrono::steady_clock::now();

    CheckpointState state{};
    state.step = 0;
    state.epoch = 0;
    state.samplerOffset = 0;

    if (cfg_.resumeFromCheckpoint) {
        std::optional<CheckpointState> loaded;
        if (usingAdamW()) {
            loaded = checkpointer_.loadLatest(backend_, *adamOptimizer_);
        }
        else {
            loaded = checkpointer_.loadLatest(backend_, *lionOptimizer_);
        }

        if (loaded) {
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
    epochStartTime_ = std::chrono::steady_clock::now();

    float accumLoss = 0.0f;
    std::int64_t accumTokens = 0;
    int accumCount = 0;

    auto params = backend_.collectTrainableParams();
    if (usingAdamW()) {
        adamOptimizer_->init(params);
    }
    else {
        lionOptimizer_->init(params);
    }

    // Short warmup then optional cosine decay; by default this is constant LR
    // after warmup for stronger small-data adaptation.
    const int autoWarmup =
        (cfg_.overfitSampleIndex >= 0)
        ? 1
        : std::min(20, std::max(1, cfg_.maxSteps / 10));
    const int warmupSteps = std::min(
        cfg_.maxSteps,
        (cfg_.warmupSteps > 0 ? cfg_.warmupSteps : autoWarmup)
    );
    const int decayStartStep = (cfg_.decayStartStep > 0)
        ? std::min(cfg_.maxSteps, std::max(warmupSteps, cfg_.decayStartStep))
        : warmupSteps;
    const int decaySteps = (cfg_.decaySteps > 0)
        ? cfg_.decaySteps
        : -1;
    const float minLr = cfg_.useCosineDecay
        ? (cfg_.learningRate * std::max(0.0f, cfg_.minLrFactor))
        : cfg_.learningRate;
    LRScheduler scheduler{ cfg_.learningRate, minLr,
                           cfg_.maxSteps, warmupSteps, decayStartStep, decaySteps };
    std::cout << "LR schedule: warmup=" << warmupSteps
        << " | decay_start=" << decayStartStep
        << " | decay_steps=" << ((decaySteps > 0) ? decaySteps : (cfg_.maxSteps - decayStartStep))
        << " | mode=" << (cfg_.useCosineDecay ? "cosine" : "constant")
        << " | peak_lr=" << cfg_.learningRate
        << " | min_lr=" << minLr << '\n';

    while (globalStep < cfg_.maxSteps) {
        const auto batch = sampleBatch(cfg_.batchSize);
        if (batch.empty()) break;

        trainStep(batch, accumLoss, accumTokens);
        accumCount += canShortcutRepeatedOverfitAccum(cfg_)
            ? cfg_.gradAccumSteps
            : 1;

        if (accumCount < cfg_.gradAccumSteps) continue;

        if (accumCount > 1) {
            const float invAccum = 1.0f / static_cast<float>(accumCount);
            scaleGradients(params, invAccum);
        }

        if (cfg_.maxGradNorm > 0.0f) {
            if (usingAdamW()) {
                adamOptimizer_->clipGradNorm(params, cfg_.maxGradNorm);
            }
            else {
                lionOptimizer_->clipGradNorm(params, cfg_.maxGradNorm);
            }
        }

        // Compute LR for this step once so debug and optimizer are identical.
        const float stepLr = scheduler.lrAt(globalStep);

        if (loraNormDebugEnabled() && globalStep % cfg_.logInterval == 0) {
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
            printf("[DEBUG] LoRA A norm: %.6f | LoRA B norm: %.6f | dLora A norm: %.6f | lr=%.2e\n",
                std::sqrt(loraA_norm), std::sqrt(loraB_norm), std::sqrt(loraA_grad_norm),
                stepLr);
        }
        if (usingAdamW()) {
            adamOptimizer_->setHyperParams(
                stepLr,
                cfg_.lionBeta1,
                cfg_.lionBeta2,
                cfg_.weightDecay
            );
            adamOptimizer_->step(params, cfg_.zeroGradAfterStep);
        }
        else {
            lionOptimizer_->setHyperParams(
                stepLr,
                cfg_.lionBeta1,
                cfg_.lionBeta2,
                cfg_.weightDecay
            );
            lionOptimizer_->step(params, cfg_.zeroGradAfterStep);
        }

        ++globalStep;

        const float avgLoss = (accumTokens > 0)
            ? (accumLoss / static_cast<float>(accumTokens))
            : 0.0f;

        reporter_.onStep(
            globalStep,
            avgLoss,
            stepLr,
            accumTokens,
            cfg_.maxSteps,
            cfg_.logInterval,
            startTime,
            epoch_,
            epochStartTime_
        );

        if (cfg_.minStepsPerSec > 0.0f &&
            globalStep >= std::max(1, cfg_.minStepsPerSecGrace))
        {
            const float stepsPerSec = computeStepsPerSec(globalStep, startTime);
            if (stepsPerSec < cfg_.minStepsPerSec) {
                std::cout << "[ABORT] throughput floor hit: steps/s=" << stepsPerSec
                    << " < minStepsPerSec=" << cfg_.minStepsPerSec
                    << " after step " << globalStep << '\n';
                break;
            }
        }

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

    if (usingAdamW()) {
        checkpointer_.save(backend_, *adamOptimizer_, state);
    }
    else {
        checkpointer_.save(backend_, *lionOptimizer_, state);
    }

    std::cout << "Training complete. LoRA adapter saved to "
        << cfg_.loraOutputPath << '\n';
}

int runCpuQLoRATrainer(const TrainerConfig& cfg,
    const std::string& datasetPath) {
    try {
        Dataset dataset{ datasetPath, cfg };
        CpuKernelBackend backend{ cfg };

        if (cfg.useAdamW) {
            AdamWOptimizer optimizer{};
            optimizer.setHyperParams(
                cfg.learningRate,
                cfg.lionBeta1,
                cfg.lionBeta2,
                cfg.weightDecay
            );
            optimizer.setZeroGradAfterStep(cfg.zeroGradAfterStep);

            CpuQLoRATrainer trainer{ cfg, dataset, backend, optimizer };
            trainer.run();
        }
        else {
            LionOptimizer optimizer{};
            optimizer.setHyperParams(
                cfg.learningRate,
                cfg.lionBeta1,
                cfg.lionBeta2,
                cfg.weightDecay
            );
            CpuQLoRATrainer trainer{ cfg, dataset, backend, optimizer };
            trainer.run();
        }
        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Fatal error in runCpuQLoRATrainer: " << ex.what() << '\n';
        return 1;
    }
}
