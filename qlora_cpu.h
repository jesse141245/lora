#pragma once

#include <chrono>
#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "checkpoint.h"
#include "config.h"
#include "cpuKernelBack.h"
#include "dataset.h"
#include "adamw_optimizer.h"
#include "lion_optimizer.h"
#include "muon_optimizer.h"

class ProgressReporter {
public:
    void onStep(
        int step,
        float loss,
        float learningRate,
        std::int64_t supervisedTokens,
        std::int64_t totalSupervisedTokens,
        int maxSteps,
        int logInterval,
        const std::chrono::steady_clock::time_point& start,
        int epoch,
        const std::chrono::steady_clock::time_point& epochStart
    );
    float currentMa8() const noexcept { return currentMa8_; }
    float currentMa32() const noexcept { return currentMa32_; }
    bool hasLoss() const noexcept { return hasLoss_; }

private:
    std::deque<float> lossWindow8_;
    std::deque<float> lossWindow32_;
    float lossSum8_ = 0.0f;
    float lossSum32_ = 0.0f;
    float currentMa8_ = 0.0f;
    float currentMa32_ = 0.0f;
    bool hasLoss_ = false;
};

class PlateauLrDecayDetector {
public:
    PlateauLrDecayDetector() = default;
    PlateauLrDecayDetector(
        bool enabled,
        int minSteps,
        int patienceSteps,
        float minRelImprovement
    );

    std::optional<int> observe(int step, float loss);
    float currentMa32() const noexcept { return currentMa32_; }
    float bestMa32() const noexcept { return bestMa32_; }
    int bestStep() const noexcept { return bestStep_; }
    int minSteps() const noexcept { return minSteps_; }
    int patienceSteps() const noexcept { return patienceSteps_; }
    float minRelImprovement() const noexcept { return minRelImprovement_; }
    bool enabled() const noexcept { return enabled_; }

private:
    bool enabled_ = false;
    int minSteps_ = 0;
    int patienceSteps_ = 32;
    float minRelImprovement_ = 0.01f;
    bool triggered_ = false;
    bool haveBest_ = false;
    std::deque<float> lossWindow32_;
    float lossSum32_ = 0.0f;
    float currentMa32_ = 0.0f;
    float bestMa32_ = 0.0f;
    int bestStep_ = 0;
};

class LossBounceStopDetector {
public:
    LossBounceStopDetector() = default;
    explicit LossBounceStopDetector(const TrainerConfig& cfg);

    bool observe(int step, float loss, std::int64_t totalSupervisedTokens);
    float bestLoss() const noexcept { return bestLoss_; }
    int bestStep() const noexcept { return bestStep_; }
    float recentRange() const noexcept { return recentRange_; }
    int recentDirectionChanges() const noexcept { return recentDirectionChanges_; }

private:
    bool enabled_ = false;
    std::int64_t minSupTokens_ = 0;
    float startLoss_ = 0.0f;
    int window_ = 8;
    int patienceSteps_ = 12;
    float minDelta_ = 0.02f;
    float minRange_ = 0.75f;
    int minDirectionChanges_ = 3;
    bool haveBest_ = false;
    float bestLoss_ = 0.0f;
    int bestStep_ = 0;
    float recentRange_ = 0.0f;
    int recentDirectionChanges_ = 0;
    std::deque<float> recentLosses_;
};

struct EvalPredictionStats {
    std::int64_t tokens = 0;
    double rankSum = 0.0;
    double targetProbSum = 0.0;
    double marginSum = 0.0;
    std::int64_t top1 = 0;
    std::int64_t top5 = 0;
    std::int64_t top10 = 0;
    std::int64_t top50 = 0;
    std::array<std::int64_t, 4> bucketTokens{};
    std::array<double, 4> bucketLossSum{};
    std::array<double, 4> bucketRankSum{};
    std::array<double, 4> bucketTargetProbSum{};
};

struct StepSampleDebugRecord {
    std::size_t sampleIndex = 0;
    std::int64_t supervisedTokens = 0;
    float avgLoss = 0.0f;
    float rawAvgLoss = 0.0f;
    float maxRawTokenLoss = 0.0f;
    float cappedFrac = 0.0f;
    float capScaleAvg = 1.0f;
    float freqScaledFrac = 0.0f;
    float freqScaleAvg = 1.0f;
    float avgTargetTrainFreq = 0.0f;
    float rare10Frac = 0.0f;
    float rare20Frac = 0.0f;
};

struct HardTokenDebugRecord {
    std::size_t sampleIndex = 0;
    std::size_t seqIndex = 0;
    std::size_t supervisedOffset = 0;
    int targetToken = 0;
    float loss = 0.0f;
    float targetProb = 0.0f;
    std::int64_t rank = 0;
    float margin = 0.0f;
    std::int64_t targetTrainFreq = 0;
    float targetTrainFreqPct = 0.0f;
};

struct TrainStepDebugMetrics {
    std::int64_t tokens = 0;
    int samples = 0;
    double lossSum = 0.0;
    double rawLossSum = 0.0;
    double targetProbSum = 0.0;
    double rankSum = 0.0;
    double marginSum = 0.0;
    double logitStdSum = 0.0;
    double dLogitsSq = 0.0;
    std::int64_t dLogitsValues = 0;
    float dLogitsMaxAbs = 0.0f;
    std::int64_t cappedTokens = 0;
    double capScaleSum = 0.0;
    float maxRawTokenLoss = 0.0f;
    std::size_t maxRawTokenSampleIndex = 0;
    std::size_t maxRawTokenSeqIndex = 0;
    std::size_t maxRawTokenSupervisedOffset = 0;
    int maxRawTokenTarget = 0;
    std::int64_t maxRawTokenRank = 0;
    float maxRawTokenMargin = 0.0f;
    std::int64_t maxRawTokenTrainFreq = 0;
    float maxRawTokenTrainFreqPct = 0.0f;
    std::int64_t freqScaledTokens = 0;
    double freqScaleSum = 0.0;
    std::int64_t wrongMarginTokens = 0;
    double wrongMarginPenaltySum = 0.0;
    float wrongMarginExcessMax = 0.0f;
    std::int64_t rareWrongMarginTokens = 0;
    double rareWrongMarginPenaltySum = 0.0;
    float rareWrongMarginExcessMax = 0.0f;
    std::int64_t hardBoostedTokens = 0;
    double hardBoostScaleSum = 0.0;
    std::int64_t rareHardBoostedTokens = 0;
    double rareHardBoostScaleSum = 0.0;
    std::int64_t rareTargetBiasUpdates = 0;
    double rareTargetBiasDeltaAbsSum = 0.0;
    float rareTargetBiasMaxAbs = 0.0f;
    int replayWeightedSamples = 0;
    double replayWeightSum = 0.0;
    int gradientOnlyReplaySamples = 0;
    std::int64_t gradientOnlyReplayTokens = 0;
    std::int64_t top1 = 0;
    std::int64_t top5 = 0;
    std::int64_t top10 = 0;
    std::int64_t top50 = 0;
    std::array<std::int64_t, 4> bucketTokens{};
    std::array<double, 4> bucketLossSum{};
    std::array<double, 4> bucketRankSum{};
    std::array<double, 4> bucketTargetProbSum{};
    float minSampleLoss = 0.0f;
    float maxSampleLoss = 0.0f;
    float minRawSampleLoss = 0.0f;
    float maxRawSampleLoss = 0.0f;
};

class CpuQLoRATrainer {
public:
    CpuQLoRATrainer(
        const TrainerConfig& cfg,
        Dataset& dataset,
        Dataset* evalDataset,
        CpuKernelBackend& backend,
        AdamWOptimizer& optimizer
    );
    CpuQLoRATrainer(
        const TrainerConfig& cfg,
        Dataset& dataset,
        Dataset* evalDataset,
        CpuKernelBackend& backend,
        LionOptimizer& optimizer
    );
    CpuQLoRATrainer(
        const TrainerConfig& cfg,
        Dataset& dataset,
        Dataset* evalDataset,
        CpuKernelBackend& backend,
        MuonOptimizer& optimizer
    );

    void run();

private:
    std::vector<std::size_t> makeEpochOrder() const;
    void buildTargetTokenFrequency();
    void observeOnlineHardReplay(
        int step,
        const std::vector<StepSampleDebugRecord>& sampleDebugRecords);
    std::vector<const TrainingSample*> sampleBatch(
        int batchSize,
        int currentStep,
        std::vector<std::size_t>* outIndices = nullptr,
        std::vector<float>* outLossWeights = nullptr,
        std::vector<std::uint8_t>* outIsReplay = nullptr);

    void trainStep(
        const std::vector<const TrainingSample*>& batch,
        const std::vector<std::size_t>& batchIndices,
        const std::vector<float>& batchLossWeights,
        const std::vector<std::uint8_t>& batchIsReplay,
        std::int64_t totalSupervisedTokensBeforeStep,
        float& accumLoss,
        std::int64_t& accumTokens,
        std::vector<StepSampleDebugRecord>* sampleDebugRecords = nullptr,
        TrainStepDebugMetrics* trainDebugMetrics = nullptr,
        std::vector<HardTokenDebugRecord>* hardTokenDebugRecords = nullptr
    );

    void maybeEval(int step, bool force = false);
    bool hasPendingTrainProbeGate(std::int64_t totalSupervisedTokens) const;
    void maybeTrainProbe(
        int step,
        std::int64_t totalSupervisedTokens,
        bool force = false);
    float datasetLoss(
        const Dataset& source,
        int maxBatches,
        std::int64_t& outTokens,
        EvalPredictionStats* outPredStats = nullptr
    );
    float sampleLossSum(
        const TrainingSample& sample,
        std::int64_t& outTokens,
        EvalPredictionStats* outPredStats = nullptr
    );
    float evalLoss(
        int maxBatches,
        std::int64_t& outTokens,
        EvalPredictionStats* outPredStats = nullptr
    );
    void maybeCheckpoint(int step, const CheckpointState& state, bool force = false);

private:
    bool usingAdamW() const noexcept { return adamOptimizer_ != nullptr; }
    bool usingMuon() const noexcept { return muonOptimizer_ != nullptr; }

private:
    TrainerConfig cfg_;
    Dataset& dataset_;
    Dataset* evalDataset_ = nullptr;
    CpuKernelBackend& backend_;
    AdamWOptimizer* adamOptimizer_ = nullptr;
    LionOptimizer* lionOptimizer_ = nullptr;
    MuonOptimizer* muonOptimizer_ = nullptr;
    Checkpointer checkpointer_;
    ProgressReporter reporter_;

    std::vector<float> logitsBuf_;
    std::vector<float> dLogitsBuf_;
    std::vector<float> rareTargetBias_;

    std::vector<std::size_t> epochOrder_;
    std::size_t samplerOffset_ = 0;
    std::size_t replaySampleOffset_ = 0;
    std::size_t replayBatchCounter_ = 0;
    std::vector<std::size_t> onlineHardReplayIndices_;
    int epoch_ = 0;
    int lastCompletedEpochLogged_ = 0;
    std::chrono::steady_clock::time_point epochStartTime_{};
    bool stopRequested_ = false;
    bool stopCheckpointRequested_ = false;
    std::string stopReason_;
    std::unordered_map<int, std::int64_t> targetTokenFrequency_;
    std::int64_t targetTokenFrequencyTotal_ = 0;
    bool trainProbeGate1Done_ = false;
    bool trainProbeGate2Done_ = false;
    bool lmHeadBootstrapCompletedLogged_ = false;
    bool attentionOnlyBootstrapCompletedLogged_ = false;
    bool attentionBOnlyBootstrapCompletedLogged_ = false;
    bool attentionABootstrapCompletedLogged_ = false;
};

int runCpuQLoRATrainer(
    const TrainerConfig& cfg,
    const std::string& datasetPath
);
