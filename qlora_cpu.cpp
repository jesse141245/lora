#include "qlora_cpu.h"
#include "env_utils.h"
#include "lora.h"
#include "loss_ce.h"
#include "lr_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <filesystem>
#include <cmath> // For std::sqrt
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace {

struct LogitStats {
    double mean = 0.0;
    double stddev = 0.0;
    float minLogit = 0.0f;
    float maxLogit = 0.0f;
    float targetLogit = 0.0f;
    std::size_t maxIndex = 0;
    std::int64_t targetRank = 0;
    bool hasNonFinite = false;
};

bool logitsDebugEnabled() {
    return envutil::envBoolOr("QLORA_DEBUG_LOGITS", false);
}

bool logitsValidationEnabled() {
    return envutil::envBoolOr("QLORA_VALIDATE_LOGITS", logitsDebugEnabled());
}

bool loraNormDebugEnabled() {
    return envutil::envBoolOr("QLORA_DEBUG_LORA_NORMS", false);
}

bool paramNormDebugEnabled() {
    return envutil::envBoolOr("QLORA_DEBUG_PARAM_NORMS", false);
}

bool stepSampleDebugEnabled() {
    return envutil::envBoolOr("QLORA_DEBUG_STEP_SAMPLES", false);
}

bool trainMetricsDebugEnabled() {
    return envutil::envBoolOr("QLORA_DEBUG_TRAIN_METRICS", false);
}

bool hardTokenDebugEnabled() {
    return envutil::envBoolOr("QLORA_DEBUG_HARD_TOKENS", false);
}

bool trainLoopTraceEnabled() {
    return envutil::envBoolOr("QLORA_TRACE_TRAIN_LOOP", false);
}

void traceTrainLoop(int step, const char* phase) {
    if (!trainLoopTraceEnabled()) return;
    std::cout << "[TRACE train_loop] step=" << step << " " << phase << '\n';
}

int hardTokenDebugTopK() {
    const auto value = envutil::tryGetEnvString("QLORA_DEBUG_HARD_TOKENS_TOPK");
    if (!value) return 8;
    try {
        return std::max(1, std::stoi(*value));
    }
    catch (...) {
        return 8;
    }
}

bool matrixNoiseDebugEnabled() {
    return envutil::envBoolOr("QLORA_DEBUG_MATRIX_NOISE", false) || paramNormDebugEnabled();
}

int matrixNoiseTopK() {
    const auto value = envutil::tryGetEnvString("QLORA_DEBUG_MATRIX_TOPK");
    if (!value) return 6;
    try {
        return std::max(1, std::stoi(*value));
    }
    catch (...) {
        return 6;
    }
}

float supervisedTokenLossWeight(const TrainerConfig& cfg, std::size_t supervisedOffset) {
    if (cfg.firstAnswerLossWeightTokens <= 0 || cfg.firstAnswerLossWeight <= 0.0f) {
        return 1.0f;
    }
    return supervisedOffset < static_cast<std::size_t>(cfg.firstAnswerLossWeightTokens)
        ? cfg.firstAnswerLossWeight
        : 1.0f;
}

float targetFrequencyLossScale(
    const TrainerConfig& cfg,
    std::int64_t targetTrainFreq,
    std::int64_t totalSupervisedTokensBeforeStep)
{
    if (cfg.targetFreqLossMinFreq <= 0) return 1.0f;
    if (totalSupervisedTokensBeforeStep < cfg.targetFreqLossMinTotalSupTokens) return 1.0f;

    const float minScale = std::clamp(cfg.targetFreqLossMinScale, 0.0f, 1.0f);
    if (minScale >= 1.0f) return 1.0f;

    const float freq = static_cast<float>(std::max<std::int64_t>(0, targetTrainFreq));
    const float minFreq = static_cast<float>(std::max(1, cfg.targetFreqLossMinFreq));
    const float t = std::clamp(freq / minFreq, 0.0f, 1.0f);
    return minScale + (1.0f - minScale) * t;
}

bool shouldNormalizeDLogitsBySampleTokens(const TrainerConfig& cfg) {
    return cfg.normalizeDLogitsBySampleTokens;
}

float logitAbsThreshold() {
    const auto value = envutil::tryGetEnvString("QLORA_LOGIT_ABS_LIMIT");
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
        if (v > maxLogit) {
            maxLogit = v;
            stats.maxIndex = i;
        }
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
    if (targetToken >= 0 && static_cast<std::size_t>(targetToken) < vocabSize) {
        std::int64_t betterCount = 0;
        for (std::size_t i = 0; i < vocabSize; ++i) {
            if (logits[i] > stats.targetLogit) {
                ++betterCount;
            }
        }
        stats.targetRank = betterCount + 1;
    }

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
        << " | maxIndex=" << stats.maxIndex
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

std::int64_t countSupervisedTokensInSample(
    const TrainingSample& sample,
    const TrainerConfig& cfg,
    std::size_t vocabSize)
{
    const std::size_t T = sample.inputIds.size();
    if (sample.attentionMask.size() != T || sample.targetIds.size() != T) {
        throw std::runtime_error("countSupervisedTokensInSample: sample sizes mismatch");
    }

    std::int64_t sampleTokenCount = 0;
    int supervisedSeen = 0;
    for (std::size_t t = 0; t < T; ++t) {
        if (sample.attentionMask[t] == 0) continue;
        const int y = sample.targetIds[t];
        if (y == cfg.ignoreIndex) continue;
        if (y < 0 || static_cast<std::size_t>(y) >= vocabSize) continue;
        if (cfg.maxSupervisedTokensPerSample > 0 &&
            supervisedSeen >= cfg.maxSupervisedTokensPerSample) {
            ++supervisedSeen;
            continue;
        }
        ++sampleTokenCount;
        ++supervisedSeen;
    }
    return sampleTokenCount;
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

struct ParamGroupNormStats {
    double gradSq = 0.0;
    double weightSq = 0.0;
    double lrScaleSum = 0.0;
    float gradMaxAbs = 0.0f;
    std::size_t values = 0;
    std::size_t tensors = 0;
};

void maybePrintParamNorms(
    const std::vector<ParamRef>& params,
    int step,
    int logInterval,
    float maxGradNorm,
    float stepLr)
{
    if (!matrixNoiseDebugEnabled()) return;
    if (logInterval > 0 && step % logInterval != 0) return;

    std::map<std::string, ParamGroupNormStats> groups;
    double globalGradSq = 0.0;
    for (const auto& p : params) {
        if (!p.w || !p.g || p.n == 0) continue;
        auto& g = groups[p.name ? p.name : "(unnamed)"];
        ++g.tensors;
        g.values += p.n;
        g.lrScaleSum += static_cast<double>(p.lrScale);
        for (std::size_t i = 0; i < p.n; ++i) {
            const double gv = static_cast<double>(p.g[i]);
            const double wv = static_cast<double>(p.w[i]);
            g.gradSq += gv * gv;
            g.weightSq += wv * wv;
            g.gradMaxAbs = std::max(g.gradMaxAbs, std::fabs(p.g[i]));
        }
    }

    globalGradSq = 0.0;
    for (const auto& [_, g] : groups) {
        globalGradSq += g.gradSq;
    }
    const double globalNorm = std::sqrt(globalGradSq);
    const double clipScale =
        (maxGradNorm > 0.0f && globalNorm > static_cast<double>(maxGradNorm))
        ? static_cast<double>(maxGradNorm) / globalNorm
        : 1.0;

    if (paramNormDebugEnabled()) {
        std::cout << "[ParamNorms step " << step << "] "
        << "globalGradNorm=" << globalNorm
        << " | maxGradNorm=" << maxGradNorm
            << " | clipScale=" << clipScale;
        for (const auto& [name, g] : groups) {
            const double denom = static_cast<double>(std::max<std::size_t>(1, g.values));
            const double tensors = static_cast<double>(std::max<std::size_t>(1, g.tensors));
            std::cout << " | " << name
                << ":gradNorm=" << std::sqrt(g.gradSq)
                << ",gradRms=" << std::sqrt(g.gradSq / denom)
                << ",gradMax=" << g.gradMaxAbs
                << ",wNorm=" << std::sqrt(g.weightSq)
                << ",lrScale=" << (g.lrScaleSum / tensors);
        }
        std::cout << '\n';
    }

    struct MatrixNoiseRow {
        std::string name;
        double gradNorm = 0.0;
        double gradRms = 0.0;
        double weightRms = 0.0;
        double updateRmsProxy = 0.0;
        double updateToWeightRatio = 0.0;
        float gradMaxAbs = 0.0f;
        double lrScale = 0.0;
        std::size_t values = 0;
    };

    std::vector<MatrixNoiseRow> rows;
    rows.reserve(groups.size());
    for (const auto& [name, g] : groups) {
        const double denom = static_cast<double>(std::max<std::size_t>(1, g.values));
        const double tensors = static_cast<double>(std::max<std::size_t>(1, g.tensors));
        const double gradNorm = std::sqrt(g.gradSq);
        const double gradRms = std::sqrt(g.gradSq / denom);
        const double weightRms = std::sqrt(g.weightSq / denom);
        const double lrScale = g.lrScaleSum / tensors;
        const double updateRmsProxy = static_cast<double>(stepLr) * lrScale * gradRms * clipScale;
        const double updateToWeightRatio =
            (weightRms > 0.0) ? (updateRmsProxy / weightRms) : updateRmsProxy;
        rows.push_back({
            name,
            gradNorm,
            gradRms,
            weightRms,
            updateRmsProxy,
            updateToWeightRatio,
            g.gradMaxAbs,
            lrScale,
            g.values
            });
    }

    const int topK = std::min<int>(matrixNoiseTopK(), static_cast<int>(rows.size()));
    std::sort(rows.begin(), rows.end(), [](const MatrixNoiseRow& a, const MatrixNoiseRow& b) {
        if (a.updateToWeightRatio != b.updateToWeightRatio) {
            return a.updateToWeightRatio > b.updateToWeightRatio;
        }
        return a.gradNorm > b.gradNorm;
        });

    std::cout << "[MatrixNoise step " << step << "] "
        << "globalGradNorm=" << globalNorm
        << " | clipScale=" << clipScale
        << " | lr=" << stepLr
        << " | topK=" << topK;
    for (int i = 0; i < topK; ++i) {
        const auto& row = rows[static_cast<std::size_t>(i)];
        std::cout << " | " << row.name
            << ":updW=" << row.updateToWeightRatio
            << ",updRms~=" << row.updateRmsProxy
            << ",gradRms=" << row.gradRms
            << ",wRms=" << row.weightRms
            << ",gradMax=" << row.gradMaxAbs
            << ",lrScale=" << row.lrScale
            << ",n=" << row.values;
    }
    std::cout << '\n';
}

void maybePrintStepSampleDebug(
    int step,
    int logInterval,
    const std::vector<StepSampleDebugRecord>& sampleDebugRecords)
{
    if (!stepSampleDebugEnabled()) return;
    if (sampleDebugRecords.empty()) return;
    if (logInterval > 0 && step % logInterval != 0) return;

    std::cout << "[StepSamples step " << step << "]";
    for (const auto& record : sampleDebugRecords) {
        std::cout << " | idx=" << record.sampleIndex
            << ",supTok=" << record.supervisedTokens
            << ",avgLoss=" << record.avgLoss
            << ",rawAvgLoss=" << record.rawAvgLoss
            << ",maxRawTok=" << record.maxRawTokenLoss
            << ",cappedFrac=" << record.cappedFrac
            << ",capScaleAvg=" << record.capScaleAvg
            << ",freqScaledFrac=" << record.freqScaledFrac
            << ",freqScaleAvg=" << record.freqScaleAvg
            << ",avgTargetFreq=" << record.avgTargetTrainFreq
            << ",rare10Frac=" << record.rare10Frac
            << ",rare20Frac=" << record.rare20Frac;
    }
    std::cout << '\n';
}

void insertHardTokenRecord(
    std::vector<HardTokenDebugRecord>& records,
    const HardTokenDebugRecord& record,
    int topK)
{
    if (topK <= 0) return;
    auto it = std::lower_bound(
        records.begin(),
        records.end(),
        record.loss,
        [](const HardTokenDebugRecord& a, float loss) {
            return a.loss > loss;
        });
    records.insert(it, record);
    if (static_cast<int>(records.size()) > topK) {
        records.pop_back();
    }
}

HardTokenDebugRecord makeHardTokenRecord(
    const float* logits,
    std::size_t vocabSize,
    std::size_t sampleIndex,
    std::size_t seqIndex,
    std::size_t supervisedOffset,
    int targetToken,
    float loss,
    float targetProb,
    std::int64_t targetTrainFreq,
    std::int64_t targetTrainFreqTotal)
{
    HardTokenDebugRecord record{};
    record.sampleIndex = sampleIndex;
    record.seqIndex = seqIndex;
    record.supervisedOffset = supervisedOffset;
    record.targetToken = targetToken;
    record.loss = loss;
    record.targetProb = targetProb;
    record.targetTrainFreq = targetTrainFreq;
    if (targetTrainFreqTotal > 0) {
        record.targetTrainFreqPct =
            100.0f * static_cast<float>(targetTrainFreq) /
            static_cast<float>(targetTrainFreqTotal);
    }

    if (!logits || vocabSize == 0 ||
        targetToken < 0 || static_cast<std::size_t>(targetToken) >= vocabSize) {
        return record;
    }

    const float targetLogit = logits[static_cast<std::size_t>(targetToken)];
    float topLogit = -std::numeric_limits<float>::max();
    std::int64_t betterCount = 0;
    for (std::size_t v = 0; v < vocabSize; ++v) {
        const float value = logits[v];
        if (value > topLogit) {
            topLogit = value;
        }
        if (value > targetLogit) {
            ++betterCount;
        }
    }
    record.rank = betterCount + 1;
    record.margin = topLogit - targetLogit;
    return record;
}

void maybePrintHardTokenDebug(
    int step,
    int logInterval,
    const std::vector<HardTokenDebugRecord>& records)
{
    if (!hardTokenDebugEnabled()) return;
    if (records.empty()) return;
    if (logInterval > 0 && step % logInterval != 0) return;

    std::cout << "[HardTokens step " << step << "]";
    for (const auto& record : records) {
        std::cout << " | idx=" << record.sampleIndex
            << ",t=" << record.seqIndex
            << ",supOff=" << record.supervisedOffset
            << ",target=" << record.targetToken
            << ",loss=" << record.loss
            << ",prob=" << record.targetProb
            << ",rank=" << record.rank
            << ",margin=" << record.margin
            << ",trainFreq=" << record.targetTrainFreq
            << ",trainFreqPct=" << record.targetTrainFreqPct;
    }
    std::cout << '\n';
}

void observeTrainPrediction(
    const float* logits,
    std::size_t vocabSize,
    int targetToken,
    std::size_t supervisedOffset,
    float loss,
    float targetProb,
    const LogitStats& logitStats,
    TrainStepDebugMetrics& metrics)
{
    if (!logits || vocabSize == 0) return;
    if (targetToken < 0 || static_cast<std::size_t>(targetToken) >= vocabSize) return;

    const float targetLogit = logits[static_cast<std::size_t>(targetToken)];
    float topLogit = -std::numeric_limits<float>::max();
    std::int64_t betterCount = 0;
    for (std::size_t v = 0; v < vocabSize; ++v) {
        const float value = logits[v];
        if (value > topLogit) {
            topLogit = value;
        }
        if (value > targetLogit) {
            ++betterCount;
        }
    }

    const std::int64_t rank = betterCount + 1;
    ++metrics.tokens;
    metrics.lossSum += static_cast<double>(loss);
    metrics.targetProbSum += static_cast<double>(targetProb);
    metrics.rankSum += static_cast<double>(rank);
    metrics.marginSum += static_cast<double>(topLogit - targetLogit);
    metrics.logitStdSum += logitStats.stddev;
    if (rank <= 1) ++metrics.top1;
    if (rank <= 5) ++metrics.top5;
    if (rank <= 10) ++metrics.top10;
    if (rank <= 50) ++metrics.top50;

    const std::size_t bucket =
        (supervisedOffset < 16) ? 0 :
        (supervisedOffset < 32) ? 1 :
        (supervisedOffset < 64) ? 2 : 3;
    ++metrics.bucketTokens[bucket];
    metrics.bucketLossSum[bucket] += static_cast<double>(loss);
    metrics.bucketRankSum[bucket] += static_cast<double>(rank);
    metrics.bucketTargetProbSum[bucket] += static_cast<double>(targetProb);
}

void maybePrintTrainMetrics(
    int step,
    int logInterval,
    const TrainStepDebugMetrics& metrics)
{
    if (!trainMetricsDebugEnabled()) return;
    if (logInterval > 0 && step % logInterval != 0) return;
    if (metrics.tokens <= 0) return;

    const double invTokens = 1.0 / static_cast<double>(metrics.tokens);
    const double dLogitsRms =
        (metrics.dLogitsValues > 0)
        ? std::sqrt(metrics.dLogitsSq / static_cast<double>(metrics.dLogitsValues))
        : 0.0;
    std::cout << "[TrainMetrics step " << step << "] "
        << "tokens=" << metrics.tokens
        << " | samples=" << metrics.samples
        << " | loss=" << (metrics.lossSum * invTokens)
        << " | rawLoss=" << (metrics.rawLossSum * invTokens)
        << " | sampleLossRange=" << (metrics.maxSampleLoss - metrics.minSampleLoss)
        << " | sampleLossMin=" << metrics.minSampleLoss
        << " | sampleLossMax=" << metrics.maxSampleLoss
        << " | rawSampleLossRange=" << (metrics.maxRawSampleLoss - metrics.minRawSampleLoss)
        << " | rawSampleLossMin=" << metrics.minRawSampleLoss
        << " | rawSampleLossMax=" << metrics.maxRawSampleLoss
        << " | maxRawTokenLoss=" << metrics.maxRawTokenLoss
        << " | maxRawTokenSample=" << metrics.maxRawTokenSampleIndex
        << " | maxRawTokenSeq=" << metrics.maxRawTokenSeqIndex
        << " | maxRawTokenSupOff=" << metrics.maxRawTokenSupervisedOffset
        << " | maxRawTokenTarget=" << metrics.maxRawTokenTarget
        << " | maxRawTokenRank=" << metrics.maxRawTokenRank
        << " | maxRawTokenMargin=" << metrics.maxRawTokenMargin
        << " | maxRawTokenTrainFreq=" << metrics.maxRawTokenTrainFreq
        << " | maxRawTokenTrainFreqPct=" << metrics.maxRawTokenTrainFreqPct
        << " | cappedFrac=" << (static_cast<double>(metrics.cappedTokens) * invTokens)
        << " | capScaleAvg=" << (
            metrics.cappedTokens > 0
            ? metrics.capScaleSum / static_cast<double>(metrics.cappedTokens)
            : 1.0)
        << " | freqScaledFrac=" << (static_cast<double>(metrics.freqScaledTokens) * invTokens)
        << " | freqScaleAvg=" << (
            metrics.freqScaledTokens > 0
            ? metrics.freqScaleSum / static_cast<double>(metrics.freqScaledTokens)
            : 1.0)
        << " | wrongMarginFrac=" << (static_cast<double>(metrics.wrongMarginTokens) * invTokens)
        << " | wrongMarginPenalty=" << (metrics.wrongMarginPenaltySum * invTokens)
        << " | wrongMarginExcessMax=" << metrics.wrongMarginExcessMax
        << " | rareWrongMarginFrac=" << (static_cast<double>(metrics.rareWrongMarginTokens) * invTokens)
        << " | rareWrongMarginPenalty=" << (metrics.rareWrongMarginPenaltySum * invTokens)
        << " | rareWrongMarginExcessMax=" << metrics.rareWrongMarginExcessMax
        << " | hardBoostFrac=" << (static_cast<double>(metrics.hardBoostedTokens) * invTokens)
        << " | hardBoostScaleAvg=" << (
            metrics.hardBoostedTokens > 0
            ? metrics.hardBoostScaleSum / static_cast<double>(metrics.hardBoostedTokens)
            : 1.0)
        << " | rareHardBoostFrac=" << (static_cast<double>(metrics.rareHardBoostedTokens) * invTokens)
        << " | rareHardBoostScaleAvg=" << (
            metrics.rareHardBoostedTokens > 0
            ? metrics.rareHardBoostScaleSum / static_cast<double>(metrics.rareHardBoostedTokens)
            : 1.0)
        << " | rareTargetBiasFrac=" << (static_cast<double>(metrics.rareTargetBiasUpdates) * invTokens)
        << " | rareTargetBiasDeltaAvg=" << (
            metrics.rareTargetBiasUpdates > 0
            ? metrics.rareTargetBiasDeltaAbsSum / static_cast<double>(metrics.rareTargetBiasUpdates)
            : 0.0)
        << " | rareTargetBiasMaxAbs=" << metrics.rareTargetBiasMaxAbs
        << " | replayWeightedSamples=" << metrics.replayWeightedSamples
        << " | replayWeightAvg=" << (
            metrics.replayWeightedSamples > 0
            ? metrics.replayWeightSum / static_cast<double>(metrics.replayWeightedSamples)
            : 1.0)
        << " | gradientOnlyReplaySamples=" << metrics.gradientOnlyReplaySamples
        << " | gradientOnlyReplayTokens=" << metrics.gradientOnlyReplayTokens
        << " | targetProb=" << (metrics.targetProbSum * invTokens)
        << " | rank=" << (metrics.rankSum * invTokens)
        << " | margin=" << (metrics.marginSum * invTokens)
        << " | logitStd=" << (metrics.logitStdSum * invTokens)
        << " | top1=" << (static_cast<double>(metrics.top1) * invTokens)
        << " | top10=" << (static_cast<double>(metrics.top10) * invTokens)
        << " | top50=" << (static_cast<double>(metrics.top50) * invTokens)
        << " | dLogitsRms=" << dLogitsRms
        << " | dLogitsMax=" << metrics.dLogitsMaxAbs;

    for (std::size_t i = 0; i < metrics.bucketTokens.size(); ++i) {
        const std::int64_t n = metrics.bucketTokens[i];
        if (n <= 0) continue;
        const double inv = 1.0 / static_cast<double>(n);
        std::cout << " | b" << i
            << ":n=" << n
            << ",loss=" << (metrics.bucketLossSum[i] * inv)
            << ",prob=" << (metrics.bucketTargetProbSum[i] * inv)
            << ",rank=" << (metrics.bucketRankSum[i] * inv);
    }
    std::cout << '\n';
}

void updateEvalPredictionStats(
    const float* logits,
    std::size_t vocabSize,
    int targetToken,
    std::size_t supervisedOffset,
    float loss,
    float targetProb,
    EvalPredictionStats& stats)
{
    if (!logits || vocabSize == 0) return;
    if (targetToken < 0 || static_cast<std::size_t>(targetToken) >= vocabSize) return;

    const float targetLogit = logits[static_cast<std::size_t>(targetToken)];
    float topLogit = -std::numeric_limits<float>::max();
    std::int64_t betterCount = 0;

    for (std::size_t v = 0; v < vocabSize; ++v) {
        const float value = logits[v];
        if (value > topLogit) {
            topLogit = value;
        }
        if (value > targetLogit) {
            ++betterCount;
        }
    }

    const std::int64_t rank = betterCount + 1;
    ++stats.tokens;
    stats.rankSum += static_cast<double>(rank);
    stats.targetProbSum += static_cast<double>(targetProb);
    stats.marginSum += static_cast<double>(topLogit - targetLogit);
    if (rank <= 1) ++stats.top1;
    if (rank <= 5) ++stats.top5;
    if (rank <= 10) ++stats.top10;
    if (rank <= 50) ++stats.top50;

    const std::size_t bucket =
        (supervisedOffset < 16) ? 0 :
        (supervisedOffset < 32) ? 1 :
        (supervisedOffset < 64) ? 2 : 3;
    ++stats.bucketTokens[bucket];
    stats.bucketLossSum[bucket] += static_cast<double>(loss);
    stats.bucketRankSum[bucket] += static_cast<double>(rank);
    stats.bucketTargetProbSum[bucket] += static_cast<double>(targetProb);
}

} // namespace

void ProgressReporter::onStep(
    int step,
    float loss,
    float learningRate,
    std::int64_t supervisedTokens,
    std::int64_t totalSupervisedTokens,
    int maxSteps,
    int logInterval,
    const std::chrono::steady_clock::time_point& start,
    int epoch,
    const std::chrono::steady_clock::time_point& epochStart) {

    if (step <= 0) return;

    const float stepsPerSec = computeStepsPerSec(step, start);
    const double elapsedSinceStart = elapsedSeconds(start);
    const double supervisedTokensPerSec =
        (elapsedSinceStart > 0.0)
        ? static_cast<double>(totalSupervisedTokens) / elapsedSinceStart
        : 0.0;

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

    currentMa8_ = lossWindow8_.empty() ? loss : (lossSum8_ / static_cast<float>(lossWindow8_.size()));
    currentMa32_ = lossWindow32_.empty() ? loss : (lossSum32_ / static_cast<float>(lossWindow32_.size()));
    hasLoss_ = true;

    if (maxSteps <= 0) return;
    if (logInterval <= 0) return;
    if (step % logInterval != 0) return;

    std::cout << "[Step " << step << "/" << maxSteps << "] "
        << "loss=" << loss
        << " | ma8=" << currentMa8_
        << " | ma32=" << currentMa32_
        << " | lr=" << learningRate
        << " | supTokens=" << supervisedTokens
        << " | totalSupTokens=" << totalSupervisedTokens
        << " | supTok/s=" << supervisedTokensPerSec
        << " | epoch=" << epoch
        << " | epochTime(s)=" << epochElapsedSec
        << " | steps/s=" << stepsPerSec
        << " | ETA(s)=" << etaSec << '\n';
}

PlateauLrDecayDetector::PlateauLrDecayDetector(
    bool enabled,
    int minSteps,
    int patienceSteps,
    float minRelImprovement)
    : enabled_{ enabled }
    , minSteps_{ std::max(0, minSteps) }
    , patienceSteps_{ std::max(1, patienceSteps) }
    , minRelImprovement_{ std::max(0.0f, minRelImprovement) }
{
}

std::optional<int> PlateauLrDecayDetector::observe(int step, float loss)
{
    if (!enabled_ || triggered_ || step <= 0) return std::nullopt;

    lossWindow32_.push_back(loss);
    lossSum32_ += loss;
    if (lossWindow32_.size() > 32) {
        lossSum32_ -= lossWindow32_.front();
        lossWindow32_.pop_front();
    }

    currentMa32_ =
        lossWindow32_.empty()
        ? loss
        : (lossSum32_ / static_cast<float>(lossWindow32_.size()));

    // Wait for a full MA32 window so the trigger does not react to noisy
    // startup loss swings.
    if (lossWindow32_.size() < 32 || step < minSteps_) {
        return std::nullopt;
    }

    const float improvementThreshold =
        haveBest_
        ? (bestMa32_ * (1.0f - minRelImprovement_))
        : currentMa32_;

    if (!haveBest_ || currentMa32_ < improvementThreshold) {
        haveBest_ = true;
        bestMa32_ = currentMa32_;
        bestStep_ = step;
        return std::nullopt;
    }

    if ((step - bestStep_) >= patienceSteps_) {
        triggered_ = true;
        return step;
    }

    return std::nullopt;
}

LossBounceStopDetector::LossBounceStopDetector(const TrainerConfig& cfg)
    : enabled_{ cfg.lossBounceStopEnabled }
    , minSupTokens_{ std::max<std::int64_t>(0, cfg.lossBounceStopMinSupTokens) }
    , startLoss_{ std::max(0.0f, cfg.lossBounceStopStartLoss) }
    , window_{ std::max(3, cfg.lossBounceStopWindow) }
    , patienceSteps_{ std::max(1, cfg.lossBounceStopPatienceSteps) }
    , minDelta_{ std::max(0.0f, cfg.lossBounceStopMinDelta) }
    , minRange_{ std::max(0.0f, cfg.lossBounceStopMinRange) }
    , minDirectionChanges_{ std::max(0, cfg.lossBounceStopMinDirectionChanges) }
{
}

bool LossBounceStopDetector::observe(
    int step,
    float loss,
    std::int64_t totalSupervisedTokens)
{
    if (!enabled_ || step <= 0) return false;

    recentLosses_.push_back(loss);
    while (static_cast<int>(recentLosses_.size()) > window_) {
        recentLosses_.pop_front();
    }

    if (!haveBest_ || loss < (bestLoss_ - minDelta_)) {
        haveBest_ = true;
        bestLoss_ = loss;
        bestStep_ = step;
    }

    if (totalSupervisedTokens < minSupTokens_) return false;
    if (startLoss_ > 0.0f && (!haveBest_ || bestLoss_ > startLoss_)) return false;
    if (static_cast<int>(recentLosses_.size()) < window_) return false;
    if ((step - bestStep_) < patienceSteps_) return false;

    auto [minIt, maxIt] = std::minmax_element(recentLosses_.begin(), recentLosses_.end());
    recentRange_ = *maxIt - *minIt;
    recentDirectionChanges_ = 0;
    int prevSign = 0;
    for (std::size_t i = 1; i < recentLosses_.size(); ++i) {
        const float delta = recentLosses_[i] - recentLosses_[i - 1];
        const int sign = (delta > minDelta_) ? 1 : ((delta < -minDelta_) ? -1 : 0);
        if (sign == 0) continue;
        if (prevSign != 0 && sign != prevSign) {
            ++recentDirectionChanges_;
        }
        prevSign = sign;
    }

    return recentRange_ >= minRange_ &&
        recentDirectionChanges_ >= minDirectionChanges_;
}

CpuQLoRATrainer::CpuQLoRATrainer(const TrainerConfig& cfg,
    Dataset& dataset,
    Dataset* evalDataset,
    CpuKernelBackend& backend,
    AdamWOptimizer& optimizer)
    : cfg_{ cfg }
    , dataset_{ dataset }
    , evalDataset_{ evalDataset }
    , backend_{ backend }
    , adamOptimizer_{ &optimizer }
    , lionOptimizer_{ nullptr }
    , checkpointer_{ cfg.checkpointDir } {
    rareTargetBias_.assign(backend_.shape().vocabSize, 0.0f);
    buildTargetTokenFrequency();
}

CpuQLoRATrainer::CpuQLoRATrainer(const TrainerConfig& cfg,
    Dataset& dataset,
    Dataset* evalDataset,
    CpuKernelBackend& backend,
    LionOptimizer& optimizer)
    : cfg_{ cfg }
    , dataset_{ dataset }
    , evalDataset_{ evalDataset }
    , backend_{ backend }
    , adamOptimizer_{ nullptr }
    , lionOptimizer_{ &optimizer }
    , muonOptimizer_{ nullptr }
    , checkpointer_{ cfg.checkpointDir } {
    rareTargetBias_.assign(backend_.shape().vocabSize, 0.0f);
    buildTargetTokenFrequency();
}

CpuQLoRATrainer::CpuQLoRATrainer(const TrainerConfig& cfg,
    Dataset& dataset,
    Dataset* evalDataset,
    CpuKernelBackend& backend,
    MuonOptimizer& optimizer)
    : cfg_{ cfg }
    , dataset_{ dataset }
    , evalDataset_{ evalDataset }
    , backend_{ backend }
    , adamOptimizer_{ nullptr }
    , lionOptimizer_{ nullptr }
    , muonOptimizer_{ &optimizer }
    , checkpointer_{ cfg.checkpointDir } {
    rareTargetBias_.assign(backend_.shape().vocabSize, 0.0f);
    buildTargetTokenFrequency();
}

void CpuQLoRATrainer::buildTargetTokenFrequency() {
    targetTokenFrequency_.clear();
    targetTokenFrequencyTotal_ = 0;

    for (std::size_t i = 0; i < dataset_.size(); ++i) {
        const TrainingSample& sample = dataset_.get(i);
        for (const int token : sample.targetIds) {
            if (token == cfg_.ignoreIndex) continue;
            ++targetTokenFrequency_[token];
            ++targetTokenFrequencyTotal_;
        }
    }

}

void CpuQLoRATrainer::observeOnlineHardReplay(
    int step,
    const std::vector<StepSampleDebugRecord>& sampleDebugRecords)
{
    if (cfg_.onlineHardReplayMinLoss <= 0.0f &&
        cfg_.onlineHardReplayMinMaxTokenLoss <= 0.0f) return;
    if (step < std::max(0, cfg_.onlineHardReplayStartStep)) return;
    if (cfg_.onlineHardReplayMaxSamples <= 0) return;
    if (sampleDebugRecords.empty()) return;

    for (const auto& record : sampleDebugRecords) {
        const float replayLoss = std::max(record.rawAvgLoss, record.avgLoss);
        const bool avgLossTrigger =
            cfg_.onlineHardReplayMinLoss > 0.0f &&
            replayLoss >= cfg_.onlineHardReplayMinLoss;
        const bool maxTokenTrigger =
            cfg_.onlineHardReplayMinMaxTokenLoss > 0.0f &&
            record.maxRawTokenLoss >= cfg_.onlineHardReplayMinMaxTokenLoss;
        if (!avgLossTrigger && !maxTokenTrigger) continue;
        if (record.sampleIndex >= dataset_.size()) continue;
        if (std::find(
                cfg_.replaySampleIndices.begin(),
                cfg_.replaySampleIndices.end(),
                record.sampleIndex) != cfg_.replaySampleIndices.end()) {
            continue;
        }
        if (std::find(
                onlineHardReplayIndices_.begin(),
                onlineHardReplayIndices_.end(),
                record.sampleIndex) != onlineHardReplayIndices_.end()) {
            continue;
        }

        onlineHardReplayIndices_.push_back(record.sampleIndex);
        while (static_cast<int>(onlineHardReplayIndices_.size()) >
               cfg_.onlineHardReplayMaxSamples) {
            onlineHardReplayIndices_.erase(onlineHardReplayIndices_.begin());
        }

        std::cout
            << "[OnlineHardReplay] added idx=" << record.sampleIndex
            << " avgLoss=" << record.avgLoss
            << " rawAvgLoss=" << record.rawAvgLoss
            << " maxRawTokenLoss=" << record.maxRawTokenLoss
            << " replayLoss=" << replayLoss
            << " trigger=" << (maxTokenTrigger ? "maxTok" : "avg")
            << " step=" << step
            << " pool=" << onlineHardReplayIndices_.size()
            << '\n';
    }
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

std::vector<const TrainingSample*> CpuQLoRATrainer::sampleBatch(
    int batchSize,
    int currentStep,
    std::vector<std::size_t>* outIndices,
    std::vector<float>* outLossWeights,
    std::vector<std::uint8_t>* outIsReplay) {
    std::vector<const TrainingSample*> batch;
    if (dataset_.size() == 0 || batchSize <= 0) return batch;
    if (outIndices) outIndices->clear();
    if (outLossWeights) outLossWeights->clear();
    if (outIsReplay) outIsReplay->clear();

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
            if (outIndices) outIndices->push_back(fixedIdx);
            if (outLossWeights) outLossWeights->push_back(1.0f);
            if (outIsReplay) outIsReplay->push_back(0);
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

    const std::size_t replayPoolSize =
        cfg_.replaySampleIndices.size() + onlineHardReplayIndices_.size();
    const int replayEverySteps = std::max(1, cfg_.replayEverySteps);
    const bool allowReplayThisBatch =
        replayPoolSize > 0 &&
        currentStep >= std::max(0, cfg_.replayInjectStartStep) &&
        (replayBatchCounter_++ % static_cast<std::size_t>(replayEverySteps)) == 0;
    const int replayCount = allowReplayThisBatch
        ? std::min<int>(batchSize, std::max(0, cfg_.replaySamplesPerBatch))
        : 0;
    const auto appendReplay = [&](int count) {
        for (int i = 0; i < count && replayPoolSize > 0; ++i) {
            const std::size_t poolOffset = replaySampleOffset_ % replayPoolSize;
            const std::size_t rawIdx =
                (poolOffset < cfg_.replaySampleIndices.size())
                ? cfg_.replaySampleIndices[poolOffset]
                : onlineHardReplayIndices_[poolOffset - cfg_.replaySampleIndices.size()];
            ++replaySampleOffset_;
            const std::size_t idx = std::min<std::size_t>(rawIdx, dataset_.size() - 1);
            batch.push_back(&dataset_.get(idx));
            if (outIndices) outIndices->push_back(idx);
            if (outLossWeights) {
                outLossWeights->push_back(std::max(0.0f, cfg_.replaySampleLossWeight));
            }
            if (outIsReplay) outIsReplay->push_back(1);
        }
    };
    if (!cfg_.appendReplaySamples) {
        appendReplay(replayCount);
    }

    for (int i = 0; i < batchSize; ++i) {
        if (static_cast<int>(batch.size()) >= batchSize) break;
        if (samplerOffset_ >= epochOrder_.size()) {
            if (lastCompletedEpochLogged_ != epoch_) {
                const double epochTimeSec = elapsedSeconds(epochStartTime_);
                std::cout << "[Epoch " << epoch_ << " complete] "
                    << "epochTime(s)=" << epochTimeSec << '\n';
                lastCompletedEpochLogged_ = epoch_;
            }
            epochOrder_ = makeEpochOrder();
            samplerOffset_ = 0;
            ++epoch_;
            epochStartTime_ = std::chrono::steady_clock::now();
        }
        const auto idx = epochOrder_[samplerOffset_++];
        batch.push_back(&dataset_.get(idx));
        if (outIndices) outIndices->push_back(idx);
        if (outLossWeights) outLossWeights->push_back(1.0f);
        if (outIsReplay) outIsReplay->push_back(0);
    }
    if (cfg_.appendReplaySamples) {
        appendReplay(replayCount);
    }

    return batch;
}

void CpuQLoRATrainer::trainStep(
    const std::vector<const TrainingSample*>& batch,
    const std::vector<std::size_t>& batchIndices,
    const std::vector<float>& batchLossWeights,
    const std::vector<std::uint8_t>& batchIsReplay,
    std::int64_t totalSupervisedTokensBeforeStep,
    float& accumLoss,
    std::int64_t& accumTokens,
    std::vector<StepSampleDebugRecord>* sampleDebugRecords,
    TrainStepDebugMetrics* trainDebugMetrics,
    std::vector<HardTokenDebugRecord>* hardTokenDebugRecords)
{
    const std::size_t V = backend_.shape().vocabSize;
    const bool collectTrainMetrics = trainDebugMetrics && trainMetricsDebugEnabled();
    const bool collectHardTokens = hardTokenDebugRecords && hardTokenDebugEnabled();
    const int hardTokenTopK = collectHardTokens ? hardTokenDebugTopK() : 0;
    const bool tokenLossCapActive =
        cfg_.tokenLossCap > 0.0f &&
        totalSupervisedTokensBeforeStep >= cfg_.tokenLossCapMinTotalSupTokens;
    const bool labelSmoothingActive =
        cfg_.labelSmoothing > 0.0f &&
        totalSupervisedTokensBeforeStep >= cfg_.labelSmoothingMinTotalSupTokens;
    const float labelSmoothing = labelSmoothingActive
        ? std::clamp(cfg_.labelSmoothing, 0.0f, 0.5f)
        : 0.0f;
    const bool logitL2Active =
        cfg_.logitL2Penalty > 0.0f &&
        totalSupervisedTokensBeforeStep >= cfg_.logitL2MinTotalSupTokens;
    const float logitL2Penalty = logitL2Active
        ? std::max(0.0f, cfg_.logitL2Penalty)
        : 0.0f;
    const bool wrongTopMarginActive =
        cfg_.wrongTopMarginPenalty > 0.0f &&
        totalSupervisedTokensBeforeStep >= cfg_.wrongTopMarginMinTotalSupTokens;
    const float wrongTopMarginPenalty = wrongTopMarginActive
        ? std::max(0.0f, cfg_.wrongTopMarginPenalty)
        : 0.0f;
    const float wrongTopMarginThreshold =
        std::max(0.0f, cfg_.wrongTopMarginThreshold);
    const bool rareWrongTopMarginActive =
        cfg_.rareWrongTopMarginMaxTrainFreq > 0 &&
        cfg_.rareWrongTopMarginPenalty > 0.0f &&
        totalSupervisedTokensBeforeStep >= cfg_.rareWrongTopMarginMinTotalSupTokens;
    const int rareWrongTopMarginMaxTrainFreq =
        std::max(0, cfg_.rareWrongTopMarginMaxTrainFreq);
    const float rareWrongTopMarginPenalty = rareWrongTopMarginActive
        ? std::max(0.0f, cfg_.rareWrongTopMarginPenalty)
        : 0.0f;
    const float rareWrongTopMarginThreshold =
        std::max(0.0f, cfg_.rareWrongTopMarginThreshold);
    const bool rareTargetBiasActive =
        cfg_.rareTargetBiasMaxTrainFreq > 0 &&
        cfg_.rareTargetBiasLrScale > 0.0f &&
        cfg_.rareTargetBiasMaxAbs > 0.0f &&
        totalSupervisedTokensBeforeStep >= cfg_.rareTargetBiasMinTotalSupTokens;
    const int rareTargetBiasMaxTrainFreq =
        std::max(0, cfg_.rareTargetBiasMaxTrainFreq);
    const float rareTargetBiasMinLoss =
        std::max(0.0f, cfg_.rareTargetBiasMinLoss);
    const float rareTargetBiasStep =
        cfg_.learningRate * std::max(0.0f, cfg_.rareTargetBiasLrScale);
    const float rareTargetBiasMaxAbs =
        std::max(0.0f, cfg_.rareTargetBiasMaxAbs);
    const bool hardTokenBoostActive =
        cfg_.hardTokenBoostThreshold > 0.0f &&
        cfg_.hardTokenBoostSlope > 0.0f &&
        cfg_.hardTokenBoostMaxScale > 1.0f &&
        totalSupervisedTokensBeforeStep >= cfg_.hardTokenBoostMinTotalSupTokens;
    const float hardTokenBoostThreshold = std::max(cfg_.hardTokenBoostThreshold, 1e-6f);
    const float hardTokenBoostSlope = cfg_.hardTokenBoostSlope;
    const float hardTokenBoostMaxScale = std::max(1.0f, cfg_.hardTokenBoostMaxScale);
    const bool rareHardTokenBoostActive =
        cfg_.rareHardTokenBoostMaxTrainFreq > 0 &&
        cfg_.rareHardTokenBoostThreshold > 0.0f &&
        cfg_.rareHardTokenBoostScale > 1.0f &&
        totalSupervisedTokensBeforeStep >= cfg_.rareHardTokenBoostMinTotalSupTokens;
    const int rareHardTokenBoostMaxTrainFreq =
        std::max(0, cfg_.rareHardTokenBoostMaxTrainFreq);
    const float rareHardTokenBoostThreshold =
        std::max(cfg_.rareHardTokenBoostThreshold, 1e-6f);
    const float rareHardTokenBoostScale =
        std::max(1.0f, cfg_.rareHardTokenBoostScale);

    for (std::size_t batchItem = 0; batchItem < batch.size(); ++batchItem) {
        const auto* sample = batch[batchItem];
        const float sampleLossWeight =
            (batchItem < batchLossWeights.size())
            ? std::max(0.0f, batchLossWeights[batchItem])
            : 1.0f;
        const bool isReplaySample =
            batchItem < batchIsReplay.size() && batchIsReplay[batchItem] != 0;
        const bool gradientOnlyReplay =
            cfg_.replayGradientOnly && isReplaySample;
        const bool includeSampleInMetrics = !gradientOnlyReplay;
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
        float sampleWeightSum = 0.0f;
        int supervisedSeen = 0;
        for (std::size_t t = 0; t < T; ++t) {
            if (sample->attentionMask[t] == 0) continue;
            const int y = sample->targetIds[t];
            if (y == cfg_.ignoreIndex) continue;
            if (y < 0 || static_cast<std::size_t>(y) >= V) continue;
            if (cfg_.maxSupervisedTokensPerSample > 0 &&
                supervisedSeen >= cfg_.maxSupervisedTokensPerSample) {
                ++supervisedSeen;
                continue;
            }
            lossMask[t] = 1;
            ++sampleTokenCount;
            sampleWeightSum += supervisedTokenLossWeight(
                cfg_,
                static_cast<std::size_t>(supervisedSeen));
            ++supervisedSeen;
        }

        if (sampleTokenCount == 0) continue;
        if (sampleWeightSum <= 0.0f) {
            sampleWeightSum = static_cast<float>(sampleTokenCount);
        }

        const std::size_t needed = backend_.logitsCount(T);
        if (logitsBuf_.size() != needed) {
            logitsBuf_.resize(needed);
        }
        if (dLogitsBuf_.size() != needed) {
            dLogitsBuf_.resize(needed);
        }
        std::fill(dLogitsBuf_.begin(), dLogitsBuf_.end(), 0.0f);

        backend_.forward(
            std::span<const int>(sample->inputIds.data(), T),
            std::span<const int>(sample->attentionMask.data(), T),
            std::span<float>(logitsBuf_.data(), logitsBuf_.size()),
            std::span<const int>(lossMask.data(), lossMask.size())
        );
        if (rareTargetBiasActive && rareTargetBias_.size() == V) {
            for (std::size_t t = 0; t < T; ++t) {
                if (lossMask[t] == 0) continue;
                float* logits_t = &logitsBuf_[t * V];
                for (std::size_t v = 0; v < V; ++v) {
                    logits_t[v] += rareTargetBias_[v];
                }
            }
        }

        float sampleLossSum = 0.0f;
        float sampleRawLossSum = 0.0f;
        float sampleMaxRawTokenLoss = 0.0f;
        std::int64_t sampleCappedTokens = 0;
        double sampleCapScaleSum = 0.0;
        std::int64_t sampleFreqScaledTokens = 0;
        double sampleFreqScaleSum = 0.0;
        double sampleTargetTrainFreqSum = 0.0;
        std::int64_t sampleRare10Tokens = 0;
        std::int64_t sampleRare20Tokens = 0;
        const float weightNorm = static_cast<float>(sampleTokenCount) / sampleWeightSum;
        const bool validateLogits = logitsValidationEnabled();
        std::size_t supervisedOffset = 0;

        // target_ids are already next-token labels for each timestep t.
        for (std::size_t t = 0; t < T; ++t) {
            if (lossMask[t] == 0) continue;

            const int y = sample->targetIds[t];

            const float* logits_t = &logitsBuf_[t * V];
            float* dlog_t = &dLogitsBuf_[t * V];
            LogitStats stats{};

            if (validateLogits ||
                collectTrainMetrics ||
                wrongTopMarginActive ||
                rareWrongTopMarginActive) {
                stats = inspectLogits(logits_t, V, y, t);
            }
            if (validateLogits) {
                maybePrintLogitStats(stats, y, t);
                validateLogitsOrThrow(stats, t, y);
            }

            const float tokenLoss = cross_entropy_dlogits_row(
                logits_t,
                dlog_t,
                V,
                y
            );
            const float targetProb = dlog_t[static_cast<std::size_t>(y)] + 1.0f;
            if (labelSmoothing > 0.0f) {
                const float uniform = labelSmoothing / static_cast<float>(V);
                for (std::size_t v = 0; v < V; ++v) {
                    dlog_t[v] -= uniform;
                }
                dlog_t[static_cast<std::size_t>(y)] += labelSmoothing;
            }
            std::int64_t targetTrainFreq = 0;
            if (const auto it = targetTokenFrequency_.find(y);
                it != targetTokenFrequency_.end()) {
                targetTrainFreq = it->second;
            }
            const float tokenWeight =
                supervisedTokenLossWeight(cfg_, supervisedOffset) * weightNorm;
            const float freqScale = targetFrequencyLossScale(
                cfg_,
                targetTrainFreq,
                totalSupervisedTokensBeforeStep);
            sampleTargetTrainFreqSum += static_cast<double>(targetTrainFreq);
            if (targetTrainFreq <= 10) {
                ++sampleRare10Tokens;
            }
            if (targetTrainFreq <= 20) {
                ++sampleRare20Tokens;
            }

            float effectiveTokenLoss = tokenLoss;
            float tokenGradScale = tokenWeight * freqScale * sampleLossWeight;
            float tokenCapScale = 1.0f;
            float hardBoostScale = 1.0f;
            float rareHardBoostScale = 1.0f;
            if (hardTokenBoostActive && tokenLoss > hardTokenBoostThreshold) {
                const float excessRatio =
                    (tokenLoss - hardTokenBoostThreshold) / hardTokenBoostThreshold;
                hardBoostScale = std::min(
                    hardTokenBoostMaxScale,
                    1.0f + hardTokenBoostSlope * excessRatio);
                tokenGradScale *= hardBoostScale;
            }
            if (rareHardTokenBoostActive &&
                tokenLoss > rareHardTokenBoostThreshold &&
                targetTrainFreq > 0 &&
                targetTrainFreq <= rareHardTokenBoostMaxTrainFreq) {
                rareHardBoostScale = rareHardTokenBoostScale;
                tokenGradScale *= rareHardBoostScale;
            }
            if (rareTargetBiasActive &&
                targetTrainFreq > 0 &&
                targetTrainFreq <= rareTargetBiasMaxTrainFreq &&
                tokenLoss >= rareTargetBiasMinLoss &&
                static_cast<std::size_t>(y) < rareTargetBias_.size()) {
                const float sparseGrad = tokenWeight * sampleLossWeight * (targetProb - 1.0f);
                const float oldBias = rareTargetBias_[static_cast<std::size_t>(y)];
                const float newBias = std::clamp(
                    oldBias - rareTargetBiasStep * sparseGrad,
                    -rareTargetBiasMaxAbs,
                    rareTargetBiasMaxAbs);
                rareTargetBias_[static_cast<std::size_t>(y)] = newBias;
                if (collectTrainMetrics && includeSampleInMetrics) {
                    ++trainDebugMetrics->rareTargetBiasUpdates;
                    trainDebugMetrics->rareTargetBiasDeltaAbsSum +=
                        std::fabs(newBias - oldBias);
                    trainDebugMetrics->rareTargetBiasMaxAbs = std::max(
                        trainDebugMetrics->rareTargetBiasMaxAbs,
                        std::fabs(newBias));
                }
            }
            if (tokenLossCapActive && tokenLoss > cfg_.tokenLossCap) {
                if (cfg_.tokenLossCapSoftness > 0.0f) {
                    const float excess = tokenLoss - cfg_.tokenLossCap;
                    const float softness = cfg_.tokenLossCapSoftness;
                    tokenCapScale = 1.0f / (1.0f + excess / softness);
                    effectiveTokenLoss =
                        cfg_.tokenLossCap +
                        softness * std::log1p(excess / softness);
                }
                else {
                    tokenCapScale = cfg_.tokenLossCap / std::max(tokenLoss, 1e-12f);
                    effectiveTokenLoss = cfg_.tokenLossCap;
                }
                tokenGradScale *= tokenCapScale;
            }

            if (tokenGradScale != 1.0f) {
                scale_inplace_avx2_or_scalar(dlog_t, V, tokenGradScale);
            }
            float tokenObjectiveLoss =
                effectiveTokenLoss *
                tokenWeight *
                freqScale *
                hardBoostScale *
                rareHardBoostScale *
                sampleLossWeight;
            if (logitL2Penalty > 0.0f) {
                double logitSqSum = 0.0;
                for (std::size_t v = 0; v < V; ++v) {
                    const float z = logits_t[v];
                    logitSqSum += static_cast<double>(z) * static_cast<double>(z);
                }
                const float invV = 1.0f / static_cast<float>(V);
                const float regWeight = tokenWeight * freqScale * sampleLossWeight;
                const float regGradScale = regWeight * logitL2Penalty * invV;
                for (std::size_t v = 0; v < V; ++v) {
                    dlog_t[v] += regGradScale * logits_t[v];
                }
                tokenObjectiveLoss +=
                    0.5f * logitL2Penalty *
                    static_cast<float>(logitSqSum * static_cast<double>(invV)) *
                    regWeight;
            }
            if (wrongTopMarginPenalty > 0.0f &&
                stats.maxIndex != static_cast<std::size_t>(y)) {
                const float margin = stats.maxLogit - stats.targetLogit;
                const float excess = margin - wrongTopMarginThreshold;
                if (excess > 0.0f) {
                    const float regWeight = tokenWeight * freqScale * tokenCapScale * sampleLossWeight;
                    const float regGrad = regWeight * wrongTopMarginPenalty * excess;
                    dlog_t[stats.maxIndex] += regGrad;
                    dlog_t[static_cast<std::size_t>(y)] -= regGrad;
                    const float regLoss =
                        0.5f * wrongTopMarginPenalty * excess * excess * regWeight;
                    tokenObjectiveLoss += regLoss;
                    if (collectTrainMetrics && includeSampleInMetrics) {
                        ++trainDebugMetrics->wrongMarginTokens;
                        trainDebugMetrics->wrongMarginPenaltySum += regLoss;
                        trainDebugMetrics->wrongMarginExcessMax = std::max(
                            trainDebugMetrics->wrongMarginExcessMax,
                            excess);
                    }
                }
            }
            if (rareWrongTopMarginPenalty > 0.0f &&
                stats.maxIndex != static_cast<std::size_t>(y) &&
                targetTrainFreq > 0 &&
                targetTrainFreq <= rareWrongTopMarginMaxTrainFreq) {
                const float margin = stats.maxLogit - stats.targetLogit;
                const float excess = margin - rareWrongTopMarginThreshold;
                if (excess > 0.0f) {
                    const float regWeight =
                        tokenWeight * freqScale * tokenCapScale * sampleLossWeight;
                    const float regGrad = regWeight * rareWrongTopMarginPenalty * excess;
                    dlog_t[stats.maxIndex] += regGrad;
                    dlog_t[static_cast<std::size_t>(y)] -= regGrad;
                    const float regLoss =
                        0.5f * rareWrongTopMarginPenalty * excess * excess * regWeight;
                    tokenObjectiveLoss += regLoss;
                    if (collectTrainMetrics && includeSampleInMetrics) {
                        ++trainDebugMetrics->rareWrongMarginTokens;
                        trainDebugMetrics->rareWrongMarginPenaltySum += regLoss;
                        trainDebugMetrics->rareWrongMarginExcessMax = std::max(
                            trainDebugMetrics->rareWrongMarginExcessMax,
                            excess);
                    }
                }
            }
            sampleLossSum += tokenObjectiveLoss;
            if (collectTrainMetrics && includeSampleInMetrics) {
                const float rawWeightedTokenLoss = tokenLoss * tokenWeight;
                trainDebugMetrics->rawLossSum += static_cast<double>(rawWeightedTokenLoss);
                if (rawWeightedTokenLoss > trainDebugMetrics->maxRawTokenLoss) {
                    trainDebugMetrics->maxRawTokenLoss = rawWeightedTokenLoss;
                    trainDebugMetrics->maxRawTokenSampleIndex =
                        (batchItem < batchIndices.size()) ? batchIndices[batchItem] : 0;
                    trainDebugMetrics->maxRawTokenSeqIndex = t;
                    trainDebugMetrics->maxRawTokenSupervisedOffset = supervisedOffset;
                    trainDebugMetrics->maxRawTokenTarget = y;
                    trainDebugMetrics->maxRawTokenRank = stats.targetRank;
                    trainDebugMetrics->maxRawTokenMargin = stats.maxLogit - stats.targetLogit;
                    trainDebugMetrics->maxRawTokenTrainFreq = targetTrainFreq;
                    if (targetTokenFrequencyTotal_ > 0) {
                        trainDebugMetrics->maxRawTokenTrainFreqPct =
                            100.0f * static_cast<float>(targetTrainFreq) /
                            static_cast<float>(targetTokenFrequencyTotal_);
                    }
                    else {
                        trainDebugMetrics->maxRawTokenTrainFreqPct = 0.0f;
                    }
                }
                if (tokenCapScale < 1.0f) {
                    ++trainDebugMetrics->cappedTokens;
                    trainDebugMetrics->capScaleSum += tokenCapScale;
                }
                if (freqScale < 1.0f) {
                    ++trainDebugMetrics->freqScaledTokens;
                    trainDebugMetrics->freqScaleSum += freqScale;
                }
                if (hardBoostScale > 1.0f) {
                    ++trainDebugMetrics->hardBoostedTokens;
                    trainDebugMetrics->hardBoostScaleSum += hardBoostScale;
                }
                if (rareHardBoostScale > 1.0f) {
                    ++trainDebugMetrics->rareHardBoostedTokens;
                    trainDebugMetrics->rareHardBoostScaleSum += rareHardBoostScale;
                }
                observeTrainPrediction(
                    logits_t,
                    V,
                    y,
                    supervisedOffset,
                    tokenObjectiveLoss,
                    targetProb,
                    stats,
                    *trainDebugMetrics);
            }
            sampleRawLossSum += tokenLoss * tokenWeight;
            sampleMaxRawTokenLoss = std::max(
                sampleMaxRawTokenLoss,
                tokenLoss * tokenWeight);
            if (tokenCapScale < 1.0f) {
                ++sampleCappedTokens;
                sampleCapScaleSum += tokenCapScale;
            }
            if (freqScale < 1.0f) {
                ++sampleFreqScaledTokens;
                sampleFreqScaleSum += freqScale;
            }
            if (includeSampleInMetrics && collectHardTokens && batchItem < batchIndices.size()) {
                insertHardTokenRecord(
                    *hardTokenDebugRecords,
                    makeHardTokenRecord(
                        logits_t,
                        V,
                        batchIndices[batchItem],
                        t,
                        supervisedOffset,
                        y,
                        tokenLoss * tokenWeight,
                        targetProb,
                        targetTrainFreq,
                        targetTokenFrequencyTotal_),
                    hardTokenTopK);
            }
            // NOTE: do NOT increment sampleTokenCount here.
            // It was already counted in the lossMask-building loop above;
            // a second increment would halve the reported loss.
            ++supervisedOffset;
        }

        if (gradientOnlyReplay && collectTrainMetrics) {
            ++trainDebugMetrics->gradientOnlyReplaySamples;
            trainDebugMetrics->gradientOnlyReplayTokens += sampleTokenCount;
        }
        if (includeSampleInMetrics) {
            accumLoss += sampleLossSum;
            accumTokens += sampleTokenCount;
        }

        const float sampleAvgLoss = (sampleTokenCount > 0)
            ? (sampleLossSum / static_cast<float>(sampleTokenCount))
            : 0.0f;
        const float sampleRawAvgLoss = (sampleTokenCount > 0)
            ? (sampleRawLossSum / static_cast<float>(sampleTokenCount))
            : 0.0f;
        if (collectTrainMetrics && includeSampleInMetrics) {
            ++trainDebugMetrics->samples;
            if (sampleLossWeight != 1.0f) {
                ++trainDebugMetrics->replayWeightedSamples;
                trainDebugMetrics->replayWeightSum += sampleLossWeight;
            }
            if (trainDebugMetrics->samples == 1) {
                trainDebugMetrics->minSampleLoss = sampleAvgLoss;
                trainDebugMetrics->maxSampleLoss = sampleAvgLoss;
                trainDebugMetrics->minRawSampleLoss = sampleRawAvgLoss;
                trainDebugMetrics->maxRawSampleLoss = sampleRawAvgLoss;
            }
            else {
                trainDebugMetrics->minSampleLoss = std::min(trainDebugMetrics->minSampleLoss, sampleAvgLoss);
                trainDebugMetrics->maxSampleLoss = std::max(trainDebugMetrics->maxSampleLoss, sampleAvgLoss);
                trainDebugMetrics->minRawSampleLoss = std::min(trainDebugMetrics->minRawSampleLoss, sampleRawAvgLoss);
                trainDebugMetrics->maxRawSampleLoss = std::max(trainDebugMetrics->maxRawSampleLoss, sampleRawAvgLoss);
            }
        }

        if (includeSampleInMetrics && sampleDebugRecords && batchItem < batchIndices.size()) {
            const float invSampleTokens = sampleTokenCount > 0
                ? 1.0f / static_cast<float>(sampleTokenCount)
                : 0.0f;
            sampleDebugRecords->push_back({
                batchIndices[batchItem],
                sampleTokenCount,
                sampleAvgLoss,
                sampleRawAvgLoss,
                sampleMaxRawTokenLoss,
                static_cast<float>(sampleCappedTokens) * invSampleTokens,
                sampleCappedTokens > 0
                    ? static_cast<float>(sampleCapScaleSum / static_cast<double>(sampleCappedTokens))
                    : 1.0f,
                static_cast<float>(sampleFreqScaledTokens) * invSampleTokens,
                sampleFreqScaledTokens > 0
                    ? static_cast<float>(sampleFreqScaleSum / static_cast<double>(sampleFreqScaledTokens))
                    : 1.0f,
                static_cast<float>(sampleTargetTrainFreqSum) * invSampleTokens,
                static_cast<float>(sampleRare10Tokens) * invSampleTokens,
                static_cast<float>(sampleRare20Tokens) * invSampleTokens
                });
        }

        // Keep per-sample logits-gradient scaling explicit and tunable.
        // The historical trusted path used summed-token gradients here.
        // Averaging by sampleTokenCount materially changes effective LR.
        if (shouldNormalizeDLogitsBySampleTokens(cfg_) && sampleTokenCount > 1) {
            const float inv_n = 1.0f / static_cast<float>(sampleTokenCount);
            const std::size_t totalDLogits = T * V;
            scale_inplace_avx2_or_scalar(dLogitsBuf_.data(), totalDLogits, inv_n);
        }

        if (collectTrainMetrics && includeSampleInMetrics) {
            const std::size_t totalDLogits = T * V;
            for (std::size_t i = 0; i < totalDLogits; ++i) {
                const float v = dLogitsBuf_[i];
                trainDebugMetrics->dLogitsSq += static_cast<double>(v) * static_cast<double>(v);
                trainDebugMetrics->dLogitsMaxAbs = std::max(
                    trainDebugMetrics->dLogitsMaxAbs,
                    static_cast<float>(std::fabs(v)));
            }
            trainDebugMetrics->dLogitsValues += static_cast<std::int64_t>(totalDLogits);
        }

        backend_.backward(
            std::span<const int>(sample->inputIds.data(), T),
            std::span<const int>(sample->attentionMask.data(), T),
            std::span<const float>(dLogitsBuf_.data(), dLogitsBuf_.size()),
            std::span<const int>(lossMask.data(), lossMask.size())
        );
    }
}

float CpuQLoRATrainer::sampleLossSum(
    const TrainingSample& sample,
    std::int64_t& outTokens,
    EvalPredictionStats* outPredStats) {
    outTokens = 0;
    const std::size_t V = backend_.shape().vocabSize;
    std::vector<int> lossMask;

    const std::size_t T = sample.inputIds.size();
    if (sample.attentionMask.size() != T || sample.targetIds.size() != T) {
        throw std::runtime_error("sampleLossSum: sample sizes mismatch");
    }

    lossMask.assign(T, 0);
    std::int64_t sampleTokenCount = 0;
    int supervisedSeen = 0;
    for (std::size_t t = 0; t < T; ++t) {
        if (sample.attentionMask[t] == 0) continue;
        const int y = sample.targetIds[t];
        if (y == cfg_.ignoreIndex) continue;
        if (y < 0 || static_cast<std::size_t>(y) >= V) continue;
        if (cfg_.maxSupervisedTokensPerSample > 0 &&
            supervisedSeen >= cfg_.maxSupervisedTokensPerSample) {
            ++supervisedSeen;
            continue;
        }
        lossMask[t] = 1;
        ++sampleTokenCount;
        ++supervisedSeen;
    }
    if (sampleTokenCount == 0) return 0.0f;

    const std::size_t needed = backend_.logitsCount(T);
    if (logitsBuf_.size() != needed) {
        logitsBuf_.resize(needed);
    }
    if (dLogitsBuf_.size() != needed) {
        dLogitsBuf_.resize(needed);
    }

    backend_.forward(
        std::span<const int>(sample.inputIds.data(), T),
        std::span<const int>(sample.attentionMask.data(), T),
        std::span<float>(logitsBuf_.data(), logitsBuf_.size()),
        std::span<const int>(lossMask.data(), lossMask.size())
    );

    float lossSum = 0.0f;
    std::size_t supervisedOffset = 0;
    for (std::size_t t = 0; t < T; ++t) {
        if (lossMask[t] == 0) continue;
        const int y = sample.targetIds[t];
        const float* logits_t = &logitsBuf_[t * V];
        float* dlog_t = &dLogitsBuf_[t * V];
        const float tokenLoss = cross_entropy_dlogits_row(
            logits_t,
            dlog_t,
            V,
            y);
        lossSum += tokenLoss;
        if (outPredStats) {
            const float targetProb = dlog_t[static_cast<std::size_t>(y)] + 1.0f;
            updateEvalPredictionStats(
                logits_t,
                V,
                y,
                supervisedOffset,
                tokenLoss,
                targetProb,
                *outPredStats);
        }
        ++supervisedOffset;
    }

    outTokens = sampleTokenCount;
    return lossSum;
}

float CpuQLoRATrainer::datasetLoss(
    const Dataset& source,
    int maxBatches,
    std::int64_t& outTokens,
    EvalPredictionStats* outPredStats) {
    outTokens = 0;
    if (outPredStats) {
        *outPredStats = {};
    }

    const std::size_t batches = (maxBatches <= 0)
        ? source.size()
        : std::min<std::size_t>(source.size(), static_cast<std::size_t>(maxBatches));

    float lossSum = 0.0f;
    std::int64_t tokenCount = 0;

    for (std::size_t i = 0; i < batches; ++i) {
        const TrainingSample& sample = source.get(i);
        std::int64_t sampleTokenCount = 0;
        const float sampleLossSum = this->sampleLossSum(sample, sampleTokenCount, outPredStats);
        if (sampleTokenCount == 0) continue;
        lossSum += sampleLossSum;
        tokenCount += sampleTokenCount;
    }

    outTokens = tokenCount;
    return (tokenCount > 0)
        ? (lossSum / static_cast<float>(tokenCount))
        : 0.0f;
}

float CpuQLoRATrainer::evalLoss(
    int maxBatches,
    std::int64_t& outTokens,
    EvalPredictionStats* outPredStats) {
    outTokens = 0;
    if (!evalDataset_) return 0.0f;
    return datasetLoss(*evalDataset_, maxBatches, outTokens, outPredStats);
}

void CpuQLoRATrainer::maybeEval(int step, bool force) {
    if (cfg_.evalInterval <= 0 || cfg_.numEvalBatches <= 0) return;
    if (!force && step % cfg_.evalInterval != 0) return;
    if (!evalDataset_) return;

    const auto evalStart = std::chrono::steady_clock::now();
    std::int64_t evalTokens = 0;
    EvalPredictionStats predStats{};
    const float loss = evalLoss(
        cfg_.numEvalBatches,
        evalTokens,
        cfg_.evalPredictionStats ? &predStats : nullptr);
    const double elapsed = elapsedSeconds(evalStart);
    const double tokPerSec = elapsed > 0.0
        ? static_cast<double>(evalTokens) / elapsed
        : 0.0;
    std::cout << "[Eval step " << step << "] loss=" << loss
        << " | tokens=" << evalTokens
        << " | batches=" << cfg_.numEvalBatches
        << " | evalTime(s)=" << elapsed
        << " | evalTok/s=" << tokPerSec << '\n';
    if (cfg_.evalPredictionStats && predStats.tokens > 0) {
        const double denom = static_cast<double>(predStats.tokens);
        std::cout << "[EvalStats step " << step << "] "
            << "avgTargetRank=" << (predStats.rankSum / denom)
            << " | top1=" << (static_cast<double>(predStats.top1) / denom)
            << " | top5=" << (static_cast<double>(predStats.top5) / denom)
            << " | top10=" << (static_cast<double>(predStats.top10) / denom)
            << " | top50=" << (static_cast<double>(predStats.top50) / denom)
            << " | avgTargetProb=" << (predStats.targetProbSum / denom)
            << " | avgTopMinusTargetLogit=" << (predStats.marginSum / denom)
            << '\n';
        static constexpr const char* labels[4] = { "0_15", "16_31", "32_63", "64_plus" };
        for (std::size_t i = 0; i < predStats.bucketTokens.size(); ++i) {
            const std::int64_t n = predStats.bucketTokens[i];
            if (n <= 0) continue;
            const double bucketDenom = static_cast<double>(n);
            std::cout << "[EvalBucket step " << step << "] "
                << "bucket=" << labels[i]
                << " | tokens=" << n
                << " | loss=" << (predStats.bucketLossSum[i] / bucketDenom)
                << " | avgTargetRank=" << (predStats.bucketRankSum[i] / bucketDenom)
                << " | avgTargetProb=" << (predStats.bucketTargetProbSum[i] / bucketDenom)
                << '\n';
        }
    }

    if (cfg_.evalLossStopThreshold > 0.0f &&
        step >= cfg_.evalLossStopMinStep &&
        std::isfinite(loss) &&
        loss <= cfg_.evalLossStopThreshold)
    {
        stopRequested_ = true;
        stopCheckpointRequested_ = cfg_.checkpointOnEvalLossStop;
        stopReason_ = "eval loss threshold reached";
        std::cout << "[STOP request] eval loss threshold reached at step "
            << step
            << " evalLoss=" << loss
            << " <= evalLossStopThreshold=" << cfg_.evalLossStopThreshold
            << " evalLossStopMinStep=" << cfg_.evalLossStopMinStep
            << '\n';
    }
}

bool CpuQLoRATrainer::hasPendingTrainProbeGate(std::int64_t totalSupervisedTokens) const {
    const auto gatePending = [totalSupervisedTokens](
        std::int64_t gateSupTokens,
        float gateMaxLoss,
        bool gateDone) {
        return !gateDone &&
            gateSupTokens > 0 &&
            gateMaxLoss > 0.0f &&
            totalSupervisedTokens >= gateSupTokens;
    };

    return gatePending(
            cfg_.trainProbeGate1SupTokens,
            cfg_.trainProbeGate1MaxLoss,
            trainProbeGate1Done_) ||
        gatePending(
            cfg_.trainProbeGate2SupTokens,
            cfg_.trainProbeGate2MaxLoss,
            trainProbeGate2Done_);
}

void CpuQLoRATrainer::maybeTrainProbe(
    int step,
    std::int64_t totalSupervisedTokens,
    bool force) {
    if (cfg_.trainProbeInterval <= 0 || cfg_.numTrainProbeBatches <= 0) return;
    if (!force && step % cfg_.trainProbeInterval != 0) return;

    const auto probeStart = std::chrono::steady_clock::now();
    std::int64_t probeTokens = 0;
    const float loss = datasetLoss(
        dataset_,
        cfg_.numTrainProbeBatches,
        probeTokens,
        nullptr);
    const double elapsed = elapsedSeconds(probeStart);
    const double tokPerSec = elapsed > 0.0
        ? static_cast<double>(probeTokens) / elapsed
        : 0.0;
    std::cout << "[TrainProbe step " << step << "] loss=" << loss
        << " | totalSupTokens=" << totalSupervisedTokens
        << " | tokens=" << probeTokens
        << " | batches=" << cfg_.numTrainProbeBatches
        << " | probeTime(s)=" << elapsed
        << " | probeTok/s=" << tokPerSec << '\n';

    if (!cfg_.trainProbeSampleIndices.empty()) {
        const auto sampleProbeStart = std::chrono::steady_clock::now();
        float sampleLossSumTotal = 0.0f;
        std::int64_t sampleProbeTokens = 0;
        int validSamples = 0;
        struct SampleProbeRow {
            std::size_t index = 0;
            std::int64_t tokens = 0;
            float loss = 0.0f;
        };
        std::vector<SampleProbeRow> rows;
        rows.reserve(cfg_.trainProbeSampleIndices.size());
        for (const std::size_t sampleIndex : cfg_.trainProbeSampleIndices) {
            if (sampleIndex >= dataset_.size()) continue;
            std::int64_t sampleTokens = 0;
            const float rowLossSum = sampleLossSum(
                dataset_.get(sampleIndex),
                sampleTokens,
                nullptr);
            if (sampleTokens <= 0) continue;
            sampleLossSumTotal += rowLossSum;
            sampleProbeTokens += sampleTokens;
            ++validSamples;
            rows.push_back({
                sampleIndex,
                sampleTokens,
                rowLossSum / static_cast<float>(sampleTokens)
            });
        }
        const double sampleProbeElapsed = elapsedSeconds(sampleProbeStart);
        const float sampleProbeLoss = sampleProbeTokens > 0
            ? sampleLossSumTotal / static_cast<float>(sampleProbeTokens)
            : 0.0f;
        std::cout << "[TrainProbeSamples step " << step << "] loss=" << sampleProbeLoss
            << " | totalSupTokens=" << totalSupervisedTokens
            << " | tokens=" << sampleProbeTokens
            << " | samples=" << validSamples
            << " | probeTime(s)=" << sampleProbeElapsed;
        for (const auto& row : rows) {
            std::cout << " | idx=" << row.index
                << ",supTok=" << row.tokens
                << ",loss=" << row.loss;
        }
        std::cout << '\n';
    }

    const auto evaluateGate = [&](int gateIndex,
                                  std::int64_t gateSupTokens,
                                  float gateMaxLoss,
                                  bool& gateDone) {
        if (gateDone) return;
        if (gateSupTokens <= 0 || gateMaxLoss <= 0.0f) return;
        if (totalSupervisedTokens < gateSupTokens) return;
        gateDone = true;
        const bool passed = loss <= gateMaxLoss;
        std::cout << "[TrainProbeGate " << gateIndex << "] status="
            << (passed ? "pass" : "fail")
            << " | totalSupTokens=" << totalSupervisedTokens
            << " | targetSupTokens=" << gateSupTokens
            << " | loss=" << loss
            << " | maxLoss=" << gateMaxLoss
            << '\n';
        if (!passed) {
            stopRequested_ = true;
            stopCheckpointRequested_ = false;
            stopReason_ = "train probe gate failed";
        }
    };

    evaluateGate(1, cfg_.trainProbeGate1SupTokens, cfg_.trainProbeGate1MaxLoss, trainProbeGate1Done_);
    evaluateGate(2, cfg_.trainProbeGate2SupTokens, cfg_.trainProbeGate2MaxLoss, trainProbeGate2Done_);
}

void CpuQLoRATrainer::maybeCheckpoint(int step,
    const CheckpointState& state,
    bool force) {
    if (!force && cfg_.checkpointInterval <= 0) return;
    if (!force && step % cfg_.checkpointInterval != 0 && step != cfg_.maxSteps) return;

    if (usingAdamW()) {
        checkpointer_.save(backend_, *adamOptimizer_, state);
    }
    else if (usingMuon()) {
        return;
    }
    else {
        checkpointer_.save(backend_, *lionOptimizer_, state);
    }
}

void CpuQLoRATrainer::run() {
    const auto startTime = std::chrono::steady_clock::now();
    const std::size_t vocabSize = backend_.shape().vocabSize;

    CheckpointState state{};
    state.step = 0;
    state.epoch = 0;
    state.samplerOffset = 0;

    if (cfg_.resumeFromCheckpoint) {
        std::optional<CheckpointState> loaded;
        if (usingAdamW()) {
            loaded = checkpointer_.loadLatest(backend_, *adamOptimizer_);
        }
        else if (!usingMuon()) {
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
    std::int64_t totalSupervisedTokens = 0;
    int accumCount = 0;

    auto params = backend_.collectTrainableParams();
    if (usingAdamW()) {
        adamOptimizer_->init(params);
    }
    else if (usingMuon()) {
        muonOptimizer_->init(params);
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
    const int configuredDecayStartStep = (cfg_.decayStartStep > 0)
        ? std::min(cfg_.maxSteps, std::max(warmupSteps, cfg_.decayStartStep))
        : warmupSteps;
    const int decaySteps = (cfg_.decaySteps > 0)
        ? cfg_.decaySteps
        : -1;
    const bool usePlateauLrDecay =
        cfg_.useCosineDecay &&
        cfg_.usePlateauLrDecay;
    const int targetSupervisedTokensPerStep =
        std::max(0, cfg_.targetSupervisedTokensPerStep);
    int referenceSupervisedTokensPerStep =
        std::max(0, cfg_.referenceSupervisedTokensPerStep);
    if (targetSupervisedTokensPerStep > 0 &&
        referenceSupervisedTokensPerStep <= 0 &&
        dataset_.size() > 0)
    {
        std::int64_t totalTokens = 0;
        for (std::size_t i = 0; i < dataset_.size(); ++i) {
            totalTokens += countSupervisedTokensInSample(dataset_.get(i), cfg_, vocabSize);
        }
        if (totalTokens > 0) {
            referenceSupervisedTokensPerStep = std::max(
                1,
                static_cast<int>((totalTokens + static_cast<std::int64_t>(dataset_.size() / 2)) /
                    static_cast<std::int64_t>(dataset_.size())));
        }
    }
    const int plateauMinSteps = (cfg_.plateauMinSteps > 0)
        ? std::min(cfg_.maxSteps, std::max(warmupSteps, cfg_.plateauMinSteps))
        : ((cfg_.decayStartStep > 0)
            ? configuredDecayStartStep
            : std::min(cfg_.maxSteps, std::max(warmupSteps, 50)));
    const float minLr = cfg_.useCosineDecay
        ? (cfg_.learningRate * std::max(0.0f, cfg_.minLrFactor))
        : cfg_.learningRate;
    int activeDecayStartStep =
        usePlateauLrDecay
        ? (cfg_.maxSteps + 1)
        : configuredDecayStartStep;
    PlateauLrDecayDetector plateauDetector{
        usePlateauLrDecay,
        plateauMinSteps,
        cfg_.plateauPatienceSteps,
        cfg_.plateauMinRelImprovement
    };
    LossBounceStopDetector bounceStopDetector{ cfg_ };
    std::cout << "LR schedule: warmup=" << warmupSteps;
    if (usePlateauLrDecay) {
        std::cout
            << " | decay_start=plateau"
            << " | plateau_min_step=" << plateauDetector.minSteps()
            << " | plateau_patience=" << plateauDetector.patienceSteps()
            << " | plateau_min_rel_improvement=" << plateauDetector.minRelImprovement()
            << " | decay_steps=" << ((decaySteps > 0) ? decaySteps : (cfg_.maxSteps - plateauDetector.minSteps()))
            << " | mode=plateau_cosine";
    }
    else {
        std::cout
            << " | decay_start=" << configuredDecayStartStep
            << " | decay_steps=" << ((decaySteps > 0) ? decaySteps : (cfg_.maxSteps - configuredDecayStartStep))
            << " | mode=" << (cfg_.useCosineDecay ? "cosine" : "constant");
    }
    std::cout
        << " | peak_lr=" << cfg_.learningRate
        << " | min_lr=" << minLr << '\n';
    if (cfg_.lmHeadOnlyBootstrapSupTokens > 0) {
        std::cout
            << "Train surface schedule: stage1=lm_head_only_until_sup_tokens="
            << cfg_.lmHeadOnlyBootstrapSupTokens
            << " (lr_scale=" << cfg_.lmHeadBootstrapLrScale << ")";
        if (cfg_.postBootstrapAttentionOnly) {
            const std::int64_t attentionOnlyUntil =
                effectiveAttentionOnlyBootstrapSupTokens(cfg_);
            if (attentionOnlyUntil > 0) {
                std::cout
                    << " | stage2=attention_lora_only_until_sup_tokens="
                    << attentionOnlyUntil
                    << " (lr_scale=" << cfg_.attentionOnlyBootstrapLrScale << ")"
                    << " | stage3=full_surface";
            }
            else {
                std::cout
                    << " | stage2=attention_lora_only"
                    << " (lr_scale=" << cfg_.attentionOnlyBootstrapLrScale << ")";
            }
        }
        else {
            std::cout << " | stage2=full_surface";
        }
        std::cout
            << '\n';
    }
    if (cfg_.attentionABootstrapSupTokens > 0 &&
        cfg_.attentionAInitialLrScale < 1.0f) {
        std::cout
            << "Attention A ramp: sup_tokens=" << cfg_.attentionABootstrapSupTokens
            << " | initial_lr_scale=" << cfg_.attentionAInitialLrScale
            << " | final_lr_scale=1"
            << '\n';
    }
    if (targetSupervisedTokensPerStep > 0) {
        std::cout
            << "Update sizing: target_sup_tokens_per_step=" << targetSupervisedTokensPerStep
            << " | ref_sup_tokens_per_step=" << referenceSupervisedTokensPerStep
            << " | grad_scale=ref_tokens/accum_tokens"
            << '\n';
    }

    while (globalStep < cfg_.maxSteps) {
        std::vector<std::size_t> batchIndices;
        std::vector<float> batchLossWeights;
        std::vector<std::uint8_t> batchIsReplay;
        const auto batch = sampleBatch(
            cfg_.batchSize,
            globalStep + 1,
            &batchIndices,
            &batchLossWeights,
            &batchIsReplay);
        if (batch.empty()) break;

        const bool shortcutRepeatedOverfitAccum =
            targetSupervisedTokensPerStep <= 0 &&
            canShortcutRepeatedOverfitAccum(cfg_);

        static std::vector<StepSampleDebugRecord> accumSampleDebugRecords;
        static TrainStepDebugMetrics accumTrainDebugMetrics;
        static std::vector<HardTokenDebugRecord> accumHardTokenDebugRecords;
        if (accumCount == 0) {
            accumSampleDebugRecords.clear();
            accumTrainDebugMetrics = {};
            accumHardTokenDebugRecords.clear();
        }

        const int traceStep = globalStep + 1;
        traceTrainLoop(traceStep, "before trainStep");
        trainStep(
            batch,
            batchIndices,
            batchLossWeights,
            batchIsReplay,
            totalSupervisedTokens,
            accumLoss,
            accumTokens,
            &accumSampleDebugRecords,
            trainMetricsDebugEnabled() ? &accumTrainDebugMetrics : nullptr,
            hardTokenDebugEnabled() ? &accumHardTokenDebugRecords : nullptr);
        traceTrainLoop(traceStep, "after trainStep");
        accumCount += shortcutRepeatedOverfitAccum
            ? cfg_.gradAccumSteps
            : 1;

        const bool reachedGradAccum = accumCount >= cfg_.gradAccumSteps;
        const bool reachedTokenTarget =
            targetSupervisedTokensPerStep > 0 &&
            accumTokens >= targetSupervisedTokensPerStep;
        const bool shouldStep =
            (targetSupervisedTokensPerStep > 0)
            ? reachedTokenTarget
            : reachedGradAccum;
        if (!shouldStep) continue;

        if (targetSupervisedTokensPerStep > 0 &&
            referenceSupervisedTokensPerStep > 0 &&
            accumTokens > 0)
        {
            traceTrainLoop(traceStep, "before scaleGradients tokenTarget");
            const float tokenScale =
                static_cast<float>(referenceSupervisedTokensPerStep) /
                static_cast<float>(accumTokens);
            scaleGradients(params, tokenScale);
            traceTrainLoop(traceStep, "after scaleGradients tokenTarget");
        }
        else if (accumCount > 1 && !shortcutRepeatedOverfitAccum) {
            traceTrainLoop(traceStep, "before scaleGradients accum");
            const float invAccum = 1.0f / static_cast<float>(accumCount);
            scaleGradients(params, invAccum);
            traceTrainLoop(traceStep, "after scaleGradients accum");
        }

        traceTrainLoop(traceStep, "before stepParams copy");
        auto stepParams = params;
        traceTrainLoop(traceStep, "after stepParams copy");
        traceTrainLoop(traceStep, "before schedules");
        applyTrainSurfaceSchedule(stepParams, cfg_, totalSupervisedTokens);
        applyAttentionAWarmRamp(stepParams, cfg_, totalSupervisedTokens);
        applyAttentionBOnlyWarmStart(stepParams, cfg_, totalSupervisedTokens);
        applyDelayedKUpdates(stepParams, cfg_, totalSupervisedTokens);
        traceTrainLoop(traceStep, "after schedules");

        // Compute LR for this step once so debug and optimizer are identical.
        const LRScheduler scheduler{
            cfg_.learningRate,
            minLr,
            cfg_.maxSteps,
            warmupSteps,
            activeDecayStartStep,
            decaySteps
        };
        const float stepLr = scheduler.lrAt(globalStep);

        traceTrainLoop(traceStep, "before param norms");
        maybePrintParamNorms(stepParams, traceStep, cfg_.logInterval, cfg_.maxGradNorm, stepLr);
        traceTrainLoop(traceStep, "after param norms");

        if (cfg_.maxGradNorm > 0.0f) {
            if (usingAdamW()) {
                traceTrainLoop(traceStep, "before adam clip");
                adamOptimizer_->clipGradNorm(stepParams, cfg_.maxGradNorm);
                traceTrainLoop(traceStep, "after adam clip");
            }
            else if (usingMuon()) {
                traceTrainLoop(traceStep, "before muon clip");
                muonOptimizer_->clipGradNorm(stepParams, cfg_.maxGradNorm);
                traceTrainLoop(traceStep, "after muon clip");
            }
            else {
                traceTrainLoop(traceStep, "before lion clip");
                lionOptimizer_->clipGradNorm(stepParams, cfg_.maxGradNorm);
                traceTrainLoop(traceStep, "after lion clip");
            }
        }

        if (loraNormDebugEnabled() && globalStep % cfg_.logInterval == 0) {
            double loraA_norm = 0.0, loraB_norm = 0.0, loraA_grad_norm = 0.0;
            for (const auto& p : stepParams) {
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
            traceTrainLoop(traceStep, "before adam hyper");
            adamOptimizer_->setHyperParams(
                stepLr,
                cfg_.adamBeta1,
                cfg_.adamBeta2,
                cfg_.adamEps,
                cfg_.weightDecay
            );
            adamOptimizer_->setParamBeta1Overrides(
                cfg_.adamAttnABeta1,
                cfg_.adamAttnBBeta1,
                cfg_.adamLmHeadMatrixBeta1
            );
            traceTrainLoop(traceStep, "before adam step");
            adamOptimizer_->step(stepParams, cfg_.zeroGradAfterStep);
            traceTrainLoop(traceStep, "after adam step");
        }
        else if (usingMuon()) {
            traceTrainLoop(traceStep, "before muon hyper");
            muonOptimizer_->setHyperParams(
                stepLr,
                cfg_.lionBeta1,
                cfg_.adamBeta1,
                cfg_.adamBeta2,
                cfg_.adamEps,
                cfg_.weightDecay
            );
            traceTrainLoop(traceStep, "before muon step");
            muonOptimizer_->step(stepParams, cfg_.zeroGradAfterStep);
            traceTrainLoop(traceStep, "after muon step");
        }
        else {
            traceTrainLoop(traceStep, "before lion hyper");
            lionOptimizer_->setHyperParams(
                stepLr,
                cfg_.lionBeta1,
                cfg_.lionBeta2,
                cfg_.weightDecay
            );
            traceTrainLoop(traceStep, "before lion step");
            lionOptimizer_->step(stepParams, cfg_.zeroGradAfterStep);
            traceTrainLoop(traceStep, "after lion step");
        }
        traceTrainLoop(traceStep, "before markTrainableParamsUpdated");
        backend_.markTrainableParamsUpdated();
        traceTrainLoop(traceStep, "after markTrainableParamsUpdated");

        ++globalStep;

        const float avgLoss = (accumTokens > 0)
            ? (accumLoss / static_cast<float>(accumTokens))
            : 0.0f;
        totalSupervisedTokens += accumTokens;
        traceTrainLoop(traceStep, "before debug prints");
        maybePrintStepSampleDebug(globalStep, cfg_.logInterval, accumSampleDebugRecords);
        maybePrintTrainMetrics(globalStep, cfg_.logInterval, accumTrainDebugMetrics);
        maybePrintHardTokenDebug(globalStep, cfg_.logInterval, accumHardTokenDebugRecords);
        observeOnlineHardReplay(globalStep, accumSampleDebugRecords);
        traceTrainLoop(traceStep, "after debug prints");
        if (cfg_.attentionBOnlyBootstrapSupTokens > 0 &&
            !attentionBOnlyBootstrapCompletedLogged_ &&
            totalSupervisedTokens >= cfg_.attentionBOnlyBootstrapSupTokens)
        {
            std::cout
                << "[Bootstrap] attention A matrices unfrozen at totalSupTokens="
                << totalSupervisedTokens
                << " | attentionBOnlyBootstrapLrScale="
                << cfg_.attentionBOnlyBootstrapLrScale
                << '\n';
            attentionBOnlyBootstrapCompletedLogged_ = true;
        }

        if (cfg_.attentionABootstrapSupTokens > 0 &&
            cfg_.attentionAInitialLrScale < 1.0f &&
            !attentionABootstrapCompletedLogged_ &&
            totalSupervisedTokens >= cfg_.attentionABootstrapSupTokens)
        {
            std::cout
                << "[Bootstrap] attention A lr ramp completed at totalSupTokens="
                << totalSupervisedTokens
                << " | attentionAInitialLrScale="
                << cfg_.attentionAInitialLrScale
                << '\n';
            attentionABootstrapCompletedLogged_ = true;
        }
        if (cfg_.lmHeadOnlyBootstrapSupTokens > 0 &&
            !lmHeadBootstrapCompletedLogged_ &&
            totalSupervisedTokens >= cfg_.lmHeadOnlyBootstrapSupTokens)
        {
            const std::int64_t attentionOnlyUntil =
                effectiveAttentionOnlyBootstrapSupTokens(cfg_);
            std::cout
                << "[Bootstrap] "
                << (cfg_.postBootstrapAttentionOnly
                    ? "attention LoRA trainable surface enabled"
                    : "full trainable surface enabled")
                << " at totalSupTokens="
                << totalSupervisedTokens;
            if (cfg_.postBootstrapAttentionOnly && attentionOnlyUntil > 0) {
                std::cout
                    << " | full surface at totalSupTokens="
                    << attentionOnlyUntil;
            }
            std::cout
                << '\n';
            lmHeadBootstrapCompletedLogged_ = true;
        }
        if (cfg_.postBootstrapAttentionOnly &&
            !attentionOnlyBootstrapCompletedLogged_) {
            const std::int64_t attentionOnlyUntil =
                effectiveAttentionOnlyBootstrapSupTokens(cfg_);
            if (attentionOnlyUntil > 0 &&
                totalSupervisedTokens >= attentionOnlyUntil) {
                std::cout
                    << "[Bootstrap] full trainable surface enabled at totalSupTokens="
                    << totalSupervisedTokens
                    << '\n';
                attentionOnlyBootstrapCompletedLogged_ = true;
            }
        }

        traceTrainLoop(traceStep, "before reporter");
        reporter_.onStep(
            globalStep,
            avgLoss,
            stepLr,
            accumTokens,
            totalSupervisedTokens,
            cfg_.maxSteps,
            cfg_.logInterval,
            startTime,
            epoch_,
            epochStartTime_
        );
        traceTrainLoop(traceStep, "after reporter");

        const auto trainStopMetricValue = [&]() {
            if (cfg_.trainLossStopMetric == "ma8" && reporter_.hasLoss()) {
                return reporter_.currentMa8();
            }
            if (cfg_.trainLossStopMetric == "ma32" && reporter_.hasLoss()) {
                return reporter_.currentMa32();
            }
            return avgLoss;
        };
        const float trainStopLoss = trainStopMetricValue();
        if (cfg_.trainLossStopThreshold > 0.0f &&
            globalStep >= std::max(1, cfg_.trainLossStopMinStep) &&
            std::isfinite(trainStopLoss) &&
            trainStopLoss <= cfg_.trainLossStopThreshold)
        {
            stopRequested_ = true;
            stopCheckpointRequested_ = cfg_.checkpointOnTrainLossStop;
            stopReason_ = "train loss target hit";
            std::cout << "[STOP request] train loss target hit: metric="
                << cfg_.trainLossStopMetric
                << " loss=" << trainStopLoss
                << " rawLoss=" << avgLoss
                << " ma8=" << (reporter_.hasLoss() ? reporter_.currentMa8() : avgLoss)
                << " ma32=" << (reporter_.hasLoss() ? reporter_.currentMa32() : avgLoss)
                << " <= threshold=" << cfg_.trainLossStopThreshold
                << " at step " << globalStep
                << " totalSupTokens=" << totalSupervisedTokens
                << '\n';
        }

        if (usePlateauLrDecay) {
            if (const std::optional<int> triggerStep = plateauDetector.observe(globalStep, avgLoss)) {
                activeDecayStartStep = *triggerStep;
                std::cout
                    << "[LR plateau trigger] step=" << *triggerStep
                    << " | ma32=" << plateauDetector.currentMa32()
                    << " | bestMa32=" << plateauDetector.bestMa32()
                    << " | bestStep=" << plateauDetector.bestStep()
                    << " | decay_steps=" << ((decaySteps > 0) ? decaySteps : (cfg_.maxSteps - activeDecayStartStep))
                    << " | min_lr=" << minLr
                    << '\n';
            }
        }

        if (cfg_.overfitSampleIndex < 0 &&
            !epochOrder_.empty() &&
            samplerOffset_ >= epochOrder_.size() &&
            lastCompletedEpochLogged_ != epoch_)
        {
            const double epochTimeSec = elapsedSeconds(epochStartTime_);
            std::cout << "[Epoch " << epoch_ << " complete] "
                << "epochTime(s)=" << epochTimeSec << '\n';
            lastCompletedEpochLogged_ = epoch_;
        }

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

        traceTrainLoop(traceStep, "before maybeEval");
        maybeEval(globalStep, stopRequested_ && cfg_.evalOnTrainLossStop);
        traceTrainLoop(traceStep, "after maybeEval");
        traceTrainLoop(traceStep, "before maybeTrainProbe");
        maybeTrainProbe(globalStep, totalSupervisedTokens);
        traceTrainLoop(traceStep, "after maybeTrainProbe");
        if (stopRequested_) {
            const std::string reason = stopReason_.empty()
                ? std::string("stop requested")
                : stopReason_;
            std::cout << "[STOP] " << reason << " at step "
                << globalStep
                << " totalSupTokens=" << totalSupervisedTokens
                << '\n';
            if (stopCheckpointRequested_) {
                maybeCheckpoint(globalStep, state, true);
            }
            break;
        }
        if (bounceStopDetector.observe(globalStep, avgLoss, totalSupervisedTokens)) {
            std::cout << "[STOP] loss bounce detected at step "
                << globalStep
                << " totalSupTokens=" << totalSupervisedTokens
                << " loss=" << avgLoss
                << " bestLoss=" << bounceStopDetector.bestLoss()
                << " bestStep=" << bounceStopDetector.bestStep()
                << " recentRange=" << bounceStopDetector.recentRange()
                << " directionChanges=" << bounceStopDetector.recentDirectionChanges()
                << '\n';
            break;
        }
        traceTrainLoop(traceStep, "before checkpoint");
        maybeCheckpoint(globalStep, state);
        traceTrainLoop(traceStep, "after checkpoint");

        if (cfg_.maxTotalSupervisedTokens > 0 &&
            totalSupervisedTokens >= cfg_.maxTotalSupervisedTokens)
        {
            if (hasPendingTrainProbeGate(totalSupervisedTokens)) {
                maybeTrainProbe(globalStep, totalSupervisedTokens, true);
                if (stopRequested_) {
                    const std::string reason = stopReason_.empty()
                        ? std::string("stop requested")
                        : stopReason_;
                    std::cout << "[STOP] " << reason << " at step "
                        << globalStep
                        << " totalSupTokens=" << totalSupervisedTokens
                        << '\n';
                    if (stopCheckpointRequested_) {
                        maybeCheckpoint(globalStep, state, true);
                    }
                    break;
                }
            }
            std::cout << "[STOP] supervised token budget hit: totalSupTokens="
                << totalSupervisedTokens
                << " >= maxTotalSupTokens=" << cfg_.maxTotalSupervisedTokens
                << " at step " << globalStep << '\n';
            break;
        }
    }

    state.step = globalStep;
    state.epoch = epoch_;
    state.samplerOffset = samplerOffset_;

    if (cfg_.saveFinalArtifacts) {
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

        if (usingAdamW()) {
            checkpointer_.save(backend_, *adamOptimizer_, state);
        }
        else if (usingMuon()) {
            std::cout << "[Checkpoint] Muon optimizer checkpoint save skipped.\n";
        }
        else {
            checkpointer_.save(backend_, *lionOptimizer_, state);
        }

        std::cout << "Training complete. LoRA adapter saved to "
            << cfg_.loraOutputPath << '\n';
    }
    else {
        std::cout << "Training complete. Final LoRA/checkpoint artifact writes skipped.\n";
    }
}

int runCpuQLoRATrainer(const TrainerConfig& cfg,
    const std::string& datasetPath) {
    try {
        Dataset dataset{ datasetPath, cfg };
        std::unique_ptr<Dataset> evalDataset;
        if (!cfg.evalDatasetPath.empty()) {
            evalDataset = std::make_unique<Dataset>(cfg.evalDatasetPath, cfg);
        }
        CpuKernelBackend backend{ cfg };

        if (cfg.useMuon) {
            MuonOptimizer optimizer{};
            optimizer.setHyperParams(
                cfg.learningRate,
                cfg.lionBeta1,
                cfg.adamBeta1,
                cfg.adamBeta2,
                cfg.adamEps,
                cfg.weightDecay
            );
            optimizer.setZeroGradAfterStep(cfg.zeroGradAfterStep);

            CpuQLoRATrainer trainer{ cfg, dataset, evalDataset.get(), backend, optimizer };
            trainer.run();
        }
        else if (cfg.useAdamW) {
            AdamWOptimizer optimizer{};
            optimizer.setHyperParams(
                cfg.learningRate,
                cfg.adamBeta1,
                cfg.adamBeta2,
                cfg.adamEps,
                cfg.weightDecay
            );
            optimizer.setParamBeta2Overrides(
                cfg.adamAttnABeta2,
                cfg.adamAttnBBeta2
            );
            optimizer.setParamBeta1Overrides(
                cfg.adamAttnABeta1,
                cfg.adamAttnBBeta1,
                cfg.adamLmHeadMatrixBeta1
            );
            optimizer.setZeroGradAfterStep(cfg.zeroGradAfterStep);
            optimizer.setClipGradNormByLrScale(cfg.clipGradNormByLrScale);
            optimizer.setClipGradNormByParamGroup(cfg.clipGradNormByParamGroup);
            optimizer.setLoraPairRmsBalance(
                cfg.loraPairRmsBalance,
                cfg.loraPairRmsBalanceTargetRatio,
                cfg.loraPairRmsBalanceMaxScale,
                cfg.loraPairRmsBalanceMinRms
            );
            optimizer.setUpdateRmsMaxRatios(
                cfg.adamAttnAUpdateRmsMaxRatio,
                cfg.adamAttnBUpdateRmsMaxRatio,
                cfg.adamLmHeadMatrixUpdateRmsMaxRatio
            );
            optimizer.setUpdateRmsCapMinWeightRms(cfg.adamUpdateRmsCapMinWeightRms);
            optimizer.setUpdateRmsCapStartStep(
                static_cast<std::uint64_t>(std::max(0, cfg.adamUpdateRmsCapStartStep))
            );

            CpuQLoRATrainer trainer{ cfg, dataset, evalDataset.get(), backend, optimizer };
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
            CpuQLoRATrainer trainer{ cfg, dataset, evalDataset.get(), backend, optimizer };
            trainer.run();
        }
        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Fatal error in runCpuQLoRATrainer: " << ex.what() << '\n';
        return 1;
    }
}
