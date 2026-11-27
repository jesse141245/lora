#include "lora.h"

#include <algorithm>

namespace {

enum class TrainSurfaceStage {
    Full,
    AttentionOnly,
    LmHeadOnly,
};

float trainSurfaceStageLrScale(
    TrainSurfaceStage stage,
    const TrainerConfig& cfg)
{
    switch (stage) {
    case TrainSurfaceStage::LmHeadOnly:
        return cfg.lmHeadBootstrapLrScale;
    case TrainSurfaceStage::AttentionOnly:
        return cfg.attentionOnlyBootstrapLrScale;
    case TrainSurfaceStage::Full:
    default:
        return 1.0f;
    }
}

TrainSurfaceStage resolveTrainSurfaceStage(
    const TrainerConfig& cfg,
    std::int64_t totalSupervisedTokensBeforeStep)
{
    if (cfg.lmHeadOnlyBootstrapSupTokens > 0 &&
        totalSupervisedTokensBeforeStep < cfg.lmHeadOnlyBootstrapSupTokens) {
        return TrainSurfaceStage::LmHeadOnly;
    }

    if (!cfg.postBootstrapAttentionOnly) {
        return TrainSurfaceStage::Full;
    }

    const std::int64_t attentionOnlyUntil =
        effectiveAttentionOnlyBootstrapSupTokens(cfg);
    if (attentionOnlyUntil <= 0 ||
        totalSupervisedTokensBeforeStep < attentionOnlyUntil) {
        return TrainSurfaceStage::AttentionOnly;
    }

    return TrainSurfaceStage::Full;
}

} // namespace

bool isLmHeadBootstrapParam(std::string_view pname) {
    return pname == "Alm" || pname == "Blm" || pname == "BiasLm";
}

bool isAttentionLoraParam(std::string_view pname) {
    return pname == "Aq" || pname == "Bq" ||
        pname == "Ak" || pname == "Bk" ||
        pname == "Av" || pname == "Bv" ||
        pname == "Ao" || pname == "Bo";
}

bool isAttentionLoraAParam(std::string_view pname) {
    return pname == "Aq" || pname == "Ak" ||
        pname == "Av" || pname == "Ao";
}

bool isAttentionLoraBParam(std::string_view pname) {
    return pname == "Bq" || pname == "Bk" ||
        pname == "Bv" || pname == "Bo";
}

bool isAttentionKParam(std::string_view pname) {
    return pname == "Ak" || pname == "Bk";
}

std::int64_t effectiveAttentionOnlyBootstrapSupTokens(const TrainerConfig& cfg) {
    if (cfg.attentionOnlyBootstrapSupTokens <= cfg.lmHeadOnlyBootstrapSupTokens) {
        return 0;
    }
    return cfg.attentionOnlyBootstrapSupTokens;
}

void applyTrainSurfaceSchedule(
    std::vector<ParamRef>& params,
    const TrainerConfig& cfg,
    std::int64_t totalSupervisedTokensBeforeStep)
{
    const TrainSurfaceStage stage =
        resolveTrainSurfaceStage(cfg, totalSupervisedTokensBeforeStep);
    const float stageLrScale = trainSurfaceStageLrScale(stage, cfg);

    switch (stage) {
    case TrainSurfaceStage::LmHeadOnly:
        for (auto& p : params) {
            if (!p.name) continue;
            const float baseLrScale = p.lrScale;
            if (!isLmHeadBootstrapParam(p.name)) {
                p.lrScale = 0.0f;
            }
            else {
                p.lrScale = baseLrScale * stageLrScale;
            }
        }
        return;
    case TrainSurfaceStage::AttentionOnly:
        for (auto& p : params) {
            if (!p.name) continue;
            const float baseLrScale = p.lrScale;
            if (!isAttentionLoraParam(p.name)) {
                p.lrScale = 0.0f;
            }
            else {
                p.lrScale = baseLrScale * stageLrScale;
            }
        }
        return;
    case TrainSurfaceStage::Full:
        return;
    }
}

void applyAttentionBOnlyWarmStart(
    std::vector<ParamRef>& params,
    const TrainerConfig& cfg,
    std::int64_t totalSupervisedTokensBeforeStep)
{
    if (cfg.attentionBOnlyBootstrapSupTokens <= 0) return;
    if (totalSupervisedTokensBeforeStep >= cfg.attentionBOnlyBootstrapSupTokens) return;

    for (auto& p : params) {
        if (!p.name) continue;
        const float baseLrScale = p.lrScale;
        if (isAttentionLoraAParam(p.name)) {
            p.lrScale = 0.0f;
        }
        else if (isAttentionLoraBParam(p.name)) {
            p.lrScale = baseLrScale * cfg.attentionBOnlyBootstrapLrScale;
        }
    }
}

void applyAttentionAWarmRamp(
    std::vector<ParamRef>& params,
    const TrainerConfig& cfg,
    std::int64_t totalSupervisedTokensBeforeStep)
{
    if (cfg.attentionABootstrapSupTokens <= 0) return;
    if (cfg.attentionAInitialLrScale >= 1.0f) return;
    if (totalSupervisedTokensBeforeStep >= cfg.attentionABootstrapSupTokens) return;

    const double progress =
        static_cast<double>(std::max<std::int64_t>(0, totalSupervisedTokensBeforeStep)) /
        static_cast<double>(cfg.attentionABootstrapSupTokens);
    const float ramp =
        cfg.attentionAInitialLrScale +
        (1.0f - cfg.attentionAInitialLrScale) * static_cast<float>(std::clamp(progress, 0.0, 1.0));

    for (auto& p : params) {
        if (!p.name) continue;
        if (!isAttentionLoraAParam(p.name)) continue;
        const float baseLrScale = p.lrScale;
        p.lrScale = baseLrScale * ramp;
    }
}

void applyDelayedKUpdates(
    std::vector<ParamRef>& params,
    const TrainerConfig& cfg,
    std::int64_t totalSupervisedTokensBeforeStep)
{
    if (cfg.loraKMinTotalSupTokens <= 0) return;

    float kScheduleScale = 1.0f;
    if (totalSupervisedTokensBeforeStep < cfg.loraKMinTotalSupTokens) {
        kScheduleScale = 0.0f;
    }
    else if (cfg.loraKWarmupSupTokens > 0) {
        const std::int64_t warmupElapsed =
            totalSupervisedTokensBeforeStep - cfg.loraKMinTotalSupTokens;
        if (warmupElapsed < cfg.loraKWarmupSupTokens) {
            kScheduleScale = static_cast<float>(
                static_cast<double>(warmupElapsed) /
                static_cast<double>(cfg.loraKWarmupSupTokens));
        }
    }

    if (kScheduleScale >= 1.0f) return;

    for (auto& p : params) {
        if (!p.name) continue;
        if (!isAttentionKParam(p.name)) continue;
        p.lrScale *= std::max(0.0f, kScheduleScale);
    }
}
