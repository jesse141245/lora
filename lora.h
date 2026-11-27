#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "config.h"
#include "lora_type.h"
#include "paramRef.h"

bool isLmHeadBootstrapParam(std::string_view pname);
bool isAttentionLoraParam(std::string_view pname);
bool isAttentionLoraAParam(std::string_view pname);
bool isAttentionLoraBParam(std::string_view pname);
bool isAttentionKParam(std::string_view pname);

std::int64_t effectiveAttentionOnlyBootstrapSupTokens(const TrainerConfig& cfg);

void applyTrainSurfaceSchedule(
    std::vector<ParamRef>& params,
    const TrainerConfig& cfg,
    std::int64_t totalSupervisedTokensBeforeStep);

void applyAttentionBOnlyWarmStart(
    std::vector<ParamRef>& params,
    const TrainerConfig& cfg,
    std::int64_t totalSupervisedTokensBeforeStep);

void applyAttentionAWarmRamp(
    std::vector<ParamRef>& params,
    const TrainerConfig& cfg,
    std::int64_t totalSupervisedTokensBeforeStep);

void applyDelayedKUpdates(
    std::vector<ParamRef>& params,
    const TrainerConfig& cfg,
    std::int64_t totalSupervisedTokensBeforeStep);
