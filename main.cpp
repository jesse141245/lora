#include <filesystem>
#include <iostream>
#include <string>
#include <exception>
#include <algorithm>
#include <thread>
#include <optional>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "config.h"
#include "env_utils.h"
#include "qlora_cpu.h"

namespace fs = std::filesystem;

namespace {

std::size_t normalizeBitNetQk(int value, std::size_t fallback) {
    if (value <= 0) return fallback;
    const int clamped = std::max(32, value);
    return static_cast<std::size_t>((clamped + 31) / 32 * 32);
}

std::optional<int> getEnvInt(const char* name) {
    const auto value = envutil::tryGetEnvString(name);
    if (!value) return std::nullopt;
    try {
        return std::stoi(*value);
    }
    catch (...) {
        return std::nullopt;
    }
}

std::optional<std::int64_t> getEnvInt64(const char* name) {
    const auto value = envutil::tryGetEnvString(name);
    if (!value) return std::nullopt;
    try {
        return std::stoll(*value);
    }
    catch (...) {
        return std::nullopt;
    }
}

std::optional<float> getEnvFloat(const char* name) {
    const auto value = envutil::tryGetEnvString(name);
    if (!value) return std::nullopt;
    try {
        return std::stof(*value);
    }
    catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> getEnvBool(const char* name) {
    return envutil::envBoolOverride(name);
}

std::vector<std::size_t> parseSizeList(const std::string& value) {
    std::vector<std::size_t> out;
    std::size_t pos = 0;
    while (pos < value.size()) {
        while (pos < value.size() &&
               (value[pos] == ',' || value[pos] == ';' || value[pos] == ' ' || value[pos] == '\t')) {
            ++pos;
        }
        if (pos >= value.size()) break;
        std::size_t end = pos;
        while (end < value.size() &&
               value[end] != ',' && value[end] != ';' && value[end] != ' ' && value[end] != '\t') {
            ++end;
        }
        try {
            const int parsed = std::stoi(value.substr(pos, end - pos));
            if (parsed >= 0) {
                out.push_back(static_cast<std::size_t>(parsed));
            }
        }
        catch (...) {
        }
        pos = end;
    }
    return out;
}

} // namespace

int main() {
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    std::cerr << "TRAINER MAIN IS RUNNING\n";
    try {
#if defined(_OPENMP)
        const int hwThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        const int ompThreads = std::max(1, getEnvInt("QLORA_TEST_OMP_THREADS").value_or(hwThreads));
        omp_set_dynamic(0);
        omp_set_num_threads(ompThreads);
#endif

        std::cout << "========================================================\n";
        std::cout << "   BITNET QLORA CPU TRAINING PIPELINE INTEGRATION\n";
        std::cout << "========================================================\n\n";
#if defined(_OPENMP)
        std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#endif

        const std::string model_path = envutil::tryGetEnvString("QLORA_TEST_MODEL_PATH").value_or("./test_model");
        const std::string dataset_path = envutil::tryGetEnvString("QLORA_TEST_DATASET_PATH").value_or(
            fs::exists("./tests/overfit_tutor_dataset.jsonl")
                ? "./tests/overfit_tutor_dataset.jsonl"
                : "./test_dataset.jsonl");
        const std::string eval_dataset_path = envutil::tryGetEnvString("QLORA_TEST_EVAL_DATASET_PATH").value_or("");

        std::cout << "[1/2] Configuring Trainer...\n";
        TrainerConfig cfg;
        cfg.modelPath = model_path;
        cfg.loraOutputPath = std::filesystem::absolute("lora_output.bin").string();
        cfg.checkpointDir = std::filesystem::absolute("checkpoints").string();
        if (const auto v = envutil::tryGetEnvString("QLORA_TEST_CHECKPOINT_DIR")) {
            cfg.checkpointDir = std::filesystem::absolute(*v).string();
        }

        cfg.maxSeqLen = 256;     // leave room for longer tutoring-style examples
        cfg.batchSize = 1;       // seq256 tutor runs are memory/bandwidth heavy
        cfg.gradAccumSteps = 4;  // smaller effective batch than the old supTokens~8 sweeps
        cfg.maxSteps = 500;
        if (const auto v = getEnvInt("QLORA_TEST_MAX_SEQ_LEN")) {
            cfg.maxSeqLen = static_cast<std::size_t>(std::max(1, *v));
        }
        // Full-depth training (0 = all model layers).
        cfg.forcedLayers = 0;

        // Higher-capacity adapter for SFT alignment.
        cfg.loraRank = 64;
        cfg.loraAlpha = 64.0f; // keep alpha ~= rank

        // Short warmup then constant LR for small-data overfit behavior.
        cfg.learningRate = 3e-3f;
        cfg.warmupSteps = -1;
        cfg.useCosineDecay = false;
        cfg.useAdamW = true; // set false to use Lion
        cfg.adamBeta1 = 0.9f;
        cfg.adamBeta2 = 0.999f;
        cfg.adamEps = 1e-8f;
        cfg.lionBeta1 = 0.9f;
        cfg.lionBeta2 = 0.999f;
        cfg.weightDecay = 0.0f;
        cfg.maxGradNorm = 1.0f;  // kept but note: has no effect on Lion step size

        cfg.logInterval = 1;     // show progress every step
        cfg.checkpointInterval = 100;
        // Start fresh after fixing label alignment; avoids loading stale optimizer state.
        cfg.resumeFromCheckpoint = false;

        // Explicitly zero gradients after each optimizer step.
        // If left at the TrainerConfig default (which may be false), gradients
        // from step N accumulate into step N+1, corrupting every subsequent update.
        cfg.zeroGradAfterStep = true;

        // Non-zero seed enables epoch shuffling so the small dataset cycles in a
        // different order each epoch rather than always repeating the same sequence.
        cfg.seed = 42;
        // Default to a softer attention-A ramp (can still be overridden via env).
        cfg.attentionBOnlyBootstrapSupTokens = 0;
        cfg.attentionABootstrapSupTokens = 2048;
        cfg.attentionAInitialLrScale = 0.05f;
        // Overfit debug mode toggle:
        // set to 0..N-1 to repeatedly train on one sample, -1 for normal sampling.
        cfg.overfitSampleIndex = -1;

        cfg.useBitNetQuant = true;
        cfg.bitnetQuantizeQKV = true;
        cfg.bitnetFuseQKV = true;
        cfg.bitnetQuantizeWo = true;
        cfg.bitnetQuantizeLmHead = false;
        cfg.tTile = 2;
        cfg.bitnetQK = 128;
        cfg.bitnetFfnQK = 0;
        cfg.bitnetJQ = 32;

        if (const auto v = getEnvInt("QLORA_TEST_MAX_STEPS")) {
            cfg.maxSteps = std::max(1, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_BATCH_SIZE")) {
            cfg.batchSize = std::max(1, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_T_TILE")) {
            cfg.tTile = static_cast<std::size_t>(std::max(1, *v));
        }
        if (const auto v = getEnvInt("QLORA_TEST_H_TILE")) {
            cfg.hTile = static_cast<std::size_t>(std::max(1, *v));
        }
        if (const auto v = getEnvInt("QLORA_TEST_V_TILE")) {
            cfg.vTile = static_cast<std::size_t>(std::max(1, *v));
        }
        if (const auto v = getEnvInt("QLORA_TEST_LORA_RANK")) {
            cfg.loraRank = static_cast<std::size_t>(std::max(1, *v));
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_ALPHA")) {
            cfg.loraAlpha = std::max(0.0f, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_USE_DORA")) {
            cfg.useDoRA = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_USE_PISSA")) {
            cfg.usePiSSA = *v;
        }
        if (const auto v = envutil::tryGetEnvString("QLORA_TEST_PISSA_PATH")) {
            cfg.pissaPath = std::filesystem::absolute(*v).string();
        }
        if (const auto v = getEnvInt("QLORA_TEST_DORA_NORM_UPDATE_INTERVAL")) {
            cfg.doraNormUpdateInterval = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_DORA_LM_HEAD_NORM_UPDATE_INTERVAL")) {
            cfg.doraLmHeadNormUpdateInterval = std::max(0, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_DORA_QK_PREROPE_RECOMPUTE")) {
            cfg.doraRecomputeQKPreRope = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_CACHE_ATTN_PROBS")) {
            cfg.cacheAttentionProbs = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_PACKED_SEGMENTS")) {
            cfg.enablePackedSegments = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_RELEASE_QUANT_DENSE_WEIGHTS")) {
            cfg.releaseQuantizedDenseWeights = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_LORA_Q")) {
            cfg.loraTargetQ = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_LORA_K")) {
            cfg.loraTargetK = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_LORA_V")) {
            cfg.loraTargetV = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_LORA_O")) {
            cfg.loraTargetO = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_ATTN_A_LR_SCALE")) {
            cfg.loraAttnALrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_ATTN_B_LR_SCALE")) {
            cfg.loraAttnBLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_K_LR_SCALE")) {
            cfg.loraKLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_LORA_K_MIN_TOTAL_SUP_TOKENS")) {
            cfg.loraKMinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_LORA_K_WARMUP_SUP_TOKENS")) {
            cfg.loraKWarmupSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_O_LR_SCALE")) {
            cfg.loraOLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_LORA_FFN_DOWN")) {
            cfg.loraTargetFfnDown = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_FFN_DOWN_LR_SCALE")) {
            cfg.loraFfnDownLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_LORA_FFN_GATE_UP")) {
            cfg.loraTargetFfnGateUp = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_FFN_GATE_UP_LR_SCALE")) {
            cfg.loraFfnGateUpLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_LORA_FFN_FIRST_LAYER")) {
            cfg.loraFfnFirstLayer = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_LORA_FFN_LAST_LAYER")) {
            cfg.loraFfnLastLayer = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_LORA_LM_HEAD")) {
            cfg.loraTargetLmHead = *v;
        }
        const bool hasLmHeadALrScale = static_cast<bool>(envutil::tryGetEnvString("QLORA_TEST_LORA_LM_HEAD_A_LR_SCALE"));
        const bool hasLmHeadBLrScale = static_cast<bool>(envutil::tryGetEnvString("QLORA_TEST_LORA_LM_HEAD_B_LR_SCALE"));
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_LM_HEAD_A_LR_SCALE")) {
            cfg.loraLmHeadALrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_LM_HEAD_B_LR_SCALE")) {
            cfg.loraLmHeadBLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_LM_HEAD_LR_SCALE")) {
            cfg.loraLmHeadLrScale = std::max(0.0f, *v);
            if (!hasLmHeadALrScale) {
                cfg.loraLmHeadALrScale = cfg.loraLmHeadLrScale;
            }
            if (!hasLmHeadBLrScale) {
                cfg.loraLmHeadBLrScale = cfg.loraLmHeadLrScale;
            }
        }
        if (const auto v = getEnvBool("QLORA_TEST_LM_HEAD_BIAS")) {
            cfg.trainLmHeadBias = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LM_HEAD_BIAS_LR_SCALE")) {
            cfg.lmHeadBiasLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_FINAL_NORM_AFFINE")) {
            cfg.trainFinalNormAffine = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_FINAL_NORM_AFFINE_LR_SCALE")) {
            cfg.finalNormAffineLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_LAYER_RMSNORM")) {
            cfg.trainLayerRmsNorm = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LAYER_RMSNORM_LR_SCALE")) {
            cfg.layerRmsNormLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_FINAL_RMSNORM")) {
            cfg.trainFinalRmsNorm = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_FINAL_RMSNORM_LR_SCALE")) {
            cfg.finalRmsNormLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_BITNET_QK")) {
            cfg.bitnetQK = normalizeBitNetQk(*v, cfg.bitnetQK);
        }
        if (const auto v = getEnvInt("QLORA_TEST_BITNET_FFN_QK")) {
            cfg.bitnetFfnQK = (*v <= 0) ? 0u : normalizeBitNetQk(*v, cfg.bitnetQK);
        }
        if (const auto v = getEnvInt("QLORA_TEST_GRAD_ACCUM")) {
            cfg.gradAccumSteps = std::max(1, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_MAX_SUPERVISED_TOKENS")) {
            cfg.maxSupervisedTokensPerSample = std::max(0, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_MAX_TOTAL_SUPERVISED_TOKENS")) {
            cfg.maxTotalSupervisedTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_LMHEAD_ONLY_BOOTSTRAP_SUP_TOKENS")) {
            cfg.lmHeadOnlyBootstrapSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_ATTN_ONLY_BOOTSTRAP_SUP_TOKENS")) {
            cfg.attentionOnlyBootstrapSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_POST_BOOTSTRAP_ATTN_ONLY")) {
            cfg.postBootstrapAttentionOnly = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LMHEAD_BOOTSTRAP_LR_SCALE")) {
            cfg.lmHeadBootstrapLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ATTN_ONLY_BOOTSTRAP_LR_SCALE")) {
            cfg.attentionOnlyBootstrapLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_ATTN_B_ONLY_BOOTSTRAP_SUP_TOKENS")) {
            cfg.attentionBOnlyBootstrapSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ATTN_B_ONLY_BOOTSTRAP_LR_SCALE")) {
            cfg.attentionBOnlyBootstrapLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_ATTN_A_BOOTSTRAP_SUP_TOKENS")) {
            cfg.attentionABootstrapSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ATTN_A_INITIAL_LR_SCALE")) {
            cfg.attentionAInitialLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_TARGET_SUPERVISED_TOKENS_PER_STEP")) {
            cfg.targetSupervisedTokensPerStep = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_REFERENCE_SUPERVISED_TOKENS_PER_STEP")) {
            cfg.referenceSupervisedTokensPerStep = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_FIRST_ANSWER_LOSS_WEIGHT_TOKENS")) {
            cfg.firstAnswerLossWeightTokens = std::max(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_FIRST_ANSWER_LOSS_WEIGHT")) {
            cfg.firstAnswerLossWeight = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_TOKEN_LOSS_CAP")) {
            cfg.tokenLossCap = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_TOKEN_LOSS_CAP_SOFTNESS")) {
            cfg.tokenLossCapSoftness = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_TOKEN_LOSS_CAP_MIN_TOTAL_SUP_TOKENS")) {
            cfg.tokenLossCapMinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_TARGET_FREQ_LOSS_MIN_FREQ")) {
            cfg.targetFreqLossMinFreq = std::max(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_TARGET_FREQ_LOSS_MIN_SCALE")) {
            cfg.targetFreqLossMinScale = std::clamp(*v, 0.0f, 1.0f);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_TARGET_FREQ_LOSS_MIN_TOTAL_SUP_TOKENS")) {
            cfg.targetFreqLossMinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LABEL_SMOOTHING")) {
            cfg.labelSmoothing = std::clamp(*v, 0.0f, 0.5f);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_LABEL_SMOOTHING_MIN_TOTAL_SUP_TOKENS")) {
            cfg.labelSmoothingMinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LOGIT_L2_PENALTY")) {
            cfg.logitL2Penalty = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_LOGIT_L2_MIN_TOTAL_SUP_TOKENS")) {
            cfg.logitL2MinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_WRONG_TOP_MARGIN_PENALTY")) {
            cfg.wrongTopMarginPenalty = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_WRONG_TOP_MARGIN_THRESHOLD")) {
            cfg.wrongTopMarginThreshold = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_WRONG_TOP_MARGIN_MIN_TOTAL_SUP_TOKENS")) {
            cfg.wrongTopMarginMinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_HARD_TOKEN_BOOST_THRESHOLD")) {
            cfg.hardTokenBoostThreshold = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_HARD_TOKEN_BOOST_SLOPE")) {
            cfg.hardTokenBoostSlope = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_HARD_TOKEN_BOOST_MAX_SCALE")) {
            cfg.hardTokenBoostMaxScale = std::max(1.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_HARD_TOKEN_BOOST_MIN_TOTAL_SUP_TOKENS")) {
            cfg.hardTokenBoostMinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_RARE_HARD_TOKEN_BOOST_MAX_TRAIN_FREQ")) {
            cfg.rareHardTokenBoostMaxTrainFreq = std::max(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_RARE_HARD_TOKEN_BOOST_THRESHOLD")) {
            cfg.rareHardTokenBoostThreshold = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_RARE_HARD_TOKEN_BOOST_SCALE")) {
            cfg.rareHardTokenBoostScale = std::max(1.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_RARE_HARD_TOKEN_BOOST_MIN_TOTAL_SUP_TOKENS")) {
            cfg.rareHardTokenBoostMinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_RARE_WRONG_TOP_MARGIN_MAX_TRAIN_FREQ")) {
            cfg.rareWrongTopMarginMaxTrainFreq = std::max(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_RARE_WRONG_TOP_MARGIN_PENALTY")) {
            cfg.rareWrongTopMarginPenalty = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_RARE_WRONG_TOP_MARGIN_THRESHOLD")) {
            cfg.rareWrongTopMarginThreshold = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_RARE_WRONG_TOP_MARGIN_MIN_TOTAL_SUP_TOKENS")) {
            cfg.rareWrongTopMarginMinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_RARE_TARGET_BIAS_MAX_TRAIN_FREQ")) {
            cfg.rareTargetBiasMaxTrainFreq = std::max(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_RARE_TARGET_BIAS_MIN_LOSS")) {
            cfg.rareTargetBiasMinLoss = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_RARE_TARGET_BIAS_LR_SCALE")) {
            cfg.rareTargetBiasLrScale = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_RARE_TARGET_BIAS_MAX_ABS")) {
            cfg.rareTargetBiasMaxAbs = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_RARE_TARGET_BIAS_MIN_TOTAL_SUP_TOKENS")) {
            cfg.rareTargetBiasMinTotalSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_NORMALIZE_DLOGITS_BY_SAMPLE_TOKENS")) {
            cfg.normalizeDLogitsBySampleTokens = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_LOG_INTERVAL")) {
            cfg.logInterval = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_CHECKPOINT_INTERVAL")) {
            cfg.checkpointInterval = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_EVAL_INTERVAL")) {
            cfg.evalInterval = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_NUM_EVAL_BATCHES")) {
            cfg.numEvalBatches = std::max(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_EVAL_LOSS_STOP_THRESHOLD")) {
            cfg.evalLossStopThreshold = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_EVAL_LOSS_STOP_MIN_STEP")) {
            cfg.evalLossStopMinStep = std::max(0, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_CHECKPOINT_ON_EVAL_LOSS_STOP")) {
            cfg.checkpointOnEvalLossStop = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_TRAIN_LOSS_STOP_THRESHOLD")) {
            cfg.trainLossStopThreshold = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_TRAIN_LOSS_STOP_MIN_STEP")) {
            cfg.trainLossStopMinStep = std::max(0, *v);
        }
        if (const auto v = envutil::tryGetEnvString("QLORA_TEST_TRAIN_LOSS_STOP_METRIC")) {
            const std::string metric = envutil::lowerCopy(*v);
            if (metric == "loss" || metric == "ma8" || metric == "ma32") {
                cfg.trainLossStopMetric = metric;
            }
        }
        if (const auto v = getEnvBool("QLORA_TEST_EVAL_ON_TRAIN_LOSS_STOP")) {
            cfg.evalOnTrainLossStop = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_CHECKPOINT_ON_TRAIN_LOSS_STOP")) {
            cfg.checkpointOnTrainLossStop = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_TRAIN_PROBE_INTERVAL")) {
            cfg.trainProbeInterval = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_NUM_TRAIN_PROBE_BATCHES")) {
            cfg.numTrainProbeBatches = std::max(0, *v);
        }
        if (const auto v = envutil::tryGetEnvString("QLORA_TEST_TRAIN_PROBE_SAMPLE_INDICES")) {
            cfg.trainProbeSampleIndices = parseSizeList(*v);
        }
        if (const auto v = envutil::tryGetEnvString("QLORA_TEST_REPLAY_SAMPLE_INDICES")) {
            cfg.replaySampleIndices = parseSizeList(*v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_REPLAY_SAMPLES_PER_BATCH")) {
            cfg.replaySamplesPerBatch = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_REPLAY_EVERY_STEPS")) {
            cfg.replayEverySteps = std::max(1, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_REPLAY_SAMPLE_LOSS_WEIGHT")) {
            cfg.replaySampleLossWeight = std::max(0.0f, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_APPEND_REPLAY_SAMPLES")) {
            cfg.appendReplaySamples = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_REPLAY_GRADIENT_ONLY")) {
            cfg.replayGradientOnly = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_REPLAY_INJECT_START_STEP")) {
            cfg.replayInjectStartStep = std::max(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ONLINE_HARD_REPLAY_MIN_LOSS")) {
            cfg.onlineHardReplayMinLoss = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ONLINE_HARD_REPLAY_MIN_MAX_TOKEN_LOSS")) {
            cfg.onlineHardReplayMinMaxTokenLoss = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_ONLINE_HARD_REPLAY_START_STEP")) {
            cfg.onlineHardReplayStartStep = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_ONLINE_HARD_REPLAY_MAX_SAMPLES")) {
            cfg.onlineHardReplayMaxSamples = std::max(0, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_TRAIN_PROBE_GATE1_SUP_TOKENS")) {
            cfg.trainProbeGate1SupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_TRAIN_PROBE_GATE1_MAX_LOSS")) {
            cfg.trainProbeGate1MaxLoss = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt64("QLORA_TEST_TRAIN_PROBE_GATE2_SUP_TOKENS")) {
            cfg.trainProbeGate2SupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_TRAIN_PROBE_GATE2_MAX_LOSS")) {
            cfg.trainProbeGate2MaxLoss = std::max(0.0f, *v);
        }
        cfg.evalDatasetPath = eval_dataset_path;
        if (const auto v = getEnvBool("QLORA_TEST_EVAL_PRED_STATS")) {
            cfg.evalPredictionStats = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_SAVE_FINAL_ARTIFACTS")) {
            cfg.saveFinalArtifacts = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_FORCED_LAYERS")) {
            cfg.forcedLayers = (*v <= 0) ? 0u : static_cast<std::size_t>(*v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_OVERFIT_SAMPLE")) {
            cfg.overfitSampleIndex = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_RESUME_CHECKPOINT")) {
            cfg.resumeFromCheckpoint = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_WARMUP_STEPS")) {
            cfg.warmupSteps = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_USE_COSINE_DECAY")) {
            cfg.useCosineDecay = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_DECAY_START_STEP")) {
            cfg.decayStartStep = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_DECAY_STEPS")) {
            cfg.decaySteps = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_MIN_LR_FACTOR")) {
            cfg.minLrFactor = std::max(0.0f, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_USE_PLATEAU_LR_DECAY")) {
            cfg.usePlateauLrDecay = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_PLATEAU_MIN_STEPS")) {
            cfg.plateauMinSteps = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_PLATEAU_PATIENCE_STEPS")) {
            cfg.plateauPatienceSteps = std::max(1, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_PLATEAU_MIN_REL_IMPROVEMENT")) {
            cfg.plateauMinRelImprovement = std::max(0.0f, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_LOSS_BOUNCE_STOP")) {
            cfg.lossBounceStopEnabled = *v;
        }
        if (const auto v = getEnvInt64("QLORA_TEST_LOSS_BOUNCE_STOP_MIN_SUP_TOKENS")) {
            cfg.lossBounceStopMinSupTokens = std::max<std::int64_t>(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LOSS_BOUNCE_STOP_START_LOSS")) {
            cfg.lossBounceStopStartLoss = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_LOSS_BOUNCE_STOP_WINDOW")) {
            cfg.lossBounceStopWindow = std::max(3, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_LOSS_BOUNCE_STOP_PATIENCE_STEPS")) {
            cfg.lossBounceStopPatienceSteps = std::max(1, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LOSS_BOUNCE_STOP_MIN_DELTA")) {
            cfg.lossBounceStopMinDelta = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LOSS_BOUNCE_STOP_MIN_RANGE")) {
            cfg.lossBounceStopMinRange = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_LOSS_BOUNCE_STOP_MIN_DIRECTION_CHANGES")) {
            cfg.lossBounceStopMinDirectionChanges = std::max(0, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_USE_ADAMW")) {
            cfg.useAdamW = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_USE_MUON")) {
            cfg.useMuon = *v;
            if (*v) cfg.useAdamW = false;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_BETA1")) {
            cfg.adamBeta1 = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_ATTN_A_BETA1")) {
            cfg.adamAttnABeta1 = (*v < 0.0f)
                ? -1.0f
                : std::clamp(*v, 0.0f, 0.999999f);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_ATTN_B_BETA1")) {
            cfg.adamAttnBBeta1 = (*v < 0.0f)
                ? -1.0f
                : std::clamp(*v, 0.0f, 0.999999f);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_LM_HEAD_MATRIX_BETA1")) {
            cfg.adamLmHeadMatrixBeta1 = (*v < 0.0f)
                ? -1.0f
                : std::clamp(*v, 0.0f, 0.999999f);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_BETA2")) {
            cfg.adamBeta2 = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_EPS")) {
            cfg.adamEps = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_ATTN_A_BETA2")) {
            cfg.adamAttnABeta2 = (*v < 0.0f)
                ? -1.0f
                : std::clamp(*v, 0.0f, 0.999999f);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_ATTN_B_BETA2")) {
            cfg.adamAttnBBeta2 = (*v < 0.0f)
                ? -1.0f
                : std::clamp(*v, 0.0f, 0.999999f);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LR")) {
            cfg.learningRate = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_MAX_GRAD_NORM")) {
            cfg.maxGradNorm = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_CLIP_GRAD_NORM_BY_LR_SCALE")) {
            cfg.clipGradNormByLrScale = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_CLIP_GRAD_NORM_BY_PARAM_GROUP")) {
            cfg.clipGradNormByParamGroup = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_LORA_PAIR_RMS_BALANCE")) {
            cfg.loraPairRmsBalance = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_PAIR_RMS_BALANCE_TARGET_RATIO")) {
            cfg.loraPairRmsBalanceTargetRatio = std::max(1e-6f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_PAIR_RMS_BALANCE_MAX_SCALE")) {
            cfg.loraPairRmsBalanceMaxScale = std::max(1.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LORA_PAIR_RMS_BALANCE_MIN_RMS")) {
            cfg.loraPairRmsBalanceMinRms = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_ATTN_A_UPDATE_RMS_MAX_RATIO")) {
            cfg.adamAttnAUpdateRmsMaxRatio = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_ATTN_B_UPDATE_RMS_MAX_RATIO")) {
            cfg.adamAttnBUpdateRmsMaxRatio = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_LM_HEAD_MATRIX_UPDATE_RMS_MAX_RATIO")) {
            cfg.adamLmHeadMatrixUpdateRmsMaxRatio = std::max(0.0f, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_ADAM_UPDATE_RMS_CAP_MIN_WEIGHT_RMS")) {
            cfg.adamUpdateRmsCapMinWeightRms = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_ADAM_UPDATE_RMS_CAP_START_STEP")) {
            cfg.adamUpdateRmsCapStartStep = std::max(0, *v);
        }
        if (const auto v = getEnvFloat("QLORA_TEST_MIN_STEPS_S")) {
            cfg.minStepsPerSec = std::max(0.0f, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_MIN_STEPS_S_GRACE")) {
            cfg.minStepsPerSecGrace = std::max(0, *v);
        }
        if (const auto v = getEnvBool("QLORA_TEST_LM_HEAD_QUANT")) {
            cfg.bitnetQuantizeLmHead = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_LM_HEAD_FLOAT_BWD")) {
            cfg.bitnetLmHeadFloatBackward = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_LM_HEAD_BWD_QUANT")) {
            cfg.bitnetLmHeadQuantBackwardOnly = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_BITNET_QKV")) {
            cfg.bitnetQuantizeQKV = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_BITNET_WO")) {
            cfg.bitnetQuantizeWo = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_BITNET_H32")) {
            cfg.bitnetHadamard32 = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_BITNET_TL2_QKV")) {
            cfg.bitnetTl2Qkv = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_BITNET_TL2_WO")) {
            cfg.bitnetTl2Wo = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_BITNET_TL2_FFN")) {
            cfg.bitnetTl2Ffn = *v;
        }
        if (const auto v = getEnvInt("QLORA_TEST_BITNET_TL2_BK")) {
            cfg.bitnetTl2Bk = static_cast<std::size_t>(std::max(1, *v));
        }
        if (const auto v = getEnvBool("QLORA_TEST_BITNET_FUSE_QKV")) {
            cfg.bitnetFuseQKV = *v;
        }
        if (const auto v = getEnvBool("QLORA_TEST_USE_BITNET_QUANT")) {
            cfg.useBitNetQuant = *v;
        }
        const bool bitnetFfnEnabled = getEnvBool("QLORA_TEST_BITNET_FFN").value_or(true);
        const float bitnetActClipRms = getEnvFloat("QLORA_TEST_BITNET_ACT_CLIP_RMS").value_or(0.0f);

        std::cout << "Test overrides:"
            << " forcedLayers=" << cfg.forcedLayers
            << " lmHeadQuant=" << (cfg.bitnetQuantizeLmHead ? "on" : "off")
            << " lmHeadFloatBwd=" << (cfg.bitnetLmHeadFloatBackward ? "on" : "off")
            << " lmHeadBwdQuant=" << (cfg.bitnetLmHeadQuantBackwardOnly ? "on" : "off")
            << " qkvQuant=" << (cfg.bitnetQuantizeQKV ? "on" : "off")
            << " woQuant=" << (cfg.bitnetQuantizeWo ? "on" : "off")
            << " bitnetFfn=" << (bitnetFfnEnabled ? "on" : "off")
            << " h32=" << (cfg.bitnetHadamard32 ? "on" : "off")
            << " tl2Qkv=" << (cfg.bitnetTl2Qkv ? "on" : "off")
            << " tl2Wo=" << (cfg.bitnetTl2Wo ? "on" : "off")
            << " tl2Ffn=" << (cfg.bitnetTl2Ffn ? "on" : "off")
            << " fuseQKV=" << (cfg.bitnetFuseQKV ? "on" : "off")
            << " maxSteps=" << cfg.maxSteps
            << " batchSize=" << cfg.batchSize
            << " gradAccum=" << cfg.gradAccumSteps
            << " maxSupTokens=" << cfg.maxSupervisedTokensPerSample
            << " maxTotalSupTokens=" << cfg.maxTotalSupervisedTokens
            << " lmHeadOnlyBootstrapSupTok=" << cfg.lmHeadOnlyBootstrapSupTokens
            << " attnOnlyBootstrapSupTok=" << cfg.attentionOnlyBootstrapSupTokens
            << " postBootstrapAttnOnly=" << (cfg.postBootstrapAttentionOnly ? "on" : "off")
            << " lmHeadBootstrapLrScale=" << cfg.lmHeadBootstrapLrScale
            << " attnOnlyBootstrapLrScale=" << cfg.attentionOnlyBootstrapLrScale
            << " attnBOnlyBootstrapSupTok=" << cfg.attentionBOnlyBootstrapSupTokens
            << " attnBOnlyBootstrapLrScale=" << cfg.attentionBOnlyBootstrapLrScale
            << " attnABootstrapSupTok=" << cfg.attentionABootstrapSupTokens
            << " attnAInitialLrScale=" << cfg.attentionAInitialLrScale
            << " targetSupTokStep=" << cfg.targetSupervisedTokensPerStep
            << " refSupTokStep=" << cfg.referenceSupervisedTokensPerStep
            << " firstAnswerWeightTokens=" << cfg.firstAnswerLossWeightTokens
            << " firstAnswerWeight=" << cfg.firstAnswerLossWeight
            << " tokenLossCap=" << cfg.tokenLossCap
            << " tokenLossCapSoftness=" << cfg.tokenLossCapSoftness
            << " tokenLossCapMinTotalSupTokens=" << cfg.tokenLossCapMinTotalSupTokens
            << " targetFreqLossMinFreq=" << cfg.targetFreqLossMinFreq
            << " targetFreqLossMinScale=" << cfg.targetFreqLossMinScale
            << " targetFreqLossMinTotalSupTokens=" << cfg.targetFreqLossMinTotalSupTokens
            << " labelSmoothing=" << cfg.labelSmoothing
            << " labelSmoothingMinTotalSupTokens=" << cfg.labelSmoothingMinTotalSupTokens
            << " logitL2Penalty=" << cfg.logitL2Penalty
            << " logitL2MinTotalSupTokens=" << cfg.logitL2MinTotalSupTokens
            << " wrongTopMarginPenalty=" << cfg.wrongTopMarginPenalty
            << " wrongTopMarginThreshold=" << cfg.wrongTopMarginThreshold
            << " wrongTopMarginMinTotalSupTokens=" << cfg.wrongTopMarginMinTotalSupTokens
            << " hardTokenBoostThreshold=" << cfg.hardTokenBoostThreshold
            << " hardTokenBoostSlope=" << cfg.hardTokenBoostSlope
            << " hardTokenBoostMaxScale=" << cfg.hardTokenBoostMaxScale
            << " hardTokenBoostMinTotalSupTokens=" << cfg.hardTokenBoostMinTotalSupTokens
            << " rareHardTokenBoostMaxTrainFreq=" << cfg.rareHardTokenBoostMaxTrainFreq
            << " rareHardTokenBoostThreshold=" << cfg.rareHardTokenBoostThreshold
            << " rareHardTokenBoostScale=" << cfg.rareHardTokenBoostScale
            << " rareHardTokenBoostMinTotalSupTokens=" << cfg.rareHardTokenBoostMinTotalSupTokens
            << " rareWrongTopMarginMaxTrainFreq=" << cfg.rareWrongTopMarginMaxTrainFreq
            << " rareWrongTopMarginPenalty=" << cfg.rareWrongTopMarginPenalty
            << " rareWrongTopMarginThreshold=" << cfg.rareWrongTopMarginThreshold
            << " rareWrongTopMarginMinTotalSupTokens=" << cfg.rareWrongTopMarginMinTotalSupTokens
            << " rareTargetBiasMaxTrainFreq=" << cfg.rareTargetBiasMaxTrainFreq
            << " rareTargetBiasMinLoss=" << cfg.rareTargetBiasMinLoss
            << " rareTargetBiasLrScale=" << cfg.rareTargetBiasLrScale
            << " rareTargetBiasMaxAbs=" << cfg.rareTargetBiasMaxAbs
            << " rareTargetBiasMinTotalSupTokens=" << cfg.rareTargetBiasMinTotalSupTokens
            << " normalizeDLogitsBySampleTokens="
            << (cfg.normalizeDLogitsBySampleTokens ? "on" : "off")
            << " logInterval=" << cfg.logInterval
            << " checkpointInterval=" << cfg.checkpointInterval
            << " evalInterval=" << cfg.evalInterval
            << " numEvalBatches=" << cfg.numEvalBatches
            << " evalLossStopThreshold=" << cfg.evalLossStopThreshold
            << " evalLossStopMinStep=" << cfg.evalLossStopMinStep
            << " checkpointOnEvalLossStop="
            << (cfg.checkpointOnEvalLossStop ? "on" : "off")
            << " trainLossStopThreshold=" << cfg.trainLossStopThreshold
            << " trainLossStopMinStep=" << cfg.trainLossStopMinStep
            << " trainLossStopMetric=" << cfg.trainLossStopMetric
            << " evalOnTrainLossStop=" << (cfg.evalOnTrainLossStop ? "on" : "off")
            << " checkpointOnTrainLossStop=" << (cfg.checkpointOnTrainLossStop ? "on" : "off")
            << " trainProbeInterval=" << cfg.trainProbeInterval
            << " numTrainProbeBatches=" << cfg.numTrainProbeBatches
            << " trainProbeSampleIndices=" << cfg.trainProbeSampleIndices.size()
            << " replaySampleIndices=" << cfg.replaySampleIndices.size()
            << " replaySamplesPerBatch=" << cfg.replaySamplesPerBatch
            << " replayEverySteps=" << cfg.replayEverySteps
            << " replaySampleLossWeight=" << cfg.replaySampleLossWeight
            << " appendReplaySamples=" << (cfg.appendReplaySamples ? "on" : "off")
            << " replayGradientOnly=" << (cfg.replayGradientOnly ? "on" : "off")
            << " replayInjectStartStep=" << cfg.replayInjectStartStep
            << " onlineHardReplayMinLoss=" << cfg.onlineHardReplayMinLoss
            << " onlineHardReplayMinMaxTokenLoss=" << cfg.onlineHardReplayMinMaxTokenLoss
            << " onlineHardReplayStartStep=" << cfg.onlineHardReplayStartStep
            << " onlineHardReplayMaxSamples=" << cfg.onlineHardReplayMaxSamples
            << " trainProbeGate1SupTokens=" << cfg.trainProbeGate1SupTokens
            << " trainProbeGate1MaxLoss=" << cfg.trainProbeGate1MaxLoss
            << " trainProbeGate2SupTokens=" << cfg.trainProbeGate2SupTokens
            << " trainProbeGate2MaxLoss=" << cfg.trainProbeGate2MaxLoss
            << " evalPredStats=" << (cfg.evalPredictionStats ? "on" : "off")
            << " saveFinalArtifacts=" << (cfg.saveFinalArtifacts ? "on" : "off")
            << " loraRank=" << cfg.loraRank
            << " loraAlpha=" << cfg.loraAlpha
            << " useDoRA=" << (cfg.useDoRA ? "on" : "off")
            << " usePiSSA=" << (cfg.usePiSSA ? "on" : "off")
            << " pissaPath=" << (cfg.pissaPath.empty() ? "<none>" : cfg.pissaPath)
            << " doraNormUpdateInterval=" << cfg.doraNormUpdateInterval
            << " doraLmHeadNormUpdateInterval=" << cfg.doraLmHeadNormUpdateInterval
            << " doraQKPreRope=" << (cfg.doraRecomputeQKPreRope ? "recompute" : "cache")
            << " attnProbMode="
            << ((cfg.cacheAttentionProbs && cfg.maxSeqLen < 1024) ? "cache" : "recompute")
            << " packedSegments=" << (cfg.enablePackedSegments ? "on" : "off")
            << " releaseQuantDense=" << (cfg.releaseQuantizedDenseWeights ? "on" : "off")
            << " loraTargets="
            << (cfg.loraTargetQ ? "q" : "")
            << (cfg.loraTargetK ? "k" : "")
            << (cfg.loraTargetV ? "v" : "")
            << (cfg.loraTargetO ? "o" : "")
            << " attnALrScale=" << cfg.loraAttnALrScale
            << " attnBLrScale=" << cfg.loraAttnBLrScale
            << " kLrScale=" << cfg.loraKLrScale
            << " kMinTotalSupTokens=" << cfg.loraKMinTotalSupTokens
            << " kWarmupSupTokens=" << cfg.loraKWarmupSupTokens
            << " oLrScale=" << cfg.loraOLrScale
            << (cfg.loraTargetFfnDown ? "fd" : "")
            << " ffnDownLrScale=" << cfg.loraFfnDownLrScale
            << (cfg.loraTargetFfnGateUp ? "fgu" : "")
            << " ffnGateUpLrScale=" << cfg.loraFfnGateUpLrScale
            << " ffnLoRALayers=[" << cfg.loraFfnFirstLayer << "," << cfg.loraFfnLastLayer << ")"
            << (cfg.loraTargetLmHead ? "lm" : "")
            << " lmHeadALrScale=" << cfg.loraLmHeadALrScale
            << " lmHeadBLrScale=" << cfg.loraLmHeadBLrScale
            << " lmHeadBias=" << (cfg.trainLmHeadBias ? "on" : "off")
            << " lmHeadBiasLrScale=" << cfg.lmHeadBiasLrScale
            << " finalNormAffine=" << (cfg.trainFinalNormAffine ? "on" : "off")
            << " finalNormAffineLrScale=" << cfg.finalNormAffineLrScale
            << " layerRmsNorm=" << (cfg.trainLayerRmsNorm ? "on" : "off")
            << " layerRmsNormLrScale=" << cfg.layerRmsNormLrScale
            << " finalRmsNorm=" << (cfg.trainFinalRmsNorm ? "on" : "off")
            << " finalRmsNormLrScale=" << cfg.finalRmsNormLrScale
            << " hTile=" << cfg.hTile
            << " tTile=" << cfg.tTile
            << " vTile=" << cfg.vTile
            << " actClipRms=" << bitnetActClipRms
            << " bitnetQK=" << cfg.bitnetQK
            << " bitnetFfnQK=" << (cfg.bitnetFfnQK == 0 ? cfg.bitnetQK : cfg.bitnetFfnQK)
            << " bitnetTl2Bk=" << cfg.bitnetTl2Bk
            << " overfitSampleIndex=" << cfg.overfitSampleIndex
            << " resume=" << (cfg.resumeFromCheckpoint ? "on" : "off")
            << " lr=" << cfg.learningRate
            << " optimizer=" << (cfg.useMuon ? "muon" : (cfg.useAdamW ? "adamw" : "lion"))
            << " adamBeta1=" << cfg.adamBeta1
            << " adamAttnABeta1=" << cfg.adamAttnABeta1
            << " adamAttnBBeta1=" << cfg.adamAttnBBeta1
            << " adamLmHeadMatrixBeta1=" << cfg.adamLmHeadMatrixBeta1
            << " adamBeta2=" << cfg.adamBeta2
            << " adamEps=" << cfg.adamEps
            << " adamAttnABeta2=" << cfg.adamAttnABeta2
            << " adamAttnBBeta2=" << cfg.adamAttnBBeta2
            << " lionBeta1=" << cfg.lionBeta1
            << " lionBeta2=" << cfg.lionBeta2
            << " maxGradNorm=" << cfg.maxGradNorm
            << " clipGradNormByLrScale=" << (cfg.clipGradNormByLrScale ? "on" : "off")
            << " clipGradNormByParamGroup=" << (cfg.clipGradNormByParamGroup ? "on" : "off")
            << " loraPairRmsBalance=" << (cfg.loraPairRmsBalance ? "on" : "off")
            << " loraPairRmsBalanceTargetRatio=" << cfg.loraPairRmsBalanceTargetRatio
            << " loraPairRmsBalanceMaxScale=" << cfg.loraPairRmsBalanceMaxScale
            << " loraPairRmsBalanceMinRms=" << cfg.loraPairRmsBalanceMinRms
            << " adamAttnAUpdateRmsMaxRatio=" << cfg.adamAttnAUpdateRmsMaxRatio
            << " adamAttnBUpdateRmsMaxRatio=" << cfg.adamAttnBUpdateRmsMaxRatio
            << " adamLmHeadMatrixUpdateRmsMaxRatio=" << cfg.adamLmHeadMatrixUpdateRmsMaxRatio
            << " adamUpdateRmsCapMinWeightRms=" << cfg.adamUpdateRmsCapMinWeightRms
            << " adamUpdateRmsCapStartStep=" << cfg.adamUpdateRmsCapStartStep
            << " warmup=" << cfg.warmupSteps
            << " decayStart=" << cfg.decayStartStep
            << " minLrFactor=" << cfg.minLrFactor
            << " cosine=" << (cfg.useCosineDecay ? "on" : "off")
            << " plateauDecay=" << (cfg.usePlateauLrDecay ? "on" : "off")
            << " plateauMinSteps=" << cfg.plateauMinSteps
            << " plateauPatience=" << cfg.plateauPatienceSteps
            << " plateauMinRelImprovement=" << cfg.plateauMinRelImprovement
            << " lossBounceStop=" << (cfg.lossBounceStopEnabled ? "on" : "off")
            << " lossBounceStopMinSupTok=" << cfg.lossBounceStopMinSupTokens
            << " lossBounceStopStartLoss=" << cfg.lossBounceStopStartLoss
            << " lossBounceStopWindow=" << cfg.lossBounceStopWindow
            << " lossBounceStopPatience=" << cfg.lossBounceStopPatienceSteps
            << " lossBounceStopMinDelta=" << cfg.lossBounceStopMinDelta
            << " lossBounceStopMinRange=" << cfg.lossBounceStopMinRange
            << " lossBounceStopDirChanges=" << cfg.lossBounceStopMinDirectionChanges
            << " minStepsS=" << cfg.minStepsPerSec
            << " minStepsGrace=" << cfg.minStepsPerSecGrace
            << "\n";

        std::cout << "[2/2] Starting Training Loop...\n";
        int result = runCpuQLoRATrainer(cfg, dataset_path);

        std::cout << "\nPipeline execution finished successfully.\n";
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << "\n";
        return 1;
    }
}
