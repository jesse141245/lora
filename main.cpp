#include <filesystem>
#include <iostream>
#include <string>
#include <exception>
#include <algorithm>
#include <thread>
#include <cctype>
#include <cstdlib>
#include <optional>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "config.h"
#include "qlora_cpu.h"

namespace fs = std::filesystem;

namespace {

std::optional<std::string> getEnvStringRaw(const char* name) {
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

std::size_t normalizeBitNetQk(int value, std::size_t fallback) {
    if (value <= 0) return fallback;
    const int clamped = std::max(32, value);
    return static_cast<std::size_t>((clamped + 31) / 32 * 32);
}

std::optional<int> getEnvInt(const char* name) {
    const auto value = getEnvStringRaw(name);
    if (!value) return std::nullopt;
    try {
        return std::stoi(*value);
    }
    catch (...) {
        return std::nullopt;
    }
}

std::optional<float> getEnvFloat(const char* name) {
    const auto value = getEnvStringRaw(name);
    if (!value) return std::nullopt;
    try {
        return std::stof(*value);
    }
    catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> getEnvBool(const char* name) {
    const auto value = getEnvStringRaw(name);
    if (!value) return std::nullopt;
    const std::string s = lowerCopy(*value);

    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    return std::nullopt;
}

std::optional<std::string> getEnvString(const char* name) {
    return getEnvStringRaw(name);
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

        const std::string model_path = getEnvString("QLORA_TEST_MODEL_PATH").value_or("./test_model");
        const std::string dataset_path = getEnvString("QLORA_TEST_DATASET_PATH").value_or(
            fs::exists("./tests/overfit_tutor_dataset.jsonl")
                ? "./tests/overfit_tutor_dataset.jsonl"
                : "./test_dataset.jsonl");

        std::cout << "[1/2] Configuring Trainer...\n";
        TrainerConfig cfg;
        cfg.modelPath = model_path;
        cfg.loraOutputPath = std::filesystem::absolute("lora_output.bin").string();
        cfg.checkpointDir = std::filesystem::absolute("checkpoints").string();

        cfg.maxSeqLen = 256;     // leave room for longer tutoring-style examples
        cfg.batchSize = 1;       // seq256 tutor runs are memory/bandwidth heavy
        cfg.gradAccumSteps = 4;  // smaller effective batch than the old supTokens~8 sweeps
        cfg.maxSteps = 500;
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
        // Reuse beta fields as optimizer betas (AdamW: 0.9 / 0.999).
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
        if (const auto v = getEnvInt("QLORA_TEST_BITNET_QK")) {
            cfg.bitnetQK = normalizeBitNetQk(*v, cfg.bitnetQK);
        }
        if (const auto v = getEnvInt("QLORA_TEST_BITNET_FFN_QK")) {
            cfg.bitnetFfnQK = (*v <= 0) ? 0u : normalizeBitNetQk(*v, cfg.bitnetQK);
        }
        if (const auto v = getEnvInt("QLORA_TEST_GRAD_ACCUM")) {
            cfg.gradAccumSteps = std::max(1, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_LOG_INTERVAL")) {
            cfg.logInterval = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_CHECKPOINT_INTERVAL")) {
            cfg.checkpointInterval = std::max(0, *v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_FORCED_LAYERS")) {
            cfg.forcedLayers = (*v <= 0) ? 0u : static_cast<std::size_t>(*v);
        }
        if (const auto v = getEnvInt("QLORA_TEST_OVERFIT_SAMPLE")) {
            cfg.overfitSampleIndex = *v;
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
        if (const auto v = getEnvBool("QLORA_TEST_USE_ADAMW")) {
            cfg.useAdamW = *v;
        }
        if (const auto v = getEnvFloat("QLORA_TEST_LR")) {
            cfg.learningRate = *v;
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
            << " logInterval=" << cfg.logInterval
            << " checkpointInterval=" << cfg.checkpointInterval
            << " loraRank=" << cfg.loraRank
            << " loraAlpha=" << cfg.loraAlpha
            << " hTile=" << cfg.hTile
            << " tTile=" << cfg.tTile
            << " vTile=" << cfg.vTile
            << " actClipRms=" << bitnetActClipRms
            << " bitnetQK=" << cfg.bitnetQK
            << " bitnetFfnQK=" << (cfg.bitnetFfnQK == 0 ? cfg.bitnetQK : cfg.bitnetFfnQK)
            << " bitnetTl2Bk=" << cfg.bitnetTl2Bk
            << " overfitSampleIndex=" << cfg.overfitSampleIndex
            << " lr=" << cfg.learningRate
            << " optimizer=" << (cfg.useAdamW ? "adamw" : "lion")
            << " warmup=" << cfg.warmupSteps
            << " decayStart=" << cfg.decayStartStep
            << " minLrFactor=" << cfg.minLrFactor
            << " cosine=" << (cfg.useCosineDecay ? "on" : "off")
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
