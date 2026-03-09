#pragma once
#include <string>
#include <vector>
#include <cstddef>

struct TrainerConfig {
    // Paths
    std::string modelPath;          // base BitNet model weights dir (contains config.json)
    std::string loraOutputPath;     // final LoRA adapter output
    std::string checkpointDir;      // periodic checkpoints
    std::string runId;              // helpful for naming

    // Train loop
    int maxSteps = 1000;
    int batchSize = 4;
    std::size_t maxSeqLen = 512;

    // (Optional / derived) Model shape info for logging/UI only.
    // Source of truth should be ModelShape loaded from modelPath/config.json.
    int vocabSize = 0;
    int hiddenSize = 0;
    int nLayers = 0;
    int nHeads = 0;

    // Backend control
    // 0 = use model config. Non-zero = force backend to run only this many layers.
    std::size_t forcedLayers = 2;

    // Optimization
    float learningRate = 1e-4f;
    float weightDecay = 0.01f;
    int   gradAccumSteps = 1;
    float maxGradNorm = 1.0f;   // <=0 means no clipping
    int   ignoreIndex = -100;

    // LoRA
    std::size_t hTile = 128;
    std::size_t tTile = 8;
    std::size_t vTile = 2048;
    std::size_t rTile = 8;       // (unused for now; reserved for rank-tiling kernels)
    std::size_t iTile = 64;
    std::size_t loraRank = 8;
    float loraAlpha = 16.0f;
    float loraDropout = 0.0f;
    std::vector<std::string> loraTargets; // e.g. {"attn.q", "attn.v"}
    bool  freezeBase = true;

    // Logging / eval
    int logInterval = 10;
    int checkpointInterval = 100;
    int evalInterval = 0;      // 0 = disabled
    int numEvalBatches = 0;

    // Repro / resume
    unsigned int seed = 42;
    bool resumeFromCheckpoint = false;
    bool mixedPrecision = false;

    bool useBitNetQuant = true;
    std::size_t bitnetBlock = 64;        // number of weights per scale
    bool bitnetQuantizeQKV = true;       // quantize Wq/Wk/Wv
    bool bitnetQuantizeWo = true;       // quantize Wo (attn output projection)
};