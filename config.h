#pragma once
#include <string>
#include <vector>
#include <cstddef>

struct TrainerConfig {
    // ============================================================
    // Paths
    // ============================================================
    std::string modelPath;          // base BitNet model weights dir (contains config.json)
    std::string loraOutputPath;     // final LoRA adapter output
    std::string checkpointDir;      // periodic checkpoints
    std::string runId;              // helpful for naming

    // ============================================================
    // Train loop
    // ============================================================
    int maxSteps = 1000;
    int batchSize = 4;
    std::size_t maxSeqLen = 512;

    // ============================================================
    // Model info (logging / UI only)
    // Source of truth should be ModelShape loaded from modelPath/config.json
    // ============================================================
    int vocabSize = 128256;
    int hiddenSize = 2560;
    int nLayers = 0;
    int nHeads = 0;

    // ============================================================
    // Backend control
    // 0 = use model config. Non-zero = force backend to run only this many layers.
    // ============================================================
    std::size_t forcedLayers = 2;

    // ============================================================
    // Optimization
    // ============================================================
    float learningRate = 1e-4f;
    float weightDecay = 0.01f;
    int   gradAccumSteps = 1;
    float maxGradNorm = 1.0f;   // <= 0 means no clipping
    int   ignoreIndex = -100;
    float lionBeta1 = 0.9f;
    float lionBeta2 = 0.99f;
    bool  zeroGradAfterStep = true;

    // ============================================================
    // Kernel tiling / LoRA
    // ============================================================
    std::size_t hTile = 128;
    std::size_t tTile = 2;
    std::size_t vTile = 2048;
    std::size_t rTile = 8;      // reserved for future rank-tiling kernels

    // Backward input tile for HH^T-style paths.
    // Current assisted BitNet backward is safest when:
    //   iTile == bitnetBwdIB
    std::size_t iTile = 32;

    std::size_t loraRank = 8;
    float loraAlpha = 8.0f;
    float loraDropout = 0.0f;
    std::vector<std::string> loraTargets; // e.g. {"attn.q", "attn.v"}
    bool freezeBase = true;

    // ============================================================
    // Logging / eval
    // ============================================================
    int logInterval = 10;
    int checkpointInterval = 100;
    int evalInterval = 0;      // 0 = disabled
    int numEvalBatches = 0;

    // ============================================================
    // Repro / resume
    // ============================================================
    unsigned int seed = 42;
    bool resumeFromCheckpoint = false;
    bool mixedPrecision = false;

    // ============================================================
    // BitNet quantization
    // ============================================================
    bool useBitNetQuant = true;

    // Legacy / stage-1 blockwise path
    std::size_t bitnetBlock = 64;   // number of weights per scale in old path

    // Which attention projections to quantize
    bool bitnetQuantizeQKV = true;  // quantize Wq/Wk/Wv
    bool bitnetQuantizeWo = true;  // quantize Wo

    // ============================================================
    // BitNet I2S layout
    // These define the forward packed/scaled layout stored in BitNetMatI2S
    // ============================================================
    std::size_t bitnetQK = 128;     // scale tile height in i/k dimension
    std::size_t bitnetJQ = 32;      // scale tile width in j/output dimension

    // ============================================================
    // Option C backward assist tile
    //
    // Current safe setting:
    //   bitnetBwdIB = 32
    //   bitnetBwdJB = 32
    //   iTile       = 32
    //
    // Current assisted backward kernels assume:
    //   iTile == bitnetBwdIB
    //
    // Later, larger iTile values can be supported by splitting runtime
    // tiles into assist-sized subtiles.
    // ============================================================
    std::size_t bitnetBwdIB = 32;
    std::size_t bitnetBwdJB = 32;
};