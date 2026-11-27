#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

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
    // 0 = use all model layers. Non-zero = use only first N layers
    // to reduce memory/startup time for experimentation.
    // ============================================================
    std::size_t forcedLayers = 0;

    // ============================================================
    // Optimization
    // ============================================================
    float learningRate = 1e-4f;
    float weightDecay = 0.01f;
    bool  useAdamW = true;    // false => use Lion unless useMuon is enabled
    bool  useMuon = false;
    int   gradAccumSteps = 1;
    float maxGradNorm = 1.0f;   // <= 0 means no clipping
    bool  clipGradNormByLrScale = false;
    bool  clipGradNormByParamGroup = false;
    bool  loraPairRmsBalance = false;
    float loraPairRmsBalanceTargetRatio = 1.0f;
    float loraPairRmsBalanceMaxScale = 2.0f;
    float loraPairRmsBalanceMinRms = 1e-6f;
    float adamAttnAUpdateRmsMaxRatio = 0.0f;
    float adamAttnBUpdateRmsMaxRatio = 0.0f;
    float adamLmHeadMatrixUpdateRmsMaxRatio = 0.0f;
    float adamUpdateRmsCapMinWeightRms = 1e-6f;
    int   adamUpdateRmsCapStartStep = 0;
    int   ignoreIndex = -100;
    // 0 => use every supervised token. Positive values cap each sample's
    // response supervision to the first N valid target positions.
    int   maxSupervisedTokensPerSample = 0;
    std::int64_t maxTotalSupervisedTokens = 0;
    int   targetSupervisedTokensPerStep = 0;
    int   referenceSupervisedTokensPerStep = 0;
    // Optional staged bootstrap: during the first N supervised tokens,
    // update only LM-head LoRA and LM-head bias parameters.
    std::int64_t lmHeadOnlyBootstrapSupTokens = 0;
    // Optional second stage: after the LM-head-only bootstrap, keep only
    // attention LoRA trainable until this total supervised-token budget is
    // reached, then enable the full trainable surface. Values <= the LM-head
    // bootstrap cutoff disable the timed second-stage cutoff.
    std::int64_t attentionOnlyBootstrapSupTokens = 0;
    // When enabled, switch from LM-head-only bootstrap into attention-LoRA-only
    // updates instead of enabling the full surface immediately.
    bool  postBootstrapAttentionOnly = false;
    // Optional stage-specific LR multipliers for the staged train-surface
    // schedule. The base per-parameter lrScale still applies; these are
    // multiplicative overlays for the active stage only.
    float lmHeadBootstrapLrScale = 1.0f;
    float attentionOnlyBootstrapLrScale = 1.0f;
    // Optional early attention LoRA warm start: keep attention A matrices
    // frozen until this supervised-token budget is reached, while attention
    // B matrices remain trainable. Useful when A updates are too noisy early.
    std::int64_t attentionBOnlyBootstrapSupTokens = 0;
    float attentionBOnlyBootstrapLrScale = 1.0f;
    // Optional softer early attention-A schedule: linearly ramp attention A
    // lrScale from attentionAInitialLrScale to 1.0 over this supervised-token
    // budget. Values <= 0 disable the ramp.
    std::int64_t attentionABootstrapSupTokens = 0;
    float attentionAInitialLrScale = 1.0f;
    // Optional training-only emphasis for the first N supervised target
    // positions in each response. A weight of 1 preserves the baseline.
    int   firstAnswerLossWeightTokens = 0;
    float firstAnswerLossWeight = 1.0f;
    // Optional robust CE cap. When > 0, token losses above the cap are
    // downweighted by cap/loss for both accumulation and logits gradients.
    float tokenLossCap = 0.0f;
    // <= 0 uses a hard cap. > 0 uses a smooth cap:
    // cap + softness * log1p((loss - cap) / softness).
    float tokenLossCapSoftness = 0.0f;
    std::int64_t tokenLossCapMinTotalSupTokens = 0;
    // Optional target-frequency weighting. When targetFreqLossMinFreq > 0,
    // tokens that appear fewer times than that in the training targets are
    // linearly scaled toward targetFreqLossMinScale.
    int targetFreqLossMinFreq = 0;
    float targetFreqLossMinScale = 1.0f;
    std::int64_t targetFreqLossMinTotalSupTokens = 0;
    // Train-only label smoothing applied to logits gradients. Logged/eval CE
    // remains the raw one-hot objective so rebound diagnostics stay comparable.
    float labelSmoothing = 0.0f;
    std::int64_t labelSmoothingMinTotalSupTokens = 0;
    // Train-only logit L2 penalty on supervised rows. Raw CE remains logged
    // separately; this is a confidence/logit-scale control, not a metric cap.
    float logitL2Penalty = 0.0f;
    std::int64_t logitL2MinTotalSupTokens = 0;
    // Train-only penalty for saturated wrong-token margins:
    // 0.5 * penalty * max(0, maxWrongLogit - targetLogit - threshold)^2.
    float wrongTopMarginPenalty = 0.0f;
    float wrongTopMarginThreshold = 8.0f;
    std::int64_t wrongTopMarginMinTotalSupTokens = 0;
    float hardTokenBoostThreshold = 0.0f;
    float hardTokenBoostSlope = 0.0f;
    float hardTokenBoostMaxScale = 1.0f;
    std::int64_t hardTokenBoostMinTotalSupTokens = 0;
    int rareHardTokenBoostMaxTrainFreq = 0;
    float rareHardTokenBoostThreshold = 0.0f;
    float rareHardTokenBoostScale = 1.0f;
    std::int64_t rareHardTokenBoostMinTotalSupTokens = 0;
    int rareWrongTopMarginMaxTrainFreq = 0;
    float rareWrongTopMarginPenalty = 0.0f;
    float rareWrongTopMarginThreshold = 8.0f;
    std::int64_t rareWrongTopMarginMinTotalSupTokens = 0;
    int rareTargetBiasMaxTrainFreq = 0;
    float rareTargetBiasMinLoss = 0.0f;
    float rareTargetBiasLrScale = 0.0f;
    float rareTargetBiasMaxAbs = 0.0f;
    std::int64_t rareTargetBiasMinTotalSupTokens = 0;
    // When enabled, average each sample's logits gradient by the number of
    // supervised tokens before backward. When disabled, keep the historical
    // summed-token gradient scale.
    bool  normalizeDLogitsBySampleTokens = false;
    float adamBeta1 = 0.9f;
    float adamBeta2 = 0.999f;
    float adamEps = 1e-8f;
    // Optional AdamW beta1 overrides. Values < 0 use adamBeta1.
    float adamAttnABeta1 = -1.0f;
    float adamAttnBBeta1 = -1.0f;
    float adamLmHeadMatrixBeta1 = -1.0f;
    // Optional AdamW beta2 overrides for attention LoRA factor matrices.
    // Values < 0 use adamBeta2. These are optimizer-only knobs; lrScale still
    // controls update magnitude.
    float adamAttnABeta2 = -1.0f;
    float adamAttnBBeta2 = -1.0f;
    float lionBeta1 = 0.9f;
    float lionBeta2 = 0.99f;
    bool  zeroGradAfterStep = true;
    // LR schedule controls:
    // warmupSteps <= 0 => auto short warmup
    // useCosineDecay=false => constant LR after warmup
    // decayStartStep > warmupSteps inserts a constant-LR hold before cosine decay
    // minLrFactor scales the floor relative to learningRate
    int   warmupSteps = -1;
    bool  useCosineDecay = false;
    int   decayStartStep = -1;
    int   decaySteps = -1;
    float minLrFactor = 0.01f;
    // Optional plateau-triggered cosine: hold peak LR until the smoothed
    // training loss stops improving enough for long enough, then begin cosine
    // decay from that step.
    bool  usePlateauLrDecay = false;
    int   plateauMinSteps = -1;
    int   plateauPatienceSteps = 32;
    float plateauMinRelImprovement = 0.01f;
    // Optional guard for single-batch loss bounce/plateau. This is intended
    // for screening runs where fixed train order starts oscillating after the
    // useful descent phase.
    bool  lossBounceStopEnabled = false;
    std::int64_t lossBounceStopMinSupTokens = 0;
    float lossBounceStopStartLoss = 0.0f; // <= 0 disables this condition
    int   lossBounceStopWindow = 8;
    int   lossBounceStopPatienceSteps = 12;
    float lossBounceStopMinDelta = 0.02f;
    float lossBounceStopMinRange = 0.75f;
    int   lossBounceStopMinDirectionChanges = 3;

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
    bool useDoRA = false;
    // PiSSA is supplied as a precomputed cache: residual frozen weights plus
    // nonzero LoRA A/B initializers. This keeps training kernels unchanged and
    // prevents double-counting the principal component.
    bool usePiSSA = false;
    std::string pissaPath;
    // Exact DoRA norm refresh recomputes ||W + BA|| and is much slower than
    // normal LoRA on large ternary matrices. Default to detached base norms for
    // the optimized path; set >0 to periodically refresh exact norms.
    int doraNormUpdateInterval = 0;
    // LM-head exact DoRA norm refresh is O(vocab * hidden * rank) and is
    // prohibitively expensive for large vocabularies. Default to cached base
    // norms for the head; set >0 only when explicitly testing exact refresh.
    int doraLmHeadNormUpdateInterval = 0;
    // Cache DoRA Q/K pre-RoPE activations by default for speed. Set true to
    // recompute those activations in backward and reduce peak memory.
    bool doraRecomputeQKPreRope = false;
    // Long-context attention can make a persistent per-layer probability cache
    // dominate memory: O(layers * heads * T^2). When enabled, keep the legacy
    // cached-probability path. When disabled, recompute probabilities from the
    // cached Q/K/V tensors during backward instead.
    bool cacheAttentionProbs = true;
    // Experimental exact packed-segment support. Datasets may mark the first
    // token of each packed sequence with attention_mask=2; when enabled, the
    // backend resets RoPE positions and causal attention at those boundaries.
    bool enablePackedSegments = false;
    // Release FP32 base matrices after packed BitNet/TL2 weights are available.
    // Keep this enabled for the normal/balanced CPU path. Disable explicitly
    // only for high-memory autotune runs that test whether retaining dense
    // copies improves throughput enough to justify the extra RAM.
    bool releaseQuantizedDenseWeights = true;
    std::vector<std::string> loraTargets; // e.g. {"attn.q", "attn.v"}
    bool loraTargetQ = true;
    bool loraTargetK = false;
    bool loraTargetV = true;
    bool loraTargetO = false;
    float loraAttnALrScale = 1.0f;
    float loraAttnBLrScale = 1.0f;
    float loraKLrScale = 1.0f;
    std::int64_t loraKMinTotalSupTokens = 0;
    std::int64_t loraKWarmupSupTokens = 0;
    float loraOLrScale = 1.0f;
    bool loraTargetFfnDown = false;
    float loraFfnDownLrScale = 1.0f;
    bool loraTargetFfnGateUp = false;
    float loraFfnGateUpLrScale = 1.0f;
    // FFN LoRA layer range is 0-based and end-exclusive. Negative => full range.
    int loraFfnFirstLayer = -1;
    int loraFfnLastLayer = -1;
    bool loraTargetLmHead = false;
    float loraLmHeadALrScale = 1.0f;
    float loraLmHeadBLrScale = 1.0f;
    float loraLmHeadLrScale = 1.0f;
    bool trainLmHeadBias = false;
    float lmHeadBiasLrScale = 1.0f;
    bool trainFinalNormAffine = false;
    float finalNormAffineLrScale = 1.0f;
    bool trainLayerRmsNorm = false;
    float layerRmsNormLrScale = 1.0f;
    bool trainFinalRmsNorm = false;
    float finalRmsNormLrScale = 1.0f;
    bool freezeBase = true;

    // ============================================================
    // Logging / eval
    // ============================================================
    int logInterval = 10;
    int checkpointInterval = 100;
    int evalInterval = 0;      // 0 = disabled
    int numEvalBatches = 0;
    // Optional eval-loss target stop. Values <= 0 disable the check.
    float evalLossStopThreshold = 0.0f;
    int evalLossStopMinStep = 0;
    bool checkpointOnEvalLossStop = true;
    // Optional train-loss target stop. Values <= 0 disable the check.
    float trainLossStopThreshold = 0.0f;
    int trainLossStopMinStep = 0;
    std::string trainLossStopMetric = "loss"; // loss, ma8, ma32
    bool evalOnTrainLossStop = false;
    bool checkpointOnTrainLossStop = false;
    int trainProbeInterval = 0;     // 0 = disabled
    int numTrainProbeBatches = 0;   // 0 = disabled
    std::vector<std::size_t> trainProbeSampleIndices;
    std::vector<std::size_t> replaySampleIndices;
    int replaySamplesPerBatch = 0;
    int replayEverySteps = 1;
    float replaySampleLossWeight = 1.0f;
    bool appendReplaySamples = false;
    bool replayGradientOnly = false;
    int replayInjectStartStep = 0;
    float onlineHardReplayMinLoss = 0.0f;
    float onlineHardReplayMinMaxTokenLoss = 0.0f;
    int onlineHardReplayStartStep = 0;
    int onlineHardReplayMaxSamples = 32;
    std::int64_t trainProbeGate1SupTokens = 0;
    float trainProbeGate1MaxLoss = 0.0f;
    std::int64_t trainProbeGate2SupTokens = 0;
    float trainProbeGate2MaxLoss = 0.0f;
    std::string evalDatasetPath;
    bool evalPredictionStats = false;
    bool saveFinalArtifacts = true;
    float minStepsPerSec = 0.0f; // <= 0 disables throughput abort
    int minStepsPerSecGrace = 0; // optimizer-step grace period before abort check

    // ============================================================
    // Repro / resume
    // ============================================================
    unsigned int seed = 42;
    bool resumeFromCheckpoint = false;
    bool mixedPrecision = false;
    // Overfit debug mode: >=0 always sample this dataset index each step.
    // -1 disables and uses normal epoch shuffle/sampling.
    int overfitSampleIndex = -1;

    // ============================================================
    // BitNet quantization
    // ============================================================
    bool useBitNetQuant = true;

    // Legacy / stage-1 blockwise path
    std::size_t bitnetBlock = 64;   // number of weights per scale in old path

    // Which attention projections to quantize
    bool bitnetQuantizeQKV = true;  // quantize Wq/Wk/Wv
    bool bitnetFuseQKV = true;      // fuse quantized Q/K/V i2s launches
    bool bitnetQuantizeWo = true;  // quantize Wo
    bool bitnetQuantizeLmHead = false; // quantize output head (very slow startup)
    bool bitnetLmHeadFloatBackward = false; // keep quantized forward, dense FP32 backward
    bool bitnetLmHeadQuantBackwardOnly = false; // keep dense forward, quantized backward
    bool bitnetHadamard32 = false; // apply 32-wide Hadamard rotation before I2S quantization
    bool bitnetTl2Qkv = false; // experimental TL2-style forward for Q/K/V
    bool bitnetTl2Wo = false; // experimental TL2-style forward for Wo
    bool bitnetTl2Ffn = false; // experimental TL2-style forward for gate/up/down
    std::size_t bitnetTl2Bk = 96; // experimental TL2 block size (currently only 96)

    // ============================================================
    // BitNet I2S layout
    // These define the forward packed/scaled layout stored in BitNetMatI2S
    // ============================================================
    std::size_t bitnetQK = 128;     // scale tile height in i/k dimension
    // 0 => reuse bitnetQK for FFN. Non-zero allows a finer FFN-only scale tile.
    std::size_t bitnetFfnQK = 0;
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
