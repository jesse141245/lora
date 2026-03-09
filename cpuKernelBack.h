#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "bitnet_weight.h"
#include "config.h"
#include "lora_type.h"
#include "modelShape.h"
#include "paramRef.h"

// ============================================================
// Model parameter containers
// ============================================================

struct LayerWeights {
    // Base (frozen) weights: row-major [H x H]
    std::vector<float> Wq, Wk, Wv, Wo;

    // Packed BitNet I2S quantized weights
    BitNetMatI2S Wq_q, Wk_q, Wv_q, Wo_q;

    // RMSNorm gamma weights [H]
    std::vector<float> rms_attn;

    // (Optional later) FFN weights...
};


// ============================================================
// CpuKernelBackend
// ============================================================

class CpuKernelBackend {
public:
    static ModelShape loadShapeFromConfig(const std::string& modelPath);

    explicit CpuKernelBackend(TrainerConfig cfg);

    const ModelShape& shape() const noexcept { return shape_; }

    std::size_t logitsCount(std::size_t seqLen) const noexcept {
        return seqLen * shape_.vocabSize;
    }

    // Forward: produces logits for given inputIds (length T) and attentionMask (length T).
    // logitsOut must be size T * vocab.
    void forward(
        std::span<const int> inputIds,
        std::span<const int> attentionMask,
        std::span<float> logitsOut
    );

    // Backward wrt logits; assumes dLogits already computed (size T * vocab).
    void backward(
        std::span<const int> inputIds,
        std::span<const int> attentionMask,
        std::span<const float> dLogits
    );

    // LoRA only
    void zeroGrad();

    void saveLoRA(const std::string& path) const;
    void loadLoRA(const std::string& path);
    std::vector<ParamRef> collectTrainableParams();

private:
    // --------------------------------------------------------
    // Backend dispatch policy
    //
    // For now:
    //   - if cfg_.useBitNetQuant is true, prefer I2S kernels
    //   - float fallback remains available
    //
    // Old non-I2S BitNet kernels should no longer be the priority path.
    // --------------------------------------------------------
    bool useBitNetI2S() const noexcept {
        return cfg_.useBitNetQuant;
    }

    bool canUseBitNetI2S(std::size_t H) const noexcept {
        return useBitNetI2S() &&
            H != 0 &&
            (H % 32) == 0 &&
            static_cast<std::size_t>(cfg_.hTile) != 0;
    }

    std::size_t bitnetI2SBlock() const noexcept {
        return static_cast<std::size_t>(cfg_.hTile);
    }

private:
    // Config is immutable after construction
    const TrainerConfig cfg_;
    ModelShape shape_{};

    // Frozen base parameters
    std::vector<float> embeddings_;   // [V x H]
    std::vector<float> W_out_;        // [H x V] row-major

    std::vector<LayerWeights> layers_;    // size = 2 for now (fixed)

    // Trainable LoRA parameters
    std::vector<LayerLoRA>     loras_;      // size = 2
    std::vector<LayerLoRAGrad> loraGrads_;  // size = 2

    // --------------------------------------------------------
    // LoRA forward caches
    // tmpQ_cache_[ (L * maxSeqLen + t) * R + r ]
    // tmpV_cache_[ (L * maxSeqLen + t) * R + r ]
    // --------------------------------------------------------
    std::vector<float> tmpQ_cache_; // [nLayers * maxSeqLen * R]
    std::vector<float> tmpV_cache_; // [nLayers * maxSeqLen * R]

    // --------------------------------------------------------
    // Forward scratch
    // --------------------------------------------------------
    std::vector<float> x_;      // [maxSeqLen * H] current hidden
    std::vector<float> xnorm_;  // [maxSeqLen * H] rmsnorm(x)
    std::vector<float> q_;      // [maxSeqLen * H]
    std::vector<float> k_;      // [maxSeqLen * H]
    std::vector<float> v_;      // [maxSeqLen * H]
    std::vector<float> ctx_;    // [maxSeqLen * H]
    std::vector<float> attn_;   // [maxSeqLen * H]

    // Attention scratch (testing / simple implementation)
    std::vector<float> scores_; // [maxSeqLen * maxSeqLen]
    std::vector<float> probs_;  // [maxSeqLen * maxSeqLen]

    // --------------------------------------------------------
    // Backward scratch
    // --------------------------------------------------------
    std::vector<float> dX_;      // [maxSeqLen * H]
    std::vector<float> dAttn_;   // [maxSeqLen * H]
    std::vector<float> dCtx_;    // [maxSeqLen * H]
    std::vector<float> dQ_;      // [maxSeqLen * H]
    std::vector<float> dK_;      // [maxSeqLen * H]
    std::vector<float> dV_;      // [maxSeqLen * H]
    std::vector<float> dXnorm_;  // [maxSeqLen * H]
    std::vector<float> dXin_;    // [maxSeqLen * H]

    // Softmax backward scratch (size T, but allocated to maxSeqLen)
    std::vector<float> tmpdP_;   // [maxSeqLen]

    // Rank scratch for LoRA backprop (per token)
    std::vector<float> dTmpRank_; // [R]

    // --------------------------------------------------------
    // Checkpoints
    //
    // x_ckpt_ layout:
    //   x_ckpt_[ (L * maxSeqLen + t) * H + h ]
    //
    // Stores x entering each layer:
    //   L = 0: embedding output
    //   L = 1: after layer 0
    //   L = 2: after layer 1 (final)
    //
    // Total size: (nLayers + 1) * maxSeqLen * H
    // --------------------------------------------------------
    std::vector<float> x_ckpt_;
};