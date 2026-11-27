#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <span>

#include "config.h"          
#include "paramRef.h"        
#include "bitnet_weight.h"   
#include "lora_type.h"

struct ModelShape {
    std::size_t vocabSize;
    std::size_t hiddenSize;
    std::size_t nLayers;
    std::size_t nHeads;
    std::size_t nKvHeads;
    std::size_t ffnSize;
    std::size_t ropeDim;
    float ropeTheta;

    std::size_t headDim() const { return hiddenSize / nHeads; }
    // kvDim is the authoritative source for key/value projection width.
    // Deriving it from a live weight tensor (Wk.size()/H) couples
    // correctness to keeping FP32 weights resident and breaks if they
    // are freed after quantization. Use this method everywhere instead.
    std::size_t kvDim()   const { return nKvHeads * headDim(); }
};

struct LayerWeights {
    std::vector<float> Wq;
    std::vector<float> Wk;
    std::vector<float> Wv;
    std::vector<float> Wo;
    std::vector<float> rms_attn;
    std::vector<float> rms_ffn;
    std::vector<float> Wgate;
    std::vector<float> Wup;
    std::vector<float> Wdown;

    BitNetMatI2S Wq_q;
    BitNetMatI2S Wk_q;
    BitNetMatI2S Wv_q;
    BitNetMatI2S Wo_q;
    BitNetMatI2S Wgate_q;
    BitNetMatI2S Wup_q;
    BitNetMatI2S Wdown_q;

    BitNetMatTL2 Wq_tl2;
    BitNetMatTL2 Wk_tl2;
    BitNetMatTL2 Wv_tl2;
    BitNetMatTL2 Wo_tl2;
    BitNetMatTL2 Wgate_tl2;
    BitNetMatTL2 Wup_tl2;
    BitNetMatTL2 Wdown_tl2;
    BitNetMatTL2Gen Wq_tl2g;
    BitNetMatTL2Gen Wk_tl2g;
    BitNetMatTL2Gen Wv_tl2g;
    BitNetMatTL2Gen Wo_tl2g;
    BitNetMatTL2Gen Wgate_tl2g;
    BitNetMatTL2Gen Wup_tl2g;
    BitNetMatTL2Gen Wdown_tl2g;
};

class CpuKernelBackend {
public:
    explicit CpuKernelBackend(TrainerConfig cfg);

    const ModelShape& shape() const { return shape_; }
    std::size_t logitsCount(std::size_t T) const { return T * shape_.vocabSize; }

    void forward(
        std::span<const int> inputIds,
        std::span<const int> attentionMask,
        std::span<float> logitsOut,
        std::span<const int> lossMask = {}
    );

    void backward(
        std::span<const int> inputIds,
        std::span<const int> attentionMask,
        std::span<const float> dLogits,
        std::span<const int> lossMask = {}
    );

    void zeroGrad();
    void markTrainableParamsUpdated();
    void saveLoRA(const std::string& path) const;
    void loadLoRA(const std::string& path);
    std::vector<ParamRef> collectTrainableParams();

private:
    ModelShape loadShapeFromConfig(const std::string& modelPath);
    bool ffnLoraLayerEnabled(std::size_t layer) const;

    TrainerConfig cfg_;
    ModelShape shape_;
    std::uint64_t doraParamVersion_ = 0;
    std::uint64_t doraNormVersion_ = 0;
    std::uint64_t doraLmHeadNormVersion_ = 0;

    std::vector<float> embeddings_;
    std::vector<float> W_out_;
    BitNetMatI2S W_out_q;
    std::vector<float> ropeCosTable_;
    std::vector<float> ropeSinTable_;

    std::vector<LayerWeights> layers_;

    std::vector<LayerLoRA> loras_;
    std::vector<LayerLoRAGrads> loraGrads_;
    LmHeadLoRA lmHeadLora_;
    LmHeadLoRAGrads lmHeadLoraGrads_;
    std::vector<float> lmHeadBias_;
    std::vector<float> lmHeadBiasGrad_;
    std::vector<float> finalNormScale_;
    std::vector<float> finalNormBias_;
    std::vector<float> finalNormScaleGrad_;
    std::vector<float> finalNormBiasGrad_;
    std::vector<float> rmsFinalGrad_;

    std::vector<float> tmpQ_cache_;
    std::vector<float> tmpK_cache_;
    std::vector<float> tmpV_cache_;
    std::vector<float> tmpO_cache_;
    std::vector<float> tmpFfnDown_cache_;
    std::vector<float> tmpFfnGate_cache_;
    std::vector<float> tmpFfnUp_cache_;
    std::vector<float> tmpLmHead_cache_;
    std::vector<float> dTmpLmHead_cache_;
    std::vector<float> q_dora_prerope_cache_;
    std::vector<float> k_dora_prerope_cache_;
    std::vector<float> q_dora_prerope_recompute_;
    std::vector<float> k_dora_prerope_recompute_;
    std::vector<float> tmpLmHeadDora_cache_;
    std::vector<float> dLogitsDoraScratch_;
    std::vector<float> xnorm_cache_;
    std::vector<float> xnorm_ffn_cache_;
    std::vector<float> q_cache_;
    std::vector<float> k_cache_;
    std::vector<float> v_cache_;
    std::vector<float> ctx_cache_;
    std::vector<float> probs_cache_;
    std::vector<float> x_mid_cache_;
    std::vector<float> gate_cache_;
    std::vector<float> up_cache_;
    std::vector<float> ffn_act_cache_;

    std::vector<float> x_;
    std::vector<float> xnorm_;
    std::vector<float> q_;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<float> ctx_;
    std::vector<float> attn_;

    std::vector<float> scores_;
    std::vector<float> probs_;

    std::vector<float> dTmpRank_;
    std::vector<float> attnLoraGradAqLocal_;
    std::vector<float> attnLoraGradBqLocal_;
    std::vector<float> attnLoraGradAvLocal_;
    std::vector<float> attnLoraGradBvLocal_;
    std::vector<float> attnLoraTmpRankLocal_;
    std::vector<float> dX_;
    std::vector<float> dAttn_;
    std::vector<float> dCtx_;
    std::vector<float> dQ_;
    std::vector<float> dK_;
    std::vector<float> dV_;
    std::vector<float> dXnorm_;
    std::vector<float> dXin_;
    std::vector<float> dXmid_;

    std::vector<float> x_mid_;
    std::vector<float> xnorm_ffn_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> ffn_act_;
    std::vector<float> ffn_out_;

    std::vector<float> dFfnOut_;
    std::vector<float> dAct_;
    std::vector<float> dGate_;
    std::vector<float> dUp_;
    std::vector<float> dXnormFfn_;

    std::vector<float> rms_final_;
    std::vector<float> xnorm_final_;
    std::vector<float> xnorm_final_base_cache_;
    std::vector<float> dXnorm_final_;
    std::vector<float> lmHeadDyCompactScratch_;
    std::vector<float> lmHeadDxCompactScratch_;

    std::vector<float> x_ckpt_;
    std::vector<float> tmpdP_;
    std::vector<std::size_t> lmHeadRowsScratch_;

    // ── Persistent scratch buffers (avoids per-step heap churn) ──────────────
    // Sized once in the constructor; reused every forward/backward call.
    std::vector<int8_t> fwdX8Scratch_;      // forward scratch rows * max input dim
    std::vector<float>  fwdXScaleScratch_;  // forward scratch rows * max(nKBlocks across mats)
    std::vector<int8_t> bwdDY8Scratch_;     // backward scratch rows * max(out-dim)
    std::vector<float>  bwdDYScaleScratch_; // backward scratch rows * max(nJBlocks)
    std::vector<int8_t> bwdDK8Scratch_;     // backward scratch rows * max(kvDim, ffnDim)
    std::vector<float>  bwdDKScaleScratch_; // backward scratch rows * ceil(max(kvDim, ffnDim)/jq)
    std::vector<int8_t> bwdDV8Scratch_;     // backward scratch rows * max(kvDim, ffnDim)
    std::vector<float>  bwdDVScaleScratch_; // backward scratch rows * ceil(max(kvDim, ffnDim)/jq)
};
