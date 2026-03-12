#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <span>

#include "config.h"          
#include "paramRef.h"        
#include "bitnet_weight.h"   

struct ModelShape {
    std::size_t vocabSize;
    std::size_t hiddenSize;
    std::size_t nLayers;
    std::size_t nHeads;
    std::size_t nKvHeads;
};

struct LayerWeights {
    std::vector<float> Wq;
    std::vector<float> Wk;
    std::vector<float> Wv;
    std::vector<float> Wo;
    std::vector<float> rms_attn;

    BitNetMatI2S Wq_q;
    BitNetMatI2S Wk_q;
    BitNetMatI2S Wv_q;
    BitNetMatI2S Wo_q;
};

struct LayerLoRA {
    std::vector<float> Aq;
    std::vector<float> Bq;
    std::vector<float> Av;
    std::vector<float> Bv;
};

struct LayerLoRAGrads {
    std::vector<float> gradAq;
    std::vector<float> gradBq;
    std::vector<float> gradAv;
    std::vector<float> gradBv;
};

class CpuKernelBackend {
public:
    explicit CpuKernelBackend(TrainerConfig cfg);

    const ModelShape& shape() const { return shape_; }
    std::size_t logitsCount(std::size_t T) const { return T * shape_.vocabSize; }

    void forward(
        std::span<const int> inputIds,
        std::span<const int> attentionMask,
        std::span<float> logitsOut
    );

    void backward(
        std::span<const int> inputIds,
        std::span<const int> attentionMask,
        std::span<const float> dLogits
    );

    void zeroGrad();
    void saveLoRA(const std::string& path) const;
    void loadLoRA(const std::string& path);
    std::vector<ParamRef> collectTrainableParams();

private:
    ModelShape loadShapeFromConfig(const std::string& modelPath);

    TrainerConfig cfg_;
    ModelShape shape_;

    std::vector<float> embeddings_;
    std::vector<float> W_out_;
    BitNetMatI2S W_out_q;

    std::vector<LayerWeights> layers_;

    std::vector<LayerLoRA> loras_;
    std::vector<LayerLoRAGrads> loraGrads_;

    std::vector<float> tmpQ_cache_;
    std::vector<float> tmpV_cache_;

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
    std::vector<float> dX_;
    std::vector<float> dAttn_;
    std::vector<float> dCtx_;
    std::vector<float> dQ_;
    std::vector<float> dK_;
    std::vector<float> dV_;
    std::vector<float> dXnorm_;
    std::vector<float> dXin_;

    std::vector<float> rms_final_;
    std::vector<float> xnorm_final_;
    std::vector<float> dXnorm_final_;

    std::vector<float> x_ckpt_;
    std::vector<float> tmpdP_;
};