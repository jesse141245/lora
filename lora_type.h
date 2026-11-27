#pragma once
#include <vector>
#include <cstdint>

struct LayerLoRA {
    std::vector<float> Aq;
    std::vector<float> Bq;
    std::vector<float> Ak;
    std::vector<float> Bk;
    std::vector<float> Av;
    std::vector<float> Bv;
    std::vector<float> Ao;
    std::vector<float> Bo;
    std::vector<float> Afd;
    std::vector<float> Bfd;
    std::vector<float> Afg;
    std::vector<float> Bfg;
    std::vector<float> Afu;
    std::vector<float> Bfu;

    // DoRA trainable magnitudes and detached forward caches. The M* vectors are
    // serialized/trainable; C* and R* are per-forward detached norm and m/C.
    std::vector<float> Mq, Cq, Rq;
    std::vector<float> Mk, Ck, Rk;
    std::vector<float> Mv, Cv, Rv;
    std::vector<float> Mo, Co, Ro;
    std::vector<float> Mfd, Cfd, Rfd;
    std::vector<float> Mfg, Cfg, Rfg;
    std::vector<float> Mfu, Cfu, Rfu;
};

struct LayerLoRAGrads {
    std::vector<float> gradAq;
    std::vector<float> gradBq;
    std::vector<float> gradAk;
    std::vector<float> gradBk;
    std::vector<float> gradAv;
    std::vector<float> gradBv;
    std::vector<float> gradAo;
    std::vector<float> gradBo;
    std::vector<float> gradAfd;
    std::vector<float> gradBfd;
    std::vector<float> gradAfg;
    std::vector<float> gradBfg;
    std::vector<float> gradAfu;
    std::vector<float> gradBfu;
    std::vector<float> gradMq;
    std::vector<float> gradMk;
    std::vector<float> gradMv;
    std::vector<float> gradMo;
    std::vector<float> gradMfd;
    std::vector<float> gradMfg;
    std::vector<float> gradMfu;
    std::vector<float> gradRmsAttn;
    std::vector<float> gradRmsFfn;
};

struct LmHeadLoRA {
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> M;
    std::vector<float> C;
    std::vector<float> R;
};

struct LmHeadLoRAGrads {
    std::vector<float> gradA;
    std::vector<float> gradB;
    std::vector<float> gradM;
};

enum class LoraDType : uint32_t {
    F32 = 0,
};
