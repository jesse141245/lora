#include "cpuKernelBack.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "attention.h"
#include "bitnet_i2s_matmul.h"
#include "bitnet_quant.h"
#include "bitnet_weight.h"
#include "kernels.h"
#include "lora_io.h"
#include "paramRef.h"

namespace {

    template <class... Args>
    concept Has_qkv_fwd_v2 = requires(Args... args) {
        linear_THH_fused_qkv_accum_blocked32_v2(args...);
    };

    template <class... Args>
    concept Has_qkv_fwd_opt = requires(Args... args) {
        linear_THH_fused_qkv_accum_blocked32_opt(args...);
    };

    template <class... Args>
    concept Has_qkv_fwd_new = requires(Args... args) {
        linear_THH_fused_qkv_accum_blocked32_new(args...);
    };

    template <class... Args>
    concept Has_qkv_fwd_old = requires(Args... args) {
        linear_THH_fused_qkv_accum_blocked32(args...);
    };

    template <class... Args>
    inline void qkv_fwd_dispatch(Args&&... args)
    {
        if constexpr (Has_qkv_fwd_v2<Args...>) {
            linear_THH_fused_qkv_accum_blocked32_v2(std::forward<Args>(args)...);
        }
        else if constexpr (Has_qkv_fwd_opt<Args...>) {
            linear_THH_fused_qkv_accum_blocked32_opt(std::forward<Args>(args)...);
        }
        else if constexpr (Has_qkv_fwd_new<Args...>) {
            linear_THH_fused_qkv_accum_blocked32_new(std::forward<Args>(args)...);
        }
        else if constexpr (Has_qkv_fwd_old<Args...>) {
            linear_THH_fused_qkv_accum_blocked32(std::forward<Args>(args)...);
        }
        else {
            static_assert(sizeof...(Args) == 0,
                "No compatible fused QKV forward kernel found.");
        }
    }

    template <class... Args>
    concept Has_linear_THH_blocked32 = requires(Args... args) {
        linear_THH_blocked32(args...);
    };

    template <class... Args>
    concept Has_linear_THH_accum_blocked32 = requires(Args... args) {
        linear_THH_accum_blocked32(args...);
    };

    template <class... Args>
    concept Has_linear_THH_old = requires(Args... args) {
        linear_THH(args...);
    };

    template <class... Args>
    inline void linear_THH_dispatch(Args&&... args)
    {
        if constexpr (Has_linear_THH_accum_blocked32<Args...>) {
            linear_THH_accum_blocked32(std::forward<Args>(args)...);
        }
        else if constexpr (Has_linear_THH_blocked32<Args...>) {
            linear_THH_blocked32(std::forward<Args>(args)...);
        }
        else if constexpr (Has_linear_THH_old<Args...>) {
            linear_THH(std::forward<Args>(args)...);
        }
        else {
            static_assert(sizeof...(Args) == 0,
                "No compatible linear_THH kernel found.");
        }
    }

    template <class... Args>
    concept Has_qkv_bwd_v2 = requires(Args... args) {
        qkv_input_backward_THH_v2(args...);
    };

    template <class... Args>
    concept Has_qkv_bwd_new = requires(Args... args) {
        qkv_input_backward_THH_blocked32(args...);
    };

    template <class... Args>
    concept Has_qkv_bwd_old = requires(Args... args) {
        qkv_input_backward_THH(args...);
    };

    template <class... Args>
    inline void qkv_bwd_dispatch(Args&&... args)
    {
        if constexpr (Has_qkv_bwd_v2<Args...>) {
            qkv_input_backward_THH_v2(std::forward<Args>(args)...);
        }
        else if constexpr (Has_qkv_bwd_new<Args...>) {
            qkv_input_backward_THH_blocked32(std::forward<Args>(args)...);
        }
        else if constexpr (Has_qkv_bwd_old<Args...>) {
            qkv_input_backward_THH(std::forward<Args>(args)...);
        }
        else {
            static_assert(sizeof...(Args) == 0,
                "No compatible QKV backward kernel found.");
        }
    }

    static inline std::size_t bitnet_bwd_iTile(std::size_t H)
    {
        (void)H;
        return 32;
    }

} // namespace

ModelShape CpuKernelBackend::loadShapeFromConfig(const std::string& modelPath)
{
    const std::string configFile = modelPath + "/config.json";

    std::ifstream in(configFile);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open config.json at " + configFile);
    }

    nlohmann::json j;
    in >> j;

    ModelShape shape{};
    shape.vocabSize = j.at("vocab_size").get<std::size_t>();
    shape.hiddenSize = j.at("hidden_size").get<std::size_t>();
    shape.nLayers = j.at("num_hidden_layers").get<std::size_t>();
    shape.nHeads = j.at("num_attention_heads").get<std::size_t>();
    return shape;
}

CpuKernelBackend::CpuKernelBackend(TrainerConfig cfg)
    : cfg_{ cfg }
{
    if (cfg_.modelPath.empty()) throw std::runtime_error("modelPath is empty");
    if (cfg_.loraRank == 0)     throw std::runtime_error("loraRank must be > 0");
    if (cfg_.vTile == 0)        throw std::runtime_error("vTile must be > 0");
    if (cfg_.maxSeqLen == 0)    throw std::runtime_error("maxSeqLen must be > 0");

    shape_ = loadShapeFromConfig(cfg_.modelPath);

    if (shape_.hiddenSize % shape_.nHeads != 0) {
        throw std::runtime_error("hiddenSize must be divisible by nHeads");
    }

    const std::size_t H = shape_.hiddenSize;
    const std::size_t V = shape_.vocabSize;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);
    const std::size_t Tm = static_cast<std::size_t>(cfg_.maxSeqLen);

    const std::size_t nLoRALayers = 2;

    embeddings_.resize(V * H);
    W_out_.resize(H * V);

    layers_.resize(nLoRALayers);
    loras_.resize(nLoRALayers);
    loraGrads_.resize(nLoRALayers);

    auto init_mat_HH = [&](std::vector<float>& W) { W.resize(H * H); };
    auto init_vec_H = [&](std::vector<float>& w) { w.resize(H); };

    for (std::size_t L = 0; L < nLoRALayers; ++L) {
        init_mat_HH(layers_[L].Wq);
        init_mat_HH(layers_[L].Wk);
        init_mat_HH(layers_[L].Wv);
        init_mat_HH(layers_[L].Wo);
        init_vec_H(layers_[L].rms_attn);

        loras_[L].Aq.resize(R * H);
        loras_[L].Bq.resize(R * H);
        loras_[L].Av.resize(R * H);
        loras_[L].Bv.resize(R * H);

        loraGrads_[L].gradAq.assign(R * H, 0.0f);
        loraGrads_[L].gradBq.assign(R * H, 0.0f);
        loraGrads_[L].gradAv.assign(R * H, 0.0f);
        loraGrads_[L].gradBv.assign(R * H, 0.0f);
    }

    tmpQ_cache_.resize(nLoRALayers * Tm * R);
    tmpV_cache_.resize(nLoRALayers * Tm * R);

    x_.resize(Tm * H);
    xnorm_.resize(Tm * H);
    q_.resize(Tm * H);
    k_.resize(Tm * H);
    v_.resize(Tm * H);
    ctx_.resize(Tm * H);
    attn_.resize(Tm * H);

    scores_.resize(Tm * Tm);
    probs_.resize(Tm * Tm);

    dTmpRank_.resize(R);

    dX_.resize(Tm * H);
    dAttn_.resize(Tm * H);
    dCtx_.resize(Tm * H);
    dQ_.resize(Tm * H);
    dK_.resize(Tm * H);
    dV_.resize(Tm * H);
    dXnorm_.resize(Tm * H);
    dXin_.resize(Tm * H);

    x_ckpt_.resize((nLoRALayers + 1) * Tm * H);
    tmpdP_.resize(Tm);

    std::mt19937 rng(42);
    std::normal_distribution<float> wdist(0.0f, 0.02f);

    for (auto& x : embeddings_) x = wdist(rng);
    for (auto& x : W_out_)      x = wdist(rng);

    for (std::size_t L = 0; L < nLoRALayers; ++L) {
        for (auto& x : layers_[L].Wq) x = wdist(rng);
        for (auto& x : layers_[L].Wk) x = wdist(rng);
        for (auto& x : layers_[L].Wv) x = wdist(rng);
        for (auto& x : layers_[L].Wo) x = wdist(rng);

        for (auto& x : layers_[L].rms_attn) x = 1.0f;

        if (useBitNetI2S()) {
            const std::size_t qk = bitnetI2SBlock();
            const std::size_t jq = 32;

            if (cfg_.bitnetQuantizeQKV) {
                quantize_ternary_i2s_blockwise(layers_[L].Wq_q, layers_[L].Wq.data(), H, qk, jq);
                quantize_ternary_i2s_blockwise(layers_[L].Wk_q, layers_[L].Wk.data(), H, qk, jq);
                quantize_ternary_i2s_blockwise(layers_[L].Wv_q, layers_[L].Wv.data(), H, qk, jq);

                layers_[L].Wq_q.bwd.ib = 32;
                layers_[L].Wq_q.bwd.jb = 32;
                build_bitnet_i2s_backward_assist(layers_[L].Wq_q);

                layers_[L].Wk_q.bwd.ib = 32;
                layers_[L].Wk_q.bwd.jb = 32;
                build_bitnet_i2s_backward_assist(layers_[L].Wk_q);

                layers_[L].Wv_q.bwd.ib = 32;
                layers_[L].Wv_q.bwd.jb = 32;
                build_bitnet_i2s_backward_assist(layers_[L].Wv_q);
            }

            if (cfg_.bitnetQuantizeWo) {
                quantize_ternary_i2s_blockwise(layers_[L].Wo_q, layers_[L].Wo.data(), H, qk, jq);

                layers_[L].Wo_q.bwd.ib = 32;
                layers_[L].Wo_q.bwd.jb = 32;
                build_bitnet_i2s_backward_assist(layers_[L].Wo_q);
            }
        }

        for (auto& x : loras_[L].Aq) x = wdist(rng) * 0.01f;
        for (auto& x : loras_[L].Av) x = wdist(rng) * 0.01f;

        std::fill(loras_[L].Bq.begin(), loras_[L].Bq.end(), 0.0f);
        std::fill(loras_[L].Bv.begin(), loras_[L].Bv.end(), 0.0f);
    }
}

void CpuKernelBackend::forward(
    std::span<const int> inputIds,
    std::span<const int> attentionMask,
    std::span<float> logitsOut)
{
    const std::size_t T = inputIds.size();
    if (attentionMask.size() != T) throw std::runtime_error("forward: attentionMask size mismatch");
    if (T > cfg_.maxSeqLen)        throw std::runtime_error("forward: seqLen exceeds maxSeqLen");

    const std::size_t H = shape_.hiddenSize;
    const std::size_t V = shape_.vocabSize;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);

    const std::size_t tTile = static_cast<std::size_t>(cfg_.tTile);
    const std::size_t hTile = static_cast<std::size_t>(cfg_.hTile);
    const float scale = cfg_.loraAlpha / float(R);

    const std::size_t CKPT_STRIDE = cfg_.maxSeqLen * H;

    for (std::size_t t = 0; t < T; ++t) {
        float* xt = &x_[t * H];

        if (attentionMask[t] == 0) {
            std::fill(xt, xt + H, 0.0f);
            continue;
        }

        const int tok = inputIds[t];
        if (tok < 0 || static_cast<std::size_t>(tok) >= V) {
            std::fill(xt, xt + H, 0.0f);
            continue;
        }

        const float* emb = &embeddings_[static_cast<std::size_t>(tok) * H];
        std::copy(emb, emb + H, xt);
    }

    std::copy(x_.begin(), x_.begin() + (T * H), &x_ckpt_[0]);

    const std::size_t nLayers = 2;
    const float eps = 1e-5f;

    for (std::size_t L = 0; L < nLayers; ++L) {
        for (std::size_t t = 0; t < T; ++t) {
            float* xn = &xnorm_[t * H];
            const float* x = &x_[t * H];

            if (attentionMask[t] == 0) {
                std::fill(xn, xn + H, 0.0f);
                continue;
            }

            rmsnorm(xn, x, layers_[L].rms_attn.data(), H, eps);
        }

        const bool useI2SQKV = canUseBitNetI2S(H) && cfg_.bitnetQuantizeQKV;
        const bool useI2SWo = canUseBitNetI2S(H) && cfg_.bitnetQuantizeWo;

        if (useI2SQKV) {
            std::fill(q_.begin(), q_.begin() + (T * H), 0.0f);
            std::fill(k_.begin(), k_.begin() + (T * H), 0.0f);
            std::fill(v_.begin(), v_.begin() + (T * H), 0.0f);

            linear_THH_fused_qkv_bitnet_i2s_accum(
                q_.data(), k_.data(), v_.data(),
                xnorm_.data(),
                layers_[L].Wq_q, layers_[L].Wk_q, layers_[L].Wv_q,
                T, H,
                tTile
            );
        }
        else {
            qkv_fwd_dispatch(
                q_.data(), k_.data(), v_.data(),
                xnorm_.data(),
                layers_[L].Wq.data(), layers_[L].Wk.data(), layers_[L].Wv.data(),
                T, H,
                hTile, tTile
            );
        }

        for (std::size_t t = 0; t < T; ++t) {
            float* tmpQ = &tmpQ_cache_[(L * cfg_.maxSeqLen + t) * R];
            float* tmpV = &tmpV_cache_[(L * cfg_.maxSeqLen + t) * R];

            if (attentionMask[t] == 0) {
                std::fill(tmpQ, tmpQ + R, 0.0f);
                std::fill(tmpV, tmpV + R, 0.0f);
                continue;
            }

            lora_add_H(
                &q_[t * H], &xnorm_[t * H],
                loras_[L].Aq.data(), loras_[L].Bq.data(),
                tmpQ, H, R, scale, hTile
            );

            lora_add_H(
                &v_[t * H], &xnorm_[t * H],
                loras_[L].Av.data(), loras_[L].Bv.data(),
                tmpV, H, R, scale, hTile
            );
        }

        attn_causal(
            ctx_.data(), q_.data(), k_.data(), v_.data(),
            T, H, shape_.nHeads, scores_.data(), probs_.data(),
            attentionMask.data()
        );

        if (useI2SWo) {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);

            linear_THH_bitnet_i2s_accum(
                attn_.data(),
                ctx_.data(),
                layers_[L].Wo_q,
                T, H,
                tTile
            );
        }
        else {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);

            linear_THH_dispatch(
                attn_.data(), ctx_.data(), layers_[L].Wo.data(), T, H, hTile
            );
        }

        for (std::size_t t = 0; t < T; ++t) {
            float* xt = &x_[t * H];
            const float* at = &attn_[t * H];
            for (std::size_t h = 0; h < H; ++h) {
                xt[h] += at[h];
            }
        }

        std::copy(x_.begin(), x_.begin() + (T * H), &x_ckpt_[(L + 1) * CKPT_STRIDE]);
    }

    const std::size_t expected = T * V;
    if (logitsOut.size() != expected) {
        throw std::runtime_error("forward: logitsOut size mismatch");
    }

    std::fill(logitsOut.begin(), logitsOut.end(), 0.0f);

    const std::size_t vTile = static_cast<std::size_t>(cfg_.vTile);

    for (std::size_t t = 0; t < T; ++t) {
        if (attentionMask[t] == 0) continue;

        const float* hidden = &x_[t * H];
        float* logits_t = &logitsOut[t * V];

        for (std::size_t v0 = 0; v0 < V; v0 += vTile) {
            const std::size_t v1 = std::min(V, v0 + vTile);
            const std::size_t len = v1 - v0;
            float* out = &logits_t[v0];

            for (std::size_t h0 = 0; h0 < H; h0 += hTile) {
                const std::size_t h1 = std::min(H, h0 + hTile);
                for (std::size_t hh = h0; hh < h1; ++hh) {
                    const float xh = hidden[hh];
                    const float* w = &W_out_[hh * V + v0];
                    for (std::size_t vv = 0; vv < len; ++vv) {
                        out[vv] += xh * w[vv];
                    }
                }
            }
        }
    }
}

void CpuKernelBackend::backward(
    std::span<const int> inputIds,
    std::span<const int> attentionMask,
    std::span<const float> dLogits)
{
    const std::size_t T = inputIds.size();
    if (attentionMask.size() != T) throw std::runtime_error("backward: attentionMask size mismatch");
    if (T > cfg_.maxSeqLen)        throw std::runtime_error("backward: seqLen exceeds maxSeqLen");

    (void)inputIds;

    const std::size_t H = shape_.hiddenSize;
    const std::size_t V = shape_.vocabSize;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);

    const std::size_t hTile = static_cast<std::size_t>(cfg_.hTile);
    const std::size_t tTile = static_cast<std::size_t>(cfg_.tTile);
    const float scale = cfg_.loraAlpha / float(R);

    const std::size_t nLayers = 2;
    const float eps = 1e-5f;
    const std::size_t CKPT_STRIDE = cfg_.maxSeqLen * H;

    std::fill(dX_.begin(), dX_.end(), 0.0f);

    for (std::size_t t = 0; t < T; ++t) {
        if (attentionMask[t] == 0) continue;

        const float* dlog_t = &dLogits[t * V];
        float* dx_t = &dX_[t * H];

        for (std::size_t h = 0; h < H; ++h) {
            const float* wrow = &W_out_[h * V];
            float acc = 0.0f;
            for (std::size_t v = 0; v < V; ++v) {
                acc += dlog_t[v] * wrow[v];
            }
            dx_t[h] += acc;
        }
    }

    for (std::size_t Lrev = 0; Lrev < nLayers; ++Lrev) {
        const std::size_t L = nLayers - 1 - Lrev;

        std::fill(dAttn_.begin(), dAttn_.end(), 0.0f);
        std::fill(dCtx_.begin(), dCtx_.end(), 0.0f);
        std::fill(dQ_.begin(), dQ_.end(), 0.0f);
        std::fill(dK_.begin(), dK_.end(), 0.0f);
        std::fill(dV_.begin(), dV_.end(), 0.0f);
        std::fill(dXnorm_.begin(), dXnorm_.end(), 0.0f);
        std::fill(dXin_.begin(), dXin_.end(), 0.0f);
        std::fill(tmpdP_.begin(), tmpdP_.end(), 0.0f);

        std::copy(
            &x_ckpt_[L * CKPT_STRIDE],
            &x_ckpt_[L * CKPT_STRIDE] + (T * H),
            x_.begin()
        );

        for (std::size_t t = 0; t < T; ++t) {
            float* xn = &xnorm_[t * H];
            const float* x = &x_[t * H];

            if (attentionMask[t] == 0) {
                std::fill(xn, xn + H, 0.0f);
                continue;
            }

            rmsnorm(xn, x, layers_[L].rms_attn.data(), H, eps);
        }

        const bool useI2SQKV = canUseBitNetI2S(H) && cfg_.bitnetQuantizeQKV;
        const bool useI2SWo = canUseBitNetI2S(H) && cfg_.bitnetQuantizeWo;

        if (useI2SQKV) {
            std::fill(q_.begin(), q_.begin() + (T * H), 0.0f);
            std::fill(k_.begin(), k_.begin() + (T * H), 0.0f);
            std::fill(v_.begin(), v_.begin() + (T * H), 0.0f);

            linear_THH_fused_qkv_bitnet_i2s_accum(
                q_.data(), k_.data(), v_.data(),
                xnorm_.data(),
                layers_[L].Wq_q, layers_[L].Wk_q, layers_[L].Wv_q,
                T, H,
                tTile
            );
        }
        else {
            qkv_fwd_dispatch(
                q_.data(), k_.data(), v_.data(),
                xnorm_.data(),
                layers_[L].Wq.data(), layers_[L].Wk.data(), layers_[L].Wv.data(),
                T, H,
                hTile, tTile
            );
        }

        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;

            const float* tmpQ = &tmpQ_cache_[((L * cfg_.maxSeqLen) + t) * R];
            const float* tmpV = &tmpV_cache_[((L * cfg_.maxSeqLen) + t) * R];

            const float* Bq = loras_[L].Bq.data();
            const float* Bv = loras_[L].Bv.data();

            float* q_t = &q_[t * H];
            float* v_t = &v_[t * H];

            for (std::size_t r = 0; r < R; ++r) {
                const float tq = tmpQ[r];
                const float tv = tmpV[r];

                const float* rowQ = &Bq[r * H];
                const float* rowV = &Bv[r * H];

                for (std::size_t h = 0; h < H; ++h) {
                    q_t[h] += scale * tq * rowQ[h];
                    v_t[h] += scale * tv * rowV[h];
                }
            }
        }

        attn_causal(
            ctx_.data(), q_.data(), k_.data(), v_.data(),
            T, H, shape_.nHeads, scores_.data(), probs_.data(),
            attentionMask.data()
        );

        if (useI2SWo) {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);

            linear_THH_bitnet_i2s_accum(
                attn_.data(), ctx_.data(), layers_[L].Wo_q, T, H, tTile
            );
        }
        else {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);

            linear_THH_dispatch(
                attn_.data(), ctx_.data(), layers_[L].Wo.data(), T, H, hTile
            );
        }

        for (std::size_t i = 0; i < T * H; ++i) {
            dAttn_[i] += dX_[i];
            dXin_[i] += dX_[i];
        }

        if (useI2SWo) {
            linear_HH_T_bitnet_i2s_assisted_block32(
                dCtx_.data(),
                dAttn_.data(),
                layers_[L].Wo_q,
                T, H,
                tTile,
                bitnet_bwd_iTile(H)
            );
        }
        else {
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                linear_HH_T_accum(
                    &dCtx_[t * H],
                    &dAttn_[t * H],
                    layers_[L].Wo.data(),
                    H,
                    hTile
                );
            }
        }

        attn_causal_backward(
            dQ_.data(), dK_.data(), dV_.data(),
            dCtx_.data(),
            q_.data(), k_.data(), v_.data(),
            T, H, shape_.nHeads, probs_.data(),
            attentionMask.data(), tmpdP_.data()
        );

        auto& gL = loraGrads_[L];

        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;

            const float* xnorm_t = &xnorm_[t * H];

            {
                const float* dy = &dQ_[t * H];
                const float* tmp = &tmpQ_cache_[((L * cfg_.maxSeqLen) + t) * R];
                const float* B = loras_[L].Bq.data();
                const float* A = loras_[L].Aq.data();

                for (std::size_t r = 0; r < R; ++r) {
                    const float tr = scale * tmp[r];
                    float* gBrow = &gL.gradBq[r * H];
                    const float* Brow = &B[r * H];

                    float acc = 0.0f;
                    for (std::size_t h = 0; h < H; ++h) {
                        const float dyh = dy[h];
                        gBrow[h] += tr * dyh;
                        acc += Brow[h] * dyh;
                    }
                    dTmpRank_[r] = scale * acc;
                }

                for (std::size_t r = 0; r < R; ++r) {
                    const float dtr = dTmpRank_[r];
                    float* gArow = &gL.gradAq[r * H];
                    for (std::size_t h = 0; h < H; ++h) {
                        gArow[h] += dtr * xnorm_t[h];
                    }
                }

                float* dxn = &dXnorm_[t * H];
                for (std::size_t r = 0; r < R; ++r) {
                    const float dr = dTmpRank_[r];
                    const float* Arow = &A[r * H];
                    for (std::size_t h = 0; h < H; ++h) {
                        dxn[h] += dr * Arow[h];
                    }
                }
            }

            {
                const float* dy = &dV_[t * H];
                const float* tmp = &tmpV_cache_[((L * cfg_.maxSeqLen) + t) * R];
                const float* B = loras_[L].Bv.data();
                const float* A = loras_[L].Av.data();

                for (std::size_t r = 0; r < R; ++r) {
                    const float tr = scale * tmp[r];
                    float* gBrow = &gL.gradBv[r * H];
                    const float* Brow = &B[r * H];

                    float acc = 0.0f;
                    for (std::size_t h = 0; h < H; ++h) {
                        const float dyh = dy[h];
                        gBrow[h] += tr * dyh;
                        acc += Brow[h] * dyh;
                    }
                    dTmpRank_[r] = scale * acc;
                }

                for (std::size_t r = 0; r < R; ++r) {
                    const float dtr = dTmpRank_[r];
                    float* gArow = &gL.gradAv[r * H];
                    for (std::size_t h = 0; h < H; ++h) {
                        gArow[h] += dtr * xnorm_t[h];
                    }
                }

                float* dxn = &dXnorm_[t * H];
                for (std::size_t r = 0; r < R; ++r) {
                    const float dr = dTmpRank_[r];
                    const float* Arow = &A[r * H];
                    for (std::size_t h = 0; h < H; ++h) {
                        dxn[h] += dr * Arow[h];
                    }
                }
            }
        }

        if (useI2SQKV) {
            qkv_input_backward_bitnet_i2s_assisted(
                dXnorm_.data(),
                dQ_.data(), dK_.data(), dV_.data(),
                layers_[L].Wq_q, layers_[L].Wk_q, layers_[L].Wv_q,
                T, H,
                tTile,
                bitnet_bwd_iTile(H)
            );
        }
        else {
            qkv_bwd_dispatch(
                dXnorm_.data(),
                dQ_.data(), dK_.data(), dV_.data(),
                layers_[L].Wq.data(), layers_[L].Wk.data(), layers_[L].Wv.data(),
                T, H,
                hTile
            );
        }

        for (std::size_t t = 0; t < T; ++t) {
            float* dxin = &dXin_[t * H];
            if (attentionMask[t] == 0) continue;

            rmsnorm_backward(
                dxin,
                &dXnorm_[t * H],
                &x_[t * H],
                layers_[L].rms_attn.data(),
                H, eps
            );
        }

        dX_.swap(dXin_);
    }
}

void CpuKernelBackend::zeroGrad()
{
    for (auto& g : loraGrads_) {
        std::fill(g.gradAq.begin(), g.gradAq.end(), 0.0f);
        std::fill(g.gradBq.begin(), g.gradBq.end(), 0.0f);
        std::fill(g.gradAv.begin(), g.gradAv.end(), 0.0f);
        std::fill(g.gradBv.begin(), g.gradBv.end(), 0.0f);
    }
}

void CpuKernelBackend::saveLoRA(const std::string& path) const
{
    const uint32_t H = static_cast<uint32_t>(shape_.hiddenSize);
    const uint32_t R = static_cast<uint32_t>(cfg_.loraRank);
    const uint32_t flags = HAS_AQ | HAS_BQ | HAS_AV | HAS_BV;

    saveLoRA_binary_v1(
        path,
        loras_,
        H,
        R,
        cfg_.loraAlpha,
        flags
    );
}

void CpuKernelBackend::loadLoRA(const std::string& path)
{
    const uint32_t expectedLayers = static_cast<uint32_t>(loras_.size());
    const uint32_t expectedH = static_cast<uint32_t>(shape_.hiddenSize);
    const uint32_t expectedR = static_cast<uint32_t>(cfg_.loraRank);

    float alpha_in_file = 0.0f;

    std::vector<LayerLoRA> loaded;
    loadLoRA_binary_v1(
        path,
        loaded,
        expectedLayers,
        expectedH,
        expectedR,
        &alpha_in_file
    );

    if (std::abs(alpha_in_file - cfg_.loraAlpha) > 1e-6f) {
        std::cerr
            << "[warn] LoRA alpha mismatch: file=" << alpha_in_file
            << " cfg=" << cfg_.loraAlpha
            << " (continuing; scaling uses cfg)\n";
    }

    loras_ = std::move(loaded);
}

std::vector<ParamRef> CpuKernelBackend::collectTrainableParams()
{
    if (loras_.size() != loraGrads_.size()) {
        throw std::runtime_error("collectTrainableParams: loras_ and loraGrads_ size mismatch");
    }

    std::vector<ParamRef> out;
    out.reserve(loras_.size() * 4);

    for (std::size_t l = 0; l < loras_.size(); ++l) {
        auto& P = loras_[l];
        auto& G = loraGrads_[l];

        if (P.Aq.size() != G.gradAq.size()) throw std::runtime_error("Aq grad size mismatch");
        if (P.Bq.size() != G.gradBq.size()) throw std::runtime_error("Bq grad size mismatch");
        if (P.Av.size() != G.gradAv.size()) throw std::runtime_error("Av grad size mismatch");
        if (P.Bv.size() != G.gradBv.size()) throw std::runtime_error("Bv grad size mismatch");

        out.push_back({ P.Aq.data(), G.gradAq.data(), P.Aq.size(), "Aq" });
        out.push_back({ P.Bq.data(), G.gradBq.data(), P.Bq.size(), "Bq" });
        out.push_back({ P.Av.data(), G.gradAv.data(), P.Av.size(), "Av" });
        out.push_back({ P.Bv.data(), G.gradBv.data(), P.Bv.size(), "Bv" });
    }

    return out;
}