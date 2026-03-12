#include "cpuKernelBack.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <cstring>

#include <nlohmann/json.hpp>

#include "attention.h"
#include "bitnet_i2s_matmul.h"
#include "bitnet_quant.h"
#include "bitnet_weight.h"
#include "kernels.h"
#include "paramRef.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

static void load_raw_bin(const std::string& filepath, std::vector<float>& buffer) {
    std::ifstream in(filepath, std::ios::binary | std::ios::ate);
    if (!in.is_open()) throw std::runtime_error("Cannot open weight file: " + filepath);

    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (size != buffer.size() * sizeof(float)) {
        throw std::runtime_error("Size mismatch in " + filepath + " expected: " + std::to_string(buffer.size() * sizeof(float)) + " got: " + std::to_string(size));
    }

    if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read " + filepath);
    }
}

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
    shape.nKvHeads = j.value("num_key_value_heads", shape.nHeads);
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

    const std::size_t kvDim = shape_.nKvHeads * (H / shape_.nHeads);
    const std::size_t nLoRALayers = shape_.nLayers;

    embeddings_.resize(V * H);
    W_out_.resize(H * V);

    layers_.resize(nLoRALayers);
    loras_.resize(nLoRALayers);
    loraGrads_.resize(nLoRALayers);

    auto init_mat_HH = [&](std::vector<float>& W) { W.resize(H * H); };
    auto init_mat_HKV = [&](std::vector<float>& W) { W.resize(H * kvDim); };
    auto init_vec_H = [&](std::vector<float>& w) { w.resize(H); };

    for (std::size_t L = 0; L < nLoRALayers; ++L) {
        init_mat_HH(layers_[L].Wq);
        init_mat_HKV(layers_[L].Wk);
        init_mat_HKV(layers_[L].Wv);
        init_mat_HH(layers_[L].Wo);
        init_vec_H(layers_[L].rms_attn);

        loras_[L].Aq.resize(R * H);
        loras_[L].Bq.resize(R * H);
        loras_[L].Av.resize(R * H);
        loras_[L].Bv.resize(R * kvDim);

        loraGrads_[L].gradAq.assign(R * H, 0.0f);
        loraGrads_[L].gradBq.assign(R * H, 0.0f);
        loraGrads_[L].gradAv.assign(R * H, 0.0f);
        loraGrads_[L].gradBv.assign(R * kvDim, 0.0f);
    }

    tmpQ_cache_.resize(nLoRALayers * Tm * R);
    tmpV_cache_.resize(nLoRALayers * Tm * R);

    x_.resize(Tm * H);
    xnorm_.resize(Tm * H);
    q_.resize(Tm * H);
    k_.resize(Tm * kvDim);
    v_.resize(Tm * kvDim);
    ctx_.resize(Tm * H);
    attn_.resize(Tm * H);

    scores_.resize(Tm * Tm);
    probs_.resize(Tm * Tm);

    dTmpRank_.resize(R);

    dX_.resize(Tm * H);
    dAttn_.resize(Tm * H);
    dCtx_.resize(Tm * H);
    dQ_.resize(Tm * H);
    dK_.resize(Tm * kvDim);
    dV_.resize(Tm * kvDim);
    dXnorm_.resize(Tm * H);
    dXin_.resize(Tm * H);

    rms_final_.resize(H);
    xnorm_final_.resize(Tm * H);
    dXnorm_final_.resize(Tm * H);

    x_ckpt_.resize((nLoRALayers + 1) * Tm * H);
    tmpdP_.resize(Tm);

    std::mt19937 rng(42);
    std::normal_distribution<float> wdist(0.0f, 0.02f);

    std::string weight_dir = cfg_.modelPath + "/weights";

    std::cout << "Loading real FP32 weights from disk...\n";
    std::cout << "Architecture: " << nLoRALayers << " Layers | H=" << H << " | kvDim=" << kvDim << "\n";

    load_raw_bin(weight_dir + "/embeddings.bin", embeddings_);
    load_raw_bin(weight_dir + "/W_out.bin", W_out_);
    load_raw_bin(weight_dir + "/rms_final.bin", rms_final_);

    for (std::size_t L = 0; L < nLoRALayers; ++L)
    {
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wq.bin", layers_[L].Wq);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wk.bin", layers_[L].Wk);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wv.bin", layers_[L].Wv);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_Wo.bin", layers_[L].Wo);
        load_raw_bin(weight_dir + "/layer_" + std::to_string(L) + "_rms_attn.bin", layers_[L].rms_attn);

        if (cfg_.useBitNetQuant) {
            const std::size_t qk = cfg_.bitnetQK;
            const std::size_t jq = cfg_.bitnetJQ;

            if (cfg_.bitnetQuantizeQKV) {
                quantize_ternary_i2s_blockwise(layers_[L].Wq_q, layers_[L].Wq.data(), H, H, qk, jq);
                quantize_ternary_i2s_blockwise(layers_[L].Wk_q, layers_[L].Wk.data(), H, kvDim, qk, jq);
                quantize_ternary_i2s_blockwise(layers_[L].Wv_q, layers_[L].Wv.data(), H, kvDim, qk, jq);
            }
            if (cfg_.bitnetQuantizeWo) {
                quantize_ternary_i2s_blockwise(layers_[L].Wo_q, layers_[L].Wo.data(), H, H, qk, jq);
            }
        }

        // Small Random Initialization for Lion Optimizer stability
        for (auto& x : loras_[L].Aq) x = wdist(rng) * 0.01f;
        for (auto& x : loras_[L].Av) x = wdist(rng) * 0.01f;
        for (auto& x : loras_[L].Bq) x = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
        for (auto& x : loras_[L].Bv) x = ((float)rand() / RAND_MAX - 0.5f) * 0.001f;
    }

    if (cfg_.useBitNetQuant && cfg_.bitnetQuantizeWo) {
        const std::size_t qk = cfg_.bitnetQK;
        const std::size_t jq = cfg_.bitnetJQ;
        quantize_ternary_i2s_blockwise(W_out_q, W_out_.data(), H, V, qk, jq);
    }
}

// ==============================================================================
// FORWARD
// ==============================================================================
void CpuKernelBackend::forward(
    std::span<const int>   inputIds,
    std::span<const int>   attentionMask,
    std::span<float>       logitsOut)
{
    const std::size_t T = inputIds.size();
    if (attentionMask.size() != T) throw std::runtime_error("forward: attentionMask size mismatch");
    if (T > cfg_.maxSeqLen)        throw std::runtime_error("forward: seqLen exceeds maxSeqLen");

    const std::size_t H = shape_.hiddenSize;
    const std::size_t V = shape_.vocabSize;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);
    const std::size_t kvDim = layers_[0].Wk.size() / H;
    const std::size_t nKvHeads = shape_.nKvHeads;
    const std::size_t tTile = static_cast<std::size_t>(cfg_.tTile);
    const std::size_t hTile = static_cast<std::size_t>(cfg_.hTile);
    const float       scale = cfg_.loraAlpha / float(R);
    const std::size_t CKPT_STRIDE = cfg_.maxSeqLen * H;

    for (std::size_t t = 0; t < T; ++t) {
        float* xt = &x_[t * H];
        if (attentionMask[t] == 0) { std::fill(xt, xt + H, 0.0f); continue; }

        const int tok = inputIds[t];
        if (tok < 0 || static_cast<std::size_t>(tok) >= V) { std::fill(xt, xt + H, 0.0f); continue; }

        const float* emb = &embeddings_[static_cast<std::size_t>(tok) * H];
        std::copy(emb, emb + H, xt);
    }

    std::copy(x_.begin(), x_.begin() + (T * H), &x_ckpt_[0]);

    const std::size_t nLayers = shape_.nLayers;
    const float eps = 1e-5f;

    for (std::size_t L = 0; L < nLayers; ++L) {
        for (std::size_t t = 0; t < T; ++t) {
            float* xn = &xnorm_[t * H];
            const float* x = &x_[t * H];
            if (attentionMask[t] == 0) { std::fill(xn, xn + H, 0.0f); continue; }
            rmsnorm(xn, x, layers_[L].rms_attn.data(), H, eps);
        }

        const bool useI2SQKV = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV;
        const bool useI2SWo = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeWo;

        // ── QKV Projection ──
        if (useI2SQKV) {
            std::fill(q_.begin(), q_.begin() + (T * H), 0.0f);
            std::fill(k_.begin(), k_.begin() + (T * kvDim), 0.0f);
            std::fill(v_.begin(), v_.begin() + (T * kvDim), 0.0f);

            // Native Microsoft I2S GQA-safe functions
            std::vector<int8_t> x8Scratch(tTile * H);
            std::vector<float>  xScaleScratch(tTile * layers_[L].Wq_q.nKBlocks());
            linear_forward_bitnet_i2s_accum(q_.data(), xnorm_.data(), layers_[L].Wq_q, T, tTile, x8Scratch.data(), xScaleScratch.data());
            linear_forward_bitnet_i2s_accum(k_.data(), xnorm_.data(), layers_[L].Wk_q, T, tTile, x8Scratch.data(), xScaleScratch.data());
            linear_forward_bitnet_i2s_accum(v_.data(), xnorm_.data(), layers_[L].Wv_q, T, tTile, x8Scratch.data(), xScaleScratch.data());
        }
        else {
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                const float* xn = &xnorm_[t * H];
                float* qt = &q_[t * H]; float* kt = &k_[t * kvDim]; float* vt = &v_[t * kvDim];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum_q = 0; for (std::size_t j = 0; j < H; ++j) sum_q += xn[j] * layers_[L].Wq[j * H + i]; qt[i] = sum_q;
                }
                for (std::size_t i = 0; i < kvDim; ++i) {
                    float sum_k = 0; for (std::size_t j = 0; j < H; ++j) sum_k += xn[j] * layers_[L].Wk[j * kvDim + i]; kt[i] = sum_k;
                    float sum_v = 0; for (std::size_t j = 0; j < H; ++j) sum_v += xn[j] * layers_[L].Wv[j * kvDim + i]; vt[i] = sum_v;
                }
            }
        }

        // ── LoRA Injection ──
        for (std::size_t t = 0; t < T; ++t) {
            float* tmpQ = &tmpQ_cache_[(L * cfg_.maxSeqLen + t) * R];
            float* tmpV = &tmpV_cache_[(L * cfg_.maxSeqLen + t) * R];

            if (attentionMask[t] == 0) {
                std::fill(tmpQ, tmpQ + R, 0.0f);
                std::fill(tmpV, tmpV + R, 0.0f);
                continue;
            }

            lora_add_H(&q_[t * H], &xnorm_[t * H], loras_[L].Aq.data(), loras_[L].Bq.data(), tmpQ, H, R, scale, hTile);

            float* v_t = &v_[t * kvDim];
            const float* xn = &xnorm_[t * H];

            for (std::size_t r = 0; r < R; ++r) {
                float acc = 0.0f;
                const float* aRow = &loras_[L].Av[r * H];
                for (std::size_t d = 0; d < H; ++d) acc += xn[d] * aRow[d];
                tmpV[r] = acc;
            }
            for (std::size_t r = 0; r < R; ++r) {
                const float tv = tmpV[r];
                const float* bRow = &loras_[L].Bv[r * kvDim];
                for (std::size_t d = 0; d < kvDim; ++d) v_t[d] += scale * tv * bRow[d];
            }
        }

        // ── Attention ──
        attn_causal(
            ctx_.data(), q_.data(), k_.data(), v_.data(),
            T, H, shape_.nHeads, nKvHeads, scores_.data(), probs_.data(),
            attentionMask.data()
        );

        // ── Output Projection (Wo) ──
        if (useI2SWo) {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);
            const std::size_t nXBlocks = layers_[L].Wo_q.nKBlocks();
            std::vector<int8_t> x8Scratch(tTile * H);
            std::vector<float>  xScaleScratch(tTile * nXBlocks);
            linear_forward_bitnet_i2s_accum(attn_.data(), ctx_.data(), layers_[L].Wo_q, T, tTile, x8Scratch.data(), xScaleScratch.data());
        }
        else {
            std::fill(attn_.begin(), attn_.begin() + (T * H), 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                const float* ct = &ctx_[t * H]; float* at = &attn_[t * H];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum = 0; for (std::size_t j = 0; j < H; ++j) sum += ct[j] * layers_[L].Wo[j * H + i]; at[i] = sum;
                }
            }
        }

        for (std::size_t t = 0; t < T; ++t) {
            float* xt = &x_[t * H];
            const float* at = &attn_[t * H];
            for (std::size_t h = 0; h < H; ++h) xt[h] += at[h];
        }

        std::copy(x_.begin(), x_.begin() + (T * H), &x_ckpt_[(L + 1) * CKPT_STRIDE]);
    }

    const std::size_t expected = T * V;
    if (logitsOut.size() != expected) throw std::runtime_error("forward: logitsOut size mismatch");
    std::fill(logitsOut.begin(), logitsOut.end(), 0.0f);

    for (std::size_t t = 0; t < T; ++t) {
        if (attentionMask[t] == 0) { std::fill(&xnorm_final_[t * H], &xnorm_final_[t * H] + H, 0.0f); continue; }
        rmsnorm(&xnorm_final_[t * H], &x_[t * H], rms_final_.data(), H, eps);
    }

    if (cfg_.useBitNetQuant && !W_out_q.empty()) {
        const std::size_t nXBlocks = W_out_q.nKBlocks();
        std::vector<int8_t> x8Scratch(tTile * H);
        std::vector<float>  xScaleScratch(tTile * nXBlocks);
        linear_forward_bitnet_i2s_accum(logitsOut.data(), xnorm_final_.data(), W_out_q, T, tTile, x8Scratch.data(), xScaleScratch.data());
    }
    else {
        const std::size_t vTile = static_cast<std::size_t>(cfg_.vTile);
#if defined(_OPENMP)
#pragma warning(suppress : 6993)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
            std::size_t t = static_cast<std::size_t>(t_i);
            if (attentionMask[t] == 0) continue;

            const float* hidden = &xnorm_final_[t * H];
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
                        for (std::size_t vv = 0; vv < len; ++vv) out[vv] += xh * w[vv];
                    }
                }
            }
        }
    }
}

// ==============================================================================
// BACKWARD
// ==============================================================================
void CpuKernelBackend::backward(
    std::span<const int>   inputIds,
    std::span<const int>   attentionMask,
    std::span<const float> dLogits)
{
    const std::size_t T = inputIds.size();
    if (attentionMask.size() != T) throw std::runtime_error("backward: attentionMask size mismatch");
    if (T > cfg_.maxSeqLen)        throw std::runtime_error("backward: seqLen exceeds maxSeqLen");

    (void)inputIds;

    const std::size_t H = shape_.hiddenSize;
    const std::size_t V = shape_.vocabSize;
    const std::size_t R = static_cast<std::size_t>(cfg_.loraRank);
    const std::size_t kvDim = layers_[0].Wk.size() / H;
    const std::size_t nKvHeads = shape_.nKvHeads;
    const std::size_t hTile = static_cast<std::size_t>(cfg_.hTile);
    const std::size_t tTile = static_cast<std::size_t>(cfg_.tTile);
    const float       scale = cfg_.loraAlpha / float(R);
    const std::size_t nLayers = shape_.nLayers;
    const float       eps = 1e-5f;
    const std::size_t CKPT_STRIDE = cfg_.maxSeqLen * H;

    // ── Backprop through LM head ──────────────────────────────────────────────
    std::fill(dXnorm_final_.begin(), dXnorm_final_.end(), 0.0f);

    if (cfg_.useBitNetQuant && !W_out_q.empty()) {
        const std::size_t nYBlocks = W_out_q.nJBlocks();
        std::vector<int8_t> dy8Scratch(tTile * V);
        std::vector<float>  dyScaleScratch(tTile * nYBlocks);
        linear_backward_bitnet_i2s_packed(dXnorm_final_.data(), dLogits.data(), W_out_q, T, tTile, dy8Scratch.data(), dyScaleScratch.data());
    }
    else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
        for (int t_i = 0; t_i < static_cast<int>(T); ++t_i) {
            std::size_t t = static_cast<std::size_t>(t_i);
            if (attentionMask[t] == 0) continue;
            const float* dlog_t = &dLogits[t * V];
            float* dx_t = &dXnorm_final_[t * H];
            for (std::size_t h = 0; h < H; ++h) {
                const float* wrow = &W_out_[h * V];
                float acc = 0.0f;
                for (std::size_t v = 0; v < V; ++v) acc += dlog_t[v] * wrow[v];
                dx_t[h] += acc;
            }
        }
    }

    // ── Backprop through final RMSNorm ────────────────────────────────────────
    std::fill(dX_.begin(), dX_.end(), 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        if (attentionMask[t] == 0) continue;
        rmsnorm_backward(&dX_[t * H], &dXnorm_final_[t * H], &x_[t * H], rms_final_.data(), H, eps);
    }

    // ── Layer loop (reverse) ─────────────────────────────────────────────────
    for (std::size_t Lrev = 0; Lrev < nLayers; ++Lrev) {
        const std::size_t L = nLayers - 1 - Lrev;

        // Clear all per-layer temporaries.
        std::fill(dAttn_.begin(), dAttn_.end(), 0.0f);
        std::fill(dCtx_.begin(), dCtx_.end(), 0.0f);
        std::fill(dQ_.begin(), dQ_.end(), 0.0f);
        std::fill(dK_.begin(), dK_.end(), 0.0f);
        std::fill(dV_.begin(), dV_.end(), 0.0f);
        std::fill(dXnorm_.begin(), dXnorm_.end(), 0.0f);
        std::fill(dXin_.begin(), dXin_.end(), 0.0f);
        std::fill(tmpdP_.begin(), tmpdP_.end(), 0.0f);

        // Restore the pre-layer activation from the checkpoint.
        std::copy(&x_ckpt_[L * CKPT_STRIDE], &x_ckpt_[L * CKPT_STRIDE] + (T * H), x_.begin());

        // Re-run forward norms/projections to rebuild activations needed for backward.
        for (std::size_t t = 0; t < T; ++t) {
            float* xn = &xnorm_[t * H];
            const float* x = &x_[t * H];
            if (attentionMask[t] == 0) { std::fill(xn, xn + H, 0.0f); continue; }
            rmsnorm(xn, x, layers_[L].rms_attn.data(), H, eps);
        }

        const bool useI2SQKV = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeQKV;
        const bool useI2SWo = cfg_.useBitNetQuant && (H % 32 == 0) && cfg_.bitnetQuantizeWo;

        if (useI2SQKV) {
            std::fill(q_.begin(), q_.begin() + (T * H), 0.0f);
            std::fill(k_.begin(), k_.begin() + (T * kvDim), 0.0f);
            std::fill(v_.begin(), v_.begin() + (T * kvDim), 0.0f);

            std::vector<int8_t> x8Scratch(tTile * H);
            std::vector<float>  xScaleScratch(tTile * layers_[L].Wq_q.nKBlocks());
            linear_forward_bitnet_i2s_accum(q_.data(), xnorm_.data(), layers_[L].Wq_q, T, tTile, x8Scratch.data(), xScaleScratch.data());
            linear_forward_bitnet_i2s_accum(k_.data(), xnorm_.data(), layers_[L].Wk_q, T, tTile, x8Scratch.data(), xScaleScratch.data());
            linear_forward_bitnet_i2s_accum(v_.data(), xnorm_.data(), layers_[L].Wv_q, T, tTile, x8Scratch.data(), xScaleScratch.data());
        }
        else {
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                const float* xn = &xnorm_[t * H];
                float* qt = &q_[t * H]; float* kt = &k_[t * kvDim]; float* vt = &v_[t * kvDim];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum_q = 0; for (std::size_t j = 0; j < H; ++j) sum_q += xn[j] * layers_[L].Wq[j * H + i]; qt[i] = sum_q;
                }
                for (std::size_t i = 0; i < kvDim; ++i) {
                    float sum_k = 0; for (std::size_t j = 0; j < H; ++j) sum_k += xn[j] * layers_[L].Wk[j * kvDim + i]; kt[i] = sum_k;
                    float sum_v = 0; for (std::size_t j = 0; j < H; ++j) sum_v += xn[j] * layers_[L].Wv[j * kvDim + i]; vt[i] = sum_v;
                }
            }
        }

        // Rebuild LoRA contributions (needed for correct cached intermediate values).
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;

            const float* tmpQ = &tmpQ_cache_[((L * cfg_.maxSeqLen) + t) * R];
            const float* tmpV = &tmpV_cache_[((L * cfg_.maxSeqLen) + t) * R];
            const float* Bq = loras_[L].Bq.data();
            const float* Bv = loras_[L].Bv.data();
            float* q_t = &q_[t * H];
            float* v_t = &v_[t * kvDim];

            for (std::size_t r = 0; r < R; ++r) {
                const float tq = tmpQ[r]; const float tv = tmpV[r];
                const float* rowQ = &Bq[r * H];  const float* rowV = &Bv[r * kvDim];
                for (std::size_t h = 0; h < H; ++h) q_t[h] += scale * tq * rowQ[h];
                for (std::size_t d = 0; d < kvDim; ++d) v_t[d] += scale * tv * rowV[d];
            }
        }

        attn_causal(
            ctx_.data(), q_.data(), k_.data(), v_.data(),
            T, H, shape_.nHeads, nKvHeads, scores_.data(), probs_.data(),
            attentionMask.data()
        );

        // ── Residual split: dAttn gets the attention branch, dX_ carries the skip ──
        for (std::size_t i = 0; i < T * H; ++i) {
            dAttn_[i] = dX_[i];   // attention branch gradient
        }

        if (useI2SWo) {
            const std::size_t nYBlocks = layers_[L].Wo_q.nJBlocks();
            std::vector<int8_t> dy8Scratch(tTile * H);
            std::vector<float>  dyScaleScratch(tTile * nYBlocks);
            linear_backward_bitnet_i2s_packed(dCtx_.data(), dAttn_.data(), layers_[L].Wo_q, T, tTile, dy8Scratch.data(), dyScaleScratch.data());
        }
        else {
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                const float* dat = &dAttn_[t * H]; float* dct = &dCtx_[t * H];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum = 0; for (std::size_t j = 0; j < H; ++j) sum += dat[j] * layers_[L].Wo[i * H + j]; dct[i] += sum;
                }
            }
        }

        attn_causal_backward(
            dQ_.data(), dK_.data(), dV_.data(),
            dCtx_.data(),
            q_.data(), k_.data(), v_.data(),
            T, H, shape_.nHeads, nKvHeads, probs_.data(),
            attentionMask.data(), tmpdP_.data()
        );

        // ── LoRA gradient accumulation ────────────────────────────────────────
        auto& gL = loraGrads_[L];

        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;

            const float* xnorm_t = &xnorm_[t * H];

            // Q LoRA grads
            {
                const float* dy = &dQ_[t * H];
                const float* tmp = &tmpQ_cache_[((L * cfg_.maxSeqLen) + t) * R];
                const float* B = loras_[L].Bq.data();
                const float* A = loras_[L].Aq.data();

                for (std::size_t r = 0; r < R; ++r) {
                    const float  tr = scale * tmp[r];
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
                    const float  dtr = dTmpRank_[r];
                    float* gArow = &gL.gradAq[r * H];
                    for (std::size_t h = 0; h < H; ++h) gArow[h] += dtr * xnorm_t[h];
                }

                float* dxn = &dXnorm_[t * H];
                for (std::size_t r = 0; r < R; ++r) {
                    const float  dr = dTmpRank_[r];
                    const float* Arow = &A[r * H];
                    for (std::size_t h = 0; h < H; ++h) dxn[h] += dr * Arow[h];
                }
            }

            // V LoRA grads
            {
                const float* dyV = &dV_[t * kvDim];
                const float* tmpV = &tmpV_cache_[((L * cfg_.maxSeqLen) + t) * R];
                const float* Bv = loras_[L].Bv.data();
                const float* Av = loras_[L].Av.data();

                for (std::size_t r = 0; r < R; ++r) {
                    const float  tr = scale * tmpV[r];
                    float* gBrow = &gL.gradBv[r * kvDim];
                    const float* Brow = &Bv[r * kvDim];
                    float acc = 0.0f;
                    for (std::size_t d = 0; d < kvDim; ++d) {
                        const float dyh = dyV[d];
                        gBrow[d] += tr * dyh;
                        acc += Brow[d] * dyh;
                    }
                    dTmpRank_[r] = scale * acc;
                }

                for (std::size_t r = 0; r < R; ++r) {
                    const float  dtr = dTmpRank_[r];
                    float* gArow = &gL.gradAv[r * H];
                    for (std::size_t h = 0; h < H; ++h) gArow[h] += dtr * xnorm_t[h];
                }

                float* dxn = &dXnorm_[t * H];
                for (std::size_t r = 0; r < R; ++r) {
                    const float  dr = dTmpRank_[r];
                    const float* Arow = &Av[r * H];
                    for (std::size_t h = 0; h < H; ++h) dxn[h] += dr * Arow[h];
                }
            }
        }

        // ── Base weight backward into dXnorm_ ────────────────────────────────
        if (useI2SQKV) {
            std::vector<int8_t> dq8Scratch(tTile * H);
            std::vector<float>  dqScaleScratch(tTile * layers_[L].Wq_q.nJBlocks());
            linear_backward_bitnet_i2s_packed(dXnorm_.data(), dQ_.data(), layers_[L].Wq_q, T, tTile, dq8Scratch.data(), dqScaleScratch.data());

            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                float* dxn = &dXnorm_[t * H];
                const float* dkt = &dK_[t * kvDim];
                const float* dvt = &dV_[t * kvDim];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum = 0.0f;
                    // Fallback to FP32 shadow weights for Key and Value backward accumulation
                    for (std::size_t j = 0; j < kvDim; ++j) {
                        sum += dkt[j] * layers_[L].Wk[i * kvDim + j] + dvt[j] * layers_[L].Wv[i * kvDim + j];
                    }
                    dxn[i] += sum;
                }
            }
        }
        else {
            for (std::size_t t = 0; t < T; ++t) {
                if (attentionMask[t] == 0) continue;
                float* dxn = &dXnorm_[t * H];
                const float* dqt = &dQ_[t * H]; const float* dkt = &dK_[t * kvDim]; const float* dvt = &dV_[t * kvDim];
                for (std::size_t i = 0; i < H; ++i) {
                    float sum = 0;
                    for (std::size_t j = 0; j < H; ++j) sum += dqt[j] * layers_[L].Wq[i * H + j];
                    for (std::size_t j = 0; j < kvDim; ++j) sum += dkt[j] * layers_[L].Wk[i * kvDim + j] + dvt[j] * layers_[L].Wv[i * kvDim + j];
                    dxn[i] += sum;
                }
            }
        }

        // ── FIX: Correct residual gradient accumulation ───────────────────────
        for (std::size_t t = 0; t < T; ++t) {
            if (attentionMask[t] == 0) continue;
            // Step 1: compute the RMSNorm gradient into the zero-initialised dXin_
            rmsnorm_backward(
                &dXin_[t * H],
                &dXnorm_[t * H],
                &x_[t * H],
                layers_[L].rms_attn.data(),
                H, eps
            );
        }

        // Step 2: accumulate the layer gradient into the residual stream
        for (std::size_t i = 0; i < T * H; ++i) {
            dX_[i] += dXin_[i];
        }
    }
}

// ==============================================================================
// UTILITIES
// ==============================================================================
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
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write to " + path);

    const uint32_t H = static_cast<uint32_t>(shape_.hiddenSize);
    const uint32_t R = static_cast<uint32_t>(cfg_.loraRank);
    const uint32_t flags = 5;
    const uint32_t nL = static_cast<uint32_t>(loras_.size());
    const float    alpha = cfg_.loraAlpha;

    out.write("LORA", 4);
    uint32_t version = 1, dtype = 0;
    out.write((char*)&version, 4);
    out.write((char*)&dtype, 4);
    out.write((char*)&flags, 4);
    out.write((char*)&nL, 4);
    out.write((char*)&H, 4);
    out.write((char*)&R, 4);
    out.write((char*)&alpha, 4);

    const std::size_t kvDim = layers_[0].Wk.size() / H;
    const uint32_t    kvD = static_cast<uint32_t>(kvDim);

    for (std::size_t L = 0; L < nL; ++L) {
        out.write((const char*)loras_[L].Aq.data(), R * H * sizeof(float));
        out.write((const char*)loras_[L].Bq.data(), H * R * sizeof(float));
        out.write((const char*)loras_[L].Av.data(), R * H * sizeof(float));
        out.write((const char*)loras_[L].Bv.data(), kvD * R * sizeof(float));
    }
}

void CpuKernelBackend::loadLoRA(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return;

    char magic[4]; in.read(magic, 4);
    if (std::strncmp(magic, "LORA", 4) != 0) return;

    uint32_t version, dtype, flags, nL, H, R; float alpha;
    in.read((char*)&version, 4); in.read((char*)&dtype, 4);
    in.read((char*)&flags, 4); in.read((char*)&nL, 4);
    in.read((char*)&H, 4); in.read((char*)&R, 4);
    in.read((char*)&alpha, 4);

    const std::size_t kvDim = layers_[0].Wk.size() / H;

    for (std::size_t L = 0; L < nL; ++L) {
        in.read((char*)loras_[L].Aq.data(), R * H * sizeof(float));
        in.read((char*)loras_[L].Bq.data(), H * R * sizeof(float));
        in.read((char*)loras_[L].Av.data(), R * H * sizeof(float));
        in.read((char*)loras_[L].Bv.data(), kvDim * R * sizeof(float));
    }
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