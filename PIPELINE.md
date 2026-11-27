# BitNet QLoRA CPU training pipeline

This document describes how configuration becomes tensor shapes, how `CpuKernelBackend` runs one forward/backward, and how `CpuQLoRATrainer` orchestrates the training loop. It reflects the C++ trainer in this repository (`main.cpp`, `qlora_cpu.cpp`, `cpuKernelBack.cpp`, `config.h`).

## 1. Configuration sources

### 1.1 `TrainerConfig` (`config.h`)

Runtime training controls: paths, loop (`maxSteps`, `batchSize`, `maxSeqLen`, `gradAccumSteps`), optimizer choice (`useAdamW` vs Lion), LR schedule, LoRA rank/alpha/targets, tiling (`hTile`, `tTile`, `vTile`, `iTile`, `rTile`), BitNet flags (`useBitNetQuant`, `bitnetQuantizeQKV`, `bitnetFuseQKV`, TL2 toggles, `bitnetQK`, `bitnetFfnQK`, `bitnetJQ`, `bitnetBwdIB`, `bitnetBwdJB`, …), logging, checkpoints, and optional `forcedLayers` (truncate the transformer to the first *N* layers after load).

Defaults in the struct can be overridden in `main()` (and via environment variables where `main` wires them).

### 1.2 Model geometry: `ModelShape` (`cpuKernelBack.h`)

**Source of truth** for width/depth is `modelPath/config.json`, parsed in `CpuKernelBackend::loadShapeFromConfig` (`cpuKernelBack.cpp`):

| JSON field | `ModelShape` member |
|------------|---------------------|
| `vocab_size` | `vocabSize` |
| `hidden_size` | `hiddenSize` |
| `num_hidden_layers` | `nLayers` (may be reduced by `cfg.forcedLayers`) |
| `num_attention_heads` | `nHeads` |
| `num_key_value_heads` (default: `nHeads`) | `nKvHeads` |
| `intermediate_size` (default: `hidden_size * 4`) | `ffnSize` |
| `rope_theta` (default `10000`) | `ropeTheta` |
| `rope_dim` (default `hidden_size / num_attention_heads`) | `ropeDim` |

Derived quantities used everywhere in kernels:

- **Head dim** `D = headDim() = hiddenSize / nHeads`
- **KV projection width** `kvDim = nKvHeads * D` (GQA)

`TrainerConfig`’s `vocabSize` / `hiddenSize` / `nLayers` / `nHeads` fields are documented as logging/UI only; backend math uses `shape_` from JSON.

### 1.3 Dataset sizing (`dataset.cpp`)

`Dataset(path, cfg)` normalizes every sample to **exactly** `cfg.maxSeqLen` tokens (truncate or pad with `ignoreIndex` / zero mask). That matches buffer sizing in the backend, which caps sequence length at `cfg.maxSeqLen`.

### 1.4 Runtime policy snapshot (`backend_runtime_policy*.`)

Each `forward` / `backward` builds a `BackendRuntimePolicy` from active sequence length and `tTile` (`buildBackendRuntimePolicy` in `backend_runtime_policy_builder.cpp`). This selects matmul variants (fused QKV, overwrite paths, LM-head row compaction, FFN streaming chunks, etc.) and is recorded for autotune logs (`*.runtime_policy.json`).

---

## 2. Tensor and kernel “shapes”

Symbols below: **T** = actual tokens this step (from attention mask prefix, ≤ `maxSeqLen`), **H** = `hiddenSize`, **V** = `vocabSize`, **F** = `ffnSize`, **D** = `headDim`, **R** = `loraRank`, **kvDim** as above.

### 2.1 Major activations (per forward)

| Buffer role | Typical layout | Notes |
|-------------|----------------|--------|
| Token embeddings | `x_` : **T × H** | From `embeddings_[tok * H]` |
| Per-layer residual stream | same `x_` updated each layer | Checkpoint strip `x_ckpt_[(L+1)*Tm*H]` |
| Attn pre-norm | `xnorm_` or per-layer cache **T × H** | RMSNorm on `x_` |
| Q / K / V projections | **T × H**, **T × kvDim**, **T × kvDim** | RoPE applied in-place on Q and K |
| Attention output / pre-Wo | `ctx_` **T × H** | `attn_causal` (`attention.cpp`) |
| Post-Wo / pre-FFN residual branch | `attn_` then `x_mid_` **T × H** | `x_mid = x + attn` |
| FFN pre-norm | `xnorm_ffn_` **T × H** | |
| Gate / SiLU path | `gate_`, `up_`, `ffn_act_` **T × F** | SiLU(gate) * up |
| FFN down output | `ffn_out_` **T × H** | Second residual: `x += x_mid + ffn_out` |
| Final RMSNorm | `xnorm_final_` **T × H** | Optional affine if `trainFinalNormAffine` |
| Logits | `logitsOut` **T × V** | Dense LM uses `vTile` over V and `hTile` over H |

`Tm = cfg.maxSeqLen` sizes persistent caches (e.g. `q_cache_` is `nLayers * Tm * H`).

### 2.2 BitNet I2S packed weights (`BitNetMatI2S` in `bitnet_weight.h`)

For a matrix **W** stored row-major with `rows` × `cols` (input along rows, output along cols for `y = x * W` style paths in code):

- **`qk`** — scale block height along the **input** (row) axis (constructor uses `TrainerConfig::bitnetQK`; FFN can use `bitnetFfnQK` when non-zero).
- **`jq`** — scale block width along the **output** (column) axis (`bitnetJQ`, default 32).
- Packed weights: 4 ternary weights per byte (`weightsPerByte = 4`).
- **`nKBlocks = ceil(rows / qk)`**, **`nJBlocks = ceil(cols / jq)`**; `w_scale_tile` length = `nKBlocks * nJBlocks`.

Forward BitNet linear passes **row tile size** `tTile` from `TrainerConfig` into kernels such as `linear_forward_bitnet_i2s_accum` (along with optional mask / prefix for skipped tiles).

### 2.3 Assisted backward tiles (`config.h`)

`bitnetBwdIB`, `bitnetBwdJB` (defaults 32) and `iTile` define the assisted backward tiling. Comments in `config.h` state the assisted path expects **`iTile == bitnetBwdIB`** for correctness today.

### 2.4 LoRA low-rank shapes

Per layer (when targets enabled): e.g. **Aq** is **R × H**, **Bq** is **R × H** for Q on hidden; **Bk**, **Bv** are **R × kvDim** for K/V; similar patterns for FFN and optional LM head **R × H**, **R × V**. Effective scale on deltas is `loraAlpha / R`.

---

## 3. Top-level function order

### 3.1 Process entry

1. **`main()`** (`main.cpp`) — OpenMP thread count, builds `TrainerConfig`, prints settings, calls **`runCpuQLoRATrainer(cfg, dataset_path)`**.

### 3.2 Trainer setup (`qlora_cpu.cpp`)

2. **`runCpuQLoRATrainer`**
   - Constructs **`Dataset`** from JSONL + `cfg`.
   - Optional eval **`Dataset`**.
   - Constructs **`CpuKernelBackend`** `backend{ cfg }` (loads weights, builds `shape_`, allocates buffers).
   - Chooses **AdamW** or **Lion**, constructs **`CpuQLoRATrainer`**, calls **`trainer.run()`**.

### 3.3 Training loop (`CpuQLoRATrainer::run`)

3. Optional **`Checkpointer::loadLatest`** if `resumeFromCheckpoint`.
4. **`backend.zeroGrad()`** then **`optimizer.init(backend.collectTrainableParams())`**.
5. Each **global step** (until `maxSteps`):
   - **`sampleBatch(batchSize)`** — shuffle epoch order unless `overfitSampleIndex` fixes one row.
   - **`trainStep(batch, accumLoss, accumTokens)`** (may run multiple micro-batches before an optimizer step).
   - When grad-accum or supervised-token target is met:
     - Optional **`scaleGradients`** (token target vs. grad-accum count).
     - Optional **grad clip** (`clipGradNorm`).
     - **`scheduler.lrAt(step)`** then **`optimizer.step(params, zeroGradAfterStep)`**.
   - **`reporter_.onStep`**; optional eval / train probe / checkpoint.

### 3.4 One micro-batch: `CpuQLoRATrainer::trainStep`

For each sample in the batch:

6. Derive **T** from `attentionMask` (last non-zero position).
7. Build **`lossMask`** from `targetIds`, `ignoreIndex`, `maxSupervisedTokensPerSample`, vocab range.
8. Resize **`logitsBuf_`** to **`T * V`**, zero **`dLogitsBuf_`**.
9. **`backend.forward(inputIds[0:T], attentionMask[0:T], logitsBuf_, lossMask)`**.
10. For each supervised **t**: **`cross_entropy_dlogits_row`** → fills **`dLogitsBuf_`** row; optional per-token weights, optional **`normalizeDLogitsBySampleTokens`** scaling.
11. **`backend.backward(..., dLogitsBuf_, lossMask)`**.

---

## 4. Forward graph order (`CpuKernelBackend::forward`)

Single call processes one sequence of length **T**. Inside `cpuKernelBack.cpp`:

1. **Token embedding** — scatter `inputIds` into `x_` (**T × H**), respecting mask.
2. **Save checkpoint slice** — `x_ckpt_[0]` = initial `x_`.
3. **For each layer `L = 0 .. nLayers-1`:**
   - **Attention RMSNorm** — `xnorm` from `x_` (+ optional fusion writing layer cache).
   - **QKV projection** — one of: TL2 generated, TL2 accum, fused I2S QKV, separate I2S lines, or scalar FP loops; optional **fused** TL2+I2S hybrid paths when flags allow.
   - **LoRA on Q/K/V** — `lora_add_*` into `q`/`k`/`v` with rank scratch `tmpQ/K/V`.
   - **RoPE** — in-place on Q (per head **D**) and K (per KV head **D**).
   - **Optional cache copy** of Q/K/V for backward.
   - **`attn_causal`** — fills `ctx_` (**T × H**), stores probs in `probs_cache_` if enabled.
   - **Wo** — TL2 / I2S / dense; optional **LoRA O**.
   - **Residual** — `x_mid = x + attn`.
   - **FFN RMSNorm** on `x_mid`.
   - **FFN gate & up** — BitNet / TL2 / optional **`linear_THF_fused_dual_accum_blocked32`** when `ffnGemm` path; optional gate/up LoRA.
   - **SiLU ×** gate and up → `ffn_act`.
   - **FFN down** → `ffn_out`; optional down LoRA.
   - **Residual** — `x = x_mid + ffn_out`.
   - **Checkpoint** — `x_ckpt_[(L+1)*Tm*H]`.
4. **Final RMSNorm** — `xnorm_final` from `x_`; optional affine.
5. **LM head** — quantized I2S path **or** blocked dense matmul over **`lmHeadRowsScratch_`** (supervised rows only when `lossMask` provided) with **`hTile`** / **`vTile`**; optional **LM head LoRA** and **bias**.

---

## 5. Backward graph order (`CpuKernelBackend::backward`)

Reverse sweep over layers. Order inside `cpuKernelBack.cpp`:

1. **LM head backward** — quantized packed backward or dense blocked path into **`dXnorm_final_`**; LM LoRA / bias grads if trained.
2. **Final norm backward** — into **`dX_`** (and optional RMS/affine grads).
3. **For each `Lrev = 0 .. nLayers-1` with `L = nLayers - 1 - Lrev`:**
   - Clear per-layer **`dCtx` / `dAttn` / `dQ` / `dK` / `dV` / FFN grads** scratch.
   - **Restore `x_` from `x_ckpt_[L]`** (activation before layer *L*).
   - Backprop **second residual** and **FFN** (down → SiLU → gate/up), BitNet or GEMM branches, LoRA accumulations as configured.
   - Backprop **first residual** and **attention**: **Wo**, **`attn_causal_backward`** (or equivalent), **LoRA O**, **RoPE backward** on Q/K, **QKV / fused QKV backward** into **`dXnorm_`**, **attention RMSNorm backward** into layer input gradient.
4. **Embedding backward** is implicit via **`dX_`** at **t=0..T-1** (scatter-add into embedding grads where implemented).

(Exact helper names vary by policy flags; the **dataflow** is: logits → final norm → stack of layers **L−1 → 0** → embeddings.)

---

## 6. Python side (`run.py`) for full repo workflow

When using the bundled script: HF export → `test_model/` weights + `test_dataset.jsonl` → launch **`qlora_test.exe`** → LoRA export to GGUF → BitNet CLI inference. That is **orchestration** around the same C++ pipeline above; geometry still comes from `test_model/config.json` and `TrainerConfig` / env as compiled into `main`.

---

## 7. Quick reference: files

| Concern | Primary files |
|---------|----------------|
| CLI / default cfg | `main.cpp` |
| Loop, loss, optimizer | `qlora_cpu.cpp`, `qlora_cpu.h`, `adamw_optimizer.*`, `lion_optimizer.*` |
| Forward / backward | `cpuKernelBack.cpp`, `cpuKernelBack.h` |
| Model JSON → shape | `cpuKernelBack.cpp` (`loadShapeFromConfig`) |
| Causal attention | `attention.cpp` |
| BitNet I2S matmul | `bitnet_i2s_matmul.*`, `bitnet_weight.h` |
| Autotune / runtime toggles | `backend_runtime_policy*.h`, `backend_runtime_policy_builder.*`, `backend_runtime_policy_json.*` |
| Config struct | `config.h` |
| JSONL data | `dataset.cpp`, `dataset.h` |

This file is descriptive; for exact branching under every flag combination, follow the `if (useTl2…)` / `if (useI2S…)` chains in `CpuKernelBackend::forward` and `::backward` in `cpuKernelBack.cpp`.
