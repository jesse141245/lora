# Kernel Architecture Research Notes

Date: 2026-04-01

This note captures upstream patterns worth stealing for the CPU LoRA training path in this repo.

## Repositories reviewed

- NVIDIA CUTLASS
  - https://github.com/NVIDIA/cutlass
  - Main pattern: separate launch policy and tile scheduling from the math kernel, then reuse prepared kernel arguments across many work items.

- Dao-AILab FlashAttention
  - https://github.com/Dao-AILab/flash-attention
  - Main pattern: pack QKV work so the kernel reuses staged activations and avoids extra reads and writes.

- Microsoft LoRA
  - https://github.com/microsoft/LoRA
  - Main pattern: use merged linear layers for packed QKV projections when only a subset of projections is adapted.

- bitsandbytes
  - https://github.com/bitsandbytes-foundation/bitsandbytes
  - Main pattern: blockwise quantization, explicit optimizer and runtime memory accounting, and practical telemetry for diagnosing pressure and paging.

- ggml / llama.cpp
  - https://github.com/ggml-org/llama.cpp
  - Main pattern: aggressively prepacked weights, avoiding useless buffer clears, and keeping quantized activation work close to the matmul path.

- Marlin
  - https://github.com/IST-DASLab/marlin
  - Main pattern: offline layout transforms and striped work partitioning so the runtime kernel can focus on a single hot access pattern.

## Patterns that map well to this repo

1. Packed projection interfaces matter more than tiny inner-loop tweaks.
   - FlashAttention and LoRA both lean on packed QKV style interfaces.
   - In this repo, that argues for fused forward and backward projection paths that quantize or stage `X` once, then reuse it across Q/K/V.

2. Kernel policy should be explicit.
   - CUTLASS exposes scheduling and tile policy separately from the kernel body.
   - In this repo, long-sequence defaults should continue moving into small policy helpers instead of being spread across call sites and env checks.

3. Avoid full-surface writes unless the math actually needs them.
   - ggml and Marlin both benefit from overwrite-first or skip-fill behavior when the kernel can prove coverage.
   - In this repo, skip-fill and overwrite are worth keeping only where validated, because several speculative versions changed the loss curve.

4. Telemetry is part of kernel work, not separate from it.
   - bitsandbytes style operational visibility is useful here because thermal drift and memory pressure have repeatedly confused quick screens.
   - Every sustained run should emit process CPU and memory traces alongside the existing clock trace.

## Applied here

- Kept: fused I2S QKV forward now uses the shared parallel tile quantizer on the defended `oldfast` branch.
- Kept: generated TL2 long-sequence forward is chunked and chunk-parallel by default.
- Kept: attention score storage defaults off unless explicitly requested.
- Added: training runner telemetry now captures process CPU and memory behavior during efficiency-mode runs.

## Next structural targets

1. Attention backward
   - Still one of the largest hot stages on the defended branch.
   - Best candidate for another packed or reused-staging design.

2. FFN `Wdown` path
   - The exact path still pays a lot of memory traffic relative to its math.
   - The next worthwhile experiment is likely structural staging, not another helper-level rewrite.

3. Shared launch policy object
   - Long-sequence heuristics for I2S and TL2 are still spread across multiple files.
   - A small runtime policy object would make the architecture easier to reason about and benchmark.
