#include <filesystem>
#include <iostream>
#include <string>
#include <exception>

#include "config.h"
#include "qlora_cpu.h"

namespace fs = std::filesystem;

int main() {
    std::cerr << "TRAINER MAIN IS RUNNING\n";
    try {
        std::cout << "========================================================\n";
        std::cout << "   BITNET QLORA CPU TRAINING PIPELINE INTEGRATION\n";
        std::cout << "========================================================\n\n";

        const std::string model_path = "./test_model";
        const std::string dataset_path = "./test_dataset.jsonl";

        std::cout << "[1/2] Configuring Trainer...\n";
        TrainerConfig cfg;
        cfg.modelPath = model_path;
        cfg.loraOutputPath = std::filesystem::absolute("lora_output.bin").string();
        cfg.checkpointDir = std::filesystem::absolute("checkpoints").string();

        // ── DIMENSIONS & MEMORY ──────────────────────────────────────────
        // 128 ensures the ENTIRE math question, answer, and EOS token fit.
        // If the model doesn't see the EOS token, it cannot converge.
        cfg.maxSeqLen = 128;

        // Batch Size 2 keeps the forward/backward intermediate buffers 
        // extremely small, allowing them to live entirely inside your 
        // Ryzen 7840U's 16MB L3 Cache. 
        cfg.batchSize = 2;

        // We accumulate 4 times to simulate a batch size of 8. 
        // This gives you the stability of a full-dataset batch, with 
        // the cache-blistering speed of micro-batches.
        cfg.gradAccumSteps = 4;
        cfg.maxSteps = 200;

        // ── OPTIMIZER (LION) ─────────────────────────────────────────────
        // 1e-6 is slightly too quiet. 5e-6 allows the Lion optimizer 
        // to actually move the needle out of the 11.9 entropy state.
        cfg.learningRate = 5e-6f;
        cfg.lionBeta1 = 0.9f;
        cfg.lionBeta2 = 0.99f;
        cfg.weightDecay = 0.0f; // Perfect. Keep at 0 for LoRA.
        cfg.maxGradNorm = 1.0f;

        // ── LOGGING ──────────────────────────────────────────────────────
        // Set to 1. Do not wait 10 steps to find out if the engine works.
        // You want to see 11.90 drop to 11.85 on step 1!
        cfg.logInterval = 1;
        cfg.checkpointInterval = 50;
        cfg.resumeFromCheckpoint = false; // Start fresh for this test

        // ── QUANTIZATION ─────────────────────────────────────────────────
        cfg.useBitNetQuant = true;
        cfg.bitnetQuantizeQKV = true;
        cfg.bitnetQuantizeWo = true;
        cfg.bitnetQK = 128;
        cfg.bitnetJQ = 32;

        std::cout << "[2/2] Starting Training Loop...\n";
        int result = runCpuQLoRATrainer(cfg, dataset_path);

        std::cout << "\nPipeline execution finished successfully.\n";
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << "\n";
        return 1;
    }
}