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

        // maxSeqLen=32: samples are ~25 tokens, 64 was wasting 60% of every
        // forward/backward pass on masked padding with zero gradient signal.
        // batchSize=8: feeds the full 8-sample dataset in one accumulation window.
        // gradAccumSteps=1: step every batch — maximize weight updates per epoch.
        // maxSteps=500: ~62 passes over 8 samples. Loss should clearly fall
        // within the first 50 steps and reach near-zero overfit by step 200-300.
        cfg.maxSeqLen = 32;
        cfg.batchSize = 8;
        cfg.gradAccumSteps = 1;
        cfg.maxSteps = 500;

        // Set explicitly — header defaults may be 0 which zeros all LoRA output.
        cfg.loraRank = 8;
        cfg.loraAlpha = 8.0f;   // scale = alpha/rank = 1.0

        // lr=1e-6, standard Lion betas, no weight decay, clip=1.0.
        // The previous instability (step-5 spike, flat loss) was caused by three
        // bugs now fixed: (1) targetIds double-shift, (2) missing backend_.zeroGrad(),
        // (3) loraAlpha not set. None of those required a conservative lr.
        cfg.learningRate = 1e-6f;
        cfg.lionBeta1 = 0.9f;
        cfg.lionBeta2 = 0.99f;
        cfg.weightDecay = 0.0f;
        cfg.maxGradNorm = 1.0f;

        cfg.logInterval = 10;
        cfg.checkpointInterval = 100;
        cfg.resumeFromCheckpoint = true;

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