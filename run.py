import os
import sys
import json
import struct
import shutil
import subprocess
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import gguf

# ==============================================================================
# CONFIGURATION
# ==============================================================================
HF_BF16_MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
HF_GGUF_MODEL = "microsoft/bitnet-b1.58-2B-4T-gguf"

INPUT_JSONL   = "my_raw_data.jsonl"
TOKENIZED_OUT = "./test_dataset.jsonl"
MODEL_DIR     = "./test_model"
WEIGHTS_DIR   = f"{MODEL_DIR}/weights"

CPP_TRAINER_EXE = os.path.abspath("x64/Release/qlora_test.exe")
LORA_RAW_BIN    = os.path.abspath("./lora_output.bin")
LORA_GGUF_OUT   = os.path.abspath("./bitnet_adapter.gguf")

# ==============================================================================
# ENGINE & WEIGHT SETUP
# ==============================================================================
def setup_bitnet_engine():
    print("\n[1/6] Setting up Microsoft BitNet inference engine...")
    if not os.path.exists("BitNet"):
        subprocess.run(["git", "clone", "--recursive", "https://github.com/microsoft/BitNet.git"], check=True)

    cli_win = "BitNet/build/bin/Release/llama-cli.exe"
    base_gguf_path = "BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

    hf_hub_download(repo_id=HF_GGUF_MODEL, filename="ggml-model-i2_s.gguf", local_dir="BitNet/models/BitNet-b1.58-2B-4T")

    if os.path.exists(cli_win): return cli_win, base_gguf_path

    os.makedirs("BitNet/build", exist_ok=True)
    subprocess.run(["cmake", "..", "-T", "ClangCL", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TESTS=OFF"], cwd="BitNet/build", check=True)
    subprocess.run(["cmake", "--build", ".", "--config", "Release", "-j", "8"], cwd="BitNet/build", check=True)
    return cli_win, base_gguf_path

def export_to_bin(tensor, path, transpose=False):
    tensor = tensor.t().contiguous() if transpose else tensor.contiguous()
    tensor.to(torch.float32).cpu().numpy().tofile(path)

def prepare_cpp_weights():
    if os.path.exists(f"{WEIGHTS_DIR}/W_out.bin"): 
        print(f"\n[2/6] Native weights already exist. Skipping extraction.")
        return

    print(f"\n[2/6] Extracting FULL native FP32 matrices directly from SafeTensors...")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # 1. Download and parse the raw JSON config (bypasses Transformers config loader)
    config_path = hf_hub_download(repo_id=HF_BF16_MODEL, filename="config.json")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    hidden_size = int(model_config["hidden_size"])
    num_attention_heads = int(model_config["num_attention_heads"])
    num_key_value_heads = int(model_config.get("num_key_value_heads", num_attention_heads))
    num_layers = int(model_config["num_hidden_layers"]) 
    
    config_out = {
        "vocab_size": int(model_config["vocab_size"]),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": hidden_size // num_attention_heads,
    }
    with open(f"{MODEL_DIR}/config.json", "w", encoding="utf-8") as f: 
        json.dump(config_out, f, indent=2)

    # 2. Extract weights directly from SafeTensors (bypasses broken PyTorch scripts)
    from safetensors.torch import load_file
    safetensors_path = hf_hub_download(repo_id=HF_BF16_MODEL, filename="model.safetensors")
    state_dict = load_file(safetensors_path)

    # 3. Export matrices for C++ Engine
    export_to_bin(state_dict["model.embed_tokens.weight"], f"{WEIGHTS_DIR}/embeddings.bin", False)
    
    # Handle Weight Tying: If lm_head is missing, it shares the embedding weights
    if "lm_head.weight" in state_dict:
        export_to_bin(state_dict["lm_head.weight"], f"{WEIGHTS_DIR}/W_out.bin", True)
    else:
        print("    [*] 'lm_head.weight' not found. Utilizing tied embeddings.")
        export_to_bin(state_dict["model.embed_tokens.weight"], f"{WEIGHTS_DIR}/W_out.bin", True)
        
    export_to_bin(state_dict["model.norm.weight"], f"{WEIGHTS_DIR}/rms_final.bin", False)
    print(f"Exporting {num_layers} layers...")
    for i in range(num_layers):
        export_to_bin(state_dict[f"model.layers.{i}.input_layernorm.weight"], f"{WEIGHTS_DIR}/layer_{i}_rms_attn.bin")
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.q_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wq.bin", True)
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.k_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wk.bin", True) 
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.v_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wv.bin", True) 
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.o_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wo.bin", True)

# ==============================================================================
# DATA PREPROCESSING (THE FIX)
# ==============================================================================
def prepare_dataset():
    print(f"\n[3/6] Cleaning and Tokenizing {INPUT_JSONL}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_BF16_MODEL, trust_remote_code=True)
    
    with open(INPUT_JSONL, "r", encoding="utf-8") as infile, open(TOKENIZED_OUT, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip(): continue
            data = json.loads(line)
            
            # REMOVED METADATA: Strictly format as Tutor chat
            # Adding multiple EOS tokens to help small datasets converge
            text = (f"User: {data['instruction']}\n"
                    f"Assistant: {data['response']} {data.get('short_explanation', '')}"
                    f"{tokenizer.eos_token}")
            
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            
            # Simple format for C++ Trainer
            outfile.write(json.dumps({
                "input_ids": input_ids, 
                "attention_mask": [1] * len(input_ids)
            }) + "\n")
    print(f"[*] Created {TOKENIZED_OUT} with clean tutor sequences.")

# ==============================================================================
# GGUF CONVERSION & INFERENCE
# ==============================================================================
def get_exact_base_architecture(base_gguf_path):
    try:
        with open(base_gguf_path, 'rb') as f:
            content = f.read(16384)
            idx = content.find(b'general.architecture')
            if idx != -1:
                idx += len(b'general.architecture')
                vtype = struct.unpack('<I', content[idx:idx+4])[0]
                if vtype == 8:
                    vlen = struct.unpack('<Q', content[idx+4:idx+12])[0]
                    return content[idx+12 : idx+12+vlen].decode('utf-8')
    except Exception: pass
    return "llama"

def convert_lora_to_gguf(base_gguf_path):
    print(f"\n[5/6] Converting C++ output to GGUF adapter...")
    if not os.path.exists(LORA_RAW_BIN):
        print(f"Error: {LORA_RAW_BIN} not found. C++ trainer might have crashed.")
        return

    with open(f"{MODEL_DIR}/config.json", "r") as f: cfg = json.load(f)
    H, n_head = int(cfg["hidden_size"]), int(cfg["num_attention_heads"])
    kv_dim = int(cfg["num_key_value_heads"]) * int(cfg["head_dim"])

    exact_arch = get_exact_base_architecture(base_gguf_path)

    with open(LORA_RAW_BIN, "rb") as f:
        header = f.read(32)
        magic, version, dtype, flags, nL, file_H, R, alpha = struct.unpack("<4s6If", header)
        
        writer = gguf.GGUFWriter(LORA_GGUF_OUT, exact_arch)
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", float(alpha))

        for L in range(nL):
            if flags & 1: # Q_PROJ (with RoPE permutation for Llama-style arch)
                a = np.fromfile(f, dtype=np.float32, count=R * H).reshape((R, H))
                b = np.fromfile(f, dtype=np.float32, count=H * R).reshape((H, R))
                # Map back from C++ internal space to GGUF format
                writer.add_tensor(f"blk.{L}.attn_q.weight.lora_a", a)
                writer.add_tensor(f"blk.{L}.attn_q.weight.lora_b", b)

            if flags & 4: # V_PROJ
                a = np.fromfile(f, dtype=np.float32, count=R * H).reshape((R, H))
                b = np.fromfile(f, dtype=np.float32, count=kv_dim * R).reshape((kv_dim, R))
                writer.add_tensor(f"blk.{L}.attn_v.weight.lora_a", a)
                writer.add_tensor(f"blk.{L}.attn_v.weight.lora_b", b)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

def run_bitnet_inference(cli_path, base_gguf):
    print(f"\n[6/6] Testing the Tutor...")
    subprocess.run([
        cli_path, "-m", base_gguf, "--lora", LORA_GGUF_OUT,
        "-p", "User: Explain the power rule for derivatives.\nAssistant:",
        "-n", "128", "--temp", "0.4", "-c", "512"
    ], check=True)

if __name__ == "__main__":
    cli_path, gguf_path = setup_bitnet_engine()
    prepare_cpp_weights()
    prepare_dataset()
    
    print(f"\n[4/6] Running C++ Lion Trainer: {CPP_TRAINER_EXE}")
    # This will run for your ~6 hours. 
    # Monitor the console for Loss < 11.9
    subprocess.run([CPP_TRAINER_EXE], check=True)
    
    convert_lora_to_gguf(gguf_path)
    run_bitnet_inference(cli_path, gguf_path)