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

MAX_SEQ_LEN = 64   # must match cfg.maxSeqLen in main.cpp
IGNORE_IDX  = -100

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
    required_probe = [
        f"{WEIGHTS_DIR}/embeddings.bin",
        f"{WEIGHTS_DIR}/W_out.bin",
        f"{WEIGHTS_DIR}/rms_final.bin",
        f"{WEIGHTS_DIR}/layer_0_rms_ffn.bin",
        f"{WEIGHTS_DIR}/layer_0_Wgate.bin",
        f"{WEIGHTS_DIR}/layer_0_Wup.bin",
        f"{WEIGHTS_DIR}/layer_0_Wdown.bin",
    ]
    config_out_path = f"{MODEL_DIR}/config.json"
    if os.path.exists(config_out_path) and all(os.path.exists(p) for p in required_probe):
        try:
            with open(config_out_path, "r", encoding="utf-8") as f:
                existing_cfg = json.load(f)
            has_required_cfg = all(k in existing_cfg for k in ("intermediate_size", "rope_theta", "rope_dim"))
            if has_required_cfg:
                print(f"\n[2/6] Native weights already exist. Skipping extraction.")
                return
        except Exception:
            pass

    print(f"\n[2/6] Extracting FULL native FP32 matrices directly from SafeTensors...")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    config_path = hf_hub_download(repo_id=HF_BF16_MODEL, filename="config.json")
    with open(config_path, "r") as f:
        model_config = json.load(f)

    hidden_size = int(model_config["hidden_size"])
    num_attention_heads = int(model_config["num_attention_heads"])
    num_key_value_heads = int(model_config.get("num_key_value_heads", num_attention_heads))
    num_layers = int(model_config["num_hidden_layers"])
    intermediate_size = int(model_config.get("intermediate_size", hidden_size * 4))
    head_dim = hidden_size // num_attention_heads
    partial_rotary_factor = float(model_config.get("partial_rotary_factor", 1.0))
    rope_dim = int(head_dim * partial_rotary_factor)
    if rope_dim <= 0:
        rope_dim = head_dim
    if rope_dim % 2 != 0:
        rope_dim -= 1
    if rope_dim <= 0:
        rope_dim = head_dim if (head_dim % 2 == 0) else max(2, head_dim - 1)
    rope_theta = float(model_config.get("rope_theta", 10000.0))
    
    config_out = {
        "vocab_size": int(model_config["vocab_size"]),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "rope_theta": rope_theta,
        "rope_dim": rope_dim,
    }
    with open(f"{MODEL_DIR}/config.json", "w", encoding="utf-8") as f: 
        json.dump(config_out, f, indent=2)

    from safetensors.torch import load_file
    safetensors_path = hf_hub_download(repo_id=HF_BF16_MODEL, filename="model.safetensors")
    state_dict = load_file(safetensors_path)

    export_to_bin(state_dict["model.embed_tokens.weight"], f"{WEIGHTS_DIR}/embeddings.bin", False)
    
    if "lm_head.weight" in state_dict:
        export_to_bin(state_dict["lm_head.weight"], f"{WEIGHTS_DIR}/W_out.bin", True)
    else:
        print("    [*] 'lm_head.weight' not found. Utilizing tied embeddings.")
        export_to_bin(state_dict["model.embed_tokens.weight"], f"{WEIGHTS_DIR}/W_out.bin", True)
        
    export_to_bin(state_dict["model.norm.weight"], f"{WEIGHTS_DIR}/rms_final.bin", False)
    print(f"Exporting {num_layers} layers...")
    for i in range(num_layers):
        export_to_bin(state_dict[f"model.layers.{i}.input_layernorm.weight"], f"{WEIGHTS_DIR}/layer_{i}_rms_attn.bin")
        export_to_bin(state_dict[f"model.layers.{i}.post_attention_layernorm.weight"], f"{WEIGHTS_DIR}/layer_{i}_rms_ffn.bin")
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.q_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wq.bin", True)
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.k_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wk.bin", True) 
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.v_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wv.bin", True) 
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.o_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wo.bin", True)
        export_to_bin(state_dict[f"model.layers.{i}.mlp.gate_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wgate.bin", True)
        export_to_bin(state_dict[f"model.layers.{i}.mlp.up_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wup.bin", True)
        export_to_bin(state_dict[f"model.layers.{i}.mlp.down_proj.weight"], f"{WEIGHTS_DIR}/layer_{i}_Wdown.bin", True)

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================
def build_sample_with_prompt_mask(tokenizer, instruction: str, response: str,
                                   short_explanation: str = "") -> dict:
    """
    Tokenize one instruction/response pair with prompt masking.

    target_ids[t] = input_ids[t+1]  for response positions  (supervised)
    target_ids[t] = IGNORE (-100)   for prompt positions     (ignored)

    The C++ trainer reads target_ids directly from the JSONL, so masked
    positions never contribute to the loss or gradients.
    """
    # Build the two text spans
    prompt_text = f"User: {instruction}\nAssistant: "
    answer_part = f"{response}"
    if short_explanation:
        answer_part += f" {short_explanation}"
    full_text = prompt_text + answer_part

    # Tokenize separately so we know the exact prompt boundary
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)
    full_ids.append(tokenizer.eos_token_id)   # teach model to stop

    # Truncate
    if len(full_ids) > MAX_SEQ_LEN:
        full_ids = full_ids[:MAX_SEQ_LEN]

    n = len(full_ids)
    num_prompt_tokens = len(prompt_ids)

    # Build target_ids: supervised only on response tokens
    target_ids = [IGNORE_IDX] * n
    for t in range(n - 1):
        if t >= num_prompt_tokens - 1:
            target_ids[t] = full_ids[t + 1]
    # Always ignore the very last position (nothing to predict after it)
    target_ids[-1] = IGNORE_IDX

    attention_mask = [1] * n

    # Pad to MAX_SEQ_LEN so all samples are the same length
    pad_len = MAX_SEQ_LEN - n
    if pad_len > 0:
        full_ids       += [0]          * pad_len
        target_ids     += [IGNORE_IDX] * pad_len
        attention_mask += [0]          * pad_len

    return {
        "input_ids":      full_ids,
        "target_ids":     target_ids,
        "attention_mask": attention_mask,
    }


def prepare_dataset():
    print(f"\n[3/6] Tokenizing {INPUT_JSONL} with prompt masking...")
    tokenizer = AutoTokenizer.from_pretrained(HF_BF16_MODEL, trust_remote_code=True)

    n_written = 0
    supervised_tokens_total = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as infile, \
         open(TOKENIZED_OUT, "w", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line)

            sample = build_sample_with_prompt_mask(
                tokenizer,
                instruction=data["instruction"],
                response=data["response"],
                short_explanation=data.get("short_explanation", ""),
            )

            outfile.write(json.dumps(sample) + "\n")
            n_written += 1

            # Count how many positions are actually supervised (not -100)
            supervised = sum(1 for t in sample["target_ids"] if t != IGNORE_IDX)
            supervised_tokens_total += supervised

    avg_supervised = supervised_tokens_total / max(n_written, 1)
    print(f"[*] Wrote {n_written} samples to {TOKENIZED_OUT}")
    print(f"[*] Avg supervised tokens per sample: {avg_supervised:.1f} "
          f"(prompt tokens excluded from loss)")

    
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
        if magic != b"LORA":
            raise ValueError(f"Unexpected LoRA magic: {magic!r}")
        if int(file_H) != H:
            raise ValueError(f"LoRA hidden size mismatch: header={file_H} config={H}")
        
        writer = gguf.GGUFWriter(LORA_GGUF_OUT, exact_arch)
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", float(alpha))

        for L in range(nL):
            if flags & 1: # Q_PROJ
                a = np.fromfile(f, dtype=np.float32, count=R * H).reshape((R, H))
                b = np.fromfile(f, dtype=np.float32, count=H * R).reshape((H, R))
                writer.add_tensor(f"blk.{L}.attn_q.weight.lora_a", a)
                writer.add_tensor(f"blk.{L}.attn_q.weight.lora_b", b)

            if flags & 4: # V_PROJ
                a = np.fromfile(f, dtype=np.float32, count=R * H).reshape((R, H))
                # GQA nuance: V projection output is kv_dim (n_kv_heads * head_dim), not H.
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
    
    print(f"\n[4/6] Running C++ Trainer: {CPP_TRAINER_EXE}")
    subprocess.run([CPP_TRAINER_EXE], check=True)
    
    convert_lora_to_gguf(gguf_path)
    run_bitnet_inference(cli_path, gguf_path)
