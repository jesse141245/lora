import os
import torch
from transformers import AutoModelForCausalLM

def export_to_bin(tensor, path, transpose=False):
    """Converts a PyTorch tensor to a raw Float32 binary file."""
    if transpose:
        tensor = tensor.t().contiguous()
    else:
        tensor = tensor.contiguous()
    
    # Cast to float32, convert to numpy, and write to raw binary file
    tensor.to(torch.float32).numpy().tofile(path)
    print(f"Saved: {path} | Shape: {list(tensor.shape)}")

def main():
    # We use a tiny model for testing. You can change this to 1.5B BitNet later.
    model_id = "microsoft/bitnet-b1.58-2B-4T" 
    
    # This must match the folder your C++ code looks for: cfg_.modelPath + "/weights"
    output_dir = "./test_model/weights"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading and loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    state_dict = model.state_dict()

    print(f"Exporting raw weights to {output_dir}...")

    # 1. Export Embeddings [V x H] (No transpose needed)
    export_to_bin(state_dict["model.embed_tokens.weight"], 
                  f"{output_dir}/embeddings.bin", transpose=False)

    # 2. Export LM Head [V x H] -> [H x V] (Needs transpose for your C++ W_out_)
    export_to_bin(state_dict["lm_head.weight"], 
                  f"{output_dir}/W_out.bin", transpose=True)

    # 3. Export Attention Layers (Currently exporting first 2 for your C++ test)
    n_layers = 2 
    for i in range(n_layers):
        print(f"\n--- Exporting Layer {i} ---")
        
        # RMSNorm (1D array, no transpose)
        export_to_bin(state_dict[f"model.layers.{i}.input_layernorm.weight"], 
                      f"{output_dir}/layer_{i}_rms_attn.bin")
        
        # Attention Projections [out, in] -> [in, out] (Require transpose)
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.q_proj.weight"], 
                      f"{output_dir}/layer_{i}_Wq.bin", transpose=True)
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.k_proj.weight"], 
                      f"{output_dir}/layer_{i}_Wk.bin", transpose=True)
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.v_proj.weight"], 
                      f"{output_dir}/layer_{i}_Wv.bin", transpose=True)
        export_to_bin(state_dict[f"model.layers.{i}.self_attn.o_proj.weight"], 
                      f"{output_dir}/layer_{i}_Wo.bin", transpose=True)

    print("\nWeight export complete!")

if __name__ == "__main__":
    main()