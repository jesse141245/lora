import json
from transformers import AutoTokenizer

def main():
    # Must match the model you exported above so the vocabulary aligns!
    model_id = "microsoft/bitnet-b1.58-2B-4T"
    
    input_jsonl = "my_raw_data.jsonl"          # Your custom math data
    output_jsonl = "./test_dataset.jsonl"      # What the C++ engine will read
    
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Tokenizing dataset...")
    
    with open(input_jsonl, 'r') as infile, open(output_jsonl, 'w') as outfile:
        for line in infile:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # 1. Extract your specific fields
            instruction = data.get("instruction", "")
            response = data.get("response", "")
            
            # (Optional: You could also include data.get("short_explanation") here if you want the model to learn it)
            
            # 2. Format into a conversational prompt
            # We add an EOS (End of Sequence) token so the model learns when to stop generating
            text = f"User: {instruction}\nAssistant: {response}{tokenizer.eos_token}"
            
            # 3. Convert English words into integer Token IDs
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            
            # 4. Create attention mask (1 for every real token)
            attention_mask = [1] * len(input_ids)
            
            # 5. Write to the format dataset.cpp expects
            cpp_ready_json = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            
            outfile.write(json.dumps(cpp_ready_json) + "\n")
            
    print(f"Done! Tokenized dataset saved to {output_jsonl}")

if __name__ == "__main__":
    main()