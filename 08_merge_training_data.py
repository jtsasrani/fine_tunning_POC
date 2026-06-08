import os
import json
import random

def get_jaccard_sim(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c)) if (len(a) + len(b) - len(c)) > 0 else 0

def merge_and_split():
    input_file = "data/llm_generated_training_data.jsonl"
    output_final = "data/final_training_data.jsonl"
    output_train = "data/train_split.jsonl"
    output_val = "data/val_split.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
        
    print(f"Loading generated training data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
        
    print(f"Loaded {len(raw_data)} total raw samples.")
    
    # Deduplicate
    unique_samples = []
    seen_instructions = set()
    dup_count = 0
    fuzzy_dup_count = 0
    
    for item in raw_data:
        inst = item["instruction"].strip()
        
        # 1. Exact match deduplication
        if inst in seen_instructions:
            dup_count += 1
            continue
            
        # 2. Fuzzy Jaccard match deduplication
        is_fuzzy_dup = False
        for existing in unique_samples:
            if get_jaccard_sim(inst, existing["instruction"]) > 0.85:
                is_fuzzy_dup = True
                fuzzy_dup_count += 1
                break
                
        if not is_fuzzy_dup:
            seen_instructions.add(inst)
            unique_samples.append(item)
            
    print(f"Deduplication results:")
    print(f"  Exact duplicates removed: {dup_count}")
    print(f"  Fuzzy duplicates removed: {fuzzy_dup_count}")
    print(f"  Unique samples remaining: {len(unique_samples)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(unique_samples)
    
    # Train / Val Split (90% / 10%)
    split_idx = int(len(unique_samples) * 0.9)
    train_split = unique_samples[:split_idx]
    val_split = unique_samples[split_idx:]
    
    print(f"Dataset split:")
    print(f"  Train split: {len(train_split)} samples")
    print(f"  Validation split: {len(val_split)} samples")
    
    # Save final merged data
    with open(output_final, 'w', encoding='utf-8') as f:
        for item in unique_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # Save train split
    with open(output_train, 'w', encoding='utf-8') as f:
        for item in train_split:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # Save val split
    with open(output_val, 'w', encoding='utf-8') as f:
        for item in val_split:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Saved files successfully to 'data/' directory.")

if __name__ == "__main__":
    merge_and_split()
