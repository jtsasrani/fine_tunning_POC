import os
import json
import random
from datasketch import MinHash, MinHashLSH

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
    
    # Deduplicate using MinHash + LSH
    unique_samples = []
    seen_instructions = set()
    dup_count = 0
    fuzzy_dup_count = 0
    
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    
    for idx, item in enumerate(raw_data):
        inst = item["instruction"].strip()
        
        # 1. Exact match deduplication
        if inst in seen_instructions:
            dup_count += 1
            continue
            
        # 2. Fuzzy match deduplication via MinHash LSH
        words = [w.strip() for w in inst.lower().split() if w.strip()]
        if not words:
            continue
            
        m = MinHash(num_perm=128)
        for w in words:
            m.update(w.encode('utf-8'))
            
        results = lsh.query(m)
        if results:
            fuzzy_dup_count += 1
            continue
            
        seen_instructions.add(inst)
        lsh.insert(f"idx_{idx}", m)
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
            
    # Save a separate gold-standard evaluation set of 35 items from the validation split
    output_eval = "data/evaluation_set.jsonl"
    eval_set = val_split[:35]
    with open(output_eval, 'w', encoding='utf-8') as f:
        for item in eval_set:
            eval_item = {
                "question": item["instruction"],
                "ground_truth": item["output"],
                "paragraph_id": item.get("paragraph_id", ""),
                "source_doc": item.get("source_doc", "")
            }
            f.write(json.dumps(eval_item, ensure_ascii=False) + '\n')
            
    print(f"Created gold-standard evaluation set at {output_eval} with {len(eval_set)} samples.")
    print(f"Saved files successfully to 'data/' directory.")

if __name__ == "__main__":
    merge_and_split()
