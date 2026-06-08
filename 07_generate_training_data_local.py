import os
import json
import random
import torch
from unsloth import FastLanguageModel

def sample_chunks(input_file):
    # Load all chunks
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]
        
    print(f"Loaded {len(chunks)} total chunks for sampling.")
    
    # Stratified sampling
    sampled = []
    
    # Group by source doc / type
    v1 = [c for c in chunks if "volume-1" in c["source_doc"].lower()]
    v2 = [c for c in chunks if "vol2" in c["source_doc"].lower() or "volume-2" in c["source_doc"].lower()]
    v3 = [c for c in chunks if "volume-3" in c["source_doc"].lower()]
    v4 = [c for c in chunks if "volume-4" in c["source_doc"].lower()]
    v5 = [c for c in chunks if "volume-5" in c["source_doc"].lower()]
    v6 = [c for c in chunks if "volume-6" in c["source_doc"].lower()]
    v7 = [c for c in chunks if "volume-7" in c["source_doc"].lower()]
    others = [c for c in chunks if c["chunk_type"] == "legislation_or_guide"]
    
    random.seed(42)
    
    # Include all of Vol 1, 3, 5, 7 and Vol 4 (critical/medium sizes)
    sampled.extend(v1)
    sampled.extend(v3)
    sampled.extend(v4)
    sampled.extend(v5)
    sampled.extend(v7)
    
    # Sample Vol 2 (calculations) and Vol 6 (enforcement)
    sampled.extend(random.sample(v2, min(len(v2), 200)))
    sampled.extend(random.sample(v6, min(len(v6), 200)))
    
    # Sample from legislation/guides
    sampled.extend(random.sample(others, min(len(others), 150)))
    
    print(f"Sampled {len(sampled)} chunks for generation:")
    print(f"  Vol 1: {len(v1)}, Vol 2: sampled 200, Vol 3: {len(v3)}, Vol 4: {len(v4)}, Vol 5: {len(v5)}, Vol 6: sampled 200, Vol 7: {len(v7)}, Others: sampled 150")
    
    return sampled

def main():
    input_file = "data/real_chunks.jsonl"
    output_file = "data/llm_generated_training_data.jsonl"
    checkpoint_file = "data/generation_checkpoint.json"
    
    # Step 1: Sample chunks
    sampled_chunks = sample_chunks(input_file)
    
    # Check for existing progress
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_ids = set(checkpoint.get("processed_ids", []))
                print(f"Resuming from checkpoint. Already processed {len(processed_ids)} chunks.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            
    # Open output file in append mode
    out_f = open(output_file, 'a', encoding='utf-8')
    
    # Step 2: Load local LLM using Unsloth
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading local model '{model_name}' in 4-bit...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
        device_map="auto"
    )
    FastLanguageModel.for_inference(model) # 2x faster inference
    
    print("Model loaded successfully. Starting generation...")
    
    system_prompt = (
        "You are an expert trainer for the DWP Child Maintenance Group (CMG). "
        "Your task is to create high-quality training materials for new decision makers. "
        "Given a specific DWP policy guidelines chunk, generate exactly one realistic question "
        "a caseworker or trainee might ask, and provide a detailed, accurate answer based SOLELY on the guidelines. "
        "Do not invent policies or extrapolate. Cite the paragraph ID or source document in the answer."
    )
    
    count = 0
    for idx, chunk in enumerate(sampled_chunks):
        chunk_id = chunk["paragraph_id"]
        if chunk_id in processed_ids:
            continue
            
        context = chunk["formatted_text"]
        user_prompt = (
            f"Context:\n{context}\n\n"
            "Based on the context above, generate exactly one realistic question-answer pair. "
            "Output your response strictly in the following JSON format without any other text or markdown formatting:\n"
            '{"instruction": "question", "output": "detailed answer"}'
        )
        
        # Format using chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.1, 
                top_p=0.9, 
                repetition_penalty=1.1
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Parse JSON output
        try:
            # Clean possible markdown block formatting
            cleaned_response = response
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            data = json.loads(cleaned_response)
            
            # Simple quality validation
            if "instruction" in data and "output" in data:
                qa_pair = {
                    "instruction": data["instruction"],
                    "output": data["output"],
                    "paragraph_id": chunk_id,
                    "source_doc": chunk["source_doc"],
                    "quality_tier": "high"
                }
                out_f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                out_f.flush()
                count += 1
                
                if count % 10 == 0:
                    print(f"Generated {count} QA pairs... (Total processed: {idx+1}/{len(sampled_chunks)})")
        except Exception as e:
            print(f"Error parsing response for chunk {chunk_id}: {e}")
            print(f"Raw response: {response}")
            
        processed_ids.add(chunk_id)
        
        # Save checkpoint periodically
        if (idx + 1) % 50 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump({"processed_ids": list(processed_ids)}, f)
                
    out_f.close()
    
    # Save final checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({"processed_ids": list(processed_ids)}, f)
        
    print(f"Completed! Generated a total of {count} training QA pairs in {output_file}")

if __name__ == "__main__":
    main()
