import os
import re
import json
import random
import torch
from unsloth import FastLanguageModel

def get_jaccard_sim(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    d = a.union(b)
    return float(len(c)) / len(d) if len(d) > 0 else 0.0

def get_containment_score(answer_str, context_str):
    a_words = set(re.findall(r'\w+', answer_str.lower()))
    b_words = set(re.findall(r'\w+', context_str.lower()))
    stopwords = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
        "to", "of", "in", "on", "at", "by", "for", "with", "about", 
        "against", "between", "into", "through", "during", "before", 
        "after", "above", "below", "from", "up", "down", "in", "out", 
        "on", "off", "over", "under", "again", "further", "then", "once"
    }
    a_words_filtered = a_words - stopwords
    if not a_words_filtered:
        return 0.0
    c = a_words_filtered.intersection(b_words)
    return float(len(c)) / len(a_words_filtered)


def run_tier2_judge_batch(model, tokenizer, qa_batch, sub_batch_size=2):
    # qa_batch: list of dicts with {"context": ..., "question": ..., "answer": ...}
    prompts = []
    for item in qa_batch:
        judge_prompt = (
            f"Context: {item['context']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}\n\n"
            "Evaluate: Is the answer fully supported by the context without outside assumptions? "
            "Respond with exactly one word: PASS or FAIL"
        )
        messages = [{"role": "user", "content": judge_prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)
        
    results = []
    # Process in small sub-batches to prevent CUDA OOM
    for i in range(0, len(prompts), sub_batch_size):
        sub_prompts = prompts[i:i+sub_batch_size]
        inputs = tokenizer(sub_prompts, padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                temperature=0.1,
                top_p=0.9
            )
            
        for idx, out in enumerate(outputs):
            generated_tokens = out[inputs.input_ids.shape[1]:]
            resp = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().upper()
            if "PASS" in resp:
                results.append("PASS")
            else:
                results.append("FAIL")
                
    return results

def main():
    input_file = "data/real_chunks.jsonl"
    output_file = "data/llm_generated_training_data.jsonl"
    checkpoint_file = "data/generation_checkpoint.json"
    
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run 01_ingest_all_pdfs.py first.")
        return
        
    print(f"Loading chunks from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(all_chunks)} total chunks.")
    
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_ids = set(checkpoint.get("processed_ids", []))
                print(f"Resuming from checkpoint. Already processed {len(processed_ids)} chunks.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            
    out_f = open(output_file, 'a', encoding='utf-8')
    
    # Load teacher model (Qwen2.5-32B-Instruct) in 4-bit via Unsloth
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    print(f"Loading teacher model '{model_name}' in 4-bit...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
        device_map="auto"
    )
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"  # Required for batching
    
    print("Model loaded successfully. Starting generation...")
    
    system_prompt_dmg = (
        "You are an expert trainer for the DWP Child Maintenance Service (CMS). "
        "Given a policy or legislative text, generate {N} realistic question-answer pairs "
        "that a caseworker or decision maker might ask to interpret the policy rules. Base "
        "answers SOLELY on the provided text. Cite specific rules or paragraph numbers where present. "
        "Each question must be distinct and test a DIFFERENT aspect of the text (e.g., eligibility criteria, "
        "general rules, exceptions, or specific edge cases). Avoid asking questions that result in "
        "overlapping answers."
    )
    
    system_prompt_proc = (
        "You are an expert trainer for the DWP Child Maintenance Service (CMS). "
        "Given an operational procedure guideline, generate exactly 1 realistic question-answer "
        "pair focusing on the business logic, eligibility criteria, decision rules, constraints, "
        "or conditions (i.e. 'when', 'why', and 'under what conditions' steps must be taken). "
        "DO NOT generate questions about click-by-click UI navigation steps (e.g. 'click this tab', "
        "'press F3'). Base answers SOLELY on the provided text."
    )
    
    count_generated = 0
    tier1_rejected = 0
    tier2_rejected = 0
    
    # Temporary list to hold Tier 2 candidates for batching
    tier2_candidates = []
    
    for idx, chunk in enumerate(all_chunks):
        chunk_id = chunk["paragraph_id"]
        if chunk_id in processed_ids:
            continue
            
        category = chunk.get("chunk_type", "procedure")
        context = chunk["formatted_text"]
        
        # Weighted settings
        if category in ["dmg", "policy_guidance"]:
            num_qa = random.choice([2, 3])
            system_prompt = system_prompt_dmg.replace("{N}", str(num_qa))
            user_prompt = (
                f"Context:\n{context}\n\n"
                f"Generate exactly {num_qa} question-answer pair(s) in this JSON format:\n"
                '[{"instruction": "question", "output": "detailed answer"}, ...]'
            )
        else:
            num_qa = 1
            system_prompt = system_prompt_proc
            user_prompt = (
                f"Context:\n{context}\n\n"
                "Generate exactly 1 question-answer pair in this JSON format:\n"
                '[{"instruction": "question", "output": "detailed answer"}]'
            )
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                temperature=0.45,  # Raised for diversity
                top_p=0.9,
                repetition_penalty=1.1
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Parse JSON
        try:
            # Clean possible markdown block formatting
            cleaned_response = response
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            data = json.loads(cleaned_response)
            if not isinstance(data, list):
                if isinstance(data, dict):
                    data = [data]
                else:
                    raise ValueError("Output is not a list or dict")
                    
            # Tier 1 filtering (Heuristics)
            valid_pairs = []
            for qa in data:
                q = qa.get("instruction", "").strip()
                a = qa.get("output", "").strip()
                
                # Length checks
                if len(q) < 20 or not q.endswith("?"):
                    tier1_rejected += 1
                    continue
                if len(a) < 50:
                    tier1_rejected += 1
                    continue
                    
                # Generic pattern filter
                generic_patterns = ["what is this document", "can you summarize", "summary of the", "outline the content"]
                if any(pat in q.lower() for pat in generic_patterns):
                    tier1_rejected += 1
                    continue
                    
                # Context overlap check
                overlap = get_containment_score(a, chunk["text"])
                if overlap < 0.25:
                    tier1_rejected += 1
                    continue
                    
                valid_pairs.append({"instruction": q, "output": a})
                
            # Inter-answer overlap filter (for multi-QA)
            if len(valid_pairs) > 1:
                final_valid = []
                for p_idx, pair in enumerate(valid_pairs):
                    is_dup = False
                    for existing in final_valid:
                        ans_overlap = get_jaccard_sim(pair["output"], existing["output"])
                        if ans_overlap > 0.60:
                            is_dup = True
                            # Keep the longer answer
                            if len(pair["output"]) > len(existing["output"]):
                                existing["instruction"] = pair["instruction"]
                                existing["output"] = pair["output"]
                            break
                    if not is_dup:
                        final_valid.append(pair)
                valid_pairs = final_valid
                
            # Queue for Tier 2 evaluation
            for pair in valid_pairs:
                tier2_candidates.append({
                    "context": chunk["text"],
                    "question": pair["instruction"],
                    "answer": pair["output"],
                    "paragraph_id": chunk_id,
                    "source_doc": chunk["source_doc"]
                })
                
        except Exception as e:
            # Silently log errors to keep output clean, but resume
            pass
            
        processed_ids.add(chunk_id)
        
        # Run Tier 2 Judge when we have a batch of 12 (or at the end)
        if len(tier2_candidates) >= 12 or (idx + 1 == len(all_chunks) and tier2_candidates):
            batch_results = run_tier2_judge_batch(model, tokenizer, tier2_candidates)
            for item, judge_result in zip(tier2_candidates, batch_results):
                if judge_result == "PASS":
                    qa_pair = {
                        "instruction": item["question"],
                        "output": item["answer"],
                        "paragraph_id": item["paragraph_id"],
                        "source_doc": item["source_doc"]
                    }
                    out_f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                    out_f.flush()
                    count_generated += 1
                else:
                    tier2_rejected += 1
            tier2_candidates = []
            
            # Print stats update
            print(f"Progress: {idx+1}/{len(all_chunks)} chunks. Generated: {count_generated} QA pairs. Tier 1 rejected: {tier1_rejected}. Tier 2 rejected: {tier2_rejected}.")
            
        # Save checkpoint periodically
        if (idx + 1) % 50 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump({"processed_ids": list(processed_ids)}, f)
                
    out_f.close()
    with open(checkpoint_file, 'w') as f:
        json.dump({"processed_ids": list(processed_ids)}, f)
        
    print(f"\nFinished Q&A Generation!")
    print(f"Total QA Pairs Generated: {count_generated}")
    print(f"Tier 1 Heuristic Rejected: {tier1_rejected}")
    print(f"Tier 2 LLM Judge Rejected: {tier2_rejected}")

if __name__ == "__main__":
    main()
