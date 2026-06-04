import os
import re
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel

# Configure thread usage and local execution (Commented out for fast multi-threaded inference on CPU)
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# torch.set_num_threads(1)

# Enable offline loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_query_embedding(query, model, tokenizer):
    encoded_input = tokenizer([query], padding=True, truncation=True, max_length=256, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

def get_keyword_score(query, paragraph_text):
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    stop_words = {"how", "are", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "under", "regarding", "about", "what", "does", "have", "when", "explain", "applies", "treated"}
    query_keywords = query_words - stop_words
    
    # Domain-specific terminology mapping to bridge vocabulary gaps
    if "paying" in query_keywords or "parent" in query_keywords:
        query_keywords.add("nrp")
    if "variance" in query_keywords:
        query_keywords.update(["differ", "differs", "difference", "change", "changed"])
        
    paragraph_words = set(re.findall(r'\b\w+\b', paragraph_text.lower()))
    match_count = 0
    for kw in query_keywords:
        if kw in paragraph_words:
            match_count += 1
            
    if not query_keywords:
        return 0.0
    return match_count / len(query_keywords)

def retrieve_context_hybrid(query, db_data, embed_model, embed_tokenizer, top_k=3):
    query_emb = get_query_embedding(query, embed_model, embed_tokenizer)
    vector_similarities = torch.matmul(db_data["embeddings"], query_emb.T).squeeze(1).tolist()
    
    hybrid_scores = []
    for idx, chunk in enumerate(db_data["chunks"]):
        v_score = vector_similarities[idx]
        k_score = get_keyword_score(query, chunk["text"])
        h_score = 0.5 * v_score + 0.5 * k_score
        hybrid_scores.append((h_score, idx))
        
    hybrid_scores.sort(key=lambda x: x[0], reverse=True)
    
    retrieved = []
    for score, idx in hybrid_scores[:top_k]:
        retrieved.append({
            "chunk": db_data["chunks"][idx],
            "score": score
        })
    return retrieved

def generate_response(model, tokenizer, prompt, max_tokens=300, temp=0.3):
    t_start = time.time()
    print(f"Tokenizing prompt...", flush=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print(f"Calling model.generate() with max_tokens={max_tokens}, temp={temp}...", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True if temp > 0.0 else False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    t_end = time.time()
    print(f"Generation complete in {t_end - t_start:.2f} seconds. (Generated {len(generated_tokens)} tokens)", flush=True)
    return response

def main():
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path = "./cmg_lora_weights"
    db_file = "vector_db.pt"
    
    if not os.path.exists(db_file):
        print(f"Error: Vector database {db_file} not found. Please build it first.", flush=True)
        return
        
    # 1. Load Vector Database
    print("Loading vector database...", flush=True)
    db_data = torch.load(db_file)
    print(f"Loaded database with {len(db_data['chunks'])} chunks.", flush=True)
    
    # 2. Load Embedding Model
    print("Loading embedding model on CPU...", flush=True)
    embed_tokenizer = AutoTokenizer.from_pretrained(db_data["model_name"])
    embed_model = AutoModel.from_pretrained(db_data["model_name"])
    embed_model.eval()
    
    # 3. Define questions
    questions = [
        "Explain how the 25% income variance rule applies to a paying parent's gross weekly income.",
        "Under the Child Maintenance (Enforcement) Act 2023, what powers does the department have regarding administrative Liability Orders?",
        "How are pension contributions treated when calculating a paying parent's child maintenance liability?"
    ]
    
    # 4. Perform Retrieval for all questions
    retrievals = {}
    print("\n--- Performing Retrieval ---", flush=True)
    for q in questions:
        print(f"Retrieving for query: '{q}'", flush=True)
        retrieved_items = retrieve_context_hybrid(q, db_data, embed_model, embed_tokenizer, top_k=3)
        retrievals[q] = retrieved_items
        for idx, item in enumerate(retrieved_items):
            chunk = item["chunk"]
            print(f"  [{idx+1}] Score: {item['score']:.4f} | Paragraph: {chunk['paragraph_id']} | Source: {chunk['source_doc']}", flush=True)
            print(f"      Text: {chunk['text'][:120]}...", flush=True)
            
    # 5. Load Base Causal Language Model
    print("\nLoading tokenizer for reader model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Loading base reader model on CPU...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    base_model.eval()
    print("Base model loaded successfully!", flush=True)
    
    # System prompt to guide RAG behavior
    system_prompt = (
        "You are an expert Decision Maker helper for the DWP CMG. Answer the user's question "
        "accurately and professionally using ONLY the provided official policy contexts. "
        "State exact rules, percentages, and paragraph numbers if they are present in the context. "
        "If the context does not contain the information needed to answer the question, state clearly "
        "that the policy manual does not provide sufficient details. Do not assume or extrapolate."
    )
    
    base_rag_responses = []
    
    # --- PHASE A: BASE MODEL + RAG INFERENCE ---
    print("\n--- Running Config B (Base Model + RAG) Inference ---", flush=True)
    for i, q in enumerate(questions):
        print(f"\nProcessing Base+RAG Question {i+1}/3: '{q}'...", flush=True)
        # Construct context string from retrieved chunks
        retrieved = retrievals[q]
        context_parts = []
        for r in retrieved:
            chunk = r["chunk"]
            context_parts.append(f"Paragraph {chunk['paragraph_id']} (from {chunk['source_doc']}):\n{chunk['text']}")
        context_str = "\n\n".join(context_parts)
        
        user_content = f"Contexts:\n{context_str}\n\nQuestion: {q}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        response = generate_response(base_model, tokenizer, prompt, temp=0.3)
        base_rag_responses.append(response)
        
    # --- PHASE B: LOAD LORA ADAPTERS ---
    print(f"\nApplying LoRA weights to base model from {adapter_path}...", flush=True)
    if not os.path.exists(adapter_path):
        print(f"Error: LoRA adapter weights not found at {adapter_path}. Cannot run tuned configurations.", flush=True)
        tuned_norag_responses = ["N/A"] * 3
        tuned_rag_responses = ["N/A"] * 3
    else:
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        peft_model.eval()
        print("LoRA weights applied successfully!", flush=True)
        
        # --- PHASE C: TUNED MODEL (NO RAG) INFERENCE ---
        print("\n--- Running Config A (Tuned Model without RAG) Inference ---", flush=True)
        tuned_norag_responses = []
        for i, q in enumerate(questions):
            print(f"\nProcessing Tuned-NoRAG Question {i+1}/3: '{q}'...", flush=True)
            # Baseline test prompt format matching fine-tuning evaluation (Phase 4)
            messages = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            response = generate_response(peft_model, tokenizer, prompt, temp=0.7) # Use 0.7 temp for non-RAG baseline
            tuned_norag_responses.append(response)
            
        # --- PHASE D: TUNED MODEL + RAG INFERENCE ---
        print("\n--- Running Config C (Tuned Model + RAG) Inference ---", flush=True)
        tuned_rag_responses = []
        for i, q in enumerate(questions):
            print(f"\nProcessing Tuned+RAG Question {i+1}/3: '{q}'...", flush=True)
            retrieved = retrievals[q]
            context_parts = []
            for r in retrieved:
                chunk = r["chunk"]
                context_parts.append(f"Paragraph {chunk['paragraph_id']} (from {chunk['source_doc']}):\n{chunk['text']}")
            context_str = "\n\n".join(context_parts)
            
            user_content = f"Contexts:\n{context_str}\n\nQuestion: {q}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            response = generate_response(peft_model, tokenizer, prompt, temp=0.3) # Use 0.3 temp for RAG
            tuned_rag_responses.append(response)
            
    # --- PHASE E: GENERATE COMPARATIVE REPORT & SAVE TO FILE ---
    print("\n" + "="*80, flush=True)
    print("                  DWP CMG RAG ARCHITECTURE COMPARISON REPORT", flush=True)
    print("="*80, flush=True)
    
    md_lines = [
        "# DWP CMG RAG Comparative Evaluation Report",
        "\nThis report provides the non-truncated outputs of the three evaluation configurations on the 3 core policy questions.",
        "\n## Comparison Matrix\n"
    ]
    
    for i, q in enumerate(questions):
        print(f"\n[QUESTION {i+1}]", flush=True)
        print(f"Question: {q}\n", flush=True)
        
        md_lines.append(f"### [QUESTION {i+1}] {q}\n")
        
        print("--- RETRIEVED CONTEXTS ---", flush=True)
        md_lines.append("#### Retrieved Contexts:")
        for idx, item in enumerate(retrievals[q]):
            chunk = item["chunk"]
            print(f"({idx+1}) [Score: {item['score']:.4f}] Paragraph {chunk['paragraph_id']} (from {chunk['source_doc']}):", flush=True)
            print(f"    {chunk['text'][:200]}...", flush=True)
            md_lines.append(f"- **Paragraph {chunk['paragraph_id']}** (Score: {item['score']:.4f}, Source: `{chunk['source_doc']}`):\n  > {chunk['text']}\n")
            
        print("-" * 50, flush=True)
        
        print("--- CONFIG A: TUNED MODEL (NO RAG) ---", flush=True)
        print(tuned_norag_responses[i], flush=True)
        print("-" * 50, flush=True)
        
        print("--- CONFIG B: BASE MODEL + RAG ---", flush=True)
        print(base_rag_responses[i], flush=True)
        print("-" * 50, flush=True)
        
        print("--- CONFIG C: TUNED MODEL + RAG ---", flush=True)
        print(tuned_rag_responses[i], flush=True)
        print("=" * 80, flush=True)
        
        md_lines.append("#### Comparative Outputs:")
        md_lines.append(f"##### Configuration A: Tuned Model (No RAG)\n```text\n{tuned_norag_responses[i]}\n```\n")
        md_lines.append(f"##### Configuration B: Base Model + RAG\n```text\n{base_rag_responses[i]}\n```\n")
        md_lines.append(f"##### Configuration C: Tuned Model + RAG\n```text\n{tuned_rag_responses[i]}\n```\n")
        md_lines.append("---\n")
        
    # Save to markdown file in artifact directory
    artifact_report_path = r"C:\Users\JitendraAsrani\.gemini\antigravity\brain\b3d94206-3538-4480-95d3-c9010fe8f009\full_answers_report.md"
    try:
        with open(artifact_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"\nSuccessfully wrote full report to {artifact_report_path}", flush=True)
    except Exception as e:
        print(f"\nWarning: Could not save report file: {e}", flush=True)

if __name__ == "__main__":
    main()
