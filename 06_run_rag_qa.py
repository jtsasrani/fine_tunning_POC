import os
import re
import sys
import time
import argparse
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import CrossEncoder

# Ensure HuggingFace can download the Cross-Encoder model if not cached.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_query_embedding(query, model, tokenizer, device):
    encoded_input = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
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

def retrieve_context_hybrid_rerank(query, db_data, faiss_index, embed_model, embed_tokenizer, reranker, device, top_k=3):
    # 1. FAISS Vector Search with BGE prefix
    query_prefixed = f"Represent this sentence for searching relevant passages: {query}"
    query_emb = get_query_embedding(query_prefixed, embed_model, embed_tokenizer, device)
    query_emb_np = query_emb.cpu().numpy().astype('float32')
    
    # Retrieve top 10 candidates from vector search
    distances, indices = faiss_index.search(query_emb_np, 10)
    faiss_hits = indices[0]
    
    # 2. Keyword search overlap
    keyword_scores = []
    for idx, chunk in enumerate(db_data["chunks"]):
        k_score = get_keyword_score(query, chunk["text"])
        keyword_scores.append((k_score, idx))
    keyword_scores.sort(key=lambda x: x[0], reverse=True)
    keyword_hits = [idx for score, idx in keyword_scores[:10] if score > 0.0]
    
    # 3. Union Candidate Pool
    candidate_indices = list(set(list(faiss_hits) + keyword_hits))
    candidate_indices = [idx for idx in candidate_indices if idx >= 0 and idx < len(db_data["chunks"])]
    
    if not candidate_indices:
        return []
        
    candidates = [db_data["chunks"][idx] for idx in candidate_indices]
    
    # 4. Cross-Encoder Reranking
    pairs = [[query, c["formatted_text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    
    ranked_results = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
    
    retrieved = []
    for score, chunk in ranked_results[:top_k]:
        retrieved.append({
            "chunk": chunk,
            "score": float(score)
        })
    return retrieved

def generate_response(model, tokenizer, prompt, device, max_tokens=300, temp=0.3):
    t_start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
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
    print(f"Generation complete in {t_end - t_start:.2f} seconds. (Generated {len(generated_tokens)} tokens)")
    return response

def main():
    parser = argparse.ArgumentParser(description="Run RAG QA using Fine-tuned Model and FAISS")
    parser.add_argument("--model_path", type=str, default="./trained_models/mistral-7b-cmg-qlora_merged", help="Path to 16bit merged model directory")
    parser.add_argument("--db_file", type=str, default="vector_db.pt", help="Path to vector database metadata file")
    parser.add_argument("--faiss_index", type=str, default="vector_db.index", help="Path to FAISS index file")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Vector Database
    if not os.path.exists(args.db_file) or not os.path.exists(args.faiss_index):
        print(f"Error: Vector DB files ({args.db_file} or {args.faiss_index}) not found. Build them first.")
        return
        
    print("Loading vector database metadata...")
    db_data = torch.load(args.db_file)
    print(f"Loaded database with {len(db_data['chunks'])} chunks.")
    
    print("Loading FAISS index...")
    faiss_index = faiss.read_index(args.faiss_index)
    
    # 2. Load Embedding & Reranker Models
    print("Loading embedding model BGE-base...")
    embed_tokenizer = AutoTokenizer.from_pretrained(db_data["model_name"])
    embed_model = AutoModel.from_pretrained(db_data["model_name"]).to(device)
    embed_model.eval()
    
    print("Loading Cross-Encoder Reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    # 3. Define test questions
    questions = [
        "Explain how the 25% income variance rule applies to a paying parent's gross weekly income.",
        "Under the Child Maintenance (Enforcement) Act 2023, what powers does the department have regarding administrative Liability Orders?",
        "How are pension contributions treated when calculating a paying parent's child maintenance liability?"
    ]
    
    # 4. Perform Retrieval for all questions
    retrievals = {}
    print("\n--- Performing Hybrid Retrieval & Reranking ---")
    for q in questions:
        print(f"\nQuery: '{q}'")
        retrieved_items = retrieve_context_hybrid_rerank(
            q, db_data, faiss_index, embed_model, embed_tokenizer, reranker, device, top_k=3
        )
        retrievals[q] = retrieved_items
        for idx, item in enumerate(retrieved_items):
            chunk = item["chunk"]
            print(f"  [{idx+1}] Rerank Score: {item['score']:.4f} | Paragraph: {chunk['paragraph_id']} | Source: {chunk['source_doc']}")
            print(f"      Text: {chunk['text'][:120]}...")
            
    # 5. Load Fine-tuned Reader Model
    print(f"\nLoading Reader Model from {args.model_path} on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    print("Reader model loaded successfully!")
    
    # System prompt to guide RAG behavior
    system_prompt = (
        "You are an expert Decision Maker helper for the DWP CMG. Answer the user's question "
        "accurately and professionally using ONLY the provided official policy contexts. "
        "State exact rules, percentages, and paragraph numbers if they are present in the context. "
        "If the context does not contain the information needed to answer the question, state clearly "
        "that the policy manual does not provide sufficient details. Do not assume or extrapolate."
    )
    
    # Run Inference
    print("\n--- Running RAG Inference ---")
    for i, q in enumerate(questions):
        print(f"\nQuestion {i+1}/3: '{q}'")
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
        
        response = generate_response(model, tokenizer, prompt, device, temp=0.3)
        print(f"Response:\n{response}")
        print("-" * 60)

if __name__ == "__main__":
    main()

