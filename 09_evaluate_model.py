import os
import re
import json
import time
import gc
import argparse
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import CrossEncoder
from rouge_score import rouge_scorer
import bert_score

# Enable offline loading only after downloads are complete
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
    # 1. FAISS Vector Search
    query_prefixed = f"Represent this sentence for searching relevant passages: {query}"
    query_emb = get_query_embedding(query_prefixed, embed_model, embed_tokenizer, device)
    query_emb_np = query_emb.cpu().numpy().astype('float32')
    
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

def generate_response(model, tokenizer, prompt, device, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0, # Greedy decoding for stable, deterministic evaluation
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response

def evaluate_model(model_path, eval_data, db_data, faiss_index, embed_model, embed_tokenizer, reranker, scorer, device):
    print(f"\n--- Starting Evaluation for Model: {os.path.basename(model_path)} ---")
    
    # Load model
    print(f"Loading reader model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    system_prompt = (
        "You are an expert Decision Maker assistant for the DWP Child Maintenance Service (CMS). "
        "Answer questions accurately using only the provided policy context. "
        "Cite specific paragraph numbers, rules, and sections where present. "
        "If the context does not contain sufficient information, state this clearly."
    )
    
    generated_answers = []
    ground_truths = []
    precisions = []
    inference_times = []
    
    for idx, sample in enumerate(eval_data):
        query = sample["question"]
        gt_text = sample["ground_truth"]
        
        # 1. Retrieve RAG Context
        retrieved = retrieve_context_hybrid_rerank(
            query, db_data, faiss_index, embed_model, embed_tokenizer, reranker, device, top_k=3
        )
        
        # 2. Check Retrieval Precision@3 (does any retrieved chunk match the ground truth text?)
        is_precision_hit = any(r["chunk"]["text"].strip() in gt_text.strip() or gt_text.strip() in r["chunk"]["text"].strip() for r in retrieved)
        precisions.append(1.0 if is_precision_hit else 0.0)
        
        # 3. Format RAG prompt
        context_parts = []
        for r in retrieved:
            chunk = r["chunk"]
            context_parts.append(f"Paragraph {chunk['paragraph_id']} (from {chunk['source_doc']}):\n{chunk['text']}")
        context_str = "\n\n".join(context_parts)
        
        user_content = f"Contexts:\n{context_str}\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 4. Generate response & time it
        t0 = time.time()
        ans = generate_response(model, tokenizer, prompt, device)
        inference_times.append(time.time() - t0)
        
        generated_answers.append(ans)
        ground_truths.append(gt_text)
        
        if (idx + 1) % 5 == 0 or idx + 1 == len(eval_data):
            print(f"  Processed {idx + 1}/{len(eval_data)} questions.")
            
    # Calculate ROUGE-L
    print("Computing ROUGE-L scores...")
    rouge_l_scores = []
    for gen, gt in zip(generated_answers, ground_truths):
        scores = scorer.score(gt, gen)
        rouge_l_scores.append(scores["rougeL"].fmeasure)
        
    # Calculate BERTScore
    print("Computing BERTScores...")
    P, R, F1 = bert_score.score(generated_answers, ground_truths, lang="en", device=device, verbose=False)
    bert_f1_scores = F1.tolist()
    
    # Aggregate Metrics
    avg_precision = np.mean(precisions)
    avg_rouge_l = np.mean(rouge_l_scores)
    avg_bert_f1 = np.mean(bert_f1_scores)
    avg_inf_time = np.mean(inference_times)
    
    # Cleanup memory
    print("Unloading model and cleaning up GPU VRAM...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "precision": avg_precision,
        "rouge_l": avg_rouge_l,
        "bert_score": avg_bert_f1,
        "avg_time": avg_inf_time,
        "answers": generated_answers
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned CMS model")
    parser.add_argument("--model_path", type=str, default="./trained_models/qwen-14b-cms-qlora_merged", help="Path to tuned model directory")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Path to base/reference model for comparison")
    parser.add_argument("--eval_file", type=str, default="data/evaluation_set.jsonl", help="Evaluation set path")
    parser.add_argument("--db_file", type=str, default="vector_db.pt", help="Vector DB file")
    parser.add_argument("--faiss_file", type=str, default="vector_db.index", help="FAISS index file")
    parser.add_argument("--report_path", type=str, default="data/evaluation_report.md", help="Output report path")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not os.path.exists(args.eval_file):
        print(f"Error: {args.eval_file} not found. Fallback to val_split.jsonl...")
        fallback = "data/val_split.jsonl"
        if os.path.exists(fallback):
            args.eval_file = fallback
        else:
            print(f"Error: {fallback} not found either. Please run generation and split first.")
            return
            
    # 1. Load evaluation set
    eval_data = []
    print(f"Loading evaluation questions from {args.eval_file}...")
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line.strip())
                # Normalize keys
                q = sample.get("question", sample.get("instruction", ""))
                gt = sample.get("ground_truth", sample.get("output", ""))
                if q and gt:
                    eval_data.append({"question": q, "ground_truth": gt})
                    
    # Cap evaluation at 40 questions to save time if val_split is used
    if len(eval_data) > 40:
        print(f"Subsampling evaluation dataset from {len(eval_data)} to 40 questions for speed.")
        np.random.seed(42)
        indices = np.random.choice(len(eval_data), 40, replace=False)
        eval_data = [eval_data[i] for i in indices]
        
    print(f"Loaded {len(eval_data)} evaluation samples.")
    
    # 2. Load Vector DB and FAISS index
    if not os.path.exists(args.db_file) or not os.path.exists(args.faiss_file):
        print(f"Error: Vector DB files {args.db_file} or {args.faiss_file} not found.")
        return
        
    db_data = torch.load(args.db_file)
    faiss_index = faiss.read_index(args.faiss_file)
    
    # 3. Load embedding model and Cross-Encoder
    embed_tokenizer = AutoTokenizer.from_pretrained(db_data["model_name"])
    embed_model = AutoModel.from_pretrained(db_data["model_name"]).to(device)
    embed_model.eval()
    
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    
    results = {}
    
    # Run evaluation on tuned model
    if os.path.exists(args.model_path):
        results["tuned"] = evaluate_model(
            args.model_path, eval_data, db_data, faiss_index, embed_model, embed_tokenizer, reranker, scorer, device
        )
    else:
        print(f"Error: Tuned model not found at {args.model_path}.")
        return
        
    # Run evaluation on base model (optional/reference)
    try:
        results["base"] = evaluate_model(
            args.base_model_path, eval_data, db_data, faiss_index, embed_model, embed_tokenizer, reranker, scorer, device
        )
    except Exception as e:
        print(f"Warning: Could not evaluate reference model {args.base_model_path}: {e}")
        
    # Print and generate report
    report_lines = [
        "# DWP CMS RAG Model Evaluation Report",
        f"\nThis report compares the performance of the fine-tuned CMS Model ({args.model_path}) against the baseline model ({args.base_model_path}).",
        "\n## Evaluation Metrics Summary\n",
        "| Model | Retrieval Precision@3 | ROUGE-L F1 Score | BERTScore F1 | Avg Inference Time (s) |",
        "|---|---|---|---|---|"
    ]
    
    if "base" in results:
        res = results["base"]
        report_lines.append(
            f"| Base Model ({os.path.basename(args.base_model_path)}) | {res['precision']:.4f} | {res['rouge_l']:.4f} | {res['bert_score']:.4f} | {res['avg_time']:.2f}s |"
        )
    if "tuned" in results:
        res = results["tuned"]
        report_lines.append(
            f"| Fine-Tuned CMS Model | {res['precision']:.4f} | {res['rouge_l']:.4f} | {res['bert_score']:.4f} | {res['avg_time']:.2f}s |"
        )
        
    report_lines.append("\n## Analysis and Verdict\n")
    if "base" in results and "tuned" in results:
        b_score = results["base"]["bert_score"]
        t_score = results["tuned"]["bert_score"]
        improvement = t_score - b_score
        verdict = "Fine-tuned model IMPROVES performance" if improvement > 0 else "Fine-tuned model does not improve performance"
        report_lines.append(
            f"Verdict: **{verdict}** (semantic similarity delta: `+{improvement:.4f}`).\n"
            f"- Base Model BERTScore: `{b_score:.4f}`\n"
            f"- Fine-Tuned CMS Model BERTScore: `{t_score:.4f}`\n"
        )
    else:
        report_lines.append("Reference baseline evaluation skipped. Results only show tuned model performance.\n")
        
    # Sample questions and outputs
    report_lines.append("\n## Sample Answers Comparison\n")
    sample_indices = [0, len(eval_data)//2, len(eval_data)-1] if len(eval_data) > 2 else range(len(eval_data))
    for idx in sample_indices:
        if idx < len(eval_data):
            q = eval_data[idx]["question"]
            gt = eval_data[idx]["ground_truth"]
            report_lines.append(f"### Question: *\"{q}\"*\n")
            report_lines.append(f"**Ground Truth Context**:\n> {gt}\n")
            if "base" in results:
                report_lines.append(f"**Base Model Answer**:\n> {results['base']['answers'][idx]}\n")
            if "tuned" in results:
                report_lines.append(f"**Fine-Tuned CMS Answer**:\n> {results['tuned']['answers'][idx]}\n")
            report_lines.append("---\n")
            
    # Save the report
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nEvaluation complete! Report written to {args.report_path}")

if __name__ == "__main__":
    main()
