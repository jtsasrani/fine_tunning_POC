import os
import re
import time
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
import faiss
from sentence_transformers import CrossEncoder
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Ensure HuggingFace models can download if not in cache (e.g. cross-encoder, base model)
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Global references for models & database
db_data = None
faiss_index = None
embed_tokenizer = None
embed_model = None
reranker = None
tokenizer_dict = {}
model_dict = {}

# System prompt for RAG answers
SYSTEM_PROMPT = (
    "You are an expert Decision Maker helper for the DWP CMG. Answer the user's question "
    "accurately and professionally using ONLY the provided official policy contexts. "
    "State exact rules, percentages, and paragraph numbers if they are present in the context. "
    "If the context does not contain the information needed to answer the question, state clearly "
    "that the policy manual does not provide sufficient details. Do not assume or extrapolate."
)

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

def retrieve_context_hybrid_rerank(query, top_k=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. FAISS Search
    query_prefixed = f"Represent this sentence for searching relevant passages: {query}"
    query_emb = get_query_embedding(query_prefixed, embed_model, embed_tokenizer, device)
    query_emb_np = query_emb.cpu().numpy().astype('float32')
    
    distances, indices = faiss_index.search(query_emb_np, 10)
    faiss_hits = indices[0]
    
    # 2. Keyword Search
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

def initialize_models():
    global db_data, faiss_index, embed_tokenizer, embed_model, reranker, tokenizer_dict, model_dict
    
    db_file = "vector_db.pt"
    faiss_file = "vector_db.index"
    
    if not os.path.exists(db_file) or not os.path.exists(faiss_file):
        raise FileNotFoundError("Vector database or FAISS index not found. Run 05_build_vector_db.py first.")
        
    print("Loading vector database metadata...", flush=True)
    db_data = torch.load(db_file)
    
    print("Loading FAISS index...", flush=True)
    faiss_index = faiss.read_index(faiss_file)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model and Cross-Encoder on {device}...", flush=True)
    embed_tokenizer = AutoTokenizer.from_pretrained(db_data["model_name"])
    embed_model = AutoModel.from_pretrained(db_data["model_name"]).to(device)
    embed_model.eval()
    
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    # Configure bitsandbytes 4-bit quantization to load models concurrently in VRAM
    print("Configuring 4-bit quantization for reader models...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    models_to_load = {
        "mistral_tuned": "./trained_models/mistral-7b-cmg-qlora_merged",
        "qwen_tuned": "./trained_models/qwen-7b-cmg-qlora_merged",
        "mistral_base": "mistralai/Mistral-7B-Instruct-v0.3"
    }
    
    for name, path in models_to_load.items():
        print(f"Loading reader model '{name}' from '{path}'...", flush=True)
        tokenizer_dict[name] = AutoTokenizer.from_pretrained(path)
        model_dict[name] = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model_dict[name].eval()
        print(f"Loaded '{name}' reader model successfully!", flush=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/query/retrieve", methods=["POST"])
def handle_retrieve():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query provided."}), 400
        
    try:
        t0 = time.time()
        retrieved_items = retrieve_context_hybrid_rerank(query, top_k=3)
        retrieval_time = time.time() - t0
        
        retrieved_data = []
        for r in retrieved_items:
            chunk = r["chunk"]
            score = r["score"]
            retrieved_data.append({
                "paragraph_id": chunk["paragraph_id"],
                "source_doc": chunk["source_doc"],
                "text": chunk["text"],
                "score": round(score, 4)
            })
            
        return jsonify({
            "query": query,
            "retrieved_contexts": retrieved_data,
            "retrieval_time": round(retrieval_time, 3)
        })
    except Exception as e:
        print(f"Error in retrieve: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/query/generate", methods=["POST"])
def handle_generate():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    model_name = data.get("model", "").strip()
    contexts = data.get("contexts", [])
    use_rag = data.get("use_rag", True)
    
    if not query or not model_name:
        return jsonify({"error": "Query and model name are required."}), 400
        
    if model_name not in model_dict:
        return jsonify({"error": f"Model '{model_name}' is not loaded."}), 400
        
    try:
        tok = tokenizer_dict[model_name]
        mod = model_dict[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if use_rag:
            # Format contexts into a single string
            context_parts = []
            for ctx in contexts:
                context_parts.append(f"Paragraph {ctx.get('paragraph_id')} (from {ctx.get('source_doc')}):\n{ctx.get('text')}")
            context_str = "\n\n".join(context_parts)
            
            user_content = f"Contexts:\n{context_str}\n\nQuestion: {query}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
        else:
            messages = [
                {"role": "user", "content": query}
            ]
            
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        t_start = time.time()
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = mod.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tok.eos_token_id
            )
        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_len:]
        response = tok.decode(generated_tokens, skip_special_tokens=True).strip()
        elapsed_time = time.time() - t_start
        
        return jsonify({
            "model": model_name,
            "response": response,
            "time": round(elapsed_time, 2)
        })
        
    except Exception as e:
        print(f"Error in generate for {model_name}: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask application. Preloading GPU models in 4-bit...", flush=True)
    try:
        initialize_models()
        print("All models loaded successfully on GPU. Web server running on http://127.0.0.1:5000", flush=True)
        app.run(host="127.0.0.1", port=5000, debug=False)
    except Exception as e:
        print(f"Failed to initialize models or start server: {e}", flush=True)

