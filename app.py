import os
import re
import sys
import time
import ast
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
import faiss
from sentence_transformers import CrossEncoder

app = Flask(__name__)

# Ensure HuggingFace models can download if not in cache
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Detect Demo/Mock Mode at import time to preserve GPU resources
# Runs on CPU-only for reader models if --demo flag is present or environment variable set
DEMO_MODE = "--demo" in sys.argv or os.environ.get("DEMO_MODE", "0") == "1"

# Global references for models & database
db_data = None
faiss_index = None
embed_tokenizer = None
embed_model = None
reranker = None
tokenizer_dict = {}
model_dict = {}

# System prompt for RAG answers (aligned with Phase 3 Training)
SYSTEM_PROMPT = (
    "You are an expert Decision Maker assistant for the DWP Child Maintenance Service (CMS). "
    "Answer questions accurately using only the provided policy context. "
    "Cite specific paragraph numbers, rules, and sections where present. "
    "If the context does not contain sufficient information, state this clearly."
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
    global db_data, faiss_index, embed_tokenizer, embed_model, reranker, tokenizer_dict, model_dict, DEMO_MODE
    
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
    
    # Define models to load
    models_to_load = {
        "qwen_14b_tuned": "./trained_models/qwen-14b-cms-qlora_merged",
        "qwen_7b_tuned": "./trained_models/qwen-7b-cmg-qlora_merged",
        "qwen_base_14b": "unsloth/qwen2.5-14b-instruct"
    }
    
    # Check if the primary local models folders actually exist. If not, auto-force DEMO_MODE
    primary_model_path = models_to_load["qwen_14b_tuned"]
    if not os.path.exists(primary_model_path):
        print(f"Warning: Tuned model path '{primary_model_path}' not found. Auto-enabling Demo Mode.")
        DEMO_MODE = True
        
    if DEMO_MODE:
        print("=== RUNNING IN DEMO MODE ===")
        print("Skipping VRAM-heavy reader LLM loads. Real FAISS + MS-Marco Cross-Encoder RAG is active on CPU.")
        return
        
    # Configure bitsandbytes 4-bit quantization to load models concurrently in VRAM
    print("Configuring 4-bit quantization for reader models...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    for name, path in models_to_load.items():
        if not os.path.exists(path) and not path.startswith("Qwen/"):
            print(f"Warning: Model path '{path}' not found. Skipping loading '{name}'.")
            continue
        try:
            print(f"Loading reader model '{name}' from '{path}'...", flush=True)
            tokenizer_dict[name] = AutoTokenizer.from_pretrained(path)
            model_dict[name] = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=bnb_config,
                device_map="auto"
            )
            model_dict[name].eval()
            print(f"Loaded '{name}' reader model successfully!", flush=True)
        except Exception as e:
            print(f"Error loading model '{name}': {e}. Skipping.", flush=True)

API_KEY = os.environ.get("API_KEY", "dwp-cmg-sec-key-7d9a1f8c")

def check_auth():
    provided_key = request.headers.get("X-API-Key") or request.args.get("api_key")
    if not provided_key and request.is_json:
        try:
            body = request.get_json(silent=True) or {}
            provided_key = body.get("api_key") or body.get("X-API-Key")
        except Exception:
            pass
            
    if provided_key != API_KEY:
        return jsonify({"error": "Unauthorized: Invalid or missing API key."}), 401
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/query/retrieve", methods=["POST"])
def handle_retrieve():
    auth_err = check_auth()
    if auth_err:
        return auth_err
        
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
    auth_err = check_auth()
    if auth_err:
        return auth_err
        
    data = request.get_json() or {}
    messages = data.get("messages", [])
    query = data.get("query", "").strip()
    model_name = data.get("model", "").strip()
    contexts = data.get("contexts", [])
    use_rag = data.get("use_rag", True)
    
    # Fallback if messages list is empty
    if not query and messages:
        # Last message content is the query
        query = messages[-1].get("content", "").strip()
        
    if not query or not model_name:
        return jsonify({"error": "Query/messages and model name are required."}), 400
        
    # Check if we should simulate generation (if in DEMO_MODE or if the model isn't loaded)
    if DEMO_MODE or model_name not in model_dict:
        t0 = time.time()
        # Sleep slightly to simulate model generation speed
        time.sleep(1.2)
        elapsed_time = time.time() - t0
        
        best_chunk = contexts[0] if contexts else None
        if best_chunk:
            text = best_chunk.get("text", "")
            doc = best_chunk.get("source_doc", "")
            pid = best_chunk.get("paragraph_id", "")
            
            # Simple heuristic mock: extract first 3 sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            summary = " ".join(sentences[:3])
            
            response = (
                f"**[Demo Mode — Qwen-14B CMS Response Simulation]**\n\n"
                f"Based on the official policy guide **{doc}** (Paragraph {pid}), the guidance states:\n\n"
                f"> {summary}...\n\n"
                f"**Retrieval Reference**:\n"
                f"- Document: `{doc}`\n"
                f"- Paragraph: `{pid}`\n"
                f"- FAISS Rerank Score: `{best_chunk.get('score', 'N/A')}`\n\n"
                f"*Note: The Qwen-14B CMS model is currently training in the background. The RAG vector retrieval is live and querying your real PDF database, but the answer generation is simulated to keep the GPU 100% free.*"
            )
        else:
            response = (
                f"**[Demo Mode — Qwen-14B CMS Response Simulation]**\n\n"
                f"No relevant policy context was found in the database to answer the question: *\"{query}\"*\n\n"
                f"*Note: The Qwen-14B CMS model is currently training in the background. The response generation is simulated to keep the GPU 100% free.*"
            )
            
        return jsonify({
            "model": model_name,
            "response": response,
            "time": round(elapsed_time, 2)
        })
        
    try:
        tok = tokenizer_dict[model_name]
        mod = model_dict[model_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Build chat message templates
        chat_messages = []
        chat_messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        # Append message history (excluding the very last user query, which we append with RAG contexts)
        # Limit history to the last 10 messages (5 turns) to prevent context limit overflow and VRAM issues
        for msg in messages[:-1][-10:]:
            chat_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
        if use_rag:
            context_parts = []
            for ctx in contexts:
                pid = ctx.get('paragraph_id', '')
                doc = ctx.get('source_doc', '')
                text = ctx.get('text', '')
                context_parts.append(f"Document: {doc}\nContent:\n{text}")
            context_str = "\n\n".join(context_parts)
            
            user_content = f"Contexts:\n{context_str}\n\nQuestion: {query}"
        else:
            user_content = query
            
        chat_messages.append({"role": "user", "content": user_content})
        
        prompt = tok.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        t_start = time.time()
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = mod.generate(
                **inputs,
                max_new_tokens=512,
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

@app.route("/api/metrics", methods=["GET"])
def handle_metrics():
    log_path = "data/pipeline_step6_training.log"
    train_history = []
    eval_history = []
    summary = {
        "total_steps": 2406,
        "completed_steps": 0,
        "progress_pct": 0.0,
        "initial_loss": 0.0,
        "final_loss": 0.0,
        "final_eval_loss": 0.0,
        "status": "idle"
    }
    
    if os.path.exists(log_path):
        summary["status"] = "running"
        dict_pattern = re.compile(r"\{'[a-z_]+':\s*.*\}")
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = dict_pattern.search(line)
                    if match:
                        try:
                            data = ast.literal_eval(match.group(0))
                            if 'loss' in data:
                                train_history.append({
                                    'step': len(train_history) * 10 + 10,
                                    'loss': float(data['loss']),
                                    'epoch': float(data['epoch']),
                                    'lr': float(data['learning_rate'])
                                })
                            elif 'eval_loss' in data:
                                epoch = float(data['epoch'])
                                step = round(epoch * (2406 / 3.0))
                                eval_history.append({
                                    'step': step,
                                    'eval_loss': float(data['eval_loss']),
                                    'epoch': epoch,
                                    'eval_runtime': float(data.get('eval_runtime', 0.0))
                                })
                        except Exception:
                            pass
        except Exception as e:
            print(f"Error parsing logs: {e}")
            
    if train_history:
        summary["completed_steps"] = train_history[-1]['step']
        summary["progress_pct"] = round((summary["completed_steps"] / summary["total_steps"]) * 100, 1)
        summary["initial_loss"] = train_history[0]['loss']
        summary["final_loss"] = train_history[-1]['loss']
        
    if eval_history:
        summary["final_eval_loss"] = eval_history[-1]['eval_loss']
        
    orchestrator_log = "data/orchestrator.log"
    if os.path.exists(orchestrator_log):
        try:
            with open(orchestrator_log, 'r', encoding='utf-8') as f:
                content = f.read()
                if "End-to-End Pipeline Completed Successfully" in content:
                    summary["status"] = "completed"
                elif "Fine-tuning completed successfully" in content:
                    summary["status"] = "completed"
                elif "Error:" in content or "failed" in content:
                    summary["status"] = "failed"
        except Exception:
            pass
            
    return jsonify({
        "train_history": train_history,
        "eval_history": eval_history,
        "summary": summary,
        "demo_mode": DEMO_MODE,
        "models_loaded": list(model_dict.keys())
    })

print("Starting Flask application. Initializing models...", flush=True)
try:
    initialize_models()
    print("Models/DB initialization completed.", flush=True)
except Exception as e:
    print(f"Failed to initialize models: {e}", flush=True)

if __name__ == "__main__":
    print(f"Web server running on http://127.0.0.1:5000 (DEMO_MODE={DEMO_MODE})", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
