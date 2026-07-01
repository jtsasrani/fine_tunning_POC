import os
import sys
import re
import time
import ast
import json
import boto3
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import AutoTokenizer, AutoModel
import faiss
from sentence_transformers import CrossEncoder

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Ensure HuggingFace models can download if not in cache
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# Detect Demo/Mock Mode at import time to preserve resources
DEMO_MODE = "--demo" in sys.argv or os.environ.get("DEMO_MODE", "0") == "1"

# SageMaker configuration
SAGEMAKER_ENDPOINT_NAME = "dwp-cmg-llama-8b-endpoint-v2"
AWS_REGION = "us-east-1"
sagemaker_runtime = None

# Global references for models & database
db_data = None
faiss_index = None
embed_tokenizer = None
embed_model = None
reranker = None
model_dict = {}  # Keeps app interface compatible with model listings

# System prompt for RAG answers (aligned with Llama 3.1 fine-tuning)
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

def get_sagemaker_runtime_client():
    global sagemaker_runtime
    if sagemaker_runtime is None:
        sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    return sagemaker_runtime

def call_sagemaker_endpoint(prompt, max_tokens=512, temperature=0.3):
    client = get_sagemaker_runtime_client()
    # TGI container parameters
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "stop": ["<|eot_id|>"]
        }
    }
    
    try:
        response = client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        
        # TGI returns [{"generated_text": "..."}]
        if isinstance(result, list) and len(result) > 0:
            gen_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            gen_text = result.get("generated_text", "")
        else:
            gen_text = str(result)
            
        # TGI might return the prompt with the generation appended — strip it if it starts with the prompt
        # (Though newer TGI versions return only the generated completion when setting return_full_text=False)
        if gen_text.startswith(prompt):
            gen_text = gen_text[len(prompt):].strip()
        return gen_text.strip()
    except Exception as e:
        print(f"Error calling SageMaker endpoint: {e}", flush=True)
        raise e


def initialize_models():
    global db_data, faiss_index, embed_tokenizer, embed_model, reranker, model_dict, DEMO_MODE
    
    db_file = "vector_db.pt"
    faiss_file = "vector_db.index"
    
    if not os.path.exists(db_file) or not os.path.exists(faiss_file):
        raise FileNotFoundError("Vector database or FAISS index not found. Run 05_build_vector_db.py first.")
        
    print("Loading vector database metadata...", flush=True)
    db_data = torch.load(db_file, map_location="cpu")
    
    print("Loading FAISS index...", flush=True)
    faiss_index = faiss.read_index(faiss_file)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model and Cross-Encoder on {device}...", flush=True)
    embed_tokenizer = AutoTokenizer.from_pretrained(db_data["model_name"])
    embed_model = AutoModel.from_pretrained(db_data["model_name"]).to(device)
    embed_model.eval()
    
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    if DEMO_MODE:
        print("=== RUNNING IN DEMO MODE ===")
        print("Skipping SageMaker endpoint connectivity checks. Mock/simulation response mode is active.")
        return
        
    print(f"Verifying SageMaker Endpoint '{SAGEMAKER_ENDPOINT_NAME}'...", flush=True)
    try:
        sm = boto3.client("sagemaker", region_name=AWS_REGION)
        desc = sm.describe_endpoint(EndpointName=SAGEMAKER_ENDPOINT_NAME)
        status = desc["EndpointStatus"]
        print(f"SageMaker Endpoint Status: {status}", flush=True)
        if status == "InService":
            model_dict[SAGEMAKER_ENDPOINT_NAME] = True
            print("SageMaker Endpoint is active and registered in dashboard.", flush=True)
        else:
            print(f"Warning: SageMaker Endpoint is '{status}' (not InService). Dashboard will fall back to simulation.", flush=True)
    except Exception as e:
        print(f"Warning: Could not connect to SageMaker endpoint: {e}. Dashboard will run in simulation mode.", flush=True)

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
    history = data.get("history", []) or data.get("messages", [])
    if not query:
        return jsonify({"error": "Empty query provided."}), 400
        
    try:
        t0 = time.time()
        
        # Query Condensation for multi-turn RAG
        search_query = query
        is_condensed = False
        clean_history = [m for m in history if m.get("role") in ["user", "assistant"]]
        
        # We only condense if there is active history beyond the current query
        if not DEMO_MODE and SAGEMAKER_ENDPOINT_NAME in model_dict and len(clean_history) > 1:
            try:
                prior_history = clean_history[:-1]
                
                system_prompt = (
                    "You are a search query optimizer for the DWP Child Maintenance Service (CMS). "
                    "Analyze the conversation history and the new follow-up question, and output a single, "
                    "concise standalone search query in plain text. "
                    "The search query must combine the context from the history and the new question to retrieve relevant policy documents. "
                    "Do not write any introductory text, explanations, or conversational responses. Only output the rephrased query."
                )
                
                # Format Llama 3.1 prompt manually
                prompt = "<|begin_of_text|>"
                prompt += f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n"
                for msg in prior_history[-4:]: # Limit to last 2 turns to keep it fast
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>\n"
                prompt += f"<|start_header_id|>user<|end_header_id|>\nRephrase this follow-up question to a standalone search query: {query}<|eot_id|>\n"
                prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
                
                gen_text = call_sagemaker_endpoint(prompt, max_tokens=40, temperature=0.0)
                # Strip any wrapping quotes
                gen_text = gen_text.replace('"', '').replace("'", "").strip()
                if gen_text:
                    search_query = gen_text
                    is_condensed = True
                    print(f"[Query Condensation] Original: '{query}' -> Condensed: '{search_query}'", flush=True)
            except Exception as cond_err:
                print(f"[Query Condensation Error] Failed to condense, falling back to original query: {cond_err}", flush=True)
        
        retrieved_items = retrieve_context_hybrid_rerank(search_query, top_k=3)
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
            "search_query": search_query,
            "is_condensed": is_condensed,
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
        
    # Fallback legacy mapping for cached frontend sessions
    if model_name == "qwen_14b_tuned":
        model_name = SAGEMAKER_ENDPOINT_NAME
        
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
                f"**[Demo Mode — Llama-3.1-8B CMS Response Simulation]**\n\n"
                f"Based on the official policy guide **{doc}** (Paragraph {pid}), the guidance states:\n\n"
                f"> {summary}...\n\n"
                f"**Retrieval Reference**:\n"
                f"- Document: `{doc}`\n"
                f"- Paragraph: `{pid}`\n"
                f"- FAISS Rerank Score: `{best_chunk.get('score', 'N/A')}`\n\n"
                f"*Note: The Llama-3.1-8B CMS model is currently training on SageMaker. The RAG vector retrieval is live and querying your real PDF database, but the answer generation is simulated to keep the GPU 100% free.*"
            )
        else:
            response = (
                f"**[Demo Mode — Llama-3.1-8B CMS Response Simulation]**\n\n"
                f"No relevant policy context was found in the database to answer the question: *\"{query}\"*\n\n"
                f"*Note: The Llama-3.1-8B CMS model is currently training on SageMaker. The response generation is simulated to keep the GPU 100% free.*"
            )
            
        return jsonify({
            "model": model_name,
            "response": response,
            "time": round(elapsed_time, 2)
        })
        
    try:
        # Build prompt manually using Llama 3.1 special tokens
        prompt = "<|begin_of_text|>"
        prompt += f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>\n"
        
        # Append message history (excluding the very last user query, which we append with RAG contexts)
        # Limit history to the last 10 messages (5 turns) to prevent context limit overflow
        for msg in messages[:-1][-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>\n"
            
        if use_rag:
            # Filter out contexts with score < -1.0 to avoid low-quality or irrelevant context
            filtered_contexts = [ctx for ctx in contexts if ctx.get('score', 0.0) >= -1.0]
            
            context_parts = []
            for ctx in filtered_contexts:
                pid = ctx.get('paragraph_id', '')
                doc = ctx.get('source_doc', '')
                text = ctx.get('text', '')
                context_parts.append(f"Paragraph ID: {pid}\nDocument: {doc}\nContent:\n{text}")
            context_str = "\n\n".join(context_parts)
            
            user_content = f"Context:\n{context_str}\n\nQuestion: {query}"
        else:
            user_content = query
            
        prompt += f"<|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>\n"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        
        t_start = time.time()
        # Generate answer using SageMaker endpoint client
        response = call_sagemaker_endpoint(prompt, max_tokens=512, temperature=0.3)
        elapsed_time = time.time() - t_start
        
        return jsonify({
            "model": model_name,
            "response": response,
            "time": round(elapsed_time, 2)
        })
        
    except Exception as e:
        print(f"Error in generate for {model_name}: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/documents/<path:filename>", methods=["GET"])
def serve_document(filename):
    auth_err = check_auth()
    if auth_err:
        return auth_err
    directory = os.path.abspath("./source_pdfs")
    try:
        return send_from_directory(directory, filename)
    except Exception as e:
        print(f"Error serving document {filename}: {e}", flush=True)
        return jsonify({"error": f"File not found: {filename}"}), 404

def get_training_metrics_from_cloudwatch(job_name):
    cw = boto3.client("logs", region_name=AWS_REGION)
    log_group = "/aws/sagemaker/TrainingJobs"
    
    # 1. Describe log streams to find the stream for this job
    try:
        streams = cw.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=job_name,
            limit=1
        )
        if not streams.get("logStreams"):
            return [], [], 0
        stream_name = streams["logStreams"][0]["logStreamName"]
    except Exception:
        return [], [], 0
        
    # 2. Get log events
    train_history = []
    eval_history = []
    completed_steps = 0
    
    try:
        events = cw.get_log_events(
            logGroupName=log_group,
            logStreamName=stream_name,
            limit=1000, # Get log lines to scan for progress
            startFromHead=True
        )
        
        # SFTTrainer logs like:
        # {'loss': 2.4931, 'grad_norm': 1.0162, 'learning_rate': 0.0001, 'epoch': 0.01}
        # tqdm output like: 0%|          | 8/1722 [02:28<8:48:59, 18.52s/it]
        tqdm_pattern = re.compile(r"(\d+)%\|.*\|\s+(\d+)/(\d+)")
        dict_pattern = re.compile(r"\{'loss':\s*.*\}")
        eval_pattern = re.compile(r"\{'eval_loss':\s*.*\}")
        
        for event in events.get("events", []):
            message = event.get("message", "")
            
            # Check for steps count
            tqdm_match = tqdm_pattern.search(message)
            if tqdm_match:
                completed_steps = int(tqdm_match.group(2))
                
            # Check for training loss dict
            dict_match = dict_pattern.search(message)
            if dict_match:
                try:
                    data = ast.literal_eval(dict_match.group(0))
                    train_history.append({
                        'step': len(train_history) * 5 + 5, # SFTTrainer logs every 5 steps
                        'loss': float(data['loss']),
                        'epoch': float(data['epoch']),
                        'lr': float(data['learning_rate'])
                    })
                except Exception:
                    pass
                    
            # Check for eval loss
            eval_match = eval_pattern.search(message)
            if eval_match:
                try:
                    data = ast.literal_eval(eval_match.group(0))
                    eval_history.append({
                        'step': len(train_history) * 5,
                        'eval_loss': float(data['eval_loss']),
                        'epoch': float(data['epoch']),
                    })
                except Exception:
                    pass
    except Exception:
        pass
        
    return train_history, eval_history, completed_steps

@app.route("/api/metrics", methods=["GET"])
def handle_metrics():
    summary = {
        "total_steps": 2296,
        "completed_steps": 0,
        "progress_pct": 0.0,
        "initial_loss": 0.0,
        "final_loss": 0.0,
        "final_eval_loss": 0.0,
        "status": "idle"
    }
    train_history = []
    eval_history = []
    
    if DEMO_MODE:
        return jsonify({
            "train_history": [],
            "eval_history": [],
            "summary": summary,
            "demo_mode": True,
            "models_loaded": []
        })
        
    try:
        sm = boto3.client("sagemaker", region_name=AWS_REGION)
        # Find latest training job matching prefix
        jobs = sm.list_training_jobs(
            NameContains="dwp-cmg-sft-",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )
        
        if jobs.get("TrainingJobSummaries"):
            job_summary = jobs["TrainingJobSummaries"][0]
            job_name = job_summary["TrainingJobName"]
            desc = sm.describe_training_job(TrainingJobName=job_name)
            
            # Map status
            sm_status = desc["TrainingJobStatus"]
            if sm_status == "InProgress":
                summary["status"] = "running"
            elif sm_status == "Completed":
                summary["status"] = "completed"
            elif sm_status == "Failed":
                summary["status"] = "failed"
            else:
                summary["status"] = sm_status.lower()
                
            # Get metrics from CloudWatch logs
            train_history, eval_history, completed_steps = get_training_metrics_from_cloudwatch(job_name)
            summary["completed_steps"] = completed_steps
            summary["progress_pct"] = round((completed_steps / summary["total_steps"]) * 100, 1)
            
            if train_history:
                summary["initial_loss"] = train_history[0]['loss']
                summary["final_loss"] = train_history[-1]['loss']
            if eval_history:
                summary["final_eval_loss"] = eval_history[-1]['eval_loss']
    except Exception as e:
        print(f"Error fetching metrics from SageMaker/CloudWatch: {e}")
        
    return jsonify({
        "train_history": train_history,
        "eval_history": eval_history,
        "summary": summary,
        "demo_mode": DEMO_MODE,
        "models_loaded": list(model_dict.keys())
    })

if __name__ == "__main__":
    print("Starting Flask application. Initializing models...", flush=True)
    try:
        initialize_models()
        print("Models/DB initialization completed.", flush=True)
    except Exception as e:
        print(f"Failed to initialize models: {e}", flush=True)
        
    print(f"Web server running on http://0.0.0.0:5000 (DEMO_MODE={DEMO_MODE})", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False)
