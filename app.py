import os
import re
import time
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel

app = Flask(__name__)

# Configure local execution (Disable offline check if HuggingFace cache is present)
# In case internet is offline, load from cached files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Configuration
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_PATH = "./cmg_lora_weights"
DB_FILE = "vector_db.pt"

# Global references for models & database
db_data = None
embed_tokenizer = None
embed_model = None
tokenizer = None
base_model = None
tuned_model = None

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
    inputs = tokenizer(prompt, return_tensors="pt")
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
    return response

def initialize_models():
    global db_data, embed_tokenizer, embed_model, tokenizer, base_model, tuned_model
    
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Vector database {DB_FILE} not found. Please run 05_build_vector_db.py first.")
        
    print("Loading vector database...", flush=True)
    db_data = torch.load(DB_FILE)
    
    print("Loading embedding model...", flush=True)
    embed_tokenizer = AutoTokenizer.from_pretrained(db_data["model_name"])
    embed_model = AutoModel.from_pretrained(db_data["model_name"])
    embed_model.eval()
    
    print("Loading base reader model tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print("Loading base reader model instance (Config B)...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    base_model.eval()
    
    print("Loading tuned reader model instance (Config A & C)...", flush=True)
    tuned_model_base = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    if os.path.exists(ADAPTER_PATH):
        tuned_model = PeftModel.from_pretrained(tuned_model_base, ADAPTER_PATH)
        tuned_model.eval()
        print("LoRA weights loaded successfully!", flush=True)
    else:
        tuned_model = None
        print("Warning: LoRA weights not found, Config A & C will run in fallback (base) mode.", flush=True)

# Endpoint: Serve homepage
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint: Run evaluation matrix for a question
@app.route("/api/query", methods=["POST"])
def handle_query():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query provided."}), 400
        
    try:
        # 1. Retrieve Context
        t0 = time.time()
        retrieved_items = retrieve_context_hybrid(query, db_data, embed_model, embed_tokenizer, top_k=3)
        retrieval_time = time.time() - t0
        
        # Prepare context blocks
        context_parts = []
        retrieved_data = []
        for r in retrieved_items:
            chunk = r["chunk"]
            score = r["score"]
            context_parts.append(f"Paragraph {chunk['paragraph_id']} (from {chunk['source_doc']}):\n{chunk['text']}")
            retrieved_data.append({
                "paragraph_id": chunk["paragraph_id"],
                "source_doc": chunk["source_doc"],
                "text": chunk["text"],
                "score": round(score, 4)
            })
        context_str = "\n\n".join(context_parts)
        
        # 2. Config A: Tuned Model (No RAG)
        t_a = time.time()
        messages_a = [{"role": "user", "content": query}]
        prompt_a = tokenizer.apply_chat_template(messages_a, tokenize=False, add_generation_prompt=True)
        if tuned_model:
            response_a = generate_response(tuned_model, tokenizer, prompt_a, temp=0.7)
        else:
            response_a = generate_response(base_model, tokenizer, prompt_a, temp=0.7) + " [Fallback: LoRA weights not loaded]"
        time_a = time.time() - t_a
        
        # 3. Config B: Base Model + RAG
        t_b = time.time()
        user_content = f"Contexts:\n{context_str}\n\nQuestion: {query}"
        messages_b = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        prompt_b = tokenizer.apply_chat_template(messages_b, tokenize=False, add_generation_prompt=True)
        response_b = generate_response(base_model, tokenizer, prompt_b, temp=0.3)
        time_b = time.time() - t_b
        
        # 4. Config C: Tuned Model + RAG
        t_c = time.time()
        if tuned_model:
            response_c = generate_response(tuned_model, tokenizer, prompt_b, temp=0.3)
        else:
            response_c = response_b + " [Fallback: LoRA weights not loaded]"
        time_c = time.time() - t_c
        
        return jsonify({
            "query": query,
            "retrieved_contexts": retrieved_data,
            "retrieval_time": round(retrieval_time, 3),
            "config_a": {
                "response": response_a,
                "time": round(time_a, 2)
            },
            "config_b": {
                "response": response_b,
                "time": round(time_b, 2)
            },
            "config_c": {
                "response": response_c,
                "time": round(time_c, 2)
            }
        })
        
    except Exception as e:
        print(f"Error handling query: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask application. Preloading models on CPU (this may take up to 20s)...", flush=True)
    try:
        initialize_models()
        print("Models loaded successfully. Web server running on http://127.0.0.1:5000", flush=True)
        app.run(host="127.0.0.1", port=5000, debug=False)
    except Exception as e:
        print(f"Failed to initialize models or start server: {e}", flush=True)
