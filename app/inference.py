import os
import re
import time
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from sentence_transformers import CrossEncoder

class DWPInferenceEngine:
    def __init__(self, db_file="vector_db.pt", faiss_file="vector_db.index"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Vector DB metadata & FAISS index
        self.db_data = torch.load(db_file)
        self.faiss_index = faiss.read_index(faiss_file)
        
        # Load Embedding Model
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.db_data["model_name"])
        self.embed_model = AutoModel.from_pretrained(self.db_data["model_name"]).to(self.device)
        self.embed_model.eval()
        
        # Load Cross-Encoder Reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device)
        
        # BitsAndBytes 4-bit Quantization Config for GPU Reader Models
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer_dict = {}
        self.model_dict = {}
        
        models_to_load = {
            "mistral_tuned": "./trained_models/mistral-7b-cmg-qlora_merged",
            "qwen_tuned": "./trained_models/qwen-7b-cmg-qlora_merged",
            "mistral_base": "mistralai/Mistral-7B-Instruct-v0.3"
        }
        
        for name, path in models_to_load.items():
            print(f"Loading reader model '{name}' from '{path}'...")
            self.tokenizer_dict[name] = AutoTokenizer.from_pretrained(path)
            self.model_dict[name] = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.model_dict[name].eval()
            print(f"Loaded '{name}' successfully.")
            
        self.system_prompt = (
            "You are an expert Decision Maker helper for the DWP CMS. Answer the user's question "
            "accurately and professionally using ONLY the provided official policy contexts. "
            "State exact rules, percentages, and paragraph numbers if they are present in the context. "
            "If the context does not contain the information needed to answer the question, state clearly "
            "that the policy manual does not provide sufficient details. Do not assume or extrapolate."
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_query_embedding(self, query):
        encoded_input = self.embed_tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def get_keyword_score(self, query, paragraph_text):
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        stop_words = {"how", "are", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "under", "regarding", "about", "what", "does", "have", "when", "explain", "applies", "treated"}
        query_keywords = query_words - stop_words
        
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

    def retrieve_context(self, query, top_k=3):
        # 1. FAISS Search
        query_prefixed = f"Represent this sentence for searching relevant passages: {query}"
        query_emb = self.get_query_embedding(query_prefixed)
        query_emb_np = query_emb.cpu().numpy().astype('float32')
        
        distances, indices = self.faiss_index.search(query_emb_np, 10)
        faiss_hits = indices[0]
        
        # 2. Keyword Search
        keyword_scores = []
        for idx, chunk in enumerate(self.db_data["chunks"]):
            k_score = self.get_keyword_score(query, chunk["text"])
            keyword_scores.append((k_score, idx))
        keyword_scores.sort(key=lambda x: x[0], reverse=True)
        keyword_hits = [idx for score, idx in keyword_scores[:10] if score > 0.0]
        
        # 3. Union Candidate Pool
        candidate_indices = list(set(list(faiss_hits) + keyword_hits))
        candidate_indices = [idx for idx in candidate_indices if idx >= 0 and idx < len(self.db_data["chunks"])]
        
        if not candidate_indices:
            return []
            
        candidates = [self.db_data["chunks"][idx] for idx in candidate_indices]
        
        # 4. Cross-Encoder Reranking
        pairs = [[query, c["formatted_text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        ranked_results = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
        
        retrieved = []
        for score, chunk in ranked_results[:top_k]:
            retrieved.append({
                "chunk": chunk,
                "score": float(score)
            })
        return retrieved

    def generate(self, model_name, query, retrieved_items, max_tokens=512):
        if model_name not in self.model_dict:
            return "Model not loaded.", 0.0
            
        t_start = time.time()
        tok = self.tokenizer_dict[model_name]
        mod = self.model_dict[model_name]
        
        # Format Context
        context_parts = []
        for r in retrieved_items:
            chunk = r["chunk"]
            pid = chunk.get('paragraph_id', '')
            doc = chunk.get('source_doc', '')
            text = chunk.get('text', '')
            if pid.startswith("L_"):
                context_parts.append(f"Context from {doc}:\n{text}")
            else:
                context_parts.append(f"Paragraph {pid} (from {doc}):\n{text}")
        context_str = "\n\n".join(context_parts)
        
        user_content = f"Contexts:\n{context_str}\n\nQuestion: {query}"
        
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True
        )
        
        inputs = tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = mod.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tok.eos_token_id
            )
        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_len:]
        response = tok.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return response, time.time() - t_start
