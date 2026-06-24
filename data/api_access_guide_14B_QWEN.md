# DWP CMS Decision Support Model — API Access Guide

This guide details how to programmatically query the fine-tuned `qwen_14b_tuned` model and RAG pipeline.

---

## 🔒 Authentication

All API endpoints are protected by an API Key. You must supply this key using one of the following methods:

1. **HTTP Header (Recommended)**:
   ```http
   X-API-Key: dwp-cmg-sec-key-7d9a1f8c
   ```
2. **JSON Payload Field**:
   Include `"api_key": "dwp-cmg-sec-key-7d9a1f8c"` inside your POST request body.
3. **Query Parameter**:
   Append `?api_key=dwp-cmg-sec-key-7d9a1f8c` to the endpoint URL.

---

## 🌐 Base URLs

* **Local Forwarded Tunnel**: `http://127.0.0.1:5000`
* **EC2 Instance Direct**: `http://35.179.154.45:5000`
* **CloudFront Distribution**: `http://d9ym4p48160x5.cloudfront.net` (subject to Origin Port and Security Group config)

---

## 🚀 API Endpoints

### 1. RAG Context Retrieval (`POST /api/query/retrieve`)
Retrieves the most semantically relevant policy contexts from the FAISS database utilizing the MiniLM embedding model and Cross-Encoder semantic reranker.

#### Request Headers
```http
Content-Type: application/json
X-API-Key: dwp-cmg-sec-key-7d9a1f8c
```

#### Request Payload
```json
{
  "query": "What are the rules regarding NRP vs RP child maintenance calculations?"
}
```

#### Response Example
```json
{
  "query": "What are the rules regarding NRP vs RP child maintenance calculations?",
  "retrieval_time": 0.612,
  "retrieved_contexts": [
    {
      "paragraph_id": "61025_chunk_0",
      "source_doc": "Volume_2_Chapter_6.pdf",
      "text": "The calculation rules regarding NRP vs RP child maintenance require evaluating the active shared care calculations...",
      "score": 0.9412
    }
  ]
}
```

---

### 2. Full Answer Generation (`POST /api/query/generate`)
Executes the full RAG pipeline: retrieves relevant contexts, constructs a prompt injected with CMS system-level guidelines, and generates a response using the fine-tuned model via the high-performance vLLM engine on the GPU.

#### Request Headers
```http
Content-Type: application/json
X-API-Key: dwp-cmg-sec-key-7d9a1f8c
```

#### Request Payload
```json
{
  "model": "qwen_14b_tuned",
  "query": "What are the rules regarding NRP vs RP child maintenance calculations?",
  "use_rag": true,
  "contexts": [],
  "messages": []
}
```
* **`model`** *(Required)*: Must be set to `"qwen_14b_tuned"`.
* **`query`** *(Required)*: The user question.
* **`use_rag`** *(Optional)*: Set to `false` to query the model directly without inserting context from the vector database.
* **`contexts`** *(Optional)*: If you already retrieved contexts from the `/api/query/retrieve` endpoint, you can pass them directly in this array to bypass the retrieval step.
* **`messages`** *(Optional)*: An array of previous message history objects (e.g. `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`) for multi-turn chat memory. The context is automatically bounded to the last 10 turns on the backend to optimize VRAM.

#### Response Example
```json
{
  "model": "qwen_14b_tuned",
  "response": "Based on CMS guideline Volume 2, Chapter 6 (Paragraph 61025), NRP vs RP calculations are determined by...\n",
  "time": 2.15
}
```

---

## 💻 Code Examples

### Curl
```bash
curl -X POST http://35.179.154.45:5000/api/query/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dwp-cmg-sec-key-7d9a1f8c" \
  -d '{
    "model": "qwen_14b_tuned",
    "query": "What are the rules regarding NRP vs RP child maintenance calculations?"
  }'
```

### Python
```python
import requests

url = "http://35.179.154.45:5000/api/query/generate"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "dwp-cmg-sec-key-7d9a1f8c"
}
payload = {
    "model": "qwen_14b_tuned",
    "query": "What are the rules regarding NRP vs RP child maintenance calculations?",
    "use_rag": True
}

response = requests.post(url, json=payload, headers=headers)
if response.status_code == 200:
    data = response.json()
    print("Response:", data["response"])
    print(f"Time Taken: {data['time']} seconds")
else:
    print("Error:", response.status_code, response.text)
```
