# Programmatic API Access Guide: DWP CMS Decision Support Suite (Llama-3.1-8B)

This guide provides technical specifications and code examples for integrating the DWP Child Maintenance Service (CMS) RAG and Inference APIs with external case management systems.

---

## 🔒 Authentication & Base Configuration

All requests to the Decision Support Suite APIs must include the API key header for authentication:

* **Base URL:** `http://13.220.148.137:5000`
* **Port:** `5000`
* **Header Name:** `X-API-Key`
* **API Key:** `dwp-cmg-sec-key-7d9a1f8c`

> [!WARNING]
> Do not expose the API key in client-side code or public repositories. Use environment variables on the client application to store the key securely.

---

## 📡 API Endpoints

### 1. Retrieve Relevant Context (`POST /api/query/retrieve`)
Performs hybrid lexical (BM25) and dense vector (FAISS via `bge-base-en-v1.5`) searches on the 603 DWP policy manuals. Candidates are reranked on CPU using the Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) before returning.

#### Request Headers:
```http
Content-Type: application/json
X-API-Key: dwp-cmg-sec-key-7d9a1f8c
```

#### Request Body (JSON):
```json
{
  "query": "What happens if the paying parent has additional income?",
  "history": []
}
```
* `query` *(string, required)*: The caseworker's query text.
* `history` *(array, optional)*: Previous conversation messages to enable query condensation for multi-turn RAG.

#### Response Body (JSON):
```json
{
  "is_condensed": false,
  "query": "What happens if the paying parent has additional income?",
  "search_query": "What happens if the paying parent has additional income?",
  "retrieval_time": 0.124,
  "retrieved_contexts": [
    {
      "paragraph_id": "24024",
      "source_doc": "Variance-Review.pdf",
      "text": "If the paying parent has additional income that was not initially considered...",
      "score": 1.4589
    }
  ]
}
```

---

### 2. Generate Model Answer (`POST /api/query/generate`)
Invokes the fine-tuned **Llama-3.1-8B-Instruct** model hosted on the SageMaker real-time endpoint (`ml.g5.2xlarge`) using the TGI container.

#### Request Headers:
```http
Content-Type: application/json
X-API-Key: dwp-cmg-sec-key-7d9a1f8c
```

#### Request Body (JSON):
```json
{
  "query": "What happens if the paying parent has additional income?",
  "model": "dwp-cmg-llama-8b-endpoint-v2",
  "contexts": [
    {
      "paragraph_id": "24024",
      "source_doc": "Variance-Review.pdf",
      "text": "If the paying parent has additional income that was not initially considered...",
      "score": 1.4589
    }
  ],
  "use_rag": true,
  "messages": []
}
```
* `query` *(string, required)*: The caseworker's question.
* `model` *(string, required)*: Must be `dwp-cmg-llama-8b-endpoint-v2`.
* `contexts` *(array, required if use_rag=true)*: Context array returned from the `/api/query/retrieve` endpoint.
* `use_rag` *(boolean, optional)*: Default is `true`. Set to `false` to query the base fine-tuned model without context.
* `messages` *(array, optional)*: Conversation history (limit to last 10 messages for context safety).

#### Response Body (JSON):
```json
{
  "model": "dwp-cmg-llama-8b-endpoint-v2",
  "response": "If the paying parent has additional income that was not initially considered in the maintenance calculation, the caseworker should request updated financial documentation to include these earnings. According to Paragraph 24024, any new income sources must be verified...",
  "time": 2.73
}
```

---

## 💻 Code Examples

### Curl (2-Step RAG Pipeline)

#### Step 1: Retrieve Context
```bash
curl -X POST http://13.220.148.137:5000/api/query/retrieve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dwp-cmg-sec-key-7d9a1f8c" \
  -d '{"query": "What happens if the paying parent has additional income?"}'
```

#### Step 2: Generate Answer
```bash
curl -X POST http://13.220.148.137:5000/api/query/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dwp-cmg-sec-key-7d9a1f8c" \
  -d '{
    "query": "What happens if the paying parent has additional income?",
    "model": "dwp-cmg-llama-8b-endpoint-v2",
    "contexts": [
      {
        "paragraph_id": "24024",
        "source_doc": "Variance-Review.pdf",
        "text": "If the paying parent has additional income...",
        "score": 1.45
      }
    ]
  }'
```

---

### Python (Combined Integration Script)

```python
import requests
import json

BASE_URL = "http://13.220.148.137:5000"
API_KEY = "dwp-cmg-sec-key-7d9a1f8c"
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def query_rag_system(question: str) -> str:
    # 1. Retrieve relevant contexts
    retrieve_url = f"{BASE_URL}/api/query/retrieve"
    retrieve_payload = {"query": question}
    
    retrieve_resp = requests.post(retrieve_url, headers=HEADERS, json=retrieve_payload)
    if retrieve_resp.status_code != 200:
        raise RuntimeError(f"Retrieval failed: {retrieve_resp.text}")
        
    contexts = retrieve_resp.json().get("retrieved_contexts", [])
    print(f"[RAG] Successfully retrieved {len(contexts)} paragraphs.")
    
    # 2. Generate answer
    generate_url = f"{BASE_URL}/api/query/generate"
    generate_payload = {
        "query": question,
        "model": "dwp-cmg-llama-8b-endpoint-v2",
        "contexts": contexts,
        "use_rag": True
    }
    
    generate_resp = requests.post(generate_url, headers=HEADERS, json=generate_payload)
    if generate_resp.status_code != 200:
        raise RuntimeError(f"Generation failed: {generate_resp.text}")
        
    return generate_resp.json().get("response")

# Execution Example
if __name__ == "__main__":
    query = "What happens if the paying parent has additional income?"
    answer = query_rag_system(query)
    print("\n--- Final Model Output ---")
    print(answer)
```
