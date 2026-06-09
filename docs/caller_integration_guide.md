# DWP CMG AI Models - Caller Integration Guide

This guide details how external applications, scripts, and callers can securely query the fine-tuned DWP CMG policy models (`mistral_tuned` and `qwen_tuned`) using the production CloudFront endpoint.

---

## 1. Authentication & Base Settings

To prevent unauthorized access, all API requests to the model endpoints require authentication.

*   **Base URL**: `https://hidden.cloudfront.net`
*   **Security Header**: `hidden - contact creator`
*   **Alternative (Query Param)**: You can append `hidden - contact creator` to the URL.

---

## 2. API Endpoints

### Endpoint 1: Retrieve Policy Contexts (RAG Search)
If you are implementing a **RAG-based** caller, use this endpoint to fetch the top 3 most relevant manual paragraphs.

*   **Route**: `POST /api/query/retrieve`
*   **Request JSON Payload**:
    ```json
    {
      "query": "What is the policy for a paying parent's 25% income variance?"
    }
    ```
*   **Response JSON (200 OK)**:
    ```json
    {
      "query": "What is the policy for a paying parent's 25% income variance?",
      "retrieval_time": 0.825,
      "retrieved_contexts": [
        {
          "paragraph_id": "44026_part1",
          "score": -7.9519,
          "source_doc": "volume-5-reviews-and-appeals.pdf",
          "text": "when dms have obtained the relevant evidence... if the current income figure is at least 25% different..."
        }
      ]
    }
    ```

---

### Endpoint 2: Generate Answer (Model Inference)
Use this endpoint to generate text answers from either the Mistral or Qwen fine-tuned models.

*   **Route**: `POST /api/query/generate`
*   **Request JSON Payload Keys**:
    *   `model` (string, **Required**): Specify `"mistral_tuned"` or `"qwen_tuned"`.
    *   `query` (string, **Required**): The question or prompt text.
    *   `use_rag` (boolean, Optional, default `true`): If `true`, the model answers using the provided contexts. If `false`, it answers purely from fine-tuned weights.
    *   `contexts` (array, Optional): The array of `retrieved_contexts` returned by the `/api/query/retrieve` endpoint. Required only if `use_rag` is `true`.

#### Example Payload (With RAG):
```json
{
  "model": "mistral_tuned",
  "query": "What is the policy for a paying parent's 25% income variance?",
  "use_rag": true,
  "contexts": [
    {
      "paragraph_id": "44026_part1",
      "source_doc": "volume-5-reviews-and-appeals.pdf",
      "text": "when dms have obtained the relevant evidence... if the current income figure is at least 25% different..."
    }
  ]
}
```

#### Example Payload (Direct Inference - No RAG):
```json
{
  "model": "qwen_tuned",
  "query": "Explain the role of administrative Liability Orders under the 2023 Act.",
  "use_rag": false
}
```

*   **Response JSON (200 OK)**:
    ```json
    {
      "model": "mistral_tuned",
      "time": 3.88,
      "response": "Under the Child Maintenance policy, a variation application is triggered when there is a 25% difference..."
    }
    ```

---

## 3. Code Integration Examples

### Python Integration (Recommended)
```python
import requests

BASE_URL = "https://hidden.cloudfront.net"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": "hidden"
}

def query_model(prompt: str, model_name: str = "mistral_tuned", use_rag: bool = True):
    # Step 1: Retrieve context if RAG is requested
    contexts = []
    if use_rag:
        ret_res = requests.post(f"{BASE_URL}/api/query/retrieve", json={"query": prompt}, headers=HEADERS)
        if ret_res.status_code == 200:
            contexts = ret_res.json().get("retrieved_contexts", [])
            
    # Step 2: Generate response
    payload = {
        "model": model_name,
        "query": prompt,
        "use_rag": use_rag,
        "contexts": contexts
    }
    gen_res = requests.post(f"{BASE_URL}/api/query/generate", json=payload, headers=HEADERS)
    if gen_res.status_code == 200:
        return gen_res.json().get("response")
    else:
        return f"Error {gen_res.status_code}: {gen_res.text}"

# Example invocation
ans = query_model("What is the pension contribution rule?", model_name="qwen_tuned", use_rag=True)
print("Answer:", ans)
```

### cURL (Command Line)
```bash
# Direct Inference on Mistral Tuned
curl -X POST https://hidden.cloudfront.net/api/query/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: hiddeb" \
  -d '{"model": "mistral_tuned", "query": "Explain pension variance", "use_rag": false}'
```

### Node.js (JavaScript fetch)
```javascript
const BASE_URL = "https://hidden.cloudfront.net";
const headers = {
  "Content-Type": "application/json",
  "X-API-Key": "hidden"
};

async function queryModel(prompt, modelName = "mistral_tuned") {
  // Direct query without RAG for simplicity
  const response = await fetch(`${BASE_URL}/api/query/generate`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: modelName,
      query: prompt,
      use_rag: false
    })
  });
  const data = await response.json();
  console.log("Model Response:", data.response);
}
```

---

## 4. Usage Rules & Constraints

1.  **Concurrency Limit**: The GPU server runs on a single process worker to prevent memory thrashing. Concurrent requests from different callers will be **queued** and processed sequentially. Avoid making concurrent burst calls.
2.  **Max Output Length**: Generation limits are capped at **300 tokens** per query to ensure quick response times (typically 2–5 seconds).
