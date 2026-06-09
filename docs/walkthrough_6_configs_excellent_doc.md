# Walkthrough - DWP CMG RAG & Fine-Tuning Comparison (Phase 1 UI Matrix)

This walkthrough documents the design, implementation, and verification of the expanded DWP CMG Comparative Matrix UI, running directly on the connected EC2 GPU instance.

The interface allows comparing **6 configurations side-by-side** in a responsive 3x2 grid, contrasting the effects of both retrieval-grounding (RAG) and model tuning.

---

## Architecture of the 6-Configuration Matrix

To support side-by-side evaluation without resource starvation, the Flask backend and the frontend script execute requests sequentially.

```mermaid
graph TD
    User[Browser UI Client] -->|1. POST /api/query/retrieve| Flask[Flask Backend]
    Flask -->|2. Search DB| VectorDB[(FAISS + Keyword DB)]
    VectorDB -->|3. Contexts| Flask
    Flask -->|4. Return Contexts| User
    
    User -->|5. POST /api/query/generate | Flask
    subgraph GPU Causal Inference (Sequential)
        direction TB
        M1[Mistral-7B Base + RAG]
        M2[Mistral-7B Base - No RAG]
        M3[Mistral-7B Tuned + RAG]
        M4[Mistral-7B Tuned - No RAG]
        Q1[Qwen-7B Tuned + RAG]
        Q2[Qwen-7B Tuned - No RAG]
    end
    Flask -->|6. Execute 1-by-1| GPU
    GPU -->|7. Card Results| User
```

---

## File Structure & Implementations

1.  **Flask Backend**: [app.py](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/app.py)
    *   Preloads vector database metadata and FAISS index.
    *   Loads three 7B reader models (`mistral_tuned`, `qwen_tuned`, and `mistral_base`) on the EC2 GPU using `bitsandbytes` 4-bit quantization to fit them all in VRAM.
    *   Exposes `POST /api/query/retrieve` for hybrid retrieval.
    *   Exposes `POST /api/query/generate` with options for `use_rag` (True/False) and specific `model` selection to allow the client to request responses card-by-card.
2.  **HTML Comparative Layout**: [templates/index.html](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/templates/index.html)
    *   Implements a 3x2 matrix comparing:
        1. **Mistral Base + RAG** vs **Mistral Base (No RAG)**
        2. **Mistral Tuned + RAG** vs **Mistral Tuned (No RAG)**
        3. **Qwen Tuned + RAG** vs **Qwen Tuned (No RAG)**
    *   Arranges them so that RAG vs. No RAG configurations are vertically aligned.
3.  **CSS Styling & Matrix Grid**: [static/style.css](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/static/style.css)
    *   Defines a responsive 3-column CSS Grid.
    *   Adjusts card height to a compact layout to minimize vertical scroll.
    *   Maintains consistent styling elements (glassmorphism, status badges, response timers).
4.  **Sequential JS Controller**: [static/script.js](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/static/script.js)
    *   Runs the retrieval step first to fetch the shared reference contexts.
    *   Dispatches the 6 generation requests **sequentially** (one-by-one) to the EC2 server.
    *   This prevents VRAM fragmentation and thread lock on the GPU, while updating cards progressively to keep the user engaged.

---

## Verification & Execution Results

### 1. Server Initialization Logs
The server successfully booted on the EC2 GPU, preloading the models in 4-bit:
```text
Starting Flask application. Preloading GPU models in 4-bit...
Loading vector database metadata...
Loading FAISS index...
Loading embedding model and Cross-Encoder on cuda...
Configuring 4-bit quantization for reader models...
Loading reader model 'mistral_tuned' from './trained_models/mistral-7b-cmg-qlora_merged'...
Loaded 'mistral_tuned' reader model successfully!
Loading reader model 'qwen_tuned' from './trained_models/qwen-7b-cmg-qlora_merged'...
Loaded 'qwen_tuned' reader model successfully!
Loading reader model 'mistral_base' from 'mistralai/Mistral-7B-Instruct-v0.3'...
Loaded 'mistral_base' reader model successfully!
All models loaded successfully on GPU. Web server running on http://127.0.0.1:5000
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

### 2. Live API Testing
A script tested the endpoint response behavior directly on the EC2 host:

*   **Hybrid Retrieval API**:
    *   Query: `"How is a variance request for assets handled?"`
    *   Status: `200 OK`
    *   Latency: `1.08s`
    *   Returns context snippets from `volume-6-collection-and-enforcement-chapters-49-95.pdf`.

*   **Grounded RAG Generation (`mistral_tuned` + RAG)**:
    *   Status: `200 OK`
    *   Latency: `3.88s`
    *   Result: *"A variance request for assets is handled by evaluating the value of the asset. Specifically, if the value of the asset exceeds £65,000, then it can be taken into consideration..."* (Grounded in context).

*   **Parametric Generation (`qwen_tuned` - No RAG)**:
    *   Status: `200 OK`
    *   Latency: `4.59s`
    *   Result: *"A variation can be requested at any time during the life of an application... Refer to Chapter 98 - Variations Overview..."* (Relies purely on model parameters).

---

## Client Access, Production (Gunicorn) & Security Verification

To transition from local testing to a secure production-ready setup for your peers, the system was upgraded with **Gunicorn** and **API Key Authentication**:

### 1. Production Gunicorn Config
The application now runs via **Gunicorn WSGI server** bound to all network interfaces on port 5000, managing timeouts up to 300 seconds (5 minutes) for slow LLM generations:
```python
# gunicorn.conf.py
bind = "0.0.0.0:5000"
workers = 1             # Protects VRAM from collisions
timeout = 300           # 5-minute timeout limit
keepalive = 5
```

### 2. API Key Protection
An authentication mechanism has been added to block unauthorized public access to the GPU resources.
* **API Key Generated**: `dwp-cmg-sec-key-7d9a1f8c`
* **Dashboard Auth Field**: A password field was added to the top-right of the dashboard. The dashboard saves the API key to the browser's `localStorage` and automatically sends it with every query.
* **Header Authorization**: Standard clients must pass `X-API-Key: dwp-cmg-sec-key-7d9a1f8c` or append `?api_key=dwp-cmg-sec-key-7d9a1f8c` as a query parameter.

---

## Live Auth Verification Results

We verified the authentication logic directly on the host using the validation script [test_api_auth.py](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/scratch/test_api_auth.py):

* **Valid API Key Request (`200 OK`)**:
  ```python
  requests.post(url, json={"query": "variance"}, headers={"X-API-Key": "dwp-cmg-sec-key-7d9a1f8c"})
  # Result: 200 OK (Returns matching RAG policy contexts correctly)
  ```

* **Invalid API Key Request (`401 Unauthorized`)**:
  ```python
  requests.post(url, json={"query": "variance"}, headers={"X-API-Key": "wrong-key"})
  # Result: 401 Unauthorized (Response: {'error': 'Unauthorized: Invalid or missing API key.'})
  ```

---

## Peer Access Setup (CloudFront + Security Groups)

Now that Gunicorn is running on port 5000, do the following to expose the app to your peers via CloudFront:

1. **Security Group Edit**:
   * Add a Custom TCP inbound rule for port **5000** allowed from anywhere (`0.0.0.0/0`) or restricted to corporate ranges.
2. **Create CloudFront Distribution**:
   * **Origin Domain**: Your EC2 DNS name.
   * **Origin HTTP Port**: Change from `80` to `5000`.
   * **Origin Protocol**: **HTTP Only** (CloudFront connects to port 5000 over HTTP, while clients connect to CloudFront over HTTPS).
   * **Viewer Protocol Policy**: **Redirect HTTP to HTTPS**.
   * **Cache Policy**: **CachingDisabled** (essential so queries do not return cached answers).
   * **Origin Request Policy**: **AllViewer** (so client headers are forwarded).
3. **Share Details**:
   * Share the `https://<cloudfront-domain>` link and the API Key (`dwp-cmg-sec-key-7d9a1f8c`) with your peers. They can paste the API Key directly into the header of the web page to run queries, or query it programmatically in their own scripts!

