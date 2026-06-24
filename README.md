# DWP CMS Decision Support Suite v2.0

> **Audience:** Caseworkers, Decision Makers, DevOps, and Machine Learning Engineers.  
> **Environment:** AWS EC2 GPU Production Mode (NVIDIA A10G VRAM $\approx$ 24GB).

---

## 🚀 Overview

The **DWP CMS Decision Support Suite v2.0** is an enterprise-grade AI assistant designed to support UK Child Maintenance Service (CMS) Decision Makers. By reading and learning from official policy guides, statutory regulations, and system manuals, the suite provides caseworkers with context-grounded, factually accurate answers to complex policy questions, accompanied by direct paragraph citations and clickable original documents.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DWP CMS Suite Architecture                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [503 Policy PDFs] ──▶ [Text Ingestion] ──▶ [Cleaned Paragraph Chunks]  │
│                                                     │                  │
│  ┌──────────────────────────────────────────────────┘                  │
│  ▼                                                                     │
│  [BGE-base-v1.5 Embed] ──▶ [FAISS Similarity Index & Metadata DB]      │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ User Casework Query                                              │  │
│  │  ├─▶ 1. Hybrid Search (FAISS Cosine Similarity + BM25 Keyword)    │  │
│  │  ├─▶ 2. Cross-Encoder Rerank (MiniLM-L6)                         │  │
│  │  ├─▶ 3. RAG Score Thresholding (Passages < -1.0 filtered out)     │  │
│  │  └─▶ 4. Answer Generation (Fine-Tuned Qwen-14B via vLLM)          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Production Web Dashboard

The web interface is structured as a **3-panel workspace** optimized for caseworker workflows:
1. **Panel 1 (AI Assistant)**: A conversation module where Decision Makers submit casework queries. It generates context-grounded answers citing paragraph reference numbers. Passages with a reranking score $< -1.0$ are visually flagged as **(Excluded)** and automatically stripped from the model's generation context to prevent hallucinations.
2. **Panel 2 (RAG Explorer)**: A search module to query the underlying policy manuals directly, display rerank scores, and inspect raw text passages.
3. **Panel 3 (Interactive Metrics)**: A live dashboard tracking fine-tuning loss convergence, validation cycles, and pipeline status.

*Document Access*: Every cited document badge is a clickable link. Clicking the name appends the API key and opens the PDF or HTML file directly in a new browser tab, rendered inline for fast reference.

---

## ⚙️ Core Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Base Language Model** | `Qwen/Qwen2.5-14B-Instruct` | Fine-tuned 14-billion parameter model optimized for policy reasoning. |
| **Teacher Generator** | `Qwen/Qwen2.5-32B-Instruct` | Used to generate synthetic Q&A pairs for dataset bootstrapping. |
| **Fine-tuning Protocol** | QLoRA (Unsloth) | GPU-optimized 4-bit base quantization with LoRA adapters ($r=32$, $\alpha=64$). |
| **Embedding Model** | `BAAI/bge-base-en-v1.5` | Generates 768-dimensional normalized similarity vectors. |
| **Vector Index** | FAISS Flat Index (`IndexFlatIP`) | Fast inner-product similarity library for vector search. |
| **Rerank Model** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Computes query-passage relevance scores. |
| **Inference Server** | vLLM Engine | High-throughput, low-latency, quantized 4-bit model hosting. |
| **Web Server** | Flask & Nginx | REST API backend and reverse proxy serving with SSL termination. |

---

## 📂 Codebase Pipeline: Script-by-Script

Here is the execution sequence for the ingestion, database creation, training, and web-serving pipeline:

### 1. `01_ingest_all_pdfs.py` — Document Processing
* **What it does:** Extracts text from 503 PDF manuals in `source_pdfs/`. Uses frequency metrics to strip recurring page headers/footers, cleans characters, and chunks text by paragraph.
* **Output:** `data/real_chunks.jsonl` (contains text paragraphs and unique `paragraph_id`s).

### 2. `01b_ingest_qa_pairs.py` — Q&A Dataset Ingestion
* **What it does:** Extracts instruction-following QA pairs matching specific policy chapter prefixes (e.g. 17xxx to 95xxx).

### 3. `05_build_vector_db.py` — Database Indexing
* **What it does:** Segments paragraphs exceeding 500 tokens using a sliding window (100 token overlap). Generates 768-dimensional BGE embeddings and builds the FAISS similarity index.
* **Output:** `vector_db.index` (vector weights) and `vector_db.pt` (metadata and text mappings).

### 4. `07_generate_training_data_local.py` — Synthetic Data Bootstrapping
* **What it does:** Feeds extracted text chunks into the **Qwen2.5-32B-Instruct** teacher model. Generates scenario-based casework questions and validates them. Requires $\approx 19.5$ GB VRAM and runs for $\approx 34$ hours on an NVIDIA A10G.
* **Output:** `data/llm_generated_training_data.jsonl`.

### 5. `08_merge_training_data.py` — Deduplication & Dataset Splits
* **What it does:** Uses MinHash LSH (Jaccard similarity threshold = 0.85) to remove duplicate questions, shuffles records, and splits the data into 90% training / 10% validation datasets.
* **Output:** `data/train_split.jsonl`, `data/val_split.jsonl`, and `data/evaluation_set.jsonl` (35 gold evaluation items).

### 6. `02_train_qlora_gpu.py` — GPU Fine-Tuning
* **What it does:** Fine-tunes the base Qwen-14B model using Unsloth. Configures adapters, bakes the DWP CMS system prompt and target context into the training sequences, runs SFTTrainer for 2400 steps, and outputs merged 16-bit weights.
* **Output:** `trained_models/qwen-14b-cms-qlora_merged/`.

### 7. `09_evaluate_model.py` — Benchmark Evaluation
* **What it does:** Computes model performance against the evaluation set, reporting ROUGE-L, BERTScore F1, and average latency.

### 8. `app.py` — Production Web Server
* **What it does:** Launches the Flask server, preloads embedding/reranking weights, initializes the vLLM engine to load the tuned model, and hosts the RAG API and document-serving routes.

---

## 🚀 How to Run on AWS

### 1. Launch instance with DLAMI
Launch an EC2 instance (e.g. `g5.xlarge` for DEV/quantized mode, or `g5.12xlarge` for PROD/16-bit Tensor Parallel mode) running the official **Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)**.

### 2. Setup the Repository & Environment
SSH into the instance, clone the repository, and activate the pre-configured PyTorch conda environment:
```bash
# Clone and enter project directory
git clone https://github.com/jtsasrani/fine_tunning_POC.git
cd fine_tunning_POC

# Activate GPU-optimized PyTorch conda environment
conda activate pytorch

# Install additional packages
pip install vllm==0.23.0 flask faiss-cpu sentence-transformers peft
```

### 3. Load Model Weights
Download the merged weights folder from your Hugging Face private repository (refer to the model transfer guide in `data/` for setup instructions) into:
`./trained_models/qwen-14b-cms-qlora_merged`

### 4. Start the Server
Configure environment flags (disabling the FlashInfer JIT sampler) and start the backend Flask server in the background:
```bash
export VLLM_USE_FLASHINFER_SAMPLER="0"

mkdir -p data
nohup python3 app.py > data/flask_server.log 2>&1 &
```

Monitor loading progress by tailing the server logs:
```bash
tail -f data/flask_server.log
```

---

## 🔒 Production Nginx Reverse Proxy with SSL

For production environments, place the application behind Nginx to handle SSL certificates and static resource caching.

### 1. Nginx Server Block Example
Create `/etc/nginx/sites-available/dwp-decision-support`:
```nginx
server {
    listen 80;
    server_name dwp-cms-suite.client.internal;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name dwp-cms-suite.client.internal;

    ssl_certificate /etc/ssl/certs/dwp_cms_suite.crt;
    ssl_certificate_key /etc/ssl/private/dwp_cms_suite.key;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Static files served directly by Nginx
    location /static/ {
        alias /home/ubuntu/dwp-cmg-finetune/static/;
        expires 30d;
    }

    # API Proxy to Flask (with streaming buffering disabled)
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

### 2. Enable Configuration
Link the configuration to active sites and reload the Nginx daemon:
```bash
sudo ln -s /etc/nginx/sites-available/dwp-decision-support /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```
