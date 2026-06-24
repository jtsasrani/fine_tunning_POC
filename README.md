# DWP CMS Decision Support Suite v2.0

> **Audience:** Caseworkers, Decision Makers, and ML Engineers.  
> **Environment:** AWS EC2 GPU Instance (NVIDIA A10G VRAM $\approx$ 24GB).

---

## 🚀 Overview

The **DWP CMS Decision Support Suite v2.0** is an AI assistant proof-of-concept designed to support UK Child Maintenance Service (CMS) Decision Makers. By reading and learning from official policy guides, statutory regulations, and system manuals, the suite provides caseworkers with context-grounded, factually accurate answers to complex policy questions, accompanied by direct paragraph citations and clickable original documents.

The system has been upgraded from the initial CPU-only Proof of Concept (SmolLM-360M) to a full **Production GPU RAG + Inference Pipeline** using a fine-tuned **Qwen2.5-14B-Instruct** model.

---

## ✨ Features (Already Accomplished)

* **Fine-Tuned Qwen-14B Model**: Trained locally on AWS GPU using QLoRA, and merged at 16-bit precision to maintain high semantic reasoning and CMS terminology accuracy.
* **Teacher-Generated Dataset**: Leveraged the larger **Qwen2.5-32B-Instruct** model as a teacher to generate synthetic instruction-following Q&A pairs from raw policy paragraphs.
* **Hybrid RAG Search**: Combines FAISS cosine similarity vector search (using `BAAI/bge-base-en-v1.5` embeddings) and BM25 lexical keyword matching to retrieve relevant context.
* **RAG Score Thresholding**: Uses a Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`) to score candidate passages. Passages with a reranking score below `-1.0` are stripped from the model's generation context to prevent hallucination, while remaining visible (dimmed) in the UI citations for user reference.
* **3-Panel Web Workspace**:
  * *Panel 1 (AI Assistant)*: Chat client featuring visual bubble streams, session resets, and citations.
  * *Panel 2 (RAG Explorer)*: Direct database search module to inspect FAISS similarity and rerank scores.
  * *Panel 3 (Interactive Metrics)*: Displays convergence training loss history and pipeline validation epochs.
* **Clickable Document Badges**: Source badges serve as links to load original policy guides inline in a new tab. Files are served securely from the server's local storage via a `/api/documents/` route.

---

## 📂 Processed Source Material

We ingested and indexed **603 official DWP policy PDF manuals** located in `source_pdfs/`.

### Core Data & Weights
* **Dataset**: Unique training (`data/train_split.jsonl`) and validation (`data/val_split.jsonl`) splits obtained after MinHash LSH deduplication.
* **Vector Database**: FAISS flat index (`vector_db.index`) and metadata tensor (`vector_db.pt`) built from segmented paragraphs.
* **Merged Model Weights**: Residing on the EC2 GPU instance at `trained_models/qwen-14b-cms-qlora_merged/`.

---

## 🛠️ The Complete Pipeline (Script-by-Script)

You can run the entire pipeline from scratch by executing these scripts in order:

### Step 1: `01_ingest_all_pdfs.py` — Ingest policy documents
* Extracts text from the 603 PDFs, dynamically filters repeating headers/footers, and outputs paragraph records to `data/real_chunks.jsonl`.

### Step 2: `07_generate_training_data_local.py` — Generate Q&A training pairs
* Uses the Qwen-32B teacher model in 4-bit to generate QA pairs. (Runs a generation pass and a validation check pass; takes $\approx 34$ hours on a single A10G GPU).

### Step 3: `08_merge_training_data.py` — Deduplicate datasets
* Applies MinHash LSH (Jaccard similarity threshold = 0.85) to remove duplicates, and splits unique pairs into 90% train / 10% validation sets.

### Step 4: `05_build_vector_db.py` — Index the search library
* Segments paragraphs exceeding 500 tokens (100 token overlap), extracts normalized BGE-base embeddings, and builds the FAISS similarity index (`vector_db.index`).

### Step 5: `02_train_qlora_gpu.py` — Fine-tune model
* Trains Qwen-14B in 4-bit via Unsloth/QLoRA on the unique splits (2400 steps), merges adapters into base weights at 16-bit, and exports the merged model.

### Step 6: `09_evaluate_model.py` — Run validation checks
* Runs benchmark evaluations against the evaluation set, reporting ROUGE-L, BERTScore F1, and average latency.

### Step 7: `app.py` — Web Dashboard Server
* Launches the API and dashboard backend.

---

## 🚀 Running the Active Service on AWS

To start, stop, or check the active web application server on the GPU EC2 instance, use these operations:

### 1. Connect to the EC2 Instance
Open your local terminal and log in using the keypair:
```bash
ssh -i .\gpu_poc_EUR.pem ubuntu@i-09e7b81b6184aebaa
```

### 2. Start the Service
Activate the pre-configured GPU PyTorch environment, set the environment paths (disabling the FlashInfer JIT compiler to prevent CUDA headers errors), and start the Flask web app in the background:
```bash
cd /home/ubuntu/dwp-cmg-finetune
conda activate pytorch

export VLLM_USE_FLASHINFER_SAMPLER="0"
nohup /opt/pytorch/bin/python3 app.py > data/flask_server.log 2>&1 &
```

Monitor model loading progress (takes $\approx$ 3-4 minutes to load shards and compile CUDA graphs):
```bash
tail -f data/flask_server.log
```

### 3. Stop the Service
To terminate the Flask app and release GPU memory (killing both Flask and vLLM subprocesses):
```bash
# Kill active python server
pkill -f app.py
```

### 4. Access the Dashboard locally
Expose the server locally by creating an SSH port-forwarding tunnel on port `5000`:
```bash
ssh -i .\gpu_poc_EUR.pem -N -L 5000:localhost:5000 ubuntu@i-09e7b81b6184aebaa
```
Once connected, load `http://localhost:5000` in your web browser.
