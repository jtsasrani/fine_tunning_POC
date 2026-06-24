# DWP CMS Decision Support Suite: RAG Database Maintenance & Re-indexing Runbook

This runbook provides step-by-step procedures to maintain, update, and re-index the RAG (Retrieval-Augmented Generation) policy database. Use this guide whenever policy manuals are revised, new guidelines are introduced, or existing documents are deprecated.

---

## 📂 Overview of RAG DB Components

The RAG search system relies on three synchronized assets in the `/home/ubuntu/dwp-cmg-finetune` directory:
1. `source_pdfs/`: The storage folder containing the original statutory PDF and HTML manuals served to user browsers.
2. `vector_db.index`: The FAISS index containing the mathematical representations (dense embeddings) of text passages.
3. `vector_db.pt`: The PyTorch metadata archive linking FAISS index positions back to original text chunks, paragraph IDs, and source filenames.

> [!IMPORTANT]
> **Constraint**: `vector_db.index` and `vector_db.pt` must always be built from the exact same execution run. Swapping one without the other will align query matches to incorrect document text (causing context mismatches).

---

## 🔄 Re-indexing Procedure (Step-by-Step)

Follow this procedure to update the knowledge base.

### Step 1: Prepare Source Files
1. Navigate to the codebase directory:
   ```bash
   cd /home/ubuntu/dwp-cmg-finetune
   ```
2. **Add New Manuals**: Place the updated or new policy PDF files directly into the `source_pdfs/` folder.
   * *Filenames*: Ensure filenames contain only alphanumeric characters, dashes, or underscores (e.g. `Volume-8-Calculation-Amendments.pdf`). Avoid spaces or special characters.
3. **Remove Old Manuals**: Delete any outdated PDF files from the `source_pdfs/` directory to prevent stale policy references from being retrieved.

### Step 2: Extract Chunks & Clean Text
Run the ingestion script to parse the PDF manuals. This script strips repeating headers and footers (page-repetition metrics), cleans character encodings, and splits pages into paragraphs:
```bash
# Activate the PyTorch environment
conda activate pytorch

# Execute ingestion
python3 01_ingest_all_pdfs.py
```
* **Inputs**: Files in `source_pdfs/`
* **Outputs**: `data/real_chunks.jsonl`
* **Verification**: Verify that the command outputs the correct number of raw chunks. Check the output file size:
  ```bash
  ls -lh data/real_chunks.jsonl
  ```

### Step 3: Segment and Embed Chunks (Rebuild FAISS Index)
Execute the database building script. This script segments large paragraphs exceeding 500 tokens (with a 100-token overlap), generates normalized embeddings using `BAAI/bge-base-en-v1.5`, and builds the FAISS inner-product flat index:
```bash
python3 05_build_vector_db.py
```
* **Inputs**: `data/real_chunks.jsonl`
* **Outputs**:
  * `vector_db.index` (new FAISS vector data)
  * `vector_db.pt` (new metadata database)

---

## ⚡ Cold Swap & Deployment (Zero-Downtime Swap)

To update the database on the live service without causing client query crashes, follow this deployment swap flow:

### 1. Build Assets in Temporary Workspace
Run the indexing pipeline inside a subfolder or local workspace first to prevent overwriting the active files while the web app is running:
```bash
# 1. Create a staging directories
mkdir -p staging/data

# 2. Copy source PDFs to staging
cp -r source_pdfs/ staging/source_pdfs/

# 3. Modify paths in script config or run locally
# (For safety, we recommend performing the swap during low-traffic windows)
```

### 2. Perform the File Swap
Once the scripts complete successfully, swap the files in the main directory:
```bash
# Move new index files into place
mv vector_db.index vector_db.index.old
mv vector_db.pt vector_db.pt.old

cp staging/vector_db.index .
cp staging/vector_db.pt .
```

### 3. Restart the Flask/vLLM App
Since the Flask application loads the FAISS index and metadata database into GPU/System RAM on startup, you **must** restart the service to apply the updates:
```bash
# 1. Terminate the active Flask server process
pkill -f app.py

# 2. Wait 5 seconds to ensure VRAM is cleared, then verify GPU status
nvidia-smi

# 3. Start the Flask server
export VLLM_USE_FLASHINFER_SAMPLER="0"
nohup /opt/pytorch/bin/python3 app.py > data/flask_server.log 2>&1 &
```

---

## 🛠️ Troubleshooting Indexing Failures

### 1. GPU Out-of-Memory (OOM) during embedding
If `05_build_vector_db.py` throws a CUDA memory error:
* *Cause*: Batch size is set too high for the available VRAM.
* *Solution*: Open `05_build_vector_db.py` and reduce the `batch_size` (e.g. change `batch_size = 64` to `batch_size = 32` or `16`).

### 2. Malformed PDF Ingestion Errors
If `01_ingest_all_pdfs.py` crashes on a specific file:
* *Cause*: The PDF might be corrupted, encrypted, or contain unreadable embedded fonts.
* *Solution*: Check the console log to identify which file caused the crash. Try opening the PDF in a local viewer, or re-save/export the PDF from standard tools to strip corruption before uploading.
