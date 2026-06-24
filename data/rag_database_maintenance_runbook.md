# Standard Operating Procedure (SOP): RAG Database Maintenance & Re-indexing Runbook

**Document Control**
* **Title**: DWP CMS Decision Support Suite - RAG Database Maintenance SOP
* **Version**: 2.0 (Production Release)
* **Author**: Lead AI Architect
* **Target Audience**: Systems Administrators, DevOps Engineers, and AI Engineers
* **Status**: Approved

---

## 📂 1. Architectural Overview & Component Interdependencies

The Retrieval-Augmented Generation (RAG) search pipeline utilizes three distinct, synchronized assets located in the project root `/home/ubuntu/dwp-cmg-finetune/`:

```
dwp-cmg-finetune/
├── source_pdfs/         # Source manuals (PDF/HTML) served directly to user web browsers
├── vector_db.index      # FAISS Flat Inner-Product index containing dense passage embeddings
└── vector_db.pt         # PyTorch binary archive storing chunks text, source metadata, and token mappings
```

### Critical Dependency Rules
* **1-to-1 Mapping**: The dense embedding index (`vector_db.index`) and the metadata catalog (`vector_db.pt`) are highly coupled. The index ID returned by FAISS matches the exact list index of the paragraph array stored inside `vector_db.pt`.
* **Desynchronization Hazard**: If `vector_db.index` is replaced without updating `vector_db.pt` (or vice-versa), search queries will fetch unrelated paragraphs and link them to incorrect document pages, causing **severe context misalignment** and **LLM hallucinations**. 
* **Zero-Downtime Requirement**: The active web server processes read these files into system and GPU RAM on initialization. Operations must build assets in a staging environment and execute atomic swaps during low-traffic windows.

---

## 🔄 2. End-to-End Re-indexing Pipeline (Step-by-Step)

Perform these steps inside the active Conda environment.

### Step 1: Pre-requisites & Environment Verification
Ensure you have the GPU-enabled PyTorch environment active and check resource availability:
```bash
# 1. Navigate to the project root directory
cd /home/ubuntu/dwp-cmg-finetune

# 2. Activate the Conda environment containing PyTorch, FAISS, and Hugging Face libraries
conda activate pytorch

# 3. Ensure system has at least 16GB free RAM and the GPU is available
nvidia-smi
```

### Step 2: Prepare Source Files
1. **Document Placement**: Store all official PDF or HTML manuals in the `source_pdfs/` folder.
2. **File Naming Conventions**: Filenames must use alphanumeric characters, dashes, and underscores only. **Never use spaces** or special characters.
   * *Correct*: `volume-6-collection-and-enforcement-chapters-49-95.pdf`
   * *Incorrect*: `Volume 6 Collection & Enforcement.pdf`
3. **Outdated Files**: Remove outdated policy files from `source_pdfs/` to prevent retired or superseded guidelines from entering the retrieval pool.

### Step 3: Text Extraction and Chunking
Run the ingestion script to parse documents, clean character encodings, strip headers/footers, and segment texts into paragraph units:
```bash
python3 01_ingest_all_pdfs.py
```
* **How it works**: The script processes pages and automatically identifies repeating headers and footers by running a frequency counter across pages (lines appearing on $> 50\%$ of pages are flagged and ignored). It strips page numbers, normalizes non-ASCII characters, and writes structured JSON lines to the data directory.
* **Output**: `data/real_chunks.jsonl`
* **Verification**: Ensure the file contains clean, non-empty chunks:
  ```bash
  tail -n 5 data/real_chunks.jsonl
  ```

### Step 4: Embeddings Generation and Indexing
Compile the dense vector index and PyTorch metadata:
```bash
python3 05_build_vector_db.py
```
* **How it works**:
  1. The script loads the raw chunks and inspects their token length using the `BAAI/bge-base-en-v1.5` tokenizer.
  2. Any chunk exceeding `500` tokens is sub-chunked using a sliding window approach with a `100` token overlap to preserve contextual boundaries.
  3. It feeds the texts to the embedding model, extracts mean-pooled outputs, applies L2 normalization, and indexes them into a `faiss.IndexFlatIP` (Flat Inner Product) vector database.
* **Outputs**:
  * `vector_db.index`
  * `vector_db.pt`

---

## 🔬 3. Database Integrity & Sanity Verification

Before deploying the newly compiled index, you **must** run an integrity check to guarantee the vector index and metadata are perfectly aligned. Run this verification command in Python:

```bash
python3 -c "
import faiss
import torch

try:
    index = faiss.read_index('vector_db.index')
    db_data = torch.load('vector_db.pt')
    
    # 1. Dimension Check
    assert index.d == 768, f'Invalid FAISS dimension: {index.d} (expected 768)'
    
    # 2. Count Alignments
    faiss_cnt = index.ntotal
    meta_cnt = len(db_data['chunks'])
    assert faiss_cnt == meta_cnt, f'Count mismatch! FAISS has {faiss_cnt} vectors, but metadata has {meta_cnt} chunks.'
    
    # 3. Embedding Shape Check
    emb_shape = db_data['embeddings'].shape
    assert emb_shape[0] == faiss_cnt, f'Embeddings tensor mismatch! Tensor rows = {emb_shape[0]}, FAISS rows = {faiss_cnt}'
    assert emb_shape[1] == 768, f'Embeddings tensor column mismatch: {emb_shape[1]}'
    
    print('\n=========================================')
    print('✅ INTEGRITY CHECK PASSED!')
    print(f'FAISS Vector Count: {faiss_cnt}')
    print(f'Metadata Chunks Count: {meta_cnt}')
    print(f'Embedding Tensor Shape: {emb_shape}')
    print('=========================================\n')
    
except AssertionError as ae:
    print('\n🚨 INTEGRITY CHECK FAILED:', str(ae))
    exit(1)
except Exception as e:
    print('\n🚨 UNEXPECTED INTEGRITY ERROR:', str(e))
    exit(1)
"
```

> [!CAUTION]
> If this script exits with an error, **do not deploy the database**. The system is out of sync. Delete `vector_db.index` and `vector_db.pt` and re-run the pipeline from Step 3.

---

## ⚡ 4. Zero-Downtime Deployment & Swap Flow

To deploy the new index without disrupting active caseworker query sessions, implement this cold swap procedure:

### 1. Build in Staging Directory
Always run the build pipeline in a staging folder so you do not overwrite files in active use:
```bash
# Create staging paths
mkdir -p staging/data

# Run compilation targeted at staging paths
# (Ensure scripts are run or temporary symbolic links are used)
```

### 2. Back Up Active Database Assets
Archive the current database assets to facilitate instant rollback if anomalies are detected post-deployment:
```bash
# Define archive timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p backups/database_$TIMESTAMP/

# Copy active database assets
cp vector_db.index backups/database_$TIMESTAMP/vector_db.index
cp vector_db.pt backups/database_$TIMESTAMP/vector_db.pt
```

### 3. Swap Assets and Restart Web Service
```bash
# 1. Swap index and metadata files atomically
mv staging/vector_db.index ./vector_db.index
mv staging/vector_db.pt ./vector_db.pt

# 2. Terminate the active Flask web app process
pkill -f app.py

# 3. Wait 5 seconds to ensure VRAM release is complete
sleep 5
nvidia-smi

# 4. Restart the web server in background production mode
export VLLM_USE_FLASHINFER_SAMPLER="0"
nohup /opt/pytorch/bin/python3 app.py > data/flask_server.log 2>&1 &

# 5. Monitor initialization logs until ready
tail -f data/flask_server.log
```
Look for this line in the logs to confirm the service is fully operational:
`Models/DB initialization completed. Web server running on http://0.0.0.0:5000`

---

## 🛠️ 5. Troubleshooting & Fallback Operations

### GPU Out-of-Memory (OOM) during embedding
* **Symptom**: `05_build_vector_db.py` crashes with `CUDA out of memory` during embedding generation.
* **Resolution**: Open `05_build_vector_db.py`, locate the `batch_size` parameter in the `main()` function (around line 102), and reduce it from `64` to `32` or `16`. Save the file and re-run.

### Malformed PDF Ingestion Crash
* **Symptom**: `01_ingest_all_pdfs.py` throws extraction errors or crashes on a specific file.
* **Resolution**: Check the terminal output to identify the exact PDF file. Verify if it is password-protected or corrupted. Open the PDF in a local browser, print/export it as a new PDF (to strip corrupted metadata/fonts), and overwrite the file in `source_pdfs/`.

### Emergency Rollback Protocol
If the web service displays internal server errors or returns misaligned RAG context post-deployment:
```bash
# 1. Stop current app
pkill -f app.py

# 2. Locate the latest stable backup
LATEST_BACKUP=$(ls -td backups/database_*/ | head -n 1)

# 3. Restore stable assets
cp ${LATEST_BACKUP}vector_db.index ./vector_db.index
cp ${LATEST_BACKUP}vector_db.pt ./vector_db.pt

# 4. Restart server
export VLLM_USE_FLASHINFER_SAMPLER="0"
nohup /opt/pytorch/bin/python3 app.py > data/flask_server.log 2>&1 &
```

