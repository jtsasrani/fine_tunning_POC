# Target Instances Terminology Reference
* **jittu account instance**: `i-01236f458ae6833ad` (Public IP: `35.179.154.33`, Key: `poc_gpu_key.pem` in local Downloads directory). This is where all future work is performed.
* **Josh TCS account**: `i-09e7b81b6184aebaa` (Public IP: `35.176.1.158`, Key: `gpu_poc_EUR.pem` in workspace). This is the legacy instance with running APIs; we must not touch it.

# Task: Phase 3 Ingestion, Generation, and Fine-Tuning Execution

- `[x]` Step 0: EC2 Instance Environment Setup (jittu account instance)
    - `[x]` Create and enable 24GB swap file to prevent CPU OOM
- `[x]` Step 1: Data Profiling
    - `[x]` Create `10_profile_source_data.py` locally with scanned PDF detection and header/footer stats
    - `[x]` Upload and run on EC2
    - `[x]` Verify output logs and check for scanned PDFs
- `[x]` Step 2: Ingestion Pipeline Upgrade
    - `[x]` Modify `01_ingest_all_pdfs.py` (recursive scan, short-doc header safety, HTML pre-processing, chunking, metadata paragraph IDs)
    - `[x]` Upload and run on EC2
    - `[x]` Verify `data/real_chunks.jsonl` output
- `[x]` Step 3: Rebuild RAG DB
    - `[x]` Run `05_build_vector_db.py` on EC2
    - `[x]` Verify vector database matches the new chunk collection
- `[ ]` Step 4: Training Data Generation (Teacher Model)
    - `[x]` Modify `07_generate_training_data_local.py` (diversity settings, temperature=0.4-0.5, context prompt integration, Tier 1 overlap filter, batched Tier 2 LLM judge)
    - `[x]` Upload and run on EC2 (Successfully launched on remote GPU)
    - `[/]` Monitor teacher generation progress (Running in background, first batch completed successfully)
- `[ ]` Step 5: Merge, Deduplicate & Split
    - `[x]` Modify `08_merge_training_data.py` (MinHash + LSH dedup using `datasketch`)
    - `[ ]` Upload and run on EC2 (Uploaded, waiting to run)
    - `[ ]` Verify train/val splits
- `[ ]` Step 6: Fine-Tune Student Model (QLoRA)
    - `[x]` Modify `02_train_qlora_gpu.py` (14B configuration, system prompt baking, RAG context in chat template, EarlyStoppingCallback)
    - `[ ]` Upload and run on EC2 (Uploaded, waiting to run)
    - `[ ]` Verify trained adapter outputs
- `[ ]` Step 7: Verification & Evaluation
    - `[x]` Update `09_evaluate_model.py` (CLI arguments, CMS system prompt, new 30-40 question evaluation set)
    - `[ ]` Run evaluation and compile results report

