# Implementation Plan — Phase 3: Data Pipeline & Model Training (FINAL - INTEGRATED)

## Target Environments Reference
* **jittu account instance**: `i-01236f458ae6833ad` (Public IP: `35.179.154.33`, Key: `poc_gpu_key.pem` in local Downloads directory). All pipeline execution and model training will run on this instance.
* **Josh TCS account**: `i-09e7b81b6184aebaa` (Public IP: `35.176.1.158`, Key: `gpu_poc_EUR.pem` in workspace). Legcy instance running active APIs; must be left untouched.

## AWS Authentication for jittu account instance
* **Status**: Configured. Profile `jittu` is configured in the AWS CLI using the credentials from `developer_tcs_accessKeys.csv`. SSH over SSM routing is active for `i-01236f458ae6833ad` using `--profile jittu`.


---

## Goal

Upgrade the entire data pipeline to produce a high-quality, domain-aware CMS Decision Support model. All ~600 documents will be used for both RAG and fine-tuning. No exclusions. This plan incorporates all 8 recommendations from the expert review to guarantee high training data quality, training-inference alignment, and pipeline efficiency.


---

## Architecture Overview

```
                        ┌─────────────────────────────────┐
                        │  Step 1: Data Profiling (NEW)    │
                        │  • Quick scan all 600 PDFs       │
                        │  • Scanned PDF/image-only check  │
                        │  • Output stats, headers         │
                        └──────────────┬──────────────────┘
                                       │
                        ┌──────────────▼──────────────────┐
                        │  Step 2: Ingestion Pipeline      │
                        │  • Recursive folder scan         │
                        │  • Repeating-line header removal │
                        │  • Semantic chunking             │
                        │  • Category tagging              │
                        └──────────────┬──────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
         ┌──────────▼──────────┐            ┌─────────────▼──────────────┐
         │  Step 3: RAG DB     │            │  Step 4: Q&A Generation     │
         │  All chunks → vector│            │  Teacher: Qwen-32B (temp=0.4│
         │  database for       │            │  Diversity prompt controls  │
         │  retrieval          │            │  Two-tier + overlap filter  │
         └─────────────────────┘            │  Batched Tier 2 LLM Judge   │
                                            └─────────────┬──────────────┘
                                                          │
                                            ┌─────────────▼──────────────┐
                                            │  Step 5: Merge & Split      │
                                            │  MinHash+LSH dedup, 90/10   │
                                            └─────────────┬──────────────┘
                                                          │
                                            ┌─────────────▼──────────────┐
                                            │  Step 6: Fine-Tune          │
                                            │  Student: Qwen-14B (QLoRA)  │
                                            │  Bake system prompt & context│
                                            │  Early stopping callback    │
                                            └─────────────────────────────┘
```

---

## Step 1: Data Profiling Script (NEW)

#### [NEW] `10_profile_source_data.py`

A quick, lightweight script (no GPU needed) that scans all PDFs and outputs:
- Total file count and size by subfolder
- Average / median / min / max page count per PDF
- Average text length per document
- **Repeating-line detection:** Identify the most common lines across pages within each PDF (these are headers/footers to strip)
- **Scanned/Image-Only PDF Detection:** Calculate average characters per page. Flag any PDF where the average text length per page is **less than 50 characters**. These zero-yield or low-yield files will be logged so they can be flagged for OCR preprocessing (e.g., via `pytesseract`) or manually checked.
- Sample extracted text (first 500 chars) from 3 random documents in each subfolder

**Purpose:** Calibrate chunking parameters, detect scanned documents needing OCR, and validate header/footer detection before committing to the full pipeline run.

**Runs on:** CPU only. Takes ~2 minutes for 600 PDFs.

---

## Step 2: Ingestion Pipeline Upgrade

#### [MODIFY] [01_ingest_all_pdfs.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/01_ingest_all_pdfs.py)

**Recursive Folder Scan:**
- Use `os.walk()` to traverse `source_pdfs/` and all subdirectories.
- Process every `.pdf` and `.html` file found.

**Repeating-Line Header/Footer Removal:**
- For each PDF, extract text per page, split into lines.
- **Short-Document safety rule:** Only run repeating-line analysis if the document has **more than 2 pages**. This prevents stripping legitimate titles/disclaimers from 1-2 page documents.
- Count how many pages each line appears on.
- Lines appearing on >50% of pages AND shorter than 100 characters are classified as noise (headers, footers, page titles).
- Also strip: standalone page numbers (`^\d+$`), URLs, and lines that are just whitespace.

**HTML Pre-processing:**
- For `.html` files, strip `<script>` and `<style>` blocks first (using case-insensitive regex) before stripping standard HTML tags. This prevents CSS/JS garbage from entering the text corpus.

**Semantic Chunking (two strategies based on document type):**

1. **DMG Volumes** (detected by `"volume"` or `"cmdmg"` in filename):
   - Keep the existing 5-digit paragraph regex splitting — this works well for these structured documents.

2. **All other documents** (policy guidance + procedures):
   - Split text by double newlines (`\n\n`) into natural paragraphs.
   - Group consecutive paragraphs into chunks of **800–1500 characters**.
   - **Never split mid-sentence:** If adding the next paragraph would exceed 1500 chars, close the current chunk and start a new one.
   - If a single paragraph exceeds 1500 chars, split it at the nearest sentence boundary (`. `).

**Metadata & Tagging:**
- Prepend filename to each chunk: `"Document: Shared-Care.pdf | [Content]"`
- Tag each chunk with a category field:
  - `"dmg"` — if filename contains "volume" or "cmdmg"
  - `"policy_guidance"` — if file is inside `policy_law_decision_making_guidance/` subfolder
  - `"procedure"` — everything else
- **Unique Paragraph ID Scheme:** Form paragraph IDs using the full sanitized filename (replacing hyphens with underscores) to completely avoid ID collisions across different files:
  `L_{sanitized_filename}_{chunk_index}` (e.g. `L_Arrears_Only_Case_CSA_Manual_Set_Up_1`).

**Output:** `data/real_chunks.jsonl` — one JSON object per line with fields:
```json
{
  "chunk_type": "policy_guidance",
  "paragraph_id": "PG_Shared_Care_3",
  "text": "The cleaned chunk text...",
  "formatted_text": "Document: Shared-Care.pdf | The cleaned chunk text...",
  "source_doc": "Shared-Care.pdf"
}
```

---

## Step 3: Rebuild RAG Vector Database

#### [RUN] [05_build_vector_db.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/05_build_vector_db.py)

- Rebuild using the new `data/real_chunks.jsonl` (all chunks from all 600 docs).
- No code changes needed if this script already reads from `data/real_chunks.jsonl`.

---

## Step 4: Training Data Generation (Teacher Model)

#### [MODIFY] [07_generate_training_data_local.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/07_generate_training_data_local.py)

**Model:** `Qwen/Qwen2.5-32B-Instruct` loaded in 4-bit via Unsloth (~17GB VRAM).

**Diversity Generation Settings:**
- **Temperature:** Set to `0.4` or `0.5` for the Q&A generation prompts (not verification) to ensure varied question styles and linguistic diversity.

**Weighted Q&A Generation:**

| Chunk Category | QA Pairs to Generate | Rationale |
|---|---|---|
| `dmg` | 2–3 per chunk | Dense legal rules; multiple questions per section |
| `policy_guidance` | 2–3 per chunk | Substantive decision-making guidance |
| `procedure` | 1 per chunk | Focused, actionable — one strong Q&A is sufficient |

**Generation Prompt (per chunk):**

- **For Policy/DMG Chunks (`dmg`, `policy_guidance`):**
  ```
  System: You are an expert trainer for the DWP Child Maintenance Service (CMS).
  Given a policy or legislative text, generate {N} realistic question-answer pairs
  that a caseworker or decision maker might ask to interpret the policy rules. Base
  answers SOLELY on the provided text. Cite specific rules or paragraph numbers where present.
  
  Each question must be distinct and test a DIFFERENT aspect of the text (e.g., eligibility criteria,
  general rules, exceptions, or specific edge cases). Avoid asking questions that result in
  overlapping answers.
  
  User: Context:
  {chunk_text}
  
  Generate exactly {N} question-answer pair(s) in this JSON format:
  [{"instruction": "question", "output": "detailed answer"}, ...]
  ```

- **For Procedure Chunks (`procedure`):**
  ```
  System: You are an expert trainer for the DWP Child Maintenance Service (CMS).
  Given an operational procedure guideline, generate exactly 1 realistic question-answer
  pair focusing on the business logic, eligibility criteria, decision rules, constraints,
  or conditions (i.e. 'when', 'why', and 'under what conditions' steps must be taken).
  DO NOT generate questions about click-by-click UI navigation steps (e.g. 'click this tab',
  'press F3'). Base answers SOLELY on the provided text.
  
  User: Context:
  {chunk_text}
  
  Generate exactly 1 question-answer pair in this JSON format:
  [{"instruction": "question", "output": "detailed answer"}]
  ```

**Two-Tier Quality Filter:**

*Tier 1 — Heuristic (instant, no GPU):*
- Reject if question < 20 characters or missing `?`
- Reject if answer < 50 characters
- Reject if answer has < 25% word overlap with source context (likely hallucinated)
- Reject if question starts with generic patterns like "What is this document" or "Can you summarize"
- **Answer Overlap Check:** For multi-QA generations (DMG, Guidance), check the overlap of answers. If two generated answers from the same chunk have a Jaccard word similarity >60%, discard the pair with the shorter answer and keep the longer one.

*Tier 2 — LLM Judge (GPU, only for pairs passing Tier 1):*
- **Batched Evaluation:** Run verification prompts in batches of **8–16** in a single forward pass instead of sequentially. This decreases Tier 2 validation time by 5-8x.
- **Verification Prompt:**
```
Context: {context}
Question: {question}
Answer: {answer}

Evaluate: Is the answer fully supported by the context without outside assumptions?
Respond with exactly one word: PASS or FAIL
```
- Discard pairs that return `FAIL`.

**Checkpoint/Resume:** Save progress every 50 chunks to allow resuming if interrupted.

**Output:** `data/llm_generated_training_data.jsonl`

---

## Step 5: Merge, Deduplicate & Split

#### [MODIFY] [08_merge_training_data.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/08_merge_training_data.py)

- Load all generated Q&A pairs from `data/llm_generated_training_data.jsonl`
- **Exact deduplication** on instruction text
- **Fuzzy deduplication (MinHash + LSH):** Replace the slow O(n²) comparison loop with an O(n) Locality-Sensitive Hashing approach using `datasketch` (add to `requirements.txt`).
  - Use MinHash with 128 permutations (`num_perm=128`).
  - Jaccard similarity threshold set to `0.85`.
  - Dedups in seconds rather than hanging for hours on large datasets.
- Shuffle with fixed seed (42)
- **90/10 train/validation split**
- Save to `data/train_split.jsonl` and `data/val_split.jsonl`

---

## Step 6: Fine-Tune Student Model (QLoRA)

#### [MODIFY] [02_train_qlora_gpu.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/02_train_qlora_gpu.py)

**Student Model:** `Qwen/Qwen2.5-14B-Instruct`

**Hyperparameters (optimised for 24GB GPU):**

| Parameter | Value | Rationale |
|---|---|---|
| `load_in_4bit` | `True` | Fits 14B in ~7.5GB VRAM |
| `max_seq_length` | `2048` | Policy docs need long context |
| `lora_r` | `32` | Higher rank for complex domain (up from 16) |
| `lora_alpha` | `64` | 2× lora_r (standard ratio) |
| `per_device_train_batch_size` | `2` | Fits comfortably in VRAM |
| `gradient_accumulation_steps` | `8` | Effective batch size = 16 |
| `num_train_epochs` | `3` | Standard for domain fine-tuning |
| `learning_rate` | `1e-4` | Slightly lower than 2e-4 for larger model stability |
| `lr_scheduler_type` | `cosine` | Smooth decay |
| `warmup_ratio` | `0.05` | 5% warmup steps |
| `optim` | `adamw_8bit` | Memory-efficient optimiser |
| `gradient_checkpointing` | `"unsloth"` | Unsloth's memory-efficient implementation |
| `eval_strategy` | `"steps"` | Track validation loss during training |
| `eval_steps` | `100` | Run validation evaluation every 100 steps |
| `save_strategy` | `"steps"` | Prevent progress loss |
| `save_steps` | `200` | Save checkpoint every 200 steps |
| `save_total_limit` | `2` | Restrict saved checkpoints to avoid disk fill |
| `load_best_model_at_end` | `True` | Retrieve the best model checkpoint based on validation loss |

**System Prompt Baking (Train-Inference Alignment):**
Prepends the standard system prompt to every training and validation example to avoid a mismatch at inference:
```python
system_prompt = (
    "You are an expert Decision Maker assistant for the DWP Child Maintenance Service (CMS). "
    "Answer questions accurately using only the provided policy context. "
    "Cite specific paragraph numbers, rules, and sections where present. "
    "If the context does not contain sufficient information, state this clearly."
)
```

**Training Format (With RAG Context):**
To train the student model to act as an accurate reading comprehension and citation extractor (rather than relying on prone-to-hallucination memory lookup), we incorporate the source context block directly into the user instruction:
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Context:\n{chunk_text}\n\nQuestion: {question}"},
    {"role": "assistant", "content": answer},
]
```
The formatting function will apply the chat template with these three messages.

**Early Stopping Callback:**
Import `EarlyStoppingCallback` from `transformers` and pass it to `SFTTrainer(..., callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])`. If validation loss does not decrease for 3 consecutive evaluations (300 steps total, given `eval_steps=100`), stop training early to prevent overfitting and save GPU resources.

**VRAM Budget:**

| Component | Estimated VRAM |
|---|---|
| Model weights (4-bit) | ~7.5 GB |
| LoRA adapters + gradients (r=32) | ~1.0 GB |
| Optimizer states (8-bit) | ~1.0 GB |
| Activations (seq=2048, batch=2, grad_ckpt) | ~5.0 GB |
| **Total** | **~14.5 GB / 24 GB** |

**Headroom:** ~9.5 GB — very comfortable. No OOM risk.

**Estimated training time:** ~10–14 hours for 3 epochs on ~8,000–10,000 examples.

---

## Verification Plan

### After Step 1 (Profiling)
- Review the profiling output to confirm header/footer patterns are correctly detected.
- Verify flagged scanned/image-only PDFs and ensure they are accounted for.
- Spot-check sample text from each category.

### After Step 2 (Ingestion)
- Count total chunks by category (`dmg`, `policy_guidance`, `procedure`).
- Inspect 5 random chunks from each category for:
  - No header/footer noise remaining
  - Complete sentences (no mid-sentence cuts)
  - Filename metadata correctly prepended

### After Step 4 (Q&A Generation)
- Review 20 random Q&A pairs across all categories.
- Check the rejection rate from Tier 1 (including answer overlap check) and Tier 2 filters.
- Verify that policy Q&As cite paragraph numbers and procedure Q&As reference specific steps.

### After Step 6 (Training & Evaluation Script Update)
- **Update Evaluation Script (`09_evaluate_model.py`):**
  - Modify the script to accept the new Qwen-14B model path via command line argument or configuration.
  - Update the system prompt from "CMG" to "CMS" to match the training setup.
  - **Create a New Evaluation Set:** Generate `evaluation_set.jsonl` with ~30-40 questions covering all three document categories (DMG, Guidance, Procedures) to ensure broad evaluation coverage.
- Monitor training loss curve — should decrease smoothly without spikes.
- Run the updated evaluation script to compare the new model against the baseline.
- Test with 10 real-world caseworker questions covering both policy and procedural topics.

---

## Timeline & Tooling Estimate

| Step | Duration | Hardware | Tooling |
|---|---|---|---|
| Step 1: Profiling | ~2 minutes | CPU | Python (PyMuPDF) |
| Step 2: Ingestion | ~5–10 minutes | CPU | Python (PyMuPDF) |
| Step 3: RAG DB rebuild | ~10–15 minutes | CPU | FAISS / Python |
| Step 4: Q&A Generation | ~15–20 hours | GPU (32B inference) | Unsloth, vLLM / Batched inference |
| Step 5: Merge & Split | ~1 minute | CPU | Python, `datasketch` (LSH) |
| Step 6: Fine-Tuning | ~10–14 hours | GPU (14B QLoRA) | Unsloth, PyTorch, SFTTrainer |
| **Total** | **~26–35 hours** | | |
