# Implementation Plan - Phase 2 Model & RAG Optimization (Local 32B Path)

This plan details the optimization steps for the next phase of the DWP CMG policy assistant. It focuses on using local, open-source models to scale data quality and size at **zero additional cost**, followed by extended fine-tuning and retrieval upgrades.

---

## User Review Required

> [!NOTE]
> **Zero Bedrock Billing / 100% Free Run**
> To avoid Bedrock billing, all QA dataset generation will run locally on the EC2 GPU using **Qwen-2.5-32B-Instruct** (loaded in 4-bit precision).
>
> **GPU VRAM Allocation during Data Generation**
> The 32B parameter model in 4-bit precision requires **~19.5 GB of VRAM**. This fits within the 24 GB VRAM of the NVIDIA A10G GPU, leaving enough room for embedding models and text caches. We will stop the web server during data generation to avoid memory contention.

---

## Proposed Changes

### Component 1: Data Generation & Refinement

#### [NEW] [07b_generate_training_data_32b.py](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/07b_generate_training_data_32b.py)
* **Model**: Load `Qwen/Qwen2.5-32B-Instruct` in 4-bit using `BitsAndBytesConfig` (NF4, bfloat16 compute type).
* **Generation**: Read the 3,174 processed chunks from `data/real_chunks.jsonl` and generate 1-2 professional Q&A pairs per chunk.
* **Local Critic Step (Self-Correction)**: 
  * After generating each QA pair, pass the question, answer, and source chunk back to the 32B model with a validation prompt: *"Is the question fully and accurately answered by the paragraph, and is the answer free of outside knowledge? Answer YES or NO."*
  * Discard any QA pairs that return NO.
* **Target Corpus**: ~2,500 - 3,000 highly verified, DWP-aligned QA pairs.

---

### Component 2: Fine-Tuning Hyperparameter Expansion

#### [MODIFY] [02_train_qlora_gpu.py](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/02_train_qlora_gpu.py)
* **Target Models**: 
  1. `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
  2. `unsloth/qwen2.5-7b-instruct-bnb-4bit`
  3. `unsloth/llama-3-8b-instruct-bnb-4bit` (added as a benchmark contender)
* **Parameters**: 
  * Increase epochs from **3 to 6**.
  * Increase LoRA Rank ($r$) from **16 to 32** (Alpha from **32 to 64**) to allow the LoRA adapters to memorize complex policy rules and citations.
  * Use a cosine decay learning rate scheduler to stabilize convergence over longer epochs.

---

### Component 3: Two-Stage Parent-Child RAG Pipeline

#### [MODIFY] [05_build_vector_db.py](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/05_build_vector_db.py) & [06_run_rag_qa.py](file:///C:/Users/JitendraAsrani/DWP_CMG_Finetune/06_run_rag_qa.py)
* **Parent-Child Chunking**:
  * Chunks indexed for FAISS vector search will be small (200 tokens) to ensure high retrieval sensitivity.
  * Once the top-k small chunks are retrieved, swap them for their larger "parent" sections (800 tokens) to supply the LLM with full context.
* **Embedding Model**: Upgrade from `BAAI/bge-base-en-v1.5` (768 dimensions) to `BAAI/bge-large-en-v1.5` (1024 dimensions) for stronger semantic separation.

---

## Verification Plan

### Metrics Target
* **Target ROUGE-L F1**: **> 0.6000** (Current: `0.5052`)
* **Target BERTScore F1**: **> 0.9400** (Current: `0.9160`)
* Evaluated on the same 147 validation samples (`data/val_split.jsonl`) to ensure clean benchmark comparisons.
