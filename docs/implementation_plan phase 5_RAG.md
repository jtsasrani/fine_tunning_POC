# Implementation Plan - Phase 5: Retrieval-Augmented Generation (RAG)

This plan outlines the design and implementation of the Retrieval-Augmented Generation (RAG) system to achieve 100% factual correctness on DWP CMG policy questions by grounding the local reader model (`SmolLM2-360M-Instruct`) with official policy manuals.

## User Review Required

> [!IMPORTANT]
> - We will use the lightweight `sentence-transformers/all-MiniLM-L6-v2` model (90MB) for vector embeddings, running locally on CPU.
> - Retrieval similarity calculations will use native PyTorch matrix operations to avoid compiling libraries like FAISS or installing complex dependencies on Windows 11.
> - The vector database (`vector_db.pt`) will be saved using standard PyTorch format (`torch.save()`), keeping all chunks, embeddings, and metadata self-contained in one file.

## Proposed Changes

### Vector Database Build System

#### [NEW] [05_build_vector_db.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/05_build_vector_db.py)
This script will construct the local policy knowledge base:
1. **Extraction**: Parse the three PDFs in `./source_pdfs/` using `pypdf` (robust fallback from PyMuPDF).
2. **Chunking**: Segment the documents along paragraph boundaries using a regex matching the 5-digit Decision Maker Guide paragraph codes: `\b((?:17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|49|50|95)\d{3})\b`.
3. **Embedding Generation**: Encode all paragraph texts using `sentence-transformers/all-MiniLM-L6-v2`. Computations will be fully vectorized and run on CPU.
4. **Serialization**: Save a dictionary containing chunks, embedding tensors, source metadata, and paragraph IDs to `vector_db.pt`.

---

### RAG Inference & Evaluation

#### [NEW] [06_run_rag_qa.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/06_run_rag_qa.py)
This script will perform RAG QA and compare models:
1. **Retrieval**: Compute the user query's embedding, calculate cosine similarity against all paragraphs in `vector_db.pt`, and retrieve the top $K=3$ contexts.
2. **Prompt Engineering**: Format a chat instruction prompt that wraps the context inside system instructions, directing the model to answer using only the provided facts.
3. **Comparative Evaluation**: Compare the answers of three configurations on the 3 core CMG test questions:
   - **Baseline**: Tuned model (LoRA) *without* RAG.
   - **RAG + Base**: Base `SmolLM2-360M-Instruct` *with* retrieved context.
   - **RAG + Tuned**: Fine-tuned `SmolLM2-360M-Instruct` (LoRA) *with* retrieved context.
4. **Reporting**: Print a clear markdown evaluation report of the outputs.

---

## Verification Plan

### Automated Tests
1. **Build Vector Database**: Run `python 05_build_vector_db.py`. Verify that the database executes successfully, creates `vector_db.pt`, and prints the total number of paragraphs stored.
2. **Execute RAG QA**: Run `python 06_run_rag_qa.py`. Verify that it retrieves context and prints answers for all three configurations.

### Manual Verification
- Review the generated answers for the 3 core questions:
  1. *25% income variance rule*
  2. *Administrative Liability Orders under the Child Maintenance (Enforcement) Act 2023*
  3. *Pension contribution calculation treatment*
- Verify that both RAG models mention exact numbers, conditions, and calculations correctly from the manual (e.g. the 25% change threshold, the enforcement act powers, and pension deduction allowance).
- Compare the tone of **RAG + Base** vs. **RAG + Tuned** to assess if the tuned model's style is preserved in RAG.
