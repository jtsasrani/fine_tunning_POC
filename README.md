# DWP CMG AI Assistant — Proof of Concept Summary

> **Audience:** Anyone who wants to understand what this POC achieved, how it works, and what comes next.  
> **Date:** June 2026  
> **Environment:** Built and run entirely on a standard Windows laptop (CPU only — no GPU required)

---

## What Did We Build?

We built an **AI-powered assistant** that can answer questions about UK Child Maintenance policy by reading and learning from official DWP Decision Makers' Guide (DMG) manuals. Think of it as a "smart search engine" that doesn't just find the right page — it reads the policy, understands the question, and writes a human-readable answer.

### In Plain English

Imagine a new Decision Maker joins the team and needs to answer a question like:

> *"How are pension contributions treated when calculating child maintenance?"*

Normally, they'd have to search through hundreds of pages of policy manuals, find the right paragraph, and interpret the rules. Our AI assistant does this automatically:

1. **Searches** through 1,779 policy paragraphs to find the 3 most relevant ones
2. **Reads** those paragraphs and understands the context
3. **Writes** a professional answer based only on official policy text

---

## How Does It Work?

The system has two core components that work together:

### Component 1: The "Brain" — A Fine-Tuned Language Model

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Pre-trained AI Model (SmolLM2-360M)               │
│   ↓                                                 │
│   + DWP Policy Training Data (80 Q&A pairs)         │
│   ↓                                                 │
│   = Fine-Tuned CMG Specialist Model                 │
│                                                     │
│   Think of it like: A general-purpose assistant      │
│   who attended a training course on DWP policy       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

- We started with a **pre-trained AI model** called SmolLM2-360M (made by HuggingFace). This is a small language model with 360 million parameters — it can understand and write English, but knows nothing about DWP policy.
- We then **fine-tuned** it (taught it) using real DWP policy content. We fed it 80 question-and-answer pairs created from the official manuals. After training, the model became better at using DWP-specific language and concepts.
- The fine-tuning technique used is called **LoRA** (Low-Rank Adaptation) — a modern, efficient method that only adjusts a small portion of the model's knowledge, making training fast even on a regular laptop CPU.

### Component 2: The "Library" — A RAG (Retrieval-Augmented Generation) Pipeline

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   User asks a question                              │
│   ↓                                                 │
│   Step 1: SEARCH the policy library                 │
│           (Vector similarity + Keyword matching)    │
│   ↓                                                 │
│   Step 2: RETRIEVE the top 3 relevant paragraphs   │
│   ↓                                                 │
│   Step 3: FEED those paragraphs to the AI model     │
│           along with the question                   │
│   ↓                                                 │
│   Step 4: AI WRITES an answer using ONLY those      │
│           paragraphs as source material              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

- RAG stands for **Retrieval-Augmented Generation**. Instead of relying solely on the AI's "memory" (which can be unreliable), we first search for the relevant policy text and hand it to the AI as reference material.
- The search uses a **hybrid approach** combining two methods:
  - **Vector Search**: Converts both the question and every policy paragraph into numerical "fingerprints" (embeddings) and finds the closest matches by mathematical similarity.
  - **Keyword Search**: Looks for exact DWP terminology matches (e.g., "NRP", "gross weekly income", "liability order").
- This hybrid approach ensures we find the right paragraphs even when the user's question uses different words from the policy text (e.g., "paying parent" = "NRP" in policy language).

---

## What Source Material Did We Use?

We processed **3 official DWP Decision Makers' Guide PDFs** covering:

| Manual | Chapters | Size | Topics |
|---|---|---|---|
| Volume 2: Maintenance Calculations | 17–26 | 0.39 MB | Income rules, pension treatment, gross weekly income, the 25% variance rule |
| Volume 3: Variations | 27–36 | 0.23 MB | Income variations, lifestyle inconsistencies, special expenses |
| Volume 6: Collection & Enforcement | 49–95 | 1.21 MB | Liability orders, deduction orders, enforcement agents, charging orders |

### By the Numbers

| Metric | Value |
|---|---|
| Total policy paragraphs extracted | **1,779** |
| Training Q&A pairs generated | **80** (AI-generated from policy text) |
| Raw training samples | **300** (direct policy text chunks) |
| Vector database size | **3.94 MB** |
| Trained LoRA adapter weights | **33 MB** |

---

## The Complete Pipeline — Script by Script

Here is every script in the project and what it does:

### Step 1: `01_ingest_local_pdfs.py` — Extract Text from PDFs
- **What it does:** Opens each PDF file, reads every page, and extracts the text.
- **Output:** `cmg_real_training_data.jsonl` — 300 policy text chunks formatted as training data.
- **In plain terms:** "Photocopy every page and type up the contents."

### Step 2: `01b_ingest_qa_pairs.py` — Generate Training Questions
- **What it does:** Takes the extracted policy text and uses the AI model itself to generate questions that each paragraph answers.
- **Output:** `cmg_qa_training_data.jsonl` — 80 question-and-answer pairs.
- **Example:** For a paragraph about pension contributions, it generates: *"What factors can affect the amount a child maintenance application is required to pay?"*
- **In plain terms:** "Write a quiz based on the textbook."

### Step 3: `02_train_lora_windows_cpu.py` — Train the AI Model
- **What it does:** Feeds the 80 Q&A pairs into the SmolLM2 model using LoRA fine-tuning. The model learns to associate DWP-style questions with policy-accurate answers.
- **Output:** `cmg_lora_weights/` — The trained adapter weights (33 MB).
- **Key settings:** LoRA rank=16, learning rate=0.0001, 2 training epochs, batch size=1 with gradient accumulation.
- **Training time:** ~2 hours on CPU.
- **In plain terms:** "Send the assistant to a DWP training course."

### Step 4: `03_test_cmg_model.py` — Quick Smoke Test
- **What it does:** Loads the fine-tuned model and asks it one test question to verify training worked.
- **In plain terms:** "After training, ask one question to make sure it learned something."

### Step 5: `04_evaluate_ab_test.py` — Compare Base vs. Tuned Model
- **What it does:** Runs the same 3 policy questions through both the **untrained** model and the **fine-tuned** model, displaying the answers side-by-side.
- **Purpose:** Proves that fine-tuning improved the model's answers.
- **In plain terms:** "Give the same exam to a trained and untrained assistant — see who does better."

### Step 6: `05_build_vector_db.py` — Build the Search Library
- **What it does:** Takes all 1,779 policy paragraphs and converts them into 384-dimensional numerical vectors using the `all-MiniLM-L6-v2` embedding model. Saves everything as `vector_db.pt`.
- **Build time:** ~60 seconds.
- **In plain terms:** "Create an index card for every paragraph in the library, organised by topic."

### Step 7: `06_run_rag_qa.py` — Full RAG Evaluation
- **What it does:** The complete pipeline — retrieves relevant paragraphs, feeds them to the model, and generates answers. Compares 3 configurations:
  - **Config A:** Fine-tuned model, no library lookup (answers from memory only)
  - **Config B:** Base model + library lookup (untrained but with reference material)
  - **Config C:** Fine-tuned model + library lookup (trained AND with reference material)
- **In plain terms:** "Full exam: test the assistant with and without reference materials."

### Step 8: `app.py` — Interactive Web Dashboard
- **What it does:** Launches a professional, modern web interface. It preloads the models once on startup and allows the user to ask custom policy questions and compare all three configurations side-by-side.
- **In plain terms:** "An interactive web page where you can type questions and see the AI answer in real-time."

---

## Results — What We Found

### The Good News ✅

1. **RAG dramatically reduces hallucination.** When the AI has access to the policy library (Configs B & C), it retrieves the correct paragraphs and bases its answers on real policy text.
2. **Retrieval accuracy is strong.** For the pension question, the system correctly retrieved Paragraph 22010 which states: *"the NRP's gross earnings figure is earnings minus pension contributions"* — exactly the right answer.
3. **The hybrid search works.** Custom terminology mapping (e.g., "paying parent" → "NRP") bridges the vocabulary gap between how users ask questions and how policies are written.
4. **CPU inference is feasible.** After multi-threading optimisation, answers are generated in 15-30 seconds on a regular laptop — fast enough for a proof of concept.

### The Limitations ⚠️

1. **The 360M model is too small for complex reasoning.** Even with the correct policy paragraphs provided as context, the model often:
   - Invents specific numbers and percentages that don't exist in the source text
   - References non-existent legislation (e.g., "Child Maintenance Act 1943" — doesn't exist)
   - Mixes up concepts from different policy areas
2. **Training data quality is limited.** The 80 Q&A pairs were generated by the same small model, creating a quality ceiling.
3. **Short context window.** The model was trained with `max_length=128` tokens — too short to capture full policy paragraphs.

### Example: Pension Contributions Question

| Configuration | Quality | Key Issue |
|---|---|---|
| **A: Tuned (No RAG)** | ❌ Poor | Hallucinates "Child Maintenance Act 2019 section 37(4)" — doesn't exist |
| **B: Base + RAG** | ⚠️ Partial | References correct concepts but cites "Child Maintenance Act 1943" — wrong |
| **C: Tuned + RAG** | ⚠️ Partial | Uses retrieved context but still adds fabricated details about "Tax Credits Register" |

> **Key Takeaway:** RAG provides the right source material, but the 360M model lacks the reasoning capacity to faithfully synthesise it into accurate answers. **Upgrading to a 7B-parameter model is the single most impactful improvement.**

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DWP CMG AI Assistant — POC Architecture          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐         ┌────────────────────┐                   │
│   │ 3 DWP Policy │────────▶│  PDF Text Extractor │                   │
│   │   PDFs       │         │  (PyMuPDF)          │                   │
│   └──────────────┘         └────────┬───────────┘                   │
│                                      │                               │
│                    ┌─────────────────┼─────────────────┐             │
│                    ▼                 ▼                  ▼             │
│           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│           │ 300 Raw      │  │ 80 QA Pairs  │  │ 1,779 Policy │      │
│           │ Text Chunks  │  │ (Training)   │  │ Paragraphs   │      │
│           └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│                  │                 │                  │               │
│                  │                 ▼                  ▼               │
│                  │        ┌──────────────┐  ┌──────────────┐        │
│                  │        │  LoRA Fine-  │  │  Embedding   │        │
│                  │        │  Tuning      │  │  Generation  │        │
│                  │        │  (CPU, 2hrs) │  │  (MiniLM)    │        │
│                  │        └──────┬───────┘  └──────┬───────┘        │
│                  │               │                  │               │
│                  │               ▼                  ▼               │
│                  │        ┌──────────────┐  ┌──────────────┐        │
│                  │        │ LoRA Adapter │  │ Vector DB    │        │
│                  │        │ Weights      │  │ (1,779 ×     │        │
│                  │        │ (33 MB)      │  │  384-dim)    │        │
│                  │        └──────┬───────┘  └──────┬───────┘        │
│                  │               │                  │               │
│                  │               ▼                  ▼               │
│                  │     ┌─────────────────────────────────┐          │
│   User ─────────────▶ │     RAG Inference Pipeline       │          │
│   Question       │     │                                 │          │
│                  │     │  1. Hybrid Search (BM25+Vector) │          │
│                  │     │  2. Retrieve Top-3 Paragraphs   │          │
│                  │     │  3. Generate Answer (SmolLM2    │          │
│                  │     │     360M + LoRA)                │          │
│                  │     └─────────────┬───────────────────┘          │
│                  │                   │                               │
│                  │                   ▼                               │
│                  │            ┌──────────────┐                      │
│                  │            │   Answer     │                      │
│                  │            │   + Source   │                      │
│                  │            │   Citations  │                      │
│                  │            └──────────────┘                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Why We Chose It |
|---|---|---|
| **Language Model** | SmolLM2-360M-Instruct | Small enough to run on CPU; designed for instruction-following |
| **Fine-tuning Method** | LoRA (PEFT library) | Efficient — trains only ~5% of model parameters; fast on CPU |
| **Embeddings** | all-MiniLM-L6-v2 | Fast, lightweight, 384-dimensional vectors |
| **PDF Extraction** | PyMuPDF (fitz) | Fastest Python PDF reader; handles complex layouts |
| **Search** | Hybrid BM25 + Vector Cosine Similarity | Combines exact keyword matching with semantic understanding |
| **Framework** | PyTorch + HuggingFace Transformers | Industry standard for ML/AI development |
| **Runtime** | Python 3.x on Windows CPU | No special hardware required |

---

## Project File Structure

```
DWP_CMG_Finetune/
├── source_pdfs/                      ← Original DWP policy PDFs (3 files, ~1.8 MB)
├── templates/                        ← HTML templates for the web dashboard
│   └── index.html
├── static/                           ← CSS stylesheet and JavaScript files
│   ├── style.css
│   └── script.js
├── app.py                            ← Web application server (Preloads models + serves UI)
├── 01_ingest_local_pdfs.py           ← Step 1: Extract text from PDFs
├── 01b_ingest_qa_pairs.py            ← Step 2: Generate Q&A training pairs
├── 02_train_lora_windows_cpu.py      ← Step 3: Fine-tune model with LoRA
├── 03_test_cmg_model.py              ← Step 4: Quick smoke test
├── 04_evaluate_ab_test.py            ← Step 5: Base vs Tuned comparison
├── 05_build_vector_db.py             ← Step 6: Build vector search library
├── 06_run_rag_qa.py                  ← Step 7: Full RAG pipeline & report
├── cmg_qa_training_data.jsonl        ← 80 Q&A pairs (40 KB)
├── cmg_real_training_data.jsonl      ← 300 raw text chunks (355 KB)
├── cmg_lora_weights/                 ← Trained model adapter (33 MB)
├── vector_db.pt                      ← Search index (3.94 MB)
├── cmg_training_temp/                ← Training checkpoints
├── requirements.txt                  ← Python dependencies
└── venv/                             ← Python virtual environment
```

---

## Key Terminology Glossary

| Term | Plain English |
|---|---|
| **LLM / Language Model** | An AI that can read and write human language. Trained on vast text data. |
| **Parameters** | The "knowledge cells" inside the model. More parameters = more capable but more compute needed. 360M = 360 million; 7B = 7 billion. |
| **Fine-tuning** | Teaching a pre-trained model about a specific topic by showing it examples (like on-the-job training). |
| **LoRA** | A technique that fine-tunes only a small part of the model (~5%), making it fast and efficient. |
| **RAG** | Retrieval-Augmented Generation — search for relevant documents first, then have the AI write an answer based on them. Reduces hallucination. |
| **Embeddings** | Numerical "fingerprints" that represent the meaning of text. Similar meanings produce similar numbers. |
| **Vector Database** | A searchable library where every document is stored as a numerical fingerprint for fast similarity search. |
| **Hallucination** | When an AI invents facts that aren't true. A major problem we're working to eliminate. |
| **Hybrid Search** | Combining two search methods (keyword + semantic) for better results than either alone. |
| **Inference** | The process of asking the AI a question and getting an answer. |
| **CPU / GPU** | CPU = standard computer processor (what we used). GPU = graphics processor (much faster for AI — what we'll use on AWS). |

---

## How to Run This Code Locally

Anyone can run this Proof of Concept on a standard Windows/Mac/Linux laptop — **no GPU required**. 

### 1. Prerequisites
- **Python 3.10+** installed
- **Git** installed

### 2. Setup the Environment
Clone the repository and install the required dependencies:
```bash
# Clone the repository
git clone https://github.com/jtsasrani/fine_tunning_POC.git
cd fine_tunning_POC

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt
```

### 3. Run the Pipeline (Step-by-Step)
You can run the entire pipeline from scratch by executing these scripts in order:

```bash
# 1. Extract text from the official DWP PDFs
python 01_ingest_local_pdfs.py

# 2. Generate the training Q&A pairs (uses the base model)
python 01b_ingest_qa_pairs.py

# 3. Train the model! (Takes ~2 hours on a standard laptop CPU)
python 02_train_lora_windows_cpu.py

# 4. Run a quick test to see if the trained model responds
python 03_test_cmg_model.py

# 5. Evaluate the Base vs Tuned model side-by-side
python 04_evaluate_ab_test.py

# 6. Build the Vector Search Database (for RAG)
python 05_build_vector_db.py

# 7. Run the full RAG Pipeline and get comprehensive answers
python 06_run_rag_qa.py
```

### 4. Run the Interactive Web Dashboard
You can launch the visual web interface to query the system interactively:
```bash
# Start the Flask web server (preloads models, takes ~15 seconds)
python app.py
```
Once started, open your web browser and navigate to:
```text
http://127.0.0.1:5000
```
This dashboard allows you to submit custom policy queries, select pre-seeded questions, view the A/B/C comparative matrix in a modern interface, and inspect the underlying retrieved policy paragraphs.

> **Note on Model Weights:** The repository includes the pre-trained LoRA weights in the `cmg_lora_weights/` folder. If you want to skip the 2-hour training process, you can skip Step 3 and immediately run the other scripts or launch the web server directly with `python app.py`!

---

## AWS EC2 Production Service Operations (Start/Stop)

If you are running the POC in the AWS production environment (using Gunicorn on the GPU EC2 instance), use the following commands to manage the server process:

### 1. Connecting to the Instance
Connect to the EC2 GPU instance from your local terminal:
```bash
ssh -i .\gpu_poc_EUR.pem ubuntu@i-09e7b81b6184aebaa
```

### 2. Starting the Service
Navigate to the project directory and start the Gunicorn WSGI server in the background (detached using `nohup`):
```bash
cd /home/ubuntu/dwp-cmg-finetune
nohup env API_KEY='dwp-cmg-sec-key-7d9a1f8c' /opt/pytorch/bin/gunicorn -c gunicorn.conf.py app:app > server_new.log 2>&1 </dev/null &
```

### 3. Stopping the Service
Kill the active Gunicorn master and worker processes:
```bash
pkill -f gunicorn
```

### 4. Monitoring Server Logs
To check preloading progress or review client HTTP query logs in real time:
```bash
tail -f /home/ubuntu/dwp-cmg-finetune/server_new.log
```

---

## What Comes Next?

The POC successfully demonstrated that:
- ✅ DWP policy PDFs can be automatically processed into a searchable AI knowledge base
- ✅ Fine-tuning makes the model use DWP-specific language and concepts
- ✅ RAG retrieval correctly finds relevant policy paragraphs
- ⚠️ The 360M model is too small for production-quality answers

**Next step:** Scale to AWS with a GPU instance and upgrade to a 7B-parameter model (20× larger) — this will dramatically improve answer quality while keeping the same RAG architecture that already works well.
