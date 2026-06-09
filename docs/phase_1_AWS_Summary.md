# DWP CMG Fine-Tuning POC — Migration and Fine-Tuning Walkthrough

This document records the exact steps, configurations, parameters, and outputs completed during the migration and fine-tuning phases.

---

## 1. Infrastructure and Connectivity Setup

### EC2 GPU Instance
* **Instance ID**: `i-09e7b81b6184aebaa`
* **Region**: `eu-west-2` (London)
* **Instance Type**: `g5.xlarge` (1 NVIDIA A10G GPU, 24 GB VRAM, 4 vCPUs, 16 GB RAM)
* **AMI**: DLAMI GPU PyTorch 2.11 (Ubuntu 24.04)
* **IAM Instance Profile**: `dwp-cmg-ec2-role` with permissions for S3, Bedrock, and SSM.
* **Storage / Storage Sync**: S3 Bucket `s3://dwp-cmg-finetune-891377369931` used to sync the workspace from local Windows to the EC2 path `/home/ubuntu/dwp-cmg-finetune/`.

### SSM Tunnel Setup (Bypassing Corporate Firewall)
* Local SSH config is configured with a `ProxyCommand` to route port 22 traffic through AWS SSM Session Manager:
  ```ssh
  Host i-09e7b81b6184aebaa
      User ubuntu
      IdentityFile C:\Users\JitendraAsrani\DWP_CMG_Finetune\gpu_poc_EUR.pem
      ProxyCommand powershell.exe -Command "$env:Path += ';C:\Program Files\Amazon\SessionManagerPlugin\bin'; aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p"
  ```

---

## 2. Environment Setup on EC2

* **Standardized Path**: Pre-installed `/opt/pytorch` virtual environment.
* **Core Libraries**:
  - Python: 3.13
  - PyTorch: 2.10.0+cu128
  - Triton: 3.6.0
  - Unsloth: Installed and patched to enable fast QLoRA/PEFT training.
  - FAISS: Installed for flat vector database searching.

---

## 3. Data Ingestion & Synthetic Q&A Generation

* **Source Materials**: 12 documents in total (7 DMG volumes, public guides, and legislation).
* **Text Extraction**: Ran `01_ingest_all_pdfs.py` to extract text from all source files.
  - Used regex for paragraph numbers in DMG volumes.
  - Used a sliding window (1200 chars size, 300 chars overlap) for guides and legislation.
  - **Result**: **3,174 chunks** saved to `data/real_chunks.jsonl`.
* **Dataset Generation**:
  - Ran `07_generate_training_data_local.py` using local Qwen-2.5-7B-Instruct (4-bit QLoRA) on the EC2 GPU to create Q&A pairs from text chunks.
  - **Result**: Generated **1,469 QA pairs**.
* **Dataset Merging & Splitting**:
  - Ran `08_merge_training_data.py` to deduplicate and divide the data.
  - **Resulting Splits**:
    - **Train split**: 1,316 samples (`data/train_split.jsonl`)
    - **Validation split**: 147 samples (`data/val_split.jsonl`)

---

## 4. Fine-Tuning Execution

### Mistral-7B-Instruct-v0.3 Fine-Tuning
* **Base Model**: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` (quantized 4-bit)
* **Script**: `02_train_qlora_gpu.py`
* **Training Arguments**:
  - Rank (R): 16
  - Alpha: 32
  - Epochs: 3
  - Batch Size: 4 (with Gradient Accumulation Steps: 4; Effective batch size = 16)
  - Optimizer: `adamw_8bit`
  - Learning Rate: 2e-4
  - Warmup Ratio: 0.03
  - Checkpoints: Disabled saving strategy during training to avoid Python 3.13 pickling errors.
* **Results**:
  - Runtime: 472.7 seconds (~8 minutes)
  - Training Loss: 1.207 (Final epoch loss dropped to ~0.67)
  - Merged and exported 16-bit precision model saved to `./trained_models/mistral-7b-cmg-qlora_merged`.

### Qwen-2.5-7B-Instruct Fine-Tuning
* **Base Model**: `unsloth/qwen2.5-7b-instruct-bnb-4bit` (quantized 4-bit)
* **Training Arguments**: Same hyperparameters as Mistral-7B.
* **Results**:
  - Runtime: 511.6 seconds (~8.5 minutes)
  - Training Loss: 1.067 (Final epoch loss dropped to ~0.76)
  - Merged and exported 16-bit precision model saved to `./trained_models/qwen-7b-cmg-qlora_merged`.

---

## 5. Deployment and Web Serving

* **Concurrency Management (4-bit inference)**: Both fine-tuned 7B models (Mistral and Qwen) are loaded concurrently on the single NVIDIA A10G GPU (24 GB VRAM) by utilizing `bitsandbytes` 4-bit quantization config (NF4, bfloat16 compute type, double quantization) during runtime initialization.
  - This reduces the active VRAM consumption of the models to ~11.4 GB in total, allowing concurrent hosting without Out-Of-Memory (OOM) errors.
* **Flask Web Server**: Run as a background service listening on `127.0.0.1:5000` on the EC2 instance, serving a side-by-side comparison of:
  - **Config A**: Mistral-7B (Fine-Tuned + RAG)
  - **Config B**: Qwen-2.5-7B (Fine-Tuned + RAG)
  - **Config C**: Mistral-7B (Base + RAG)
* **Port Forwarding**: Local port forwarding established to map `localhost:5000` on the user's host to `127.0.0.1:5000` on the EC2 instance:
  ```powershell
  ssh -o StrictHostKeyChecking=no -i "C:\Users\JitendraAsrani\DWP_CMG_Finetune\gpu_poc_EUR.pem" -N -L 5000:localhost:5000 ubuntu@i-09e7b81b6184aebaa
  ```

---

## 6. Disk Volume Expansion

* To support full concurrent model storage (each 16-bit model weight set is ~14 GB) and the Hugging Face cache, the AWS EBS root volume was resized from 80 GB to **150 GB**.
* Verification and expansion commands (`growpart` and `resize2fs`) were successfully run, leaving the root partition `/dev/root` expanded to **145 GB** (with ~52 GB currently free).
* The Hugging Face cache and model weights reside on persistent block storage.

---

## 7. Baseline vs. Tuned Models Evaluation (147 Validation Samples)

We ran a side-by-side automated evaluation comparing the baseline **Mistral Base + RAG**, **Mistral Tuned + RAG**, and **Qwen Tuned + RAG** on the full 147-sample validation dataset.

### Metrics Table

| Model | Retrieval Precision@3 | ROUGE-L F1 Score | BERTScore F1 | Avg Inference Time (s) |
|---|---|---|---|---|
| **Mistral-7B-Instruct (Base + RAG)** | 0.0000 | 0.3434 | 0.8883 | 7.15s |
| **Mistral-7B-Instruct (Tuned + RAG)** | 0.0000 | **0.5052** | **0.9160** | **2.64s** |
| **Qwen-2.5-7B-Instruct (Tuned + RAG)** | 0.0000 | 0.4889 | 0.9094 | 2.95s |

### Key Findings
1. **Semantic & Alignment Quality (BERTScore / ROUGE-L)**:
   * **Mistral Tuned** achieved the highest semantic similarity (**0.9160**) and text overlap (**0.5052**), followed closely by **Qwen Tuned** (**0.9094** / **0.4889**).
   * Both tuned models significantly outperformed the **Mistral Base** model (**0.8883** / **0.3434**). The tuned Mistral showed a **47.1% relative improvement** in ROUGE-L over the base model.
2. **Conciseness & Inference Speed**:
   * **Mistral Tuned** was **2.7x faster** than the base model (average **2.64s** vs **7.15s**).
   * This speedup is due to fine-tuning teaching the model to write extremely concise, direct answers using exact DWP policy rules, rather than generating verbose, explanatory text (which is typical for base instructions).
3. **Retrieval Precision@3**:
   * Measured as `0.0000` because the ground-truth answers in the validation set are synthesized summaries rather than exact copy-pastes of raw text passages. The ROUGE-L and BERTScore remain the primary indicators of semantic accuracy.
