# Standard Technical Manual: Model Fine-Tuning Operations & Hyperparameter Guide

**Document Control**
* **Title**: DWP CMS Decision Support Suite - Fine-Tuning & Model Lifecycle Guide
* **Version**: 2.0 (Production Release)
* **Author**: Lead AI Researcher
* **Target Audience**: AI/ML Engineers, Data Scientists
* **Status**: Approved

---

## 🧠 1. Fine-Tuning Methodology

The DWP CMS Decision Support model is trained using **QLoRA (Quantized Low-Rank Adaptation)** on top of `Qwen/Qwen2.5-14B-Instruct`. 

### Key Concepts
* **4-Bit Base Quantization**: The base 14B parameter model is loaded in 4-bit NormalFloat (NF4) precision to minimize the VRAM footprint.
* **Low-Rank Adapters (LoRA)**: Trainable low-rank weight matrices are injected into the attention layers and feed-forward projections. This freezes the baseline model's extensive parametric knowledge while adapting it to the structured formats, styles, and policy citation constraints required by DWP casework.
* **Target Modules**: Adapters are applied across all linear layers to ensure optimal capacity representation:
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

---

## ⚙️ 2. Synthetic Data Generation & Preprocessing Pipeline

Before fine-tuning, training data is synthetically generated from policy manuals and filtered for high accuracy.

```
01_ingest_all_pdfs.py ──> real_chunks.jsonl ──> 07_generate_training_data_local.py
                                                           │
                                                           ▼
08_merge_training_data.py <── MinHash LSH <── llm_generated_training_data.jsonl
         │
         ├──> train_split.jsonl (90%)
         ├──> val_split.jsonl   (10%)
         └──> evaluation_set.jsonl (Golden 35-item split)
```

### A. Data Generation (`07_generate_training_data_local.py`)
To bootstrap domain-specific instruction pairs, a **Qwen2.5-32B-Instruct** teacher model is utilized to run a dual-pass dataset generation pipeline:
1. **Pass 1: Question & Answer Generation**: For each chunk in `data/real_chunks.jsonl`, the model generates a realistic caseworker query along with a statutory response grounded strictly in the chunk's text.
2. **Pass 2: Tier-2 Judge Verification**: The generated question and answer are passed back to the model in a low-temperature inference pass (`temperature=0.1`). The model is prompted to judge if the answer contains external assumptions or is fully supported. Non-conforming samples (labeled `FAIL`) are immediately discarded.
* *Resource Footprint*: The teacher model uses $\approx 19.5\text{ GB}$ of VRAM (4-bit quantization). Processing the full manual set takes **$\approx 34$ hours** on a single A10G GPU.

### B. Deduplication and Splitting (`08_merge_training_data.py`)
To prevent the student model from overfitting on similar synthetic constructs:
* **Fuzzy Deduplication**: Implements **MinHash LSH** (Locality Sensitive Hashing) via the `datasketch` library with a Jaccard similarity threshold of `0.85` across 128 permutations.
* **Dataset Splitting**: Shuffles the deduplicated dataset and splits it into:
  * **Train Split (90%)**: `data/train_split.jsonl` - Used by the trainer.
  * **Validation Split (10%)**: `data/val_split.jsonl` - Used for epoch evaluation.
  * **Golden Evaluation Set**: Extracts the first 35 items from the validation split into `data/evaluation_set.jsonl` as a gold-standard benchmarking set.

---

## 📋 3. Training Dataset Schema & Prompt Baking

The training dataset splits are stored as JSON Lines (`.jsonl`). Each object contains:
```json
{
  "instruction": "Caseworker query text...",
  "output": "Statutory policy-grounded answer text...",
  "paragraph_id": "document_name_paragraph_index"
}
```

### Context & System Prompt Baking
During data collation in the SFT Trainer, the context paragraph matching the `paragraph_id` is fetched and formatted as:
```text
System: You are an expert Decision Maker assistant for the DWP Child Maintenance Service (CMS). Answer questions accurately using only the provided policy context. Cite specific paragraph numbers, rules, and sections where present. If the context does not contain sufficient information, state this clearly.

User: Context: [Fetched Paragraph Text]
Question: [Instruction]

Assistant: [Output]
```
This formats the training examples exactly like the active application prompt, ensuring consistent instruction-following behavior during production inference.

---

## ⚙️ 4. Hyperparameter Reference Table

These parameters are defined in [02_train_qlora_gpu.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/02_train_qlora_gpu.py):

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `max_seq_length` | `2048` | Max token context window. Restricts VRAM spikes during long generations. |
| `lora_r` | `32` | Rank dimension. Provides a balanced capacity for learning citation structures. |
| `lora_alpha` | `64` | Scale multiplier. Usually set to $2 \times$ `lora_r` for scaling gradient updates. |
| `learning_rate` | `1e-4` | Peak learning rate. Prevents catastrophic forgetting of base model reasoning. |
| `lr_scheduler_type` | `"cosine"` | Cosine decay schedule. Gradually decreases learning rate to stabilize convergence. |
| `warmup_ratio` | `0.05` | Linear warmup over the first 5% of training steps to prevent gradient explosion. |
| `epochs` | `3` | Number of complete passes over the training set. |
| `batch_size` | `1` | Per-device batch size. Kept at `1` to avoid VRAM fragmentation. |
| `gradient_accumulation_steps` | `16` | Accumulates gradients over 16 steps before updating weights. Effective batch size = 16. |
| `weight_decay` | `0.01` | L2 weight regularizer applied to prevent overfitting on specific phrasing. |
| `optim` | `"adamw_8bit"`| 8-bit AdamW optimizer. Saves 75% VRAM compared to 32-bit. |

---

## ⚡ 5. GPU VRAM Optimization Techniques

Training a 14B model on a single 24GB VRAM GPU (A10G) is made possible through:
1. **Unsloth Triton Kernels**: Replaces standard PyTorch linear layers with highly optimized Triton kernels, reducing VRAM usage by ~60% and speeding up training by $2\times$.
2. **Expandable Segments**: Pre-allocates memory segments to prevent allocation fragmentation:
   ```python
   os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
   ```
3. **Gradient Checkpointing**: Activates `unsloth` gradient checkpointing to recompute activations during backpropagation instead of caching them in GPU memory.

---

## 🏃 6. Running Training & Weight Merging

### Execute Training
Run the training script on the host machine:
```bash
conda activate pytorch
python3 02_train_qlora_gpu.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" \
    --train_data "data/train_split.jsonl" \
    --val_data "data/val_split.jsonl" \
    --output_dir "./trained_models/qwen-14b-cms-qlora" \
    --epochs 3 \
    --save_merged True
```

### Weight Merging
On completion, the script merges the LoRA parameters back into the base 16-bit model weights:
```python
# Internal script logic executing unsloth merged save
model.save_pretrained_merged(
    "./trained_models/qwen-14b-cms-qlora_merged",
    tokenizer,
    save_method="merged_16bit"
)
```
This outputs a standalone folder `./trained_models/qwen-14b-cms-qlora_merged/` containing standard Hugging Face weights (`.safetensors`), ready for deployment in vLLM.

---

## 🔬 7. Model Evaluation Suite & Metrics

Post-training, run the evaluation suite to measure performance metrics:
```bash
python3 09_evaluate_model.py --model_path "./trained_models/qwen-14b-cms-qlora_merged"
```
The script runs the model in deterministic greedy mode (`temperature=0.0`) against the Golden Evaluation Set and outputs:

1. **Generation Speed**: Measured in tokens per second. (Target: $> 25\text{ tok/sec}$ under vLLM).
2. **ROUGE-L Score**: Measures exact n-gram matching and LCS sequence overlap against golden ground truth answers.
3. **BERTScore F1**: Evaluates semantic similarity using contextual embeddings, capturing correct policy statements even if formulated using different terminology.
4. **VRAM Footprint**: Logs peak VRAM usage during model generation.

