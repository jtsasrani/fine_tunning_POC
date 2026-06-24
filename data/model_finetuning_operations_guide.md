# DWP CMS Decision Support Suite: Model Fine-Tuning Operations & Hyperparameter Guide

This guide details the fine-tuning methodology, dataset formatting, hyperparameter configurations, and GPU memory optimizations implemented in the DWP CMS Decision Support Suite. Use this reference when executing future training cycles to update or expand the model's domain expertise.

---

## 🧠 Fine-Tuning Methodology

The model is trained using **QLoRA (Quantized Low-Rank Adaptation)** on the base model `Qwen/Qwen2.5-14B-Instruct`. QLoRA loads the base model weights at 4-bit precision (NF4 quantization) and injects trainable low-rank adapters into the attention and projection layers, drastically reducing the required training VRAM while preserving downstream model reasoning.

---

## 📋 Training Dataset Formatting

The training pipeline requires a JSON Lines (`.jsonl`) dataset with the following structure:
```json
{
  "instruction": "Caseworker question text...",
  "output": "Statutory policy-grounded answer text...",
  "paragraph_id": "document_name_paragraph_index"
}
```
* **System Prompt Baking**: During ingestion, the script automatically formats each record using Qwen's chat template, embedding the official CMS system prompt:
  ```text
  You are an expert Decision Maker assistant for the DWP Child Maintenance Service (CMS). Answer questions accurately using only the provided policy context. Cite specific paragraph numbers, rules, and sections where present. If the context does not contain sufficient information, state this clearly.
  ```
* **Context Baking**: The text content corresponding to the `paragraph_id` is retrieved from `data/real_chunks.jsonl` and appended to the user instruction under a `Context:` prefix, training the model to prioritize retrieved reference data over its pre-trained parametric memory.

---

## ⚙️ Hyperparameter Configuration Reference

The following parameters are configured in [02_train_qlora_gpu.py](file:///c:/Users/JitendraAsrani/DWP_CMG_Finetune/02_train_qlora_gpu.py) for the fine-tuning run:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `max_seq_length` | `2048` | Maximum token sequence length allowed per training sample. |
| `lora_r` | `32` | LoRA rank (dimension of the low-rank update matrices). |
| `lora_alpha` | `64` | LoRA scaling factor (typically set to $2 \times$ `lora_r`). |
| `learning_rate` | `1e-4` | Peak learning rate for the AdamW optimizer. |
| `lr_scheduler_type` | `"cosine"` | Cosine decay schedule for adjusting learning rate during epochs. |
| `warmup_ratio` | `0.05` | First 5% of training steps are used to warm up learning rate linearly. |
| `epochs` | `3` | Total number of passes over the training dataset. |
| `batch_size` | `1` | Per-device training batch size. Keep at `1` to avoid VRAM OOM errors. |
| `gradient_accumulation_steps` | `16` | Number of steps to accumulate gradients before executing an optimizer step. Effective batch size = $1 \times 16 = 16$. |
| `weight_decay` | `0.01` | L2 weight regularization factor to prevent overfitting. |

---

## ⚡ GPU VRAM Optimizations

To train a 14-billion parameter model on a single NVIDIA A10G (24GB VRAM) GPU, the following memory optimizations must be active:

1. **Unsloth Fast Language Model**: Utilizes specialized Triton kernels that reduce VRAM overhead by 60% compared to standard Hugging Face PEFT.
2. **Expandable Segments**: Enabled via PyTorch environment flags:
   ```python
   os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
   ```
   This prevents memory fragmentation, keeping memory allocation contiguous.
3. **8-Bit Optimizer**: Configured with `optim="adamw_8bit"` to reduce optimizer state VRAM footprint by 75% compared to 32-bit AdamW.
4. **Gradient Checkpointing**: Configured to recalculate intermediate activations during backward passes instead of saving them:
   ```python
   use_gradient_checkpointing="unsloth"
   ```

---

## 🏃 Execution & Merging

Follow these commands to run a fine-tuning cycle:

```bash
# 1. Activate the GPU conda environment
conda activate pytorch

# 2. Run the training script
# The script will save the LoRA adapter AND export the merged 16-bit model
python3 02_train_qlora_gpu.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" \
    --train_data "data/train_split.jsonl" \
    --val_data "data/val_split.jsonl" \
    --output_dir "./trained_models/qwen-14b-cms-qlora" \
    --epochs 3 \
    --save_merged True
```

### Weight Merging
The training script automatically executes `model.save_pretrained_merged()` to combine base model weights and LoRA adapters:
* **Merged folder output**: `./trained_models/qwen-14b-cms-qlora_merged`
* **Format**: Standard FP16/BF16 Hugging Face weight tensors (`.safetensors`).
* **Deployability**: Ready to be loaded immediately by the backend `vLLM` engine for zero-downtime server updates.
