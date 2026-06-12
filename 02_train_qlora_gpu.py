import os
import json
import argparse
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback

def train(args):
    print(f"=== Starting QLoRA training for {args.model_name} ===")
    
    # 1. Load model and tokenizer in 4-bit
    print("Loading base model in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None, # Auto-detect (will use bf16 on A10G)
        device_map="auto"
    )
    
    # 2. Add LoRA adapters
    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0, # Optimized to 0 by Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # 3. Load datasets
    print(f"Loading datasets: {args.train_data} / {args.val_data}")
    dataset = load_dataset("json", data_files={"train": args.train_data, "validation": args.val_data})
    
    # Load original chunks to map paragraph_id -> context
    print("Loading chunks for RAG context mapping...")
    chunk_map = {}
    chunks_file = "data/real_chunks.jsonl"
    if os.path.exists(chunks_file):
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    chunk_map[item["paragraph_id"]] = item.get("formatted_text", item.get("text", ""))
        print(f"Loaded {len(chunk_map)} chunks for mapping.")
    else:
        print(f"Warning: {chunks_file} not found. Fallback to empty context.")
        
    # 4. Format dataset using tokenizer's chat template with system prompt and context baking
    system_prompt = (
        "You are an expert Decision Maker assistant for the DWP Child Maintenance Service (CMS). "
        "Answer questions accurately using only the provided policy context. "
        "Cite specific paragraph numbers, rules, and sections where present. "
        "If the context does not contain sufficient information, state this clearly."
    )
    
    def formatting_func(examples):
        texts = []
        for instruction, output, paragraph_id in zip(examples["instruction"], examples["output"], examples["paragraph_id"]):
            context = chunk_map.get(paragraph_id, "")
            user_content = f"Context:\n{context}\n\nQuestion: {instruction}" if context else instruction
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}
        
    train_dataset = dataset["train"].map(formatting_func, batched=True)
    val_dataset = dataset["validation"].map(formatting_func, batched=True)
    
    # 5. Set up Trainer with validation tracking and Early Stopping
    print("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False, # Set to False for instruction tuning to keep sequences separate
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        args=TrainingArguments(
            output_dir=args.output_dir + "_checkpoints",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none" # Disable external logging for simplicity
        ),
    )
    
    # 6. Train model
    print("Starting training loop...")
    trainer.train()
    print("Training completed successfully!")
    
    # 7. Save LoRA adapter
    print(f"Saving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 8. Save merged 16-bit model for inference (vLLM / llama.cpp)
    if args.save_merged:
        merged_dir = args.output_dir + "_merged"
        print(f"Saving merged 16bit model to {merged_dir}...")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        
    print("=== Fine-Tuning Process Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using Unsloth and QLoRA")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="Base model name")
    parser.add_argument("--train_data", type=str, default="data/train_split.jsonl", help="Path to training split")
    parser.add_argument("--val_data", type=str, default="data/val_split.jsonl", help="Path to validation split")
    parser.add_argument("--output_dir", type=str, default="./trained_models/qwen-14b-cms-qlora", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_merged", type=bool, default=True, help="Save merged 16bit model")
    
    args = parser.parse_args()
    train(args)
