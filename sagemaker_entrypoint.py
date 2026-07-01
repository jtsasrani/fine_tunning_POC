import os
import argparse
import json
import torch
from datasets import load_dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# Llama 3.1 is a gated HuggingFace model — authenticate via env var set by the launcher
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
    print("HuggingFace Hub: authenticated successfully.")
else:
    print("WARNING: No HUGGING_FACE_HUB_TOKEN set. Gated models (e.g. Llama) may fail to download.")

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker paths
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--val-dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    
    # Hyperparameters
    parser.add_argument("--model_id", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=-1)
    
    return parser.parse_known_args()[0]

def main():
    args = parse_args()
    print("CUDA available:", torch.cuda.is_available())
    print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    # Reduce memory fragmentation on the 22.3GB A10G in bfloat16 mode
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # Version diagnostics — confirm actual installed versions at runtime
    import transformers as _tf; import tokenizers as _tk
    print(f"[VERSION] transformers={_tf.__version__}  tokenizers={_tk.__version__}")
    
    # Determine if Unsloth is available in the environment
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
        print("Using Unsloth optimization backend.")
    except ImportError:
        use_unsloth = False
        print("Unsloth not found. Falling back to standard Hugging Face PEFT/BitsAndBytes.")

    max_seq_length = 512   # Halved from 1024 — bfloat16 full weights use ~16GB, leaving ~6GB for activations

    if use_unsloth:
        # Load via Unsloth in bfloat16
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True
        )
    else:
        # Standard bfloat16 loading — no quantization, clean weights for serving
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16  # Load directly in bfloat16, fits within 24GB A10G VRAM
        )

        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.config.use_cache = False  # Required for gradient checkpointing
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 3. Format datasets
    train_path = os.path.join(args.train_dir, "train_split.jsonl")
    val_path = os.path.join(args.val_dir, "val_split.jsonl")
    
    print(f"Loading training data from: {train_path}")
    print(f"Loading validation data from: {val_path}")
    
    dataset_files = {
        "train": train_path,
        "validation": val_path
    }
    raw_dataset = load_dataset("json", data_files=dataset_files)
    
    def format_prompts(batch):
        """Format using Llama 3.1 chat template tokens."""
        prompts = []
        for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            text = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n"
                f"You are a DWP Child Maintenance caseworker assistant. Answer accurately using the provided context.<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"Context: {inp}\nQuestion: {inst}<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
                f"{out}<|eot_id|>"
            )
            prompts.append(text)
        return {"text": prompts}
        
    formatted_dataset = raw_dataset.map(format_prompts, batched=True)
    
    # 4. SFTTrainer config
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["validation"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,   # Pack multiple short sequences into one — eliminates padding waste, ~30% throughput boost
        args=TrainingArguments(
            per_device_train_batch_size=1,  # Must stay at 1 for bfloat16 on 24GB A10G
            gradient_accumulation_steps=8,  # 8 steps * batch 1 = effective batch size 8
            per_device_eval_batch_size=1,   # Prevents OOM during evaluation at step 50 (defaults to 8!)
            eval_accumulation_steps=4,      # Reduces memory build-up during evaluation
            warmup_steps=10,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="no",
            optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="/tmp/outputs",
            report_to="none",
            dataloader_pin_memory=False,  # Reduces CPU↔GPU memory pressure
        ),
    )
    
    print("Starting SageMaker SFT Training Job...")
    trainer.train()
    print("Training complete!")
    
    # 5. Merge LoRA adapters into clean bfloat16 weights and save to SM_MODEL_DIR
    print(f"Merging LoRA adapters and saving merged bfloat16 model to {args.model_dir}...")
    if use_unsloth:
        model.save_pretrained_merged(args.model_dir, tokenizer, save_method="merged_16bit")
    else:
        # Merge and explicitly cast to bfloat16 before saving — guarantees clean float tensors
        merged_model = model.merge_and_unload()
        merged_model = merged_model.to(torch.bfloat16)
        merged_model.save_pretrained(args.model_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.model_dir)

    print(f"Model saved successfully in {args.model_dir}!")

if __name__ == "__main__":
    main()
