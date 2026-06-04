import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import torch
# Set PyTorch thread count to 1 for absolute stability on CPU during backpropagation
torch.set_num_threads(1)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

def train():
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    data_path = "cmg_qa_training_data.jsonl"
    output_dir = "./cmg_lora_weights"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run the ingestion script first.")
        return
        
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading base model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id
    )
    
    # Configure PEFT / LoRA with expanded modules and rank 16
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load and format the training dataset
    print(f"Loading data from {data_path}...")
    raw_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
                
    print(f"Loaded {len(raw_data)} samples. Formatting using chat template...")
    
    formatted_data = []
    for item in raw_data:
        # Use Hugging Face recommended Chat Template structure
        messages = [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_data.append({"text": text})
        
    dataset = Dataset.from_list(formatted_data)
    
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding=False
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized
        
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Calculate steps for exactly 2 epochs
    per_device_batch_size = 1
    gradient_accumulation_steps = 2
    num_epochs = 2
    total_samples = len(tokenized_dataset)
    steps_per_epoch = total_samples / (per_device_batch_size * gradient_accumulation_steps)
    max_steps = int(steps_per_epoch * num_epochs)
    print(f"Dataset has {total_samples} samples. Running for {num_epochs} epochs. Calculated max_steps: {max_steps}")
    
    # Setup Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Define training arguments with CPU constraints and gradient checkpointing to save memory
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir="./cmg_training_temp",
        use_cpu=True,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        logging_steps=5,
        learning_rate=1e-4,
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,  # Disable to speed up execution (360M fits easily in 16GB RAM)
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        report_to="none"
    )
    
    import inspect
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset,
        "data_collator": data_collator,
    }
    sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
        
    trainer = Trainer(**trainer_kwargs)
    
    # Disable cache to avoid warnings during training when using gradient checkpointing
    model.config.use_cache = False
    
    # Check if there are checkpoints to resume from
    resume_from_checkpoint = False
    if os.path.exists("./cmg_training_temp"):
        checkpoints = [d for d in os.listdir("./cmg_training_temp") if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from_checkpoint = True
            print(f"Found checkpoints: {checkpoints}. Resuming training from checkpoint...")
            
    print("Starting training...")
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    print(f"Saving trained adapter weights to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    train()
