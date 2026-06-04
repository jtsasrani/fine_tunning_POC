import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
# Set PyTorch thread count to 1 for absolute stability on CPU during inference
torch.set_num_threads(1)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path = "./cmg_lora_weights"
    
    if not os.path.exists(adapter_path):
        print(f"Error: LoRA weights not found at {adapter_path}. Please run training first.")
        return
        
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading base model on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id
    )
    
    # 3 domain-specific UK child maintenance questions
    questions = [
        "Explain how the 25% income variance rule applies to a paying parent's gross weekly income.",
        "Under the Child Maintenance (Enforcement) Act 2023, what powers does the department have regarding administrative Liability Orders?",
        "How are pension contributions treated when calculating a paying parent's child maintenance liability?"
    ]
    
    base_responses = []
    tuned_responses = []
    
    # --- PHASE 1: BASE MODEL INFERENCE ---
    print("\n--- Running Base Model Inference ---")
    base_model.eval()
    for i, q in enumerate(questions):
        print(f"Generating base response for question {i+1}/3...")
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        base_responses.append(response.strip())
        
    # --- PHASE 2: LOAD LORA ADAPTERS ---
    print("\nApplying LoRA weights to base model...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()
    
    # --- PHASE 3: TUNED MODEL INFERENCE ---
    print("--- Running Tuned Model Inference ---")
    for i, q in enumerate(questions):
        print(f"Generating tuned response for question {i+1}/3...")
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        input_len = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tuned_responses.append(response.strip())
        
    # --- PHASE 4: DISPLAY REPORT ---
    print("\n" + "="*80)
    print("                  DWP CMG FINE-TUNING A/B TEST REPORT")
    print("="*80)
    
    for i, q in enumerate(questions):
        print(f"\n[QUESTION {i+1}]")
        print(f"Question: {q}\n")
        
        print("--- BASE MODEL OUTPUT (UNTRAINED) ---")
        print(base_responses[i])
        print("-" * 50)
        
        print("--- CMG TUNED MODEL OUTPUT (LORA) ---")
        print(tuned_responses[i])
        print("=" * 80)

if __name__ == "__main__":
    main()
