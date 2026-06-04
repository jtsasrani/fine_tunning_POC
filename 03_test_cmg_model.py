import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path = "./cmg_lora_weights"
    
    if not os.path.exists(adapter_path):
        print(f"Error: LoRA weights not found at {adapter_path}. Run training script first.")
        return
        
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading base model on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": "cpu"}
    )
    
    print(f"Loading LoRA weights from {adapter_path} and merging...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Define a test prompt asking a question relevant to UK Child Maintenance
    test_question = "Detail the DWP CMG policy regarding the following context: maintenance calculations and variations."
    print(f"\nPrompt: {test_question}")
    
    # Format with chat template
    messages = [
        {"role": "user", "content": test_question}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Extract only the generated response
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("\n" + "="*40)
    print("GENERATED RESPONSE:")
    print("="*40)
    print(response.strip())
    print("="*40)

if __name__ == "__main__":
    main()
