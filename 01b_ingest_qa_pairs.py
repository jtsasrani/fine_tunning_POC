import os
import re
import json
import gc
import torch
# Let PyTorch use default threads (multi-threaded) for fast inference on CPU
# torch.set_num_threads(1)
import ctypes
try:
    # Set process priority to ABOVE_NORMAL_PRIORITY_CLASS (0x8000) to prevent Windows 11 from suspending/throttling the CPU background task
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x8000)
    print("Process priority set to ABOVE NORMAL to prevent Windows background throttling.")
except Exception as e:
    print(f"Warning: Could not set process priority: {e}")

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import fitz  # PyMuPDF
    USE_PYMUPDF = True
except ImportError:
    print("Warning: PyMuPDF import failed. Falling back to pure-Python 'pypdf'...")
    import pypdf
    USE_PYMUPDF = False

def clean_text(text):
    # Replace curly quotes and hyphens
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    # Remove non-breaking spaces and tabs
    text = text.replace("\xa0", " ").replace("\t", " ")
    # Strip any remaining non-ascii characters
    text = text.encode("ascii", "ignore").decode("ascii")
    # Normalize spacing
    return " ".join(text.split())

def extract_policy_chunks():
    source_dir = "./source_pdfs"
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} directory does not exist.")
        return []
    
    pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        return []
        
    print(f"Found {len(pdf_files)} PDF files in {source_dir}")
    raw_sections = []
    
    section_pattern = re.compile(r'\b((?:17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|49|50|95)\d{3})\b')
    keywords = ["variance", "pension", "liability", "gross weekly", "calculation", "enforcement", "deduction", "regulation", "effective date", "shared care", "historic income", "current income"]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(source_dir, pdf_file)
        print(f"Reading text from {pdf_file}...")
        
        full_text = []
        try:
            if USE_PYMUPDF:
                doc = fitz.open(pdf_path)
                for page in doc:
                    text = page.get_text()
                    if text:
                        full_text.append(text)
                doc.close()
            else:
                reader = pypdf.PdfReader(pdf_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
            continue
            
        merged_text = "\n".join(full_text)
        matches = list(section_pattern.finditer(merged_text))
        print(f"Found {len(matches)} section numbers in {pdf_file}")
        
        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i+1].start() if i + 1 < len(matches) else len(merged_text)
            chunk = clean_text(merged_text[start:end])
            
            if len(chunk) > 80:
                has_keyword = any(kw in chunk.lower() for kw in keywords)
                if has_keyword:
                    raw_sections.append(chunk)
                    
    print(f"Extracted {len(raw_sections)} raw policy sections matching keywords.")
    seen = set()
    deduped_sections = []
    for s in raw_sections:
        if s not in seen:
            seen.add(s)
            deduped_sections.append(s)
            
    print(f"Deduplicated to {len(deduped_sections)} policy sections.")
    return deduped_sections

def generate_qa_dataset():
    output_file = "cmg_qa_training_data.jsonl"
    
    # 1. Check existing progress (Resume functionality)
    existing_count = 0
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_count += 1
        print(f"Found existing {output_file} with {existing_count} records. Resuming...")
        
    # 2. Extract sections
    sections = extract_policy_chunks()
    if not sections:
        print("No sections extracted. Exiting.")
        return
        
    max_dataset_size = 80
    selected_sections = [s[:1000] for s in sections[:max_dataset_size]]
    print(f"Selected {len(selected_sections)} sections (truncated to max 1000 chars) for question generation.")
    
    if existing_count >= len(selected_sections):
        print("Dataset generation already fully complete.")
        return
        
    # 3. Load model
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    print(f"Loading local helper model {model_id} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"})
    
    print("Starting Q&A generation loop...")
    # Open file in append mode to save incrementally
    with open(output_file, "a", encoding="utf-8") as f_out:
        for idx, section in enumerate(selected_sections):
            # Skip already processed sections
            if idx < existing_count:
                continue
                
            # Strip prepended paragraph codes (e.g. 18001) from the section text
            clean_section = re.sub(r'^\s*\d{5}\s*', '', section)
            
            prompt = f"Given the following DWP policy context, write one clear, direct question that is answered by this text. Do not write multiple questions, write only one. Keep it professional.\n\nContext: {clean_section}\n\nQuestion:"
            
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Generate question with torch.no_grad()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            question = generated_text.strip()
            
            question = re.sub(r'^(Question|Q):\s*', '', question, flags=re.IGNORECASE)
            
            # Also clean question if any paragraph numbers leaked into it
            question = re.sub(r'\b\d{5}\b', '', question).strip()
            
            if len(question) > 15 and "?" in question:
                item = {
                    "instruction": question,
                    "input": "",
                    "output": clean_section
                }
                print(f"[{idx+1}/{len(selected_sections)}] Q: {question}")
            else:
                fallback_q = f"What is the DWP policy regarding the section: {clean_section[:80]}..."
                item = {
                    "instruction": fallback_q,
                    "input": "",
                    "output": clean_section
                }
                print(f"[{idx+1}/{len(selected_sections)}] Fallback Q: {fallback_q}")
                
            # Write line immediately
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()
            
            # Clear memory caches
            del inputs
            del outputs
            gc.collect()
            
    print(f"\nSuccessfully completed dataset generation and updated {output_file}")

if __name__ == "__main__":
    generate_qa_dataset()
