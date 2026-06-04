import os
import json

# Fallback mechanism to handle Windows PyMuPDF DLL loading issues
try:
    import fitz  # PyMuPDF
    USE_PYMUPDF = True
except ImportError as e:
    print(f"Warning: PyMuPDF import failed: {e}")
    print("Falling back to pure-Python 'pypdf' library...")
    import pypdf
    USE_PYMUPDF = False

def generate_instruction(chunk_text):
    # Extract the first sentence as the core topic/context
    sentences = [s.strip() for s in chunk_text.split('.') if s.strip()]
    if sentences:
        topic = sentences[0]
        # Keep the topic description concise
        if len(topic) > 120:
            topic = topic[:117] + "..."
    else:
        topic = "maintenance calculations and variations"
        
    templates = [
        f"What is the DWP CMG policy regarding: {topic}",
        f"Explain the DWP Decision Makers' Guide rules for: {topic}",
        f"Detail the official DWP CMG guidelines on: {topic}"
    ]
    # Pick a template deterministically
    idx = len(chunk_text) % len(templates)
    return templates[idx]

def ingest_pdfs():
    source_dir = "./source_pdfs"
    output_file = "cmg_real_training_data.jsonl"
    max_chunks = 300
    
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} directory does not exist.")
        return
    
    pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        return
        
    print(f"Found {len(pdf_files)} PDF files in {source_dir}")
    
    chunks = []
    
    for pdf_file in pdf_files:
        if len(chunks) >= max_chunks:
            break
            
        pdf_path = os.path.join(source_dir, pdf_file)
        print(f"Processing {pdf_file}...")
        
        all_text = []
        try:
            if USE_PYMUPDF:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text:
                        cleaned_text = " ".join(text.split())
                        if cleaned_text:
                            all_text.append(cleaned_text)
                doc.close()
            else:
                reader = pypdf.PdfReader(pdf_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        cleaned_text = " ".join(text.split())
                        if cleaned_text:
                            all_text.append(cleaned_text)
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
            continue
            
        full_text = " ".join(all_text)
        words = full_text.split()
        
        # Chunk into blocks of roughly 175 words
        chunk_size = 175
        for i in range(0, len(words), chunk_size):
            if len(chunks) >= max_chunks:
                break
            chunk_words = words[i:i+chunk_size]
            # Keep chunks that have a substantial amount of text (at least 50 words)
            if len(chunk_words) < 50 and len(chunks) > 0:
                continue
            chunk_text = " ".join(chunk_words)
            
            # Restructured instruction generation
            instruction = generate_instruction(chunk_text)
            
            item = {
                "instruction": instruction,
                "input": "",
                "output": chunk_text
            }
            chunks.append(item)
        
    print(f"Generated {len(chunks)} chunks (Limit: {max_chunks}).")
    
    # Save to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    print(f"Saved dataset to {output_file}")

if __name__ == "__main__":
    ingest_pdfs()
