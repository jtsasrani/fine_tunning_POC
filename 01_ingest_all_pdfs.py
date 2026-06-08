import os
import re
import json
import fitz

def clean_text(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())

def ingest_all():
    source_dir = "./source_pdfs"
    output_file = "data/real_chunks.jsonl"
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} directory does not exist.")
        return
    
    pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf') or f.lower().endswith('.html')]
    print(f"Found {len(pdf_files)} source files.")
    
    chunks = []
    seen = set()
    
    # Regex to match 5-digit paragraph numbers from chapters 03 to 99
    dmg_pattern = re.compile(r'\b((?:0[3-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9])\d{3})\b')
    
    for filename in pdf_files:
        filepath = os.path.join(source_dir, filename)
        print(f"Processing {filename}...")
        
        # Load text content
        text_content = ""
        try:
            if filename.lower().endswith('.pdf'):
                doc = fitz.open(filepath)
                text_content = "\n".join([page.get_text() for page in doc])
                doc.close()
            elif filename.lower().endswith('.html'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Strip basic HTML tags
                    raw_html = f.read()
                    text_content = re.sub('<[^<]+?>', '', raw_html)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        # Check if the file is a DWP DMG Volume
        is_dmg = "volume" in filename.lower() or "cmdmg" in filename.lower()
        
        if is_dmg:
            matches = list(dmg_pattern.finditer(text_content))
            print(f"  DMG Volume: Found {len(matches)} paragraph section headers")
            
            for i in range(len(matches)):
                paragraph_id = matches[i].group(1)
                start = matches[i].start()
                end = matches[i+1].start() if i + 1 < len(matches) else len(text_content)
                
                raw_chunk = clean_text(text_content[start:end])
                
                if len(raw_chunk) > 80:
                    clean_chunk_text = re.sub(r'^\s*\d{5}\s*', '', raw_chunk).strip()
                    dup_key = (paragraph_id, clean_chunk_text)
                    if dup_key not in seen:
                        seen.add(dup_key)
                        formatted_text = f"Paragraph {paragraph_id}: {clean_chunk_text}"
                        chunks.append({
                            "chunk_type": "dmg",
                            "paragraph_id": paragraph_id,
                            "text": clean_chunk_text,
                            "formatted_text": formatted_text,
                            "source_doc": filename
                        })
        else:
            # Non-DMG document (Legislation, Public Leaflet, ICE report): Use sliding window
            cleaned_full_text = clean_text(text_content)
            print(f"  Non-DMG doc: Text length {len(cleaned_full_text)} characters")
            
            # Sliding window of 1200 characters (~200 words) with 300 characters overlap
            window_size = 1200
            overlap = 300
            step = window_size - overlap
            
            doc_chunks = 0
            for start_idx in range(0, len(cleaned_full_text), step):
                chunk_text = cleaned_full_text[start_idx : start_idx + window_size].strip()
                if len(chunk_text) > 100:
                    formatted_text = f"Document [{filename}], Section {doc_chunks + 1}: {chunk_text}"
                    chunks.append({
                        "chunk_type": "legislation_or_guide",
                        "paragraph_id": f"L_{filename.split('-')[0]}_{doc_chunks+1}",
                        "text": chunk_text,
                        "formatted_text": formatted_text,
                        "source_doc": filename
                    })
                    doc_chunks += 1
            print(f"  Generated {doc_chunks} sliding window chunks")
            
    print(f"\nTotal chunks generated: {len(chunks)}")
    
    # Save to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    # Also save to old POC filename for backward compatibility
    with open("cmg_real_training_data.jsonl", 'w', encoding='utf-8') as f:
        for chunk in chunks:
            # Format to old POC style
            old_format = {
                "instruction": f"Explain the DWP guidelines for {chunk.get('paragraph_id', 'general policy')}",
                "input": "",
                "output": chunk["text"]
            }
            f.write(json.dumps(old_format, ensure_ascii=False) + '\n')
            
    print(f"Saved database chunks to {output_file} and cmg_real_training_data.jsonl")

if __name__ == "__main__":
    ingest_all()
