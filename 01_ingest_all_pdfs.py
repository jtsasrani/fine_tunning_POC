import os
import re
import json
import fitz
from collections import Counter

def clean_text(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())

def sanitize_filename(filename):
    # Strip extension
    base = os.path.splitext(filename)[0]
    # Replace non-alphanumeric with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', base)
    # Remove duplicate underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_')

def get_repeating_lines(doc):
    doc_pages = len(doc)
    if doc_pages <= 2:
        return set()
    
    line_frequencies = Counter()
    for page_num in range(doc_pages):
        page_text = doc[page_num].get_text()
        lines = [l.strip() for l in page_text.split('\n') if l.strip()]
        unique_lines = set(lines)
        for line in unique_lines:
            line_frequencies[line] += 1
            
    repeating_lines = set()
    for line, count in line_frequencies.items():
        ratio = count / doc_pages
        if ratio > 0.5 and len(line) < 100:
            repeating_lines.add(line)
    return repeating_lines

def clean_page_lines(page_text, repeating_lines):
    lines = page_text.split('\n')
    cleaned_lines = []
    
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    page_num_pattern = re.compile(r'^\d+$')
    
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        # Skip repeating header/footers
        if cleaned_line in repeating_lines:
            continue
        # Skip standalone page numbers
        if page_num_pattern.match(cleaned_line):
            continue
        # Skip URLs
        if url_pattern.match(cleaned_line):
            continue
        cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines)

def split_into_sentences(text):
    # Split on sentence boundaries (period/exclamation/question followed by space and capital letter)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text_by_paragraphs(text):
    # Split by double newlines to find paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_len = 0
    
    for paragraph in paragraphs:
        p_len = len(paragraph)
        
        # If a single paragraph is too large, split it by sentence boundary
        if p_len > 1500:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
                
            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                s_len = len(sentence)
                if current_len + s_len + (1 if current_chunk else 0) > 1500:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_len = s_len
                else:
                    current_chunk.append(sentence)
                    current_len += s_len + (1 if current_chunk else 0)
        else:
            if current_len + p_len + (2 if current_chunk else 0) > 1500:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [paragraph]
                current_len = p_len
            else:
                current_chunk.append(paragraph)
                current_len += p_len + (2 if current_chunk else 0)
                
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def ingest_all():
    source_dir = "./source_pdfs"
    output_file = "data/real_chunks.jsonl"
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} directory does not exist.")
        return
        
    chunks = []
    seen = set()
    total_files = 0
    
    # Regex to match 5-digit paragraph numbers from chapters 03 to 99
    dmg_pattern = re.compile(r'\b((?:0[3-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9])\d{3})\b')
    
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if not (filename.lower().endswith('.pdf') or filename.lower().endswith('.html')):
                continue
                
            filepath = os.path.join(root, filename)
            rel_dir = os.path.relpath(root, source_dir)
            total_files += 1
            print(f"Processing {filename}...")
            
            # Determine category based on path and name
            is_dmg = "volume" in filename.lower() or "cmdmg" in filename.lower()
            if is_dmg:
                category = "dmg"
            elif "policy_law_decision_making_guidance" in root or "policy_law_decision_making_guidance" in rel_dir:
                category = "policy_guidance"
            else:
                category = "procedure"
                
            text_content = ""
            try:
                if filename.lower().endswith('.pdf'):
                    doc = fitz.open(filepath)
                    repeating_lines = get_repeating_lines(doc)
                    if repeating_lines:
                        print(f"  Detected {len(repeating_lines)} repeating lines (headers/footers)")
                    
                    cleaned_pages = []
                    for page_num in range(len(doc)):
                        page_text = doc[page_num].get_text()
                        cleaned_page = clean_page_lines(page_text, repeating_lines)
                        cleaned_pages.append(cleaned_page)
                    
                    text_content = "\n\n".join(cleaned_pages)
                    doc.close()
                elif filename.lower().endswith('.html'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        raw_html = f.read()
                        # Strip script and style blocks first (case insensitive)
                        html_cleaned = re.sub(r'(?i)<script.*?>.*?</script>', '', raw_html, flags=re.DOTALL)
                        html_cleaned = re.sub(r'(?i)<style.*?>.*?</style>', '', html_cleaned, flags=re.DOTALL)
                        text_content = re.sub('<[^<]+?>', '', html_cleaned)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
                
            sanitized_name = sanitize_filename(filename)
            
            if category == "dmg":
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
                            formatted_text = f"Document: {filename} | Paragraph {paragraph_id}: {clean_chunk_text}"
                            chunks.append({
                                "chunk_type": "dmg",
                                "paragraph_id": f"L_{sanitized_name}_{paragraph_id}",
                                "text": clean_chunk_text,
                                "formatted_text": formatted_text,
                                "source_doc": filename
                            })
            else:
                # Use semantic paragraph chunking for policy_guidance and procedure
                raw_chunks = chunk_text_by_paragraphs(text_content)
                doc_chunks = 0
                for chunk_text in raw_chunks:
                    clean_chunk = clean_text(chunk_text)
                    if len(clean_chunk) < 80:
                        continue
                        
                    dup_key = (sanitized_name, clean_chunk[:100])
                    if dup_key not in seen:
                        seen.add(dup_key)
                        doc_chunks += 1
                        formatted_text = f"Document: {filename} | {clean_chunk}"
                        chunks.append({
                            "chunk_type": category,
                            "paragraph_id": f"L_{sanitized_name}_{doc_chunks}",
                            "text": clean_chunk,
                            "formatted_text": formatted_text,
                            "source_doc": filename
                        })
                print(f"  Generated {doc_chunks} semantic chunks (Category: {category})")
                
    print(f"\nTotal files processed: {total_files}")
    print(f"Total chunks generated: {len(chunks)}")
    
    # Save to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    # Also save to old POC filename for backward compatibility
    with open("cmg_real_training_data.jsonl", 'w', encoding='utf-8') as f:
        for chunk in chunks:
            old_format = {
                "instruction": f"Explain the DWP guidelines for {chunk.get('paragraph_id', 'general policy')}",
                "input": "",
                "output": chunk["text"]
            }
            f.write(json.dumps(old_format, ensure_ascii=False) + '\n')
            
    print(f"Saved database chunks to {output_file} and cmg_real_training_data.jsonl")

if __name__ == "__main__":
    ingest_all()
