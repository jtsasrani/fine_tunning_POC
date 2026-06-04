import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModel

# Configure thread usage and local execution (Commented out for fast multi-threaded embedding generation on CPU)
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# torch.set_num_threads(1)

# Enable offline loading for models that are already cached
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    import fitz  # PyMuPDF
    USE_PYMUPDF = True
except (ImportError, Exception):
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

def extract_policy_paragraphs():
    source_dir = "./source_pdfs"
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} directory does not exist.")
        return []
    
    pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {source_dir}")
        return []
        
    print(f"Found {len(pdf_files)} PDF files in {source_dir} (Using PyMuPDF: {USE_PYMUPDF})")
    
    # Pattern to match 5-digit DWP paragraph numbers (e.g. 17001, 18234, 63001) covering chapters 17-36 and 49-95
    section_pattern = re.compile(r'\b((?:1[7-9]|2[0-9]|3[0-6]|4[9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-5])\d{3})\b')
    
    chunks = []
    seen = set()
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(source_dir, pdf_file)
        print(f"Processing {pdf_file}...")
        
        full_text = []
        try:
            if USE_PYMUPDF:
                try:
                    doc = fitz.open(pdf_path)
                    for page in doc:
                        text = page.get_text()
                        if text:
                            full_text.append(text)
                    doc.close()
                except Exception as e:
                    print(f"PyMuPDF open failed: {e}. Falling back to pypdf...")
                    reader = pypdf.PdfReader(pdf_path)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            full_text.append(text)
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
        print(f"Found {len(matches)} section headers in {pdf_file}")
        
        for i in range(len(matches)):
            paragraph_id = matches[i].group(1)
            start = matches[i].start()
            end = matches[i+1].start() if i + 1 < len(matches) else len(merged_text)
            
            raw_chunk = clean_text(merged_text[start:end])
            
            if len(raw_chunk) > 80:
                # Strip the prepended paragraph number from raw chunk to get clean text
                clean_chunk_text = re.sub(r'^\s*\d{5}\s*', '', raw_chunk).strip()
                
                # Check for duplicates using paragraph_id and the text content
                dup_key = (paragraph_id, clean_chunk_text)
                if dup_key not in seen:
                    seen.add(dup_key)
                    
                    # Formatted text explicitly tags the paragraph ID for retrieval clarity
                    formatted_text = f"Paragraph {paragraph_id}: {clean_chunk_text}"
                    
                    chunks.append({
                        "paragraph_id": paragraph_id,
                        "text": clean_chunk_text,
                        "formatted_text": formatted_text,
                        "source_doc": pdf_file
                    })
                    
    print(f"Extracted and deduplicated {len(chunks)} paragraph chunks.")
    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    # 1. Extract chunks from PDFs
    chunks = extract_policy_paragraphs()
    if not chunks:
        print("No paragraphs extracted. Vector DB build aborted.")
        return
        
    # 2. Load embedding model
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model '{embedding_model_name}' on CPU...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        model = AutoModel.from_pretrained(embedding_model_name)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Make sure you run the test script first to populate cache, or check network.")
        return
        
    model.eval()
    
    # 3. Generate embeddings
    print("Generating embeddings for all chunks...")
    batch_size = 32
    all_embeddings = []
    
    # Wrap in torch.no_grad for inference
    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            # Embed the formatted text (includes paragraph ID) to represent the chunk fully
            texts = [c["formatted_text"] for c in batch]
            
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            model_output = model(**inputs)
            
            # Mean pooling
            embeddings = mean_pooling(model_output, inputs["attention_mask"])
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings)
            print(f"Embedded batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
    # Concatenate all embeddings into a single tensor
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    print("Completed embedding generation. Final tensor shape:", embeddings_tensor.shape)
    
    # 4. Save to vector_db.pt
    db_file = "vector_db.pt"
    db_data = {
        "model_name": embedding_model_name,
        "chunks": chunks,
        "embeddings": embeddings_tensor
    }
    
    print(f"Saving vector database to {db_file}...")
    torch.save(db_data, db_file)
    print("Vector database successfully built and saved!")

if __name__ == "__main__":
    main()
