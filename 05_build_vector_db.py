import os
import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Disable offline check to ensure HuggingFace downloads BGE-base if not cached.
# Once cached, it can run offline.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

def clean_text(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())

def sub_chunk_if_needed(chunks, tokenizer, max_tokens=500, overlap=100):
    new_chunks = []
    print(f"Checking {len(chunks)} chunks for sub-chunking...")
    
    for c in chunks:
        formatted_text = c.get("formatted_text", "")
        tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            new_chunks.append(c)
        else:
            text = c.get("text", "")
            text_tokens = tokenizer.encode(text, add_special_tokens=False)
            
            print(f"  Sub-chunking chunk {c.get('paragraph_id')} from {c.get('source_doc')} ({len(tokens)} tokens)")
            
            step = max_tokens - overlap
            part_idx = 1
            for start_idx in range(0, len(text_tokens), step):
                end_idx = min(start_idx + max_tokens, len(text_tokens))
                sub_tokens = text_tokens[start_idx:end_idx]
                
                # Skip tiny trailing fragments
                if len(sub_tokens) < 30 and part_idx > 1:
                    continue
                    
                sub_text = tokenizer.decode(sub_tokens, skip_special_tokens=True)
                paragraph_id = c.get("paragraph_id", "")
                source_doc = c.get("source_doc", "")
                chunk_type = c.get("chunk_type", "dmg")
                
                if chunk_type == "dmg":
                    sub_formatted = f"Paragraph {paragraph_id} (Part {part_idx}): {sub_text}"
                else:
                    sub_formatted = f"Document [{source_doc}], Section {paragraph_id.split('_')[-1]} (Part {part_idx}): {sub_text}"
                
                new_chunks.append({
                    "chunk_type": chunk_type,
                    "paragraph_id": f"{paragraph_id}_part{part_idx}",
                    "text": sub_text,
                    "formatted_text": sub_formatted,
                    "source_doc": source_doc
                })
                part_idx += 1
                
    print(f"Total chunks after sub-chunking: {len(new_chunks)}")
    return new_chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    # 1. Load pre-ingested chunks from real_chunks.jsonl
    chunks_file = "data/real_chunks.jsonl"
    if not os.path.exists(chunks_file):
        print(f"Error: {chunks_file} not found. Please run 01_ingest_all_pdfs.py first.")
        return
        
    print(f"Loading chunks from {chunks_file}...")
    raw_chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_chunks.append(json.loads(line.strip()))
    print(f"Loaded {len(raw_chunks)} raw chunks.")

    # 2. Load embedding model BGE-base
    embedding_model_name = "BAAI/bge-base-en-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model '{embedding_model_name}' on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name).to(device)
    model.eval()

    # 3. Apply sub-chunking
    chunks = sub_chunk_if_needed(raw_chunks, tokenizer, max_tokens=500, overlap=100)

    # 4. Generate embeddings
    print("Generating embeddings for all chunks...")
    batch_size = 64
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["formatted_text"] for c in batch]
            
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            model_output = model(**inputs)
            
            embeddings = mean_pooling(model_output, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Keep on CPU to collect
            all_embeddings.append(embeddings.cpu())
            if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(chunks):
                print(f"Embedded batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    print("Completed embedding generation. Final tensor shape:", embeddings_tensor.shape)
    
    # 5. Build and save FAISS index
    print("Building FAISS flat index (IndexFlatIP, dimension 768)...")
    embeddings_np = embeddings_tensor.numpy().astype('float32')
    
    index = faiss.IndexFlatIP(768)
    index.add(embeddings_np)
    
    faiss_file = "vector_db.index"
    print(f"Saving FAISS index to {faiss_file}...")
    faiss.write_index(index, faiss_file)
    
    # 6. Save metadata and embeddings tensor to vector_db.pt (for compatibility)
    db_file = "vector_db.pt"
    db_data = {
        "model_name": embedding_model_name,
        "chunks": chunks,
        "embeddings": embeddings_tensor
    }
    
    print(f"Saving vector database metadata to {db_file}...")
    torch.save(db_data, db_file)
    print("Vector database successfully built, indexed, and saved!")

if __name__ == "__main__":
    main()

