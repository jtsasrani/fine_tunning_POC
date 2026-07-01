import os
import re
import json
import boto3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError

# Bedrock Configuration
MODEL_ID = "amazon.nova-lite-v1:0"  # Nova Lite
REGION = "us-east-1"
MAX_WORKERS = 10  # Parallel threads

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=REGION)

def generate_qa_pair(chunk_text, bedrock_client):
    prompt = f"""You are a senior DWP child maintenance decision maker and training supervisor.
Based on the following policy/procedural extract, generate exactly 3 realistic, high-quality caseworker scenario questions and detailed answers.
The questions should represent complex scenarios, appeals, or calculations that caseworkers face.
The answers must be authoritative, directly grounded in the text, and reference specific paragraphs or rules from the text where possible.

Policy Extract:
{chunk_text}

Output format must be a valid JSON array of objects with keys "question" and "answer". Do not include markdown code block syntax, backticks, or any conversational text. Return ONLY the raw JSON array.

Example:
[
  {{
    "question": "A paying parent asserts they are currently in prison and cannot make payments. How should the case worker verify this and what action is taken on the case?",
    "answer": "The case worker must verify the imprisonment status using the internal prison lookup interface or direct contact with the prison service. Once verified, the case is placed into a suspended status with a nil rate assessment applied for the duration of the imprisonment."
  }}
]
"""

    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]
    
    # Retry configuration with backoff
    for attempt in range(5):
        try:
            # We use Bedrock Converse API which is clean and standardized across models
            response = bedrock_client.converse(
                modelId=MODEL_ID,
                messages=messages,
                inferenceConfig={
                    "temperature": 0.2,
                    "maxTokens": 1024
                }
            )
            
            output_text = response['output']['message']['content'][0]['text']
            
            # Clean up potential markdown formatting wrapping the JSON
            cleaned_text = output_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            qa_pairs = json.loads(cleaned_text)
            if isinstance(qa_pairs, list):
                return qa_pairs
            else:
                raise ValueError("Response is not a JSON list")
                
        except (ClientError, Exception) as e:
            # Check for ThrottlingException
            is_throttled = False
            if hasattr(e, 'response') and 'Error' in e.response:
                error_code = e.response['Error'].get('Code', '')
                if error_code in ['ThrottlingException', 'LimitExceededException']:
                    is_throttled = True
            
            wait_time = (2 ** attempt) + (0.5 * attempt)
            print(f"Error calling Bedrock (attempt {attempt+1}/5): {e}. "
                  f"{'Throttled, waiting' if is_throttled else 'Waiting'} {wait_time:.1f}s...")
            time.sleep(wait_time)
            
    return None

def process_chunk(chunk):
    bedrock_client = get_bedrock_client()
    chunk_id = chunk.get("paragraph_id", "unknown")
    text = chunk.get("formatted_text", chunk.get("text", ""))
    source = chunk.get("source_doc", "unknown")
    
    qa_list = generate_qa_pair(text, bedrock_client)
    results = []
    
    if qa_list:
        for item in qa_list:
            question = item.get("question")
            answer = item.get("answer")
            if question and answer:
                results.append({
                    "instruction": question,
                    "input": text,
                    "output": answer,
                    "chunk_id": chunk_id,
                    "source_doc": source
                })
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process (for dry-runs)")
    args = parser.parse_args()

    input_file = "data/real_chunks.jsonl"
    output_file = "data/synthetic_qa.jsonl"
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found. Run ingestion first.")
        return
        
    chunks = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
                
    if args.limit:
        chunks = chunks[:args.limit]
        print(f"Dry-run mode: limited to first {args.limit} chunks.")
        
    print(f"Loaded {len(chunks)} document chunks. Starting Bedrock QA generation...")
    
    generated_count = 0
    # Process concurrently using thread pool
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
            
            for i, future in enumerate(as_completed(future_to_chunk)):
                chunk = future_to_chunk[future]
                chunk_id = chunk.get("paragraph_id", "unknown")
                try:
                    qa_results = future.result()
                    if qa_results:
                        for qa in qa_results:
                            outfile.write(json.dumps(qa, ensure_ascii=False) + '\n')
                        generated_count += len(qa_results)
                    
                    if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                        print(f"Progress: {i+1}/{len(chunks)} chunks processed. Generated {generated_count} QA pairs so far.")
                        
                except Exception as exc:
                    print(f"Chunk {chunk_id} generated an exception: {exc}")
                    
    print(f"Bedrock QA Generation complete. Saved {generated_count} pairs to {output_file}")

if __name__ == "__main__":
    main()
