import os
import json
import random
import re
from collections import defaultdict

# Jaccard similarity threshold for duplicate detection
SIMILARITY_THRESHOLD = 0.85

def tokenize(text):
    # Convert to lowercase and get words
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words

def get_shingles(text, k=2):
    # Word k-grams
    words = tokenize(text)
    if len(words) < k:
        return set(words)
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i:i+k])
        shingles.add(shingle)
    return shingles

def compute_jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Pure Python MinHash LSH implementation
class SimpleMinHashLSH:
    def __init__(self, num_perm=64, threshold=SIMILARITY_THRESHOLD):
        self.num_perm = num_perm
        self.threshold = threshold
        # Find optimal bands and rows
        # threshold ~ (1/b)**(1/r)
        # For threshold = 0.85 and num_perm = 64, b=16, r=4 is a good approximation:
        # (1/16)**(1/4) = 0.50 (low false negatives)
        self.b = 16
        self.r = 4
        assert self.b * self.r == self.num_perm
        
        # Initialize permutation parameters (hash coefficients)
        # Using prime numbers for linear hashing: h(x) = (a*x + b) % p
        self.p = 2**31 - 1
        self.a = [random.randint(1, self.p - 1) for _ in range(self.num_perm)]
        self.b_coeff = [random.randint(0, self.p - 1) for _ in range(self.num_perm)]
        
        # Buckets for LSH bands: band_index -> bucket_key -> list of item_ids
        self.buckets = [defaultdict(list) for _ in range(self.b)]
        self.signatures = {}
        self.shingle_sets = {}

    def _hash_string(self, s):
        # Python's built-in hash is salted/non-deterministic across runs,
        # but we use it deterministically within a single run.
        # To avoid salt issues, we use a custom deterministic string hash.
        h = 0
        for char in s:
            h = (31 * h + ord(char)) & 0xFFFFFFFF
        return h

    def compute_signature(self, shingle_set):
        sig = [self.p] * self.num_perm
        for shingle in shingle_set:
            h_val = self._hash_string(shingle)
            for i in range(self.num_perm):
                # Apply permutation hash
                perm_hash = (self.a[i] * h_val + self.b_coeff[i]) % self.p
                if perm_hash < sig[i]:
                    sig[i] = perm_hash
        return sig

    def add(self, item_id, shingle_set):
        self.shingle_sets[item_id] = shingle_set
        sig = self.compute_signature(shingle_set)
        self.signatures[item_id] = sig
        
        # Hash bands
        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_sig = tuple(sig[start:end])
            bucket_key = hash(band_sig)
            self.buckets[band_idx][bucket_key].append(item_id)

    def query_duplicates(self, shingle_set):
        sig = self.compute_signature(shingle_set)
        candidates = set()
        
        # Collect candidates from matching buckets
        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_sig = tuple(sig[start:end])
            bucket_key = hash(band_sig)
            for candidate_id in self.buckets[band_idx][bucket_key]:
                candidates.add(candidate_id)
                    
        # Compute exact Jaccard similarity for candidates
        duplicates = []
        for candidate_id in candidates:
            cand_shingles = self.shingle_sets[candidate_id]
            sim = compute_jaccard(shingle_set, cand_shingles)
            if sim >= self.threshold:
                duplicates.append((candidate_id, sim))
        return duplicates

def main():
    input_file = "data/synthetic_qa.jsonl"
    output_train = "data/train_split.jsonl"
    output_val = "data/val_split.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found. Run QA generation first.")
        return
        
    print("Loading generated QA dataset...")
    items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                
    print(f"Loaded {len(items)} raw QA pairs. Starting LSH Deduplication...")
    
    lsh = SimpleMinHashLSH()
    deduplicated_items = []
    duplicate_count = 0
    
    for idx, item in enumerate(items):
        question = item["instruction"]
        shingles = get_shingles(question, k=2)
        
        # Query existing duplicates
        dups = lsh.query_duplicates(shingles) if idx > 0 else []
        
        if not dups:
            # Not a duplicate, add to LSH and save
            lsh.add(idx, shingles)
            deduplicated_items.append(item)
        else:
            duplicate_count += 1
            # Optional: Log duplicates
            if duplicate_count <= 5:
                dup_idx, sim = dups[0]
                print(f"Duplicate found (Sim: {sim:.2f}):")
                print(f"  Q1: {question[:80]}...")
                print(f"  Q2: {items[dup_idx]['instruction'][:80]}...")
                
        if (idx + 1) % 2000 == 0 or (idx + 1) == len(items):
            print(f"Processed {idx+1}/{len(items)} items. Found {duplicate_count} duplicates.")
            
    print(f"Deduplication complete. Retained {len(deduplicated_items)} unique QA pairs (removed {duplicate_count} duplicates).")
    
    # Shuffle and split into 90% training, 10% validation
    random.seed(42)
    random.shuffle(deduplicated_items)
    
    split_idx = int(len(deduplicated_items) * 0.9)
    train_set = deduplicated_items[:split_idx]
    val_set = deduplicated_items[split_idx:]
    
    # Write Train
    with open(output_train, 'w', encoding='utf-8') as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # Write Val
    with open(output_val, 'w', encoding='utf-8') as f:
        for item in val_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Splits generated successfully:")
    print(f"  Training Split: {len(train_set)} samples -> saved to {output_train}")
    print(f"  Validation Split: {len(val_set)} samples -> saved to {output_val}")

if __name__ == "__main__":
    main()
