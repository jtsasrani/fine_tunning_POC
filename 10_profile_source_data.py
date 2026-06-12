import os
import re
import json
import fitz  # PyMuPDF
from collections import Counter

def clean_line(line):
    return line.strip()

def profile_source_data():
    source_dir = "./source_pdfs"
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} directory does not exist.")
        return
        
    print(f"Starting data profiling on '{source_dir}'...\n")
    
    file_stats = []
    scanned_pdfs = []
    all_repeating_lines = {}
    
    total_files = 0
    total_pdf_files = 0
    total_html_files = 0
    total_pages = 0
    
    # Traverse directory recursively
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if not (filename.lower().endswith('.pdf') or filename.lower().endswith('.html')):
                continue
                
            filepath = os.path.join(root, filename)
            rel_dir = os.path.relpath(root, source_dir)
            file_size = os.path.getsize(filepath)
            total_files += 1
            
            is_pdf = filename.lower().endswith('.pdf')
            if is_pdf:
                total_pdf_files += 1
            else:
                total_html_files += 1
                
            text_content = ""
            doc_pages = 0
            text_len = 0
            
            try:
                if is_pdf:
                    doc = fitz.open(filepath)
                    doc_pages = len(doc)
                    total_pages += doc_pages
                    
                    page_texts = []
                    line_frequencies = Counter()
                    
                    for page_num in range(doc_pages):
                        page_text = doc[page_num].get_text()
                        page_texts.append(page_text)
                        
                        # Collect lines for repeating-line analysis
                        lines = [clean_line(l) for l in page_text.split('\n') if clean_line(l)]
                        # Use unique lines per page to avoid counting multiple identical lines on the same page
                        unique_lines = set(lines)
                        for line in unique_lines:
                            line_frequencies[line] += 1
                            
                    text_content = "\n".join(page_texts)
                    text_len = len(text_content)
                    doc.close()
                    
                    # Detect scanned/image-only PDFs
                    avg_chars_per_page = text_len / doc_pages if doc_pages > 0 else 0
                    if doc_pages > 0 and avg_chars_per_page < 50:
                        scanned_pdfs.append({
                            "filename": filename,
                            "path": filepath,
                            "pages": doc_pages,
                            "avg_chars_per_page": round(avg_chars_per_page, 2)
                        })
                        
                    # Find repeating lines if doc has more than 2 pages
                    if doc_pages > 2:
                        repeating_candidates = []
                        for line, count in line_frequencies.items():
                            ratio = count / doc_pages
                            if ratio > 0.5 and len(line) < 100:
                                repeating_candidates.append({
                                    "line": line,
                                    "frequency_ratio": round(ratio, 2),
                                    "page_count": count
                                })
                        if repeating_candidates:
                            all_repeating_lines[filename] = sorted(
                                repeating_candidates, 
                                key=lambda x: x["frequency_ratio"], 
                                reverse=True
                            )
                else:
                    # HTML processing
                    doc_pages = 1  # count HTML as 1 page equivalent
                    with open(filepath, 'r', encoding='utf-8') as f:
                        raw_html = f.read()
                        # Strip standard scripts/styles before tags
                        html_cleaned = re.sub(r'(?i)<script.*?>.*?</script>', '', raw_html, flags=re.DOTALL)
                        html_cleaned = re.sub(r'(?i)<style.*?>.*?</style>', '', html_cleaned, flags=re.DOTALL)
                        text_content = re.sub('<[^<]+?>', '', html_cleaned)
                    text_len = len(text_content)
                    
            except Exception as e:
                print(f"Error profiling {filename}: {e}")
                continue
                
            file_stats.append({
                "filename": filename,
                "path": filepath,
                "relative_dir": rel_dir,
                "size_bytes": file_size,
                "pages": doc_pages,
                "text_length_chars": text_len,
                "is_pdf": is_pdf
            })
            
    # Compile aggregate stats
    total_size = sum(f["size_bytes"] for f in file_stats)
    avg_text_len = sum(f["text_length_chars"] for f in file_stats) / total_files if total_files > 0 else 0
    pdf_pages = [f["pages"] for f in file_stats if f["is_pdf"]]
    
    summary = {
        "total_files": total_files,
        "total_pdf_files": total_pdf_files,
        "total_html_files": total_html_files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "total_pages": total_pages,
        "avg_text_length_chars": round(avg_text_len, 2),
        "scanned_pdfs_count": len(scanned_pdfs),
        "scanned_pdfs": scanned_pdfs,
        "median_pages_per_pdf": sorted(pdf_pages)[len(pdf_pages)//2] if pdf_pages else 0,
        "max_pages_per_pdf": max(pdf_pages) if pdf_pages else 0,
        "min_pages_per_pdf": min(pdf_pages) if pdf_pages else 0,
    }
    
    # Save profiling output
    with open("data/profiling_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "repeating_lines": all_repeating_lines,
            "files": file_stats
        }, f, indent=2, ensure_ascii=False)
        
    print("=== PROFILING REPORT SUMMARY ===")
    print(f"Total Files Found:          {total_files} ({total_pdf_files} PDFs, {total_html_files} HTMLs)")
    print(f"Total Size:                 {summary['total_size_mb']} MB")
    print(f"Total Pages Processed:      {total_pages}")
    print(f"Average Text Length:        {summary['avg_text_length_chars']} characters")
    if pdf_pages:
        print(f"Page range per PDF:         {summary['min_pages_per_pdf']} to {summary['max_pages_per_pdf']} (Median: {summary['median_pages_per_pdf']})")
    print(f"Scanned/Low-Yield PDFs:     {len(scanned_pdfs)}")
    for sp in scanned_pdfs[:10]:
        print(f"  - {sp['filename']} ({sp['pages']} pages, {sp['avg_chars_per_page']} chars/page)")
    if len(scanned_pdfs) > 10:
        print(f"  - ... and {len(scanned_pdfs) - 10} more.")
        
    print("\nRepeating Lines Detected:")
    sample_repeating = list(all_repeating_lines.keys())[:3]
    for doc_name in sample_repeating:
        print(f"  - {doc_name}:")
        for rl in all_repeating_lines[doc_name][:3]:
            print(f"    * [{rl['frequency_ratio'] * 100}% of pages] \"{rl['line']}\"")
            
    print(f"\nProfiling complete! Saved summary and details to 'data/profiling_summary.json'.")

if __name__ == "__main__":
    profile_source_data()
