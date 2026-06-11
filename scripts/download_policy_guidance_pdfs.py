import os
import re
import urllib.request
import urllib.parse
from pathlib import Path

def main():
    content_file = r"C:\Users\JitendraAsrani\.gemini\antigravity\brain\7751c5ad-b015-47f0-98e0-47c9d97c740e\.system_generated\steps\40\content.md"
    output_dir = Path(r"c:\Users\JitendraAsrani\DWP_CMG_Finetune\source_pdfs\policy_law_decision_making_guidance")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(content_file):
        print(f"Error: Content file {content_file} does not exist.")
        return

    print("Reading content file...")
    with open(content_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Find all links ending with .pdf
    pdf_regex = re.compile(r'(?:href=["\']|\]\()([^"\')]+\.pdf)(?:["\']|\))', re.IGNORECASE)
    pdf_urls = pdf_regex.findall(html_content)

    # De-duplicate links and clean them
    unique_urls = []
    seen = set()
    base_url = "https://www.voiceofthechild.org.uk/kb/child-maintenance-service-policy-law-and-decision-making-guidance/"

    for url in pdf_urls:
        full_url = urllib.parse.urljoin(base_url, url.strip())
        if full_url not in seen:
            seen.add(full_url)
            unique_urls.append(full_url)

    print(f"Found {len(unique_urls)} unique PDF links.")
    
    # Print a few examples
    for i, url in enumerate(unique_urls[:10]):
        print(f"  {i+1}: {url}")
    if len(unique_urls) > 10:
        print("  ...")

    print("\nStarting downloads...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    success_count = 0
    failure_count = 0

    for idx, url in enumerate(unique_urls):
        parsed = urllib.parse.urlparse(url)
        quoted_path = urllib.parse.quote(parsed.path)
        quoted_url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, quoted_path, parsed.params, parsed.query, parsed.fragment))

        filename = os.path.basename(parsed.path)
        clean_filename = filename.replace('\u2013', '-')
        
        target_path = output_dir / clean_filename
        
        if target_path.exists():
            if target_path.stat().st_size > 1024:
                success_count += 1
                continue

        print(f"[{idx+1}/{len(unique_urls)}] Downloading {clean_filename}...")
        try:
            req = urllib.request.Request(quoted_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(target_path, "wb") as out_file:
                    out_file.write(response.read())
            print(f"  Successfully downloaded: {clean_filename}")
            success_count += 1
        except Exception as e:
            print(f"  Failed to download {url}: {e}")
            failure_count += 1

    print(f"\nDownload process complete: {success_count} succeeded/skipped, {failure_count} failed.")

if __name__ == "__main__":
    main()
