"""
Fast Multi-Threaded Genuine arXiv PDF Downloader (Phase 1 / PRD §7.1.1).
Downloads genuine PDFs for all 200 snowball-sampled papers using 5 concurrent workers.
Enforces %PDF- header validation and zero synthetic fallback.
"""

import sys
import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DOWNLOAD_DIR = "data/pdfs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_single_paper(paper: dict) -> tuple:
    aid = paper.get("arxiv_id", "")
    title = paper.get("title", "Paper")
    if not aid:
        return (aid, False, "No arXiv ID")

    clean_id = aid.replace("/", "_").replace(":", "_")
    pdf_path = os.path.join(DOWNLOAD_DIR, f"{clean_id}.pdf")

    # Check if already downloaded and valid
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10240:
        try:
            with open(pdf_path, "rb") as f:
                if f.read(5) == b"%PDF-":
                    return (aid, True, "Already Exists")
        except Exception:
            pass

    url = paper.get("pdf_url") or f"https://arxiv.org/pdf/{aid}.pdf"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=12.0)
        if resp.status_code == 200 and len(resp.content) > 10240 and resp.content.startswith(b"%PDF-"):
            with open(pdf_path, "wb") as f:
                f.write(resp.content)
            return (aid, True, f"Downloaded ({len(resp.content):,} bytes)")
        else:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            return (aid, False, f"HTTP {resp.status_code}")
    except Exception as e:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        return (aid, False, str(e))


def download_full_corpus_fast(target_count: int = 200):
    print("==========================================================================")
    print(f"FAST MULTI-THREADED DOWNLOAD OF ALL {target_count} GENUINE ARXIV PDFs")
    print("==========================================================================")

    file_path = "data/metadata/sampled_papers.json"
    with open(file_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} candidate paper entries. Spawning 5 parallel workers...")

    downloaded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_single_paper, p): p for p in papers[:target_count]}
        for idx, future in enumerate(as_completed(futures), 1):
            aid, success, status = future.result()
            if success:
                downloaded += 1
            else:
                failed += 1
            if idx % 20 == 0 or idx == target_count:
                print(f"[Download Progress] {idx}/{target_count} papers processed | Verified Valid PDFs: {downloaded}")

    print("\n--------------------------------------------------------------------------")
    print(f"FAST DOWNLOAD SUMMARY:")
    print(f"  • Verified Valid Genuine PDFs: {downloaded} / {target_count}")
    print(f"  • Failed Downloads:            {failed}")
    print("==========================================================================")
    return downloaded


if __name__ == "__main__":
    download_full_corpus_fast(200)
