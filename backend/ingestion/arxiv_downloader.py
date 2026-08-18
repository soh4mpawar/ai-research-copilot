"""
ArXiv PDF Batch Downloader (Phase 1 / PRD §7.1).
Downloads genuine paper PDFs from arXiv with rate limiting and robust retry headers.
Zero synthetic or dummy text fallback.
"""

import os
import time
import requests
from typing import Dict, List, Any


class ArxivDownloader:
    """Batch downloader for genuine arXiv paper PDFs."""

    def __init__(self, download_dir: str = "data/pdfs", min_delay_sec: float = 3.0):
        self.download_dir = download_dir
        self.min_delay_sec = min_delay_sec
        os.makedirs(self.download_dir, exist_ok=True)

    def get_pdf_path(self, arxiv_id: str) -> str:
        """Return target local file path for paper PDF."""
        clean_id = arxiv_id.replace("/", "_").replace(":", "_")
        return os.path.join(self.download_dir, f"{clean_id}.pdf")

    def download_paper(self, paper: Dict[str, Any]) -> str:
        """Download paper PDF if not already present on disk."""
        arxiv_id = paper.get("arxiv_id", "")
        if not arxiv_id:
            return ""

        pdf_path = self.get_pdf_path(arxiv_id)

        # Verify existing file is a valid PDF (>10 KB and starts with %PDF)
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10240:
            try:
                with open(pdf_path, "rb") as f:
                    header = f.read(5)
                    if header == b"%PDF-":
                        return pdf_path
            except Exception:
                pass

        url = paper.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"[Downloader] Downloading genuine PDF for {arxiv_id} from {url}...")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf"
        }

        try:
            resp = requests.get(url, headers=headers, timeout=15.0)
            if resp.status_code == 200 and len(resp.content) > 10240 and resp.content.startswith(b"%PDF-"):
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
                print(f"[Downloader] Successfully saved genuine PDF {pdf_path} ({len(resp.content):,} bytes)")
                time.sleep(self.min_delay_sec)
                return pdf_path
            else:
                print(f"[Downloader Warning] Failed to download valid PDF for {arxiv_id} (HTTP {resp.status_code}).")
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                return ""
        except Exception as e:
            print(f"[Downloader Error] Download exception for {arxiv_id}: {e}")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            return ""
