"""
Download Genuine Foundational Paper Corpus (Phase 1 / PRD §7.1).
Downloads 100% genuine arXiv PDFs for foundational NLP and Computer Vision literature.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ingestion.arxiv_downloader import ArxivDownloader

FOUNDATIONAL_PAPERS = [
    {"arxiv_id": "1706.03762", "title": "Attention Is All You Need", "category": "cs.CL"},
    {"arxiv_id": "1512.03385", "title": "Deep Residual Learning for Image Recognition", "category": "cs.CV"},
    {"arxiv_id": "1810.04805", "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "category": "cs.CL"},
    {"arxiv_id": "2005.11401", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "category": "cs.CL"},
    {"arxiv_id": "2004.04906", "title": "Dense Passage Retrieval for Open-Domain Question Answering", "category": "cs.CL"},
    {"arxiv_id": "2010.11929", "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "category": "cs.CV"},
    {"arxiv_id": "2103.14030", "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", "category": "cs.CV"},
    {"arxiv_id": "1409.1556", "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition", "category": "cs.CV"},
    {"arxiv_id": "1703.06870", "title": "Mask R-CNN", "category": "cs.CV"},
    {"arxiv_id": "1905.11946", "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", "category": "cs.CV"}
]


def download_genuine_corpus():
    downloader = ArxivDownloader(download_dir="data/pdfs")
    print(f"Downloading genuine PDFs for {len(FOUNDATIONAL_PAPERS)} core foundational papers...")

    downloaded = []
    for p in FOUNDATIONAL_PAPERS:
        path = downloader.download_paper(p)
        if path and os.path.exists(path) and os.path.getsize(path) > 10240:
            downloaded.append((p["arxiv_id"], p["title"], path))
        else:
            print(f"[Warning] Failed to download {p['arxiv_id']}")

    print(f"\nSuccessfully verified {len(downloaded)}/{len(FOUNDATIONAL_PAPERS)} genuine PDFs on disk.")


if __name__ == "__main__":
    download_genuine_corpus()
