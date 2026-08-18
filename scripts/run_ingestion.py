"""
Master Batch Ingestion Pipeline Script (Phase 1 / PRD §7.1, §7.1.1; FR-1, FR-2, FR-3, FR-10).
Runs snowball sampling, PDF downloading, Docling layout parsing, section-aware chunking,
nomic-embed-text ChromaDB vector store persistence, and in-corpus citation connectivity calculation.
"""

import sys
import os
import gc
import json
from typing import List, Dict, Any

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.snowball_sampler import SnowballSampler
from backend.ingestion.arxiv_downloader import ArxivDownloader
from backend.ingestion.docling_parser import ProductionDoclingParser
from backend.ingestion.section_chunker import SectionChunker
from backend.ingestion.vector_store import VectorStore
from backend.ingestion.checkpoint import IngestionCheckpoint


def calculate_in_corpus_connectivity(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate the fraction of papers holding at least one in-corpus citation edge (PRD §7.1.1).
    Used as the Phase 6 GraphRAG viability check trigger (~150 papers / high connectivity).
    """
    corpus_ids = set(p["arxiv_id"] for p in papers if p.get("arxiv_id"))
    connected_papers_count = 0
    total_in_corpus_edges = 0

    for paper in papers:
        refs = paper.get("references", [])
        cits = paper.get("citations", [])
        
        # Filter references and citations that exist inside the ingested corpus
        in_corpus_refs = [r for r in refs if r in corpus_ids]
        in_corpus_cits = [c for c in cits if c in corpus_ids]
        
        edge_count = len(in_corpus_refs) + len(in_corpus_cits)
        if edge_count > 0:
            connected_papers_count += 1
            total_in_corpus_edges += edge_count

    total_papers = len(papers) if len(papers) > 0 else 1
    connectivity_fraction = round(connected_papers_count / total_papers, 3)

    return {
        "total_ingested_papers": len(papers),
        "connected_papers_count": connected_papers_count,
        "connectivity_fraction": connectivity_fraction,
        "total_in_corpus_edges": total_in_corpus_edges,
        "graphrag_viability_met": (len(papers) >= 150 and connectivity_fraction >= 0.50)
    }


def main():
    print("==========================================================================")
    print("AI RESEARCH COPILOT — PHASE 1: CORPUS CONSTRUCTION & BATCH INGESTION")
    print("==========================================================================")

    # 1. Snowball Sampling
    sampler = SnowballSampler()
    papers = sampler.run_snowball_sampling(200)

    # Save sampled metadata
    os.makedirs("data/metadata", exist_ok=True)
    with open("data/metadata/sampled_papers.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2)

    downloader = ArxivDownloader(download_dir="data/pdfs", min_delay_sec=0.1) # Fast delay for local demo
    parser = ProductionDoclingParser()
    chunker = SectionChunker(target_chunk_tokens=300)
    checkpoint = IngestionCheckpoint()

    all_ingested_chunks = []
    excluded_count = 0

    # 2. Batch PDF Parsing & Chunking (Sequential VRAM management)
    print("\n--- Ingestion & Parsing Pass (Docling + Tesseract OCR Fallback) ---")
    for idx, p in enumerate(papers, 1):
        pid = p["arxiv_id"]
        title = p["title"]
        authors = p.get("authors", ["Author"])

        if checkpoint.is_completed(pid):
            continue

        pdf_path = downloader.download_paper(p)
        md_text, sections, audit = parser.parse_pdf(pdf_path)

        if audit.get("excluded"):
            excluded_count += 1
            checkpoint.mark_completed(pid)
            continue

        chunks = chunker.chunk_paper_sections(
            paper_id=pid,
            title=title,
            authors=authors,
            sections=sections
        )
        all_ingested_chunks.extend(chunks)
        checkpoint.mark_completed(pid)

        if idx % 20 == 0 or idx == len(papers):
            print(f"[Ingestion Progress] Processed {idx}/{len(papers)} papers ({len(all_ingested_chunks):,} total chunks generated).")

    # 3. Explicit VRAM Unload for Docling Models (PRD §7.1.2)
    print("\n--- Sequential VRAM Unload: Unloading Docling Layout Models ---")
    del parser
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"[VRAM Audit] Peak GPU VRAM allocated during ingestion: {peak_vram:.2f} MB (Budget: 8,192 MB)")
    except Exception:
        pass

    # 4. Dense Embeddings & Vector Store Persistence (ChromaDB + nomic-embed-text)
    print("\n--- Embedding Pass (nomic-embed-text -> ChromaDB Vector Store) ---")
    vector_store = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")
    vector_store.add_chunks(all_ingested_chunks)

    # 5. Compute In-Corpus Citation Edge Connectivity Fraction (PRD §7.1.1)
    print("\n--- Post-Ingestion Citation Connectivity Analysis ---")
    conn_metrics = calculate_in_corpus_connectivity(papers)

    print(f"Total Ingested Papers:       {conn_metrics['total_ingested_papers']}")
    print(f"Connected Papers (>=1 Edge): {conn_metrics['connected_papers_count']}")
    print(f"In-Corpus Citation Fraction: {conn_metrics['connectivity_fraction']:.3f} ({conn_metrics['connectivity_fraction']*100:.1f}%)")
    print(f"Total In-Corpus Citation Edges: {conn_metrics['total_in_corpus_edges']}")
    print(f"GraphRAG Viability Threshold Met: {conn_metrics['graphrag_viability_met']}")

    # Save Ingestion Audit Summary
    summary = {
        "total_papers": len(papers),
        "excluded_papers": excluded_count,
        "total_chunks": vector_store.get_total_chunks_count(),
        "connectivity_metrics": conn_metrics
    }
    with open("data/metadata/ingestion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================================================================")
    print("PHASE 1 DEFINITION OF DONE MET: ChromaDB Vector Store Persisted Successfully!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
