"""
Resumable Two-Stage Pipeline for Corpus Parsing and Vector Store Ingestion (Phase 1 / FR-3).

Stage 1 (Parse): Extracts and chunks PDFs into per-paper JSON files in data/parsed/<arxiv_id>.json.
                  Resumable: checks if data/parsed/<arxiv_id>.json exists before parsing.
Stage 2 (Ingest): Reads all parsed JSONs in data/parsed/, generates GPU embeddings, and persists into ChromaDB.
"""

import sys
import os
import glob
import json
import argparse
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ingestion.docling_parser import ProductionDoclingParser
from backend.ingestion.section_chunker import SectionChunker
from backend.ingestion.vector_store import VectorStore
from scripts.download_real_corpus import FOUNDATIONAL_PAPERS


PARSED_DIR = "data/parsed"
METADATA_PATH = "data/metadata/papers_corpus.json"
PDFS_DIR = "data/pdfs"


def clean_invalid_pdfs() -> Tuple[int, int]:
    """Clean corrupt / non-PDF files from the PDF directory."""
    pdf_files = glob.glob(os.path.join(PDFS_DIR, "*.pdf"))
    removed_count = 0
    valid_count = 0

    for path in pdf_files:
        try:
            with open(path, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    f.close()
                    os.remove(path)
                    removed_count += 1
                else:
                    valid_count += 1
        except Exception:
            if os.path.exists(path):
                os.remove(path)
                removed_count += 1

    return removed_count, valid_count


def load_corpus_metadata() -> Dict[str, Dict[str, Any]]:
    """Load canonical paper metadata mapping arxiv_id -> paper info."""
    meta_map = {}
    
    # 1. Base foundational papers
    for p in FOUNDATIONAL_PAPERS:
        meta_map[p["arxiv_id"]] = p

    # 2. Existing papers_corpus.json metadata
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                for p in cdata.get("papers", []):
                    aid = p.get("arxiv_id")
                    title = p.get("title", "")
                    if aid and title and not title.startswith("Scientific Paper"):
                        meta_map[aid] = p
        except Exception:
            pass

    return meta_map


def run_parse_stage(force_reparse: bool = False) -> List[Dict[str, Any]]:
    """
    Stage 1: Resumable PDF Parsing & Section Chunking.
    Saves each paper to data/parsed/<arxiv_id>.json.
    """
    print("==========================================================================")
    print("STAGE 1: RESUMABLE PDF EXTRACTION & SECTION CHUNKING")
    print("==========================================================================")

    os.makedirs(PARSED_DIR, exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)

    removed, valid = clean_invalid_pdfs()
    print(f"[Cleanup] Cleaned {removed} invalid files. Found {valid} genuine PDFs.")

    meta_map = load_corpus_metadata()
    parser = ProductionDoclingParser()
    chunker = SectionChunker(target_chunk_tokens=300)

    pdf_files = sorted(glob.glob(os.path.join(PDFS_DIR, "*.pdf")))
    processed_papers = []
    total_chunks_count = 0
    cached_count = 0
    parsed_count = 0

    for pdf_path in pdf_files:
        basename = os.path.basename(pdf_path)
        aid = basename.replace(".pdf", "")
        parsed_file = os.path.join(PARSED_DIR, f"{aid}.json")

        meta = meta_map.get(aid, {})
        title = meta.get("title", f"Paper {aid}")
        cat = meta.get("category", "cs.CL")

        # Check if already parsed
        if not force_reparse and os.path.exists(parsed_file):
            try:
                with open(parsed_file, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                    chunks = pdata.get("chunks", [])
                    if chunks and len(chunks) > 0:
                        # Ensure title is canonical
                        if not pdata.get("title") or pdata["title"].startswith("Scientific Paper"):
                            pdata["title"] = title
                            with open(parsed_file, "w", encoding="utf-8") as fw:
                                json.dump(pdata, fw, indent=2)
                        
                        processed_papers.append({
                            "arxiv_id": aid,
                            "title": pdata.get("title", title),
                            "category": pdata.get("category", cat),
                            "abstract": pdata.get("abstract", "")
                        })
                        total_chunks_count += len(chunks)
                        cached_count += 1
                        print(f"[Cached JSON] {aid:12s} | Title: '{pdata.get('title', title)[:45]:45s}' | Chunks: {len(chunks)}")
                        continue
            except Exception:
                pass  # Fall through to re-parse

        # Parse PDF
        full_text, sections, audit = parser.parse_pdf(pdf_path)
        if not full_text or audit.get("excluded"):
            print(f"[Skip] {aid}: {audit.get('exclusion_reason')}")
            continue

        abstract = sections.get("Abstract", full_text[:400])
        chunks = chunker.chunk_paper_sections(
            paper_id=aid,
            title=title,
            authors=["Researcher"],
            sections=sections
        )

        paper_record = {
            "arxiv_id": aid,
            "title": title,
            "category": cat,
            "abstract": abstract,
            "sections": sections,
            "chunks": chunks
        }

        with open(parsed_file, "w", encoding="utf-8") as f:
            json.dump(paper_record, f, indent=2)

        processed_papers.append({
            "arxiv_id": aid,
            "title": title,
            "category": cat,
            "abstract": abstract
        })
        total_chunks_count += len(chunks)
        parsed_count += 1
        print(f"[Parsed PDF]  {aid:12s} | Title: '{title[:45]:45s}' | Chunks: {len(chunks)}")

    # Update canonical corpus metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump({"papers": processed_papers}, f, indent=2)

    print(f"\nStage 1 Complete: {cached_count} cached + {parsed_count} newly parsed = {len(processed_papers)} papers ({total_chunks_count} total chunks).")
    return processed_papers


def run_ingest_stage():
    """
    Stage 2: ChromaDB Dense Vector Store Ingestion.
    Reads all parsed JSON files from data/parsed/ and generates GPU embeddings.
    """
    print("==========================================================================")
    print("STAGE 2: CHROMADB VECTOR STORE INGESTION (GPU BATCH_SIZE=16)")
    print("==========================================================================")

    parsed_files = sorted(glob.glob(os.path.join(PARSED_DIR, "*.json")))
    if not parsed_files:
        print("[Error] No parsed JSON files found in data/parsed/. Run Stage 1 (parse) first!")
        return

    all_chunks = []
    for pfile in parsed_files:
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                pdata = json.load(f)
                all_chunks.extend(pdata.get("chunks", []))
        except Exception as e:
            print(f"[Warning] Failed to read {pfile}: {e}")

    print(f"Loaded {len(all_chunks)} total chunks from {len(parsed_files)} parsed papers.")

    # Initialize ChromaDB vector store
    vector_store = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")

    print(f"\nUpserting {len(all_chunks)} genuine chunks into ChromaDB...")
    vector_store.add_chunks(all_chunks)

    final_count = vector_store.get_total_chunks_count()
    print("==========================================================================")
    print(f"STAGE 2 SUCCESS! ChromaDB collection '{vector_store.collection_name}' contains {final_count} chunks.")
    print("==========================================================================")


def main():
    parser = argparse.ArgumentParser(description="Corpus Parsing & ChromaDB Vector Ingestion Pipeline")
    parser.add_argument("--stage", choices=["parse", "ingest", "all"], default="all", help="Pipeline stage to execute")
    parser.add_argument("--force-reparse", action="store_true", help="Force re-parsing of PDFs even if cached JSON exists")
    args = parser.parse_args()

    if args.stage in ["parse", "all"]:
        run_parse_stage(force_reparse=args.force_reparse)

    if args.stage in ["ingest", "all"]:
        run_ingest_stage()


if __name__ == "__main__":
    main()
