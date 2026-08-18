"""
File Verification & Line Count Sanity Check Script (Checkpoint Item 6).
Inspects file sizes, row counts, and timestamps for all database & metadata artifacts.
"""

import os
import json
import time

def verify_system_artifacts():
    print("==========================================================================")
    print("SYSTEM ARTIFACTS & FILE SIZE SANITY CHECK (CHECKPOINT ITEM 6)")
    print("==========================================================================")

    paths = [
        ("Vector Store Database", "data/chroma_db/chroma.sqlite3"),
        ("QA Test Dataset", "data/metadata/draft_qa_dataset.json"),
        ("RAGAS Evaluation Results", "data/metadata/ragas_eval_results.json"),
        ("Ablation Study Results", "data/metadata/ablation_eval_results.json"),
        ("Baseline Evaluation Results", "data/metadata/baseline_eval_results.json"),
        ("Citation Edges CSV", "data/metadata/citation_edges.csv"),
        ("Sampled Papers Metadata", "data/metadata/sampled_papers.json"),
        ("Paper Corpus Metadata", "data/metadata/papers_corpus.json")
    ]

    for label, path in paths:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024.0
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
            
            detail = ""
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        d = json.load(f)
                        if isinstance(d, list):
                            detail = f"| Rows: {len(d)}"
                        elif isinstance(d, dict):
                            detail = f"| Keys: {len(d)} (samples: {d.get('total_eval_samples', 'N/A')})"
                    except Exception:
                        pass
            elif path.endswith(".csv"):
                with open(path, "r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                    detail = f"| CSV Lines: {lines}"

            print(f"[FOUND] {label:26s} | Path: {path:38s} | Size: {size_kb:8.1f} KB | ModTime: {mtime} {detail}")
        else:
            print(f"[MISSING] {label:24s} | Path: {path}")

    # Check PDF directory
    pdf_dir = "data/pdfs"
    if os.path.exists(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        print(f"\n[FOUND] Ingested PDFs Directory   | Path: {pdf_dir:38s} | Total PDFs: {len(pdf_files)} files")

    print("==========================================================================")

if __name__ == "__main__":
    verify_system_artifacts()
