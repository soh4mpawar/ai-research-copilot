"""
Real Semantic Scholar & Docling PDF Bibliography Citation Graph Rebuilder (FR-14).
Fetches ground-truth academic citation networks for all 200 ingested corpus papers.

Features:
1. Semantic Scholar Graph API query with safe rate-limiting (~1.2s pace) and exponential backoff on 429s
2. Docling PDF parsed bibliography fallback (matching references against corpus via arXiv IDs and title matching)
3. Exact corpus-scoped edge filtering (both source and target must belong to the 200-paper corpus)
4. Saves verified edges to data/metadata/citation_edges.csv with quote-wrapping
"""

import sys
import os
import json
import csv
import re
import time
import urllib.request
import urllib.error
from typing import Dict, List, Set, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def normalize_title(title: str) -> str:
    """Normalize paper title for robust string matching."""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def build_corpus_lookup(corpus_file: str = "data/metadata/papers_corpus.json") -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Load corpus papers and build lookup maps."""
    with open(corpus_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    papers = data.get("papers", data) if isinstance(data, dict) else data

    id_to_paper = {}
    norm_title_to_id = {}

    for p in papers:
        aid = p.get("arxiv_id", "").strip()
        title = p.get("title", "").strip()
        if aid:
            id_to_paper[aid] = p
            if title:
                norm_title_to_id[normalize_title(title)] = aid

    return papers, id_to_paper, norm_title_to_id


def fetch_semantic_scholar_data(arxiv_id: str) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch paper references and citations from Semantic Scholar API with exponential backoff.
    Returns (success, references_list, citations_list).
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}?fields=title,references.externalIds,references.title,citations.externalIds,citations.title"
    headers = {"User-Agent": "AcademicResearchBot/1.0 (mailto:academic-rag@deepmind.com)"}

    max_retries = 5
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as response:
                data = json.loads(response.read().decode("utf-8"))
                refs = data.get("references", []) or []
                cits = data.get("citations", []) or []
                return True, refs, cits
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"    [S2 429 RateLimit] ArXiv {arxiv_id} (Attempt {attempt+1}/{max_retries}) -> Backing off {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2.0, 30.0)
            elif e.code == 404:
                # Paper not indexed in S2
                return False, [], []
            else:
                print(f"    [S2 HTTP {e.code}] ArXiv {arxiv_id}: {e}")
                return False, [], []
        except Exception as e:
            print(f"    [S2 Err] ArXiv {arxiv_id}: {e}")
            time.sleep(1.0)

    return False, [], []


def extract_pdf_references_fallback(arxiv_id: str, norm_title_to_id: Dict[str, str], id_to_paper: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Fallback: Parse references section from local parsed JSON file.
    Matches referenced papers against corpus via arXiv ID regex or title matching.
    """
    parsed_path = f"data/parsed/{arxiv_id}.json"
    if not os.path.exists(parsed_path):
        return []

    try:
        with open(parsed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ref_text = ""
        sections = data.get("sections", {})
        for sec_name, sec_content in sections.items():
            if "reference" in sec_name.lower() or "bibliography" in sec_name.lower():
                ref_text += " " + sec_content

        if not ref_text:
            # Check entire text if no explicit references section
            ref_text = " ".join(sections.values())

        matched_target_ids = set()

        # 1. Match explicit arXiv ID patterns: arXiv:YYMM.NNNNN or arxiv.org/abs/YYMM.NNNNN
        arxiv_matches = re.findall(r"(?:arxiv\.org/abs/|arXiv:)(\d{4}\.\d{4,5})", ref_text, re.IGNORECASE)
        for target_aid in arxiv_matches:
            if target_aid in id_to_paper and target_aid != arxiv_id:
                matched_target_ids.add(target_aid)

        # 2. Match canonical foundational and corpus paper titles
        ref_text_lower = ref_text.lower()
        for target_aid, target_paper in id_to_paper.items():
            if target_aid == arxiv_id:
                continue
            title = target_paper.get("title", "").strip().lower()
            if len(title) > 15:
                # Substring check for paper title in bibliography text
                if title in ref_text_lower:
                    matched_target_ids.add(target_aid)

        return list(matched_target_ids)
    except Exception as e:
        print(f"    [PDF Fallback Err] {arxiv_id}: {e}")
        return []


def run_rebuild():
    print("=" * 80)
    print("SEMANTIC SCHOLAR & DOCLING PDF CITATION GRAPH REBUILD (200 PAPERS)")
    print("=" * 80)

    papers, id_to_paper, norm_title_to_id = build_corpus_lookup()
    print(f"Loaded corpus: {len(papers)} papers indexed.\n")

    edges = []
    seen_edge_pairs = set()

    s2_success_count = 0
    pdf_fallback_count = 0

    t_start = time.time()

    for idx, p in enumerate(papers, 1):
        aid = p.get("arxiv_id", "")
        src_title = p.get("title", "Paper")
        src_cat = p.get("category", "cs.CL")

        if not aid:
            continue

        print(f"[{idx:03d}/200] Processing arXiv:{aid} ('{src_title[:38]}...')...", end="", flush=True)

        s2_ok, refs, cits = fetch_semantic_scholar_data(aid)
        
        found_targets_for_paper = set()

        if s2_ok:
            s2_success_count += 1
            print(f" [S2 OK: {len(refs)} refs, {len(cits)} cits]", flush=True)
            
            # Process S2 References (aid CITES target)
            for r in refs:
                ext_ids = r.get("externalIds") or {}
                ref_aid = ext_ids.get("ArXiv", "")
                ref_title = r.get("title", "")

                target_id = None
                if ref_aid and ref_aid in id_to_paper:
                    target_id = ref_aid
                elif ref_title:
                    norm_t = normalize_title(ref_title)
                    if norm_t in norm_title_to_id:
                        target_id = norm_title_to_id[norm_t]

                if target_id and target_id != aid:
                    found_targets_for_paper.add((aid, target_id))

            # Process S2 Citations (target CITES aid)
            for c in cits:
                ext_ids = c.get("externalIds") or {}
                cit_aid = ext_ids.get("ArXiv", "")
                cit_title = c.get("title", "")

                source_id = None
                if cit_aid and cit_aid in id_to_paper:
                    source_id = cit_aid
                elif cit_title:
                    norm_t = normalize_title(cit_title)
                    if norm_t in norm_title_to_id:
                        source_id = norm_title_to_id[norm_t]

                if source_id and source_id != aid:
                    found_targets_for_paper.add((source_id, aid))

        else:
            pdf_fallback_count += 1
            matched_targets = extract_pdf_references_fallback(aid, norm_title_to_id, id_to_paper)
            print(f" [S2 404/Fail -> PDF Fallback: {len(matched_targets)} matched corpus refs]", flush=True)
            for target_aid in matched_targets:
                found_targets_for_paper.add((aid, target_aid))

        # Add unique edges
        for u, v in found_targets_for_paper:
            if (u, v) not in seen_edge_pairs:
                seen_edge_pairs.add((u, v))
                u_paper = id_to_paper.get(u, {})
                v_paper = id_to_paper.get(v, {})
                u_cat = u_paper.get("category", "cs.CL")
                v_cat = v_paper.get("category", "cs.CL")
                is_cross = (u_cat != v_cat)

                edges.append({
                    "source_paper_id": u,
                    "source_paper_title": u_paper.get("title", u),
                    "source_category": u_cat,
                    "target_paper_id": v,
                    "target_paper_title": v_paper.get("title", v),
                    "target_category": v_cat,
                    "is_cross_category": is_cross,
                    "weight": 1.0 if not is_cross else 0.9
                })

        # Respect Semantic Scholar rate limits (~1.2s delay between calls)
        time.sleep(1.2)

    total_time = time.time() - t_start

    # Export to CSV with QUOTE_ALL
    out_csv = "data/metadata/citation_edges.csv"
    fieldnames = [
        "source_paper_id", "source_paper_title", "source_category",
        "target_paper_id", "target_paper_title", "target_category",
        "is_cross_category", "weight"
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in edges:
            writer.writerow(row)

    print("\n" + "=" * 80)
    print("REBUILD COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  • Total Time Elapsed: {total_time/60:.2f} minutes")
    print(f"  • Total Papers Processed: {len(papers)}")
    print(f"  • Semantic Scholar API Success: {s2_success_count} papers")
    print(f"  • Docling PDF Fallback Used: {pdf_fallback_count} papers")
    print(f"  • Real Verified Citation Edges: {len(edges)}")
    
    within_cnt = sum(1 for e in edges if not e["is_cross_category"])
    cross_cnt = sum(1 for e in edges if e["is_cross_category"])
    print(f"  • Within-Category Edges: {within_cnt} ({within_cnt/len(edges)*100:.1f}%)" if edges else "  • Within-Category: 0")
    print(f"  • Cross-Category Edges: {cross_cnt} ({cross_cnt/len(edges)*100:.1f}%)" if edges else "  • Cross-Category: 0")

    print("\nSample of 5 Verified Edges:")
    for i, e in enumerate(edges[:5], 1):
        print(f"  {i}. [{e['source_paper_id']}] '{e['source_paper_title'][:35]}...' -> [{e['target_paper_id']}] '{e['target_paper_title'][:35]}...' (Cross-Cat: {e['is_cross_category']})")
    print("=" * 80)


if __name__ == "__main__":
    run_rebuild()
