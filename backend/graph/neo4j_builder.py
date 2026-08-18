"""
Citation Graph Engine (Phase 6 / FR-14, FR-15, FR-16, PRD §7.1).
Builds a NetworkX / Neo4j-compatible directed citation graph from:
1. data/metadata/papers_corpus.json (200 genuine paper nodes with full metadata)
2. data/metadata/citation_edges.csv (171 clean, quote-wrapped citation network edges)
"""

import os
import json
import csv
import networkx as nx
from typing import Dict, List, Any, Optional


class CitationGraphEngine:
    """Citation graph builder and NetworkX / Neo4j graph engine (FR-14)."""

    def __init__(
        self,
        corpus_path: str = "data/metadata/papers_corpus.json",
        edges_path: str = "data/metadata/citation_edges.csv"
    ):
        self.corpus_path = corpus_path
        self.edges_path = edges_path
        self.graph = nx.DiGraph()
        self.paper_lookup: Dict[str, Dict[str, Any]] = {}
        self.build_graph()

    def build_graph(self):
        """Build citation graph from corpus metadata and citation edges CSV."""
        # 1. Load Paper Nodes from papers_corpus.json
        if os.path.exists(self.corpus_path):
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            papers = raw_data.get("papers", raw_data) if isinstance(raw_data, dict) else raw_data

            for p in papers:
                aid = p.get("arxiv_id", "")
                if not aid:
                    continue

                title = p.get("title", "Paper")
                cat = p.get("category", "cs.CL")
                cits = p.get("citation_count", 50)
                year = p.get("year", 2026)
                authors = p.get("authors", [])
                if isinstance(authors, list):
                    authors_str = ", ".join(authors[:3])
                else:
                    authors_str = str(authors)

                self.paper_lookup[aid] = p
                self.graph.add_node(
                    aid,
                    label=title,
                    title=title,
                    category=cat,
                    citation_count=cits,
                    year=year,
                    authors=authors_str,
                    arxiv_id=aid
                )

        # 2. Load Citation Edges from citation_edges.csv
        edges_loaded = 0
        if os.path.exists(self.edges_path):
            with open(self.edges_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = row.get("source_paper_id", "").strip()
                    tgt = row.get("target_paper_id", "").strip()
                    if src and tgt:
                        is_cross = row.get("is_cross_category", "False").lower() == "true"
                        weight = float(row.get("weight", 1.0))
                        
                        # Ensure nodes exist even if outside sampled list
                        if src not in self.graph:
                            self.graph.add_node(src, label=row.get("source_paper_title", src), category=row.get("source_category", "cs.CL"))
                        if tgt not in self.graph:
                            self.graph.add_node(tgt, label=row.get("target_paper_title", tgt), category=row.get("target_category", "cs.CL"))

                        self.graph.add_edge(
                            src,
                            tgt,
                            relationship="CITES",
                            is_cross_category=is_cross,
                            weight=weight
                        )
                        edges_loaded += 1

        print(f"[GraphEngine] Initialized citation graph: {self.graph.number_of_nodes()} paper nodes, {self.graph.number_of_edges()} citation edges (FR-14).")

    def get_subgraph_data(self, seed_arxiv_id: str, depth: int = 1) -> Dict[str, Any]:
        """Fetch 1-hop subgraph around seed paper for PyVis / Streamlit visualization (FR-16)."""
        if not self.graph.nodes:
            return {"nodes": [], "edges": []}

        if seed_arxiv_id not in self.graph:
            seed_arxiv_id = "1706.03762" if "1706.03762" in self.graph else list(self.graph.nodes())[0]

        sub_nodes = {seed_arxiv_id}
        for successor in self.graph.successors(seed_arxiv_id):
            sub_nodes.add(successor)
        for predecessor in self.graph.predecessors(seed_arxiv_id):
            sub_nodes.add(predecessor)

        nodes_list = []
        for nid in sub_nodes:
            attr = self.graph.nodes[nid]
            title = attr.get("title", attr.get("label", nid))
            cat = attr.get("category", "cs.CL")
            cits = attr.get("citation_count", 50)
            
            # Category-based color scheme
            color = "#3B82F6" if "CV" in cat else "#10B981"  # Blue for CV, Green for CL
            if nid == seed_arxiv_id:
                color = "#F59E0B"  # Amber for seed node

            nodes_list.append({
                "id": nid,
                "label": f"[{nid}] {title[:28]}...",
                "title": f"Title: {title}\nArXiv ID: {nid}\nCategory: {cat}\nCitations: {cits}",
                "category": cat,
                "color": color,
                "value": min(35, max(12, int(cits) // 400 + 10))
            })

        edges_list = []
        sub_g = self.graph.subgraph(sub_nodes)
        for u, v, data in sub_g.edges(data=True):
            is_cross = data.get("is_cross_category", False)
            edges_list.append({
                "from": u,
                "to": v,
                "label": "CITES",
                "arrows": "to",
                "color": "#EF4444" if is_cross else "#9CA3AF",
                "dashes": is_cross
            })

        return {"nodes": nodes_list, "edges": edges_list, "seed": seed_arxiv_id}

    def get_citation_stats(self) -> Dict[str, Any]:
        """Compute citation network topology metrics."""
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        cross_edges = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("is_cross_category", False))
        
        in_degrees = dict(self.graph.in_degree())
        top_cited = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        top_hubs = [
            {
                "paper_id": pid,
                "title": self.graph.nodes[pid].get("title", pid),
                "in_citations": in_deg
            }
            for pid, in_deg in top_cited
        ]

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "within_category_edges": total_edges - cross_edges,
            "cross_category_edges": cross_edges,
            "cross_category_pct": round((cross_edges / total_edges * 100) if total_edges > 0 else 0, 2),
            "top_citation_hubs": top_hubs
        }
