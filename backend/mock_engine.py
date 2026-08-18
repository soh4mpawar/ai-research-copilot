"""
Mock Engine implementation of Backend Contract (A's component interface).
Provides realistic research data, dense/sparse/RRF/reranker retrieval inspection,
literature review synthesis, citation graph network, and RAGAS evaluation metrics.
"""

import time
import random
from typing import List, Dict, Any
from backend.contract import (
    QueryResult,
    SourcePaper,
    RetrievedChunk,
    PipelineMetrics,
    LitReviewResult,
    CitationGraphData,
    EvalMetrics,
)

# ---------------------------------------------------------------------------
# Pre-defined Mock Papers Corpus (Representative subset of 180+ paper corpus)
# ---------------------------------------------------------------------------

SAMPLE_PAPERS = [
    SourcePaper(
        paper_id="paper_001",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"],
        year=2017,
        arxiv_id="1706.03762",
        category="cs.CL",
        venue="NeurIPS 2017",
        citation_count=112000,
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture relying entirely on an attention mechanism to draw global dependencies between input and output."
    ),
    SourcePaper(
        paper_id="paper_002",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal", "Heinrich Küttler", "Mike Lewis", "Wen-tau Yih", "Tim Rocktäschel", "Sebastian Riedel", "Douwe Kiela"],
        year=2020,
        arxiv_id="2005.11401",
        category="cs.CL",
        venue="NeurIPS 2020",
        citation_count=4800,
        pdf_url="https://arxiv.org/pdf/2005.11401.pdf",
        abstract="Large language models (LLMs) can store implicit knowledge in their parameters. However, their ability to access and precisely manipulate knowledge is limited. We build RAG models where parametric memory is combined with non-parametric (retrieved) memory."
    ),
    SourcePaper(
        paper_id="paper_003",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        year=2019,
        arxiv_id="1810.04805",
        category="cs.CL",
        venue="NAACL 2019",
        citation_count=98000,
        pdf_url="https://arxiv.org/pdf/1810.04805.pdf",
        abstract="We introduce BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text."
    ),
    SourcePaper(
        paper_id="paper_004",
        title="Dense Passage Retrieval for Open-Domain Question Answering",
        authors=["Vladimir Karpukhin", "Barlas Oğuz", "Sewell Sewell", "Patrick Lewis", "Ledell Wu", "Sergey Edunov", "Danqi Chen", "Wen-tau Yih"],
        year=2020,
        arxiv_id="2004.04906",
        category="cs.CL",
        venue="EMNLP 2020",
        citation_count=3200,
        pdf_url="https://arxiv.org/pdf/2004.04906.pdf",
        abstract="Open-domain question answering relies on efficient passage retrieval. We show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of passages using a dual-encoder framework."
    ),
    SourcePaper(
        paper_id="paper_005",
        title="Reciprocal Rank Fusion Outperforms Vector Similarity Search in Hybrid Retrieval",
        authors=["Gordon V. Cormack", "Charles L. A. Clarke", "Stefan Büttcher"],
        year=2009,
        arxiv_id="0905.1234",
        category="cs.IR",
        venue="SIGIR 2009",
        citation_count=1450,
        pdf_url="https://arxiv.org/pdf/0905.1234.pdf",
        abstract="Reciprocal Rank Fusion (RRF) is a simple method for combining multiple rank lists to produce a single unified ranking. We demonstrate that RRF consistently outperforms standard vector search when combining dense and sparse retrieval signals."
    ),
    SourcePaper(
        paper_id="paper_006",
        title="BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings",
        authors=["Jianlyu Chen", "Shitao Xiao", "Peitian Zhang", "Zheng Liu", "Defu Lian"],
        year=2024,
        arxiv_id="2402.03216",
        category="cs.CL",
        venue="arXiv preprint",
        citation_count=650,
        pdf_url="https://arxiv.org/pdf/2402.03216.pdf",
        abstract="We introduce BGE M3-Embedding, which supports multi-linguality, multi-functionality (dense retrieval, multi-vector, and sparse retrieval), and multi-granularity (from short sentences to long documents up to 8192 tokens)."
    ),
    SourcePaper(
        paper_id="paper_007",
        title="GraphRAG: Unlocking LLM Discovery on Narrative Networks",
        authors=["Darren Edge", "Ha Trinh", "Xing Niu", "Robert L. T. Cheng", "Jonathan Bradley", "Alex Chao", "Aron M. Inman", "Cheng-Yu Lee"],
        year=2024,
        arxiv_id="2404.16130",
        category="cs.AI",
        venue="Microsoft Research 2024",
        citation_count=890,
        pdf_url="https://arxiv.org/pdf/2404.16130.pdf",
        abstract="The use of Retrieval-Augmented Generation (RAG) to answer queries over private or domain-specific document collections is widespread. We present GraphRAG, combining knowledge graph extraction and graph summarization for holistic dataset synthesis."
    ),
    SourcePaper(
        paper_id="paper_008",
        title="RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        authors=["Shahul Es", "Jithin James", "Luis Espinosa-Anke", "Steven Schockaert"],
        year=2023,
        arxiv_id="2309.15217",
        category="cs.CL",
        venue="EACL 2024",
        citation_count=1200,
        pdf_url="https://arxiv.org/pdf/2309.15217.pdf",
        abstract="Evaluating Retrieval-Augmented Generation (RAG) pipelines is challenging due to the lack of reference answers. We present RAGAS, a framework for reference-free evaluation of RAG systems measuring Faithfulness, Answer Relevance, Context Precision, and Recall."
    ),
    SourcePaper(
        paper_id="paper_009",
        title="Docling: Efficient Document Processing and Layout Extraction for LLM Ingestion",
        authors=["Peter W. J. Staar", "Maksym Lysak", "Michele Dolfi", "Christoph Auer"],
        year=2024,
        arxiv_id="2408.09871",
        category="cs.CV",
        venue="IBM Research 2024",
        citation_count=430,
        pdf_url="https://arxiv.org/pdf/2408.09871.pdf",
        abstract="Docling parses complex PDF documents including multi-column text, mathematical equations, figures, and tables into cleanly structured Markdown format optimized for downstream RAG chunking."
    ),
    SourcePaper(
        paper_id="paper_010",
        title="ChromaDB: Open-Source AI Native Vector Database Architecture",
        authors=["Jeff Huber", "Anton Troynikov"],
        year=2023,
        arxiv_id="2301.00001",
        category="cs.DB",
        venue="Open Source",
        citation_count=2100,
        pdf_url="https://chromadb.com",
        abstract="ChromaDB provides fast HNSW indexed vector similarity search, metadata filtering, and persistent document storage tailored for LLM context retrieval systems."
    ),
]

# Add additional papers dynamically to reach 150+ corpus simulation
for i in range(11, 185):
    cat = "cs.CL" if i % 2 == 0 else "cs.CV"
    SAMPLE_PAPERS.append(
        SourcePaper(
            paper_id=f"paper_{i:03d}",
            title=f"Advanced Deep Learning & RAG Frontiers Vol. {i}: Methodological Innovations in Neural Retrieval",
            authors=[f"Researcher {chr(65 + (i % 26))}. Smith", f"Co-Author {chr(90 - (i % 26))}. Davis"],
            year=2020 + (i % 5),
            arxiv_id=f"{2000 + (i % 5)}.{10000 + i}",
            category=cat,
            venue="arXiv preprint",
            citation_count=15 + (i * 7) % 400,
            pdf_url="https://arxiv.org",
            abstract=f"This paper explores section-aware chunking, hybrid BM25 + dense embedding fusion, and cross-encoder reranking over scientific literature corpora in {cat} domains."
        )
    )


class MockEngine:
    """Mock implementation of the backend contract for rapid S frontend development."""

    @staticmethod
    def get_corpus_papers() -> List[SourcePaper]:
        return SAMPLE_PAPERS

    @staticmethod
    def query(query_text: str, mode: str = "qa") -> QueryResult:
        """Simulate RAG processing pipeline execution."""
        # Simulated latency timings
        retrieval_t = round(random.uniform(0.8, 1.4), 2)
        rerank_t = round(random.uniform(0.4, 0.9), 2)
        gen_t = round(random.uniform(3.2, 5.5), 2)
        total_t = round(retrieval_t + rerank_t + gen_t, 2)

        # Select relevant sample papers
        p0 = SAMPLE_PAPERS[0]
        p1 = SAMPLE_PAPERS[1]
        p2 = SAMPLE_PAPERS[4]
        p3 = SAMPLE_PAPERS[3]
        sources = [p0, p1, p2, p3]

        # Generate realistic chunks
        chunks = [
            RetrievedChunk(
                chunk_id="chk_001_m07",
                paper_id=p1.paper_id,
                paper_title=p1.title,
                authors=", ".join(p1.authors[:3]) + " et al.",
                section="Methodology & Architecture",
                text="Retrieval-Augmented Generation (RAG) models combine parametric memory (a pre-trained seq2seq transformer) with non-parametric memory (a dense vector index of Wikipedia passages accessed via a neural retriever DPR). During generation, the retriever returns top-K passages that condition the autoregressive generator [1].",
                score=0.94,
                dense_rank=1,
                bm25_rank=3,
                rrf_rank=1,
                rerank_score=0.96,
                page=4
            ),
            RetrievedChunk(
                chunk_id="chk_004_r02",
                paper_id=p3.paper_id,
                paper_title=p3.title,
                authors=", ".join(p3.authors[:3]) + " et al.",
                section="Dense Passage Retrieval",
                text="DPR uses a dual-encoder framework where two independent BERT encoders embed passages and questions into a shared 768-dimensional continuous vector space. Similarity is calculated using dot product, outperforming traditional sparse BM25 retrieval on top-k retrieval accuracy by 9-12% [2].",
                score=0.89,
                dense_rank=2,
                bm25_rank=12,
                rrf_rank=2,
                rerank_score=0.91,
                page=3
            ),
            RetrievedChunk(
                chunk_id="chk_005_f01",
                paper_id=p2.paper_id,
                paper_title=p2.title,
                authors=", ".join(p2.authors[:3]) + " et al.",
                section="Reciprocal Rank Fusion",
                text="RRF scores each document d by calculating RRF_score(d) = sum_{m in M} 1 / (k + r_m(d)), where k=60 is a constant and r_m(d) is the rank of document d in rank list m. Combining BM25 keyword matching with dense embedding semantic search prevents failure on technical terminology [3].",
                score=0.85,
                dense_rank=6,
                bm25_rank=1,
                rrf_rank=3,
                rerank_score=0.88,
                page=2
            ),
            RetrievedChunk(
                chunk_id="chk_001_i02",
                paper_id=p0.paper_id,
                paper_title=p0.title,
                authors=", ".join(p0.authors[:3]) + " et al.",
                section="Introduction & Attention",
                text="The Transformer architecture eliminates recurrence, relying entirely on scaled dot-product self-attention mechanisms QK^T / sqrt(d_k) to capture global context across tokens in parallel [4].",
                score=0.78,
                dense_rank=4,
                bm25_rank=5,
                rrf_rank=4,
                rerank_score=0.82,
                page=2
            )
        ]

        if "rag" in query_text.lower() or "retrieval" in query_text.lower():
            answer = (
                "### Summary of Grounded Research Findings\n\n"
                "**Retrieval-Augmented Generation (RAG)** was introduced by Lewis et al. (2020) to address key limitations in parametric large language models, specifically knowledge freshness, domain specificity, and factual hallucinations **[1]**. \n\n"
                "#### Core Architecture & Mechanics:\n"
                "1. **Dense Retrieval Engine**: DPR (Karpukhin et al., 2020) employs dual BERT encoders to map queries and document passages into a 768-dimensional dense vector space, optimizing dot-product similarity search **[2]**.\n"
                "2. **Hybrid Fusion & Sparse BM25**: To overcome dense retrieval failures on exact technical terms or acronyms, Reciprocal Rank Fusion (RRF) merges BM25 keyword ranks with ChromaDB vector distance ranks **[3]**.\n"
                "3. **Cross-Encoder Reranking**: Candidate passages returned by RRF are passed through a `bge-reranker-base` cross-encoder to produce fine-grained attention relevance scores before injecting context into the Gemini generator.\n"
                "4. **Attention-Conditioned Generation**: The generator conditions its output strictly on retrieved text passages, guaranteeing verifiable academic citations **[4]**.\n\n"
                "> **Verification Note**: All statement assertions above are derived directly from the indexed paper corpus chunks shown in the Evidence Drawer below."
            )
            strength = "Strong"
        else:
            answer = (
                f"### Analysis for Query: '{query_text}'\n\n"
                f"Based on the indexed scientific corpus across NLP and Computer Vision domains, the literature demonstrates that hybrid retrieval pipelines combining **ChromaDB dense search** with **BM25 keyword search** significantly increase context precision **[1]**.\n\n"
                "#### Key Insights:\n"
                "* **Section-aware Chunking**: Parsing PDF papers into 250–350 token structural sections preserves context boundaries better than arbitrary fixed token splitting **[2]**.\n"
                "* **Reranking Utility**: Cross-encoder reranking (`bge-reranker-base`) filters out noisy candidate chunks, raising top-5 context precision by **~18%** **[3]**.\n"
                "* **Transformer Backbone**: Scaled dot-product self-attention allows parallel context computation across long academic documents **[4]**.\n"
            )
            strength = "Strong" if len(query_text) > 10 else "Moderate"

        metrics = PipelineMetrics(
            retrieval_time_sec=retrieval_t,
            reranking_time_sec=rerank_t,
            generation_time_sec=gen_t,
            total_time_sec=total_t,
            dense_candidates_count=20,
            bm25_candidates_count=20,
            rrf_candidates_count=25,
            reranked_candidates_count=10,
            final_context_chunks_count=4
        )

        return QueryResult(
            query=query_text,
            mode=mode,
            answer=answer,
            evidence_strength=strength,
            sources=sources,
            retrieved_chunks=chunks,
            metrics=metrics
        )

    @staticmethod
    def generate_lit_review(topic: str) -> LitReviewResult:
        """Simulate structured literature review generation."""
        sources = SAMPLE_PAPERS[:6]
        
        table = [
            {
                "Paper Title": "Attention Is All You Need",
                "Year": 2017,
                "Core Approach": "Transformer Self-Attention",
                "Key Contribution": "Replaced RNNs/CNNs with parallel scaled dot-product attention.",
                "Limitations": "Quadratic memory complexity O(N^2) relative to sequence length."
            },
            {
                "Paper Title": "BERT",
                "Year": 2019,
                "Core Approach": "Bidirectional Masked LM",
                "Key Contribution": "Deep bidirectional pre-training for contextual language representations.",
                "Limitations": "High computational fine-tuning cost; context length capped at 512 tokens."
            },
            {
                "Paper Title": "Retrieval-Augmented Gen (RAG)",
                "Year": 2020,
                "Core Approach": "Hybrid Parametric + Non-Parametric",
                "Key Contribution": "Combines pre-trained seq2seq generator with DPR neural retriever.",
                "Limitations": "Retriever latency bottleneck and vector store sync overhead."
            },
            {
                "Paper Title": "Dense Passage Retrieval (DPR)",
                "Year": 2020,
                "Core Approach": "Dual BERT Encoders",
                "Key Contribution": "Replaced BM25 keyword matching with dense dot-product embeddings.",
                "Limitations": "Fails on rare out-of-vocabulary technical jargon and exact numbers."
            },
            {
                "Paper Title": "GraphRAG",
                "Year": 2024,
                "Core Approach": "Knowledge Graph Extraction",
                "Key Contribution": "Extracts entity-relation graphs and hierarchical communities for holistic dataset synthesis.",
                "Limitations": "High LLM prompt token cost during graph extraction stage."
            }
        ]

        gaps = [
            "Lack of unified benchmarking for section-aware PDF parsing vs fixed-token chunking in complex mathematical documents.",
            "High computational latency during cross-encoder reranking over multi-thousand passage candidate pools.",
            "Limited evaluation protocols for detecting subtle implicit hallucinations in long multi-paper synthesis.",
            "Absence of dynamic graph updating mechanisms when incorporating daily new arXiv submissions."
        ]

        return LitReviewResult(
            topic=topic,
            introduction=(
                f"## Comprehensive Literature Review: {topic}\n\n"
                "Scientific literature synthesis requires retrieving, parsing, and critically analyzing dense technical documents. "
                "This review synthesizes key architectural developments across Transformer models, Dense Passage Retrieval, Hybrid Fusion, and Graph-based RAG architectures."
            ),
            comparison_table=table,
            architectural_evolution=(
                "### Architectural Evolution\n\n"
                "1. **Paradigm Shift to Self-Attention (2017)**: Vaswani et al. introduced the Transformer, discarding recurrent connections in favor of self-attention mechanisms.\n"
                "2. **Pre-training & Bidirectionality (2019)**: Devlin et al. introduced BERT, demonstrating that bidirectional representations capture deep token semantics.\n"
                "3. **Retrieval Augmentation (2020)**: Lewis et al. combined seq2seq generators with non-parametric DPR dense retrieval, establishing the foundation for modern RAG pipelines.\n"
                "4. **Hybrid Sparse-Dense & Graph Summarization (2024)**: Recent advances integrate BM25 sparse keyword matching via Reciprocal Rank Fusion (RRF) and GraphRAG citation networks."
            ),
            methodology_synthesis=(
                "### Methodological Synthesis\n\n"
                "The core trade-off in modern retrieval architectures lies between **Semantic Generalization** (dense embedding search) and **Exact Lexical Precision** (BM25 sparse search). "
                "Hybrid retrieval with Reciprocal Rank Fusion (RRF) combined with `bge-reranker-base` cross-encoders consistently yields optimal Context Precision and Context Recall."
            ),
            identified_research_gaps=gaps,
            conclusion=(
                "### Conclusion & Strategic Recommendations\n\n"
                "To build a state-of-the-art AI Research Copilot, systems must enforce section-aware chunking (250–350 tokens), "
                "leverage hybrid RRF dense/sparse search, apply cross-encoder reranking, and measure performance using reference-free RAGAS metrics."
            ),
            sources=sources
        )

    @staticmethod
    def get_citation_graph() -> CitationGraphData:
        """Simulate Neo4j Paper Citation Graph for PyVis interaction."""
        nodes = [
            {"id": "p1", "label": "Vaswani et al. (2017)", "group": "Transformer", "title": "Attention Is All You Need (112k citations)", "val": 35},
            {"id": "p2", "label": "Devlin et al. (2019)", "group": "Encoder", "title": "BERT: Pre-training (98k citations)", "val": 30},
            {"id": "p3", "label": "Lewis et al. (2020)", "group": "RAG Core", "title": "Retrieval-Augmented Generation (4.8k citations)", "val": 28},
            {"id": "p4", "label": "Karpukhin et al. (2020)", "group": "Dense Retrieval", "title": "Dense Passage Retrieval (3.2k citations)", "val": 24},
            {"id": "p5", "label": "Cormack et al. (2009)", "group": "Hybrid Search", "title": "Reciprocal Rank Fusion (1.4k citations)", "val": 20},
            {"id": "p6", "label": "Es et al. (2023)", "group": "Evaluation", "title": "RAGAS Evaluation Framework (1.2k citations)", "val": 22},
            {"id": "p7", "label": "Edge et al. (2024)", "group": "GraphRAG", "title": "GraphRAG Microsoft (890 citations)", "val": 20},
            {"id": "p8", "label": "Staar et al. (2024)", "group": "Ingestion", "title": "Docling PDF Parser (430 citations)", "val": 18},
            {"id": "p9", "label": "Huber et al. (2023)", "group": "Vector DB", "title": "ChromaDB Architecture (2.1k citations)", "val": 19},
            {"id": "p10", "label": "Chen et al. (2024)", "group": "Embeddings", "title": "BGE M3-Embedding (650 citations)", "val": 17},
        ]

        edges = [
            {"from": "p2", "to": "p1", "label": "cites", "weight": 5},
            {"from": "p3", "to": "p1", "label": "cites", "weight": 4},
            {"from": "p3", "to": "p2", "label": "cites", "weight": 4},
            {"from": "p3", "to": "p4", "label": "uses retriever", "weight": 5},
            {"from": "p4", "to": "p2", "label": "cites", "weight": 3},
            {"from": "p6", "to": "p3", "label": "evaluates", "weight": 4},
            {"from": "p7", "to": "p3", "label": "extends", "weight": 4},
            {"from": "p7", "to": "p1", "label": "cites", "weight": 3},
            {"from": "p8", "to": "p1", "label": "parses", "weight": 2},
            {"from": "p9", "to": "p4", "label": "stores vectors", "weight": 3},
            {"from": "p10", "to": "p5", "label": "uses RRF", "weight": 3},
        ]

        return CitationGraphData(nodes=nodes, edges=edges)

    @staticmethod
    def get_eval_metrics() -> EvalMetrics:
        """Return RAGAS framework evaluation metrics and baseline comparisons."""
        rag_vs_non_rag = {
            "Faithfulness": {"Hybrid RAG Pipeline": 0.86, "Non-RAG Gemini Baseline": 0.34},
            "Context Precision": {"Hybrid RAG Pipeline": 0.74, "Non-RAG Gemini Baseline": 0.12},
            "Context Recall": {"Hybrid RAG Pipeline": 0.78, "Non-RAG Gemini Baseline": 0.15},
            "Answer Relevance": {"Hybrid RAG Pipeline": 0.82, "Non-RAG Gemini Baseline": 0.71},
        }

        stage_comparisons = {
            "Dense Search (ChromaDB)": {"Precision@5": 0.54, "Recall@5": 0.58, "Latency (s)": 0.45},
            "Sparse Search (BM25)": {"Precision@5": 0.48, "Recall@5": 0.51, "Latency (s)": 0.12},
            "Hybrid Fusion (RRF)": {"Precision@5": 0.67, "Recall@5": 0.71, "Latency (s)": 0.62},
            "RRF + Cross-Encoder Reranker": {"Precision@5": 0.84, "Recall@5": 0.78, "Latency (s)": 1.25},
        }

        samples = [
            {
                "id": "QA-01",
                "question": "What problem does RAG solve in LLMs?",
                "ground_truth": "RAG reduces dependence on parametric memory, mitigates factual hallucinations, and integrates up-to-date non-parametric document knowledge.",
                "retrieved_papers": ["Lewis et al. 2020", "Karpukhin et al. 2020"],
                "faithfulness": 0.92,
                "precision": 0.85,
                "recall": 0.88,
                "relevance": 0.90,
                "status": "PASSED"
            },
            {
                "id": "QA-02",
                "question": "How does Reciprocal Rank Fusion calculate document scores?",
                "ground_truth": "RRF sums the reciprocal of constant k (60) plus the document rank across multiple search strategy rank lists.",
                "retrieved_papers": ["Cormack et al. 2009"],
                "faithfulness": 0.88,
                "precision": 0.80,
                "recall": 0.82,
                "relevance": 0.86,
                "status": "PASSED"
            },
            {
                "id": "QA-03",
                "question": "What is the token chunk size target in section-aware parsing?",
                "ground_truth": "The target chunk size is 250 to 350 tokens with section metadata headers.",
                "retrieved_papers": ["Docling Staar et al. 2024"],
                "faithfulness": 0.84,
                "precision": 0.72,
                "recall": 0.75,
                "relevance": 0.79,
                "status": "PASSED"
            },
            {
                "id": "QA-04",
                "question": "What are the limitations of self-attention mechanisms?",
                "ground_truth": "Quadratic memory and computational complexity O(N^2) with respect to input sequence length.",
                "retrieved_papers": ["Vaswani et al. 2017", "Devlin et al. 2019"],
                "faithfulness": 0.90,
                "precision": 0.82,
                "recall": 0.84,
                "relevance": 0.87,
                "status": "PASSED"
            }
        ]

        return EvalMetrics(
            faithfulness=0.86,
            context_precision=0.74,
            context_recall=0.78,
            answer_relevance=0.82,
            rag_vs_non_rag=rag_vs_non_rag,
            stage_comparisons=stage_comparisons,
            eval_samples=samples
        )
