"""
Unit tests for SplitRestrictedHybridRetriever (Protocol Amendment v1.1).
Verifies that:
1. Both dense and BM25 streams are strictly restricted to the assigned paper pool.
2. The genuine hybrid pipeline (Dense + BM25 + RRF + Cross-Encoder) executes without lexical-overlap fallback.
3. Candidate component ranks, fusion ranks, rerank scores, payload hashes, and budgets are correctly computed.
"""

import os
import sys
import unittest
import json

from backend.ingestion.vector_store import VectorStore
from backend.selective.retrieval_adapter import SplitRestrictedHybridRetriever
from sentence_transformers import CrossEncoder


class TestSplitRestrictedHybridRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manifest_path = "data/metadata/full_benchmark_split_manifest.json"
        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        cls.train_pids = [p["paper_id"] for p in manifest["paper_allocations"]["train"][:5]]
        cls.test_pids = [p["paper_id"] for p in manifest["paper_allocations"]["test"][:3]]
        
        cls.vs = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")
        cls.reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        
        cls.retriever = SplitRestrictedHybridRetriever(
            vector_store=cls.vs,
            reranker=cls.reranker,
            allowed_paper_ids=cls.train_pids,
            dense_top_k=10,
            sparse_top_k=10,
            fusion_top_k=15,
            final_max_passages=6,
            final_max_tokens=2000
        )

    def test_pool_restriction_integrity(self):
        """Verify that all chunks loaded in pool belong strictly to allowed paper IDs."""
        for c in self.retriever.pool_chunks:
            self.assertIn(
                c["paper_id"],
                self.train_pids,
                f"Chunk {c['chunk_id']} from paper {c['paper_id']} leaked into train pool!"
            )
            for forbidden_pid in self.test_pids:
                self.assertNotEqual(
                    c["paper_id"],
                    forbidden_pid,
                    f"Test paper {forbidden_pid} leaked into train pool!"
                )

    def test_hybrid_pipeline_execution_and_ranks(self):
        """Verify execution of dense, sparse BM25, RRF fusion, and cross-encoder reranking."""
        query = "How does on-policy distillation compare with pre-OPD base models as sampling budget increases?"
        res = self.retriever.retrieve_hybrid_context(query)
        
        # 1. Check timings
        timings = res["timings"]
        self.assertIn("dense_sec", timings)
        self.assertIn("sparse_sec", timings)
        self.assertIn("fusion_sec", timings)
        self.assertIn("rerank_sec", timings)
        self.assertGreater(timings["total_sec"], 0.0)
        
        # 2. Check candidate trace
        trace = res["candidate_trace"]
        self.assertGreater(len(trace), 0)
        for cand in trace:
            self.assertIn("chunk_id", cand)
            self.assertIn(cand["paper_id"], self.train_pids)
            self.assertIn("rerank_score", cand)
            self.assertIn("rrf_rank", cand)
            self.assertIn("rrf_score", cand)
            self.assertIsInstance(cand["rerank_score"], float)
            
        # 3. Check final record
        rec = res["final_record"]
        self.assertLessEqual(len(rec.passages), 6)
        self.assertLessEqual(rec.total_tokens, 2000)
        self.assertGreater(len(rec.passages), 0)
        self.assertTrue(len(rec.payload_hash) == 64)
        
        # Verify passage scores are sorted descending
        scores = [p.score for p in rec.passages]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_forbidden_paper_retrieved(self):
        """Test with a query designed to hit a test paper, ensuring the filter blocks it."""
        # Query specifically about test paper concepts
        query = "PRISM lesion-symptom mapping for post-stroke aphasia patients"
        res = self.retriever.retrieve_hybrid_context(query)
        rec = res["final_record"]
        for p in rec.passages:
            self.assertIn(
                p.paper_id,
                self.train_pids,
                f"Retrieved passage from unallowed paper {p.paper_id}"
            )

    def test_filtered_query_failure_raises_and_never_queries_global(self):
        """(a) Confirm that filtered-query failure raises an explicit RuntimeError and never queries global collection."""
        from unittest.mock import MagicMock
        
        # Save original query method
        orig_query = self.retriever.vector_store.collection.query
        mock_query = MagicMock(side_effect=Exception("Simulated ChromaDB where-filter failure"))
        self.retriever.vector_store.collection.query = mock_query
        
        try:
            with self.assertRaises(RuntimeError) as ctx:
                self.retriever.search_dense("test query", top_k=5)
            self.assertIn("Split-filtered dense query failed", str(ctx.exception))
            # Assert query was called exactly once with the where filter, and never fell back to an unfiltered query
            self.assertEqual(mock_query.call_count, 1)
            call_kwargs = mock_query.call_args[1]
            self.assertIn("where", call_kwargs)
            self.assertIn("paper_id", call_kwargs["where"])
        finally:
            self.retriever.vector_store.collection.query = orig_query

    def test_missing_or_failing_embeddings_raise_error_not_hashed_vector(self):
        """(b) Confirm missing or failing embeddings trigger an explicit error, not silent fallback to hashed vectors."""
        # 1. Missing embed_model on VectorStore with strict=True
        orig_model = self.vs.embed_model
        try:
            self.vs.embed_model = None
            with self.assertRaises(RuntimeError) as ctx:
                self.vs._compute_dense_vector("test query", strict=True)
            self.assertIn("Dense embedding model is not loaded in strict experiment mode", str(ctx.exception))
            
            # Initializing retriever with missing embed_model must raise RuntimeError
            with self.assertRaises(RuntimeError) as ctx_init:
                SplitRestrictedHybridRetriever(
                    vector_store=self.vs,
                    reranker=self.reranker,
                    allowed_paper_ids=self.train_pids
                )
            self.assertIn("Experiment mode violation: vector_store.embed_model is None", str(ctx_init.exception))
        finally:
            self.vs.embed_model = orig_model

        # 2. Failing embed_model.encode with strict=True
        from unittest.mock import MagicMock
        failing_model = MagicMock()
        failing_model.encode.side_effect = RuntimeError("CUDA out of memory during embedding")
        self.vs.embed_model = failing_model
        try:
            with self.assertRaises(RuntimeError) as ctx:
                self.vs._compute_dense_vector("test query", strict=True)
            self.assertIn("Dense vector embedding computation failed in strict mode", str(ctx.exception))
        finally:
            self.vs.embed_model = orig_model

    def test_dense_and_sparse_chunk_eligibility_consistent(self):
        """(c) Confirm dense and sparse retrieval index and return from the exact same eligible non-reference chunk set."""
        # Check pool definition: no references in pool_chunks or pool_chunk_ids
        for c in self.retriever.pool_chunks:
            self.assertNotIn("ref", c["section"].lower(), f"Reference chunk {c['chunk_id']} present in pool_chunks")
            self.assertTrue(len(c["text"].strip()) > 0, f"Empty chunk {c['chunk_id']} present in pool_chunks")
            self.assertIn(c["chunk_id"], self.retriever.pool_chunk_ids)

        query = "on-policy distillation test-time scaling loss"
        dense_results = self.retriever.search_dense(query, top_k=10)
        sparse_results = self.retriever.search_sparse(query, top_k=10)

        self.assertGreater(len(dense_results), 0)
        self.assertGreater(len(sparse_results), 0)

        # Dense candidates must strictly be members of pool_chunk_ids
        for d in dense_results:
            self.assertIn(
                d["chunk_id"],
                self.retriever.pool_chunk_ids,
                f"Dense retrieval returned chunk {d['chunk_id']} which is not in eligible pool_chunks!"
            )
            self.assertNotIn("ref", d["section"].lower(), f"Dense candidate {d['chunk_id']} is a reference chunk!")

        # Sparse candidates must strictly be members of pool_chunk_ids
        for s in sparse_results:
            self.assertIn(
                s["chunk_id"],
                self.retriever.pool_chunk_ids,
                f"Sparse retrieval returned chunk {s['chunk_id']} which is not in eligible pool_chunks!"
            )
            self.assertNotIn("ref", s["section"].lower(), f"Sparse candidate {s['chunk_id']} is a reference chunk!")

    def test_progressive_retrieval_does_not_lose_candidates_behind_references(self):
        """Confirm that when many high-ranking reference chunks precede eligible passages,
        progressive query expansion retrieves deeper into the allowed pool so eligible candidates
        are not silently lost due to an initial retrieval limit.
        """
        from unittest.mock import MagicMock
        
        orig_query = self.retriever.vector_store.collection.query
        
        # Pick 5 real eligible chunks from pool_chunks
        eligible_sample = self.retriever.pool_chunks[:5]
        eligible_cids = [c["chunk_id"] for c in eligible_sample]
        
        # Create 35 simulated reference / ineligible chunk IDs
        ineligible_cids = [f"simulated_ref_chunk_{i:03d}" for i in range(35)]
        
        # The full ranking puts all 35 ineligible chunks FIRST, followed by the 5 eligible chunks
        full_ranking_cids = ineligible_cids + eligible_cids
        
        def mock_chroma_query(query_embeddings, n_results, where):
            # Return top n_results from full_ranking_cids
            requested_cids = full_ranking_cids[:n_results]
            return {
                "ids": [requested_cids],
                "documents": [["Text for " + cid for cid in requested_cids]],
                "metadatas": [[{"paper_id": "test_paper", "section": "References" if "ref" in cid else "Method"} for cid in requested_cids]],
                "distances": [[0.01 * (i + 1) for i in range(len(requested_cids))]]
            }
        
        mock = MagicMock(side_effect=mock_chroma_query)
        self.retriever.vector_store.collection.query = mock
        
        try:
            # Request top_k = 5.
            # Initial batch_size will be min(max(5 * 4, 30), max_possible) = 30.
            # The top 30 chunks are all ineligible references!
            # Progressive expansion must expand batch_size to 60, successfully retrieving the 5 eligible chunks.
            results = self.retriever.search_dense("test query", top_k=5)
            
            self.assertEqual(len(results), 5, "Progressive retrieval failed to return 5 eligible candidates!")
            returned_cids = [r["chunk_id"] for r in results]
            self.assertEqual(returned_cids, eligible_cids, "Eligible candidates were not returned in expected order!")
            # Confirm that query was called more than once due to progressive expansion
            self.assertGreater(mock.call_count, 1, "ChromaDB query was not progressively expanded!")
        finally:
            self.retriever.vector_store.collection.query = orig_query


if __name__ == "__main__":
    unittest.main()
