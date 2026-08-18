import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from backend.pipeline import ResearchPipelineOrchestrator

def main():
    print("=" * 80)
    print("LATENCY SPREAD BENCHMARK: GraphRAG ON vs GraphRAG OFF (3 Runs Each)")
    print("=" * 80)

    orchestrator = ResearchPipelineOrchestrator()
    query = "How did Dense Passage Retrieval and BERT contribute to Retrieval-Augmented Generation models?"

    # 1. Benchmark GraphRAG ON
    print("\n--- Testing GraphRAG ON (enable_graph_rag=True) ---")
    on_results = []
    for i in range(1, 4):
        t0 = time.time()
        res = orchestrator.execute_query(query, enable_graph_rag=True)
        dur = time.time() - t0
        m = res.metrics
        on_results.append({
            "run": i,
            "total_sec": round(dur, 2),
            "retrieval_sec": round(m.retrieval_time_sec, 3),
            "reranking_sec": round(m.reranking_time_sec, 3),
            "generation_sec": round(m.generation_time_sec, 2),
            "chunks_fed_to_llm": m.final_context_chunks_count,
            "answer_len": len(res.answer)
        })
        print(f"Run {i}: Total={dur:.2f}s | Retrieval={m.retrieval_time_sec:.3f}s | Rerank={m.reranking_time_sec:.3f}s | Gen={m.generation_time_sec:.2f}s | AnsLen={len(res.answer)}")

    # 2. Benchmark GraphRAG OFF
    print("\n--- Testing GraphRAG OFF (enable_graph_rag=False) ---")
    off_results = []
    for i in range(1, 4):
        t0 = time.time()
        res = orchestrator.execute_query(query, enable_graph_rag=False)
        dur = time.time() - t0
        m = res.metrics
        off_results.append({
            "run": i,
            "total_sec": round(dur, 2),
            "retrieval_sec": round(m.retrieval_time_sec, 3),
            "reranking_sec": round(m.reranking_time_sec, 3),
            "generation_sec": round(m.generation_time_sec, 2),
            "chunks_fed_to_llm": m.final_context_chunks_count,
            "answer_len": len(res.answer)
        })
        print(f"Run {i}: Total={dur:.2f}s | Retrieval={m.retrieval_time_sec:.3f}s | Rerank={m.reranking_time_sec:.3f}s | Gen={m.generation_time_sec:.2f}s | AnsLen={len(res.answer)}")

if __name__ == "__main__":
    main()
