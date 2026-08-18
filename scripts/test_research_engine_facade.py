import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from backend import research_engine

def main():
    print("Testing research_engine facade...")
    papers = research_engine.get_corpus_papers()
    print(f"Corpus papers: {len(papers)}")
    
    graph = research_engine.get_citation_graph()
    print(f"Citation Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    eval_m = research_engine.get_eval_metrics()
    print(f"Eval metrics: Faithfulness={eval_m.faithfulness}, Precision={eval_m.context_precision}, Recall={eval_m.context_recall}, Relevancy={eval_m.answer_relevance}")

if __name__ == "__main__":
    main()
