import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from backend.pipeline import ResearchPipelineOrchestrator

def main():
    print("Testing live PointQA pipeline with Gemini 3.7 Flash...")
    orchestrator = ResearchPipelineOrchestrator()
    query = "How does Scaled Dot-Product Attention work in Transformer and why is the scaling factor applied?"
    res = orchestrator.execute_query(query)
    
    print("\n" + "="*80)
    print("LIVE GENERATED ANSWER (Gemini 3.7 Flash):")
    print("="*80)
    print(res.answer)
    print("="*80)
    print("Retrieved chunks count:", len(res.retrieved_chunks))
    print("Latency:", getattr(res, "metrics", {}))

if __name__ == "__main__":
    main()
