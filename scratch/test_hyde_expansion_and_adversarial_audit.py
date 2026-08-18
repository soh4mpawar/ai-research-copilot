import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

import json
import time
from backend.pipeline import ResearchPipelineOrchestrator
from backend.retrieval.threshold_gate import RelevanceThresholdGate

orch = ResearchPipelineOrchestrator()
gate = RelevanceThresholdGate()

# The 6 VGG phrasings
vgg_phrasings = [
    ("Q1 (User Natural)", "Why does VGG use small filters instead of big ones?"),
    ("Q2 (Paraphrase A)", "why not just use bigger kernels in VGG"),
    ("Q3 (Paraphrase B)", "what's the benefit of small kernel sizes in CNNs"),
    ("Q4 (Paraphrase C)", "why stack small convolutions instead of one large one"),
    ("Q5 (qa_008 Textbook)", "What is the theoretical benefit of stacking two 3x3 convolution layers instead of using a single 5x5 convolution layer in VGGNet?"),
    ("Q6 (Literal Paper)", "How does the architecture of VGGNet use very small 3x3 convolution filters and stacks of conv layers?")
]

# The 23 adversarial OOD queries
adversarial_ood = [
    ("Weather Forecast", "What's the weather going to be like tomorrow?"),
    ("Bicycle Tire", "How do I fix a flat bicycle tire?"),
    ("Champions League", "Who won the UEFA Champions League last season?"),
    ("Lasagna Recipe", "How to bake a classic meat lasagna with ricotta cheese and tomato sauce?"),
    ("Tax Filing", "What is the deadline for filing federal income taxes in the United States?"),
    ("Capital of Australia", "What is the capital city of Australia?"),
    ("Poem", "Write a rhyming poem about a cat and a dog becoming best friends."),
    ("Workout Plan", "What is the best 5-day workout split for building muscle hypertrophy?"),
    ("Sci-Fi Movies", "What are the best sci-fi movies released in the 1990s?"),
    ("Gravity Spacetime", "What is Einstein's general theory of relativity and spacetime curvature?"),
    ("Bread Algorithm", "What is the algorithm for kneading sourdough bread dough?"),
    ("Photosynthesis", "How does chlorophyll in plant leaves absorb light energy during photosynthesis?"),
    ("Honda Oil Change", "How do you change the engine oil on a 2018 Honda Civic?"),
    ("Texas Hold'em", "What are the complete rules and hand rankings in Texas Hold'em poker?"),
    ("Roman Empire", "What caused the fall of the Western Roman Empire in 476 AD?"),
    ("Paris Vacation", "What are the top 10 tourist attractions to visit in Paris France?"),
    ("Airplane Lift", "How does Bernoulli's principle explain aerodynamic lift on airplane wings?"),
    ("Volcano Magma", "What causes volcanic eruptions and magma formation inside active volcanoes?"),
    ("Zillow Scraper", "How do I write a web scraper in Python to extract real estate prices from Zillow?"),
    ("Acoustic Guitar", "How do you tune an acoustic guitar using standard EADGBE tuning?"),
    ("Human Digestion", "How does the human digestive system absorb nutrients in the small intestine?"),
    ("Japan Inflation", "What economic policies did the Bank of Japan use to fight deflation?"),
    ("Elden Ring Plot", "What is the main storyline and lore of the video game Elden Ring?")
]

def generate_hyde_expansion(query: str) -> str:
    """
    Generate hypothetical scientific passage / domain expansion using Gemini SDK.
    If offline or API fails, returns keyword fallback.
    """
    prompt = (
        f"You are a computer science & AI research search engine. Given the user question: '{query}', "
        f"write a concise 1-2 sentence hypothetical scientific textbook passage that directly answers it "
        f"using formal academic terminology, paper titles, and technical concepts."
    )
    try:
        resp = orch.qa_engine.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )
        return f"{query} {resp.text.strip()}"
    except Exception as e:
        # Fallback to domain synonyms
        return f"{query} academic paper methodology architecture experimental results"

def run_pipeline_with_expansion(raw_query: str, use_hyde: bool = True, k_dense: int = 50, k_sparse: int = 50, k_fused: int = 50):
    search_query = generate_hyde_expansion(raw_query) if use_hyde else raw_query
    
    dense = orch.vector_store.search_dense(search_query, top_k=k_dense)
    sparse = orch.bm25_retriever.search_sparse(search_query, top_k=k_sparse)
    fused = orch.fusion_retriever.fuse_results(dense, sparse, top_k=k_fused)
    graph = orch.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
    pool = list(fused)
    if graph:
        pool.extend(graph)
        
    # We rerank against the raw_query (or search_query) with top_k=15
    reranked_raw = orch.reranker.rerank_chunks(raw_query, pool, top_k=15)
    passed_raw, valid_raw, msg_raw = gate.evaluate_chunks(reranked_raw)
    
    reranked_exp = orch.reranker.rerank_chunks(search_query, pool, top_k=15)
    passed_exp, valid_exp, msg_exp = gate.evaluate_chunks(reranked_exp)
    
    return {
        "raw_query": raw_query,
        "search_query": search_query,
        "reranked_raw": reranked_raw,
        "passed_raw": passed_raw,
        "msg_raw": msg_raw,
        "reranked_exp": reranked_exp,
        "passed_exp": passed_exp,
        "msg_exp": msg_exp
    }

print("=" * 115)
print("PART 1: AUDITING THE 6 VGG PHRASINGS WITH DEEPENED POOL (K=50) + HyDE EXPANSION")
print("=" * 115)

for label, q in vgg_phrasings:
    res = run_pipeline_with_expansion(q, use_hyde=True, k_dense=50, k_sparse=50, k_fused=50)
    top_raw = res["reranked_raw"][0] if res["reranked_raw"] else {}
    top_exp = res["reranked_exp"][0] if res["reranked_exp"] else {}
    
    print(f"\n>>> [{label}]: '{q}'")
    print(f"  • HyDE Generated: '{res['search_query'][:120]}...'")
    print(f"  • Rerank vs Raw Query:        Top-1: {top_raw.get('rerank_score', 0):.4f} (Paper: {top_raw.get('paper_id')}) | Gate: {'PASS' if res['passed_raw'] else 'BLOCKED'}")
    print(f"    - Status Msg: {res['msg_raw']}")
    print(f"  • Rerank vs Expanded Query:   Top-1: {top_exp.get('rerank_score', 0):.4f} (Paper: {top_exp.get('paper_id')}) | Gate: {'PASS' if res['passed_exp'] else 'BLOCKED'}")
    print(f"    - Status Msg: {res['msg_exp']}")

print("\n" + "=" * 115)
print("PART 2: AUDITING THE 23 ADVERSARIAL OOD QUERIES WITH HyDE EXPANSION")
print("=" * 115)

ood_blocked_raw = 0
ood_blocked_exp = 0

for label, q in adversarial_ood:
    res = run_pipeline_with_expansion(q, use_hyde=True, k_dense=50, k_sparse=50, k_fused=50)
    top_raw = res["reranked_raw"][0] if res["reranked_raw"] else {}
    top_exp = res["reranked_exp"][0] if res["reranked_exp"] else {}
    
    if not res["passed_raw"]:
        ood_blocked_raw += 1
    if not res["passed_exp"]:
        ood_blocked_exp += 1
    else:
        print(f"  ⚠️ OOD FALSE POSITIVE on [{label}] vs Expanded! Top: {top_exp.get('rerank_score', 0):.4f} (Paper: {top_exp.get('paper_id')})")
        
    print(f"  [{label}] -> Raw Gate: {'BLOCKED' if not res['passed_raw'] else 'PASS'} ({top_raw.get('rerank_score', 0):.3f}) | Exp Gate: {'BLOCKED' if not res['passed_exp'] else 'PASS'} ({top_exp.get('rerank_score', 0):.3f})")

print("\n" + "=" * 115)
print(f"OOD BLOCKED SUMMARY: Raw Rerank: {ood_blocked_raw}/23 | Exp Rerank: {ood_blocked_exp}/23")
print("=" * 115)
