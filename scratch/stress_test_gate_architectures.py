import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

import json
from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

# ==============================================================================
# 1. Narrow Single-Fact In-Domain Queries (Testing for False Negatives)
# ==============================================================================
narrow_in_domain_queries = [
    ("Narrow ID 1 (VGG Batch Size)", "What batch size was used for training ConvNets in paper 1409.1556?"),
    ("Narrow ID 2 (BERT Token Masking)", "What percentage of tokens are masked in BERT masked language modeling?"),
    ("Narrow ID 3 (Swin Resolution)", "What input image resolution is used for Swin Transformer pre-training?"),
    ("Narrow ID 4 (AdamW Beta2)", "What beta2 hyperparameter value is used for Adam optimizer in Attention Is All You Need?"),
    ("Narrow ID 5 (ResNet Stride)", "What stride is used in downsampling convolutional layers in ResNet?")
]

# ==============================================================================
# 2. Broad In-Domain Controls
# ==============================================================================
broad_in_domain_queries = [
    ("Broad ID 1 (Transformer)", "How does multi-head self-attention work in the Transformer architecture?"),
    ("Broad ID 2 (ResNet)", "Why do residual shortcut connections in ResNet prevent vanishing gradients?"),
    ("Broad ID 3 (VGG)", "What is the theoretical benefit of using a stack of three 3x3 convolution layers instead of a single 7x7 layer in VGGNet?")
]

# ==============================================================================
# 3. Original 10 Adversarial + Champions League Queries
# ==============================================================================
original_ood_queries = [
    ("OOD Weather", "What's the weather going to be like tomorrow?"),
    ("OOD Bicycle Tire", "How do I fix a flat bicycle tire?"),
    ("OOD Champions League", "Who won the Champions League last season?"),
    ("OOD Lasagna", "How to bake a classic meat lasagna with ricotta cheese and tomato sauce?"),
    ("OOD Taxes", "How do I file my personal income taxes online?"),
    ("OOD Capital Australia", "What is the capital city of Australia?"),
    ("OOD Stock Price", "What is the current stock price of Apple Inc (AAPL)?"),
    ("OOD Poem", "Write a rhyming poem about a cat sitting on a windowsill."),
    ("OOD Workout Plan", "What is a 4-day gym split workout plan for building muscle?"),
    ("OOD Sci-Fi Movies", "What are the top 10 best sci-fi movies of all time?")
]

# ==============================================================================
# 4. Expanded 15 Out-of-Domain Queries (including scientific-sounding pseudo queries)
# ==============================================================================
expanded_ood_queries = [
    ("OOD Gravity Spacetime", "What is the theory of general relativity and how does gravity curve spacetime?"),
    ("OOD Bread Algorithm", "Explain the algorithm for baking sourdough bread using yeast fermentation."),
    ("OOD Photosynthesis", "What is the molecular formula for photosynthesis in plants?"),
    ("OOD Honda Oil Change", "How do I change the engine oil in a 2018 Honda Civic?"),
    ("OOD Texas Holdem", "What are the rules and hand rankings of Texas Hold'em poker?"),
    ("OOD Roman Empire", "Who was the first emperor of the Roman Empire and when did he rule?"),
    ("OOD Paris Vacation", "What is the best 3-day travel itinerary for visiting Paris France?"),
    ("OOD Airplane Lift", "How do airplane wings generate aerodynamic lift using Bernoulli's principle?"),
    ("OOD Appendicitis", "What are the early medical symptoms and clinical diagnosis of acute appendicitis?"),
    ("OOD Zillow Scraper", "Write a Python script using BeautifulSoup to scrape real estate listings from Zillow."),
    ("OOD Acoustic Guitar", "What is the physical difference between an acoustic guitar and an electric guitar?"),
    ("OOD Human Digestion", "How does the human digestive system absorb carbohydrates and nutrients?"),
    ("OOD Japan Inflation", "What is the current annual inflation rate and central bank monetary policy in Japan?"),
    ("OOD Elden Ring Plot", "What is the lore and main storyline of the video game Elden Ring?"),
    ("OOD Solar Panels", "How do photovoltaic solar panels convert sunlight into electrical current using semiconductors?")
]

all_test_sets = [
    ("1. NARROW SINGLE-FACT IN-DOMAIN QUERIES", narrow_in_domain_queries, True),
    ("2. BROAD IN-DOMAIN QUERIES", broad_in_domain_queries, True),
    ("3. ORIGINAL ADVERSARIAL OOD QUERIES", original_ood_queries, False),
    ("4. EXPANDED ADVERSARIAL OOD QUERIES (15)", expanded_ood_queries, False)
]

def run_retrieval_and_rerank(q):
    dense_candidates = orch.vector_store.search_dense(q, top_k=20)
    sparse_candidates = orch.bm25_retriever.search_sparse(q, top_k=20)
    fused_candidates = orch.fusion_retriever.fuse_results(dense_candidates, sparse_candidates, top_k=25)
    graph_candidates = orch.graph_retriever.traverse_and_fetch_chunks(fused_candidates, max_graph_candidates=10)
    candidate_pool = list(fused_candidates)
    if graph_candidates:
        candidate_pool.extend(graph_candidates)
    reranked = orch.reranker.rerank_chunks(q, candidate_pool, top_k=10)
    scores = [c.get("rerank_score", 0.0) for c in reranked]
    return reranked, scores

print("=" * 115)
print("EXECUTING STRESS TEST ON RETRIEVAL SCORE DISTRIBUTIONS")
print("=" * 115)

collected_results = []

for group_name, q_list, is_in_domain in all_test_sets:
    print(f"\n>>> {group_name} ({len(q_list)} queries) <<<")
    for label, q in q_list:
        reranked, scores = run_retrieval_and_rerank(q)
        top1 = scores[0] if len(scores) > 0 else 0.0
        top2 = scores[1] if len(scores) > 1 else 0.0
        top3 = scores[2] if len(scores) > 2 else 0.0
        valid_025 = sum(1 for s in scores if s >= 0.25)
        valid_035 = sum(1 for s in scores if s >= 0.35)
        gap = round(top1 - top2, 4)
        top_paper = reranked[0].get("paper_id", "") if reranked else ""
        top_sec = reranked[0].get("section", "") if reranked else ""
        
        collected_results.append({
            "group": group_name,
            "label": label,
            "query": q,
            "is_in_domain": is_in_domain,
            "top1": top1,
            "top2": top2,
            "top3": top3,
            "gap": gap,
            "valid_025": valid_025,
            "valid_035": valid_035,
            "top_paper": top_paper,
            "top_sec": top_sec,
            "top_text": reranked[0].get("text", "")[:120].replace("\n", " ") if reranked else ""
        })
        
        print(f"[{label}] Top-1:{top1:.4f} | Top-2:{top2:.4f} | Top-3:{top3:.4f} | Gap:{gap:.4f} | Chunks>=0.35:{valid_035} | Paper:{top_paper} ({top_sec})")

# Save results to JSON for multi-gate formulation simulation
with open("scratch/stress_test_scores.json", "w", encoding="utf-8") as f:
    json.dump(collected_results, f, indent=2)

print("\n" + "=" * 115)
print("STRESS TEST COMPLETED — RESULTS SAVED TO scratch/stress_test_scores.json")
print("=" * 115)
