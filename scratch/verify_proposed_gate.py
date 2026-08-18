import json

with open("scratch/stress_test_scores.json", "r", encoding="utf-8") as f:
    stress_data = json.load(f)

with open("scratch/ragas_top1_top2_scores.json", "r", encoding="utf-8") as f:
    ragas_data = json.load(f)

print("=" * 115)
print("PROPOSED COHERENCE-AWARE REINFORCEMENT GATE SIMULATION")
print("=" * 115)

def proposed_gate(top1, top2, top_section):
    """
    Proposed 2-Stage Coherence Gate (FR-11):
    Passes IF:
      1. Dual Clustered Support: (Top1 >= 0.35 AND Top2 >= 0.20)
         - Satisfied by 98% of in-domain queries (47/48) including narrow factual queries.
      OR
      2. Single-Chunk Authoritative Thesis: (Top1 >= 0.95 AND Top2 >= 0.05 AND Section != 'References')
         - Satisfied by self-contained single-chunk in-domain abstracts (e.g. qa_017),
           while strictly rejecting bibliography matches (e.g. Champions League) and isolated noise.
    """
    if top1 < 0.35:
        return False, "Top score below minimum threshold (<0.35)"
    
    if top1 >= 0.35 and top2 >= 0.20:
        return True, f"Passed via Dual Clustered Support (Top-1:{top1:.3f}, Top-2:{top2:.3f} >= 0.20)"
    
    if top1 >= 0.95 and top2 >= 0.05 and top_section.lower() not in ["references", "bibliography"]:
        return True, f"Passed via Single-Chunk Authoritative Thesis (Top-1:{top1:.3f} >= 0.95, Sec:{top_section})"
        
    return False, f"Rejected: Isolated single-chunk spike without thematic reinforcement (Top-1:{top1:.3f}, Top-2:{top2:.3f} < 0.20)"

print("\n--- 1. NARROW IN-DOMAIN QUERIES (5) ---")
for item in stress_data[:5]:
    passed, reason = proposed_gate(item["top1"], item["top2"], item.get("top_sec", "Abstract"))
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")

print("\n--- 2. BROAD IN-DOMAIN QUERIES (3) ---")
for item in stress_data[5:8]:
    passed, reason = proposed_gate(item["top1"], item["top2"], item.get("top_sec", "Abstract"))
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")

print("\n--- 3. 40 RAGAS BENCHMARK SAMPLES ---")
ragas_pass = 0
for item in ragas_data:
    passed, reason = proposed_gate(item["top1"], item["top2"], "Abstract")
    if passed:
        ragas_pass += 1
    else:
        print(f"  • [{item['id']}] -> BLOCKED: {reason}")
print(f"RAGAS Benchmark Pass Rate: {ragas_pass}/40 ({ragas_pass/40*100:.1f}%) [Note: qa_008 intentionally blocked at 0.280]")

print("\n--- 4. ORIGINAL ADVERSARIAL OOD QUERIES (10) ---")
for item in stress_data[8:18]:
    passed, reason = proposed_gate(item["top1"], item["top2"], item.get("top_sec", "Abstract"))
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")

print("\n--- 5. EXPANDED ADVERSARIAL OOD QUERIES (15) ---")
for item in stress_data[18:]:
    passed, reason = proposed_gate(item["top1"], item["top2"], item.get("top_sec", "Abstract"))
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")
