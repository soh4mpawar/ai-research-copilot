# Fabricated Infrastructure Fixtures — NOT Corpus Evidence or Research Results

> **QUARANTINE NOTICE / AUDIT TRAIL**
> The files in this directory are preserved strictly as an audit trail of the initial Stage 2 harness development.
>
> **Known Defects of these Fixtures:**
> 1. Passages contain hand-authored text rather than verified source spans loaded from the corpus.
> 2. Cross-encoder scores were hard-coded (e.g. 0.92, 0.74, 0.45, 0.78) rather than computed by the cross-encoder.
> 3. Sufficiency labels were mapped deterministically from variant names rather than assessed from complete context.
> 4. "Evidence removed" variants substituted invented contradictory statements rather than omitting supporting text.
>
> **THESE FIXTURES MUST NEVER BE USED AS RESEARCH EVIDENCE, BENCHMARK GROUND TRUTH, OR MODEL TRAINING DATA.**
> Genuine corpus-grounded evaluation uses the read-only corpus extractor and real cross-encoder reranker in `backend/selective/dataset.py`.
