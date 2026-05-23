# Mira-Embeddings-V1 — Boundary-Aware Job–Resume Embeddings

## Links

- **Paper (HTML):** https://arxiv.org/html/2604.17738
- **arXiv:** https://arxiv.org/abs/2604.17738 (check for PDF when available)

## Summary

Mira-Embeddings-V1 (April 2026) trains job-description and resume embeddings without manual labels by using an LLM-driven synthetic data pipeline and two-round LoRA adaptation. A **BoundaryHead MLP** reduces role-boundary confusion (e.g., similar titles across different seniority domains). Recall@50 improves from **68.89% → 77.55%**.

## Key Architecture Details

| Component | Detail |
|-----------|--------|
| **Synthetic data pipeline** | Five-stage LLM prompt pipeline generates contrastive training pairs (no human labels) |
| **Two-round LoRA** | Round 1: JD–JD contrastive (job similarity); Round 2: JD–CV alignment (resume–job matching) |
| **BoundaryHead MLP** | Auxiliary head that penalizes embeddings that confuse adjacent but distinct role boundaries |
| **Cold start** | Entire training signal from LLM-synthesized data — no labeled hire/no-hire pairs required |
| **Metrics** | Recall@50: 68.89% → 77.55% on job–resume retrieval |

### Five-Stage LLM Prompt Pipeline (conceptual)

1. Job description parsing and normalization
2. Synthetic resume generation conditioned on JD
3. Hard negative mining (similar but non-matching roles)
4. Contrastive pair construction
5. Quality filtering and deduplication

## What We Borrow

1. **LLM-synthesized training data** — For cold start when we lack HiringCafe hire labels; generate JD–resume pairs from public job postings.
2. **Boundary-aware reranking concept** — Phase 2 enhancement: detect when two roles look semantically similar but belong to different seniority/function clusters.
3. **Two-stage fine-tuning strategy** — First align job–job space, then align job–candidate space.

## Relevance to HiringCafe MVP

- **Phase 1 (MVP):** Use off-the-shelf encoders (`all-MiniLM-L6-v2`, BGE).
- **Phase 2:** Explore Mira-style synthetic data + LoRA if off-the-shelf embeddings plateau on JobSearch-XS or internal eval.

## Phase 2 Implementation Sketch

```
Public JD corpus → LLM synthetic resumes → LoRA round 1 (JD-JD) → LoRA round 2 (JD-CV) → BoundaryHead reranker
```
