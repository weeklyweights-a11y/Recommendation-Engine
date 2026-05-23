# Two-Tower Recsys with Diffusion Cross-Interaction (WWW 2025)

## Links

- **Paper (PDF):** https://arxiv.org/pdf/2502.20687
- **arXiv abstract:** https://arxiv.org/abs/2502.20687
- **Venue:** The Web Conference (WWW) 2025

## Summary

Standard two-tower recommender models encode users and items independently for fast ANN retrieval, but cannot model fine-grained cross-features between user and item representations at encoding time. This paper introduces **diffusion-based cross-interaction** to enhance two-tower models, improving candidate matching quality while preserving approximate retrieval efficiency.

## Key Architecture Details

| Component | Detail |
|-----------|--------|
| **Two-tower baseline** | Separate encoders for user (candidate) and item (job); dot product or cosine for scoring |
| **Limitation** | No interaction between towers until late scoring — misses cross-feature patterns |
| **Diffusion cross-interaction** | Lightweight diffusion module injects cross-signal during training without full cross-attention at inference |
| **Retrieval** | Still supports ANN-friendly embeddings for Stage 1 recall |
| **Reranking** | Cross-interaction features improve Stage 2 ranking over pure dot-product |

## What We Borrow

1. **Understanding two-tower limitations** — Our bi-encoder + FAISS approach is a two-tower pattern; know when it will underperform.
2. **Justification for Stage 2 cross-encoder** — Cross-encoder reranking compensates for the lack of early cross-interaction (same motivation as this paper).
3. **Phase 2 architecture option** — If bi-encoder plateaus, explore diffusion-enhanced towers or full cross-encoder reranking.

## Relevance to HiringCafe MVP

| MVP stage | Two-tower role |
|-----------|----------------|
| Stage 1 (FAISS recall) | Classic two-tower: candidate embedding × job embedding ANN |
| Stage 2 (rerank) | Cross-encoder / utility function adds the "cross-interaction" missing from towers |

## Related Repo

See `resources/repos/repos_index.md` → **Two-Tower Recommender Models (PyTorch)**
