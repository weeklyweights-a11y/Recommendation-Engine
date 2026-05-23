# CareerBuilder Two-Stage System for Job Recommendation

## Links

- **Paper (PDF):** https://arxiv.org/pdf/2107.00221
- **arXiv abstract:** https://arxiv.org/abs/2107.00221

## Summary

CareerBuilder describes a production-scale job recommendation system deployed at one of the largest job boards. The system addresses the cold-start and scale challenges of matching millions of job seekers to millions of postings in real time.

## Key Architecture Details

| Component | Detail |
|-----------|--------|
| **Fused embedding** | Combines raw job/candidate text, semantic entities (skills, titles), and geolocation into a single dense representation |
| **ANN indexing** | FAISS for approximate nearest-neighbor search over fused embeddings at production scale |
| **Two-stage retrieval** | Stage 1: fast ANN recall (top-K candidates); Stage 2: reranking with richer cross-features |
| **Multi-signal fusion** | Text + structured entities + location signals encoded together rather than a single bag-of-words vector |
| **Production scale** | Designed for CareerBuilder's full traffic; emphasizes latency and recall tradeoffs |

## What We Borrow

1. **Two-stage architecture pattern** — Fast ANN recall first, then expensive reranking on a smaller candidate set.
2. **FAISS for ANN retrieval** — Local, free, proven at CareerBuilder scale; ideal for MVP.
3. **Fused multi-signal embeddings** — Separate vectors or fused features for text, skills, and location rather than one generic embedding.

## Relevance to HiringCafe MVP

- Validates our Stage 1 (FAISS recall) → Stage 2 (cross-encoder + utility rerank) pipeline.
- Supports multi-vector indexing: one index per signal type (full text, skills, geo) with late fusion at scoring time.
