# JobMatchAI — Intelligent Job Matching with Knowledge Graphs and Explainable AI

## Links

- **Paper (HTML):** https://arxiv.org/html/2603.14558v2
- **Paper (PDF):** https://arxiv.org/pdf/2603.14558v2
- **arXiv abstract:** https://arxiv.org/abs/2603.14558
- **GitHub (installable package):** https://github.com/coral-lab-asu/job-hunt-AI
- **Live demo video:** https://youtu.be/4jPKvzxZIYU
- **License:** MIT (code); JobSearch-XS benchmark CC-BY-4.0 (upon acceptance)

## Summary

JobMatchAI (Arizona State University, March 2026) is a production-ready, microservices-based platform for explainable job search. It integrates hybrid retrieval (BM25 + semantic + knowledge graph), multi-factor utility reranking, and grounded LLM explanations — achieving **NDCG@10 of 0.81** (~7% over BM25-only) at **sub-100 ms** median latency on the JobSearch-XS benchmark.

## Key Architecture Details

| Component | Detail |
|-----------|--------|
| **Hybrid retrieval** | Parallel BM25 (Elasticsearch, k=150) + ANN semantic (k=150) + Neo4j graph traversal (k=75) |
| **Dense encoder** | `all-MiniLM-L6-v2` (384 dimensions) |
| **Knowledge graph** | Neo4j with Candidate, Job, Skill, Location, Company nodes; ESCO-style RELATED_TO edges |
| **Fusion** | Query-adaptive Reciprocal Rank Fusion (RRF); short queries favor KG, long queries favor text |
| **Scoring layer** | Deterministic multi-factor utility: Skill (0.35), Experience (0.25), Location (0.15), Salary (0.10), Semantic, Company |
| **Explanation layer** | LLM receives only pre-computed scores + KG paths — explains rankings but never inflates them |
| **Benchmark** | JobSearch-XS: 1,283 NYC civil-service roles, 30 queries, 29K silver labels |

### Five-Stage Online Pipeline (from paper)

1. **Query enrichment** — Entity/skill extraction, KG expansion (depth-2 RELATED_TO), dense embedding
2. **Parallel multi-source retrieval** — BM25 + ANN + graph (thread pool)
3. **RRF fusion & deduplication** — Union by job ID, max ~400 candidates
4. **Hard-constraint filtering** — Visa, degree, certifications
5. **Multi-factor reranking** — Weighted utility function with user-adjustable sliders

## What We Borrow

1. **Hybrid three-signal retrieval** — BM25 + semantic + knowledge graph in parallel, then fuse.
2. **ESCO knowledge graph in Neo4j** — Skill expansion and synonym bridging (e.g., Kubernetes ↔ container orchestration).
3. **Explainability architecture** — LLM explains but does not score; scoring is fully auditable and deterministic.
4. **Query-adaptive fusion weights** — Short queries → graph-heavy; long queries → text-heavy.
5. **JobSearch-XS** as our primary offline evaluation benchmark.

## Stack (from paper & repo)

Sentence Transformers, Neo4j, Elasticsearch, FastAPI, spaCy (skill extraction)

## Notes for MVP

- Check `coral-lab-asu/job-hunt-AI` for installation scripts and service layout.
- For MVP we may swap Elasticsearch → local BM25 (`rank_bm25`) and Elasticsearch ANN → FAISS, keeping the same pipeline shape.
