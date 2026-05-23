# Models Comparison — HiringCafe Job Recommendation MVP

Models to evaluate for each pipeline component. See `resources/papers/resume2vec.md` for encoder benchmark context.

---

## 1. Embedding Models (Bi-Encoder / Indexing)

Used to encode candidate resumes and job descriptions into dense vectors for FAISS ANN search.

| Model | Dims | Speed | Quality | Cost | Notes |
|-------|------|-------|---------|------|-------|
| **all-MiniLM-L6-v2** | 384 | ⚡ Fast | Good | Free (local) | **MVP default** — used by JobMatchAI; best latency/quality tradeoff |
| **BAAI/bge-large-en-v1.5** | 1024 | Medium | Better | Free (local) | Higher recall; slower indexing; use for quality channel |
| **Cohere embed-v3** | 1024 | API | Strong | Paid API | Strong multilingual; good if we expand beyond EN |
| **OpenAI text-embedding-3-small** | 1536 | API | Strong | Paid API | Easy integration; vendor lock-in |

### MVP recommendation

- **Index (bulk):** `all-MiniLM-L6-v2` — fast FAISS build, proven in JobMatchAI
- **Quality channel (optional):** `BAAI/bge-large-en-v1.5` — second FAISS index for fusion
- **Defer APIs** until offline eval shows local models insufficient

### Evaluation criteria

| Metric | Target |
|--------|--------|
| Recall@50 (JobSearch-XS) | ≥ 0.70 |
| Index build time (100K jobs) | < 30 min on CPU |
| Query encode latency | < 10 ms |

---

## 2. Cross-Encoder Models (Reranking / Stage 2)

Score (resume, job) pairs jointly — slower but more accurate than bi-encoder dot product.

| Model | Speed | Quality | Cost | Notes |
|-------|-------|---------|------|-------|
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | ⚡ Fast | Baseline | Free | **MVP rerank baseline** — MS MARCO trained |
| **BAAI/bge-reranker-large** | Medium | High | Free | Best open-source reranker quality |
| **Cohere rerank-v3** | API | High | Paid | Strong; use if open models insufficient |

### MVP recommendation

- Rerank top-50 from Stage 1 with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Upgrade to `BAAI/bge-reranker-large` if NDCG@10 gain > 2%

### Evaluation criteria

| Metric | Target |
|--------|--------|
| NDCG@10 lift over bi-encoder only | ≥ 5% |
| Rerank latency (50 pairs, CPU) | < 500 ms |

---

## 3. LLMs (Profile Extraction & Explanation)

| Model | Use case | Cost | Notes |
|-------|----------|------|-------|
| **Claude Sonnet** | Structured profile extraction + natural-language explanations | Medium | **Primary** — strong JSON/structured output |
| **GPT-4o-mini** | High-volume extraction (skills, entities) | Low | Cheaper alternative for batch ingestion |

### Strict separation of concerns (from JobMatchAI)

| Layer | Model role |
|-------|-----------|
| **Scoring** | No LLM — deterministic utility function only |
| **Extraction** | LLM parses resume → structured skills, experience, location, salary prefs |
| **Explanation** | LLM receives pre-computed scores + KG paths → generates human-readable rationale |

> The LLM must never assign or modify relevance scores.

---

## 4. Vector Databases (ANN Search)

| Store | Type | Cost | Filtering | Notes |
|-------|------|------|-----------|-------|
| **FAISS** | Local library | Free | Basic | **MVP choice** — CareerBuilder-proven; no infra |
| **Qdrant** | Local or cloud | Free tier | Rich metadata filters | Good Phase 2 if we need filter-heavy queries |
| **Pinecone** | Managed cloud | Paid | Good | Easiest ops; avoid for MVP cost |

### MVP recommendation

**FAISS** with `IndexFlatIP` or `IndexHNSWFlat` for ≤ 500K jobs. Persist index to disk; load on FastAPI startup.

```python
# Minimal FAISS pattern
import faiss
import numpy as np

dim = 384
index = faiss.IndexHNSWFlat(dim, 32)
embeddings = np.array(job_embeddings).astype('float32')
faiss.normalize_L2(embeddings)
index.add(embeddings)
```

---

## 5. Lexical Search (BM25)

| Option | Notes |
|--------|-------|
| **rank_bm25** (Python) | **MVP** — no Elasticsearch dependency; good for ≤ 200K docs |
| **Elasticsearch** | JobMatchAI production choice; add if corpus > 500K |
| **Tantivy / Whoosh** | Alternatives if we need persistent lexical index |

---

## 6. Component → Model Matrix (MVP defaults)

| Pipeline stage | Component | Default model |
|----------------|-----------|---------------|
| Ingestion | Skill/entity extraction | Claude Sonnet (or GPT-4o-mini batch) |
| Ingestion | ESCO linking | Tabiya livelihoods classifier |
| Indexing | Job/candidate embedding | all-MiniLM-L6-v2 |
| Indexing | ANN store | FAISS (HNSW) |
| Indexing | Lexical store | rank_bm25 |
| Indexing | Knowledge graph | Neo4j + ESCO |
| Retrieval | Semantic channel | FAISS k=150 |
| Retrieval | Lexical channel | BM25 k=150 |
| Retrieval | Graph channel | Neo4j Cypher k=75 |
| Rerank | Cross-encoder | ms-marco-MiniLM-L-6-v2 |
| Scoring | Utility function | Deterministic (no LLM) |
| Explanation | Narrative | Claude Sonnet |

---

## 7. Evaluation Plan

| Benchmark | Models compared |
|-----------|----------------|
| JobSearch-XS | MiniLM vs BGE-large vs BM25-only vs hybrid |
| LinkedIn Kaggle (holdout) | Hybrid vs TF-IDF baseline (smart-job-recommendation pattern) |
| Latency | End-to-end p50/p95 with FAISS + BM25 + Neo4j parallel |

See `resources/datasets/datasets_index.md` for dataset download links.
