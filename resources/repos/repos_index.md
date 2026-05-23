# Reference Repositories Index

> Star counts fetched 2026-05-23 via GitHub API. Re-check before citing in docs.

## Primary References (directly aligned with MVP)

### JobMatchAI (ASU)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/coral-lab-asu/job-hunt-AI |
| **Paper** | https://arxiv.org/html/2603.14558v2 |
| **Stars** | 1 |
| **License** | MIT |
| **What it does** | Installable package for the JobMatchAI system: hybrid BM25 + semantic + Neo4j KG retrieval, multi-factor reranking, explainable LLM layer |
| **Stack** | Sentence Transformers, Neo4j, Elasticsearch, FastAPI, spaCy |
| **What we learn** | Full hybrid retrieval implementation, microservice layout, query-adaptive RRF fusion, separated scoring/explanation layers, JobSearch-XS eval setup |
| **Demo** | https://youtu.be/4jPKvzxZIYU |

---

### SkillAlign — ESCO Career Recommendation Engine

| Field | Value |
|-------|-------|
| **URL** | https://github.com/Y4SSERk/SkillAlign |
| **Stars** | 2 |
| **License** | Check repo |
| **What it does** | AI career recommendation engine mapping user skills to ESCO occupations via Neo4j KG + FAISS semantic search |
| **Stack** | Neo4j 5.x, FAISS, FastAPI, Next.js 14, SentenceTransformers (`all-mpnet-base-v2`), ESCO CSVs (17 files) |
| **Tags** | `python` `machine-learning` `neo4j` `nextjs` `knowledge-graph` `semantic-search` `faiss` `fastapi` `vector-search` `career-recommendation` `esco-taxonomy` |
| **What we learn** | ESCO CSV → Neo4j ETL pipeline, FAISS index generation from skill/occupation embeddings, skill expansion via graph traversal, sub-800ms recommendation pattern |

> **Note:** This is the closest match to the tagged "ESCO knowledge graph FAISS Neo4j career recommendation" search. Use as reference for KG loading; our MVP uses Streamlit instead of Next.js.

---

### Tabiya Livelihoods Classifier

| Field | Value |
|-------|-------|
| **URL** | https://github.com/tabiya-tech/tabiya-livelihoods-classifier |
| **Paper** | https://arxiv.org/pdf/2512.03195 |
| **Stars** | 6 |
| **License** | Check repo |
| **What it does** | Entity-linking of job descriptions to ESCO/ISCO taxonomies (sentence linking + entity linking) |
| **Stack** | Python, transformer models, ESCO/ISCO taxonomies |
| **What we learn** | How to link free-text skills and occupations to ESCO taxonomy nodes for our knowledge graph ingestion layer |

---

## Retrieval & Reranking Patterns

### Two-Tower Recommender Models (PyTorch)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/gauravchak/two_tower_models |
| **Stars** | 113 |
| **What it does** | Sample two-tower retrieval code + RLHF/RLAIF alignment with a ranking model on top |
| **Stack** | PyTorch |
| **What we learn** | Two-tower implementation pattern, how to align retrieval embeddings with a downstream ranker |

---

### Cross-Encoder & Reranking Demo

| Field | Value |
|-------|-------|
| **URL** | https://github.com/ianhohoho/cross-encoder-and-reranking-demo |
| **Stars** | 18 |
| **What it does** | Eight runnable examples: bi-encoder vs cross-encoder, fine-tuning, caching, distillation, ColBERT late interaction |
| **Stack** | Python, Sentence Transformers |
| **What we learn** | Bi-encoder vs cross-encoder tradeoffs, multi-stage reranking patterns for our Stage 2 pipeline |

---

### LLM Semantic Search Pipeline

| Field | Value |
|-------|-------|
| **URL** | https://github.com/saleena-18/llm-semantic-search |
| **Stars** | 1 |
| **What it does** | Multi-stage semantic retrieval: BM25 → Cohere embeddings + Annoy → UMAP visualization → Weaviate hybrid rerank |
| **Stack** | BM25, Cohere embeddings, Annoy, UMAP, Weaviate, Jupyter |
| **What we learn** | Hybrid BM25 + vector retrieval pipeline structure; reranking via managed vector DB (Weaviate pattern — we use FAISS locally) |

---

## Baselines & Comparisons

### Smart Job Recommendation (Django + scikit-learn)

| Field | Value |
|-------|-------|
| **URL** | https://github.com/hossein-sa/smart-job-recommendation |
| **Stars** | 4 |
| **What it does** | Classic ML job recommender: TF-IDF + cosine similarity on user profiles |
| **Stack** | Django, DRF, PostgreSQL, scikit-learn, NLTK |
| **What we learn** | Simple TF-IDF + cosine baseline to benchmark against; expect our hybrid system to significantly outperform on synonym/generalization cases |

---

## Quick Reference Matrix

| Repo | Hybrid retrieval | KG | FAISS | Rerank | ESCO | MVP priority |
|------|-----------------|-----|-------|--------|------|--------------|
| coral-lab-asu/job-hunt-AI | ✅ | ✅ Neo4j | ❌ (ES ANN) | ✅ utility | ✅ | **P0** |
| Y4SSERk/SkillAlign | partial | ✅ Neo4j | ✅ | partial | ✅ | **P0** |
| tabiya-tech/tabiya-livelihoods-classifier | ❌ | linking only | ❌ | ❌ | ✅ | **P0** |
| gauravchak/two_tower_models | towers only | ❌ | ❌ | ✅ RLHF | ❌ | P1 |
| ianhohoho/cross-encoder-and-reranking-demo | ❌ | ❌ | ❌ | ✅ | ❌ | **P0** |
| saleena-18/llm-semantic-search | ✅ | ❌ | Annoy | ✅ | ❌ | P1 |
| hossein-sa/smart-job-recommendation | ❌ | ❌ | ❌ | ❌ | ❌ | baseline |

## Clone Commands (optional)

```bash
cd D:\LinkedIn\Hiringcafe_demo\resources\repos

git clone https://github.com/coral-lab-asu/job-hunt-AI.git
git clone https://github.com/Y4SSERk/SkillAlign.git
git clone https://github.com/tabiya-tech/tabiya-livelihoods-classifier.git
git clone https://github.com/gauravchak/two_tower_models.git
git clone https://github.com/ianhohoho/cross-encoder-and-reranking-demo.git
git clone https://github.com/saleena-18/llm-semantic-search.git
git clone https://github.com/hossein-sa/smart-job-recommendation.git
```
