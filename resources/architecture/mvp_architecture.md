# HiringCafe Personalized Job Recommendation Engine — MVP Architecture

Five-stage pipeline for a personalized, explainable job matching MVP. Synthesized from CareerBuilder (two-stage + FAISS), JobMatchAI (hybrid retrieval + separated scoring/explanation), ESCO knowledge graph, and HiringCafe product requirements.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Streamlit Demo Frontend                              │
│   Resume upload │ Preference sliders │ Ranked jobs │ Factor explanations    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ REST
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                            FastAPI Backend                                     │
│  /ingest  /search  /explain  /health                                          │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     ▼                             ▼                             ▼
┌─────────┐                 ┌───────────┐                 ┌───────────┐
│  FAISS  │                 │   BM25    │                 │   Neo4j   │
│  ANN    │                 │  (lexical)│                 │ ESCO KG   │
└─────────┘                 └───────────┘                 └───────────┘
```

**Design principles:**
- Hybrid retrieval beats pure embedding (~7% NDCG gain per JobMatchAI)
- Two-stage: fast recall → accurate rerank (CareerBuilder pattern)
- LLM explains but never scores (JobMatchAI auditability)
- Multi-vector embeddings capture different fit dimensions

---

## Stage 1 — Profile & Job Ingestion

**Goal:** Transform unstructured resumes and job postings into structured, indexable representations.

### Inputs
- Candidate resume (PDF/DOCX/text)
- Optional: preference form (location, salary, remote, company size)
- Job corpus: Kaggle LinkedIn postings (MVP) → HiringCafe live feed (production)

### Processing
| Step | Component | Tech |
|------|-----------|------|
| Document parsing | PDF/DOCX extractor | `pypdf`, `python-docx` |
| Profile extraction | Structured JSON output | **Gemini 2.5 Pro** (primary) or Gemini 2.5 Flash (batch) |
| Skill linking | Map free-text → ESCO URIs | **Tabiya livelihoods classifier** |
| Job normalization | Title, description, salary, location, level | Rule-based + LLM |
| Graph population | Candidate/Job → Skill edges | **Neo4j** Cypher |

### Outputs (per candidate)
```json
{
  "skills": ["http://data.europa.eu/esco/skill/..."],
  "skills_expanded": ["..."],
  "experience_level": "mid",
  "locations": ["New York, NY"],
  "salary_min": 120000,
  "preferences": { "remote": true, "company_size": "startup" }
}
```

### Embeddings generated (multi-vector)
| Vector | Source text | Index |
|--------|-------------|-------|
| `v_text` | Full resume / JD | FAISS `index_text` |
| `v_skills` | Concatenated skill labels | FAISS `index_skills` |
| `v_prefs` | Location + seniority + domain | FAISS `index_prefs` |

**Encoder:** `all-MiniLM-L6-v2` (384d) for all channels in MVP.

---

## Stage 2 — Multi-Signal Indexing

**Goal:** Build retrieval indexes offline so online queries stay under 100ms.

### Indexes

| Index | Technology | Contents | Approx. recall@K |
|-------|------------|----------|------------------|
| Semantic (×3) | **FAISS** HNSW | `v_text`, `v_skills`, `v_prefs` per job | k=150 each |
| Lexical | **BM25** (`rank_bm25`) | Job title + description + skills | k=150 |
| Knowledge graph | **Neo4j** | ESCO skills, RELATED_TO, REQUIRES_SKILL | k=75 |

### Ingestion pipeline (offline)
```
Job CSV/API → normalize → ESCO link → embed (MiniLM) → FAISS.add()
                       → ESCO link → Neo4j MERGE
                       → tokenize → BM25 index
```

### Storage layout
```
data/
├── faiss/
│   ├── index_text.bin
│   ├── index_skills.bin
│   └── job_id_map.json
├── bm25/
│   └── corpus.pkl
└── neo4j/   (running instance)
```

---

## Stage 3 — Hybrid Two-Stage Retrieval

**Goal:** Maximize recall with three complementary signals, then fuse into a single candidate pool.

### Stage 3A — Query enrichment
```
Resume / query text
  → entity + skill extraction
  → Neo4j 2-hop RELATED_TO expansion → S+
  → encode query → e_q (MiniLM)
  → extract keywords K
```

### Stage 3B — Parallel retrieval (thread pool)
| Channel | Method | k |
|---------|--------|---|
| Lexical | BM25(field-boosted title, description, skills) | 150 |
| Semantic | FAISS ANN on `e_q` vs `index_text` (+ optional `index_skills`) | 150 |
| Graph | Cypher: `S+` → REQUIRES_SKILL → Job | 75 |

### Stage 3C — Fusion
- Union candidates by `job_id`
- **Reciprocal Rank Fusion (RRF):** `RRF(d) = Σ w_r / (60 + rank_r(d))`
- Query-adaptive weights (from JobMatchAI):
  - Short query (≤2 tokens): `w_kg=0.7`, `w_sem=0.2`, `w_lex=0.1`
  - Long query: `w_lex=0.6`, `w_sem=0.3`, `w_kg=0.1`
- Cap fused pool at **~400** candidates

### Stage 3D — Hard filters
- Remote requirement, visa, minimum degree, salary floor
- Applied post-fusion to preserve recall diversity

---

## Stage 4 — Scoring & Reranking (Deterministic)

**Goal:** Produce auditable, personalized rankings. **No LLM in this stage.**

### 4A — Cross-encoder rerank (two-stage pattern)
- Input: top-50 from fused pool
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Output: `semantic_cross_score` per (candidate, job) pair

### 4B — Multi-factor utility function
```
U(c, j) = Σ w_f · φ_f(c, j)

F = { Skill, Experience, Location, Salary, Semantic, Company }
```

| Factor | Default weight | Feature φ |
|--------|---------------|-----------|
| Skill match | 0.35 | Jaccard(skill sets) + KG relatedness bonus |
| Experience | 0.25 | Level-distance penalty (junior/mid/senior) |
| Location | 0.15 | Exact city > state > remote OK |
| Salary | 0.10 | Overlap between expected and offered range |
| Semantic | 0.10 | Cross-encoder score (normalized) |
| Company | 0.05 | Size/industry preference match |

- User-adjustable weights via Streamlit sliders
- Utility recomputes in real time on weight change

### Output
```json
{
  "job_id": "...",
  "score": 0.87,
  "factors": {
    "skill": 0.92,
    "experience": 0.85,
    "location": 1.0,
    "salary": 0.70,
    "semantic": 0.88,
    "company": 0.60
  },
  "kg_evidence": ["Kubernetes → container orchestration (RELATED_TO)"]
}
```

---

## Stage 5 — Explanation & Demo Layer

**Goal:** Human-readable match explanations without hallucinated relevance.

### Explanation generation
- **Input to LLM:** Pre-computed factor scores + KG paths + job title/company (no raw scoring authority)
- **Model:** Gemini 2.5 Pro (`LLM_MODEL_PRO`)
- **Output:** 2–3 sentence explanation per job; factor-wise breakdown

```
Prompt template:
  Given scores: {skill: 0.92, experience: 0.85, ...}
  And KG path: Kubernetes ↔ container orchestration
  Explain why this job matches the candidate. Do NOT change or invent scores.
```

### Streamlit demo features
| Feature | Description |
|---------|-------------|
| Resume upload | Triggers Stage 1 extraction |
| Smart search | Resume-driven recommendations (full pipeline) |
| Keyword search | Text query + KG enrichment |
| Weight sliders | Adjust `w_f` in real time |
| Explain button | Per-job LLM explanation |
| Skill gap view | Required skills missing from profile |

### FastAPI endpoints
| Endpoint | Method | Stage |
|----------|--------|-------|
| `POST /ingest/resume` | Upload + extract profile | 1 |
| `POST /ingest/jobs` | Bulk job index | 1–2 |
| `POST /search` | Full pipeline → ranked jobs | 3–4 |
| `POST /explain/{job_id}` | LLM explanation | 5 |
| `GET /health` | Index status | — |

---

## Tech Stack Summary

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit |
| **API** | FastAPI (async) |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) |
| **ANN index** | FAISS (HNSW) |
| **Lexical search** | rank_bm25 |
| **Knowledge graph** | Neo4j + ESCO |
| **ESCO linking** | Tabiya livelihoods classifier |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Scoring** | Deterministic Python (no LLM) |
| **Extraction + explanation** | Gemini 2.5 Pro; batch extraction via Gemini 2.5 Flash |
| **Evaluation** | JobSearch-XS (NDCG@10), LinkedIn Kaggle holdout |
| **Data (MVP)** | ESCO CSVs, Kaggle LinkedIn postings |

---

## Latency Budget (target)

| Step | Target |
|------|--------|
| Query enrichment | < 15 ms |
| Parallel retrieval (BM25 + FAISS + Neo4j) | < 60 ms |
| RRF fusion + filter | < 10 ms |
| Cross-encoder rerank (top-50) | < 200 ms |
| Utility scoring | < 5 ms |
| **Total (excl. LLM explain)** | **< 300 ms** |
| LLM explanation (on demand) | < 2 s |

---

## Evaluation Targets

| Metric | Baseline | MVP target |
|--------|----------|------------|
| NDCG@10 (JobSearch-XS) | BM25-only ~0.74 | **≥ 0.78** (JobMatchAI: 0.81) |
| Recall@50 | — | ≥ 0.70 |
| p50 latency | — | < 300 ms |

---

## Phase 2 Roadmap (post-MVP)

| Enhancement | Source |
|-------------|--------|
| Mira-style synthetic training data + LoRA | mira_embeddings.md |
| BGE-large second semantic channel | resume2vec.md |
| BoundaryHead role-confusion reranking | mira_embeddings.md |
| Diffusion cross-interaction towers | two_tower_diffusion.md |
| HiringCafe live data integration | production |
| Qdrant migration if metadata filtering needed | models_comparison.md |
