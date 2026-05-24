# Technology Decisions — HiringCafe Job Recommendation MVP

Rationale for each major component choice. Each decision maps to evidence from our reference papers and repos.

---

## Vector Search: FAISS over Pinecone (MVP)

| | FAISS | Pinecone |
|---|-------|----------|
| **Cost** | Free, local | Paid managed |
| **Latency** | In-process, no network | Network round-trip |
| **Proven at scale** | CareerBuilder production (arxiv:2107.00221) | Many startups, not our reference arch |
| **MVP fit** | ✅ | ❌ (cost + vendor lock-in) |

**Decision:** FAISS with HNSW index for MVP. Migrate to Qdrant or Pinecone only if we need managed infra, multi-tenant isolation, or rich metadata filtering at scale.

**Reference:** `papers/careerbuilder_two_stage.md`, `models_info/models_comparison.md`

---

## Knowledge Graph: Neo4j + ESCO

| | Neo4j | Alternatives (NetworkX, PostgreSQL ltree) |
|---|-------|------------------------------------------|
| **ESCO ecosystem** | Multiple open projects already load ESCO into Neo4j | No standard pattern |
| **Query language** | Cypher — expressive for multi-hop skill expansion | Manual graph traversal code |
| **Skill expansion** | `(PyTorch)-[:RELATED_TO*1..2]-(skill)` natively | Possible but painful |
| **JobMatchAI validation** | Production architecture in JobMatchAI paper | — |

**Decision:** Neo4j (local Desktop for dev, Aura free tier for demos) with ESCO taxonomy as the skill backbone.

**Reference:** `papers/jobmatchai.md`, `knowledge_graph/neo4j_esco_setup.md`, `repos/repos_index.md` (SkillAlign)

---

## Scoring vs Explanation: Strict Separation

| Concern | Combined LLM scoring | Separated layers |
|---------|---------------------|------------------|
| **Auditability** | Opaque; scores can drift | Deterministic, reproducible |
| **Hallucinated relevance** | LLM may inflate match quality | LLM only narrates pre-computed scores |
| **Regulatory/compliance** | Risky in hiring contexts | Factor-wise decomposition auditable |
| **User trust** | "Why am I seeing this?" unanswered | Per-factor breakdown + narrative |

**Decision:** Deterministic multi-factor utility function for all ranking. Gemini Pro receives only `{factor_scores, kg_paths, job_metadata}` for explanations — never raw documents for scoring.

**Reference:** `papers/jobmatchai.md` — "the strict separation of a deterministic scoring layer from a generative explanation layer"

---

## Embeddings: Multi-Vector over Single Vector

| | Single vector | Multi-vector (text + skills + prefs) |
|---|--------------|--------------------------------------|
| **Signal capture** | Averages away skill vs domain vs location | Each dimension indexed separately |
| **CareerBuilder pattern** | — | Fused/multi-signal embeddings |
| **Fusion flexibility** | One similarity score | RRF across channels; channel can be down-weighted |
| **Cold start** | Poor when one field is sparse | Skills channel still works if summary is thin |

**Decision:** Three FAISS indexes per job/candidate: `v_text`, `v_skills`, `v_prefs`. Fuse at retrieval with RRF, not at embedding time.

**Reference:** `papers/careerbuilder_two_stage.md` (fused multi-signal embeddings)

---

## Retrieval: BM25 + Semantic + Graph Hybrid over Pure Embedding

| Approach | NDCG@10 (JobSearch-XS) | Synonym handling |
|----------|------------------------|------------------|
| BM25 only | ~0.74 (baseline) | Poor |
| Semantic only | Better than BM25 | Good |
| **Hybrid (BM25 + semantic + KG)** | **0.81** | Best |

**Decision:** Three parallel retrievers with RRF fusion. Pure embedding-only is insufficient — JobMatchAI shows ~7% NDCG improvement from hybrid.

**Reference:** `papers/jobmatchai.md`

### Channel-specific roles
| Channel | Catches |
|---------|---------|
| BM25 | Exact title/keyword matches |
| FAISS semantic | Paraphrase, similar descriptions |
| Neo4j graph | Skill synonyms, implicit requirements (Kubernetes → container orchestration) |

---

## Two-Stage Retrieval: ANN Recall → Cross-Encoder Rerank

| | Single-stage cross-encoder | Two-stage (FAISS → rerank) |
|---|---------------------------|---------------------------|
| **Scale** | O(jobs) per query — too slow | O(top-K) rerank only |
| **Quality** | Best possible | Near-best with 50× less compute |
| **Production pattern** | Rare at scale | CareerBuilder, JobMatchAI, industry standard |

**Decision:** Stage 1: FAISS + BM25 + Neo4j → top-400 fused → top-50 reranked by cross-encoder. Stage 2: utility function on top-50.

**Reference:** `papers/careerbuilder_two_stage.md`, `papers/two_tower_diffusion.md` (two-tower limitation motivates cross-interaction at rerank)

---

## Frontend: Streamlit over Next.js (MVP)

| | Streamlit | Next.js |
|---|-----------|---------|
| **Time to demo** | Hours | Days–weeks |
| **ML integration** | Native Python | Requires API layer |
| **Interactivity** | Sliders, uploads, tables | Full custom UI |
| **MVP goal** | Impressive interactive demo fast | Production UI |

**Decision:** Streamlit for MVP demo. SkillAlign and JobMatchAI use Next.js for production — migrate to Next.js or embed Streamlit in iframe for production if needed.

---

## Backend: FastAPI

| | FastAPI | Flask/Django |
|---|---------|--------------|
| **Async** | Native — parallel retrieval thread pool | Sync by default |
| **ML ecosystem** | Python-native | Same |
| **JobMatchAI / SkillAlign** | Both use FastAPI | — |
| **OpenAPI docs** | Auto-generated | Manual |

**Decision:** FastAPI for all backend endpoints. Async wrapper around parallel BM25 + FAISS + Neo4j retrieval.

**Reference:** `repos/repos_index.md`

---

## Encoder: all-MiniLM-L6-v2 (MVP default)

| | MiniLM-L6-v2 | BGE-large | API embeddings |
|---|-------------|-----------|----------------|
| **Dims** | 384 | 1024 | 1024–1536 |
| **Speed** | Fastest | 3× slower | Network latency |
| **JobMatchAI validated** | ✅ (NDCG@10 0.81) | Not in their paper | — |
| **Cost** | Free | Free | Per-token |

**Decision:** MiniLM for all MVP indexing. Evaluate BGE-large as a second semantic channel in Phase 2 if Recall@50 is below target.

**Reference:** `papers/jobmatchai.md`, `papers/resume2vec.md`, `models_info/models_comparison.md`

---

## ESCO Linking: Tabiya Classifier

| | Tabiya livelihoods classifier | Raw LLM prompting |
|---|------------------------------|-------------------|
| **Standardization** | Maps to ESCO URIs | Inconsistent labels |
| **Open source** | ✅ | — |
| **Paper validation** | arxiv:2512.03195 | — |
| **Graph integration** | Direct Neo4j node creation | Manual cleanup |

**Decision:** Tabiya for entity linking at ingestion. Sentence linking for occupation classification when needed.

**Reference:** `papers/esco_job_matching.md`

---

## Lexical Search: rank_bm25 over Elasticsearch (MVP)

| | rank_bm25 | Elasticsearch |
|---|-----------|---------------|
| **Infra** | Zero — pure Python | Separate service |
| **JobMatchAI** | — | Production choice |
| **MVP corpus** | < 200K jobs (Kaggle) | Overkill |

**Decision:** `rank_bm25` for MVP. Add Elasticsearch if corpus exceeds 500K or we need field-boosted multi-index search.

---

## LLM: Gemini Pro + Flash

| Use | Model |
|-----|-------|
| Resume → structured profile | Gemini 2.5 Pro (`LLM_MODEL_PRO`) |
| Match explanations | Gemini 2.5 Pro (`LLM_MODEL_PRO`) |
| Batch job field extraction (cost-sensitive) | Gemini 2.5 Flash (`LLM_MODEL_FLASH`) |

**Decision:** Pro for quality-critical paths; Flash for high-volume batch extraction only.

---

## Evaluation: JobSearch-XS + LinkedIn Kaggle

| Dataset | Role |
|---------|------|
| **JobSearch-XS** | Standardized NDCG@10 benchmark (compare to JobMatchAI 0.81) |
| **LinkedIn Kaggle** | Volume for index build + TF-IDF baseline comparison |
| **ESCO** | Graph correctness (skill expansion spot checks) |

---

## Decision Summary Table

| Component | Choice | Why (one line) |
|-----------|--------|----------------|
| ANN index | FAISS | Free, local, CareerBuilder-proven |
| Knowledge graph | Neo4j + ESCO | Skill expansion, industry standard |
| Scoring | Deterministic utility | Auditable, no hallucinated scores |
| Explanation | Gemini Pro (scores only) | Narrates, never ranks |
| Embeddings | Multi-vector MiniLM | Different fit dimensions |
| Retrieval | BM25 + semantic + graph | +7% NDCG (JobMatchAI) |
| Reranking | Cross-encoder → utility | Two-stage quality at scale |
| Frontend | Streamlit | Fast interactive demo |
| Backend | FastAPI | Async, ML-native |
| ESCO linking | Tabiya | Open-source, paper-validated |
| Lexical | rank_bm25 | Zero infra for MVP |

See `mvp_architecture.md` for the full five-stage pipeline diagram and endpoint spec.
