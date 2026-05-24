# PersonalMatch

AI-powered personalized job recommendation engine using hybrid retrieval (BM25 + semantic embeddings + ESCO knowledge graph) with explainable multi-factor scoring.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- 6GB+ RAM allocated to Docker Desktop (Neo4j + Elasticsearch)

## Setup

```bash
cp .env.example .env
# Edit .env — set NEO4J_PASSWORD and NEO4J_AUTH to match

docker compose up -d
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install -e .
```

## Phase 1 runbook

```bash
python scripts/seed_db.py

# Manual: download Kaggle dataset to data/sample_jobs/
python scripts/load_kaggle_jobs.py --file-path data/sample_jobs/job_postings.csv

python scripts/download_esco.py
python scripts/load_esco_neo4j.py --clean
python scripts/precompute_esco_embeddings.py

python scripts/scrape_jobs.py --source <name> --insert-db --limit 100

python scripts/index_jobs_elasticsearch.py --recreate

set RUN_INTEGRATION=1
pytest tests/test_skill_expander.py tests/test_entity_linker.py -m integration
```

## Phase 2 runbook

Prerequisites: Phase 1 complete (Docker, ESCO loaded, `precompute_esco_embeddings.py` run). Set `GOOGLE_AI_API_KEY` (or `GOOGLE_API_KEY`) and optionally `GITHUB_TOKEN` in `.env`.

```bash
# Unit tests (mocked Gemini + GitHub)
pytest tests/test_resume_parser.py tests/test_llm_extractor.py tests/test_github_fetcher.py tests/test_profile_builder.py tests/test_candidate_embedder.py -v

# Pipeline integration (mocked APIs)
pytest tests/test_phase2_pipeline.py -m integration -v

# Build profile locally (no DB write)
python scripts/build_candidate_profile.py --resume path/to/resume.pdf --github YOUR_USERNAME

# Build and persist to Postgres
python scripts/build_candidate_profile.py --resume path/to/resume.pdf --github YOUR_USERNAME --save
```

Optional full-stack linker test:

```bash
set RUN_INTEGRATION=1
pytest tests/test_entity_linker.py -m integration
```

## Phase 3 runbook

Prerequisites: Phase 1–2 complete, Docker running, jobs in PostgreSQL.

```bash
# Unit tests (mocked encoder / FAISS fixtures)
pytest tests/test_job_embedder.py tests/test_faiss_manager.py tests/test_vector_retriever.py tests/test_graph_retriever.py tests/test_hybrid_fuser.py tests/test_hybrid_pipeline.py -v

# Smoke embed (adjust --limit as needed)
python scripts/embed_jobs.py --limit 500

# Full embed + FAISS indexes (hours on CPU for ~97k jobs)
python scripts/embed_jobs.py

# Rebuild FAISS only from existing PG embeddings
python scripts/embed_jobs.py --rebuild-index

# Refresh Elasticsearch skills_extracted after embed
python scripts/index_jobs_elasticsearch.py --only-embedded

# Hybrid retrieval demo (candidate must exist with embeddings in DB)
python scripts/demo_hybrid_retrieval.py --email you@example.com

set RUN_INTEGRATION=1
pytest tests/test_vector_retriever.py tests/test_graph_retriever.py tests/test_hybrid_pipeline.py -m integration -v
```

Manual checks: `SELECT count(*) FROM jobs WHERE is_embedded = false` → 0; four `*_index.faiss` files under `FAISS_INDEX_PATH`; compare `FUSION_STRATEGY=rrf` vs `weighted_sum` on the same candidate.

## Phase 4 runbook

Prerequisites: Phase 1–3 complete, embedded jobs, FAISS indexes, Elasticsearch index, `GOOGLE_AI_API_KEY` for explanations.

```bash
# Unit tests (mocked LLM / pipeline stages)
pytest tests/test_hard_filters.py tests/test_reranker.py tests/test_explainer.py tests/test_matching_engine.py tests/test_api.py -v

# Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Full pipeline for a candidate UUID (first run may take minutes)
python scripts/demo_recommendations.py --candidate-id <uuid> --refresh

# Filter funnel only
python scripts/demo_hard_filters.py --candidate-id <uuid>

# Hybrid fuse + rerank comparison
python scripts/demo_rerank.py --candidate-id <uuid>
```

API base path: `/api/v1`. Key routes: `GET /health`, `POST /candidates` (resume upload), `GET /recommendations/{candidate_id}?refresh=1`, `PATCH /candidates/{id}/preferences`, `POST /feedback`.

Integration health check (requires Docker services):

```bash
set RUN_INTEGRATION=1
pytest tests/test_api.py -m integration -v
```

API smoke test (server must be running on port 8000):

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
python scripts/e2e_smoke.py --candidate-id <uuid>
```

## Phase 5 runbook

Prerequisites: Phase 1–4 complete, API running, `GOOGLE_AI_API_KEY` set.

```bash
docker compose up -d
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Separate terminal — set API URL for Streamlit
set API_BASE_URL=http://localhost:8000
streamlit run frontend/app.py
```

Open `http://localhost:8501`, complete onboarding (resume → optional GitHub → preferences → review → feed).

Manual checks:

- Save / dismiss / apply on feed cards; refresh feed after 5+ feedback actions to see ranking shift
- Profile: edit preferences, re-upload resume, view saved/dismissed jobs
- `pytest tests/test_weight_adjuster.py -v`

## Project structure

- `config/` — settings and logging
- `src/` — application code
- `scripts/` — CLI data pipelines
- `frontend/` — Streamlit UI (Phase 5)
- `resources/` — research notes and architecture references

See [PROJECT.md](PROJECT.md) and [PHASE1_SPEC.md](PHASE1_SPEC.md) for the full implementation plan.
