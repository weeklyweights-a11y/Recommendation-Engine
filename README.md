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

## Project structure

- `config/` — settings and logging
- `src/` — application code
- `scripts/` — CLI data pipelines
- `frontend/` — Streamlit UI (Phase 5)
- `resources/` — research notes and architecture references

See [PROJECT.md](PROJECT.md) and [PHASE1_SPEC.md](PHASE1_SPEC.md) for the full implementation plan.
