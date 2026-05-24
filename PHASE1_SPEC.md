# Phase 1 — Foundation & Data Layer

## Objective
By the end of Phase 1, the following must be true:
- Git repo initialized with full project structure
- Docker services running: PostgreSQL, Neo4j, Elasticsearch, Redis
- PostgreSQL has schema with all tables migrated
- PostgreSQL has 100,000+ job listings loaded (Kaggle + scraped)
- Neo4j has the complete ESCO taxonomy loaded (13,890 skills, 3,008 occupations, all relationships)
- Skill expansion queries working and tested (input "PyTorch" → returns "deep learning", "neural networks", "TensorFlow" with decay weights)
- Entity linker working (free text skill → best matching ESCO node)
- Elasticsearch has all jobs indexed for BM25 search
- BM25 retrieval returning results for test queries
- Every step committed individually with the specified message

---

## Step 1.1 — Project Initialization

### What to do
Create the git repository and full folder structure. This is the skeleton that everyone else builds on top of.

### Instructions

Initialize a new git repository at the project root.

Create the complete folder structure. Every directory listed in PROJECT.md must exist. Create empty `__init__.py` files in every Python package directory under `src/` and `tests/`. The structure must include `config/`, `data/esco/`, `data/sample_jobs/`, `scripts/`, `src/ingestion/`, `src/embeddings/`, `src/knowledge_graph/`, `src/matching/`, `src/feedback/`, `src/api/`, `src/api/routes/`, `src/api/schemas/`, `src/db/`, `src/db/migrations/`, `frontend/`, `frontend/pages/`, `frontend/components/`, and `tests/`.

Create `.gitignore` that excludes: `.env`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `data/esco/`, `data/sample_jobs/`, `*.faiss`, `*.idx`, `node_modules/`, `.venv/`, `venv/`, `dist/`, `build/`, `*.egg-info/`, `.DS_Store`, `Thumbs.db`, `logs/`, `*.log`.

Create `.env.example` with every environment variable the project needs. Include placeholders for: `DATABASE_URL` (PostgreSQL async connection string), `REDIS_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `ELASTICSEARCH_URL`, `GOOGLE_AI_API_KEY`, `LLM_MODEL_PRO` (default `gemini-2.5-pro`), `LLM_MODEL_FLASH` (default `gemini-2.5-flash`), `LLM_MAX_TOKENS`, `GITHUB_TOKEN`, `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`), `FAISS_INDEX_PATH`, `ESCO_DATA_PATH`, `API_HOST`, `API_PORT`, `FRONTEND_PORT`, `LOG_LEVEL`. Every value should be a descriptive placeholder, never a real credential.

Create `config/__init__.py` and `config/settings.py`. The settings module must use `pydantic-settings` BaseSettings class to load all configuration from environment variables. Group settings logically: DatabaseSettings, Neo4jSettings, ElasticsearchSettings, LLMSettings, EmbeddingSettings, GitHubSettings, AppSettings. Create a single `get_settings()` function that returns a cached settings instance. Every setting must have a type annotation and a sensible default where appropriate (except credentials which have no default). Add validators where needed (e.g., DATABASE_URL must start with `postgresql`).

Create `requirements.txt` with pinned versions for all dependencies: `fastapi`, `uvicorn[standard]`, `sqlalchemy[asyncio]`, `asyncpg`, `alembic`, `pydantic`, `pydantic-settings`, `neo4j`, `elasticsearch`, `faiss-cpu`, `sentence-transformers`, `google-genai`, `PyMuPDF`, `python-docx`, `httpx`, `streamlit`, `redis`, `pytest`, `pytest-asyncio`, `python-dotenv`, `python-multipart`, `aiofiles`, `tenacity` (for retry logic), `beautifulsoup4`, `lxml`.

Create `docker-compose.yml` with four services:

PostgreSQL service: Use `postgres:16` image. Map port 5432. Set `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` from environment. Volume for data persistence at `./docker-data/postgres`. Add healthcheck using `pg_isready`.

Neo4j service: Use `neo4j:5` image. Map ports 7474 (browser) and 7687 (bolt). Set `NEO4J_AUTH` from environment. Set `NEO4J_PLUGINS` to include APOC. Volume for data persistence at `./docker-data/neo4j`. Increase memory settings: `NEO4J_server_memory_heap_max__size=1G`. Add healthcheck using cypher-shell.

Elasticsearch service: Use `elasticsearch:8.12.0` image. Map port 9200. Set `discovery.type=single-node`, `xpack.security.enabled=false`, `ES_JAVA_OPTS=-Xms512m -Xmx512m`. Volume for data persistence at `./docker-data/elasticsearch`. Add healthcheck using curl to `localhost:9200`.

Redis service: Use `redis:7-alpine` image. Map port 6379. Volume for data persistence at `./docker-data/redis`. Add healthcheck using `redis-cli ping`.

All four services on the same Docker network named `personalmatch-network`. Add `docker-data/` to `.gitignore`.

Create a basic `README.md` with: project name (PersonalMatch), one-line description, prerequisites (Python 3.11+, Docker, Docker Compose), setup instructions (copy `.env.example` to `.env`, fill in values, run `docker-compose up -d`, install Python deps, run migrations), and project structure overview.

Verify: run `docker-compose up -d` and confirm all four services start and pass healthchecks.

**Commit:** `init: project structure, docker-compose, config management`

---

## Step 1.2 — Database Schema

### What to do
Define all database tables, set up SQLAlchemy ORM models, configure Alembic for migrations, and run the initial migration.

### Instructions

Create `src/db/database.py`. Set up an async SQLAlchemy engine using the `DATABASE_URL` from settings. Create an async session factory using `async_sessionmaker`. Write a `get_db()` async generator function that yields a session and handles cleanup. Write an `init_db()` function that creates all tables (used for initial setup).

Create `src/db/models.py` with the following SQLAlchemy ORM models:

**Job model:**
- `id` — UUID primary key, generated server-side
- `title` — String, not nullable, indexed
- `company` — String, not nullable, indexed
- `description` — Text, not nullable (full job description)
- `location` — String, nullable (city, state, country)
- `salary_min` — Integer, nullable
- `salary_max` — Integer, nullable
- `currency` — String, nullable, default "USD"
- `remote_type` — String, nullable (enum-like: "remote", "hybrid", "onsite")
- `sponsorship_available` — Boolean, nullable
- `company_size` — String, nullable (enum-like: "1-10", "11-50", "51-200", "201-1000", "1000+")
- `company_stage` — String, nullable (enum-like: "pre-seed", "seed", "series-a", "series-b", "growth", "enterprise")
- `industry` — String, nullable
- `experience_level` — String, nullable (enum-like: "entry", "mid", "senior", "lead", "executive")
- `skills_extracted` — JSONB, nullable (list of skills extracted from description)
- `source_url` — String, nullable (original listing URL)
- `source_platform` — String, nullable (where the job was scraped from)
- `posted_date` — DateTime, nullable
- `embedding_skill` — LargeBinary, nullable (384-dim float32 vector serialized)
- `embedding_domain` — LargeBinary, nullable
- `embedding_role` — LargeBinary, nullable
- `embedding_environment` — LargeBinary, nullable
- `is_embedded` — Boolean, default False
- `created_at` — DateTime, server default now
- `updated_at` — DateTime, server default now, on update now

Add indexes on: `title`, `company`, `location`, `remote_type`, `industry`, `experience_level`, `posted_date`, `is_embedded`. Add a composite index on `(company, title)` for deduplication checks.

**Candidate model:**
- `id` — UUID primary key
- `name` — String, nullable
- `email` — String, nullable, unique
- `resume_text` — Text, nullable (raw extracted text)
- `resume_filename` — String, nullable
- `github_username` — String, nullable
- `github_data` — JSONB, nullable (raw GitHub API response data)
- `profile` — JSONB, nullable (the structured candidate profile built by LLM)
- `preferences` — JSONB, nullable (explicit user preferences from onboarding)
- `embedding_skill` — LargeBinary, nullable
- `embedding_domain` — LargeBinary, nullable
- `embedding_role` — LargeBinary, nullable
- `embedding_environment` — LargeBinary, nullable
- `utility_weights` — JSONB, nullable (per-candidate reranker weights, adjusted by feedback)
- `created_at` — DateTime, server default now
- `updated_at` — DateTime, server default now, on update now

**Recommendation model:**
- `id` — UUID primary key
- `candidate_id` — UUID, foreign key to Candidate, not nullable, indexed
- `job_id` — UUID, foreign key to Job, not nullable, indexed
- `match_score` — Float, not nullable (final combined score 0-1)
- `factor_scores` — JSONB, not nullable (per-factor breakdown: skill_fit, experience_alignment, domain_relevance, role_shape_match, location_fit, company_stage_alignment, semantic_similarity)
- `retrieval_scores` — JSONB, nullable (per-source scores: bm25, vector, graph)
- `explanation` — Text, nullable (LLM-generated explanation)
- `rank` — Integer, nullable (position in the feed)
- `created_at` — DateTime, server default now

Add unique constraint on `(candidate_id, job_id)` to prevent duplicate recommendations. Add index on `(candidate_id, rank)` for feed queries.

**Feedback model:**
- `id` — UUID primary key
- `candidate_id` — UUID, foreign key to Candidate, not nullable, indexed
- `job_id` — UUID, foreign key to Job, not nullable, indexed
- `action` — String, not nullable (enum-like: "saved", "dismissed", "applied")
- `created_at` — DateTime, server default now

Add index on `(candidate_id, action)` for feedback aggregation queries. Add unique constraint on `(candidate_id, job_id, action)` to prevent duplicate feedback.

Set up Alembic: run `alembic init src/db/migrations`. Configure `alembic.ini` to read `DATABASE_URL` from environment (not hardcoded). Update `migrations/env.py` to import your models and use the async engine. Generate the initial migration with `alembic revision --autogenerate -m "initial schema"`. Run the migration with `alembic upgrade head`.

Create `src/api/schemas/job.py` with Pydantic models: `JobBase`, `JobCreate`, `JobResponse`, `JobListResponse` (paginated).

Create `src/api/schemas/candidate.py` with Pydantic models: `CandidatePreferences` (all the preference fields from the onboarding form), `CandidateProfile` (the structured profile output), `CandidateCreate`, `CandidateResponse`.

Create `src/api/schemas/recommendation.py` with Pydantic models: `RecommendationResponse` (includes job details, match score, factor breakdown, explanation), `RecommendationListResponse` (paginated feed).

Create `src/api/schemas/feedback.py` with Pydantic models: `FeedbackCreate` (candidate_id, job_id, action), `FeedbackResponse`.

Verify: run the migration, connect to PostgreSQL, confirm all four tables exist with correct columns and indexes.

**Commit:** `feat(db): PostgreSQL schema with jobs, candidates, recommendations, feedback tables`

---

## Step 1.3 — Job Data Scraper

### What to do
Build a configurable scraper that collects job listings from public career pages and stores them in PostgreSQL.

### Instructions

Create `scripts/scrape_jobs.py` as a standalone CLI script.

The scraper must be fully configurable. Create a YAML or JSON config file at `config/scraper_config.yaml` that defines: target base URLs, CSS selectors or XPath patterns for extracting title/company/description/location/salary/posted_date from listing pages, pagination patterns, rate limit delay (seconds between requests), maximum pages to scrape per source, user agent string, and output format.

The scraper logic:
- Read scraper config from the YAML file
- For each configured source:
  - Crawl listing pages following the pagination pattern
  - For each listing, extract: title, company, full description text, location, salary range (parse from text if not structured), remote/hybrid/onsite (infer from description keywords if not explicit), posted date, source URL
  - Clean and normalize extracted text: strip HTML tags, normalize whitespace, decode HTML entities
  - Rate limit: sleep for the configured delay between requests
  - Handle errors gracefully: log failed pages, continue with next listing
- Deduplicate: before inserting, check if a job with the same (company, title, source_url) already exists
- Bulk insert into PostgreSQL Job table
- Log statistics: total pages crawled, total listings found, duplicates skipped, errors encountered, time elapsed

Use `httpx` for HTTP requests (async capable). Use `beautifulsoup4` with `lxml` parser for HTML parsing. Use `tenacity` for retry on transient failures (HTTP 429, 500, 503).

Important: the scraper must work as a general-purpose tool. The config file defines what to scrape and how. No source-specific logic hardcoded in the scraper itself. Someone should be able to add a new source by adding an entry to the config file.

Add a `--dry-run` flag that fetches and parses but doesn't insert into the database (prints to stdout instead). Add a `--limit` flag to cap the number of listings per source. Add a `--source` flag to run only a specific configured source.

Verify: run the scraper against at least one configured source, confirm jobs appear in PostgreSQL with all fields populated.

**Commit:** `feat(scraper): configurable job listing scraper with rate limiting`

---

## Step 1.4 — Kaggle Dataset Loader

### What to do
Load the LinkedIn job postings Kaggle dataset (124K+ postings) into PostgreSQL, cleaned and normalized to match our schema.

### Instructions

Create `scripts/load_kaggle_jobs.py` as a standalone CLI script.

This script assumes the Kaggle dataset has been downloaded and extracted to a path specified via environment variable or CLI argument (default `data/sample_jobs/`). The dataset is a CSV with columns that may include: job title, company, description, location, salary, listed time, and various other fields depending on the dataset version.

The loader logic:
- Read the CSV file(s) using pandas
- Map Kaggle columns to our Job schema fields:
  - `title` ← job title column
  - `company` ← company name column
  - `description` ← job description column (this is the full text)
  - `location` ← location column, normalize to "City, State, Country" format where possible
  - `salary_min`, `salary_max` ← parse from salary column if available (handle formats like "$100k-$150k", "$100,000 - $150,000/yr", etc.)
  - `remote_type` ← infer from location or description ("Remote" in location → "remote", scan description for "hybrid" or "on-site" keywords)
  - `experience_level` ← infer from title keywords ("Senior" → "senior", "Lead" → "lead", "Junior"/"Entry" → "entry", "Staff"/"Principal" → "lead")
  - `industry` ← if available in dataset, otherwise leave null
  - `posted_date` ← parse from listed time column
  - `source_platform` ← set to "kaggle-linkedin"
- Clean descriptions: remove HTML tags, normalize unicode, strip excessive whitespace, truncate descriptions longer than 50,000 characters
- Drop rows with missing title or missing description (these are unusable)
- Deduplicate within the dataset: group by (company, title), keep the most recent posting
- Deduplicate against existing database entries: check for existing (company, title) pairs before inserting
- Bulk insert into PostgreSQL in batches of 1000 rows for efficiency
- Log statistics: total rows in CSV, rows after cleaning, rows after dedup, rows inserted, rows skipped (already exist), time elapsed

Add CLI arguments: `--file-path` (path to CSV), `--batch-size` (default 1000), `--dry-run` (print stats without inserting), `--limit` (max rows to process, useful for testing).

Verify: run the loader, confirm 100K+ jobs in PostgreSQL, spot-check 10 random records for data quality (title present, description non-empty, location normalized).

**Commit:** `feat(data): Kaggle LinkedIn job dataset loader with deduplication`

---

## Step 1.5 — ESCO Taxonomy Download & Neo4j Import

### What to do
Download the complete ESCO taxonomy from the EU portal and load all skills, occupations, and relationships into Neo4j. This is the foundation of the semantic skill matching layer.

### Instructions

**Download script:**

Create `scripts/download_esco.py` as a standalone CLI script.

The ESCO dataset is downloadable from `https://esco.ec.europa.eu/en/use-esco/download`. The script should:
- Download the CSV version of the ESCO classification (skills, occupations, skill-skill relationships, occupation-skill relationships)
- The download requires selecting: topic (all), format (CSV), languages (English at minimum)
- Save all CSV files to the path configured in `ESCO_DATA_PATH` environment variable (default `data/esco/`)
- If files already exist, skip download unless `--force` flag is set
- Validate downloaded files: check expected CSVs are present, check row counts are reasonable (skills CSV should have 13,000+ rows, occupations should have 3,000+ rows)
- Log: files downloaded, sizes, row counts

Note: the ESCO download may require manual steps from the web portal. If automated download isn't feasible, the script should check if the files exist and print clear instructions for manual download if they're missing.

The key ESCO CSV files you need:
- `skills_en.csv` — all skills with URI, preferred label, description, skill type (skill vs knowledge vs competence)
- `occupations_en.csv` — all occupations with URI, preferred label, description, ISCO group code
- `skillRelations.csv` or `broaderRelationsSkillPillar.csv` — skill-to-skill relationships (broader, narrower, related)
- `occupationSkillRelations.csv` — which skills each occupation requires (with relationship type: essential vs optional)

**Neo4j import script:**

Create `scripts/load_esco_neo4j.py` as a standalone CLI script.

Connect to Neo4j using the official Python driver with URI, user, and password from environment variables.

Step 1 — Clear existing ESCO data (optional, behind `--clean` flag):
- Delete all nodes with label Skill, Occupation, and their relationships
- Log: nodes deleted, relationships deleted

Step 2 — Create constraints and indexes:
- Uniqueness constraint on `Skill.uri`
- Uniqueness constraint on `Occupation.uri`
- Index on `Skill.label` for text search
- Index on `Skill.skill_type` for filtering
- Index on `Occupation.label` for text search
- Index on `Occupation.isco_group` for grouping

Step 3 — Load Skill nodes:
- Read skills CSV
- For each skill, create a node with properties: `uri`, `label` (preferred label), `description`, `skill_type` (skill/knowledge/competence), `alt_labels` (alternative labels as a list)
- Use UNWIND with batch Cypher for performance (batches of 500)
- Log: total skills loaded

Step 4 — Load Occupation nodes:
- Read occupations CSV
- For each occupation, create a node with properties: `uri`, `label`, `description`, `isco_group`, `alt_labels`
- Batch Cypher, batches of 500
- Log: total occupations loaded

Step 5 — Load skill-to-skill relationships:
- Read the skill relations CSV
- For each relationship, create an edge between two Skill nodes
- Relationship types to create:
  - `BROADER_THAN` — from narrower skill to broader skill
  - `NARROWER_THAN` — from broader to narrower
  - `RELATED_TO` — bidirectional relatedness (create in both directions or use undirected semantics)
- Match on skill URIs
- Batch Cypher, batches of 500
- Log: total relationships loaded, breakdown by type

Step 6 — Load occupation-skill relationships:
- Read the occupation-skill relations CSV
- For each relationship, create a `REQUIRES_SKILL` edge from Occupation to Skill
- Edge properties: `relationship_type` ("essential" or "optional")
- Batch Cypher
- Log: total occupation-skill links loaded

Step 7 — Validate:
- Run count queries: total Skill nodes, total Occupation nodes, total relationships by type
- Run sample queries:
  - Find skill "Python (programming language)" and its direct relationships
  - Find occupation "Data scientist" and its required skills
  - Find all skills within 2 hops of "machine learning"
- Print validation results
- Fail with clear error if counts are below expected minimums (skills < 10,000 → something went wrong)

Add CLI flags: `--clean` (delete existing before loading), `--validate-only` (skip loading, just run validation), `--batch-size` (default 500).

**Neo4j client:**

Create `src/knowledge_graph/neo4j_client.py`:
- Wrapper class around the Neo4j Python driver
- Constructor takes URI, user, password from settings
- `run_query(cypher, params) -> list[dict]` — execute a read query, return results as list of dicts
- `run_write(cypher, params) -> None` — execute a write query
- `run_batch_write(cypher, batch_params) -> None` — execute a parameterized query for a batch of data using UNWIND
- `health_check() -> bool` — verify connection is alive
- `close()` — close the driver
- Context manager support (`__aenter__`, `__aexit__`)
- Connection pooling (the Neo4j driver handles this natively, just configure pool size)
- Retry on transient failures (ServiceUnavailable, SessionExpired)
- Log connection events and query timing

Verify: run the import script, then open Neo4j Browser at `localhost:7474`, visually explore the graph. Run `MATCH (n:Skill) RETURN count(n)` and confirm 13,000+ nodes. Run `MATCH (n:Occupation) RETURN count(n)` and confirm 3,000+ nodes. Run `MATCH ()-[r]->() RETURN type(r), count(r)` and confirm multiple relationship types with thousands of edges each.

**Commit:** `feat(kg): ESCO taxonomy download and Neo4j import (13,890 skills, 3,008 occupations)`

---

## Step 1.6 — Skill Expansion & Entity Linking

### What to do
Build the skill expansion engine that traverses the ESCO graph to find semantically related skills, and the entity linker that maps free-text skill strings to ESCO nodes.

### Instructions

**Skill Expander:**

Create `src/knowledge_graph/skill_expander.py`:

`expand_skill(skill_uri: str, max_hops: int = 2) -> list[ExpandedSkill]`
- Takes an ESCO skill URI (already linked) and traverses the graph outward
- Returns a list of related skills with decay-weighted relevance scores
- Decay weights: 1-hop = 1.0, 2-hop = 0.5, 3-hop = 0.25
- Traverses these relationship types: BROADER_THAN, NARROWER_THAN, RELATED_TO
- BROADER_THAN traversal (going to more general skills) gets a slight penalty (multiply weight by 0.8) because broader skills are less specific matches
- NARROWER_THAN traversal (going to more specific skills) gets no penalty because specific skills imply general capability
- RELATED_TO traversal gets the standard decay
- Deduplicate: if a skill is reachable via multiple paths, keep the highest weight
- Return ExpandedSkill objects with: `uri`, `label`, `weight`, `hop_distance`, `path` (how it was reached)
- Cypher query pattern: variable-length path matching `(start)-[r*1..N]-(end)` with relationship type filtering

`expand_skills(skill_uris: list[str], max_hops: int = 2) -> dict[str, list[ExpandedSkill]]`
- Batch expansion for multiple skills
- Run expansions in parallel or in a single batched Cypher query for efficiency
- Merge results: if two input skills expand to the same skill, take the maximum weight
- Return a dict mapping each input skill URI to its expanded skills
- Also return a flat merged list of all unique expanded skills with their best weights

`get_skills_for_occupation(occupation_uri: str) -> list[OccupationSkill]`
- Given an occupation, return all required skills with their relationship type (essential/optional)
- Useful for expanding job matching: if a job title maps to an ESCO occupation, we get its full skill requirements

Caching: wrap expansion results in an in-memory LRU cache (use `functools.lru_cache` or a Redis cache). Skills and their relationships don't change at runtime, so cache aggressively. Cache key is `(skill_uri, max_hops)`. Cache size: 10,000 entries should cover the most common skills.

**Entity Linker:**

Create `src/knowledge_graph/entity_linker.py`:

This is the critical bridge between free-text skills (from resumes and job descriptions) and ESCO graph nodes. Without this, the knowledge graph is useless.

`link_skill(free_text: str) -> LinkedSkill | None`
- Takes a free-text skill string like "PyTorch", "machine learning", "React.js", "project management"
- Returns the best matching ESCO Skill node, or None if no good match exists
- Matching strategy (in priority order):
  1. **Exact match** — case-insensitive match against ESCO skill labels and alternative labels. "Python" → matches "Python (programming language)" label. This is the fastest path.
  2. **Fuzzy match** — if no exact match, use fuzzy string matching (Levenshtein distance or similar) against ESCO labels. Threshold: similarity > 0.85. "Pytorch" → "PyTorch". "k8s" → "Kubernetes". This catches typos and abbreviations.
  3. **Embedding match** — if no fuzzy match, encode the free-text skill using the sentence transformer model and find the nearest ESCO skill by cosine similarity. This requires pre-computing embeddings for all ESCO skill labels+descriptions and storing them for fast lookup. Threshold: cosine similarity > 0.75. "container orchestration tools" → "Kubernetes" or "Docker Swarm". This catches semantic paraphrases.
- Return LinkedSkill with: `esco_uri`, `esco_label`, `match_type` ("exact"/"fuzzy"/"semantic"), `confidence` (0-1)

`link_skills(free_texts: list[str]) -> list[LinkedSkill | None]`
- Batch linking for efficiency
- For embedding matching, batch encode all inputs at once

**Pre-computation for embedding matching:**
- Create a script or initialization function that:
  - Loads all ESCO skill labels and descriptions from Neo4j
  - Encodes each skill's "label: description" string using the sentence transformer model
  - Stores the resulting matrix (13,890 × 384) as a numpy array saved to disk
  - Also stores the mapping from matrix index to skill URI
  - At runtime, entity linker loads these from disk (fast) rather than recomputing
- This pre-computation should run once after ESCO is loaded into Neo4j
- Add a `scripts/precompute_esco_embeddings.py` script for this

**ESCO skill label index for fast exact/fuzzy matching:**
- At initialization, load all ESCO skill labels and alt_labels from Neo4j into an in-memory dictionary: `{lowercase_label: skill_uri}`
- This dict typically has ~50,000 entries (13,890 skills × ~3.5 labels each)
- For fuzzy matching, use a library like `rapidfuzz` for fast approximate string matching against all labels

**Tests:**

Create `tests/test_skill_expander.py`:
- Test that expanding "machine learning" returns skills like "deep learning", "supervised learning", "data science"
- Test that hop decay works: 1-hop results have weight 1.0, 2-hop have weight 0.5
- Test that BROADER_THAN traversal has the 0.8 penalty
- Test that batch expansion deduplicates properly
- Test that caching returns the same results on repeated calls
- Test with a skill that has no relationships: should return empty list

Create `tests/test_entity_linker.py`:
- Test exact match: "Python" → matches an ESCO Python node
- Test fuzzy match: "Pytorch" → matches "PyTorch"
- Test semantic match: "container orchestration" → matches something related to Kubernetes/Docker
- Test no match: "asdfghjkl" → returns None
- Test batch linking: multiple skills in one call

Note: tests need Neo4j running with ESCO loaded. Mark these as integration tests that require the Docker services. Use pytest markers: `@pytest.mark.integration`.

Verify: run the tests. Also manually test in a Python REPL:
- `link_skill("Python")` → returns a linked ESCO node
- `expand_skill(linked_python.esco_uri)` → returns related skills
- `link_skill("building ML pipelines")` → should semantically match to something relevant
- `expand_skills([python_uri, pytorch_uri])` → returns merged expansion

**Commit:** `feat(kg): skill expansion with hop-decay weighting and ESCO entity linking`

---

## Step 1.7 — Elasticsearch Setup & BM25 Retrieval

### What to do
Index all job listings in Elasticsearch for BM25 lexical search and build the retrieval module.

### Instructions

**Elasticsearch indexing script:**

Create `scripts/index_jobs_elasticsearch.py` as a standalone CLI script.

Connect to Elasticsearch using the URL from environment variables.

Step 1 — Create the index:
- Index name: read from config (default "jobs")
- Delete existing index if `--recreate` flag is set
- Define index mapping:
  - `title` — text field with standard analyzer, boost 2.0 (title matches are more important)
  - `description` — text field with standard analyzer
  - `company` — text field with keyword sub-field for exact matching
  - `location` — text field with keyword sub-field
  - `skills_extracted` — text field (for matching skill keywords)
  - `industry` — keyword field
  - `remote_type` — keyword field
  - `experience_level` — keyword field
  - `posted_date` — date field
  - `job_id` — keyword field (our PostgreSQL UUID, for joining back)
- Configure index settings: 1 shard (single node), 0 replicas (dev mode), custom analyzer if needed

Step 2 — Bulk index all jobs:
- Read all jobs from PostgreSQL
- For each job, create an Elasticsearch document with the mapped fields
- Extract skills from description if `skills_extracted` is null in the database:
  - Simple approach for now: look for common skill patterns in the text
  - Or set this field to the full description text (BM25 will handle the matching)
- Use Elasticsearch bulk API for performance
- Batch size: 500 documents per bulk request
- Progress bar showing indexing progress
- Log: total documents indexed, time elapsed, any errors

Step 3 — Verify:
- Run a test search query: "machine learning engineer"
- Print top 5 results with scores
- Run count query, confirm document count matches PostgreSQL job count

Add CLI flags: `--recreate` (delete and recreate index), `--batch-size` (default 500), `--limit` (index only N jobs for testing).

**BM25 Retriever:**

Create `src/matching/bm25_retriever.py`:

`BM25Retriever` class:
- Constructor: takes Elasticsearch client, index name from settings
- `retrieve(query_text: str, top_k: int = 500, filters: dict | None = None) -> list[ScoredJob]`
  - Build an Elasticsearch query:
    - `multi_match` query across `title`, `description`, `skills_extracted` fields
    - `title` field boosted (weight 2.0)
    - Use `best_fields` type for multi_match
  - If filters provided, add them as `filter` clauses in a `bool` query:
    - `remote_type` filter
    - `experience_level` filter
    - `location` filter (match query, not exact)
    - `posted_date` range filter
  - Return list of ScoredJob objects with: `job_id` (our UUID), `bm25_score` (Elasticsearch _score), `title`, `company`
  - Normalize scores to [0, 1] range: divide each score by the max score in the result set
- `health_check() -> bool` — verify Elasticsearch connection and index exists

Create a `ScoredJob` Pydantic model (can go in `src/api/schemas/recommendation.py` or a shared models file):
- `job_id: str`
- `score: float`
- `source: str` (which retrieval method produced this: "bm25", "vector", "graph")

Verify: instantiate the retriever, run `retrieve("senior data scientist")`, confirm results come back with scores. Run `retrieve("React frontend developer")`, confirm different results. Run `retrieve("Python machine learning", filters={"remote_type": "remote"})`, confirm filtering works.

**Commit:** `feat(search): Elasticsearch BM25 indexing and retrieval for job listings`

---

## Phase 1 Completion Checklist

Before moving to Phase 2, verify ALL of the following:

- [ ] Git repo has 7 commits, one for each step
- [ ] `docker-compose up -d` starts all 4 services successfully
- [ ] PostgreSQL has all 4 tables with correct schema (run `\dt` in psql)
- [ ] PostgreSQL has 100,000+ jobs loaded (run `SELECT count(*) FROM jobs`)
- [ ] Neo4j has 13,000+ Skill nodes (run `MATCH (n:Skill) RETURN count(n)`)
- [ ] Neo4j has 3,000+ Occupation nodes (run `MATCH (n:Occupation) RETURN count(n)`)
- [ ] Neo4j has skill relationships loaded (run `MATCH ()-[r]->() RETURN type(r), count(r)`)
- [ ] Skill expansion works: `expand_skill("machine learning URI")` returns related skills with decay weights
- [ ] Entity linking works: `link_skill("Python")` returns an ESCO node
- [ ] Entity linking semantic fallback works: `link_skill("container orchestration")` returns something relevant
- [ ] ESCO skill embeddings are pre-computed and saved to disk
- [ ] Elasticsearch index has all jobs (count matches PostgreSQL)
- [ ] BM25 retrieval returns reasonable results for test queries
- [ ] All tests pass: `pytest tests/test_skill_expander.py tests/test_entity_linker.py`
- [ ] No hardcoded values in any file — all config from environment variables
- [ ] `.env.example` is up to date with all required variables
