# PersonalMatch — AI-Powered Personalized Job Recommendation Engine

## Overview

A production-grade job recommendation engine that builds a deep understanding of candidates from multiple signals (resume, GitHub, explicit preferences) and delivers a personalized job feed ranked by genuine fit — not keywords. The system uses hybrid retrieval (BM25 + semantic embeddings + ESCO skill knowledge graph), multi-factor transparent scoring, and LLM-powered explainability to surface jobs candidates would never find through manual search.

## Problem

Job seekers spend hours manually searching and filtering across platforms. Every user gets the same results. Existing platforms match on keywords, missing that someone who "built anomaly detection at a manufacturing startup" could thrive at a fintech company doing fraud detection. Job discovery is broken because platforms don't understand *who you are* — they only understand *what you type*.

## Solution

Upload your resume, connect your GitHub, set your preferences. Get a personalized feed of jobs ranked by multidimensional fit with clear explanations for each match. No searching required. The jobs find you.

## Core Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION                        │
│  Resume (PDF/DOCX) + GitHub API + Preferences Form       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               CANDIDATE PROFILE BUILDER                  │
│  LLM Structured Extraction → Multi-Vector Embeddings     │
│  Structured Fields + Skill/Domain/Role/Env Embeddings    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  MATCHING ENGINE                         │
│  Phase A: Hard Filters (location, visa, salary, remote)  │
│  Phase B: Hybrid Retrieval (BM25 + FAISS + ESCO Graph)   │
│  Phase C: Multi-Factor Reranking + LLM Explanations      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 PERSONALIZED FEED                        │
│  Ranked jobs + match scores + "why this job" + feedback   │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend API | FastAPI | Async, fast, native Python ML ecosystem |
| Database | PostgreSQL | Structured candidate/job storage, mature, reliable |
| Knowledge Graph | Neo4j + ESCO Taxonomy | 13,890 skills, 3,008 occupations, semantic skill expansion |
| Vector Search | FAISS | Free, local, proven at million-scale (CareerBuilder) |
| Lexical Search | Elasticsearch | BM25 scoring, catches exact matches embeddings miss |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | Fast, 384 dims, proven in JobMatchAI |
| LLM | Claude API (via Anthropic SDK) | Structured extraction, preference inference, explanations |
| Resume Parsing | PyMuPDF + python-docx | PDF and DOCX text extraction |
| GitHub Data | GitHub REST API | Public repos, languages, activity |
| Frontend | Streamlit | Fast to build, interactive, good for data apps |
| Containerization | Docker + Docker Compose | Reproducible environment, easy setup |

## Project Structure

```
personalmatch/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── PROJECT.md
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # All config from env vars, zero hardcoding
│
├── data/
│   ├── esco/                        # ESCO taxonomy CSVs (gitignored, downloaded via script)
│   └── sample_jobs/                 # Sample job data for testing (gitignored)
│
├── scripts/
│   ├── download_esco.py             # Download ESCO taxonomy from EU portal
│   ├── load_esco_neo4j.py           # Cypher import: skills, occupations, relationships
│   ├── scrape_jobs.py               # Job listing scraper (configurable source)
│   ├── load_kaggle_jobs.py          # Load LinkedIn Kaggle dataset
│   ├── embed_jobs.py                # Generate embeddings for all jobs, build FAISS index
│   ├── index_jobs_elasticsearch.py  # Index jobs in Elasticsearch for BM25
│   └── seed_db.py                   # Initialize PostgreSQL schema and seed data
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── resume_parser.py         # PDF/DOCX → raw text extraction
│   │   ├── llm_extractor.py         # LLM structured extraction → candidate JSON
│   │   ├── github_fetcher.py        # GitHub API → repos, languages, activity
│   │   ├── portfolio_scraper.py     # URL → HTML → text extraction
│   │   └── profile_builder.py      # Merge all signals → unified candidate profile
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── encoder.py               # Sentence transformer encoding (multi-vector)
│   │   ├── job_embedder.py          # Batch embed job descriptions
│   │   └── candidate_embedder.py   # Generate candidate embedding vectors
│   │
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── neo4j_client.py          # Neo4j connection and query helpers
│   │   ├── skill_expander.py        # Multi-hop skill expansion with decay weights
│   │   └── entity_linker.py         # Map free-text skills → ESCO nodes
│   │
│   ├── matching/
│   │   ├── __init__.py
│   │   ├── hard_filters.py          # Non-negotiable constraint filtering
│   │   ├── bm25_retriever.py        # Elasticsearch BM25 retrieval
│   │   ├── vector_retriever.py      # FAISS approximate nearest neighbor search
│   │   ├── graph_retriever.py       # Knowledge graph-enhanced retrieval
│   │   ├── hybrid_fuser.py          # Fuse three retrieval signals into ranked list
│   │   ├── reranker.py              # Multi-factor utility scoring
│   │   └── explainer.py             # LLM-generated match explanations
│   │
│   ├── feedback/
│   │   ├── __init__.py
│   │   ├── tracker.py               # Track user actions (save, dismiss, apply)
│   │   └── weight_adjuster.py       # Adjust utility weights from feedback signals
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── candidates.py        # POST /candidates (upload resume, create profile)
│   │   │   ├── recommendations.py   # GET /recommendations/{candidate_id}
│   │   │   ├── feedback.py          # POST /feedback (save, dismiss, apply actions)
│   │   │   └── health.py            # GET /health
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── candidate.py         # Pydantic models for candidate profile
│   │   │   ├── job.py               # Pydantic models for job listings
│   │   │   ├── recommendation.py    # Pydantic models for recommendations
│   │   │   └── feedback.py          # Pydantic models for feedback
│   │   └── dependencies.py          # Shared dependencies (db sessions, clients)
│   │
│   └── db/
│       ├── __init__.py
│       ├── models.py                # SQLAlchemy ORM models
│       ├── database.py              # Database connection and session management
│       └── migrations/              # Alembic migrations
│
├── frontend/
│   ├── app.py                       # Streamlit main app
│   ├── pages/
│   │   ├── onboarding.py            # Resume upload + GitHub + preferences form
│   │   ├── feed.py                  # Personalized job feed with match cards
│   │   └── profile.py               # View/edit candidate profile
│   └── components/
│       ├── job_card.py              # Job recommendation card with score + explanation
│       ├── preference_form.py       # Preferences input form
│       └── feedback_buttons.py      # Save / Dismiss / Apply action buttons
│
└── tests/
    ├── __init__.py
    ├── test_resume_parser.py
    ├── test_llm_extractor.py
    ├── test_github_fetcher.py
    ├── test_skill_expander.py
    ├── test_matching_engine.py
    ├── test_reranker.py
    └── test_api.py
```

---

## Phase Breakdown

### Phase 1 — Foundation & Data Layer

**Goal:** Project skeleton, all infrastructure running, job data loaded, ESCO knowledge graph operational.

**Parallelizable across 3 people:**
- Person A: Project setup + PostgreSQL schema + Docker
- Person B: Job data ingestion (scraper + Kaggle loader)
- Person C: Neo4j + ESCO taxonomy + skill expansion

#### Step 1.1 — Project Initialization
- Initialize git repo
- Create project folder structure as defined above
- Set up `requirements.txt` with all dependencies
- Create `.env.example` with all required environment variables (API keys, DB URLs, Neo4j creds)
- Create `config/settings.py` that reads ALL config from environment variables — zero hardcoded values
- Create `docker-compose.yml` with services: PostgreSQL, Neo4j, Elasticsearch, Redis (for caching)
- Write `README.md` with setup instructions
- **Commit:** `init: project structure, docker-compose, config management`

#### Step 1.2 — Database Schema
- Create SQLAlchemy models in `src/db/models.py`:
  - `Job` — id, title, company, description, location, salary_min, salary_max, remote_type (remote/hybrid/onsite), sponsorship, company_size, company_stage, source_url, posted_date, embedding_vector (stored as binary), created_at
  - `Candidate` — id, name, email, resume_text, github_username, profile_json (JSONB for structured profile), preferences_json (JSONB), embedding_vectors (JSONB storing multiple named vectors), created_at, updated_at
  - `Recommendation` — id, candidate_id, job_id, match_score, factor_scores (JSONB for per-factor breakdown), explanation_text, created_at
  - `Feedback` — id, candidate_id, job_id, action (saved/dismissed/applied), created_at
- Set up Alembic for migrations
- Create `src/db/database.py` with async session management
- Run initial migration
- **Commit:** `feat(db): PostgreSQL schema with jobs, candidates, recommendations, feedback tables`

#### Step 1.3 — Job Data Scraper
- Build `scripts/scrape_jobs.py`:
  - Configurable scraper that collects job listings from public career pages
  - Extract: title, company, full description, location, salary range (if available), remote/hybrid/onsite, posted date, source URL
  - Rate limiting and polite scraping (configurable delay between requests)
  - Output as JSON lines for bulk loading
  - All target URLs and selectors loaded from config, not hardcoded
- **Commit:** `feat(scraper): configurable job listing scraper with rate limiting`

#### Step 1.4 — Kaggle Dataset Loader
- Build `scripts/load_kaggle_jobs.py`:
  - Load LinkedIn job postings dataset from Kaggle CSV
  - Clean and normalize fields to match our Job schema
  - Deduplicate by title + company
  - Bulk insert into PostgreSQL
  - Log stats: total loaded, duplicates skipped, errors
- **Commit:** `feat(data): Kaggle LinkedIn job dataset loader with deduplication`

#### Step 1.5 — ESCO Taxonomy Download & Neo4j Import
- Build `scripts/download_esco.py`:
  - Download full ESCO dataset from EU portal (CSV format)
  - Skills, competences, occupations, skill relationships, occupation-skill mappings
  - Save to `data/esco/` (gitignored)
- Build `scripts/load_esco_neo4j.py`:
  - Connect to Neo4j instance
  - Create constraints and indexes for Skill, Occupation nodes
  - Load all skills as Skill nodes with properties: uri, label, description, skill_type
  - Load all occupations as Occupation nodes with properties: uri, label, description, isco_group
  - Load RELATED_TO edges between skills (broader/narrower/related)
  - Load REQUIRES_SKILL edges from occupations to skills
  - Log stats: nodes created, relationships created
- Build `src/knowledge_graph/neo4j_client.py`:
  - Neo4j driver wrapper with connection pooling
  - Query helper methods
  - Health check
- **Commit:** `feat(kg): ESCO taxonomy download and Neo4j import (13,890 skills, 3,008 occupations)`

#### Step 1.6 — Skill Expansion Queries
- Build `src/knowledge_graph/skill_expander.py`:
  - `expand_skill(skill_label, max_hops=3)` — given a skill, traverse the ESCO graph and return related skills with decay weights
    - 1-hop: weight 1.0
    - 2-hop: weight 0.5
    - 3-hop: weight 0.25
  - `expand_skills(skill_list, max_hops=2)` — batch expansion for a list of skills, deduplicate and merge weights
  - Cypher queries that traverse RELATED_TO, broaderSkill, narrowerSkill edges
  - Cache results (skills don't change often)
- Build `src/knowledge_graph/entity_linker.py`:
  - `link_skill_to_esco(free_text_skill)` — map a free-text skill string to the best matching ESCO node
  - Use embedding similarity between the input skill text and ESCO skill labels/descriptions
  - Fallback: fuzzy string matching if embedding match is below threshold
  - `link_skills_batch(skill_list)` — batch linking
- Write tests: `tests/test_skill_expander.py`
  - Test: "PyTorch" expands to include "deep learning", "neural networks", etc.
  - Test: "Kubernetes" expands to include "container orchestration", "DevOps", etc.
  - Test: decay weights decrease with hops
  - Test: batch expansion deduplicates properly
- **Commit:** `feat(kg): skill expansion with hop-decay weighting and ESCO entity linking`

#### Step 1.7 — Elasticsearch Setup
- Add Elasticsearch to docker-compose.yml
- Build `scripts/index_jobs_elasticsearch.py`:
  - Create index with proper analyzers for job descriptions
  - Bulk index all jobs from PostgreSQL
  - Field mapping: title, description, company, location, skills (extracted)
- Build `src/matching/bm25_retriever.py`:
  - `retrieve(query_text, top_k=500)` — BM25 search against job index
  - Returns list of (job_id, bm25_score)
- **Commit:** `feat(search): Elasticsearch BM25 indexing and retrieval for job listings`

---

### Phase 2 — Candidate Profile Ingestion

**Goal:** A user can upload resume + GitHub + preferences and get back a rich structured candidate profile with multi-dimensional embeddings.

#### Step 2.1 — Resume Parser
- Build `src/ingestion/resume_parser.py`:
  - `parse_pdf(file_path) -> str` — extract text from PDF using PyMuPDF (fitz)
  - `parse_docx(file_path) -> str` — extract text from DOCX using python-docx
  - `parse_resume(file_path) -> str` — auto-detect format and extract
  - Handle edge cases: multi-column PDFs, tables, headers/footers
  - Clean extracted text: normalize whitespace, remove artifacts
- Write tests: `tests/test_resume_parser.py`
- **Commit:** `feat(ingestion): resume parser for PDF and DOCX with text cleaning`

#### Step 2.2 — LLM Structured Extractor
- Build `src/ingestion/llm_extractor.py`:
  - `extract_profile(resume_text, github_data=None) -> CandidateProfile`
  - Send resume text to Claude API with a structured extraction prompt
  - Prompt instructs the LLM to return JSON with:
    - `name`, `email`, `phone` (if present)
    - `skills` — list of {skill_name, proficiency: junior/mid/senior/expert, years_used, context}
    - `experience` — list of {company, title, start_date, end_date, description, domain, company_size_estimate, company_stage_estimate}
    - `education` — list of {institution, degree, field, graduation_year}
    - `domains` — list of industries/verticals they've worked in
    - `role_archetype` — inferred: founding_builder / platform_engineer / research_scientist / full_stack / manager / specialist
    - `career_trajectory` — inferred: ascending / lateral / career_change / early_career
    - `inferred_preferences` — inferred from career history: {preferred_company_stage, preferred_team_size, preferred_work_style}
  - If GitHub data is provided, merge signals (skill depth from commit frequency, project complexity)
  - Validate LLM output against Pydantic schema, retry on parse failure (max 2 retries)
- Write tests: `tests/test_llm_extractor.py`
- **Commit:** `feat(ingestion): LLM-powered structured profile extraction with inference`

#### Step 2.3 — GitHub Fetcher
- Build `src/ingestion/github_fetcher.py`:
  - `fetch_github_profile(username) -> GitHubProfile`
  - Hit GitHub REST API endpoints:
    - `/users/{username}` — bio, public repos count, followers
    - `/users/{username}/repos?sort=updated&per_page=100` — repos with languages, stars, forks, updated_at
    - `/repos/{owner}/{repo}/languages` — language breakdown per repo
    - `/repos/{owner}/{repo}/readme` — README content (base64 decode)
  - Aggregate into:
    - `languages` — dict of {language: total_bytes} across all repos
    - `repos` — list of {name, description, stars, forks, primary_language, languages, last_commit, has_tests, has_docker, has_ci, readme_summary}
    - `activity_score` — based on recency and frequency of commits
    - `project_complexity` — inferred from repo size, language diversity, presence of infra files
  - Respect GitHub rate limits (5000 req/hr for authenticated, 60 for unauthenticated)
  - Use personal access token from env vars
- Write tests: `tests/test_github_fetcher.py`
- **Commit:** `feat(ingestion): GitHub profile fetcher with repo analysis and activity scoring`

#### Step 2.4 — Profile Builder (Merge All Signals)
- Build `src/ingestion/profile_builder.py`:
  - `build_profile(resume_file, github_username=None, preferences=None) -> CandidateProfile`
  - Orchestrates the full pipeline:
    1. Parse resume → raw text
    2. Fetch GitHub data (if username provided)
    3. Send to LLM extractor with both resume text and GitHub data
    4. Merge explicit preferences from onboarding form
    5. Cross-validate: if LLM inferred "prefers startups" but user explicitly set "enterprise", user preference wins
    6. Compute skill depth scores: skill mentioned in resume (0.3) + found in GitHub repos (0.3) + recent GitHub activity (0.2) + endorsed on LinkedIn PDF (0.2)
    7. Link all extracted skills to ESCO nodes via entity linker
    8. Return unified CandidateProfile object
  - CandidateProfile Pydantic model with all fields defined in schemas
- **Commit:** `feat(ingestion): unified profile builder merging resume + GitHub + preferences`

#### Step 2.5 — Candidate Embeddings
- Build `src/embeddings/candidate_embedder.py`:
  - `embed_candidate(profile: CandidateProfile) -> CandidateEmbeddings`
  - Generate four embedding vectors using sentence-transformers:
    - `skill_embedding` — encode concatenated skill descriptions and contexts
    - `domain_embedding` — encode domain/industry experience descriptions
    - `role_embedding` — encode role descriptions, responsibilities, career narrative
    - `environment_embedding` — encode company descriptions, team sizes, work style signals
  - Each vector is 384-dim (all-MiniLM-L6-v2)
  - Store all four vectors in the candidate record
- Build `src/embeddings/encoder.py`:
  - Wrapper around sentence-transformers model
  - `encode(text) -> np.ndarray`
  - `encode_batch(texts) -> np.ndarray`
  - Model loaded once, reused across calls
- **Commit:** `feat(embeddings): multi-vector candidate embedding generation (skill, domain, role, environment)`

---

### Phase 3 — Embedding Pipeline & Vector Index

**Goal:** All jobs embedded, FAISS index built, hybrid retrieval operational.

#### Step 3.1 — Job Embedding Pipeline
- Build `src/embeddings/job_embedder.py`:
  - `embed_job(job: Job) -> JobEmbeddings`
  - Generate same four embedding dimensions for jobs:
    - `skill_embedding` — from required skills and technologies in description
    - `domain_embedding` — from industry/vertical context
    - `role_embedding` — from role title, responsibilities, level
    - `environment_embedding` — from company description, size, stage, culture signals
  - Use LLM to first extract structured fields from raw job description (skills required, domain, company stage, etc.) then embed the structured text
- Build `scripts/embed_jobs.py`:
  - Batch process all jobs in PostgreSQL
  - Generate embeddings in batches of 64
  - Store embeddings back in PostgreSQL (binary column)
  - Build FAISS index for each embedding dimension
  - Save FAISS indexes to disk
  - Progress bar, resume from last processed job on restart
  - Log stats: total embedded, time per batch, index size
- **Commit:** `feat(embeddings): batch job embedding pipeline with four FAISS indexes`

#### Step 3.2 — FAISS Vector Retriever
- Build `src/matching/vector_retriever.py`:
  - `retrieve(candidate_embeddings, top_k=500, dimension_weights=None) -> List[ScoredJob]`
  - Load four FAISS indexes (one per dimension)
  - Search each index with the corresponding candidate embedding
  - Fuse results with configurable dimension weights:
    - Default: skill=0.35, domain=0.25, role=0.25, environment=0.15
    - Weights adjustable per candidate based on preferences and feedback
  - Return deduplicated ranked list of (job_id, vector_score)
  - Support for filtering by job IDs (to apply after hard filters)
- **Commit:** `feat(matching): FAISS multi-vector retrieval with configurable dimension weights`

#### Step 3.3 — Knowledge Graph Retriever
- Build `src/matching/graph_retriever.py`:
  - `retrieve(candidate_skills, top_k=500) -> List[ScoredJob]`
  - For each candidate skill:
    - Link to ESCO node via entity linker
    - Expand skill through graph (2 hops with decay)
    - Find jobs that require any of the expanded skills
    - Score by: direct match (1.0) > 1-hop match (0.5) > 2-hop match (0.25)
  - Aggregate scores per job across all candidate skills
  - Return ranked list of (job_id, graph_score)
  - Cypher query: match candidate skills → expanded skills → jobs requiring those skills
- **Commit:** `feat(matching): knowledge graph retriever with ESCO skill expansion`

#### Step 3.4 — Hybrid Fusion
- Build `src/matching/hybrid_fuser.py`:
  - `fuse(bm25_results, vector_results, graph_results, weights=None) -> List[ScoredJob]`
  - Normalize scores from each retrieval source to [0, 1] range
  - Default fusion weights: bm25=0.25, vector=0.45, graph=0.30
  - Reciprocal Rank Fusion (RRF) as alternative fusion strategy
  - Return top_k fused results sorted by combined score
  - Support configurable weights and fusion strategy
- **Commit:** `feat(matching): hybrid retrieval fusion (BM25 + vector + graph) with RRF`

---

### Phase 4 — Matching Engine & Explainability

**Goal:** Full matching pipeline operational — hard filters → hybrid retrieval → reranking → explanations.

#### Step 4.1 — Hard Filters
- Build `src/matching/hard_filters.py`:
  - `filter_jobs(candidate_preferences, job_queryset) -> filtered_job_ids`
  - Filter on non-negotiable constraints:
    - Location: if candidate set specific locations, filter to those + remote jobs
    - Remote type: if candidate requires remote, exclude onsite-only
    - Visa/sponsorship: if candidate needs sponsorship, exclude jobs that don't offer it
    - Salary: if candidate set minimum, exclude jobs with max below that
    - Company size: if candidate set preference, filter accordingly
    - Excluded industries: remove jobs in industries candidate wants to avoid
  - Each filter is optional — only applied if candidate set the preference
  - Return set of job IDs that pass all filters
  - Log: total jobs, jobs after each filter, final count
- **Commit:** `feat(matching): hard constraint filters (location, visa, salary, remote, company size)`

#### Step 4.2 — Multi-Factor Reranker
- Build `src/matching/reranker.py`:
  - `rerank(candidate_profile, scored_jobs, top_k=50) -> List[RankedJob]`
  - Takes the hybrid retrieval results and computes a multi-factor utility score:
    - `skill_fit` (0-1): weighted overlap between candidate skills and job requirements, boosted by graph expansion
    - `experience_alignment` (0-1): years of experience vs job requirements, level match
    - `domain_relevance` (0-1): same industry or transferable domain based on graph distance
    - `role_shape_match` (0-1): founding→founding, IC→IC, manager→manager alignment
    - `location_fit` (0-1): exact location match, same city, same state, remote compatibility
    - `company_stage_alignment` (0-1): candidate history/preference vs company stage
    - `semantic_similarity` (0-1): the raw embedding similarity score from vector retrieval
  - Utility function: `final_score = Σ(weight_i × factor_i)` with default weights configurable
  - Default weights: skill_fit=0.25, experience=0.15, domain=0.15, role_shape=0.15, location=0.10, company_stage=0.10, semantic=0.10
  - Freshness boost: jobs posted in last 48h get +0.05, last week +0.03
  - Diversity injection: ensure top 20 results include at least 3 different industries and 3 different company stages
  - Return top_k with per-factor score breakdown
- **Commit:** `feat(matching): multi-factor reranker with adjustable weights and diversity injection`

#### Step 4.3 — LLM Explainer
- Build `src/matching/explainer.py`:
  - `explain_match(candidate_profile, job, factor_scores) -> str`
  - Takes the pre-computed factor scores (NOT the raw texts) and generates a natural language explanation
  - The LLM receives:
    - Candidate summary (skills, experience, preferences)
    - Job summary (title, company, requirements)
    - Factor score breakdown with labels
    - Knowledge graph evidence (which skills matched directly, which matched via expansion)
  - The LLM explains WHY the scores are what they are — it does not compute its own scores
  - Strict separation: scoring is deterministic, explanation is generative
  - Output format: 2-3 sentence explanation + bullet list of top match reasons + any gaps/stretches
  - `explain_batch(candidate_profile, ranked_jobs, top_k=20) -> List[str]`
  - Batch explanations for efficiency (can batch multiple jobs in one LLM call)
- **Commit:** `feat(matching): LLM-powered match explanations with separated scoring/explanation layers`

#### Step 4.4 — Recommendation Pipeline Orchestrator
- Build orchestration in `src/api/routes/recommendations.py`:
  - `GET /recommendations/{candidate_id}?page=1&per_page=20`
  - Full pipeline:
    1. Load candidate profile from DB
    2. Apply hard filters → get eligible job IDs
    3. Run hybrid retrieval (BM25 + vector + graph) on filtered jobs
    4. Rerank with multi-factor utility function
    5. Generate explanations for top results
    6. Store recommendations in DB
    7. Return paginated results with scores and explanations
  - Cache recommendations (invalidate on preference change or new feedback)
  - Log: pipeline timing per stage
- **Commit:** `feat(api): recommendation pipeline orchestrator with full matching flow`

---

### Phase 5 — Frontend & Feedback Loop

**Goal:** Complete Streamlit app with onboarding, personalized feed, and feedback actions.

#### Step 5.1 — Onboarding Page
- Build `frontend/pages/onboarding.py`:
  - Step 1: Resume upload (drag-and-drop, PDF/DOCX)
  - Step 2: GitHub username input (optional, with preview of repos found)
  - Step 3: Preferences form:
    - Job type: Full-time / Contract / Part-time (multi-select)
    - Work model: Remote / Hybrid / Onsite (multi-select)
    - Locations: text input with autocomplete
    - Visa sponsorship needed: Yes / No
    - Salary range: dual slider (min/max)
    - Company stage: Pre-seed / Seed / Series A / Series B / Growth / Enterprise (multi-select)
    - Company size: 1-10 / 10-50 / 50-200 / 200-1000 / 1000+ (multi-select)
    - Target roles: text input, comma separated
    - Industries to target: multi-select from common list
    - Industries to avoid: multi-select
    - What matters most: rank ordering of (compensation, equity, learning, mission, team, title, work-life balance)
  - Step 4: "Build My Profile" button → calls API, shows profile summary
  - Step 5: Review inferred preferences — "We think you prefer early-stage companies. Is that right?" → confirm or edit
  - Step 6: "Show My Jobs" → redirect to feed
- Build `frontend/components/preference_form.py` — reusable form component
- **Commit:** `feat(frontend): onboarding flow with resume upload, GitHub, and preferences`

#### Step 5.2 — Personalized Feed Page
- Build `frontend/pages/feed.py`:
  - Call `GET /recommendations/{candidate_id}`
  - Display job cards in a scrollable feed
  - Each card shows:
    - Job title + company name
    - Location + remote/hybrid/onsite badge
    - Salary range (if available)
    - Match percentage (circular progress or bar)
    - Top 3 match reasons (from explanation)
    - Skills that matched (highlighted) and skills that are gaps
    - Posted date + freshness indicator
    - Source link to original listing
  - Two sections: "Strong matches" and "Worth exploring"
  - Filter/sort options: by match score, by date, by salary, by company stage
- Build `frontend/components/job_card.py` — the recommendation card component
- **Commit:** `feat(frontend): personalized job feed with match cards and explanations`

#### Step 5.3 — Feedback Actions
- Build `frontend/components/feedback_buttons.py`:
  - Three buttons per job card: Save (bookmark icon), Dismiss (X icon), Apply (arrow icon)
  - On action → call `POST /feedback` API
  - Visual feedback: saved cards get bookmark highlight, dismissed cards fade and collapse
  - Apply opens the source URL in new tab and records the action
- Build `src/api/routes/feedback.py`:
  - `POST /feedback` — record action (candidate_id, job_id, action type)
  - Store in Feedback table with timestamp
- **Commit:** `feat(frontend): feedback actions (save, dismiss, apply) with API integration`

#### Step 5.4 — Feedback Loop / Weight Adjustment
- Build `src/feedback/tracker.py`:
  - `get_feedback_summary(candidate_id) -> FeedbackSummary`
  - Aggregate all feedback for a candidate:
    - Saved jobs: extract common patterns (skills, domains, company stages, role types)
    - Dismissed jobs: extract patterns of what they don't want
    - Applied jobs: strongest signal patterns
- Build `src/feedback/weight_adjuster.py`:
  - `adjust_weights(candidate_id, current_weights) -> adjusted_weights`
  - Analyze feedback patterns and adjust utility function weights:
    - If candidate consistently saves founding roles → increase role_shape_match weight
    - If candidate dismisses all enterprise jobs → increase company_stage_alignment weight
    - If candidate saves across diverse domains → decrease domain_relevance weight
  - Simple heuristic-based adjustment for MVP (not ML-based)
  - Weights bounded: no factor goes below 0.05 or above 0.40
- Integrate into recommendation pipeline: load adjusted weights before reranking
- **Commit:** `feat(feedback): weight adjustment from user feedback patterns`

#### Step 5.5 — Profile Page
- Build `frontend/pages/profile.py`:
  - Display candidate profile summary:
    - Extracted skills with depth scores
    - Experience timeline
    - GitHub stats overview
    - Current preferences (editable)
    - Inferred traits (role archetype, career trajectory)
  - "Update Preferences" → re-triggers recommendation generation
  - "Re-upload Resume" → re-processes profile
- **Commit:** `feat(frontend): candidate profile view with editable preferences`

---

### Phase 6 — Integration, Testing & Polish

**Goal:** Everything works end-to-end, edge cases handled, demo-ready.

#### Step 6.1 — End-to-End Integration Tests
- Write `tests/test_api.py`:
  - Test full pipeline: upload resume → create profile → get recommendations → submit feedback → get updated recommendations
  - Test with multiple candidate types: senior ML engineer, junior dev, career changer
  - Test edge cases: resume with no skills section, GitHub with no public repos, all preferences set to "any"
- **Commit:** `test: end-to-end integration tests for full recommendation pipeline`

#### Step 6.2 — Performance Optimization
- Profile the recommendation pipeline, identify bottlenecks
- Add caching:
  - Cache ESCO skill expansions (Redis, TTL 24h)
  - Cache job embeddings (loaded once at startup)
  - Cache recommendations per candidate (invalidate on feedback or preference change)
- Optimize FAISS search: use IVF index for faster search on large datasets
- Batch LLM calls for explanations (multiple jobs per API call)
- Target: full recommendation pipeline under 5 seconds for 10,000 jobs
- **Commit:** `perf: caching, FAISS optimization, batched LLM calls`

#### Step 6.3 — Error Handling & Edge Cases
- Add graceful error handling throughout:
  - LLM extraction fails → retry with simplified prompt, fallback to rule-based extraction
  - GitHub API rate limited → queue and retry, return partial profile
  - Neo4j connection fails → fall back to embedding-only matching (skip graph retrieval)
  - FAISS index not found → rebuild from PostgreSQL embeddings
  - Resume parsing fails → return clear error message to user
- Add logging throughout the pipeline with structured logs
- Add input validation on all API endpoints
- **Commit:** `fix: comprehensive error handling, fallbacks, and input validation`

#### Step 6.4 — Docker & Deployment
- Finalize `docker-compose.yml` with all services properly configured
- Create `Dockerfile` for the FastAPI backend
- Create `Dockerfile.frontend` for the Streamlit app
- Write `scripts/setup.sh` — one-command setup:
  1. Start Docker services
  2. Run migrations
  3. Download ESCO (if not present)
  4. Load ESCO into Neo4j
  5. Load sample job data
  6. Embed jobs and build indexes
  7. Start API and frontend
- Test clean setup from zero on a fresh machine
- **Commit:** `ops: Docker setup, one-command deployment script`

#### Step 6.5 — Demo Polish
- Create three sample resumes for demo (ML engineer, full-stack dev, career changer) in `data/sample_resumes/`
- Pre-compute recommendations for demo candidates so the demo loads instantly
- Add loading animations and progress indicators in Streamlit
- Write demo script in `DEMO.md`:
  - Open app → upload resume → enter GitHub → fill preferences → see feed → interact → show feed improving
  - Highlight: knowledge graph expansion, multi-factor scoring, explainability
- Final pass on UI: consistent styling, clear labels, mobile-responsive
- **Commit:** `docs: demo script, sample data, UI polish`

---

## Environment Variables

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/personalmatch
REDIS_URL=redis://localhost:6379/0

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200

# LLM
ANTHROPIC_API_KEY=
LLM_MODEL=claude-sonnet-4-20250514
LLM_MAX_TOKENS=4096

# GitHub
GITHUB_TOKEN=

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
FAISS_INDEX_PATH=./data/faiss_indexes/

# App
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PORT=8501
LOG_LEVEL=INFO
```

## Key Design Principles

1. **Zero hardcoding** — every URL, credential, threshold, and weight comes from config/env vars
2. **Separated concerns** — scoring is deterministic, explanation is generative, never mix them
3. **Multi-signal > single signal** — always prefer fusing multiple weak signals over one strong one
4. **Transparent scoring** — every recommendation has a per-factor score breakdown, not a black box number
5. **Graceful degradation** — if Neo4j is down, fall back to embedding-only; if LLM fails, show scores without explanations
6. **Git discipline** — commit after every meaningful step, clear commit messages
