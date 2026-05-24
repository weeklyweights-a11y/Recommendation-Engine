# Phase 2 — Candidate Profile Ingestion

## Objective
By the end of Phase 2, the following must be true:
- A user can upload a PDF or DOCX resume and get clean extracted text
- The LLM parses the resume into a structured candidate profile with skills, experience, domains, role archetype, and inferred preferences
- GitHub username input returns structured repo analysis with language distribution, project complexity, activity scores, and production-readiness signals
- All signals merge into a unified CandidateProfile object with cross-validated data
- Multi-dimensional embeddings (skill, domain, role, environment) are generated for the candidate
- All extracted skills are linked to ESCO nodes via the entity linker from Phase 1
- The full pipeline is tested end-to-end: upload resume → get structured profile with embeddings
- Every step committed individually

## Prerequisites
- Phase 1 complete: Docker services running, PostgreSQL schema, Neo4j with ESCO loaded, entity linker working
- Google AI API key (`GOOGLE_AI_API_KEY`) in `.env`
- GitHub personal access token in `.env`
- Sentence-transformers model downloadable (internet access needed on first run)

---

## Step 2.1 — Resume Parser

### What to do
Build a module that takes a PDF or DOCX file and returns clean, normalized plain text. This is pure text extraction — no intelligence yet.

### Instructions

Create `src/ingestion/resume_parser.py`:

**PDF parsing — `parse_pdf(file_path: str) -> str`:**
- Use PyMuPDF (imported as `fitz`) to open the PDF
- Iterate through all pages
- For each page, extract text using `page.get_text("text")`
- Handle multi-column layouts: PyMuPDF extracts text in reading order by default, but some resumes use two-column layouts where the extraction order gets jumbled. Use `page.get_text("blocks")` to get text blocks with their bounding box coordinates. Sort blocks by: first by vertical position (top to bottom), then by horizontal position (left to right). This produces a more natural reading order for multi-column resumes.
- Handle tables: if the resume has tables (common for skills sections), `page.get_text("text")` may lose structure. Use `page.find_tables()` if available in your PyMuPDF version to extract table content separately and format as "Column1: Value1, Column2: Value2" inline text.
- Concatenate all pages with double newline between them
- Return the full text string

**DOCX parsing — `parse_docx(file_path: str) -> str`:**
- Use `python-docx` to open the document
- Extract text from all paragraphs: `[para.text for para in doc.paragraphs]`
- Also extract text from tables: iterate through `doc.tables`, for each table iterate through rows and cells, extract cell text
- Format table content as "Column1: Value1 | Column2: Value2" to preserve structure
- Concatenate paragraphs and table content
- Return the full text string

**Unified parser — `parse_resume(file_path: str) -> str`:**
- Detect file type from extension (`.pdf` or `.docx`/`.doc`)
- Call the appropriate parser
- Run text cleaning on the result:
  - Normalize unicode characters (NFKD normalization)
  - Replace multiple consecutive whitespace/newlines with single space or double newline (preserve paragraph breaks)
  - Remove null bytes and control characters
  - Strip leading/trailing whitespace
  - Remove common PDF artifacts: header/footer repetitions, page numbers if they appear on every page
  - Decode HTML entities if any are present (&amp; → &, etc.)
- Validate: if extracted text is less than 50 characters, raise a clear error ("Could not extract meaningful text from resume")
- If extracted text is longer than 100,000 characters, truncate with a warning log
- Return cleaned text

**File validation — `validate_resume_file(file_path: str) -> None`:**
- Check file exists
- Check file extension is `.pdf`, `.docx`, or `.doc`
- Check file size is under 10MB (configurable from settings)
- Check file is not empty (size > 0)
- Raise specific exceptions: `FileNotFoundError`, `UnsupportedFileTypeError`, `FileTooLargeError`

Create custom exception classes in `src/ingestion/exceptions.py`:
- `ResumeParsingError` — base exception
- `UnsupportedFileTypeError(ResumeParsingError)`
- `FileTooLargeError(ResumeParsingError)`
- `ExtractionFailedError(ResumeParsingError)`

**Tests:**

Create `tests/test_resume_parser.py`:
- Create a small test PDF and test DOCX file in `tests/fixtures/` directory (or generate them programmatically in the test using reportlab for PDF and python-docx for DOCX)
- Test PDF parsing: create a simple PDF with known text, parse it, verify extracted text contains the expected content
- Test DOCX parsing: create a simple DOCX with known text, parse it, verify
- Test DOCX with tables: create a DOCX with a skills table, verify table content is extracted
- Test text cleaning: pass text with excessive whitespace, unicode artifacts, HTML entities — verify cleaning works
- Test file validation: test with wrong extension, oversized file, empty file — verify correct exceptions
- Test unsupported format: pass a `.txt` file — verify UnsupportedFileTypeError

Verify: take a real resume PDF (your own or a sample), run `parse_resume("path/to/resume.pdf")`, inspect the output text for quality. Check that skills sections, job titles, company names, and dates are all present and readable.

**Commit:** `feat(ingestion): resume parser for PDF and DOCX with text cleaning`

---

## Step 2.2 — LLM Structured Extractor

### What to do
Build the module that sends resume text (and optionally GitHub data) to the Google Gemini API and gets back a structured JSON profile. This is where raw text becomes intelligence.

### Instructions

Create `src/ingestion/llm_extractor.py`:

**Prompt design:**

Define the extraction prompt as a constant string in the module (or in a separate `src/ingestion/prompts.py` file if it's long). The prompt must instruct the LLM to:

System prompt: "You are an expert resume analyst and career profiler. Your job is to extract structured information from resume text and infer career patterns. You must respond ONLY with valid JSON matching the specified schema. No markdown, no preamble, no explanation — just the JSON object."

User prompt template: Include the resume text and (if available) a summary of GitHub data. Instruct the LLM to extract and return a JSON object with these exact fields:

```
{
  "name": "string or null",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null — city, state/country",

  "skills": [
    {
      "name": "string — the skill name as written",
      "category": "string — one of: programming_language, framework, tool, platform, methodology, domain_knowledge, soft_skill, other",
      "proficiency": "string — one of: beginner, intermediate, advanced, expert — inferred from context",
      "years_used": "number or null — estimated years if inferable",
      "context": "string — brief note on how/where this skill was used"
    }
  ],

  "experience": [
    {
      "company": "string",
      "title": "string",
      "start_date": "string — YYYY-MM or YYYY format",
      "end_date": "string or null — YYYY-MM, YYYY, or 'present'",
      "duration_months": "number — estimated total months",
      "description": "string — what they did in this role",
      "domain": "string — industry/vertical: healthcare, fintech, manufacturing, saas, ecommerce, etc.",
      "company_size_estimate": "string — one of: 1-10, 11-50, 51-200, 201-1000, 1000+ — infer from context",
      "company_stage_estimate": "string — one of: pre-seed, seed, series-a, series-b, growth, enterprise — infer from context",
      "role_type": "string — one of: ic, tech-lead, manager, founding, co-founder, executive",
      "key_achievements": ["string — notable achievements or metrics"]
    }
  ],

  "education": [
    {
      "institution": "string",
      "degree": "string — BS, MS, PhD, MBA, etc.",
      "field": "string — field of study",
      "graduation_year": "number or null"
    }
  ],

  "total_years_experience": "number — total professional years",

  "domains": ["string — list of all industries/verticals they have experience in"],

  "role_archetype": "string — one of: founding_builder (builds from zero at startups), platform_engineer (builds infrastructure and tools), research_scientist (research-focused), product_engineer (full-stack product work), specialist (deep expertise in narrow area), generalist (broad skills across many areas), manager (people/team management focus)",

  "career_trajectory": "string — one of: ascending (consistently growing in scope/seniority), lateral (moving across domains at similar level), career_changer (shifting to a new field), early_career (less than 3 years total), seasoned (10+ years, stable trajectory)",

  "inferred_preferences": {
    "preferred_company_stage": "string or null — based on career history",
    "preferred_team_size": "string or null",
    "preferred_work_style": "string or null — fast-paced/methodical/research-driven/etc.",
    "likely_looking_for": "string — brief inference of what kind of role they probably want next"
  },

  "summary": "string — 2-3 sentence career summary capturing who this person is professionally"
}
```

If GitHub data is provided, add to the prompt: "Additionally, here is their GitHub profile data: {github_summary}. Use this to validate and enhance skill proficiency ratings — if a skill appears in multiple repos with recent commits, rate it higher. Note any skills visible in GitHub but missing from the resume."

**Extraction function:**

`extract_profile(resume_text: str, github_data: dict | None = None) -> ExtractedProfile`

- Build the prompt from the template, inserting resume text and optionally GitHub data summary
- Call the `google-genai` SDK: `client.models.generate_content` with the system instruction and user prompt
- Model name from `settings.llm.llm_model_pro` (default: `gemini-2.5-pro`) — quality-critical extraction
- Max tokens from settings (default: 4096)
- Parse the response: extract the text content from the response
- Clean the response: strip markdown code fences if present (```json ... ```), strip any preamble text before the JSON
- Parse as JSON: `json.loads(cleaned_response)`
- Validate against Pydantic model `ExtractedProfile` — this catches missing fields, wrong types
- If JSON parsing fails or validation fails:
  - Log the error and the raw response for debugging
  - Retry once with a simpler prompt that explicitly says "Your previous response was not valid JSON. Please respond with ONLY a JSON object, no other text."
  - If retry also fails, raise `ExtractionFailedError` with context
- Return the validated `ExtractedProfile` object

**Pydantic models for extraction output:**

Define these in `src/api/schemas/candidate.py` or a dedicated `src/ingestion/schemas.py`:

- `ExtractedSkill` — name, category, proficiency, years_used, context
- `ExtractedExperience` — all the experience fields listed above
- `ExtractedEducation` — institution, degree, field, graduation_year
- `InferredPreferences` — preferred_company_stage, preferred_team_size, preferred_work_style, likely_looking_for
- `ExtractedProfile` — the complete profile with all fields, using the sub-models above

All fields that could be null should be `Optional[T] = None` in the Pydantic model. Use `model_validator` to add custom validation:
- `skills` list should not be empty (if it is, the extraction probably failed)
- `experience` list should not be empty
- `total_years_experience` should be >= 0

**Cost tracking:**

Log the token usage from each API call: input tokens, output tokens, total. Store this in the function's return value or log it separately. This helps track API costs during development and production.

**Tests:**

Create `tests/test_llm_extractor.py`:
- Mock the Google GenAI client — do NOT make real API calls in tests
- Create a mock response that returns a valid JSON profile
- Test that the extraction function correctly parses the mock response into an ExtractedProfile
- Test retry logic: mock first call returning invalid JSON, second call returning valid JSON — verify retry works
- Test that extraction failure after retries raises ExtractionFailedError
- Test with GitHub data provided: verify the prompt includes GitHub data
- Test Pydantic validation: pass JSON with missing required fields, verify validation catches it

Verify: with a real API key, run the extractor on a sample resume text. Inspect the output JSON for quality: are skills complete? Are experience entries reasonable? Does the role archetype make sense? Do inferred preferences seem right?

**Commit:** `feat(ingestion): LLM-powered structured profile extraction with inference`

---

## Step 2.3 — GitHub Fetcher

### What to do
Build a module that takes a GitHub username and returns a structured analysis of their public profile, repos, languages, and activity patterns.

### Instructions

Create `src/ingestion/github_fetcher.py`:

**GitHub API client setup:**
- Use `httpx.AsyncClient` for all GitHub API calls
- Base URL: `https://api.github.com`
- Set headers: `Accept: application/vnd.github.v3+json`, `Authorization: Bearer {token}` (token from settings)
- If no token configured, make unauthenticated requests (60 req/hr limit instead of 5000)
- Handle rate limiting: check `X-RateLimit-Remaining` header on each response. If remaining < 10, log a warning. If remaining = 0, read `X-RateLimit-Reset` header and sleep until that timestamp.

**Main fetch function:**

`fetch_github_profile(username: str) -> GitHubProfile`

Step 1 — Fetch user profile:
- `GET /users/{username}`
- Extract: name, bio, public_repos count, followers, following, created_at
- If 404, raise `GitHubUserNotFoundError`

Step 2 — Fetch repositories:
- `GET /users/{username}/repos?sort=updated&per_page=100&type=owner`
- If more than 100 repos, paginate (follow `Link` header) up to 300 repos max
- Filter out forks unless they have significant additional commits (stargazers_count > 0 or has been updated recently)
- For each repo, extract: name, description, language (primary), stargazers_count, forks_count, size, created_at, updated_at, pushed_at, topics, has_wiki, default_branch

Step 3 — Fetch languages per repo (for top 20 repos by recency):
- `GET /repos/{username}/{repo}/languages`
- Returns dict of {language: bytes_of_code}
- Don't fetch for all repos — only the 20 most recently updated to save API calls

Step 4 — Fetch README for top 10 repos:
- `GET /repos/{username}/{repo}/readme`
- Decode from base64
- Truncate to first 2000 characters (we only need a summary, not the full README)
- If no README exists (404), set to null

Step 5 — Check for production-readiness indicators in top 20 repos:
- Check if repo contains specific files (use repo contents API or check common patterns):
  - `Dockerfile` or `docker-compose.yml` → has_docker = true
  - `.github/workflows/` directory or `.circleci/` or `Jenkinsfile` → has_ci = true
  - `tests/` or `test_*.py` or `*_test.go` etc. → has_tests = true (check by listing root directory)
  - `requirements.txt` or `pyproject.toml` or `package.json` → has_dependency_management = true
- This step requires additional API calls so only do it for the top 10-20 repos

**Aggregation and analysis:**

After fetching raw data, compute derived metrics:

`languages_distribution: dict[str, float]` — aggregate language bytes across all repos, compute percentage per language. Example: {"Python": 0.65, "JavaScript": 0.20, "TypeScript": 0.10, "Shell": 0.05}

`activity_metrics`:
- `total_repos: int`
- `repos_last_6_months: int` — repos with pushed_at in last 6 months
- `repos_last_year: int`
- `most_active_language: str`
- `avg_stars: float`
- `total_stars: int`

`project_complexity_scores: list[RepoAnalysis]` — for each of the top 20 repos:
- `name: str`
- `complexity: str` — "low" (single file or trivial), "medium" (multi-file project), "high" (has tests, CI, Docker, multiple languages)
  - Scoring: +1 for has_tests, +1 for has_docker, +1 for has_ci, +1 for multiple languages, +1 for >1000 lines (infer from size), +1 for >5 stars. 0-1 = low, 2-3 = medium, 4+ = high
- `languages: list[str]`
- `description: str`
- `stars: int`
- `last_active: str` — relative time ("2 weeks ago", "3 months ago")
- `readme_summary: str | None` — first 500 chars of README
- `production_signals: list[str]` — which production indicators were found ("docker", "ci", "tests")

`inferred_skills: list[str]` — skills inferred from GitHub that might not be on the resume. Derive from: languages used (Python → "Python"), framework indicators in repo names or topics (e.g., topic "fastapi" → "FastAPI"), README content keywords.

**GitHubProfile Pydantic model:**

Define in `src/api/schemas/candidate.py` or `src/ingestion/schemas.py`:
- `username: str`
- `name: str | None`
- `bio: str | None`
- `public_repos: int`
- `followers: int`
- `account_age_years: float`
- `languages_distribution: dict[str, float]`
- `activity_metrics: ActivityMetrics`
- `top_repos: list[RepoAnalysis]`
- `inferred_skills: list[str]`
- `overall_assessment: str` — one of: "active_builder" (consistent recent activity, diverse projects), "portfolio_focused" (few but polished repos), "contributor" (lots of forks, open source contributions), "inactive" (no activity in 6+ months), "beginner" (<5 repos, all simple)

**Error handling:**
- GitHub user not found (404) → raise `GitHubUserNotFoundError`, don't crash the pipeline — profile can still be built from resume alone
- Rate limit hit → log warning, sleep and retry with exponential backoff
- Individual repo API call fails → log error, skip that repo, continue with others
- No public repos → return a valid GitHubProfile with empty repos list and "inactive" assessment

**Tests:**

Create `tests/test_github_fetcher.py`:
- Mock all GitHub API responses using httpx mock or respx library
- Test successful profile fetch with mock data for user, repos, languages, readmes
- Test language distribution calculation: given mock repos with known language bytes, verify percentages
- Test complexity scoring: given repos with known production signals, verify correct complexity levels
- Test user not found: mock 404 response, verify GitHubUserNotFoundError
- Test rate limiting: mock rate limit headers, verify sleep/retry behavior
- Test with zero public repos: verify empty but valid profile returned

Verify: with a real GitHub token, run `fetch_github_profile("your-username")`. Inspect the output: are languages correct? Are repos ranked by recency? Do complexity scores make sense? Are inferred skills reasonable?

**Commit:** `feat(ingestion): GitHub profile fetcher with repo analysis and activity scoring`

---

## Step 2.4 — Profile Builder (Merge All Signals)

### What to do
Build the orchestrator that takes all raw inputs (resume file, GitHub username, user preferences), runs them through the individual processors, and produces a single unified CandidateProfile with cross-validated, merged data.

### Instructions

Create `src/ingestion/profile_builder.py`:

**Main orchestration function:**

`async build_profile(resume_file_path: str, github_username: str | None = None, preferences: CandidatePreferences | None = None) -> CandidateProfile`

This is the top-level function that the API endpoint will call. It orchestrates the full pipeline:

Step 1 — Parse resume:
- Call `resume_parser.parse_resume(resume_file_path)`
- If parsing fails, raise error immediately (resume is required, can't build a profile without it)
- Store raw text for later use

Step 2 — Fetch GitHub data (if username provided):
- Call `github_fetcher.fetch_github_profile(github_username)`
- If GitHub fetch fails (user not found, rate limited, network error), log the error and continue without GitHub data — it's optional
- If successful, prepare a GitHub summary string for the LLM: top languages, top repos with descriptions, production signals. Keep it under 1000 tokens to not bloat the LLM prompt.

Step 3 — LLM structured extraction:
- Call `llm_extractor.extract_profile(resume_text, github_data=github_summary_or_none)`
- This returns the ExtractedProfile with skills, experience, domains, role archetype, inferred preferences
- If LLM extraction fails after retries, attempt a minimal fallback: extract basic info using rule-based parsing (find email with regex, extract skill-like keywords, find company names). This fallback produces a much weaker profile but is better than nothing.

Step 4 — Link skills to ESCO:
- Take all skills from the LLM extraction (skill names)
- Also take inferred skills from GitHub (language names, framework names from topics)
- Combine into a deduplicated list
- Call `entity_linker.link_skills(all_skill_names)` from Phase 1
- For each skill, store: the original text, the ESCO URI (if linked), the ESCO label, the match confidence
- Skills that don't link to any ESCO node are kept as-is (they might be too specific or too new for ESCO)

Step 5 — Compute skill depth scores:
- For each skill, compute a depth score (0.0 to 1.0) based on multiple signals:
  - Resume mention: +0.25 (the skill appears on the resume)
  - Resume context depth: +0.15 if the LLM extracted proficiency as "advanced" or "expert"
  - GitHub presence: +0.25 if the skill (or its language) appears in GitHub repos
  - GitHub recency: +0.15 if found in repos updated within the last 6 months
  - GitHub production signals: +0.10 if found in repos with tests/CI/Docker
  - GitHub volume: +0.10 if found in 3+ repos
- Cap at 1.0
- Skills from GitHub that are NOT on the resume get a base score of 0.2 (proven but not highlighted)
- Skills on the resume but NOT in GitHub get a base score of 0.3 (claimed but unverified)

Step 6 — Merge preferences:
- Start with the LLM's inferred preferences (from career history analysis)
- If the user provided explicit preferences, those override the inferred ones on every field where the user set a value
- If the user didn't set a particular preference, keep the LLM's inference
- Construct the final preferences object with clear source tracking: `{field: value, source: "explicit" | "inferred"}`

Step 7 — Build the final CandidateProfile:
- Merge everything into the CandidateProfile Pydantic model
- Fields:
  - `name`, `email`, `phone`, `location` — from LLM extraction
  - `skills` — merged list with depth scores, ESCO URIs, and source tracking
  - `experience` — from LLM extraction, sorted by date descending
  - `education` — from LLM extraction
  - `domains` — from LLM extraction, deduplicated
  - `total_years_experience` — from LLM extraction
  - `role_archetype` — from LLM extraction
  - `career_trajectory` — from LLM extraction
  - `github_summary` — condensed GitHub analysis (if available)
  - `preferences` — merged preferences with source tracking
  - `esco_linked_skills` — list of skills successfully linked to ESCO nodes
  - `summary` — from LLM extraction

**CandidateProfile Pydantic model:**

Define the complete model. This is the canonical representation of a candidate used by the matching engine. Every downstream component reads from this model.

```
CandidateProfile:
  # Identity
  name: str | None
  email: str | None
  phone: str | None
  location: str | None

  # Skills (merged from all sources)
  skills: list[ProfileSkill]
    # ProfileSkill:
    #   name: str
    #   category: str
    #   proficiency: str
    #   depth_score: float (0-1)
    #   years_used: int | None
    #   context: str | None
    #   esco_uri: str | None
    #   esco_label: str | None
    #   sources: list[str] (e.g., ["resume", "github"])

  # Experience
  experience: list[ProfileExperience]
  education: list[ProfileEducation]
  total_years_experience: float
  domains: list[str]

  # Archetypes and trajectory
  role_archetype: str
  career_trajectory: str

  # GitHub (optional)
  github_summary: GitHubSummary | None

  # Preferences (merged explicit + inferred)
  preferences: MergedPreferences
    # MergedPreferences:
    #   job_types: list[str] | None (full-time, contract, part-time)
    #   work_models: list[str] | None (remote, hybrid, onsite)
    #   locations: list[str] | None
    #   needs_sponsorship: bool | None
    #   salary_min: int | None
    #   salary_max: int | None
    #   company_stages: list[str] | None
    #   company_sizes: list[str] | None
    #   target_roles: list[str] | None
    #   target_industries: list[str] | None
    #   avoid_industries: list[str] | None
    #   priorities: list[str] | None (ranked: compensation, equity, learning, mission, etc.)
    #   --- each field also has a source: "explicit" | "inferred" | "default"

  # ESCO linkage
  esco_linked_skills: list[ESCOLinkedSkill]

  # Summary
  summary: str
```

**Store the profile:**
- After building the profile, serialize it to JSON and store in the PostgreSQL Candidate record's `profile` JSONB column
- Also store the raw resume text in the `resume_text` column
- Also store the GitHub username and raw GitHub data in their respective columns
- Return the profile to the caller

**Tests:**

Note: this module orchestrates multiple components, so tests should mock the sub-components and test the orchestration logic:
- Mock resume_parser to return known text
- Mock llm_extractor to return a known ExtractedProfile
- Mock github_fetcher to return a known GitHubProfile
- Mock entity_linker to return known ESCO links
- Test full pipeline with all inputs: verify the output CandidateProfile has merged data from all sources
- Test with GitHub unavailable: verify profile builds from resume only
- Test with LLM failure + fallback: verify minimal profile is produced
- Test skill depth scoring: given known resume skills and GitHub skills, verify scores are computed correctly
- Test preference merging: given inferred preferences and explicit overrides, verify explicit wins
- Test ESCO linking integration: verify skills have ESCO URIs attached

**Commit:** `feat(ingestion): unified profile builder merging resume + GitHub + preferences`

---

## Step 2.5 — Candidate Embeddings

### What to do
Generate multi-dimensional embedding vectors for a candidate profile. These vectors are what the FAISS vector retriever will use to find matching jobs.

### Instructions

**Encoder wrapper:**

Create `src/embeddings/encoder.py`:

`EmbeddingEncoder` class:
- Constructor: load the sentence-transformer model specified in settings (default `all-MiniLM-L6-v2`)
- Model is loaded once and kept in memory for reuse
- Use a class-level or module-level singleton pattern so the model isn't loaded multiple times
- `encode(text: str) -> np.ndarray` — encode a single text string, return a 384-dim float32 numpy array
- `encode_batch(texts: list[str]) -> np.ndarray` — encode multiple texts at once, return a (N, 384) numpy array. Use the model's built-in batch encoding for efficiency.
- Handle empty strings: if input is empty or only whitespace, return a zero vector and log a warning
- Handle very long texts: sentence-transformers have a max token limit (typically 512 tokens for MiniLM). If the input text is longer, truncate to the token limit. Alternatively, split into chunks, encode each chunk, and average the vectors (mean pooling across chunks). The averaging approach preserves more information for long texts like job descriptions.
- Set `device` based on availability: use CUDA if available (for faster encoding), otherwise CPU. Read from settings to allow forcing CPU mode.

**Candidate embedding generator:**

Create `src/embeddings/candidate_embedder.py`:

`embed_candidate(profile: CandidateProfile) -> CandidateEmbeddings`

Generate four separate embedding vectors, each capturing a different dimension of who the candidate is:

**Skill embedding:**
- Construct a text representation focused on skills:
  - Take all skills with their category, proficiency, and context
  - Format: "Skills: {skill1} (advanced, used for building ML pipelines), {skill2} (intermediate, used in data processing), ..."
  - Weight by depth score: skills with higher depth scores should appear first and potentially be repeated or emphasized
  - Include ESCO expanded labels for linked skills: "Python (programming language), also related to: software development, scripting, automation"
- Encode this text → 384-dim vector
- This vector captures WHAT they can do

**Domain embedding:**
- Construct a text representation focused on industry and domain experience:
  - Take all experience entries' domains and descriptions
  - Format: "Domain experience: {duration} in {domain1} working on {description}. {duration} in {domain2} working on {description}."
  - Also include the domains list from the profile
  - Weight by recency: most recent domains first
- Encode this text → 384-dim vector
- This vector captures WHERE they've worked

**Role embedding:**
- Construct a text representation focused on the shape and level of their work:
  - Take role archetype, career trajectory, job titles, and key achievements
  - Format: "Role profile: {role_archetype}. Career trajectory: {career_trajectory}. Recent roles: {title1} at {company1} ({role_type}, {company_stage}), {title2} at {company2} ({role_type}, {company_stage}). Key achievements: {achievement1}, {achievement2}."
  - Include team size and scope indicators from experience
- Encode this text → 384-dim vector
- This vector captures HOW they work

**Environment embedding:**
- Construct a text representation focused on work environment and culture preferences:
  - Take company sizes, stages, and work styles from experience history
  - Take explicit and inferred preferences
  - Format: "Work environment: Has worked at {company_stage} companies with {company_size} teams. Prefers {preferred_work_style}. Values {priorities}. Looking for {likely_looking_for}."
- Encode this text → 384-dim vector
- This vector captures WHERE they want to be

Return a `CandidateEmbeddings` object:
```
CandidateEmbeddings:
  skill: np.ndarray (384,)
  domain: np.ndarray (384,)
  role: np.ndarray (384,)
  environment: np.ndarray (384,)
```

**Serialization helpers:**
- `serialize_embedding(vector: np.ndarray) -> bytes` — convert numpy array to bytes for PostgreSQL storage
- `deserialize_embedding(data: bytes) -> np.ndarray` — convert bytes back to numpy array
- These go in the encoder module for shared use

**Store embeddings:**
- After generating embeddings, serialize and store in the Candidate record's embedding columns
- Update the candidate record in PostgreSQL

**Tests:**

Create tests (can be in `tests/test_candidate_embedder.py` or extend existing test files):
- Test encoder: encode a known text, verify output shape is (384,), verify it's a float32 array
- Test batch encoding: encode multiple texts, verify output shape is (N, 384)
- Test empty input handling: verify zero vector returned for empty string
- Test candidate embedding: given a mock CandidateProfile, verify four embeddings are generated with correct shapes
- Test that different profiles produce different embeddings (cosine similarity < 1.0)
- Test serialization roundtrip: serialize → deserialize → verify arrays are equal
- Test that the skill embedding of an ML engineer is more similar to a data scientist than to a frontend developer (basic semantic sanity check)

Verify: generate embeddings for a real candidate profile. Check that the four vectors are different from each other (they should capture different aspects). Compute cosine similarity between skill embeddings of two different profiles — verify it's a reasonable number between 0 and 1.

**Commit:** `feat(embeddings): multi-vector candidate embedding generation (skill, domain, role, environment)`

---

## Phase 2 Completion Checklist

Before moving to Phase 3, verify ALL of the following:

- [ ] Git repo has 5 new commits from Phase 2 (12 total with Phase 1's 7)
- [ ] Resume parser handles PDF and DOCX files correctly
- [ ] Resume parser produces clean text from a real resume
- [ ] LLM extractor returns a valid ExtractedProfile from resume text
- [ ] ExtractedProfile has skills, experience, domains, role archetype, inferred preferences
- [ ] GitHub fetcher returns a valid GitHubProfile with repos, languages, activity metrics
- [ ] GitHub fetcher handles user not found and rate limiting gracefully
- [ ] Profile builder merges resume + GitHub + preferences into a single CandidateProfile
- [ ] Skill depth scores are computed correctly (resume + GitHub signals combined)
- [ ] All extracted skills are linked to ESCO nodes where possible
- [ ] Explicit preferences override inferred preferences
- [ ] Four embedding vectors (skill, domain, role, environment) are generated for each candidate
- [ ] Embeddings are 384-dim float32 numpy arrays
- [ ] Embeddings are serialized and stored in PostgreSQL
- [ ] All tests pass
- [ ] Pipeline works end-to-end: `build_profile("resume.pdf", "github-user", preferences)` returns a complete CandidateProfile with embeddings
- [ ] No hardcoded values — all config from environment variables
- [ ] LLM token usage is logged for cost tracking
