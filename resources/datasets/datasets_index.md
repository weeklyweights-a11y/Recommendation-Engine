# Datasets Index

Reference datasets for building, training, and evaluating the HiringCafe personalized job recommendation engine MVP.

---

## ESCO Taxonomy Dataset

| Field | Value |
|-------|-------|
| **Download** | https://esco.ec.europa.eu/en/use-esco/download |
| **Format** | CSV / RDF / JSON — skills, competences, occupations, qualifications, and relationships |
| **Size** | ~3,000 occupations, ~13,000+ skills (version-dependent) |
| **License** | EU ESCO terms (open for reuse with attribution) |
| **Use in MVP** | Load into Neo4j as the skill knowledge graph backbone |
| **Key files** | `skills_en.csv`, `occupations_en.csv`, `skillSkillRelations_en.csv`, `occupationSkillRelations_en.csv` |
| **Setup guide** | See `resources/knowledge_graph/neo4j_esco_setup.md` |

### Loading checklist

- [ ] Download latest ESCO package (English)
- [ ] Parse CSVs → Cypher import scripts
- [ ] Create `Skill`, `Occupation` nodes and `RELATED_TO`, `REQUIRES` edges
- [ ] Validate skill expansion query (PyTorch → deep learning, TensorFlow, etc.)

---

## JobSearch-XS Benchmark (JobMatchAI)

| Field | Value |
|-------|-------|
| **Source** | JobMatchAI paper (ASU, arXiv:2603.14558) |
| **Paper** | https://arxiv.org/html/2603.14558v2 |
| **Repo** | https://github.com/coral-lab-asu/job-hunt-AI |
| **License** | CC-BY-4.0 (stated as releasing upon paper acceptance) |
| **Contents** | 1,283 NYC civil-service job roles, 30 search queries, ~29K silver relevance labels |
| **Splits** | Skill-disjoint train/dev/test for zero-shot skill generalization eval |
| **Use in MVP** | Primary offline evaluation benchmark for retrieval pipeline (target: NDCG@10 ≥ 0.75) |
| **Metrics** | NDCG@10, MRR, Recall@K; compare against BM25-only baseline |

### Download status

- Check `coral-lab-asu/job-hunt-AI` releases/README for benchmark download instructions.
- Fallback: NYC Open Data civil-service postings (same source as paper) + manual query set.

---

## LinkedIn Job Postings Dataset (Kaggle)

| Field | Value |
|-------|-------|
| **URL** | https://www.kaggle.com/datasets/arshkon/linkedin-job-postings |
| **Title** | LinkedIn Job Postings (2023–2024) |
| **Author** | arshkon |
| **Size** | 124,000+ job postings |
| **Format** | CSV — `job_postings.csv`, `companies.csv`, `benefits.csv`, etc. |
| **Key columns** | `title`, `description`, `location`, `max_salary`, `min_salary`, `med_salary`, `remote_allowed`, `formatted_work_type`, `formatted_experience_level`, `skills_desc` |
| **Use in MVP** | Training and testing the matching engine before connecting to live HiringCafe data |
| **Notes** | Requires Kaggle account; use for embedding index build, ESCO linking eval, and TF-IDF baseline comparison |

### Suggested MVP splits

| Split | % | Use |
|-------|---|-----|
| Index corpus | 80% | Build FAISS + BM25 indexes |
| Dev eval | 10% | Tune fusion weights and reranker |
| Holdout test | 10% | Final offline metrics report |

---

## Stack Overflow Developer Survey

| Field | Value |
|-------|-------|
| **URL** | https://survey.stackoverflow.co/ |
| **Data downloads** | https://survey.stackoverflow.co/ (annual results + CSV exports) |
| **Format** | CSV per survey year |
| **Use in MVP** | Skills distribution data, developer preferences for enriching synthetic candidate profiles |
| **Example uses** | Popular language/skill co-occurrence priors; remote-work preference defaults; experience-level calibration |

---

## Dataset Priority for MVP Phases

| Phase | Dataset | Goal |
|-------|---------|------|
| **Week 1** | ESCO | Neo4j knowledge graph online |
| **Week 1–2** | LinkedIn Kaggle (subset) | Build FAISS + BM25 indexes, test ingestion |
| **Week 2** | JobSearch-XS | Standardized retrieval eval (NDCG@10) |
| **Week 3+** | HiringCafe live data | Replace Kaggle corpus with production postings |
| **Optional** | Stack Overflow Survey | Enrich candidate preference defaults |

---

## Download Commands

```bash
# ESCO — manual download from browser:
# https://esco.ec.europa.eu/en/use-esco/download
# Save to: D:\LinkedIn\Hiringcafe_demo\resources\datasets\esco\

# Kaggle (requires API credentials):
# pip install kaggle
# kaggle datasets download -d arshkon/linkedin-job-postings -p resources/datasets/linkedin/

# JobSearch-XS — follow coral-lab-asu/job-hunt-AI instructions
```
