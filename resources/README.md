# HiringCafe Job Recommendation — Reference Materials

Research papers, repos, datasets, and architecture docs for the personalized job recommendation engine MVP.

## Folder Structure

```
resources/
├── papers/           # Paper summaries with links and borrowings
├── repos/            # GitHub repo index (repos_index.md)
├── datasets/         # Dataset catalog (datasets_index.md)
├── knowledge_graph/  # Neo4j + ESCO setup guide
├── models_info/      # Encoder/reranker/vector DB comparison
└── architecture/     # MVP architecture + tech decisions
```

## Quick Start

1. Read `architecture/mvp_architecture.md` for the five-stage pipeline.
2. Read `architecture/tech_decisions.md` for rationale.
3. Set up Neo4j: `knowledge_graph/neo4j_esco_setup.md`
4. Clone reference repos: see `repos/repos_index.md`
5. Download datasets: see `datasets/datasets_index.md`

## Papers Index

| File | Topic |
|------|-------|
| `papers/careerbuilder_two_stage.md` | FAISS + two-stage retrieval at scale |
| `papers/jobmatchai.md` | Hybrid BM25 + semantic + KG + explainable AI |
| `papers/mira_embeddings.md` | Synthetic training data, boundary-aware embeddings |
| `papers/resume2vec.md` | Encoder benchmark for resume–JD matching |
| `papers/esco_job_matching.md` | ESCO linking with LLMs |
| `papers/two_tower_diffusion.md` | Two-tower limitations + cross-interaction |

## Key External Links

- JobMatchAI repo: https://github.com/coral-lab-asu/job-hunt-AI
- ESCO download: https://esco.ec.europa.eu/en/use-esco/download
- Tabiya classifier: https://github.com/tabiya-tech/tabiya-livelihoods-classifier
- SkillAlign (ESCO + FAISS + Neo4j): https://github.com/Y4SSERk/SkillAlign
