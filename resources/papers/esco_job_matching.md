# ESCO Job Matching with LLMs — Sentence vs Entity Linking

## Links

- **Paper (PDF):** https://arxiv.org/pdf/2512.03195
- **arXiv abstract:** https://arxiv.org/abs/2512.03195
- **Open-source tool:** https://github.com/tabiya-tech/tabiya-livelihoods-classifier

## Summary

This work (December 2025) compares two approaches for mapping free-text job descriptions to the **ESCO** (European Skills, Competences, Qualifications and Occupations) taxonomy using LLMs and transformer-based linking:

1. **Sentence Linking** — Embed full sentences/phrases and match to ESCO labels via similarity.
2. **Entity Linking** — Extract skill/occupation entities first, then link each entity to ESCO nodes.

Tabiya released an open-source classifier implementing both approaches.

## Key Architecture Details

| Approach | Strength | Weakness |
|----------|----------|----------|
| **Sentence Linking** | Captures context; good for ambiguous phrases | Slower; may over-match broad labels |
| **Entity Linking** | Precise per-skill mapping; graph-friendly | Misses implicit skills not explicitly mentioned |
| **ESCO taxonomy** | Standardized skills, occupations, relationships | Large (13K+ skills); requires efficient indexing |
| **Output** | ESCO URIs attached to job/candidate profiles | Enables Neo4j graph population and skill expansion |

## What We Borrow

1. **ESCO linking methodology** — Map extracted resume/job skills to ESCO nodes before graph traversal.
2. **Tabiya classifier** — Use `tabiya-livelihoods-classifier` for entity linking in our ingestion pipeline.
3. **Dual linking strategy** — Sentence linking for occupation classification; entity linking for skill nodes in Neo4j.

## Integration with Knowledge Graph

```
Job description / Resume
    → Tabiya ESCO linker → ESCO skill/occupation URIs
    → Neo4j nodes (Skill, Occupation) + RELATED_TO edges
    → Graph expansion queries at retrieval time
```

## Related Repo

See `resources/repos/repos_index.md` → **Tabiya Livelihoods Classifier**
