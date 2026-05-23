# Neo4j + ESCO Knowledge Graph Setup

Guide for loading the ESCO taxonomy into Neo4j and running skill-expansion queries for the HiringCafe job recommendation MVP.

## Reference

- **ESCO download:** https://esco.ec.europa.eu/en/use-esco/download
- **Blog walkthrough:** https://blog.bruggen.com/2018/08/esco-database-in-neo4j-skills.html
- **Example repo:** https://github.com/Y4SSERk/SkillAlign (ETL + Cypher patterns)

---

## 1. Install Neo4j

### Option A — Local (recommended for development)

1. Download **Neo4j Desktop** or **Community Edition**: https://neo4j.com/download/
2. Create a new database (default password on first login).
3. Open **Neo4j Browser** at http://localhost:7474
4. Default bolt URI: `bolt://localhost:7687`

### Option B — Neo4j Aura Free Tier (cloud)

1. Sign up: https://neo4j.com/cloud/aura/
2. Create a free AuraDB instance.
3. Save connection URI, username, and password.
4. Connect from Python: `Graph("neo4j+s://xxxx.databases.neo4j.io", auth=(user, pass))`

### Python driver

```bash
pip install neo4j py2neo
```

---

## 2. Download ESCO Taxonomy

1. Go to https://esco.ec.europa.eu/en/use-esco/download
2. Download the **CSV** package (English).
3. Extract to `resources/datasets/esco/`

### Key CSV files

| File | Contents |
|------|----------|
| `skills_en.csv` | Skill labels, URIs, descriptions |
| `occupations_en.csv` | Occupation labels, URIs |
| `skillSkillRelations_en.csv` | Skill ↔ skill relationships |
| `occupationSkillRelations_en.csv` | Occupation → required/optional skills |
| `broaderRelationsSkillPillar_en.csv` | Skill hierarchy (broader/narrower) |

---

## 3. Graph Schema

```cypher
// Constraints
CREATE CONSTRAINT skill_uri IF NOT EXISTS FOR (s:Skill) REQUIRE s.uri IS UNIQUE;
CREATE CONSTRAINT occupation_uri IF NOT EXISTS FOR (o:Occupation) REQUIRE o.uri IS UNIQUE;

// Node properties
// Skill: { uri, label, description, skillType }
// Occupation: { uri, label, description }

// Relationships
// (Skill)-[:RELATED_TO { relationType }]->(Skill)
// (Skill)-[:BROADER_THAN]->(Skill)
// (Occupation)-[:REQUIRES_SKILL { essential }]->(Skill)
```

---

## 4. Cypher Import Scripts

### 4.1 Load skills

```cypher
// Run via neo4j-admin import or LOAD CSV (file:/// must be in Neo4j import dir)
LOAD CSV WITH HEADERS FROM 'file:///skills_en.csv' AS row
MERGE (s:Skill {uri: row.conceptUri})
SET s.label = row.preferredLabel,
    s.description = row.description,
    s.skillType = row.skillType;
```

### 4.2 Load skill–skill relations

```cypher
LOAD CSV WITH HEADERS FROM 'file:///skillSkillRelations_en.csv' AS row
MATCH (a:Skill {uri: row.originalSkillUri})
MATCH (b:Skill {uri: row.relatedSkillUri})
MERGE (a)-[r:RELATED_TO]->(b)
SET r.relationType = row.relationType;
```

### 4.3 Load occupations and required skills

```cypher
LOAD CSV WITH HEADERS FROM 'file:///occupations_en.csv' AS row
MERGE (o:Occupation {uri: row.conceptUri})
SET o.label = row.preferredLabel;

LOAD CSV WITH HEADERS FROM 'file:///occupationSkillRelations_en.csv' AS row
MATCH (o:Occupation {uri: row.occupationUri})
MATCH (s:Skill {uri: row.skillUri})
MERGE (o)-[r:REQUIRES_SKILL]->(s)
SET r.essential = row.essentialSkill;
```

> **Tip:** Copy CSVs to Neo4j's `import/` directory, or use `neo4j-admin database import full` for large datasets.

---

## 5. Skill Expansion Queries

### 5.1 Given a skill, find related skills (1 hop)

```cypher
MATCH (s:Skill {label: 'PyTorch'})-[:RELATED_TO]-(related:Skill)
RETURN DISTINCT related.label AS skill, related.uri AS uri
LIMIT 20;
```

### 5.2 Two-hop expansion (for query enrichment)

```cypher
MATCH (s:Skill)
WHERE toLower(s.label) = 'pytorch'
CALL {
  WITH s
  MATCH (s)-[:RELATED_TO*1..2]-(expanded:Skill)
  RETURN DISTINCT expanded
}
RETURN expanded.label AS skill, expanded.uri AS uri
LIMIT 50;
```

Expected neighbors for **PyTorch** (illustrative — actual results depend on ESCO version):

- deep learning
- neural networks
- TensorFlow
- machine learning
- model training
- Python

### 5.3 Expand candidate skills → matching jobs

```cypher
// Given expanded skill set from resume
UNWIND $skillUris AS skillUri
MATCH (s:Skill {uri: skillUri})<-[:REQUIRES_SKILL]-(o:Occupation)
RETURN o.uri AS occupationUri, o.label AS occupation, count(*) AS skillOverlap
ORDER BY skillOverlap DESC
LIMIT 75;
```

### 5.4 Bridge vocabulary gaps (JobMatchAI pattern)

```cypher
// Candidate has "Kubernetes", job requires "container orchestration"
MATCH (cSkill:Skill {label: 'Kubernetes'})-[:RELATED_TO*1..2]-(bridge:Skill)
      <-[:REQUIRES_SKILL]-(j:Job {id: $jobId})
RETURN bridge.label AS matchedVia, j.title AS jobTitle;
```

---

## 6. Integration with Retrieval Pipeline

```
Resume / Job text
    → Tabiya ESCO linker → skill URIs
    → Neo4j 2-hop RELATED_TO expansion → expanded skill set S+
    → Graph retriever: traverse REQUIRES_SKILL to jobs
    → Merge with BM25 + FAISS results via RRF
```

### Python example (py2neo)

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def expand_skills(skill_labels: list[str], hops: int = 2) -> list[str]:
    query = """
    UNWIND $labels AS label
    MATCH (s:Skill) WHERE toLower(s.label) = toLower(label)
    MATCH (s)-[:RELATED_TO*1..$hops]-(e:Skill)
    RETURN DISTINCT e.label AS skill
  """
    with driver.session() as session:
        result = session.run(query, labels=skill_labels, hops=hops)
        return [r["skill"] for r in result]
```

---

## 7. MVP Graph Extensions (beyond raw ESCO)

Add application-specific nodes on top of ESCO:

```cypher
// Candidate and Job nodes (JobMatchAI pattern)
CREATE (c:Candidate {id: $candidateId})
CREATE (j:Job {id: $jobId, title: $title})

// Link to ESCO skills
MATCH (s:Skill {uri: $skillUri})
CREATE (c)-[:HAS_SKILL]->(s)
CREATE (j)-[:REQUIRES_SKILL]->(s)
```

---

## 8. Verification Checklist

- [ ] Neo4j running (local or Aura)
- [ ] ESCO CSVs downloaded and imported
- [ ] `Skill` and `Occupation` node counts match ESCO docs (~13K skills)
- [ ] `RELATED_TO` edges present (skillSkillRelations)
- [ ] PyTorch 2-hop expansion returns expected related skills
- [ ] Python driver connects from FastAPI backend
- [ ] Graph retriever returns jobs in < 50ms for MVP scale

---

## 9. Performance Tips

| Tip | Detail |
|-----|--------|
| Indexes | Create indexes on `Skill.label`, `Skill.uri`, `Job.id` |
| Hop limit | Cap expansion at 2 hops to control query fan-out |
| Cache | Cache expanded skill sets per candidate session (Redis or in-memory) |
| Batch import | Use `neo4j-admin import` for initial load, not row-by-row MERGE |
