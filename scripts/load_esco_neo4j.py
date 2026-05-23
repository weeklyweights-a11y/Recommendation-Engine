"""Load ESCO taxonomy CSVs into Neo4j."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings
from src.knowledge_graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


def find_csv(data_dir: Path, name: str) -> Path:
    """Find CSV by filename under data directory."""
    matches = list(data_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Missing {name} under {data_dir}")
    return matches[0]


def setup_constraints(client: Neo4jClient) -> None:
    """Create constraints and indexes."""
    statements = [
        "CREATE CONSTRAINT skill_uri IF NOT EXISTS FOR (s:Skill) REQUIRE s.uri IS UNIQUE",
        "CREATE CONSTRAINT occupation_uri IF NOT EXISTS FOR (o:Occupation) REQUIRE o.uri IS UNIQUE",
        "CREATE INDEX skill_label IF NOT EXISTS FOR (s:Skill) ON (s.label)",
        "CREATE INDEX occupation_label IF NOT EXISTS FOR (o:Occupation) ON (o.label)",
    ]
    for stmt in statements:
        client.run_write(stmt)


def clean_graph(client: Neo4jClient) -> None:
    """Remove ESCO nodes and relationships."""
    client.run_write("MATCH (n:Skill) DETACH DELETE n")
    client.run_write("MATCH (n:Occupation) DETACH DELETE n")


def load_skills(client: Neo4jClient, path: Path, batch_size: int) -> int:
    """Load Skill nodes from CSV."""
    df = pd.read_csv(path)
    uri_col = "conceptUri" if "conceptUri" in df.columns else "conceptURI"
    label_col = "preferredLabel"
    total = 0
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "uri": str(row[uri_col]),
                "label": str(row.get(label_col, "")),
                "description": str(row.get("description", "") or ""),
                "skill_type": str(row.get("skillType", "") or ""),
            },
        )
        if len(rows) >= batch_size:
            client.run_batch_write(
                """
                UNWIND $rows AS row
                MERGE (s:Skill {uri: row.uri})
                SET s.label = row.label, s.description = row.description,
                    s.skill_type = row.skill_type
                """,
                rows,
            )
            total += len(rows)
            rows = []
    if rows:
        client.run_batch_write(
            """
            UNWIND $rows AS row
            MERGE (s:Skill {uri: row.uri})
            SET s.label = row.label, s.description = row.description,
                s.skill_type = row.skill_type
            """,
            rows,
        )
        total += len(rows)
    return total


def load_occupations(client: Neo4jClient, path: Path, batch_size: int) -> int:
    """Load Occupation nodes from CSV."""
    df = pd.read_csv(path)
    uri_col = "conceptUri" if "conceptUri" in df.columns else "conceptURI"
    total = 0
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "uri": str(row[uri_col]),
                "label": str(row.get("preferredLabel", "")),
                "description": str(row.get("description", "") or ""),
                "isco_group": str(row.get("iscoGroup", "") or ""),
            },
        )
        if len(rows) >= batch_size:
            client.run_batch_write(
                """
                UNWIND $rows AS row
                MERGE (o:Occupation {uri: row.uri})
                SET o.label = row.label, o.description = row.description,
                    o.isco_group = row.isco_group
                """,
                rows,
            )
            total += len(rows)
            rows = []
    if rows:
        client.run_batch_write(
            """
            UNWIND $rows AS row
            MERGE (o:Occupation {uri: row.uri})
            SET o.label = row.label, o.description = row.description,
                o.isco_group = row.isco_group
            """,
            rows,
        )
        total += len(rows)
    return total


def load_skill_relations(client: Neo4jClient, path: Path, batch_size: int) -> int:
    """Load skill-to-skill relationships."""
    df = pd.read_csv(path)
    if "originalSkillUri" in df.columns:
        src_col, tgt_col = "originalSkillUri", "relatedSkillUri"
        rel_col = "relationType"
    else:
        src_col, tgt_col = "broaderUri", "narrowerUri"
        rel_col = None

    total = 0
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rel_type = str(row.get(rel_col, "related")) if rel_col else "narrower"
        rows.append(
            {
                "src": str(row[src_col]),
                "tgt": str(row[tgt_col]),
                "rel_type": rel_type,
            },
        )
        if len(rows) >= batch_size:
            client.run_batch_write(
                """
                UNWIND $rows AS row
                MATCH (a:Skill {uri: row.src}), (b:Skill {uri: row.tgt})
                MERGE (a)-[r:RELATED_TO]->(b)
                SET r.relation_type = row.rel_type
                """,
                rows,
            )
            total += len(rows)
            rows = []
    if rows:
        client.run_batch_write(
            """
            UNWIND $rows AS row
            MATCH (a:Skill {uri: row.src}), (b:Skill {uri: row.tgt})
            MERGE (a)-[r:RELATED_TO]->(b)
            SET r.relation_type = row.rel_type
            """,
            rows,
        )
        total += len(rows)
    return total


def load_occupation_skills(client: Neo4jClient, path: Path, batch_size: int) -> int:
    """Load occupation to skill requirements."""
    df = pd.read_csv(path)
    occ_col = "occupationUri" if "occupationUri" in df.columns else "occupationURI"
    skill_col = "skillUri" if "skillUri" in df.columns else "skillURI"
    essential_col = "essentialSkill" if "essentialSkill" in df.columns else None

    total = 0
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        essential = "essential"
        if essential_col:
            val = row.get(essential_col)
            essential = "essential" if str(val).lower() in ("true", "1", "yes") else "optional"
        rows.append(
            {
                "occ_uri": str(row[occ_col]),
                "skill_uri": str(row[skill_col]),
                "essential": essential,
            },
        )
        if len(rows) >= batch_size:
            client.run_batch_write(
                """
                UNWIND $rows AS row
                MATCH (o:Occupation {uri: row.occ_uri}), (s:Skill {uri: row.skill_uri})
                MERGE (o)-[r:REQUIRES_SKILL]->(s)
                SET r.relationship_type = row.essential
                """,
                rows,
            )
            total += len(rows)
            rows = []
    if rows:
        client.run_batch_write(
            """
            UNWIND $rows AS row
            MATCH (o:Occupation {uri: row.occ_uri}), (s:Skill {uri: row.skill_uri})
            MERGE (o)-[r:REQUIRES_SKILL]->(s)
            SET r.relationship_type = row.essential
            """,
            rows,
        )
        total += len(rows)
    return total


def validate_counts(client: Neo4jClient) -> None:
    """Validate node and relationship counts."""
    skills = client.run_query("MATCH (n:Skill) RETURN count(n) AS c")[0]["c"]
    occupations = client.run_query("MATCH (n:Occupation) RETURN count(n) AS c")[0]["c"]
    rels = client.run_query("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c")
    logger.info("Skills: %s, Occupations: %s", skills, occupations)
    for row in rels:
        logger.info("Relationship %s: %s", row["t"], row["c"])
    if skills < 10000:
        raise RuntimeError(f"Skill count too low: {skills}")
    if occupations < 3000:
        raise RuntimeError(f"Occupation count too low: {occupations}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Load ESCO into Neo4j")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level)
    data_dir = Path(settings.paths.esco_data_path)

    with Neo4jClient() as client:
        if not client.health_check():
            logger.error("Neo4j is not reachable")
            sys.exit(1)

        if args.validate_only:
            validate_counts(client)
            return

        if args.clean:
            logger.info("Cleaning existing ESCO graph")
            clean_graph(client)

        setup_constraints(client)

        skills_path = find_csv(data_dir, "skills_en.csv")
        occ_path = find_csv(data_dir, "occupations_en.csv")

        n_skills = load_skills(client, skills_path, args.batch_size)
        logger.info("Loaded %s skills", n_skills)

        n_occ = load_occupations(client, occ_path, args.batch_size)
        logger.info("Loaded %s occupations", n_occ)

        try:
            rel_path = find_csv(data_dir, "skillSkillRelations_en.csv")
        except FileNotFoundError:
            rel_path = find_csv(data_dir, "broaderRelationsSkillPillar_en.csv")
        n_rels = load_skill_relations(client, rel_path, args.batch_size)
        logger.info("Loaded %s skill relations", n_rels)

        try:
            occ_skill_path = find_csv(data_dir, "occupationSkillRelations_en.csv")
            n_os = load_occupation_skills(client, occ_skill_path, args.batch_size)
            logger.info("Loaded %s occupation-skill links", n_os)
        except FileNotFoundError:
            logger.warning("occupationSkillRelations_en.csv not found — skipping")

        validate_counts(client)


if __name__ == "__main__":
    main()
