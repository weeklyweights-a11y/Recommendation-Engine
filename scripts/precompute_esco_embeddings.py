"""Precompute ESCO skill embeddings for semantic linking."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_scripts_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_scripts_dir.parent))
import scripts._bootstrap  # noqa: F401

from config.logging import setup_logging
from config.settings import get_settings
from src.knowledge_graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Precompute ESCO skill embeddings")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level)
    matrix_path = Path(settings.embedding.esco_embeddings_path)
    index_path = Path(settings.embedding.esco_uri_index_path)

    if matrix_path.exists() and index_path.exists() and not args.force:
        logger.info("Embeddings already exist — use --force to rebuild")
        return

    start = time.time()
    with Neo4jClient() as client:
        rows = client.run_query(
            "MATCH (s:Skill) RETURN s.uri AS uri, s.label AS label, s.description AS description",
        )

    texts: list[str] = []
    uris: list[str] = []
    for row in rows:
        label = str(row.get("label", ""))
        desc = str(row.get("description", "") or "")
        texts.append(f"{label}: {desc}".strip())
        uris.append(str(row["uri"]))

    model = SentenceTransformer(settings.embedding.embedding_model)
    matrix = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    matrix = np.asarray(matrix, dtype=np.float32)

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(uris, handle)

    elapsed = time.time() - start
    logger.info("Saved %s embeddings to %s (%.1fs)", len(uris), matrix_path, elapsed)


if __name__ == "__main__":
    main()
