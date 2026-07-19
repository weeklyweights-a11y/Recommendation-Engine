"""Map free-text skills to ESCO taxonomy nodes."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rapidfuzz import fuzz, process

from config.settings import Settings, get_settings
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.schemas import LinkedSkill

logger = logging.getLogger(__name__)

_label_index: dict[str, str] = {}
_uri_labels: dict[str, str] = {}
_embedding_model: Any = None
_skill_matrix: Optional[np.ndarray] = None
_uri_by_index: list[str] = []


def _register_label(index: dict[str, str], label: str, uri: str) -> None:
    """Add a lowercase label alias to the index."""
    key = label.strip().lower()
    if key:
        index[key] = uri


def _load_label_index(client: Neo4jClient) -> None:
    """Build lowercase label -> uri index from Neo4j preferred and alt labels."""
    global _label_index, _uri_labels
    if _label_index:
        return
    records = client.run_query(
        "MATCH (s:Skill) RETURN s.uri AS uri, s.label AS label, s.alt_labels AS alt_labels",
    )
    index: dict[str, str] = {}
    uri_labels: dict[str, str] = {}
    for row in records:
        label = str(row.get("label", "")).strip()
        uri = str(row.get("uri", "")).strip()
        if not label or not uri:
            continue
        uri_labels[uri] = label
        _register_label(index, label, uri)
        alt_labels = row.get("alt_labels") or []
        if isinstance(alt_labels, str):
            alt_labels = [part.strip() for part in alt_labels.replace("|", "\n").splitlines() if part.strip()]
        for alt in alt_labels:
            _register_label(index, str(alt), uri)
    _label_index = index
    _uri_labels = uri_labels
    logger.info("Loaded %s skill label aliases for exact matching", len(_label_index))


def _load_embeddings(settings: Settings) -> None:
    """Load precomputed skill embedding matrix."""
    global _embedding_model, _skill_matrix, _uri_by_index
    if _skill_matrix is not None:
        return
    matrix_path = Path(settings.embedding.esco_embeddings_path)
    index_path = Path(settings.embedding.esco_uri_index_path)
    if not matrix_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            "ESCO embeddings not found — run scripts/precompute_esco_embeddings.py first",
        )
    from sentence_transformers import SentenceTransformer

    _skill_matrix = np.load(matrix_path)
    with index_path.open(encoding="utf-8") as handle:
        _uri_by_index = json.load(handle)
    _embedding_model = SentenceTransformer(settings.embedding.embedding_model)


def reset_linker_caches() -> None:
    """Clear in-memory linker caches (for tests or after graph reload)."""
    global _label_index, _uri_labels, _embedding_model, _skill_matrix, _uri_by_index
    _label_index = {}
    _uri_labels = {}
    _embedding_model = None
    _skill_matrix = None
    _uri_by_index = []


def link_skill_to_esco(free_text: str) -> Optional[LinkedSkill]:
    """Alias for link_skill (PROJECT.md naming)."""
    return link_skill(free_text)


def link_skills_batch(free_texts: list[str]) -> list[Optional[LinkedSkill]]:
    """Alias for link_skills (PROJECT.md naming)."""
    return link_skills(free_texts)


def link_skill(free_text: str) -> Optional[LinkedSkill]:
    """Map free-text to best ESCO skill: exact, fuzzy, then semantic."""
    from src.cache.redis_client import get_redis_cache

    settings = get_settings()
    text = free_text.strip()
    if not text:
        return None

    redis_key = f"esco:link:{hashlib.sha256(text.lower().encode()).hexdigest()}"
    cache = get_redis_cache()
    cached = cache.get_json(redis_key)
    if cached is not None:
        if not cached:
            return None
        return LinkedSkill.model_validate(cached)

    with Neo4jClient() as client:
        _load_label_index(client)

    lower = text.lower()
    if lower in _label_index:
        uri = _label_index[lower]
        linked = LinkedSkill(
            esco_uri=uri,
            esco_label=_uri_labels.get(uri, text),
            match_type="exact",
            confidence=1.0,
        )
        cache.set_json(redis_key, linked.model_dump(), settings.cache.esco_link_ttl_seconds)
        return linked

    if _label_index and len(lower) >= 4:
        match = process.extractOne(
            lower,
            _label_index.keys(),
            scorer=fuzz.token_set_ratio,
        )
        if match and match[1] / 100.0 >= settings.skill_graph.fuzzy_match_threshold:
            matched_label = match[0]
            if len(matched_label) >= 4 and len(matched_label) >= len(lower) * 0.4:
                uri = _label_index[matched_label]
                linked = LinkedSkill(
                    esco_uri=uri,
                    esco_label=_uri_labels.get(uri, matched_label),
                    match_type="fuzzy",
                    confidence=match[1] / 100.0,
                )
                cache.set_json(redis_key, linked.model_dump(), settings.cache.esco_link_ttl_seconds)
                return linked

    try:
        _load_embeddings(settings)
    except FileNotFoundError:
        logger.warning("Semantic linking unavailable — embeddings missing")
        cache.set_json(redis_key, {}, settings.cache.esco_link_ttl_seconds)
        return None

    assert _embedding_model is not None and _skill_matrix is not None
    query_vec = _embedding_model.encode([text], normalize_embeddings=True)[0]
    scores = _skill_matrix @ query_vec
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < settings.skill_graph.semantic_match_threshold:
        cache.set_json(redis_key, {}, settings.cache.esco_link_ttl_seconds)
        return None
    uri = _uri_by_index[best_idx]
    label = _uri_labels.get(uri, uri)
    linked = LinkedSkill(
        esco_uri=uri,
        esco_label=label,
        match_type="semantic",
        confidence=best_score,
    )
    cache.set_json(redis_key, linked.model_dump(), settings.cache.esco_link_ttl_seconds)
    return linked


def link_skills(free_texts: list[str]) -> list[Optional[LinkedSkill]]:
    """Batch link multiple free-text skills."""
    settings = get_settings()
    results: list[Optional[LinkedSkill]] = [None] * len(free_texts)
    pending_semantic: list[tuple[int, str]] = []

    with Neo4jClient() as client:
        _load_label_index(client)

    for idx, text in enumerate(free_texts):
        stripped = text.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower in _label_index:
            uri = _label_index[lower]
            results[idx] = LinkedSkill(
                esco_uri=uri,
                esco_label=_uri_labels.get(uri, stripped),
                match_type="exact",
                confidence=1.0,
            )
            continue
        if _label_index and len(lower) >= 4:
            match = process.extractOne(
                lower,
                _label_index.keys(),
                scorer=fuzz.token_set_ratio,
            )
            if match and match[1] / 100.0 >= settings.skill_graph.fuzzy_match_threshold:
                matched_label = match[0]
                if len(matched_label) >= 4 and len(matched_label) >= len(lower) * 0.4:
                    uri = _label_index[matched_label]
                    results[idx] = LinkedSkill(
                        esco_uri=uri,
                        esco_label=_uri_labels.get(uri, matched_label),
                        match_type="fuzzy",
                        confidence=match[1] / 100.0,
                    )
                    continue
        pending_semantic.append((idx, stripped))

    if not pending_semantic:
        return results

    try:
        _load_embeddings(settings)
    except FileNotFoundError:
        return results

    assert _embedding_model is not None and _skill_matrix is not None
    texts = [t for _, t in pending_semantic]
    query_matrix = _embedding_model.encode(texts, normalize_embeddings=True)
    for (idx, _), query_vec in zip(pending_semantic, query_matrix):
        scores = _skill_matrix @ query_vec
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < settings.skill_graph.semantic_match_threshold:
            continue
        uri = _uri_by_index[best_idx]
        results[idx] = LinkedSkill(
            esco_uri=uri,
            esco_label=_uri_labels.get(uri, uri),
            match_type="semantic",
            confidence=best_score,
        )
    return results
