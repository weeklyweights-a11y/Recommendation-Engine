"""BM25 job retrieval via Elasticsearch."""

import logging
from typing import Any, Optional

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ApiError, TransportError

from config.settings import Settings, get_settings
from src.api.schemas.recommendation import ScoredJob

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Retrieve jobs using Elasticsearch BM25 scoring."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize ES client and index name from settings."""
        self._settings = settings or get_settings()
        self._client = Elasticsearch(self._settings.elasticsearch.elasticsearch_url)
        self._index = self._settings.elasticsearch.es_index_name

    def health_check(self) -> bool:
        """Return True if cluster is up and index exists."""
        try:
            if not self._client.ping():
                return False
            return self._client.indices.exists(index=self._index)
        except (ApiError, TransportError, OSError):
            return False

    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        allowed_job_ids: Optional[set[str]] = None,
    ) -> list[ScoredJob]:
        """Run BM25 search and return normalized ScoredJob results."""
        k = top_k if top_k is not None else self._settings.retrieval.bm25_top_k
        merged_filters = dict(filters or {})
        if allowed_job_ids is not None:
            cap = self._settings.retrieval.es_terms_filter_max
            if len(allowed_job_ids) <= cap:
                merged_filters["job_ids"] = list(allowed_job_ids)
        body = self._build_query(query_text, k, merged_filters)
        try:
            response = self._client.search(index=self._index, body=body)
        except (ApiError, TransportError) as exc:
            logger.exception("Elasticsearch search failed for query=%r", query_text)
            raise RuntimeError(f"BM25 search failed: {exc}") from exc

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            return []

        raw_scores = [float(hit["_score"]) for hit in hits]
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score <= 0:
            max_score = 1.0

        results: list[ScoredJob] = []
        for hit, raw in zip(hits, raw_scores):
            source = hit.get("_source", {})
            job_id = source.get("job_id") or hit.get("_id")
            results.append(
                ScoredJob(
                    job_id=str(job_id),
                    score=raw / max_score,
                    source="bm25",
                    title=source.get("title"),
                    company=source.get("company"),
                ),
            )
        return results

    def _build_query(
        self,
        query_text: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> dict[str, Any]:
        """Build Elasticsearch query DSL."""
        must: list[dict[str, Any]] = [
            {
                "multi_match": {
                    "query": query_text,
                    "fields": [
                        "title^2.0",
                        "description",
                        "company",
                        "location",
                        "skills_extracted",
                    ],
                    "type": "best_fields",
                },
            },
        ]
        filter_clauses: list[dict[str, Any]] = []
        if remote := filters.get("remote_type"):
            filter_clauses.append({"term": {"remote_type": remote}})
        if experience := filters.get("experience_level"):
            filter_clauses.append({"term": {"experience_level": experience}})
        if location := filters.get("location"):
            filter_clauses.append({"term": {"location": location}})
        if posted_range := filters.get("posted_date"):
            filter_clauses.append({"range": {"posted_date": posted_range}})
        if job_ids := filters.get("job_ids"):
            filter_clauses.append({"terms": {"job_id": job_ids}})

        query: dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": must,
                    "filter": filter_clauses,
                },
            },
        }
        return query
