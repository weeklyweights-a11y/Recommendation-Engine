"""Redis cache client tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.cache.redis_client import RedisCache


@patch("redis.from_url")
def test_redis_get_json(mock_from_url) -> None:
    client = MagicMock()
    client.ping.return_value = True
    client.get.return_value = '{"ok": true}'
    mock_from_url.return_value = client

    cache = RedisCache()
    assert cache.get_json("key") == {"ok": True}


@patch("redis.from_url", side_effect=ConnectionError("down"))
def test_redis_graceful_degradation(_mock_from_url) -> None:
    cache = RedisCache()
    assert cache.get("key") is None
    cache.set("key", "value", 60)  # should not raise


@patch("src.cache.invalidation.delete_recommendations_for_candidate")
@patch("src.cache.invalidation.get_sync_session")
@patch("src.cache.invalidation.get_redis_cache")
def test_invalidate_candidate_recommendation_cache(
    mock_get_cache,
    mock_session,
    mock_delete,
) -> None:
    from uuid import uuid4

    from src.cache.invalidation import invalidate_candidate_recommendation_cache

    cache = MagicMock()
    mock_get_cache.return_value = cache
    mock_session.return_value.__enter__.return_value = MagicMock()

    candidate_id = uuid4()
    invalidate_candidate_recommendation_cache(candidate_id)

    cache.delete_pattern.assert_any_call(f"recs:{candidate_id}*")
    mock_delete.assert_called_once()
