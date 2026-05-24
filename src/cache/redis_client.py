"""Redis cache wrapper with graceful degradation."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Optional

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Thin Redis wrapper; all operations no-op when Redis is unavailable."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any = None
        self._available: Optional[bool] = None

    def _connect(self) -> bool:
        if self._available is False:
            return False
        if self._client is not None:
            return True
        try:
            import redis

            self._client = redis.from_url(
                self._settings.redis.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._client.ping()
            self._available = True
            return True
        except Exception as exc:
            logger.warning("Redis unavailable, caching disabled: %s", exc)
            self._client = None
            self._available = False
            return False

    def health_check(self) -> bool:
        """Return True if Redis responds to PING."""
        return self._connect()

    def get(self, key: str) -> str | None:
        if not self._connect():
            return None
        try:
            return self._client.get(key)
        except Exception as exc:
            logger.warning("Redis GET failed for %s: %s", key, exc)
            return None

    def get_json(self, key: str) -> Any | None:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Redis value for %s is not valid JSON", key)
            return None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        if not self._connect():
            return
        try:
            self._client.setex(key, ttl_seconds, value)
        except Exception as exc:
            logger.warning("Redis SET failed for %s: %s", key, exc)

    def set_json(self, key: str, payload: Any, ttl_seconds: int) -> None:
        self.set(key, json.dumps(payload), ttl_seconds)

    def delete(self, key: str) -> None:
        if not self._connect():
            return
        try:
            self._client.delete(key)
        except Exception as exc:
            logger.warning("Redis DELETE failed for %s: %s", key, exc)

    def delete_pattern(self, pattern: str) -> None:
        if not self._connect():
            return
        try:
            for key in self._client.scan_iter(match=pattern, count=200):
                self._client.delete(key)
        except Exception as exc:
            logger.warning("Redis delete_pattern failed for %s: %s", pattern, exc)


@lru_cache
def get_redis_cache() -> RedisCache:
    """Shared Redis cache instance."""
    return RedisCache()
