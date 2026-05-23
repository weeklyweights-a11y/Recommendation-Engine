"""Neo4j driver wrapper for knowledge graph operations."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Optional

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Sync Neo4j client with retries and helpers."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        """Initialize client from settings or explicit credentials."""
        settings = get_settings()
        self._uri = uri or settings.neo4j.neo4j_uri
        self._user = user or settings.neo4j.neo4j_user
        self._password = password or settings.neo4j.neo4j_password
        self._driver: Driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )

    def close(self) -> None:
        """Close the underlying driver."""
        self._driver.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def run_query(self, cypher: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Execute a read query and return records as dicts."""
        start = time.perf_counter()
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            records = [record.data() for record in result]
        logger.debug("Query completed in %.3fs", time.perf_counter() - start)
        return records

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def run_write(self, cypher: str, params: Optional[dict[str, Any]] = None) -> None:
        """Execute a write query."""
        with self._driver.session() as session:
            session.run(cypher, params or {})

    def run_batch_write(self, cypher: str, batch_params: list[dict[str, Any]]) -> None:
        """Execute a parameterized write for a batch using UNWIND."""
        if not batch_params:
            return
        with self._driver.session() as session:
            session.run(cypher, {"rows": batch_params})

    def health_check(self) -> bool:
        """Return True if Neo4j is reachable."""
        try:
            self.run_query("RETURN 1 AS ok")
            return True
        except (ServiceUnavailable, SessionExpired, OSError):
            return False

    @contextmanager
    def session_scope(self) -> Any:
        """Provide a raw Neo4j session context."""
        session: Session = self._driver.session()
        try:
            yield session
        finally:
            session.close()

    def __enter__(self) -> "Neo4jClient":
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        self.close()
