"""SQLite-backed cache layer (MVP).

Design note: implement against CacheProvider protocol so PostgreSQL/Redis
can replace SQLite without changing agent code.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from app.data.exceptions import CacheError
from app.data.interfaces import CacheProvider

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    namespace   TEXT NOT NULL,
    cache_key   TEXT NOT NULL,
    value       TEXT NOT NULL,
    expires_at  REAL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (namespace, cache_key)
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries (expires_at);
"""


class SQLiteCache:
    """SQLite cache implementation with TTL support."""

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise CacheError(f"Failed to initialize cache database: {exc}") from exc

    def get(self, namespace: str, key: str) -> str | None:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT value, expires_at
                    FROM cache_entries
                    WHERE namespace = ? AND cache_key = ?
                    """,
                    (namespace, key),
                ).fetchone()

                if row is None:
                    return None

                expires_at = row["expires_at"]
                if expires_at is not None and expires_at <= time.time():
                    # Keep the row so get_allow_stale() can still recover it
                    # after Yahoo rate limits. purge_expired() handles cleanup.
                    return None

                return row["value"]
        except sqlite3.Error as exc:
            raise CacheError(f"Cache read failed for {namespace}/{key}: {exc}") from exc

    def get_allow_stale(self, namespace: str, key: str) -> str | None:
        """Return a cached value even if expired (does not delete stale entries)."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT value
                    FROM cache_entries
                    WHERE namespace = ? AND cache_key = ?
                    """,
                    (namespace, key),
                ).fetchone()
                if row is None:
                    return None
                return row["value"]
        except sqlite3.Error as exc:
            raise CacheError(f"Stale cache read failed for {namespace}/{key}: {exc}") from exc

    def set(
        self,
        namespace: str,
        key: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds is not None else None
        now = time.time()

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO cache_entries (namespace, cache_key, value, expires_at, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, cache_key) DO UPDATE SET
                        value = excluded.value,
                        expires_at = excluded.expires_at,
                        created_at = excluded.created_at
                    """,
                    (namespace, key, value, expires_at, now),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Cache write failed for {namespace}/{key}: {exc}") from exc

    def delete(self, namespace: str, key: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?",
                    (namespace, key),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Cache delete failed for {namespace}/{key}: {exc}") from exc

    def clear_namespace(self, namespace: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
                conn.commit()
        except sqlite3.Error as exc:
            raise CacheError(f"Cache clear failed for namespace {namespace}: {exc}") from exc

    def purge_expired(self) -> int:
        """Remove all expired entries. Returns count of deleted rows."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (time.time(),),
                )
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as exc:
            raise CacheError(f"Cache purge failed: {exc}") from exc


def create_cache(db_path: str) -> CacheProvider:
    """Factory for the default cache provider."""
    return SQLiteCache(db_path=db_path)
