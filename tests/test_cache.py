"""Tests for SQLite cache layer."""

import time

import pytest

from app.data.cache import SQLiteCache
from app.data.exceptions import CacheError


def test_set_and_get(cache: SQLiteCache):
    cache.set("quotes", "RELIANCE.NS", '{"price": 1450.25}', ttl_seconds=60)
    assert cache.get("quotes", "RELIANCE.NS") == '{"price": 1450.25}'


def test_get_missing_key_returns_none(cache: SQLiteCache):
    assert cache.get("quotes", "MISSING.NS") is None


def test_delete_removes_entry(cache: SQLiteCache):
    cache.set("quotes", "INFY.NS", "data")
    cache.delete("quotes", "INFY.NS")
    assert cache.get("quotes", "INFY.NS") is None


def test_clear_namespace(cache: SQLiteCache):
    cache.set("quotes", "A.NS", "a")
    cache.set("quotes", "B.NS", "b")
    cache.set("historical", "A.NS:1y", "hist")
    cache.clear_namespace("quotes")
    assert cache.get("quotes", "A.NS") is None
    assert cache.get("quotes", "B.NS") is None
    assert cache.get("historical", "A.NS:1y") == "hist"


def test_ttl_expiry(cache: SQLiteCache):
    cache.set("quotes", "RELIANCE.NS", "expired", ttl_seconds=1)
    assert cache.get("quotes", "RELIANCE.NS") == "expired"
    time.sleep(1.1)
    assert cache.get("quotes", "RELIANCE.NS") is None


def test_expired_entry_remains_available_via_get_allow_stale(cache: SQLiteCache):
    cache.set("quotes", "RELIANCE.NS", '{"price": 1450.25}', ttl_seconds=1)
    time.sleep(1.1)
    assert cache.get("quotes", "RELIANCE.NS") is None
    assert cache.get_allow_stale("quotes", "RELIANCE.NS") == '{"price": 1450.25}'


def test_set_without_ttl_persists(cache: SQLiteCache):
    cache.set("quotes", "RELIANCE.NS", "persistent", ttl_seconds=None)
    assert cache.get("quotes", "RELIANCE.NS") == "persistent"


def test_upsert_overwrites_value(cache: SQLiteCache):
    cache.set("quotes", "RELIANCE.NS", "v1")
    cache.set("quotes", "RELIANCE.NS", "v2")
    assert cache.get("quotes", "RELIANCE.NS") == "v2"


def test_purge_expired(cache: SQLiteCache):
    cache.set("quotes", "OLD.NS", "old", ttl_seconds=1)
    cache.set("quotes", "NEW.NS", "new", ttl_seconds=300)
    time.sleep(1.1)
    removed = cache.purge_expired()
    assert removed >= 1
    assert cache.get("quotes", "OLD.NS") is None
    assert cache.get("quotes", "NEW.NS") == "new"


def test_creates_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "cache.db"
    cache = SQLiteCache(db_path=str(db_path))
    cache.set("test", "key", "value")
    assert db_path.exists()


def test_cache_implements_protocol(cache: SQLiteCache):
    from app.data.interfaces import CacheProvider

    assert isinstance(cache, CacheProvider)
