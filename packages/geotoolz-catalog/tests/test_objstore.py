"""Tests for the process-global obstore client pool."""

from __future__ import annotations

import pytest


# Skip the whole module when obstore isn't installed — the pool's
# public surface raises ImportError on first call, and several tests
# invoke get_obstore directly.
pytest.importorskip("obstore")

from geocatalog._src import objstore


@pytest.fixture(autouse=True)
def _clear_pool():
    """Each test starts with an empty pool — keeps key collisions independent."""
    objstore.clear_obstore_pool()
    yield
    objstore.clear_obstore_pool()


# --- key construction ----------------------------------------------------


def test_pool_key_s3_basic():
    key = objstore._pool_key("s3://my-bucket/path/to/file.tif")
    assert key[0] == "s3"
    assert key[1] == "my-bucket"


def test_pool_key_gs_basic():
    key = objstore._pool_key("gs://my-bucket/path/to/file.tif")
    assert key == ("gs", "my-bucket", None, None)


def test_pool_key_https_basic():
    key = objstore._pool_key("https://example.com/data/file.tif")
    assert key == ("https", "example.com", None, None)


def test_pool_key_includes_aws_region_from_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-3")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    key = objstore._pool_key("s3://bucket-a/key")
    assert key[2] == "eu-west-3"


def test_pool_key_falls_back_to_default_region(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    key = objstore._pool_key("s3://bucket-a/key")
    assert key[2] == "us-east-1"


def test_pool_key_different_buckets_distinct():
    a = objstore._pool_key("s3://bucket-a/key")
    b = objstore._pool_key("s3://bucket-b/key")
    assert a != b


# --- pool identity -------------------------------------------------------


def test_get_obstore_returns_same_instance_for_same_key():
    # Use HTTPStore since it doesn't require credentials at construction.
    a = objstore.get_obstore("https://example.com/foo")
    b = objstore.get_obstore("https://example.com/bar")  # same host → same key
    assert a is b


def test_get_obstore_different_hosts_get_different_instances():
    a = objstore.get_obstore("https://example.com/foo")
    b = objstore.get_obstore("https://other.example.com/bar")
    assert a is not b


def test_clear_pool_drops_all_entries():
    a = objstore.get_obstore("https://example.com/foo")
    objstore.clear_obstore_pool()
    b = objstore.get_obstore("https://example.com/foo")
    assert a is not b  # fresh client after clear


# --- LRU eviction --------------------------------------------------------


def test_lru_eviction_at_maxsize():
    """Inserting past the cap evicts the least-recently-used entry."""
    objstore.set_obstore_pool_maxsize(2)
    try:
        a = objstore.get_obstore("https://host-a.example.com/x")
        objstore.get_obstore("https://host-b.example.com/x")
        objstore.get_obstore("https://host-c.example.com/x")  # evicts a
        # a was evicted: a fresh fetch builds a new instance
        a2 = objstore.get_obstore("https://host-a.example.com/x")
        assert a2 is not a
    finally:
        objstore.set_obstore_pool_maxsize(64)


def test_lru_touch_on_access():
    objstore.set_obstore_pool_maxsize(2)
    try:
        a = objstore.get_obstore("https://host-a.example.com/x")
        objstore.get_obstore("https://host-b.example.com/x")
        # touch a so it becomes the most-recently-used
        objstore.get_obstore("https://host-a.example.com/x")
        # now insert c — should evict b, not a
        objstore.get_obstore("https://host-c.example.com/x")
        a2 = objstore.get_obstore("https://host-a.example.com/x")
        assert a2 is a
    finally:
        objstore.set_obstore_pool_maxsize(64)


def test_set_maxsize_rejects_zero():
    with pytest.raises(ValueError, match=">= 1"):
        objstore.set_obstore_pool_maxsize(0)


# --- unsupported scheme --------------------------------------------------


def test_get_obstore_rejects_unsupported_scheme():
    with pytest.raises(ValueError, match="unsupported scheme"):
        objstore.get_obstore("ftp://example.com/foo")
