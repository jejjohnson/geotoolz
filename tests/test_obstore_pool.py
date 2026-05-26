"""Tests for the geotoolz obstore client pool."""

from __future__ import annotations

import pytest


pytest.importorskip("obstore")

from geotoolz import _obstore


@pytest.fixture(autouse=True)
def _clear_pool():
    _obstore.clear_obstore_pool()
    yield
    _obstore.clear_obstore_pool()


# --- key construction ---------------------------------------------------


def test_pool_key_s3_basic():
    key = _obstore._pool_key("s3://my-bucket/path/to/file.tif")
    assert key[0] == "s3"
    assert key[1] == "my-bucket"


def test_pool_key_gs_basic():
    key = _obstore._pool_key("gs://my-bucket/path/to/file.tif")
    assert key == ("gs", "my-bucket", None, None)


def test_pool_key_https_basic():
    key = _obstore._pool_key("https://example.com/data/file.tif")
    assert key == ("https", "example.com", None, None)


def test_pool_key_includes_aws_region_from_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-3")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    key = _obstore._pool_key("s3://bucket-a/key")
    assert key[2] == "eu-west-3"


def test_pool_key_different_buckets_distinct():
    a = _obstore._pool_key("s3://bucket-a/key")
    b = _obstore._pool_key("s3://bucket-b/key")
    assert a != b


# --- pool identity ------------------------------------------------------


def test_get_obstore_returns_same_instance_for_same_key():
    a = _obstore.get_obstore("https://example.com/foo")
    b = _obstore.get_obstore("https://example.com/bar")
    assert a is b


def test_get_obstore_different_hosts_get_different_instances():
    a = _obstore.get_obstore("https://example.com/foo")
    b = _obstore.get_obstore("https://other.example.com/bar")
    assert a is not b


def test_clear_pool_drops_all_entries():
    a = _obstore.get_obstore("https://example.com/foo")
    _obstore.clear_obstore_pool()
    b = _obstore.get_obstore("https://example.com/foo")
    assert a is not b


# --- shared-pool soft import -------------------------------------------


def test_shared_pool_returns_none_when_no_sibling_packages(monkeypatch):
    """With no sibling packages installed, ``_try_shared_pool`` is ``None``.

    Verified at module-import time by simulating both siblings being
    absent — replaces both module imports with ``ImportError`` to keep
    the test deterministic regardless of what's actually on the path.
    """
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name in (
            "geocatalog._src.objstore",
            "geopatcher._src.objstore",
        ):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert _obstore._try_shared_pool() is None


# --- unsupported scheme -------------------------------------------------


def test_get_obstore_rejects_unsupported_scheme():
    with pytest.raises(ValueError, match="unsupported scheme"):
        _obstore.get_obstore("ftp://example.com/foo")
