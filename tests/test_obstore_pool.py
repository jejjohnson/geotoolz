"""Tests for the geotoolz obstore client pool."""

from __future__ import annotations

import pytest


pytest.importorskip("obstore")

from geotoolz import _obstore


@pytest.fixture(autouse=True)
def _isolate_pool(monkeypatch):
    """Hermetic pool state for every test.

    Three things:

    1. Clear the local pool before and after each test so identity
       tests don't see leakage between test ordering.
    2. Force ``_try_shared_pool`` to return ``None`` so the local
       pool is always under test — otherwise ``clear_obstore_pool``
       is a no-op when geocatalog or geopatcher is installed and
       owns the shared cache, which would make assertions like
       ``test_clear_pool_drops_all_entries`` flaky depending on the
       Python path.
    3. Strip endpoint env vars that would otherwise contaminate the
       pool key assertions below.
    """
    monkeypatch.setattr(_obstore, "_try_shared_pool", lambda: None)
    for var in (
        "AWS_S3_ENDPOINT",
        "AWS_ENDPOINT_URL",
        "GOOGLE_SERVICE_ENDPOINT",
        "AZURE_STORAGE_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
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


# --- unsupported scheme -------------------------------------------------


def test_get_obstore_rejects_unsupported_scheme():
    with pytest.raises(ValueError, match="unsupported scheme"):
        _obstore.get_obstore("ftp://example.com/foo")
