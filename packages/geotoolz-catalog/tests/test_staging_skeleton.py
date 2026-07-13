"""Smoke tests for the scaffolded `geocatalog.staging` surface."""

from __future__ import annotations

import pytest

import geocatalog.staging as staging_ns
from geocatalog._src.staging import LocalCache, stage


class TestReexports:
    def test_subnamespace_reexports(self) -> None:
        assert staging_ns.stage is stage
        assert staging_ns.LocalCache is LocalCache


class TestLocalCache:
    def test_default_construction(self) -> None:
        c = LocalCache()
        assert c.root is None
        assert c.ttl_days is None

    def test_custom_construction(self) -> None:
        c = LocalCache(root="/tmp/cat", ttl_days=30)
        assert c.root == "/tmp/cat"
        assert c.ttl_days == 30


class TestStage:
    def test_rejects_non_inmemory_catalog(self) -> None:
        # `stage()` is implemented; behaviour coverage lives in
        # `tests/test_staging.py`. Skeleton locks the guard.
        with pytest.raises(TypeError, match="InMemoryGeoCatalog"):
            stage(catalog=object(), dest="/tmp/staged")  # type: ignore[arg-type]
