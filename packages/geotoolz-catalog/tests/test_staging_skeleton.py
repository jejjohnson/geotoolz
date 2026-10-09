"""Smoke tests for the scaffolded `geocatalog.staging` surface."""

from __future__ import annotations

import pytest

import geocatalog.staging as staging_ns
from geocatalog._src.staging.stage import stage


class TestReexports:
    def test_subnamespace_reexports(self) -> None:
        assert staging_ns.stage is stage
        assert staging_ns.__all__ == ["stage"]


class TestStage:
    def test_rejects_non_inmemory_catalog(self) -> None:
        # `stage()` is implemented; behaviour coverage lives in
        # `tests/test_staging.py`. Skeleton locks the guard.
        with pytest.raises(TypeError, match="InMemoryGeoCatalog"):
            stage(catalog=object(), dest="/tmp/staged")  # type: ignore[arg-type]
