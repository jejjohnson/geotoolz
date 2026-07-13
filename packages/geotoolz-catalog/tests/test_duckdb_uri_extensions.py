"""Unit tests for DuckDB URI extension setup."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import geocatalog._src.duckdb_backend as duckdb_backend
from geocatalog import DuckDBGeoCatalog


class _FakeRelation:
    pass


class _FakeConnection:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.closed = False

    def execute(self, command: str) -> None:
        self.commands.append(command)

    def close(self) -> None:
        self.closed = True

    def sql(self, query: str, *, params: dict[str, Any]) -> _FakeRelation:
        return _FakeRelation()


@pytest.fixture
def fake_duckdb(monkeypatch: pytest.MonkeyPatch) -> _FakeConnection:
    con = _FakeConnection()
    monkeypatch.setattr(
        duckdb_backend,
        "duckdb",
        SimpleNamespace(
            connect=lambda: con,
            BinderException=Exception,
            IOException=Exception,
        ),
    )
    monkeypatch.setattr(
        duckdb_backend,
        "_read_geoparquet_crs",
        lambda source, *, default, strict=False: default,
    )
    monkeypatch.setattr(
        duckdb_backend,
        "_read_backend_tag",
        lambda con, source, *, default, partitioned=False, strict=False: default,
    )
    monkeypatch.setattr(
        duckdb_backend,
        "_check_schema_version",
        lambda con, source, *, partitioned=False: None,
    )
    return con


@pytest.mark.parametrize(
    ("source", "scheme"),
    [
        ("s3://bucket/cat.parquet", "s3"),
        ("HTTPS://example.test/cat.parquet", "https"),
        ("az://container/cat.parquet", "az"),
        (Path("cat.parquet"), None),
        ("", None),
        ("://missing-scheme", None),
        ("C:/data/cat.parquet", None),
        ("C:\\data\\cat.parquet", None),
        # `name:foo.parquet` is a local path in shell semantics — `urlsplit`
        # would happily parse a ``scheme`` out of it, so guard against
        # triggering remote-extension installs for those.
        ("s3:catalog.parquet", None),
        ("foo:bar.parquet", None),
    ],
)
def test_scheme(source: str | Path, scheme: str | None) -> None:
    assert duckdb_backend._scheme(source) == scheme


def test_open_does_not_install_extensions_for_colon_local_paths(
    fake_duckdb: _FakeConnection,
) -> None:
    """`s3:catalog.parquet` is a local path, not a URI — no httpfs install."""
    cat = DuckDBGeoCatalog.open("s3:catalog.parquet")
    assert isinstance(cat, DuckDBGeoCatalog)
    assert fake_duckdb.commands == ["LOAD spatial"]


def test_open_warns_when_uri_source_and_no_crs(
    fake_duckdb: _FakeConnection,
) -> None:
    """URI sources can't auto-detect CRS; the user should be nudged."""
    with pytest.warns(UserWarning, match="cannot auto-detect CRS"):
        DuckDBGeoCatalog.open("s3://bucket/cat.parquet")


def test_open_no_warning_when_uri_source_and_explicit_crs(
    fake_duckdb: _FakeConnection,
) -> None:
    """No warning when the caller heeds the docstring and passes `crs=`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DuckDBGeoCatalog.open("s3://bucket/cat.parquet", crs="EPSG:4326")


def test_open_no_warning_for_local_path_without_crs(
    fake_duckdb: _FakeConnection,
) -> None:
    """Local paths use the real metadata reader; no warning needed."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DuckDBGeoCatalog.open(Path("cat.parquet"))


@pytest.mark.parametrize(
    ("source", "extension_commands"),
    [
        ("s3://bucket/cat.parquet", ["INSTALL httpfs", "LOAD httpfs"]),
        ("gs://bucket/cat.parquet", ["INSTALL httpfs", "LOAD httpfs"]),
        ("gcs://bucket/cat.parquet", ["INSTALL httpfs", "LOAD httpfs"]),
        ("https://example.test/cat.parquet", ["INSTALL httpfs", "LOAD httpfs"]),
        ("http://example.test/cat.parquet", ["INSTALL httpfs", "LOAD httpfs"]),
        ("r2://bucket/cat.parquet", ["INSTALL httpfs", "LOAD httpfs"]),
        ("hf://datasets/org/cat.parquet", ["INSTALL httpfs", "LOAD httpfs"]),
        ("az://container/cat.parquet", ["INSTALL azure", "LOAD azure"]),
        ("azure://container/cat.parquet", ["INSTALL azure", "LOAD azure"]),
        ("foo://bucket/cat.parquet", []),
        (Path("cat.parquet"), []),
    ],
)
def test_open_loads_extension_for_supported_uri_schemes(
    source: str | Path,
    extension_commands: list[str],
    fake_duckdb: _FakeConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_source: list[str] = []

    def fake_sql(query: str, *, params: dict[str, Any]) -> _FakeRelation:
        assert query == "SELECT * FROM read_parquet($src, hive_partitioning = $hive)"
        captured_source.append(params["src"])
        return _FakeRelation()

    monkeypatch.setattr(fake_duckdb, "sql", fake_sql)

    cat = DuckDBGeoCatalog.open(source)

    assert isinstance(cat, DuckDBGeoCatalog)
    assert fake_duckdb.commands == [
        "LOAD spatial",
        *extension_commands,
    ]
    assert captured_source == [str(source)]


def test_strict_remote_uri_without_crs_raises(fake_duckdb: _FakeConnection) -> None:
    """strict=True can't silently default CRS for URIs it can't introspect."""
    from geocatalog import CatalogMetadataError

    with pytest.raises(CatalogMetadataError, match="Pass crs="):
        DuckDBGeoCatalog.open("s3://bucket/cat.parquet", strict=True)


def test_strict_remote_uri_with_explicit_crs_opens(
    fake_duckdb: _FakeConnection,
) -> None:
    cat = DuckDBGeoCatalog.open(
        "s3://bucket/cat.parquet", crs="EPSG:32629", strict=True
    )
    assert cat.crs == "EPSG:32629"
