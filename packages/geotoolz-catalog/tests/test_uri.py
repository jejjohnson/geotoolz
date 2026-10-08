"""One parser for paths and URIs, shared constants and extras hints (#247)."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pytest

from geocatalog._src._extras import install_hint, missing_extra, require_extra
from geocatalog._src._schema import BACKEND_TAGS, check_schema_versions, empty_frame
from geocatalog._src.base import CatalogSchemaError
from geocatalog._src.utils.uri import (
    DUCKDB_EXTENSIONS,
    FSSPEC_SCHEMES,
    parse_uri,
    query_params,
    with_query,
)


SRC = Path(__file__).resolve().parents[1] / "src" / "geocatalog"


# ---------------------------------------------------------------------------
# parse_uri
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "scheme", "local", "name"),
    [
        ("/data/scene.tif", "", True, "scene.tif"),
        ("relative/scene.tif", "", True, "scene.tif"),
        (Path("/data/scene.tif"), "", True, "scene.tif"),
        (PurePosixPath("a/b.nc"), "", True, "b.nc"),
        ("C:/data/scene.tif", "", True, "scene.tif"),
        ("C:\\data\\scene.tif", "", True, "scene.tif"),
        ("s3:catalog.parquet", "", True, "s3:catalog.parquet"),
        ("://missing-scheme", "", True, "missing-scheme"),
        ("file:///data/scene.tif", "file", True, "scene.tif"),
        ("s3://bucket/key/scene.tif", "s3", False, "scene.tif"),
        ("S3A://bucket/scene.tif", "s3a", False, "scene.tif"),
        ("HTTPS://host/x/scene.tif?sig=1", "https", False, "scene.tif"),
        ("hf://datasets/org/repo/a.parquet", "hf", False, "a.parquet"),
        ("abfss://c@acct.dfs.core.windows.net/a.tif", "abfss", False, "a.tif"),
        ("gs://bucket/store.zarr/", "gs", False, "store.zarr"),
    ],
)
def test_parse_uri_table(uri: object, scheme: str, local: bool, name: str) -> None:
    parsed = parse_uri(uri)  # type: ignore[arg-type]
    assert parsed.scheme == scheme
    assert parsed.is_local is local
    assert parsed.is_remote is (not local)
    assert parsed.name == name


@pytest.mark.parametrize(
    ("uri", "is_zarr"),
    [
        ("s3://bucket/store.zarr", True),
        ("s3://bucket/store.zarr/", True),  # trailing slash
        ("/data/store.zarr/", True),
        ("/data/store.zarr.tif", False),
        ("/data/scene.nc", False),
    ],
)
def test_zarr_detection(uri: str, is_zarr: bool) -> None:
    assert parse_uri(uri).is_zarr is is_zarr


def test_local_path_of_file_uris() -> None:
    assert parse_uri("file:///data/a%20b.tif").local_path() == Path("/data/a b.tif")
    assert parse_uri("file://localhost/x.tif").local_path() == Path("/x.tif")
    assert parse_uri("file://server/share/x.tif").local_path() == Path(
        "//server/share/x.tif"
    )
    assert parse_uri("s3://b/k").local_path() is None


def test_local_path_keeps_hash_and_question_marks() -> None:
    parsed = parse_uri("/data/run#3/scene?.tif")
    assert parsed.local_path() == Path("/data/run#3/scene?.tif")
    assert parsed.suffix == ".tif"


@pytest.mark.parametrize(
    ("uri", "gdal"),
    [
        ("s3://bucket/key.tif", "/vsis3/bucket/key.tif"),
        ("s3a://bucket/key.tif", "/vsis3/bucket/key.tif"),
        ("gs://bucket/key.tif", "/vsigs/bucket/key.tif"),
        ("az://container/key.tif", "/vsiaz/container/key.tif"),
        ("abfss://cont@acct.dfs.core.windows.net/k.tif", "/vsiaz/cont/k.tif"),
        ("https://host/k.tif", "/vsicurl/https://host/k.tif"),
        ("hf://datasets/x/k.tif", None),
        ("/local/k.tif", None),
    ],
)
def test_gdal_path(uri: str, gdal: str | None) -> None:
    assert parse_uri(uri).gdal_path() == gdal


def test_scheme_tables_cover_the_documented_schemes() -> None:
    assert {"s3", "s3a", "abfs", "abfss", "hf"} <= FSSPEC_SCHEMES
    assert DUCKDB_EXTENSIONS["s3a"] == "httpfs"
    assert DUCKDB_EXTENSIONS["abfss"] == "azure"


def test_query_helpers_round_trip() -> None:
    uri = "https://host/a.tif?b=2&sig=x&a=1"
    assert query_params(uri) == [("b", "2"), ("sig", "x"), ("a", "1")]
    assert with_query(uri, [("a", "1")]) == "https://host/a.tif?a=1"
    assert query_params("/local/a?b=1.tif") == []  # not a URL


# ---------------------------------------------------------------------------
# One parser, one Literal, one hint format
# ---------------------------------------------------------------------------


def _sources() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def test_url_parsing_lives_only_in_uri_module() -> None:
    offenders = [
        p.relative_to(SRC)
        for p in _sources()
        if p.name != "uri.py"
        and re.search(r"\burl(?:parse|split)\(|\.split\(\"://\"\)", p.read_text())
    ]
    assert offenders == []


def test_backend_literals_are_declared_once() -> None:
    pattern = re.compile(r'Literal\[\s*"(?:raster|memory)"')
    offenders = [
        p.relative_to(SRC)
        for p in _sources()
        if p.name != "_schema.py" and pattern.search(p.read_text())
    ]
    assert offenders == []
    assert BACKEND_TAGS == ("raster", "xarray", "vector")


def test_install_hints_name_the_distribution() -> None:
    offenders = [
        p.relative_to(SRC)
        for p in _sources()
        if re.search(r"pip install '?geocatalog\[", p.read_text())
    ]
    assert offenders == []


def test_missing_extra_message() -> None:
    err = missing_extra("`STACSource`", "stac", packages="pystac-client")
    assert isinstance(err, ImportError)
    assert str(err) == (
        "`STACSource` requires the [stac] extra; install with "
        "`pip install 'geotoolz-catalog[stac]'` (or `pip install pystac-client`)."
    )
    assert install_hint("duckdb") == "pip install 'geotoolz-catalog[duckdb]'"


def test_require_extra_imports_or_names_the_extra() -> None:
    assert require_extra("json", "none").dumps({}) == "{}"
    with pytest.raises(ImportError, match=r"geotoolz-catalog\[nope\]"):
        require_extra("geocatalog_no_such_module", "nope")


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def test_schema_version_checks() -> None:
    check_schema_versions("a", 0, 0, reader=0, can_migrate=False)
    check_schema_versions("a", 0, 0, reader=1, can_migrate=True)
    with pytest.raises(CatalogSchemaError, match="mixed"):
        check_schema_versions("a", 0, 1, reader=1, can_migrate=True)
    with pytest.raises(CatalogSchemaError, match="exceeds reader v0"):
        check_schema_versions("a", 1, 1, reader=0, can_migrate=True)
    with pytest.raises(CatalogSchemaError, match="geocatalog migrate"):
        check_schema_versions("a", 0, 0, reader=1, can_migrate=False)


def test_empty_frame_layout() -> None:
    gdf = empty_frame("EPSG:4326", {"filepath": "object", "flag": "bool"})
    assert list(gdf.columns) == ["filepath", "flag", "geometry"]
    assert gdf["flag"].dtype == bool
    assert gdf.index.name == "datetime"
    assert gdf.index.closed == "both"
    assert gdf.crs.to_epsg() == 4326
