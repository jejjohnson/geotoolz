"""Tests for transient I/O retry/backoff handling."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry
from loguru import logger
from tenacity import wait_none

from geocatalog import (
    GeoSlice,
    InMemoryGeoCatalog,
    build_raster_catalog,
    from_geoparquet,
    load_raster,
    to_geoparquet,
)
from geocatalog._src import (
    duckdb_backend as duckdb_module,
    parquet as parquet_module,
    raster as raster_module,
    retry as retry_module,
)


REGEX = r"S2_T29SND_(?P<date>\d{8}).*\.tif"


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry_module, "_RETRY_WAIT", wait_none())


@pytest.fixture
def retry_log_sink() -> io.StringIO:
    buf = io.StringIO()
    handler_id = logger.add(buf, level="WARNING", format="{level} | {message}")
    logger.enable("geocatalog")
    try:
        yield buf
    finally:
        logger.disable("geocatalog")
        logger.remove(handler_id)


def _toy_catalog() -> InMemoryGeoCatalog:
    gdf = gpd.GeoDataFrame(
        {
            "filepath": ["a.tif"],
            "geometry": [shapely.geometry.box(0, 0, 100, 100)],
            "start_time": [pd.Timestamp("2024-01-01")],
            "end_time": [pd.Timestamp("2024-01-02")],
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")


def test_from_geoparquet_retries_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_log_sink: io.StringIO,
) -> None:
    path = tmp_path / "cat.parquet"
    to_geoparquet(_toy_catalog(), path)
    original = parquet_module.gpd.read_parquet
    attempts = 0

    def flaky_read_parquet(path: Path) -> gpd.GeoDataFrame:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("temporary read failure")
        return original(path)

    monkeypatch.setattr(parquet_module.gpd, "read_parquet", flaky_read_parquet)

    recovered = from_geoparquet(path, retries=2)

    assert len(recovered) == 1
    assert attempts == 3
    assert "WARNING | Transient I/O error" in retry_log_sink.getvalue()


def test_from_geoparquet_exhausts_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cat.parquet"
    to_geoparquet(_toy_catalog(), path)
    attempts = 0

    def always_fails(path: Path) -> gpd.GeoDataFrame:
        nonlocal attempts
        attempts += 1
        raise OSError("permanent read failure")

    monkeypatch.setattr(parquet_module.gpd, "read_parquet", always_fails)

    with pytest.raises(OSError, match="permanent read failure"):
        from_geoparquet(path, retries=3)

    assert attempts == 4


def test_retries_zero_disables_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cat.parquet"
    to_geoparquet(_toy_catalog(), path)
    attempts = 0

    def always_fails(path: Path) -> gpd.GeoDataFrame:
        nonlocal attempts
        attempts += 1
        raise OSError("first failure")

    monkeypatch.setattr(parquet_module.gpd, "read_parquet", always_fails)

    with pytest.raises(OSError, match="first failure"):
        from_geoparquet(path, retries=0)

    assert attempts == 1


def test_build_raster_catalog_retries_metadata_open(
    utm29_tile_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = utm29_tile_factory((500_000, 4_000_000, 500_320, 4_000_320), "20240115")
    original = raster_module.rasterio.open
    attempts = 0

    def flaky_open(*args: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("temporary raster metadata failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(raster_module.rasterio, "open", flaky_open)

    catalog = build_raster_catalog([path], filename_regex=REGEX)

    assert len(catalog) == 1
    assert attempts == 3


def test_load_raster_retries_data_open(
    utm29_tile_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = utm29_tile_factory(
        (500_000, 4_000_000, 500_320, 4_000_320), "20240115", value=7
    )
    catalog = build_raster_catalog([path], filename_regex=REGEX)
    original = raster_module.rasterio.open
    attempts = 0

    def flaky_open(*args: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("temporary raster data failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(raster_module.rasterio, "open", flaky_open)
    sl = GeoSlice(
        bounds=(500_000, 4_000_000, 500_320, 4_000_320),
        interval=pd.Interval(
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-31"),
            closed="both",
        ),
        resolution=(10.0, 10.0),
        crs="EPSG:32629",
    )

    tensor = load_raster(catalog, sl, retries=2)

    assert attempts == 3
    np.testing.assert_array_equal(tensor.values, 7)


def test_duckdb_open_retries_read_parquet_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relation = object()

    class FakeConnection:
        def __init__(self) -> None:
            self.attempts = 0

        def execute(self, *args: object, **kwargs: object) -> None:
            # httpfs auto-load (INSTALL/LOAD) goes through `con.execute`;
            # the retry path under test runs `con.sql` for `read_parquet`.
            return None

        def sql(self, *args: object, **kwargs: object) -> object:
            self.attempts += 1
            if self.attempts < 3:
                raise OSError("temporary duckdb read failure")
            return relation

    class FakeDuckDB:
        def __init__(self, con: FakeConnection) -> None:
            self.con = con

        def connect(self) -> FakeConnection:
            return self.con

    con = FakeConnection()
    monkeypatch.setattr(duckdb_module, "_require_duckdb", lambda: FakeDuckDB(con))
    monkeypatch.setattr(duckdb_module, "_ensure_spatial", lambda con: None)
    monkeypatch.setattr(
        duckdb_module,
        "_check_schema_version",
        lambda con, src, *, partitioned=False: None,
    )

    catalog = duckdb_module.DuckDBGeoCatalog.open(
        "s3://bucket/catalog.parquet",
        backend="raster",
        crs="EPSG:4326",
        retries=2,
    )

    assert catalog.relation is relation
    assert con.attempts == 3


# ---------------------------------------------------------------------------
# Direct unit tests of `retry_transient_io` — exception filtering, validation,
# and the success-fast path.
# ---------------------------------------------------------------------------


def test_retry_rejects_negative_retries() -> None:
    with pytest.raises(ValueError, match="retries must be >= 0"):
        retry_module.retry_transient_io(lambda: None, retries=-1)


def test_retry_success_first_try() -> None:
    calls = 0

    def ok() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert retry_module.retry_transient_io(ok, retries=3) == "ok"
    assert calls == 1


def test_retry_does_not_retry_file_not_found() -> None:
    attempts = 0

    def missing() -> None:
        nonlocal attempts
        attempts += 1
        raise FileNotFoundError("nope")

    with pytest.raises(FileNotFoundError):
        retry_module.retry_transient_io(missing, retries=5)

    # Fatal OSError subclasses should not be retried — paying 6x wall time
    # for a path that will never exist is a foot-gun.
    assert attempts == 1


@pytest.mark.parametrize(
    "exc_type",
    [PermissionError, IsADirectoryError, NotADirectoryError, InterruptedError],
)
def test_retry_does_not_retry_other_fatal_oserrors(
    exc_type: type[OSError],
) -> None:
    attempts = 0

    def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise exc_type("nope")

    with pytest.raises(exc_type):
        retry_module.retry_transient_io(fail, retries=5)

    assert attempts == 1


def test_retry_handles_rasterio_io_error() -> None:
    from rasterio.errors import RasterioIOError

    attempts = 0

    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RasterioIOError("HTTP 503")
        return "ok"

    assert retry_module.retry_transient_io(flaky, retries=3) == "ok"
    assert attempts == 3


def test_retry_handles_urllib3_read_timeout() -> None:
    urllib3_exc = pytest.importorskip("urllib3.exceptions")
    attempts = 0

    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise urllib3_exc.ReadTimeoutError(None, "url", "timed out")
        return "ok"

    assert retry_module.retry_transient_io(flaky, retries=2) == "ok"
    assert attempts == 2


def test_retry_does_not_swallow_non_io_exception() -> None:
    attempts = 0

    def boom() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("not an IO error")

    with pytest.raises(ValueError, match="not an IO error"):
        retry_module.retry_transient_io(boom, retries=5)

    assert attempts == 1
