"""Backend parity and GeoParquet round trips (#235).

Every result is compared against the in-memory backend's answer for the
same inputs with `assert_catalogs_equal`, so any divergence in rows,
extras, time handling or CRS between `InMemoryGeoCatalog`,
`DuckDBGeoCatalog.from_memory` and `DuckDBGeoCatalog.open` fails here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import shapely

from geocatalog import InMemoryGeoCatalog, from_geoparquet, to_geoparquet
from geocatalog._src.base import GeoCatalog
from geocatalog._src.streaming import StreamingParquetWriter
from tests.conftest import assert_catalogs_equal, needs_duckdb


Convert = Callable[[InMemoryGeoCatalog], GeoCatalog]


def _catalog(
    rows: list[dict], *, crs: str = "EPSG:32629", kind: str = "raster"
) -> InMemoryGeoCatalog:
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return InMemoryGeoCatalog(gdf, kind=kind)  # type: ignore[arg-type]


def _scenes() -> InMemoryGeoCatalog:
    """Three scenes with the extras types catalogs carry in practice."""
    return _catalog(
        [
            {
                "geometry": shapely.box(0, 0, 100, 100),
                "start_time": pd.Timestamp("2024-01-01T10:00"),
                "end_time": pd.Timestamp("2024-01-01T10:05"),
                "filepath": "a.tif",
                "sensor": "S2A",
                "cloud": 12.5,
                "orbit": 7,
                "acquired": pd.Timestamp("2024-01-01T10:02", tz="UTC"),
            },
            {
                "geometry": shapely.box(80, 0, 180, 100),
                "start_time": pd.Timestamp("2024-01-02T10:00"),
                "end_time": pd.Timestamp("2024-01-02T10:05"),
                "filepath": "b.tif",
                "sensor": "S2B",
                "cloud": None,
                "orbit": 8,
                "acquired": pd.Timestamp("2024-01-02T10:02", tz="UTC"),
            },
            {
                "geometry": shapely.MultiPolygon(
                    [shapely.box(300, 0, 350, 50), shapely.box(360, 0, 400, 50)]
                ),
                "start_time": pd.Timestamp("2024-02-01T10:00"),
                "end_time": pd.Timestamp("2024-02-01T10:05"),
                "filepath": "c.tif",
                "sensor": "S2A",
                "cloud": 80.0,
                "orbit": 9,
                "acquired": pd.Timestamp("2024-02-01T10:02", tz="UTC"),
            },
        ]
    )


def _labels() -> InMemoryGeoCatalog:
    return _catalog(
        [
            {
                "geometry": shapely.box(50, 50, 150, 150),
                "start_time": pd.Timestamp("2024-01-01"),
                "end_time": pd.Timestamp("2024-01-31"),
                "filepath": "labels.gpkg",
                "label": "crop",
            },
            {
                "geometry": shapely.box(320, 10, 370, 40),
                "start_time": pd.Timestamp("2024-01-15"),
                "end_time": pd.Timestamp("2024-02-15"),
                "filepath": "labels2.gpkg",
                "label": "forest",
            },
        ],
        kind="vector",
    )


# ---------------------------------------------------------------------------
# Every operation agrees with the in-memory answer
# ---------------------------------------------------------------------------


def test_catalog_matches_memory(as_backend: Convert) -> None:
    assert_catalogs_equal(as_backend(_scenes()), _scenes())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": (0, 0, 50, 50), "crs": "EPSG:32629"},
        {"bounds": (90, 0, 95, 10), "crs": "EPSG:32629"},
        {"time": ("2024-01-02", "2024-03-01")},
        {"bounds": (1e6, 1e6, 2e6, 2e6), "crs": "EPSG:32629"},
    ],
    ids=["one-tile", "overlap", "time", "empty"],
)
def test_query_matches_memory(as_backend: Convert, kwargs: dict) -> None:
    assert_catalogs_equal(
        as_backend(_scenes()).query(**kwargs), _scenes().query(**kwargs)
    )


@pytest.mark.parametrize("spatial_only", [False, True])
def test_intersect_matches_memory(as_backend: Convert, spatial_only: bool) -> None:
    got = as_backend(_scenes()).intersect(_labels(), spatial_only=spatial_only)
    want = _scenes().intersect(_labels(), spatial_only=spatial_only)
    assert len(want) > 0
    assert_catalogs_equal(got, want)


def test_intersect_keeps_geometry_family(as_backend: Convert) -> None:
    got = as_backend(_scenes()).intersect(_labels())
    want = _scenes().intersect(_labels())
    families = {r.filepath: r.geometry.geom_type for r in got.iter_rows()}
    assert families == {r.filepath: r.geometry.geom_type for r in want.iter_rows()}


def test_union_matches_memory(as_backend: Convert) -> None:
    got = as_backend(_scenes()).union(_labels())
    want = _scenes().union(_labels())
    assert_catalogs_equal(got, want)


def test_iter_slices_match_memory(as_backend: Convert) -> None:
    got = sorted(as_backend(_scenes()).iter_slices(resolution=(10.0, 10.0)), key=repr)
    want = sorted(_scenes().iter_slices(resolution=(10.0, 10.0)), key=repr)
    assert got == want


def test_get_config_matches_memory(as_backend: Convert) -> None:
    got = as_backend(_scenes()).get_config()
    want = _scenes().get_config()
    assert got.keys() == want.keys()
    assert {k: v for k, v in got.items() if k != "engine"} == {
        k: v for k, v in want.items() if k != "engine"
    }


def test_empty_results_agree(as_backend: Convert) -> None:
    empty = as_backend(_scenes()).query(bounds=(1e6, 1e6, 2e6, 2e6), crs="EPSG:32629")
    assert len(empty) == 0
    assert list(empty.iter_rows()) == []
    assert list(empty.iter_slices(resolution=(10.0, 10.0))) == []
    assert empty.temporal_extent is None
    assert empty.crs == _scenes().crs


def test_geometry_column_under_another_name(as_backend: Convert) -> None:
    gdf = _scenes().gdf.rename_geometry("footprint")
    renamed = InMemoryGeoCatalog(gdf, kind="raster")
    assert_catalogs_equal(as_backend(renamed), _scenes())


# ---------------------------------------------------------------------------
# Time zones: aware inputs in any zone mean the same instants everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("catalog_tz", [None, "UTC", "America/New_York"])
@pytest.mark.parametrize("query_tz", [None, "Asia/Tokyo"])
def test_tz_by_tz(
    as_backend: Convert, catalog_tz: str | None, query_tz: str | None
) -> None:
    def ts(text: str, tz: str | None) -> pd.Timestamp:
        utc = pd.Timestamp(text, tz="UTC")
        return utc.tz_localize(None) if tz is None else utc.tz_convert(tz)

    rows = [
        {
            "geometry": shapely.box(i * 10, 0, i * 10 + 5, 5),
            "start_time": ts(f"2024-01-0{i + 1}T23:30", catalog_tz),
            "end_time": ts(f"2024-01-0{i + 1}T23:45", catalog_tz),
            "filepath": f"{i}.tif",
        }
        for i in range(3)
    ]
    window = (ts("2024-01-02T23:00", query_tz), ts("2024-01-03T00:00", query_tz))

    got = as_backend(_catalog(rows)).query(time=window)
    want = _catalog(rows).query(time=window)

    assert [r.filepath for r in want.iter_rows()] == ["1.tif"]
    assert_catalogs_equal(got, want)


# ---------------------------------------------------------------------------
# GeoParquet round trips
# ---------------------------------------------------------------------------


def _bbox_matches_geometry(path: Path) -> None:
    """The GeoParquet 1.1 covering bbox holds each row's geometry bounds."""
    table = pq.read_table(path, columns=["geometry", "bbox"])
    bounds = shapely.bounds(shapely.from_wkb(table.column("geometry").to_numpy()))
    bbox = table.column("bbox").combine_chunks()
    got = np.column_stack(
        [bbox.field(k).to_numpy() for k in ("xmin", "ymin", "xmax", "ymax")]
    )
    np.testing.assert_allclose(got, bounds)

    # Row-group statistics on the bbox leaves are what DuckDB prunes on.
    meta = pq.ParquetFile(path).metadata
    names = [meta.schema.column(i).path for i in range(meta.num_columns)]
    for leaf, reduce in (("bbox.xmin", np.min), ("bbox.xmax", np.max)):
        stats = meta.row_group(0).column(names.index(leaf)).statistics
        assert stats is not None and stats.has_min_max
        side = stats.min if leaf.endswith("min") else stats.max
        assert side == pytest.approx(reduce(got[:, 0 if leaf.endswith("xmin") else 2]))


@needs_duckdb
@pytest.mark.parametrize("engine", ["memory", "duckdb"])
def test_write_open_query_write_open(tmp_path: Path, engine: str) -> None:
    from geocatalog import open_catalog

    first = tmp_path / "first.parquet"
    to_geoparquet(_scenes(), first)
    _bbox_matches_geometry(first)

    opened = open_catalog(first, engine=engine)
    assert_catalogs_equal(opened, _scenes())

    subset = opened.query(time=("2024-01-01", "2024-01-31"))
    second = tmp_path / "second.parquet"
    to_geoparquet(subset, second)  # type: ignore[arg-type]
    _bbox_matches_geometry(second)

    reopened = open_catalog(second, engine=engine)
    assert_catalogs_equal(reopened, _scenes().query(time=("2024-01-01", "2024-01-31")))


@needs_duckdb
def test_streamed_artifact_reads_the_same_on_both_engines(tmp_path: Path) -> None:
    from geocatalog._src.duckdb_backend import DuckDBGeoCatalog

    path = tmp_path / "streamed.parquet"
    with StreamingParquetWriter(path, crs="EPSG:32629", kind="raster") as w:
        for row in _scenes().iter_rows():
            w.write_row(
                {
                    "filepath": row.filepath,
                    "geometry": row.geometry,
                    "start_time": row.interval.left,
                    "end_time": row.interval.right,
                    **row.extras,
                }
            )
    _bbox_matches_geometry(path)

    memory = from_geoparquet(path)
    duck = DuckDBGeoCatalog.open(path)
    assert_catalogs_equal(memory, _scenes())
    assert_catalogs_equal(duck, _scenes())


# ---------------------------------------------------------------------------
# get_config()["crs"] is one string everywhere (#249)
# ---------------------------------------------------------------------------

_LAEA = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m"


@pytest.mark.parametrize("crs", ["EPSG:32629", "epsg:4326", "OGC:CRS84", _LAEA])
def test_config_crs_is_stable_across_backends_and_round_trips(
    tmp_path: Path, crs: str
) -> None:
    import pyproj

    from geocatalog._src._schema import crs_config_string

    memory = _catalog(_rows_simple(), crs=crs)
    path = tmp_path / "c.parquet"
    to_geoparquet(memory, path)
    configs = [memory.get_config()["crs"], from_geoparquet(path).get_config()["crs"]]
    try:
        from geocatalog._src.duckdb_backend import DuckDBGeoCatalog

        configs.append(DuckDBGeoCatalog.open(path).get_config()["crs"])
    except ImportError:
        pass
    assert len(set(configs)) == 1
    # It names the same CRS, and an authority id (not a guess) when there is one.
    assert pyproj.CRS.from_user_input(configs[0]).equals(
        pyproj.CRS.from_user_input(crs)
    )
    if crs == _LAEA:
        assert configs[0].startswith("PROJCRS[")
    else:
        assert configs[0] == crs_config_string(crs) == crs.upper()


def _rows_simple() -> list[dict]:
    return [
        {
            "geometry": shapely.box(0, 0, 1, 1),
            "start_time": pd.Timestamp("2024-01-01"),
            "end_time": pd.Timestamp("2024-01-02"),
            "filepath": "a.tif",
        }
    ]
