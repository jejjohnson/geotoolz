"""Protocol conformance, `iter_slices` edge footprints, `CatalogDomain` (#232)."""

from __future__ import annotations

import warnings

import geopandas as gpd
import pandas as pd
import pyproj
import pytest
import shapely

from geocatalog import CatalogDomain, GeoCatalog, InMemoryGeoCatalog
from geocatalog._src.base import GeoCatalog as _ProtocolCls


def _mem(geoms: list[shapely.Geometry | None]) -> InMemoryGeoCatalog:
    n = len(geoms)
    gdf = gpd.GeoDataFrame(
        {
            "geometry": geoms,
            "start_time": [pd.Timestamp("2024-01-01")] * n,
            "end_time": [pd.Timestamp("2024-01-02")] * n,
            "filepath": [f"f{i}.tif" for i in range(n)],
        },
        geometry="geometry",
        crs="EPSG:32629",
    )
    return InMemoryGeoCatalog(gdf, backend="raster")


def _as(backend: str, mem: InMemoryGeoCatalog) -> GeoCatalog:
    if backend == "memory":
        return mem
    pytest.importorskip("duckdb")
    from geocatalog import DuckDBGeoCatalog

    return DuckDBGeoCatalog.from_memory(mem)


BACKENDS = ["memory", "duckdb"]


def _protocol_members() -> set[str]:
    members = {
        name
        for name in vars(_ProtocolCls)
        if not name.startswith("_") and name not in {"gdf"}
    }
    members |= set(_ProtocolCls.__annotations__) - {"gdf"}
    return members | {"__len__"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_implements_every_protocol_member(backend: str) -> None:
    cat = _as(backend, _mem([shapely.box(0, 0, 10, 10)]))
    missing = sorted(m for m in _protocol_members() if not hasattr(cat, m))
    assert missing == []
    assert isinstance(cat, GeoCatalog)


@pytest.mark.parametrize("backend", BACKENDS)
def test_crs_is_pyproj(backend: str) -> None:
    cat = _as(backend, _mem([shapely.box(0, 0, 10, 10)]))
    assert isinstance(cat.crs, pyproj.CRS)
    assert cat.crs.to_epsg() == 32629


@pytest.mark.parametrize("backend", BACKENDS)
def test_iter_slices_point_and_line_footprints(backend: str) -> None:
    cat = _as(
        backend,
        _mem(
            [
                shapely.Point(100.0, 200.0),
                shapely.LineString([(0.0, 50.0), (40.0, 50.0)]),  # zero height
                shapely.box(0, 0, 20, 30),
            ]
        ),
    )
    slices = list(cat.iter_slices(resolution=(10.0, 10.0)))
    assert [s.bounds for s in slices] == [
        (95.0, 195.0, 105.0, 205.0),
        (0.0, 45.0, 40.0, 55.0),
        (0.0, 0.0, 20.0, 30.0),
    ]
    assert slices[0].shape == (1, 1)


def test_iter_slices_skips_missing_footprint_with_one_warning() -> None:
    cat = _mem([shapely.box(0, 0, 10, 10), None, shapely.Polygon()])
    with pytest.warns(UserWarning, match="skipped 2 row"):
        slices = list(cat.iter_slices(resolution=(1.0, 1.0)))
    assert len(slices) == 1


def test_iter_slices_no_warning_when_all_footprints_present() -> None:
    cat = _mem([shapely.box(0, 0, 10, 10)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(list(cat.iter_slices(resolution=(1.0, 1.0)))) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_catalog_domain_over_either_backend(backend: str) -> None:
    cat = _as(backend, _mem([shapely.box(0, 0, 10, 10), shapely.Point(50, 50)]))
    domain = CatalogDomain(cat, resolution=(10.0, 10.0))
    assert domain.crs == pyproj.CRS.from_epsg(32629)
    assert len(domain) == 2
    assert len(domain.slices()) == 2


def test_catalog_domain_crs_does_not_materialise_duckdb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("duckdb")
    from geocatalog import DuckDBGeoCatalog

    duck = DuckDBGeoCatalog.from_memory(_mem([shapely.box(0, 0, 10, 10)]))

    def _boom(self: object) -> None:
        raise AssertionError("gdf materialised")

    monkeypatch.setattr(DuckDBGeoCatalog, "gdf", property(_boom))
    assert CatalogDomain(duck, resolution=(1.0, 1.0)).crs.to_epsg() == 32629
