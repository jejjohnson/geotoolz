"""Pickle round trips: the YAML-safe axes and patchers, and everything the
ML recipes hand to worker processes.

`SpatialCustom`, `TemporalFold`, and `SpatialLearned` carry closures and
are intentionally excluded.

`IndexedPatchView` (every cache mode), `PatchCache` and every `Field`
adapter must survive ``pickle`` — spawn / forkserver DataLoader workers
and Grain's multiprocess data sources pickle them — and a pickled field
must split to the same patches as the original (#197, epic #172).
"""

from __future__ import annotations

import multiprocessing
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _helpers import (
    init_view_worker,
    make_raster_field,
    make_rasterio_reader_field,
    read_view_item,
)
from test_adapter_matrix import ADAPTERS, _chip_transform

from geopatcher import (
    IndexedPatchView,
    PatchCache,
    SpatialBoxcar,
    SpatialHann,
    SpatialKNNGraph,
    SpatialMean,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialSum,
    SpatialTukey,
    TemporalCausalBoxcar,
    TemporalExponentialDecay,
    TemporalFixedLookback,
    TemporalLookbackHorizon,
    TemporalMean,
    TemporalPatcher,
    TemporalRegularStride,
)


@pytest.mark.parametrize(
    "op",
    [
        SpatialRectangular(size=(8, 8)),
        SpatialRegularStride(step=8),
        SpatialBoxcar(),
        SpatialHann(),
        SpatialTukey(alpha=0.5),
        SpatialSum(),
        SpatialMean(),
        SpatialOverlapAdd(),
        TemporalFixedLookback(length=5),
        TemporalLookbackHorizon(lookback=3, horizon=2),
        TemporalRegularStride(step=2),
        TemporalCausalBoxcar(),
        TemporalExponentialDecay(tau=2.0),
        TemporalMean(),
    ],
)
def test_axis_pickle_roundtrip(op) -> None:
    blob = pickle.dumps(op)
    clone = pickle.loads(blob)
    assert type(clone) is type(op)
    # assert_equal: aggregation configs carry a NaN default ``fill_value``.
    np.testing.assert_equal(clone.get_config(), op.get_config())


class TestSpatialPatcherPickle:
    def test_roundtrip(self) -> None:
        sp = SpatialPatcher(
            geometry=SpatialRectangular(size=(8, 8)),
            sampler=SpatialRegularStride(step=8),
            window=SpatialHann(),
            aggregation=SpatialOverlapAdd(),
        )
        clone = pickle.loads(pickle.dumps(sp))
        np.testing.assert_equal(clone.get_config(), sp.get_config())


class TestTemporalPatcherPickle:
    def test_roundtrip(self) -> None:
        tp = TemporalPatcher(
            geometry=TemporalFixedLookback(length=4),
            sampler=TemporalRegularStride(step=2),
            window=TemporalCausalBoxcar(),
            aggregation=TemporalMean(),
        )
        clone = pickle.loads(pickle.dumps(tp))
        assert clone.get_config() == tp.get_config()


def _patcher(size: int = 16, boundary: str = "drop") -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(size, size), boundary=boundary),  # type: ignore[arg-type]
        sampler=SpatialRegularStride(step=size),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def _assert_same_patches(got: list[Any], ref: list[Any]) -> None:
    assert [p.anchor for p in got] == [p.anchor for p in ref]
    for g, r in zip(got, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(g.data), np.asarray(r.data))
        assert np.asarray(g.data).dtype == np.asarray(r.data).dtype
        if hasattr(r.data, "crs"):
            assert g.data.crs == r.data.crs
        if hasattr(r.data, "transform") or hasattr(r.data, "rio"):
            assert _chip_transform(g.data) == _chip_transform(r.data)


# --------------------------------------------------------------------------
# IndexedPatchView
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["off", "memory", "patch_cache"])
def test_indexed_patch_view_roundtrip(mode: str, tmp_path: Path) -> None:
    field = make_raster_field(64)
    patcher = _patcher()
    cache: Any = {"off": False, "memory": True}.get(mode)
    if mode == "patch_cache":
        cache = PatchCache(tmp_path / "cache", field_id="scene")
    view = IndexedPatchView(patcher, field, cache=cache)
    _ = [view[i] for i in range(3)]  # populate the cache / bind field_id

    clone = pickle.loads(pickle.dumps(view))

    assert clone.anchors == view.anchors
    assert clone._cache == {}  # in-memory entries are process-local
    clone[0]  # the recreated lock works
    clone.clear_cache()
    if mode == "patch_cache":
        # The bound identity travels: the clone hits what the parent wrote
        # without re-deriving the field id.
        assert clone._field_id == view._field_id is not None
        hits = clone._disk_cache.stats()["hits"]
        _ = [clone[i] for i in range(3)]
        assert clone._disk_cache.stats()["hits"] == hits + 3
    _assert_same_patches(list(clone), list(patcher.split(field)))


def test_indexed_patch_view_over_temporal_patcher_pickles() -> None:
    from geopatcher import (
        TemporalCausalBoxcar,
        TemporalMean,
        TemporalMultiScale,
        TemporalPatcher,
        TemporalRegularStride,
    )

    tp = TemporalPatcher(
        geometry=TemporalMultiScale(scales=[3, 8]),
        sampler=TemporalRegularStride(step=5),
        window=TemporalCausalBoxcar(),
        aggregation=TemporalMean(),
    )
    series = np.arange(40.0)
    clone = pickle.loads(pickle.dumps(IndexedPatchView(tp, series, cache=True)))
    expected = list(tp.split(series))
    assert len(clone) == len(expected)
    for got, ref in zip(clone, expected, strict=True):
        np.testing.assert_array_equal(got.data, ref.data)


# --------------------------------------------------------------------------
# PatchCache
# --------------------------------------------------------------------------


def test_patch_cache_roundtrip_serves_split(tmp_path: Path) -> None:
    field = make_raster_field(32)
    patcher = _patcher(8)
    cache = PatchCache(tmp_path, field_id="scene")
    ref = list(patcher.split(field, cache=cache))  # populate

    clone = pickle.loads(pickle.dumps(cache))
    assert clone.stats()["entries"] == len(ref)
    _assert_same_patches(list(patcher.split(field, cache=clone)), ref)
    assert clone.stats()["hits"] == len(ref)


# --------------------------------------------------------------------------
# Field adapters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", list(ADAPTERS))
def test_field_adapter_roundtrip(adapter: str, tmp_path: Path) -> None:
    field = ADAPTERS[adapter](tmp_path).field
    clone = pickle.loads(pickle.dumps(field))
    patcher = _patcher(boundary="pad")
    _assert_same_patches(list(patcher.split(clone)), list(patcher.split(field)))


def _knn_patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialKNNGraph(k=2),
        sampler=SpatialRandom(n_samples=3, seed=0),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def _point_fields() -> dict[str, Any]:
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    from geopatcher._src.fields.geopandas import GeoPandasField

    pts = [shapely.Point(float(i), float(i % 3)) for i in range(6)]
    gdf = gpd.GeoDataFrame({"v": np.arange(6.0)}, geometry=pts, crs="EPSG:4326")
    fields: dict[str, Any] = {"GeoPandasField": GeoPandasField(gdf, as_points=True)}
    try:
        import xarray as xr

        pytest.importorskip("xvec")
        from geopatcher._src.fields.xvec import XvecField

        geoms = np.array(pts, dtype=object)
        ds = xr.Dataset(
            {"v": (("geometry",), np.arange(6.0))}, coords={"geometry": geoms}
        )
        fields["XvecField"] = XvecField(
            ds.xvec.set_geom_indexes("geometry", crs="EPSG:4326")
        )
    except pytest.skip.Exception:
        pass
    return fields


@pytest.mark.parametrize("name", ["GeoPandasField", "XvecField"])
def test_point_field_roundtrip(name: str) -> None:
    fields = _point_fields()
    if name not in fields:
        pytest.skip(f"{name} extra not installed")
    field = fields[name]
    clone = pickle.loads(pickle.dumps(field))
    patcher = _knn_patcher()
    got, ref = list(patcher.split(clone)), list(patcher.split(field))
    assert [p.anchor for p in got] == [p.anchor for p in ref]
    for g, r in zip(got, ref, strict=True):
        payload = getattr(r.data, "gdf", None)
        if payload is not None:
            assert g.data.gdf.equals(payload)
        else:
            assert g.data.ds.identical(r.data.ds)


# --------------------------------------------------------------------------
# A real spawn worker
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["geotensor", "rasterio"])
def test_view_read_in_spawn_worker(source: str, tmp_path: Path) -> None:
    """A spawn-context pool (no inherited memory) reads ``view[i]``.

    Mirrors a torch DataLoader / Grain worker: the view is pickled into
    each worker once (the pool initializer), then indexed there.
    """
    if source == "geotensor":
        field = make_raster_field(48)
    else:
        field = make_rasterio_reader_field(tmp_path / "src.tif")
    patcher = _patcher()
    view = IndexedPatchView(patcher, field, cache=True)
    expected = list(patcher.split(field))

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2, initializer=init_view_worker, initargs=(view,)) as pool:
        results = pool.map(read_view_item, range(len(view)))

    assert len(results) == len(expected)
    for (values, transform, crs), ref in zip(results, expected, strict=True):
        np.testing.assert_array_equal(values, np.asarray(ref.data))
        assert transform == tuple(ref.data.transform)[:6]
        assert crs == str(ref.data.crs)
