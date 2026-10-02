"""`DaskField` and the Dask helpers against the real ``dask`` package.

#178: ``select`` returns the materialised `xarray.DataArray` chip (not
another `DaskField`), so the patches feed the dense aggregations and
split → merge round-trips.

#179: ``from_zarr`` opens a variable of a zarr store, ``to_dask_bag``
computes, and ``to_delayed`` puts the field in the graph once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _helpers import ArrayField


dask = pytest.importorskip("dask")
pytest.importorskip("dask.array")
pytest.importorskip("dask.bag")
xr = pytest.importorskip("xarray")

from geopatcher import (
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)
from geopatcher._src.fields.dask import DaskField


def _cube(n: int = 12) -> xr.DataArray:
    cols = np.arange(n) + 0.5
    return xr.DataArray(
        np.arange(n * n, dtype=np.float32).reshape(n, n),
        dims=("y", "x"),
        coords={"y": 4600000.0 - 10.0 * cols, "x": 500000.0 + 10.0 * cols},
    ).chunk({"y": 5, "x": 5})


def _patcher(step: int = 4) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(4, 4)),
        sampler=SpatialRegularStride(step=step),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def test_select_returns_materialised_dataarray() -> None:
    da = _cube()
    chip = DaskField(da).select({"y": slice(4, 8), "x": slice(0, 4)})
    assert isinstance(chip, xr.DataArray)
    assert isinstance(chip.data, np.ndarray)  # computed, not a dask graph
    np.testing.assert_array_equal(chip.values, da.values[4:8, 0:4])
    np.testing.assert_array_equal(chip["x"].values, da["x"].values[0:4])
    np.testing.assert_array_equal(chip["y"].values, da["y"].values[4:8])


def test_select_writes_window_transform_when_georeferenced() -> None:
    pytest.importorskip("rioxarray")
    from rasterio import Affine
    from rasterio.windows import Window, transform as window_transform

    t = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4600000.0)
    da = (
        xr.DataArray(np.zeros((12, 12), dtype=np.float32), dims=("y", "x"))
        .rio.write_crs("EPSG:32630")
        .rio.write_transform(t)
        .chunk()
    )
    chip = DaskField(da).select({"y": slice(4, 8), "x": slice(6, 10)})
    assert chip.rio.transform() == window_transform(Window(6, 4, 4, 4), t)


@pytest.mark.parametrize("step", [2, 4])
def test_dask_field_split_merge(step: int) -> None:
    da = _cube()
    field = DaskField(da)
    patcher = _patcher(step)
    merged = patcher.merge(patcher.split(field), field.domain)
    np.testing.assert_allclose(np.asarray(merged), da.values)


def _graph(tasks: list[object]) -> dict[object, object]:
    graph: dict[object, object] = {}
    for task in tasks:
        graph.update(dict(task.__dask_graph__()))  # type: ignore[attr-defined]
    return graph


def test_from_zarr_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    da = _cube().rename("sst")
    store = tmp_path / "cube.zarr"
    da.to_dataset().to_zarr(store)
    field = DaskField.from_zarr(store)
    assert isinstance(field.array, xr.DataArray)
    assert field.array.name == "sst"
    np.testing.assert_array_equal(field.array.values, da.values)
    patcher = _patcher(4)
    merged = patcher.merge(patcher.split(field), field.domain)
    np.testing.assert_allclose(np.asarray(merged), da.values)


def test_from_zarr_selects_named_variable(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    da = _cube()
    ds = xr.Dataset({"a": da, "b": da * 2})
    store = tmp_path / "two.zarr"
    ds.to_zarr(store)
    field = DaskField.from_zarr(store, var="b")
    np.testing.assert_array_equal(field.array.values, da.values * 2)
    with pytest.raises(ValueError, match="pass var="):
        DaskField.from_zarr(store)
    with pytest.raises(KeyError, match="'c' is not a data variable"):
        DaskField.from_zarr(store, var="c")


@pytest.mark.parametrize("kind", ["array", "dask"])
def test_to_dask_bag_compute(kind: str) -> None:
    field = (
        DaskField(_cube())
        if kind == "dask"
        else ArrayField(np.arange(144, dtype=np.float32).reshape(12, 12))
    )
    patcher = _patcher(4)
    bag = patcher.to_dask_bag(field)
    n = patcher.n_anchors(field)
    assert bag.npartitions == n
    assert bag.count().compute(scheduler="sync") == n
    patches = bag.compute(scheduler="threads")
    assert [p.anchor for p in patches] == patcher.anchors(field)
    merged = patcher.merge(patches, field.domain)
    np.testing.assert_allclose(np.asarray(merged), _values(field))


def _values(field: object) -> np.ndarray:
    if isinstance(field, DaskField):
        return np.asarray(field.array.values)
    return np.asarray(field.array)  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ["array", "dask"])
def test_to_delayed_compute(kind: str) -> None:
    field = (
        DaskField(_cube())
        if kind == "dask"
        else ArrayField(np.arange(144, dtype=np.float32).reshape(12, 12))
    )
    patcher = _patcher(4)
    tasks = patcher.to_delayed(field)
    assert len(tasks) == patcher.n_anchors(field)
    patches = dask.compute(*tasks, scheduler="threads")
    merged = patcher.merge(patches, field.domain)
    np.testing.assert_allclose(np.asarray(merged), _values(field))

    sums = dask.compute(
        *patcher.to_delayed(field, operator=lambda p: float(np.asarray(p.data).sum())),
        scheduler="sync",
    )
    assert sum(sums) == pytest.approx(float(_values(field).sum()))


def test_to_delayed_embeds_field_once() -> None:
    field = ArrayField(np.arange(144, dtype=np.float32).reshape(12, 12))
    patcher = _patcher(4)
    tasks = patcher.to_delayed(field)
    graph = _graph(tasks)
    field_keys = [k for k in graph if str(k).startswith("ArrayField-")]
    # One shared field node (+ one patcher node) and one task per anchor —
    # not a copy of the field inlined into every task.
    assert len(field_keys) == 1
    assert len(graph) == len(tasks) + 2
