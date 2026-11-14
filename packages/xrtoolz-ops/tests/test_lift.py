"""`xrops.lift`: a geoops geometry operator per item over a cube (#363)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


geoarray = pytest.importorskip("geoarray")
gg = pytest.importorskip("geoops.geom")
from geoarray import GeoArray

import xrops


TRANSFORM = (0.5, 0.0, 10.0, 0.0, -0.5, 50.0)


def _items(n: int = 4) -> list[GeoArray]:
    rng = np.random.default_rng(0)
    return [
        GeoArray(
            rng.normal(size=(2, 8, 10)),
            TRANSFORM,
            "EPSG:4326",
            dims=("band", "y", "x"),
            fill_value=np.nan,
        )
        for _ in range(n)
    ]


def _cube(items: list[GeoArray]) -> xr.DataArray:
    stacked = xr.concat([g.to_xarray() for g in items], dim="time")
    return stacked.assign_coords(time=np.arange(len(items)))


OPS = {
    "reproject": lambda: gg.Reproject(dst_crs="EPSG:3857"),
    "resample": lambda: gg.Resample(resolution=(1.0, 1.0)),
    "gradient": lambda: gg.Gradient2D(nan_policy="adaptive"),
}


def _no_compute(*args, **kwargs):
    raise AssertionError("the cube was computed while the graph was built")


@pytest.mark.parametrize("name", list(OPS))
@pytest.mark.parametrize("lazy", [False, True])
def test_the_lift_equals_the_per_item_result_restacked(name, lazy) -> None:
    items = _items()
    cube = _cube(items)
    want = xr.concat([OPS[name]()(g).to_xarray() for g in items], dim="time")
    if lazy:
        dask = pytest.importorskip("dask")
        cube = cube.chunk({"time": 2})
        with dask.config.set(scheduler=_no_compute):
            out = xrops.lift(OPS[name](), dim="time")(cube)
        assert out.chunks is not None
        with dask.config.set(scheduler="sync"):
            out = out.compute()
    else:
        out = xrops.lift(OPS[name](), dim="time")(cube)
    assert out.dims[0] == "time"
    np.testing.assert_array_equal(out["time"].values, np.arange(4))
    np.testing.assert_allclose(out.values, want.values, equal_nan=True)
    for c in ("x", "y"):
        np.testing.assert_allclose(out[c].values, want[c].values)


def test_one_task_per_time_chunk() -> None:
    pytest.importorskip("dask")
    cube = _cube(_items(6)).chunk({"time": 2})
    out = xrops.lift(gg.Gradient2D(), dim="time")(cube)
    assert out.chunks[0] == (2, 2, 2)
    graph = out.data.__dask_graph__()
    lift_layer = graph.layers[out.data.name]
    assert len(lift_layer) == 3  # one task per time chunk, nothing per item


def test_a_spatially_chunked_cube_raises_naming_the_fix() -> None:
    pytest.importorskip("dask")
    cube = _cube(_items()).chunk({"time": 1, "y": 4})
    with pytest.raises(ValueError, match=r"\.chunk\(\{'y': -1\}\)"):
        xrops.lift(gg.Gradient2D(), dim="time")(cube)


def test_config_round_trip_and_bad_inputs() -> None:
    op = xrops.lift(gg.Reproject(dst_crs="EPSG:3857"), dim="time")
    rebuilt = type(op)(**op.get_config())
    assert rebuilt.get_config()["dim"] == "time"
    assert type(rebuilt.op) is gg.Reproject
    with pytest.raises(ValueError, match="not a dim"):
        xrops.lift(gg.Gradient2D(), dim="member")(_cube(_items()))
    with pytest.raises(TypeError, match="DataArray"):
        xrops.lift(gg.Gradient2D(), dim="time")(_cube(_items()).to_dataset(name="v"))


def test_building_a_long_lazy_lift_allocates_one_item_not_the_cube() -> None:
    import tracemalloc

    dask = pytest.importorskip("dask")
    import dask.array as dsa

    item = _items(1)[0].to_xarray()
    n = 20_000  # the output would be 20_000 x 2 x 2 x 8 x 10 float64 = 51 MB
    data = dsa.zeros((n, *item.shape), chunks=(500, *item.shape))
    cube = xr.DataArray(
        data, dims=("time", *item.dims), coords={**item.coords, "time": np.arange(n)}
    )
    tracemalloc.start()
    with dask.config.set(scheduler=_no_compute):
        out = xrops.lift(gg.Gradient2D(), dim="time")(cube)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert out.shape == (n, 2, 2, 8, 10) and out.chunks is not None
    assert peak < 20e6, f"building the lift allocated {peak / 1e6:.0f} MB"
