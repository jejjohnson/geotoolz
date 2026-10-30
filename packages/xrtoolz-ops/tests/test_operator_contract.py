"""The labeled-lane contract on every xrops operator (#366; LL-H1, LL-H2, OP-O6)."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import numpy as np
import pipekit
import pytest
import xarray as xr
from pipekit.adapters import BackendError

import xrops
from xrops.metrics import RMSE


VOCABULARY = {"numpy", "jax", "dask"}


def _public_operators() -> list[type]:
    found: dict[str, type] = {}
    for info in pkgutil.walk_packages(xrops.__path__, prefix="xrops."):
        if "._src" in info.name:
            continue
        module = importlib.import_module(info.name)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if inspect.isclass(obj) and issubclass(obj, pipekit.Operator):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    for name in xrops.__all__:
        obj = getattr(xrops, name)
        if inspect.isclass(obj) and issubclass(obj, pipekit.Operator):
            found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return list(found.values())


OPERATORS = _public_operators()


def test_every_operator_declares_backends_from_the_vocabulary() -> None:
    assert len(OPERATORS) > 200
    undeclared = sorted(op.__name__ for op in OPERATORS if op.backends is None)
    assert not undeclared, f"declare backends on {undeclared}"
    outside = {
        op.__name__: op.backends
        for op in OPERATORS
        if not set(op.backends or ()) <= VOCABULARY
        or "numpy" not in (op.backends or ())
    }
    assert not outside, (
        f"backends outside {sorted(VOCABULARY)} or without numpy: {outside}"
    )


def _field() -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.normal(size=(6, 4, 5)),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(6), "lat": np.arange(4.0), "lon": np.arange(5.0)},
        name="ssh",
    )


def test_a_jax_backed_dataarray_is_refused_naming_the_conversion() -> None:
    jnp = pytest.importorskip("jax.numpy")
    field = _field()
    on_jax = field.copy(data=jnp.asarray(field.values))
    with pytest.raises(BackendError, match="not jax"):
        xrops.Gradient(dims=("lat", "lon"))(on_jax)  # a DataArray
    ds = on_jax.to_dataset()
    with pytest.raises(BackendError, match="not jax"):
        RMSE(variable="ssh", dims=("time",))(ds, ds)  # a Dataset, by its variables


def test_numpy_beside_dask_is_one_lazy_computation() -> None:
    pytest.importorskip("dask")
    field = _field()
    ds = field.to_dataset()
    lazy = ds.chunk({"time": 2})
    rmse = RMSE(variable="ssh", dims=("time",))
    out = rmse(lazy, ds)  # a lazy prediction, an in-memory reference
    assert out.chunks is not None
    xr.testing.assert_allclose(out.compute(), rmse(ds, ds))


def test_a_geometry_operator_on_a_dataarray_names_lift_and_lift_runs_it() -> None:
    geoarray = pytest.importorskip("geoarray")
    gg = pytest.importorskip("geoops.geom")
    items = [
        geoarray.GeoArray(
            np.full((4, 5), float(t)),
            (1.0, 0, 0, 0, -1.0, 4.0),
            "EPSG:4326",
            dims=("y", "x"),
        ).to_xarray()
        for t in range(3)
    ]
    cube = xr.concat(items, dim="time").assign_coords(time=[0, 1, 2])
    with pytest.raises(TypeError, match=r"xrops\.lift"):
        gg.Gradient2D()(cube)
    assert xrops.lift(gg.Gradient2D(), dim="time")(cube).dims == (
        "time",
        "component",
        "y",
        "x",
    )


def test_a_jax_backed_datatree_is_refused_too() -> None:
    jnp = pytest.importorskip("jax.numpy")
    field = _field()
    ds = field.copy(data=jnp.asarray(field.values)).to_dataset()
    tree = xr.DataTree.from_dict({"a": ds, "b": ds})
    rmse = RMSE(variable="ssh", dims=("time",))
    with pytest.raises(BackendError, match="not jax"):
        rmse(tree, tree)  # the leaf-wise map
    from xrops.learn import XarrayEstimator

    sklearn = pytest.importorskip("sklearn")
    est = XarrayEstimator(sklearn.preprocessing.StandardScaler(), sample_dims="time")
    with pytest.raises(BackendError, match="not jax"):
        est(tree)  # a whole-tree override
    numpy_tree = xr.DataTree.from_dict({"a": field.to_dataset()})
    assert isinstance(rmse(numpy_tree, numpy_tree), xr.DataTree)
