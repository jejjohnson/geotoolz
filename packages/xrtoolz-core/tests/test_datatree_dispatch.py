"""Tests for the xarray-aware :class:`xrcore.Operator`.

Covers the three ``__call__`` dispatch modes added on top of
``pipekit.Operator``:

1. Symbolic ``Node`` construction passes through to pipekit unchanged.
2. ``DataTree`` arguments map ``_apply`` over every leaf, returning a
   tree with the same structure.
3. Plain ``Dataset`` / ``DataArray`` arguments hit ``_apply`` directly
   (the existing eager path).

The DataTree path is exercised end-to-end via ``Sequential`` and the
functional ``Graph`` API to confirm composition flows through without
per-step changes.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pipekit import Graph, Input, Node, Sequential

from xrcore import Operator


# ---------- toy operators --------------------------------------------------


class _ScaleVar(Operator):
    """Multiply ``variable`` by ``factor`` in-place on the Dataset."""

    def __init__(self, variable: str, *, factor: float) -> None:
        self.variable = variable
        self.factor = factor

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return ds.assign({self.variable: ds[self.variable] * self.factor})


class _Diff(Operator):
    """Two-input op: ``pred - ref`` of ``variable``, returns a Dataset."""

    def __init__(self, variable: str) -> None:
        self.variable = variable

    def _apply(self, pred: xr.Dataset, ref: xr.Dataset) -> xr.Dataset:
        return xr.Dataset({self.variable: pred[self.variable] - ref[self.variable]})


class _DiffArray(Operator):
    """Two-input op that returns a DataArray (e.g. an RMSE-style metric)."""

    def __init__(self, variable: str) -> None:
        self.variable = variable

    def _apply(self, pred: xr.Dataset, ref: xr.Dataset) -> xr.DataArray:
        out = pred[self.variable] - ref[self.variable]
        return out.rename("delta")


class _ReturnsObject(Operator):
    """Op that returns a non-xarray object (stand-in for terminal viz)."""

    def _apply(self, ds: xr.Dataset) -> object:
        return object()


# ---------- fixtures -------------------------------------------------------


@pytest.fixture
def ds() -> xr.Dataset:
    return xr.Dataset(
        {"x": (("t",), np.arange(4, dtype=float))},
        coords={"t": np.arange(4)},
    )


@pytest.fixture
def dt() -> xr.DataTree:
    leaf_a = xr.Dataset(
        {"x": (("t",), np.array([1.0, 2.0, 3.0, 4.0]))},
        coords={"t": np.arange(4)},
    )
    leaf_b = xr.Dataset(
        {"x": (("t",), np.array([10.0, 20.0, 30.0, 40.0]))},
        coords={"t": np.arange(4)},
    )
    return xr.DataTree.from_dict({"a": leaf_a, "b": leaf_b})


# ---------- eager Dataset mode (no regression) -----------------------------


def test_dataset_path_unchanged(ds: xr.Dataset) -> None:
    op = _ScaleVar("x", factor=2.0)
    out = op(ds)
    assert isinstance(out, xr.Dataset)
    np.testing.assert_array_equal(out["x"].values, ds["x"].values * 2.0)


# ---------- single-input DataTree dispatch ---------------------------------


def test_single_input_datatree_dispatch(dt: xr.DataTree) -> None:
    op = _ScaleVar("x", factor=3.0)
    out = op(dt)
    assert isinstance(out, xr.DataTree)
    assert set(out.children) == {"a", "b"}
    np.testing.assert_array_equal(
        out["a"].dataset["x"].values, np.array([3.0, 6.0, 9.0, 12.0])
    )
    np.testing.assert_array_equal(
        out["b"].dataset["x"].values, np.array([30.0, 60.0, 90.0, 120.0])
    )


def test_datatree_preserves_structure(dt: xr.DataTree) -> None:
    op = _ScaleVar("x", factor=1.0)
    out = op(dt)
    assert set(out.children) == set(dt.children)
    for path in dt.children:
        assert out[path].dataset.equals(dt[path].dataset)


# ---------- multi-input DataTree dispatch ----------------------------------


def test_multi_input_datatree_dispatch(dt: xr.DataTree) -> None:
    op = _Diff("x")
    out = op(dt, dt)  # identical trees → zero everywhere
    assert isinstance(out, xr.DataTree)
    for path in ("a", "b"):
        np.testing.assert_array_equal(
            out[path].dataset["x"].values, np.zeros(4, dtype=float)
        )


def test_multi_input_mismatched_structure_raises(dt: xr.DataTree) -> None:
    other = xr.DataTree.from_dict(
        {
            "a": xr.Dataset({"x": (("t",), np.zeros(4))}, coords={"t": np.arange(4)}),
        }
    )
    op = _Diff("x")
    with pytest.raises(ValueError):
        op(dt, other)


# ---------- Sequential threads DataTrees end-to-end ------------------------


def test_sequential_threads_datatree(dt: xr.DataTree) -> None:
    pipeline = Sequential([_ScaleVar("x", factor=2.0), _ScaleVar("x", factor=5.0)])
    out = pipeline(dt)
    assert isinstance(out, xr.DataTree)
    np.testing.assert_array_equal(
        out["a"].dataset["x"].values, np.array([10.0, 20.0, 30.0, 40.0])
    )
    np.testing.assert_array_equal(
        out["b"].dataset["x"].values, np.array([100.0, 200.0, 300.0, 400.0])
    )


def test_sequential_still_works_on_dataset(ds: xr.Dataset) -> None:
    pipeline = Sequential([_ScaleVar("x", factor=2.0), _ScaleVar("x", factor=5.0)])
    out = pipeline(ds)
    assert isinstance(out, xr.Dataset)
    np.testing.assert_array_equal(out["x"].values, ds["x"].values * 10.0)


# ---------- Graph dispatch over DataTrees ----------------------------------


def test_graph_over_datatree(dt: xr.DataTree) -> None:
    inp = Input("dt")
    node = _ScaleVar("x", factor=4.0)(inp)
    graph = Graph(inputs={"dt": inp}, outputs={"out": node})
    out = graph(dt=dt)["out"]
    assert isinstance(out, xr.DataTree)
    np.testing.assert_array_equal(
        out["a"].dataset["x"].values, np.array([4.0, 8.0, 12.0, 16.0])
    )


# ---------- symbolic Node dispatch still works -----------------------------


def test_node_construction_still_works() -> None:
    inp = Input("ds")
    result = _ScaleVar("x", factor=2.0)(inp)
    assert isinstance(result, Node)


# ---------- DataArray returns get wrapped into a Dataset -------------------


def test_datatree_dispatch_wraps_dataarray_returns(dt: xr.DataTree) -> None:
    """A metric returning DataArray maps cleanly over a DataTree."""
    op = _DiffArray("x")
    out = op(dt, dt)
    assert isinstance(out, xr.DataTree)
    for path in ("a", "b"):
        ds_leaf = out[path].dataset
        assert "delta" in ds_leaf.data_vars
        np.testing.assert_array_equal(ds_leaf["delta"].values, np.zeros(4, dtype=float))


# ---------- non-Dataset / non-DataArray returns are rejected ---------------


def test_datatree_dispatch_rejects_non_xarray_return(dt: xr.DataTree) -> None:
    op = _ReturnsObject()
    with pytest.raises(TypeError, match="DataTree dispatch"):
        op(dt)


# ---------- partial-empty multi-input surfaces a real error ----------------


def test_partial_empty_multi_input_surfaces_apply_error() -> None:
    """If one tree has data and the other does not, ``_apply`` runs and
    its KeyError surfaces rather than being silently swallowed.
    """
    ds_full = xr.Dataset(
        {"x": (("t",), np.arange(4, dtype=float))},
        coords={"t": np.arange(4)},
    )
    tree_full = xr.DataTree.from_dict({"a": ds_full})
    tree_empty = xr.DataTree.from_dict({"a": xr.Dataset()})
    op = _Diff("x")
    with pytest.raises(KeyError):
        op(tree_full, tree_empty)


# ---------- _apply_tree hook ----------------------------------------------


class _WholeTree(Operator):
    """Sees the whole tree: subtracts the mean of ``x`` over every leaf."""

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        raise AssertionError("_apply must not be called when _apply_tree is overridden")

    def _apply_tree(self, tree: xr.DataTree) -> xr.DataTree:
        mean = float(np.mean([float(n["x"].mean()) for n in tree.leaves]))
        return tree.map_over_datasets(lambda d: d - mean if d.data_vars else d)


def test_apply_tree_override_receives_whole_tree(dt: xr.DataTree) -> None:
    out = _WholeTree()(dt)

    leaf_means = [float(dt[p]["x"].mean()) for p in ("a", "b")]
    expected = dt["a"]["x"] - np.mean(leaf_means)
    xr.testing.assert_allclose(out["a"]["x"], expected)


def test_apply_tree_override_is_used_inside_sequential(dt: xr.DataTree) -> None:
    out = Sequential([_ScaleVar("x", factor=2.0), _WholeTree()])(dt)

    assert isinstance(out, xr.DataTree)
    assert float(out["a"]["x"].mean() + out["b"]["x"].mean()) == pytest.approx(0.0)


def test_default_apply_tree_is_the_leaf_map(dt: xr.DataTree) -> None:
    op = _ScaleVar("x", factor=3.0)

    xr.testing.assert_identical(op._apply_tree(dt), op(dt))
