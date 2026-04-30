"""Behavioral tests for :class:`xr_toolz.inference.ModelOp`.

Covers the framework-agnostic core: xarray <-> array marshalling,
``method=`` dispatch, batched non-feature dims, raw-array inputs, and
JSON-safe :meth:`get_config`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xr_toolz.inference import ModelOp


class _DummyModel:
    """Minimal duck-typed model: ``predict`` doubles, ``transform`` halves."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * x.sum(axis=-1)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x


@pytest.fixture
def da_2d() -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.standard_normal((6, 3)),
        dims=("sample", "feature"),
        coords={"sample": np.arange(6), "feature": ["a", "b", "c"]},
    )


@pytest.fixture
def da_batched() -> xr.DataArray:
    rng = np.random.default_rng(1)
    return xr.DataArray(
        rng.standard_normal((4, 5, 3)),
        dims=("time", "sample", "feature"),
        coords={"time": np.arange(4), "feature": ["a", "b", "c"]},
    )


def test_predict_default_method(da_2d: xr.DataArray) -> None:
    op = ModelOp(_DummyModel())
    out = op(da_2d)
    assert isinstance(out, xr.DataArray)
    assert out.dims == ("sample",)
    np.testing.assert_allclose(out.values, 2.0 * da_2d.values.sum(axis=-1))
    assert out.name == "prediction"


def test_method_dispatch_transform(da_2d: xr.DataArray) -> None:
    op = ModelOp(_DummyModel(), method="transform", output_name="halved")
    out = op(da_2d)
    assert out.dims == ("sample", "output")
    np.testing.assert_allclose(out.values, 0.5 * da_2d.values)
    assert out.name == "halved"


def test_batched_leading_dim(da_batched: xr.DataArray) -> None:
    op = ModelOp(_DummyModel())
    out = op(da_batched)
    assert out.dims == ("time", "sample")
    expected = 2.0 * da_batched.values.sum(axis=-1)
    np.testing.assert_allclose(out.values, expected)


def test_dataset_input_stacks_along_feature() -> None:
    rng = np.random.default_rng(2)
    ds = xr.Dataset(
        {
            "a": (("sample",), rng.standard_normal(4)),
            "b": (("sample",), rng.standard_normal(4)),
        },
        coords={"sample": np.arange(4)},
    )
    op = ModelOp(_DummyModel())
    out = op(ds)
    assert out.dims == ("sample",)
    expected = 2.0 * (ds["a"].values + ds["b"].values)
    np.testing.assert_allclose(out.values, expected)


def test_raw_numpy_input_2d() -> None:
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((5, 3))
    op = ModelOp(_DummyModel())
    out = op(arr)
    assert isinstance(out, xr.DataArray)
    np.testing.assert_allclose(out.values, 2.0 * arr.sum(axis=-1))


def test_missing_feature_dim_raises() -> None:
    da = xr.DataArray(np.zeros((4, 3)), dims=("sample", "channel"))
    op = ModelOp(_DummyModel())
    with pytest.raises(ValueError, match="feature dim"):
        op(da)


def test_unknown_method_raises() -> None:
    op = ModelOp(_DummyModel(), method="nope")
    with pytest.raises(AttributeError, match="nope"):
        op(np.zeros((2, 2)))


def test_get_config_is_json_serializable() -> None:
    op = ModelOp(_DummyModel(), method="predict", feature_dim="band")
    cfg = op.get_config()
    assert cfg["model"] == "<model>"
    assert cfg["method"] == "predict"
    assert cfg["feature_dim"] == "band"
    assert json.loads(json.dumps(cfg)) == cfg


def test_repr_uses_config() -> None:
    op = ModelOp(_DummyModel(), feature_dim="band")
    r = repr(op)
    assert "ModelOp" in r and "feature_dim='band'" in r


def test_attrs_are_preserved(da_2d: xr.DataArray) -> None:
    da = da_2d.assign_attrs(units="K", long_name="surface temperature")
    op = ModelOp(_DummyModel())
    out = op(da)
    assert out.attrs.get("units") == "K"
    assert out.attrs.get("long_name") == "surface temperature"


def test_auxiliary_sample_coords_preserved() -> None:
    rng = np.random.default_rng(7)
    da = xr.DataArray(
        rng.standard_normal((5, 3)),
        dims=("station", "feature"),
        coords={
            "station": np.arange(5),
            "feature": ["a", "b", "c"],
            "lat": ("station", np.linspace(40.0, 42.0, 5)),
            "station_id": ("station", [f"S{i}" for i in range(5)]),
        },
    )
    op = ModelOp(_DummyModel())
    out = op(da)
    assert "lat" in out.coords
    assert "station_id" in out.coords
    np.testing.assert_allclose(out.coords["lat"].values, da.coords["lat"].values)
    # The feature-dim coord should NOT carry through.
    assert "feature" not in out.coords


def test_auxiliary_multi_dim_coords_preserved() -> None:
    rng = np.random.default_rng(8)
    da = xr.DataArray(
        rng.standard_normal((2, 4, 3)),
        dims=("time", "station", "feature"),
        coords={
            "time": np.arange(2),
            "station": np.arange(4),
            "feature": ["a", "b", "c"],
            "obs_id": (("time", "station"), np.arange(8).reshape(2, 4)),
        },
    )
    op = ModelOp(_DummyModel())
    out = op(da)
    assert "obs_id" in out.coords
    assert out.coords["obs_id"].dims == ("time", "station")


def test_raw_1d_array_rejected() -> None:
    op = ModelOp(_DummyModel())
    with pytest.raises(ValueError, match="2-D"):
        op(np.zeros(5))


def test_raw_3d_array_rejected() -> None:
    op = ModelOp(_DummyModel())
    with pytest.raises(ValueError, match="2-D"):
        op(np.zeros((2, 3, 4)))


def test_higher_rank_model_output_rejected(da_2d: xr.DataArray) -> None:
    class _BadModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros((x.shape[0], 1, 2))

    op = ModelOp(_BadModel())
    with pytest.raises(ValueError, match=r"1-D .* or 2-D"):
        op(da_2d)


def test_output_dim_collision_with_sample_dim_rejected() -> None:
    rng = np.random.default_rng(9)
    # Input has a non-feature dim already named "output" — collides
    # with the default ``output_dim="output"``.
    da = xr.DataArray(
        rng.standard_normal((3, 5, 2)),
        dims=("output", "sample", "feature"),
    )

    class _MultiOutputModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros((x.shape[0], 4))

    op = ModelOp(_MultiOutputModel())
    with pytest.raises(ValueError, match="collides"):
        op(da)
    # Workaround: pass a distinct output_dim.
    op_ok = ModelOp(_MultiOutputModel(), output_dim="channel")
    out = op_ok(da)
    assert "channel" in out.dims
    assert "output" in out.dims  # original sample dim preserved


def test_works_inside_graph(da_2d: xr.DataArray) -> None:
    from xr_toolz.core import Graph, Input

    inp = Input("x")
    out = ModelOp(_DummyModel())(inp)
    graph = Graph(inputs={"x": inp}, outputs={"y": out})
    result = graph(x=da_2d)
    np.testing.assert_allclose(result["y"].values, 2.0 * da_2d.values.sum(axis=-1))
