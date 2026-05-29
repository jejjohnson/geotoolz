"""Layer-1 operator tests: apply parity, config, signature, DataTree."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

import xrtoolz.einx as xnx
from xrtoolz import Signature
from xrtoolz.einx import (
    BatchMatmul,
    Einsum,
    Matmul,
    Outer,
    Rearrange,
    Reduce,
    Repeat,
)


@pytest.fixture
def field() -> xr.DataArray:
    return xr.DataArray(
        np.arange(18.0).reshape(3, 2, 3),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(3), "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
        name="ssh",
    )


def test_reduce_operator_matches_function(field) -> None:
    op = Reduce("time lat lon -> lat lon", op="mean")
    xr.testing.assert_allclose(
        op(field), xnx.reduce("time lat lon -> lat lon", field, op="mean")
    )


def test_einsum_operator_variadic(field) -> None:
    mask = field.isel(time=0).drop_vars("time")
    op = Einsum("time lat lon, lat lon -> time")
    out = op(field, mask)
    assert out.dims == ("time",)


def test_rearrange_operator(field) -> None:
    op = Rearrange("time lat lon -> lat lon time")
    assert op(field).dims == ("lat", "lon", "time")


def test_matmul_operator() -> None:
    a = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("time", "k"))
    b = xr.DataArray(np.arange(10.0).reshape(2, 5), dims=("k", "mode"))
    assert Matmul(dim="k")(a, b).dims == ("time", "mode")


@pytest.mark.parametrize(
    "op",
    [
        Einsum("time lat lon, lat lon -> time"),
        Rearrange("a b -> b a"),
        Reduce("time lat lon -> lat lon", op="mean"),
        Repeat("lat lon -> month lat lon", month=12),
        Matmul(dim="k"),
        Outer(),
        BatchMatmul(dim="k", batch_dims=["ens"]),
    ],
    ids=lambda op: type(op).__name__,
)
def test_get_config_json_round_trips(op) -> None:
    cfg = op.get_config()
    assert json.loads(json.dumps(cfg)) == cfg
    # Reconstruct from config (operators take pattern positionally).
    rebuilt = type(op)(**cfg)
    assert rebuilt.get_config() == cfg


def test_compute_output_signature_einsum() -> None:
    op = Einsum("time lat lon, lat lon -> time")
    sigs = (
        Signature({"time": 3, "lat": 2, "lon": 3}, dtype="float64"),
        Signature({"lat": 2, "lon": 3}, dtype="float64"),
    )
    out = op.compute_output_signature(sigs)
    assert out == Signature({"time": 3}, dtype="float64")


def test_compute_output_signature_reduce() -> None:
    op = Reduce("time lat lon -> lat lon", op="mean")
    out = op.compute_output_signature(Signature({"time": 3, "lat": 2, "lon": 3}))
    assert dict(out.dims) == {"lat": 2, "lon": 3}


def test_config_round_trips_coords() -> None:
    """coords passed to __init__ must survive get_config (JSON-safe) and
    cls(**get_config()) must reproduce the operator."""
    op = Repeat("lat lon -> month lat lon", month=3, coords={"month": [1, 2, 3]})
    cfg = op.get_config()
    assert json.loads(json.dumps(cfg)) == cfg
    assert cfg["coords"] == {"month": [1, 2, 3]}

    mask = xr.DataArray(np.ones((2, 3)), dims=("lat", "lon"))
    rebuilt = Repeat(**cfg)
    xr.testing.assert_identical(rebuilt(mask), op(mask))


def test_config_round_trips_numpy_coords() -> None:
    """numpy-array coord values are serialized to lists for JSON safety."""
    op = Repeat("lat lon -> month lat lon", month=2, coords={"month": np.array([5, 6])})
    cfg = op.get_config()
    assert cfg["coords"] == {"month": [5, 6]}
    assert json.loads(json.dumps(cfg)) == cfg
    op = Matmul(dim="k")
    sigs = (
        Signature({"time": 3, "k": 2}),
        Signature({"k": 2, "mode": 5}),
    )
    out = op.compute_output_signature(sigs)
    assert dict(out.dims) == {"time": 3, "mode": 5}


def test_operators_compose_in_sequential(field) -> None:
    """einx operators are DataArray-level; they chain in a Sequential
    (carrier-agnostic) DataArray pipeline."""
    from pipekit import Sequential

    pipeline = Sequential(
        [
            Rearrange("time lat lon -> lat lon time"),
            Reduce("lat lon time -> lat lon", op="mean"),
        ]
    )
    out = pipeline(field)
    assert out.dims == ("lat", "lon")
    np.testing.assert_allclose(out.values, field.mean("time").values)
