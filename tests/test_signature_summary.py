"""Shape-signature propagation and structural summary tests."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pipekit import Operator

from xrtoolz.geo.operators import (
    CalculateClimatology,
    Reduce,
    RenameCoords,
    RenameVariables,
    SelectVariables,
    SubsetBBox,
    SubsetTime,
)
from xrtoolz.interpolate.operators import Coarsen, Refine, RegridLike, ResampleTime
from xrtoolz.metrics.operators import RMSE
from xrtoolz.signature import Signature


def test_signature_renders_unknown_dimensions() -> None:
    signature = Signature({"time": None, "lat": 3}, dtype="float32")

    assert signature.format() == "(time=?, lat=3); dtype=float32"


def test_operator_default_signature_is_shape_preserving() -> None:
    signature = Signature({"time": 10}, dtype="float32")

    assert Operator().compute_output_signature(signature) == signature


def test_geo_shape_changing_operator_signatures() -> None:
    signature = Signature({"time": 365, "lat": 181, "lon": 360}, dtype="float32")

    renamed = RenameCoords({"lon": "longitude"}).compute_output_signature(signature)
    assert renamed == Signature(
        {"time": 365, "lat": 181, "longitude": 360},
        dtype="float32",
    )
    assert (
        RenameVariables({"tas": "air_temperature"}).compute_output_signature(signature)
        == signature
    )

    subset = SubsetBBox((-125, -65), (25, 50)).compute_output_signature(signature)
    assert subset == Signature({"time": 365, "lat": None, "lon": None}, "float32")

    time_subset = SubsetTime("2000-01-01", "2000-01-31").compute_output_signature(
        signature
    )
    assert time_subset == Signature({"time": None, "lat": 181, "lon": 360}, "float32")

    reduced = Reduce(dim=("time", "lat")).compute_output_signature(signature)
    assert reduced == Signature({"lon": 360}, "float32")

    assert SelectVariables("tas").compute_output_signature(signature) == signature

    climatology = CalculateClimatology(freq="month").compute_output_signature(signature)
    assert climatology == Signature(
        {"month": None, "lat": 181, "lon": 360},
        "float32",
    )


def test_interpolation_shape_changing_operator_signatures() -> None:
    signature = Signature({"time": 24, "lat": 10, "lon": 20}, dtype="float32")
    target = xr.Dataset(
        coords={
            "lat": np.linspace(-90, 90, 5),
            "lon": np.linspace(0, 360, 8, endpoint=False),
        }
    )

    assert ResampleTime("1D").compute_output_signature(signature) == Signature(
        {"time": None, "lat": 10, "lon": 20},
        "float32",
    )
    assert RegridLike(target).compute_output_signature(signature) == Signature(
        {"time": 24, "lat": 5, "lon": 8},
        "float32",
    )
    assert Coarsen({"lat": 2, "lon": 5}).compute_output_signature(signature) == (
        Signature({"time": 24, "lat": 5, "lon": 4}, "float32")
    )
    assert Coarsen({"lat": 3}, boundary="pad").compute_output_signature(
        signature
    ) == Signature({"time": 24, "lat": 4, "lon": 20}, "float32")
    with pytest.raises(
        ValueError,
        match=(
            "coarsen boundary='exact' requires 'lat' size 10 to be "
            "divisible by factor 3"
        ),
    ):
        Coarsen({"lat": 3}, boundary="exact").compute_output_signature(signature)
    assert Refine({"lat": 2}).compute_output_signature(signature) == Signature(
        {"time": 24, "lat": 19, "lon": 20},
        "float32",
    )


def test_metric_signature_rejects_mismatched_inputs() -> None:
    op = RMSE(variable="tas", dims=("time",))
    pred = Signature({"time": 12, "lat": 4}, "float32")
    ref = Signature({"time": 10, "lat": 4}, "float32")

    with pytest.raises(ValueError, match="sizes do not match"):
        op.compute_output_signature((pred, ref))


def test_signature_dtype_canonicalization_unifies_string_and_numpy_forms() -> None:
    string_form = Signature({"time": 12}, dtype="float32")
    numpy_form = Signature({"time": 12}, dtype=np.float32)
    dtype_obj = Signature({"time": 12}, dtype=np.dtype("float32"))

    assert string_form == numpy_form == dtype_obj
    # Canonical name is the np.dtype.name string regardless of input form.
    assert string_form.dtype == "float32"
    assert numpy_form.dtype == "float32"


def test_signature_equality_is_dim_order_insensitive() -> None:
    a = Signature({"time": 12, "lat": 4}, dtype="float32")
    b = Signature({"lat": 4, "time": 12}, dtype="float32")

    assert a == b
    assert hash(a) == hash(b)


def test_signature_dims_are_immutable() -> None:
    sig = Signature({"time": 12, "lat": 4}, dtype="float32")

    # MappingProxyType disallows __setitem__ on the proxy, so a buggy
    # operator override that tries to mutate in place fails loudly
    # rather than corrupting cached signatures inside Graph.
    with pytest.raises(TypeError):
        sig.dims["time"] = 5  # type: ignore[index]


def test_signature_validates_dim_types() -> None:
    with pytest.raises(TypeError, match="dim name must be a str"):
        Signature({1: 5})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="must be int or None"):
        Signature({"time": 1.5})  # type: ignore[dict-item]


def test_signature_replace_dims_strict_raises_on_unknown() -> None:
    sig = Signature({"time": 12, "lat": 4}, dtype="float32")

    # Default (strict=False) silently ignores unknown keys — useful for
    # shape-preserving paths that don't care if the dim is present.
    assert sig.replace_dims({"missing": 7}) == sig
    # strict=True surfaces typos in operator overrides.
    with pytest.raises(KeyError, match="not in signature dims"):
        sig.replace_dims({"missing": 7}, strict=True)


def test_coarsen_rejects_unknown_boundary() -> None:
    with pytest.raises(ValueError, match="boundary must be one of"):
        Coarsen({"lat": 2}, boundary="trip")


def test_calculate_climatology_smoothed_uses_canonical_dim() -> None:
    # The override pulls the dim name from CLIMATOLOGY_DIMS rather than
    # hardcoding "dayofyear", so this test catches divergence between
    # the runtime and the inferred signature.
    from xrtoolz.geo._src import detrend
    from xrtoolz.geo.operators import CalculateClimatologySmoothed

    signature = Signature({"time": 365, "lat": 4}, dtype="float32")
    smoothed = CalculateClimatologySmoothed().compute_output_signature(signature)

    assert detrend.CLIMATOLOGY_DIMS["day"] in smoothed.dims
    assert "time" not in smoothed.dims
