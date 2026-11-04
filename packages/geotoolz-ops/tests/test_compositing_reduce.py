"""Nodata-aware reduce / argreduce in geoops.compositing (GA-B6, #632)."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from types import ModuleType

import numpy as np
import pipekit_array as pa
import pytest
from geoarray import GeoArray, GeoPoints

from geoops.compositing import argreduce, reduce


T = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
CRS = "EPSG:32613"


@pytest.fixture
def x64() -> Iterator[ModuleType]:
    """JAX with ``jax_enable_x64`` on for one test, restored afterwards."""
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)


def _stack(fill: float | None = 0.0) -> GeoArray:
    """(time=3, y=2, x=2): frame 1 is all fill, pixel (1, 1) is fill everywhere."""
    data = np.array(
        [
            [[1.0, 2.0], [3.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 0.0]],
        ]
    )
    return GeoArray(data, T, CRS, fill_value=fill, dims="time y x")


@pytest.mark.parametrize(
    ("fn", "want"),
    [
        ("mean", [[3.0, 4.0], [5.0, np.nan]]),
        ("median", [[3.0, 4.0], [5.0, np.nan]]),
        ("max", [[5.0, 6.0], [7.0, np.nan]]),
        ("min", [[1.0, 2.0], [3.0, np.nan]]),
        # nansum would give 0 where nothing is valid; the reduction gives NaN
        ("sum", [[6.0, 8.0], [10.0, np.nan]]),
    ],
)
def test_reduce_leaves_the_fill_out(fn: str, want: list[list[float]]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an all-invalid slice warns nothing
        out = reduce(_stack(), fn, "time")
    assert isinstance(out, GeoArray) and out.dims == ("y", "x")
    np.testing.assert_array_equal(out.data, want)
    assert np.isnan(out.fill_value)


def test_pipekit_reduce_would_average_the_fill_in() -> None:
    """The trap the geoops version exists for."""
    assert pa.reduce(_stack(), "mean", "time").data[0, 0] == pytest.approx(2.0)
    assert reduce(_stack(), "mean", "time").data[0, 0] == pytest.approx(3.0)


def test_nan_names_are_accepted_and_non_finite_is_invalid() -> None:
    g = _stack(fill=None).with_data(
        np.where(_stack().data == 0.0, np.inf, _stack().data)
    )
    np.testing.assert_array_equal(
        reduce(g, "nanmean", "time").data, [[3.0, 4.0], [5.0, np.nan]]
    )


def test_integer_data_is_reduced_as_float() -> None:
    g = _stack().with_data(_stack().data.astype(np.uint16))
    out = reduce(g, "mean", "time")
    assert out.dtype == np.float64
    assert out.data[0, 0] == pytest.approx(3.0)


def test_reduce_over_band_and_over_space() -> None:
    g = _stack().with_data(_stack().data, dims=("band", "y", "x"))
    np.testing.assert_array_equal(reduce(g, "max", "band").data, [[5, 6], [7, np.nan]])
    profile = reduce(_stack(), "mean", "x")  # consumes x: no grid any more
    assert type(profile).__name__ == "DimArray"
    np.testing.assert_array_equal(
        profile.data, [[1.5, 3.0], [np.nan, np.nan], [5.5, 7.0]]
    )


def test_a_callable_sees_nan_at_invalid_entries() -> None:
    seen: list[np.ndarray] = []

    def spy(data: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
        seen.append(data)
        return np.fmax.reduce(data, axis=axis)

    reduce(_stack(), spy, "time")
    assert np.isnan(seen[0][1]).all()


def test_an_unknown_reduction_is_refused() -> None:
    with pytest.raises(ValueError, match="no NaN-skipping form"):
        reduce(_stack(), "ptp", "time")


def test_a_plain_named_array_is_refused() -> None:
    with pytest.raises(TypeError, match=r"pipekit_array\.reduce"):
        reduce(pa.DimArray(np.ones((2, 2)), ("time", "x")), "mean", "time")


def test_points_leave_invalid_samples_out() -> None:
    pts = GeoPoints(
        np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
        xy=np.zeros((3, 2)),
        valid=np.array([True, False, True]),
        dims=("time", "obs"),
        crs=CRS,
    )
    out = reduce(pts, "mean", "obs")
    np.testing.assert_array_equal(out.data, [2.0, 4.0])


def test_argreduce_never_picks_the_fill() -> None:
    ndvi = GeoArray(
        np.array([[[0.4, 0.2]], [[9.0, 9.0]], [[0.7, 0.1]]]),
        T,
        CRS,
        fill_value=9.0,
        dims="time y x",
    )
    assert argreduce(ndvi, "argmax", "time").data.tolist() == [[2, 0]]
    assert argreduce(ndvi, "argmin", "time").data.tolist() == [[0, 2]]
    # pipekit-array's own would pick the fill frame
    assert pa.argreduce(ndvi, "argmax", "time").data.tolist() == [[1, 1]]


def test_argreduce_where_nothing_is_valid_gathers_nodata() -> None:
    idx = argreduce(_stack(), "argmax", "time")
    picked = pa.take_along(_stack(), idx, "time")
    assert picked.data[1, 1] == 0.0  # the fill: still nodata


def test_argreduce_refuses_other_reductions() -> None:
    with pytest.raises(ValueError, match="argmax"):
        argreduce(_stack(), "max", "time")


@pytest.mark.parametrize("fn", ["mean", "median", "sum"])
def test_jax_matches_numpy_under_jit(fn: str, x64: ModuleType) -> None:
    want = reduce(_stack(), fn, "time").data
    got = x64.jit(lambda g: reduce(g, fn, "time").data)(_stack().to_jax())
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-14)
    idx = x64.jit(lambda g: argreduce(g, "argmax", "time").data)(_stack().to_jax())
    assert (
        np.asarray(idx).tolist() == argreduce(_stack(), "argmax", "time").data.tolist()
    )
