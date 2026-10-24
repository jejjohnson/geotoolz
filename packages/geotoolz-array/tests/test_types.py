"""Readable types and check_dims (#521, GA-T5)."""

from __future__ import annotations

import typing
from types import ModuleType
from typing import Annotated, Any

import numpy as np
import pytest
from pipekit_array import DimArray

from geoarray import GeoArray, GeoPoints, GeoSwath
from geoarray._src import types as types_impl
from geoarray.coords import Vertical
from geoarray.types import (
    DimsError,
    GeoImage,
    GeoImageSeries,
    GeoMap,
    GeoMapSeries,
    GeoPatches,
    GeoProfiles,
    GeoSwathImage,
    GeoSwathProfile,
    GeoTimeSeries,
    GeoTracks,
    GeoVolume,
    ReadableType,
    check_dims,
    kind_of,
    matches,
)


T = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
DEPTH = Vertical("depth", "m", "down")


def _grid(dims: str) -> GeoArray:
    shape = (2, 3, 4, 5, 6)[-len(dims.split()) :]
    return GeoArray(np.zeros(shape), T, 32613, dims=dims)


def _swath(dims: str) -> GeoSwath:
    shape = (3, 4, 5)[-len(dims.split()) :]
    lat = np.zeros(shape[-2:])
    return GeoSwath(np.zeros(shape), lat, lat, dims=dims)


def _points(dims: str, **leaves: Any) -> GeoPoints:
    shape = (3, 2, 4)[-len(dims.split()) :]
    return GeoPoints(np.zeros(shape), xy=np.zeros((4, 2)), dims=dims, **leaves)


CASES = [
    (GeoMap, lambda: _grid("y x")),
    (GeoImage, lambda: _grid("band y x")),
    (GeoMapSeries, lambda: _grid("time y x")),
    (GeoImageSeries, lambda: _grid("time band y x")),
    (GeoVolume, lambda: _grid("level y x")),
    (GeoPatches, lambda: _grid("patch band y x")),
    (GeoSwathImage, lambda: _swath("band row col")),
    (GeoSwathProfile, lambda: _swath("level row col")),
    (GeoTimeSeries, lambda: _points("time variable obs")),
    (GeoTracks, lambda: _points("variable obs", time=np.zeros(4, "datetime64[ns]"))),
    (GeoProfiles, lambda: _points("variable obs", z=np.zeros(4), vertical=DEPTH)),
]


@pytest.mark.parametrize(
    ("alias", "make"), CASES, ids=lambda c: getattr(c, "__name__", "")
)
def test_each_type_matches_its_carrier_and_no_other(alias: Any, make: Any) -> None:
    x = make()
    assert matches(x, alias)
    others = [a for a, _ in CASES if a is not alias]
    # GeoTracks and GeoProfiles share a pattern; their leaves tell them apart
    assert not any(matches(x, a) for a in others), [
        typing.get_args(a)[1] for a in others if matches(x, a)
    ]


def test_aliases_read_as_their_carrier() -> None:
    assert typing.get_args(GeoImage)[0] is GeoArray
    assert typing.get_args(GeoSwathImage)[0] is GeoSwath
    assert typing.get_args(GeoTracks)[0] is GeoPoints
    assert (
        repr(typing.get_args(GeoTracks)[1]) == "GeoTracks[points: variable obs + time]"
    )


def test_slicing_a_series_is_an_image() -> None:
    stack = _grid("time band y x")
    assert matches(stack, GeoImageSeries)
    assert matches(stack.isel(time=0), GeoImage)
    assert not matches(stack, GeoImage)


def test_a_dimarray_has_no_readable_type() -> None:
    d = DimArray(np.zeros((2, 3)), ("y", "x"))
    assert kind_of(d) is None
    assert not matches(d, GeoMap)
    assert not matches(np.zeros((2, 3)), GeoMap)


def test_subscript_is_an_annotated_pattern() -> None:
    ann = GeoArray["band y x"]
    assert typing.get_origin(ann) is Annotated
    spec = typing.get_args(ann)[1]
    assert isinstance(spec, ReadableType) and spec.pattern == "band y x"
    assert matches(_grid("band y x"), ann)
    assert matches(_swath("band row col"), GeoSwath["... row col"])
    with pytest.raises(ValueError, match="ends in"):
        GeoArray["band row col"]


def test_the_0_2_generic_parameter_warns() -> None:
    with pytest.warns(DeprecationWarning, match="array-type"):
        assert GeoArray[np.ndarray] is GeoArray


def test_bad_readable_types() -> None:
    with pytest.raises(ValueError, match="kind"):
        ReadableType("X", "y x", "mesh")
    with pytest.raises(TypeError, match="not a readable type"):
        matches(_grid("y x"), int)


# -- check_dims ----------------------------------------------------------------------


@check_dims
def ndvi(scene: GeoImage) -> GeoMap:
    return scene.isel(band=0)


@check_dims
def broken(scene: GeoImage) -> GeoMap:
    return scene


def test_check_dims_passes_matching_calls() -> None:
    assert ndvi(_grid("band y x")).dims == ("y", "x")
    assert ndvi(scene=_grid("time band y x").isel(time=1)).dims == ("y", "x")


def test_check_dims_names_the_fix() -> None:
    with pytest.raises(
        DimsError,
        match=(
            r"scene is a GeoImage.*got dims \('time', 'band', 'y', 'x'\)"
            r".*map_over\(x, ndvi, 'time'\)"
        ),
    ):
        ndvi(_grid("time band y x"))
    with pytest.raises(DimsError, match="is a swath, not a grid"):
        ndvi(_swath("band row col"))


def test_check_dims_checks_the_return() -> None:
    with pytest.raises(DimsError, match="returns a GeoMap"):
        broken(_grid("band y x"))


def test_optional_and_unannotated_arguments() -> None:
    @check_dims
    def f(a: GeoMap | None, b: int, c: Any = None) -> int:
        return b

    assert f(None, 1) == 1
    assert f(_grid("y x"), 2) == 2
    with pytest.raises(DimsError):
        f(_grid("band y x"), 3)


def test_check_dims_runs_once_per_trace(
    x64: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    jax = x64
    calls = 0
    original = types_impl._mismatch

    def counting(x: Any, t: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(x, t)

    monkeypatch.setattr(types_impl, "_mismatch", counting)
    f = jax.jit(lambda g: ndvi(g).data.sum())
    scene = _grid("band y x").to_jax()
    for k in range(3):
        f(scene.shift_origin((float(k), 0.0)))
    assert calls == 2  # the argument and the return, once, at the first trace
