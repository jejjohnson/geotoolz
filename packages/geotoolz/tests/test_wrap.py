"""Tests for the shared rewrap helpers (``geotoolz._src.wrap`` / ``_src.bands``)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from _helpers import DEFAULT_TRANSFORM, toy_geotensor
from georeader.geotensor import GeoTensor

from geotoolz._src.bands import (
    DEFAULT_BAND_KEYS,
    PER_BAND_KEYS,
    band_count,
    concat_band_attrs,
    per_band_values,
    strip_band_attrs,
    take_band_attrs,
)
from geotoolz._src.wrap import INHERIT, adopt_attrs, rewrap_attrs, wrap_like


def _scene(n_bands: int = 3, **extra) -> GeoTensor:
    names = [f"b{i}" for i in range(n_bands)]
    return toy_geotensor(
        np.ones((n_bands, 4, 4), dtype=np.float32),
        attrs={
            "band_names": names,
            "descriptions": list(names),
            "bands": list(names),
            "band_descriptions": list(names),
            "wavelengths": [500.0 + i for i in range(n_bands)],
            "wavelengths_nm": [500.0 + i for i in range(n_bands)],
            "sensor": "toy",
            **extra,
        },
    )


# --- bands ------------------------------------------------------------------


def test_default_band_keys_are_per_band_keys() -> None:
    assert set(DEFAULT_BAND_KEYS) < set(PER_BAND_KEYS)
    assert "wavelengths" in PER_BAND_KEYS


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((4, 4), 1), ((1, 4, 4), 1), ((5, 4, 4), 5), ((2, 5, 4, 4), 5)],
)
def test_band_count(shape: tuple[int, ...], expected: int) -> None:
    assert band_count(shape) == expected


def test_per_band_values_rejects_misfits() -> None:
    attrs = {"a": ["x", "y"], "s": "xy", "m": {"x": 0, "y": 1}, "n": 3}
    assert per_band_values(attrs, "a", 2) == ["x", "y"]
    assert per_band_values(attrs, "a", 3) is None
    assert per_band_values(attrs, "s", 2) is None
    assert per_band_values(attrs, "m", 2) is None
    assert per_band_values(attrs, "n", 2) is None
    assert per_band_values(None, "a", 2) is None
    values = per_band_values({"w": np.array([1.0, 2.0])}, "w", 2)
    assert values == [1.0, 2.0]
    assert all(type(v) is float for v in values or [])


def test_strip_take_and_concat_band_attrs() -> None:
    attrs = dict(_scene().attrs)
    assert strip_band_attrs(attrs) == {"sensor": "toy"}
    taken = take_band_attrs(attrs, [2, 0], n_bands=3)
    assert taken["band_names"] == ["b2", "b0"]
    assert taken["descriptions"] == ["b2", "b0"]
    assert taken["wavelengths"] == [502.0, 500.0]
    assert taken["sensor"] == "toy"
    # A per-band key that does not fit the band count is dropped, not kept stale.
    assert "band_names" not in take_band_attrs({"band_names": ["a"]}, [0], n_bands=3)
    joined = concat_band_attrs(
        [{"band_names": ["a"], "wavelengths": [1.0]}, {"band_names": ["b", "c"]}],
        [1, 2],
    )
    assert joined == {"band_names": ["a", "b", "c"]}


# --- wrap_like ----------------------------------------------------------------


def test_wrap_like_plain_array_passthrough() -> None:
    out = wrap_like(np.ones((2, 4, 4)), np.zeros((4, 4)), band_names=["x"])
    assert type(out) is np.ndarray


def test_wrap_like_attrs_are_a_fresh_copy() -> None:
    gt = _scene()
    out = wrap_like(gt, np.asarray(gt) * 2)
    assert out.attrs is not gt.attrs
    assert out.attrs == gt.attrs
    out.attrs["sensor"] = "changed"
    assert gt.attrs["sensor"] == "toy"


@pytest.mark.parametrize("out_shape", [(4, 4), (1, 4, 4), (2, 4, 4)])
def test_wrap_like_drops_per_band_keys_when_band_count_changes(
    out_shape: tuple[int, ...],
) -> None:
    gt = _scene()
    out = wrap_like(gt, np.zeros(out_shape))
    assert out.attrs == {"sensor": "toy"}


def test_wrap_like_writes_band_names_under_canonical_key() -> None:
    gt = _scene()
    out = wrap_like(gt, np.zeros((2, 4, 4)), band_names=("p", "q"))
    assert out.attrs == {"sensor": "toy", "band_names": ["p", "q"]}
    # Same band count: aliases are replaced, spectral metadata survives.
    same = wrap_like(gt, np.zeros((3, 4, 4)), band_names=["x", "y", "z"])
    assert same.attrs["band_names"] == ["x", "y", "z"]
    assert "descriptions" not in same.attrs
    assert same.attrs["wavelengths"] == [500.0, 501.0, 502.0]
    with pytest.raises(ValueError, match="band_names has 1 entries"):
        wrap_like(gt, np.zeros((2, 4, 4)), band_names=["p"])


def test_wrap_like_explicit_attrs_are_copied_verbatim() -> None:
    gt = _scene()
    attrs = {"band_names": ["a", "b"], "k": 1}
    out = wrap_like(gt, np.zeros((2, 4, 4)), attrs=attrs)
    assert out.attrs == attrs
    assert out.attrs is not attrs
    named = wrap_like(gt, np.zeros((2, 4, 4)), attrs={"k": 1}, band_names=["x", "y"])
    assert named.attrs == {"k": 1, "band_names": ["x", "y"]}


def test_wrap_like_fill_value_default_inherit_vs_explicit() -> None:
    gt = _scene()
    assert gt.fill_value_default == -9999
    assert wrap_like(gt, np.asarray(gt)).fill_value_default == -9999
    assert (
        wrap_like(gt, np.asarray(gt), fill_value_default=INHERIT).fill_value_default
        == -9999
    )
    assert (
        wrap_like(gt, np.asarray(gt), fill_value_default=None).fill_value_default
        is None
    )
    assert np.isnan(
        wrap_like(gt, np.asarray(gt), fill_value_default=np.nan).fill_value_default
    )
    assert (
        wrap_like(gt, np.asarray(gt), fill_value_default=False).fill_value_default
        is False
    )
    assert wrap_like(gt, np.asarray(gt), fill_value_default=0).fill_value_default == 0


def test_wrap_like_transform_override_allows_new_grid() -> None:
    gt = _scene()
    with pytest.raises(ValueError, match="altered spatial dimensions"):
        wrap_like(gt, np.zeros((3, 2, 2)))
    new = DEFAULT_TRANSFORM * rasterio.Affine.translation(1, 1)
    out = wrap_like(gt, np.zeros((3, 2, 2)), transform=new)
    assert out.transform == new
    assert out.crs == gt.crs
    assert out.attrs["band_names"] == ["b0", "b1", "b2"]


def test_wrap_like_four_d_uses_band_axis() -> None:
    gt = toy_geotensor(np.zeros((2, 3, 4, 4)), attrs={"band_names": ["a", "b", "c"]})
    assert wrap_like(gt, np.zeros((5, 3, 4, 4))).attrs == gt.attrs
    assert wrap_like(gt, np.zeros((2, 1, 4, 4))).attrs == {}


# --- rewrap_attrs / adopt_attrs -------------------------------------------------


def test_rewrap_attrs_same_band_count_keeps_everything() -> None:
    attrs = dict(_scene().attrs)
    assert rewrap_attrs(attrs, in_bands=3, out_bands=3) == attrs
    assert rewrap_attrs(None, in_bands=3, out_bands=1) == {}


def test_adopt_attrs_keeps_result_grid_and_never_mutates_input() -> None:
    gt = _scene()
    window = rasterio.windows.Window(col_off=1, row_off=1, width=2, height=2)
    cropped = gt.read_from_window(window, boundless=False)
    assert cropped.attrs is gt.attrs  # georeader aliases the dict
    out = adopt_attrs(gt, cropped)
    assert out.attrs is not gt.attrs
    assert out.attrs == gt.attrs
    assert out.transform == cropped.transform
    # georeader sometimes hands back the input itself; it must stay untouched.
    same = adopt_attrs(gt, gt, band_names=["x", "y", "z"])
    assert same is not gt
    assert gt.attrs["band_names"] == ["b0", "b1", "b2"]
    plain = np.zeros((2, 2))
    assert adopt_attrs(gt, plain) is plain
