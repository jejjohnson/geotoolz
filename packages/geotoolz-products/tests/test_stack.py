"""``geoproducts.stack``: each reader's nodata survives onto the stacked grid."""

from __future__ import annotations

import math

import numpy as np
from geoarray import GeoArray

from geoproducts import stack


TRANSFORM = (0.1, 0.0, 10.0, 0.0, -0.1, 50.0)


def _part(data: np.ndarray, fill: float | int | None, name: str) -> GeoArray:
    return GeoArray(
        data, TRANSFORM, 4326, fill, {"band_names": (name,)}, dims="band y x"
    )


def _nan_fill(g: GeoArray) -> bool:
    return isinstance(g.fill_value, float) and math.isnan(g.fill_value)


def test_a_float_fill_that_is_not_nan_becomes_nan() -> None:
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    values[0, 0, 0] = -9999.0
    sentinel = _part(values, -9999.0, "a")
    clean = _part(np.ones((1, 3, 4), np.float32), np.nan, "b")
    out = stack([clean, sentinel], resampling="nearest")
    assert out.dtype == np.float32
    assert _nan_fill(out)
    data = np.asarray(out)
    assert np.isnan(data[1, 0, 0])
    assert (data != -9999.0).all()
    assert not out.valid_mask()[0, 0]
    assert out.valid_mask().sum() == 11
    np.testing.assert_array_equal(data[1].ravel()[1:], np.arange(1, 12))


def test_float_parts_sharing_a_sentinel_fill_still_get_nan() -> None:
    values = np.ones((1, 3, 4), np.float32)
    values[0, 2, 3] = -9999.0
    out = stack(
        [_part(values, -9999.0, "a"), _part(values.copy(), -9999.0, "b")],
        resampling="nearest",
    )
    assert _nan_fill(out)
    assert np.isnan(np.asarray(out)[:, 2, 3]).all()
    assert out.valid_mask().sum() == 11


def test_an_integer_part_promoted_to_float_gets_nan_at_its_fill() -> None:
    counts = np.full((1, 3, 4), 7, np.uint16)
    counts[0, 1, 1] = 65535
    floats = _part(np.ones((1, 3, 4), np.float32), np.nan, "f")
    out = stack([floats, _part(counts, 65535, "n")], resampling="nearest")
    assert out.dtype == np.float32
    data = np.asarray(out)
    assert np.isnan(data[1, 1, 1])
    assert np.nansum(data[1] == 7) == 11


def test_mixed_integer_dtypes_go_float_with_each_own_fill_as_nan() -> None:
    a = np.full((1, 3, 4), 3, np.uint8)
    a[0, 0, 1] = 255
    b = np.full((1, 3, 4), 4, np.int16)
    b[0, 2, 0] = -1
    out = stack([_part(a, 255, "a"), _part(b, -1, "b")], resampling="nearest")
    assert out.dtype == np.float32 and _nan_fill(out)
    data = np.asarray(out)
    assert np.isnan(data[0, 0, 1]) and np.isnan(data[1, 2, 0])
    assert np.isnan(data).sum() == 2


def test_integer_parts_sharing_dtype_and_fill_stay_integer() -> None:
    a = np.full((1, 3, 4), 2, np.uint8)
    a[0, 0, 0] = 255
    out = stack([_part(a, 255, "a"), _part(a.copy(), 255, "b")], resampling="nearest")
    assert out.dtype == np.uint8
    assert out.fill_value == 255
    assert (np.asarray(out)[:, 0, 0] == 255).all()
