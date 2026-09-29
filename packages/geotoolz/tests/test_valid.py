"""Tests for the shared invalid-pixel helpers (``geotoolz._src.valid``)."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import fill_pixel_mask, toy_geotensor

from geotoolz._src.valid import (
    carrier_fill_value,
    invalid_values,
    is_fill,
    mask_invalid_to_nan,
    restore_fill,
    valid_pixels,
)
from geotoolz._src.wrap import INHERIT


def test_is_fill_numeric_nan_and_none():
    arr = np.array([1.0, np.nan, -9999.0])
    assert is_fill(arr, -9999).tolist() == [False, False, True]
    assert is_fill(arr, np.nan).tolist() == [False, True, False]
    assert is_fill(arr, None).tolist() == [False, False, False]
    # A NaN fill on an integer array matches nothing.
    assert not is_fill(np.array([1, 2]), np.nan).any()


def test_carrier_fill_value():
    gt = toy_geotensor(np.zeros((2, 2)), fill_value_default=-1)
    assert carrier_fill_value(gt) == -1
    assert carrier_fill_value(gt, 5) == 5
    assert carrier_fill_value(np.zeros((2, 2))) is None
    assert carrier_fill_value(gt, INHERIT) == -1


def test_invalid_values_includes_non_finite_under_numeric_fill():
    gt = toy_geotensor(np.array([[1.0, np.inf], [np.nan, -9999.0]]))
    assert invalid_values(gt).tolist() == [[False, True], [True, True]]


def test_valid_pixels_handles_nan_fill_that_validmask_misses():
    vals = np.ones((2, 3, 3), dtype=np.float32)
    gt = toy_geotensor(vals, fill_value_default=np.nan, with_fill_pixels=True)
    # georeader's validmask compares with != and never matches a NaN fill.
    assert np.asarray(gt.validmask()).all()
    np.testing.assert_array_equal(valid_pixels(gt), ~fill_pixel_mask(gt.shape))


def test_valid_pixels_any_band_invalidates_the_pixel():
    vals = np.ones((3, 2, 2), dtype=np.int16)
    vals[1, 0, 1] = -9999
    gt = toy_geotensor(vals)
    assert valid_pixels(gt).tolist() == [[True, False], [True, True]]


def test_valid_pixels_plain_array_is_finite_only():
    arr = np.array([[-9999.0, np.nan]])
    assert valid_pixels(arr).tolist() == [[True, False]]
    assert valid_pixels(arr, fill_value=-9999).tolist() == [[False, False]]


def test_valid_pixels_4d_keep_time():
    vals = np.ones((2, 2, 2, 2))
    vals[1, 0, 0, 0] = np.nan
    assert valid_pixels(vals).tolist() == [[False, True], [True, True]]
    per_frame = valid_pixels(vals, keep_time=True)
    assert per_frame.shape == (2, 2, 2)
    assert per_frame[0].all()
    assert not per_frame[1, 0, 0]


def test_valid_pixels_rejects_1d():
    with pytest.raises(ValueError, match=">= 2 dims"):
        valid_pixels(np.ones(3))


def test_mask_invalid_to_nan_promotes_and_masks_all_bands():
    vals = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
    vals[0, 1, 1] = -9999
    out = mask_invalid_to_nan(toy_geotensor(vals))
    assert out.dtype == np.float32
    assert np.isnan(out[:, 1, 1]).all()
    assert np.isfinite(out[:, 0, 0]).all()
    assert vals[0, 1, 1] == -9999  # input untouched


def test_mask_invalid_to_nan_explicit_valid_and_dtype():
    vals = np.ones((2, 2), dtype=np.float32)
    valid = np.array([[True, False], [True, True]])
    out = mask_invalid_to_nan(vals, valid=valid, dtype=np.float64)
    assert out.dtype == np.float64
    assert np.isnan(out[0, 1])


def test_restore_fill_numeric_nan_and_4d():
    out = np.zeros((2, 2, 2), dtype=np.float32)
    valid = np.array([[True, False], [True, True]])
    filled = restore_fill(out, valid, -9999)
    assert (filled[:, 0, 1] == -9999).all()
    assert (out == 0).all()  # copy, not in place
    assert np.isnan(restore_fill(out, valid, None)[:, 0, 1]).all()
    assert np.isnan(restore_fill(out, valid, np.nan)[:, 0, 1]).all()

    per_frame = np.ones((2, 2, 2), dtype=bool)
    per_frame[1, 0, 0] = False
    filled4 = restore_fill(np.zeros((2, 3, 2, 2)), per_frame, -1)
    assert (filled4[1, :, 0, 0] == -1).all()
    assert (filled4[0] == 0).all()


def test_restore_fill_all_valid_is_passthrough():
    out = np.zeros((2, 2), dtype=np.uint8)
    assert restore_fill(out, np.ones((2, 2), bool), None) is out


def test_restore_fill_rejects_unrepresentable_fill():
    out = np.zeros((2, 2), dtype=np.uint8)
    valid = np.array([[True, False], [True, True]])
    with pytest.raises(ValueError, match="Cannot write"):
        restore_fill(out, valid, np.nan)
    with pytest.raises(ValueError, match="Cannot write"):
        restore_fill(out, valid, -9999)
    assert restore_fill(out, valid, 0)[0, 1] == 0


def test_toy_geotensor_with_fill_pixels_copies_values():
    vals = np.ones((2, 3, 3))
    gt = toy_geotensor(vals, with_fill_pixels=True)
    assert (np.asarray(gt)[:, 0, 0] == -9999).all()
    assert (np.asarray(gt)[:, -1, -1] == -9999).all()
    assert (vals == 1).all()
    with pytest.raises(ValueError, match="fill_value_default"):
        toy_geotensor(vals, fill_value_default=None, with_fill_pixels=True)
