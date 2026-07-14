"""Tests for interpolation mask-cleanup primitives and operators."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr


pytest.importorskip("skimage")

from skimage import morphology

from xrtoolz.interpolate import (
    CleanMask,
    MaskBinaryClosing,
    MaskBinaryOpening,
    MaskRemoveSmallHoles,
    MaskRemoveSmallObjects,
    binary_closing_2d,
    binary_opening_2d,
    clean_mask,
    remove_small_holes_2d,
    remove_small_objects_2d,
)
from xrtoolz.interpolate._src.mask_ops import _resolve_footprint
from xrtoolz.transforms import clean_mask as transforms_clean_mask


def _mask(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values.astype(bool),
        dims=("lat", "lon"),
        coords={"lat": np.arange(values.shape[0]), "lon": np.arange(values.shape[1])},
        name="mask",
    )


def test_mask_ops_reexport_from_transforms() -> None:
    assert clean_mask is transforms_clean_mask


def test_remove_small_holes_fills_small_unmasked_island() -> None:
    mask = _mask(np.ones((100, 100), dtype=bool))
    mask.loc[{"lat": 50, "lon": 50}] = False

    filled = remove_small_holes_2d(mask, area=2)
    unchanged = remove_small_holes_2d(mask, area=1)

    assert bool(filled.sel(lat=50, lon=50))
    assert not bool(unchanged.sel(lat=50, lon=50))


def test_remove_small_objects_drops_masked_speck() -> None:
    mask = _mask(np.zeros((100, 100), dtype=bool))
    mask.loc[{"lat": 50, "lon": 50}] = True

    cleaned = remove_small_objects_2d(mask, area=2)

    assert not bool(cleaned.sel(lat=50, lon=50))


def test_binary_opening_and_closing_match_skimage() -> None:
    mask = _mask(np.zeros((7, 7), dtype=bool))
    mask.loc[{"lat": slice(2, 4), "lon": slice(2, 4)}] = True
    mask.loc[{"lat": 3, "lon": 3}] = False
    footprint = np.ones((3, 3), dtype=bool)

    xr.testing.assert_equal(
        binary_closing_2d(mask, footprint=footprint),
        _mask(morphology.closing(mask.values, footprint=footprint)),
    )
    xr.testing.assert_equal(
        binary_opening_2d(mask, footprint=footprint),
        _mask(morphology.opening(mask.values, footprint=footprint)),
    )


def test_resolve_footprint_shapes() -> None:
    array = np.ones((1, 5), dtype=bool)
    assert _resolve_footprint(array) is array
    np.testing.assert_array_equal(_resolve_footprint(2), morphology.disk(2))
    expected = {
        "disk": morphology.disk(1),
        # Radius-1 square: 3×3. The previous 1×1 default made opening /
        # closing a no-op, defeating the helper.
        "square": morphology.footprint_rectangle((3, 3)),
        "diamond": morphology.diamond(1),
        "star": morphology.star(1),
    }
    for name, footprint in expected.items():
        np.testing.assert_array_equal(
            _resolve_footprint(name),
            footprint,
        )


def test_clean_mask_defaults_match_remove_small_holes_only() -> None:
    mask = _mask(np.ones((5, 5), dtype=bool))
    mask.loc[{"lat": 2, "lon": 2}] = False

    xr.testing.assert_equal(clean_mask(mask), remove_small_holes_2d(mask, area=4))


def test_clean_mask_ordering() -> None:
    mask = _mask(np.zeros((7, 7), dtype=bool))
    mask.loc[{"lat": slice(1, 5), "lon": slice(1, 5)}] = True
    mask.loc[{"lat": 3, "lon": 3}] = False
    mask.loc[{"lat": 0, "lon": 0}] = True

    expected = remove_small_holes_2d(mask, area=2)
    expected = remove_small_objects_2d(expected, area=2)
    expected = binary_closing_2d(expected, footprint=1)
    expected = binary_opening_2d(expected, footprint=1)

    xr.testing.assert_equal(
        clean_mask(
            mask,
            fill_holes_area=2,
            drop_objects_area=2,
            closing_footprint=1,
            opening_footprint=1,
        ),
        expected,
    )


def test_leading_dim_broadcast_matches_manual_loop() -> None:
    values = np.zeros((2, 5, 5), dtype=bool)
    values[:, 1:4, 1:4] = True
    values[0, 2, 2] = False
    values[1, 0, 0] = True
    stack = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": np.arange(5), "lon": np.arange(5)},
    )

    result = remove_small_holes_2d(stack, area=2)
    expected = xr.concat(
        [remove_small_holes_2d(stack.isel(time=i), area=2) for i in range(2)],
        dim=stack.time,
    )

    xr.testing.assert_equal(result, expected)


def test_mask_validation_errors_are_clear() -> None:
    with pytest.raises(TypeError, match="mask must be boolean"):
        remove_small_holes_2d(
            xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
        )
    with pytest.raises(ValueError, match="missing"):
        remove_small_holes_2d(
            xr.DataArray(np.ones((2, 2), dtype=bool), dims=("y", "x")),
        )


def test_dask_time_chunks_preserved_and_core_chunks_rejected() -> None:
    pytest.importorskip("dask.array")
    stack = xr.DataArray(
        np.ones((10, 4, 4), dtype=bool),
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(10), "lat": np.arange(4), "lon": np.arange(4)},
    )

    result = remove_small_holes_2d(stack.chunk({"time": 5}), area=2)
    assert result.chunks is not None
    assert result.chunks[0] == (5, 5)

    with pytest.raises(ValueError, match=r"[Cc]ore dimension"):
        remove_small_holes_2d(stack.chunk({"lat": 2}), area=2)


@pytest.mark.parametrize(
    "mask_cleanup_operator",
    [
        MaskRemoveSmallHoles(area=2),
        MaskRemoveSmallObjects(area=2),
        MaskBinaryOpening(footprint="disk"),
        MaskBinaryClosing(footprint=np.ones((1, 3), dtype=bool)),
        CleanMask(
            fill_holes_area=2,
            drop_objects_area=2,
            closing_footprint=np.ones((1, 3), dtype=bool),
            opening_footprint="diamond",
        ),
    ],
    ids=lambda mask_cleanup_operator: type(mask_cleanup_operator).__name__,
)
def test_mask_operator_config_is_json_safe(mask_cleanup_operator) -> None:
    cfg = mask_cleanup_operator.get_config()
    serialized = json.dumps(cfg)
    assert json.loads(serialized) == cfg
