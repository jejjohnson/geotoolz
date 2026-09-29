"""Tests for the shared shape helpers (``geotoolz._src.shape``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _helpers import TIME_STACK_SHAPE, frames, time_stack, toy_geotensor
from georeader.geotensor import GeoTensor

from geotoolz._src.shape import (
    BAND_AXIS,
    band_axis,
    keep_band_axis,
    map_frames,
    over_frames,
    require_ndim,
    single_band,
)


def test_time_stack_fixture_is_time_band_y_x() -> None:
    stack = time_stack()
    assert isinstance(stack, GeoTensor)
    assert stack.shape == TIME_STACK_SHAPE == (2, 3, 4, 4)
    assert stack.dims == ("time", "band", "y", "x")
    assert stack.attrs["band_names"] == ["b0", "b1", "b2"]
    assert len(stack.attrs["wavelengths"]) == 3
    # Frames and bands differ, so a time/band mix-up is observable.
    values = np.asarray(stack)
    assert not np.allclose(values[0], values[1])
    assert not np.allclose(values[:, 0], values[:, 1])


def test_frames_keep_grid_and_attrs() -> None:
    stack = time_stack()
    out = frames(stack)
    assert [f.shape for f in out] == [(3, 4, 4), (3, 4, 4)]
    assert all(f.transform == stack.transform for f in out)
    assert all(f.attrs == stack.attrs and f.attrs is not stack.attrs for f in out)
    np.testing.assert_array_equal(np.asarray(out[1]), np.asarray(stack)[1])


@pytest.mark.parametrize("shape", [(3, 4, 4), (2, 3, 4, 4), (1, 2, 3, 4, 4)])
def test_band_axis_is_minus_three(shape: tuple[int, ...]) -> None:
    arr = np.zeros(shape)
    assert band_axis(arr) == BAND_AXIS == -3
    assert arr.shape[band_axis(arr)] == shape[-3]


def test_band_axis_rejects_two_d() -> None:
    with pytest.raises(ValueError, match="2-D array has no band axis"):
        band_axis(np.zeros((4, 4)))


def test_require_ndim_returns_rank() -> None:
    assert require_ndim(np.zeros((4, 4)), (2, 3), "Op") == 2
    assert require_ndim(np.zeros((2, 3, 4, 4)), 4, "Op") == 4


def test_require_ndim_message_names_operator_ranks_and_shape() -> None:
    with pytest.raises(ValueError) as info:
        require_ndim(time_stack(), (2, 3), "OpticalFlowTVL1")
    assert str(info.value) == (
        "OpticalFlowTVL1 accepts 2-D (H, W) or 3-D (C, H, W) input; "
        "got a 4-D array of shape (2, 3, 4, 4)"
    )
    with pytest.raises(ValueError, match=r"MNF accepts 3-D \(C, H, W\) or 4-D"):
        require_ndim(np.zeros((4, 4)), [4, 3], "MNF")


def test_keep_band_axis() -> None:
    stack = np.zeros((2, 3, 4, 4))
    assert keep_band_axis(np.zeros((2, 4, 4)), stack).shape == (2, 1, 4, 4)
    # Only a 3-D result of a 4-D input changes.
    assert keep_band_axis(np.zeros((4, 4)), np.zeros((3, 4, 4))).shape == (4, 4)
    assert keep_band_axis(np.zeros((2, 5, 4, 4)), stack).shape == (2, 5, 4, 4)


def test_map_frames_restacks_per_frame_results() -> None:
    stack = time_stack(attrs={"band_names": ["a", "b", "c"], "sensor": "toy"})

    def band_sum(frame: Any) -> GeoTensor:
        return toy_geotensor(np.asarray(frame).sum(axis=0), fill_value_default=0)

    out = map_frames(band_sum, stack, name="BandSum")
    assert isinstance(out, GeoTensor)
    assert out.shape == (2, 1, 4, 4)
    assert out.transform == stack.transform
    assert out.fill_value_default == 0
    np.testing.assert_allclose(np.asarray(out)[:, 0], np.asarray(stack).sum(axis=1))


def test_map_frames_passes_other_ranks_through() -> None:
    cube = np.ones((3, 4, 4))
    np.testing.assert_array_equal(map_frames(lambda x: x * 2, cube, name="Op"), 2)


def test_map_frames_plain_arrays() -> None:
    stack = np.arange(2 * 3 * 4 * 4, dtype=float).reshape(2, 3, 4, 4)
    out = map_frames(lambda frame: frame[::-1], stack, name="Reverse")
    assert type(out) is np.ndarray
    np.testing.assert_array_equal(out, stack[:, ::-1])


def test_map_frames_rejects_non_carrier_results() -> None:
    with pytest.raises(ValueError, match="Stats returns float per frame"):
        map_frames(lambda frame: float(np.mean(frame)), time_stack(), name="Stats")


def test_map_frames_rejects_results_on_different_grids() -> None:
    sizes = iter([2, 3])

    def crop(frame: Any) -> np.ndarray:
        n = next(sizes)
        return np.asarray(frame)[:, :n, :n]

    with pytest.raises(ValueError, match="Crop produced per-frame results"):
        map_frames(crop, time_stack(), name="Crop")


def test_over_frames_decorates_apply() -> None:
    class Double:
        calls = 0

        @over_frames
        def _apply(self, x: Any, *, scale: float = 2.0) -> np.ndarray:
            type(self).calls += 1
            assert np.ndim(x) == 3
            return np.asarray(x) * scale

    op = Double()
    out = op._apply(np.ones((2, 3, 4, 4)), scale=3.0)
    assert out.shape == (2, 3, 4, 4) and np.all(out == 3.0)
    assert Double.calls == 2
    assert Double._apply.__name__ == "_apply"


def test_single_band() -> None:
    assert single_band(np.zeros((1, 4, 4))).shape == (4, 4)
    with pytest.raises(ValueError, match="Otsu expects a single-band map"):
        single_band(np.zeros((2, 4, 4)), name="Otsu")
