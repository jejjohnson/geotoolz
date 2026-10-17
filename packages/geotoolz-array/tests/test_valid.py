"""`GeoArray.valid_mask` (gate GB4): the nodata rule geoops uses."""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from conftest import CRS, TRANSFORM

from geoarray import GeoArray


def test_nan_fill_and_non_finite() -> None:
    data = np.ones((2, 3, 3), dtype=np.float32)
    data[0, 0, 0] = np.nan
    data[1, 1, 1] = np.inf
    mask = GeoArray(data, TRANSFORM, CRS, fill_value=np.nan).valid_mask()
    assert mask.shape == (3, 3)
    assert mask.sum() == 7
    assert not mask[0, 0]
    assert not mask[1, 1]


def test_numeric_fill_any_band() -> None:
    data = np.ones((2, 2, 2), dtype=np.uint16)
    data[1, 0, 1] = 0
    mask = GeoArray(data, TRANSFORM, CRS, fill_value=0).valid_mask()
    assert mask.tolist() == [[True, False], [True, True]]


def test_no_fill_on_integers_is_all_valid() -> None:
    data = np.zeros((4, 4), dtype=np.int32)
    assert GeoArray(data, TRANSFORM, CRS).valid_mask().all()


def test_time_axis() -> None:
    data = np.ones((3, 2, 2, 2))
    data[1, 0, 0, 0] = -9999
    g = GeoArray(data, TRANSFORM, CRS, fill_value=-9999)
    assert g.valid_mask().shape == (2, 2)
    assert not g.valid_mask()[0, 0]
    per_time = g.valid_mask(keep_time=True)
    assert per_time.shape == (3, 2, 2)
    assert per_time[:, 0, 0].tolist() == [True, False, True]


def test_batched_keeps_the_batch_axis() -> None:
    data = np.ones((2, 3, 2, 2))
    data[1, 2, 1, 1] = np.nan
    transforms = np.stack([np.array(TRANSFORM)] * 2)
    mask = GeoArray(data, transforms, CRS, fill_value=np.nan).valid_mask()
    assert mask.shape == (2, 2, 2)
    assert mask.sum() == 7


def test_jit(x64: ModuleType) -> None:
    data = np.ones((2, 3, 3))
    data[0, 2, 2] = 0.0
    g = GeoArray(data, TRANSFORM, CRS, fill_value=0.0)
    mask = x64.jit(lambda c: c.valid_mask())(g.to_jax())
    np.testing.assert_array_equal(np.asarray(mask), g.valid_mask())


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    ("dtype", "fill"),
    [(np.float32, np.nan), (np.float64, -9999.0), (np.uint16, 0), (np.int16, -1)],
)
def test_parity_with_geoops(seed: int, dtype: type, fill: float) -> None:
    """GB4: the same pixels as geoops' nodata rule, on random scenes."""
    valid = pytest.importorskip("geoops._src.valid")
    from affine import Affine
    from georeader.geotensor import GeoTensor

    rng = np.random.default_rng(seed)
    data = rng.integers(0, 5, size=(3, 16, 16)).astype(dtype)
    holes = rng.random(data.shape) < 0.1
    data[holes] = fill
    if np.issubdtype(dtype, np.floating):
        data[rng.random(data.shape) < 0.05] = np.inf
    theirs = GeoTensor(data, Affine(*TRANSFORM), CRS, fill_value_default=fill)
    expected = np.asarray(valid.valid_pixels(theirs))
    mine = GeoArray(data, TRANSFORM, CRS, fill_value=fill).valid_mask()
    np.testing.assert_array_equal(mine, expected)
