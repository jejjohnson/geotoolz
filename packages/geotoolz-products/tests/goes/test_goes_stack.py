"""``geoproducts.goes.stack`` — readers onto one grid."""

from __future__ import annotations

import numpy as np
import pytest
from _goes_abi import STEP_RAD, X0_RAD, Y0_RAD
from affine import Affine
from georeader.geotensor import GeoTensor

from geoproducts import goes


def _ht(l2_file, shape, step, value=1000.0):
    data = np.full(shape, int(value / 0.5), dtype=np.uint16)
    attrs = {
        "_FillValue": np.uint16(65535),
        "scale_factor": np.float32(0.5),
        "add_offset": np.float32(0.0),
        "units": np.bytes_(b"m"),
    }
    # Pixel centres half a fine pixel in, so the coarse cells nest exactly.
    origin = (X0_RAD + (step - STEP_RAD) / 2, Y0_RAD - (step - STEP_RAD) / 2)
    return goes.L2Reader(l2_file({"HT": (data, attrs)}, step=step, origin=origin))


def test_same_grid_readers_stack_band_by_band(abi_file) -> None:
    c13 = goes.Reader(abi_file(), calibration="brightness_temperature")
    c07 = goes.Reader(abi_file(band_id=7), calibration="brightness_temperature")
    out = goes.stack([c13, c07])
    assert out.shape == (2, 6, 8)
    assert out.attrs["band_names"] == ("C13", "C07")
    assert out.attrs["units"] == ("K", "K")
    assert out.transform == c13.transform
    np.testing.assert_allclose(np.asarray(out)[1], np.asarray(c07.load())[0])


def test_finer_reader_is_averaged_onto_the_reference(abi_file, l2_file) -> None:
    coarse = _ht(l2_file, (3, 4), 2 * STEP_RAD)
    fine_counts = np.arange(48, dtype=np.uint16).reshape(6, 8) + 100
    fine = goes.Reader(abi_file(counts=fine_counts))
    out = goes.stack([coarse, fine])
    assert out.shape == (2, 3, 4)
    radiance = np.asarray(fine.load())[0]
    blocks = radiance.reshape(3, 2, 4, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(np.asarray(out)[1], blocks, rtol=1e-5)


def test_integer_masks_mix_into_a_float_stack(abi_file, l2_file) -> None:
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[0, 0] = 255  # fill
    mask[1, 1] = 1
    acm = goes.L2Reader(
        l2_file({"BCM": (mask, {"_FillValue": np.uint8(255)})}, code="ACM"),
        variables="BCM",
    )
    out = goes.stack([goes.Reader(abi_file()), acm])
    assert out.dtype == np.float32
    assert np.isnan(np.asarray(out)[1, 0, 0])
    assert np.asarray(out)[1, 1, 1] == 1.0
    assert np.isnan(out.fill_value_default)


def test_all_integer_readers_stay_integer(abi_file) -> None:
    path = abi_file()
    quality = goes.Reader(path).quality
    out = goes.stack([quality, goes.QualityReader(path)])
    assert out.dtype == np.uint8
    assert out.fill_value_default == 255


def test_bounds_select_the_reference_area(abi_file) -> None:
    reader = goes.Reader(abi_file())
    t = reader.transform
    left, top = t * (2, 1)
    right, bottom = t * (6, 4)
    out = goes.stack([reader], bounds=(left, bottom, right, top))
    assert out.shape == (1, 3, 4)


def test_like_picks_the_reference_grid(abi_file, l2_file) -> None:
    coarse = _ht(l2_file, (3, 4), 2 * STEP_RAD)
    fine = goes.Reader(abi_file())
    out = goes.stack([coarse, fine], like=1)
    assert out.shape == (2, 6, 8)
    assert out.attrs["band_names"] == ("HT", "C13")
    np.testing.assert_allclose(np.nanmean(np.asarray(out)[0]), 1000.0)


def test_generic_geodata_without_per_band_units(abi_file) -> None:
    reader = goes.Reader(abi_file())
    other = reader.load()
    plain = GeoTensor(
        np.asarray(other),
        transform=other.transform,
        crs=other.crs,
        fill_value_default=np.nan,
    )
    out = goes.stack([reader, plain])
    assert "band_names" not in out.attrs
    assert out.shape == (2, 6, 8)


def test_explicit_resampling_and_errors(abi_file) -> None:
    reader = goes.Reader(abi_file())
    out = goes.stack([reader, reader], resampling="nearest")
    assert out.shape == (2, 6, 8)
    with pytest.raises(ValueError, match="at least one"):
        goes.stack([])
    with pytest.raises(ValueError, match="out of range"):
        goes.stack([reader], like=1)
    assert isinstance(reader.transform, Affine)
