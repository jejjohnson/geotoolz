"""``geoproducts.goes.Reader`` on synthetic ABI L1b files."""

from __future__ import annotations

import builtins
import io
import pickle
from datetime import UTC, datetime

import h5py
import numpy as np
import pytest
from _goes_abi import KAPPA0, PERSPECTIVE_HEIGHT_M, PLANCK, STEP_RAD, X0_RAD, Y0_RAD
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geoproducts import ProductReader, goes
from geoproducts.goes import constants


def _radiance(counts: np.ndarray, scale: float = 0.5, offset: float = -1.0):
    return counts.astype(np.float32) * np.float32(scale) + np.float32(offset)


class TestGrid:
    def test_affine_is_recovered_from_the_scan_angles(self, abi_file) -> None:
        reader = goes.Reader(abi_file())
        t = reader.transform
        step_m = STEP_RAD * PERSPECTIVE_HEIGHT_M
        assert t.a == pytest.approx(step_m)
        assert t.e == pytest.approx(-step_m)
        assert t.c == pytest.approx((X0_RAD - STEP_RAD / 2) * PERSPECTIVE_HEIGHT_M)
        assert t.f == pytest.approx((Y0_RAD + STEP_RAD / 2) * PERSPECTIVE_HEIGHT_M)
        # Pixel centres land back on the file's own scan angles.
        x, y = t * (3.5, 2.5)
        assert x / PERSPECTIVE_HEIGHT_M == pytest.approx(X0_RAD + 3 * STEP_RAD)
        assert y / PERSPECTIVE_HEIGHT_M == pytest.approx(Y0_RAD - 2 * STEP_RAD)

    def test_crs_is_the_fixed_grid_projection(self, abi_file) -> None:
        # rasterio's ``to_proj4`` drops ``+sweep``; the WKT keeps the full
        # PROJ string (and GDAL honours it when reprojecting).
        proj4 = goes.Reader(abi_file()).crs.to_wkt()
        assert "+proj=geos" in proj4
        assert "+lon_0=-75" in proj4
        assert f"+h={PERSPECTIVE_HEIGHT_M:.0f}" in proj4
        assert "+sweep=x" in proj4

    def test_reader_contract(self, abi_file) -> None:
        reader = goes.Reader(abi_file())
        assert isinstance(reader, ProductReader)
        assert reader.shape == (1, 6, 8)
        assert reader.dims == ("band", "y", "x")
        assert reader.bands == ("C13",)
        assert reader.track == "A"


class TestCalibration:
    def test_radiance_scales_counts_and_masks_the_fill(self, abi_file) -> None:
        counts = np.full((4, 5), 200, dtype=np.uint16)
        counts[1, 2] = 16383
        reader = goes.Reader(abi_file(counts=counts))
        out = np.asarray(reader.load())[0]
        assert reader.dtype == np.float32
        assert np.isnan(out[1, 2])
        assert np.isnan(reader.fill_value_default)
        expected = _radiance(counts)
        expected[1, 2] = np.nan
        np.testing.assert_array_equal(out, expected)

    def test_counts_are_read_unsigned(self, abi_file) -> None:
        counts = np.array([[1, 40_000], [65_000, 7]], dtype=np.uint16)
        reader = goes.Reader(abi_file(counts=counts, fill=65_535), calibration="counts")
        out = reader.load()
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(np.asarray(out)[0], counts)
        assert reader.fill_value_default == 65_535

    def test_reflectance_is_kappa0_times_radiance(self, abi_file) -> None:
        counts = np.arange(12, dtype=np.uint16).reshape(3, 4) + 300
        reader = goes.Reader(
            abi_file(band_id=1, counts=counts, fill=1023), calibration="reflectance"
        )
        np.testing.assert_allclose(
            np.asarray(reader.load())[0], _radiance(counts) * np.float32(KAPPA0)
        )
        assert reader.units == "1"
        assert reader.channel == "C01"

    def test_brightness_temperature_uses_the_planck_coefficients(
        self, abi_file
    ) -> None:
        counts = np.array([[200, 150], [180, 2]], dtype=np.uint16)
        reader = goes.Reader(
            abi_file(counts=counts), calibration="brightness_temperature"
        )
        out = np.asarray(reader.load())[0]
        rad = _radiance(counts).astype(np.float64)
        with np.errstate(divide="ignore"):
            expected = (
                PLANCK["fk2"] / np.log(PLANCK["fk1"] / rad + 1.0) - PLANCK["bc1"]
            ) / PLANCK["bc2"]
        expected[rad <= 0] = np.nan  # counts == 2 → radiance 0: no temperature
        np.testing.assert_allclose(out, expected, rtol=1e-6)
        assert reader.units == "K"
        assert 200.0 < np.nanmin(out) < np.nanmax(out) < 400.0

    def test_unknown_calibration_is_rejected(self, abi_file) -> None:
        with pytest.raises(ValueError, match="calibration must be one of"):
            goes.Reader(abi_file(), calibration="albedo")  # type: ignore[arg-type]

    def test_reflectance_needs_a_reflective_channel(self, abi_file) -> None:
        with pytest.raises(ValueError, match="C13 has no kappa0"):
            goes.Reader(abi_file(band_id=13), calibration="reflectance")

    def test_brightness_temperature_needs_an_emissive_channel(self, abi_file) -> None:
        with pytest.raises(ValueError, match="C02 has no Planck"):
            goes.Reader(
                abi_file(band_id=2, fill=4095), calibration="brightness_temperature"
            )

    def test_calibration_coefficients_report_nan_when_not_applicable(
        self, abi_file
    ) -> None:
        coeffs = goes.Reader(abi_file(band_id=13)).calibration_coefficients
        assert np.isnan(coeffs["kappa0"])
        assert coeffs["planck_fk1"] == pytest.approx(PLANCK["fk1"])
        assert coeffs["scale_factor"] == 0.5


class TestWindows:
    def test_window_matches_the_full_load(self, abi_file) -> None:
        reader = goes.Reader(abi_file())
        full = np.asarray(reader.load())
        chip = reader.read_from_window(Window(2, 1, 4, 3))
        assert isinstance(chip, GeoTensor)
        assert chip.shape == (1, 3, 4)
        np.testing.assert_array_equal(np.asarray(chip), full[:, 1:4, 2:6])
        assert chip.transform.c == pytest.approx(reader.transform.c + 2 * reader.res[0])

    def test_boundless_window_pads_with_the_fill(self, abi_file) -> None:
        reader = goes.Reader(abi_file())
        chip = np.asarray(reader.read_from_window(Window(-2, -1, 4, 3)))
        assert np.isnan(chip[0, 0, :]).all()
        assert np.isnan(chip[0, :, :2]).all()
        np.testing.assert_array_equal(
            chip[0, 1:, 2:], np.asarray(reader.load())[0, :2, :2]
        )

    def test_window_outside_the_file_is_all_fill(self, abi_file) -> None:
        chip = goes.Reader(abi_file(), calibration="counts").read_from_window(
            Window(100, 100, 2, 2)
        )
        assert (np.asarray(chip) == 16383).all()

    def test_bounded_read_clips_to_the_file(self, abi_file) -> None:
        chip = goes.Reader(abi_file()).read_from_window(
            Window(6, 4, 10, 10), boundless=False
        )
        assert chip.shape == (1, 2, 2)

    def test_read_from_bounds_matches_the_window(self, abi_file) -> None:
        reader = goes.Reader(abi_file())
        t = reader.transform
        left, top = t * (1, 2)
        right, bottom = t * (5, 5)
        out = reader.read_from_bounds((left, bottom, right, top))
        np.testing.assert_array_equal(
            np.asarray(out), np.asarray(reader.load())[:, 2:5, 1:5]
        )

    def test_output_attrs_name_the_band_and_quantity(self, abi_file) -> None:
        attrs = (
            goes.Reader(abi_file(), calibration="brightness_temperature").load().attrs
        )
        assert attrs["band_names"] == ("C13",)
        assert attrs["wavelengths"] == (pytest.approx(10300.0),)
        assert attrs["units"] == "K"
        assert attrs["calibration"] == "brightness_temperature"


class TestQuality:
    def test_dqf_is_read_as_uint8_flags(self, abi_file) -> None:
        dqf = np.zeros((6, 8), dtype=np.uint8)
        dqf[0, 0] = constants.DQF_NO_VALUE
        dqf[5, 7] = constants.DQF_FILL
        quality = goes.Reader(abi_file(dqf=dqf)).quality
        assert isinstance(quality, goes.QualityReader)
        out = quality.load()
        assert out.dtype == np.uint8
        assert out.attrs["band_names"] == ("DQF",)
        np.testing.assert_array_equal(np.asarray(out)[0], dqf)
        padded = np.asarray(quality.read_from_window(Window(-1, 0, 2, 1)))
        assert padded.tolist() == [[[constants.DQF_FILL, constants.DQF_NO_VALUE]]]

    def test_quality_reader_shares_the_grid(self, abi_file) -> None:
        path = abi_file()
        reader = goes.Reader(path)
        direct = goes.QualityReader(path)
        assert direct.transform == reader.quality.transform == reader.transform
        assert "QualityReader" in repr(direct)

    def test_flag_table(self) -> None:
        assert constants.DQF_FLAGS[constants.DQF_GOOD] == "good"
        assert len(constants.DQF_FLAGS) == 5


class TestMetadata:
    def test_file_metadata(self, abi_file) -> None:
        path = abi_file(platform="G18", scene="Full Disk")
        reader = goes.Reader(path)
        assert reader.path == path
        assert reader.platform == "G18"
        assert reader.scene == "Full Disk"
        assert reader.wavelength_um == pytest.approx(10.3)
        assert reader.start_time == datetime(2026, 10, 7, 12, 1, 17, 800_000, UTC)
        assert reader.end_time == datetime(2026, 10, 7, 12, 3, 55, 100_000, UTC)
        assert reader.satellite_lon_deg == pytest.approx(-75.2)
        assert reader.satellite_height_m == pytest.approx(35_786_023.0, abs=1.0)
        assert "channel='C13'" in repr(reader)

    def test_band_table(self) -> None:
        bands = goes.BANDS
        assert [row["name"] for row in bands] == list(constants.CHANNELS)
        assert {row["kind"] for row in bands[:6]} == {"reflective"}
        assert {row["kind"] for row in bands[6:]} == {"emissive"}
        assert float(bands[1]["resolution_km"]) == 0.5
        with pytest.raises(AttributeError):
            _ = goes.NOT_A_NAME  # type: ignore[attr-defined]


class TestSources:
    def test_file_like_source(self, abi_file) -> None:
        path = abi_file()
        reader = goes.Reader(io.BytesIO(path.read_bytes()))
        assert reader.path is None
        np.testing.assert_array_equal(
            np.asarray(reader.load()), np.asarray(goes.Reader(path).load())
        )

    def test_path_reader_pickles(self, abi_file) -> None:
        reader = goes.Reader(abi_file(), calibration="brightness_temperature")
        clone = pickle.loads(pickle.dumps(reader))
        np.testing.assert_array_equal(
            np.asarray(clone.load()), np.asarray(reader.load())
        )

    def test_non_abi_file_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "other.nc"
        with h5py.File(path, "w") as f:
            f.create_dataset("temperature", data=np.zeros((2, 2)))
        with pytest.raises(ValueError, match="not a GOES-R ABI L1b radiance file"):
            goes.Reader(path)

    def test_missing_h5py_names_the_extra(self, abi_file, monkeypatch) -> None:
        path = abi_file()
        real_import = builtins.__import__

        def no_h5py(name, *args, **kwargs):
            if name == "h5py":
                raise ImportError("No module named 'h5py'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_h5py)
        with pytest.raises(ImportError, match=r"geotoolz-products\[goes\]"):
            goes.Reader(path)
