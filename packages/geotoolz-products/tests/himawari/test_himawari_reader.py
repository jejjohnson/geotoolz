"""``himawari.Reader`` on synthetic HSD segments."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime

import numpy as np
import pytest
from _hsd import CFAC, IR_CALIBRATION, OFFSET, SIZE, VIS_CALIBRATION
from rasterio.windows import Window

from geoproducts import ProductReader, himawari
from geoproducts._src.geostationary import FixedGrid
from geoproducts.himawari._src import hsd as hsd_module


def _counts(row: int, col: int, segments: int = 4) -> int:
    """The fixture's count at area pixel ``(row, col)``."""
    return 1000 + col + 100 * (row // (SIZE // segments) + 1)


def test_segments_assemble_in_any_order(full_disk) -> None:
    paths = full_disk(segments=4)
    reader = himawari.Reader(paths[::-1], calibration="counts")
    assert isinstance(reader, ProductReader)
    assert reader.segments == (1, 2, 3, 4)
    assert reader.paths == tuple(paths)
    assert reader.shape == (1, SIZE, SIZE)
    assert reader.dtype == np.uint16
    out = np.asarray(reader.load())[0]
    rows, cols = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    np.testing.assert_array_equal(out, _counts(rows, cols))


def test_missing_segments_read_as_fill(full_disk) -> None:
    paths = full_disk(segments=4)
    reader = himawari.Reader([paths[1], paths[3]], calibration="counts")
    out = np.asarray(reader.load())[0]
    assert (out[:25] == 65535).all() and (out[50:75] == 65535).all()
    assert out[25, 7] == _counts(25, 7) and out[99, 99] == _counts(99, 99)
    radiance = np.asarray(himawari.Reader([paths[1]]).load())[0]
    assert np.isnan(radiance[:25]).all()


def test_windows_decode_only_the_segments_they_touch(full_disk, monkeypatch) -> None:
    reader = himawari.Reader(full_disk(segments=4), calibration="counts")
    touched: list[int] = []
    real = hsd_module.read_lines

    def spy(path, header, rows, cols):
        touched.append(header.segment)
        return real(path, header, rows, cols)

    monkeypatch.setattr("geoproducts.himawari.reader.read_lines", spy)
    tile = reader.read_from_window(Window(10, 20, 30, 10))  # rows 20-29
    assert sorted(set(touched)) == [1, 2]
    expected = _counts(
        *np.meshgrid(np.arange(20, 30), np.arange(10, 40), indexing="ij")
    )
    np.testing.assert_array_equal(np.asarray(tile)[0], expected)
    # Boundless: the part past the grid is fill.
    edge = np.asarray(reader.read_from_window(Window(95, 95, 10, 10)))[0]
    assert (edge[5:, :] == 65535).all() and (edge[:, 5:] == 65535).all()
    assert edge[0, 0] == _counts(95, 95)


def test_compressed_segments_cache_one_decompressed_segment(full_disk) -> None:
    reader = himawari.Reader(full_disk(segments=4, bz2=True), calibration="counts")
    first = np.asarray(reader.read_from_window(Window(0, 30, 50, 5)))
    assert reader._cache is not None and reader._cache[0] == 1  # segment 2
    again = np.asarray(reader.read_from_window(Window(0, 30, 50, 5)))
    np.testing.assert_array_equal(first, again)
    clone = pickle.loads(pickle.dumps(reader))
    assert clone._cache is None
    np.testing.assert_array_equal(np.asarray(clone.load()), np.asarray(reader.load()))


def test_grid_is_the_cgms_projection(full_disk) -> None:
    reader = himawari.Reader(full_disk(segments=2))
    expected = FixedGrid.from_cgms(
        columns=SIZE,
        lines=SIZE,
        cfac=CFAC,
        lfac=CFAC,
        coff=OFFSET,
        loff=OFFSET,
        lon_0=140.7,
        height_m=35_785_863.0,
        semi_major_m=6_378_137.0,
        semi_minor_m=6_356_752.3,
    )
    assert reader.transform == expected.transform
    assert reader.crs == expected.crs
    assert "Geostationary" in reader.crs.to_wkt()
    # The disk spans the grid: the centre pixel looks at the sub-satellite
    # point, the pixel size is 55 x 2 km.
    assert reader.res[0] == pytest.approx(110_000.0, rel=1e-4)
    assert reader.transform.c == pytest.approx(-SIZE / 2 * reader.res[0])
    assert reader.satellite_lon_deg == 140.7
    assert reader.satellite_height_m == pytest.approx(35_785_863.0)


def test_metadata_and_attrs(full_disk) -> None:
    reader = himawari.Reader(
        full_disk(segments=2), calibration="brightness_temperature"
    )
    assert reader.band == "B13"
    assert reader.area == "FLDK"
    assert reader.satellite == "Himawari-9"
    assert reader.wavelength_um == pytest.approx(IR_CALIBRATION["wavelength_um"])
    assert reader.start_time < reader.end_time
    assert reader.units == "K"
    attrs = reader.read_from_window(Window(40, 40, 2, 2)).attrs
    assert attrs["band_names"] == ("B13",)
    assert attrs["units"] == ("K",)
    assert attrs["wavelengths"] == (pytest.approx(10407.3),)
    assert attrs["calibration"] == "brightness_temperature"
    assert "B13" in repr(reader) and "segments=(1, 2)" in repr(reader)
    assert reader.header.band == 13


def _single(hsd, value: int, *, band: int = 13, **kwargs) -> np.ndarray:
    counts = np.full((SIZE, SIZE), value, dtype=np.uint16)
    return hsd(counts, band=band, name=f"b{band}_{value}.DAT", **kwargs)


def test_radiance_and_brightness_temperature(hsd) -> None:
    path = _single(hsd, 3000)
    centre = Window(48, 48, 4, 4)
    radiance = np.asarray(himawari.Reader(path).read_from_window(centre))
    c = IR_CALIBRATION
    expected_l = 3000 * c["gain"] + c["offset"]
    np.testing.assert_allclose(radiance, expected_l, rtol=1e-6)
    bt = np.asarray(
        himawari.Reader(path, calibration="brightness_temperature").read_from_window(
            centre
        )
    )
    # Planck inversion at the band's central wavelength, by hand.
    speed, planck, boltzmann = c["planck"]
    lam = c["wavelength_um"] * 1e-6
    te = (planck * speed / (boltzmann * lam)) / np.log(
        2 * planck * speed**2 / (lam**5 * expected_l * 1e6) + 1
    )
    c0, c1, c2 = c["tb"]
    np.testing.assert_allclose(bt, c0 + c1 * te + c2 * te**2, rtol=1e-6)
    assert 200.0 < bt.mean() < 320.0


def test_reflectance_uses_the_updated_visible_calibration(hsd) -> None:
    path = _single(hsd, 500, band=3)
    centre = Window(48, 48, 4, 4)
    refl = np.asarray(
        himawari.Reader(path, calibration="reflectance").read_from_window(centre)
    )
    gain, offset = VIS_CALIBRATION["updated"]
    np.testing.assert_allclose(
        refl, (500 * gain + offset) * VIS_CALIBRATION["albedo"], rtol=1e-6
    )
    nominal = _single(hsd, 500, band=3, updated_vis=False)
    plain = np.asarray(himawari.Reader(nominal).read_from_window(centre))
    expected = 500 * VIS_CALIBRATION["gain"] + VIS_CALIBRATION["offset"]
    np.testing.assert_allclose(plain, expected, rtol=1e-6)


def test_error_outside_and_off_disk_pixels_are_nan(hsd) -> None:
    counts = np.full((SIZE, SIZE), 3000, dtype=np.uint16)
    counts[50, 50] = 65535
    counts[50, 51] = 65534
    reader = himawari.Reader(hsd(counts), calibration="brightness_temperature")
    bt = np.asarray(reader.load())[0]
    assert np.isnan(bt[50, 50]) and np.isnan(bt[50, 51])
    # Corners look past the limb into space: no geolocation, so NaN.
    assert np.isnan(bt[0, 0]) and np.isnan(bt[-1, -1]) and np.isnan(bt[50, 0])
    assert np.isfinite(bt[50, 5]) and np.isfinite(bt[5, 50])
    # Counts stay as stored, off-disk included.
    raw = np.asarray(
        himawari.Reader(hsd(counts, name="raw.DAT"), calibration="counts").load()
    )
    assert raw[0, 0, 0] == 3000


@pytest.mark.parametrize(
    ("band", "calibration", "match"),
    [
        (13, "reflectance", "reflectance applies to bands 1-6"),
        (3, "brightness_temperature", "brightness_temperature applies to bands 7-16"),
        (13, "albedo", "calibration must be one of"),
    ],
)
def test_calibration_must_fit_the_band(hsd, band, calibration, match) -> None:
    path = _single(hsd, 100, band=band)
    with pytest.raises(ValueError, match=match):
        himawari.Reader(path, calibration=calibration)


def test_segments_of_different_bands_or_scans_are_refused(hsd, full_disk) -> None:
    b13 = full_disk(13, segments=2)
    b14 = full_disk(14, segments=2)
    with pytest.raises(ValueError, match="'band'"):
        himawari.Reader([b13[0], b14[1]])
    with pytest.raises(ValueError, match="duplicated segments"):
        himawari.Reader([b13[0], b13[0]])
    with pytest.raises(ValueError, match="at least one"):
        himawari.Reader([])


def test_segments_of_different_scans_are_refused(hsd) -> None:
    rows = np.zeros((50, SIZE), np.uint16)
    first = hsd(rows, segment=1, total_segments=2, name="s1.DAT")
    later = hsd(rows, segment=2, total_segments=2, timeline=310, name="s2.DAT")
    with pytest.raises(ValueError, match="'timeline'"):
        himawari.Reader([first, later])
    next_day = hsd(
        rows,
        segment=2,
        total_segments=2,
        start=datetime(2026, 10, 8, 3, 4, tzinfo=UTC),
        name="s2b.DAT",
    )
    with pytest.raises(ValueError, match="different days"):
        himawari.Reader([first, next_day])


def test_navigation_corrections_warn(hsd) -> None:
    path = hsd(np.zeros((SIZE, SIZE), np.uint16), shift_px=1.5)
    with pytest.warns(UserWarning, match="navigation corrections"):
        reader = himawari.Reader(path)
    assert reader.header.max_shift_px == pytest.approx(1.5)


def test_each_segment_uses_its_own_calibration(hsd) -> None:
    rows = np.full((50, SIZE), 3000, np.uint16)
    top = hsd(rows, segment=1, total_segments=2, name="t.DAT")
    bottom = hsd(rows, segment=2, total_segments=2, gain=-0.002, name="b.DAT")
    radiance = np.asarray(himawari.Reader([top, bottom]).load())[0]
    offset = IR_CALIBRATION["offset"]
    assert radiance[49, 50] == pytest.approx(3000 * IR_CALIBRATION["gain"] + offset)
    assert radiance[50, 50] == pytest.approx(3000 * -0.002 + offset)
