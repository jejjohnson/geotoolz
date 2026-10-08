"""End-to-end checks against real ABI files from NOAA's public AWS buckets.

The buckets are anonymous and free, so these need only network access:
they list a fixed hour of mesoscale scans, download two small files
(~0.4 MB C13 and ~1-2 MB C01, one minute's scan each) and check the
reader against the file's own metadata. Marked ``integration``: they run
from the "Extended Tests" workflow or ``make test-slow``, never in the
automatic CI tier::

    pytest -m integration tests/goes
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
from rasterio.warp import transform as warp_transform

from geoproducts import goes, stack
from geoproducts.goes import aws, constants


pytestmark = pytest.mark.integration

# A fixed, archived hour: the buckets keep the full record.
START = datetime(2026, 10, 7, 12, 0)
END = datetime(2026, 10, 7, 12, 5)


def _first(channel: int, satellite: str = "G19") -> aws.ABIFile:
    files = aws.list_files(
        satellite=satellite,
        product="ABI-L1b-RadM",
        start=START,
        end=END,
        channel=channel,
        sector="M1",
    )
    assert files, f"no {satellite} RadM1 C{channel:02d} files between {START} and {END}"
    return files[0]


@pytest.fixture(scope="module")
def downloads(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    dest = tmp_path_factory.mktemp("goes")
    return {
        "C13": aws.download(_first(13), dest),
        "C01": aws.download(_first(1), dest),
    }


def test_listing_finds_one_minute_mesoscale_scans() -> None:
    files = aws.list_files(
        satellite="G19", product="ABI-L1b-RadM", start=START, end=END, channel=13
    )
    assert {f.sector for f in files} == {"M1", "M2"}
    assert all(START <= f.start.replace(tzinfo=None) < END for f in files)
    assert all(f.size and f.size > 0 for f in files)


def test_brightness_temperature_scene(downloads: dict[str, Path]) -> None:
    reader = goes.Reader(downloads["C13"], calibration="brightness_temperature")
    assert reader.channel == "C13"
    assert reader.platform == "G19"
    assert reader.scene == "Mesoscale"
    assert reader.shape == (1, 500, 500)  # 2 km channel, 1000 km sector
    assert reader.res[0] == pytest.approx(2004.0, abs=1.0)
    assert reader.satellite_lon_deg == pytest.approx(constants.GOES_EAST_LON_DEG)
    bt = np.asarray(reader.load())
    assert bt.dtype == np.float32
    assert np.isfinite(bt).mean() > 0.99
    assert 150.0 < np.nanmin(bt) < np.nanmax(bt) < 340.0


def test_grid_matches_the_files_own_coordinates(downloads: dict[str, Path]) -> None:
    reader = goes.Reader(downloads["C13"])
    with h5py.File(downloads["C13"]) as f:
        x, y = f["x"], f["y"]
        x_rad = x[()] * x.attrs["scale_factor"][0] + x.attrs["add_offset"][0]
        y_rad = y[()] * y.attrs["scale_factor"][0] + y.attrs["add_offset"][0]
        height = f["goes_imager_projection"].attrs["perspective_point_height"][0]
        extent = f["geospatial_lat_lon_extent"].attrs
        lon_c = float(extent["geospatial_lon_center"][0])
        lat_c = float(extent["geospatial_lat_center"][0])
    cols = np.arange(reader.width) + 0.5
    rows = np.arange(reader.height) + 0.5
    xs, _ = reader.transform * (cols, np.full_like(cols, 0.5))
    _, ys = reader.transform * (np.full_like(rows, 0.5), rows)
    # Within 1 % of a pixel of the file's scan angles.
    np.testing.assert_allclose(xs / height, x_rad, atol=0.01 * reader.res[0] / height)
    np.testing.assert_allclose(ys / height, y_rad, atol=0.01 * reader.res[1] / height)
    # The grid centre projects onto the file's reported lat/lon centre.
    t = reader.transform
    cx, cy = t * (reader.width / 2, reader.height / 2)
    (lon,), (lat,) = warp_transform(reader.crs, "EPSG:4326", [cx], [cy])
    assert lon == pytest.approx(lon_c, abs=0.05)
    assert lat == pytest.approx(lat_c, abs=0.05)


def test_reflectance_and_quality(downloads: dict[str, Path]) -> None:
    reader = goes.Reader(downloads["C01"], calibration="reflectance")
    assert reader.shape == (1, 1000, 1000)  # 1 km channel
    refl = np.asarray(reader.load())
    assert -0.05 < np.nanmin(refl) < np.nanmax(refl) < 1.6
    flags = np.unique(np.asarray(reader.quality.load()))
    assert set(flags.tolist()) <= {*constants.DQF_FLAGS, constants.DQF_FILL}


def test_read_from_lon_lat_bounds(downloads: dict[str, Path]) -> None:
    reader = goes.Reader(downloads["C13"], calibration="brightness_temperature")
    t = reader.transform
    cx, cy = t * (reader.width / 2, reader.height / 2)
    (lon,), (lat,) = warp_transform(reader.crs, "EPSG:4326", [cx], [cy])
    chip = reader.read_from_bounds(
        (lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5), crs_bounds="EPSG:4326"
    )
    assert chip.crs == reader.crs
    assert 40 < chip.shape[-1] < 80  # ~1 degree of 2 km pixels
    assert np.isfinite(np.asarray(chip)).all()


# -- L2 products ------------------------------------------------------------

L2_START = datetime(2026, 10, 7, 18, 0)  # daytime over the Americas
L2_END = datetime(2026, 10, 7, 18, 2)


def _l2(product: str) -> aws.ABIFile:
    files = aws.list_files(
        satellite="G19", product=product, start=L2_START, end=L2_END, sector="M1"
    )
    assert files, f"no {product} M1 files between {L2_START} and {L2_END}"
    return files[0]


@pytest.fixture(scope="module")
def l2_downloads(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    dest = tmp_path_factory.mktemp("goes-l2")
    return {
        code: aws.download(_l2(f"ABI-L2-{code}M"), dest)
        for code in ("ACM", "ACHA", "MCMIP")  # ~0.25, 0.25 and 4.8 MB
    }


def test_l2_file_names_parse(l2_downloads: dict[str, Path]) -> None:
    item = aws.parse_key(l2_downloads["ACM"].name)
    assert item.product == "ABI-L2-ACMM"
    assert item.sector == "M1"
    assert item.channel is None


def test_clear_sky_mask_matches_the_files_own_counts(
    l2_downloads: dict[str, Path],
) -> None:
    reader = goes.L2Reader(l2_downloads["ACM"])
    assert reader.product == "ACM"
    assert reader.bands == ("BCM", "ACM")
    assert reader.flags("ACM") == {
        0: "clear",
        1: "probably_clear",
        2: "probably_cloudy",
        3: "cloudy",
    }
    acm = np.asarray(reader.load())[1]
    with h5py.File(l2_downloads["ACM"]) as f:
        expected = {
            constants.ACM_CLEAR: f["percent_clear_pixels"][()],
            constants.ACM_PROBABLY_CLEAR: f["percent_probably_clear_pixels"][()],
            constants.ACM_PROBABLY_CLOUDY: f["percent_probably_cloudy_pixels"][()],
            constants.ACM_CLOUDY: f["percent_cloudy_pixels"][()],
        }
    valid = acm != reader.fill_value_default
    for code, fraction in expected.items():
        assert (acm[valid] == code).mean() == pytest.approx(float(fraction), abs=1e-5)


def test_mcmip_has_every_channel_calibrated(l2_downloads: dict[str, Path]) -> None:
    reader = goes.L2Reader(l2_downloads["MCMIP"])
    assert reader.bands == constants.CHANNELS
    assert reader.shape == (16, 500, 500)
    cmi = np.asarray(reader.load())
    reflectance, bt = cmi[:6], cmi[6:]
    assert -0.05 < np.nanmin(reflectance) < np.nanmax(reflectance) < 1.6
    assert 150.0 < np.nanmin(bt) < np.nanmax(bt) < 340.0
    assert reader.quality.bands[0] == "DQF_C01"


def test_cloud_top_height_stacks_onto_the_imagery_grid(
    l2_downloads: dict[str, Path],
) -> None:
    imagery = goes.L2Reader(l2_downloads["MCMIP"])
    height = goes.L2Reader(l2_downloads["ACHA"])
    assert height.shape == (1, 250, 250)  # 4 km product
    mask = goes.L2Reader(l2_downloads["ACM"], variables="BCM")
    scene = stack([imagery, mask, height])
    assert scene.shape == (18, 500, 500)
    assert scene.attrs["band_names"][-2:] == ("BCM", "HT")
    ht = np.asarray(scene)[-1]
    assert 0.0 <= np.nanmin(ht) < np.nanmax(ht) < 20_000.0
    # Heights are retrieved over cloud: far more coverage where the (2 km)
    # mask says cloudy than where it says clear, which on a 4 km product is
    # mostly cloud edges.
    bcm = np.asarray(scene)[-2]
    over_cloud = np.isfinite(ht[bcm == constants.BCM_CLOUDY]).mean()
    over_clear = np.isfinite(ht[bcm == constants.BCM_CLEAR]).mean()
    assert over_cloud > 0.8
    assert over_cloud - over_clear > 0.4


def test_rgb_recipes_on_real_imagery(l2_downloads: dict[str, Path]) -> None:
    pytest.importorskip("geotoolz")
    cmi = goes.L2Reader(l2_downloads["MCMIP"]).load()
    for name in goes.recipes.RECIPES:
        rgb = np.asarray(goes.presets.Recipe(name)(cmi))
        assert rgb.shape == (3, 500, 500)
        assert np.nanmin(rgb) >= 0.0 and np.nanmax(rgb) <= 1.0
        assert np.isfinite(rgb).mean() > 0.99
