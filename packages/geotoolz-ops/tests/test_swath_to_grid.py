"""`geom.coregister.SwathToGrid`: a swath gridded by one functax IDW plan (#645)."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import toy_geoarray
from affine import Affine
from geoarray import GeoArray, GeoSwath

from geoops.geom.coregister import PointCloudToRaster, SwathToGrid


pytest.importorskip("functax")

_UTM = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)


def _like(shape: tuple[int, int] = (8, 10)) -> GeoArray:
    return toy_geoarray(np.zeros(shape), transform=_UTM, crs="EPSG:32629")


def _swath(bands: int = 2, seed: int = 0, crs: str = "EPSG:32629") -> GeoSwath:
    """A rotated 6 × 7 swath over the grid, geolocated in ``crs``."""
    rng = np.random.default_rng(seed)
    rows, cols = np.meshgrid(np.arange(6.0), np.arange(7.0), indexing="ij")
    x = 500_007.0 + 13.0 * cols + 3.0 * rows
    y = 3_999_993.0 - 12.0 * rows + 2.0 * cols
    data = rng.normal(size=(bands, 6, 7))
    return GeoSwath(
        data,
        y,
        x,
        dims="band row col",
        crs=crs,
        fill_value=np.nan,
        attrs={"band_names": tuple(f"b{i}" for i in range(bands))},
    )


def test_swath_to_grid_matches_point_cloud_to_raster_per_band() -> None:
    """(C, Nr, Nc) -> (C, Ny, Nx): each band is the point-cloud IDW of its pixels."""
    swath, like = _swath(), _like()
    out = SwathToGrid(k=4, power=2.0)(swath, like)
    assert isinstance(out, GeoArray)
    assert out.shape == (2, 8, 10)
    assert out.attrs["band_names"] == ("b0", "b1")
    np.testing.assert_array_equal(np.asarray(out.transform), np.asarray(like.transform))
    xy = np.column_stack([np.ravel(swath.lon), np.ravel(swath.lat)])
    for band in range(2):
        want = PointCloudToRaster(method="idw", k=4, power=2.0)(
            (xy, np.asarray(swath)[band].ravel()), like
        )
        np.testing.assert_allclose(np.asarray(out)[band], np.asarray(want), atol=1e-12)


def test_swath_to_grid_skips_a_masked_pixel() -> None:
    """An invalid swath pixel is never a neighbour: targets read the next valid ones."""
    swath, like = _swath(), _like()
    data = np.asarray(swath).copy()
    data[1, 2, 3] = np.nan  # invalid in one band: the whole pixel is skipped
    masked = swath.with_data(data)
    out = np.asarray(SwathToGrid(k=4)(masked, like))
    assert np.isfinite(out).all()
    keep = np.ones(42, dtype=bool)
    keep[2 * 7 + 3] = False
    xy = np.column_stack([np.ravel(swath.lon), np.ravel(swath.lat)])[keep]
    want = PointCloudToRaster(method="idw", k=4)((xy, data[0].ravel()[keep]), like)
    np.testing.assert_allclose(out[0], np.asarray(want), atol=1e-12)


def test_swath_to_grid_transforms_the_geolocation_into_the_target_crs() -> None:
    import pyproj

    utm = _swath()
    to_lonlat = pyproj.Transformer.from_crs("EPSG:32629", "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(np.asarray(utm.lon), np.asarray(utm.lat))
    lonlat = GeoSwath(np.asarray(utm), lat, lon, dims="band row col", fill_value=np.nan)
    like = _like()
    np.testing.assert_allclose(
        np.asarray(SwathToGrid()(lonlat, like)),
        np.asarray(SwathToGrid()(utm, like)),
        atol=1e-6,
    )


def test_swath_to_grid_nearest_and_max_radius() -> None:
    swath, like = _swath(bands=1), _like((20, 20))
    out = np.asarray(SwathToGrid(method="nearest", max_radius=10.0)(swath, like))
    assert np.isnan(out).any()
    assert set(out[np.isfinite(out)]) <= set(np.asarray(swath).ravel())


def test_swath_to_grid_refuses_a_raster_and_bad_settings() -> None:
    with pytest.raises(ValueError, match="method"):
        SwathToGrid(method="bilinear")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must be"):
        SwathToGrid(k=0)
    with pytest.raises(TypeError, match="GeoSwath"):
        SwathToGrid()(_like(), _like())


def test_the_bands_and_their_coords_come_from_the_swath_not_the_target() -> None:
    """A 3-D target with dims (time, y, x) lends its grid only, not its dims."""
    from geoarray.coords import SpectralAxis

    swath = _swath()
    swath = GeoSwath(
        np.asarray(swath),
        swath.lat,
        swath.lon,
        dims="band row col",
        coords=(SpectralAxis(("b0", "b1")),),
        crs="EPSG:32629",
        fill_value=np.nan,
    )
    like = GeoArray(
        np.zeros((3, 8, 10)), _UTM, "EPSG:32629", dims="time y x", fill_value=0.0
    )
    out = SwathToGrid(k=4)(swath, like)
    assert out.dims == ("band", "y", "x")
    assert out.labels("band") == ("b0", "b1")
    assert out.shape == (2, 8, 10)
    np.testing.assert_array_equal(np.asarray(out.transform), np.asarray(like.transform))
    two_d = SwathToGrid(k=4)(swath, _like())
    np.testing.assert_allclose(np.asarray(out), np.asarray(two_d), atol=1e-12)
