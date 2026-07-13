"""CRS-aware patching — issue #20.

Two independent levels:

- Level 1: ``crs=`` on the coordinate-consuming samplers
  (`SpatialAlongTrack`, `SpatialExplicitCoords`) reprojects anchors to
  the domain CRS before the pixel mapping.
- Level 2: `ReprojectingRasterField` presents the destination grid as
  its domain, so every axis works on the reprojected grid unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher import (
    RasterField,
    ReprojectingRasterField,
    SpatialAlongTrack,
    SpatialBoxcar,
    SpatialExplicitCoords,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)


pyproj = pytest.importorskip("pyproj")


# UTM zone 30N, 10 m pixels, 100x100 — a realistic projected raster.
_UTM_TRANSFORM = rasterio.Affine(10, 0, 500_000, 0, -10, 4_600_000)
_UTM_CRS = "EPSG:32630"


def _utm_field(n: int = 100) -> RasterField:
    arr = np.arange(n * n, dtype=np.float32).reshape(n, n)
    return RasterField(GeoTensor(values=arr, transform=_UTM_TRANSFORM, crs=_UTM_CRS))


def _pixel_center_lonlat(row: int, col: int) -> tuple[float, float]:
    """lon/lat of the centre of pixel ``(row, col)`` in the UTM field."""
    x = _UTM_TRANSFORM.c + (col + 0.5) * _UTM_TRANSFORM.a
    y = _UTM_TRANSFORM.f + (row + 0.5) * _UTM_TRANSFORM.e
    to_lonlat = pyproj.Transformer.from_crs(_UTM_CRS, "EPSG:4326", always_xy=True)
    return to_lonlat.transform(x, y)


class TestAnchorReprojection:
    def test_lonlat_coord_lands_on_hand_computed_pixel(self) -> None:
        field = _utm_field()
        geom = SpatialRectangular(size=(8, 8))
        # A lon/lat that is exactly the centre of pixel (50, 50).
        lon, lat = _pixel_center_lonlat(50, 50)
        sampler = SpatialExplicitCoords(coords=[(lon, lat)], crs="EPSG:4326")
        anchors = list(sampler.anchors(field.domain, geom))
        # Centred UL for a pixel at (50, 50) with an 8x8 patch → (46, 46).
        assert anchors == [(46, 46)]

    def test_crs_none_and_equal_are_noops(self) -> None:
        field = _utm_field()
        geom = SpatialRectangular(size=(8, 8))
        coords = [(500_505.0, 4_599_495.0), (500_105.0, 4_599_895.0)]  # UTM already
        none = list(SpatialExplicitCoords(coords=coords).anchors(field.domain, geom))
        same = list(
            SpatialExplicitCoords(coords=coords, crs=_UTM_CRS).anchors(
                field.domain, geom
            )
        )
        assert none == same
        assert none  # sanity: the coords are in-domain

    def test_alongtrack_crs_matches_manual_transform(self) -> None:
        field = _utm_field()
        geom = SpatialRectangular(size=(8, 8))
        # A short lon/lat track over the UTM field.
        track_lonlat = [_pixel_center_lonlat(r, r) for r in (20, 40, 60)]
        to_utm = pyproj.Transformer.from_crs("EPSG:4326", _UTM_CRS, always_xy=True)
        track_utm = [tuple(to_utm.transform(lon, lat)) for lon, lat in track_lonlat]

        with_crs = list(
            SpatialAlongTrack(track_lonlat, spacing=50.0, crs="EPSG:4326").anchors(
                field.domain, geom
            )
        )
        manual = list(
            SpatialAlongTrack(track_utm, spacing=50.0).anchors(field.domain, geom)
        )
        # Spacing (50 m) is applied in domain units *after* the transform,
        # so both paths place identical anchors.
        assert with_crs == manual
        assert with_crs

    def test_config_records_crs(self) -> None:
        cfg = SpatialExplicitCoords(coords=[(0.0, 0.0)], crs="EPSG:4326").get_config()
        assert cfg["crs"] == "EPSG:4326"
        assert cfg["n_coords"] == 1
        assert SpatialAlongTrack([(0.0, 0.0), (1.0, 1.0)]).get_config()["crs"] is None


class TestPolarDatelineGuard:
    def _sampler(self, guard: str) -> SpatialExplicitCoords:
        # Geographic coords near the pole, over a UTM (non-geographic) domain
        # so the reprojection path — and its guard — actually runs.
        return SpatialExplicitCoords(
            coords=[(0.0, 85.0)], crs="EPSG:4326", polar_guard=guard
        )

    def test_warns_beyond_80_degrees(self) -> None:
        field = _utm_field()
        geom = SpatialRectangular(size=(8, 8))
        with pytest.warns(RuntimeWarning, match="latitude beyond"):
            list(self._sampler("warn").anchors(field.domain, geom))

    def test_raise_mode_errors(self) -> None:
        field = _utm_field()
        geom = SpatialRectangular(size=(8, 8))
        with pytest.raises(ValueError, match="±80"):
            list(self._sampler("raise").anchors(field.domain, geom))

    def test_ignore_mode_is_silent(self) -> None:
        field = _utm_field()
        geom = SpatialRectangular(size=(8, 8))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            list(self._sampler("ignore").anchors(field.domain, geom))

    def test_antimeridian_track_warns(self) -> None:
        field = _utm_field()
        geom = SpatialRectangular(size=(8, 8))
        track = [(179.5, 0.5), (-179.5, 0.5)]  # a step across ±180°
        sampler = SpatialAlongTrack(track, crs="EPSG:4326")
        with pytest.warns(RuntimeWarning, match="antimeridian"):
            list(sampler.anchors(field.domain, geom))

    def test_invalid_guard_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid polar_guard"):
            SpatialExplicitCoords(coords=[(0.0, 0.0)], polar_guard="mirror")


class TestReprojectingRasterField:
    def test_domain_reports_destination_crs(self) -> None:
        reader = _utm_field().reader
        field = ReprojectingRasterField(reader, dst_crs="EPSG:3857")
        assert str(field.domain.crs) == "EPSG:3857"
        assert len(field.domain.shape) == 2

    def test_identity_crs_is_passthrough(self) -> None:
        reader = _utm_field().reader
        field = ReprojectingRasterField(reader, dst_crs=_UTM_CRS)
        # Same CRS + native resolution → the destination grid matches the
        # source grid, and a full read reproduces the source values.
        assert field.domain.shape == tuple(reader.shape[-2:])
        from rasterio.windows import Window

        chip = field.select(Window(0, 0, 100, 100))
        np.testing.assert_allclose(
            np.asarray(chip.values), np.asarray(reader.values), rtol=1e-4
        )

    def test_chips_are_on_destination_grid(self) -> None:
        reader = _utm_field().reader
        field = ReprojectingRasterField(reader, dst_crs="EPSG:3857")
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        chips = list(patcher.split(field))
        assert chips
        for chip in chips:
            assert chip.data.values.shape == (16, 16)
            assert str(chip.data.crs) == "EPSG:3857"

    def test_stitched_output_matches_destination_grid(self) -> None:
        reader = _utm_field().reader
        field = ReprojectingRasterField(reader, dst_crs="EPSG:3857")
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16)),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        merged = patcher.merge((p for p in patcher.split(field)), field.domain)
        assert np.asarray(merged).shape[-2:] == field.domain.shape

    def test_unknown_resampling_rejected(self) -> None:
        reader = _utm_field().reader
        with pytest.raises(ValueError, match="unknown resampling"):
            ReprojectingRasterField(reader, dst_crs="EPSG:3857", resampling="quantic")
