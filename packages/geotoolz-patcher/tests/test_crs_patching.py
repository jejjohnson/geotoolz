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

    def test_unordered_catalogue_across_antimeridian_is_silent(self) -> None:
        # #187: an event catalogue holding 179.5 and -179.5 is not a track
        # crossing the dateline; consecutive differences mean nothing.
        field = _utm_field()
        geom = SpatialRectangular(size=(8, 8))
        sampler = SpatialExplicitCoords(
            coords=[(179.5, 0.5), (-179.5, 0.5)], crs="EPSG:4326"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            list(sampler.anchors(field.domain, geom))

    def test_polar_destination_crs_is_silent(self) -> None:
        # #187: the polar check ignored the destination CRS; a polar
        # stereographic domain (EPSG:3413, valid to 90N) is exactly where
        # 85N coordinates belong.
        arr = np.zeros((10, 10), dtype=np.float32)
        polar = RasterField(
            GeoTensor(
                values=arr,
                transform=rasterio.Affine(1000, 0, -5000, 0, -1000, 5000),
                crs="EPSG:3413",
            )
        )
        geom = SpatialRectangular(size=(2, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            list(self._sampler("warn").anchors(polar.domain, geom))

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


def _utm_3d_geotensor(n: int = 100, bands: int = 2) -> GeoTensor:
    arr = np.stack(
        [
            np.arange(n * n, dtype=np.float32).reshape(n, n) + 1e5 * b
            for b in range(bands)
        ]
    )
    return GeoTensor(values=arr, transform=_UTM_TRANSFORM, crs=_UTM_CRS)


def _stride_patcher(size: int = 16) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(size, size)),
        sampler=SpatialRegularStride(step=size),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


def _write_tif(path, gt: GeoTensor) -> str:
    values = np.asarray(gt.values)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[-2],
        width=values.shape[-1],
        count=values.shape[0],
        dtype=values.dtype,
        crs=gt.crs,
        transform=gt.transform,
    ) as dst:
        dst.write(values)
    return str(path)


class TestReprojectingRasterField3D:
    def test_domain_shape_keeps_leading_dims(self) -> None:
        field = ReprojectingRasterField(_utm_3d_geotensor(), dst_crs="EPSG:3857")
        assert len(field.domain.shape) == 3
        assert field.domain.shape[0] == 2

    @pytest.mark.parametrize("backend", ["geotensor", "rasterio"])
    def test_reprojecting_field_3d_merge(self, backend: str, tmp_path) -> None:
        from georeader.rasterio_reader import RasterioReader
        from georeader.read import read_reproject

        gt = _utm_3d_geotensor()
        reader = (
            gt
            if backend == "geotensor"
            else RasterioReader(_write_tif(tmp_path / "src.tif", gt))
        )
        field = ReprojectingRasterField(reader, dst_crs="EPSG:3857")
        patcher = _stride_patcher()
        chips = list(patcher.split(field))
        assert chips
        assert all(c.data.values.shape == (2, 16, 16) for c in chips)
        merged = np.asarray(patcher.merge(iter(chips), field.domain))
        assert merged.shape == field.domain.shape

        # Reference: one full warp of the whole source onto the same grid.
        # Chips are warped independently, so allow the documented
        # approximate-transformer drift (well under a ramp step of 1).
        full = read_reproject(
            gt,
            dst_crs="EPSG:3857",
            dst_transform=field.domain.transform,
            window_out=rasterio.windows.Window(
                0, 0, field.domain.shape[-1], field.domain.shape[-2]
            ),
            resampling=rasterio.warp.Resampling.bilinear,
        )
        # Compare only the region tiled by full chips (stride "drop").
        hh = (field.domain.shape[-2] // 16) * 16
        ww = (field.domain.shape[-1] // 16) * 16
        np.testing.assert_allclose(
            merged[:, :hh, :ww], np.asarray(full.values)[:, :hh, :ww], atol=1.0
        )

    def test_per_chip_warp_reads_only_chip_footprint(self, monkeypatch) -> None:
        # Work guard (not wall-clock): every warp call must see a source
        # crop around the chip, never the whole 200x200 source.
        import rasterio.warp

        seen: list[tuple[int, ...]] = []
        real = rasterio.warp.reproject

        def spy(source, *args, **kwargs):
            seen.append(tuple(np.shape(source)))
            return real(source, *args, **kwargs)

        monkeypatch.setattr(rasterio.warp, "reproject", spy)
        field = ReprojectingRasterField(_utm_3d_geotensor(n=200), dst_crs="EPSG:3857")
        chips = list(_stride_patcher().split(field))
        assert len(chips) > 50
        assert seen
        # 16x16 chip ≈ 16x16 source pixels (+ curvature + 3 px margin per side).
        assert max(s[-2] for s in seen) <= 32
        assert max(s[-1] for s in seen) <= 32

    @pytest.mark.parametrize("resolution", [None, 40.0])
    def test_cropped_chip_matches_uncropped_warp(self, resolution) -> None:
        # The pre-crop is an optimisation only: each chip must equal a warp
        # of the *whole* source onto the same chip grid, including when the
        # destination is coarser than the source (widened kernel margin).
        from georeader.read import read_reproject
        from rasterio.windows import Window, transform as window_transform

        gt = _utm_3d_geotensor()
        field = ReprojectingRasterField(gt, dst_crs="EPSG:3857", resolution=resolution)
        h, w = field.domain.shape[-2:]
        for window in (Window(0, 0, 8, 8), Window(w // 3, h // 3, 8, 8)):
            chip = field.select(window)
            ref = read_reproject(
                gt,
                dst_crs="EPSG:3857",
                dst_transform=window_transform(window, field.domain.transform),
                window_out=Window(0, 0, 8, 8),
                resampling=rasterio.warp.Resampling.bilinear,
            )
            np.testing.assert_array_equal(
                np.asarray(chip.values), np.asarray(ref.values)
            )

    def test_chip_outside_source_is_nodata(self) -> None:
        from rasterio.windows import Window

        field = ReprojectingRasterField(_utm_3d_geotensor(), dst_crs="EPSG:3857")
        h, w = field.domain.shape[-2:]
        chip = field.select(Window(w + 50, h + 50, 8, 8))
        assert chip.values.shape == (2, 8, 8)
        assert np.all(np.asarray(chip.values) == chip.fill_value_default)
