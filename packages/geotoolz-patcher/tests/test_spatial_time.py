"""Tests for `SpatioTemporalPatcher` (product + coupled coupling)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher import (
    RasterField,
    SpatialPatcher,
    SpatioTemporalPatch,
    SpatioTemporalPatcher,
    TemporalPatcher,
    spatial,
    temporal,
)
from geopatcher.temporal.stencils import TimeStencil


@pytest.fixture
def time_field() -> RasterField:
    # (time=8, H=16, W=16); the spatial select returns the full time series.
    arr = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


@pytest.fixture
def sp() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=8),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )


@pytest.fixture
def tp() -> TemporalPatcher:
    return TemporalPatcher(
        geometry=temporal.geometry.FixedLookback(length=4),
        sampler=temporal.sampler.RegularStride(step=4),
        window=temporal.window.CausalBoxcar(),
        aggregation=temporal.aggregation.Mean(),
    )


class TestProductCoupling:
    def test_yields_spatial_x_temporal_patches(
        self, time_field: RasterField, sp: SpatialPatcher, tp: TemporalPatcher
    ) -> None:
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp, coupling="product")
        patches = list(stp.split(time_field))
        # Spatial: 4 (2x2); Temporal: anchors 0, 4 — anchor 0's 4-step
        # lookback overflows and is dropped, anchor 4 reads [1, 5); Product: 4
        assert len(patches) == 4
        assert all(isinstance(p, SpatioTemporalPatch) for p in patches)
        for p in patches:
            assert p.time == 4
            assert p.temporal_indices == slice(1, 5)
            assert p.data.shape == (4, 8, 8)


class TestCoupledCoupling:
    def test_requires_paired_anchors(self) -> None:
        # Couple coupling expects spatial.sampler.anchors_ to be (space, time)
        sp_explicit = SpatialPatcher(
            geometry=spatial.geometry.Rectangular(size=(8, 8)),
            sampler=spatial.sampler.Explicit(anchors_=[((0, 0), 1), ((0, 8), 4)]),
            window=spatial.window.Boxcar(),
            aggregation=spatial.aggregation.OverlapAdd(),
        )
        tp = TemporalPatcher(
            geometry=temporal.geometry.FixedLookback(length=2),
            sampler=temporal.sampler.RegularStride(step=1),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Mean(),
        )
        stp = SpatioTemporalPatcher(
            spatial=sp_explicit, temporal=tp, coupling="coupled"
        )

        arr = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
        field = RasterField(
            GeoTensor(
                values=arr,
                transform=rasterio.Affine.identity(),
                crs="EPSG:32630",
            )
        )
        patches = list(stp.split(field))
        assert len(patches) == 2
        anchors = [p.space for p in patches]
        assert (0, 0) in anchors
        assert (0, 8) in anchors

    def test_coupled_requires_anchors_(
        self, time_field: RasterField, sp: SpatialPatcher, tp: TemporalPatcher
    ) -> None:
        # sp's sampler is RegularStride, not Explicit — coupled must raise.
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp, coupling="coupled")
        with pytest.raises(TypeError, match="anchors_"):
            list(stp.split(time_field))


class TestUnknownCoupling:
    def test_raises(
        self, time_field: RasterField, sp: SpatialPatcher, tp: TemporalPatcher
    ) -> None:
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp, coupling="weird")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown coupling"):
            list(stp.split(time_field))


class TestCoordAwareTemporal:
    """coord= threading through SpatioTemporalPatcher (gh #58)."""

    @pytest.fixture
    def coord(self) -> np.ndarray:
        # 8 three-hourly steps, matching time_field's time axis.
        return np.arange(
            "2020-01-01T00", "2020-01-02T00", 3, dtype="datetime64[h]"
        ).astype("datetime64[ns]")

    @pytest.fixture
    def tp_coord(self) -> TemporalPatcher:
        # -3h..+3h at 3-hourly cadence: 3-point windows, origins 1..6.
        stencil = TimeStencil("-3h", "3h", "3h", closed="both")
        return TemporalPatcher(
            geometry=temporal.geometry.StencilGeometry(
                stencil=stencil, source_step=np.timedelta64(3, "h")
            ),
            sampler=temporal.sampler.StencilSampler(stencil=stencil),
            window=temporal.window.CausalBoxcar(),
            aggregation=temporal.aggregation.Forecast(horizon=1),
        )

    def test_missing_coord_raises(
        self, time_field: RasterField, sp: SpatialPatcher, tp_coord: TemporalPatcher
    ) -> None:
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp_coord, coupling="product")
        with pytest.raises(ValueError, match="requires coord="):
            list(stp.split(time_field))

    def test_wrong_length_coord_raises(
        self,
        time_field: RasterField,
        sp: SpatialPatcher,
        tp_coord: TemporalPatcher,
        coord: np.ndarray,
    ) -> None:
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp_coord, coupling="product")
        with pytest.raises(ValueError, match="coord length"):
            list(stp.split(time_field, coord=coord[:-1]))

    def test_product_end_to_end(
        self,
        time_field: RasterField,
        sp: SpatialPatcher,
        tp_coord: TemporalPatcher,
        coord: np.ndarray,
    ) -> None:
        from geopatcher._src.temporal.stencils import valid_origin_points

        stencil = TimeStencil("-3h", "3h", "3h", closed="both")
        origins = valid_origin_points(coord, stencil)
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp_coord, coupling="product")
        patches = list(stp.split(time_field, coord=coord))
        # 4 spatial anchors x one 3-point window per valid origin.
        assert len(patches) == 4 * len(origins)
        for p in patches:
            assert p.data.shape == (3, 8, 8)
            s = p.temporal_indices
            assert s.stop - s.start == 3
            # Window is centred on the anchor (lookback 1, horizon 1).
            assert s.start == p.time - 1

    def test_coupled_with_stencil_geometry(
        self, time_field: RasterField, tp_coord: TemporalPatcher, coord: np.ndarray
    ) -> None:
        sp = SpatialPatcher(
            geometry=spatial.geometry.Rectangular(size=(8, 8)),
            sampler=spatial.sampler.Explicit(anchors_=[((0, 0), 3), ((8, 8), 5)]),
            window=spatial.window.Boxcar(),
            aggregation=spatial.aggregation.OverlapAdd(),
        )
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp_coord, coupling="coupled")
        patches = list(stp.split(time_field, coord=coord))
        assert len(patches) == 2
        assert [p.time for p in patches] == [3, 5]
        for p in patches:
            assert p.data.shape == (3, 8, 8)
            assert p.temporal_indices == slice(p.time - 1, p.time + 2)

    def test_asplit_matches_split(
        self,
        time_field: RasterField,
        sp: SpatialPatcher,
        tp_coord: TemporalPatcher,
        coord: np.ndarray,
    ) -> None:
        import asyncio

        class _AsyncField:
            def __init__(self, inner: RasterField) -> None:
                self.inner = inner

            @property
            def domain(self):
                return self.inner.domain

            async def aselect(self, window):
                await asyncio.sleep(0)
                return self.inner.select(window)

        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp_coord, coupling="product")
        sync_patches = list(stp.split(time_field, coord=coord))

        async def collect() -> list[SpatioTemporalPatch]:
            return [p async for p in stp.asplit(_AsyncField(time_field), coord=coord)]

        async_patches = asyncio.run(collect())
        assert len(async_patches) == len(sync_patches)
        for a, b in zip(async_patches, sync_patches, strict=True):
            assert a.space == b.space
            assert a.time == b.time
            np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))

    def test_integer_pipeline_ignores_coord(
        self,
        time_field: RasterField,
        sp: SpatialPatcher,
        tp: TemporalPatcher,
        coord: np.ndarray,
    ) -> None:
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp, coupling="product")
        with_coord = list(stp.split(time_field, coord=coord))
        without = list(stp.split(time_field))
        assert len(with_coord) == len(without)
        for a, b in zip(with_coord, without, strict=True):
            np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))

    def test_coupled_short_coord_raises_value_error(
        self, time_field: RasterField, tp_coord: TemporalPatcher, coord: np.ndarray
    ) -> None:
        # A paired time anchor beyond the coord's length must surface as
        # the documented ValueError, not an IndexError from the hook
        # payload lookup.
        sp = SpatialPatcher(
            geometry=spatial.geometry.Rectangular(size=(8, 8)),
            sampler=spatial.sampler.Explicit(anchors_=[((0, 0), 5)]),
            window=spatial.window.Boxcar(),
            aggregation=spatial.aggregation.OverlapAdd(),
        )
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp_coord, coupling="coupled")
        with pytest.raises(ValueError, match="coord length"):
            list(stp.split(time_field, coord=coord[:4]))


def test_coupled_polygon_and_reflect(
    time_field: RasterField, tp: TemporalPatcher
) -> None:
    """The coupled split reads each chip through SpatialPatcher's pipeline (#188).

    It used to call ``field.select`` itself: a polygon geometry's masked
    window crashed the raster read, ``boundary="reflect"`` returned the
    nodata fill instead of the mirror, and every patch got the unmasked,
    uncropped base weights.
    """
    import geopandas as gpd
    import shapely

    # North-up grid: polygon footprints need a negative y pixel size.
    north_up = RasterField(
        GeoTensor(
            values=np.asarray(time_field.reader.values),
            transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 16.0),
            crs="EPSG:32630",
        )
    )
    tri = shapely.Polygon([(1, 9), (7, 9), (1, 15)])
    pairs = [(0, 4)]
    polygon = SpatialPatcher(
        geometry=spatial.geometry.PolygonIntersection(polygons=gpd.GeoSeries([tri])),
        sampler=spatial.sampler.Explicit(anchors_=pairs),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    reflect = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8), boundary="reflect"),
        sampler=spatial.sampler.Explicit(anchors_=[((-4, -4), 4)]),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    for sp, space in [(polygon, 0), (reflect, (-4, -4))]:
        stp = SpatioTemporalPatcher(spatial=sp, temporal=tp, coupling="coupled")
        (patch,) = list(stp.split(north_up))
        chip = sp.patch_at(north_up, space)
        np.testing.assert_array_equal(
            np.asarray(patch.data), np.asarray(chip.data)[1:5]
        )
        np.testing.assert_array_equal(patch.weights, chip.weights)
