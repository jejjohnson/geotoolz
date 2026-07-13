"""Regression tests for bugs surfaced during the v0.0.1 extraction review.

Each test pins a specific code path that was either silently incorrect or
crashed at runtime before the review-comment fixes landed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely.geometry
from georeader.geotensor import GeoTensor

from geopatcher import (
    PointDomain,
    RasterField,
    SpatialBoxcar,
    SpatialExplicit,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialPolygonIntersection,
    SpatialSphericalCap,
    SpatioTemporalPatch,
    SpatioTemporalPatcher,
    TemporalFixedLookback,
    TemporalForecast,
    TemporalPatcher,
    TemporalRegularStride,
)


def _ones_field(h: int = 32, w: int = 32) -> RasterField:
    arr = np.ones((h, w), dtype=np.float32)
    # north-up affine: y decreases as the row index grows. `from_origin`
    # produces `Affine(1, 0, 0, 0, -1, h)`, which is what `from_bounds`
    # downstream expects.
    transform = rasterio.transform.from_origin(
        west=0.0, north=float(h), xsize=1.0, ysize=1.0
    )
    gt = GeoTensor(values=arr, transform=transform, crs="EPSG:32630")
    return RasterField(gt)


class TestPolygonIntersectionUnwrap:
    """`SpatialPolygonIntersection.neighborhood` returns a `_MaskedWindow`;
    the previous code path forwarded that wrapper directly to
    `Field.select`, which downstream readers (e.g. `RasterField.read_from_window`)
    can't consume. The fix is in the `_unwrap_for_select` helper inside
    `SpatialPatcher.split`.
    """

    def test_unwrap_for_select_strips_masked_window(self) -> None:
        from rasterio.windows import Window

        from geopatcher._src.spatial.geometry import _MaskedWindow
        from geopatcher._src.spatial.patcher import _unwrap_for_select

        win = Window(col_off=0, row_off=0, width=4, height=4)
        wrapped = _MaskedWindow(window=win, mask=np.ones((4, 4), dtype=bool))

        # Wrapped → bare window; other shapes (dict, list, plain Window)
        # pass through unchanged so the helper is safe to call at the
        # `field.select` boundary regardless of geometry type.
        assert _unwrap_for_select(wrapped) is win
        assert _unwrap_for_select(win) is win
        d = {"x": slice(0, 4)}
        assert _unwrap_for_select(d) is d

    def test_split_calls_field_select_with_unwrapped_window(self) -> None:
        # Drive SpatialPatcher.split with a stub Field that records the
        # `indices` it gets; verify the recorded value is a plain Window
        # (i.e. unwrapped) when geometry is SpatialPolygonIntersection.
        from rasterio.windows import Window

        from geopatcher._src.spatial.geometry import _MaskedWindow

        seen: list[Any] = []

        class _StubDomain:
            # `_is_raster_domain` checks for transform/shape/crs attrs.
            transform = rasterio.transform.from_origin(0.0, 20.0, 1.0, 1.0)
            shape = (20, 20)
            crs = "EPSG:32630"

        class _StubField:
            domain = _StubDomain()

            def select(self, indices: Any) -> str:
                seen.append(indices)
                return "data"

            def with_data(self, array: Any) -> Any:
                return array

        poly = shapely.geometry.box(4, 4, 12, 12)
        patcher = SpatialPatcher(
            geometry=SpatialPolygonIntersection(polygons=pd.Series([poly])),
            sampler=SpatialExplicit(anchors_=[0]),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )

        patches = list(patcher.split(_StubField()))

        assert len(patches) == 1
        assert len(seen) == 1
        # `field.select` saw a bare rasterio Window — not the wrapper.
        assert isinstance(seen[0], Window)
        assert not isinstance(seen[0], _MaskedWindow)
        # The wrapper is still on `Patch.indices` so aggregation can
        # recover the polygon mask via `_resolve_indices`.
        assert isinstance(patches[0].indices, _MaskedWindow)
        assert patches[0].weights is not None


class TestSpatioTemporalMergeDictAnchor:
    """`SpatioTemporalPatcher.merge` used `dict` anchors as dict keys, which
    is a TypeError on Python (`unhashable type: 'dict'`).
    """

    def _build_patcher(self) -> SpatioTemporalPatcher:
        # The patcher's split() is not exercised here; we hand-craft patches.
        # Construction is enough to validate the merge wiring.
        spatial = SpatialPatcher(
            geometry=SpatialPolygonIntersection(
                polygons=pd.Series([shapely.geometry.box(0, 0, 1, 1)])
            ),
            sampler=SpatialExplicit(anchors_=[0]),
            window=SpatialOverlapAdd(),
            aggregation=SpatialOverlapAdd(),
        )
        temporal = TemporalPatcher(
            geometry=TemporalFixedLookback(length=2),
            sampler=TemporalRegularStride(step=1),
            window=None,  # type: ignore[arg-type]
            aggregation=TemporalForecast(horizon=1),
        )
        return SpatioTemporalPatcher(spatial=spatial, temporal=temporal)

    def test_dict_anchors_merge_without_typeerror(self) -> None:
        patcher = self._build_patcher()
        # Two patches share the same dict-shaped spatial anchor; one has a
        # different anchor. The merge should group them by anchor and
        # apply TemporalForecast per group.
        anchor_a = {"lat": 0.0, "lon": 0.0}
        anchor_b = {"lat": 1.0, "lon": 0.0}
        patches = [
            SpatioTemporalPatch(
                data=np.ones((3, 4)),
                space=anchor_a,
                time=0,
                spatial_indices=None,
                temporal_indices=slice(0, 3),
                weights=None,
            ),
            SpatioTemporalPatch(
                data=np.ones((3, 4)) * 2,
                space=anchor_a,
                time=1,
                spatial_indices=None,
                temporal_indices=slice(1, 4),
                weights=None,
            ),
            SpatioTemporalPatch(
                data=np.ones((3, 4)) * 3,
                space=anchor_b,
                time=0,
                spatial_indices=None,
                temporal_indices=slice(0, 3),
                weights=None,
            ),
        ]

        out = patcher.merge(patches, field=None)

        # `merge` returns a list of (anchor, merged) pairs because dict
        # anchors aren't hashable as dict keys. The original dict anchor
        # objects round-trip verbatim.
        assert len(out) == 2
        anchors_out = [a for a, _ in out]
        assert anchor_a in anchors_out
        assert anchor_b in anchors_out

    def test_temporal_forecast_reads_anchor_via_rebox(self) -> None:
        """TemporalForecast reads `p.anchor` / `p.indices`, which don't exist
        on SpatioTemporalPatch (it has `time` / `temporal_indices`). The
        merge should rebox each group as TemporalPatch before passing it
        through.
        """
        patcher = self._build_patcher()
        patches = [
            SpatioTemporalPatch(
                data=np.arange(12).reshape(3, 4).astype(np.float64),
                space=(0.0, 0.0),
                time=7,
                spatial_indices=None,
                temporal_indices=slice(0, 3),
                weights=None,
            )
        ]

        out = patcher.merge(patches, field=None)

        # TemporalForecast returns {anchor: horizon_block}, and the anchor
        # is read from the rebox'd TemporalPatch.anchor (== p.time == 7).
        result = dict(out)[(0.0, 0.0)]
        assert 7 in result


class TestSpatialExplicitConfigDoesNotConsumeAnchors:
    """`SpatialExplicit.get_config()` materialised `anchors_` with
    ``list(...)``, which consumed one-shot iterators and left
    ``anchors()`` empty.
    """

    def test_get_config_then_anchors_still_yields(self) -> None:
        # Pass a generator — the previous Iterable typing accepted it.
        def gen():
            yield from [(0, 0), (1, 1), (2, 2)]

        sampler = SpatialExplicit(anchors_=gen())

        cfg = sampler.get_config()
        assert cfg == {"n_anchors": 3}

        # After get_config(), anchors() must still walk all three anchors —
        # __post_init__ materialises the iterable to a list so multiple
        # consumers can co-exist.
        out = list(sampler.anchors(domain=None, geometry=None))  # type: ignore[arg-type]
        assert out == [(0, 0), (1, 1), (2, 2)]


class TestSphericalCapPointDomainAnchorConvention:
    """`SpatialSphericalCap.neighborhood` on a `PointDomain` previously
    unpacked the anchor as `(lat, lon)`, but `PointDomain.coords` and the
    KNN/radius haversine paths use the `(x, y) = (lon, lat)` convention.
    A spherical cap centred on the natural `(lon, lat)` anchor therefore
    computed distances from swapped coordinates.
    """

    def test_anchor_uses_x_y_convention_consistent_with_knn(self) -> None:
        # Three points around (lon=10, lat=5) in (x, y) = (lon, lat) order.
        coords = np.array(
            [
                [10.0, 5.0],  # exactly at the anchor
                [10.5, 5.0],  # ~55 km east (close in longitude)
                [10.0, 45.0],  # ~4400 km north (far in latitude)
            ]
        )
        # Build PointDomain with a stub kdtree (this geometry doesn't use it).
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            pytest.skip("scipy not available")
        domain = PointDomain(coords=coords, kdtree=cKDTree(coords))

        # Anchor in the natural (x, y) = (lon, lat) convention.
        cap = SpatialSphericalCap(radius_km=200.0)
        idx = cap.neighborhood(domain, anchor=(10.0, 5.0))

        # The point at (10, 5) (exactly at anchor) and the near one at
        # (10.5, 5) must be inside the 200 km cap; the far point at
        # (10, 45) must be outside. Before the fix the anchor was
        # interpreted as (lat=10, lon=5), so the cap was centred at
        # (lat=10, lon=5) and gave a different (wrong) result.
        assert 0 in idx
        assert 1 in idx
        assert 2 not in idx
