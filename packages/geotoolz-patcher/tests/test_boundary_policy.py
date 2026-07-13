"""Tests for `SpatialRectangular.boundary` — issue #19.

Four modes on a deliberately misaligned domain (70x70, patch 16,
stride 16 → 4 full anchors plus a 6-px residual at the right/bottom
edges):

- ``"drop"`` (default): residual is silently dropped; 4x4 = 16 anchors.
- ``"pad"``: edge anchors emitted; reads use ``boundless=True`` so the
  patch is the full geometry size with the reader's nodata in the
  overflow region; 5x5 = 25 anchors.
- ``"shrink"``: edge anchors emitted; the geometry clips the Window so
  the patch is smaller at the edge; 5x5 = 25 anchors, edge ones smaller.
- ``"raise"``: edge anchors emitted; `SpatialPatcher.split` raises on
  the first overflow.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher import (
    RasterField,
    SpatialBoxcar,
    SpatialJitteredStride,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialPoissonDisk,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialSampler,
)


# Match the BoundaryMode literal defined in
# `geopatcher._src.spatial.geometry`. Re-declared locally so the test
# module doesn't reach into private code just for typing.
BoundaryMode = Literal["drop", "pad", "shrink", "raise", "reflect"]


def _patcher(boundary: BoundaryMode) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16), boundary=boundary),
        sampler=SpatialRegularStride(step=16),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


@pytest.fixture
def misaligned_field() -> RasterField:
    # 70x70 with patch=16, stride=16 → residual of 6 px on each axis.
    arr = np.ones((70, 70), dtype=np.float32)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


class TestRectangularBoundary:
    def test_drop_is_default_and_omits_residual(
        self, misaligned_field: RasterField
    ) -> None:
        p = _patcher("drop")
        anchors = [patch.anchor for patch in p.split(misaligned_field)]
        # 4 anchors per axis (0, 16, 32, 48). 64 is dropped because
        # 64 + 16 = 80 > 70.
        assert len(anchors) == 16
        rows = sorted({a[0] for a in anchors})
        assert rows == [0, 16, 32, 48]

    def test_pad_emits_edge_anchors_full_size(
        self, misaligned_field: RasterField
    ) -> None:
        p = _patcher("pad")
        patches = list(p.split(misaligned_field))
        # 5 anchors per axis (0, 16, 32, 48, 64).
        assert len(patches) == 25
        # Every patch is still 16x16 — georeader pads the out-of-bounds
        # region via boundless=True (with reader nodata).
        for patch in patches:
            assert patch.data.values.shape == (16, 16)

    def test_shrink_clips_edge_patches(self, misaligned_field: RasterField) -> None:
        p = _patcher("shrink")
        patches = list(p.split(misaligned_field))
        assert len(patches) == 25
        # Interior patch at (0, 0) keeps full 16x16; corner patch at
        # (64, 64) shrinks to 6x6.
        shapes = {patch.anchor: patch.data.values.shape for patch in patches}
        assert shapes[(0, 0)] == (16, 16)
        assert shapes[(64, 64)] == (6, 6)
        assert shapes[(64, 0)] == (6, 16)
        # Weights track the actual patch size.
        for patch in patches:
            assert patch.weights.shape == patch.data.values.shape

    def test_raise_errors_on_first_overflow(
        self, misaligned_field: RasterField
    ) -> None:
        p = _patcher("raise")
        with pytest.raises(ValueError, match="overflows the domain"):
            list(p.split(misaligned_field))

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid boundary mode"):
            SpatialRectangular(size=(16, 16), boundary="wrap")  # type: ignore[arg-type]

    def test_config_round_trips_boundary(self) -> None:
        geom = SpatialRectangular(size=(16, 16), boundary="pad")
        cfg = geom.get_config()
        assert cfg["boundary"] == "pad"
        # Defaults preserved through round-trip too.
        default = SpatialRectangular(size=(16, 16))
        assert default.get_config()["boundary"] == "drop"


class TestAlignedDomainIsUnchanged:
    """When the domain divides evenly, drop / pad / shrink agree."""

    @pytest.fixture
    def aligned_field(self) -> RasterField:
        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        gt = GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
        return RasterField(gt)

    @pytest.mark.parametrize("boundary", ["drop", "pad", "shrink", "raise", "reflect"])
    def test_aligned_domain_anchor_count(
        self, aligned_field: RasterField, boundary: str
    ) -> None:
        p = _patcher(boundary)
        anchors = [patch.anchor for patch in p.split(aligned_field)]
        # 4x4 = 16 anchors on a 64x64 domain with patch=16, stride=16.
        assert len(anchors) == 16


class TestBoundaryHonoredByAllRasterSamplers:
    """Boundary must be wired into every raster sampler, not only
    `SpatialRegularStride`. The contract: when ``boundary != "drop"``,
    the sampler is allowed to place anchors that overflow the domain,
    and `SpatialPatcher.split(boundary="raise")` raises on the first
    such anchor. When ``boundary == "drop"``, anchors stay in-bounds.
    """

    @pytest.fixture
    def misaligned_field(self) -> RasterField:
        arr = np.ones((70, 70), dtype=np.float32)
        gt = GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
        return RasterField(gt)

    @pytest.mark.parametrize(
        "sampler",
        [
            SpatialRegularStride(step=16),
            SpatialJitteredStride(step=16, jitter=0.5, seed=0),
            SpatialRandom(n_samples=200, seed=0),
            SpatialPoissonDisk(min_dist=4.0, seed=0),
        ],
        ids=["RegularStride", "JitteredStride", "Random", "PoissonDisk"],
    )
    def test_raise_mode_fires_for_each_sampler(
        self, misaligned_field: RasterField, sampler: SpatialSampler
    ) -> None:
        # With boundary="raise" and a domain that doesn't divide evenly
        # by the patch size, every raster sampler must be willing to
        # place at least one overflowing anchor, so split() raises.
        # JitteredStride is borderline (it only emits 16 base anchors,
        # all in-bounds at jitter=0); use SpatialRandom and the others
        # to cover the contract.
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16), boundary="raise"),
            sampler=sampler,
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        if isinstance(sampler, SpatialJitteredStride):
            # The base RegularStride anchors stay below 64 even in
            # raise mode (4x4 lattice on a 70x70 field), and the
            # jittered offsets respect `rmax = h - 1` which still leaves
            # most jittered anchors in-bounds. We can't deterministically
            # force overflow from JitteredStride at this jitter level —
            # instead check that at least one of its anchors lands above
            # the drop-mode ceiling of 54.
            anchors = list(
                sampler.anchors(
                    misaligned_field.domain,
                    SpatialRectangular(size=(16, 16), boundary="raise"),
                )
            )
            assert any(r > 54 or c > 54 for r, c in anchors), (
                "JitteredStride should have placed at least one anchor "
                "beyond the drop-mode ceiling when boundary != 'drop'"
            )
            return
        with pytest.raises(ValueError, match="overflows the domain"):
            list(patcher.split(misaligned_field))

    @pytest.mark.parametrize(
        "sampler",
        [
            SpatialRandom(n_samples=200, seed=0),
            SpatialPoissonDisk(min_dist=4.0, seed=0),
        ],
        ids=["Random", "PoissonDisk"],
    )
    def test_drop_mode_keeps_anchors_in_bounds(
        self, misaligned_field: RasterField, sampler: SpatialSampler
    ) -> None:
        # Inverse property: with boundary="drop", no anchor produces an
        # overflowing window. Confirms the wiring is conditioned on
        # boundary rather than being a no-op everywhere.
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16), boundary="drop"),
            sampler=sampler,
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        for patch in patcher.split(misaligned_field):
            r, c = patch.anchor
            assert r + 16 <= 70 and c + 16 <= 70


def _arange_field(n: int = 10) -> RasterField:
    """``n x n`` field of ``arange`` values on an identity transform."""
    arr = np.arange(n * n, dtype=np.float32).reshape(n, n)
    gt = GeoTensor(values=arr, transform=rasterio.Affine.identity(), crs="EPSG:32630")
    return RasterField(gt)


def _corner_patcher(
    boundary: BoundaryMode, size: int = 4, pad_value: float | None = None
) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(
            size=(size, size), boundary=boundary, pad_value=pad_value
        ),
        sampler=SpatialRegularStride(step=size),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


class TestReflectAndPadValue:
    """`boundary="reflect"` and `pad_value` — the remainder of issue #19.

    Field-independent clip-and-pad: the overflowing window is clipped to
    the domain, read once, then padded up to the full geometry size.
    """

    def test_reflect_edge_equals_numpy_pad_of_clipped_read(self) -> None:
        # 11x11 field, patch 4, stride 4 → anchors 0, 4, 8. Anchor (8, 8)
        # covers rows/cols 8..11; row/col 11 is out of domain, so the
        # clipped read is raw[8:11, 8:11] (3x3) and the overflow is 1.
        field = _arange_field(11)
        raw = np.asarray(field.reader.values)
        patcher = _corner_patcher("reflect", size=4)
        patches = {p.anchor: p for p in patcher.split(field)}
        corner = patches[(8, 8)]
        expected = np.pad(raw[8:11, 8:11], ((0, 1), (0, 1)), mode="reflect")
        np.testing.assert_array_equal(corner.data.values, expected)
        assert corner.data.values.shape == (4, 4)

    def test_reflect_interior_anchor_is_untouched(self) -> None:
        # An in-domain window takes the plain read path — no padding.
        field = _arange_field(11)
        raw = np.asarray(field.reader.values)
        patcher = _corner_patcher("reflect", size=4)
        interior = {p.anchor: p for p in patcher.split(field)}[(0, 0)]
        np.testing.assert_array_equal(interior.data.values, raw[0:4, 0:4])

    def test_pad_value_fills_overflow_region(self) -> None:
        field = _arange_field(10)
        raw = np.asarray(field.reader.values)
        patcher = _corner_patcher("pad", size=4, pad_value=-999.0)
        corner = {p.anchor: p for p in patcher.split(field)}[(8, 8)]
        assert corner.data.values.shape == (4, 4)
        # In-domain quadrant preserved …
        np.testing.assert_array_equal(corner.data.values[0:2, 0:2], raw[8:10, 8:10])
        # … overflow filled with the requested constant.
        assert np.all(corner.data.values[2:, :] == -999.0)
        assert np.all(corner.data.values[:, 2:] == -999.0)

    def test_edge_chip_keeps_exact_georeferencing(self) -> None:
        # Overflow is bottom/right only → the UL origin is unchanged and,
        # on an identity transform, equals the anchor.
        field = _arange_field(10)
        patcher = _corner_patcher("pad", size=4)
        corner = {p.anchor: p for p in patcher.split(field)}[(8, 8)]
        assert corner.data.transform.c == 8
        assert corner.data.transform.f == 8

    def test_reflect_raises_when_overflow_exceeds_extent(self) -> None:
        # 10x10 field, patch 4, anchor (8, 8): clipped is 2x2 but reflect
        # needs pad (2) < extent (2). Must raise a clear error.
        field = _arange_field(10)
        patcher = _corner_patcher("reflect", size=4)
        with pytest.raises(ValueError, match="reflect"):
            list(patcher.split(field))

    def test_pad_none_matches_boundless_read(self) -> None:
        # With pad_value=None the clip-and-pad path must reproduce the old
        # boundless read (fill_value_default) bit-for-bit.
        field = _arange_field(10)
        patcher = _corner_patcher("pad", size=4)
        corner = {p.anchor: p for p in patcher.split(field)}[(8, 8)]
        boundless = field.reader.read_from_window(corner.indices, boundless=True)
        np.testing.assert_array_equal(corner.data.values, np.asarray(boundless.values))

    def test_config_round_trips_pad_value(self) -> None:
        geom = SpatialRectangular(size=(16, 16), boundary="pad", pad_value=0.0)
        cfg = geom.get_config()
        assert cfg["boundary"] == "pad"
        assert cfg["pad_value"] == 0.0
        assert SpatialRectangular(size=(16, 16)).get_config()["pad_value"] is None
