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

#185: every mode survives split → merge (the merge crops chip data and
weights to the in-domain part of the window), non-drop samplers stop at
the first anchor whose patch reaches the edge, ``shrink`` clips negative
anchors, ``drop`` places nothing for a patch larger than the domain,
``check_full_scan`` only applies under ``drop`` and ``pad_value`` must fit
the field's dtype.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import (
    Window,
    bounds as window_bounds,
    transform as window_transform,
)

from geopatcher import (
    IncompleteScanConfiguration,
    Patch,
    RasterField,
    SpatialBoxcar,
    SpatialExplicit,
    SpatialExplicitCoords,
    SpatialHann,
    SpatialHardVote,
    SpatialInvVarWeightedMean,
    SpatialJitteredStride,
    SpatialMax,
    SpatialMean,
    SpatialMedian,
    SpatialMin,
    SpatialMode,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialPoissonDisk,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialSampler,
    SpatialSoftVote,
    SpatialSum,
    SpatialVariance,
    SpatialWeightedSum,
)


# Match the BoundaryMode literal defined in
# `geopatcher._src.spatial.geometry`. Re-declared locally so the test
# module doesn't reach into private code just for typing.
BoundaryMode = Literal["drop", "pad", "shrink", "raise", "reflect"]


def _patcher(boundary: BoundaryMode, step: int = 16) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16), boundary=boundary),
        sampler=SpatialRegularStride(step=step),
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
    """When the domain divides evenly, every mode places the same anchors.

    Parametrised over ``step < size`` too (#185): the non-drop stop rule
    used to add a trailing anchor (56 for 64/16/8) whose patch overflows
    although 48 already reaches the edge.
    """

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
    @pytest.mark.parametrize(
        ("step", "starts"), [(16, [0, 16, 32, 48]), (8, list(range(0, 49, 8)))]
    )
    def test_aligned_domain_anchor_count(
        self,
        aligned_field: RasterField,
        boundary: BoundaryMode,
        step: int,
        starts: list[int],
    ) -> None:
        p = _patcher(boundary, step=step)
        anchors = [patch.anchor for patch in p.split(aligned_field)]
        # 64x64 domain, patch 16: the last anchor is 48 (48 + 16 = 64)
        # whatever the step — no overflowing trailing anchor, so "raise"
        # is happy and "pad" emits no half-padding row.
        assert sorted({a[0] for a in anchors}) == starts
        assert len(anchors) == len(starts) ** 2
        merged = p.merge(p.split(aligned_field), aligned_field.domain)
        np.testing.assert_array_equal(merged, aligned_field.reader.values)


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
    def test_raise_mode_fires_for_each_sampler(self, sampler: SpatialSampler) -> None:
        # A patch larger than the domain must overflow under every
        # non-drop raster sampler (anchored at the origin), so
        # boundary="raise" fires.
        small = RasterField(
            GeoTensor(
                values=np.ones((10, 10), dtype=np.float32),
                transform=rasterio.Affine.identity(),
                crs="EPSG:32630",
            )
        )
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16), boundary="raise"),
            sampler=sampler,
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        with pytest.raises(ValueError, match="overflows the domain"):
            list(patcher.split(small))

    @pytest.mark.parametrize(
        "sampler",
        [
            SpatialJitteredStride(step=16, jitter=0.5, seed=0),
            SpatialRandom(n_samples=200, seed=0),
            SpatialPoissonDisk(min_dist=4.0, seed=0),
        ],
        ids=["JitteredStride", "Random", "PoissonDisk"],
    )
    @pytest.mark.parametrize("boundary", ["pad", "shrink", "raise", "reflect"])
    def test_non_drop_random_anchors_stay_within_last_fitting_anchor(
        self,
        misaligned_field: RasterField,
        sampler: SpatialSampler,
        boundary: BoundaryMode,
    ) -> None:
        # #185: non-drop random/jittered samplers used to draw anchors up
        # to h - 1, emitting chips that are mostly padding (or a 1x1
        # shrink sliver). Anchors are now drawn from [0, h - size].
        geom = SpatialRectangular(size=(16, 16), boundary=boundary)
        anchors = list(sampler.anchors(misaligned_field.domain, geom))
        assert anchors
        assert all(0 <= r <= 54 and 0 <= c <= 54 for r, c in anchors)
        if isinstance(sampler, SpatialJitteredStride):
            # The edge lattice anchor (64) jitters to 56..72 and clamps
            # to 54, so the trailing strip is still covered.
            assert max(r for r, _ in anchors) == 54
            assert max(c for _, c in anchors) == 54

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

    @pytest.mark.parametrize(("n", "size"), [(10, 4), (70, 16), (5, 16)])
    def test_reflect_overflow_beyond_in_domain_extent(self, n: int, size: int) -> None:
        # #185: the overflow may exceed the in-domain part of the window
        # (10x10 / 4 → 2 in, 2 out; the documented 70x70 / 16 → 6 in,
        # 10 out; a 5x5 domain under a 16-px patch). The chip equals the
        # reflect-extended domain, i.e. np.pad(raw, mode="reflect").
        field = _arange_field(n)
        raw = np.asarray(field.reader.values)
        patcher = _corner_patcher("reflect", size=size)
        reference = np.pad(raw, ((0, size), (0, size)), mode="reflect")
        patches = list(patcher.split(field))
        for patch in patches:
            r, c = patch.anchor
            np.testing.assert_array_equal(
                patch.data.values, reference[r : r + size, c : c + size]
            )
            assert (patch.data.transform.c, patch.data.transform.f) == (c, r)
        merged = patcher.merge(patches, field.domain)
        np.testing.assert_array_equal(merged, raw)

    def test_reflect_needs_two_cells(self) -> None:
        field = _arange_field(1)
        patcher = _corner_patcher("reflect", size=4)
        with pytest.raises(ValueError, match="cannot mirror"):
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


# ---------------------------------------------------------------------------
# #185 — split → merge matrix and the remaining boundary contracts
# ---------------------------------------------------------------------------

# A north-up, non-unit, offset grid so bounds checks catch any transform
# that silently assumes the identity. 37 x 45 is not a multiple of 16 or 8.
_T = rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_600_000.0)
_H, _W, _SIZE = 37, 45, 16
_MATRIX_MODES: tuple[BoundaryMode, ...] = ("drop", "pad", "reflect", "shrink")


def _ramp(h: int = _H, w: int = _W) -> np.ndarray:
    return np.arange(h * w, dtype=np.float32).reshape(h, w)


def _raster_case() -> RasterField:
    return RasterField(GeoTensor(values=_ramp(), transform=_T, crs="EPSG:32630"))


def _rioxarray_case() -> Any:
    xr = pytest.importorskip("xarray")
    pytest.importorskip("rioxarray")
    from geopatcher import RioXarrayField

    da = xr.DataArray(
        _ramp(),
        dims=("y", "x"),
        coords={
            "y": _T.f + (np.arange(_H) + 0.5) * _T.e,
            "x": _T.c + (np.arange(_W) + 0.5) * _T.a,
        },
    )
    return RioXarrayField(da.rio.write_crs("EPSG:32630").rio.write_transform(_T))


_MATRIX_FIELDS = {"RasterField": _raster_case, "RioXarrayField": _rioxarray_case}


def _chip_georef(data: Any) -> tuple[rasterio.Affine, tuple[float, ...]]:
    if isinstance(data, GeoTensor):
        return data.transform, tuple(data.bounds)
    return data.rio.transform(), tuple(data.rio.bounds())


def _requested_window(boundary: BoundaryMode, anchor: tuple[int, int]) -> Window:
    """The window a chip must cover: full under pad/reflect, clipped under shrink."""
    r, c = anchor
    if boundary == "shrink":
        return Window(c, r, min(c + _SIZE, _W) - c, min(r + _SIZE, _H) - r)
    return Window(c, r, _SIZE, _SIZE)


@pytest.mark.parametrize("field_name", list(_MATRIX_FIELDS))
@pytest.mark.parametrize("step", [_SIZE, _SIZE // 2], ids=["step==size", "step<size"])
@pytest.mark.parametrize("boundary", _MATRIX_MODES)
def test_split_merge_matrix(field_name: str, step: int, boundary: BoundaryMode) -> None:
    """Epic #169 DoD: every boundary mode round-trips content and bounds."""
    field = _MATRIX_FIELDS[field_name]()
    raw = _ramp()
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(_SIZE, _SIZE), boundary=boundary),
        sampler=SpatialRegularStride(step=step),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    reflected = np.pad(raw, ((0, _SIZE), (0, _SIZE)), mode="reflect")
    patches = list(patcher.split(field))
    covered = np.zeros(raw.shape, dtype=bool)
    for patch in patches:
        window = _requested_window(boundary, patch.anchor)
        chip = np.asarray(patch.data)
        assert chip.shape == (window.height, window.width)
        transform, bounds = _chip_georef(patch.data)
        assert transform == window_transform(window, _T)
        np.testing.assert_allclose(bounds, window_bounds(window, _T))
        r, c = int(window.row_off), int(window.col_off)
        in_h, in_w = min(window.height, _H - r), min(window.width, _W - c)
        np.testing.assert_array_equal(
            chip[:in_h, :in_w], raw[r : r + in_h, c : c + in_w]
        )
        if boundary == "reflect":
            np.testing.assert_array_equal(
                chip, reflected[r : r + window.height, c : c + window.width]
            )
        if boundary == "pad":
            assert np.all(chip[in_h:, :] == 0) and np.all(chip[:, in_w:] == 0)
        covered[r : r + in_h, c : c + in_w] = True

    # Non-drop modes cover the domain, and the last anchor is the first
    # whose patch reaches the edge — no redundant trailing row/column.
    rows = sorted({p.anchor[0] for p in patches})
    cols = sorted({p.anchor[1] for p in patches})
    if boundary == "drop":
        assert rows[-1] + _SIZE <= _H < rows[-1] + _SIZE + step
        assert cols[-1] + _SIZE <= _W < cols[-1] + _SIZE + step
    else:
        assert covered.all()
        assert rows[-1] + _SIZE >= _H > rows[-2] + _SIZE
        assert cols[-1] + _SIZE >= _W > cols[-2] + _SIZE

    merged = np.asarray(patcher.merge(patches, field.domain))
    np.testing.assert_array_equal(merged[covered], raw[covered])
    assert np.all(merged[~covered] == 0)


def _pad_patches(step: int = _SIZE) -> tuple[RasterField, list[Patch]]:
    field = _raster_case()
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(
            size=(_SIZE, _SIZE), boundary="pad", pad_value=-1.0
        ),
        sampler=SpatialRegularStride(step=step),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    return field, list(patcher.split(field))


@pytest.mark.parametrize(
    "aggregation",
    [
        SpatialOverlapAdd(),
        SpatialSum(),
        SpatialMean(),
        SpatialMax(),
        SpatialMin(),
        SpatialWeightedSum(),
        SpatialMedian(),
    ],
    ids=lambda a: type(a).__name__,
)
def test_every_dense_aggregation_merges_pad_chips(aggregation: Any) -> None:
    # #185: the overhanging pad window used to be forwarded as the
    # accumulator slice (numpy clips 32:48 to 32:37) → broadcast error.
    # Disjoint tiling, so every aggregation reproduces the source, and
    # the -1 fill never leaks in.
    field, patches = _pad_patches()
    merged = aggregation.merge(patches, field.domain)
    np.testing.assert_array_equal(merged, _ramp())


def test_streaming_overlap_add_merges_pad_chips(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    field, patches = _pad_patches(step=_SIZE // 2)
    agg = SpatialOverlapAdd(streaming=True, target_path=str(tmp_path))
    np.testing.assert_allclose(np.asarray(agg.merge(patches, field.domain)[:]), _ramp())


def test_variance_and_categorical_aggregations_crop_pad_chips() -> None:
    field, patches = _pad_patches(step=_SIZE // 2)
    labels = [
        Patch(
            data=(np.asarray(p.data) % 3).astype(np.int64),
            anchor=p.anchor,
            indices=p.indices,
        )
        for p in patches
    ]
    expected_labels = (_ramp() % 3).astype(np.int64)
    np.testing.assert_array_equal(SpatialVariance().merge(patches, field.domain), 0.0)
    np.testing.assert_array_equal(
        SpatialMode().merge(labels, field.domain), expected_labels
    )
    np.testing.assert_array_equal(
        SpatialHardVote(n_classes=3).merge(labels, field.domain), expected_labels
    )
    one_hot = [
        Patch(
            data=np.stack([np.asarray(p.data) == k for k in range(3)]).astype(float),
            anchor=p.anchor,
            indices=p.indices,
        )
        for p in labels
    ]
    np.testing.assert_array_equal(
        SpatialSoftVote(n_classes=3).merge(one_hot, field.domain), expected_labels
    )
    posteriors = [
        Patch(
            data=(np.asarray(p.data), np.ones_like(np.asarray(p.data))),
            anchor=p.anchor,
            indices=p.indices,
            weights=p.weights,
        )
        for p in patches
    ]
    out = SpatialInvVarWeightedMean().merge(posteriors, field.domain)
    np.testing.assert_allclose(out["mu"], _ramp())


def test_negative_pad_window_does_not_wrap() -> None:
    # A window starting at -2 used to become acc[-2:2] — a slice numpy
    # reads from the end — instead of the in-domain rows 0..1.
    field = _arange_field(10)
    raw = np.asarray(field.reader.values)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(4, 4), boundary="pad", pad_value=-1.0),
        sampler=SpatialExplicit([(-2, -2)]),
        window=SpatialBoxcar(),
        aggregation=SpatialSum(),
    )
    merged = patcher.merge(patcher.split(field), field.domain)
    expected = np.zeros(raw.shape, dtype=np.float64)
    expected[:2, :2] = raw[:2, :2]
    np.testing.assert_array_equal(merged, expected)


class TestShrinkClipsNegativeAnchors:
    """#185: ``shrink`` used to keep a negative anchor's full window and let
    the boundless read pad it — pad semantics under the shrink label."""

    def test_explicit_negative_anchor(self) -> None:
        field = _arange_field(10)
        raw = np.asarray(field.reader.values)
        window = SpatialHann()
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(4, 4), boundary="shrink"),
            sampler=SpatialExplicit([(-2, -1)]),
            window=window,
            aggregation=SpatialOverlapAdd(),
        )
        (patch,) = list(patcher.split(field))
        assert patch.indices == Window(0, 0, 3, 2)
        np.testing.assert_array_equal(patch.data.values, raw[0:2, 0:3])
        assert (patch.data.transform.c, patch.data.transform.f) == (0.0, 0.0)
        # The weights are the in-domain part of the full taper, not its
        # top-left corner.
        full = window.weights(patcher.geometry)
        np.testing.assert_array_equal(patch.weights, full[2:4, 1:4])

    def test_centred_coords_near_corner(self) -> None:
        field = _arange_field(20)
        raw = np.asarray(field.reader.values)
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(4, 4), boundary="shrink"),
            sampler=SpatialExplicitCoords([(0.5, 0.5)]),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        (patch,) = list(patcher.split(field))
        # Centred on pixel (0, 0): anchor (-2, -2) clips to rows/cols 0..1.
        assert patch.anchor == (-2, -2)
        assert patch.indices == Window(0, 0, 2, 2)
        np.testing.assert_array_equal(patch.data.values, raw[:2, :2])
        assert patch.weights.shape == (2, 2)


class TestDropPatchLargerThanDomain:
    """#185: ``drop`` emitted one overflowing anchor when ``size > domain``.

    It now places none (the mode's literal meaning: only patches wholly
    in-domain) and warns, rather than raising, so a loop over mixed-size
    scenes keeps running.
    """

    @pytest.mark.parametrize(
        "sampler",
        [
            SpatialRegularStride(step=4),
            SpatialJitteredStride(step=4, seed=0),
            SpatialRandom(n_samples=5, seed=0),
            SpatialPoissonDisk(min_dist=2.0, seed=0),
            SpatialExplicitCoords([(5.5, 5.5)]),
        ],
        ids=[
            "RegularStride",
            "JitteredStride",
            "Random",
            "PoissonDisk",
            "ExplicitCoords",
        ],
    )
    def test_no_anchors_and_a_warning(self, sampler: SpatialSampler) -> None:
        field = _arange_field(10)
        geom = SpatialRectangular(size=(16, 16))
        with pytest.warns(RuntimeWarning, match="exceeds the domain length"):
            anchors = list(sampler.anchors(field.domain, geom))
        assert anchors == []

    def test_one_axis_oversize_is_enough(self) -> None:
        field = _raster_case()  # 37 x 45
        geom = SpatialRectangular(size=(40, 16))
        with pytest.warns(RuntimeWarning, match="exceeds the domain length"):
            anchors = list(SpatialRegularStride(step=8).anchors(field.domain, geom))
        assert anchors == []

    @pytest.mark.parametrize("boundary", ["pad", "shrink", "reflect"])
    def test_non_drop_modes_cover_a_small_domain(self, boundary: BoundaryMode) -> None:
        field = _arange_field(10)
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(16, 16), boundary=boundary),
            sampler=SpatialRegularStride(step=16),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        patches = list(patcher.split(field))
        assert [p.anchor for p in patches] == [(0, 0)]
        merged = patcher.merge(patches, field.domain)
        np.testing.assert_array_equal(merged, field.reader.values)


class TestCheckFullScanOnlyUnderDrop:
    @pytest.mark.parametrize("boundary", ["pad", "shrink", "reflect", "raise"])
    def test_non_drop_modes_do_not_raise(
        self, misaligned_field: RasterField, boundary: BoundaryMode
    ) -> None:
        # 70 / 16 / 16 leaves a 6-px residual that drop loses — but the
        # other modes cover it, so the strict-tiling check must not fire.
        geom = SpatialRectangular(size=(16, 16), boundary=boundary)
        sampler = SpatialRegularStride(step=16, check_full_scan=True)
        assert len(list(sampler.anchors(misaligned_field.domain, geom))) == 25

    def test_drop_still_raises(self, misaligned_field: RasterField) -> None:
        sampler = SpatialRegularStride(step=16, check_full_scan=True)
        geom = SpatialRectangular(size=(16, 16))
        with pytest.raises(IncompleteScanConfiguration, match=r"\(70 - 16\) % 16"):
            list(sampler.anchors(misaligned_field.domain, geom))

    def test_oversize_message_names_the_size(self) -> None:
        field = _arange_field(10)
        sampler = SpatialRegularStride(step=4, check_full_scan=True)
        geom = SpatialRectangular(size=(16, 16))
        with pytest.raises(IncompleteScanConfiguration, match="patch size 16 exceeds"):
            list(sampler.anchors(field.domain, geom))


class TestPadValueValidation:
    """#185: ``pad_value`` used to be cast silently (-999 → 64537 in uint16)."""

    @staticmethod
    def _field(dtype: Any) -> RasterField:
        arr = np.arange(100).reshape(10, 10).astype(dtype)
        return RasterField(
            GeoTensor(
                values=arr, transform=rasterio.Affine.identity(), crs="EPSG:32630"
            )
        )

    @pytest.mark.parametrize(
        ("dtype", "pad_value"),
        [
            (np.uint16, -999.0),
            (np.uint16, 70_000),
            (np.int16, float("nan")),
            (np.uint8, 0.5),
            (np.float32, 1e300),
        ],
    )
    def test_unrepresentable_value_raises(self, dtype: Any, pad_value: float) -> None:
        patcher = _corner_patcher("pad", size=4, pad_value=pad_value)
        with pytest.raises(ValueError, match="cannot be represented"):
            list(patcher.split(self._field(dtype)))

    @pytest.mark.parametrize(
        ("dtype", "pad_value"),
        [
            (np.uint16, 0.0),
            (np.uint16, 65535),
            (np.int16, -999),
            (np.float32, float("nan")),
        ],
    )
    def test_representable_value_fills(self, dtype: Any, pad_value: float) -> None:
        patcher = _corner_patcher("pad", size=4, pad_value=pad_value)
        corner = {p.anchor: p for p in patcher.split(self._field(dtype))}[(8, 8)]
        assert corner.data.values.dtype == dtype
        np.testing.assert_array_equal(
            corner.data.values[2:, :], np.full((2, 4), pad_value, dtype=dtype)
        )

    def test_non_numeric_rejected_at_construction(self) -> None:
        with pytest.raises(TypeError, match="pad_value"):
            SpatialRectangular(size=(4, 4), boundary="pad", pad_value="0")  # type: ignore[arg-type]


class TestGridDomainBoundary:
    """#185: the `GridDomain` path honours the boundary policy too."""

    @pytest.fixture
    def field(self) -> Any:
        xr = pytest.importorskip("xarray")
        from geopatcher import XarrayField

        da = xr.DataArray(
            _ramp(),
            dims=("y", "x"),
            coords={
                "y": 100.0 - 2.0 * np.arange(_H),
                "x": 10.0 + 0.5 * np.arange(_W),
            },
        )
        return XarrayField(da)

    @pytest.mark.parametrize(
        "step", [_SIZE, _SIZE // 2], ids=["step==size", "step<size"]
    )
    @pytest.mark.parametrize("boundary", _MATRIX_MODES)
    def test_split_merge(self, field: Any, step: int, boundary: BoundaryMode) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(_SIZE, _SIZE), boundary=boundary),
            sampler=SpatialRegularStride(step=step),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        raw = _ramp()
        reflected = np.pad(raw, ((0, _SIZE), (0, _SIZE)), mode="reflect")
        patches = list(patcher.split(field))
        covered = np.zeros(raw.shape, dtype=bool)
        for patch in patches:
            r, c = patch.anchor["y"], patch.anchor["x"]
            h = min(_SIZE, _H - r) if boundary == "shrink" else _SIZE
            w = min(_SIZE, _W - c) if boundary == "shrink" else _SIZE
            assert patch.data.shape == (h, w)
            # Coordinates continue the source grid past the edge.
            np.testing.assert_allclose(
                patch.data["y"].values, 100.0 - 2.0 * np.arange(r, r + h)
            )
            np.testing.assert_allclose(
                patch.data["x"].values, 10.0 + 0.5 * np.arange(c, c + w)
            )
            if boundary == "reflect":
                np.testing.assert_array_equal(
                    patch.data.values, reflected[r : r + h, c : c + w]
                )
            covered[r : r + h, c : c + w] = True
        if boundary != "drop":
            assert covered.all()
        merged = patcher.merge(patches, field.domain)
        np.testing.assert_array_equal(merged[covered], raw[covered])

    def test_raise_on_grid_overflow(self, field: Any) -> None:
        patcher = SpatialPatcher(
            geometry=SpatialRectangular(size=(_SIZE, _SIZE), boundary="raise"),
            sampler=SpatialRegularStride(step=_SIZE),
            window=SpatialBoxcar(),
            aggregation=SpatialOverlapAdd(),
        )
        with pytest.raises(ValueError, match="overflows the domain"):
            list(patcher.split(field))
