"""Tests for the spatial `Aggregation` family."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geopatcher import Patch, RasterField, SpatialPatcher, spatial
from geopatcher._src.spatial.geometry import _MaskedWindow


def _patch(data, row: int, col: int, weights=None, *, shape=None) -> Patch:
    if shape is None:
        shape = data.shape[-2:]  # type: ignore[union-attr]
    h, w = shape
    win = Window(col_off=col, row_off=row, width=w, height=h)
    return Patch(data=data, anchor=(row, col), indices=win, weights=weights)


@pytest.fixture
def empty_field() -> GeoTensor:
    return GeoTensor(
        values=np.zeros((4, 4), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )


class TestSpatialSum:
    def test_disjoint_patches(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.ones((2, 2)), 0, 0)
        p2 = _patch(np.full((2, 2), 3.0), 2, 2)
        out = spatial.aggregation.Sum().merge([p1, p2], empty_field)
        assert np.nansum(out) == 4 * 1 + 4 * 3
        assert out[0, 0] == 1.0
        assert out[3, 3] == 3.0
        # Cells no patch reached are the NaN fill, not a 0.0 sum.
        assert np.isnan(out[0, 2:]).all() and np.isnan(out[2:, 0]).all()


class TestSpatialMean:
    def test_overlap_mean(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.full((2, 2), 2.0), 0, 0)
        p2 = _patch(np.full((2, 2), 6.0), 1, 1)  # overlaps p1 at (1,1)
        out = spatial.aggregation.Mean().merge([p1, p2], empty_field)
        assert out[1, 1] == pytest.approx(4.0)


class TestSpatialVariance:
    def test_variance_zero_for_constant_patches(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.full((2, 2), 5.0), 0, 0)
        p2 = _patch(np.full((2, 2), 5.0), 0, 0)
        out = spatial.aggregation.Variance().merge([p1, p2], empty_field)
        assert out[0, 0] == pytest.approx(0.0)


class TestSpatialOverlapAdd:
    def test_uniform_weights_reconstruct(self, empty_field: GeoTensor) -> None:
        # Two patches, no overlap, boxcar weights -> exact reconstruction
        p1 = _patch(np.ones((2, 2)), 0, 0, weights=np.ones((2, 2)))
        p2 = _patch(np.full((2, 2), 5.0), 2, 2, weights=np.ones((2, 2)))
        out = spatial.aggregation.OverlapAdd().merge([p1, p2], empty_field)
        assert out[0, 0] == 1.0
        assert out[3, 3] == 5.0

    def test_multiband_raster_trailing_axis(self) -> None:
        """The (row, col) slicer must target the trailing two axes when the
        domain has a leading band/time dim. Regression for the bug where
        ``acc[(row_slice, col_slice)]`` indexed the first two axes instead.
        """
        domain = GeoTensor(
            values=np.zeros((3, 4, 4), dtype=np.float32),  # (band, H, W)
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
        # Two non-overlapping (3-band, 2x2) patches at (0,0) and (2,2).
        p1 = _patch(np.full((3, 2, 2), 1.0), 0, 0, weights=np.ones((2, 2)))
        p2 = _patch(np.full((3, 2, 2), 5.0), 2, 2, weights=np.ones((2, 2)))
        out = spatial.aggregation.OverlapAdd().merge([p1, p2], domain)
        assert out.shape == (3, 4, 4)
        np.testing.assert_array_equal(out[:, :2, :2], 1.0)
        np.testing.assert_array_equal(out[:, 2:, 2:], 5.0)

    def test_normalisation_with_overlap(self, empty_field: GeoTensor) -> None:
        # Two identical patches with non-uniform weights at the same location.
        # OverlapAdd normalises (sum w*x / sum w) → the per-cell average, which
        # equals x because both patches have the same data.
        w = np.array([[0.5, 1.0], [1.0, 0.5]])
        p1 = _patch(np.full((2, 2), 3.0), 0, 0, weights=w)
        p2 = _patch(np.full((2, 2), 3.0), 0, 0, weights=w)
        out = spatial.aggregation.OverlapAdd().merge([p1, p2], empty_field)
        np.testing.assert_allclose(out[:2, :2], 3.0)


class TestSpatialInvVarWeightedMean:
    def test_two_patches_gaussian_merge(self, empty_field: GeoTensor) -> None:
        # Two overlapping Gaussian patches with the same mean -> global mean
        # equals it; global variance equals 1 / (1/var1 + 1/var2).
        mu = np.full((2, 2), 3.0)
        var1 = np.full((2, 2), 4.0)
        var2 = np.full((2, 2), 1.0)
        p1 = _patch((mu, var1), 0, 0, shape=(2, 2))
        p2 = _patch((mu, var2), 0, 0, shape=(2, 2))
        out = spatial.aggregation.InvVarWeightedMean().merge([p1, p2], empty_field)
        np.testing.assert_allclose(out["mu"][0, 0], 3.0)
        np.testing.assert_allclose(out["var"][0, 0], 1.0 / (1 / 4 + 1 / 1))


class TestSpatialByIndex:
    def test_returns_anchor_data_pairs(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.array([[1.0]]), 0, 0)
        p2 = _patch(np.array([[2.0]]), 1, 1)
        out = spatial.aggregation.ByIndex().merge([p1, p2], empty_field)
        assert [anchor for anchor, _ in out] == [(0, 0), (1, 1)]
        assert dict(out)[(1, 1)][0, 0] == 2.0


class TestSpatialHardVote:
    def test_majority(self, empty_field: GeoTensor) -> None:
        # Three patches at (0,0); 2 vote for class 1, 1 votes for class 0
        p1 = _patch(np.zeros((2, 2), dtype=int), 0, 0)
        p2 = _patch(np.ones((2, 2), dtype=int), 0, 0)
        p3 = _patch(np.ones((2, 2), dtype=int), 0, 0)
        out = spatial.aggregation.HardVote(n_classes=2).merge([p1, p2, p3], empty_field)
        assert (out[:2, :2] == 1).all()


class TestSpatialMedian:
    def test_warns_when_streaming(self, empty_field: GeoTensor) -> None:
        # spatial.aggregation.Median.streaming_safe == False —
        # `_warn_if_unsafe_streaming` is called from SpatialPatcher.merge, not from
        # Median.merge itself.
        from geopatcher._src.spatial.aggregation import (
            _warn_if_unsafe_streaming,
        )

        with pytest.warns(RuntimeWarning, match="streaming_safe = False"):
            _warn_if_unsafe_streaming(spatial.aggregation.Median())

    def test_strict_raises_on_streaming_unsafe(self, empty_field: GeoTensor) -> None:
        # ADR-003: gp.observe.set_strict(True) promotes the streaming_safe
        # warning to a hard RuntimeError so batch / CI callers fail fast.
        # Capture and restore the initial value so we don't leak state
        # if the caller has GEOPATCHER_STRICT=1 set or another test left
        # strict mode flipped on.
        import geopatcher as gp
        from geopatcher._src.spatial.aggregation import (
            _warn_if_unsafe_streaming,
        )

        original = gp.observe.get_strict()
        gp.observe.set_strict(True)
        try:
            assert gp.observe.get_strict() is True
            with pytest.raises(RuntimeError, match="streaming_safe = False"):
                _warn_if_unsafe_streaming(spatial.aggregation.Median())
        finally:
            gp.observe.set_strict(original)
        assert gp.observe.get_strict() is original

    def test_median_value(self, empty_field: GeoTensor) -> None:
        p1 = _patch(np.full((2, 2), 1.0), 0, 0)
        p2 = _patch(np.full((2, 2), 5.0), 0, 0)
        p3 = _patch(np.full((2, 2), 3.0), 0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = spatial.aggregation.Median().merge([p1, p2, p3], empty_field)
        assert out[0, 0] == 3.0


# ---------------------------------------------------------------------------
# #192 — masks, NaN, Σw = 0 fill, grid anchors, band domains, unknown indices
# ---------------------------------------------------------------------------


def _masked_patch(data, mask, row: int = 0, col: int = 0) -> Patch:
    h, w = np.shape(mask)
    win = Window(col_off=col, row_off=row, width=w, height=h)
    return Patch(data=data, anchor=(row, col), indices=_MaskedWindow(win, mask))


# Only (0, 0) is inside the mask; the masked-out 100s must never leak in.
_MASK = np.array([[True, False], [False, False]])
_DATA = np.array([[7.0, 100.0], [100.0, 100.0]])

_DENSE = [
    spatial.aggregation.Sum,
    spatial.aggregation.Mean,
    spatial.aggregation.Max,
    spatial.aggregation.Min,
    spatial.aggregation.WeightedSum,
    spatial.aggregation.OverlapAdd,
    spatial.aggregation.Median,
    spatial.aggregation.Mode,
]


@pytest.mark.parametrize("cls", _DENSE, ids=lambda c: c.__name__)
def test_masked_window_respected_by_all_aggregations(
    cls: type, empty_field: GeoTensor
) -> None:
    out = cls().merge([_masked_patch(_DATA, _MASK)], empty_field)
    expected = np.full((4, 4), np.nan)
    expected[0, 0] = 7.0
    np.testing.assert_array_equal(out, expected)


def test_masked_window_respected_by_variance_votes_and_posteriors(
    empty_field: GeoTensor,
) -> None:
    twice = [_masked_patch(_DATA, _MASK), _masked_patch(_DATA + 2, _MASK)]
    var = spatial.aggregation.Variance().merge(twice, empty_field)
    assert var[0, 0] == pytest.approx(2.0)
    assert np.isnan(var.reshape(-1)[1:]).all()

    labels = _masked_patch(np.array([[1, 0], [0, 0]]), _MASK)
    hard = spatial.aggregation.HardVote(n_classes=2).merge([labels], empty_field)
    assert hard[0, 0] == 1
    assert (hard.reshape(-1)[1:] == -1).all()

    probs = np.stack([1.0 - _MASK, _MASK.astype(float)])  # class 1 only at (0,0)
    soft = spatial.aggregation.SoftVote(n_classes=2).merge(
        [_masked_patch(probs, _MASK)], empty_field
    )
    assert soft[0, 0] == 1
    assert (soft.reshape(-1)[1:] == -1).all()

    post = _masked_patch((_DATA, np.ones((2, 2))), _MASK)
    out = spatial.aggregation.InvVarWeightedMean().merge([post], empty_field)
    assert out["mu"][0, 0] == 7.0
    assert out["var"][0, 0] == 1.0
    assert np.isnan(out["mu"].reshape(-1)[1:]).all()

    minmax = spatial.aggregation.MinMax().merge(
        [_masked_patch(_DATA, _MASK)], empty_field
    )
    assert minmax == {"min": 7.0, "max": 7.0}
    assert spatial.aggregation.MeanStd().merge(twice, empty_field)[
        "mean"
    ] == pytest.approx(8.0)


def test_masked_mean_through_polygon_patcher() -> None:
    # A triangle over an 8x8 raster: the mean reproduces the source at the
    # pixels whose centre lies inside it and is NaN everywhere else —
    # previously the whole bounding box was written.
    t = rasterio.transform.from_origin(0.0, 8.0, 1.0, 1.0)
    source = np.arange(64, dtype=np.float32).reshape(8, 8)
    field = RasterField(GeoTensor(values=source, transform=t, crs="EPSG:32630"))
    # The hypotenuse passes through no pixel centre (no edge ambiguity).
    tri = shapely.Polygon([(0.0, 8.0), (8.0, 8.0), (0.0, 0.5)])
    patcher = SpatialPatcher(
        geometry=spatial.geometry.PolygonIntersection(polygons=pd.Series([tri])),
        sampler=spatial.sampler.Explicit(anchors_=[0]),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.Mean(),
    )
    patches = [p.with_data(np.asarray(p.data.values)) for p in patcher.split(field)]
    merged = patcher.merge(patches, field.domain)
    rows, cols = np.mgrid[0:8, 0:8]
    # Centre (c + .5, 7.5 - r) above the line y = 0.5 + 0.9375 x.
    inside = 7.5 - rows > 0.5 + 0.9375 * (cols + 0.5)
    np.testing.assert_array_equal(merged[inside], source[inside])
    assert np.isnan(merged[~inside]).all()


@pytest.mark.parametrize("cls", _DENSE, ids=lambda c: c.__name__)
def test_zero_weight_cells_are_nodata(cls: type, empty_field: GeoTensor) -> None:
    covered = _patch(np.full((2, 2), 4.0), 0, 0)
    out = cls().merge([covered], empty_field)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out[:2, :2], 4.0)
    assert np.isnan(out[2:, :]).all()
    assert np.isnan(out[:, 2:]).all()
    # A covered-but-all-NaN cell is no sample either.
    hole = _patch(np.full((2, 2), np.nan), 2, 2)
    assert np.isnan(cls().merge([hole], empty_field)).all()
    # fill_value overrides the NaN default.
    filled = cls(fill_value=-9999.0).merge([covered], empty_field)
    np.testing.assert_array_equal(filled[2:, :], -9999.0)
    np.testing.assert_array_equal(filled[:2, :2], 4.0)


def test_zero_weight_fill_on_votes_and_posteriors(empty_field: GeoTensor) -> None:
    labels = _patch(np.ones((2, 2), dtype=int), 0, 0)
    hard = spatial.aggregation.HardVote(n_classes=2).merge([labels], empty_field)
    assert hard.dtype == np.int64
    assert (hard[2:, :] == -1).all()
    nan_hard = spatial.aggregation.HardVote(n_classes=2, fill_value=np.nan).merge(
        [labels], empty_field
    )
    assert nan_hard.dtype == np.float64
    assert np.isnan(nan_hard[2:, :]).all()
    assert (nan_hard[:2, :2] == 1).all()

    probs = _patch(np.stack([np.zeros((2, 2)), np.ones((2, 2))]), 0, 0, shape=(2, 2))
    soft = spatial.aggregation.SoftVote(n_classes=2, fill_value=255).merge(
        [probs], empty_field
    )
    assert soft.dtype == np.int64
    assert (soft[:2, :2] == 1).all()
    assert (soft[2:, :] == 255).all()

    post = _patch((np.full((2, 2), 3.0), np.ones((2, 2))), 0, 0, shape=(2, 2))
    out = spatial.aggregation.InvVarWeightedMean(fill_value=-1.0).merge(
        [post], empty_field
    )
    assert (out["mu"][2:, :] == -1.0).all()
    assert (out["var"][2:, :] == -1.0).all()

    # One sample: the ddof=1 variance is undefined (as np.nanvar) → fill.
    var = spatial.aggregation.Variance().merge(
        [_patch(np.ones((2, 2)), 0, 0)], empty_field
    )
    assert np.isnan(var).all()


def _hann_constant_field(fill: float | None = None) -> RasterField:
    kwargs = {} if fill is None else {"fill_value_default": fill}
    return RasterField(
        GeoTensor(
            values=np.full((24, 24), 2.5),
            transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 240.0),
            crs="EPSG:32630",
            **kwargs,
        )
    )


def _hann_merge(field: RasterField, aggregation: Any) -> np.ndarray:
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(8, 8)),
        sampler=spatial.sampler.RegularStride(step=(4, 4)),
        window=spatial.window.Hann(),
        aggregation=aggregation,
    )
    patches = [p.with_data(np.asarray(p.data.values)) for p in patcher.split(field)]
    return np.asarray(patcher.merge(patches, field.domain))


@pytest.mark.parametrize(
    "cls", [spatial.aggregation.OverlapAdd, spatial.aggregation.WeightedSum]
)
def test_periodic_hann_leading_ring_is_nodata(cls: type) -> None:
    # Periodic Hann + drop: the sole chip covering row 0 / col 0 puts its
    # w[0] = 0 sample there, so Σw = 0 — the leading ring used to be 0.0,
    # indistinguishable from data.
    out = _hann_merge(_hann_constant_field(), cls())
    assert np.isnan(out[0, :]).all()
    assert np.isnan(out[:, 0]).all()
    assert not np.isnan(out[1:, 1:]).any()
    if cls is spatial.aggregation.OverlapAdd:
        np.testing.assert_allclose(out[1:, 1:], 2.5, rtol=1e-14)

    # The domain's nodata as the override.
    field = _hann_constant_field(fill=-9999.0)
    out = _hann_merge(field, cls(fill_value=field.domain.fill_value_default))
    np.testing.assert_array_equal(out[0, :], -9999.0)
    np.testing.assert_array_equal(out[:, 0], -9999.0)
    assert (out[1:, 1:] > 0).all()


def test_by_index_grid_anchors() -> None:
    # GridDomain anchors are dicts (unhashable) — the old `{anchor: data}`
    # raised TypeError. Pairs keep every anchor object as-is, in order.
    xr = pytest.importorskip("xarray")
    from geopatcher.fields import XarrayField

    da = xr.DataArray(
        np.arange(8 * 12, dtype=np.float32).reshape(8, 12),
        dims=("latitude", "longitude"),
        coords={"latitude": np.arange(8.0), "longitude": np.arange(12.0)},
    )
    field = XarrayField(da)
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(4, 4)),
        sampler=spatial.sampler.RegularStride(step=(4, 4)),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.ByIndex(),
    )
    patches = list(patcher.split(field))
    out = patcher.merge(patches, field.domain)
    assert len(out) == len(patches) == 6
    for (anchor, data), p in zip(out, patches, strict=True):
        assert isinstance(anchor, dict)
        assert anchor is p.anchor
        assert data is p.data
    # Array anchors (graph geometries) and repeated anchors survive too.
    arr = Patch(data=1.0, anchor=np.array([3, 4]), indices=np.array([3, 4]))
    pairs = spatial.aggregation.ByIndex().merge([arr, arr], None)
    assert len(pairs) == 2
    assert pairs[0][0] is arr.anchor


def test_soft_vote_band_domain() -> None:
    # (K=2, 4, 4) probability chips on a (band=3, 8, 8) domain give one
    # (8, 8) label map: the cells hold what the patches carry, not three
    # copies of it.
    domain = GeoTensor(
        values=np.zeros((3, 8, 8), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    probs = np.stack([np.full((4, 4), 0.2), np.full((4, 4), 0.8)])
    p1 = _patch(probs, 0, 0)
    p2 = _patch(probs[::-1], 4, 4)
    out = spatial.aggregation.SoftVote(n_classes=2).merge([p1, p2], domain)
    assert out.shape == (8, 8)
    assert (out[:4, :4] == 1).all()
    assert (out[4:, 4:] == 0).all()
    assert (out[:4, 4:] == -1).all()
    # A full (K, band, h, w) chip keeps the band axis.
    full = np.broadcast_to(probs[:, None], (2, 3, 4, 4))
    out = spatial.aggregation.SoftVote(n_classes=2).merge([_patch(full, 0, 0)], domain)
    assert out.shape == (3, 8, 8)
    assert (out[:, :4, :4] == 1).all()


@pytest.mark.parametrize(
    "aggregation",
    [
        spatial.aggregation.Sum(),
        spatial.aggregation.Max(),
        spatial.aggregation.Min(),
        spatial.aggregation.Mean(),
        spatial.aggregation.WeightedSum(),
        spatial.aggregation.OverlapAdd(),
        spatial.aggregation.Median(),
        spatial.aggregation.Mode(),
        spatial.aggregation.HardVote(n_classes=4),
    ],
    ids=lambda agg: type(agg).__name__,
)
@pytest.mark.parametrize("cells", [(1, 4, 4), (4, 4), (2, 4, 4)], ids=str)
def test_dense_merge_takes_band_axes_from_patches(aggregation, cells) -> None:
    # A per-patch operator that changes the band count (4 bands → 1 index,
    # a 2-D map, 2 features) merges into the patches' band axes on the
    # domain grid instead of being broadcast into the domain's 4 bands.
    domain = GeoTensor(
        values=np.zeros((4, 8, 8), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    patches = [
        _patch(np.full(cells, float(r // 4 + c // 4)), r, c)
        for r in (0, 4)
        for c in (0, 4)
    ]
    out = aggregation.merge(patches, domain)
    assert out.shape == (*cells[:-2], 8, 8)
    np.testing.assert_array_equal(np.asarray(out)[..., 4:, 4:], 2)


def test_dense_merge_rejects_patches_with_different_band_axes() -> None:
    domain = GeoTensor(
        values=np.zeros((4, 8, 8), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    patches = [_patch(np.ones((1, 4, 4)), 0, 0), _patch(np.ones((2, 4, 4)), 0, 4)]
    with pytest.raises(
        ValueError, match=r"leading \(band / time\) axes: \(2,\) after \(1,\)"
    ):
        spatial.aggregation.Mean().merge(patches, domain)


@pytest.mark.parametrize(
    "aggregation",
    [
        *(cls() for cls in _DENSE),
        spatial.aggregation.HardVote(n_classes=2),
        spatial.aggregation.SoftVote(n_classes=2),
        spatial.aggregation.InvVarWeightedMean(),
        spatial.aggregation.MeanStd(),
        spatial.aggregation.MinMax(),
    ],
    ids=lambda a: type(a).__name__,
)
def test_unknown_indices_raise(aggregation: Any, empty_field: GeoTensor) -> None:
    # A point-index array is not a dense placement: every dense aggregation
    # used to skip it silently and return an empty field.
    if isinstance(aggregation, spatial.aggregation.InvVarWeightedMean):
        data: Any = (np.ones(2), np.ones(2))
    else:
        data = np.ones((2, 2))
    patch = Patch(data=data, anchor=0, indices=np.array([1, 2]))
    with pytest.raises(TypeError, match="dense aggregations need"):
        aggregation.merge([patch], empty_field)


def test_mean_std_and_min_max_are_nan_aware_and_crop_pad_fill() -> None:
    # One on_error="mask" (all-NaN) patch used to poison the global
    # stats ({'mean': nan}, {'min': inf, 'max': -inf}); pad fill (-1) used
    # to count as data.
    field = RasterField(
        GeoTensor(
            values=np.arange(36, dtype=np.float64).reshape(6, 6) + 1,
            transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 6.0),
            crs="EPSG:32630",
        )
    )
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(
            size=(4, 4), boundary="pad", pad_value=-1.0
        ),
        sampler=spatial.sampler.RegularStride(step=(4, 4)),
        window=spatial.window.Boxcar(),
        aggregation=spatial.aggregation.MinMax(),
    )
    patches = [p.with_data(np.asarray(p.data.values)) for p in patcher.split(field)]
    patches.append(patches[0].with_data(np.full((4, 4), np.nan)))
    assert spatial.aggregation.MinMax().merge(patches, field.domain) == {
        "min": 1.0,
        "max": 36.0,
    }
    stats = spatial.aggregation.MeanStd().merge(patches, field.domain)
    values = np.arange(36) + 1.0
    assert stats["mean"] == pytest.approx(values.mean())
    assert stats["std"] == pytest.approx(values.std(ddof=1))
    # Non-dense domains (no shape) still reduce every non-NaN sample.
    loose = [Patch(data=np.array([1.0, np.nan, 3.0]), anchor=0, indices=None)]
    assert spatial.aggregation.MinMax().merge(loose, None) == {"min": 1.0, "max": 3.0}


def test_inv_var_zero_variance_is_exact(empty_field: GeoTensor) -> None:
    exact = _patch((np.full((2, 2), 5.0), np.zeros((2, 2))), 0, 0, shape=(2, 2))
    noisy = _patch((np.full((2, 2), 1.0), np.full((2, 2), 2.0)), 0, 0, shape=(2, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = spatial.aggregation.InvVarWeightedMean().merge(
            [exact, noisy], empty_field
        )
    np.testing.assert_array_equal(out["mu"][:2, :2], 5.0)
    np.testing.assert_array_equal(out["var"][:2, :2], 0.0)


def _random_overlapping(seed: int, n: int = 20) -> list[Patch]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r, c = (int(v) for v in rng.integers(-1, 6, size=2))
        data = rng.integers(0, 4, size=(3, 3)).astype(float)
        data[rng.random((3, 3)) < 0.2] = np.nan
        out.append(_patch(data, r, c))
    return out


def _reference_stack(patches: list[Patch], shape: tuple[int, int]) -> np.ndarray:
    """One full NaN layer per patch — the old O(n_patches * cells) layout."""
    layers = []
    for p in patches:
        full = np.full((shape[0] + 8, shape[1] + 8), np.nan)
        r, c = p.anchor
        full[r + 4 : r + 7, c + 4 : c + 7] = p.data
        layers.append(full[4 : 4 + shape[0], 4 : 4 + shape[1]])
    return np.stack(layers)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_median_and_mode_match_reference(seed: int) -> None:
    domain = GeoTensor(
        values=np.zeros((8, 8)), transform=rasterio.Affine.identity(), crs="EPSG:32630"
    )
    patches = _random_overlapping(seed)
    ref = _reference_stack(patches, (8, 8))
    covered = (~np.isnan(ref)).any(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref_median = np.nanmedian(ref, axis=0)
    np.testing.assert_array_equal(
        spatial.aggregation.Median().merge(patches, domain), ref_median
    )

    ref_mode = np.full((8, 8), np.nan)
    for i, j in zip(*np.nonzero(covered), strict=True):
        cell = ref[:, i, j]
        vals, counts = np.unique(cell[~np.isnan(cell)], return_counts=True)
        ref_mode[i, j] = vals[np.argmax(counts)]  # ties → smallest value
    np.testing.assert_array_equal(
        spatial.aggregation.Mode().merge(patches, domain), ref_mode
    )
    labels = spatial.aggregation.Mode(fill_value=-1).merge(patches, domain)
    assert labels.dtype == np.int64
    np.testing.assert_array_equal(labels, np.where(covered, ref_mode, -1))


def test_median_buffer_bounded_by_overlap() -> None:
    # 50 1x1 patches over a 4x4 domain: the per-cell history is as deep as
    # the largest overlap (4), not one full-domain layer per patch (50).
    from geopatcher._src.spatial.aggregation import _overlap_stack

    patches = [_patch(np.array([[float(i)]]), i % 4, (i // 4) % 4) for i in range(50)]
    stack, count = _overlap_stack(patches, (4, 4))
    assert stack.shape[0] == count.max() == 4


def test_vote_ties_go_to_lowest_class(empty_field: GeoTensor) -> None:
    zero = _patch(np.zeros((2, 2), dtype=int), 0, 0)
    one = _patch(np.ones((2, 2), dtype=int), 0, 0)
    hard = spatial.aggregation.HardVote(n_classes=2).merge([one, zero], empty_field)
    assert (hard[:2, :2] == 0).all()
    mode = spatial.aggregation.Mode().merge([one, zero], empty_field)
    assert (mode[:2, :2] == 0).all()
