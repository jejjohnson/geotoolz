"""Tests for the spatial `*Window` family."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from scipy.signal.windows import hann, tukey

from geopatcher import (
    RasterField,
    SpatialBoxcar,
    SpatialCustom,
    SpatialGaussian,
    SpatialHann,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialTukey,
)


@pytest.fixture
def geom() -> SpatialRectangular:
    return SpatialRectangular(size=(8, 8))


class TestSpatialBoxcar:
    def test_uniform(self, geom: SpatialRectangular) -> None:
        w = SpatialBoxcar().weights(geom)
        assert w.shape == (8, 8)
        np.testing.assert_array_equal(w, 1.0)


class TestSpatialHann:
    def test_periodic_values_pinned(self, geom: SpatialRectangular) -> None:
        # Periodic (DFT-even) Hann, n = 8: 0.5 - 0.5 cos(2 pi k / 8).
        s = np.sqrt(2.0) / 4.0
        ref_1d = np.array([0.0, 0.5 - s, 0.5, 0.5 + s, 1.0, 0.5 + s, 0.5, 0.5 - s])
        w = SpatialHann().weights(geom)
        assert w.shape == (8, 8)
        np.testing.assert_allclose(w, np.outer(ref_1d, ref_1d), rtol=0, atol=1e-15)
        # Leading endpoint is zero, trailing endpoint is not (periodic).
        assert w[0, 0] == 0.0
        assert w[-1, -1] > 0.02

    def test_cola_at_half_hop(self) -> None:
        # w[k] + w[k + N/2] == 1 exactly for every k: true COLA at hop N/2.
        for n in (4, 8, 16, 256):
            w = SpatialHann().weights(SpatialRectangular(size=(n, n)))[n // 2]
            np.testing.assert_allclose(
                w[: n // 2] + w[n // 2 :], 1.0, rtol=0, atol=1e-15
            )


def _ref(n: int, alpha: float | None) -> np.ndarray:
    if n < 3:
        return np.ones(n)
    if alpha is None:
        return hann(n, sym=False)
    return tukey(n, alpha=alpha, sym=False)


@pytest.mark.parametrize("size", [(8, 8), (16, 12), (5, 7), (3, 32)])
@pytest.mark.parametrize("alpha", [None, 0.0, 0.25, 0.5, 0.75, 1.0])
def test_hann_tukey_convention(size: tuple[int, int], alpha: float | None) -> None:
    """Both tapers are scipy's periodic (sym=False) windows, outer-producted."""
    geom = SpatialRectangular(size=size)
    window = SpatialHann() if alpha is None else SpatialTukey(alpha=alpha)
    expected = np.outer(_ref(size[0], alpha), _ref(size[1], alpha))
    np.testing.assert_allclose(window.weights(geom), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("size", [(8, 8), (16, 12), (5, 7)])
def test_tukey_alpha_one_is_hann(size: tuple[int, int]) -> None:
    geom = SpatialRectangular(size=size)
    np.testing.assert_array_equal(
        SpatialTukey(alpha=1.0).weights(geom), SpatialHann().weights(geom)
    )


@pytest.mark.parametrize("window", [SpatialHann(), SpatialTukey(alpha=0.5)])
@pytest.mark.parametrize("n", [1, 2])
def test_short_axis_falls_back_to_boxcar(window, n: int) -> None:
    # A 1- or 2-sample axis cannot carry a taper (np.hanning(2) was all
    # zeros); it is boxcar, while the long axis keeps its taper.
    w = window.weights(SpatialRectangular(size=(n, 8)))
    assert w.shape == (n, 8)
    long_axis = _ref(8, None if isinstance(window, SpatialHann) else 0.5)
    np.testing.assert_allclose(w, np.tile(long_axis, (n, 1)), rtol=0, atol=1e-15)
    assert (w.sum(axis=1) > 0).all()


class TestSpatialTukey:
    def test_alpha_zero_is_boxcar(self, geom: SpatialRectangular) -> None:
        w = SpatialTukey(alpha=0.0).weights(geom)
        np.testing.assert_allclose(w, 1.0)


class TestSpatialGaussian:
    def test_peak_centre(self, geom: SpatialRectangular) -> None:
        w = SpatialGaussian(sigma=0.5).weights(geom)
        ctr = w[w.shape[0] // 2, w.shape[1] // 2]
        edge = w[0, 0]
        assert ctr > edge

    @pytest.mark.parametrize("n", [8, 13, 64])
    @pytest.mark.parametrize("sigma", [0.25, 0.5, 1.0])
    def test_sigma_is_fraction_of_half_width(self, n: int, sigma: float) -> None:
        # sigma is a fraction of the half-width: std = sigma * n / 2 pixels,
        # centred at (n - 1) / 2.
        x = np.arange(n) - (n - 1) / 2.0
        ref = np.exp(-0.5 * (x / (sigma * n / 2.0)) ** 2)
        w = SpatialGaussian(sigma=sigma).weights(SpatialRectangular(size=(n, n)))
        np.testing.assert_allclose(w, np.outer(ref, ref), rtol=1e-14, atol=0)


class TestSpatialCustom:
    def test_calls_user_fn(self, geom: SpatialRectangular) -> None:
        called: list[bool] = []

        def fn(g):
            called.append(True)
            return np.full(g.size, 0.5)

        w = SpatialCustom(fn=fn).weights(geom)
        assert called == [True]
        np.testing.assert_array_equal(w, 0.5)


# ---------------------------------------------------------------------------
# Merged constant field: interior seams vs the leading border ring
# ---------------------------------------------------------------------------

_C = 3.25


def _constant_field(n: int) -> RasterField:
    return RasterField(
        GeoTensor(
            values=np.full((n, n), _C, dtype=np.float64),
            transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
            crs="EPSG:32630",
        )
    )


def _merge_identity(patcher: SpatialPatcher, field: RasterField) -> np.ndarray:
    patches = [
        type(p)(
            data=np.asarray(p.data.values),
            anchor=p.anchor,
            indices=p.indices,
            weights=p.weights,
        )
        for p in patcher.split(field)
    ]
    return np.asarray(patcher.merge(patches, field.domain))


def test_overlap_add_constant_field_border() -> None:
    """Periodic Hann at hop N/2 on a tileable 40x40 field (16/8, drop).

    Every interior seam is the constant exactly. The only defect is the
    leading border ring — row 0 and column 0 — where the sole covering chip
    has ``w[0] = 0``, so the accumulated weight is zero and OverlapAdd fills
    ``0.0``. This pins today's anchor behaviour (regular samplers start at
    anchor 0, never negative), not a property of the window alone.
    """
    field = _constant_field(40)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16)),
        sampler=SpatialRegularStride(step=(8, 8)),
        window=SpatialHann(),
        aggregation=SpatialOverlapAdd(),
    )
    out = _merge_identity(patcher, field)
    assert out.shape == (40, 40)
    # Everything but the leading ring — including all interior seams (rows
    # / cols 8, 16, 24, 32) and the trailing row/col — is exactly C.
    np.testing.assert_allclose(out[1:, 1:], _C, rtol=1e-15, atol=0)
    # Leading border ring: zero weight -> 0.0 fill.
    np.testing.assert_array_equal(out[0, :], 0.0)
    np.testing.assert_array_equal(out[:, 0], 0.0)

    # True COLA: without normalisation the doubly-covered interior is
    # already C (sum of shifted periodic Hann == 1); only the outer band
    # of width `step`, covered by a single chip, carries the bare taper.
    raw_patcher = SpatialPatcher(
        geometry=patcher.geometry,
        sampler=patcher.sampler,
        window=patcher.window,
        aggregation=SpatialOverlapAdd(normalize_by_window=False),
    )
    raw = _merge_identity(raw_patcher, field)
    np.testing.assert_allclose(raw[8:32, 8:32], _C, rtol=1e-15, atol=0)
    w1 = hann(16, sym=False)
    np.testing.assert_allclose(raw[0:8, 20], _C * w1[0:8], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(raw[32:40, 20], _C * w1[8:16], rtol=1e-14, atol=0)


def test_overlap_add_hann_step_equals_size_zero_seams() -> None:
    """``step == size`` with Hann leaves every chip's first row/col at zero
    weight — documented reason to overlap chips (or use Boxcar)."""
    field = _constant_field(32)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16)),
        sampler=SpatialRegularStride(step=(16, 16)),
        window=SpatialHann(),
        aggregation=SpatialOverlapAdd(),
    )
    out = _merge_identity(patcher, field)
    seam = np.zeros((32, 32), dtype=bool)
    seam[[0, 16], :] = True
    seam[:, [0, 16]] = True
    np.testing.assert_array_equal(out[seam], 0.0)
    np.testing.assert_allclose(out[~seam], _C, rtol=1e-15, atol=0)
