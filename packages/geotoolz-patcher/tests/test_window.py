"""Tests for the spatial `*Window` family."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from geoarray import GeoArray

from geopatcher import RasterField, SpatialPatcher, spatial


@pytest.fixture
def geom() -> spatial.geometry.Rectangular:
    return spatial.geometry.Rectangular(size=(8, 8))


class TestSpatialBoxcar:
    def test_uniform(self, geom: spatial.geometry.Rectangular) -> None:
        w = spatial.window.Boxcar().weights(geom)
        assert w.shape == (8, 8)
        np.testing.assert_array_equal(w, 1.0)


class TestSpatialHann:
    def test_values_pinned(self, geom: spatial.geometry.Rectangular) -> None:
        # Periodic Hann sampled at pixel centres, n = 8: sin²(π (k + ½) / 8).
        ref_1d = np.sin(np.pi * (np.arange(8) + 0.5) / 8) ** 2
        w = spatial.window.Hann().weights(geom)
        assert w.shape == (8, 8)
        np.testing.assert_allclose(w, np.outer(ref_1d, ref_1d), rtol=0, atol=1e-15)
        # Symmetric and never zero: no edge pixel is dropped.
        np.testing.assert_allclose(w, w[::-1, ::-1], rtol=0, atol=1e-15)
        assert w.min() == pytest.approx(np.sin(np.pi / 16) ** 4)

    def test_cola_at_half_hop(self) -> None:
        # w[k] + w[k + N/2] == 1 exactly for every k: true COLA at hop N/2.
        for n in (4, 8, 16, 256):
            w = spatial.window.Hann().weights(
                spatial.geometry.Rectangular(size=(1, n))
            )[0]
            np.testing.assert_allclose(
                w[: n // 2] + w[n // 2 :], 1.0, rtol=0, atol=1e-15
            )


def _ref(n: int, alpha: float | None) -> np.ndarray:
    """Periodic Tukey / Hann of period ``n`` at the pixel centres (independent form)."""
    if n < 3:
        return np.ones(n)
    t = (np.arange(n) + 0.5) / n
    a = 1.0 if alpha is None else alpha
    if a <= 0:
        return np.ones(n)
    out = np.empty(n)
    for i, ti in enumerate(t):
        if ti < a / 2:
            out[i] = 0.5 * (1 - np.cos(2 * np.pi * ti / a))
        elif ti > 1 - a / 2:
            out[i] = 0.5 * (1 - np.cos(2 * np.pi * (1 - ti) / a))
        else:
            out[i] = 1.0
    return out


@pytest.mark.parametrize("size", [(8, 8), (16, 12), (5, 7), (3, 32)])
@pytest.mark.parametrize("alpha", [None, 0.0, 0.25, 0.5, 0.75, 1.0])
def test_hann_tukey_convention(size: tuple[int, int], alpha: float | None) -> None:
    """Both tapers are periodic windows sampled at pixel centres, outer-producted."""
    geom = spatial.geometry.Rectangular(size=size)
    window = (
        spatial.window.Hann() if alpha is None else spatial.window.Tukey(alpha=alpha)
    )
    expected = np.outer(_ref(size[0], alpha), _ref(size[1], alpha))
    np.testing.assert_allclose(window.weights(geom), expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_tukey_cola_at_its_hop(alpha: float) -> None:
    # Overlap-add at hop n (1 - alpha / 2) sums to 1 exactly in the interior.
    n = 16
    hop = int(n * (1 - alpha / 2))
    w = spatial.window.Tukey(alpha=alpha).weights(
        spatial.geometry.Rectangular(size=(1, n))
    )[0]
    total = np.zeros(4 * n)
    for start in range(0, 3 * n, hop):
        total[start : start + n] += w
    np.testing.assert_allclose(total[n : 2 * n], 1.0, rtol=0, atol=1e-15)


@pytest.mark.parametrize("size", [(8, 8), (16, 12), (5, 7)])
def test_tukey_alpha_one_is_hann(size: tuple[int, int]) -> None:
    geom = spatial.geometry.Rectangular(size=size)
    np.testing.assert_array_equal(
        spatial.window.Tukey(alpha=1.0).weights(geom),
        spatial.window.Hann().weights(geom),
    )


@pytest.mark.parametrize(
    "window", [spatial.window.Hann(), spatial.window.Tukey(alpha=0.5)]
)
@pytest.mark.parametrize("n", [1, 2])
def test_short_axis_falls_back_to_boxcar(window, n: int) -> None:
    # A 1- or 2-sample axis cannot carry a taper (np.hanning(2) was all
    # zeros); it is boxcar, while the long axis keeps its taper.
    w = window.weights(spatial.geometry.Rectangular(size=(n, 8)))
    assert w.shape == (n, 8)
    long_axis = _ref(8, None if isinstance(window, spatial.window.Hann) else 0.5)
    np.testing.assert_allclose(w, np.tile(long_axis, (n, 1)), rtol=0, atol=1e-15)
    assert (w.sum(axis=1) > 0).all()


class TestSpatialTukey:
    def test_alpha_zero_is_boxcar(self, geom: spatial.geometry.Rectangular) -> None:
        w = spatial.window.Tukey(alpha=0.0).weights(geom)
        np.testing.assert_allclose(w, 1.0)


class TestSpatialGaussian:
    def test_peak_centre(self, geom: spatial.geometry.Rectangular) -> None:
        w = spatial.window.Gaussian(sigma=0.5).weights(geom)
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
        w = spatial.window.Gaussian(sigma=sigma).weights(
            spatial.geometry.Rectangular(size=(n, n))
        )
        np.testing.assert_allclose(w, np.outer(ref, ref), rtol=1e-14, atol=0)


class TestSpatialCustom:
    def test_calls_user_fn(self, geom: spatial.geometry.Rectangular) -> None:
        called: list[bool] = []

        def fn(g):
            called.append(True)
            return np.full(g.size, 0.5)

        w = spatial.window.Custom(fn=fn).weights(geom)
        assert called == [True]
        np.testing.assert_array_equal(w, 0.5)


# ---------------------------------------------------------------------------
# Merged constant field: interior seams vs the leading border ring
# ---------------------------------------------------------------------------

_C = 3.25


def _constant_field(n: int) -> RasterField:
    return RasterField(
        GeoArray(
            data=np.full((n, n), _C, dtype=np.float64),
            transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
            crs="EPSG:32630",
            dims="y x",
        )
    )


def _merge_identity(patcher: SpatialPatcher, field: RasterField) -> np.ndarray:
    patches = [
        type(p)(
            data=np.asarray(p.data),
            anchor=p.anchor,
            indices=p.indices,
            weights=p.weights,
        )
        for p in patcher.split(field)
    ]
    return np.asarray(patcher.merge(patches, field.domain))


def test_overlap_add_constant_field_border() -> None:
    """Hann at hop N/2 on a tileable 40x40 field (16/8, drop).

    Every pixel — interior seams and the scene's outer edge alike — is the
    constant exactly: no taper sample is zero, so even a pixel only one chip
    covers keeps a non-zero weight.
    """
    field = _constant_field(40)
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(16, 16)),
        sampler=spatial.sampler.RegularStride(step=(8, 8)),
        window=spatial.window.Hann(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    out = _merge_identity(patcher, field)
    assert out.shape == (40, 40)
    np.testing.assert_allclose(out, _C, rtol=1e-15, atol=0)

    # True COLA: without normalisation the doubly-covered interior is
    # already C (sum of shifted Hann == 1); only the outer band of width
    # `step`, covered by a single chip, carries the bare taper.
    raw_patcher = SpatialPatcher(
        geometry=patcher.geometry,
        sampler=patcher.sampler,
        window=patcher.window,
        aggregation=spatial.aggregation.OverlapAdd(normalize_by_window=False),
    )
    raw = _merge_identity(raw_patcher, field)
    np.testing.assert_allclose(raw[8:32, 8:32], _C, rtol=1e-15, atol=0)
    w1 = _ref(16, None)
    np.testing.assert_allclose(raw[0:8, 20], _C * w1[0:8], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(raw[32:40, 20], _C * w1[8:16], rtol=1e-14, atol=0)


def test_overlap_add_hann_step_equals_size_tiles_exactly() -> None:
    """``step == size`` with Hann: every pixel is covered by one chip with a
    non-zero weight, so the normalised merge reproduces the field."""
    field = _constant_field(32)
    patcher = SpatialPatcher(
        geometry=spatial.geometry.Rectangular(size=(16, 16)),
        sampler=spatial.sampler.RegularStride(step=(16, 16)),
        window=spatial.window.Hann(),
        aggregation=spatial.aggregation.OverlapAdd(),
    )
    out = _merge_identity(patcher, field)
    np.testing.assert_allclose(out, _C, rtol=1e-15, atol=0)
