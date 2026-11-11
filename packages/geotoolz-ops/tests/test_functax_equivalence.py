"""Equivalence pins for the gridders and gap fills that run on functax plans (#645).

Each ``_legacy_*`` function is the implementation at ``1e7308b``, copied
verbatim so the comparison outlives the code it pins: the KD-tree IDW of
``points_to_raster_idw`` / ``raster_to_point_cloud``, the per-pixel
``query_ball_point`` loop of ``gap_fill_idw``, the EDT ``gap_fill_nearest`` and
the Jacobi ``gap_fill_laplacian``. The tolerances are the ones the issue
states (``atol=1e-9`` in float64) unless a test says otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from scipy import ndimage
from scipy.spatial import KDTree, cKDTree

from geoops.geom._src.coregister.array import (
    _pixel_center_xy,
    points_to_raster_idw,
    raster_to_point_cloud,
)
from geoops.restore import gap_fill_idw, gap_fill_nearest


pytest.importorskip("functax")

_TRANSFORM = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0)


# -- the implementations at 1e7308b ---------------------------------------------------


def _legacy_points_to_raster_idw(
    xy, values, *, dst_shape, dst_transform, k=8, power=2.0, max_radius=None
):
    xy = np.asarray(xy, dtype=np.float64)
    values = np.asarray(values)
    h, w = dst_shape
    if xy.shape[0] == 0:
        return np.full((h, w), np.nan, dtype=np.float64)
    tree = KDTree(xy)
    eff_k = min(k, xy.shape[0])
    distances, indices = tree.query(_pixel_center_xy(dst_transform, (h, w)), k=eff_k)
    if eff_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    sampled = values[indices]
    weights = 1.0 / np.maximum(distances**power, 1e-12)
    num = (sampled * weights).sum(axis=1)
    den = weights.sum(axis=1)
    result = (num / den).reshape(h, w).astype(np.float64)
    if max_radius is not None:
        nearest_dist = distances[:, 0].reshape(h, w)
        result = np.where(nearest_dist > max_radius, np.nan, result)
    return result


def _legacy_raster_to_point_cloud(arr, transform, xy, *, k, max_radius, method, power):
    arr = np.asarray(arr)
    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    tree = KDTree(_pixel_center_xy(transform, arr.shape))
    eff_k = min(k, tree.n)
    distances, indices = tree.query(xy, k=eff_k)
    if eff_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    flat = arr.reshape(*arr.shape[:-2], -1)
    gathered = flat[..., indices]
    if method == "nearest" or eff_k == 1:
        result = gathered[..., 0].astype(np.float64)
    else:
        weights = 1.0 / np.maximum(distances**power, 1e-12)
        num = (gathered * weights).sum(axis=-1)
        den = weights.sum(axis=-1)
        result = (num / den).astype(np.float64)
    if max_radius is not None:
        result[..., distances[:, 0] > max_radius] = np.nan
    return result


def _legacy_gap_fill_idw(plane, *, power, radius):
    out = plane.copy()
    missing = ~np.isfinite(plane)
    valid = np.argwhere(np.isfinite(plane))
    tree = cKDTree(valid)
    for row, col in np.argwhere(missing):
        neighbours = tree.query_ball_point([row, col], r=radius)
        if not neighbours:
            continue
        coords = valid[neighbours]
        dist = np.linalg.norm(coords - np.array([row, col]), axis=1)
        weights = 1.0 / np.maximum(dist, 1e-12) ** power
        out[row, col] = np.sum(weights * plane[coords[:, 0], coords[:, 1]]) / np.sum(
            weights
        )
    return out


def _legacy_gap_fill_nearest(plane, *, max_distance=None):
    mask = ~np.isfinite(plane)
    distances, nearest = ndimage.distance_transform_edt(
        mask, return_distances=True, return_indices=True
    )
    filled = plane[tuple(nearest)]
    if max_distance is not None:
        filled = np.where(distances <= max_distance, filled, np.nan)
    return np.where(mask, filled, plane)


def _legacy_gap_fill_laplacian(values, *, iterations):
    out = np.stack(
        [_legacy_gap_fill_nearest(p) for p in values.reshape(-1, *values.shape[-2:])]
    ).reshape(values.shape)
    missing = ~np.isfinite(values)
    pad_width = [(0, 0)] * (out.ndim - 2) + [(1, 1), (1, 1)]
    for _ in range(iterations):
        padded = np.pad(out, pad_width, mode="edge")
        avg = (
            padded[..., :-2, 1:-1]
            + padded[..., 2:, 1:-1]
            + padded[..., 1:-1, :-2]
            + padded[..., 1:-1, 2:]
        ) / 4.0
        out = np.where(missing, avg, values)
    return out


# -- inputs ---------------------------------------------------------------------------


def _cloud(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A random cloud over the 12 × 16 grid of ``_TRANSFORM``, off every centre."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(500_000.0, 500_160.0, n)
    y = rng.uniform(3_999_880.0, 4_000_000.0, n)
    return np.column_stack([x, y]), rng.normal(size=n)


def _gappy(shape: tuple[int, ...] = (24, 24), frac: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=shape)
    arr[rng.uniform(size=shape) < frac] = np.nan
    return arr


# -- the pins -------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 4, 8])
@pytest.mark.parametrize("power", [1.0, 2.0, 3.5])
def test_points_to_raster_idw_matches_legacy(k: int, power: float) -> None:
    xy, values = _cloud()
    kwargs = {
        "dst_shape": (12, 16),
        "dst_transform": _TRANSFORM,
        "k": k,
        "power": power,
    }
    np.testing.assert_allclose(
        points_to_raster_idw(xy, values, **kwargs),
        _legacy_points_to_raster_idw(xy, values, **kwargs),
        rtol=0,
        atol=1e-9,
    )


def test_points_to_raster_idw_max_radius_matches_legacy() -> None:
    xy, values = _cloud(n=12)
    kwargs = {"dst_shape": (12, 16), "dst_transform": _TRANSFORM, "max_radius": 25.0}
    got = points_to_raster_idw(xy, values, **kwargs)
    want = _legacy_points_to_raster_idw(xy, values, **kwargs)
    assert np.isnan(want).any() and np.isfinite(want).any()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    ("method", "k"), [("nearest", 1), ("idw", 1), ("idw", 4), ("idw", 9)]
)
@pytest.mark.parametrize("max_radius", [None, 6.0])
def test_raster_to_point_cloud_matches_legacy(
    method: str, k: int, max_radius: float | None
) -> None:
    rng = np.random.default_rng(1)
    raster = rng.normal(size=(3, 12, 16))
    xy, _ = _cloud(n=50, seed=2)
    kwargs = {"k": k, "max_radius": max_radius, "method": method, "power": 2.0}
    got = raster_to_point_cloud(raster, _TRANSFORM, xy, **kwargs)
    want = _legacy_raster_to_point_cloud(raster, _TRANSFORM, xy, **kwargs)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


@pytest.mark.parametrize("radius", [1, 2, 5])
@pytest.mark.parametrize("power", [1.0, 2.0, 3.0])
def test_gap_fill_idw_matches_legacy(radius: int, power: float) -> None:
    arr = _gappy(frac=0.5)
    got = gap_fill_idw(arr, power=power, radius=radius)
    want = _legacy_gap_fill_idw(arr, power=power, radius=radius)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


def test_gap_fill_idw_matches_legacy_per_plane() -> None:
    arr = _gappy(shape=(2, 3, 16, 16), seed=3)
    got = gap_fill_idw(arr, power=2.0, radius=3)
    for idx in np.ndindex(arr.shape[:-2]):
        want = _legacy_gap_fill_idw(arr[idx], power=2.0, radius=3)
        np.testing.assert_allclose(got[idx], want, rtol=0, atol=1e-9)


@pytest.mark.parametrize("max_distance", [None, 1, 2])
def test_gap_fill_nearest_matches_legacy_where_the_nearest_pixel_is_unique(
    max_distance: int | None,
) -> None:
    """Off ties, the fill is the legacy EDT one; on a tie it is one of the tied."""
    arr = _gappy(frac=0.6, seed=4)
    got = gap_fill_nearest(arr, max_distance=max_distance)
    want = _legacy_gap_fill_nearest(arr, max_distance=max_distance)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    valid = np.argwhere(np.isfinite(arr))
    for row, col in np.argwhere(~np.isfinite(arr)):
        d2 = ((valid - (row, col)) ** 2).sum(axis=1)
        tied = valid[d2 == d2.min()]
        if not np.isfinite(want[row, col]):
            continue
        if len(tied) == 1:
            assert got[row, col] == want[row, col]
        else:
            assert got[row, col] in arr[tied[:, 0], tied[:, 1]]


def test_points_to_raster_idw_matches_a_brute_force_reference() -> None:
    xy, values = _cloud(n=60, seed=5)
    k, power = 5, 2.0
    got = points_to_raster_idw(
        xy, values, dst_shape=(12, 16), dst_transform=_TRANSFORM, k=k, power=power
    )
    centres = _pixel_center_xy(_TRANSFORM, (12, 16))
    d = np.linalg.norm(centres[:, None, :] - xy[None, :, :], axis=-1)
    nearest = np.argsort(d, axis=1)[:, :k]
    dk = np.take_along_axis(d, nearest, axis=1)
    w = dk**-power
    want = (w * values[nearest]).sum(axis=1) / w.sum(axis=1)
    np.testing.assert_allclose(got.ravel(), want, rtol=0, atol=1e-12)


def test_plan_rows_sum_to_one() -> None:
    from functax.geometry import Points
    from functax.interp import InverseDistance

    from geoops._src.scattered import x64

    xy, _ = _cloud(n=80, seed=6)
    with x64("test"):
        plan = InverseDistance(k=8, power=2.0).plan(
            Points(xy), Points(_pixel_center_xy(_TRANSFORM, (12, 16)))
        )
        weights = np.asarray(plan.read.weights)
    assert weights.dtype == np.float64
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_a_target_on_a_source_takes_its_value_exactly() -> None:
    """Today: weight 1 / max(0, 1e-12) beside the others, so the value is only
    close to the source's. After: exactly the source's value."""
    centres = _pixel_center_xy(_TRANSFORM, (12, 16))
    xy, values = _cloud(n=30, seed=7)
    xy[0] = centres[17]
    values[0] = 0.123456789
    kwargs = {"dst_shape": (12, 16), "dst_transform": _TRANSFORM, "k": 8}
    got = points_to_raster_idw(xy, values, **kwargs).ravel()[17]
    legacy = _legacy_points_to_raster_idw(xy, values, **kwargs).ravel()[17]
    assert got == values[0]
    assert legacy != values[0]
    assert abs(legacy - values[0]) < 1e-9


def test_points_to_raster_idw_max_radius_beyond_is_nan() -> None:
    xy = np.array([[500_005.0, 3_999_995.0]])
    got = points_to_raster_idw(
        xy,
        np.array([2.0]),
        dst_shape=(12, 16),
        dst_transform=_TRANSFORM,
        max_radius=15.0,
    )
    centres = _pixel_center_xy(_TRANSFORM, (12, 16))
    far = np.linalg.norm(centres - xy[0], axis=1) > 15.0
    assert np.isnan(got.ravel()[far]).all()
    np.testing.assert_array_equal(got.ravel()[~far], 2.0)


def test_gap_fill_laplacian_matches_converged_legacy() -> None:
    """Scattered gaps: at 1e-7 against the legacy 200 Jacobi sweeps (measured
    4e-9). One 24 × 28 hole: 200 sweeps are 0.03 short of the harmonic
    fill (measured), so the pin is the converged legacy, 20 000 sweeps."""
    from geoops.restore import gap_fill_laplacian

    arr = _gappy(frac=0.3)
    np.testing.assert_allclose(
        gap_fill_laplacian(arr),
        _legacy_gap_fill_laplacian(arr, iterations=200),
        rtol=0,
        atol=1e-7,
    )
    yy, xx = np.mgrid[0:48, 0:48]
    hole = np.sin(yy / 7.0) + np.cos(xx / 5.0)
    hole[12:36, 10:38] = np.nan
    np.testing.assert_allclose(
        gap_fill_laplacian(hole),
        _legacy_gap_fill_laplacian(hole, iterations=20_000),
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize("method", ["laplacian", "biharmonic"])
def test_gap_fills_reproduce_a_linear_ramp(method: str) -> None:
    """Δu = 0 and Δ²u = 0 both hold for a plane: a hole 2 px off the edge is exact.

    Today the biharmonic fill was the Laplace fill smoothed by a σ = 1
    Gaussian, which bends a ramp; after, it is Δ²u = 0 (a different fill on
    purpose: up to 1.7–2.0 apart on the random-gap rasters above, measured).
    """
    from geoops.restore import gap_fill_biharmonic, gap_fill_laplacian

    fill = gap_fill_laplacian if method == "laplacian" else gap_fill_biharmonic
    yy, xx = np.mgrid[0:16, 0:20]
    ramp = 0.5 * yy - 0.25 * xx + 3.0
    arr = ramp.copy()
    arr[4:11, 5:14] = np.nan
    np.testing.assert_allclose(fill(arr), ramp, rtol=0, atol=1e-6)


def test_fill_gaps_is_compiled_once_and_still_names_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One cached jit per process; the extra is checked on every call anyway."""
    import importlib

    from geoops._src import scattered
    from geoops.restore import gap_fill_laplacian

    arr = _gappy(frac=0.2)
    want = gap_fill_laplacian(arr)
    assert scattered._compiled_fill_gaps() is scattered._compiled_fill_gaps()
    np.testing.assert_array_equal(gap_fill_laplacian(arr), want)

    real_import_module = importlib.import_module

    def without_functax(name: str, package: str | None = None):
        if name.split(".")[0] == "functax":
            raise ImportError(f"simulated missing {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", without_functax)
    with pytest.raises(ImportError, match=r"pip install 'geotoolz-ops\[interp\]'"):
        gap_fill_laplacian(arr)
