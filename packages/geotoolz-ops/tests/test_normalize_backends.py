"""The normalizers and their statistics on every carrier (#533, #590).

`LogScale`, `AsinhScale`, `PowerScale` and fixed-statistics `Normalize`
run one kernel through pipekit-array's ``KernelOp`` with the band dim as
its core dim, so a pixel with any invalid band is ``NaN`` in every band,
as on the NumPy path, on a NumPy `GeoArray`, a JAX one (eagerly and under
``jax.jit``), its ``to_xarray()`` (eagerly and dask-backed) and a plain
array.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import toy_geoarray
from geoarray import GeoArray

import geoops.normalize as gn


jax = pytest.importorskip("jax")
xr = pytest.importorskip("xarray")
dask = pytest.importorskip("dask")

OPS = [
    pytest.param(gn.LogScale(base=2.0), id="LogScale"),
    pytest.param(gn.AsinhScale(a=0.3), id="AsinhScale"),
    pytest.param(gn.PowerScale(gamma=0.7), id="PowerScale"),
    pytest.param(
        gn.Normalize(mean=[0.1, 0.2, 0.3], std=[0.5, 0.0, 2.0]), id="Normalize"
    ),
    pytest.param(gn.Normalize(mean=0.2, std=0.4), id="Normalize-scalar"),
]


@pytest.fixture(autouse=True)
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def scene() -> GeoArray:
    """Three bands, 6 x 5, with two fill pixels and one NaN in one band."""
    rng = np.random.default_rng(9)
    values = rng.uniform(-0.2, 1.0, size=(3, 6, 5)).astype(np.float32)
    values[1, 2, 3] = np.nan
    return toy_geoarray(values, with_fill_pixels=True)


@pytest.mark.parametrize("op", OPS)
def test_normalizer_agrees_on_every_carrier(op: object, scene: GeoArray) -> None:
    want = op(scene)
    assert isinstance(want, GeoArray)
    assert want.dims == scene.dims
    assert np.isnan(want.fill_value)
    values = np.asarray(want)
    assert np.isnan(values[:, 2, 3]).all()  # one bad band: the whole pixel
    assert np.isnan(values[:, 0, 0]).all()  # a fill pixel

    jitted = jax.jit(lambda g: op(g))(scene.to_jax())
    da = scene.to_xarray()
    on_dask = op(da.chunk({"y": 3}))
    assert jitted.backend == "jax"
    assert hasattr(on_dask.data, "dask")
    for got in (op(scene.to_jax()), jitted, op(da), on_dask.compute()):
        np.testing.assert_allclose(np.asarray(got), values, rtol=1e-6, equal_nan=True)
    # a plain array declares no fill: only its fill pixels differ
    plain = op(np.asarray(scene))
    finite = np.isfinite(values)
    np.testing.assert_allclose(plain[finite], values[finite], rtol=1e-6)


def test_bad_parameters_fail_before_any_compute(scene: GeoArray) -> None:
    lazy = scene.to_xarray().chunk({"y": 3})
    with pytest.raises(ValueError, match="base must be positive"):
        gn.LogScale(base=1.0)(lazy)
    with pytest.raises(ValueError, match="does not match the kept axes"):
        gn.Normalize(mean=[0.0, 0.0], std=[1.0, 1.0])(lazy)


def test_fitting_scalers_stay_numpy_only(scene: GeoArray) -> None:
    from pipekit import BackendError

    with pytest.raises(BackendError, match="StandardScaler runs on numpy only"):
        gn.StandardScaler(fit_on_call=True)(scene.to_jax())


# --- mask-aware per-band statistics (#590) ---------------------------------


def _numpy_reference(g: GeoArray, q: tuple[float, ...]) -> dict[str, np.ndarray]:
    """Today's NumPy fit path: masked float64 copy, ``stat_axes``, ``np.nan*``."""
    from geoops.normalize._src.array import stat_axes
    from geoops.normalize._src.operators import _masked

    arr, _ = _masked(g)
    axis = stat_axes(arr)
    return {
        "mean": np.nanmean(arr, axis=axis),
        "std": np.nanstd(arr, axis=axis),
        "min": np.nanmin(arr, axis=axis),
        "max": np.nanmax(arr, axis=axis),
        "quantiles": np.nanpercentile(arr, [100 * level for level in q], axis=axis),
    }


@pytest.fixture(params=["cube", "stack"])
def stats_scene(request: pytest.FixtureRequest, scene: GeoArray) -> GeoArray:
    if request.param == "cube":
        return scene
    rng = np.random.default_rng(3)
    values = rng.uniform(0.0, 2.0, size=(2, 3, 6, 5)).astype(np.float32)
    values[1, 0, 4, 1] = np.nan  # one band of one frame: that pixel only
    return toy_geoarray(values, with_fill_pixels=True)


def test_band_stats_agree_on_every_carrier(stats_scene: GeoArray) -> None:
    q = (0.1, 0.5, 0.9)
    want = _numpy_reference(stats_scene, q)
    n_bands = stats_scene.shape[-3]
    da = stats_scene.to_xarray()
    lazy = gn.carrier_band_stats(da.chunk({"y": 2, "x": 2}), quantiles=q)
    assert all(hasattr(value, "dask") for value in lazy.values())
    jitted = jax.jit(lambda g: gn.carrier_band_stats(g, quantiles=q))(
        stats_scene.to_jax()
    )
    for got in (
        gn.carrier_band_stats(stats_scene, quantiles=q),
        gn.carrier_band_stats(stats_scene.to_jax(), quantiles=q),
        jitted,
        gn.carrier_band_stats(da, quantiles=q),
        dict(zip(lazy, dask.compute(*lazy.values()), strict=True)),
    ):
        assert set(got) == set(want)
        assert got["mean"].shape == (n_bands,)
        assert got["quantiles"].shape == (len(q), n_bands)
        for key, value in want.items():
            np.testing.assert_allclose(np.asarray(got[key]), value, rtol=1e-12)


def test_band_stats_on_a_plain_array_judge_only_non_finite(scene: GeoArray) -> None:
    # a plain array declares no fill: its fill pixels count as data
    values = np.asarray(scene, dtype=float)
    got = gn.carrier_band_stats(values, stats=("max",))
    pixel_ok = np.isfinite(values).all(axis=0)
    np.testing.assert_allclose(got["max"], [b[pixel_ok].max() for b in values])


def test_band_stats_global_and_without_a_band_dim(scene: GeoArray) -> None:
    arr = np.where(scene.valid_mask, np.asarray(scene, dtype=float), np.nan)
    pooled = gn.carrier_band_stats(scene, per_band=False, stats=("min", "max"))
    assert np.shape(pooled["min"]) == ()
    np.testing.assert_allclose(pooled["max"], np.nanmax(arr))
    band = scene.isel(band=1)  # (y, x): judged element by element
    values = np.asarray(band, dtype=float)
    want = np.nanmean(np.where(values != band.fill_value, values, np.nan))
    single = gn.carrier_band_stats(band, stats=("mean",))
    np.testing.assert_allclose(single["mean"], want)


def test_band_stats_l1_takes_the_band_axis_by_position() -> None:
    x = np.arange(24.0).reshape(2, 3, 4)  # band last
    got = gn.band_stats(x, band_axis=-1, stats=("mean",))
    np.testing.assert_allclose(got["mean"], x.mean(axis=(0, 1)))
    q = gn.band_quantiles(x, x < 20, band_axis=-1, q=(0.5,))
    np.testing.assert_allclose(q[0], np.nanmedian(np.where(x < 20, x, np.nan), (0, 1)))


def test_band_stats_leave_infinities_out() -> None:
    x = np.array([[[1.0, np.inf]], [[2.0, -np.inf]]])  # (C=2, H=1, W=2)
    for valid in (None, np.ones(x.shape, dtype=bool)):
        got = gn.band_stats(x, valid)
        np.testing.assert_array_equal(got["mean"], [1.0, 2.0])
        np.testing.assert_array_equal(got["std"], [0.0, 0.0])
        np.testing.assert_array_equal(gn.band_quantiles(x, valid, q=(1.0,))[0], [1, 2])


def test_band_stats_reject_bad_requests(scene: GeoArray) -> None:
    with pytest.raises(ValueError, match="unknown statistics"):
        gn.carrier_band_stats(scene, stats=("median",))
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        gn.carrier_band_stats(scene, quantiles=(25.0,))
