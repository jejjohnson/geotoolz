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


# --- per-scene stretches: ZeroOne and HistogramStretch (#591) --------------


def _old_stretch(g: GeoArray, op: object) -> np.ndarray:
    """The NumPy-only implementation before #591, as the reference."""
    from geoops.normalize._src.array import stat_axes
    from geoops.normalize._src.operators import _masked

    arr, _ = _masked(g)
    if isinstance(op, gn.ZeroOne):
        axis = stat_axes(arr, per_band=op.per_band)
        lo = np.nanmin(arr, axis=axis, keepdims=True)
        hi = np.nanmax(arr, axis=axis, keepdims=True)
        return (arr - lo) / np.where(hi > lo, hi - lo, 1.0)
    axis = stat_axes(arr)
    lo = np.nanpercentile(arr, op.lower, axis=axis, keepdims=True)
    hi = np.nanpercentile(arr, op.upper, axis=axis, keepdims=True)
    unit = np.clip((arr - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0, 1.0)
    out_min, out_max = op.out_range
    return unit * (out_max - out_min) + out_min


STRETCHES = [
    pytest.param(gn.ZeroOne(), id="ZeroOne"),
    pytest.param(gn.ZeroOne(per_band=False), id="ZeroOne-global"),
    pytest.param(gn.HistogramStretch(), id="HistogramStretch"),
    pytest.param(
        gn.HistogramStretch(out_range=(0.0, 255.0), lower=5.0, upper=90.0),
        id="HistogramStretch-255",
    ),
]


@pytest.mark.parametrize("op", STRETCHES)
def test_stretch_agrees_on_every_carrier(op: object, stats_scene: GeoArray) -> None:
    want = _old_stretch(stats_scene, op)
    got = op(stats_scene)
    assert isinstance(got, GeoArray)
    assert np.isnan(got.fill_value)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, equal_nan=True)

    jitted = jax.jit(lambda g: op(g))(stats_scene.to_jax())
    da = stats_scene.to_xarray()
    on_dask = op(da.chunk({"y": 2}))
    assert jitted.backend == "jax"
    assert hasattr(on_dask.data, "dask")
    for out in (op(stats_scene.to_jax()), jitted, op(da), on_dask.compute()):
        np.testing.assert_allclose(np.asarray(out), want, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("op", STRETCHES)
def test_stretch_maps_a_constant_band_to_its_low_end(op: object) -> None:
    values = np.stack([np.full((4, 4), 0.3), np.linspace(0, 1, 16).reshape(4, 4)])
    out = np.asarray(op(toy_geoarray(values)))
    low = op.out_range[0] if isinstance(op, gn.HistogramStretch) else 0.0
    if getattr(op, "per_band", True):
        np.testing.assert_allclose(out[0], low)
    np.testing.assert_allclose(out, _old_stretch(toy_geoarray(values), op))


# --- fitted scalers: StandardScaler, RobustScaler, MinMaxScaler (#592) -----

SCALERS = {
    "StandardScaler": (gn.StandardScaler, ("mean_", "std_")),
    "RobustScaler": (gn.RobustScaler, ("median_", "iqr_")),
    "MinMaxScaler": (gn.MinMaxScaler, ("vmin_", "vmax_")),
}


def _old_fit(name: str, g: GeoArray) -> tuple[np.ndarray, np.ndarray]:
    """The NumPy-only fit before #592, as the reference."""
    from geoops.normalize._src.array import stat_axes
    from geoops.normalize._src.operators import _masked

    arr, _ = _masked(g)
    axis = stat_axes(arr)
    if name == "StandardScaler":
        return np.nanmean(arr, axis=axis), np.nanstd(arr, axis=axis)
    if name == "RobustScaler":
        q25, q50, q75 = np.nanpercentile(arr, [25.0, 50.0, 75.0], axis=axis)
        return q50, q75 - q25
    return np.nanmin(arr, axis=axis), np.nanmax(arr, axis=axis)


@pytest.mark.parametrize("name", SCALERS)
def test_scaler_fit_stores_host_arrays_on_every_carrier(
    name: str, stats_scene: GeoArray
) -> None:
    cls, attrs = SCALERS[name]
    want = _old_fit(name, stats_scene)
    da = stats_scene.to_xarray()
    for carrier in (stats_scene, stats_scene.to_jax(), da, da.chunk({"y": 2})):
        fitted = cls().fit(carrier)
        for attr, value in zip(attrs, want, strict=True):
            got = getattr(fitted, attr)
            assert type(got) is np.ndarray
            np.testing.assert_allclose(got, value, rtol=1e-12)


@pytest.mark.parametrize("name", SCALERS)
def test_fitted_scaler_transforms_and_inverts_on_every_carrier(
    name: str, stats_scene: GeoArray
) -> None:
    import pickle

    cls, _ = SCALERS[name]
    scaler = pickle.loads(pickle.dumps(cls().fit(stats_scene)))
    want = np.asarray(scaler(stats_scene))
    da = stats_scene.to_xarray()
    lazy = scaler(da.chunk({"y": 2}))
    assert hasattr(lazy.data, "dask")
    jitted = jax.jit(lambda g: scaler(g))(stats_scene.to_jax())
    for got in (scaler(stats_scene.to_jax()), jitted, scaler(da), lazy.compute()):
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, equal_nan=True)

    back = jax.jit(lambda g: scaler.inverse(g))(scaler(stats_scene.to_jax()))
    lazy_back = scaler.inverse(lazy)
    assert hasattr(lazy_back.data, "dask")
    valid = np.isfinite(want)
    original = np.asarray(stats_scene, dtype=float)
    for got in (scaler.inverse(scaler(stats_scene)), back, lazy_back.compute()):
        np.testing.assert_allclose(np.asarray(got)[valid], original[valid], rtol=1e-9)
        assert np.isnan(np.asarray(got)[~valid]).all()


@pytest.mark.parametrize("name", SCALERS)
def test_scaler_fit_refuses_a_jax_trace(name: str, scene: GeoArray) -> None:
    cls, attrs = SCALERS[name]
    with pytest.raises(TypeError, match=rf"{name}.fit cannot run under jax.jit"):
        jax.jit(lambda g: cls().fit(g).transform(g))(scene.to_jax())
    lazy = cls(fit_on_call=True)
    with pytest.raises(TypeError, match=r"Call .*fit\(x\) eagerly"):
        jax.jit(lambda g: lazy(g))(scene.to_jax())
    assert all(getattr(lazy, attr) is None for attr in attrs)  # nothing written
    # fitted eagerly, the same operator then runs inside the trace
    lazy(scene.to_jax())
    jax.jit(lambda g: lazy(g))(scene.to_jax())


def test_scaler_statistics_must_match_the_band_dim_on_dask(scene: GeoArray) -> None:
    lazy = scene.to_xarray().chunk({"y": 3})
    with pytest.raises(ValueError, match="does not match the kept axes"):
        gn.RobustScaler(median=[0.0, 0.0], iqr=[1.0, 1.0])(lazy)
    with pytest.raises(ValueError, match="does not match the kept axes"):
        gn.MinMaxScaler(vmin=[0.0, 0.0], vmax=[1.0, 1.0]).inverse(lazy)


# --- PerBandStats (#593) -----------------------------------------------------


def test_per_band_stats_agree_on_every_carrier(stats_scene: GeoArray) -> None:
    from geoops.normalize._src.array import stat_axes
    from geoops.normalize._src.operators import _masked

    arr, _ = _masked(stats_scene)
    want = gn.per_band_stats(arr, percentiles=(2.0, 50.0), reduce_axes=stat_axes(arr))
    da = stats_scene.to_xarray()
    for carrier in (stats_scene, stats_scene.to_jax(), da, da.chunk({"y": 2})):
        op = gn.PerBandStats(percentiles=[2.0, 50.0])
        assert op(carrier) is carrier
        assert set(op.stats) == set(want)
        for key, value in want.items():
            assert isinstance(op.stats[key], list)
            np.testing.assert_allclose(op.stats[key], value, rtol=1e-12)


def test_per_band_stats_compute_a_dask_cube_once(scene: GeoArray) -> None:
    computes = 0

    def counting(dsk: object, keys: object, **kwargs: object) -> object:
        nonlocal computes
        computes += 1
        return dask.get(dsk, keys, **kwargs)

    lazy = scene.to_xarray().chunk({"y": 3})
    op = gn.PerBandStats()
    with dask.config.set(scheduler=counting):
        out = op(lazy)
    assert computes == 1
    assert out is lazy
    assert hasattr(out.data, "dask")


def test_per_band_stats_refuse_a_jax_trace(scene: GeoArray) -> None:
    op = gn.PerBandStats()
    with pytest.raises(TypeError, match=r"PerBandStats cannot run under jax\.jit"):
        jax.jit(lambda g: op(g))(scene.to_jax())
    assert op.stats == {}
    # the functional form runs inside the trace
    stats = jax.jit(lambda g: gn.carrier_band_stats(g))(scene.to_jax())
    np.testing.assert_allclose(stats["mean"], gn.carrier_band_stats(scene)["mean"])
