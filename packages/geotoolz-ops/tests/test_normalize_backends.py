"""The value-wise normalizers on every carrier (#533).

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
pytest.importorskip("dask")

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
