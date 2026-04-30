"""Integration tests for :class:`xr_toolz.inference.JaxModelOp`.

Skipped if ``jax`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


jax = pytest.importorskip("jax")
jnp = jax.numpy


from xr_toolz.inference import JaxModelOp


def _linear_model(w: np.ndarray, b: float):
    """Return a closure that mimics a JAX/equinox-style callable."""

    w_jnp = jnp.asarray(w)

    def fn(x):
        return jnp.dot(x, w_jnp) + b

    return fn


def test_callable_jit(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 3))
    w = np.array([0.5, -1.0, 2.0])
    fn = _linear_model(w, 0.25)

    op = JaxModelOp(fn, jit=True)
    da = xr.DataArray(x, dims=("sample", "feature"))
    out = op(da)

    expected = x @ w + 0.25
    np.testing.assert_allclose(np.asarray(out.values), expected, rtol=1e-5)


def test_callable_no_jit() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 2))
    w = np.array([1.0, -1.0])
    fn = _linear_model(w, 0.0)

    op = JaxModelOp(fn, jit=False)
    out = op(xr.DataArray(x, dims=("sample", "feature")))
    np.testing.assert_allclose(np.asarray(out.values), x @ w, rtol=1e-5)


def test_pytree_object_with_method() -> None:
    """A pytree-style object exposing a named method."""

    class Linear:
        def __init__(self, w, b):
            self.w = jnp.asarray(w)
            self.b = b

        def predict(self, x):
            return jnp.dot(x, self.w) + self.b

    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 2))
    model = Linear(np.array([0.3, -0.7]), 1.5)

    op = JaxModelOp(model, method="predict")
    out = op(xr.DataArray(x, dims=("sample", "feature")))
    expected = x @ np.array([0.3, -0.7]) + 1.5
    np.testing.assert_allclose(np.asarray(out.values), expected, rtol=1e-5)


def test_get_config() -> None:
    op = JaxModelOp(_linear_model(np.zeros(2), 0.0), jit=False)
    cfg = op.get_config()
    assert cfg["model"] == "<model>"
    assert cfg["method"] is None
    assert cfg["jit"] is False
