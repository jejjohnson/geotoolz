"""Integration tests for :class:`xr_toolz.inference.SklearnModelOp`.

Skipped if ``scikit-learn`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


pytest.importorskip("sklearn")


from sklearn.linear_model import LinearRegression, LogisticRegression

from xr_toolz.inference import SklearnModelOp


def test_predict_round_trip() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.1
    model = LinearRegression().fit(x, y)

    op = SklearnModelOp(model)
    da = xr.DataArray(x, dims=("sample", "feature"))
    out = op(da)

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("sample",)
    np.testing.assert_allclose(out.values, model.predict(x), rtol=1e-6)


def test_predict_proba_dispatch() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((20, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(x, y)

    op = SklearnModelOp(model, method="predict_proba")
    da = xr.DataArray(x, dims=("sample", "feature"))
    out = op(da)

    assert out.dims == ("sample", "output")
    assert out.shape == (20, 2)
    np.testing.assert_allclose(out.values, model.predict_proba(x), rtol=1e-6)


def test_missing_method_raises_immediately() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((10, 2))
    y = rng.standard_normal(10)
    model = LinearRegression().fit(x, y)
    with pytest.raises(AttributeError, match="predict_proba"):
        SklearnModelOp(model, method="predict_proba")
