"""NaN and dask-chunking semantics for the Tier B pixel metrics.

Regression tests for PR #101 review: Tier A kernels must skip NaNs to
match the previous xarray ``skipna=True`` default, and the Tier B
``apply_ufunc`` plumbing must work on dask-backed inputs whose core
(reduce) dim is split across multiple chunks.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.metrics._src import pixel


@pytest.fixture
def ds_with_nans() -> tuple[xr.Dataset, xr.Dataset]:
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((4, 12))
    ref = rng.standard_normal((4, 12))
    pred[0, 3] = np.nan  # one sample missing in pred
    ref[2, 7] = np.nan  # one sample missing in ref
    coords = {"sample": np.arange(4), "time": np.arange(12)}
    ds_p = xr.Dataset({"x": (("sample", "time"), pred)}, coords=coords)
    ds_r = xr.Dataset({"x": (("sample", "time"), ref)}, coords=coords)
    return ds_p, ds_r


@pytest.mark.parametrize(
    "fn",
    [pixel.mse, pixel.rmse, pixel.mae, pixel.bias, pixel.nrmse, pixel.r2_score],
    ids=lambda fn: fn.__name__,
)
def test_metrics_skip_nans(ds_with_nans: tuple[xr.Dataset, xr.Dataset], fn) -> None:
    """A NaN sample shouldn't poison the entire reduction."""
    ds_p, ds_r = ds_with_nans
    out = fn(ds_p, ds_r, "x", "time")
    assert np.isfinite(out.values).all(), (
        f"{fn.__name__} returned NaN despite skipna semantics: {out.values}"
    )


def test_metrics_promote_int_input_to_float() -> None:
    """Integer inputs should not truncate the output dtype."""
    coords = {"time": np.arange(8)}
    ds_p = xr.Dataset({"x": (("time",), np.arange(8, dtype=np.int64))}, coords=coords)
    ds_r = xr.Dataset(
        {"x": (("time",), np.arange(8, dtype=np.int64) + 1)}, coords=coords
    )
    out = pixel.mse(ds_p, ds_r, "x", "time")
    assert np.issubdtype(out.dtype, np.floating)
    np.testing.assert_allclose(out.values, 1.0)


def test_metrics_handle_chunked_core_dim() -> None:
    """Dask-backed input with multi-chunk reduce dim must not error."""
    pytest.importorskip("dask")
    coords = {"sample": np.arange(3), "time": np.arange(12)}
    rng = np.random.default_rng(1)
    pred = rng.standard_normal((3, 12))
    ref = rng.standard_normal((3, 12))
    ds_p = xr.Dataset({"x": (("sample", "time"), pred)}, coords=coords).chunk(
        {"time": 4}
    )
    ds_r = xr.Dataset({"x": (("sample", "time"), ref)}, coords=coords).chunk(
        {"time": 4}
    )
    # Should not raise; values should match the eager computation.
    chunked = pixel.mse(ds_p, ds_r, "x", "time").compute()
    eager_p = xr.Dataset({"x": (("sample", "time"), pred)}, coords=coords)
    eager_r = xr.Dataset({"x": (("sample", "time"), ref)}, coords=coords)
    eager = pixel.mse(eager_p, eager_r, "x", "time")
    np.testing.assert_allclose(chunked.values, eager.values)
