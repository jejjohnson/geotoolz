"""Tests for the xrtoolz.einx linalg conveniences."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import xrtoolz.einx as xnx
from xrtoolz.einx import PatternError


def test_matmul_contracts_shared_dim() -> None:
    a = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("time", "k"))
    b = xr.DataArray(np.arange(10.0).reshape(2, 5), dims=("k", "mode"))
    out = xnx.matmul(a, b, dim="k")
    assert out.dims == ("time", "mode")
    np.testing.assert_allclose(out.values, a.values @ b.values)


def test_matmul_requires_shared_dim() -> None:
    a = xr.DataArray(np.ones((3, 2)), dims=("time", "k"))
    b = xr.DataArray(np.ones((2, 5)), dims=("j", "mode"))
    with pytest.raises(PatternError, match="present on both"):
        xnx.matmul(a, b, dim="k")


def test_matmul_rejects_overlapping_free_dims() -> None:
    a = xr.DataArray(np.ones((3, 2)), dims=("time", "k"))
    b = xr.DataArray(np.ones((2, 3)), dims=("k", "time"))
    with pytest.raises(PatternError, match="disjoint"):
        xnx.matmul(a, b, dim="k")


def test_outer_product() -> None:
    a = xr.DataArray(np.arange(2.0), dims="lat")
    b = xr.DataArray(np.arange(3.0), dims="lon")
    out = xnx.outer(a, b)
    assert out.dims == ("lat", "lon")
    np.testing.assert_allclose(out.values, np.outer(a.values, b.values))


def test_outer_rejects_shared_dims() -> None:
    a = xr.DataArray(np.arange(2.0), dims="lat")
    b = xr.DataArray(np.arange(2.0), dims="lat")
    with pytest.raises(PatternError, match="disjoint"):
        xnx.outer(a, b)


def test_batch_matmul_broadcasts_over_batch_dim() -> None:
    a = xr.DataArray(np.arange(2 * 3 * 2.0).reshape(2, 3, 2), dims=("ens", "time", "k"))
    b = xr.DataArray(np.arange(2 * 2 * 5.0).reshape(2, 2, 5), dims=("ens", "k", "mode"))
    out = xnx.batch_matmul(a, b, dim="k", batch_dims=["ens"])
    assert out.dims == ("ens", "time", "mode")
    expected = np.einsum("eik,ekj->eij", a.values, b.values)
    np.testing.assert_allclose(out.values, expected)
