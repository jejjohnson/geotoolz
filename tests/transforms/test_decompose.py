"""Tests for :mod:`xr_toolz.transforms._src.decompose`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xr_toolz.transforms import eof, ica, kmeans, nmf, pca


@pytest.fixture
def da_3d() -> xr.DataArray:
    rng = np.random.default_rng(0)
    time = pd.date_range("2020-01-01", periods=24, freq="1ME")
    lat = np.linspace(-10.0, 10.0, 4)
    lon = np.linspace(-20.0, 20.0, 5)
    tt = np.arange(len(time))[:, None, None]
    base = np.sin(2 * np.pi * tt / 12.0) * np.ones((1, len(lat), len(lon)))
    data = base + 0.1 * rng.standard_normal((len(time), len(lat), len(lon)))
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="signal",
    )


def test_pca_fit_transform_returns_components(da_3d):
    est = pca(sample_dim="time", n_components=3)
    scores = est.fit_transform(da_3d)
    assert scores.dims == ("time", "component")
    assert scores.sizes["component"] == 3
    assert est.components_.shape == (3, da_3d.sizes["lat"] * da_3d.sizes["lon"])


def test_pca_inverse_transform_returns_grid(da_3d):
    est = pca(sample_dim="time", n_components=3)
    scores = est.fit_transform(da_3d)
    recon = est.inverse_transform(scores)
    assert set(recon.dims) == {"time", "lat", "lon"}
    assert recon.shape == da_3d.shape


def test_eof_uses_mode_axis_name(da_3d):
    """EOF is just PCA with new_feature_dim='mode' (geophysics convention)."""
    est = eof(sample_dim="time", n_modes=2)
    scores = est.fit_transform(da_3d)
    assert scores.dims == ("time", "mode")
    assert scores.sizes["mode"] == 2


def test_ica_fits_and_transforms(da_3d):
    est = ica(sample_dim="time", n_components=2, random_state=0)
    scores = est.fit_transform(da_3d)
    assert scores.dims == ("time", "component")
    assert scores.sizes["component"] == 2


def test_nmf_requires_nonnegative_input(da_3d):
    """Sanity check that NMF actually runs on a non-negative version."""
    da_pos = da_3d - da_3d.min() + 1e-6
    est = nmf(sample_dim="time", n_components=2, init="random", random_state=0)
    scores = est.fit_transform(da_pos)
    assert scores.dims == ("time", "component")
    assert (scores.values >= 0).all()


def test_kmeans_predicts_labels(da_3d):
    est = kmeans(sample_dim="time", n_clusters=3, n_init=4, random_state=0)
    est.fit(da_3d)
    labels = est.predict(da_3d)
    assert labels.dims == ("time",)
    assert set(np.unique(labels.values)).issubset({0, 1, 2})
