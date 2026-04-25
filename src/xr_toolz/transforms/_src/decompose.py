"""Linear and matrix-factorisation decompositions.

These are thin presets over :class:`xr_toolz.utils.XarrayEstimator` that
wrap the corresponding :mod:`sklearn` estimators. The wrappers preserve
the sklearn API (``fit / transform / fit_transform / inverse_transform``)
on N-D xarray inputs by stacking non-sample dims into a feature axis,
delegating to sklearn, and unstacking the result.

Naming convention:

- ``new_feature_dim="component"`` for PCA / ICA / NMF / KMeans
- ``new_feature_dim="mode"`` for EOF (Empirical Orthogonal Functions —
  the geophysics dialect of PCA, where the convention is "modes" not
  "components")
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from sklearn.cluster import KMeans as _SkKMeans
from sklearn.decomposition import (
    NMF as _SkNMF,
    PCA as _SkPCA,
    FastICA as _SkICA,
)

from xr_toolz.utils import XarrayEstimator


def pca(
    sample_dim: Hashable | None = None,
    n_components: int | float | str | None = None,
    *,
    whiten: bool = False,
    random_state: int | None = None,
    **kwargs: Any,
) -> XarrayEstimator:
    """Principal Component Analysis on N-D xarray inputs.

    Returns:
        An unfitted :class:`XarrayEstimator` wrapping
        :class:`sklearn.decomposition.PCA`. Call ``.fit(da)`` /
        ``.transform(da)`` / ``.fit_transform(da)`` /
        ``.inverse_transform(scores)`` as you would on the raw sklearn
        object — inputs and outputs are :class:`xr.DataArray`.
    """
    return XarrayEstimator(
        _SkPCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state,
            **kwargs,
        ),
        sample_dim=sample_dim,
        new_feature_dim="component",
    )


def eof(
    sample_dim: Hashable | None = None,
    n_modes: int | None = None,
    *,
    whiten: bool = False,
    random_state: int | None = None,
    **kwargs: Any,
) -> XarrayEstimator:
    """Empirical Orthogonal Functions — geophysical PCA.

    Identical numerics to :func:`pca`, but the new feature axis is
    named ``"mode"`` rather than ``"component"`` to match the
    geosciences convention.
    """
    return XarrayEstimator(
        _SkPCA(
            n_components=n_modes,
            whiten=whiten,
            random_state=random_state,
            **kwargs,
        ),
        sample_dim=sample_dim,
        new_feature_dim="mode",
    )


def ica(
    sample_dim: Hashable | None = None,
    n_components: int | None = None,
    *,
    random_state: int | None = None,
    **kwargs: Any,
) -> XarrayEstimator:
    """Independent Component Analysis (FastICA)."""
    return XarrayEstimator(
        _SkICA(n_components=n_components, random_state=random_state, **kwargs),
        sample_dim=sample_dim,
        new_feature_dim="component",
    )


def nmf(
    sample_dim: Hashable | None = None,
    n_components: int | None = None,
    *,
    init: str | None = None,
    random_state: int | None = None,
    **kwargs: Any,
) -> XarrayEstimator:
    """Non-negative Matrix Factorisation. Input must be non-negative."""
    return XarrayEstimator(
        _SkNMF(
            n_components=n_components,
            init=init,
            random_state=random_state,
            **kwargs,
        ),
        sample_dim=sample_dim,
        new_feature_dim="component",
    )


def kmeans(
    sample_dim: Hashable | None = None,
    n_clusters: int = 8,
    *,
    n_init: int | str = "auto",
    random_state: int | None = None,
    **kwargs: Any,
) -> XarrayEstimator:
    """K-Means clustering. ``predict(da)`` returns 1-D labels along
    ``sample_dim``."""
    return XarrayEstimator(
        _SkKMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
            **kwargs,
        ),
        sample_dim=sample_dim,
        new_feature_dim="component",
    )
