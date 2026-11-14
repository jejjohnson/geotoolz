"""Linear and matrix-factorisation decompositions.

These are thin presets over :class:`xrops.learn.XarrayEstimator` that
wrap the corresponding :mod:`sklearn` estimators. The wrappers preserve
the sklearn API (``fit / transform / fit_transform / inverse_transform``)
on N-D xarray inputs by flattening the non-sample dims into features,
delegating to sklearn, and rebuilding the result. scikit-learn is imported
when a preset is built: install the ``xrtoolz-ops[sklearn]`` extra.

Naming convention:

- ``feature_dim="component"`` for PCA / ICA / NMF / KMeans
- ``feature_dim="mode"`` for EOF (Empirical Orthogonal Functions —
  the geophysics dialect of PCA, where the convention is "modes" not
  "components")
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from xrops.learn import XarrayEstimator
from xrops.utils._src.optional_imports import _require_optional


def _sklearn(module: str, name: str) -> Any:
    mod = _require_optional(
        f"sklearn.{module}", extra="sklearn", feature=name, package="scikit-learn"
    )
    return getattr(mod, name)


def pca(
    sample_dims: Hashable | Sequence[Hashable],
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
        _sklearn("decomposition", "PCA")(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state,
            **kwargs,
        ),
        sample_dims=sample_dims,
        feature_dim="component",
    )


def eof(
    sample_dims: Hashable | Sequence[Hashable],
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
        _sklearn("decomposition", "PCA")(
            n_components=n_modes,
            whiten=whiten,
            random_state=random_state,
            **kwargs,
        ),
        sample_dims=sample_dims,
        feature_dim="mode",
    )


def ica(
    sample_dims: Hashable | Sequence[Hashable],
    n_components: int | None = None,
    *,
    random_state: int | None = None,
    **kwargs: Any,
) -> XarrayEstimator:
    """Independent Component Analysis (FastICA)."""
    return XarrayEstimator(
        _sklearn("decomposition", "FastICA")(
            n_components=n_components, random_state=random_state, **kwargs
        ),
        sample_dims=sample_dims,
        feature_dim="component",
    )


def nmf(
    sample_dims: Hashable | Sequence[Hashable],
    n_components: int | None = None,
    *,
    init: str | None = None,
    random_state: int | None = None,
    **kwargs: Any,
) -> XarrayEstimator:
    """Non-negative Matrix Factorisation. Input must be non-negative."""
    return XarrayEstimator(
        _sklearn("decomposition", "NMF")(
            n_components=n_components,
            init=init,
            random_state=random_state,
            **kwargs,
        ),
        sample_dims=sample_dims,
        feature_dim="component",
    )


def kmeans(
    sample_dims: Hashable | Sequence[Hashable],
    n_clusters: int = 8,
    *,
    n_init: int | str = "auto",
    random_state: int | None = None,
    **kwargs: Any,
) -> XarrayEstimator:
    """K-Means clustering. ``predict(da)`` returns 1-D labels along
    ``sample_dims``."""
    return XarrayEstimator(
        _sklearn("cluster", "KMeans")(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
            **kwargs,
        ),
        sample_dims=sample_dims,
        feature_dim="component",
    )
