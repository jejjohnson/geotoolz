"""xarray accessors for the sklearn bridge.

Registers the ``da.sklearn`` and ``ds.sklearn`` accessors at import time
so users can run sklearn-style verbs directly on xarray objects without
constructing an :class:`XarrayEstimator` by hand. Each accessor method
constructs an ``XarrayEstimator`` under the hood and delegates — there
is no parallel marshalling path. Importing :mod:`xrsklearn` is
enough to register the accessors.

Example:
    ```pycon
    >>> import numpy as np
    >>> import xarray as xr
    >>> from sklearn.preprocessing import StandardScaler
    >>> import xrsklearn  # registers the .sklearn accessor
    >>> ssh = xr.DataArray(np.arange(12.0).reshape(4, 3), dims=("time", "space"))
    >>> scaled = ssh.sklearn.fit_transform(StandardScaler(), sample_dim="time")
    >>> scaled.dims, scaled.shape
    (('time', 'space'), (4, 3))

    ```
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import xarray as xr
from sklearn.base import clone

from xrsklearn._src.wrap import (
    NanPolicy,
    XarrayEstimator,
    _check_no_conflict,
    _explicit,
)


class _SklearnAccessor:
    """Thin xarray accessor that delegates to :class:`XarrayEstimator`.

    Provides ``fit``, ``fit_transform``, ``transform``, ``inverse_transform``,
    ``predict``, ``predict_proba``, and ``score`` directly on
    :class:`xarray.DataArray` / :class:`xarray.Dataset` objects via the
    registered ``.sklearn`` namespace. The accessor never re-implements
    the stack→delegate→unstack marshalling — every method constructs an
    :class:`XarrayEstimator` and forwards.

    Methods that need a *fitted* estimator (``transform``,
    ``inverse_transform``, ``predict``, ``predict_proba``, ``score``)
    accept either a fitted raw sklearn estimator or a fitted
    :class:`XarrayEstimator`. **For ``inverse_transform`` to recover the
    original feature grid, you must pass a fitted ``XarrayEstimator``** —
    only the wrapper carries the fit-time layout needed to rebuild
    ``(sample_dim, *feature_dims)``. A raw sklearn estimator goes
    through the shortcut path and produces a generic
    ``(sample_dim, component)`` layout instead.

    ``sample_dim`` / ``new_feature_dim`` / ``nan_policy`` configure the
    wrapper built around a raw estimator. With an ``XarrayEstimator``
    they may be omitted (the wrapper's own settings apply); ``fit`` /
    ``fit_transform`` fit a reconfigured *clone* of it, and the fitted-
    estimator methods raise ``ValueError`` if an explicit value disagrees
    with the fitted wrapper's.

    Example:
        Fit once, reuse via the accessor — a fitted ``XarrayEstimator``
        is forwarded as-is, so ``inverse_transform`` can rebuild the
        original ``(time, lat, lon)`` grid:

        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from sklearn.decomposition import PCA
        >>> from xrsklearn import XarrayEstimator
        >>> rng = np.random.default_rng(0)
        >>> da = xr.DataArray(
        ...     rng.normal(size=(8, 3, 4)), dims=("time", "lat", "lon"),
        ... )
        >>> wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(da)
        >>> scores = da.sklearn.transform(wrap)
        >>> scores.dims
        ('time', 'component')
        >>> recon = scores.sklearn.inverse_transform(wrap)
        >>> recon.dims
        ('time', 'lat', 'lon')

        ```

        Fit-and-transform directly off the DataArray:

        ```pycon
        >>> scores = da.sklearn.fit_transform(
        ...     PCA(n_components=3),
        ...     sample_dim="time",
        ...     nan_policy="mask",
        ... )
        >>> scores.shape
        (8, 3)

        ```

        Score a fitted estimator, and the Dataset variant (data_vars are
        column-concatenated into one ``(sample, feature)`` matrix; a
        one-to-one transformer hands back a Dataset on the same grid):

        ```pycon
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.preprocessing import StandardScaler
        >>> X = xr.DataArray(rng.normal(size=(8, 3)), dims=("time", "feat"))
        >>> y = X.values @ np.array([1.0, 2.0, 3.0])
        >>> fitted = XarrayEstimator(LinearRegression(), sample_dim="time").fit(X, y)
        >>> round(X.sklearn.score(fitted, y, sample_dim="time"), 6)
        1.0
        >>> ds = xr.Dataset({"u": da, "v": da})
        >>> scaled = ds.sklearn.fit_transform(StandardScaler(), sample_dim="time")
        >>> list(scaled.data_vars), scaled["u"].dims
        (['u', 'v'], ('time', 'lat', 'lon'))

        ```
    """

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset) -> None:
        self._obj = xarray_obj

    def _wrap(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
    ) -> XarrayEstimator:
        """An *unfitted* wrapper to fit — a fresh clone if given a wrapper."""
        overrides = _explicit(sample_dim, new_feature_dim, nan_policy)
        if isinstance(estimator, XarrayEstimator):
            # Fit a configured clone: never mutate the caller's wrapper, and
            # never nest an XarrayEstimator inside another.
            return clone(estimator).set_params(**overrides)
        return XarrayEstimator(estimator, **overrides)

    def _wrap_fitted(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
    ) -> XarrayEstimator:
        """A wrapper around an already-fitted estimator."""
        overrides = _explicit(sample_dim, new_feature_dim, nan_policy)
        # Pre-fitted XarrayEstimator: pass it through. Re-wrapping would
        # construct a fresh wrapper without its fit-time layout, so
        # `inverse_transform` would fall into the generic
        # `(sample_dim, new_feature_dim)` layout instead of restoring the
        # original feature grid. Settings are fixed at fit time, so an
        # explicit override that disagrees with the wrapper is an error
        # rather than something to drop silently.
        if isinstance(estimator, XarrayEstimator):
            _check_no_conflict(estimator, overrides)
            return estimator
        wrap = XarrayEstimator(estimator, **overrides)
        wrap.estimator_ = estimator
        return wrap

    def fit(
        self,
        estimator: Any,
        y: xr.DataArray | xr.Dataset | Any | None = None,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
        **kwargs: Any,
    ) -> XarrayEstimator:
        """Fit ``estimator`` on this xarray object via ``XarrayEstimator``."""
        return self._wrap(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).fit(self._obj, y=y, **kwargs)

    def fit_transform(
        self,
        estimator: Any,
        y: xr.DataArray | xr.Dataset | Any | None = None,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | Any:
        """Fit and transform this xarray object via ``XarrayEstimator``."""
        return self._wrap(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).fit_transform(self._obj, y=y, **kwargs)

    def transform(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
    ) -> xr.DataArray | Any:
        """Transform this xarray object with a fitted sklearn estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).transform(self._obj)

    def inverse_transform(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
    ) -> xr.DataArray | Any:
        """Inverse-transform this xarray object with a fitted estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).inverse_transform(self._obj)

    def predict(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
    ) -> xr.DataArray | Any:
        """Predict from this xarray object with a fitted estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).predict(self._obj)

    def predict_proba(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
    ) -> xr.DataArray | Any:
        """Predict class probabilities with a fitted estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).predict_proba(self._obj)

    def score(
        self,
        estimator: Any,
        y: xr.DataArray | xr.Dataset | Any | None = None,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
    ) -> float:
        """Score this xarray object with a fitted estimator."""
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
        ).score(self._obj, y=y)


@xr.register_dataarray_accessor("sklearn")
class SklearnDataArrayAccessor(_SklearnAccessor):
    """``DataArray.sklearn`` adapter for :class:`XarrayEstimator`."""


@xr.register_dataset_accessor("sklearn")
class SklearnDatasetAccessor(_SklearnAccessor):
    """``Dataset.sklearn`` adapter for :class:`XarrayEstimator`."""
