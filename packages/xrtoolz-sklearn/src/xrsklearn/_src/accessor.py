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

from collections.abc import Hashable, Sequence
from typing import Any

import xarray as xr
from sklearn.base import clone

from xrsklearn._src.nan import MissingKind, NanPolicy
from xrsklearn._src.tree import TreeMode
from xrsklearn._src.wrap import (
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

    ``sample_dim`` / ``new_feature_dim`` / ``nan_policy`` / ``missing`` configure the
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

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset | xr.DataTree) -> None:
        self._obj = xarray_obj

    def _wrap(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
        missing: MissingKind | None = None,
        tree_mode: TreeMode | None = None,
        tree_paths: Sequence[str] | None = None,
    ) -> XarrayEstimator:
        """An *unfitted* wrapper to fit — a fresh clone if given a wrapper."""
        overrides = _explicit(
            sample_dim, new_feature_dim, nan_policy, missing, tree_mode, tree_paths
        )
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
        missing: MissingKind | None = None,
        tree_mode: TreeMode | None = None,
    ) -> XarrayEstimator:
        """A wrapper around an already-fitted estimator."""
        overrides = _explicit(
            sample_dim, new_feature_dim, nan_policy, missing, tree_mode
        )
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
        missing: MissingKind | None = None,
        tree_mode: TreeMode | None = None,
        tree_paths: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> XarrayEstimator:
        """Fit ``estimator`` on this object and return the fitted wrapper.

        Args:
            estimator: An unfitted sklearn estimator, or an
                :class:`XarrayEstimator` (a reconfigured clone of it is fitted;
                the original is left untouched).
            y: Optional target, as for :meth:`XarrayEstimator.fit`.
            **kwargs: Extra arguments for the estimator's ``fit``.
            sample_dim: Sample dimension of the wrapper (see
                :class:`XarrayEstimator`); ``None`` keeps the wrapper's setting
                or infers it.
            new_feature_dim: Name of a reduced output's feature dim.
            nan_policy: Missing-value policy (see :class:`XarrayEstimator`).
            missing: What counts as missing (``"nan"`` / ``"nonfinite"``).
            tree_mode: What fitting on a DataTree means (see
                :class:`XarrayEstimator`).
            tree_paths: DataTree nodes to fit on; by default every node with
                data variables.

        Returns:
            The fitted :class:`XarrayEstimator` — keep it to transform other data.

        Raises:
            ValueError: If the data does not suit the wrapper settings (e.g. a
                missing ``sample_dim``).

        Example:
            ```pycon
            >>> import numpy as np
            >>> import xarray as xr
            >>> import xrsklearn  # registers the .sklearn accessors
            >>> rng = np.random.default_rng(0)
            >>> da = xr.DataArray(
            ...     rng.normal(size=(12, 2, 3)), dims=("time", "lat", "lon")
            ... )
            >>> from sklearn.decomposition import PCA
            >>> wrap = da.sklearn.fit(PCA(n_components=2), sample_dim="time")
            >>> wrap.components_.shape
            (2, 6)

            ```
        """
        return self._wrap(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
            missing=missing,
            tree_mode=tree_mode,
            tree_paths=tree_paths,
        ).fit(self._obj, y=y, **kwargs)

    def fit_transform(
        self,
        estimator: Any,
        y: xr.DataArray | xr.Dataset | Any | None = None,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
        missing: MissingKind | None = None,
        tree_mode: TreeMode | None = None,
        tree_paths: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | Any:
        """Fit ``estimator`` on this object and return the transformed data.

        Args:
            estimator: An unfitted sklearn estimator or :class:`XarrayEstimator`.
            y: Optional target, as for :meth:`XarrayEstimator.fit`.
            **kwargs: Extra arguments for the estimator's ``fit``.
            sample_dim: Sample dimension of the wrapper (see
                :class:`XarrayEstimator`); ``None`` keeps the wrapper's setting
                or infers it.
            new_feature_dim: Name of a reduced output's feature dim.
            nan_policy: Missing-value policy (see :class:`XarrayEstimator`).
            missing: What counts as missing (``"nan"`` / ``"nonfinite"``).
            tree_mode: What fitting on a DataTree means (see
                :class:`XarrayEstimator`).
            tree_paths: DataTree nodes to fit on; by default every node with
                data variables.

        Returns:
            The transformed data (layout as in :meth:`XarrayEstimator.transform`).

        Raises:
            ValueError: If the data does not suit the wrapper settings.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import xarray as xr
            >>> import xrsklearn  # registers the .sklearn accessors
            >>> rng = np.random.default_rng(0)
            >>> da = xr.DataArray(
            ...     rng.normal(size=(12, 2, 3)), dims=("time", "lat", "lon")
            ... )
            >>> from sklearn.preprocessing import StandardScaler
            >>> scaled = da.sklearn.fit_transform(StandardScaler(), sample_dim="time")
            >>> scaled.dims, bool(abs(scaled.mean("time")).max() < 1e-12)
            (('time', 'lat', 'lon'), True)

            ```
        """
        return self._wrap(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
            missing=missing,
            tree_mode=tree_mode,
            tree_paths=tree_paths,
        ).fit_transform(self._obj, y=y, **kwargs)

    def transform(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
        missing: MissingKind | None = None,
        tree_mode: TreeMode | None = None,
    ) -> xr.DataArray | Any:
        """Transform this object with a fitted estimator.

        Args:
            estimator: A fitted sklearn estimator, or a fitted
                :class:`XarrayEstimator` (which also validates this object
                against its fit-time grid).
            sample_dim: Sample dimension of the wrapper (see
                :class:`XarrayEstimator`); ``None`` keeps the wrapper's setting
                or infers it.
            new_feature_dim: Name of a reduced output's feature dim.
            nan_policy: Missing-value policy (see :class:`XarrayEstimator`).
            missing: What counts as missing (``"nan"`` / ``"nonfinite"``).
            tree_mode: What fitting on a DataTree means (see
                :class:`XarrayEstimator`).

        Returns:
            The transformed data.

        Raises:
            ValueError: If a setting disagrees with a fitted
                :class:`XarrayEstimator`'s own, or the input does not match the
                fit-time layout.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import xarray as xr
            >>> import xrsklearn  # registers the .sklearn accessors
            >>> rng = np.random.default_rng(0)
            >>> da = xr.DataArray(
            ...     rng.normal(size=(12, 2, 3)), dims=("time", "lat", "lon")
            ... )
            >>> from sklearn.decomposition import PCA
            >>> from xrsklearn import XarrayEstimator
            >>> wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(da)
            >>> da.sklearn.transform(wrap).dims
            ('time', 'component')

            ```
        """
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
            missing=missing,
            tree_mode=tree_mode,
        ).transform(self._obj)

    def inverse_transform(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
        missing: MissingKind | None = None,
        tree_mode: TreeMode | None = None,
    ) -> xr.DataArray | Any:
        """Map this object back to a fitted estimator's input space.

        Args:
            estimator: A fitted :class:`XarrayEstimator` (restores the
                fit-time grid) or a fitted raw estimator (generic
                ``(sample_dim, new_feature_dim)`` output).
            sample_dim: Sample dimension of the wrapper (see
                :class:`XarrayEstimator`); ``None`` keeps the wrapper's setting
                or infers it.
            new_feature_dim: Name of a reduced output's feature dim.
            nan_policy: Missing-value policy (see :class:`XarrayEstimator`).
            missing: What counts as missing (``"nan"`` / ``"nonfinite"``).
            tree_mode: What fitting on a DataTree means (see
                :class:`XarrayEstimator`).

        Returns:
            The reconstruction.

        Raises:
            ValueError: If a setting disagrees with a fitted
                :class:`XarrayEstimator`'s own, or the input does not match the
                fit-time layout.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import xarray as xr
            >>> import xrsklearn  # registers the .sklearn accessors
            >>> rng = np.random.default_rng(0)
            >>> da = xr.DataArray(
            ...     rng.normal(size=(12, 2, 3)), dims=("time", "lat", "lon")
            ... )
            >>> from sklearn.decomposition import PCA
            >>> from xrsklearn import XarrayEstimator
            >>> wrap = XarrayEstimator(PCA(n_components=2), sample_dim="time").fit(da)
            >>> scores = da.sklearn.transform(wrap)
            >>> scores.sklearn.inverse_transform(wrap).dims
            ('time', 'lat', 'lon')

            ```
        """
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
            missing=missing,
            tree_mode=tree_mode,
        ).inverse_transform(self._obj)

    def predict(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
        missing: MissingKind | None = None,
        tree_mode: TreeMode | None = None,
    ) -> xr.DataArray | Any:
        """Predict from this object with a fitted estimator.

        Args:
            estimator: A fitted sklearn estimator or :class:`XarrayEstimator`.
            sample_dim: Sample dimension of the wrapper (see
                :class:`XarrayEstimator`); ``None`` keeps the wrapper's setting
                or infers it.
            new_feature_dim: Name of a reduced output's feature dim.
            nan_policy: Missing-value policy (see :class:`XarrayEstimator`).
            missing: What counts as missing (``"nan"`` / ``"nonfinite"``).
            tree_mode: What fitting on a DataTree means (see
                :class:`XarrayEstimator`).

        Returns:
            The predictions (layout as in :meth:`XarrayEstimator.predict`).

        Raises:
            ValueError: If a setting disagrees with a fitted
                :class:`XarrayEstimator`'s own, or the input does not match the
                fit-time layout.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import xarray as xr
            >>> import xrsklearn  # registers the .sklearn accessors
            >>> rng = np.random.default_rng(0)
            >>> da = xr.DataArray(
            ...     rng.normal(size=(12, 2, 3)), dims=("time", "lat", "lon")
            ... )
            >>> from sklearn.cluster import KMeans
            >>> kmeans = KMeans(n_clusters=2, n_init=1, random_state=0)
            >>> wrap = da.sklearn.fit(kmeans, sample_dim="time")
            >>> da.sklearn.predict(wrap).dims
            ('time',)

            ```
        """
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
            missing=missing,
            tree_mode=tree_mode,
        ).predict(self._obj)

    def predict_proba(
        self,
        estimator: Any,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
        missing: MissingKind | None = None,
        tree_mode: TreeMode | None = None,
    ) -> xr.DataArray | Any:
        """Predict class probabilities for this object with a fitted classifier.

        Args:
            estimator: A fitted sklearn classifier or :class:`XarrayEstimator`.
            sample_dim: Sample dimension of the wrapper (see
                :class:`XarrayEstimator`); ``None`` keeps the wrapper's setting
                or infers it.
            new_feature_dim: Name of a reduced output's feature dim.
            nan_policy: Missing-value policy (see :class:`XarrayEstimator`).
            missing: What counts as missing (``"nan"`` / ``"nonfinite"``).
            tree_mode: What fitting on a DataTree means (see
                :class:`XarrayEstimator`).

        Returns:
            ``(sample_dim, "class")`` probabilities labeled by ``classes_``.

        Raises:
            ValueError: If a setting disagrees with a fitted
                :class:`XarrayEstimator`'s own, or the input does not match the
                fit-time layout.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import xarray as xr
            >>> import xrsklearn  # registers the .sklearn accessors
            >>> rng = np.random.default_rng(0)
            >>> da = xr.DataArray(
            ...     rng.normal(size=(12, 2, 3)), dims=("time", "lat", "lon")
            ... )
            >>> from sklearn.linear_model import LogisticRegression
            >>> labels = xr.where(da.mean(("lat", "lon")) > 0, "warm", "cold")
            >>> wrap = da.sklearn.fit(LogisticRegression(), labels, sample_dim="time")
            >>> da.sklearn.predict_proba(wrap)["class"].values.tolist()
            ['cold', 'warm']

            ```
        """
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
            missing=missing,
            tree_mode=tree_mode,
        ).predict_proba(self._obj)

    def score(
        self,
        estimator: Any,
        y: xr.DataArray | xr.Dataset | Any | None = None,
        *,
        sample_dim: Hashable | None = None,
        new_feature_dim: str | None = None,
        nan_policy: NanPolicy | None = None,
        missing: MissingKind | None = None,
        tree_mode: TreeMode | None = None,
    ) -> float:
        """Score a fitted estimator on this object.

        Args:
            estimator: A fitted sklearn estimator or :class:`XarrayEstimator`.
            y: Target, as for :meth:`XarrayEstimator.score`.
            sample_dim: Sample dimension of the wrapper (see
                :class:`XarrayEstimator`); ``None`` keeps the wrapper's setting
                or infers it.
            new_feature_dim: Name of a reduced output's feature dim.
            nan_policy: Missing-value policy (see :class:`XarrayEstimator`).
            missing: What counts as missing (``"nan"`` / ``"nonfinite"``).
            tree_mode: What fitting on a DataTree means (see
                :class:`XarrayEstimator`).

        Returns:
            The score (``{path: score}`` for a per-node DataTree fit).

        Raises:
            ValueError: If a setting disagrees with a fitted
                :class:`XarrayEstimator`'s own, or the input does not match the
                fit-time layout.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import xarray as xr
            >>> import xrsklearn  # registers the .sklearn accessors
            >>> rng = np.random.default_rng(0)
            >>> da = xr.DataArray(
            ...     rng.normal(size=(12, 2, 3)), dims=("time", "lat", "lon")
            ... )
            >>> from sklearn.linear_model import LinearRegression
            >>> y = da.mean(("lat", "lon"))
            >>> wrap = da.sklearn.fit(LinearRegression(), y, sample_dim="time")
            >>> round(da.sklearn.score(wrap, y), 6)
            1.0

            ```
        """
        return self._wrap_fitted(
            estimator,
            sample_dim=sample_dim,
            new_feature_dim=new_feature_dim,
            nan_policy=nan_policy,
            missing=missing,
            tree_mode=tree_mode,
        ).score(self._obj, y=y)


@xr.register_dataarray_accessor("sklearn")
class SklearnDataArrayAccessor(_SklearnAccessor):
    """``DataArray.sklearn`` adapter for :class:`XarrayEstimator`.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from sklearn.preprocessing import StandardScaler
        >>> import xrsklearn  # registers the .sklearn accessors
        >>> da = xr.DataArray(np.arange(6.0).reshape(3, 2), dims=("time", "x"))
        >>> da.sklearn.fit_transform(StandardScaler(), sample_dim="time").dims
        ('time', 'x')

        ```
    """


@xr.register_dataset_accessor("sklearn")
class SklearnDatasetAccessor(_SklearnAccessor):
    """``Dataset.sklearn`` adapter for :class:`XarrayEstimator`.

    Data variables are column-concatenated into one feature matrix; a
    one-to-one transformer hands back a Dataset on the same grids.

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from sklearn.preprocessing import StandardScaler
        >>> import xrsklearn  # registers the .sklearn accessors
        >>> ds = xr.Dataset({
        ...     "u": (("time", "x"), np.arange(6.0).reshape(3, 2)),
        ...     "v": ("time", [1.0, 5.0, 2.0]),
        ... })
        >>> out = ds.sklearn.fit_transform(StandardScaler(), sample_dim="time")
        >>> {name: out[name].dims for name in out.data_vars}
        {'u': ('time', 'x'), 'v': ('time',)}

        ```
    """


@xr.register_datatree_accessor("sklearn")
class SklearnDataTreeAccessor(_SklearnAccessor):
    """``DataTree.sklearn`` adapter for :class:`XarrayEstimator`.

    ``tree_mode`` picks what fitting on a tree means (``"per_node"``,
    ``"pool_samples"`` or ``"concat_features"``; see
    :class:`XarrayEstimator`).

    Example:
        ```pycon
        >>> import numpy as np
        >>> import xarray as xr
        >>> from sklearn.preprocessing import StandardScaler
        >>> import xrsklearn  # registers the .sklearn accessors
        >>> rng = np.random.default_rng(0)
        >>> tree = xr.DataTree.from_dict({
        ...     "coarse": xr.Dataset({"ssh": (("time", "x"), rng.normal(size=(6, 2)))}),
        ...     "fine": xr.Dataset({"ssh": (("time", "x"), rng.normal(size=(6, 4)))}),
        ... })
        >>> scaled = tree.sklearn.fit_transform(StandardScaler(), sample_dim="time")
        >>> scaled["fine"]["ssh"].dims, scaled["fine"]["ssh"].shape
        (('time', 'x'), (6, 4))

        ```
    """
