"""The ``.sklearn`` accessor on DataArray, Dataset and DataTree.

Importing `xrops.learn` registers it. Each verb builds an `XarrayEstimator`
(or uses a fitted one) and forwards, so there is no second marshalling path.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import xarray as xr
from pipekit_array.estimator import NanPolicy

from xrops.learn._src.estimator import TreeMode, XarrayEstimator


class _SklearnAccessor:
    """``obj.sklearn``: scikit-learn verbs on this xarray object."""

    def __init__(self, obj: xr.DataArray | xr.Dataset | xr.DataTree) -> None:
        self._obj = obj

    def fit(
        self,
        estimator: Any,
        y: Any = None,
        *,
        sample_dims: Hashable | Sequence[Hashable],
        feature_dim: str | None = None,
        nan_policy: NanPolicy = "propagate",
        tree_mode: TreeMode = "per_node",
    ) -> XarrayEstimator:
        """Fit ``estimator`` on this object and return the fitted wrapper.

        Args:
            estimator: An unfitted scikit-learn-style estimator.
            y: Targets, as for `XarrayEstimator.fit`.
            sample_dims: The dim (or dims) whose entries are samples.
            feature_dim: The name of a new output dim (``"component"``).
            nan_policy: As for `XarrayEstimator`.
            tree_mode: As for `XarrayEstimator`.

        Returns:
            The fitted `XarrayEstimator`; keep it to transform other data.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import xarray as xr
            >>> from sklearn.decomposition import PCA
            >>> import xrops.learn  # registers .sklearn
            >>> rng = np.random.default_rng(0)
            >>> da = xr.DataArray(
            ...     rng.normal(size=(12, 2, 3)), dims=("time", "lat", "lon")
            ... )
            >>> est = da.sklearn.fit(PCA(2), sample_dims="time", feature_dim="mode")
            >>> da.sklearn.transform(est).dims
            ('time', 'mode')

            ```
        """
        return XarrayEstimator(
            estimator,
            sample_dims=sample_dims,
            feature_dim=feature_dim,
            nan_policy=nan_policy,
            tree_mode=tree_mode,
        ).fit(self._obj, y)

    def fit_transform(
        self,
        estimator: Any,
        y: Any = None,
        *,
        sample_dims: Hashable | Sequence[Hashable],
        feature_dim: str | None = None,
        nan_policy: NanPolicy = "propagate",
        tree_mode: TreeMode = "per_node",
    ) -> Any:
        """Fit ``estimator`` on this object and return its transform.

        Args:
            estimator: An unfitted scikit-learn-style estimator.
            y: Targets, as for `XarrayEstimator.fit`.
            sample_dims: The dim (or dims) whose entries are samples.
            feature_dim: The name of a new output dim (``"component"``).
            nan_policy: As for `XarrayEstimator`.
            tree_mode: As for `XarrayEstimator`.

        Returns:
            The transformed data, as for `XarrayEstimator.transform`.

        Example:
            ```pycon
            >>> import numpy as np
            >>> import xarray as xr
            >>> from sklearn.preprocessing import StandardScaler
            >>> import xrops.learn  # registers .sklearn
            >>> ds = xr.Dataset({
            ...     "u": (("time", "x"), np.arange(6.0).reshape(3, 2)),
            ...     "v": ("time", [1.0, 5.0, 2.0]),
            ... })
            >>> out = ds.sklearn.fit_transform(StandardScaler(), sample_dims="time")
            >>> {name: out[name].dims for name in out.data_vars}
            {'u': ('time', 'x'), 'v': ('time',)}

            ```
        """
        est = self.fit(
            estimator,
            y,
            sample_dims=sample_dims,
            feature_dim=feature_dim,
            nan_policy=nan_policy,
            tree_mode=tree_mode,
        )
        return est.transform(self._obj)

    def transform(self, fitted: XarrayEstimator) -> Any:
        """``fitted.transform`` of this object."""
        return _fitted(fitted).transform(self._obj)

    def inverse_transform(self, fitted: XarrayEstimator) -> Any:
        """``fitted.inverse_transform`` of this object."""
        return _fitted(fitted).inverse_transform(self._obj)

    def predict(self, fitted: XarrayEstimator) -> Any:
        """``fitted.predict`` of this object."""
        return _fitted(fitted).predict(self._obj)

    def predict_proba(self, fitted: XarrayEstimator) -> Any:
        """``fitted.predict_proba`` of this object."""
        return _fitted(fitted).predict_proba(self._obj)


def _fitted(est: Any) -> XarrayEstimator:
    if not isinstance(est, XarrayEstimator):
        raise TypeError(
            "pass the XarrayEstimator returned by .sklearn.fit(...); it carries the "
            f"fit-time layout. Got {type(est).__name__}."
        )
    return est


@xr.register_dataarray_accessor("sklearn")
class SklearnDataArrayAccessor(_SklearnAccessor):
    """``DataArray.sklearn``: scikit-learn verbs through `XarrayEstimator`."""


@xr.register_dataset_accessor("sklearn")
class SklearnDatasetAccessor(_SklearnAccessor):
    """``Dataset.sklearn``: the variables side by side as one feature table."""


@xr.register_datatree_accessor("sklearn")
class SklearnDataTreeAccessor(_SklearnAccessor):
    """``DataTree.sklearn``: per node or pooled over nodes (``tree_mode``)."""


__all__ = [
    "SklearnDataArrayAccessor",
    "SklearnDataTreeAccessor",
    "SklearnDatasetAccessor",
]
