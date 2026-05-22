"""Operator wrapper for sklearn-style estimators."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Literal

import numpy as np
import xarray as xr

from xrtoolz._operator import Operator
from xrtoolz.utils import XarrayEstimator
from xrtoolz.utils._src.sklearn_wrap import NanPolicy


SklearnMethod = Literal[
    "fit_transform",
    "transform",
    "predict",
    "predict_proba",
    "inverse_transform",
]
_MAX_JSON_RECURSION_DEPTH = 10


class SklearnOp(Operator):
    """Layer-1 operator that delegates sklearn marshalling to ``XarrayEstimator``.

    Drops a fitted (or fittable) sklearn-style estimator into a
    :class:`pipekit.Sequential` chain. The op marshals the input via
    :class:`XarrayEstimator` (stack → delegate → unstack), so the underlying
    estimator only ever sees a 2-D numpy array.

    Behaviour by ``method``:

    - ``"transform"`` (default), ``"predict"``, ``"predict_proba"`` —
      ``estimator`` must already be fitted. Pass either a fitted sklearn
      estimator or a fitted :class:`XarrayEstimator`.
    - ``"inverse_transform"`` — also requires fitted state, **and** requires
      ``estimator`` to be a fitted :class:`XarrayEstimator` if you want the
      original ``(sample_dim, *feature_dims)`` grid back. The raw-estimator
      path produces a generic ``(sample_dim, new_feature_dim)`` layout.
    - ``"fit_transform"`` — fits a fresh clone on every call. Avoid inside
      :class:`Sequential` (re-fitting on every pipeline invocation is rarely
      what you want); use it for one-shot fit-and-transform ops.

    ``get_config()`` stores primitive estimator parameters directly and
    serialises non-primitive parameters as ``repr(...)`` strings, so the
    config is JSON-safe (works with ``Sequential.get_config()``).

    Args:
        estimator: A raw sklearn estimator (``BaseEstimator``) or an
            :class:`XarrayEstimator`. For methods other than
            ``"fit_transform"`` it must already be fitted.
        variable: Dataset variable to operate on. Required when input is a
            multi-variable :class:`xarray.Dataset` (or any Dataset where the
            whole-Dataset stacked path is not desired).
        output_variable: Name of the variable to assign the result to.
            Defaults to ``variable``. When input is a Dataset and neither
            ``variable`` nor ``output_variable`` is set, ``__call__`` raises
            :class:`ValueError` (the result would be a DataArray, breaking
            :class:`Sequential` chains).
        sample_dim: xarray dim indexing samples. Defaults to the first dim.
        new_feature_dim: Name for the feature dim when the estimator changes
            feature count (e.g. PCA component count).
        nan_policy: ``"propagate"`` (default), ``"raise"``, or ``"mask"``. See
            :class:`XarrayEstimator` for the masking semantics on Dataset
            input.
        method: Which sklearn-style method to call on each invocation.

    Example:
        Compose a fitted scaler + a fitted PCA in a Sequential::

            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            from pipekit import Sequential
            from xrtoolz.transforms import SklearnOp
            from xrtoolz.utils import XarrayEstimator

            # Fit upstream, then drop the fitted estimators into the chain.
            scaler = XarrayEstimator(StandardScaler(), sample_dim="time").fit(ds["ssh"])
            pca = XarrayEstimator(PCA(n_components=10), sample_dim="time").fit(
                scaler.transform(ds["ssh"])
            )

            pipeline = Sequential([
                SklearnOp(scaler, variable="ssh"),
                SklearnOp(pca, variable="ssh", output_variable="pcs"),
            ])
            ds_out = pipeline(ds)            # ds_out has both "ssh" and "pcs"

        One-shot fit-and-transform of a single var (no Sequential reuse)::

            op = SklearnOp(StandardScaler(), variable="ssh", sample_dim="time",
                           method="fit_transform")
            ds_scaled = op(ds)

        Recover the original feature grid via ``inverse_transform``::

            pca = XarrayEstimator(PCA(n_components=3), sample_dim="time").fit(ssh)
            inv = SklearnOp(pca, method="inverse_transform")
            # → (time, lat, lon), not (time, component)
            ssh_recon = inv(pca.transform(ssh))

        NaN-laden ocean grid with the masking policy::

            op = SklearnOp(StandardScaler(), variable="ssh", sample_dim="time",
                           method="fit_transform", nan_policy="mask")
            # Land/NaN rows are dropped pre-fit, then re-inserted as NaN on output.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        variable: Hashable | None = None,
        output_variable: Hashable | None = None,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
        method: SklearnMethod = "transform",
    ) -> None:
        self.estimator = estimator
        self.variable = variable
        self.output_variable = output_variable
        self.sample_dim = sample_dim
        self.new_feature_dim = new_feature_dim
        self.nan_policy = nan_policy
        self.method = method

    def _apply(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        if (
            isinstance(data, xr.Dataset)
            and self.variable is None
            and self.output_variable is None
        ):
            raise ValueError(
                "SklearnOp received a Dataset input but neither `variable` nor "
                "`output_variable` is set. The whole-Dataset path produces a "
                "DataArray, which would break a Sequential chain. Pass "
                "`variable=...` to select a single variable, or "
                "`output_variable=...` to name the result when stacking all "
                "variables together."
            )
        target = (
            data[self.variable]
            if isinstance(data, xr.Dataset) and self.variable is not None
            else data
        )
        out = self._run(target)
        if not isinstance(data, xr.Dataset):
            return out

        name = self.output_variable or self.variable
        return data.assign({name: out})

    def _run(self, data: xr.DataArray | xr.Dataset) -> xr.DataArray:
        wrap = self._resolve_wrap()
        method = getattr(wrap, self.method)
        return method(data)

    def _resolve_wrap(self) -> XarrayEstimator:
        """Return an ``XarrayEstimator`` appropriate for ``self.method``.

        Pre-fitted ``XarrayEstimator`` instances are reused as-is so their
        ``_fitted_meta_`` is preserved (needed for ``inverse_transform`` to
        recover the original feature grid). Raw sklearn estimators are wrapped
        on each call; for ``method="fit_transform"`` the wrapper fits a fresh
        clone, otherwise the estimator is treated as already fitted.
        """
        if isinstance(self.estimator, XarrayEstimator):
            return self.estimator
        wrap = XarrayEstimator(
            self.estimator,
            sample_dim=self.sample_dim,
            new_feature_dim=self.new_feature_dim,
            nan_policy=self.nan_policy,
        )
        if self.method != "fit_transform":
            wrap.estimator_ = self.estimator
        return wrap

    def get_config(self) -> dict[str, Any]:
        return {
            "estimator": self._estimator_class_name(),
            "estimator_params": self._estimator_params(),
            "variable": self.variable,
            "output_variable": self.output_variable,
            "sample_dim": self.sample_dim,
            "new_feature_dim": self.new_feature_dim,
            "nan_policy": self.nan_policy,
            "method": self.method,
        }

    def _estimator_class_name(self) -> str:
        if isinstance(self.estimator, XarrayEstimator):
            return self.estimator.estimator.__class__.__name__
        return self.estimator.__class__.__name__

    def _estimator_params(self) -> dict[str, Any]:
        # XarrayEstimator wraps the underlying sklearn estimator; pull params
        # off the wrapped instance so the config reflects what sklearn sees.
        target = (
            self.estimator.estimator
            if isinstance(self.estimator, XarrayEstimator)
            else self.estimator
        )
        if not hasattr(target, "get_params"):
            return {}
        return {
            key: _json_safe(value)
            for key, value in target.get_params(deep=False).items()
        }


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    if _depth > _MAX_JSON_RECURSION_DEPTH:
        return repr(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        # numpy scalar (np.int64, np.float64, np.bool_, ...) → native Python.
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(v, _depth=_depth + 1) for v in value]
    if isinstance(value, list):
        return [_json_safe(v, _depth=_depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth=_depth + 1) for k, v in value.items()}
    return repr(value)
