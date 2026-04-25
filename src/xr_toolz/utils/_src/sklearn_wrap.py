"""Generic xarray ↔ scikit-learn bridge.

The :class:`XarrayEstimator` wraps any sklearn ``BaseEstimator`` and lets
it operate on N-D :class:`xr.DataArray` / :class:`xr.Dataset` inputs.
The data flow is **stack → delegate → unstack**:

    Phase 1 (stack):    xr.DataArray  →  (n_samples, n_features) numpy
    Phase 2 (delegate): sklearn.fit / .transform / .predict on numpy
    Phase 3 (unstack):  numpy result  →  xr.DataArray (re-gridded)

The wrapper does not monkey-patch sklearn and does not pass xarray
objects into the estimator — sklearn sees a plain 2-D numpy array. All
metadata (dim names, coords, attrs) is captured in a marshalling
``meta`` dict and used to reconstruct the output.

This is the foundation used by :mod:`xr_toolz.transforms.decompose` to
expose PCA / EOF / ICA / NMF / KMeans as thin presets.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.base import BaseEstimator, clone


NanPolicy = Literal["propagate", "raise"]


@dataclass
class _Meta:
    """Marshalling metadata captured during ``_to_2d``."""

    sample_dim: Hashable
    sample_coord: np.ndarray | None
    feature_dims: list[Hashable]
    feature_index: pd.MultiIndex | pd.Index | None
    attrs: dict[str, Any]
    name: Hashable | None
    n_features: int


def _to_2d(da: xr.DataArray, sample_dim: Hashable) -> tuple[np.ndarray, _Meta]:
    """Reshape ``da`` into a ``(n_samples, n_features)`` numpy array.

    All non-sample dimensions are stacked into a single ``__features__``
    MultiIndex. The metadata required to invert the operation — dim
    names, coordinate arrays, attrs, name — is captured in the returned
    :class:`_Meta`.

    1-D inputs (only the sample dim) become ``(n_samples, 1)`` since
    sklearn requires 2-D ``X``.
    """
    if sample_dim not in da.dims:
        raise ValueError(
            f"sample_dim={sample_dim!r} not found on DataArray with dims={da.dims}."
        )

    feature_dims: list[Hashable] = [d for d in da.dims if d != sample_dim]

    if not feature_dims:
        arr = np.asarray(da.values)[:, None]
        sample_coord = (
            np.asarray(da[sample_dim].values) if sample_dim in da.coords else None
        )
        meta = _Meta(
            sample_dim=sample_dim,
            sample_coord=sample_coord,
            feature_dims=[],
            feature_index=None,
            attrs=dict(da.attrs),
            name=da.name,
            n_features=1,
        )
        return arr, meta

    stacked = da.stack(__features__=feature_dims).transpose(sample_dim, "__features__")
    arr = np.asarray(stacked.values)
    feature_index = stacked.indexes["__features__"]
    sample_coord = (
        np.asarray(stacked[sample_dim].values) if sample_dim in stacked.coords else None
    )

    meta = _Meta(
        sample_dim=sample_dim,
        sample_coord=sample_coord,
        feature_dims=list(feature_dims),
        feature_index=feature_index,
        attrs=dict(da.attrs),
        name=da.name,
        n_features=arr.shape[1],
    )
    return arr, meta


def _from_2d(
    arr: np.ndarray,
    meta: _Meta,
    *,
    new_feature_dim: str = "component",
) -> xr.DataArray:
    """Inverse of :func:`_to_2d`. Three reconstruction paths:

    1. **1-D output** (``predict`` returning ``(n_samples,)``): wrap as a
       1-D DataArray indexed by the sample dim.
    2. **Same-feature-count output**: rebuild the stacked MultiIndex and
       ``unstack`` to recover the original N-D grid.
    3. **Changed-feature-count output**: cannot unstack to the original
       grid. Return a 2-D DataArray ``(sample_dim, new_feature_dim)``
       with an integer index along the new feature axis.
    """
    sample_dim = meta.sample_dim
    sample_coord = meta.sample_coord
    sample_coords = {sample_dim: sample_coord} if sample_coord is not None else {}

    if arr.ndim == 1:
        return xr.DataArray(
            arr,
            dims=(sample_dim,),
            coords=sample_coords,
            attrs=dict(meta.attrs),
            name=meta.name,
        )

    if arr.ndim != 2:
        raise ValueError(
            f"Cannot reconstruct DataArray from sklearn output with shape "
            f"{arr.shape}; expected 1-D or 2-D."
        )

    _n_samples, n_features = arr.shape
    same_grid = (
        n_features == meta.n_features
        and meta.feature_index is not None
        and meta.feature_dims
    )
    if same_grid:
        coords: dict[Hashable, Any] = {"__features__": meta.feature_index}
        if sample_coord is not None:
            coords[sample_dim] = sample_coord
        stacked = xr.DataArray(
            arr,
            dims=(sample_dim, "__features__"),
            coords=coords,
            attrs=dict(meta.attrs),
            name=meta.name,
        )
        return stacked.unstack("__features__")

    return xr.DataArray(
        arr,
        dims=(sample_dim, new_feature_dim),
        coords={
            **sample_coords,
            new_feature_dim: np.arange(n_features),
        },
        attrs=dict(meta.attrs),
        name=meta.name,
    )


def _prepare_y(
    y: xr.DataArray | xr.Dataset | np.ndarray | None,
    sample_dim: Hashable,
) -> np.ndarray | None:
    """Convert a target ``y`` into the numpy form sklearn expects."""
    if y is None:
        return None
    if isinstance(y, xr.DataArray):
        if sample_dim not in y.dims:
            raise ValueError(
                f"y is missing sample_dim={sample_dim!r} (has dims={y.dims})."
            )
        # If y has extra dims, stack them as a feature axis.
        extra = [d for d in y.dims if d != sample_dim]
        if extra:
            stacked = y.stack(__features__=extra).transpose(sample_dim, "__features__")
            return np.asarray(stacked.values)
        return np.asarray(y.values)
    if isinstance(y, xr.Dataset):
        cols = []
        for name in y.data_vars:
            arr = _prepare_y(y[name], sample_dim)
            assert arr is not None
            cols.append(arr if arr.ndim == 2 else arr[:, None])
        return np.column_stack(cols)
    return np.asarray(y)


def _check_no_nan(arr: np.ndarray, *, label: str) -> None:
    """Raise if ``arr`` contains any NaN."""
    if np.isnan(arr).any():
        raise ValueError(
            f"{label} contains NaN values; pass nan_policy='propagate' to forward "
            f"them to the underlying estimator, or impute upstream."
        )


def _dataset_to_2d(
    ds: xr.Dataset, sample_dim: Hashable
) -> tuple[np.ndarray, list[_Meta], list[Hashable], list[int]]:
    """Stack every data variable in ``ds`` into a single 2-D array.

    Variables are column-concatenated in iteration order. Returns the
    stacked array, a per-variable ``_Meta`` list (in the same order),
    the variable names, and the column boundaries between variables in
    the concatenated output.
    """
    metas: list[_Meta] = []
    names: list[Hashable] = []
    blocks: list[np.ndarray] = []
    bounds: list[int] = [0]
    for name in ds.data_vars:
        arr, meta = _to_2d(ds[name], sample_dim)
        blocks.append(arr)
        metas.append(meta)
        names.append(name)
        bounds.append(bounds[-1] + arr.shape[1])
    if not blocks:
        raise ValueError("Cannot stack an empty Dataset (no data variables).")
    n_samples = blocks[0].shape[0]
    for blk, name in zip(blocks, names, strict=True):
        if blk.shape[0] != n_samples:
            raise ValueError(
                f"Variable {name!r} has {blk.shape[0]} samples but expected "
                f"{n_samples}; all variables must share sample_dim={sample_dim!r}."
            )
    return np.concatenate(blocks, axis=1), metas, names, bounds


class XarrayEstimator(BaseEstimator):
    """Wrap an sklearn estimator so it operates on xarray inputs.

    The estimator is cloned on :meth:`fit` (so the original is never
    mutated) and stored as :attr:`estimator_`. After fitting, attribute
    access on the wrapper transparently proxies to ``estimator_`` —
    that means ``wrap.components_``, ``wrap.cluster_centers_``,
    ``wrap.coef_``, etc. work as you would expect.

    Args:
        estimator: Any unfitted sklearn ``BaseEstimator``.
        sample_dim: The xarray dimension indexing samples. If ``None``,
            defaults to the first dim of the input on first ``fit`` /
            ``transform`` call.
        new_feature_dim: Name of the feature dimension when the
            estimator changes the number of features (e.g. PCA reducing
            150 features to 5 components).
        nan_policy: ``"propagate"`` (default) hands NaN to the estimator
            unchanged; ``"raise"`` errors out before delegating.

    Example:
        >>> from sklearn.decomposition import PCA
        >>> wrap = XarrayEstimator(PCA(n_components=3), sample_dim="time")
        >>> scores = wrap.fit_transform(da)            # (time, component)
        >>> recon  = wrap.inverse_transform(scores)    # (time, lat, lon)
        >>> wrap.components_.shape                      # passthrough attr
        (3, lat*lon)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        sample_dim: Hashable | None = None,
        new_feature_dim: str = "component",
        nan_policy: NanPolicy = "propagate",
    ) -> None:
        self.estimator = estimator
        self.sample_dim = sample_dim
        self.new_feature_dim = new_feature_dim
        self.nan_policy = nan_policy

    # ---------- internals -------------------------------------------------

    def _resolve_sample_dim(self, x: xr.DataArray | xr.Dataset) -> Hashable:
        if self.sample_dim is not None:
            return self.sample_dim
        if isinstance(x, xr.DataArray):
            return x.dims[0]
        # Dataset: use the first dim of the first variable.
        first = next(iter(x.data_vars))
        return x[first].dims[0]

    def _stack(
        self,
        x: xr.DataArray | xr.Dataset | np.ndarray,
    ) -> tuple[np.ndarray, _Meta | list[_Meta] | None, Hashable | None]:
        """Marshal the input into a 2-D numpy array.

        Returns ``(arr, meta_or_None, sample_dim_or_None)``. If ``x`` is
        already a numpy array, ``meta`` and ``sample_dim`` are ``None``
        and the wrapper passes the array through unmodified.
        """
        if isinstance(x, np.ndarray):
            return x, None, None
        sample_dim = self._resolve_sample_dim(x)
        if isinstance(x, xr.DataArray):
            arr, meta = _to_2d(x, sample_dim)
            if self.nan_policy == "raise":
                _check_no_nan(arr, label="X")
            return arr, meta, sample_dim
        if isinstance(x, xr.Dataset):
            arr, metas, _, _ = _dataset_to_2d(x, sample_dim)
            if self.nan_policy == "raise":
                _check_no_nan(arr, label="X")
            return arr, metas, sample_dim
        raise TypeError(
            f"X must be xr.DataArray, xr.Dataset, or np.ndarray; got {type(x)}."
        )

    def _prepare_y(
        self,
        y: xr.DataArray | xr.Dataset | np.ndarray | None,
        sample_dim: Hashable | None,
    ) -> np.ndarray | None:
        """Marshal ``y`` to numpy. Errors clearly when ``x`` was numpy
        (no ``sample_dim``) but ``y`` is an xarray object."""
        if sample_dim is None and isinstance(y, xr.DataArray | xr.Dataset):
            raise TypeError(
                "When x is a NumPy array, y must also be a NumPy array or None; "
                "xarray y requires a sample dimension carried on x."
            )
        if sample_dim is None:
            return _prepare_y(y, "")
        return _prepare_y(y, sample_dim)

    def _unstack(
        self,
        arr: np.ndarray,
        meta: _Meta | list[_Meta] | None,
    ) -> xr.DataArray | np.ndarray:
        """Inverse of :meth:`_stack`. Numpy passthrough when ``meta`` is None."""
        if meta is None:
            return arr
        if isinstance(meta, list):
            # Dataset input → return a single DataArray with the new
            # feature axis. Inverse-transforming back to a multi-variable
            # Dataset is intentionally not attempted (lossy when the
            # estimator changed feature count).
            primary = meta[0]
            return _from_2d(arr, primary, new_feature_dim=self.new_feature_dim)
        return _from_2d(arr, meta, new_feature_dim=self.new_feature_dim)

    # ---------- sklearn-style verbs ---------------------------------------

    def fit(
        self,
        x: xr.DataArray | xr.Dataset | np.ndarray,
        y: xr.DataArray | xr.Dataset | np.ndarray | None = None,
        **kwargs: Any,
    ) -> XarrayEstimator:
        """Fit the wrapped estimator to ``x`` (and optional ``y``)."""
        arr, meta, sample_dim = self._stack(x)
        y_np = self._prepare_y(y, sample_dim)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(arr, y_np, **kwargs)
        self._fitted_sample_dim_ = sample_dim
        self._fitted_meta_ = meta
        return self

    def transform(
        self, x: xr.DataArray | xr.Dataset | np.ndarray
    ) -> xr.DataArray | np.ndarray:
        """Transform ``x`` via the fitted estimator."""
        self._require_fitted()
        arr, meta, _ = self._stack(x)
        out = self.estimator_.transform(arr)
        return self._unstack(out, meta)

    def fit_transform(
        self,
        x: xr.DataArray | xr.Dataset | np.ndarray,
        y: xr.DataArray | xr.Dataset | np.ndarray | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | np.ndarray:
        """Fit then transform ``x``."""
        arr, meta, sample_dim = self._stack(x)
        y_np = self._prepare_y(y, sample_dim)
        self.estimator_ = clone(self.estimator)
        if hasattr(self.estimator_, "fit_transform"):
            out = self.estimator_.fit_transform(arr, y_np, **kwargs)
        else:
            self.estimator_.fit(arr, y_np, **kwargs)
            out = self.estimator_.transform(arr)
        self._fitted_sample_dim_ = sample_dim
        self._fitted_meta_ = meta
        return self._unstack(out, meta)

    def inverse_transform(
        self, x: xr.DataArray | xr.Dataset | np.ndarray
    ) -> xr.DataArray | np.ndarray:
        """Map back to the original feature space via the fitted estimator.

        When the inverse output's feature count matches the training
        feature count, the training-time meta is used to re-grid the
        result back to the original ``(sample_dim, *feature_dims)``
        layout. This is the common case for PCA / EOF / NMF.
        """
        self._require_fitted()
        if not hasattr(self.estimator_, "inverse_transform"):
            raise AttributeError(
                f"{self.estimator_.__class__.__name__} does not implement "
                "inverse_transform."
            )
        arr, meta, _ = self._stack(x)
        out = self.estimator_.inverse_transform(arr)
        train_meta = self.__dict__.get("_fitted_meta_")
        if (
            isinstance(train_meta, _Meta)
            and isinstance(meta, _Meta)
            and out.ndim == 2
            and out.shape[1] == train_meta.n_features
        ):
            # Recover the training feature grid (dims, coords, MultiIndex)
            # but keep the *current* input's sample axis/coords — the
            # caller may be inverse-transforming scores from a different
            # sample period than the training set.
            hybrid = _Meta(
                sample_dim=meta.sample_dim,
                sample_coord=meta.sample_coord,
                feature_dims=train_meta.feature_dims,
                feature_index=train_meta.feature_index,
                attrs=train_meta.attrs,
                name=train_meta.name,
                n_features=train_meta.n_features,
            )
            return _from_2d(out, hybrid, new_feature_dim=self.new_feature_dim)
        return self._unstack(out, meta)

    def predict(
        self, x: xr.DataArray | xr.Dataset | np.ndarray
    ) -> xr.DataArray | np.ndarray:
        """Predict via the fitted estimator (regression / classification)."""
        self._require_fitted()
        arr, meta, _ = self._stack(x)
        out = self.estimator_.predict(arr)
        return self._unstack(out, meta)

    def predict_proba(
        self, x: xr.DataArray | xr.Dataset | np.ndarray
    ) -> xr.DataArray | np.ndarray:
        """Class-probability prediction (classifiers only)."""
        self._require_fitted()
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError(
                f"{self.estimator_.__class__.__name__} does not implement "
                "predict_proba."
            )
        arr, meta, _ = self._stack(x)
        out = self.estimator_.predict_proba(arr)
        return self._unstack(out, meta)

    def score(
        self,
        x: xr.DataArray | xr.Dataset | np.ndarray,
        y: xr.DataArray | xr.Dataset | np.ndarray | None = None,
    ) -> float:
        """Scalar score from the wrapped estimator. Not re-wrapped — sklearn
        ``.score`` returns a Python float."""
        self._require_fitted()
        arr, _, sample_dim = self._stack(x)
        y_np = self._prepare_y(y, sample_dim)
        return float(self.estimator_.score(arr, y_np))

    # ---------- proxy + dunder --------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only invoked if the attribute is not found the normal way.
        # Forward fitted-state attributes (``components_``, ``coef_``,
        # ``cluster_centers_``, ``n_iter_`` …) to the wrapped estimator.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            est = self.__dict__["estimator_"]
        except KeyError as exc:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r} "
                "(estimator has not been fitted yet)."
            ) from exc
        return getattr(est, name)

    def _require_fitted(self) -> None:
        if "estimator_" not in self.__dict__:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted; call .fit(...) first."
            )

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        return (
            f"XarrayEstimator(estimator={self.estimator!r}, "
            f"sample_dim={self.sample_dim!r}, "
            f"new_feature_dim={self.new_feature_dim!r}, "
            f"nan_policy={self.nan_policy!r})"
        )
