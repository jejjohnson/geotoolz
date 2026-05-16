"""scikit-learn estimator marshalling for GeoTensor inputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import joblib
import numpy as np


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


ReshapeMode = Literal["pixel", "pixel_time", "spectral", "temporal", "patch", "custom"]
NanStrategy = Literal[
    "drop",
    "propagate",
    "error",
    "impute_simple",
    "impute_knn",
    "impute_iterative",
]
Task = Literal[
    "transform",
    "predict",
    "predict_proba",
    "decision_function",
    "fit_predict",
    "inverse_transform",
]


_IMPUTE_STRATEGIES = {"impute_simple", "impute_knn", "impute_iterative"}


class GeoTensorEstimator:
    """Marshall a scikit-learn estimator to and from GeoTensor-shaped data.

    Args:
        estimator: scikit-learn-compatible object to fit/apply.
        mode: Named reshape mode.
        sample_axes: Explicit sample axes for ``mode="custom"``.
        feature_axes: Explicit feature axes for ``mode="custom"``.
        nan_fit: NaN strategy used while fitting.
        nan_transform: NaN strategy used while applying the estimator.
        impute_simple_strategy: Strategy passed to ``SimpleImputer``.
        impute_knn_n_neighbors: Neighbour count passed to ``KNNImputer``.
        impute_iterative_max_iter: Iteration cap passed to ``IterativeImputer``.

    Examples:
        >>> from sklearn.decomposition import PCA
        >>> est = GeoTensorEstimator(PCA(n_components=2), mode="pixel")
        >>> projected = est.fit(scene).transform(scene)
    """

    def __init__(
        self,
        estimator: Any,
        *,
        mode: ReshapeMode = "pixel",
        sample_axes: tuple[str | int, ...] | None = None,
        feature_axes: tuple[str | int, ...] | None = None,
        nan_fit: NanStrategy = "drop",
        nan_transform: NanStrategy = "propagate",
        impute_simple_strategy: str = "mean",
        impute_knn_n_neighbors: int = 5,
        impute_iterative_max_iter: int = 10,
    ) -> None:
        self.estimator = estimator
        self.mode = mode
        self.sample_axes = sample_axes
        self.feature_axes = feature_axes
        self.nan_fit = nan_fit
        self.nan_transform = nan_transform
        self.impute_simple_strategy = impute_simple_strategy
        self.impute_knn_n_neighbors = impute_knn_n_neighbors
        self.impute_iterative_max_iter = impute_iterative_max_iter
        self.imputer: Any | None = None
        self.is_fitted = False
        self.fit_geotensor_shape: tuple[int, ...] | None = None
        self.fit_n_samples: int | None = None

        _validate_nan_strategy(nan_fit)
        _validate_nan_strategy(nan_transform)
        if mode == "custom" and (sample_axes is None or feature_axes is None):
            raise ValueError(
                "GeoTensorEstimator requires sample_axes and feature_axes "
                'when mode="custom"'
            )

    def fit(self, gt: GeoTensor) -> Self:
        """Fit the wrapped estimator from a GeoTensor."""
        flat = self._flatten(gt)
        x_fit, _ = self._prepare_fit(flat.x)
        self.estimator.fit(x_fit)
        self.is_fitted = True
        self.fit_geotensor_shape = tuple(np.asarray(gt).shape)
        self.fit_n_samples = int(x_fit.shape[0])
        return self

    def partial_fit(self, gt: GeoTensor) -> Self:
        """Incrementally fit the wrapped estimator from a GeoTensor."""
        if not hasattr(self.estimator, "partial_fit"):
            raise TypeError(
                f"{type(self.estimator).__name__} does not support partial_fit"
            )
        flat = self._flatten(gt)
        x_fit, _ = self._prepare_fit(flat.x)
        self.estimator.partial_fit(x_fit)
        self.is_fitted = True
        self.fit_geotensor_shape = tuple(np.asarray(gt).shape)
        self.fit_n_samples = int(x_fit.shape[0])
        return self

    def transform(self, gt: GeoTensor) -> GeoTensor:
        """Apply ``estimator.transform`` and unflatten the result."""
        return self._apply_task(gt, "transform")

    def predict(self, gt: GeoTensor) -> GeoTensor:
        """Apply ``estimator.predict`` and unflatten the result."""
        return self._apply_task(gt, "predict")

    def predict_proba(self, gt: GeoTensor) -> GeoTensor:
        """Apply ``estimator.predict_proba`` and unflatten the result."""
        return self._apply_task(gt, "predict_proba")

    def decision_function(self, gt: GeoTensor) -> GeoTensor:
        """Apply ``estimator.decision_function`` and unflatten the result."""
        return self._apply_task(gt, "decision_function")

    def inverse_transform(self, gt: GeoTensor) -> GeoTensor:
        """Apply ``estimator.inverse_transform`` and unflatten the result."""
        return self._apply_task(gt, "inverse_transform")

    def fit_predict(self, gt: GeoTensor) -> GeoTensor:
        """Apply ``estimator.fit_predict`` and unflatten the result."""
        if not hasattr(self.estimator, "fit_predict"):
            raise TypeError(
                f"{type(self.estimator).__name__} does not support fit_predict"
            )
        flat = self._flatten(gt)
        x_fit, valid = self._prepare_fit(flat.x)
        y_fit = self.estimator.fit_predict(x_fit)
        self.is_fitted = True
        self.fit_geotensor_shape = tuple(np.asarray(gt).shape)
        self.fit_n_samples = int(x_fit.shape[0])
        return self._unflatten_apply_result(gt, flat, y_fit, valid)

    def save_state(self, path: str | Path) -> None:
        """Persist the fitted estimator and any fitted imputer with joblib."""
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "imputer": self.imputer,
                "is_fitted": self.is_fitted,
                "fit_geotensor_shape": self.fit_geotensor_shape,
                "fit_n_samples": self.fit_n_samples,
            },
            state_path,
        )
        _write_metadata(
            state_path,
            fit_geotensor_shape=self.fit_geotensor_shape,
            fit_n_samples=self.fit_n_samples,
        )

    def load_state(self, path: str | Path) -> None:
        """Load a fitted estimator and imputer previously saved with joblib."""
        state_path = Path(path)
        if not state_path.exists():
            raise FileNotFoundError(f"Sklearn state file not found: {state_path}")
        state = joblib.load(state_path)
        self.estimator = state["estimator"]
        self.imputer = state.get("imputer")
        self.is_fitted = bool(state.get("is_fitted", True))
        shape = state.get("fit_geotensor_shape")
        self.fit_geotensor_shape = None if shape is None else tuple(shape)
        n_samples = state.get("fit_n_samples")
        self.fit_n_samples = None if n_samples is None else int(n_samples)

    def _flatten(self, gt: GeoTensor) -> _FlatGeoTensor:
        arr = np.asarray(gt)
        axes = _resolve_axes(arr.ndim, self.mode, self.sample_axes, self.feature_axes)
        moved = np.moveaxis(arr, axes.sample_axes + axes.feature_axes, range(arr.ndim))
        sample_shape = tuple(arr.shape[axis] for axis in axes.sample_axes)
        feature_shape = tuple(arr.shape[axis] for axis in axes.feature_axes)
        n_samples = prod(sample_shape)
        n_features = prod(feature_shape) if feature_shape else 1
        return _FlatGeoTensor(
            x=moved.reshape(n_samples, n_features),
            axes=axes,
            sample_shape=sample_shape,
            feature_shape=feature_shape,
        )

    def _prepare_fit(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        strategy = self.nan_fit
        valid = _valid_rows(x)
        if strategy == "error" and not valid.all():
            raise ValueError("GeoTensorEstimator.fit received NaN values")
        if strategy == "error":
            return x, valid
        if strategy == "drop":
            return x[valid], valid
        if strategy == "propagate":
            return x, np.ones(x.shape[0], dtype=bool)
        self.imputer = _make_imputer(
            strategy,
            simple_strategy=self.impute_simple_strategy,
            knn_n_neighbors=self.impute_knn_n_neighbors,
            iterative_max_iter=self.impute_iterative_max_iter,
        )
        return self.imputer.fit_transform(x), np.ones(x.shape[0], dtype=bool)

    def _prepare_transform(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        strategy = self.nan_transform
        valid = _valid_rows(x)
        if strategy == "error" and not valid.all():
            raise ValueError("GeoTensorEstimator.transform received NaN values")
        if strategy == "error":
            return x, valid
        if strategy in {"drop", "propagate"}:
            return x[valid], valid
        if self.imputer is None:
            self.imputer = _make_imputer(
                strategy,
                simple_strategy=self.impute_simple_strategy,
                knn_n_neighbors=self.impute_knn_n_neighbors,
                iterative_max_iter=self.impute_iterative_max_iter,
            )
            self.imputer.fit(x)
        return self.imputer.transform(x), np.ones(x.shape[0], dtype=bool)

    def _apply_task(self, gt: GeoTensor, task: Task) -> GeoTensor:
        if not hasattr(self.estimator, task):
            raise TypeError(f"{type(self.estimator).__name__} does not support {task}")
        flat = self._flatten(gt)
        x_apply, valid = self._prepare_transform(flat.x)
        y_apply = getattr(self.estimator, task)(x_apply)
        return self._unflatten_apply_result(gt, flat, y_apply, valid)

    def _unflatten_apply_result(
        self,
        gt: GeoTensor,
        flat: _FlatGeoTensor,
        y_apply: np.ndarray,
        valid: np.ndarray,
    ) -> GeoTensor:
        y = np.asarray(y_apply)
        if valid.all():
            dense = y
        else:
            out_shape = (valid.shape[0], *y.shape[1:])
            dense = np.full(out_shape, np.nan, dtype=np.result_type(y.dtype, float))
            dense[valid] = y
        out = _restore_shape(dense, flat)
        if out.ndim < 2 or out.shape[-2:] != np.asarray(gt).shape[-2:]:
            return out
        return gt.array_as_geotensor(out)


class _ResolvedAxes:
    def __init__(
        self,
        *,
        axis_order: tuple[str, ...],
        sample_axes: tuple[int, ...],
        feature_axes: tuple[int, ...],
    ) -> None:
        self.axis_order = axis_order
        self.sample_axes = sample_axes
        self.feature_axes = feature_axes


class _FlatGeoTensor:
    def __init__(
        self,
        *,
        x: np.ndarray,
        axes: _ResolvedAxes,
        sample_shape: tuple[int, ...],
        feature_shape: tuple[int, ...],
    ) -> None:
        self.x = x
        self.axes = axes
        self.sample_shape = sample_shape
        self.feature_shape = feature_shape


def _resolve_axes(
    ndim: int,
    mode: ReshapeMode,
    sample_axes: tuple[str | int, ...] | None,
    feature_axes: tuple[str | int, ...] | None,
) -> _ResolvedAxes:
    if mode == "patch":
        return _ResolvedAxes(
            axis_order=tuple(str(i) for i in range(ndim)),
            sample_axes=(0,),
            feature_axes=tuple(range(1, ndim)),
        )

    axis_order = _canonical_axis_order(ndim)
    if mode == "custom":
        assert sample_axes is not None
        assert feature_axes is not None
        samples = _axis_indices(sample_axes, axis_order, ndim)
        features = _axis_indices(feature_axes, axis_order, ndim)
    elif mode == "pixel":
        samples = _axis_indices(("H", "W"), axis_order, ndim)
        features = tuple(axis for axis in range(ndim) if axis not in samples)
    elif mode == "pixel_time":
        samples = _axis_indices(("T", "H", "W"), axis_order, ndim)
        features = tuple(axis for axis in range(ndim) if axis not in samples)
    elif mode == "spectral":
        samples = _axis_indices(("C",), axis_order, ndim)
        features = tuple(axis for axis in range(ndim) if axis not in samples)
    elif mode == "temporal":
        samples = _axis_indices(("T",), axis_order, ndim)
        features = tuple(axis for axis in range(ndim) if axis not in samples)
    else:
        raise ValueError(f"Unknown reshape mode: {mode!r}")

    if set(samples) & set(features):
        raise ValueError("sample_axes and feature_axes must be disjoint")
    if len(samples) + len(features) != ndim:
        raise ValueError("sample_axes and feature_axes must cover every input axis")
    return _ResolvedAxes(
        axis_order=axis_order,
        sample_axes=samples,
        feature_axes=features,
    )


def _canonical_axis_order(ndim: int) -> tuple[str, ...]:
    if ndim < 2:
        raise ValueError("GeoTensorEstimator expects at least 2 dimensions")
    if ndim == 2:
        return ("H", "W")
    labels = ("T", "C", "H", "W")
    if ndim <= 4:
        return labels[-ndim:]
    extra = tuple(f"X{i}" for i in range(ndim - 4))
    return extra + labels


def _axis_indices(
    axes: Iterable[str | int],
    axis_order: tuple[str, ...],
    ndim: int,
) -> tuple[int, ...]:
    indices: list[int] = []
    for axis in axes:
        if isinstance(axis, str):
            if axis not in axis_order:
                raise ValueError(f"Axis label {axis!r} is not present in this input")
            idx = axis_order.index(axis)
        else:
            idx = axis % ndim
        indices.append(idx)
    if len(set(indices)) != len(indices):
        raise ValueError("Axes must not contain duplicates")
    return tuple(indices)


def _restore_shape(y: np.ndarray, flat: _FlatGeoTensor) -> np.ndarray:
    if y.ndim == 1:
        return y.reshape(flat.sample_shape)
    if y.ndim != 2:
        raise ValueError("Estimator output must be 1-D or 2-D")

    sample_axes = set(flat.axes.sample_axes)
    first_feature_axis = min(flat.axes.feature_axes) if flat.axes.feature_axes else None
    restored = y.reshape((*flat.sample_shape, y.shape[1]))
    source_tokens: list[int | str] = [*flat.axes.sample_axes, "__out__"]
    target_tokens: list[int | str] = []
    out_axis_inserted = False
    for axis in range(len(flat.axes.axis_order)):
        if axis in sample_axes:
            target_tokens.append(axis)
        elif axis == first_feature_axis and not out_axis_inserted:
            target_tokens.append("__out__")
            out_axis_inserted = True
    if not out_axis_inserted:
        target_tokens.append("__out__")
    return restored.transpose([source_tokens.index(token) for token in target_tokens])


def _valid_rows(x: np.ndarray) -> np.ndarray:
    return ~np.isnan(np.asarray(x, dtype=float)).any(axis=1)


def _validate_nan_strategy(strategy: NanStrategy) -> None:
    valid = {"drop", "propagate", "error", *_IMPUTE_STRATEGIES}
    if strategy not in valid:
        raise ValueError(f"Unknown NaN strategy: {strategy!r}")


def _make_imputer(
    strategy: NanStrategy,
    *,
    simple_strategy: str,
    knn_n_neighbors: int,
    iterative_max_iter: int,
) -> Any:
    if strategy == "impute_simple":
        from sklearn.impute import SimpleImputer

        return SimpleImputer(strategy=simple_strategy)
    if strategy == "impute_knn":
        from sklearn.impute import KNNImputer

        return KNNImputer(n_neighbors=knn_n_neighbors)
    if strategy == "impute_iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        return IterativeImputer(max_iter=iterative_max_iter)
    raise ValueError(f"NaN strategy {strategy!r} is not an imputer strategy")


def _write_metadata(
    state_path: Path,
    *,
    fit_geotensor_shape: tuple[int, ...] | None,
    fit_n_samples: int | None,
) -> None:
    import sklearn

    metadata = {
        "sklearn_version": sklearn.__version__,
        "fit_geotensor_shape": fit_geotensor_shape,
        "fit_n_samples": fit_n_samples,
        "fit_timestamp": datetime.now(UTC).isoformat(),
    }
    state_path.with_suffix(state_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
