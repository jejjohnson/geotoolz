"""Operator wrappers for scikit-learn estimators."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from geotoolz.core import Operator
from geotoolz.learn._src.estimators import (
    GeoTensorEstimator,
    NanStrategy,
    ReshapeMode,
    Task,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


FitMode = Literal["pre_fit", "fit_on_call", "refit", "fit_streaming", "fit_only"]


class SklearnOp(Operator):
    """Universal adapter for scikit-learn-compatible estimators on GeoTensors.

    Args:
        estimator: scikit-learn-compatible object to wrap.
        mode: Named reshape mode passed to :class:`GeoTensorEstimator`.
        sample_axes: Explicit sample axes for ``mode="custom"``.
        feature_axes: Explicit feature axes for ``mode="custom"``.
        fit_mode: Fitting lifecycle: pre-fit, first-call fit, refit,
            streaming ``partial_fit``, or fit-only.
        task: Estimator method used at apply time. ``None`` auto-detects.
        nan_fit: NaN strategy used while fitting.
        nan_transform: NaN strategy used while applying the estimator.
        state_path: Optional joblib path to load immediately.

    Examples:
        >>> from sklearn.decomposition import PCA
        >>> op = SklearnOp(PCA(n_components=3), mode="pixel")
        >>> projected = op(scene)
    """

    def __init__(
        self,
        estimator: Any,
        *,
        mode: ReshapeMode = "pixel",
        sample_axes: tuple[str | int, ...] | None = None,
        feature_axes: tuple[str | int, ...] | None = None,
        fit_mode: FitMode = "fit_on_call",
        task: Task | None = None,
        nan_fit: NanStrategy = "drop",
        nan_transform: NanStrategy = "propagate",
        impute_simple_strategy: str = "mean",
        impute_knn_n_neighbors: int = 5,
        impute_iterative_max_iter: int = 10,
        state_path: str | Path | None = None,
        out_band_names: list[str] | None = None,
    ) -> None:
        _validate_fit_mode(fit_mode)
        resolved_task = _resolve_task(estimator, task)
        if fit_mode == "fit_streaming" and not hasattr(estimator, "partial_fit"):
            raise TypeError(
                f"{type(estimator).__name__} does not support fit_streaming "
                "because it has no partial_fit method"
            )
        if fit_mode == "pre_fit" and resolved_task == "fit_predict":
            raise ValueError('fit_mode="pre_fit" is not valid with task="fit_predict"')

        self.estimator = estimator
        self.mode = mode
        self.sample_axes = sample_axes
        self.feature_axes = feature_axes
        self.fit_mode = fit_mode
        self.task = task
        self.nan_fit = nan_fit
        self.nan_transform = nan_transform
        self.impute_simple_strategy = impute_simple_strategy
        self.impute_knn_n_neighbors = impute_knn_n_neighbors
        self.impute_iterative_max_iter = impute_iterative_max_iter
        self.state_path = None if state_path is None else str(state_path)
        self.out_band_names = out_band_names
        self._task = resolved_task
        self._geo_estimator = GeoTensorEstimator(
            estimator,
            mode=mode,
            sample_axes=sample_axes,
            feature_axes=feature_axes,
            nan_fit=nan_fit,
            nan_transform=nan_transform,
            impute_simple_strategy=impute_simple_strategy,
            impute_knn_n_neighbors=impute_knn_n_neighbors,
            impute_iterative_max_iter=impute_iterative_max_iter,
        )
        if state_path is not None:
            self.load_state(state_path)

    def _apply(self, gt: GeoTensor) -> GeoTensor:
        if self._task == "fit_predict":
            if self.fit_mode == "pre_fit":
                raise ValueError('fit_mode="pre_fit" is not valid with fit_predict')
            return self._geo_estimator.fit_predict(gt)

        should_fit = (
            self.fit_mode == "fit_on_call" and not self._geo_estimator.is_fitted
        ) or self.fit_mode == "refit"
        if should_fit:
            self._geo_estimator.fit(gt)
        elif self.fit_mode == "fit_streaming":
            self._geo_estimator.partial_fit(gt)
        elif self.fit_mode == "fit_only":
            self._geo_estimator.fit(gt)
            return gt

        return getattr(self._geo_estimator, self._task)(gt)

    def save_state(self, path: str | Path) -> None:
        """Persist fitted estimator state to ``path``."""
        self._geo_estimator.save_state(path)
        self.estimator = self._geo_estimator.estimator
        self.state_path = str(path)

    def load_state(self, path: str | Path) -> None:
        """Load fitted estimator state from ``path``."""
        self._geo_estimator.load_state(path)
        self.estimator = self._geo_estimator.estimator
        self.state_path = str(path)

    def get_config(self) -> dict[str, Any]:
        params = (
            self.estimator.get_params(deep=False)
            if hasattr(self.estimator, "get_params")
            else {}
        )
        estimator_path = (
            f"{type(self.estimator).__module__}.{type(self.estimator).__name__}"
        )
        return {
            "estimator": estimator_path,
            "estimator_params": _jsonable_params(params),
            "mode": self.mode,
            "sample_axes": self.sample_axes,
            "feature_axes": self.feature_axes,
            "fit_mode": self.fit_mode,
            "task": self.task,
            "nan_fit": self.nan_fit,
            "nan_transform": self.nan_transform,
            "impute_simple_strategy": self.impute_simple_strategy,
            "impute_knn_n_neighbors": self.impute_knn_n_neighbors,
            "impute_iterative_max_iter": self.impute_iterative_max_iter,
            "state_path": self.state_path,
            "out_band_names": self.out_band_names,
        }


def _resolve_task(estimator: Any, task: Task | None) -> Task:
    if task is not None:
        if not hasattr(estimator, task):
            raise TypeError(f"{type(estimator).__name__} does not support {task}")
        return task
    for candidate in ("transform", "predict", "decision_function", "fit_predict"):
        if hasattr(estimator, candidate):
            return candidate  # type: ignore[return-value]
    raise TypeError(
        f"{type(estimator).__name__} must expose transform, predict, "
        "decision_function, or fit_predict"
    )


def _validate_fit_mode(fit_mode: FitMode) -> None:
    if fit_mode not in {"pre_fit", "fit_on_call", "refit", "fit_streaming", "fit_only"}:
        raise ValueError(f"Unknown fit mode: {fit_mode!r}")


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    jsonable: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or isinstance(value, (bool, int, float, str)):
            jsonable[key] = value
        elif isinstance(value, (list, tuple)):
            jsonable[key] = list(value)
    return jsonable
