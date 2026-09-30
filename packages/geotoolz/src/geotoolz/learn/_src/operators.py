"""Tier-B operators of :mod:`geotoolz.learn` -- every ``pipekit.Operator`` here.

Module rule for ``learn/_src``:

* ``array.py`` -- Tier A: axis bookkeeping (reshape modes, output fill
  values); no sklearn, no carriers.
* ``estimators.py`` -- :class:`GeoTensorEstimator`, the non-Operator
  adapter that marshals a carrier to ``(n_samples, n_features)``, applies
  the NaN strategy, calls the estimator and restores the grid, plus its
  joblib state I/O.
* ``operators.py`` (this module) -- the Operators: :class:`SklearnOp`
  (fitting lifecycle over a :class:`GeoTensorEstimator`), the named
  convenience wrappers (``PixelwisePCA``, ``PixelwiseKMeans``, ...), and the
  framework-agnostic :class:`ModelOp`, which wraps any callable (torch,
  JAX, sklearn ``predict``, a plain function) without importing a
  framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pipekit import Carrier, Operator

from geotoolz._src.config import jsonable
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

    Delegates the ``(C, H, W) <-> (n_samples, n_features)`` marshalling
    to :class:`GeoTensorEstimator` and adds the fitting lifecycle
    (``fit_mode``) plus joblib state persistence. Metadata-independent:
    accepts a ``GeoTensor`` or a plain ``np.ndarray`` and returns a
    matching carrier whenever the estimator output preserves the input's
    spatial ``(H, W)`` layout; sample-only outputs (and
    ``nan_transform="drop"`` with invalid rows) come back as bare ndarrays.

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
        state_path: Optional joblib path to load immediately, in the
            constructor. Loading unpickles the file, which can execute
            arbitrary code: only pass files from a trusted source (see
            :meth:`load_state`).
        out_band_names: Names of the output bands, written to
            ``attrs["band_names"]`` of GeoTensor outputs (must match the
            output band count). ``None`` keeps the input's band keys when
            the band count is unchanged and drops them otherwise.
        label_fill_value: Fill value of integer outputs such as
            cluster-label maps (default ``-1``). Float outputs always use
            ``NaN``. See :class:`GeoTensorEstimator` for the nodata rules
            (fill pixels of the input are excluded from fits and come back
            as the output fill).

    Examples:
        >>> from sklearn.decomposition import PCA
        >>> op = SklearnOp(estimator=PCA(n_components=3), mode="pixel")
        >>> projected = op(scene)
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        estimator: Any,
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
        label_fill_value: int = -1,
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
        if fit_mode == "pre_fit" and state_path is None:
            raise ValueError(
                'fit_mode="pre_fit" requires state_path to point at a '
                "previously fitted estimator state"
            )

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
        self.label_fill_value = label_fill_value
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
            out_band_names=out_band_names,
            label_fill_value=label_fill_value,
        )
        if state_path is not None:
            self.load_state(state_path)

    def _apply(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        if self._task == "fit_predict":
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

    def save_state(self, path: str | Path, *, write_meta: bool = False) -> None:
        """Persist fitted estimator state to ``path`` (a joblib pickle).

        Args:
            path: Destination file.
            write_meta: Also write a timestamped ``<path>.meta.json``
                sidecar (scikit-learn version, fit shape / sample count,
                UTC ``fit_timestamp``); off by default. See
                :meth:`GeoTensorEstimator.save_state`.
        """
        self._geo_estimator.save_state(path, write_meta=write_meta)
        self.estimator = self._geo_estimator.estimator
        self.state_path = str(path)

    def load_state(self, path: str | Path) -> None:
        """Load fitted estimator state from ``path``.

        Warning:
            The state file is a joblib pickle and unpickling can execute
            arbitrary code. Only load files you created or that come from
            a source you trust as much as the code you run.
        """
        self._geo_estimator.load_state(path)
        self.estimator = self._geo_estimator.estimator
        self.state_path = str(path)

    def get_config(self) -> dict[str, Any]:
        # Keys mirror the constructor; the estimator (a runtime object) is
        # summarised as a debug payload with the task it resolves to.
        params = (
            self.estimator.get_params(deep=False)
            if hasattr(self.estimator, "get_params")
            else {}
        )
        estimator_path = (
            f"{type(self.estimator).__module__}.{type(self.estimator).__name__}"
        )
        return {
            "estimator": {
                "class": estimator_path,
                # ``strict``: nested estimators (``Pipeline.steps``),
                # ``RandomState`` and callables become their ``repr`` and
                # non-finite floats ``'nan'`` / ``'inf'``, so the payload
                # always survives strict ``json.dumps`` with no key dropped.
                "params": jsonable(params, strict=True),
                "resolved_task": self._task,
            },
            "mode": self.mode,
            "sample_axes": jsonable(self.sample_axes),
            "feature_axes": jsonable(self.feature_axes),
            "fit_mode": self.fit_mode,
            "task": self.task,
            "nan_fit": self.nan_fit,
            "nan_transform": self.nan_transform,
            "impute_simple_strategy": self.impute_simple_strategy,
            "impute_knn_n_neighbors": self.impute_knn_n_neighbors,
            "impute_iterative_max_iter": self.impute_iterative_max_iter,
            "state_path": self.state_path,
            "out_band_names": self.out_band_names,
            "label_fill_value": self.label_fill_value,
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


class PixelwisePCA(SklearnOp):
    """Pixel-wise PCA convenience operator.

    Pins ``mode="pixel"``, ``task="transform"``, ``nan_fit="drop"``;
    every other :class:`SklearnOp` keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "transform")
        kwargs.setdefault("nan_fit", "drop")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseIPCA(SklearnOp):
    """Streaming IncrementalPCA convenience operator.

    Pins ``mode="pixel"``, ``fit_mode="fit_streaming"`` (each call routes
    through ``partial_fit``), ``task="transform"``; every other
    :class:`SklearnOp` keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("fit_mode", "fit_streaming")
        kwargs.setdefault("task", "transform")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseNMF(SklearnOp):
    """Pixel-wise NMF convenience operator.

    Pins ``mode="pixel"`` and ``task="transform"``; every other
    :class:`SklearnOp` keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "transform")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseKMeans(SklearnOp):
    """Pixel-wise KMeans label convenience operator.

    Pins ``mode="pixel"`` and ``task="predict"`` — the output is a
    single-band cluster-label map; every other :class:`SklearnOp`
    keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "predict")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseMiniBatchKMeans(SklearnOp):
    """Streaming MiniBatchKMeans convenience operator.

    Pins ``mode="pixel"``, ``fit_mode="fit_streaming"`` (each call routes
    through ``partial_fit``), ``task="predict"``; every other
    :class:`SklearnOp` keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("fit_mode", "fit_streaming")
        kwargs.setdefault("task", "predict")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseGMM(SklearnOp):
    """Gaussian mixture convenience operator.

    Pins ``mode="pixel"`` and ``task="predict_proba"`` — the output has
    one band per mixture component; every other :class:`SklearnOp`
    keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "predict_proba")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseIsolationForest(SklearnOp):
    """Pixel-wise IsolationForest anomaly-score convenience operator.

    Pins ``mode="pixel"`` and ``task="decision_function"`` — the output
    is a single-band anomaly-score map; every other :class:`SklearnOp`
    keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "decision_function")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseOneClassSVM(SklearnOp):
    """Pixel-wise OneClassSVM anomaly-score convenience operator.

    Pins ``mode="pixel"`` and ``task="decision_function"``; every other
    :class:`SklearnOp` keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "decision_function")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseLocalOutlierFactor(SklearnOp):
    """Pixel-wise LocalOutlierFactor convenience operator.

    Pins ``mode="pixel"`` and ``task="decision_function"``; requires the
    estimator to be constructed with ``novelty=True`` (sklearn only
    exposes ``decision_function`` in novelty mode). Every other
    :class:`SklearnOp` keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "decision_function")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseKNNImputer(SklearnOp):
    """Pixel-wise KNNImputer convenience operator.

    Pins ``mode="pixel"``, ``task="transform"``, and
    ``nan_fit=nan_transform="propagate"`` so the NaN-tolerant imputer
    fits on the clean rows and NaN placement is preserved for it to
    fill; every other :class:`SklearnOp` keyword argument passes
    through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "transform")
        kwargs.setdefault("nan_fit", "propagate")
        kwargs.setdefault("nan_transform", "propagate")
        super().__init__(estimator=estimator, **kwargs)


class PixelwiseIterativeImputer(SklearnOp):
    """Pixel-wise IterativeImputer convenience operator.

    Pins ``mode="pixel"``, ``task="transform"``, and
    ``nan_fit=nan_transform="propagate"``; every other
    :class:`SklearnOp` keyword argument passes through.
    """

    def __init__(self, *, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "transform")
        kwargs.setdefault("nan_fit", "propagate")
        kwargs.setdefault("nan_transform", "propagate")
        super().__init__(estimator=estimator, **kwargs)


class ModelOp(Operator):
    """Wrap any callable as an Operator.

    Materialises the GeoTensor to a plain ``np.ndarray`` (via
    ``np.asarray``) before handing it to the model — frameworks that
    strip the subclass (torch, JAX, sklearn) don't care, and frameworks
    that preserve it (numpy proper) still see something sensible.

    Args:
        model: Any object that can be called as ``model(arr)`` or whose
            ``method`` attribute can be called as
            ``model.predict(arr)``. No isinstance / framework imports.
        method: Method name to invoke on ``model``. Default
            ``"__call__"`` — equivalent to ``model(arr)``. Set to
            ``"predict"`` for sklearn estimators.
        batch_size: If set, split the input along axis 0 into chunks of
            this size, call the model once per chunk, concatenate the
            results along axis 0. Useful when the model can't fit the
            whole input in GPU memory.

    Note:
        ``forbid_in_yaml = True`` — the model is a runtime object and
        won't round-trip to YAML. Users typically pin a model artifact
        (state-dict + class config) themselves.

    Examples:
        Inference with a sklearn classifier::

            op = ModelOp(model=rf_clf, method="predict")
            preds = op(features_gt)

        Batched inference with a torch model::

            op = ModelOp(model=unet_model, batch_size=8)
            preds = op(chips_gt)  # iterates 8 chips at a time
    """

    forbid_in_yaml: ClassVar[bool] = True
    # ConfigMixin would auto-derive `model` from `__init__` as a non-JSON
    # opaque object; override with a curated debug repr below.
    __config_mixin_auto__: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: Any,
        method: str = "__call__",
        batch_size: int | None = None,
    ) -> None:
        self.model = model
        self.method = method
        self.batch_size = batch_size

    def _resolve_callable(self) -> Any:
        if self.method == "__call__":
            return self.model
        return getattr(self.model, self.method)

    def _apply(self, gt: Carrier) -> Any:
        arr = np.asarray(gt)
        fn = self._resolve_callable()
        if self.batch_size is None:
            return fn(arr)
        return self._batched(fn, arr)

    def _batched(self, fn: Any, arr: np.ndarray) -> np.ndarray:
        """Split ``arr`` along axis 0, call ``fn`` per chunk, concatenate.

        Plain ``np.concatenate`` along axis 0 — works when the model's
        output preserves the batch dimension (the common case).

        Empty inputs (``arr.shape[0] == 0``) are passed straight to the
        model in one call: ``np.concatenate`` cannot accept an empty list
        of chunks, and the model is free to return a meaningful
        zero-length result.
        """
        n = arr.shape[0]
        if n == 0:
            return fn(arr)
        chunks: list[Any] = []
        bs = int(self.batch_size or n)
        for start in range(0, n, bs):
            chunks.append(fn(arr[start : start + bs]))
        return np.concatenate(chunks, axis=0)

    def get_config(self) -> dict[str, Any]:
        return {
            "model_type": type(self.model).__name__,
            "method": self.method,
            "batch_size": self.batch_size,
        }


__all__ = [
    "ModelOp",
    "PixelwiseGMM",
    "PixelwiseIPCA",
    "PixelwiseIsolationForest",
    "PixelwiseIterativeImputer",
    "PixelwiseKMeans",
    "PixelwiseKNNImputer",
    "PixelwiseLocalOutlierFactor",
    "PixelwiseMiniBatchKMeans",
    "PixelwiseNMF",
    "PixelwiseOneClassSVM",
    "PixelwisePCA",
    "SklearnOp",
]
