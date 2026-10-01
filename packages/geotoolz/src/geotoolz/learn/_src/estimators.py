"""scikit-learn estimator marshalling for GeoTensor inputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import joblib
import numpy as np
from jaxtyping import Bool, Num, Shaped

from geotoolz._src.dtype import as_float
from geotoolz._src.samples import SampleLayout, cube_to_samples, samples_to_cube
from geotoolz._src.valid import invalid_values
from geotoolz._src.wrap import wrap_like
from geotoolz.learn._src.array import (
    ReshapeMode,
    output_fill_value,
    resolve_axes,
)


if TYPE_CHECKING:
    from georeader.geotensor import GeoTensor


NanStrategy = Literal[
    "drop",
    "propagate",
    "propagate_raw",
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

    scikit-learn works on 2-D ``(n_samples, n_features)`` matrices while
    remote-sensing cubes are channel-first rasters. This class owns the
    round trip between the two layouts. Axes follow the canonical
    channel-first order — ``(H, W)`` for 2-D input, ``(C, H, W)`` for
    3-D, ``(T, C, H, W)`` for 4-D — and ``mode`` picks which axes become
    samples (rows) and which become features (columns):

    - ``"pixel"``: samples ``(H, W)``, features the rest. A ``(C, H, W)``
      cube flattens to ``(H * W, C)`` — one row per pixel, one column
      per band.
    - ``"pixel_time"``: samples ``(T, H, W)``; ``(T, C, H, W)`` flattens
      to ``(T * H * W, C)``.
    - ``"spectral"``: samples ``(C,)``; ``(C, H, W)`` flattens to
      ``(C, H * W)`` — one row per band.
    - ``"temporal"``: samples ``(T,)``; features are all remaining axes.
    - ``"patch"``: samples axis ``0``; ``(N, ...)`` flattens to
      ``(N, prod(rest))``.
    - ``"custom"``: explicit ``sample_axes`` / ``feature_axes`` (axis
      labels ``"T"/"C"/"H"/"W"`` or integer indices) that together must
      cover every input axis exactly once.

    On the way back, a 1-D estimator output of length ``n_samples``
    unflattens to the sample shape (``"pixel"`` -> ``(H, W)``); a 2-D
    output ``(n_samples, k)`` restores the sample axes to their original
    positions and places the ``k`` output-feature axis where the first
    input feature axis was (``"pixel"`` on ``(C, H, W)`` -> ``(k, H, W)``),
    or in front of the sample axes when there is no feature axis
    (``"pixel"`` on a single-band ``(H, W)`` -> ``(k, H, W)``), so outputs
    stay channel-first.

    Nodata: a sample row is *invalid* when any of its elements is
    non-finite or equals the carrier's ``fill_value_default`` (see
    :mod:`geotoolz._src.valid`; plain ndarrays have no fill, so only
    non-finite values count). The ``nan_*`` strategies apply to every
    invalid row, not only NaN ones, and invalid elements are handed to
    NaN-tolerant estimators and imputers as ``NaN``. Following georeader,
    a carrier with ``fill_value_default=0`` treats ``0`` as nodata; pass
    ``fill_value_default=None`` (or ``NaN``) on the input when ``0`` is
    real data.

    Output fill values: rows that were not handed to the estimator are
    written back as nodata, and every GeoTensor output declares an
    explicit ``fill_value_default`` that matches its dtype -- ``NaN`` for
    floating-point outputs (scores, projections, probabilities),
    ``label_fill_value`` (default ``-1``) for integer outputs such as
    cluster-label maps (the dtype is kept), ``False`` for boolean outputs.

    Carrier behavior: every method accepts a ``GeoTensor`` or a plain
    ``np.ndarray``. When the unflattened output's trailing two axes still
    match the input's spatial ``(H, W)``, the result is rewrapped to
    match the input carrier (GeoTensor in -> GeoTensor out with a fresh
    ``attrs`` copy; ndarray in -> ndarray out). Otherwise a bare ndarray is
    returned (e.g. sample-only outputs from ``mode="custom"`` or
    ``mode="spectral"``, or ``nan_transform="drop"`` with invalid rows).

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
        out_band_names: Names of the output bands, written to
            ``attrs["band_names"]`` of GeoTensor outputs (one per output
            band; a mismatch raises ``ValueError``). ``None`` keeps the
            input's band keys when the band count is unchanged and drops
            them otherwise.
        label_fill_value: Fill value for integer outputs (e.g. cluster
            labels). ``-1`` by default -- sklearn's "no cluster" label and
            never a valid KMeans / GMM component index. Pick another value
            if ``-1`` is a real class of the wrapped estimator (e.g.
            ``IsolationForest.predict`` outliers).

    Attributes:
        estimator: The wrapped scikit-learn estimator (replaced on
            ``load_state``).
        imputer: Fitted imputer for the ``impute_*`` NaN strategies, or
            ``None``. Refitted by :meth:`fit`; in streaming mode it is
            fitted on the first :meth:`partial_fit` batch and reused
            (frozen) for later batches.
        is_fitted: Whether ``fit`` / ``partial_fit`` / ``fit_predict``
            (or ``load_state``) has run.
        fit_geotensor_shape: Shape of the last cube seen at fit time.
        fit_n_samples: Number of sample rows the estimator was fitted on.

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
        out_band_names: Sequence[str] | None = None,
        label_fill_value: int = -1,
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
        self.out_band_names = None if out_band_names is None else list(out_band_names)
        self.label_fill_value = int(label_fill_value)
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

    def fit(self, gt: GeoTensor | np.ndarray) -> Self:
        """Fit the wrapped estimator on the flattened sample matrix.

        The cube is reshaped to ``(n_samples, n_features)`` according to
        ``mode`` (e.g. ``"pixel"`` sends ``(C, H, W)`` to ``(H * W, C)``),
        the ``nan_fit`` strategy is applied to the rows, and the result
        is handed to ``estimator.fit``.

        Args:
            gt: Input cube — a ``GeoTensor`` or plain ``np.ndarray``.

        Returns:
            This estimator, for chaining.
        """
        flat = self._flatten(gt)
        x_fit, _ = self._prepare_fit(flat)
        self.estimator.fit(x_fit)
        self.is_fitted = True
        self.fit_geotensor_shape = tuple(np.asarray(gt).shape)
        self.fit_n_samples = int(x_fit.shape[0])
        return self

    def partial_fit(self, gt: GeoTensor | np.ndarray) -> Self:
        """Incrementally fit the wrapped estimator on one flattened batch.

        Same ``(n_samples, n_features)`` marshalling and ``nan_fit``
        handling as :meth:`fit`, but routed to ``estimator.partial_fit``
        so streaming estimators accumulate state across calls. With an
        ``impute_*`` ``nan_fit`` strategy the imputer is fitted on the
        first batch only and reused (``imputer.transform``) for every
        later batch, so no batch re-derives the imputation statistics.

        Args:
            gt: Input cube — a ``GeoTensor`` or plain ``np.ndarray``.

        Returns:
            This estimator, for chaining.

        Raises:
            TypeError: If the wrapped estimator has no ``partial_fit``.
        """
        if not hasattr(self.estimator, "partial_fit"):
            raise TypeError(
                f"{type(self.estimator).__name__} does not support partial_fit"
            )
        flat = self._flatten(gt)
        x_fit, _ = self._prepare_fit(flat, reuse_imputer=True)
        self.estimator.partial_fit(x_fit)
        self.is_fitted = True
        self.fit_geotensor_shape = tuple(np.asarray(gt).shape)
        self.fit_n_samples = int(x_fit.shape[0])
        return self

    def transform(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        """Apply ``estimator.transform`` and unflatten the result.

        The cube is flattened to ``(n_samples, n_features)`` per
        ``mode``, ``nan_transform`` is applied, and the estimator output
        of shape ``(n_samples,)`` or ``(n_samples, k)`` is reshaped back
        (the ``k`` output-feature axis replaces the first input feature
        axis — ``"pixel"`` mode maps ``(C, H, W)`` input to ``(k, H, W)``
        output).

        Returns a carrier matching the input (GeoTensor in -> GeoTensor
        out, ndarray in -> ndarray out) when the output's trailing axes
        match the input's spatial ``(H, W)`` shape; otherwise returns a
        bare :class:`numpy.ndarray` (e.g. for sample-only outputs from
        ``mode="custom"``, or ``nan_transform="drop"`` with invalid rows).
        """
        return self._apply_task(gt, "transform")

    def predict(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        """Apply ``estimator.predict`` and unflatten the result.

        See :meth:`transform` for the reshaping and carrier contract.
        """
        return self._apply_task(gt, "predict")

    def predict_proba(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        """Apply ``estimator.predict_proba`` and unflatten the result.

        See :meth:`transform` for the reshaping and carrier contract.
        """
        return self._apply_task(gt, "predict_proba")

    def decision_function(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        """Apply ``estimator.decision_function`` and unflatten the result.

        See :meth:`transform` for the reshaping and carrier contract.
        """
        return self._apply_task(gt, "decision_function")

    def inverse_transform(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        """Apply ``estimator.inverse_transform`` and unflatten the result.

        See :meth:`transform` for the reshaping and carrier contract.
        """
        return self._apply_task(gt, "inverse_transform")

    def fit_predict(self, gt: GeoTensor | np.ndarray) -> GeoTensor | np.ndarray:
        """Apply ``estimator.fit_predict`` and unflatten the result.

        Fitting uses the ``nan_fit`` strategy (like :meth:`fit`); see
        :meth:`transform` for the reshaping and carrier contract.

        Raises:
            TypeError: If the wrapped estimator has no ``fit_predict``.
        """
        if not hasattr(self.estimator, "fit_predict"):
            raise TypeError(
                f"{type(self.estimator).__name__} does not support fit_predict"
            )
        flat = self._flatten(gt)
        x_fit, valid = self._prepare_fit(flat)
        y_fit = self.estimator.fit_predict(x_fit)
        self.is_fitted = True
        self.fit_geotensor_shape = tuple(np.asarray(gt).shape)
        self.fit_n_samples = int(x_fit.shape[0])
        return self._unflatten_apply_result(gt, flat, y_fit, valid)

    def save_state(self, path: str | Path, *, write_meta: bool = False) -> None:
        """Persist the fitted estimator and any fitted imputer with joblib.

        The file is a joblib pickle: load it only where you would run code
        from its author (see :meth:`load_state`).

        Args:
            path: Destination file (parent directories are created).
            write_meta: Also write a ``<path>.meta.json`` sidecar recording
                the scikit-learn version, the fit input shape / sample
                count and a UTC ``fit_timestamp``. Off by default so saving
                is deterministic and writes exactly one file.
        """
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
        if write_meta:
            _write_metadata(
                state_path,
                fit_geotensor_shape=self.fit_geotensor_shape,
                fit_n_samples=self.fit_n_samples,
            )

    def load_state(self, path: str | Path) -> None:
        """Load a fitted estimator and imputer previously saved with joblib.

        Warning:
            ``joblib.load`` unpickles the file, and unpickling can execute
            arbitrary code. Only load state files you created yourself or
            obtained from a source you trust as much as the code you run;
            never load one from an untrusted upload, download or share.
        """
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

    def _flatten(self, gt: GeoTensor | np.ndarray) -> _FlatGeoTensor:
        arr = np.asarray(gt)
        axes = resolve_axes(arr.ndim, self.mode, self.sample_axes, self.feature_axes)
        x, layout = cube_to_samples(
            arr, band_axis=axes.feature_axes, sample_axes=axes.sample_axes
        )
        invalid, _ = cube_to_samples(
            invalid_values(gt),
            band_axis=axes.feature_axes,
            sample_axes=axes.sample_axes,
        )
        if invalid.any():
            # Fill values become NaN so imputers / NaN-tolerant estimators
            # (``propagate_raw``) recognise them as missing.
            x = as_float(x).copy()
            x[invalid] = np.nan
        return _FlatGeoTensor(x=x, valid=~invalid.any(axis=1), layout=layout)

    def _prepare_fit(
        self, flat: _FlatGeoTensor, *, reuse_imputer: bool = False
    ) -> tuple[Num[np.ndarray, "m c"], Bool[np.ndarray, " n"]]:
        strategy = self.nan_fit
        x, valid = flat.x, flat.valid
        if strategy == "error" and not valid.all():
            raise ValueError(
                "GeoTensorEstimator.fit received NaN / non-finite / fill values"
            )
        if strategy == "error":
            return x, valid
        if strategy == "drop":
            return x[valid], valid
        if strategy in {"propagate", "propagate_raw"}:
            # At fit time both pass the array through unchanged; the
            # apply-time behavior diverges (see ``_prepare_transform``).
            return x, np.ones(x.shape[0], dtype=bool)
        if reuse_imputer and self.imputer is not None:
            return self.imputer.transform(x), np.ones(x.shape[0], dtype=bool)
        self.imputer = _make_imputer(
            strategy,
            simple_strategy=self.impute_simple_strategy,
            knn_n_neighbors=self.impute_knn_n_neighbors,
            iterative_max_iter=self.impute_iterative_max_iter,
        )
        return self.imputer.fit_transform(x), np.ones(x.shape[0], dtype=bool)

    def _prepare_transform(
        self, flat: _FlatGeoTensor
    ) -> tuple[Num[np.ndarray, "m c"], Bool[np.ndarray, " n"]]:
        """Apply ``nan_transform`` strategy to apply-time input.

        Invalid rows are those with a NaN, non-finite or fill element.
        ``"drop"`` strips invalid rows and the unflatten step truncates the
        sample axis to the valid rows only (output has fewer samples).
        ``"propagate"`` also strips invalid rows from the estimator input
        but the unflatten step writes the output fill (``NaN`` or
        ``label_fill_value``) at those positions so the spatial layout is
        preserved -- callers using raster carriers want this.
        ``"propagate_raw"`` passes rows through unchanged (fill elements
        as ``NaN``) so NaN-tolerant estimators (e.g. imputers) can see the
        missing values.
        Imputer strategies require a previously fitted imputer to avoid
        leaking inference-batch statistics into preprocessing.
        """
        strategy = self.nan_transform
        x, valid = flat.x, flat.valid
        if strategy == "error" and not valid.all():
            raise ValueError(
                "GeoTensorEstimator.transform received NaN / non-finite / fill values"
            )
        if strategy == "error":
            return x, valid
        if strategy == "drop":
            return x[valid], valid
        if strategy == "propagate":
            # Strip invalid rows from the estimator input; unflatten will
            # refill those positions with the output fill value.
            return x[valid], valid
        if strategy == "propagate_raw":
            # Pass rows through unchanged so NaN-tolerant estimators receive
            # the NaNs directly.
            return x, np.ones(x.shape[0], dtype=bool)
        if self.imputer is None:
            raise RuntimeError(
                f"nan_transform={strategy!r} requires a fitted imputer. Fit "
                f"the estimator first with a matching nan_fit strategy, or "
                f"load a saved state via load_state(). Fitting a fresh "
                f"imputer at transform time would leak inference-batch "
                f"statistics into preprocessing."
            )
        return self.imputer.transform(x), np.ones(x.shape[0], dtype=bool)

    def _apply_task(
        self, gt: GeoTensor | np.ndarray, task: Task
    ) -> GeoTensor | np.ndarray:
        if not hasattr(self.estimator, task):
            raise TypeError(f"{type(self.estimator).__name__} does not support {task}")
        flat = self._flatten(gt)
        x_apply, valid = self._prepare_transform(flat)
        y_apply = getattr(self.estimator, task)(x_apply)
        if self.nan_transform == "drop" and not valid.all():
            # ``drop`` returns only the rows the estimator actually saw,
            # collapsing the sample axis -- the result can no longer be
            # reshaped into the original spatial layout, so we return a
            # bare ndarray. Use ``propagate`` to preserve the layout.
            return np.asarray(y_apply)
        return self._unflatten_apply_result(gt, flat, y_apply, valid)

    def _unflatten_apply_result(
        self,
        gt: GeoTensor | np.ndarray,
        flat: _FlatGeoTensor,
        y_apply: Shaped[np.ndarray, " m"] | Shaped[np.ndarray, "m k"],
        valid: Bool[np.ndarray, " n"],
    ) -> GeoTensor | np.ndarray:
        y = np.asarray(y_apply)
        fill_value = output_fill_value(y.dtype, self.label_fill_value)
        if valid.all():
            dense = y
        else:
            out_shape = (valid.shape[0], *y.shape[1:])
            if fill_value is None:
                dtype = np.result_type(y.dtype, float)
            elif y.dtype.kind in "iu":
                # Widen (e.g. uint8 labels with a -1 fill) only when needed.
                dtype = np.promote_types(y.dtype, np.min_scalar_type(fill_value))
            else:
                dtype = y.dtype
            dense = np.full(
                out_shape, np.nan if fill_value is None else fill_value, dtype=dtype
            )
            dense[valid] = y
        out = _restore_shape(dense, flat)
        # When the trailing axes do not match the input spatial (H, W),
        # ``GeoTensor.array_as_geotensor`` cannot attach CRS/transform
        # meaningfully (e.g. sample-only outputs from ``mode="custom"`` or
        # ``mode="spectral"``). Return a bare ndarray in that case; callers
        # / wrappers are responsible for re-attaching geo metadata.
        if out.ndim < 2 or out.shape[-2:] != np.asarray(gt).shape[-2:]:
            return out
        return wrap_like(
            gt,
            out,
            fill_value_default=np.nan if fill_value is None else fill_value,
            band_names=self.out_band_names,
        )


class _FlatGeoTensor:
    def __init__(
        self,
        *,
        x: Num[np.ndarray, "n c"],
        valid: Bool[np.ndarray, " n"],
        layout: SampleLayout,
    ) -> None:
        self.x = x
        self.valid = valid
        self.layout = layout


def _restore_shape(
    y: Shaped[np.ndarray, " n"] | Shaped[np.ndarray, "n k"],
    flat: _FlatGeoTensor,
) -> np.ndarray:
    # Sample axes return to their input order; a 2-D output's features
    # become one channel axis at the first feature axis (or in front when
    # there is none, e.g. ``"pixel"`` on a single-band ``(H, W)``
    # raster), so the result stays channel-first ``(k, H, W)``.
    if y.ndim not in (1, 2):
        raise ValueError("Estimator output must be 1-D or 2-D")
    return samples_to_cube(y, flat.layout)


def _validate_nan_strategy(strategy: NanStrategy) -> None:
    valid = {"drop", "propagate", "propagate_raw", "error", *_IMPUTE_STRATEGIES}
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
        # Importing this module intentionally enables sklearn's experimental
        # IterativeImputer before importing the estimator class.
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
