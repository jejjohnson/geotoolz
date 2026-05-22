"""scikit-learn integration for GeoTensor operator pipelines.

.. warning::

    **Phase-1 API — subject to change.** This module ships a universal
    ``SklearnOp`` adapter plus named-algorithm convenience wrappers as a
    first cut. The full design (tracked in geotoolz issue #32) calls for
    a ``PixelTable`` carrier, first-class shape adapters (``ToPixelMajor``,
    ``ToTemporalPixelMajor``, ``ToChipMajor``), and type-named estimator
    wrappers (``PixelwiseClassifier`` / ``Regressor`` / ``Transformer`` /
    ``Clusterer`` / ``Proba`` / ``Decision``) backed by a ``NanPolicy``
    dataclass. Those are not in this PR; the current ``mode=`` /
    ``nan_fit=`` / ``nan_transform=`` keyword surface will be replaced
    with that vocabulary in a follow-up, with deprecation shims.

    **Supervised estimators (classifiers / regressors) must be pre-fit**
    out-of-graph and loaded via ``state_path=``. The operator's in-graph
    fit modes (``fit_on_call`` / ``refit`` / ``fit_streaming``) take a
    single ``GeoTensor`` and have no path for a label tensor ``y``.
    Out-of-graph fitting matches the design doc's preferred pattern; the
    follow-up issue tracks adding fit helpers (``fit_pixelwise``,
    ``fit_pixelwise_incremental``) for the inside-the-graph case.
"""

from __future__ import annotations

from typing import Any

from geotoolz.learn._src.estimators import GeoTensorEstimator
from geotoolz.learn._src.operators import SklearnOp


class PCA(SklearnOp):
    """Pixel-wise PCA convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "transform")
        kwargs.setdefault("nan_fit", "drop")
        super().__init__(estimator, **kwargs)


class IPCA(SklearnOp):
    """Streaming IncrementalPCA convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("fit_mode", "fit_streaming")
        kwargs.setdefault("task", "transform")
        super().__init__(estimator, **kwargs)


class NMF(SklearnOp):
    """Pixel-wise NMF convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "transform")
        super().__init__(estimator, **kwargs)


class KMeans(SklearnOp):
    """Pixel-wise KMeans label convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "predict")
        super().__init__(estimator, **kwargs)


class MiniBatchKMeans(SklearnOp):
    """Streaming MiniBatchKMeans convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("fit_mode", "fit_streaming")
        kwargs.setdefault("task", "predict")
        super().__init__(estimator, **kwargs)


class GMM(SklearnOp):
    """Gaussian mixture convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "predict_proba")
        super().__init__(estimator, **kwargs)


class IsolationForest(SklearnOp):
    """Pixel-wise IsolationForest anomaly-score convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "decision_function")
        super().__init__(estimator, **kwargs)


class OneClassSVM(SklearnOp):
    """Pixel-wise OneClassSVM anomaly-score convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "decision_function")
        super().__init__(estimator, **kwargs)


class LocalOutlierFactor(SklearnOp):
    """Pixel-wise LocalOutlierFactor convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "decision_function")
        super().__init__(estimator, **kwargs)


class KNNImputer(SklearnOp):
    """Pixel-wise KNNImputer convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "transform")
        kwargs.setdefault("nan_fit", "propagate")
        kwargs.setdefault("nan_transform", "propagate")
        super().__init__(estimator, **kwargs)


class IterativeImputer(SklearnOp):
    """Pixel-wise IterativeImputer convenience operator."""

    def __init__(self, estimator: Any, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "pixel")
        kwargs.setdefault("task", "transform")
        kwargs.setdefault("nan_fit", "propagate")
        kwargs.setdefault("nan_transform", "propagate")
        super().__init__(estimator, **kwargs)


__all__ = [
    "GMM",
    "IPCA",
    "NMF",
    "PCA",
    "GeoTensorEstimator",
    "IsolationForest",
    "IterativeImputer",
    "KMeans",
    "KNNImputer",
    "LocalOutlierFactor",
    "MiniBatchKMeans",
    "OneClassSVM",
    "SklearnOp",
]
