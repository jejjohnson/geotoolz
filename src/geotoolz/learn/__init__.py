"""scikit-learn integration for GeoTensor operator pipelines."""

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
