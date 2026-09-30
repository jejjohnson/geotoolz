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

from geotoolz.learn._src.estimators import GeoTensorEstimator
from geotoolz.learn._src.operators import (
    GMM,
    IPCA,
    NMF,
    PCA,
    IsolationForest,
    IterativeImputer,
    KMeans,
    KNNImputer,
    LocalOutlierFactor,
    MiniBatchKMeans,
    ModelOp,
    OneClassSVM,
    SklearnOp,
)


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
    "ModelOp",
    "OneClassSVM",
    "SklearnOp",
]
