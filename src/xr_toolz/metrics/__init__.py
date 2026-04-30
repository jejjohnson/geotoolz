"""Evaluation metrics — pixel, spectral, and (forthcoming) view-specific.

Submodules group metrics by *scientific diagnostic family*:

- :mod:`xr_toolz.metrics._src.pixel` — pointwise (mse, rmse, …)
- :mod:`xr_toolz.metrics._src.spectral` — PSD-based scores
- :mod:`xr_toolz.metrics._src.multiscale`, :mod:`forecast` — V1
- :mod:`xr_toolz.metrics._src.structural`, :mod:`probabilistic`,
  :mod:`distributional`, :mod:`masked` — V2
- :mod:`xr_toolz.metrics._src.lagrangian` — V3
- :mod:`xr_toolz.metrics._src.physical` — V4
- :mod:`xr_toolz.metrics._src.object` — V5

Layer-1 ``Operator`` wrappers are re-exported flat from this package
and from :mod:`xr_toolz.metrics.operators`.
"""

from xr_toolz.metrics._src.distributional import (
    CRPS,
    EnergyDistance,
    Wasserstein1,
    crps_ensemble,
    energy_distance,
    wasserstein_1,
)
from xr_toolz.metrics._src.forecast import SkillByLeadTime, skill_by_lead_time
from xr_toolz.metrics._src.masked import MaskedMetric, masked_metric
from xr_toolz.metrics._src.multiscale import (
    EvaluateByRegion,
    evaluate_by_region,
    normalize_regions,
)
from xr_toolz.metrics._src.pixel import (
    MAE,
    MSE,
    NRMSE,
    RMSE,
    Bias,
    Correlation,
    R2Score,
    bias,
    correlation,
    mae,
    mse,
    nrmse,
    r2_score,
    rmse,
)
from xr_toolz.metrics._src.probabilistic import (
    EnsembleCoverage,
    RankHistogram,
    ReliabilityCurve,
    SpreadSkillRatio,
    ensemble_coverage,
    rank_histogram,
    reliability_curve,
    spread_skill_ratio,
)
from xr_toolz.metrics._src.spectral import (
    PSDScore,
    find_intercept_1D,
    psd_error,
    psd_score,
    resolved_scale,
)
from xr_toolz.metrics._src.structural import (
    SSIM,
    CentroidDisplacement,
    GradientDifference,
    PhaseShiftError,
    centroid_displacement,
    gradient_difference,
    phase_shift_error,
    ssim,
)


__all__ = [
    "CRPS",
    "MAE",
    "MSE",
    "NRMSE",
    "RMSE",
    "SSIM",
    "Bias",
    "CentroidDisplacement",
    "Correlation",
    "EnergyDistance",
    "EnsembleCoverage",
    "EvaluateByRegion",
    "GradientDifference",
    "MaskedMetric",
    "PSDScore",
    "PhaseShiftError",
    "R2Score",
    "RankHistogram",
    "ReliabilityCurve",
    "SkillByLeadTime",
    "SpreadSkillRatio",
    "Wasserstein1",
    "bias",
    "centroid_displacement",
    "correlation",
    "crps_ensemble",
    "energy_distance",
    "ensemble_coverage",
    "evaluate_by_region",
    "find_intercept_1D",
    "gradient_difference",
    "mae",
    "masked_metric",
    "mse",
    "normalize_regions",
    "nrmse",
    "phase_shift_error",
    "psd_error",
    "psd_score",
    "r2_score",
    "rank_histogram",
    "reliability_curve",
    "resolved_scale",
    "rmse",
    "skill_by_lead_time",
    "spread_skill_ratio",
    "ssim",
    "wasserstein_1",
]
