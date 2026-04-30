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

from xr_toolz.metrics._src.forecast import SkillByLeadTime, skill_by_lead_time
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
from xr_toolz.metrics._src.spectral import (
    PSDScore,
    find_intercept_1D,
    psd_error,
    psd_score,
    resolved_scale,
)


__all__ = [
    "MAE",
    "MSE",
    "NRMSE",
    "RMSE",
    "Bias",
    "Correlation",
    "EvaluateByRegion",
    "PSDScore",
    "R2Score",
    "SkillByLeadTime",
    "bias",
    "correlation",
    "evaluate_by_region",
    "find_intercept_1D",
    "mae",
    "mse",
    "normalize_regions",
    "nrmse",
    "psd_error",
    "psd_score",
    "r2_score",
    "resolved_scale",
    "rmse",
    "skill_by_lead_time",
]
