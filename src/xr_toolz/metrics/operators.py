"""Layer-1 ``Operator`` wrappers for evaluation metrics.

This module re-exports the operator classes from
:mod:`xr_toolz.metrics._src.pixel` and :mod:`xr_toolz.metrics._src.spectral`
for ergonomic ``from xr_toolz.metrics.operators import RMSE`` access.
"""

from xr_toolz.metrics._src.forecast import SkillByLeadTime
from xr_toolz.metrics._src.multiscale import EvaluateByRegion
from xr_toolz.metrics._src.pixel import (
    MAE,
    MSE,
    NRMSE,
    RMSE,
    Bias,
    Correlation,
    R2Score,
)
from xr_toolz.metrics._src.spectral import PSDScore


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
]
