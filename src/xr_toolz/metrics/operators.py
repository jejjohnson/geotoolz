"""Layer-1 ``Operator`` wrappers for evaluation metrics.

This module re-exports the operator classes from
:mod:`xr_toolz.metrics._src.pixel` and friends for ergonomic
``from xr_toolz.metrics.operators import RMSE`` access.
"""

from xr_toolz.metrics._src.distributional import (
    CRPS,
    EnergyDistance,
    Wasserstein1,
)
from xr_toolz.metrics._src.forecast import SkillByLeadTime
from xr_toolz.metrics._src.masked import MaskedMetric
from xr_toolz.metrics._src.multiscale import EvaluateByRegion
from xr_toolz.metrics._src.physical import (
    DensityInversionFraction,
    DivergenceError,
    GeostrophicBalanceError,
    PVConservationError,
)
from xr_toolz.metrics._src.pixel import (
    MAE,
    MSE,
    NRMSE,
    RMSE,
    Bias,
    Correlation,
    R2Score,
)
from xr_toolz.metrics._src.probabilistic import (
    EnsembleCoverage,
    RankHistogram,
    ReliabilityCurve,
    SpreadSkillRatio,
)
from xr_toolz.metrics._src.segmented_psd import SegmentedPSDScore
from xr_toolz.metrics._src.residuals import BinnedResiduals2D, RegionScores
from xr_toolz.metrics._src.spectral import (
    BandLimitedRMSE,
    FrequencyBandSkill,
    PSDScore,
)
from xr_toolz.metrics._src.structural import (
    SSIM,
    CentroidDisplacement,
    GradientDifference,
    PhaseShiftError,
)


__all__ = [
    "CRPS",
    "MAE",
    "MSE",
    "NRMSE",
    "RMSE",
    "SSIM",
    "BandLimitedRMSE",
    "Bias",
    "BinnedResiduals2D",
    "CentroidDisplacement",
    "Correlation",
    "DensityInversionFraction",
    "DivergenceError",
    "EnergyDistance",
    "EnsembleCoverage",
    "EvaluateByRegion",
    "FrequencyBandSkill",
    "GeostrophicBalanceError",
    "GradientDifference",
    "MaskedMetric",
    "PSDScore",
    "PVConservationError",
    "PhaseShiftError",
    "R2Score",
    "RankHistogram",
    "RegionScores",
    "ReliabilityCurve",
    "SegmentedPSDScore",
    "SkillByLeadTime",
    "SpreadSkillRatio",
    "Wasserstein1",
]
