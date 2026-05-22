"""Layer-1 ``Operator`` wrappers for evaluation metrics.

This module re-exports the operator classes from
:mod:`xrtoolz.metrics._src.pixel` and friends for ergonomic
``from xrtoolz.metrics.operators import RMSE`` access.
"""

from xrtoolz.metrics._src.distributional import (
    CRPS,
    EnergyDistance,
    Wasserstein1,
)
from xrtoolz.metrics._src.forecast import SkillByLeadTime
from xrtoolz.metrics._src.masked import MaskedMetric
from xrtoolz.metrics._src.multiscale import EvaluateByRegion
from xrtoolz.metrics._src.physical import (
    DensityInversionFraction,
    DivergenceError,
    GeostrophicBalanceError,
    PVConservationError,
)
from xrtoolz.metrics._src.pixel import (
    MAE,
    MSE,
    NRMSE,
    RMSE,
    Bias,
    Correlation,
    R2Score,
)
from xrtoolz.metrics._src.probabilistic import (
    EnsembleCoverage,
    RankHistogram,
    ReliabilityCurve,
    SpreadSkillRatio,
)
from xrtoolz.metrics._src.residuals import BinnedResiduals2D, RegionScores
from xrtoolz.metrics._src.segmented_psd import SegmentedPSDScore
from xrtoolz.metrics._src.spectral import (
    BandLimitedRMSE,
    FrequencyBandSkill,
    PSDScore,
    WaveletPSDScore,
)
from xrtoolz.metrics._src.structural import (
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
    "WaveletPSDScore",
]
