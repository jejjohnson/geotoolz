"""Layer-1 ``Operator`` wrappers for evaluation metrics.

This module re-exports the operator classes from
:mod:`xrops.metrics._src.pixel` and friends for ergonomic
``from xrops.metrics.operators import RMSE`` access.
"""

from xrops.metrics._src.distributional import (
    CRPS,
    EnergyDistance,
    Wasserstein1,
)
from xrops.metrics._src.dm import DieboldMariano
from xrops.metrics._src.forecast import SkillByLeadTime
from xrops.metrics._src.instance import (
    AveragePrecisionMatched,
    InstanceF1AtIoU,
    InstanceMatcher,
    MaskIoU,
)
from xrops.metrics._src.masked import MaskedMetric
from xrops.metrics._src.multiscale import EvaluateByRegion
from xrops.metrics._src.physical import (
    DensityInversionFraction,
    DivergenceError,
    GeostrophicBalanceError,
    GeostrophicImbalance,
    PVConservationError,
)
from xrops.metrics._src.pixel import (
    MAE,
    MSE,
    NRMSE,
    RMSE,
    Bias,
    Correlation,
    NRMSEScore,
    R2Score,
)
from xrops.metrics._src.probabilistic import (
    EnsembleCoverage,
    RankHistogram,
    ReliabilityCurve,
    SpreadSkillRatio,
)
from xrops.metrics._src.residuals import BinnedResiduals2D, RegionScores
from xrops.metrics._src.segmented_psd import SegmentedPSDScore
from xrops.metrics._src.spectral import (
    BandLimitedRMSE,
    FrequencyBandSkill,
    PSDScore,
    WaveletPSDScore,
)
from xrops.metrics._src.structural import (
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
    "AveragePrecisionMatched",
    "BandLimitedRMSE",
    "Bias",
    "BinnedResiduals2D",
    "CentroidDisplacement",
    "Correlation",
    "DensityInversionFraction",
    "DieboldMariano",
    "DivergenceError",
    "EnergyDistance",
    "EnsembleCoverage",
    "EvaluateByRegion",
    "FrequencyBandSkill",
    "GeostrophicBalanceError",
    "GeostrophicImbalance",
    "GradientDifference",
    "InstanceF1AtIoU",
    "InstanceMatcher",
    "MaskIoU",
    "MaskedMetric",
    "NRMSEScore",
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
