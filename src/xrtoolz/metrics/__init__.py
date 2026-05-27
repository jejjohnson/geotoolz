"""Evaluation metrics — pixel, spectral, and (forthcoming) view-specific.

Submodules group metrics by *scientific diagnostic family*:

- :mod:`xrtoolz.metrics._src.pixel` — pointwise (mse, rmse, …)
- :mod:`xrtoolz.metrics._src.spectral` — PSD-based scores
- :mod:`xrtoolz.metrics._src.multiscale`, :mod:`forecast` — V1
- :mod:`xrtoolz.metrics._src.structural`, :mod:`probabilistic`,
  :mod:`distributional`, :mod:`masked` — V2
- :mod:`xrtoolz.metrics._src.lagrangian` — V3
- :mod:`xrtoolz.metrics._src.physical` — V4
- :mod:`xrtoolz.metrics._src.object` — V5

Layer-1 ``Operator`` wrappers are re-exported flat from this package
and from :mod:`xrtoolz.metrics.operators`.
"""

from xrtoolz.metrics._src.composite import psd_score_spacetime, rmse_skill_scores
from xrtoolz.metrics._src.distributional import (
    CRPS,
    EnergyDistance,
    Wasserstein1,
    crps_ensemble,
    energy_distance,
    wasserstein_1,
)
from xrtoolz.metrics._src.dm import dm_test
from xrtoolz.metrics._src.forecast import SkillByLeadTime, skill_by_lead_time
from xrtoolz.metrics._src.instance import (
    AveragePrecisionMatched,
    InstanceF1AtIoU,
    InstanceMatcher,
    MaskIoU,
    average_precision_matched,
    instance_f1_at_iou,
    mask_iou_matrix,
    match_instances,
)
from xrtoolz.metrics._src.leaderboard import rank_methods
from xrtoolz.metrics._src.masked import MaskedMetric, masked_metric
from xrtoolz.metrics._src.multiscale import (
    EvaluateByRegion,
    evaluate_by_region,
    normalize_regions,
)
from xrtoolz.metrics._src.physical import (
    DensityInversionFraction,
    DivergenceError,
    GeostrophicBalanceError,
    PVConservationError,
    density_inversion_fraction,
    divergence_error,
    geostrophic_balance_error,
    pv_conservation_error,
)
from xrtoolz.metrics._src.pixel import (
    MAE,
    MSE,
    NRMSE,
    RMSE,
    Bias,
    Correlation,
    NRMSEScore,
    R2Score,
    bias,
    correlation,
    mae,
    mse,
    nrmse,
    nrmse_score,
    r2_score,
    rmse,
)
from xrtoolz.metrics._src.probabilistic import (
    EnsembleCoverage,
    RankHistogram,
    ReliabilityCurve,
    SpreadSkillRatio,
    ensemble_coverage,
    rank_histogram,
    reliability_curve,
    spread_skill_ratio,
)
from xrtoolz.metrics._src.residuals import (
    BinnedResiduals2D,
    RegionScores,
    bin_residuals_2d,
    scores_by_region,
)
from xrtoolz.metrics._src.segmented_psd import (
    SegmentedPSDScore,
    along_track_psd_score,
    psd_score_by_region,
)
from xrtoolz.metrics._src.spectral import (
    BandLimitedRMSE,
    FrequencyBandSkill,
    PSDScore,
    WaveletPSDScore,
    band_limited_rmse,
    evaluate_by_frequency_band,
    find_intercept_1D,
    find_intercept_2D,
    psd_error,
    psd_score,
    resolved_scale,
    resolved_scale_2d,
    wavelet_psd_score,
    wavelet_resolved_scale_map,
)
from xrtoolz.metrics._src.structural import (
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
    "AveragePrecisionMatched",
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
    "along_track_psd_score",
    "average_precision_matched",
    "band_limited_rmse",
    "bias",
    "bin_residuals_2d",
    "centroid_displacement",
    "correlation",
    "crps_ensemble",
    "density_inversion_fraction",
    "divergence_error",
    "dm_test",
    "energy_distance",
    "ensemble_coverage",
    "evaluate_by_frequency_band",
    "evaluate_by_region",
    "find_intercept_1D",
    "find_intercept_2D",
    "geostrophic_balance_error",
    "gradient_difference",
    "instance_f1_at_iou",
    "mae",
    "mask_iou_matrix",
    "masked_metric",
    "match_instances",
    "mse",
    "normalize_regions",
    "nrmse",
    "nrmse_score",
    "phase_shift_error",
    "psd_error",
    "psd_score",
    "psd_score_by_region",
    "psd_score_spacetime",
    "pv_conservation_error",
    "r2_score",
    "rank_histogram",
    "rank_methods",
    "reliability_curve",
    "resolved_scale",
    "resolved_scale_2d",
    "rmse",
    "rmse_skill_scores",
    "scores_by_region",
    "skill_by_lead_time",
    "spread_skill_ratio",
    "ssim",
    "wasserstein_1",
    "wavelet_psd_score",
    "wavelet_resolved_scale_map",
]
