"""Evaluation metrics, grouped by scientific diagnostic family.

- :mod:`xrops.metrics.pixel` — pointwise scores (mse, rmse, bias, …)
- :mod:`xrops.metrics.spectral` — PSD-based scores and resolved scale
- :mod:`xrops.metrics.multiscale` / :mod:`forecast` — region and
  lead-time composites
- :mod:`xrops.metrics.structural` / :mod:`probabilistic` /
  :mod:`distributional` / :mod:`masked` — image structure, ensemble,
  distribution, and masked-domain scores
- :mod:`xrops.metrics.physical` — physical-consistency diagnostics
- :mod:`xrops.metrics.object` — object-based verification
- :mod:`xrops.metrics.lagrangian` — reserved namespace (no content yet)

Families without a facade module (Diebold–Mariano, instance matching,
residual binning, leaderboard, composite skill scores) are exported flat
from this package. Layer-1 ``Operator`` wrappers are re-exported flat
from this package and from :mod:`xrops.metrics.operators`.
"""

from xrops.metrics._src.composite import psd_score_spacetime, rmse_skill_scores
from xrops.metrics._src.distributional import (
    CRPS,
    EnergyDistance,
    Wasserstein1,
    crps_ensemble,
    energy_distance,
    wasserstein_1,
)
from xrops.metrics._src.dm import DieboldMariano, dm_test
from xrops.metrics._src.forecast import SkillByLeadTime, skill_by_lead_time
from xrops.metrics._src.instance import (
    AveragePrecisionMatched,
    InstanceF1AtIoU,
    InstanceMatcher,
    MaskIoU,
    average_precision_matched,
    instance_f1_at_iou,
    mask_iou_matrix,
    match_instances,
)
from xrops.metrics._src.leaderboard import rank_methods
from xrops.metrics._src.masked import MaskedMetric, masked_metric
from xrops.metrics._src.multiscale import (
    EvaluateByRegion,
    evaluate_by_region,
    normalize_regions,
)
from xrops.metrics._src.physical import (
    DensityInversionFraction,
    DivergenceError,
    GeostrophicBalanceError,
    GeostrophicImbalance,
    PVConservationError,
    density_inversion_fraction,
    divergence_error,
    geostrophic_balance_error,
    geostrophic_imbalance,
    pv_conservation_error,
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
    bias,
    correlation,
    mae,
    mse,
    nrmse,
    nrmse_score,
    r2_score,
    rmse,
)
from xrops.metrics._src.probabilistic import (
    EnsembleCoverage,
    RankHistogram,
    ReliabilityCurve,
    SpreadSkillRatio,
    ensemble_coverage,
    rank_histogram,
    reliability_curve,
    spread_skill_ratio,
)
from xrops.metrics._src.residuals import (
    BinnedResiduals2D,
    RegionScores,
    bin_residuals_2d,
    scores_by_region,
)
from xrops.metrics._src.segmented_psd import (
    SegmentedPSDScore,
    along_track_psd_score,
    psd_score_by_region,
)
from xrops.metrics._src.spectral import (
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
from xrops.metrics._src.structural import (
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
    "geostrophic_imbalance",
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
