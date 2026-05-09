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

from xr_toolz.metrics._src.array_segmented_psd import (
    segment_signal,
    segmented_coherence,
    segmented_csd,
    segmented_psd,
)
from xr_toolz.metrics._src.distributional import (
    CRPS,
    EnergyDistance,
    Wasserstein1,
    crps_ensemble,
    energy_distance,
    wasserstein_1,
)
from xr_toolz.metrics._src.dm import dm_test
from xr_toolz.metrics._src.forecast import SkillByLeadTime, skill_by_lead_time
from xr_toolz.metrics._src.leaderboard import rank_methods
from xr_toolz.metrics._src.masked import MaskedMetric, masked_metric
from xr_toolz.metrics._src.multiscale import (
    EvaluateByRegion,
    evaluate_by_region,
    normalize_regions,
)
from xr_toolz.metrics._src.physical import (
    DensityInversionFraction,
    DivergenceError,
    GeostrophicBalanceError,
    PVConservationError,
    density_inversion_fraction,
    divergence_error,
    geostrophic_balance_error,
    pv_conservation_error,
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
from xr_toolz.metrics._src.residuals import (
    BinnedResiduals2D,
    RegionScores,
    bin_residuals_2d,
    scores_by_region,
)
from xr_toolz.metrics._src.segmented_psd import (
    SegmentedPSDScore,
    along_track_psd_score,
    psd_score_by_region,
)
from xr_toolz.metrics._src.spectral import (
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
    wavelet_psd_score,
    wavelet_resolved_scale_map,
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
    "along_track_psd_score",
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
    "mae",
    "masked_metric",
    "mse",
    "normalize_regions",
    "nrmse",
    "phase_shift_error",
    "psd_error",
    "psd_score",
    "psd_score_by_region",
    "pv_conservation_error",
    "r2_score",
    "rank_histogram",
    "rank_methods",
    "reliability_curve",
    "resolved_scale",
    "rmse",
    "scores_by_region",
    "segment_signal",
    "segmented_coherence",
    "segmented_csd",
    "segmented_psd",
    "skill_by_lead_time",
    "spread_skill_ratio",
    "ssim",
    "wasserstein_1",
    "wavelet_psd_score",
    "wavelet_resolved_scale_map",
]
