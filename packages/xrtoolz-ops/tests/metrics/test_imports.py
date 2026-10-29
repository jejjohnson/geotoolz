"""Import-surface tests for :mod:`xrops.metrics` — guards against
regressions in the public API as the validation framework lands.
"""

from __future__ import annotations

import importlib

import pytest


# ---- New canonical surface ------------------------------------------------


def test_metrics_package_root_exposes_layer0_and_operators():
    from xrops.metrics import (
        MAE,
        MSE,
        NRMSE,
        RMSE,
        Bias,
        Correlation,
        PSDScore,
        R2Score,
        SegmentedPSDScore,
        along_track_psd_score,
        bias,
        correlation,
        find_intercept_1D,
        mae,
        mse,
        nrmse,
        psd_error,
        psd_score,
        psd_score_by_region,
        psd_score_spacetime,
        r2_score,
        resolved_scale,
        resolved_scale_2d,
        rmse,
        rmse_skill_scores,
    )

    # Sanity-check at least one is callable.
    assert callable(rmse)
    assert RMSE.__name__ == "RMSE"
    assert PSDScore.__name__ == "PSDScore"
    # Hit unused-import suppression via tuple to keep the comprehensive list.
    _ = (
        MAE,
        MSE,
        NRMSE,
        Bias,
        Correlation,
        R2Score,
        SegmentedPSDScore,
        along_track_psd_score,
        bias,
        correlation,
        find_intercept_1D,
        mae,
        mse,
        nrmse,
        psd_error,
        psd_score,
        psd_score_spacetime,
        psd_score_by_region,
        r2_score,
        resolved_scale,
        resolved_scale_2d,
        rmse_skill_scores,
    )


def test_metrics_pixel_submodule_imports():
    from xrops.metrics.pixel import (
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

    assert callable(mse)
    _ = (
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
        nrmse,
        r2_score,
        rmse,
    )


def test_metrics_spectral_submodule_imports():
    from xrops.metrics.spectral import (
        BandLimitedRMSE,
        FrequencyBandSkill,
        PSDScore,
        SegmentedPSDScore,
        along_track_psd_score,
        band_limited_rmse,
        evaluate_by_frequency_band,
        find_intercept_1D,
        psd_error,
        psd_score,
        psd_score_by_region,
        psd_score_spacetime,
        resolved_scale,
        resolved_scale_2d,
    )

    assert callable(psd_score)
    assert callable(evaluate_by_frequency_band)
    assert callable(band_limited_rmse)
    _ = (
        PSDScore,
        SegmentedPSDScore,
        FrequencyBandSkill,
        BandLimitedRMSE,
        along_track_psd_score,
        find_intercept_1D,
        psd_error,
        psd_score_spacetime,
        psd_score_by_region,
        resolved_scale,
        resolved_scale_2d,
    )


def test_metrics_structural_submodule_imports():
    from xrops.metrics.structural import (
        SSIM,
        CentroidDisplacement,
        GradientDifference,
        PhaseShiftError,
        centroid_displacement,
        gradient_difference,
        phase_shift_error,
        ssim,
    )

    assert callable(ssim)
    _ = (
        SSIM,
        CentroidDisplacement,
        GradientDifference,
        PhaseShiftError,
        centroid_displacement,
        gradient_difference,
        phase_shift_error,
    )


def test_metrics_probabilistic_submodule_imports():
    from xrops.metrics.probabilistic import (
        EnsembleCoverage,
        RankHistogram,
        ReliabilityCurve,
        SpreadSkillRatio,
        ensemble_coverage,
        rank_histogram,
        reliability_curve,
        spread_skill_ratio,
    )

    assert callable(spread_skill_ratio)
    _ = (
        EnsembleCoverage,
        RankHistogram,
        ReliabilityCurve,
        SpreadSkillRatio,
        ensemble_coverage,
        rank_histogram,
        reliability_curve,
    )


def test_metrics_distributional_submodule_imports():
    from xrops.metrics.distributional import (
        CRPS,
        EnergyDistance,
        Wasserstein1,
        crps_ensemble,
        energy_distance,
        wasserstein_1,
    )

    assert callable(crps_ensemble)
    _ = (CRPS, EnergyDistance, Wasserstein1, energy_distance, wasserstein_1)


def test_metrics_masked_submodule_imports():
    from xrops.metrics.masked import MaskedMetric, masked_metric

    assert callable(masked_metric)
    _ = (MaskedMetric,)


def test_metrics_operators_submodule_imports():
    from xrops.metrics.operators import (
        MAE,
        MSE,
        NRMSE,
        RMSE,
        Bias,
        Correlation,
        PSDScore,
        R2Score,
    )

    assert RMSE.__name__ == "RMSE"
    _ = (MAE, MSE, NRMSE, PSDScore, Bias, Correlation, R2Score)


@pytest.mark.parametrize(
    "submodule",
    [
        "lagrangian",
    ],
)
def test_metrics_view_stub_submodules_are_importable(submodule):
    """Importing any V1–V5 stub submodule must succeed even before its
    epic lands its bodies — this is what unblocks the additive merge order.
    The stub must also expose no public names today; the V epic that
    fills it in is responsible for the public surface.

    ``object`` is excluded: F1.3 pre-reserves the canonical long-form
    V5 class names there. See ``test_metrics_object_reserved_names``.
    ``physical`` is excluded: V4.1 has landed its public surface.
    """
    mod = importlib.import_module(f"xrops.metrics.{submodule}")
    public_names = [n for n in dir(mod) if not n.startswith("_")]
    assert public_names == [], (
        f"xrops.metrics.{submodule} unexpectedly exports public names: {public_names}"
    )


def test_metrics_physical_submodule_imports():
    from xrops.metrics.physical import (
        DensityInversionFraction,
        DivergenceError,
        GeostrophicBalanceError,
        PVConservationError,
        density_inversion_fraction,
        divergence_error,
        geostrophic_balance_error,
        pv_conservation_error,
    )

    assert callable(geostrophic_balance_error)
    _ = (
        DensityInversionFraction,
        DivergenceError,
        GeostrophicBalanceError,
        PVConservationError,
        density_inversion_fraction,
        divergence_error,
        pv_conservation_error,
    )


_RESERVED_OBJECT_CLASSES = (
    "ProbabilityOfDetection",
    "FalseAlarmRatio",
    "CriticalSuccessIndex",
    "IntersectionOverUnion",
    "DurationError",
    "IntensityBias",
    "CentroidDistance",
)


def test_metrics_object_exposes_all_reserved_names():
    """``metrics.object`` reserves the V5 long-form names per D14."""
    import xrops.metrics.object as obj

    assert set(_RESERVED_OBJECT_CLASSES) <= set(dir(obj))


@pytest.mark.parametrize("name", _RESERVED_OBJECT_CLASSES)
def test_metrics_object_reserved_class_raises_notimplementederror(name):
    """Each V5-reserved class must raise ``NotImplementedError`` on
    instantiation, with a message that names the class and points at V5.
    """
    from xrops.metrics import object as obj_module

    cls = getattr(obj_module, name)
    with pytest.raises(NotImplementedError) as exc_info:
        cls()
    msg = str(exc_info.value)
    assert name in msg, f"error message for {name} does not name the class: {msg!r}"
    assert "V5" in msg, f"error message for {name} does not point at V5: {msg!r}"


# ---- The old xrtoolz.geo metric paths are gone (no shims, #359) ----------


def test_metrics_are_not_reexported_from_geo():
    """Metrics live only in :mod:`xrops.metrics`."""
    import xrops.geo as geo
    import xrops.geo.operators as geo_ops

    for name in ("rmse", "mse", "psd_score"):
        assert not hasattr(geo, name)
    for name in ("RMSE", "MSE", "PSDScore"):
        assert not hasattr(geo_ops, name)
    for module in ("xrops.geo._src.metrics", "xrops.geo.plot"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
