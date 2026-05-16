"""Tests for plume detection and quantification operators."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

import geotoolz as gz
from geotoolz.plume import convert_column_units, resolve_threshold


def _gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0),
        crs="EPSG:32629",
    )


def test_plume_module_exports_at_top_level() -> None:
    assert gz.plume.PlumeMask is gz.PlumeMask
    assert gz.plume.ColumnToMass is gz.ColumnToMass


def test_plume_mask_otsu_matches_resolved_threshold() -> None:
    values = np.r_[np.zeros(50), np.full(50, 10.0)].reshape(10, 10)
    gt = _gt(values)

    out = gz.plume.PlumeMask(threshold="otsu", min_area=1)(gt)

    assert np.array_equal(np.asarray(out), values > resolve_threshold(values, "otsu"))
    assert out.transform == gt.transform
    assert str(out.crs) == "EPSG:32629"


def test_plume_mask_percentile_keeps_top_half_percent() -> None:
    values = np.arange(10_000, dtype=float).reshape(100, 100)

    out = gz.plume.PlumeMask(threshold="percentile:99.5", min_area=1)(_gt(values))

    assert int(np.asarray(out).sum()) == 50


def test_plume_contours_labels_connected_components() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0:2, 0:2] = True
    mask[4, 4] = True

    labels = gz.plume.PlumeContours(min_area=2)(_gt(mask))

    assert set(np.unique(np.asarray(labels))) == {0, 1}
    assert np.asarray(labels)[0, 0] == 1
    assert np.asarray(labels)[4, 4] == 0


def test_plume_footprint_area_and_enhancement_stats() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:4] = True
    enhancement = np.arange(16, dtype=float).reshape(4, 4)

    gdf = gz.plume.PlumeFootprint(
        min_area_m2=1.0,
        simplify_tolerance=None,
        enhancement=_gt(enhancement),
    )(_gt(mask))

    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row.area_m2 == pytest.approx(6 * 100.0)
    assert row.n_pixels == 6
    assert row.mean_enhancement == pytest.approx(float(enhancement[mask].mean()))
    assert row.max_enhancement == pytest.approx(float(enhancement[mask].max()))
    assert gdf.crs == "EPSG:32629"

    no_stats = gz.plume.PlumeFootprint(
        min_area_m2=1.0,
        simplify_tolerance=None,
    )(_gt(mask))
    assert no_stats.iloc[0].mean_enhancement is None
    assert no_stats.iloc[0].max_enhancement is None


def test_wind_advection_cone_follows_wind_orientation() -> None:
    transform = rasterio.Affine.translation(-5.5, 5.5) * rasterio.Affine.scale(
        1.0, -1.0
    )
    gt = GeoTensor(values=np.zeros((11, 11)), transform=transform, crs="EPSG:32629")

    east = gz.plume.WindAdvectionCone(
        source=(0.0, 0.0),
        wind_u=1.0,
        wind_v=0.0,
        half_angle_deg=20.0,
        max_distance=5.0,
    )(gt)
    north = gz.plume.WindAdvectionCone(
        source=(0.0, 0.0),
        wind_u=0.0,
        wind_v=1.0,
        half_angle_deg=20.0,
        max_distance=5.0,
    )(gt)

    assert bool(np.asarray(east)[5, 8])
    assert not bool(np.asarray(east)[2, 5])
    assert bool(np.asarray(north)[2, 5])
    assert not bool(np.asarray(north)[5, 8])


def test_ime_estimate_uses_integrated_mass_and_max_axis_length() -> None:
    mask = _gt(np.array([[True, True, True]]))
    enhancement = _gt(np.array([[2.0, 2.0, 2.0]]))

    estimate = gz.plume.IMEEstimate(
        plume_mask=mask,
        wind_speed=2.0,
        return_uncertainty=False,
    )(enhancement)

    assert estimate["ime_kg"] == pytest.approx(600.0)
    assert estimate["length_m"] == pytest.approx(20.0)
    assert estimate["emission_rate_kg_s"] == pytest.approx(60.0)


def test_ime_skeleton_length_and_uncertainty_fraction() -> None:
    mask_arr = np.array([[True, True], [False, True]])
    mask = _gt(mask_arr)
    enhancement = _gt(np.ones((2, 2), dtype=float))

    estimate = gz.plume.IMEEstimate(
        plume_mask=mask,
        wind_speed=2.0,
        length_method="skeleton",
        uncertainty_fraction=0.2,
    )(enhancement)
    max_axis = gz.plume.IMEEstimate(
        plume_mask=mask,
        wind_speed=2.0,
        length_method="max_axis",
    )(enhancement)

    assert estimate["ime_kg"] == pytest.approx(300.0)
    assert estimate["length_m"] == pytest.approx(20.0)
    assert estimate["length_m"] != pytest.approx(max_axis["length_m"])
    assert estimate["emission_rate_kg_s"] == pytest.approx(30.0)
    assert estimate["emission_rate_uncertainty_kg_s"] == pytest.approx(6.0)


def test_cross_sectional_flux_returns_transect_geodataframe() -> None:
    mask = _gt(np.ones((3, 3), dtype=bool))
    enhancement = _gt(np.ones((3, 3), dtype=float))

    gdf = gz.plume.CrossSectionalFlux(
        plume_mask=mask,
        source=(0.0, 100.0),
        wind_u=1.0,
        wind_v=0.0,
        n_transects=2,
        transect_spacing_m=10.0,
    )(enhancement)

    assert list(gdf["transect_id"]) == [1, 2]
    assert (gdf["flux_kg_s"] >= 0.0).all()
    assert gdf.crs == "EPSG:32629"


def test_column_to_mass_round_trip_from_ppm_m() -> None:
    values = np.array([[100.0, 250.0]])
    kg_m2 = convert_column_units(values, gas="CH4", units_in="ppm_m", units_out="kg_m2")
    recovered = convert_column_units(
        kg_m2, gas="CH4", units_in="kg_m2", units_out="ppm_m"
    )

    assert np.allclose(recovered, values)


def test_column_to_mass_operator_preserves_metadata() -> None:
    gt = _gt(np.array([[1.0, 2.0]]))

    out = gz.plume.ColumnToMass(gas="CO2", units_in="mol_m2")(gt)

    assert np.allclose(np.asarray(out), np.array([[0.04401, 0.08802]]))
    assert out.transform == gt.transform
    assert str(out.crs) == "EPSG:32629"


def test_sbmp_reference_scene_correlates_with_injected_signal() -> None:
    truth = np.linspace(0.0, 0.2, 25).reshape(5, 5)
    reference = np.stack([np.ones_like(truth), np.ones_like(truth)], axis=0)
    scene = np.stack([np.exp(truth), np.ones_like(truth)], axis=0)

    out = gz.plume.SBMP(swir1=0, swir2=1, reference_scene=_gt(reference))(_gt(scene))

    assert np.allclose(np.asarray(out), truth)
    corr = np.corrcoef(np.asarray(out).ravel(), truth.ravel())[0, 1]
    assert corr > 0.99


def test_sbmp_default_sentinel2_swir_band_names() -> None:
    truth = np.linspace(0.0, 0.2, 25).reshape(5, 5)
    reference = np.ones((12, 5, 5), dtype=float)
    scene = reference.copy()
    scene[10] = np.exp(truth)

    out = gz.plume.SBMP(reference_scene=_gt(reference))(_gt(scene))

    assert np.allclose(np.asarray(out), truth)


def test_sbmp_clips_non_positive_swir_values_before_log() -> None:
    scene = np.array(
        [
            [[0.0, -1.0], [2.0, 3.0]],
            [[0.0, 1.0], [-2.0, 3.0]],
        ]
    )

    out = gz.plume.SBMP(swir1=0, swir2=1)(_gt(scene))

    assert np.isfinite(np.asarray(out)).all()
