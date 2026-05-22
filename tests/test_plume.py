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
    assert row.area_m2 == pytest.approx(row.area * 100.0)
    assert row.mean_enhancement == pytest.approx(float(enhancement[mask].mean()))
    assert row.max_enhancement == pytest.approx(float(enhancement[mask].max()))
    assert row.area == pytest.approx(6.0)
    assert row.major_axis_length > 0.0
    assert row["bbox-0"] == 1
    assert row["bbox-1"] == 1
    assert row["bbox-2"] == 3
    assert row["bbox-3"] == 4
    assert gdf.crs == "EPSG:32629"

    no_stats = gz.plume.PlumeFootprint(
        min_area_m2=1.0,
        simplify_tolerance=None,
    )(_gt(mask))
    assert no_stats.iloc[0].mean_enhancement is None
    assert no_stats.iloc[0].max_enhancement is None


def test_plume_footprint_empty_mask_returns_empty_geodataframe() -> None:
    """Empty masks (or all-filtered components) must yield a valid empty GDF.

    Regression test: previously ``gpd.GeoDataFrame(rows, geometry="geometry")``
    raised ``ValueError`` when ``rows`` was empty because the ``geometry``
    column did not exist.
    """
    empty_mask = _gt(np.zeros((4, 4), dtype=bool))

    gdf = gz.plume.PlumeFootprint(
        min_area_m2=1.0,
        simplify_tolerance=None,
    )(empty_mask)

    assert len(gdf) == 0
    assert gdf.crs == "EPSG:32629"
    expected_cols = {
        "geometry",
        "area_m2",
        "centroid",
        "mean_enhancement",
        "max_enhancement",
        "n_pixels",
        "label_id",
    }
    assert expected_cols.issubset(set(gdf.columns))
    assert gdf.geometry.name == "geometry"

    # Also covers the "all components filtered by min_area_m2" path: a
    # single-pixel mask filtered out by a huge ``min_area_m2``.
    one_pixel = np.zeros((4, 4), dtype=bool)
    one_pixel[0, 0] = True
    filtered = gz.plume.PlumeFootprint(
        min_area_m2=1.0e9,
        simplify_tolerance=None,
    )(_gt(one_pixel))
    assert len(filtered) == 0
    assert filtered.crs == "EPSG:32629"


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


def test_skeleton_plume_length_is_deterministic_for_multi_component_mask() -> None:
    """Skeleton length must not depend on set iteration order.

    Regression test: previously ``_longest_active_pixel_path`` seeded the
    double-BFS from ``next(iter(nodes))`` where ``nodes`` was a Python
    set. Across two disconnected components, the arbitrary starting node
    made the result depend on insertion order (effectively the hash of
    the tuple coordinates).
    """
    from geotoolz.plume._src.array import _longest_active_pixel_path, plume_length

    # Two disconnected components: a long horizontal arm (10 px) and a
    # short isolated blob (1 px). The longest path should come from the
    # long arm, regardless of which component the algorithm visits first.
    mask = np.zeros((6, 12), dtype=bool)
    mask[1, 1:11] = True  # 10-pixel horizontal arm
    mask[4, 10] = True  # isolated single pixel
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)

    length_a = _longest_active_pixel_path(mask, transform)
    length_b = _longest_active_pixel_path(mask, transform)
    length_via_api = plume_length(mask, transform, method="skeleton")

    # 10 pixels at 10 m spacing => 9 * 10 m between endpoint centroids.
    assert length_a == pytest.approx(90.0)
    assert length_b == pytest.approx(90.0)
    assert length_via_api == pytest.approx(90.0)

    # Permuting the mask via row/col reflection still picks the same
    # connected component as the dominant one.
    reflected = mask[::-1, ::-1].copy()
    assert _longest_active_pixel_path(reflected, transform) == pytest.approx(90.0)


def test_ime_rejects_negative_uncertainty_fraction() -> None:
    with pytest.raises(ValueError, match="uncertainty_fraction"):
        gz.plume.IMEEstimate(
            plume_mask=_gt(np.ones((1, 1), dtype=bool)),
            wind_speed=1.0,
            uncertainty_fraction=-0.1,
        )


def test_ime_convex_hull_length_agrees_with_max_axis_on_convex_plume() -> None:
    mask = _gt(np.ones((2, 2), dtype=bool))
    enhancement = _gt(np.ones((2, 2), dtype=float))

    convex_hull = gz.plume.IMEEstimate(
        plume_mask=mask,
        wind_speed=1.0,
        length_method="convex_hull",
        return_uncertainty=False,
    )(enhancement)
    max_axis = gz.plume.IMEEstimate(
        plume_mask=mask,
        wind_speed=1.0,
        length_method="max_axis",
        return_uncertainty=False,
    )(enhancement)

    assert convex_hull["length_m"] == pytest.approx(max_axis["length_m"])


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
    op = gz.plume.SBMP(swir1=0, swir2=1)
    scene = np.array(
        [
            [[0.0, -1.0], [2.0, 3.0]],
            [[0.0, 1.0], [-2.0, 3.0]],
        ]
    )

    out = op(_gt(scene))

    clipped_swir1 = np.maximum(scene[0], 0.0)
    clipped_swir2 = np.maximum(scene[1], 0.0)
    expected = (clipped_swir1 - clipped_swir2) / (
        clipped_swir1 + clipped_swir2 + op.eps
    )
    assert np.isfinite(np.asarray(out)).all()
    assert np.allclose(np.asarray(out), expected)


def test_ime_recovers_injected_mass_on_synthetic_plume() -> None:
    """Synthetic plume on a flat background: IME equals injected mass."""
    enhancement = np.zeros((20, 20), dtype=float)
    # Linear plume: 1 px tall, 10 px long, uniform 0.5 kg/m^2 column.
    enhancement[10, 5:15] = 0.5
    mask_arr = enhancement > 0
    # 10 m pixels -> pixel area 100 m^2 -> per-pixel mass 50 kg ->
    # total injected mass over 10 pixels = 500 kg.
    expected_ime = 10 * 100.0 * 0.5
    # Length: 10 pixels at 10 m spacing => 9 * 10 m between centroids.
    expected_length = 9 * 10.0
    wind = 4.0

    result = gz.plume.IMEEstimate(
        plume_mask=_gt(mask_arr),
        wind_speed=wind,
        return_uncertainty=False,
    )(_gt(enhancement))

    assert result["ime_kg"] == pytest.approx(expected_ime)
    assert result["length_m"] == pytest.approx(expected_length)
    assert result["emission_rate_kg_s"] == pytest.approx(
        wind * expected_ime / expected_length
    )


def test_ime_empty_mask_yields_zero_emission() -> None:
    mask = _gt(np.zeros((4, 4), dtype=bool))
    enhancement = _gt(np.ones((4, 4), dtype=float))

    result = gz.plume.IMEEstimate(
        plume_mask=mask,
        wind_speed=5.0,
        return_uncertainty=False,
    )(enhancement)

    assert result["ime_kg"] == 0.0
    assert result["length_m"] == 0.0
    assert result["emission_rate_kg_s"] == 0.0


def test_cross_sectional_flux_matches_uniform_plume_analytical() -> None:
    """Uniform Omega along a 1-pixel-tall plume gives Q = U * Omega * W * N.

    For a 1-pixel-tall row at the source y, the across-wind integral of
    Omega = const is Omega * (n_active_pixels) * dx. Each transect picks
    up exactly one pixel along the row, so flux = U * Omega * dx.
    """
    enhancement = np.zeros((11, 21), dtype=float)
    enhancement[5, :] = 1.0  # kg/m^2, full row
    mask_arr = enhancement > 0

    # Place source at the row centre.
    source_x = 5.5 * 10.0  # column 5 centre
    source_y = 100.0 - 5.5 * 10.0  # row 5 centre (transform.f=100, e=-10)
    wind_u = 1.0
    wind_v = 0.0

    gdf = gz.plume.CrossSectionalFlux(
        plume_mask=_gt(mask_arr),
        source=(source_x, source_y),
        wind_u=wind_u,
        wind_v=wind_v,
        n_transects=3,
        transect_spacing_m=10.0,
    )(_gt(enhancement))

    # Each transect crosses exactly one pixel (1 kg/m^2) of along-wind
    # half-width 5 m. Flux = Omega * |U| * dx_across = 1 * 1 * 10 = 10.
    for flux in gdf["flux_kg_s"]:
        assert flux == pytest.approx(10.0)
    assert (gdf["n_pixels"] == 1).all()


def test_cross_sectional_flux_rejects_zero_wind() -> None:
    mask = _gt(np.ones((3, 3), dtype=bool))
    enhancement = _gt(np.ones((3, 3), dtype=float))
    op = gz.plume.CrossSectionalFlux(
        plume_mask=mask,
        source=(0.0, 0.0),
        wind_u=0.0,
        wind_v=0.0,
    )
    with pytest.raises(ValueError, match="wind vector"):
        op(enhancement)


def test_column_unit_round_trip_through_mol_m2() -> None:
    """ppm_m -> mol_m2 -> kg_m2 -> ppm_m round-trips to within float eps."""
    values = np.array([[10.0, 100.0, 1000.0]])
    mol = convert_column_units(values, gas="CH4", units_in="ppm_m", units_out="mol_m2")
    kg = convert_column_units(mol, gas="CH4", units_in="mol_m2", units_out="kg_m2")
    back = convert_column_units(kg, gas="CH4", units_in="kg_m2", units_out="ppm_m")
    assert np.allclose(back, values)


def test_plume_operators_get_config_is_json_safe() -> None:
    """get_config dicts must be JSON-serialisable (no GeoTensors)."""
    import json

    mask_gt = _gt(np.ones((2, 2), dtype=bool))

    ops = [
        gz.plume.PlumeMask(threshold="otsu", min_area=10),
        gz.plume.PlumeContours(min_area=5),
        gz.plume.PlumeFootprint(min_area_m2=100.0),
        gz.plume.WindAdvectionCone(
            source=(0.0, 0.0),
            wind_u=1.0,
            wind_v=0.0,
        ),
        gz.plume.ColumnToMass(gas="CO2", units_in="ppm_m"),
        gz.plume.IMEEstimate(plume_mask=mask_gt, wind_speed=3.0),
        gz.plume.CrossSectionalFlux(
            plume_mask=mask_gt,
            source=(0.0, 0.0),
            wind_u=1.0,
            wind_v=0.0,
        ),
        gz.plume.SBMP(reference_scene=_gt(np.ones((12, 2, 2)))),
    ]
    for op in ops:
        # Must round-trip through JSON without raising.
        config = op.get_config()
        json.dumps(config)


def test_label_components_drops_small_and_renumbers_contiguously() -> None:
    """label_components must yield contiguous 1..K labels after dropping."""
    from geotoolz.plume._src.array import label_components

    mask = np.zeros((7, 7), dtype=bool)
    # Three isolated components separated by >=2 pixels on every side so
    # 8-connectivity does not merge them.
    mask[0:2, 0:2] = True  # 4 px component
    mask[6, 6] = True  # 1 px component (dropped at min_area=2)
    mask[5:7, 0:3] = True  # 6 px component (top-right diagonal of [6,6]
    # is (5,5)=False, so [6,6] stays isolated)
    mask[5, 5] = False

    labels = label_components(mask, min_area=2, connectivity=4)
    unique = sorted(set(np.unique(labels).tolist()))
    assert unique == [0, 1, 2]
    assert int((labels == 1).sum()) + int((labels == 2).sum()) == 4 + 6


def test_ime_convex_hull_handles_collinear_points() -> None:
    """Degenerate convex hull (1-pixel row) must not crash."""
    mask = _gt(np.array([[True, True, True, True]]))
    enhancement = _gt(np.ones((1, 4), dtype=float))

    result = gz.plume.IMEEstimate(
        plume_mask=mask,
        wind_speed=1.0,
        length_method="convex_hull",
        return_uncertainty=False,
    )(enhancement)

    # Four pixels at 10 m spacing -> length = 3 * 10 = 30 m.
    assert result["length_m"] == pytest.approx(30.0)
