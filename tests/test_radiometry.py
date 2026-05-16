"""Tests for `geotoolz.radiometry`.

Three layers per module convention: Tier-A math, Tier-B carrier
round-trip, Hydra-zen `builds()` round-trip.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geotoolz.radiometry import (
    DOS1,
    ApplySRF,
    BTFromRadiance,
    ComputeSZA,
    DNToRadiance,
    DNToReflectance,
    EarthSunDistanceCorrection,
    Gamma,
    IntegratedIrradiance,
    MinMax,
    PercentileClip,
    RadianceToDN,
    RadianceToReflectance,
    ReflectanceToRadiance,
    SimpleAtmosphericCorrection,
    ToFloat32,
    bt_from_radiance,
    dn_to_radiance,
    dn_to_reflectance,
    dos1,
    gamma_correct,
    min_max_normalize,
    percentile_clip,
    radiance_to_dn,
)
from geotoolz.radiometry._src.solar import (
    earth_sun_distance_correction_factor,
    observation_date_correction_factor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_geotensor(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
        crs="EPSG:32629",
        fill_value_default=0,
    )


@pytest.fixture
def dn_4band() -> GeoTensor:
    """4-band uint16 DN raster mimicking Sentinel-2 L1C."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 12_000, size=(4, 8, 8), dtype=np.uint16)
    return _toy_geotensor(arr)


# ---------------------------------------------------------------------------
# Tier-A — primitive math
# ---------------------------------------------------------------------------


def test_dn_to_radiance_linear() -> None:
    dn = np.array([[0.0, 1000.0], [2000.0, 4095.0]])
    out = dn_to_radiance(dn, gain=0.01, offset=-1.0)
    expected = 0.01 * dn - 1.0
    np.testing.assert_allclose(out, expected, rtol=1e-9)


def test_radiance_to_dn_inverts_dn_to_radiance_with_scale() -> None:
    dn = np.array([0.0, 1000.0, 2000.0])
    radiance = dn_to_radiance(dn, gain=2.0, offset=1.0, scale=10.0)
    out = radiance_to_dn(radiance, gain=2.0, offset=1.0, scale=10.0)
    np.testing.assert_allclose(out, dn, rtol=1e-12)


def test_dn_to_reflectance_s2_l1c() -> None:
    """S2 L1C pre-2022 convention: rho = 1e-4 * DN. Round-trip a known reflectance."""
    rho = np.array([[0.05, 0.30], [0.55, 0.80]])
    dn = (rho / 1e-4).astype(np.uint16)
    out = dn_to_reflectance(dn.astype(np.float64), scale=1e-4, offset=0.0)
    np.testing.assert_allclose(out, rho, atol=1e-4)  # uint16 quantisation


def test_dn_to_reflectance_landsat_c2_sr_with_additive_offset() -> None:
    """Landsat-8/9 C2 SR convention: rho = scale * DN + offset (reflectance units).

    The offset is in reflectance units, not DN units — so the formula
    is the canonical y = m*x + b. A scaled-reflectance DN of 6000 with
    scale=2.75e-5 and offset=-0.2 must give rho = 0.165 - 0.2 = -0.035
    (negative is physically reasonable here, signalling water in NIR
    after atmospheric over-correction).
    """
    dn = np.array([0.0, 6000.0, 30_000.0])
    out = dn_to_reflectance(dn, scale=2.75e-5, offset=-0.2)
    expected = 2.75e-5 * dn - 0.2
    np.testing.assert_allclose(out, expected, rtol=1e-9)


def test_dn_to_reflectance_s2_l1c_post_2022_offset() -> None:
    """S2 L1C post-2022: RADIO_ADD_OFFSET=-1000 (DN) collapses to offset=-0.1.

    Pre-2022 DN=11000 -> rho = 1.1. Post-2022 the published RADIO_ADD_OFFSET
    is -1000 DN; multiplied through scale=1e-4 it becomes -0.1 reflectance,
    so DN=11000 -> rho = 1.1 - 0.1 = 1.0 — the conversion handles the
    bias correctly under the affine-decode convention.
    """
    dn = np.array([11_000.0])
    out = dn_to_reflectance(dn, scale=1e-4, offset=-0.1)
    np.testing.assert_allclose(out, [1.0], rtol=1e-9)


def test_min_max_normalize() -> None:
    arr = np.array([0.0, 0.5, 1.0, 2.0])
    out = min_max_normalize(arr, vmin=0.0, vmax=1.0, clip=True)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0, 1.0])
    # clip=False leaves the tail untouched.
    out_unclipped = min_max_normalize(arr, vmin=0.0, vmax=1.0, clip=False)
    np.testing.assert_allclose(out_unclipped, [0.0, 0.5, 1.0, 2.0])


def test_min_max_normalize_rejects_degenerate_range() -> None:
    with pytest.raises(ValueError, match="vmax > vmin"):
        min_max_normalize(np.zeros(4), vmin=1.0, vmax=1.0)


def test_percentile_clip_global() -> None:
    arr = np.linspace(0.0, 100.0, 101)  # 0, 1, ..., 100
    # With p_min=10, p_max=90 percentiles are 10 and 90.
    out = percentile_clip(arr, p_min=10.0, p_max=90.0, axis=None)
    # The percentile-bound values map to exactly 0 and 1.
    np.testing.assert_allclose(out[10], 0.0, atol=1e-9)
    np.testing.assert_allclose(out[90], 1.0, atol=1e-9)
    # Below the lower percentile -> 0; above the upper -> 1.
    assert out[0] == 0.0
    assert out[100] == 1.0


def test_percentile_clip_per_band_axis() -> None:
    """axis=(-2,-1) stretches each band independently."""
    arr = np.stack(
        [
            np.linspace(0, 10, 100).reshape(10, 10),  # band 0: range 0-10
            np.linspace(0, 1000, 100).reshape(10, 10),  # band 1: range 0-1000
        ]
    )
    out = percentile_clip(arr, p_min=0.0, p_max=100.0, axis=(-2, -1))
    # Both bands should now be normalised into [0, 1].
    assert np.isclose(out[0].max(), 1.0)
    assert np.isclose(out[1].max(), 1.0)
    assert np.isclose(out[0].min(), 0.0)
    assert np.isclose(out[1].min(), 0.0)


def test_gamma_correct_monotone() -> None:
    arr = np.linspace(0.0, 1.0, 11)
    out = gamma_correct(arr, g=2.2)
    # Strictly monotone increasing.
    assert np.all(np.diff(out) > 0)
    # g > 1 brightens midtones -> 0.5 -> 0.5**(1/2.2) ~ 0.73
    np.testing.assert_allclose(
        gamma_correct(np.array([0.5]), g=2.2), [0.5 ** (1 / 2.2)]
    )


def test_gamma_correct_handles_negatives() -> None:
    arr = np.array([-0.5, 0.0, 0.5])
    out = gamma_correct(arr, g=1.2)
    assert out[0] == 0.0  # negative clipped to zero before power
    assert np.isfinite(out).all()


def test_bt_from_radiance_planck_inversion() -> None:
    radiance = np.array([10.0])
    out = bt_from_radiance(radiance, k1=774.8853, k2=1321.0789)
    expected = 1321.0789 / np.log((774.8853 / radiance) + 1.0)
    np.testing.assert_allclose(out, expected)


def test_bt_from_radiance_round_trip_against_planck() -> None:
    """The inversion ``T = K2 / ln(K1/L + 1)`` must invert ``L = K1 / (exp(K2/T) - 1)``.

    Construct radiance synthetically from a target temperature using
    the forward Planck form, push it through ``bt_from_radiance``, and
    verify the recovered temperature matches the input to float
    precision.
    """
    k1, k2 = 774.8853, 1321.0789  # Landsat-8 TIRS Band 10
    targets = np.array([260.0, 290.0, 310.0])  # cold / temperate / warm
    radiance = k1 / (np.exp(k2 / targets) - 1.0)
    recovered = bt_from_radiance(radiance, k1=k1, k2=k2)
    np.testing.assert_allclose(recovered, targets, rtol=1e-12)


def test_earth_sun_distance_perihelion_aphelion() -> None:
    """Perihelion (~Jan 3) gives d ≈ 0.983; aphelion (~Jul 4) gives d ≈ 1.017."""
    d_perihelion = earth_sun_distance_correction_factor(datetime(2024, 1, 3))
    d_aphelion = earth_sun_distance_correction_factor(datetime(2024, 7, 4))
    np.testing.assert_allclose(d_perihelion, 0.9833, atol=1e-3)
    np.testing.assert_allclose(d_aphelion, 1.0167, atol=1e-3)
    # Inverse-square irradiance ratio matches the well-known ~6.5%.
    ratio = (d_aphelion / d_perihelion) ** 2
    np.testing.assert_allclose(ratio, 1.069, atol=2e-3)


def test_observation_date_correction_factor_from_sza() -> None:
    """``π·d²/cos(θ_z)`` factor must match the algebraic formula."""
    date = datetime(2024, 6, 21)
    sza_deg = 30.0
    out = observation_date_correction_factor(date, sza_deg=sza_deg)
    d = earth_sun_distance_correction_factor(date)
    expected = np.pi * (d**2) / np.cos(np.deg2rad(sza_deg))
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_observation_date_correction_factor_returns_none_without_inputs() -> None:
    """No sza_deg and no center_coords -> defer to georeader (return None)."""
    out = observation_date_correction_factor(datetime(2024, 6, 21))
    assert out is None


def test_dos1_subtracts_dark_percentile() -> None:
    reflectance = np.array([[[0.02, 0.10], [0.20, 0.30]]])
    out = dos1(reflectance, dark_percentile=0.0)
    np.testing.assert_allclose(out, [[[0.0, 0.08], [0.18, 0.28]]])


# ---------------------------------------------------------------------------
# Tier-B — Operator + GeoTensor round-trip
# ---------------------------------------------------------------------------


def test_to_float32(dn_4band: GeoTensor) -> None:
    op = ToFloat32()
    out = op(dn_4band)
    assert isinstance(out, GeoTensor)
    assert out.dtype == np.float32
    assert out.transform == dn_4band.transform
    assert str(out.crs) == "EPSG:32629"


def test_dn_to_reflectance_operator_preserves_metadata(dn_4band: GeoTensor) -> None:
    op = DNToReflectance(scale=1e-4)
    out = op(dn_4band)
    assert isinstance(out, GeoTensor)
    assert out.transform == dn_4band.transform
    assert out.shape == dn_4band.shape
    np.testing.assert_allclose(np.asarray(out), 1e-4 * np.asarray(dn_4band))


def test_dn_to_radiance_per_band(dn_4band: GeoTensor) -> None:
    gains = np.array([0.01, 0.02, 0.015, 0.012])
    offsets = np.array([-1.0, -2.0, -1.5, -1.2])
    op = DNToRadiance(gain=gains, offset=offsets)
    out = op(dn_4band)
    arr = np.asarray(out)
    dn = np.asarray(dn_4band)
    for b in range(4):
        np.testing.assert_allclose(arr[b], gains[b] * dn[b] + offsets[b])


def test_radiance_to_dn_operator_inverts_dn_to_radiance(dn_4band: GeoTensor) -> None:
    gains = np.array([0.01, 0.02, 0.015, 0.012])
    offsets = np.array([-1.0, -2.0, -1.5, -1.2])
    radiance = DNToRadiance(gain=gains, offset=offsets, scale=2.0)(dn_4band)
    out = RadianceToDN(gain=gains, offset=offsets, scale=2.0)(radiance)
    np.testing.assert_allclose(np.asarray(out), np.asarray(dn_4band))


def test_dn_to_radiance_rejects_wrong_band_count(dn_4band: GeoTensor) -> None:
    op = DNToRadiance(gain=np.array([0.01, 0.02]))  # only 2 entries, need 4
    with pytest.raises(ValueError, match="doesn't match"):
        op(dn_4band)


def test_min_max_operator(dn_4band: GeoTensor) -> None:
    rho = DNToReflectance(scale=1e-4)(dn_4band)
    out = MinMax(vmin=0.0, vmax=1.0)(rho)
    assert np.all(np.asarray(out) <= 1.0)
    assert np.all(np.asarray(out) >= 0.0)
    assert out.transform == rho.transform


def test_percentile_clip_operator(dn_4band: GeoTensor) -> None:
    rho = DNToReflectance(scale=1e-4)(dn_4band)
    op = PercentileClip(p_min=2.0, p_max=98.0)
    out = op(rho)
    assert out.shape == rho.shape
    assert np.all(np.asarray(out) >= 0.0)
    assert np.all(np.asarray(out) <= 1.0)
    assert out.transform == rho.transform


def test_gamma_operator(dn_4band: GeoTensor) -> None:
    rho = DNToReflectance(scale=1e-4)(dn_4band)
    out = Gamma(g=1.2)(rho)
    assert out.shape == rho.shape
    assert out.transform == rho.transform


def test_radiometry_pipeline_composes(dn_4band: GeoTensor) -> None:
    """The README's display pipeline should work as a single Sequential."""
    pipe = (
        ToFloat32()
        | DNToReflectance(scale=1e-4)
        | PercentileClip(p_min=2.0, p_max=98.0)
        | Gamma(g=1.2)
    )
    out = pipe(dn_4band)
    assert isinstance(out, GeoTensor)
    assert out.shape == dn_4band.shape
    assert np.all(np.asarray(out) >= 0.0)
    assert np.all(np.asarray(out) <= 1.0)


def test_radiance_reflectance_roundtrip_preserves_metadata() -> None:
    radiance = _toy_geotensor(
        np.array(
            [
                [[10.0, 12.0], [14.0, 16.0]],
                [[20.0, 22.0], [24.0, 26.0]],
            ]
        )
    )
    date = datetime(2024, 7, 14, 11, 32)
    solar_irradiance = np.array([1.95, 1.85])
    to_reflectance = RadianceToReflectance(
        solar_irradiance=solar_irradiance,
        acquisition_date=date,
        sza_deg=30.0,
        units="W/m2/sr/nm",
    )
    to_radiance = ReflectanceToRadiance(
        solar_irradiance=solar_irradiance,
        acquisition_date=date,
        sza_deg=30.0,
    )
    out = to_radiance(to_reflectance(radiance))
    np.testing.assert_allclose(np.asarray(out), np.asarray(radiance), rtol=1e-12)
    assert out.transform == radiance.transform


def test_earth_sun_distance_correction_operator() -> None:
    date = datetime(2024, 1, 3)
    out = EarthSunDistanceCorrection(acquisition_date=date)()
    expected = earth_sun_distance_correction_factor(date)
    np.testing.assert_allclose(out, expected)


def test_bt_from_radiance_operator_preserves_fill() -> None:
    radiance = _toy_geotensor(np.array([[[10.0, 0.0], [12.0, 14.0]]]))
    op = BTFromRadiance(K1=774.8853, K2=1321.0789)
    out = op(radiance)
    assert np.asarray(out)[0, 0, 1] == radiance.fill_value_default
    expected = 1321.0789 / np.log((774.8853 / 10.0) + 1.0)
    np.testing.assert_allclose(np.asarray(out)[0, 0, 0], expected)


def test_dos1_operator_preserves_fill() -> None:
    reflectance = _toy_geotensor(np.array([[[0.02, 0.0], [0.10, 0.20]]]))
    out = DOS1(dark_percentile=0.0)(reflectance)
    np.testing.assert_allclose(np.asarray(out), [[[0.0, 0.0], [0.08, 0.18]]])


def test_simple_atmospheric_correction_dos1() -> None:
    reflectance = _toy_geotensor(np.array([[[0.02, 0.10], [0.20, 0.30]]]))
    out = SimpleAtmosphericCorrection(method="dos1", dark_percentile=0.0)(reflectance)
    np.testing.assert_allclose(np.asarray(out), [[[0.0, 0.08], [0.18, 0.28]]])


def test_apply_srf_flat_spectrum_preserves_flat_signal() -> None:
    hyperspectral = _toy_geotensor(np.ones((5, 2, 2), dtype=np.float32) * 7.0)
    out = ApplySRF(
        target_center_wavelengths=[500.0, 520.0],
        target_fwhm=[20.0, 20.0],
        source_wavelengths=[480.0, 490.0, 500.0, 510.0, 520.0],
    )(hyperspectral)
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(np.asarray(out), 7.0)


def test_apply_srf_preserves_per_band_validity_under_partial_fill() -> None:
    """A target band whose SRF has *no* weight on a fill source band
    must remain finite (non-fill) at pixels where that unrelated source
    band is fill. The previous global "any source fill -> fill all
    targets" mask was too aggressive (Codex P1 review on PR #37).
    """
    # Source bands far apart in wavelength: 450, 550, 650, 750 nm.
    # Targets centred at 450 and 750 with narrow FWHM=10 → each target's
    # SRF support is entirely one source band (no overlap).
    source_wavelengths = [450.0, 550.0, 650.0, 750.0]
    values = np.full((4, 2, 2), 5.0, dtype=np.float32)
    # Pixel (0, 0) is fill in source band 1 (550 nm) only. This band
    # has zero SRF weight on both target bands (450 and 750), so both
    # target outputs at (0, 0) must remain non-fill.
    values[1, 0, 0] = 0.0  # fill_value_default for this fixture is 0
    hyperspectral = _toy_geotensor(values)
    out = ApplySRF(
        target_center_wavelengths=[450.0, 750.0],
        target_fwhm=[10.0, 10.0],
        source_wavelengths=source_wavelengths,
    )(hyperspectral)
    out_arr = np.asarray(out)
    # Both target bands should have finite, non-zero output at (0, 0)
    # because the fill pixel in source band 1 does not contribute to
    # either target's SRF.
    assert out_arr[0, 0, 0] != 0.0, (
        "target band 0 should not be fill at (0,0): "
        "source band 1 (fill) has no SRF weight on it"
    )
    assert out_arr[1, 0, 0] != 0.0, (
        "target band 1 should not be fill at (0,0): "
        "source band 1 (fill) has no SRF weight on it"
    )
    # And the actual aggregated value should reflect the valid source
    # bands (which are all 5.0), so the SRF integral should yield 5.0.
    np.testing.assert_allclose(out_arr[0, 0, 0], 5.0, rtol=1e-5)
    np.testing.assert_allclose(out_arr[1, 0, 0], 5.0, rtol=1e-5)


def test_apply_srf_propagates_fill_on_contributing_source_band() -> None:
    """Conversely, if a *contributing* source band is fill at a pixel,
    the corresponding target band must become fill at that pixel.
    """
    source_wavelengths = [450.0, 550.0, 650.0, 750.0]
    values = np.full((4, 2, 2), 5.0, dtype=np.float32)
    # Pixel (1, 1) is fill in source band 0 (450 nm). This is the only
    # band contributing to target 0 (centred at 450, FWHM 10), so
    # target 0 at (1, 1) must become fill. Target 1 (750 nm) is
    # unaffected and must stay finite.
    values[0, 1, 1] = 0.0
    hyperspectral = _toy_geotensor(values)
    out = ApplySRF(
        target_center_wavelengths=[450.0, 750.0],
        target_fwhm=[10.0, 10.0],
        source_wavelengths=source_wavelengths,
    )(hyperspectral)
    out_arr = np.asarray(out)
    assert out_arr[0, 1, 1] == 0.0, "fill should propagate to contributing target"
    assert out_arr[1, 1, 1] != 0.0, "non-contributing target must remain valid"


def test_integrated_irradiance_operator_with_flat_solar_spectrum() -> None:
    srf_df = pd.DataFrame({"B1": [1.0, 1.0, 1.0]}, index=[499.0, 500.0, 501.0])
    solar = pd.DataFrame(
        {
            "Nanometer": [499.0, 500.0, 501.0],
            "Radiance(mW/m2/nm)": [2.0, 2.0, 2.0],
        }
    )
    out = IntegratedIrradiance(srf=srf_df, solar_irradiance=solar)()
    np.testing.assert_allclose(out, [2.0])


# ---------------------------------------------------------------------------
# Parity with georeader.reflectance (smoke check on a synthetic radiance)
# ---------------------------------------------------------------------------


def test_radiance_to_reflectance_parity_with_georeader() -> None:
    """`RadianceToReflectance` must produce exactly the values that come
    out of `georeader.reflectance.radiance_to_reflectance` when given
    the same inputs — the operator is a thin wrapper, so any drift
    would mean the wrapper is double-applying a correction (a class of
    bug we explicitly want to rule out).
    """
    from georeader.reflectance import radiance_to_reflectance as g_r2r

    radiance = _toy_geotensor(
        np.array(
            [
                [[10.0, 12.0], [14.0, 16.0]],
                [[20.0, 22.0], [24.0, 26.0]],
            ]
        )
    )
    date = datetime(2024, 7, 14, 11, 32)
    solar = np.array([1.95, 1.85])
    sza_deg = 30.0
    # Same observation-date factor on both sides — bypasses pysolar so
    # the comparison is hermetic.
    obs_factor = observation_date_correction_factor(date, sza_deg=sza_deg)
    expected = g_r2r(
        radiance,
        solar_irradiance=solar,
        observation_date_corr_factor=obs_factor,
        units="W/m2/sr/nm",
    )
    op = RadianceToReflectance(
        solar_irradiance=solar,
        acquisition_date=date,
        sza_deg=sza_deg,
        units="W/m2/sr/nm",
    )
    out = op(radiance)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-12)
    assert out.transform == radiance.transform
    assert str(out.crs) == str(radiance.crs)


def test_radiance_to_reflectance_accepts_iso_string_for_yaml_roundtrip() -> None:
    """`acquisition_date` must accept either a datetime or an ISO string
    so that hydra-zen / YAML configs can round-trip without datetime
    serialisers.
    """
    date = datetime(2024, 7, 14, 11, 32)
    radiance = _toy_geotensor(np.ones((2, 2, 2)) * 5.0)
    solar = np.array([1.95, 1.85])
    out_dt = RadianceToReflectance(
        solar_irradiance=solar,
        acquisition_date=date,
        sza_deg=30.0,
    )(radiance)
    out_iso = RadianceToReflectance(
        solar_irradiance=solar,
        acquisition_date=date.isoformat(),
        sza_deg=30.0,
    )(radiance)
    np.testing.assert_allclose(np.asarray(out_iso), np.asarray(out_dt))


def test_compute_sza_operator_summer_solstice_noon() -> None:
    """Solar noon at the equator on the summer solstice gives SZA ≈ 23.4°
    (the Earth's axial tilt). Loose tolerance because pysolar applies
    higher-order corrections (equation of time, atmospheric refraction).
    """
    pytest.importorskip("pysolar")
    # June 21 ~12:00 UTC at 0°N, 0°E -- close to local solar noon.
    # pysolar requires a tz-aware datetime.
    op = ComputeSZA(
        center_coords=(0.0, 0.0),
        acquisition_date=datetime(2024, 6, 21, 12, 0, tzinfo=UTC),
    )
    sza = op()
    assert 22.0 < sza < 25.0


def test_dn_to_reflectance_parity_with_georeader_scalar_path() -> None:
    """For pre-scaled-reflectance products (S2 L1C), DNToReflectance is exact.

    There's no georeader equivalent of the L1C shortcut (georeader's
    `radiance_to_reflectance` handles the proper solar-geometry path,
    which DNToReflectance deliberately doesn't), so this test just
    confirms our primitive matches the documented S2 L1C formula
    rho = (DN + offset) / 10000.
    """
    dn = np.array([1500.0, 3000.0, 6000.0])
    out = dn_to_reflectance(dn, scale=1e-4, offset=0.0)
    np.testing.assert_allclose(out, dn / 10_000.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Hydra-zen round-trip
# ---------------------------------------------------------------------------


try:
    import hydra_zen
except ImportError:  # pragma: no cover - exercised via the [hydra] extra
    hydra_zen = None  # type: ignore[assignment]


@pytest.mark.skipif(hydra_zen is None, reason="requires hydra-zen extra")
@pytest.mark.parametrize(
    "op",
    [
        ToFloat32(),
        DNToRadiance(gain=0.012, offset=-60.0),
        RadianceToDN(gain=0.012, offset=-60.0),
        DNToReflectance(scale=1e-4),
        DNToReflectance(scale=1e-4, offset=-0.1),  # S2 L1C post-2022
        DNToReflectance(scale=2.75e-5, offset=-0.2),  # Landsat-8/9 C2 SR
        MinMax(vmin=0.0, vmax=0.3),
        PercentileClip(p_min=2.0, p_max=98.0),
        BTFromRadiance(K1=774.8853, K2=1321.0789),
        DOS1(dark_percentile=1.0),
        SimpleAtmosphericCorrection(method="dos1", dark_percentile=1.0),
        Gamma(g=1.4),
        EarthSunDistanceCorrection(acquisition_date=datetime(2024, 6, 21)),
        ComputeSZA(
            center_coords=(0.0, 0.0),
            acquisition_date=datetime(2024, 6, 21, 12, 0),
        ),
        RadianceToReflectance(
            solar_irradiance=[1.95, 1.85],
            acquisition_date=datetime(2024, 6, 21),
            sza_deg=30.0,
        ),
        ReflectanceToRadiance(
            solar_irradiance=[1.95, 1.85],
            acquisition_date=datetime(2024, 6, 21),
            sza_deg=30.0,
        ),
        ApplySRF(
            target_center_wavelengths=[500.0, 520.0],
            target_fwhm=[20.0, 20.0],
            source_wavelengths=[480.0, 490.0, 500.0, 510.0, 520.0],
        ),
    ],
)
def test_radiometry_hydra_zen_roundtrip(op: object) -> None:
    cfg = hydra_zen.builds(type(op), **op.get_config())  # type: ignore[attr-defined]
    restored = hydra_zen.instantiate(cfg)
    assert type(restored) is type(op)
    assert restored.get_config() == op.get_config()  # type: ignore[attr-defined]


def test_integrated_irradiance_forbid_in_yaml() -> None:
    """`IntegratedIrradiance` holds a pandas DataFrame and is correctly
    flagged as not YAML-serialisable.
    """
    assert IntegratedIrradiance.forbid_in_yaml is True


@pytest.mark.skipif(hydra_zen is None, reason="requires hydra-zen extra")
def test_dn_to_radiance_per_band_coef_jsonable() -> None:
    """Per-band ndarray coefficients should round-trip as plain lists."""
    op = DNToRadiance(gain=np.array([0.012, 0.013]), offset=[-60.0, -61.0])
    cfg = op.get_config()
    assert cfg["gain"] == [0.012, 0.013]
    assert cfg["offset"] == [-60.0, -61.0]
    # And builds()/instantiate() reconstruct an equivalent op.
    cfg_builds = hydra_zen.builds(DNToRadiance, **cfg)  # type: ignore[union-attr]
    restored = hydra_zen.instantiate(cfg_builds)
    assert restored.get_config() == cfg
