"""Methane column operators: averaging kernels and VMR ↔ column conversions.

Column amounts are expressed in molecules m⁻² throughout, so
``mixing_ratio_to_column`` and ``dry_air_column`` divide to a dry-air mole
fraction directly.
"""

from __future__ import annotations

import xarray as xr
from xrreader.types import Variable, apply_cf_attrs

from xrtoolz.atm._src.constants import K_B, M_CH4, M_DRY, N_A, G
from xrtoolz.atm._src.vertical import column_integral


#: ``N_A / (g M_dry)``: molecules of dry air per m² per Pa of hydrostatic
#: pressure.
_MOLECULES_PER_PA = N_A / (G * M_DRY)

_DRY_AIR_COLUMN = Variable(
    name="dry_air_column",
    long_name="Dry-air column number density",
    units="m-2",
)
# No CF ``standard_name``: ``atmosphere_mole_content_of_methane`` is in
# mol m⁻², whereas this column carries molecules per square metre.
_CH4_COLUMN = Variable(
    name="ch4_column",
    long_name="Methane column number density (molecules per square metre)",
    units="m-2",
)


def apply_column_averaging_kernel(
    ds: xr.Dataset,
    *,
    profile: str,
    prior: str,
    averaging_kernel: str,
    pressure_weights: str,
    level: str = "layer",
    name: str = "xch4_smoothed",
) -> xr.Dataset:
    """Smooth a model profile with a satellite column averaging kernel.

    TROPOMI / OCO convention, with pressure weights ``h`` summing to one,
    column averaging kernel ``A`` per layer and retrieval prior ``x_a``:

    ``ŷ = hᵀ x_a + Σ_l h_l A_l (x_l − x_{a,l})``

    ``A ≡ 1`` reduces to the column average ``hᵀ x``; ``x = x_a`` returns
    the prior column ``hᵀ x_a``.

    Args:
        ds: Dataset with the four per-layer fields.
        profile: Model profile ``x`` per layer (mole fraction).
        prior: Retrieval prior ``x_a`` per layer, same units as ``profile``.
        averaging_kernel: Column averaging kernel ``A`` per layer.
        pressure_weights: Pressure weights ``h`` per layer (sum to one).
        level: Layer dimension.
        name: Name of the output variable.

    Returns:
        ``ds`` with ``name`` added (CF
        ``dry_atmosphere_mole_fraction_of_methane``, units copied from
        ``profile``); the ``level`` dim is reduced away. A NaN in any layer
        (e.g. a QA-masked pixel) makes the result NaN.

    Raises:
        ValueError: If ``profile`` lacks ``level``.
    """
    x = ds[profile]
    if level not in x.dims:
        raise ValueError(f"{profile!r} does not carry the layer dim {level!r}")
    x_a = ds[prior]
    h = ds[pressure_weights]
    a = ds[averaging_kernel]
    smoothed = (h * x_a).sum(level, skipna=False) + (h * a * (x - x_a)).sum(
        level, skipna=False
    )
    variable = Variable(
        name=name,
        standard_name="dry_atmosphere_mole_fraction_of_methane",
        long_name="Averaging-kernel-smoothed XCH4",
        units=x.attrs.get("units"),
    )
    out = ds.copy()
    out[name] = apply_cf_attrs(smoothed, variable, overwrite=True)
    return out


def dry_air_column(
    ds: xr.Dataset,
    *,
    surface_pressure: str = "sp",
    water_vapour_column: str | None = None,
    name: str = "dry_air_column",
) -> xr.Dataset:
    """Dry-air column number density from surface pressure.

    Hydrostatic balance gives the total-air column as ``p_s / g`` kg m⁻²;
    removing the water-vapour mass column ``W`` (total column water vapour,
    kg m⁻²) and converting to molecules with ``N_A / M_dry``:

    ``N_dry = N_A / (g M_dry) ∫₀^{p_s} (1 − q) dp = N_A (p_s / g − W) / M_dry``

    With ``W = 0`` and ``p_s = 10⁵ Pa`` this is ``≈ 2.12 × 10²⁹ m⁻²``.

    Args:
        ds: Dataset with the surface pressure (and optional water column).
        surface_pressure: Surface pressure [Pa].
        water_vapour_column: Optional total column water vapour [kg m⁻²]
            (CF ``atmosphere_mass_content_of_water_vapor``, ERA5 ``tcwv``);
            ``None`` treats the column as dry.
        name: Name of the output variable.

    Returns:
        ``ds`` with ``name`` added [molecules m⁻²].
    """
    mass = ds[surface_pressure] / G
    if water_vapour_column is not None:
        mass = mass - ds[water_vapour_column]
    n_dry = mass * (N_A / M_DRY)
    out = ds.copy()
    out[name] = apply_cf_attrs(n_dry, _DRY_AIR_COLUMN, overwrite=True)
    return out


def mixing_ratio_to_column(
    ds: xr.Dataset,
    *,
    vmr: str,
    pressure: str = "pressure",
    specific_humidity: str | None = None,
    level: str = "level",
    name: str = "column",
) -> xr.Dataset:
    """Column number density of a gas from its dry-air mole-fraction profile.

    Trapezoid integral over the pressure levels provided (hydrostatic):

    ``Ω = N_A / (g M_dry) ∫ χ (1 − q) dp``

    so a uniform profile ``χ ≡ c`` over the full column gives
    ``Ω = c N_dry``. The integral spans the levels present in ``ds`` only —
    include ``p = 0`` and ``p = p_s`` levels to cover the whole column. The
    result is orientation-independent (absolute value).

    Args:
        ds: Dataset with the mole-fraction profile and pressure per level.
        vmr: Dry-air mole fraction ``χ`` per level (dimensionless; ppb in
            gives ppb-scaled molecules out).
        pressure: Pressure per level [Pa]; may be the ``level`` coordinate.
        specific_humidity: Optional specific humidity [kg kg⁻¹] per level;
            ``None`` treats the air as dry.
        level: Vertical dimension.
        name: Name of the output variable.

    Returns:
        ``ds`` with ``name`` added (molecules m⁻², ``units="m-2"``; no CF
        ``standard_name`` since the CF mole-content name is in mol m⁻²);
        the ``level`` dim is reduced away.
    """
    chi = ds[vmr]
    if specific_humidity is not None:
        chi = chi * (1.0 - ds[specific_humidity])
    integral = column_integral(chi, dim=level, coord=ds[pressure], method="trapezoid")
    column = abs(integral) * _MOLECULES_PER_PA
    out = ds.copy()
    out[name] = apply_cf_attrs(column, _CH4_COLUMN, overwrite=True)
    return out


def column_mass_to_delta_vmr(
    column_mass: xr.DataArray,
    *,
    pressure_pa: float | xr.DataArray,
    temperature_k: float | xr.DataArray,
    path_length_m: float | xr.DataArray,
    molar_mass_kg_mol: float = M_CH4,
) -> xr.DataArray:
    """Column mass enhancement → volume-mixing-ratio enhancement.

    Spreads the column mass ``m`` [kg m⁻²] over a single well-mixed layer
    of length ``L`` at the ideal-gas number density ``n_air = p / (k_B T)``:

    ``Δχ = (m N_A / M_gas) / (n_air L)``

    Args:
        column_mass: Column mass enhancement [kg m⁻²].
        pressure_pa: Layer pressure [Pa].
        temperature_k: Layer temperature [K].
        path_length_m: Vertical path length ``L`` [m] over which the
            enhancement is distributed.
        molar_mass_kg_mol: Molar mass of the gas [kg mol⁻¹] (methane).

    Returns:
        Dimensionless mixing-ratio enhancement, same shape as
        ``column_mass`` broadcast against the layer parameters.
    """
    n_air = pressure_pa / (K_B * temperature_k)
    molecules_per_m2 = column_mass / molar_mass_kg_mol * N_A
    delta = molecules_per_m2 / (n_air * path_length_m)
    return delta.assign_attrs(long_name="Mixing-ratio enhancement", units="1")
