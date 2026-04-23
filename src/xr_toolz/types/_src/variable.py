"""Rich ``Variable`` type with CF metadata and a small built-in registry.

A ``Variable`` carries everything a well-behaved data request needs:

- ``name`` — canonical short name used inside ``xr_toolz`` (e.g. ``"sst"``).
- ``standard_name`` — CF-1.x standard name (e.g. ``"sea_surface_temperature"``).
- ``long_name`` — human-readable description.
- ``units`` — canonical CF units string (e.g. ``"K"``, ``"m s-1"``).
- ``aliases`` — per-source identifiers, e.g. ``{"cmems": "thetao",
  "cds": "sea_surface_temperature"}`` so each adapter can translate.
- ``valid_range`` — optional ``(min, max)`` tuple used by validators.
- ``dtype`` — optional expected dtype, used by validators.

The registry (``REGISTRY``) holds curated variables. Users are free to
construct their own ``Variable`` instances; the registry is a
convenience, not a boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Variable:
    """CF-aware variable descriptor used across adapters and validators.

    The class is an immutable dataclass so instances are hashable and
    safe to share; per-source identifiers live in ``aliases`` and are
    resolved with :meth:`for_source`.
    """

    name: str
    standard_name: str | None = None
    long_name: str | None = None
    units: str | None = None
    # ``hash=False`` so the mutable dict default doesn't leak into the
    # auto-generated ``__hash__``; equality still compares aliases.
    aliases: Mapping[str, str] = field(default_factory=dict, hash=False)
    valid_range: tuple[float, float] | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        # Wrap any Mapping input in a read-only view so callers can't
        # mutate ``var.aliases`` after construction (would otherwise
        # silently change equality/hashing semantics).
        if not isinstance(self.aliases, MappingProxyType):
            frozen = MappingProxyType(dict(self.aliases))
            object.__setattr__(self, "aliases", frozen)

    def for_source(self, source: str) -> str:
        """Return the identifier this variable uses in ``source``.

        Falls back to :attr:`name` if no alias is registered.
        """
        return self.aliases.get(source, self.name)

    def cf_attrs(self) -> dict[str, str]:
        """Return the CF-1.x metadata dict suitable for ``da.attrs``."""
        attrs: dict[str, str] = {}
        if self.standard_name is not None:
            attrs["standard_name"] = self.standard_name
        if self.long_name is not None:
            attrs["long_name"] = self.long_name
        if self.units is not None:
            attrs["units"] = self.units
        return attrs


# ---- Curated registry ----------------------------------------------------


SST = Variable(
    name="sst",
    standard_name="sea_surface_temperature",
    long_name="Sea surface temperature",
    units="K",
    aliases={"cmems": "thetao", "cds": "sea_surface_temperature"},
    valid_range=(270.0, 320.0),
)

SSH = Variable(
    name="ssh",
    standard_name="sea_surface_height_above_geoid",
    long_name="Sea surface height above geoid",
    units="m",
    aliases={"cmems": "zos"},
    valid_range=(-5.0, 5.0),
)

SLA = Variable(
    name="sla",
    standard_name="sea_surface_height_above_sea_level",
    long_name="Sea level anomaly",
    units="m",
    aliases={"cmems": "sla"},
    valid_range=(-2.0, 2.0),
)

MDT = Variable(
    name="mdt",
    standard_name="sea_surface_height_above_geoid",
    long_name="Mean dynamic topography",
    units="m",
    aliases={"cmems": "mdt"},
)

UO = Variable(
    name="uo",
    standard_name="eastward_sea_water_velocity",
    long_name="Eastward sea water velocity",
    units="m s-1",
    aliases={"cmems": "uo"},
    valid_range=(-5.0, 5.0),
)

VO = Variable(
    name="vo",
    standard_name="northward_sea_water_velocity",
    long_name="Northward sea water velocity",
    units="m s-1",
    aliases={"cmems": "vo"},
    valid_range=(-5.0, 5.0),
)

SO = Variable(
    name="so",
    standard_name="sea_water_salinity",
    long_name="Sea water salinity",
    units="1e-3",
    aliases={"cmems": "so"},
    valid_range=(0.0, 45.0),
)

T2M = Variable(
    name="t2m",
    standard_name="air_temperature",
    long_name="2 metre temperature",
    units="K",
    aliases={"cds": "2m_temperature"},
    valid_range=(150.0, 350.0),
)

D2M = Variable(
    name="d2m",
    standard_name="dew_point_temperature",
    long_name="2 metre dewpoint temperature",
    units="K",
    aliases={"cds": "2m_dewpoint_temperature"},
    valid_range=(150.0, 350.0),
)

U10 = Variable(
    name="u10",
    standard_name="eastward_wind",
    long_name="10 metre U wind component",
    units="m s-1",
    aliases={"cds": "10m_u_component_of_wind"},
    valid_range=(-100.0, 100.0),
)

V10 = Variable(
    name="v10",
    standard_name="northward_wind",
    long_name="10 metre V wind component",
    units="m s-1",
    aliases={"cds": "10m_v_component_of_wind"},
    valid_range=(-100.0, 100.0),
)

MSL = Variable(
    name="msl",
    standard_name="air_pressure_at_mean_sea_level",
    long_name="Mean sea level pressure",
    units="Pa",
    aliases={"cds": "mean_sea_level_pressure"},
    valid_range=(87000.0, 108500.0),
)

TP = Variable(
    name="tp",
    standard_name="lwe_thickness_of_precipitation_amount",
    long_name="Total precipitation",
    units="m",
    aliases={"cds": "total_precipitation"},
    valid_range=(0.0, 1.0),
)

SP = Variable(
    name="sp",
    standard_name="surface_air_pressure",
    long_name="Surface pressure",
    units="Pa",
    aliases={"cds": "surface_pressure"},
    valid_range=(50000.0, 110000.0),
)

SSRD = Variable(
    name="ssrd",
    standard_name="surface_downwelling_shortwave_flux_in_air",
    long_name="Surface solar radiation downwards",
    units="J m-2",
    aliases={"cds": "surface_solar_radiation_downwards"},
)

# ---- Ocean — altimetry-derived (DUACS) ----------------------------------

ADT = Variable(
    name="adt",
    standard_name="sea_surface_height_above_geoid",
    long_name="Absolute dynamic topography",
    units="m",
    aliases={"cmems": "adt"},
    valid_range=(-3.0, 3.0),
)

UGOS = Variable(
    name="ugos",
    standard_name="surface_geostrophic_eastward_sea_water_velocity",
    long_name="Surface geostrophic eastward velocity",
    units="m s-1",
    aliases={"cmems": "ugos"},
    valid_range=(-5.0, 5.0),
)

VGOS = Variable(
    name="vgos",
    standard_name="surface_geostrophic_northward_sea_water_velocity",
    long_name="Surface geostrophic northward velocity",
    units="m s-1",
    aliases={"cmems": "vgos"},
    valid_range=(-5.0, 5.0),
)

# ---- Ocean — salinity companion -----------------------------------------

SOS = Variable(
    name="sos",
    standard_name="sea_surface_salinity",
    long_name="Sea surface salinity",
    units="1e-3",
    aliases={"cmems": "sos"},
    valid_range=(0.0, 45.0),
)

DENS = Variable(
    name="dens",
    standard_name="sea_water_density",
    long_name="Sea surface density",
    units="kg m-3",
    aliases={"cmems": "dos"},
    valid_range=(1000.0, 1050.0),
)

# ---- Ocean — sea-ice ----------------------------------------------------

ICE_CONC = Variable(
    name="ice_conc",
    standard_name="sea_ice_area_fraction",
    long_name="Sea ice area fraction",
    units="1",
    aliases={"cmems": "sea_ice_fraction"},
    valid_range=(0.0, 1.0),
)

# ---- Ocean colour -------------------------------------------------------

CHL = Variable(
    name="chl",
    standard_name="mass_concentration_of_chlorophyll_a_in_sea_water",
    long_name="Chlorophyll-a concentration",
    units="mg m-3",
    aliases={"cmems": "CHL"},
    valid_range=(0.0, 100.0),
)

KD490 = Variable(
    name="kd490",
    standard_name="volume_attenuation_coefficient_of_downwelling_radiative_flux_in_sea_water",
    long_name="Diffuse attenuation coefficient at 490 nm",
    units="m-1",
    aliases={"cmems": "KD490"},
    valid_range=(0.0, 10.0),
)

ZSD = Variable(
    name="zsd",
    standard_name="secchi_depth_of_sea_water",
    long_name="Secchi disk depth",
    units="m",
    aliases={"cmems": "ZSD"},
    valid_range=(0.0, 100.0),
)

SPM = Variable(
    name="spm",
    standard_name="mass_concentration_of_suspended_matter_in_sea_water",
    long_name="Total suspended matter",
    units="g m-3",
    aliases={"cmems": "SPM"},
    valid_range=(0.0, 1000.0),
)

BBP443 = Variable(
    name="bbp443",
    standard_name="volume_backwards_scattering_coefficient_of_radiative_flux_in_sea_water",
    long_name="Particulate backscattering at 443 nm",
    units="m-1",
    aliases={"cmems": "BBP443"},
    valid_range=(0.0, 1.0),
)

PP = Variable(
    name="pp",
    standard_name="net_primary_production_of_biomass_expressed_as_carbon_per_unit_area_in_sea_water",
    long_name="Primary production",
    units="mg m-2 day-1",
    aliases={"cmems": "PP"},
    valid_range=(0.0, 10000.0),
)

# Remote-sensing reflectance — one Variable per wavelength so CF metadata
# and ``valid_range`` stay per-band (Rrs spectra differ across sensors).

RRS412 = Variable(
    name="rrs412",
    standard_name="surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air",
    long_name="Remote sensing reflectance at 412 nm",
    units="sr-1",
    aliases={"cmems": "RRS412"},
    valid_range=(0.0, 0.1),
)

RRS443 = Variable(
    name="rrs443",
    standard_name="surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air",
    long_name="Remote sensing reflectance at 443 nm",
    units="sr-1",
    aliases={"cmems": "RRS443"},
    valid_range=(0.0, 0.1),
)

RRS490 = Variable(
    name="rrs490",
    standard_name="surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air",
    long_name="Remote sensing reflectance at 490 nm",
    units="sr-1",
    aliases={"cmems": "RRS490"},
    valid_range=(0.0, 0.1),
)

RRS510 = Variable(
    name="rrs510",
    standard_name="surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air",
    long_name="Remote sensing reflectance at 510 nm",
    units="sr-1",
    aliases={"cmems": "RRS510"},
    valid_range=(0.0, 0.1),
)

RRS555 = Variable(
    name="rrs555",
    standard_name="surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air",
    long_name="Remote sensing reflectance at 555 nm",
    units="sr-1",
    aliases={"cmems": "RRS555"},
    valid_range=(0.0, 0.1),
)

RRS670 = Variable(
    name="rrs670",
    standard_name="surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air",
    long_name="Remote sensing reflectance at 670 nm",
    units="sr-1",
    aliases={"cmems": "RRS670"},
    valid_range=(0.0, 0.1),
)

# ---- Ocean biogeochemistry ----------------------------------------------

NO3 = Variable(
    name="no3",
    standard_name="mole_concentration_of_nitrate_in_sea_water",
    long_name="Nitrate concentration",
    units="mmol m-3",
    aliases={"cmems": "no3"},
    valid_range=(0.0, 100.0),
)

PO4 = Variable(
    name="po4",
    standard_name="mole_concentration_of_phosphate_in_sea_water",
    long_name="Phosphate concentration",
    units="mmol m-3",
    aliases={"cmems": "po4"},
    valid_range=(0.0, 10.0),
)

SI = Variable(
    name="si",
    standard_name="mole_concentration_of_silicate_in_sea_water",
    long_name="Silicate concentration",
    units="mmol m-3",
    aliases={"cmems": "si"},
    valid_range=(0.0, 200.0),
)

O2 = Variable(
    name="o2",
    standard_name="mole_concentration_of_dissolved_molecular_oxygen_in_sea_water",
    long_name="Dissolved oxygen",
    units="mmol m-3",
    aliases={"cmems": "o2"},
    valid_range=(0.0, 500.0),
)

PHYC = Variable(
    name="phyc",
    standard_name="mole_concentration_of_phytoplankton_expressed_as_carbon_in_sea_water",
    long_name="Phytoplankton carbon concentration",
    units="mmol m-3",
    aliases={"cmems": "phyc"},
    valid_range=(0.0, 50.0),
)

ZOOC = Variable(
    name="zooc",
    standard_name="mole_concentration_of_zooplankton_expressed_as_carbon_in_sea_water",
    long_name="Zooplankton carbon concentration",
    units="mmol m-3",
    aliases={"cmems": "zooc"},
    valid_range=(0.0, 50.0),
)

PH = Variable(
    name="ph",
    standard_name="sea_water_ph_reported_on_total_scale",
    long_name="Sea water pH",
    units="1",
    aliases={"cmems": "ph"},
    valid_range=(7.0, 9.0),
)

SPCO2 = Variable(
    name="spco2",
    standard_name="surface_partial_pressure_of_carbon_dioxide_in_sea_water",
    long_name="Surface partial pressure of CO2",
    units="Pa",
    aliases={"cmems": "spco2"},
    valid_range=(0.0, 100.0),
)


REGISTRY: dict[str, Variable] = {
    v.name: v
    for v in (
        SST,
        SSH,
        SLA,
        MDT,
        ADT,
        UO,
        VO,
        UGOS,
        VGOS,
        SO,
        SOS,
        DENS,
        ICE_CONC,
        T2M,
        D2M,
        U10,
        V10,
        MSL,
        TP,
        SP,
        SSRD,
        # Ocean colour
        CHL,
        KD490,
        ZSD,
        SPM,
        BBP443,
        PP,
        RRS412,
        RRS443,
        RRS490,
        RRS510,
        RRS555,
        RRS670,
        # Biogeochemistry
        NO3,
        PO4,
        SI,
        O2,
        PHYC,
        ZOOC,
        PH,
        SPCO2,
    )
}


def resolve(v: str | Variable) -> Variable:
    """Return a :class:`Variable` given an instance or its canonical name.

    Raises:
        KeyError: if ``v`` is a string not present in :data:`REGISTRY`.
    """
    if isinstance(v, Variable):
        return v
    try:
        return REGISTRY[v]
    except KeyError as exc:
        raise KeyError(
            f"Unknown variable {v!r}. Known: {sorted(REGISTRY)}. "
            "Pass a Variable instance to use a variable outside the registry."
        ) from exc


def register(variable: Variable) -> Variable:
    """Add ``variable`` to :data:`REGISTRY` (keyed on :attr:`Variable.name`)."""
    REGISTRY[variable.name] = variable
    return variable
