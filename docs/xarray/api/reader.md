# API Reference

## Variable registry additions

**ERA5 pressure levels.** `reanalysis-era5-pressure-levels` (catalog short
name `era5.pressure_levels`) now describes the CDS pressure-level fields
`u`, `v`, `w` (omega, Pa s⁻¹), `t`, `z` (geopotential) and `q`, each with a
`cds` alias so
`CDSSource(...).download("reanalysis-era5-pressure-levels", Path("uv.nc"), variables=["u", "v"], levels=PressureLevels([1000, 925]))`
emits the CDS form names without hand-typing them (`era5.pressure_levels`
is the catalog short name for that dataset id — see
`describe("era5.pressure_levels")`). The single-level preset gains `blh`
(boundary-layer height). The `wrf` aliases on `u`, `v` and `blh` name the
raw `wrfout` variables that map onto the same CF quantity; `q` carries no
`wrf` alias because `QVAPOR` is a mixing ratio w.r.t. dry air and must be
converted (`q = r / (1 + r)`) before it is CF specific humidity.

**Methane / trace gas.** `xch4`, `xch4_bias_corrected`, `xch4_precision`,
`column_averaging_kernel`, `ch4_profile_apriori`, `dry_air_subcolumns` and
`qa_value` carry the TROPOMI L2 product names as `tropomi` aliases;
`ch4_enhancement` (ppm m) is the EMIT L2B plume-enhancement quantity, and
`ch4_column` (kg m⁻², `atmosphere_mass_content_of_methane`) is the canonical
column-mass output of the `xrtoolz.atm.gas.ch4` operators. All of them
resolve through `xrreader.resolve` and stamp CF attributes through
`apply_cf_attrs`.

::: xrreader
