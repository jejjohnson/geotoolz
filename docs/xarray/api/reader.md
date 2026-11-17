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
column-mass output of the `xrops.atm.gas.ch4` operators. All of them
resolve through `xrproducts.resolve` and stamp CF attributes through
`apply_cf_attrs`.

## Local L2 products

`LocalL2Source` is the `DataSource` for satellite L2 product files that are
already on disk — nothing is fetched (`download()` raises
`NotImplementedError`). The catalog short names `tropomi.ch4`, `emit.ch4`,
`av3.ch4` and `ghgsat.ch4` (`source="local"`) each map to a product *opener*
that turns one file into a flat CF Dataset. The openers read NetCDF, and the core install
ships no backend for it — install the `local` extra
(`pip install 'xrtoolz-products[local]'`, which adds `netcdf4`):

| Short name | Opener | Kind | Layout |
|---|---|---|---|
| `tropomi.ch4` | `open_tropomi_ch4_l2(path, qa_min=0.5, keep_groups=False)` | `DatasetKind.SWATH` | `(time, scanline, ground_pixel[, layer, level, corner])` |
| `emit.ch4` | `open_emit_ch4_l2b(path, sensitivity_path=None, uncertainty_path=None, glt_path=None)` | `DatasetKind.SCENE` | `(time, y, x)` |
| `av3.ch4` | `open_av3_ch4_l2b(path, sensitivity_path=None, uncertainty_path=None)` | `DatasetKind.SCENE` | `(time, y, x)` |
| `ghgsat.ch4` | `open_ghgsat_ch4_l2(path)` | `DatasetKind.SCENE` | `(time, y, x)` |

**TROPOMI.** The four product groups (`PRODUCT` and
`PRODUCT/SUPPORT_DATA/{DETAILED_RESULTS,INPUT_DATA,GEOLOCATIONS}`) are read
with `xarray.open_datatree` and merged on the shared swath dims. Variables are
renamed through the registry's `tropomi` aliases (`methane_mixing_ratio` →
`xch4`, `methane_mixing_ratio_bias_corrected` → `xch4_bias_corrected`,
`methane_mixing_ratio_precision` → `xch4_precision`,
`methane_profile_apriori` → `ch4_profile_apriori`, `surface_pressure` → `sp`;
`column_averaging_kernel`, `dry_air_subcolumns` and `qa_value` keep their
names, `pressure_interval` has no registry entry and is passed through), and
`latitude` / `longitude` become the 2-D `lat` / `lon` coordinates. Time keeps
the product's shape: the length-1 `time` index is the granule reference time
(`PRODUCT/time`, seconds since 2010-01-01) and the per-scanline `delta_time`
offset is folded into a `scanline_time` `(time, scanline)` datetime64
coordinate. `qa_value < qa_min` pixels are set to NaN (`subset_where`, shape
preserved); `keep_groups=True` keeps the raw product names.

**EMIT.** The L2B enhancement raster (`methane_plume_complex` as distributed,
or the registry alias `ch4_enhancement` — either becomes `ch4_enhancement` in
ppm m — plus `ch4_uncertainty` when present) is read from a NetCDF container;
`rioxarray` is not an `xrproducts` dependency, so GeoTIFF distributions have to
be converted first. With `glt_path` the scene is orthorectified through the
geometric lookup table (`glt_x` / `glt_y`, 1-based, `0` = no data) and the
2-D `lon` / `lat` of the orthorectified grid come from the GLT's own `lon` /
`lat` or its GDAL `geotransform` attribute (full six-coefficient affine). A
scalar `time` variable on the raster is carried through as the scene
timestamp. Several scenes take one lookup table each via
`glt_paths=[...]`, zipped with `paths=[...]`.

`sensitivity_path` and `uncertainty_path` add `ch4_sensitivity` (unitless) and `ch4_uncertainty` (ppm m) from the product's separate `EMIT_L2B_CH4SENS` and `EMIT_L2B_CH4UNCERT` files. Each companion holds the registry name or a single `(y, x)` variable, and must have the enhancement's `(y, x)` shape and the same `lon` / `lat` / `x` / `y` to 1e-6 wherever both files carry them; otherwise `ProductGridError` names the file, and nothing is resampled. With a lookup table, the companions go through the same table as the enhancement. The enhancement is returned as stored (the matched-filter score, never divided by the sensitivity), and all three rasters are floats with the products' -9999 no-data value as `NaN` (recorded as `source_fill_value`). The uncertainty carries `uncertainty_of = "truth"`, meaning the standard deviation of the enhancement divided by the sensitivity, which is how JPL's SPIIRE uses it; the user guides do not say, so treat it as unverified.

**AVIRIS-3.** `open_av3_ch4_l2b` reads the AVIRIS-3 L2B greenhouse-gas scenes (ORNL DAAC, doi:10.3334/ORNLDAAC/2358: `*_CH4_ORT`, `*_CH4_SNS_ORT` and `*_CH4_UNC_ORT`, orthocorrected in UTM) converted to NetCDF, and returns the EMIT opener's variables, dims and units. It has no lookup table, squeezes a length-1 `band` axis, sets the companions' flare marker -1 to `NaN` (`flare_value`), and takes the scene time from the file or else from the `AV3YYYYMMDDthhmmss` flight prefix of the file name.

**Plume complexes.** `open_ch4_plume_complexes(path)` reads a plume-complex GeoJSON (the EMIT DAAC `EMIT_L2B_CH4PLMMETA_*.json`, or JPL's multi-plume lists) into a Dataset on `plume`, indexed by `"Plume ID"`, with `time` (`"UTC Time Observed"`), `scene_ids` (`"Scene FIDs"` or `"fids"`, comma-joined), the pseudo-origin `origin_lon` / `origin_lat` (`"Psuedo-Origin"`, `NaN` when absent, as in the DAAC files) and the exterior rings as a CF contiguous ragged array (`vertex_count` on `plume`, `outline_lon` / `outline_lat` on `vertex`). `plume_outline(plumes, plume_id)` returns one ring as `(lon, lat)` rows. `Point` features are skipped, a multi-polygon keeps its largest polygon (`attrs["dropped_polygons"]`), and a feature without an identifier or a geometry raises `MissingPropertyError`. See the [methane plume products recipe](../recipes/methane_plume_products.md).

**GHGSat.** A per-plume NetCDF with `xch4` (ppb) and/or `ch4_enhancement` on
`(y, x)` and 2-D `latitude` / `longitude`, promoted to `lon` / `lat` coords.

`LocalL2Source.open(dataset_id, path=..., bbox=..., time=..., qa_min=...,
variables=...)` applies the screening after the opener: `bbox` through
`subset_bbox` on the 2-D coordinates (`where(..., drop=True)`) after the box
is normalised to the dataset's longitude convention (`[-180, 180]` when any
`lon` is negative, `[0, 360]` when any exceeds 180 — a box that still crosses
the antimeridian afterwards raises), `time` as a per-scanline mask on
`scanline_time` for swaths or a mask on the `time` coordinate for scenes, and
`paths=[a, b]` concatenates granules along `time` in the order given.
`subset_bbox`, `subset_where` and `subset_time` now live in
`xrproducts.types` (pure xarray) and are re-exported unchanged by
`xrops.geom`, so the reader can screen without importing `xrtoolz`. See the
[TROPOMI recipe](../recipes/tropomi_ch4_screen.md) for a worked example.

**Emission inventories.** `open_ch4_inventory(path, variable=None, units=None,
source=None)` opens a gridded CH₄ inventory (EDGAR, GFEI, EPA) as
`ch4_emission_flux` in kg m⁻² s⁻¹ on `([time,] lat, lon)`, with `lat`
ascending and `lon` wrapped to `[-180, 180)`. The file's area-flux units are
converted through `UNIT_FACTORS` (kg m⁻² s⁻¹, g m⁻² s⁻¹, Mg km⁻² a⁻¹ with a
365.25-day year); the factor and the original units are kept as attributes.
Per-cell totals such as EDGAR's `emi` files (tonnes per cell, per year or per
month) are rejected because the period is not in the file: read the `flx`
files instead. It is a plain opener, not a catalog entry.

::: xrproducts
