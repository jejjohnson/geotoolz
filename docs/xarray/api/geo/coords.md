# Validation & Coordinates

Coordinate-system hygiene for incoming datasets: validate longitude /
latitude / time axes, decode CF time, rename coordinates and variables to
or from CF standard names, and measure grid resolution.

!!! note "CF standard names"
    `RenameToCFStandardNames` / `RenameFromCFStandardNames` use the curated
    `Variable` registry that now lives in
    [`xrproducts.types`](../reader.md) — the same
    registry that drives the colormap lookup in `xrops.viz`.

## Coordinate validation

::: xrops.cf.operators.ValidateLongitude

::: xrops.cf.operators.ValidateLatitude

::: xrops.cf.operators.ValidateCoords

::: xrops.cf.operators.ValidateTime

::: xrops.cf.operators.DecodeCFTime

## Renaming & CF standard names

::: xrops.cf.operators.RenameCoords

::: xrops.cf.operators.RenameVariables

::: xrops.cf.operators.RenameToCFStandardNames

::: xrops.cf.operators.RenameFromCFStandardNames

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.cf.check_dataset_coords

::: xrops.cf.validate_longitude

::: xrops.cf.validate_latitude

::: xrops.cf.validate_time

::: xrops.cf.decode_cf_time

::: xrops.cf.rename_coords

::: xrops.cf.rename_variables

::: xrops.cf.rename_to_cf_standard_names

::: xrops.cf.rename_from_cf_standard_names

::: xrops.geom.calc_latlon

::: xrops.geom.get_dataset_resolution

::: xrops.wavelets.median_dx_km

::: xrops.wavelets.geometric_scales

::: xrops.wavelets.scale_to_wavenumber

::: xrops.wavelets.wavenumber_to_scale
