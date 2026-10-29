# Validation & Coordinates

Coordinate-system hygiene for incoming datasets: validate longitude /
latitude / time axes, decode CF time, rename coordinates and variables to
or from CF standard names, and measure grid resolution.

!!! note "CF standard names"
    `RenameToCFStandardNames` / `RenameFromCFStandardNames` use the curated
    `Variable` registry that now lives in
    [`xrreader.types`](../reader.md) — the same
    registry that drives the colormap lookup in `xrops.viz`.

## Coordinate validation

::: xrops.geo.operators.ValidateLongitude

::: xrops.geo.operators.ValidateLatitude

::: xrops.geo.operators.ValidateCoords

::: xrops.geo.operators.ValidateTime

::: xrops.geo.operators.DecodeCFTime

## Renaming & CF standard names

::: xrops.geo.operators.RenameCoords

::: xrops.geo.operators.RenameVariables

::: xrops.geo.operators.RenameToCFStandardNames

::: xrops.geo.operators.RenameFromCFStandardNames

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.geo.check_dataset_coords

::: xrops.geo.validate_longitude

::: xrops.geo.validate_latitude

::: xrops.geo.validate_time

::: xrops.geo.decode_cf_time

::: xrops.geo.rename_coords

::: xrops.geo.rename_variables

::: xrops.geo.rename_to_cf_standard_names

::: xrops.geo.rename_from_cf_standard_names

::: xrops.geo.calc_latlon

::: xrops.geo.get_dataset_resolution

::: xrops.geo.median_dx_km

::: xrops.geo.geometric_scales

::: xrops.geo.scale_to_wavenumber

::: xrops.geo.wavenumber_to_scale
