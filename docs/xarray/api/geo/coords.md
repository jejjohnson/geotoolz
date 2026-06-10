# Validation & Coordinates

Coordinate-system hygiene for incoming datasets: validate longitude /
latitude / time axes, decode CF time, rename coordinates and variables to
or from CF standard names, and measure grid resolution.

!!! note "CF standard names"
    `RenameToCFStandardNames` / `RenameFromCFStandardNames` use the curated
    `Variable` registry that now lives in
    [`xrreader.types`](https://github.com/jejjohnson/xrreader) — the same
    registry that drives the colormap lookup in `xrtoolz.viz`.

## Coordinate validation

::: xrtoolz.geo.operators.ValidateLongitude

::: xrtoolz.geo.operators.ValidateLatitude

::: xrtoolz.geo.operators.ValidateCoords

::: xrtoolz.geo.operators.ValidateTime

::: xrtoolz.geo.operators.DecodeCFTime

## Renaming & CF standard names

::: xrtoolz.geo.operators.RenameCoords

::: xrtoolz.geo.operators.RenameVariables

::: xrtoolz.geo.operators.RenameToCFStandardNames

::: xrtoolz.geo.operators.RenameFromCFStandardNames

## Functional primitives (Layer 0)

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.geo.check_dataset_coords

::: xrtoolz.geo.validate_longitude

::: xrtoolz.geo.validate_latitude

::: xrtoolz.geo.validate_time

::: xrtoolz.geo.decode_cf_time

::: xrtoolz.geo.rename_coords

::: xrtoolz.geo.rename_variables

::: xrtoolz.geo.rename_to_cf_standard_names

::: xrtoolz.geo.rename_from_cf_standard_names

::: xrtoolz.geo.calc_latlon

::: xrtoolz.geo.get_dataset_resolution

::: xrtoolz.geo.median_dx_km

::: xrtoolz.geo.geometric_scales

::: xrtoolz.geo.scale_to_wavenumber

::: xrtoolz.geo.wavenumber_to_scale
