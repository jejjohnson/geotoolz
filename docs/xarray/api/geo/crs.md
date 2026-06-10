# CRS & Reprojection

Assign or read a coordinate reference system (via `rioxarray` / `pyproj`),
reproject a dataset, and convert between geographic (lon/lat) and projected
(x/y) coordinates.

## Functions

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.geo.assign_crs

::: xrtoolz.geo.get_crs

::: xrtoolz.geo.reproject

::: xrtoolz.geo.lonlat_to_xy

::: xrtoolz.geo.xy_to_lonlat
