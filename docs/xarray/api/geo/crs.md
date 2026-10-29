# CRS & Reprojection

Assign or read a coordinate reference system (via `rioxarray` / `pyproj`),
reproject a dataset, convert between geographic (lon/lat) and projected
(x/y) coordinates, and express coordinates in a metric local frame anchored
at a site (UTM zone picked from the origin).

## Operators

::: xrops.geo.operators.AssignLocalXY

## Local metric frames

`local_frame(origin_lon, origin_lat)` picks the UTM zone containing the
origin and returns a frozen, JSON-serialisable `LocalFrame` whose `to_xy`
gives metres east / north of the origin and whose `to_lonlat` inverts it.
`assign_local_xy` stamps those coordinates onto a dataset (1-D rectilinear,
2-D swath, or paired 1-D along-track lon/lat on one dim); `calc_latlon` is
the inverse direction for gridded
rasters that already carry projected `x`/`y` axes.

::: xrops.geo.utm_crs_for

::: xrops.geo.LocalFrame

::: xrops.geo.local_frame

::: xrops.geo.assign_local_xy

## Functions

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrops.geo.assign_crs

::: xrops.geo.get_crs

::: xrops.geo.reproject

::: xrops.geo.lonlat_to_xy

::: xrops.geo.xy_to_lonlat
