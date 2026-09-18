# CRS & Reprojection

Assign or read a coordinate reference system (via `rioxarray` / `pyproj`),
reproject a dataset, convert between geographic (lon/lat) and projected
(x/y) coordinates, and express coordinates in a metric local frame anchored
at a site (UTM zone picked from the origin).

## Operators

::: xrtoolz.geo.operators.AssignLocalXY

## Local metric frames

`local_frame(origin_lon, origin_lat)` picks the UTM zone containing the
origin and returns a frozen, JSON-serialisable `LocalFrame` whose `to_xy`
gives metres east / north of the origin and whose `to_lonlat` inverts it.
`assign_local_xy` stamps those coordinates onto a dataset (1-D rectilinear,
2-D swath, or paired 1-D along-track lon/lat on one dim); `calc_latlon` is
the inverse direction for gridded
rasters that already carry projected `x`/`y` axes.

::: xrtoolz.geo.utm_crs_for

::: xrtoolz.geo.LocalFrame

::: xrtoolz.geo.local_frame

::: xrtoolz.geo.assign_local_xy

## Functions

These pure functions back the operators above; each takes `xr.DataArray`/`xr.Dataset` and a `dim:` argument.

::: xrtoolz.geo.assign_crs

::: xrtoolz.geo.get_crs

::: xrtoolz.geo.reproject

::: xrtoolz.geo.lonlat_to_xy

::: xrtoolz.geo.xy_to_lonlat
