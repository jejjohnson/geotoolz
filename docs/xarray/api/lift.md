# Lifting geoops operators over cubes (`xrops.lift`)

geoops geometry operators (`Reproject`, `Resample`, `Gradient2D`, …) take their geometry from each item's grid, so they need a `geoarray.GeoArray` per item and refuse a `DataArray`.
`xrops.lift(op, dim=...)` runs such an operator on every item of a cube along `dim` (geostack OP-O4): each item becomes a `GeoArray` (`GeoArray.from_xarray`, reading the CF `spatial_ref`), goes through the operator, and comes back with `to_xarray`.

```python
import geoops.geom as gg
import xrops

# cube: DataArray["time band y x"] with a CF spatial_ref (geoarray's to_xarray layout)
reprojected = xrops.lift(gg.Reproject(dst_crs="EPSG:3857"), dim="time")(cube)
gradient = xrops.lift(gg.Gradient2D(nan_policy="adaptive"), dim="time")(cube)
#   DataArray["time band component y x"]
```

On a dask cube the lift is lazy, with one task per `dim` chunk; the output's grid comes from running the operator once on a zero-filled item, so building the graph computes nothing.
Each item must be whole, so a cube chunked along any other dim raises and names the `.chunk({...: -1})` that fixes it.
The coordinate along `dim` (the time stamps) is carried through untouched.

::: xrops.lift

::: xrops.Lift
