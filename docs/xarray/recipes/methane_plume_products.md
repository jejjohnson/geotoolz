# Open a methane plume scene and select one plume

An emission-rate fit for one catalogued plume needs four things from the product files: the matched-filter enhancement, its per-pixel sensitivity and uncertainty on the same grid, and the plume's outline and source point. EMIT ships the three rasters as separate COGs (`EMIT_L2B_CH4ENH`, `EMIT_L2B_CH4SENS`, `EMIT_L2B_CH4UNCERT`) and the outline as a plume-complex GeoJSON; AVIRIS-3 ships `*_CH4_ORT`, `*_CH4_SNS_ORT` and `*_CH4_UNC_ORT`. The openers read the rasters from NetCDF, so convert the GeoTIFFs first (for example with `rioxarray.open_rasterio(...).to_netcdf`), and install the NetCDF backend with `pip install 'xrtoolz-products[local]'`.

## The scene with its companions

```python
import xarray as xr
from xrproducts import open_emit_ch4_l2b

scene: xr.Dataset = open_emit_ch4_l2b(
    "EMIT_L2B_CH4ENH_002_20231006T190153_2327912_020.nc",
    sensitivity_path="EMIT_L2B_CH4SENS_002_20231006T190153_2327912_020.nc",
    uncertainty_path="EMIT_L2B_CH4UNCERT_002_20231006T190153_2327912_020.nc",
)
enhancement = scene["ch4_enhancement"]   # (1, N_y, N_x) float, ppm m, as stored
sensitivity = scene["ch4_sensitivity"]   # (1, N_y, N_x) float, unitless
uncertainty = scene["ch4_uncertainty"]   # (1, N_y, N_x) float, ppm m
```

The three share one `(time, y, x)` grid: a companion of another shape, or with coordinates more than 1e-6 away from the enhancement's, raises `ProductGridError` naming the file. The products' -9999 no-data value is `NaN` in all three. The enhancement is never divided by the sensitivity, and `uncertainty.attrs["uncertainty_of"]` says which quantity the uncertainty is the standard deviation of (`"truth"` by default, which is unverified against the user guides). For AVIRIS-3, `open_av3_ch4_l2b` takes the same two keywords and returns the same names, dims and units.

## One plume

```python
import numpy as np
from xrproducts import open_ch4_plume_complexes, plume_outline

plumes: xr.Dataset = open_ch4_plume_complexes("EMIT_L2B_CH4PLMMETA_002_20231006T190153_001234.json")
plume_id = str(plumes["plume"].values[0])
plume = plumes.sel(plume=plume_id)
observed = plume["time"].values           # () datetime64[ns], "UTC Time Observed"
origin_lon = plume["origin_lon"].item()   # () float, NaN when the file has no pseudo-origin
ring: np.ndarray = plume_outline(plumes, plume_id)   # (V_p, 2) float64, (lon, lat) rows
```

The outlines are a CF contiguous ragged array: `outline_lon` / `outline_lat` hold every exterior ring end to end on `vertex`, and `vertex_count` says how many vertices each plume owns. The DAAC plume files carry no pseudo-origin, so `origin_lon` / `origin_lat` are `NaN` for them; JPL's plume lists carry one as `"Psuedo-Origin"` (sic), in either axis order, and the opener picks the order that falls on the outline and records it in `plumes.attrs["origin_axis_order"]`.
