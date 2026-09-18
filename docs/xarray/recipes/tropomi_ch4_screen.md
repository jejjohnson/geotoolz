# Open a TROPOMI CH₄ granule and screen it

A Sentinel-5P TROPOMI methane L2 granule (`S5P_*_L2__CH4____*.nc`) spreads one
retrieval over four NetCDF groups. `LocalL2Source` flattens them into a single
CF Dataset, renames the variables to the registry names, and applies the
quality, bounding-box and time screening in one call. The file itself has to
be on disk already (Earthdata acquisition is a separate step), and reading it
needs a NetCDF backend: `pip install 'xrtoolz-reader[local]'`.

## One granule, screened

```python
from xrreader import BBox, LocalL2Source, TimeRange

src = LocalL2Source()
ds = src.open(
    "tropomi.ch4",
    path="S5P_RPRO_L2__CH4____20190601T104423_20190601T122553_08532_03_020400.nc",
    bbox=BBox(lon_min=4.0, lon_max=6.0, lat_min=51.0, lat_max=53.0),
    time=TimeRange.parse("2019-06-01T11:00", "2019-06-01T12:00"),
    qa_min=0.5,   # PUM recommendation; None keeps the opener default (0.5), 0.0 keeps all
)
ds
```

```text
<xarray.Dataset>
Dimensions:                  (time: 1, scanline: 212, ground_pixel: 31, layer: 12, ...)
Coordinates:
    time                     (time) datetime64[ns] 2019-06-01
    scanline_time            (time, scanline) datetime64[ns] ...
    lon                      (time, scanline, ground_pixel) float32 ...
    lat                      (time, scanline, ground_pixel) float32 ...
Data variables:
    xch4                     (time, scanline, ground_pixel) float32 ...
    xch4_bias_corrected      (time, scanline, ground_pixel) float32 ...
    xch4_precision           (time, scanline, ground_pixel) float32 ...
    qa_value                 (time, scanline, ground_pixel) float32 ...
    column_averaging_kernel  (time, scanline, ground_pixel, layer) float32 ...
    ch4_profile_apriori      (time, scanline, ground_pixel, layer) float32 ...
    dry_air_subcolumns       (time, scanline, ground_pixel, layer) float32 ...
    sp                       (time, scanline, ground_pixel) float32 ...
    pressure_interval        (time, scanline, ground_pixel) float32 ...
    ...
```

What happened:

- `PRODUCT`, `DETAILED_RESULTS`, `INPUT_DATA` and `GEOLOCATIONS` were merged
  on the shared `(time, scanline, ground_pixel)` swath; the averaging kernel
  and a-priori profile keep their `layer` axis.
- `latitude` / `longitude` became the 2-D `lat` / `lon` coordinates, so
  `bbox` was applied with `subset_bbox` (`where(..., drop=True)` — scanlines
  and ground pixels entirely outside the box are dropped).
- `time` is the granule reference time; the per-scanline `delta_time` became
  `scanline_time`, which the `time=` window masks on.
- Pixels with `qa_value < 0.5` are NaN in every data variable (shape kept).

## Only the opener

Skip the screening and keep the whole swath — or the raw product names — with
the opener directly:

```python
from xrreader import open_tropomi_ch4_l2

raw = open_tropomi_ch4_l2(path, qa_min=None, keep_groups=True)   # product names
ds = open_tropomi_ch4_l2(path)                                    # canonical, qa >= 0.5
```

## Several granules

`paths=[...]` concatenates granules along `time` in the order given (they must
share the swath shape):

```python
ds = src.open("tropomi.ch4", paths=[granule_a, granule_b], bbox=box)
```

## Pressure per layer

The averaging kernel is applied on the product's pressure grid, which is not
stored per layer. Rebuild it from `sp` and `pressure_interval`:

```python
p_layer = ds["sp"] - ds["layer"] * ds["pressure_interval"]   # Pa, (time, scanline, ground_pixel, layer)
```

## The same call for EMIT and GHGSat scenes

```python
emit = src.open("emit.ch4", path="EMIT_L2B_CH4ENH_...nc", glt_path="EMIT_..._GLT.nc", bbox=box)
scenes = src.open("emit.ch4", paths=[scene_a, scene_b], glt_paths=[glt_a, glt_b])
plume = src.open("ghgsat.ch4", path="GHGSat_plume.nc")
```

Both come back on `(time, y, x)` with 2-D `lon` / `lat`, so `subset_bbox`
and the `xrtoolz.geo` operators run on them unchanged. See
[Local L2 products](../api/reader.md#local-l2-products) for the full API.
