---
status: draft
version: 0.1.0
---

!!! note "Imports in this page are from the original `geo_toolz` layout"
    These design docs were adapted from the `geo_toolz` design study.
    Code snippets use the original feature-based import paths
    (`geo_toolz.<module>`). In `xr_toolz`, the domain-agnostic
    operations live under `xr_toolz.geo.<module>`; physics-specific
    operations live under `xr_toolz.ocn` / `xr_toolz.atm` /
    `xr_toolz.rs`. See `xr_toolz/__init__.py` and
    `xr_toolz/geo/__init__.py` for the current export surface.

# Layer 1 — Component Examples

Sequential pipelines and operator composition. *(P1: everything is an operator, P2: progressive disclosure)*

---

## Full Preprocessing Pipeline

### Sequential: linear chain of single-input operators

```python
from geo_toolz.core import Sequential
from geo_toolz.validation import ValidateCoords
from geo_toolz.subset import SubsetBBox, SubsetTime
from geo_toolz.regrid import Regrid
from geo_toolz.detrend import CalculateClimatology, RemoveClimatology
from geo_toolz.masks import AddOceanMask

# Learn climatology from training period (D2: split-object pattern)
ds_train = xr.open_dataset("ssh_2000_2020.nc")
clim = CalculateClimatology(freq="day", smoothing=60)(ds_train)

# Build pipeline — P1: every step is an Operator
preprocess = Sequential([
    ValidateCoords(),
    SubsetBBox(lon_bnds=(-30, 45), lat_bnds=(25, 65)),
    SubsetTime(time_min="2020-01", time_max="2023-12"),
    Regrid(target_lon=target_lon, target_lat=target_lat, method="linear"),
    AddOceanMask(ocean="mediterranean"),
    RemoveClimatology(clim),
])

# Apply — P3: xarray in, xarray out
ds_clean = preprocess(ds_raw)

# Introspect
print(preprocess.describe())
```

---

## Pipe Syntax (Operator |)

### Sugar for Sequential via `__or__`

```python
pipeline = (
    ValidateCoords()
    | Regrid(target_lon, target_lat)
    | RemoveClimatology(clim)
    | SubsetBBox(lon_bnds=(-30, 45), lat_bnds=(25, 65))
)

ds_clean = pipeline(ds_raw)
```

---

## Evaluation Pipeline

### Multi-input operators at Layer 1 (eager)

```python
from geo_toolz.metrics import RMSE, NRMSE, PSDScore

# P1: metrics are operators too — same __call__ interface
rmse = RMSE(variable="ssh", dims=["time"])(ds_pred, ds_ref)
nrmse = NRMSE(variable="ssh", dims=["time"])(ds_pred, ds_ref)
psd_score = PSDScore(variable="ssh", dims=["lat", "lon"])(ds_pred, ds_ref)
```

---

## Hydra / hydra-zen Integration

### Auto-generate configs from operators (Hydra-serializable via get_config)

```python
from hydra_zen import builds, instantiate

# Auto-generate configs from operators
RegridConf = builds(Regrid, target_lon="${target_lon}", target_lat="${target_lat}", method="linear")
SubsetConf = builds(SubsetBBox, lon_bnds=(-30, 45), lat_bnds=(25, 65))

# Store as YAML, load and instantiate later
regrid_op = instantiate(RegridConf, target_lon=target_lon, target_lat=target_lat)
```

---

## Nested Pipelines

### Sequential is an Operator — it nests

```python
# P1: a pipeline is itself an operator — it composes
preprocess = Sequential([ValidateCoords(), Regrid(grid)])
full = Sequential([preprocess, RemoveClimatology(clim), SubsetBBox(region)])

ds_clean = full(ds_raw)
```
