# Index and load model states

Index an ensemble forecast archive (one NetCDF per run and member) and
load it back as an `(init_time, member, lead_time, y, x)` cube, with every
run kept. `geocatalog.states` adds the run time, member, kind, producer and
parents to an ordinary xarray catalog, and queries them by type.

## Why not `load_xarray`

A model state has two times: the **run time** $r$ (when the run started)
and the **valid time** $v$ (when the state is valid). Their difference is
the **lead time**, $\tau = v - r$. Two runs started a day apart are both
valid tomorrow, at leads 1 and 2 days. `load_xarray` treats files that
cover the same time as duplicates and keeps one; `load_states` keeps both,
because they are two different states.

## Index and load an ensemble history

Write one file per (run, member), index the files with
`build_state_catalog`, and load the forecasts:

```python
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import geocatalog as gc
from geocatalog.backends import InMemoryGeoCatalog
from geocatalog.build import build_state_catalog
from geocatalog.states import CF_NAMES, forecasts, load_states

# Shapes: R = 3 runs, N_e = 4 members, L = 2 leads, H x W = 4 x 5 cells
runs: pd.DatetimeIndex = pd.date_range("2026-01-01", periods=3, freq="D")
history: np.ndarray = np.random.default_rng(0).standard_normal((3, 4, 2, 4, 5))  # (R, N_e, L, H, W) float64
root: Path = Path(tempfile.mkdtemp())
paths: list[Path] = []
for r, run in enumerate(runs):
    for m in range(4):
        path: Path = root / f"fc_{run:%Y%m%d}_m{m}.nc"
        xr.Dataset(
            {"xch4": (("time", "lat", "lon"), history[r, m])},                # (L, H, W) float64
            coords={
                "time": run + pd.to_timedelta([1, 2], unit="D"),             # valid times v = r + τ
                "lat": 31.5 + 0.5 * np.arange(4),
                "lon": -104.0 + 0.5 * np.arange(5),
            },
            attrs={                                                          # CF names, plus three of ours
                "forecast_reference_time": run.isoformat(),
                "realization": m,
                "kind": "forecast",
                "producer": "regional-ch4-v1",
                "parents": f"ana_{run:%Y%m%d}.nc",
            },
        ).to_netcdf(path)
        paths.append(path)

catalog: InMemoryGeoCatalog = build_state_catalog(paths, crs="EPSG:4326", names=CF_NAMES)  # 12 rows
aoi: gc.GeoSlice = gc.GeoSlice(
    bounds=(-104.5, 31.0, -101.5, 33.5),
    interval=pd.Interval(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-06"), closed="both"),
    resolution=(0.5, 0.5),
    crs="EPSG:4326",
)
cube: xr.Dataset = load_states(forecasts(catalog), aoi)                     # (3, 4, 2, 4, 5) float64
assert np.array_equal(cube["xch4"].values, history)                         # every run, exactly
```

`cube` has dimensions `init_time`, `member` and `lead_time` before the
grid, and a 2-D `valid_time = init_time + lead_time` coordinate. Members
are text (`"0"` … `"3"`): a CF realization is a number, but a CMIP member
is a label such as `r1i1p1f1`. A member or lead missing from a run comes
back as NaN, so the cube keeps its shape.

## Ask questions with typed queries

The queries take `pd.Timestamp` and `pd.Timedelta`, never query strings,
and return a catalog on the same backend (DuckDB stays lazy):

```python
import pandas as pd

from geocatalog import open_catalog
from geocatalog.backends import CatalogRow
from geocatalog.states import (
    at_time, ensemble_at, forecast_chain, latest_analysis, lead_window,
)

catalog = open_catalog("states.parquet")
run = ensemble_at(catalog, run_time=pd.Timestamp("2026-01-01"))            # every member of one run
chain = forecast_chain(catalog, run_time=pd.Timestamp("2026-01-02"), member="0")  # one trajectory
day_two = lead_window(catalog, pd.Timedelta("2D"), pd.Timedelta("2D"))    # rows that reach lead 2 days
jan3 = at_time(catalog, pd.Timestamp("2026-01-03"))                       # every state valid then
latest: CatalogRow | None = latest_analysis(catalog, before=pd.Timestamp("2026-01-03 06:00"))
```

Strings are refused on purpose. On pandas 3,
`catalog.where("run_time == '2026-01-02'")` on a datetime column matches
nothing, silently; `member=0` raises `TypeError` for the same reason, since
the member column is text. `latest_analysis` filters on the `kind` column,
not on a zero lead: a lead-0 forecast is valid at its run time too.

## Other archives: cfgrib and file names

`names` says where each field is read: a coordinate or variable, then a
global attribute, or a callable on the file path.

- `CF_NAMES` (default): `forecast_reference_time`, `realization`, the
  `time` axis.
- `CFGRIB_NAMES`: cfgrib's `time` (the run), `number` (the member) and
  `valid_time` along `step`.
- Your own map, for archives that encode the run in the file name:

<!-- docs-check: skip -->
```python
names = {
    **CF_NAMES,
    "run_time": lambda p: pd.Timestamp(Path(p).name[3:11]),   # fc_20260102_m0.nc
    "member": lambda p: Path(p).stem.split("_m")[1],
}
catalog = build_state_catalog(paths, crs="EPSG:4326", names=names)
```

`kind`, `producer` and `parents` have no CF name. Write them as global
attributes when you save the states (as above), or map them to callables.

## Producers on different grids

`load_states` refuses rows on different grids rather than interpolating
them. Pass `group_by="producer"` to get an `xarray.DataTree` with one
leaf per producer, each a cube on its own grid.

## Pitfalls

- **One run and member per file.** A file whose run-time or member source
  holds several values is rejected; so is a (run, member) pair indexed
  twice.
- **Rows need a run time.** Analyses written without
  `forecast_reference_time` have no run; load them with
  `geocatalog.load.load_xarray`.
- **Lead time is derived, not stored.** A row's leads are its valid-time
  interval minus `run_time`.
- **Older catalogs.** The state columns are reserved in schema v1. A v0
  catalog loads in memory and is migrated on read; for DuckDB, run
  `geocatalog migrate <path>` first ([schema versions](../schema-versions.md)).

The reserved columns and their types are on the
[States API](../api/states.md) page.
