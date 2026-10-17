"""`geocatalog.states` — query and load model states: analyses, forecasts, ensembles.

A model state has two times (the run's start, ``run_time``, and the time
it describes, the valid time), an ensemble member, a producer and parent
states. A state catalog is an ordinary xarray-kind catalog whose rows
carry five reserved columns (`STATE_COLUMNS`), filled by
`geocatalog.build.build_state_catalog` from each file's metadata.

- `load_states` — stack one file per (run, member) into an
  ``(init_time, member, lead_time, y, x)`` cube; overlapping runs are
  never dropped (``[xarray-raster]`` extra; loads lazily).
- Typed queries, on either backend: `forecasts`, `ensemble_at`,
  `forecast_chain`, `lead_window`, `at_time`, `latest_analysis`.
- Name maps for the builder: `CF_NAMES`, `CFGRIB_NAMES`.
- `StateFields` — a row's state fields, typed (`CatalogRow.state`).

```python
import pandas as pd

from geocatalog import GeoSlice, open_catalog
from geocatalog.states import ensemble_at, load_states

catalog = open_catalog("states.parquet")
run = ensemble_at(catalog, run_time=pd.Timestamp("2026-01-01"))  # one run
aoi = GeoSlice(
    bounds=(-104.0, 31.0, -101.0, 34.0),
    interval=pd.Interval(pd.Timestamp("2026-01-01"),
                         pd.Timestamp("2026-01-05"), closed="both"),
    resolution=(1.0, 1.0),
    crs="EPSG:4326",
)
cube = load_states(run, aoi)   # (init_time, member, lead_time, lat, lon)
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from geocatalog._src._lazy import lazy_getattr
from geocatalog._src._schema import STATE_COLUMNS
from geocatalog._src.base import StateFields
from geocatalog._src.states import (
    CF_NAMES,
    CFGRIB_NAMES,
    at_time,
    ensemble_at,
    forecast_chain,
    forecasts,
    latest_analysis,
    lead_window,
)


if TYPE_CHECKING:
    from geocatalog._src.formats.states import load_states


__all__ = [
    "CFGRIB_NAMES",
    "CF_NAMES",
    "STATE_COLUMNS",
    "StateFields",
    "at_time",
    "ensemble_at",
    "forecast_chain",
    "forecasts",
    "latest_analysis",
    "lead_window",
    "load_states",
]


# Extras-gated names resolve lazily (`geocatalog._src._lazy.LAZY`).
__getattr__ = lazy_getattr(
    globals(),
    [
        "load_states",
    ],
)
