# xrtoolz-reader — authenticated archive readers

```bash
# Pre-PyPI, install from the workspace via git. The core is light, so
# this resolves on its own; add an extra for the service you need. Once
# published this becomes a plain `pip install 'xrtoolz-reader[cmems]'`.
uv pip install "xrtoolz-reader @ git+https://github.com/jejjohnson/xrtoolz@main#subdirectory=packages/xrtoolz-reader"
```

`xrreader` is the **data-acquisition** layer of the stack. It turns a typed
*request* — variables, a bounding box, a time window — into the exact payload a
remote archive expects, and hands back a ready-to-use `xarray.Dataset`. Where
the rest of the workspace operates *on* data, this package is what *fetches* it.

It imports as `xrreader` (the distribution is `xrtoolz-reader`, matching the
`xrtoolz-grad` → `xrgrad` convention), and was developed for a time in its own
repository before being folded back into this workspace.

## Where it sits

```text
 geocatalog ──▶ xrreader ──▶ xrtoolz ──▶ geopatcher
  index files    fetch the     operate     split → op
  / STAC         arrays        on data     → stitch
```

`xrreader` complements [`geocatalog`](https://github.com/jejjohnson/geocatalog):
geocatalog *indexes where data is* (files / STAC); `xrreader` *fetches the data
itself* from API services that have their own authentication and server-side
subsetting.

## The light core

The `[project] dependencies` are only `numpy` / `pandas` / `xarray`. Every
service client is an optional extra, imported lazily at first use:

| Extra | Pulls | Covers |
|---|---|---|
| `cmems` | `copernicusmarine` | Copernicus Marine — `CMEMSSource` |
| `cds` | `cdsapi`, `netcdf4` | Climate Data Store gridded reanalysis (ERA5) — `CDSSource` |
| `cds-insitu` | + `geopandas`, `pyarrow`, `loguru` | CDS in-situ surface-land / surface-marine + `CDSInsituArchive` |
| `aemet` | `httpx`, `geopandas`, `pyarrow`, `loguru`, `netcdf4` | AEMET OpenData — `AemetSource`, `AemetArchive` |
| `grib` | `cfgrib` | GRIB reader backend, for `CDSSource(format="grib")` |

That light core is load-bearing for the workspace: `xrtoolz` depends on
`xrtoolz-reader` for the shared typed request vocabulary and the CF `Variable`
registry (`xrtoolz.geo` and `xrtoolz.viz` import from `xrreader.types`), so the
archive clients must stay out of a plain `pip install xrtoolz`.

## The request vocabulary

The typed primitives know how to serialize *themselves* into each service's
payload dialect — the per-service knowledge lives on the type, not scattered
through the adapters:

```python
from xrreader import BBox, TimeRange

bbox = BBox(lon_min=-80, lon_max=-50, lat_min=30, lat_max=45)   # Gulf Stream
time = TimeRange.parse("2020-01-01", "2020-01-31")

bbox.as_cmems()     # {'minimum_longitude': -80, 'maximum_longitude': -50, ...}
bbox.as_cds_area()  # CDS wants [North, West, South, East] → [45, -80, 30, -50]
time.as_cds_form()  # CDS wants exploded lists → {'year': [...], 'month': [...]}
```

`BBox` understands the antimeridian, normalizes between `[-180, 180]` and
`[0, 360]`, and can emit an `xarray.sel()` selector. `TimeRange` carries an
optional sampling frequency. `DepthRange` / `PressureLevels` cover the vertical
axis, and the whole lot composes into a single `Request`.

## Credentials

Each `load_*` helper resolves in a fixed order — explicit arguments, then
environment variables, then a `.env` walked up from the working directory, then
the service's own dotfile (`~/.cmems`, `~/.cdsapirc`). Adapters constructed with
`credentials=None` run that resolution automatically, so a notebook started
under `docs/` still picks up the project-root `.env`.

## API reference

- [Readers](../api/reader.md)
- [Reader architecture](../design/reader/architecture.md)
