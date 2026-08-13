# xrtoolz-reader

[![Tests](https://github.com/jejjohnson/xrtoolz/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/xrtoolz/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/xrtoolz/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/xrtoolz/actions/workflows/typecheck.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Authenticated archive readers for xarray geodata (import name: `xrreader`).

Author: J. Emmanuel Johnson · Repo: <https://github.com/jejjohnson/xrtoolz>

**Authenticated archive readers for xarray geodata.** `xrreader` turns a typed
*request* — variables, a bounding box, a time window — into the exact payload a
remote data service expects, and returns a ready-to-use `xarray.Dataset`.

It is the data-acquisition layer of a small, single-responsibility ecosystem:

```text
 geocatalog ──▶ xrreader ──▶ xrtoolz ──▶ geopatcher
  index files    fetch the     operate     split → op
  / STAC         arrays        on data     → stitch
```

`xrreader` complements [`geocatalog`](https://github.com/jejjohnson/geocatalog):
geocatalog *indexes where data is* (files / STAC); xrreader *fetches the data
itself* from API services (CMEMS, CDS, AEMET) that have their own authentication
and server-side subsetting.

## Install

The core stays light (`numpy` / `pandas` / `xarray`); each service client is an
optional extra and is imported lazily:

```bash
pip install 'xrtoolz-reader[cmems]'        # Copernicus Marine (copernicusmarine)
pip install 'xrtoolz-reader[cds]'          # Climate Data Store / ERA5 (cdsapi)
pip install 'xrtoolz-reader[cds-insitu]'   # CDS in-situ + GeoParquet archive
pip install 'xrtoolz-reader[aemet]'        # AEMET OpenData + GeoParquet archive
pip install 'xrtoolz-reader[grib]'         # GRIB reader backend (cfgrib/ecCodes)
```

While the ecosystem is pre-PyPI, install from source with `uv`.

## Usage

```python
import xrreader

src = xrreader.CMEMSSource()                  # creds from env / ~/.cmems

# Short names live in xrreader.CATALOG; resolve to the service dataset id.
entry = xrreader.CATALOG["glorys12.daily"]
sel = dict(
    variables=["thetao", "so"],
    bbox=xrreader.BBox(-80, -50, 30, 45),     # Gulf Stream
    time=xrreader.TimeRange.parse("2020-01-01", "2020-01-31"),
)

src.list_datasets()                                   # discovery (no download)
src.describe(entry.dataset_id)                        # full metadata for the id
ds   = src.open(entry.dataset_id, **sel)              # lazy xr.Dataset
path = src.download(entry.dataset_id, "glorys.nc", **sel)   # NetCDF on disk
```

Request fields (`variables`, `bbox`, `time`, `depth`, `levels`) are keyword-only
after `dataset_id`. Each typed primitive (`BBox`, `TimeRange`, `DepthRange`,
`PressureLevels`) knows how to serialize itself into the chosen service's
dialect, so the same selection works across every adapter. `"glorys12.daily"`
is a short name from the bundled `xrreader.CATALOG`; a fully-qualified
`dataset_id` works too.

## Adapters

| Source | Class | Extra | Notes |
|---|---|---|---|
| Copernicus Marine | `CMEMSSource` | `cmems` | ocean physics / BGC / SSH / SST / ocean colour |
| Climate Data Store | `CDSSource` | `cds`, `cds-insitu` | ERA5 reanalysis + in-situ; `CDSInsituArchive` |
| AEMET OpenData | `AemetSource` | `aemet` | Spanish station observations; `AemetArchive` |

## Documentation

The full design — architecture, protocols, and the
`xrtoolz.data` → `xrreader` migration — lives in
[`docs/design/architecture.md`](docs/design/architecture.md).

## Development

```bash
make install      # uv sync --all-groups + pre-commit hooks
make test         # uv run pytest -v
make lint         # ruff check .
make typecheck    # ty check src/xrreader
```

## License

MIT — see [LICENSE](LICENSE).
