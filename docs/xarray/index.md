# xr_toolz

> Composable operator library for geoprocessing Earth System Data Cubes.

`xr_toolz` provides a uniform `Operator` abstraction for preprocessing, inference, and evaluation of xarray datasets, organised around Earth-science domains.

## Layout

```
xr_toolz/
├── core   # Operator, Sequential, Input, Node, Graph
├── geo    # Generic xarray geoprocessing (validation, subset, regrid,
│          # detrend, masks, metrics, spectral, ...)
├── ocn    # Oceanography physics (streamfunction, geostrophic velocity, ...)
├── atm    # Atmospheric physics (potential temperature, wind, ...)
│   └── gas/ch4   # Trace-gas physics (column averaging kernel, ...)
├── rs     # Remote sensing (NDVI, radiance/reflectance, ...)
└── ice    # Cryosphere (reserved; no content yet)
```

See the [Design](design/README.md) section for the full architecture and
roadmap.

## Installation

```bash
pip install xr_toolz
```

Or with `uv`:

```bash
uv add xr_toolz
```

## Quickstart

```python
import xr_toolz
```

## Links

- [API Reference](api/reference.md)
- [Changelog](CHANGELOG.md)
- [GitHub](https://github.com/jejjohnson/xr_toolz)
