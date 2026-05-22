# xrtoolz

> Composable operator library for geoprocessing Earth System Data Cubes.

`xrtoolz` provides a uniform `Operator` abstraction for preprocessing, inference, and evaluation of xarray datasets, organised around Earth-science domains.

## Layout

```
xrtoolz/
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
pip install xrtoolz
```

Or with `uv`:

```bash
uv add xrtoolz
```

## Quickstart

```python
import xrtoolz
```

## Links

- [API Reference](api/reference.md)
- [Changelog](CHANGELOG.md)
- [GitHub](https://github.com/jejjohnson/xrtoolz)
