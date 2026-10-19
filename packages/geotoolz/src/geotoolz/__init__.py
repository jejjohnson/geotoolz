"""geotoolz — the geostack in one install.

This distribution holds no code. It installs the five packages of the
stack, each imported under its own name:

| Distribution | Import | Use it for |
|---|---|---|
| ``geotoolz-ops`` | ``geoops`` | Operator families on ``GeoTensor`` |
| ``geotoolz-patcher`` | ``geopatcher`` | Split a field into patches, process, stitch |
| ``geotoolz-catalog`` | ``geocatalog`` | Find, index, join and load files |
| ``geotoolz-cloud`` | ``geocloud`` | Object storage and COG reads |
| ``geotoolz-products`` | ``geoproducts`` | Readers for Earth-observation products |

    import geoops as gz
    import geopatcher as gp
    import geocatalog as gc
"""

from __future__ import annotations


__version__ = "0.12.0"

__all__ = ["__version__"]
