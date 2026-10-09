"""Tier-A primitives for :mod:`geotoolz.io` — none of its own.

The io family wraps readers and writers; the array helpers its HDF /
NetCDF readers need (1-based band selection, CF fill lookup, GDAL
``GeoTransform`` parsing) are geotoolz-cloud's, in `geocloud.hdf`, next
to the readers themselves.
"""

from __future__ import annotations


__all__: list[str] = []
