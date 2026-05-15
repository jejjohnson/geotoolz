"""`Field` / `Domain` Protocols — the substrate the Patcher splits over.

Two intentionally tiny Protocols. `Field` is "something the Patcher can
read from"; `Domain` is its I/O-free metadata twin — bounds, CRS, shape
— consulted by `Sampler.anchors` and `PatchGeometry.neighborhood`
without triggering any reads.

For rasters, the raster-side surface already exists in
`georeader.abstract_reader` as `GeoData` / `AsyncGeoData` / `GeoDataBase`.
A thin `RasterField` adapter in `_src/fields/raster.py` bridges the
naming difference (`select` vs `read_from_window`). The non-raster
fields (`XarrayField`, `GeoPandasField`, `XvecField`) are new substrate.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Domain(Protocol):
    """Metadata view of a field — bounds, CRS, no I/O.

    Concrete subclasses (`RasterDomain`, `GridDomain`, `VectorDomain`,
    `PointDomain`) carry the backend-specific shape that a
    `PatchGeometry.neighborhood(domain, anchor)` dispatch needs.
    """

    @property
    def crs(self) -> Any: ...

    @property
    def bounds(self) -> Any: ...


@runtime_checkable
class Field(Protocol):
    """A substrate the Patcher reads patches out of.

    Three operations:
    - ``domain`` exposes the I/O-free metadata view used by the
      `Sampler` and `PatchGeometry`.
    - ``select(indexer)`` reads a slice of the field. The shape of
      ``indexer`` is decided by ``domain`` (a ``rasterio.windows.Window``
      for `RasterDomain`, a ``dict[str, slice]`` for `GridDomain`, a list
      of row indices for `PointDomain` / `VectorDomain`).
    - ``with_data(array)`` reconstructs a field-shaped value from an
      operator output; used by `Aggregation.merge` to rebuild a global
      field.
    """

    @property
    def domain(self) -> Domain: ...

    def select(self, indexer: Any) -> Any: ...

    def with_data(self, array: Any) -> Any: ...


@runtime_checkable
class AsyncField(Protocol):
    """Async mirror of `Field` — concurrent `select` over remote tiles.

    `AsyncSpatialPatcher.split` consumes this Protocol; the sync surface
    is otherwise identical.
    """

    @property
    def domain(self) -> Domain: ...

    async def select(self, indexer: Any) -> Any: ...

    def with_data(self, array: Any) -> Any: ...
