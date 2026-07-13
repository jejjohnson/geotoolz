"""`CatalogDomain` — adapter that lets a `GeoCatalog` act as a Patcher domain.

A downstream `SpatialPatcher` (e.g. `geotoolz.patch.SpatialPatcher`)
operates on a `Domain` — an object with ``bounds``, an iterable tiling,
and (downstream) a `Field` that knows how to read bytes for a given
sub-region. The single-file case is `RasterDomain` + `RasterField`. The
*multi-file* case is this shim: wrap a catalog, get one sub-domain per
row, and the same operator pipeline that handled a single raster now
handles an archive.

`CatalogDomain` duck-types the `Domain` protocol; the package has no
runtime dependency on any patcher. The integration stays decoupled as
long as the `GeoSlice` shape stays put.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from geocatalog._src.geoslice import GeoSlice
    from geocatalog._src.memory import InMemoryGeoCatalog


@dataclass(frozen=True)
class CatalogDomain:
    """A `geotoolz.patch.Domain` view of a `GeoCatalog`.

    Use this when you want to run a `SpatialPatcher` (or any consumer
    that takes a `Domain`) across a multi-file archive instead of a
    single in-RAM raster. Each catalog row becomes one sub-domain; the
    patcher iterates them, the loader opens each in turn, the operator
    runs per file, and stitching reassembles the result.

    The dataclass is ``frozen=True`` so it can be cached, hashed, and
    safely shipped across function boundaries — useful when feeding the
    same domain into multiple parallel pipelines.

    Args:
        catalog: The catalog being adapted. Any backend tag (``"raster"``,
            ``"xarray"``, ``"vector"``) is accepted — the loader you pair
            this with downstream is what makes it concrete. Must be
            non-empty for `slices` to return anything useful.
        resolution: Target ``(x_res, y_res)`` in CRS units, baked into
            every `GeoSlice` produced by `slices`. Drives the pixel
            shape a downstream loader will materialise; pick whatever
            your operator expects.

    Example:
        Tiled per-row inference across a multi-file S2 archive::

            catalog = gz.build_raster_catalog(paths, ...)
            domain  = gz.CatalogDomain(catalog, resolution=(10.0, 10.0))
            for slice_ in domain.slices():
                chip = gz.load_raster(catalog, slice_)
                yield model(chip.values)
    """

    catalog: InMemoryGeoCatalog
    resolution: tuple[float, float]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Union bbox over the catalog — the outer envelope of every row.

        Returns:
            ``(xmin, ymin, xmax, ymax)`` in catalog-CRS units. Four NaNs
            if the catalog is empty.
        """
        return self.catalog.total_bounds

    @property
    def crs(self) -> Any:
        """The catalog's CRS — proxied from ``catalog.gdf.crs``."""
        return self.catalog.gdf.crs

    def __len__(self) -> int:
        """Number of sub-domains (== number of catalog rows)."""
        return len(self.catalog)

    def slices(self) -> Sequence[GeoSlice]:
        """Materialise the catalog's per-row `GeoSlice` list.

        Eager (returns a list) rather than lazy because most callers
        want ``len(slices())`` for progress bars or to shard work
        upfront. Iterate the catalog's `iter_slices` directly if you
        want streaming.

        Returns:
            A list of `GeoSlice` instances, one per catalog row, each
            carrying this domain's ``resolution`` and the row's
            footprint + interval.
        """
        return list(self.catalog.iter_slices(resolution=self.resolution))

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary — for logging and audits.

        Returns:
            ``{"catalog": <catalog.get_config()>, "resolution": (x, y)}``.
            Round-trippable through ``json.dumps`` provided the catalog's
            own ``get_config`` is.
        """
        return {
            "catalog": self.catalog.get_config(),
            "resolution": self.resolution,
        }
