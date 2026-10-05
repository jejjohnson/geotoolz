"""`GeoPandasField` — adapter for vector geometries (polygons, lines, points).

Wraps a `geopandas.GeoDataFrame`. The domain reports either a
`VectorDomain` (general polygons) or a `PointDomain` (when every
geometry is a `shapely.Point` — useful for KNN/RadiusGraph patching of
station data).

Optional extra: ``pip install 'geotoolz-patcher[vector]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from geopatcher._src._extras import missing_extra
from geopatcher._src.domains import PointDomain, VectorDomain


try:
    import geopandas as gpd  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    gpd = None  # type: ignore[assignment]

# scipy is a base dependency — cKDTree is always present, but we still
# type the import this way so static analysers don't trip when the
# bundled scipy stubs are stale.
from scipy.spatial import cKDTree  # type: ignore[import-untyped]


@dataclass(eq=False)
class GeoPandasField:
    """Wrap a `geopandas.GeoDataFrame` as a `Field[VectorDomain]`.

    If every geometry is a `shapely.Point`, callers can ask for a
    `PointDomain` view via the ``as_points=True`` flag — that branch
    builds a `cKDTree` on the point coordinates so `KNNGraph` /
    `RadiusGraph` queries are cheap.

    Args:
        gdf: The underlying `geopandas.GeoDataFrame`.
        as_points: If ``True``, expose a `PointDomain` instead of a
            `VectorDomain`. The GDF must hold Point geometries.

    ``domain`` (including its spatial index / ``cKDTree``) is built on
    first access and cached; treat ``gdf`` as immutable once wrapped.
    """

    gdf: Any
    as_points: bool = False

    def __post_init__(self) -> None:
        if gpd is None:
            raise missing_extra(
                "GeoPandasField", "vector", "geopandas>=0.14 shapely>=2"
            )

    @cached_property
    def domain(self) -> VectorDomain | PointDomain:
        if self.as_points:
            coords = np.c_[self.gdf.geometry.x, self.gdf.geometry.y]
            return PointDomain(coords=coords, kdtree=cKDTree(coords), crs=self.gdf.crs)
        return VectorDomain(
            geometry=self.gdf.geometry, sindex=self.gdf.sindex, crs=self.gdf.crs
        )

    def select(self, indexer: Any) -> GeoPandasField:
        """Return the sub-field at ``indexer`` — positional indexing only.

        ``indexer`` is a boolean mask, an integer position array, or a
        positional slice, all resolved via ``gdf.iloc``. Row *labels* are
        never consulted: samplers emit positions into the domain
        (``PointDomain.coords`` / ``VectorDomain.geometry`` order), so a
        label fallback would silently read the wrong rows whenever the
        index is not a ``RangeIndex``.
        """
        return GeoPandasField(self.gdf.iloc[indexer].copy(), as_points=self.as_points)

    def with_data(self, array: Any) -> GeoPandasField:
        """Attach one value per row as a ``_value`` column.

        Args:
            array: A 1-D array with exactly one entry per row of ``gdf``.
                Multi-dimensional payloads (e.g. per-row feature vectors)
                are not supported — a GeoDataFrame column holds scalars.

        Raises:
            ValueError: If ``array`` is not 1-D or its length differs
                from the number of rows.
        """
        values = np.asarray(array)
        if values.ndim != 1 or values.shape[0] != len(self.gdf):
            raise ValueError(
                "GeoPandasField.with_data expects a 1-D array with one value "
                f"per row ({len(self.gdf)}); got shape {values.shape}."
            )
        new = self.gdf.copy()
        new["_value"] = values
        return GeoPandasField(new, as_points=self.as_points)
