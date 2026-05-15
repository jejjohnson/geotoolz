"""`GeoPandasField` — adapter for vector geometries (polygons, lines, points).

Wraps a `geopandas.GeoDataFrame`. The domain reports either a
`VectorDomain` (general polygons) or a `PointDomain` (when every
geometry is a `shapely.Point` — useful for KNN/RadiusGraph patching of
station data).

Optional extra: ``pip install 'geotoolz[vector]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geotoolz.patch._src.domains import PointDomain, VectorDomain
from geotoolz.patch._src.fields._extras import _missing_extra


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
    """

    gdf: Any
    as_points: bool = False

    def __post_init__(self) -> None:
        if gpd is None:
            raise _missing_extra(
                "GeoPandasField", "vector", "geopandas>=0.14 shapely>=2"
            )

    @property
    def domain(self) -> VectorDomain | PointDomain:
        if self.as_points:
            coords = np.c_[self.gdf.geometry.x, self.gdf.geometry.y]
            return PointDomain(coords=coords, kdtree=cKDTree(coords), crs=self.gdf.crs)
        return VectorDomain(
            geometry=self.gdf.geometry, sindex=self.gdf.sindex, crs=self.gdf.crs
        )

    def select(self, indexer: Any) -> GeoPandasField:
        # `indexer` may be a boolean mask, an iloc-style index array, or a
        # list of row labels. `iloc` covers the first two; `loc` is the
        # fallback for labels.
        try:
            sub = self.gdf.iloc[indexer]
        except (TypeError, IndexError, KeyError):
            sub = self.gdf.loc[indexer]
        return GeoPandasField(sub.copy(), as_points=self.as_points)

    def with_data(self, array: Any) -> GeoPandasField:
        new = self.gdf.copy()
        new["_value"] = np.asarray(array)
        return GeoPandasField(new, as_points=self.as_points)
