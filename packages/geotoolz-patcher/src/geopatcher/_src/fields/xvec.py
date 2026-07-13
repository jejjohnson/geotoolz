"""`XvecField` — adapter for vector data cubes (xarray + shapely points).

The modern answer for stations / floats / swath samples with multiple
variables and times. Built on `xvec`, which exposes a `geometry`
coordinate on an `xarray.Dataset`.

Optional extra: ``pip install 'geopatcher[point]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geopatcher._src.domains import PointDomain
from geopatcher._src.fields._extras import _missing_extra


try:
    import xvec  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    xvec = None  # type: ignore[assignment]

# scipy is a base dependency — see `geopandas.py` for the same import shape.
from scipy.spatial import cKDTree  # type: ignore[import-untyped]


@dataclass(eq=False)
class XvecField:
    """Wrap a vector data cube (xarray + xvec) as a `Field[PointDomain]`.

    Args:
        ds: An `xarray.Dataset` with a ``geometry`` coordinate of
            `shapely.Point` instances (the xvec convention).
    """

    ds: Any
    _geometry_dim: str = "geometry"

    def __post_init__(self) -> None:
        if xvec is None:
            raise _missing_extra("XvecField", "point", "xvec>=0.4 scipy")

    @property
    def domain(self) -> PointDomain:
        geom_var = self.ds[self._geometry_dim]
        coords = np.c_[[g.x for g in geom_var.values], [g.y for g in geom_var.values]]
        # xvec stashes the CRS on the geometry coord's attrs (set_geom_indexes
        # writes `crs=...` there). Newer xvec versions may expose it through
        # the xvec accessor — try the attr path first, fall back to None.
        crs = geom_var.attrs.get("crs")
        return PointDomain(coords=coords, kdtree=cKDTree(coords), crs=crs)

    def select(self, indexer: Any) -> XvecField:
        return XvecField(self.ds.isel({self._geometry_dim: indexer}))

    def with_data(self, array: Any) -> XvecField:
        new = self.ds.copy()
        new["_value"] = ((self._geometry_dim,), np.asarray(array))
        return XvecField(new)
