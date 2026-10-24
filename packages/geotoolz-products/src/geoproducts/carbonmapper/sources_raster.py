"""Rasterize Carbon Mapper *sources* (point clusters) onto a target grid.

Carbon Mapper sources are point geometries (DBSCAN-clustered plume
locations). For training labels, QA overlays, and source-prior features
it is useful to project them onto the same grid as an L2B scene as a
binary mask. This module provides:

- :func:`rasterize_sources` — one-shot function: list of points →
  `geoarray.GeoArray` mask.
- :class:`CMSourceRaster` — lazy wrapper that mirrors
  :class:`~geoproducts.carbonmapper.rasters.CMImageRaster` shape
  (``read_polygon`` / ``read_window`` / ``read_window_to_crs``) so
  callers can compose the source mask with the L2B rasters.

Both delegate the actual burn-in to georeader (the internal engine:
:func:`georeader.rasterize.rasterize_geopandas_like` /
:func:`~georeader.rasterize.rasterize_from_geopandas`) and return a
`geoarray.GeoArray` — no custom rasterio.features call lives in this
module.

The Carbon Mapper API does not publish a sources raster — these helpers
build it client-side from :func:`list_sources` (or any iterable of
:class:`~geoproducts.carbonmapper.source.CMSource`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from geoarray import DIMS, GeoArray, Grid, RasterSource
from geoarray.interop.georeader import from_georeader, source_to_georeader
from georeader import read
from georeader.rasterize import rasterize_from_geopandas, rasterize_geopandas_like
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

from geoproducts._src.carrier import require_source
from geoproducts._src.query import validate_lonlat_bbox
from geoproducts.carbonmapper.api_queries import CMTileItem
from geoproducts.carbonmapper.products import rio_env_options_for
from geoproducts.carbonmapper.source import CMSource


BBox = tuple[float, float, float, float]  # (W, S, E, N) in WGS-84

#: Anything we know how to extract a (lon, lat) pair from.
SourceLike = CMSource | Point | tuple[float, float]


def _to_point(src: SourceLike) -> Point:
    if isinstance(src, CMSource):
        return src.point
    if isinstance(src, Point):
        return src
    lon, lat = src
    return Point(float(lon), float(lat))


def _sources_gdf(sources: Iterable[SourceLike]) -> gpd.GeoDataFrame:
    """Build an EPSG:4326 GeoDataFrame with a ``value=1`` column."""
    pts = [_to_point(s) for s in sources]
    return gpd.GeoDataFrame(
        {"value": np.ones(len(pts), dtype=np.uint8)},
        geometry=pts,
        crs="EPSG:4326",
    )


def _apply_buffer(
    gdf: gpd.GeoDataFrame, target_crs: rasterio.crs.CRS, buffer_m: float
) -> gpd.GeoDataFrame:
    """Reproject to ``target_crs`` and buffer each point by ``buffer_m`` metres.

    Requires ``target_crs`` to be projected; ``buffer_m`` is converted to
    its linear unit (e.g. US survey feet for State Plane grids).
    """
    if not target_crs.is_projected:
        raise ValueError(
            "buffer_m > 0 requires a projected CRS (metres). Got geographic "
            f"CRS {target_crs}. Reproject your grid first or use buffer_m=0."
        )
    _, metres_per_unit = target_crs.linear_units_factor
    gdf = gdf.to_crs(target_crs)
    gdf = gdf.assign(geometry=gdf.geometry.buffer(buffer_m / metres_per_unit))  # ty: ignore[invalid-assignment]
    return gdf


def rasterize_sources(
    sources: Iterable[SourceLike],
    *,
    transform: rasterio.Affine,
    shape: tuple[int, int],
    crs: str | rasterio.crs.CRS,
    buffer_m: float = 0.0,
) -> GeoArray:
    """Rasterize source points onto a target grid as a binary mask.

    Each source contributes a value of ``1`` at its pixel; if
    ``buffer_m > 0`` a disk of that radius (in metres) is stamped
    instead. Sources falling outside the grid are silently dropped.

    Delegates to
    :func:`georeader.rasterize.rasterize_from_geopandas`.

    Parameters
    ----------
    sources:
        Iterable of :class:`CMSource`, Shapely :class:`Point`, or
        ``(lon, lat)`` tuples — all interpreted as WGS-84 lon/lat.
    transform:
        Affine transform of the target grid.
    shape:
        ``(height, width)`` of the target grid.
    crs:
        CRS of the target grid. Must be projected when
        ``buffer_m > 0``.
    buffer_m:
        Buffer radius in metres applied around each source point.
        ``0`` (default) → ``all_touched`` single-pixel stamp per source.

    Returns
    -------
    GeoArray
        2D ``uint8`` mask of ``shape`` with values in ``{0, 1}`` (fill ``0``).

    Raises
    ------
    ValueError
        If ``buffer_m > 0`` and ``crs`` is geographic, or if ``shape``
        is not 2D.
    """
    if len(shape) != 2:
        raise ValueError(f"Expected (H, W) shape, got {shape}")
    crs_obj = rasterio.crs.CRS.from_user_input(crs)

    gdf = _sources_gdf(sources)
    if len(gdf) == 0:
        return GeoArray(
            np.zeros(shape, dtype=np.uint8),
            transform,
            crs_obj,
            fill_value=0,
            dims=DIMS[-len(shape) :],
        )

    if buffer_m > 0:
        gdf = _apply_buffer(gdf, crs_obj, buffer_m)
        all_touched = False
    else:
        gdf = gdf.to_crs(crs_obj)
        all_touched = True  # stamp the pixel containing each point

    height, width = shape
    window_out = rasterio.windows.Window(
        col_off=0, row_off=0, width=width, height=height
    )
    return from_georeader(
        rasterize_from_geopandas(
            gdf,
            column="value",
            transform=transform,
            window_out=window_out,
            crs_out=crs_obj,
            fill=0,
            all_touched=all_touched,
        )
    )


def rasterize_sources_like(
    sources: Iterable[SourceLike],
    data_like: RasterSource,
    *,
    buffer_m: float = 0.0,
) -> GeoArray:
    """Rasterize sources onto the grid of an existing raster.

    Thin wrapper around
    :func:`georeader.rasterize.rasterize_geopandas_like`.

    Args:
        sources: Source points (see :func:`rasterize_sources`).
        data_like: The raster whose grid the mask takes: any
            `geoarray.RasterSource` (a reader, a lazy CM band, a GeoArray).
        buffer_m: Disk radius in metres around each point (``0``: one pixel).

    Returns:
        ``(H, W)`` ``uint8`` mask on ``data_like.grid``, values in ``{0, 1}``.

    Raises:
        TypeError: ``data_like`` is a georeader ``GeoTensor`` / ``GeoData``
            (convert it with ``from_georeader`` / ``source_from_georeader``).
    """
    require_source(data_like, "rasterize_sources_like")
    grid = data_like.grid
    crs_obj = rasterio.crs.CRS.from_user_input(grid.crs)
    gdf = _sources_gdf(sources)
    if len(gdf) == 0:
        return GeoArray(
            np.zeros(grid.shape, dtype=np.uint8),
            grid.transform,
            grid.crs,
            fill_value=0,
            dims=DIMS[-2:],
        )

    if buffer_m > 0:
        gdf = _apply_buffer(gdf, crs_obj, buffer_m)
        all_touched = False
    else:
        gdf = gdf.to_crs(crs_obj)
        all_touched = True

    return from_georeader(
        rasterize_geopandas_like(
            gdf,
            data_like=source_to_georeader(data_like),
            column="value",
            fill=0,
            all_touched=all_touched,
        )
    )


# ─────────────────────────────────────────────────────────────────────
#  Lazy class wrapper
# ─────────────────────────────────────────────────────────────────────


@dataclass(repr=False)
class CMSourceRaster:
    """Lazy binary-mask raster of Carbon Mapper sources on a target grid.

    Mirrors the read-helper surface of
    :class:`~geoproducts.carbonmapper.rasters.CMImageRaster` so
    callers can compose source masks with L2B reads.

    Attributes
    ----------
    sources:
        Source points to rasterize.
    transform, shape, crs:
        Target grid spec. Use :meth:`from_cmtileitem` or
        :meth:`from_source` to inherit the spec from an existing
        raster.
    buffer_m:
        Per-point disk radius in metres. ``0`` → single pixel.
    """

    sources: Sequence[SourceLike]
    transform: rasterio.Affine
    shape: tuple[int, int]
    crs: rasterio.crs.CRS
    buffer_m: float = 0.0

    # ---- Constructors ----

    @classmethod
    def from_source(
        cls,
        sources: Sequence[SourceLike],
        template: RasterSource,
        *,
        buffer_m: float = 0.0,
    ) -> CMSourceRaster:
        """Build a source raster on the grid of an existing raster.

        Args:
            sources: Source points to rasterize.
            template: Any `geoarray.RasterSource` (a reader, a lazy CM
                band such as ``CMImageRaster.cmf``, a GeoArray).
            buffer_m: Per-point disk radius in metres.

        Raises:
            TypeError: ``template`` is a georeader ``GeoTensor`` /
                ``GeoData`` (convert it with ``from_georeader`` /
                ``source_from_georeader``).
        """
        grid = require_source(template, "CMSourceRaster.from_source").grid
        return cls(
            sources=sources,
            transform=grid.affine,
            shape=grid.shape,
            crs=rasterio.crs.CRS.from_user_input(grid.crs),
            buffer_m=buffer_m,
        )

    @classmethod
    def from_cmtileitem(
        cls,
        sources: Sequence[SourceLike],
        tile: CMTileItem,
        *,
        buffer_m: float = 0.0,
        token: str | None = None,
    ) -> CMSourceRaster:
        """Build a source raster aligned to an L2B :class:`CMTileItem`.

        Resolves the tile's ``cmf`` GeoTIFF header to inherit
        ``(transform, shape, crs)``. Issues one HEAD/GET-range read.
        STAC items key the asset with its extension (``cmf.tif``); the
        extension-less form is accepted too. Pass ``token`` for an href
        on Carbon Mapper's Bearer-gated asset proxy; it is sent only
        inside this read's ``rasterio.Env``.
        """
        assets = tile.asset_urls
        cmf_url = assets.get("cmf.tif") or assets.get("cmf") or assets.get("ch4-mfa")
        if not cmf_url:
            raise ValueError(
                f"CMTileItem {tile.scene_id!r} has no 'cmf' asset to align to."
            )
        env_options = rio_env_options_for(str(cmf_url), token) or {}
        with rasterio.Env(**env_options), rasterio.open(cmf_url) as ds:
            return cls(
                sources=sources,
                transform=ds.transform,
                shape=(ds.height, ds.width),
                crs=ds.crs,
                buffer_m=buffer_m,
            )

    # ---- Eager render ----

    @property
    def grid(self) -> Grid:
        """The target pixel grid."""
        return Grid.from_affine(self.transform, self.crs, self.shape)

    def load(self) -> GeoArray:
        """Rasterize all sources onto the full grid: ``(H, W)`` ``uint8``."""
        return rasterize_sources(
            self.sources,
            transform=self.transform,
            shape=self.shape,
            crs=self.crs,
            buffer_m=self.buffer_m,
        )

    # ---- Read helpers (mirror CMImageRaster) ----

    def read_polygon(
        self,
        polygon: BaseGeometry,
        *,
        crs_polygon: str = "EPSG:4326",
    ) -> GeoArray:
        """Read a polygon clip of the source mask."""
        full = self.load()
        return from_georeader(
            read.read_from_polygon(
                source_to_georeader(full),
                polygon=polygon,
                crs_polygon=crs_polygon,
            )
        )

    def read_window(self, bounds_4326: BBox) -> GeoArray:
        """Read a WGS-84 bbox window of the source mask.

        Raises:
            ValueError: If the bbox is malformed or crosses the
                antimeridian (``W > E``) — split such windows in two.
        """
        validate_lonlat_bbox(bounds_4326)
        return self.read_polygon(box(*bounds_4326))

    def read_window_to_crs(
        self,
        bounds_4326: BBox,
        crs_dst: str,
    ) -> GeoArray:
        """Read a window then reproject the mask to ``crs_dst``."""
        crop = self.read_window(bounds_4326)
        return from_georeader(read.read_to_crs(source_to_georeader(crop), crs_dst))

    # ---- Repr ----

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_sources={len(self.sources)}, "
            f"shape={self.shape}, buffer_m={self.buffer_m}, crs={self.crs})"
        )
