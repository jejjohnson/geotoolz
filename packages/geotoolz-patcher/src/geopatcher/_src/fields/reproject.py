"""`ReprojectingRasterField` — on-the-fly reprojection as a `Field` adapter.

Level 2 of CRS-aware patching (issue jejjohnson/geopatcher#20): rather than teaching the
patcher core about CRSs, present the *destination* grid as the field's
domain. Every existing sampler / geometry / aggregation then works on the
target grid unchanged, and each `select` warps the source into the chip.

The heavy lifting is georeader's ``read_reproject`` / the
``calculate_transform_window`` grid computation, used as an internal engine
behind `geoarray.interop.georeader` (``source_to_georeader`` in,
``from_georeader`` out) — geopatcher only wires the destination window back
to source pixels per chip. Chips and merged outputs are `geoarray.GeoArray`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from geoarray import GeoArray, Grid, RasterSource
from geoarray.interop.georeader import from_georeader, source_to_georeader

from geopatcher._src.fields.raster import (
    _pack_reader_state,
    _require_source,
    _rewrap,
    _unpack_reader_state,
)


@dataclass(frozen=True)
class _ReprojectedDomain:
    """The destination grid as a `RasterDomain`: ``grid`` plus full ``shape``."""

    grid: Grid
    shape: tuple[int, ...]

    @property
    def crs(self) -> Any:
        return self.grid.crs

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.grid.bounds


@dataclass(eq=False)
class ReprojectingRasterField:
    """Wrap a `geoarray.RasterSource`, presenting a reprojected destination grid.

    The domain reports the *destination* CRS / transform / shape, so
    samplers place anchors on the reprojected grid; each `select` warps
    the corresponding source pixels into that chip.

    Args:
        reader: Any `geoarray.RasterSource` (a `GeoArray`, a lazy reader,
            …) in the source CRS. A georeader reader crosses with
            ``geoarray.interop.georeader.source_from_georeader``.
        dst_crs: Destination CRS the domain (and every chip) is expressed
            in — ``"EPSG:3857"``, a CRS object, or WKT.
        resolution: Destination pixel size in ``dst_crs`` units. ``None``
            (default) lets georeader pick a resolution that matches the
            source.
        resampling: Resampling method — ``"nearest"``, ``"bilinear"``,
            ``"cubic"``, ``"cubic_spline"``, or ``"lanczos"``.

    The domain shape keeps the source's leading (band / time) dims —
    ``(*reader.shape[:-2], height, width)`` — so chips and the merged
    output agree on rank.

    Each `select` first crops the source to the chip's footprint (plus a
    small pixel margin for the resampling kernel, scaled up when the
    destination is coarser than the source) and warps only that crop, so
    per-chip cost scales with the chip, not the whole source.

    Note:
        Chips are warped independently, so a stitched result can differ
        slightly from one full-image warp of the same source — GDAL's
        approximate transformer linearises per warp call, which shifts
        sample positions by a small fraction of a pixel (with bilinear
        resampling, differences stay below one ramp step on a
        unit-increment ramp). The source crop itself does not change the
        result: a chip is identical to warping the uncropped source onto
        the same chip grid. Warp the whole source once (``geoops.geom``'s
        reprojection) when bit-exactness against a full warp matters.

    Raises:
        TypeError: ``reader`` is a georeader object or not a `RasterSource`.
    """

    reader: RasterSource
    dst_crs: Any
    resolution: float | tuple[float, float] | None = None
    resampling: str = "bilinear"

    def __post_init__(self) -> None:
        from georeader.read import calculate_transform_window

        _require_source(self.reader, "ReprojectingRasterField", RasterSource)
        window_data, dst_transform = calculate_transform_window(
            self._engine, self.dst_crs, self.resolution
        )
        # `frozen`-friendly private state on an `eq=False` dataclass.
        object.__setattr__(self, "_transform", dst_transform)
        object.__setattr__(
            self,
            "_shape",
            (
                *(int(d) for d in self.reader.shape[:-2]),
                int(window_data.height),
                int(window_data.width),
            ),
        )
        object.__setattr__(self, "_resampling", _resampling_enum(self.resampling))
        object.__setattr__(
            self,
            "_grid",
            Grid.from_affine(dst_transform, self.dst_crs, self._shape[-2:]),
        )

    @property
    def _engine(self) -> Any:
        """The source as a georeader ``GeoData``, for georeader's ``read``."""
        return source_to_georeader(self.reader)

    def __getstate__(self) -> dict[str, Any]:
        return _pack_reader_state(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(_unpack_reader_state(state))

    @property
    def domain(self) -> _ReprojectedDomain:
        return _ReprojectedDomain(grid=self._grid, shape=self._shape)

    def select(self, window: Any) -> GeoArray:
        """Warp the source into ``window`` of the destination grid.

        Returns:
            ``(..., window.height, window.width)`` on the window's grid in
            ``dst_crs``.
        """
        from georeader.read import read_reproject
        from rasterio.windows import Window, transform as window_transform

        chip_transform = window_transform(window, self._transform)
        out = Window(
            col_off=0, row_off=0, width=int(window.width), height=int(window.height)
        )
        warped = read_reproject(
            self._crop_source(out, chip_transform),
            dst_crs=self.dst_crs,
            dst_transform=chip_transform,
            window_out=out,
            resampling=self._resampling,
        )
        return from_georeader(warped)

    def _crop_source(self, out: Any, chip_transform: Any) -> Any:
        """Source pixels covering the chip footprint, plus a kernel margin.

        ``read_reproject`` only crops lazy readers (and never an in-memory
        carrier), so without this each chip would warp the whole source.
        The chip polygon is densified before being mapped into the source
        CRS so curved edges of the reprojected footprint are
        covered; the margin is `_CROP_MARGIN` source pixels, scaled by
        the downsampling factor (GDAL widens the kernel when the
        destination is coarser). Returns the reader unchanged when the
        chip does not overlap the source — ``read_reproject`` then
        short-circuits to a nodata chip without reading anything — or
        when the footprint cannot be mapped into the source CRS.
        """
        from georeader import window_utils
        from georeader.read import read_from_window, window_from_polygon

        footprint = window_utils.window_polygon(out, chip_transform)
        step = max(footprint.length / 64.0, 1e-12)
        footprint = footprint.segmentize(step)
        engine = self._engine
        src_window = window_from_polygon(engine, footprint, crs_polygon=self.dst_crs)
        extent = (src_window.col_off, src_window.row_off)
        extent += (src_window.width, src_window.height)
        if not all(math.isfinite(float(v)) for v in extent):
            # Footprint not representable in the source CRS — let
            # `read_reproject` handle the full source.
            return engine
        scale = max(
            float(src_window.width) / max(float(out.width), 1.0),
            float(src_window.height) / max(float(out.height), 1.0),
            1.0,
        )
        pad = math.ceil(_CROP_MARGIN * scale)
        src_window = window_utils.round_outer_window(
            window_utils.pad_window(src_window, (pad, pad))
        )
        cropped = read_from_window(
            engine, src_window, trigger_load=True, boundless=False
        )
        return engine if cropped is None else cropped

    def cache_id(self) -> str:
        """`PatchCache` identity: the source plus the warp parameters.

        Folds in the source reader's identity (``None`` for an in-memory
        source — `PatchCache` then requires an explicit ``field_id``),
        ``dst_crs``, ``resolution`` and ``resampling``, so two
        reprojections of one source never share cache entries.

        Examples:
            >>> reader = source_from_georeader(RasterioReader("a.tif"))
            >>> ReprojectingRasterField(reader, "EPSG:3857").cache_id()
            '{"dst_crs": "EPSG:3857", "resampling": "bilinear", ...}'
            >>> ReprojectingRasterField(scene, "EPSG:3857").cache_id()
            '{"dst_crs": "EPSG:3857", ..., "source": null}'
        """
        from geopatcher._src.cache import _crs_text, _source_id

        resolution = self.resolution
        if isinstance(resolution, (tuple, list)):
            resolution = [float(r) for r in resolution]
        elif resolution is not None:
            resolution = float(resolution)
        return json.dumps(
            {
                "source": _source_id(self.reader),
                "dst_crs": _crs_text(self.dst_crs),
                "resolution": resolution,
                "resampling": self.resampling,
            },
            sort_keys=True,
        )

    def with_data(self, array: Any) -> GeoArray:
        """``array`` (``(..., H, W)`` on the destination grid) as a `GeoArray`."""
        return _rewrap(self.reader, array, self._grid)


# Source pixels kept around each chip footprint before warping — enough
# for the widest supported kernel (lanczos: 3 px radius) at 1:1 scale.
_CROP_MARGIN = 3


def _resampling_enum(name: str) -> Any:
    import rasterio.warp

    try:
        return getattr(rasterio.warp.Resampling, name)
    except AttributeError as exc:
        raise ValueError(
            f"unknown resampling {name!r}; expected one of 'nearest', "
            f"'bilinear', 'cubic', 'cubic_spline', 'lanczos'."
        ) from exc


__all__ = ["ReprojectingRasterField"]
