"""`ReprojectingRasterField` — on-the-fly reprojection as a `Field` adapter.

Level 2 of CRS-aware patching (issue #20): rather than teaching the
patcher core about CRSs, present the *destination* grid as the field's
domain. Every existing sampler / geometry / aggregation then works on the
target grid unchanged, and each `select` warps the source into the chip.

The heavy lifting is georeader's ``read_reproject`` / the
``calculate_transform_window`` grid computation — geopatcher only wires
the destination window back to source pixels per chip.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor

from geopatcher._src.fields.raster import _rewrap


@dataclass(frozen=True)
class _ReprojectedDomain:
    """`GeoDataBase`-shaped view of the destination grid."""

    crs: Any
    transform: Any
    shape: tuple[int, ...]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        from rasterio.windows import Window, bounds

        return bounds(
            Window(
                col_off=0,
                row_off=0,
                width=int(self.shape[-1]),
                height=int(self.shape[-2]),
            ),
            self.transform,
        )


@dataclass(eq=False)
class ReprojectingRasterField:
    """Wrap a georeader reader, presenting a reprojected destination grid.

    The domain reports the *destination* CRS / transform / shape, so
    samplers place anchors on the reprojected grid; each `select` warps
    the corresponding source pixels into that chip.

    Args:
        reader: Any georeader `GeoData` (a `RasterioReader`, a
            `GeoTensor`, …) in the source CRS.
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
        the same chip grid. Use a single ``georeader.read.read_reproject``
        call when bit-exactness against a full warp matters.
    """

    reader: GeoData
    dst_crs: Any
    resolution: float | tuple[float, float] | None = None
    resampling: str = "bilinear"

    def __post_init__(self) -> None:
        from georeader.read import calculate_transform_window

        window_data, dst_transform = calculate_transform_window(
            self.reader, self.dst_crs, self.resolution
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

    @property
    def domain(self) -> _ReprojectedDomain:
        return _ReprojectedDomain(
            crs=self.dst_crs, transform=self._transform, shape=self._shape
        )

    def select(self, window: Any) -> GeoTensor:
        from georeader.read import read_reproject
        from rasterio.windows import Window, transform as window_transform

        chip_transform = window_transform(window, self._transform)
        out = Window(
            col_off=0, row_off=0, width=int(window.width), height=int(window.height)
        )
        return read_reproject(
            self._crop_source(out, chip_transform),
            dst_crs=self.dst_crs,
            dst_transform=chip_transform,
            window_out=out,
            resampling=self._resampling,
        )

    def _crop_source(self, out: Any, chip_transform: Any) -> Any:
        """Source pixels covering the chip footprint, plus a kernel margin.

        ``read_reproject`` only crops lazy readers (and never a
        `GeoTensor`), so without this each chip would warp the whole
        source. The chip polygon is densified before being mapped into
        the source CRS so curved edges of the reprojected footprint are
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
        src_window = window_from_polygon(
            self.reader, footprint, crs_polygon=self.dst_crs
        )
        extent = (src_window.col_off, src_window.row_off)
        extent += (src_window.width, src_window.height)
        if not all(math.isfinite(float(v)) for v in extent):
            # Footprint not representable in the source CRS — let
            # `read_reproject` handle the full source.
            return self.reader
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
            self.reader, src_window, trigger_load=True, boundless=False
        )
        return self.reader if cropped is None else cropped

    def cache_id(self) -> str:
        """`PatchCache` identity: the source plus the warp parameters.

        Folds in the source reader's identity (``None`` for an in-memory
        source — `PatchCache` then requires an explicit ``field_id``),
        ``dst_crs``, ``resolution`` and ``resampling``, so two
        reprojections of one source never share cache entries.

        Examples:
            >>> f = ReprojectingRasterField(RasterioReader("a.tif"), "EPSG:3857")
            >>> f.cache_id()
            '{"dst_crs": "EPSG:3857", "resampling": "bilinear", ...}'
            >>> ReprojectingRasterField(geotensor, "EPSG:3857").cache_id()
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

    def with_data(self, array: Any) -> GeoTensor:
        return _rewrap(self.reader, array, self._transform, self.dst_crs)


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
