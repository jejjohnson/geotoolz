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

from dataclasses import dataclass
from typing import Any

from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor


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
            self, "_shape", (int(window_data.height), int(window_data.width))
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
            self.reader,
            dst_crs=self.dst_crs,
            dst_transform=chip_transform,
            window_out=out,
            resampling=self._resampling,
        )

    def with_data(self, array: Any) -> GeoTensor:
        return GeoTensor(values=array, transform=self._transform, crs=self.dst_crs)


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
