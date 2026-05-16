"""`GeoSlice` — the unit of work between catalog, sampler, loader, operator.

A `GeoSlice` is a **bounded request for data**: a bbox, a time interval,
a target resolution, and a CRS. Catalogs produce them; loaders consume
them; `geopatcher` composes them. It is the cross-cutting wire
format that decouples the catalog layer from the patcher layer from the
reader layer.

The dataclass is ``frozen=True`` so it can be hashed, used as a dict
key, or shipped across function boundaries without anyone mutating it
in flight. Code that wants to "change" a slice uses
``dataclasses.replace(slice_, bounds=new_bounds)``.

See ``research_journal_v2/notes/geotoolz/plans/types/geoslice.md`` for
the design report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyproj
from rasterio import Affine
from rasterio.windows import (
    Window,
    bounds as window_bounds,
    from_bounds,
)


# Number of decimal digits within which ``(bounds, resolution)`` must
# round to integer pixel counts. The catalog/sampler layer guarantees
# this; loaders may rely on it without re-checking.
PIXEL_PRECISION: int = 3


@dataclass(frozen=True)
class GeoSlice:
    """A bounded request for data — produced by samplers, consumed by loaders.

    Carries everything a loader needs to fetch a chip without consulting
    the catalog: bbox in CRS units, time interval, target resolution, CRS.

    Args:
        bounds: ``(xmin, ymin, xmax, ymax)`` in ``crs`` units.
        interval: A ``pd.Interval`` (``closed='both'``) over the time axis.
        resolution: ``(x_res, y_res)`` in CRS units. Both positive.
        crs: A ``pyproj.CRS``. String / EPSG-int inputs are coerced.
    """

    bounds: tuple[float, float, float, float]
    interval: pd.Interval
    resolution: tuple[float, float]
    crs: pyproj.CRS

    def __post_init__(self) -> None:
        xmin, ymin, xmax, ymax = self.bounds
        if not (xmin < xmax and ymin < ymax):
            raise ValueError(
                f"GeoSlice.bounds must satisfy xmin < xmax and ymin < ymax; "
                f"got {self.bounds!r}"
            )
        if not isinstance(self.interval, pd.Interval):
            raise TypeError(
                "GeoSlice.interval must be a pd.Interval; "
                f"got {type(self.interval).__name__}"
            )
        if self.interval.closed != "both":
            raise ValueError(
                "GeoSlice.interval must be closed='both' for consistency with "
                f"the catalog IntervalIndex; got closed={self.interval.closed!r}"
            )
        x_res, y_res = self.resolution
        if x_res <= 0 or y_res <= 0:
            raise ValueError(
                f"GeoSlice.resolution must be positive on both axes; "
                f"got {self.resolution!r}"
            )
        # Coerce string / EPSG-int CRS to pyproj.CRS so equality + serialisation
        # are well-defined. Frozen dataclasses forbid normal assignment;
        # object.__setattr__ is the standard escape hatch in __post_init__.
        if not isinstance(self.crs, pyproj.CRS):
            object.__setattr__(self, "crs", pyproj.CRS.from_user_input(self.crs))

    @property
    def shape(self) -> tuple[int, int]:
        """Output grid shape ``(height, width)`` from bounds + resolution."""
        x_res, y_res = self.resolution
        xmin, ymin, xmax, ymax = self.bounds
        return (
            round((ymax - ymin) / y_res),
            round((xmax - xmin) / x_res),
        )

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def transform(self) -> Affine:
        """North-up affine from bounds + resolution."""
        x_res, y_res = self.resolution
        xmin, _ymin, _xmax, ymax = self.bounds
        return Affine(x_res, 0.0, xmin, 0.0, -y_res, ymax)

    def to_crs(self, target_crs: Any) -> GeoSlice:
        """Reproject this slice's bounds into ``target_crs``.

        The resolution is conservatively rescaled by the ratio of the
        projected-vs-source bbox widths so the output ``shape`` stays
        roughly stable. Antimeridian-crossing reprojections are not
        handled — split the slice first.
        """
        target = (
            target_crs
            if isinstance(target_crs, pyproj.CRS)
            else pyproj.CRS.from_user_input(target_crs)
        )
        if target == self.crs:
            return self
        transformer = pyproj.Transformer.from_crs(self.crs, target, always_xy=True)
        xmin, ymin, xmax, ymax = self.bounds
        new_bounds = transformer.transform_bounds(xmin, ymin, xmax, ymax)
        # Preserve output shape: scale resolution by the bbox width ratio.
        old_w = xmax - xmin
        new_w = new_bounds[2] - new_bounds[0]
        old_h = ymax - ymin
        new_h = new_bounds[3] - new_bounds[1]
        x_res = self.resolution[0] * (new_w / old_w)
        y_res = self.resolution[1] * (new_h / old_h)
        return GeoSlice(
            bounds=new_bounds,
            interval=self.interval,
            resolution=(x_res, y_res),
            crs=target,
        )


def slice_to_window(slice_: GeoSlice, transform: Affine) -> Window:
    """Convert a CRS-unit ``GeoSlice`` to a pixel-space ``Window``.

    Lossy in the time / CRS axes — the returned ``Window`` carries
    neither.
    """
    return from_bounds(*slice_.bounds, transform=transform)


def window_to_slice(
    window: Window,
    transform: Affine,
    crs: Any,
    interval: pd.Interval,
    resolution: tuple[float, float],
) -> GeoSlice:
    """Inverse of `slice_to_window`. Mostly useful in tests and debugging."""
    b = window_bounds(window, transform)
    return GeoSlice(bounds=b, interval=interval, resolution=resolution, crs=crs)
