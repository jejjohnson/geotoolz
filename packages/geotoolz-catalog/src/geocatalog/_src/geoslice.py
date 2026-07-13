"""`GeoSlice` — the unit of work between catalog, sampler, loader, operator.

A `GeoSlice` is a **bounded request for data**: a bbox, a time interval,
a target resolution, and a CRS. Catalogs produce them; loaders consume
them; downstream patchers (e.g. `geotoolz.patch`) compose them. It is
the cross-cutting wire format that decouples the catalog layer from the
patcher layer from the reader layer.

The dataclass is ``frozen=True`` so it can be hashed, used as a dict
key, or shipped across function boundaries without anyone mutating it
in flight. Code that wants to "change" a slice uses
``dataclasses.replace(slice_, bounds=new_bounds)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, get_args

import numpy as np
import pandas as pd
import pyproj
from rasterio import Affine
from rasterio.windows import (
    Window,
    bounds as window_bounds,
    from_bounds,
)

from geocatalog._src._align import Align, GridAlignmentWarning, divide_evenly


_VALID_ALIGN_MODES: frozenset[str] = frozenset(get_args(Align))


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
    # Construction-time alignment policy. NOT part of identity:
    # two slices with the same bounds/interval/resolution/crs compare
    # equal and hash equal regardless of ``align`` (see #6.7 of the
    # design doc / geopatcher#59).
    align: Align = field(
        default="off",
        compare=False,
        hash=False,
        repr=False,
    )

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
        # Reject unknown modes at construction so a typo like
        # ``align="warning"`` doesn't silently disable validation.
        # The `Literal` annotation is documentation, not runtime
        # enforcement.
        if self.align not in _VALID_ALIGN_MODES:
            raise ValueError(
                f"GeoSlice.align must be one of "
                f"{sorted(_VALID_ALIGN_MODES)!r}; got {self.align!r}"
            )
        if self.align != "off":
            self._check_or_snap_alignment()

    def _check_or_snap_alignment(self) -> None:
        """Validate or snap ``bounds`` against the alignment policy.

        Dispatches per-axis on the current ``align`` mode:

        - ``"error"`` — raise on the first misaligned axis.
        - ``"warn"`` — emit a `GridAlignmentWarning` per misaligned
          axis, leave bounds alone.
        - ``"snap"`` — round outward, **preserving the affine origin**
          for that axis. For north-up rasters the affine maps pixel
          ``(0,0)`` to ``(xmin, ymax)``, so snap holds ``xmin`` and
          ``ymax`` fixed and extends ``xmax`` rightward / ``ymin``
          downward. The resulting bounds fully cover the original AOI.
        """
        xmin, ymin, xmax, ymax = self.bounds
        rx, ry = self.resolution
        # For north-up rasters the affine origin is (xmin, ymax). To
        # preserve it under snap, hold those fixed and move xmax
        # rightward / ymin downward.
        new_xmax, new_ymin = xmax, ymin
        axes = (
            (xmax - xmin, rx, "x"),
            (ymax - ymin, ry, "y"),
        )
        for length, step, axis in axes:
            try:
                divide_evenly(length, step, label=f"{axis}-extent")
            except ValueError as exc:
                if self.align == "error":
                    raise
                if self.align == "warn":
                    warnings.warn(
                        f"GeoSlice grid misalignment: {exc}",
                        GridAlignmentWarning,
                        stacklevel=4,
                    )
                    continue
                if self.align == "snap":
                    n_up = int(np.ceil(length / step))
                    if axis == "x":
                        old = xmax
                        new_xmax = xmin + n_up * step
                        new = new_xmax
                    else:
                        old = ymin
                        new_ymin = ymax - n_up * step
                        new = new_ymin
                    warnings.warn(
                        f"GeoSlice snap: {axis}-extent edge "
                        f"{old:.6g} -> {new:.6g} (n={n_up}, "
                        f"step={step:.6g})",
                        GridAlignmentWarning,
                        stacklevel=4,
                    )
        if self.align == "snap" and (new_xmax != xmax or new_ymin != ymin):
            object.__setattr__(self, "bounds", (xmin, new_ymin, new_xmax, ymax))

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

    def aligned_shape(self) -> tuple[int, int]:
        """Strict ``shape``: raises on misalignment regardless of mode.

        Use this when you need a guaranteed-exact pixel count without
        flipping the constructor's default ``align`` mode. ``.shape``
        stays `round`-based for backwards compatibility with existing
        loaders.

        Raises:
            ValueError: if either axis's extent is not an integer
                multiple of its resolution (within tolerance).
        """
        rx, ry = self.resolution
        xmin, ymin, xmax, ymax = self.bounds
        return (
            divide_evenly(ymax - ymin, ry, label="y-extent"),
            divide_evenly(xmax - xmin, rx, label="x-extent"),
        )

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
        roughly stable.

        Raises:
            NotImplementedError: The reprojection produces a degenerate
                or non-finite box — a pole-wrapping, antimeridian-adjacent,
                or out-of-domain transform. Split or clip the slice first.
                (Antimeridian-*crossing* input bounds are already rejected
                at construction: ``__post_init__`` requires
                ``xmin < xmax``.)
        """
        target = (
            target_crs
            if isinstance(target_crs, pyproj.CRS)
            else pyproj.CRS.from_user_input(target_crs)
        )
        if target == self.crs:
            return self
        xmin, ymin, xmax, ymax = self.bounds
        transformer = pyproj.Transformer.from_crs(self.crs, target, always_xy=True)
        new_bounds = transformer.transform_bounds(xmin, ymin, xmax, ymax)
        if not all(np.isfinite(b) for b in new_bounds) or not (
            new_bounds[0] < new_bounds[2] and new_bounds[1] < new_bounds[3]
        ):
            raise NotImplementedError(
                f"GeoSlice.to_crs produced a degenerate box reprojecting "
                f"bounds={self.bounds} from {self.crs} to {target}: got "
                f"{tuple(new_bounds)}. This usually means the bounds wrap a "
                f"pole or leave the target CRS's domain — split or clip the "
                f"slice first."
            )
        # Preserve output shape: scale resolution by the bbox width ratio.
        old_w = xmax - xmin
        new_w = new_bounds[2] - new_bounds[0]
        old_h = ymax - ymin
        new_h = new_bounds[3] - new_bounds[1]
        x_res = self.resolution[0] * (new_w / old_w)
        y_res = self.resolution[1] * (new_h / old_h)
        # Reprojected bounds are *generically* non-integer multiples
        # of the rescaled resolution; force align="off" so a strict
        # parent's policy doesn't cause to_crs to raise on its own
        # output. Callers wanting validation can reconstruct.
        return GeoSlice(
            bounds=new_bounds,
            interval=self.interval,
            resolution=(x_res, y_res),
            crs=target,
            align="off",
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
