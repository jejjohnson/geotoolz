"""The raster carrier at geopatcher's boundary: `geoarray.GeoArray`.

Every public raster in and out of geopatcher is a `geoarray.GeoArray`, and
every lazy raster a `geoarray.RasterSource`. georeader stays an internal
engine (reprojection, polygon windows), reached through
`geoarray.interop.georeader`; a georeader ``GeoTensor`` or reader passed
in by a user is refused with a `TypeError` naming the adapter.

This module holds the few helpers every raster path shares:

- `reject_georeader` — the refusal (reads the class hierarchy only, so it
  never imports georeader).
- `reject_georeader_data` — the same refusal on a patch payload (a dict's
  members too).
- `holds_fill` / `output_fill` — whether a dtype can hold a fill, and the
  fill an output of that dtype declares (ADR-012).
- `raster_grid` — the `geoarray.Grid` of a raster domain, or ``None`` for
  any other domain. A raster domain is a `geoarray.Grid` or anything with a
  ``grid`` (every `RasterSource`, `geocloud.cog.CogDomain`).
- `as_pixel_window` — a rasterio ``Window`` (what the geometries produce)
  as a `geoarray.PixelWindow`.
- `pad_carrier` — ``np.pad`` on a GeoArray's pixels, the transform moved
  with them and the fill kept.
- `to_wire` / `from_wire` — a compact picklable form of a GeoArray that
  is rebuilt through the constructor on the other side.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from geoarray import GeoArray, Grid, PixelWindow
from rasterio import Affine


__all__ = [
    "as_pixel_window",
    "from_wire",
    "holds_fill",
    "output_fill",
    "pad_carrier",
    "raster_grid",
    "reject_georeader",
    "reject_georeader_data",
    "to_wire",
]


def _georeader_classes(x: Any) -> list[type]:
    return [
        cls
        for cls in type(x).__mro__
        if cls.__module__ == "georeader" or cls.__module__.startswith("georeader.")
    ]


def reject_georeader(x: Any, where: str) -> None:
    """Raise `TypeError` when ``x`` is a georeader ``GeoTensor`` or reader.

    Args:
        x: The value a public geopatcher entry point received.
        where: The entry point's name, for the message.

    Raises:
        TypeError: ``x`` is a georeader object; the message names
            ``geoarray.interop.georeader.from_georeader`` for a
            ``GeoTensor`` and ``source_from_georeader`` for a lazy reader.
    """
    classes = _georeader_classes(x)
    if not classes:
        return
    if any(cls.__name__ == "GeoTensor" for cls in classes):
        raise TypeError(
            f"{where} takes a geoarray.GeoArray, got a georeader GeoTensor; "
            "convert it with geoarray.interop.georeader.from_georeader(gt)"
        )
    raise TypeError(
        f"{where} takes a geoarray.RasterSource, got georeader's "
        f"{type(x).__name__}; wrap it with "
        "geoarray.interop.georeader.source_from_georeader(reader)"
    )


def reject_georeader_data(data: Any, where: str) -> None:
    """`reject_georeader` on a patch payload: the value, or each of a dict's.

    A matched patch carries ``{source name: carrier}``; every member is
    checked.
    """
    reject_georeader(data, where)
    if isinstance(data, dict):
        for value in data.values():
            reject_georeader(value, where)


def holds_fill(dtype: Any, fill: Any) -> bool:
    """Whether data of ``dtype`` holds ``fill`` exactly (geoarray's own check).

    ``None`` always fits. Float and complex data hold any fill, NaN
    included; integer data a finite integral value in range; bool data
    ``0`` / ``1``. Anything else (a string fill, an object dtype) does not.

    Example:
        >>> holds_fill(np.uint8, 255), holds_fill(np.uint8, -9999.0)
        (True, False)
    """
    if fill is None:
        return True
    dtype = np.dtype(dtype)
    if dtype.kind in "fc":
        return True
    try:
        value = float(fill)
    except (TypeError, ValueError):
        return False
    if dtype.kind == "b":
        return value in (0.0, 1.0)
    if dtype.kind in "iu":
        info = np.iinfo(dtype.type)
        return (
            bool(np.isfinite(value))
            and value.is_integer()
            and (info.min <= value <= info.max)
        )
    return False


def output_fill(fill: Any, dtype: Any) -> Any:
    """The fill an output of ``dtype`` declares when ``fill`` is the wanted one.

    The rule every raster ``with_data``, `merge_to_field` and
    ``MergePatches`` follow (ADR-012): keep ``fill`` when ``dtype`` holds
    it; else NaN for a float / complex output; else ``None`` (no nodata) —
    never a fill the data cannot hold, which `geoarray.GeoArray` refuses.

    Example:
        >>> output_fill(-9999.0, np.uint8) is None
        True
        >>> output_fill(-9999.0, np.float32)
        -9999.0
    """
    if holds_fill(dtype, fill):
        return fill
    return np.nan if np.dtype(dtype).kind in "fc" else None


def raster_grid(domain: Any) -> Grid | None:
    """The pixel grid of a raster domain, or ``None`` for any other domain.

    A raster domain is a `geoarray.Grid` itself or an object whose ``grid``
    is one: every `geoarray.RasterSource` (a `GeoArray`, a lazy reader),
    `geocloud.cog.CogDomain`, the reprojected and rioxarray domains.

    Example:
        >>> grid = Grid((10.0, 0.0, 0.0, 0.0, -10.0, 0.0), "EPSG:32630", (64, 64))
        >>> raster_grid(grid) is grid
        True
    """
    if isinstance(domain, Grid):
        return domain
    try:
        grid = getattr(domain, "grid", None)
    except (AttributeError, ValueError, TypeError):  # a traced / batched carrier
        return None
    return grid if isinstance(grid, Grid) else None


def _whole_number(value: Any, name: str) -> int:
    as_int = round(float(value))
    if abs(float(value) - as_int) > 1e-9:
        raise ValueError(
            f"window {name}={value} is not a whole number of pixels; round the "
            "window first (rasterio.windows.Window.round_offsets / round_lengths)"
        )
    return int(as_int)


def as_pixel_window(window: Any) -> PixelWindow:
    """A rasterio ``Window`` (or a `PixelWindow`) as a `geoarray.PixelWindow`.

    Raises:
        ValueError: An offset or size is not a whole number of pixels.

    Example:
        >>> from rasterio.windows import Window
        >>> as_pixel_window(Window(col_off=8, row_off=0, width=16, height=16))
        PixelWindow(col_off=8, row_off=0, width=16, height=16)
    """
    if isinstance(window, PixelWindow):
        return window
    return PixelWindow(
        col_off=_whole_number(window.col_off, "col_off"),
        row_off=_whole_number(window.row_off, "row_off"),
        width=_whole_number(window.width, "width"),
        height=_whole_number(window.height, "height"),
    )


def pad_carrier(
    g: GeoArray, pads: tuple[int, int, int, int], mode: str, fill: Any
) -> GeoArray:
    """Pad a GeoArray's spatial axes, moving the transform with the pixels.

    Unlike `geoarray.pad` (constant fill only, and the fill becomes the
    carrier's), this pads with any ``np.pad`` mode and keeps the carrier's
    own ``fill_value``: a chip padded with ``pad_value`` still declares the
    source's nodata.

    Args:
        g: The carrier, ``(..., H, W)``.
        pads: ``(top, bottom, left, right)`` widths in pixels.
        mode: ``"constant"``; anything else mirrors (``"reflect"``).
        fill: The constant for ``mode="constant"``.

    Returns:
        ``(..., top + H + bottom, left + W + right)`` with origin
        T(-left, -top).
    """
    top, bottom, left, right = pads
    widths = [(0, 0)] * (g.ndim - 2) + [(top, bottom), (left, right)]
    values = np.asarray(g)
    if mode == "constant":
        padded = np.pad(values, widths, mode="constant", constant_values=fill)
    else:
        padded = np.pad(values, widths, mode="reflect")
    # c' = c - a·left - b·top, f' = f - d·left - e·top
    transform = g.grid.affine * Affine.translation(-left, -top)
    return g.with_data(padded, transform=transform)


class _WireGeoArray:
    """Compact picklable form of a `GeoArray`, rebuilt by its constructor.

    The pixels plus the grid's strict-JSON config (transform, CRS string,
    shape), the fill and the attrs as a plain dict. Rebuilding through the
    constructor recomputes everything derived (the attrs' hash, the NaN
    fill), so a carrier that crossed a process boundary compares equal to
    one built there.
    """

    __slots__ = ("attrs", "data", "fill_value", "grid")

    def __init__(self, g: GeoArray) -> None:
        self.data = g.data
        self.grid = g.grid.get_config()
        self.fill_value = g.fill_value
        self.attrs = g.attrs.to_dict()

    def __getstate__(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in self.__slots__)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        for name, value in zip(self.__slots__, state, strict=True):
            setattr(self, name, value)

    def rebuild(self) -> GeoArray:
        grid = Grid.from_config(self.grid)
        return GeoArray(
            self.data,
            grid.transform,
            grid.crs,
            fill_value=self.fill_value,
            attrs=self.attrs,
        )


def to_wire(data: Any) -> Any:
    """``data`` ready to pickle: a `GeoArray` as `_WireGeoArray`, else as is."""
    if isinstance(data, GeoArray) and not data.batch_shape:
        return _WireGeoArray(data)
    return data


def from_wire(data: Any) -> Any:
    """Inverse of `to_wire`."""
    return data.rebuild() if isinstance(data, _WireGeoArray) else data
