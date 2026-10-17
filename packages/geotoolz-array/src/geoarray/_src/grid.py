"""`Grid` and `PixelWindow`: where pixels are, without the pixels.

A grid is the triple (T, crs, (H, W)). The affine transform
T = (a, b, c, d, e, f) maps a pixel corner (column j, row i) to map
coordinates:

    x = a·j + b·i + c
    y = d·j + e·i + f

so the centre of pixel (i, j) is T(j + ½, i + ½). For a north-up grid
b = d = 0, a is the pixel width and e is minus the pixel height. The
transform is stored in this (a, b, c, d, e, f) order — the order of
``affine.Affine`` and rasterio, not GDAL's (c, a, b, f, d, e).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyproj
from affine import Affine

from geoarray._src.backend import namespace


__all__ = [
    "Grid",
    "PixelWindow",
    "Transform",
    "apply_transform",
    "as_crs",
    "as_transform",
    "shift_scale_transform",
]

Transform = tuple[float, float, float, float, float, float]


def as_transform(transform: Any) -> Transform:
    """Coerce a transform to a tuple of six Python floats (a, b, c, d, e, f).

    Args:
        transform: An ``affine.Affine`` (rasterio's ``Affine`` is the same
            class), six numbers, the nine numbers of a 3 × 3 affine matrix,
            or a ``(2, 3)`` / ``(3, 3)`` array.

    Returns:
        ``(a, b, c, d, e, f)`` as Python floats.

    Raises:
        ValueError: The input does not describe an affine transform.

    Example:
        >>> as_transform(Affine(20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0))
        (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
    """
    if isinstance(transform, Affine):
        return (
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        )
    try:
        arr = np.asarray(transform, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not an affine transform: {transform!r}") from exc
    flat = arr.reshape(-1)
    if flat.size == 9:
        if not np.allclose(flat[6:], (0.0, 0.0, 1.0)):
            raise ValueError(
                f"the last row of an affine matrix must be (0, 0, 1): {flat[6:]}"
            )
        flat = flat[:6]
    if flat.size != 6:
        raise ValueError(
            "an affine transform has 6 numbers (a, b, c, d, e, f); "
            f"got shape {arr.shape}"
        )
    a, b, c, d, e, f = (float(v) for v in flat)
    return (a, b, c, d, e, f)


def as_crs(crs: Any) -> pyproj.CRS | None:
    """Coerce ``crs`` to a ``pyproj.CRS``; ``None`` stays ``None``.

    Accepts anything ``pyproj.CRS.from_user_input`` does: ``"EPSG:32613"``,
    an EPSG integer, WKT, PROJ JSON, or an object with ``to_wkt()`` such as
    a rasterio ``CRS``.

    Raises:
        ValueError: ``crs`` is not a CRS pyproj understands.

    Example:
        >>> as_crs(32613).to_epsg()
        32613
    """
    if crs is None or isinstance(crs, pyproj.CRS):
        return crs
    try:
        return pyproj.CRS.from_user_input(crs)
    except pyproj.exceptions.CRSError as exc:
        raise ValueError(f"not a CRS: {crs!r}") from exc


def _crs_string(crs: pyproj.CRS | None) -> str | None:
    """The shortest exact string for ``crs``: ``"EPSG:32613"`` or its WKT."""
    if crs is None:
        return None
    authority = crs.to_authority()
    if authority is not None:
        candidate = f"{authority[0]}:{authority[1]}"
        if pyproj.CRS.from_user_input(candidate) == crs:
            return candidate
    return crs.to_wkt()


def _as_index(value: Any, name: str) -> int:
    """A non-negative Python int, or `ValueError`."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def apply_transform(transform: Any, cols: Any, rows: Any) -> tuple[Any, Any]:
    """Map pixel positions to map coordinates: (x, y) = T(cols, rows).

    x = a·j + b·i + c, y = d·j + e·i + f, for a column j and a row i
    (corners at integers, centres at + ½). Works on a tuple or a ``(6,)``
    NumPy / JAX transform and on scalars or arrays of positions.
    """
    a, b, c, d, e, f = (transform[k] for k in range(6))
    return (a * cols + b * rows + c, d * cols + e * rows + f)


def shift_scale_transform(
    transform: Any,
    row0: Any,
    col0: Any,
    row_step: Any = 1,
    col_step: Any = 1,
) -> Any:
    """The transform of a slice starting at (row0, col0) with the given steps.

    a′ = a·s_c, b′ = b·s_r, c′ = c + a·c₀ + b·r₀,
    d′ = d·s_c, e′ = e·s_r, f′ = f + d·c₀ + e·r₀.

    Works on ``(6,)`` and batched ``(*batch, 6)`` NumPy or JAX transforms,
    with traced offsets.
    """
    xp = namespace(transform)
    a, b, c, d, e, f = (transform[..., i] for i in range(6))
    columns = (
        a * col_step,
        b * row_step,
        c + a * col0 + b * row0,
        d * col_step,
        e * row_step,
        f + d * col0 + e * row0,
    )
    return xp.stack(columns, axis=-1)


def _snap(value: float, atol: float) -> float:
    """``value`` rounded to the nearest integer when within ``atol`` of it."""
    nearest = round(value)
    return float(nearest) if abs(value - nearest) <= atol else value


def _invert(transform: Transform) -> Transform:
    """The inverse affine: map coordinates → (column, row).

    Raises:
        ValueError: The transform is singular (a·e − b·d = 0).
    """
    a, b, c, d, e, f = transform
    det = a * e - b * d
    if det == 0.0:
        raise ValueError(f"a singular transform cannot be inverted: {transform}")
    return (
        e / det,
        -b / det,
        (b * f - c * e) / det,
        -d / det,
        a / det,
        (c * d - a * f) / det,
    )


@dataclass(frozen=True)
class PixelWindow:
    """A rectangle of pixels: ``col_off``, ``row_off``, ``width``, ``height``.

    rasterio's ``Window`` without importing rasterio, with its field order
    (columns before rows), so ``PixelWindow(*rasterio_window.flatten())``
    is the same rectangle. Offsets may be negative (a window that starts
    outside the array); sizes may not.

    Example:
        >>> w = PixelWindow.from_slices(slice(10, 42), slice(5, 37))
        >>> w
        PixelWindow(col_off=5, row_off=10, width=32, height=32)
        >>> w.to_slices()
        (slice(10, 42, None), slice(5, 37, None))
    """

    col_off: int
    row_off: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("col_off", "row_off"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise TypeError(f"{name} must be an integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "height", _as_index(self.height, "height"))
        object.__setattr__(self, "width", _as_index(self.width, "width"))

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)``."""
        return (self.height, self.width)

    @classmethod
    def from_slices(cls, rows: slice, cols: slice) -> PixelWindow:
        """The window two step-1 slices with explicit bounds select.

        Raises:
            ValueError: A slice has a step other than 1 or an open bound.
        """
        bounds = []
        for name, s in (("rows", rows), ("cols", cols)):
            if s.step not in (None, 1):
                raise ValueError(f"{name}: a window needs step 1, got {s.step}")
            if s.start is None or s.stop is None:
                raise ValueError(
                    f"{name}: a window needs explicit start and stop, got {s}"
                )
            bounds.append((int(s.start), int(s.stop)))
        (r0, r1), (c0, c1) = bounds
        return cls(
            col_off=c0, row_off=r0, width=max(c1 - c0, 0), height=max(r1 - r0, 0)
        )

    def to_slices(self) -> tuple[slice, slice]:
        """``(rows, cols)`` slices selecting this window."""
        return (
            slice(self.row_off, self.row_off + self.height),
            slice(self.col_off, self.col_off + self.width),
        )

    def intersection(self, other: PixelWindow) -> PixelWindow | None:
        """The overlap of two windows, or ``None`` when they do not overlap.

        Example:
            >>> a = PixelWindow(col_off=0, row_off=0, width=10, height=10)
            >>> a.intersection(PixelWindow(col_off=5, row_off=5, width=10, height=10))
            PixelWindow(col_off=5, row_off=5, width=5, height=5)
        """
        r0 = max(self.row_off, other.row_off)
        c0 = max(self.col_off, other.col_off)
        r1 = min(self.row_off + self.height, other.row_off + other.height)
        c1 = min(self.col_off + self.width, other.col_off + other.width)
        if r1 <= r0 or c1 <= c0:
            return None
        return PixelWindow(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)


@dataclass(frozen=True)
class Grid:
    """A pixel grid with no data: transform, CRS and spatial shape.

    Hashable, so it can be a warp target, an operator's static config or a
    dictionary key. ``==`` is exact equality; `isclose` allows a tolerance.

    Args:
        transform: The affine transform (anything `as_transform` accepts).
        crs: The CRS (anything `as_crs` accepts), or ``None``.
        shape: ``(H, W)``.

    Example:
        >>> transform = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
        >>> grid = Grid(transform, "EPSG:32613", (600, 600))
        >>> grid.res
        (20.0, 20.0)
        >>> grid.bounds
        (600000.0, 3538000.0, 612000.0, 3550000.0)
    """

    transform: Transform
    crs: pyproj.CRS | None
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", as_transform(self.transform))
        object.__setattr__(self, "crs", as_crs(self.crs))
        shape = tuple(self.shape)
        if len(shape) != 2:
            raise ValueError(f"a grid's shape is (H, W), got {self.shape!r}")
        object.__setattr__(
            self, "shape", (_as_index(shape[0], "H"), _as_index(shape[1], "W"))
        )

    @property
    def height(self) -> int:
        """H, the number of rows."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """W, the number of columns."""
        return self.shape[1]

    @property
    def affine(self) -> Affine:
        """The transform as an ``affine.Affine`` (what rasterio expects)."""
        return Affine(*self.transform)

    @property
    def is_axis_aligned(self) -> bool:
        """Whether the grid has no rotation or shear: b = d = 0."""
        _, b, _, d, _, _ = self.transform
        return b == 0.0 and d == 0.0

    @property
    def res(self) -> tuple[float, float]:
        """Pixel size ``(x, y)`` in CRS units, always positive.

        For a rotated grid, the lengths of the column and row vectors:
        (√(a² + d²), √(b² + e²)).
        """
        a, b, _, d, e, _ = self.transform
        return (math.hypot(a, d), math.hypot(b, e))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` of the grid's outer edges."""
        h, w = self.shape
        corners = [
            apply_transform(self.transform, j, i) for j in (0, w) for i in (0, h)
        ]
        xs, ys = zip(*corners, strict=True)
        return (min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def from_affine(cls, affine: Affine, crs: Any, shape: Sequence[int]) -> Grid:
        """A grid from an ``affine.Affine`` (or rasterio ``Affine``)."""
        h, w = shape
        return cls(as_transform(affine), crs, (h, w))

    def shift(self, rows: float, cols: float) -> Grid:
        """The same grid with its origin moved by ``rows`` × ``cols`` pixels.

        c′ = c + a·cols + b·rows, f′ = f + d·cols + e·rows.

        Example:
            >>> grid = Grid((20.0, 0.0, 0.0, 0.0, -20.0, 0.0), None, (4, 4))
            >>> grid.shift(1, 2).transform
            (20.0, 0.0, 40.0, 0.0, -20.0, -20.0)
        """
        a, b, _, d, e, _ = self.transform
        c, f = apply_transform(self.transform, cols, rows)
        return Grid((a, b, c, d, e, f), self.crs, self.shape)

    @classmethod
    def from_bounds(
        cls,
        bounds: Sequence[float],
        res: float | Sequence[float],
        crs: Any,
        *,
        atol: float = 1e-6,
    ) -> Grid:
        """The north-up grid that covers ``bounds`` with square-edged pixels.

        The origin is the top-left corner (x_min, y_max); the shape rounds
        up to whole pixels, ignoring overshoots below ``atol`` pixels.

        Args:
            bounds: ``(x_min, y_min, x_max, y_max)`` in CRS units.
            res: Pixel size, one number or ``(x, y)``, positive.
            crs: The CRS (anything `as_crs` accepts).
            atol: Tolerance, in pixels, for a size that is almost whole.

        Example:
            >>> Grid.from_bounds((600000, 3538000, 612000, 3550000), 20, 32613).shape
            (600, 600)
        """
        x_min, y_min, x_max, y_max = (float(v) for v in bounds)
        rx, ry = (float(v) for v in np.broadcast_to(np.asarray(res, float), (2,)))
        if rx <= 0 or ry <= 0 or x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f"need positive res and non-empty bounds, got {bounds}, {res}"
            )
        width = math.ceil(_snap((x_max - x_min) / rx, atol))
        height = math.ceil(_snap((y_max - y_min) / ry, atol))
        return cls((rx, 0.0, x_min, 0.0, -ry, y_max), crs, (height, width))

    def window(self, bounds: Sequence[float], *, atol: float = 1e-6) -> PixelWindow:
        """The pixels that cover ``bounds``, rounded outward to whole pixels.

        Corners within ``atol`` pixels of a pixel edge snap to it, so bounds
        on the grid map to exactly the pixels inside them. The window may
        extend outside the grid; `PixelWindow.intersection` clips it.

        Args:
            bounds: ``(x_min, y_min, x_max, y_max)`` in this grid's CRS.
            atol: Snapping tolerance, in pixels.

        Example:
            >>> transform = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
            >>> grid = Grid(transform, None, (600, 600))
            >>> grid.window((600100, 3549000, 600400, 3549900))
            PixelWindow(col_off=5, row_off=5, width=15, height=45)
        """
        x_min, y_min, x_max, y_max = (float(v) for v in bounds)
        corners = ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
        ia, ib, ic, id_, ie, if_ = _invert(self.transform)
        cols = [ia * x + ib * y + ic for x, y in corners]
        rows = [id_ * x + ie * y + if_ for x, y in corners]
        r0 = math.floor(_snap(min(rows), atol))
        r1 = math.ceil(_snap(max(rows), atol))
        c0 = math.floor(_snap(min(cols), atol))
        c1 = math.ceil(_snap(max(cols), atol))
        return PixelWindow(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)

    def crop(self, window: PixelWindow) -> Grid:
        """The grid of ``window``: its origin and its shape.

        Example:
            >>> grid = Grid((20.0, 0.0, 0.0, 0.0, -20.0, 0.0), None, (8, 8))
            >>> window = PixelWindow(col_off=3, row_off=2, width=4, height=4)
            >>> grid.crop(window).transform
            (20.0, 0.0, 60.0, 0.0, -20.0, -40.0)
        """
        shifted = self.shift(window.row_off, window.col_off)
        return Grid(shifted.transform, self.crs, window.shape)

    def bounds_of(self, window: PixelWindow) -> tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` of ``window`` on this grid.

        The inverse of `window` for bounds on pixel edges:
        ``grid.bounds_of(grid.window(b)) == b``.
        """
        return self.crop(window).bounds

    def isclose(self, other: Grid, atol: float = 1e-6) -> bool:
        """Equal CRS and shape, and transforms within ``atol`` (map units).

        Example:
            >>> g = Grid((1.0, 0.0, 0.0, 0.0, -1.0, 0.0), None, (2, 2))
            >>> g.isclose(g.shift(1e-9, 0.0))
            True
        """
        return (
            self.shape == other.shape
            and self.crs == other.crs
            and all(
                abs(u - v) <= atol
                for u, v in zip(self.transform, other.transform, strict=True)
            )
        )

    def get_config(self) -> dict[str, Any]:
        """A strict-JSON config: the transform, the CRS string and the shape.

        Example:
            >>> grid = Grid((20.0, 0.0, 0.0, 0.0, -20.0, 0.0), "EPSG:32613", (2, 3))
            >>> grid.get_config()["crs"], grid.get_config()["shape"]
            ('EPSG:32613', [2, 3])
        """
        return {
            "transform": list(self.transform),
            "crs": _crs_string(self.crs),
            "shape": list(self.shape),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Grid:
        """Rebuild a grid from `get_config`; ``from_config(g.get_config()) == g``."""
        return cls(tuple(config["transform"]), config["crs"], tuple(config["shape"]))
