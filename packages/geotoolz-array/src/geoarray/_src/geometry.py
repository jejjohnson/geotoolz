"""Operations that re-index or re-describe pixels without computing new values.

The membership test for this module (and for geoarray): the output holds
only input values, fill values or coordinates. Each function is written
once against the data's backend (`namespace`), so it runs on NumPy and on
JAX, and inside ``jit`` where noted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from geoarray._src.array import INHERIT, GeoArray, _normalize_fill
from geoarray._src.attrs import concat_bands
from geoarray._src.backend import backend_of, is_tracer, namespace, require_x64
from geoarray._src.errors import GridMismatchError
from geoarray._src.grid import PixelWindow, apply_transform, shift_scale_transform


__all__ = [
    "combine",
    "concatenate",
    "crop",
    "footprint",
    "pad",
    "pixel_coords",
    "read_window",
    "stack",
    "write_window",
]

_SHAPELY_HINT = "geoarray.footprint needs shapely: pip install 'geotoolz-array[vector]'"


def _require_carriers(carriers: Sequence[Any], what: str) -> list[GeoArray]:
    if not carriers:
        raise ValueError(f"{what} needs at least one GeoArray")
    for g in carriers:
        if not isinstance(g, GeoArray):
            raise TypeError(f"{what} takes GeoArrays, got {type(g).__name__}")
    return list(carriers)


def _require_same_grid(carriers: list[GeoArray], what: str) -> None:
    """Raise `GridMismatchError` unless every carrier shares the first's grid.

    CRS, spatial shape and batch shape are always compared; the transforms
    are compared exactly when concrete (under a trace they are values JAX
    cannot read, so the caller is responsible for them).
    """
    first = carriers[0]
    for i, g in enumerate(carriers[1:], start=1):
        if g.crs != first.crs:
            raise GridMismatchError(
                f"{what}: carrier {i} has CRS {g.crs}, not {first.crs}"
            )
        if g.shape[-2:] != first.shape[-2:] or g.batch_shape != first.batch_shape:
            raise GridMismatchError(
                f"{what}: carrier {i} has spatial shape {g.shape[-2:]} and batch "
                f"{g.batch_shape}, not {first.shape[-2:]} and {first.batch_shape}"
            )
        if not (is_tracer(g.transform) or is_tracer(first.transform)) and not (
            np.array_equal(np.asarray(g.transform), np.asarray(first.transform))
        ):
            theirs = np.asarray(g.transform).tolist()
            ours = np.asarray(first.transform).tolist()
            raise GridMismatchError(
                f"{what}: carrier {i} has transform {theirs}, not {ours}"
            )


def _same_fill(a: Any, b: Any) -> bool:
    """Equal fills; every NaN fill is the one ``math.nan`` object."""
    return a is b or a == b


def _require_same_fill(carriers: list[GeoArray], what: str) -> None:
    """Raise `ValueError` unless every carrier has the first's fill value.

    The result keeps one fill, so a carrier with another one would have its
    nodata pixels read as data.
    """
    first = carriers[0].fill_value
    for i, g in enumerate(carriers[1:], start=1):
        if not _same_fill(g.fill_value, first):
            raise ValueError(
                f"{what}: carrier {i} has fill_value {g.fill_value!r}, not "
                f"{first!r}; give the carriers one fill first"
            )


def _promote(carriers: list[GeoArray]) -> list[GeoArray]:
    """All carriers on JAX when any is (C1: mixed inputs promote, as ``jnp``).

    Raises:
        X64Error: Promotion needs ``jax_enable_x64`` (float64 transforms).
    """
    if all(g.backend == "numpy" for g in carriers):
        return carriers
    return [g.to_jax() if g.backend == "numpy" else g for g in carriers]


def _leading_axis(g: GeoArray, axis: int, what: str, *, new: bool = False) -> int:
    """A leading axis index after the batch axes, normalised to be >= 0."""
    ndim = g.ndim + (1 if new else 0)
    n_batch = len(g.batch_shape)
    normalised = axis + ndim if axis < 0 else axis
    if not n_batch <= normalised < ndim - 2:
        raise ValueError(
            f"{what}: axis {axis} is not a leading axis of a {ndim}-D result "
            f"(after the {n_batch} batch axes, before the two spatial axes)"
        )
    return normalised


def combine(f: Callable[..., Any], *carriers: GeoArray) -> GeoArray:
    """Apply ``f`` to the data of carriers that share one pixel grid.

    The result keeps the first carrier's grid, CRS, fill value and attrs.

    Args:
        f: A function of the data arrays, returning one ``(..., H, W)``
            array (``jnp.divide``, ``lambda a, b: a - b``, …).
        *carriers: GeoArrays on the same grid; NumPy carriers join JAX ones
            on JAX.

    Raises:
        GridMismatchError: The grids differ (CRS, shape, or concrete
            transform).
        ValueError: The fill values differ.
        X64Error: A NumPy carrier must move to JAX and x64 is off.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> b11 = GeoArray(np.full((1, 2, 2), 2.0), (20, 0, 0, 0, -20, 0), 32613)
        >>> b12 = b11.with_data(np.full((1, 2, 2), 1.0))
        >>> combine(np.divide, b12, b11).data[0, 0, 0]
        np.float64(0.5)
    """
    gs = _require_carriers(carriers, "combine")
    gs = _promote(gs)
    _require_same_grid(gs, "combine")
    _require_same_fill(gs, "combine")
    return gs[0].with_data(f(*(g.data for g in gs)))


def stack(carriers: Sequence[GeoArray], axis: int = 0) -> GeoArray:
    """Stack carriers on one grid along a new leading axis.

    Three ``(C, H, W)`` scenes become one ``(3, C, H, W)`` time stack. The
    result keeps the first carrier's fill value and attrs.

    Raises:
        GridMismatchError: The grids differ.
        ValueError: The fill values differ, the result would have more
            than 4 dims, or ``axis`` is a batch or spatial axis.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> g = GeoArray(np.zeros((2, 4, 4)), (20, 0, 0, 0, -20, 0), 32613)
        >>> stack([g, g, g]).shape
        (3, 2, 4, 4)
    """
    gs = _require_carriers(carriers, "stack")
    gs = _promote(gs)
    _require_same_grid(gs, "stack")
    _require_same_fill(gs, "stack")
    first = gs[0]
    position = _leading_axis(first, axis, "stack", new=True)
    data = first.xp.stack([g.data for g in gs], axis=position)
    return first.with_data(data)


def concatenate(carriers: Sequence[GeoArray], axis: int = -3) -> GeoArray:
    """Join carriers on one grid along an existing leading axis.

    Along the band axis (−3, the default) the per-band attrs are joined
    too, when every carrier has them; otherwise they are dropped.

    Raises:
        GridMismatchError: The grids differ.
        ValueError: The fill values differ, or ``axis`` is a batch or
            spatial axis.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> t = (20, 0, 0, 0, -20, 0)
        >>> b04 = GeoArray(np.zeros((1, 4, 4)), t, 32613, attrs={"band_names": ["B04"]})
        >>> b08 = b04.with_data(np.ones((1, 4, 4)), attrs={"band_names": ["B08"]})
        >>> concatenate([b04, b08]).attrs["band_names"]
        ('B04', 'B08')
    """
    gs = _require_carriers(carriers, "concatenate")
    gs = _promote(gs)
    _require_same_grid(gs, "concatenate")
    _require_same_fill(gs, "concatenate")
    first = gs[0]
    position = _leading_axis(first, axis, "concatenate")
    data = first.xp.concatenate([g.data for g in gs], axis=position)
    on_bands = position == first.ndim - 3 and not first.batch_shape
    attrs = concat_bands([(g.attrs, g.count) for g in gs]) if on_bands else None
    return first.with_data(data, attrs=attrs)


def _whole(g: GeoArray) -> PixelWindow:
    """The window covering every pixel of ``g``."""
    return PixelWindow(col_off=0, row_off=0, width=g.width, height=g.height)


def _shifted(g: GeoArray, row0: Any, col0: Any) -> Any:
    return shift_scale_transform(g.transform, row0, col0)


def crop(g: GeoArray, offset: Any, size: tuple[int, int]) -> GeoArray:
    """The ``size`` pixels starting at ``offset``, with the matching transform.

    ``offset`` may be traced (inside ``jit`` / ``vmap``); ``size`` must be
    static. Like ``jax.lax.dynamic_slice``, an offset that would run past
    the edge is clamped so the crop fits, and the transform follows the
    pixels actually returned, so georeferencing is always right.

    x_p = x[..., r₀ : r₀ + h, c₀ : c₀ + w], c′ = c + a·c₀ + b·r₀,
    f′ = f + d·c₀ + e·r₀.

    Args:
        g: The carrier, ``(..., H, W)``.
        offset: ``(row, col)`` of the crop's top-left pixel.
        size: ``(h, w)``, at most ``(H, W)``.

    Raises:
        ValueError: ``size`` exceeds the spatial shape.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> g = GeoArray(np.zeros((2, 8, 8)), (20, 0, 600000, 0, -20, 3550000), 32613)
        >>> patch = crop(g, (2, 4), (4, 4))
        >>> patch.shape, patch.bounds[:2]
        ((2, 4, 4), (600080.0, 3549880.0))
    """
    h, w = (int(v) for v in size)
    height, width = g.height, g.width
    if not (0 < h <= height and 0 < w <= width):
        raise ValueError(
            f"crop size {(h, w)} must fit in the spatial shape {(height, width)}"
        )
    lead = tuple(g.data.shape[:-2])
    traced = is_tracer(offset) or any(is_tracer(v) for v in offset)
    if g.backend == "numpy" and not traced:
        row = min(max(int(offset[0]), 0), height - h)
        col = min(max(int(offset[1]), 0), width - w)
        data = g.data[..., row : row + h, col : col + w]
        return g.with_data(data, transform=_shifted(g, row, col))

    require_x64()
    import jax.numpy as jnp
    from jax import lax

    data = g.data if backend_of(g.data) == "jax" else jnp.asarray(g.data)
    offsets = jnp.asarray(offset)
    row = jnp.clip(offsets[0], 0, height - h)
    col = jnp.clip(offsets[1], 0, width - w)
    starts = (0,) * len(lead) + (row, col)
    data = lax.dynamic_slice(data, starts, (*lead, h, w))
    transform = shift_scale_transform(
        jnp.asarray(g.transform),
        row.astype(jnp.float64),
        col.astype(jnp.float64),
    )
    return g.with_data(data, transform=transform)


def _pad_widths(pad_width: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    if isinstance(pad_width, int | np.integer):
        pad_width = ((pad_width, pad_width), (pad_width, pad_width))
    (top, bottom), (left, right) = pad_width
    widths = (int(top), int(bottom), int(left), int(right))
    if min(widths) < 0:
        raise ValueError(f"pad widths must be >= 0, got {pad_width}")
    return ((widths[0], widths[1]), (widths[2], widths[3]))


def _fill_for(g: GeoArray, fill_value: Any, what: str) -> Any:
    """The value of new pixels, which is also the result's fill value.

    Raises:
        ValueError: There is no fill; ``fill_value`` differs from the
            carrier's own (whose nodata pixels would then read as data); or
            a NaN fill for integer data.
    """
    value = g.fill_value
    if fill_value is not INHERIT:
        value = _normalize_fill(fill_value)
        if g.fill_value is not None and not _same_fill(value, g.fill_value):
            raise ValueError(
                f"{what}: fill_value={fill_value!r} differs from the carrier's "
                f"{g.fill_value!r}, whose nodata pixels would then read as data"
            )
    if value is None:
        raise ValueError(
            f"{what} writes fill pixels, but the carrier has no fill_value"
        )
    if isinstance(value, float) and math.isnan(value) and g.dtype.kind not in "fc":
        raise ValueError(f"{what}: a NaN fill cannot be written into {g.dtype} data")
    return value


def pad(g: GeoArray, pad_width: Any, *, fill_value: Any = INHERIT) -> GeoArray:
    """Add fill pixels around the spatial axes; the transform moves with them.

    Args:
        g: The carrier.
        pad_width: One int for every side, or ``((top, bottom), (left, right))``.
        fill_value: The value of the new pixels, and the result's fill;
            defaults to ``g.fill_value``. A carrier without a fill takes
            this one; a carrier with a different fill raises.

    Raises:
        ValueError: No fill value, or a NaN fill for integer data.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> transform = (20, 0, 600000, 0, -20, 3550000)
        >>> g = GeoArray(np.ones((1, 2, 2)), transform, 32613, fill_value=0.0)
        >>> padded = pad(g, ((1, 0), (2, 0)))
        >>> padded.shape, padded.transform[2], padded.transform[5]
        ((1, 3, 4), np.float64(599960.0), np.float64(3550020.0))
    """
    (top, bottom), (left, right) = _pad_widths(pad_width)
    value = _fill_for(g, fill_value, "pad")
    widths = [(0, 0)] * (g.ndim - 2) + [(top, bottom), (left, right)]
    data = g.xp.pad(g.data, widths, mode="constant", constant_values=value)
    return g.with_data(data, transform=_shifted(g, -top, -left), fill_value=value)


def read_window(
    g: GeoArray,
    window: PixelWindow,
    *,
    boundless: bool = True,
    fill_value: Any = INHERIT,
) -> GeoArray:
    """The pixels of ``window``, padding with fill where it leaves the array.

    The result's fill value is the padding value, so padded pixels read as
    nodata (`GeoArray.valid_mask`).

    Args:
        g: The carrier.
        window: The pixels to read; may extend past the edges.
        boundless: ``True`` returns exactly ``window.shape``, padded with
            fill outside the array. ``False`` returns only the part inside.
        fill_value: The padding value, and the result's fill; defaults to
            ``g.fill_value`` (rules as in `pad`).

    Raises:
        ValueError: ``boundless=False`` and the window misses the array, or
            padding is needed and there is no fill value.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray, PixelWindow
        >>> g = GeoArray(np.ones((4, 4)), (20, 0, 0, 0, -20, 0), 32613, fill_value=0.0)
        >>> window = PixelWindow(col_off=-1, row_off=-1, width=3, height=3)
        >>> read_window(g, window).data
        array([[0., 0., 0.],
               [0., 1., 1.],
               [0., 1., 1.]])
    """
    inside = window.intersection(_whole(g))
    if not boundless:
        if inside is None:
            raise ValueError(
                f"window {window} does not overlap the array {g.shape[-2:]}"
            )
        rows, cols = inside.to_slices()
        return g[..., rows, cols]
    if inside == window:
        rows, cols = window.to_slices()
        return g[..., rows, cols]
    value = _fill_for(g, fill_value, "read_window")
    lead = tuple(g.data.shape[:-2])
    if inside is None:
        data = g.xp.full((*lead, *window.shape), value, dtype=g.dtype)
        transform = _shifted(g, window.row_off, window.col_off)
        return g.with_data(data, transform=transform, fill_value=value)
    rows, cols = inside.to_slices()
    top = inside.row_off - window.row_off
    left = inside.col_off - window.col_off
    bottom = window.height - top - inside.height
    right = window.width - left - inside.width
    return pad(g[..., rows, cols], ((top, bottom), (left, right)), fill_value=value)


def write_window(g: GeoArray, values: Any, window: PixelWindow) -> GeoArray:
    """A copy of ``g`` with ``values`` written into ``window``.

    The part of ``window`` outside the array is ignored, and ``values``
    are cast to the carrier's dtype. JAX values promote a NumPy carrier to
    JAX (C1), so this works inside ``jit``. JAX data is
    immutable, so this always returns a new carrier (the NumPy input is
    not modified either).

    Args:
        g: The carrier.
        values: ``(..., h, w)`` with ``(h, w) == window.shape`` and the
            carrier's leading shape; a GeoArray's data is used.
        window: Where to write.

    Raises:
        ValueError: ``values`` does not match ``window`` and the leading
            shape.
        TypeError: ``values`` would change kind when cast (float into
            integer data, say).

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray, PixelWindow
        >>> g = GeoArray(np.zeros((3, 3)), (20, 0, 0, 0, -20, 0), 32613)
        >>> window = PixelWindow(col_off=2, row_off=2, width=2, height=2)
        >>> write_window(g, np.ones((2, 2)), window).data
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 1.]])
    """
    values = values.data if isinstance(values, GeoArray) else values
    lead = tuple(g.data.shape[:-2])
    if tuple(values.shape) != (*lead, *window.shape):
        raise ValueError(
            f"values of shape {tuple(values.shape)} do not fit window {window.shape} "
            f"with leading shape {lead}"
        )
    inside = window.intersection(_whole(g))
    if inside is None:
        return g
    rows, cols = inside.to_slices()
    v_rows = slice(
        inside.row_off - window.row_off, inside.row_off - window.row_off + inside.height
    )
    v_cols = slice(
        inside.col_off - window.col_off, inside.col_off - window.col_off + inside.width
    )
    part = values[..., v_rows, v_cols]
    if not np.can_cast(part.dtype, g.dtype, casting="same_kind"):
        raise TypeError(
            f"write_window: {part.dtype} values cannot be written into "
            f"{g.dtype} data without changing their kind; cast them first"
        )
    if g.backend == "numpy" and backend_of(part) == "jax":
        g = g.to_jax()
    if g.backend == "numpy":
        data = np.array(g.data, copy=True)
        data[..., rows, cols] = np.asarray(part)
    else:
        data = g.data.at[..., rows, cols].set(g.xp.asarray(part, dtype=g.dtype))
    return g.with_data(data)


def pixel_coords(g: GeoArray, *, center: bool = True) -> Any:
    """Map coordinates of every pixel, ``(2, H, W)``: x then y.

    x_ij = a·(j + ½) + b·(i + ½) + c, y_ij = d·(j + ½) + e·(i + ½) + f
    (drop the ½ with ``center=False`` for the top-left corners). Runs
    inside ``jit``; float64 like the transform.

    Raises:
        ValueError: The carrier is batched; use ``jax.vmap``.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> g = GeoArray(np.zeros((2, 3)), (20, 0, 600000, 0, -20, 3550000), 32613)
        >>> pixel_coords(g)[:, 0, 0]
        array([ 600010., 3549990.])
    """
    if g.batch_shape:
        raise ValueError(
            "pixel_coords is per scene; use jax.vmap for a batched carrier"
        )
    xp = namespace(g.transform)
    offset = 0.5 if center else 0.0
    cols = xp.arange(g.width, dtype=g.transform.dtype) + offset
    rows = xp.arange(g.height, dtype=g.transform.dtype) + offset
    jj, ii = xp.meshgrid(cols, rows)
    return xp.stack(apply_transform(g.transform, jj, ii))


def footprint(g: GeoArray, crs: Any = None, *, densify: int = 21) -> Any:
    """The outline of the carrier's grid as a shapely ``Polygon``.

    Args:
        g: The carrier (concrete, not batched).
        crs: Reproject the outline to this CRS; ``None`` keeps the
            carrier's. Each edge is densified to ``densify`` points first,
            so curved edges stay accurate.
        densify: Points per edge when reprojecting.

    Raises:
        ImportError: shapely is not installed (``geotoolz-array[vector]``).
        ValueError: ``crs`` is given but the carrier has no CRS.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray
        >>> g = GeoArray(np.zeros((2, 2)), (20, 0, 600000, 0, -20, 3550000), 32613)
        >>> footprint(g).bounds
        (600000.0, 3549960.0, 600040.0, 3550000.0)
    """
    try:
        from shapely.geometry import Polygon
    except ImportError as exc:  # pragma: no cover - exercised by the base-install job
        raise ImportError(_SHAPELY_HINT) from exc
    grid = g.grid
    h, w = grid.shape
    if crs is None:
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        return Polygon([apply_transform(grid.transform, j, i) for j, i in corners])
    if grid.crs is None:
        raise ValueError("footprint(crs=...) needs a carrier with a CRS")
    import pyproj

    steps = np.linspace(0.0, 1.0, densify)[:-1]
    ring = (
        [(w * s, 0.0) for s in steps]
        + [(w, h * s) for s in steps]
        + [(w * (1 - s), h) for s in steps]
        + [(0.0, h * (1 - s)) for s in steps]
    )
    xs, ys = apply_transform(grid.transform, *np.array(ring).T)
    to_crs = pyproj.Transformer.from_crs(grid.crs, crs, always_xy=True)
    tx, ty = to_crs.transform(xs, ys)
    return Polygon(list(zip(tx, ty, strict=True)))
