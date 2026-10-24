"""Map coordinates to pixels, and a grid sampled at points: H(x) (#522).

`to_pixel` inverts the transform; `interp_at` reads a `GeoArray` at the
positions of a `GeoPoints`, which is the observation operator H(x) of data
assimilation and of training against in-situ data. Both are written once
against the array namespace, so they run on NumPy and under ``jit`` /
``grad`` on JAX.

``interp_at`` returns input values only with ``method="nearest"``; bilinear
weights blend four pixels, so strictly it computes new values (the
membership test of geoarray). It lives here because it is the one
interpolation every carrier-level cost needs, and it reads nothing but the
georeference (``geoops.geom`` holds the general resamplers).
"""

from __future__ import annotations

from typing import Any

from geoarray._src.array import SPATIAL_DIMS, GeoArray
from geoarray._src.backend import namespace
from geoarray._src.carriers import GeoPoints


__all__ = ["interp_at", "to_pixel"]


def to_pixel(transform: Any, xy: Any) -> Any:
    """Map coordinates ``(x, y)`` to fractional pixel ``(row, col)`` centres.

    With T = (a, b, c, d, e, f), column i and row j map to x = a·i + b·j + c,
    y = d·i + e·j + f. Inverting the 2 × 2 part,

        (i, j)ᵀ = [[a, b], [d, e]]⁻¹ (x − c, y − f)ᵀ,

    and subtracting ½ from both puts integer values at pixel centres (area
    convention), so ``(0, 0)`` is the centre of the top-left pixel.

    Args:
        transform: ``(6,)`` affine transform (NumPy or JAX).
        xy: ``(N, 2)`` positions in the transform's CRS.

    Returns:
        ``(N, 2)`` fractional ``(row, col)``.

    Example:
        >>> import numpy as np
        >>> t = np.array([20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0])
        >>> to_pixel(t, np.array([[600010.0, 3549990.0]])).tolist()
        [[0.0, 0.0]]
    """
    xp = namespace(xy)
    a, b, c, d, e, f = (transform[k] for k in range(6))
    det = a * e - b * d
    dx = xy[..., 0] - c
    dy = xy[..., 1] - f
    # [[a, b], [d, e]]⁻¹ = [[e, −b], [−d, a]] / det
    col = (e * dx - b * dy) / det - 0.5
    row = (-d * dx + a * dy) / det - 0.5
    return xp.stack([row, col], axis=-1)


def interp_at(
    grid: GeoArray,
    points: GeoPoints,
    method: str = "bilinear",
    *,
    frame: Any = None,
    along: str = "time",
) -> GeoPoints:
    """H(x): ``grid`` read at the positions of ``points``.

    Bilinear sampling clips the neighbour indices to [0, R − 2] × [0, C − 2]
    (exact on the last row and column) and the weights to [0, 1] (the half
    pixel at the border takes the edge value). A point outside the grid's
    extent gives NaN and ``valid=False``; a padded point (``valid=False``
    on ``points``) gives 0 and stays invalid, so a masked cost and its
    gradient are finite and unchanged by the padding (GA-G16).

    Args:
        grid: A carrier ``(..., y, x)``; its leading dims stay, before ``obs``.
        points: Where to read, in ``grid``'s CRS.
        method: ``"bilinear"`` or ``"nearest"``.
        frame: Optional ``(obs,)`` integer index into ``along`` per point:
            each observation reads its own frame of a series (tracks over a
            ``GeoMapSeries``); ``along`` is then dropped from the result.
        along: The dim ``frame`` indexes.

    Returns:
        A `GeoPoints` with ``points``' leaves, ``valid`` combining the
        points' validity and being inside the grid, and the grid's leading
        dims and coordinates: a ``(time, variable, y, x)`` stack sampled at
        stations is a ``GeoTimeSeries``.

    Raises:
        ValueError: The CRSs differ, the grid is batched, or ``method`` is
            unknown.

    Example:
        >>> import numpy as np
        >>> from geoarray import GeoArray, GeoPoints
        >>> t = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)
        >>> g = GeoArray(np.arange(16.0).reshape(4, 4), t, None, dims="y x")
        >>> p = GeoPoints(np.zeros(2), xy=np.array([[0.5, 3.5], [9.0, 9.0]]),
        ...               dims="obs", crs=None)
        >>> out = interp_at(g, p)
        >>> out.data.tolist(), out.valid.tolist()
        ([0.0, nan], [True, False])
    """
    if grid.crs != points.crs:
        raise ValueError(
            f"interp_at: the grid is in {grid.crs} and the points in {points.crs}; "
            "transform the points first"
        )
    if grid.batch_shape:
        raise ValueError(
            "interp_at reads one grid; map a batched carrier with geoarray.map_over"
        )
    if method not in ("bilinear", "nearest"):
        raise ValueError(
            f"interp_at: method is 'bilinear' or 'nearest', got {method!r}"
        )
    xp = grid.xp
    xy = xp.asarray(points.xy)
    given = (
        xp.asarray(points.valid)
        if points.valid is not None
        else xp.ones(points.size, dtype=bool)
    )
    # padded positions may be NaN: read them at the origin, and mask after
    safe_xy = xp.where(given[:, None] & xp.isfinite(xy), xy, 0.0)
    rc = to_pixel(xp.asarray(grid.transform), safe_xy)
    r, c = rc[:, 0], rc[:, 1]
    height, width = grid.height, grid.width
    inside = (r >= -0.5) & (r <= height - 0.5) & (c >= -0.5) & (c <= width - 0.5)
    data = grid.data
    if frame is not None:
        if along not in grid.dims:
            raise ValueError(
                f"interp_at: frame indexes {along!r}, not a dim of {grid.dims}"
            )
        data = xp.moveaxis(data, grid.dims.index(along), 0)
    if method == "nearest":
        ri = xp.clip(xp.round(r), 0, height - 1).astype(int)
        ci = xp.clip(xp.round(c), 0, width - 1).astype(int)
        values = _gather(data, ri, ci, frame, xp)
    else:
        r0 = xp.clip(xp.floor(r), 0, max(height - 2, 0)).astype(int)
        c0 = xp.clip(xp.floor(c), 0, max(width - 2, 0)).astype(int)
        r1 = xp.minimum(r0 + 1, height - 1)
        c1 = xp.minimum(c0 + 1, width - 1)
        # w_r = r − r₀, w_c = c − c₀, clipped to [0, 1] at the border
        wr = xp.clip(r - r0, 0.0, 1.0)
        wc = xp.clip(c - c0, 0.0, 1.0)
        v00 = _gather(data, r0, c0, frame, xp)
        v01 = _gather(data, r0, c1, frame, xp)
        v10 = _gather(data, r1, c0, frame, xp)
        v11 = _gather(data, r1, c1, frame, xp)
        # (1 − w_r)(1 − w_c) v₀₀ + (1 − w_r) w_c v₀₁ + w_r (1 − w_c) v₁₀ + w_r w_c v₁₁
        values = (1 - wr) * ((1 - wc) * v00 + wc * v01) + wr * (
            (1 - wc) * v10 + wc * v11
        )
    floating = values if values.dtype.kind in "fc" else values.astype(float)
    nan = xp.asarray(float("nan"), dtype=floating.dtype)
    values = xp.where(inside, floating, nan)
    values = xp.where(given, values, xp.zeros_like(values))
    leading = tuple(
        d
        for d in grid.dims
        if d not in SPATIAL_DIMS and not (frame is not None and d == along)
    )
    coords = tuple(c for c in grid.coords if c.dim in leading)
    return GeoPoints(
        values,
        points.xy,
        z=points.z,
        vertical=points.vertical,
        time=points.time,
        valid=given & inside,
        dims=(*leading, "obs"),
        coords=coords,
        crs=points.crs,
        fill_value=float("nan"),
        attrs=grid.attrs,
    )


def _gather(data: Any, rows: Any, cols: Any, frame: Any, xp: Any) -> Any:
    """``data[..., rows[k], cols[k]]`` for each point k, as ``(..., obs)``."""
    if frame is None:
        return data[..., rows, cols]
    # data is (frame, ..., y, x): each point reads its own frame
    picked = data[xp.asarray(frame), ..., rows, cols]
    return xp.moveaxis(picked, 0, -1)
