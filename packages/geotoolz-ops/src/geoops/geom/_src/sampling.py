"""H(x): a grid read at points, the observation operator (GA-B6, #631).

`interp_at` reads a `GeoArray` at the positions of a `GeoPoints`: the
observation operator H(x) of data assimilation and of training against
in-situ data. Bilinear weights blend the four pixels around each point, so
it computes new values and lives in geoops; `geoarray.to_pixel`, which maps
positions to pixels, is coordinate maths and stays in geoarray. Written once
against the array namespace, it runs on NumPy and under ``jit`` / ``grad``
on JAX.
"""

from __future__ import annotations

import warnings
from typing import Any

from geoarray import SPATIAL_DIMS, GeoArray, GeoPoints, to_pixel
from pipekit import backend_of


__all__ = ["interp_at"]

_PROMOTION_WARNING = (
    "interp_at: a NumPy grid read at JAX points is moved to JAX, which is "
    "deprecated (it hides a host-to-device copy); call .to_jax() on the grid "
    "first."
)


def _is_tracer(x: Any) -> bool:
    """Whether ``x`` is a JAX tracer (JAX is already imported for JAX data)."""
    import jax

    return isinstance(x, jax.core.Tracer)


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
    extent, at a non-finite position, or reading a nodata pixel (non-finite
    or the grid's fill, in any band, with a non-zero weight) gives NaN and
    ``valid=False``; nodata never blends into a value. A padded point
    (``valid=False`` on ``points``) gives 0 and stays invalid, so a masked
    cost and its gradient are finite and unchanged by the padding (GA-G16).
    JAX points read a NumPy grid on JAX (a constant under ``jit``), so the
    values can be differentiated with respect to the positions.

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
        >>> from geoops.geom import interp_at
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
    if grid.backend == "numpy" and backend_of(points.xy) == "jax":
        # compute where the positions live; eagerly that hides a host-to-
        # device copy (C1's deprecation), under a trace the grid is a constant
        if not _is_tracer(points.xy):
            warnings.warn(_PROMOTION_WARNING, DeprecationWarning, stacklevel=2)
        grid = grid.to_jax()
    xp = grid.xp
    xy = xp.asarray(points.xy)
    given = (
        xp.asarray(points.valid)
        if points.valid is not None
        else xp.ones(points.size, dtype=bool)
    )
    # a missing position is not an observation at some other place
    located = given & xp.all(xp.isfinite(xy), axis=-1)
    # padded or missing positions are read at the origin, and masked after
    safe_xy = xp.where(located[:, None], xy, 0.0)
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
    fill = grid.fill_value

    def usable(v: Any) -> Any:
        """Neighbour values that are data: finite and not the fill."""
        ok = xp.isfinite(v) if v.dtype.kind in "fc" else xp.ones(v.shape, dtype=bool)
        if fill is not None and fill == fill:
            ok = ok & (v != fill)
        return ok

    if method == "nearest":
        ri = xp.clip(xp.round(r), 0, height - 1).astype(int)
        ci = xp.clip(xp.round(c), 0, width - 1).astype(int)
        values = _gather(data, ri, ci, frame, xp)
        supported = usable(values)
        values = xp.where(supported, values, xp.zeros_like(values))
    else:
        r0 = xp.clip(xp.floor(r), 0, max(height - 2, 0)).astype(int)
        c0 = xp.clip(xp.floor(c), 0, max(width - 2, 0)).astype(int)
        r1 = xp.minimum(r0 + 1, height - 1)
        c1 = xp.minimum(c0 + 1, width - 1)
        # w_r = r − r₀, w_c = c − c₀, clipped to [0, 1] at the border
        wr = xp.clip(r - r0, 0.0, 1.0)
        wc = xp.clip(c - c0, 0.0, 1.0)
        corners = [
            (_gather(data, ri, ci, frame, xp), w)
            for ri, ci, w in (
                (r0, c0, (1 - wr) * (1 - wc)),
                (r0, c1, (1 - wr) * wc),
                (r1, c0, wr * (1 - wc)),
                (r1, c1, wr * wc),
            )
        ]
        # a nodata neighbour with weight > 0 leaves the point unsupported;
        # with weight 0 it is ignored (and zeroed, so 0 · NaN never spreads)
        supported = xp.ones(corners[0][0].shape, dtype=bool)
        for v, w in corners:
            supported = supported & (usable(v) | (w == 0))
        # Σₖ wₖ vₖ: (1 − w_r)(1 − w_c) v₀₀ + (1 − w_r) w_c v₀₁ + w_r (1 − w_c) v₁₀ + …
        values = sum(w * xp.where(usable(v), v, xp.zeros_like(v)) for v, w in corners)
    # a point is data when every band it reads is (valid_mask's rule)
    lead_axes = tuple(range(supported.ndim - 1))
    supported = xp.all(supported, axis=lead_axes) if lead_axes else supported
    observed = located & inside & supported
    floating = values if values.dtype.kind in "fc" else values.astype(float)
    nan = xp.asarray(float("nan"), dtype=floating.dtype)
    values = xp.where(observed, floating, nan)
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
        valid=observed,
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
