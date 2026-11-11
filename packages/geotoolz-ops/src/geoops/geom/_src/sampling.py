"""H(x): a grid read at points, the observation operator (GA-B6, #631).

`interp_at` reads a `GeoArray` at the positions of a `GeoPoints`: the
observation operator H(x) of data assimilation and of training against
in-situ data. Bilinear weights blend the four pixels around each point, so
it computes new values and lives in geoops; `geoarray.to_pixel`, which maps
positions to pixels, is coordinate maths and stays in geoarray.

The stencil is a functax plan (#643): ``Linear`` or ``Nearest`` on the
grid's ``AffineGrid``, with the policies that are this function's
convention (``outside="nan"``, ``extent="cells"``, ``boundary="edge"``).
The plan carries H and its adjoint; this module adds the nodata rule, the
per-point frames and the `GeoPoints` wrapping. functax plans are JAX, so
this needs the ``[interp]`` extra; a NumPy grid is read on JAX in 64-bit
mode and comes back as NumPy. It runs under ``jit`` / ``grad`` on JAX.
"""

from __future__ import annotations

import sys
import warnings
from types import ModuleType
from typing import Any

from geoarray import SPATIAL_DIMS, GeoArray, GeoPoints
from pipekit_ops import backend_of

from geoops._src.scattered import functax_module, x64


__all__ = ["interp_at"]

_PROMOTION_WARNING = (
    "interp_at: a NumPy grid read at JAX points is moved to JAX, which is "
    "deprecated (it hides a host-to-device copy); call .to_jax() on the grid "
    "first."
)


def is_tracer(x: Any) -> bool:
    """Whether ``x`` is a JAX tracer (JAX is already imported for JAX data)."""
    jax = sys.modules.get("jax")
    return jax is not None and isinstance(x, jax.core.Tracer)


def interp_at(
    grid: GeoArray,
    points: GeoPoints,
    method: str = "bilinear",
    *,
    frame: Any = None,
    along: str = "time",
) -> GeoPoints:
    """H(x): ``grid`` read at the positions of ``points``.

    The read-out is a functax plan on the grid (`functax.interp.Linear` or
    `Nearest`). A point is inside when its fractional pixel position lies in
    [−½, R − ½] × [−½, C − ½] (``extent="cells"``), and taps past the last
    row or column read the edge pixel (``boundary="edge"``), so the half
    pixel at the border takes the edge value. Nearest-neighbour ties (a
    position exactly half way between two pixel centres) go to the lower
    pixel, as functax's ``Nearest`` does. A point outside the grid's
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
        ImportError: functax's plans are missing (the ``[interp]`` extra).

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
    check_args(grid, points, method)
    return interp_with_plan(grid, points, method, frame=frame, along=along)


def check_args(grid: GeoArray, points: GeoPoints, method: str) -> None:
    """Raise unless `interp_at` can read ``grid`` at ``points`` by ``method``."""
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


def interp_with_plan(
    grid: GeoArray,
    points: GeoPoints,
    method: str,
    *,
    frame: Any = None,
    along: str = "time",
    plan: Any = None,
) -> GeoPoints:
    """`interp_at` after its argument checks, reading through ``plan`` if given.

    ``plan`` is `plan_at`'s for this grid and these points (`InterpAt`
    passes its cached one); ``None`` builds it.
    """
    interp = functax_module("interp", feature="geoops.geom.interp_at")
    import jax

    if grid.backend == "numpy" and backend_of(points.xy) == "jax":
        # compute where the positions live; eagerly that hides a host-to-
        # device copy (C1's deprecation), under a trace the grid is a constant
        if not is_tracer(points.xy):
            warnings.warn(_PROMOTION_WARNING, DeprecationWarning, stacklevel=2)
        grid = grid.to_jax()
    if frame is not None and along not in grid.dims:
        raise ValueError(
            f"interp_at: frame indexes {along!r}, not a dim of {grid.dims}"
        )
    if grid.backend == "jax":
        values, observed = _read(interp, grid, points, method, frame, along, plan)
    else:
        # a NumPy grid is read on JAX in 64-bit mode, then comes back
        with x64("geoops.geom.interp_at"):
            values, observed = _read(interp, grid, points, method, frame, along, plan)
            values, observed = jax.device_get((values, observed))
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


def _plan(
    interp: ModuleType, grid: GeoArray, xy: Any, located: Any, method: str
) -> Any:
    """The functax plan H of ``grid`` read at ``xy`` ``(obs, 2)``: interp_at's.

    ``Linear`` (bilinear) or ``Nearest`` on the grid's ``AffineGrid``, with
    ``outside="nan"`` (rows off the grid invalid, zero weights),
    ``extent="cells"`` (the domain is the pixels' union, half a pixel past
    the edge centres) and ``boundary="edge"`` (taps past the edge read it).
    Built from the carrier's transform leaf, so it traces under ``jit``.
    """
    import jax.numpy as jnp
    from functax.geometry import AffineGrid, Points

    scheme = interp.Linear() if method == "bilinear" else interp.Nearest()
    policies = interp.Policies(
        outside="nan", missing="propagate", boundary="edge", extent="cells"
    )
    source = AffineGrid(jnp.asarray(grid.transform), (grid.height, grid.width))
    return scheme.plan(source, Points(xy, valid=located), policies=policies)


def _located(points: GeoPoints) -> tuple[Any, Any, Any]:
    """``given``, ``located`` and the positions to plan at, on JAX.

    A padded point (``valid=False``) or a non-finite position is not
    located, and is planned at the origin so nothing NaN enters the plan.
    """
    import jax.numpy as jnp

    xy = jnp.asarray(points.xy)
    given = (
        jnp.asarray(points.valid)
        if points.valid is not None
        else jnp.ones(points.size, dtype=bool)
    )
    # a missing position is not an observation at some other place
    located = given & jnp.all(jnp.isfinite(xy), axis=-1)
    # padded or missing positions are read at the origin, and masked after
    safe_xy = jnp.where(located[:, None], xy, 0.0)
    return given, located, safe_xy


def plan_at(interp: ModuleType, grid: GeoArray, points: GeoPoints, method: str) -> Any:
    """The functax plan H that `interp_at` reads ``grid`` at ``points`` with.

    Runs on JAX: a NumPy grid's caller enters 64-bit mode first.
    """
    _, located, safe_xy = _located(points)
    return _plan(interp, grid, safe_xy, located, method)


def _read(
    interp: ModuleType,
    grid: GeoArray,
    points: GeoPoints,
    method: str,
    frame: Any,
    along: str,
    plan: Any = None,
) -> tuple[Any, Any]:
    """The values ``(..., obs)`` and ``valid`` ``(obs,)`` of H(x), on JAX."""
    import jax.numpy as jnp

    given, located, safe_xy = _located(points)
    if plan is None:
        plan = _plan(interp, grid, safe_xy, located, method)
    stencil = plan.read
    data = jnp.asarray(grid.data)
    if frame is not None:
        data = jnp.moveaxis(data, grid.dims.index(along), 0)
    # y: the grid flattened row-major, (..., y, x) -> (..., N)
    flat = data.reshape(*data.shape[:-2], grid.height * grid.width)
    fill = grid.fill_value
    # samples that are data: finite and not the fill
    usable = (
        jnp.isfinite(flat)
        if flat.dtype.kind in "fc"
        else jnp.ones(flat.shape, dtype=bool)
    )
    if fill is not None and fill == fill:
        usable = usable & (flat != fill)
    # nodata is zeroed, so 0 · NaN never spreads; it is masked below
    clean = jnp.where(usable, flat, jnp.zeros((), flat.dtype))
    if frame is None:
        # f̂ = H y over the leading dims, and the usable flag of every tap
        values = stencil.apply(clean)
        taps_usable = usable[..., stencil.indices]
    else:
        # each point reads its own frame: y[frame[m], ..., i_ms], (..., M, S)
        rows = jnp.asarray(frame)[:, None]
        gathered = jnp.moveaxis(clean[rows, ..., stencil.indices], (0, 1), (-2, -1))
        taps_usable = jnp.moveaxis(usable[rows, ..., stencil.indices], (0, 1), (-2, -1))
        # f̂ₘ = Σₛ wₘₛ y[frame[m], iₘₛ]
        values = jnp.einsum("...ms,ms->...m", gathered, stencil.weights)
    if method == "nearest" and flat.dtype.kind in "fc":
        # one tap of weight 1: the pixel's own value, in the grid's dtype
        values = values.astype(flat.dtype)
    # a nodata tap with weight > 0 leaves the point unsupported; with
    # weight 0 it is ignored
    supported = jnp.all(taps_usable | (stencil.weights == 0), axis=-1)
    # a point is data when every band it reads is (valid_mask's rule)
    lead_axes = tuple(range(supported.ndim - 1))
    supported = jnp.all(supported, axis=lead_axes) if lead_axes else supported
    observed = located & stencil.valid & supported
    floating = values if values.dtype.kind in "fc" else values.astype(float)
    nan = jnp.asarray(float("nan"), dtype=floating.dtype)
    values = jnp.where(observed, floating, nan)
    values = jnp.where(given, values, jnp.zeros_like(values))
    return values, observed
