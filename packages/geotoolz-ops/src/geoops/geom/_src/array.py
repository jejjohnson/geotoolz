"""Tier-A primitives for geometry operators — pure-numpy helpers.

The carrier-aware Operator wrappers in
:mod:`geoops.geom._src.operators` lift these into the ``GeoArray``
pipeline. Each helper here is deliberately framework-agnostic: it
operates on ``numpy.ndarray`` (and tuples of plain ints / floats) so the
same primitive is callable from notebooks and tests without constructing
a ``GeoArray``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np
from affine import Affine
from jaxtyping import Bool, Float, Shaped
from pipekit_array import array_namespace

from geoops._src.blending import triangular_weights
from geoops._src.geotensor import reject_geotensor
from geoops._src.optional import optional_name
from geoops.geom._src.stencils import (
    METHODS,
    Method,
    difference_coeffs,
    difference_offsets,
    nonuniform_coeffs,
)


if TYPE_CHECKING:
    from rasterio.enums import Resampling


# Aliases between the user-friendly resampling names and the
# rasterio / skimage naming conventions used downstream.
_RASTERIO_ALIASES: dict[str, str] = {"bicubic": "cubic", "linear": "bilinear"}
_SKIMAGE_ALIASES: dict[str, str] = {
    "cubic": "bicubic",
    "cubic_spline": "bicubic",
    "average": "bilinear",
}


def resolve_resampling(name: str | Resampling) -> Resampling:
    """Translate a string or `Resampling` to the canonical `rasterio` enum.

    Accepts the user-friendly aliases ``"linear"`` (= ``"bilinear"``) and
    ``"bicubic"`` (= ``"cubic"``) in addition to all
    :class:`rasterio.enums.Resampling` members.

    Args:
        name: Either a `Resampling` enum value or a string alias.

    Returns:
        The matching :class:`rasterio.enums.Resampling` enum member.
    """
    Resampling = optional_name(
        "rasterio.enums",
        "Resampling",
        "raster",
        feature="geoops.geom",
    )
    if isinstance(name, Resampling):
        return name
    return Resampling[_RASTERIO_ALIASES.get(name, name)]


def resolve_interpolation(name: str) -> str:
    """Translate a resampling name into a ``skimage.transform.resize`` mode.

    ``GeoArray.resize`` delegates to ``skimage.transform.resize`` whose
    ``interpolation`` parameter uses different names than rasterio
    (``"bicubic"`` instead of ``"cubic"``, etc.). This helper keeps the
    geom operators idiomatic from the user's side while passing the
    correct names downstream.

    Args:
        name: Resampling alias.

    Returns:
        The matching ``skimage`` interpolation name.
    """
    return _SKIMAGE_ALIASES.get(name, name)


def center_offsets(
    current: tuple[int, int], target: tuple[int, int]
) -> tuple[int, int]:
    """Pixel offsets to center-anchor a ``target`` shape inside ``current``.

    Returns ``((current_h - target_h) // 2, (current_w - target_w) // 2)``,
    the row/col offsets used by `CropTo(anchor="center")`.

    Args:
        current: Source spatial shape ``(H, W)``.
        target: Target spatial shape ``(H', W')``.

    Returns:
        ``(row_off, col_off)`` integer pair.
    """
    return ((current[0] - target[0]) // 2, (current[1] - target[1]) // 2)


def is_north_up(transform: Affine) -> bool:
    """Return ``True`` iff the affine transform is axis-aligned, north-up.

    Equivalent to ``b == 0 and d == 0 and a > 0 and e < 0`` on
    :class:`affine.Affine` — no rotation or shear, columns increasing
    eastward and rows increasing southward. North-up transforms are the
    only ones that admit pixel-aligned mosaicking with the `Stitch`
    operator (a south-up ``e > 0`` grid would be placed upside down).

    Args:
        transform: An :class:`affine.Affine` transform.

    Returns:
        ``True`` if the transform is axis-aligned and north-up.
    """
    return transform.b == 0 and transform.d == 0 and transform.a > 0 and transform.e < 0


def feather_weights(shape: tuple[int, int], width: int) -> Float[np.ndarray, "h w"]:
    r"""Edge-feathered weight kernel for tile blending.

    Builds a 2-D weight array where each pixel's weight is the minimum
    of:

    .. math::

        w(i, j) \;=\; \min\!\left(
            1,\,
            \frac{\min(i + 1,\,H - i)}{W_f},\,
            \frac{\min(j + 1,\,W - j)}{W_f}
        \right)

    so the centre of the tile is at full weight ``1`` and a band of
    ``width`` pixels along each edge ramps linearly down to
    ``1 / W_f``. Used by ``Stitch(blend="feather")``.

    Args:
        shape: Tile spatial shape ``(H, W)``.
        width: Feather band width in pixels. ``<= 0`` returns
            ``np.ones(shape)``.

    Returns:
        ``float32`` array of shape ``(H, W)`` with values in ``[0, 1]``.
    """
    return triangular_weights(shape, width)


def target_slices(
    tile_transform: Affine,
    tile_shape: tuple[int, int],
    target_transform: Affine,
    target_shape: tuple[int, int],
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Project a tile onto a target grid as ``(out_slice, tile_slice)``.

    Computes the pixel-space slices into ``target`` and ``tile`` arrays
    that overlap. Returns empty slices when the tile lies fully outside
    the target. Both transforms are assumed north-up (call
    :func:`is_north_up` first).

    Args:
        tile_transform: Affine of the tile.
        tile_shape: Tile spatial shape ``(h, w)``.
        target_transform: Affine of the target mosaic.
        target_shape: Target spatial shape ``(H, W)``.

    Returns:
        ``((row_slice, col_slice), (tile_row_slice, tile_col_slice))``.
        When the tile is fully outside the target grid (e.g. when
        ``target_shape``/``target_transform`` describe a region smaller
        than the union of tile bounds), all four slices are empty
        ``slice(0, 0)`` so downstream blending becomes a no-op.
    """
    row = round((tile_transform.f - target_transform.f) / target_transform.e)
    col = round((tile_transform.c - target_transform.c) / target_transform.a)
    height, width = tile_shape
    target_h, target_w = target_shape
    # Clamp the output window to ``[0, target_*]`` so we never produce
    # negative starts/stops (which numpy would silently interpret as
    # "count from the end" and wrap around to invalid memory).
    row0 = min(max(row, 0), target_h)
    col0 = min(max(col, 0), target_w)
    row1 = min(max(row + height, 0), target_h)
    col1 = min(max(col + width, 0), target_w)
    # Empty intersection: return empty slices for both output and tile.
    if row1 <= row0 or col1 <= col0:
        empty = slice(0, 0)
        return ((empty, empty), (empty, empty))
    tile_row0 = row0 - row
    tile_col0 = col0 - col
    tile_row1 = tile_row0 + (row1 - row0)
    tile_col1 = tile_col0 + (col1 - col0)
    # Clamp the tile window too, defensively, so a bad caller cannot
    # index past the tile's own array bounds.
    tile_row0 = min(max(tile_row0, 0), height)
    tile_col0 = min(max(tile_col0, 0), width)
    tile_row1 = min(max(tile_row1, 0), height)
    tile_col1 = min(max(tile_col1, 0), width)
    return (
        (slice(row0, row1), slice(col0, col1)),
        (slice(tile_row0, tile_row1), slice(tile_col0, tile_col1)),
    )


def valid_pixel_mask(
    tile: Shaped[np.ndarray, "*batch h w"], fill: float | int | None
) -> Bool[np.ndarray, "h w"]:
    """Boolean mask of pixels not equal to ``fill``, collapsed over band axis.

    Tiles produced by ``Tile(include_incomplete=True, boundless=True)``
    pad the right/bottom edges with the carrier's ``fill_value``;
    blending must not aggregate those sentinel pixels into the output.

    Args:
        tile: Numpy view of the tile values, shape ``(..., H, W)``.
        fill: Sentinel value. ``None`` short-circuits to all-True.

    Returns:
        Boolean ``(H, W)`` mask where ``True`` means "real data".
    """
    if fill is None:
        return np.ones(tile.shape[-2:], dtype=bool)
    if np.issubdtype(tile.dtype, np.floating) and np.isnan(fill):
        per_band = ~np.isnan(tile)
    else:
        per_band = tile != fill
    if per_band.ndim == 2:
        return per_band
    return per_band.reshape(-1, *per_band.shape[-2:]).any(axis=0)


# ----------------------------------------------------------------------------
# Whiskbroom scan geometry (bowtie) — used by ``BowtieCorrection``
# ----------------------------------------------------------------------------


def scan_angles(
    n_columns: int,
    max_scan_angle_deg: float,
    aggregation_zones: tuple[tuple[int, int], ...] | None = None,
) -> Float[np.ndarray, " w"]:
    """Scan angle θ (radians, signed, nadir = 0) of every swath column.

    The scan mirror sweeps at a constant rate, so the *native* samples are
    uniform in angle. ``max_scan_angle_deg`` is the outer edge of the
    outermost native sample, so with ``n`` native samples per half-scan the
    angular sampling interval is ``Δθ = θ_max / n`` and native sample ``i``
    (counted outward from nadir) is centred at ``θᵢ = (i + ½)·Δθ``.

    * Without ``aggregation_zones`` every column is one native sample
      (MODIS): ``θ_c = θ_max · (2c + 1 − W) / W``; an odd ``W`` has a
      nadir column at exactly ``θ = 0``.
    * With ``aggregation_zones`` (VIIRS) each half-scan is a sequence of
      ``(native_samples, factor)`` zones from nadir outward; every output
      column averages ``factor`` consecutive native samples, so its angle
      is the mean of their centres. The two halves mirror each other.

    Args:
        n_columns: Swath width ``W`` (columns).
        max_scan_angle_deg: Scan-edge angle ``θ_max`` in degrees.
        aggregation_zones: Optional on-board aggregation table, e.g.
            ``((1776, 3), (736, 2), (640, 1))`` for VIIRS M-bands.

    Returns:
        ``(W,)`` scan angles in radians, increasing with column index.

    Raises:
        ValueError: If ``n_columns`` does not match the aggregation table,
            or a zone's sample count is not a multiple of its factor.

    Examples:
        >>> np.rad2deg(scan_angles(3, 60.0)).round(6).tolist()
        [-40.0, 0.0, 40.0]
        >>> scan_angles(4, 10.0, ((4, 2),)).shape
        (4,)
    """
    theta_max = np.deg2rad(max_scan_angle_deg)
    if aggregation_zones is None:
        cols = np.arange(n_columns, dtype=np.float64)
        return theta_max * (2.0 * cols + 1.0 - n_columns) / n_columns
    for native, factor in aggregation_zones:
        if factor < 1 or native < 1 or native % factor:
            raise ValueError(
                f"aggregation zone ({native}, {factor}) must hold a positive, "
                "whole number of aggregated samples"
            )
    n_native = sum(native for native, _ in aggregation_zones)
    step = theta_max / n_native  # Δθ = θ_max / n
    half: list[np.ndarray] = []
    start = 0
    for native, factor in aggregation_zones:
        centres = (start + np.arange(native, dtype=np.float64) + 0.5) * step
        half.append(centres.reshape(-1, factor).mean(axis=1))
        start += native
    right = np.concatenate(half)
    if 2 * right.size != n_columns:
        raise ValueError(
            f"aggregation_zones describe {2 * right.size} columns, got {n_columns}"
        )
    return np.concatenate([-right[::-1], right])


def scan_pixel_growth(
    scan_angle: Float[np.ndarray, "*shape"] | float,
    altitude_km: float,
    earth_radius_km: float,
) -> tuple[
    Float[np.ndarray, "*shape"],
    Float[np.ndarray, "*shape"],
    Float[np.ndarray, "*shape"],
]:
    """Pixel-footprint growth of a scanner over a spherical Earth.

    For a sensor at altitude ``h`` over a sphere of radius ``R`` looking at
    scan angle θ (off nadir, in the cross-track scan plane):

    * Earth central angle between nadir and the ground point
      ``β = arcsin(((R + h)/R)·sin θ) − θ``;
    * slant range ``s = (R + h)·cos θ − √(R² − (R + h)²·sin² θ)``;
    * a detector tilted by a small angle δφ out of the scan plane moves
      the ground point along track by ``s·δφ`` (the along-track direction
      is tangent to the sphere and perpendicular to the line of sight), so
      the along-track *ground* size grows as ``s / h``;
    * the orbit advances by the same orbit angle ψ at every scan angle, and
      the ground point sits on a small circle of radius ``R·cos β`` about
      the orbit axis, so in orbit angle a detector subtends
      ``δψ = s·δφ / (R·cos β)``; relative to nadir (``δψ₀ = h·δφ / R``)
      the along-track growth *in orbit-angle units* is
      ``g(θ) = s / (h·cos β)``;
    * cross-track ground size grows as ``R·(dβ/dθ) / h`` with
      ``dβ/dθ = ((R + h)/R)·cos θ / √(1 − ((R + h)/R)²·sin² θ) − 1``
      (equivalently ``s / (h·cos(θ + β))``).

    The flat-Earth limit ``R → ∞`` gives ``s/h = g = 1/cos θ`` along track
    and ``1/cos² θ`` across. For MODIS (h = 705 km, θ = 55°) this yields
    the published ≈ 2.0 km × 4.8 km edge-of-scan pixel (1 km at nadir).

    Args:
        scan_angle: Scan angle(s) θ in radians.
        altitude_km: Orbit altitude ``h`` (km).
        earth_radius_km: Earth radius ``R`` (km).

    Returns:
        ``(along_ground, along_orbit, cross_ground)`` growth factors, each
        1 at nadir: ``s/h``, ``g = s/(h·cos β)`` and ``R·(dβ/dθ)/h``.

    Raises:
        ValueError: If any |θ| reaches the horizon, ``sin θ ≥ R/(R + h)``.

    Examples:
        >>> along, orbit, cross = scan_pixel_growth(0.0, 705.0, 6371.0)
        >>> round(float(along), 9), round(float(orbit), 9), round(float(cross), 9)
        (1.0, 1.0, 1.0)
        >>> round(float(scan_pixel_growth(np.deg2rad(55.0), 705.0, 6371.0)[0]), 2)
        2.01
    """
    theta = np.abs(np.asarray(scan_angle, dtype=np.float64))
    h = float(altitude_km)
    radius = float(earth_radius_km)
    k = (radius + h) / radius
    k_sin = k * np.sin(theta)
    if np.any(k_sin >= 1.0):
        raise ValueError(
            f"scan angle reaches the horizon: need sin θ < R / (R + h) = {1.0 / k:.6f}"
        )
    beta = np.arcsin(k_sin) - theta  # β = arcsin(k·sin θ) − θ
    root = np.sqrt(1.0 - k_sin**2)
    # s = (R + h)·cos θ − √(R² − (R + h)²·sin² θ)
    slant = (radius + h) * np.cos(theta) - radius * root
    along_ground = slant / h
    along_orbit = along_ground / np.cos(beta)  # g = s / (h·cos β)
    cross_ground = radius * (k * np.cos(theta) / root - 1.0) / h
    return along_ground, along_orbit, cross_ground


def bowtie_detector_positions(
    along_orbit_growth: Float[np.ndarray, " w"], detectors_per_scan: int
) -> tuple[Float[np.ndarray, "n w"], Float[np.ndarray, "n w"], np.ndarray]:
    """Fractional detector index feeding each de-overlapped row of a scan.

    Along-track positions are measured in orbit angle, in units of one
    nadir detector (``δψ₀``); scans are contiguous at nadir, so scan ``j``
    advances by ``N`` units and its centre is at ``cⱼ = j·N + (N − 1)/2``.
    Detector ``k`` of scan ``j`` at a column with growth ``g`` (see
    :func:`scan_pixel_growth`) sits at

        ``y(j, k) = cⱼ + (k − (N − 1)/2)·g``

    so a scan covers ``N·g ≥ N`` units and neighbouring scans overlap by
    ``N·(g − 1)``. The de-overlapped grid places output row ``r = j·N + m``
    at ``y = r`` (the nadir grid). The scan whose nadir rows contain ``r``
    — scan ``j``, the one whose centre is nearest — is the primary source,
    at detector

        ``k₀(m) = (N − 1)/2 + (m − (N − 1)/2) / g``;

    the adjacent scan ``j ± 1`` on the side of ``r`` (``+1`` when
    ``m > (N − 1)/2``) also covers ``r`` when it overlaps, at

        ``k₁(m) = (N − 1)/2 + (m − (N − 1)/2 ∓ N) / g``,

    and serves as the fallback when the primary sample is nodata (e.g.
    VIIRS bowtie-deleted rows). A position is inside a scan's footprint
    when ``−½ ≤ k ≤ N − ½``.

    Args:
        along_orbit_growth: ``(W,)`` growth ``g ≥ 1`` per column.
        detectors_per_scan: Detectors per scan ``N``.

    Returns:
        ``(k0, k1, step)``: ``(N, W)`` primary and fallback detector
        positions and the ``(N,)`` scan offset (``±1``) of the fallback.

    Examples:
        >>> k0, k1, step = bowtie_detector_positions(np.array([1.0, 2.0]), 4)
        >>> k0[:, 0].tolist(), k0[:, 1].tolist()
        ([0.0, 1.0, 2.0, 3.0], [0.75, 1.25, 1.75, 2.25])
        >>> step.tolist()
        [-1, -1, 1, 1]
    """
    n = int(detectors_per_scan)
    centre = (n - 1) / 2.0
    offset = np.arange(n, dtype=np.float64) - centre  # m − (N − 1)/2
    step = np.where(offset > 0, 1, -1)
    g = np.asarray(along_orbit_growth, dtype=np.float64)[None, :]
    k0 = centre + offset[:, None] / g
    k1 = centre + (offset - step * n)[:, None] / g
    return k0, k1, step


# ---------------------------------------------------------------------------
# Finite differences (L1). The stencils come from `stencils.py` (L0); the
# Gradient operator in `operators.py` (L2) derives the geometry from a carrier.
# ---------------------------------------------------------------------------


def difference(
    f: Float[np.ndarray, "*shape"],
    *,
    axis: int = -1,
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    coords: Float[np.ndarray, " n"] | None = None,
    derivative: int = 1,
    accuracy: int = 1,
    method: Method = "central",
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    r"""Finite-difference derivative along one axis; a NaN propagates.

    The plain stencil, finitediffx's ``difference`` on any backend: with
    ``method="central"`` the centred stencil of order ``accuracy + 1`` inside
    and forward / backward stencils of order ``accuracy`` on the edges,
    nothing wrapping around (``accuracy=1`` is ``numpy.gradient``'s
    scheme). ``"forward"`` / ``"backward"`` use that stencil everywhere it
    fits and the other one-sided stencil where it does not.

    A stencil that reads a NaN returns NaN, so a masked cell costs a halo of
    about ``accuracy`` cells; :func:`adaptive_difference` recovers them.

    .. math::

        \partial^n_s f(x_i) \approx h^{-n} \sum_k c_k\, f(x_{i + s_k})

    Args:
        f: The field; any shape.
        axis: The axis to differentiate.
        step: The signed step :math:`h` between centres along ``axis``: a
            scalar, or an array broadcasting against ``f``.
        coords: The positions of the centres along ``axis``, one per entry,
            for an unevenly spaced axis (depth or pressure levels, irregular
            times); replaces ``step``. Each point's stencil is then solved for
            its actual distances (`nonuniform_coeffs`), exact for polynomials
            of the stencil's degree.
        derivative: The derivative order :math:`n`.
        accuracy: The accuracy order of the one-sided stencils.
        method: ``"central"``, ``"forward"`` or ``"backward"``.
        periodic: The axis wraps (a global longitude): every point uses the
            ``method`` stencil, reading across the seam. Not with ``coords``.

    Returns:
        :math:`\partial^n f / \partial s^n`, ``f``'s shape.

    Raises:
        ValueError: The axis is shorter than the stencil needs, ``coords`` has
            the wrong length, or ``coords`` meets ``periodic``.

    Examples:
        >>> f = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x², x = 0…4
        >>> difference(f)
        array([1., 2., 4., 6., 7.])
        >>> difference(f, accuracy=2)
        array([0., 2., 4., 6., 8.])
        >>> difference(f, derivative=2)
        array([2., 2., 2., 2., 2.])
        >>> z = np.array([0.0, 1.0, 3.0, 7.0, 15.0])  # uneven levels
        >>> difference(z**2, coords=z, derivative=2)  # exact for a quadratic
        array([2., 2., 2., 2., 2.])
    """
    reject_geotensor(f, "difference")
    xp = array_namespace(f)
    axis = axis % f.ndim
    if coords is not None:
        if periodic:
            raise ValueError("difference: coords= does not combine with periodic=True")
        if coords.shape != (f.shape[axis],):
            raise ValueError(
                f"difference: coords has shape {tuple(coords.shape)}; the axis has "
                f"{f.shape[axis]} entries"
            )
        return _bounded(xp, f, axis, derivative, accuracy, method, coords)
    if periodic:
        out = _periodic(xp, f, axis, derivative, accuracy, method)
    else:
        out = _bounded(xp, f, axis, derivative, accuracy, method)
    return out / step**derivative


def adaptive_difference(
    f: Float[np.ndarray, "*shape"],
    *,
    mask: Bool[np.ndarray, "*shape"] | None = None,
    axis: int = -1,
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    coords: Float[np.ndarray, " n"] | None = None,
    derivative: int = 1,
    accuracy: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    r"""Mask-aware finite difference: the widest stencil that fits on valid data.

    Near a mask (land, a swath edge, a cloud) the centred stencil reads a
    masked cell and :func:`difference` loses the valid cells next to it.
    This keeps, per point, the first finite result among the central,
    forward and backward stencils at ``accuracy``, then at each lower
    accuracy down to 1, so a narrow channel still gets a first-order
    estimate. Recovered cells carry one order less than the interior.

    NaN is its own support detector: a stencil that reads a masked cell
    returns NaN (``0 * nan`` is ``nan``), so the selection needs no mask
    arithmetic (xrtoolz ``docs/design/xrgrad-nan-masks.md``, where it beat
    fill-then-differentiate by ~90x at the coast). Every candidate is
    computed, so the call costs ``3 * accuracy`` stencils and runs under
    ``jax.jit``. Masked cells are ``NaN`` in the result.

    Args:
        f: The field; any shape.
        mask: ``True`` drops the cell (geoops' mask convention), on ``f``'s
            shape; ``None`` drops only non-finite cells.
        axis: The axis to differentiate.
        step: The signed step between centres along ``axis``.
        coords: The centres' positions along ``axis``, for an unevenly spaced
            axis (see :func:`difference`); replaces ``step``.
        derivative: The derivative order.
        accuracy: The highest accuracy order to try.
        periodic: The axis wraps.

    Returns:
        The derivative, ``f``'s shape, ``NaN`` where ``f`` is masked or no
        stencil fits on valid data.

    Example:
        >>> f = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0])  # f = x
        >>> difference(f)
        array([ 1.,  1., nan, nan, nan,  1.,  1.])
        >>> adaptive_difference(f)
        array([ 1.,  1.,  1., nan,  1.,  1.,  1.])
    """
    reject_geotensor(f, "adaptive_difference")
    reject_geotensor(mask, "adaptive_difference")
    xp = array_namespace(f)
    dropped = ~xp.isfinite(f)
    if mask is not None:
        dropped = dropped | mask
    field = xp.where(dropped, xp.nan, f)
    out = None
    for acc in range(accuracy, 0, -1):
        for method in METHODS:
            try:
                candidate = difference(
                    field,
                    axis=axis,
                    step=step,
                    coords=coords,
                    derivative=derivative,
                    accuracy=acc,
                    method=method,
                    periodic=periodic,
                )
            except ValueError:
                continue  # a stencil wider than the axis: try a narrower one
            out = (
                candidate if out is None else xp.where(xp.isfinite(out), out, candidate)
            )
    if out is None:
        raise ValueError(
            f"axis of length {f.shape[axis]} is too short for derivative {derivative}"
        )
    return xp.where(dropped, xp.nan, out)


def gradient_2d(
    f: Float[np.ndarray, "*batch h w"],
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "2 *batch h w"]:
    r"""``(∂f/∂y, ∂f/∂x)`` over the last two axes; a NaN propagates.

    :func:`difference` along rows (``axis=-2``, step ``dy``) and columns
    (``axis=-1``, step ``dx``). At ``accuracy=1``: second-order centred
    differences inside, first-order one-sided on the edges, exact for a
    linear field everywhere.

    Args:
        f: The field, rows and columns last; leading axes are mapped.
        dy: The signed step between row centres, a scalar or ``(H, W)``.
        dx: The signed step between column centres, a scalar or ``(H, W)``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic_x: The columns wrap (a global longitude).

    Returns:
        ``(∂f/∂y, ∂f/∂x)`` stacked on a new leading axis.

    Example:
        >>> y, x = np.mgrid[0:3, 0:4].astype(float)
        >>> d_y, d_x = gradient_2d(2.0 * y + 3.0 * x)
        >>> float(d_y.min()), float(d_y.max()), float(d_x.min()), float(d_x.max())
        (2.0, 2.0, 3.0, 3.0)
    """
    reject_geotensor(f, "gradient_2d")
    xp = array_namespace(f)
    d_y = difference(f, axis=-2, step=dy, accuracy=accuracy)
    d_x = difference(f, axis=-1, step=dx, accuracy=accuracy, periodic=periodic_x)
    return xp.stack([d_y, d_x])


def adaptive_gradient_2d(
    f: Float[np.ndarray, "*batch h w"],
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    mask: Bool[np.ndarray, "*batch h w"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "2 *batch h w"]:
    """Mask-aware ``(∂f/∂y, ∂f/∂x)``: :func:`adaptive_difference` on both axes.

    Args:
        f: The field, rows and columns last.
        dy: The signed step between row centres.
        dx: The signed step between column centres.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic_x: The columns wrap.

    Returns:
        ``(∂f/∂y, ∂f/∂x)`` stacked on a new leading axis, ``NaN`` on masked
        cells only (a linear field stays exact up to the coast).

    Example:
        >>> y, x = np.mgrid[0:4, 0:5].astype(float)
        >>> land = (y == 1) & (x == 2)
        >>> d_y, d_x = adaptive_gradient_2d(y + x, mask=land)
        >>> int(np.isnan(d_x).sum()), float(np.nanmax(np.abs(d_x - 1.0)))
        (1, 0.0)
    """
    reject_geotensor(f, "adaptive_gradient_2d")
    reject_geotensor(mask, "adaptive_gradient_2d")
    xp = array_namespace(f)
    d_y = adaptive_difference(f, mask=mask, axis=-2, step=dy, accuracy=accuracy)
    d_x = adaptive_difference(
        f, mask=mask, axis=-1, step=dx, accuracy=accuracy, periodic=periodic_x
    )
    return xp.stack([d_y, d_x])


def divergence_2d(
    u: Float[np.ndarray, "*batch h w"],
    v: Float[np.ndarray, "*batch h w"],
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch h w"]:
    r"""Divergence ``∂u/∂x + ∂v/∂y`` over the last two axes; a NaN propagates.

    .. math::

        \nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x}
                                + \frac{\partial v}{\partial y}

    :func:`difference` of ``u`` along columns (``axis=-1``, step ``dx``) plus
    that of ``v`` along rows (``axis=-2``, step ``dy``), so a uniform-grid
    field needs no carrier.

    Args:
        u: The component along columns, rows and columns last.
        v: The component along rows, on ``u``'s shape.
        dy: The signed step between row centres, a scalar or ``(H, W)``.
        dx: The signed step between column centres, a scalar or ``(H, W)``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic_x: The columns wrap (a global longitude).

    Returns:
        The divergence, ``u``'s shape.

    Examples:
        >>> y, x = np.mgrid[0:4, 0:5].astype(float)
        >>> float(np.abs(divergence_2d(-y, x)).max())  # solid-body rotation
        0.0
        >>> float(divergence_2d(x, y).min())  # a uniform expansion
        2.0
    """
    reject_geotensor(u, "divergence_2d")
    reject_geotensor(v, "divergence_2d")
    d_u = difference(u, axis=-1, step=dx, accuracy=accuracy, periodic=periodic_x)
    return d_u + difference(v, axis=-2, step=dy, accuracy=accuracy)


def adaptive_divergence_2d(
    u: Float[np.ndarray, "*batch h w"],
    v: Float[np.ndarray, "*batch h w"],
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    mask: Bool[np.ndarray, "*batch h w"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch h w"]:
    """Mask-aware divergence: :func:`adaptive_difference` for each derivative.

    Each term uses, per point, the widest stencil lying on valid data, so a
    linear field stays exact up to the coast. A cell is ``NaN`` where ``mask``
    drops it or either component is non-finite.

    Args:
        u: The component along columns.
        v: The component along rows.
        dy: The signed step between row centres.
        dx: The signed step between column centres.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic_x: The columns wrap.

    Returns:
        The divergence, ``u``'s shape.

    Example:
        >>> y, x = np.mgrid[0:5, 0:6].astype(float)
        >>> land = (y == 2) & (x == 3)
        >>> div = adaptive_divergence_2d(x, y, mask=land)
        >>> int(np.isnan(div).sum()), float(np.nanmax(np.abs(div - 2.0)))
        (1, 0.0)
    """
    reject_geotensor(u, "adaptive_divergence_2d")
    reject_geotensor(v, "adaptive_divergence_2d")
    reject_geotensor(mask, "adaptive_divergence_2d")
    d_u = adaptive_difference(
        u, mask=mask, axis=-1, step=dx, accuracy=accuracy, periodic=periodic_x
    )
    return d_u + adaptive_difference(v, mask=mask, axis=-2, step=dy, accuracy=accuracy)


def curl_2d(
    u: Float[np.ndarray, "*batch h w"],
    v: Float[np.ndarray, "*batch h w"],
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch h w"]:
    r"""Curl2D (relative vorticity) ``∂v/∂x − ∂u/∂y`` over the last two axes.

    .. math::

        \zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}

    A NaN propagates through the stencils that read it.

    Args:
        u: The component along columns, rows and columns last.
        v: The component along rows, on ``u``'s shape.
        dy: The signed step between row centres, a scalar or ``(H, W)``.
        dx: The signed step between column centres, a scalar or ``(H, W)``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic_x: The columns wrap (a global longitude).

    Returns:
        The curl, ``u``'s shape.

    Examples:
        >>> y, x = np.mgrid[0:4, 0:5].astype(float)
        >>> float(curl_2d(-y, x).min()), float(curl_2d(-y, x).max())  # solid-body rotation
        (2.0, 2.0)
        >>> float(np.abs(curl_2d(x, y)).max())  # an irrotational expansion
        0.0
    """
    reject_geotensor(u, "curl_2d")
    reject_geotensor(v, "curl_2d")
    d_v = difference(v, axis=-1, step=dx, accuracy=accuracy, periodic=periodic_x)
    return d_v - difference(u, axis=-2, step=dy, accuracy=accuracy)


def adaptive_curl_2d(
    u: Float[np.ndarray, "*batch h w"],
    v: Float[np.ndarray, "*batch h w"],
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    mask: Bool[np.ndarray, "*batch h w"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch h w"]:
    """Mask-aware curl: :func:`adaptive_difference` for each derivative.

    Args:
        u: The component along columns.
        v: The component along rows.
        dy: The signed step between row centres.
        dx: The signed step between column centres.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic_x: The columns wrap.

    Returns:
        The curl, ``u``'s shape, ``NaN`` only where a cell is dropped.

    Example:
        >>> y, x = np.mgrid[0:5, 0:6].astype(float)
        >>> wall = x == 2
        >>> zeta = adaptive_curl_2d(-y, x, mask=wall)
        >>> int(np.isnan(zeta).sum()), float(np.nanmax(np.abs(zeta - 2.0)))
        (5, 0.0)
    """
    reject_geotensor(u, "adaptive_curl_2d")
    reject_geotensor(v, "adaptive_curl_2d")
    reject_geotensor(mask, "adaptive_curl_2d")
    d_v = adaptive_difference(
        v, mask=mask, axis=-1, step=dx, accuracy=accuracy, periodic=periodic_x
    )
    return d_v - adaptive_difference(u, mask=mask, axis=-2, step=dy, accuracy=accuracy)


def laplacian_2d(
    f: Float[np.ndarray, "*batch h w"],
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch h w"]:
    r"""Laplacian ``∂²f/∂x² + ∂²f/∂y²`` over the last two axes; a NaN propagates.

    .. math::

        \nabla^2 f = \frac{\partial^2 f}{\partial x^2}
                   + \frac{\partial^2 f}{\partial y^2}

    Second-derivative stencils (``derivative=2``), not two first derivatives:
    the 3-point centred stencil inside and 3-point one-sided ones on the edges
    at ``accuracy=1``, all exact for a quadratic.

    Args:
        f: The field, rows and columns last; leading axes are mapped.
        dy: The step between row centres, a scalar or ``(H, W)``.
        dx: The step between column centres, a scalar or ``(H, W)``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic_x: The columns wrap (a global longitude).

    Returns:
        The Laplacian, ``f``'s shape.

    Examples:
        >>> y, x = np.mgrid[0:4, 0:5].astype(float)
        >>> lap = laplacian_2d(x**2 + y**2)
        >>> float(lap.min()), float(lap.max())  # exact, edges included
        (4.0, 4.0)
        >>> float(laplacian_2d(x**2, dx=0.5).max())  # steps in coordinate units
        8.0
    """
    reject_geotensor(f, "laplacian_2d")
    d_xx = difference(
        f, axis=-1, step=dx, derivative=2, accuracy=accuracy, periodic=periodic_x
    )
    return d_xx + difference(f, axis=-2, step=dy, derivative=2, accuracy=accuracy)


def adaptive_laplacian_2d(
    f: Float[np.ndarray, "*batch h w"],
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    mask: Bool[np.ndarray, "*batch h w"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch h w"]:
    """Mask-aware Laplacian: :func:`adaptive_difference` of second order on each axis.

    Args:
        f: The field, rows and columns last.
        dy: The step between row centres.
        dx: The step between column centres.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic_x: The columns wrap.

    Returns:
        The Laplacian, ``f``'s shape; a quadratic stays exact next to the mask.

    Example:
        >>> y, x = np.mgrid[0:7, 0:8].astype(float)
        >>> island = (y == 3) & (x == 4)  # two cells from every edge
        >>> lap = adaptive_laplacian_2d(x**2 + y**2, mask=island)
        >>> int(np.isnan(lap).sum()), float(np.nanmax(np.abs(lap - 4.0)))
        (1, 0.0)
    """
    reject_geotensor(f, "adaptive_laplacian_2d")
    reject_geotensor(mask, "adaptive_laplacian_2d")
    d_xx = adaptive_difference(
        f,
        mask=mask,
        axis=-1,
        step=dx,
        derivative=2,
        accuracy=accuracy,
        periodic=periodic_x,
    )
    return d_xx + adaptive_difference(
        f, mask=mask, axis=-2, step=dy, derivative=2, accuracy=accuracy
    )


# ---------------------------------------------------------------------------
# 1-D and 3-D calculus (L1). One axis at a time: `axis` with `step` or
# `coords`; three axes: z (``axis=-3``, step `dz` or positions `z`), y, x.
# ---------------------------------------------------------------------------


def gradient_1d(
    f: Float[np.ndarray, "*shape"],
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    *,
    axis: int = -1,
    coords: Float[np.ndarray, " n"] | None = None,
    accuracy: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    r"""``∂f/∂s`` along one axis; a NaN propagates.

    :func:`difference` of first order, named for the calculus family.

    Args:
        f: The field; any shape.
        step: The signed step between centres along ``axis``.
        axis: The axis to differentiate.
        coords: The centres' positions along ``axis`` for an unevenly spaced
            axis (depth or pressure levels, irregular times); replaces ``step``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic: The axis wraps.

    Returns:
        :math:`\partial f / \partial s`, ``f``'s shape.

    Examples:
        >>> gradient_1d(np.array([0.0, 2.0, 4.0, 6.0]), 2.0)
        array([1., 1., 1., 1.])
        >>> z = np.array([0.0, 10.0, 30.0, 70.0])  # uneven levels
        >>> gradient_1d(3.0 * z, coords=z)
        array([3., 3., 3., 3.])
    """
    reject_geotensor(f, "gradient_1d")
    return difference(
        f, axis=axis, step=step, coords=coords, accuracy=accuracy, periodic=periodic
    )


def adaptive_gradient_1d(
    f: Float[np.ndarray, "*shape"],
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    *,
    axis: int = -1,
    coords: Float[np.ndarray, " n"] | None = None,
    mask: Bool[np.ndarray, "*shape"] | None = None,
    accuracy: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    """Mask-aware ``∂f/∂s`` along one axis: :func:`adaptive_difference`.

    Args:
        f: The field; any shape.
        step: The signed step between centres along ``axis``.
        axis: The axis to differentiate.
        coords: The centres' positions along ``axis``; replaces ``step``.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic: The axis wraps.

    Returns:
        The derivative, ``f``'s shape, ``NaN`` only where a cell is dropped.

    Example:
        >>> adaptive_gradient_1d(np.array([0.0, 1.0, np.nan, 3.0, 4.0]))
        array([ 1.,  1., nan,  1.,  1.])
    """
    reject_geotensor(f, "adaptive_gradient_1d")
    return adaptive_difference(
        f,
        mask=mask,
        axis=axis,
        step=step,
        coords=coords,
        accuracy=accuracy,
        periodic=periodic,
    )


def divergence_1d(
    u: Float[np.ndarray, "*shape"],
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    *,
    axis: int = -1,
    coords: Float[np.ndarray, " n"] | None = None,
    accuracy: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    r"""The 1-D divergence ``∂u/∂s`` of a flux along one axis; a NaN propagates.

    In one dimension the divergence of a flux is its derivative, so this is
    :func:`gradient_1d` read as a conservation term (``∂F/∂z`` of a vertical
    flux, ``∂q/∂x`` along a channel).

    Args:
        u: The flux component along ``axis``.
        step: The signed step between centres along ``axis``.
        axis: The axis to differentiate.
        coords: The centres' positions along ``axis``; replaces ``step``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic: The axis wraps.

    Returns:
        :math:`\partial u / \partial s`, ``u``'s shape.

    Example:
        >>> divergence_1d(np.array([0.0, 1.0, 4.0, 9.0]))  # u = s², s = 0…3
        array([1., 2., 4., 5.])
    """
    reject_geotensor(u, "divergence_1d")
    return difference(
        u, axis=axis, step=step, coords=coords, accuracy=accuracy, periodic=periodic
    )


def adaptive_divergence_1d(
    u: Float[np.ndarray, "*shape"],
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    *,
    axis: int = -1,
    coords: Float[np.ndarray, " n"] | None = None,
    mask: Bool[np.ndarray, "*shape"] | None = None,
    accuracy: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    """Mask-aware 1-D divergence: :func:`adaptive_difference` of ``u``.

    Args:
        u: The flux component along ``axis``.
        step: The signed step between centres along ``axis``.
        axis: The axis to differentiate.
        coords: The centres' positions along ``axis``; replaces ``step``.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic: The axis wraps.

    Returns:
        The divergence, ``u``'s shape, ``NaN`` only where a cell is dropped.

    Example:
        >>> adaptive_divergence_1d(np.array([0.0, 2.0, np.nan, 6.0, 8.0]), 2.0)
        array([ 1.,  1., nan,  1.,  1.])
    """
    reject_geotensor(u, "adaptive_divergence_1d")
    return adaptive_difference(
        u,
        mask=mask,
        axis=axis,
        step=step,
        coords=coords,
        accuracy=accuracy,
        periodic=periodic,
    )


def laplacian_1d(
    f: Float[np.ndarray, "*shape"],
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    *,
    axis: int = -1,
    coords: Float[np.ndarray, " n"] | None = None,
    accuracy: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    r"""``∂²f/∂s²`` along one axis, a second-derivative stencil; a NaN propagates.

    Args:
        f: The field; any shape.
        step: The step between centres along ``axis``.
        axis: The axis to differentiate.
        coords: The centres' positions along ``axis``; replaces ``step``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic: The axis wraps.

    Returns:
        :math:`\partial^2 f / \partial s^2`, ``f``'s shape.

    Examples:
        >>> laplacian_1d(np.array([0.0, 1.0, 4.0, 9.0]))  # f = s²
        array([2., 2., 2., 2.])
        >>> z = np.array([0.0, 5.0, 15.0, 30.0])  # uneven levels
        >>> laplacian_1d(z**2, coords=z)
        array([2., 2., 2., 2.])
    """
    reject_geotensor(f, "laplacian_1d")
    return difference(
        f,
        axis=axis,
        step=step,
        coords=coords,
        derivative=2,
        accuracy=accuracy,
        periodic=periodic,
    )


def adaptive_laplacian_1d(
    f: Float[np.ndarray, "*shape"],
    step: Float[np.ndarray, "*shape"] | float = 1.0,
    *,
    axis: int = -1,
    coords: Float[np.ndarray, " n"] | None = None,
    mask: Bool[np.ndarray, "*shape"] | None = None,
    accuracy: int = 1,
    periodic: bool = False,
) -> Float[np.ndarray, "*shape"]:
    """Mask-aware ``∂²f/∂s²`` along one axis: :func:`adaptive_difference` of order 2.

    Args:
        f: The field; any shape.
        step: The step between centres along ``axis``.
        axis: The axis to differentiate.
        coords: The centres' positions along ``axis``; replaces ``step``.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic: The axis wraps.

    Returns:
        The second derivative, ``f``'s shape, ``NaN`` only where dropped.

    Example:
        >>> s = np.arange(8.0)
        >>> f = s**2
        >>> f[3] = np.nan
        >>> adaptive_laplacian_1d(f)
        array([ 2.,  2.,  2., nan,  2.,  2.,  2.,  2.])
    """
    reject_geotensor(f, "adaptive_laplacian_1d")
    return adaptive_difference(
        f,
        mask=mask,
        axis=axis,
        step=step,
        coords=coords,
        derivative=2,
        accuracy=accuracy,
        periodic=periodic,
    )


def gradient_3d(
    f: Float[np.ndarray, "*batch d h w"],
    dz: Float[np.ndarray, "d h w"] | float = 1.0,
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    z: Float[np.ndarray, " d"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "3 *batch d h w"]:
    r"""``(∂f/∂z, ∂f/∂y, ∂f/∂x)`` over the last three axes; a NaN propagates.

    The components follow the axes, levels (``axis=-3``) then rows then
    columns, as :func:`gradient_2d` gives rows then columns.

    Args:
        f: The field, levels, rows and columns last.
        dz: The signed step between level centres.
        dy: The signed step between row centres.
        dx: The signed step between column centres.
        z: The level centres' positions for uneven levels; replaces ``dz``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic_x: The columns wrap (a global longitude).

    Returns:
        ``(∂f/∂z, ∂f/∂y, ∂f/∂x)`` stacked on a new leading axis.

    Example:
        >>> z, y, x = np.mgrid[0:3, 0:4, 0:5].astype(float)
        >>> tuple(float(d.max()) for d in gradient_3d(4.0 * z + 2.0 * y + x))
        (4.0, 2.0, 1.0)
    """
    reject_geotensor(f, "gradient_3d")
    xp = array_namespace(f)
    kw = {"accuracy": accuracy}
    d_z = difference(f, axis=-3, step=dz, coords=z, **kw)
    d_y = difference(f, axis=-2, step=dy, **kw)
    d_x = difference(f, axis=-1, step=dx, periodic=periodic_x, **kw)
    return xp.stack([d_z, d_y, d_x])


def adaptive_gradient_3d(
    f: Float[np.ndarray, "*batch d h w"],
    dz: Float[np.ndarray, "d h w"] | float = 1.0,
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    z: Float[np.ndarray, " d"] | None = None,
    mask: Bool[np.ndarray, "*batch d h w"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "3 *batch d h w"]:
    """Mask-aware ``(∂f/∂z, ∂f/∂y, ∂f/∂x)``: :func:`adaptive_difference` per axis.

    Args:
        f: The field, levels, rows and columns last.
        dz: The signed step between level centres.
        dy: The signed step between row centres.
        dx: The signed step between column centres.
        z: The level centres' positions; replaces ``dz``.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic_x: The columns wrap.

    Returns:
        The three components stacked on a new leading axis.

    Example:
        >>> z, y, x = np.mgrid[0:4, 0:4, 0:5].astype(float)
        >>> seabed = (z == 3) & (x == 2)
        >>> d_z, _, _ = adaptive_gradient_3d(z + y + x, mask=seabed)
        >>> int(np.isnan(d_z).sum()), float(np.nanmax(np.abs(d_z - 1.0)))
        (4, 0.0)
    """
    reject_geotensor(f, "adaptive_gradient_3d")
    reject_geotensor(mask, "adaptive_gradient_3d")
    xp = array_namespace(f)
    kw = {"mask": mask, "accuracy": accuracy}
    d_z = adaptive_difference(f, axis=-3, step=dz, coords=z, **kw)
    d_y = adaptive_difference(f, axis=-2, step=dy, **kw)
    d_x = adaptive_difference(f, axis=-1, step=dx, periodic=periodic_x, **kw)
    return xp.stack([d_z, d_y, d_x])


def divergence_3d(
    u: Float[np.ndarray, "*batch d h w"],
    v: Float[np.ndarray, "*batch d h w"],
    w: Float[np.ndarray, "*batch d h w"],
    dz: Float[np.ndarray, "d h w"] | float = 1.0,
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    z: Float[np.ndarray, " d"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch d h w"]:
    r"""Divergence ``∂u/∂x + ∂v/∂y + ∂w/∂z`` over the last three axes.

    .. math::

        \nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x}
          + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z}

    ``w`` is the component along increasing ``z`` (the level coordinate as
    stored), so on pressure levels this is the pressure-coordinate divergence
    with :math:`\omega` in ``w``. A NaN propagates.

    Args:
        u: The component along columns.
        v: The component along rows.
        w: The component along levels.
        dz: The signed step between level centres.
        dy: The signed step between row centres.
        dx: The signed step between column centres.
        z: The level centres' positions for uneven levels; replaces ``dz``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic_x: The columns wrap.

    Returns:
        The divergence, ``u``'s shape.

    Example:
        >>> z, y, x = np.mgrid[0:3, 0:4, 0:5].astype(float)
        >>> float(divergence_3d(x, y, -2.0 * z).max())  # incompressible
        0.0
    """
    for c in (u, v, w):
        reject_geotensor(c, "divergence_3d")
    kw = {"accuracy": accuracy}
    return (
        difference(u, axis=-1, step=dx, periodic=periodic_x, **kw)
        + difference(v, axis=-2, step=dy, **kw)
        + difference(w, axis=-3, step=dz, coords=z, **kw)
    )


def adaptive_divergence_3d(
    u: Float[np.ndarray, "*batch d h w"],
    v: Float[np.ndarray, "*batch d h w"],
    w: Float[np.ndarray, "*batch d h w"],
    dz: Float[np.ndarray, "d h w"] | float = 1.0,
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    z: Float[np.ndarray, " d"] | None = None,
    mask: Bool[np.ndarray, "*batch d h w"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch d h w"]:
    """Mask-aware 3-D divergence: :func:`adaptive_difference` for each term.

    Args:
        u: The component along columns.
        v: The component along rows.
        w: The component along levels.
        dz: The signed step between level centres.
        dy: The signed step between row centres.
        dx: The signed step between column centres.
        z: The level centres' positions; replaces ``dz``.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic_x: The columns wrap.

    Returns:
        The divergence, ``u``'s shape, ``NaN`` only where a cell is dropped.

    Example:
        >>> z, y, x = np.mgrid[0:5, 0:5, 0:6].astype(float)
        >>> rock = (z == 2) & (y == 2) & (x == 3)  # off every edge
        >>> div = adaptive_divergence_3d(x, y, z, mask=rock)
        >>> int(np.isnan(div).sum()), float(np.nanmax(np.abs(div - 3.0)))
        (1, 0.0)
    """
    for c in (u, v, w, mask):
        reject_geotensor(c, "adaptive_divergence_3d")
    kw = {"mask": mask, "accuracy": accuracy}
    return (
        adaptive_difference(u, axis=-1, step=dx, periodic=periodic_x, **kw)
        + adaptive_difference(v, axis=-2, step=dy, **kw)
        + adaptive_difference(w, axis=-3, step=dz, coords=z, **kw)
    )


def curl_3d(
    u: Float[np.ndarray, "*batch d h w"],
    v: Float[np.ndarray, "*batch d h w"],
    w: Float[np.ndarray, "*batch d h w"],
    dz: Float[np.ndarray, "d h w"] | float = 1.0,
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    z: Float[np.ndarray, " d"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "3 *batch d h w"]:
    r"""Curl (vorticity vector) over the last three axes; a NaN propagates.

    .. math::

        \boldsymbol{\omega} = \nabla \times \mathbf{u}, \quad
        \omega_z = \partial_x v - \partial_y u, \;
        \omega_y = \partial_z u - \partial_x w, \;
        \omega_x = \partial_y w - \partial_z v

    Components in axis order, ``(ω_z, ω_y, ω_x)``, as :func:`gradient_3d`.
    The curl is right-handed in ``(x, y, z)`` with ``z`` the level
    coordinate as stored and ``w`` along increasing ``z``: on a
    ``positive="down"`` axis (depth, pressure) that system is left-handed, so
    ``ω_y`` and ``ω_x`` have the opposite sign of the curl in upward height.

    Args:
        u: The component along columns.
        v: The component along rows.
        w: The component along levels.
        dz: The signed step between level centres.
        dy: The signed step between row centres.
        dx: The signed step between column centres.
        z: The level centres' positions for uneven levels; replaces ``dz``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic_x: The columns wrap.

    Returns:
        ``(ω_z, ω_y, ω_x)`` stacked on a new leading axis.

    Example:
        >>> z, y, x = np.mgrid[0:3, 0:4, 0:5].astype(float)
        >>> zeta_z, zeta_y, zeta_x = curl_3d(-y, x, 0.0 * z)  # rotation about z
        >>> float(zeta_z.min()), float(np.abs(zeta_y).max()), float(np.abs(zeta_x).max())
        (2.0, 0.0, 0.0)
    """
    for c in (u, v, w):
        reject_geotensor(c, "curl_3d")
    xp = array_namespace(u)
    kw = {"accuracy": accuracy}
    d_x = functools.partial(difference, axis=-1, step=dx, periodic=periodic_x, **kw)
    d_y = functools.partial(difference, axis=-2, step=dy, **kw)
    d_z = functools.partial(difference, axis=-3, step=dz, coords=z, **kw)
    return xp.stack([d_x(v) - d_y(u), d_z(u) - d_x(w), d_y(w) - d_z(v)])


def adaptive_curl_3d(
    u: Float[np.ndarray, "*batch d h w"],
    v: Float[np.ndarray, "*batch d h w"],
    w: Float[np.ndarray, "*batch d h w"],
    dz: Float[np.ndarray, "d h w"] | float = 1.0,
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    z: Float[np.ndarray, " d"] | None = None,
    mask: Bool[np.ndarray, "*batch d h w"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "3 *batch d h w"]:
    """Mask-aware 3-D curl: :func:`adaptive_difference` for each derivative.

    Args:
        u: The component along columns.
        v: The component along rows.
        w: The component along levels.
        dz: The signed step between level centres.
        dy: The signed step between row centres.
        dx: The signed step between column centres.
        z: The level centres' positions; replaces ``dz``.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic_x: The columns wrap.

    Returns:
        ``(ω_z, ω_y, ω_x)`` stacked on a new leading axis.

    Example:
        >>> z, y, x = np.mgrid[0:4, 0:5, 0:6].astype(float)
        >>> rock = (z == 2) & (y == 2) & (x == 3)
        >>> zeta_z = adaptive_curl_3d(-y, x, 0.0 * z, mask=rock)[0]
        >>> int(np.isnan(zeta_z).sum()), float(np.nanmax(np.abs(zeta_z - 2.0)))
        (1, 0.0)
    """
    for c in (u, v, w, mask):
        reject_geotensor(c, "adaptive_curl_3d")
    xp = array_namespace(u)
    kw = {"mask": mask, "accuracy": accuracy}
    d_x = functools.partial(
        adaptive_difference, axis=-1, step=dx, periodic=periodic_x, **kw
    )
    d_y = functools.partial(adaptive_difference, axis=-2, step=dy, **kw)
    d_z = functools.partial(adaptive_difference, axis=-3, step=dz, coords=z, **kw)
    return xp.stack([d_x(v) - d_y(u), d_z(u) - d_x(w), d_y(w) - d_z(v)])


def laplacian_3d(
    f: Float[np.ndarray, "*batch d h w"],
    dz: Float[np.ndarray, "d h w"] | float = 1.0,
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    z: Float[np.ndarray, " d"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch d h w"]:
    r"""Laplacian ``∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²``, second-derivative stencils.

    Args:
        f: The field, levels, rows and columns last.
        dz: The step between level centres.
        dy: The step between row centres.
        dx: The step between column centres.
        z: The level centres' positions for uneven levels; replaces ``dz``.
        accuracy: The accuracy order (see :func:`difference`).
        periodic_x: The columns wrap.

    Returns:
        The Laplacian, ``f``'s shape.

    Example:
        >>> z, y, x = np.mgrid[0:3, 0:4, 0:5].astype(float)
        >>> lap = laplacian_3d(x**2 + y**2 + z**2)
        >>> float(lap.min()), float(lap.max())  # exact, edges included
        (6.0, 6.0)
    """
    reject_geotensor(f, "laplacian_3d")
    kw = {"derivative": 2, "accuracy": accuracy}
    return (
        difference(f, axis=-1, step=dx, periodic=periodic_x, **kw)
        + difference(f, axis=-2, step=dy, **kw)
        + difference(f, axis=-3, step=dz, coords=z, **kw)
    )


def adaptive_laplacian_3d(
    f: Float[np.ndarray, "*batch d h w"],
    dz: Float[np.ndarray, "d h w"] | float = 1.0,
    dy: Float[np.ndarray, "h w"] | float = 1.0,
    dx: Float[np.ndarray, "h w"] | float = 1.0,
    *,
    z: Float[np.ndarray, " d"] | None = None,
    mask: Bool[np.ndarray, "*batch d h w"] | None = None,
    accuracy: int = 1,
    periodic_x: bool = False,
) -> Float[np.ndarray, "*batch d h w"]:
    """Mask-aware 3-D Laplacian: :func:`adaptive_difference` of order 2 per axis.

    Args:
        f: The field, levels, rows and columns last.
        dz: The step between level centres.
        dy: The step between row centres.
        dx: The step between column centres.
        z: The level centres' positions; replaces ``dz``.
        mask: ``True`` drops the cell; ``None`` drops only non-finite cells.
        accuracy: The highest accuracy order to try.
        periodic_x: The columns wrap.

    Returns:
        The Laplacian, ``f``'s shape; a quadratic stays exact next to the mask.

    Example:
        >>> z, y, x = np.mgrid[0:7, 0:7, 0:8].astype(float)
        >>> rock = (z == 3) & (y == 3) & (x == 4)
        >>> lap = adaptive_laplacian_3d(x**2 + y**2 + z**2, mask=rock)
        >>> int(np.isnan(lap).sum()), float(np.nanmax(np.abs(lap - 6.0)))
        (1, 0.0)
    """
    reject_geotensor(f, "adaptive_laplacian_3d")
    reject_geotensor(mask, "adaptive_laplacian_3d")
    kw = {"mask": mask, "derivative": 2, "accuracy": accuracy}
    return (
        adaptive_difference(f, axis=-1, step=dx, periodic=periodic_x, **kw)
        + adaptive_difference(f, axis=-2, step=dy, **kw)
        + adaptive_difference(f, axis=-3, step=dz, coords=z, **kw)
    )


def _bounded(
    xp: Any,
    f: Any,
    axis: int,
    derivative: int,
    accuracy: int,
    method: Method,
    coords: Any = None,
) -> Any:
    """The stencils of ``method`` with one-sided ones where it would leave the axis."""
    size = f.shape[axis]
    forward = difference_offsets(
        derivative=derivative, accuracy=accuracy, method="forward"
    )
    backward = difference_offsets(
        derivative=derivative, accuracy=accuracy, method="backward"
    )
    if method == "central":
        centre = difference_offsets(
            derivative=derivative, accuracy=accuracy, method="central"
        )
        if size < len(centre):
            raise ValueError(
                f"an axis of length {size} is shorter than the {len(centre)}-point "
                f"central stencil (derivative {derivative}, accuracy {accuracy})"
            )
        regions = [
            (0, -centre[0], forward),
            (-centre[0], size - centre[-1], centre),
            (size - centre[-1], size, backward),
        ]
    elif method in ("forward", "backward"):
        if size < len(forward):
            raise ValueError(
                f"an axis of length {size} is shorter than the {len(forward)}-point "
                f"{method} stencil (derivative {derivative}, accuracy {accuracy})"
            )
        if method == "forward":
            regions = [
                (0, size - forward[-1], forward),
                (size - forward[-1], size, backward),
            ]
        else:
            regions = [(0, -backward[0], forward), (-backward[0], size, backward)]
    else:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    regions = [
        (start, stop, offsets) for start, stop, offsets in regions if stop > start
    ]
    for start, stop, offsets in regions:
        # every point of a region must read inside the axis
        if start + offsets[0] < 0 or stop - 1 + offsets[-1] > size - 1:
            raise ValueError(
                f"an axis of length {size} is too short for the {len(offsets)}-point "
                f"boundary stencils of method {method!r} (derivative {derivative}, "
                f"accuracy {accuracy}); lower the accuracy"
            )
    parts = [
        _apply_stencil(f, axis, start, stop, offsets, derivative, coords)
        for start, stop, offsets in regions
    ]
    return xp.concat(parts, axis=axis)


def _apply_stencil(
    f: Any,
    axis: int,
    start: int,
    stop: int,
    offsets: tuple[int, ...],
    derivative: int,
    coords: Any = None,
) -> Any:
    """``Σ c_k f[i + s_k]`` for ``i`` in ``[start, stop)`` along ``axis``."""
    if coords is None:
        coeffs: Any = difference_coeffs(offsets, derivative)
    else:
        xp = array_namespace(coords)
        centres = coords[start:stop]
        dz = xp.stack(
            [coords[start + o : stop + o] - centres for o in offsets], axis=-1
        )
        # one coefficient per point, laid along `axis` to broadcast against f
        shape = [1] * f.ndim
        shape[axis] = stop - start
        weights = nonuniform_coeffs(xp.astype(dz, xp.float64), derivative)
        coeffs = [xp.reshape(weights[:, k], shape) for k in range(len(offsets))]
    out = None
    for offset, coeff in zip(offsets, coeffs, strict=True):
        index = [slice(None)] * f.ndim
        index[axis] = slice(start + offset, stop + offset)
        term = coeff * f[tuple(index)]  # zero coefficients too: 0 * nan is nan
        out = term if out is None else out + term
    return out


def _periodic(
    xp: Any, f: Any, axis: int, derivative: int, accuracy: int, method: Method
) -> Any:
    """The ``method`` stencil at every point, reading across the seam."""
    offsets = difference_offsets(
        derivative=derivative, accuracy=accuracy, method=method
    )
    if f.shape[axis] < len(offsets):
        raise ValueError(
            f"a periodic axis of length {f.shape[axis]} is shorter than the "
            f"{len(offsets)}-point {method} stencil"
        )
    coeffs = difference_coeffs(offsets, derivative)
    out = None
    for offset, coeff in zip(offsets, coeffs, strict=True):
        term = coeff * xp.roll(f, -offset, axis=axis)
        out = term if out is None else out + term
    return out
