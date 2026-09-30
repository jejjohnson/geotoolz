"""Tier-A primitives for geometry operators — pure-numpy helpers.

The carrier-aware Operator wrappers in
:mod:`geotoolz.geom._src.operators` lift these into the ``GeoTensor``
pipeline. Each helper here is deliberately framework-agnostic: it
operates on ``numpy.ndarray`` (and tuples of plain ints / floats) so the
same primitive is callable from notebooks and tests without constructing
a ``GeoTensor``.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from jaxtyping import Bool, Float, Shaped
from rasterio.enums import Resampling

from geotoolz._src.blending import triangular_weights


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
    if isinstance(name, Resampling):
        return name
    return Resampling[_RASTERIO_ALIASES.get(name, name)]


def resolve_interpolation(name: str) -> str:
    """Translate a resampling name into a ``skimage.transform.resize`` mode.

    ``GeoTensor.resize`` delegates to ``skimage.transform.resize`` whose
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
    pad the right/bottom edges with the carrier's ``fill_value_default``;
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
