"""Pure-numpy plume detection and quantification primitives.

This module hosts the Tier-A primitives (Google-style docstrings, no
``GeoTensor``) that the Tier-B operators in ``operators.py`` wrap.
Algorithms and unit conventions follow the trace-gas plume literature
referenced per-function:

- Varon et al. (2018), AMT — Integrated Mass Enhancement (IME) method.
- Varon et al. (2021), RSE — Sentinel-2 SWIR single-band methane retrieval.
- Frankenberg et al. (2016), PNAS — airborne CH4 plume detection.
- Thompson et al. (2015), AMT — matched-filter CH4 retrieval.
- Foote et al. (2020), TGRS — fast matched-filter for CH4.
- Ehret et al. (2022), TGRS — S2 multi-band methane retrieval.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from jaxtyping import Bool, Float, Num, Shaped
from shapely.geometry import MultiPoint

from geotoolz._src.geo import pixel_xy
from geotoolz._src.labels import Connectivity, label_components, skeleton_length
from geotoolz._src.shape import single_band
from geotoolz.segment._src.array import ThresholdMode, threshold_mask


ColumnUnit = Literal["ppm_m", "mol_m2", "kg_m2"]

# Molar masses (kg/mol) for supported trace gases.
MOLAR_MASS_KG_PER_MOL = {
    "CH4": 0.01604,
    "CO2": 0.04401,
}
# Standard molar volume of an ideal gas at 298.15 K, 1 atm (m^3/mol).
# Used only by ``convert_column_units`` to translate ppm m (mixing ratio
# integrated along path) into mol/m^2 for typical near-surface conditions.
STANDARD_MOLAR_VOLUME_M3_PER_MOL = 0.024465


def squeeze_single_band(
    values: Shaped[np.ndarray, "h w"] | Shaped[np.ndarray, "1 h w"],
) -> Shaped[np.ndarray, "h w"]:
    """Return a 2-D plume map from a 2-D or singleton-band array.

    Thin delegate to :func:`geotoolz._src.shape.single_band`, kept under
    its historical public name for the ``geotoolz.plume`` API.

    Args:
        values: A ``(H, W)`` array or a ``(1, H, W)`` singleton-band
            cube. Any array-like is accepted.

    Returns:
        The ``(H, W)`` array. No copy is made for ndarray input.

    Raises:
        ValueError: If ``values`` is neither ``(H, W)`` nor ``(1, H, W)``.
    """
    return single_band(values, name="plume")


def plume_mask(
    values: Num[np.ndarray, "h w"] | Num[np.ndarray, "1 h w"],
    *,
    threshold: ThresholdMode = "otsu",
    min_area: int = 50,
    connectivity: Connectivity = 8,
    nbins: int = 256,
) -> Bool[np.ndarray, "h w"]:
    """Threshold an enhancement map and remove small connected components.

    Args:
        values: Single-band enhancement map, ``(H, W)`` or ``(1, H, W)``.
        threshold: Absolute number, ``"otsu"``, or ``"percentile:<p>"``;
            see :func:`geotoolz.segment.resolve_threshold`.
        min_area: Minimum connected-component size in pixels.
        connectivity: 4 or 8 connectivity for component labelling.
        nbins: Histogram bins for the ``"otsu"`` threshold. Default ``256``.

    Returns:
        Boolean ``(H, W)`` mask of the surviving plume pixels.
    """
    arr = squeeze_single_band(values)
    raw = threshold_mask(arr, threshold, nbins=nbins)
    return label_components(raw, min_area=min_area, connectivity=connectivity) > 0


def pixel_area(transform: Any) -> float:
    """Return pixel area from an affine-like transform determinant.

    Args:
        transform: Affine-like object exposing ``a``, ``b``, ``d``, ``e``
            coefficients (e.g. ``rasterio.Affine``).

    Returns:
        ``|a*e - b*d|`` — the pixel area in squared CRS units (m^2 for a
        projected metric CRS).
    """
    return float(abs(transform.a * transform.e - transform.b * transform.d))


def pixel_centers(
    shape: tuple[int, int], transform: Any
) -> tuple[Float[np.ndarray, "h w"], Float[np.ndarray, "h w"]]:
    """Return x/y coordinate grids for pixel centers.

    Args:
        shape: Raster shape ``(H, W)``.
        transform: Affine-like geotransform mapping ``(col, row)`` pixel
            indices to CRS coordinates.

    Returns:
        Tuple ``(xs, ys)`` of ``(H, W)`` float arrays holding the CRS
        coordinates of each pixel center.
    """
    rows, cols = np.indices(shape, dtype=float)
    return pixel_xy(transform, rows, cols)


def wind_advection_cone(
    shape: tuple[int, int],
    transform: Any,
    *,
    source: tuple[float, float],
    wind_u: float,
    wind_v: float,
    half_angle_deg: float = 30.0,
    max_distance: float = 5000.0,
) -> Bool[np.ndarray, "h w"]:
    """Rasterize an analytical downwind sector mask.

    Marks pixels whose center lies within ``max_distance`` of ``source``
    and whose bearing from the source is within ``half_angle_deg`` of the
    wind direction ``(wind_u, wind_v)`` — a geometric prior for where an
    advected plume can be. The source pixel itself is included.

    With ``d = (x − x₀, y − y₀)`` the offset of a pixel center from the
    source and ``ŵ = (u, v) / ‖(u, v)‖`` the unit wind vector, a pixel is
    inside when

        ‖d‖ ≤ max_distance  and  d·ŵ ≥ ‖d‖ · cos(half_angle_deg)

    i.e. the angle between ``d`` and ``ŵ`` is at most ``half_angle_deg``.
    For ``half_angle_deg ≤ 90`` the angle test alone keeps the sector in
    the downwind half-plane (``cos ≥ 0 ⇒ d·ŵ ≥ 0``); ``half_angle_deg > 90``
    widens it past the crosswind line, and ``half_angle_deg = 180`` is the
    full disc of radius ``max_distance``, upwind pixels included.

    Args:
        shape: Raster shape ``(H, W)``.
        transform: Affine-like geotransform of the raster; its CRS units
            must match ``source`` and ``max_distance``.
        source: ``(x, y)`` source coordinates in CRS units.
        wind_u: Eastward wind component.
        wind_v: Northward wind component.
        half_angle_deg: Half-angle of the sector in degrees, in [0, 180].
        max_distance: Sector radius in CRS units.

    Returns:
        Boolean ``(H, W)`` mask, True inside the downwind sector.

    Raises:
        ValueError: If the wind vector is zero, ``half_angle_deg`` is
            outside [0, 180], or ``max_distance`` is not positive.
    """
    wind_norm = float(np.hypot(wind_u, wind_v))
    if wind_norm == 0.0:
        raise ValueError("wind vector must be non-zero")
    if not 0.0 <= half_angle_deg <= 180.0:
        raise ValueError("half_angle_deg must be in [0, 180]")
    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive")

    xs, ys = pixel_centers(shape, transform)
    dx = xs - source[0]
    dy = ys - source[1]
    distances = np.hypot(dx, dy)
    projection = (dx * wind_u + dy * wind_v) / wind_norm
    cos_angle = np.divide(
        projection,
        distances,
        out=np.ones_like(distances, dtype=float),
        where=distances > 0,
    )
    # Clip rounding overshoot so half_angle_deg=180 (cos = −1) is the full disc.
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    min_cos = np.cos(np.deg2rad(half_angle_deg))
    return (distances <= max_distance) & (cos_angle >= min_cos)


def convert_column_units(
    values: Num[np.ndarray, "*dims"],
    *,
    gas: str = "CH4",
    units_in: ColumnUnit = "ppm_m",
    units_out: ColumnUnit = "kg_m2",
) -> Float[np.ndarray, "*dims"]:
    r"""Convert column enhancement among ppm m, mol/m^2, and kg/m^2.

    The conversions are

    .. math::

        \Omega_{\mathrm{mol/m^2}} \;=\; \frac{X \cdot 10^{-6}}{V_m}, \qquad
        \Omega_{\mathrm{kg/m^2}}  \;=\; M_{\mathrm{gas}}
                                        \cdot \Omega_{\mathrm{mol/m^2}}

    where ``X`` is the column-integrated volume mixing ratio in ppm m,
    :math:`V_m = 0.024465` m^3/mol is the standard molar volume of an
    ideal gas at 298.15 K and 1 atm, and :math:`M_{\mathrm{gas}}` is the
    molar mass (CH4: 0.01604 kg/mol; CO2: 0.04401 kg/mol).

    The ppm m branch therefore assumes near-surface conditions. Pass
    ``mol_m2`` or ``kg_m2`` inputs when retrieval-specific pressure and
    temperature corrections have already been applied upstream (e.g. by
    a matched-filter retrieval per Thompson 2015 / Foote 2020).
    """
    gas_key = gas.upper()
    if gas_key not in MOLAR_MASS_KG_PER_MOL:
        supported = ", ".join(sorted(MOLAR_MASS_KG_PER_MOL))
        raise ValueError(f"unsupported gas {gas!r}; expected one of {supported}")
    molar_mass = MOLAR_MASS_KG_PER_MOL[gas_key]

    arr = np.asarray(values, dtype=float)
    if units_in == "ppm_m":
        # 1e-6 converts ppm to a fraction; molar volume assumes standard air.
        mol_m2 = arr * 1e-6 / STANDARD_MOLAR_VOLUME_M3_PER_MOL
    elif units_in == "mol_m2":
        mol_m2 = arr
    elif units_in == "kg_m2":
        mol_m2 = arr / molar_mass
    else:
        raise ValueError("units_in must be 'ppm_m', 'mol_m2', or 'kg_m2'")

    if units_out == "ppm_m":
        return mol_m2 * STANDARD_MOLAR_VOLUME_M3_PER_MOL * 1e6
    if units_out == "mol_m2":
        return mol_m2
    if units_out == "kg_m2":
        return mol_m2 * molar_mass
    raise ValueError("units_out must be 'ppm_m', 'mol_m2', or 'kg_m2'")


def plume_length(
    mask: Bool[np.ndarray, "h w"],
    transform: Any,
    *,
    method: Literal["max_axis", "convex_hull", "skeleton"] = "max_axis",
) -> float:
    """Estimate plume length ``L`` from active pixel centers.

    ``L`` is the effective length used by Varon et al. (2018) in the IME
    method ``Q = U_eff * IME / L``. Three estimators are supported:

    - ``"max_axis"``: maximum pairwise distance between active pixels.
      Fast and robust; matches the original Varon 2018 definition for
      reasonably linear plumes.
    - ``"convex_hull"``: max diameter of the convex hull of the active
      pixels. For degenerate hulls (single point, collinear points) we
      fall back to the point-set diameter so the result is always well
      defined.
    - ``"skeleton"``: length of the plume centreline, via
      :func:`geotoolz.measure.skeleton_length` with the transform as the
      pixel step. The mask is skeletonised and ``L`` is the longest
      geodesic path through the 8-connected skeleton,
      with each step weighted by its Euclidean length in CRS units
      (so diagonal steps and anisotropic pixels are measured
      correctly). Better than the chord for curved plumes. The path
      runs between skeleton pixel centres, so for thick plumes it is
      shorter than the mask extent by roughly the plume width.

    Lengths are in CRS units (m for a projected metric CRS) and measured
    between pixel centres.

    A single-pixel mask returns one linear pixel size,
    ``sqrt(pixel_area)``, for every method: it has no geometry to
    measure, and the one-pixel proxy is the documented convention.

    Args:
        mask: Boolean ``(H, W)`` plume mask.
        transform: Affine-like geotransform (``a``, ``b``, ``d``, ``e``
            coefficients) mapping pixel indices to CRS coordinates.
        method: Length estimator (see above).

    Returns:
        The plume length in CRS units; ``0.0`` for an empty mask.

    Raises:
        ValueError: If ``method`` is unknown, or if ``method="skeleton"``
            and a multi-pixel mask yields ``L == 0`` (every skeleton
            component collapses to a single pixel, e.g. a compact blob a
            few pixels across or scattered isolated pixels). A one-pixel
            proxy would badly under-estimate ``L`` (and over-estimate
            ``Q``) there, so use ``"max_axis"`` or ``"convex_hull"``.
    """
    active = np.asarray(mask, dtype=bool)
    if not active.any():
        return 0.0
    xs, ys = pixel_centers(active.shape, transform)
    points = np.column_stack([xs[active], ys[active]])
    if points.shape[0] == 1:
        # Single pixel: use one linear pixel size as the length proxy.
        return float(np.sqrt(pixel_area(transform)))
    if method == "max_axis":
        diff = points[:, None, :] - points[None, :, :]
        return float(np.sqrt(np.max(np.sum(diff**2, axis=-1))))
    if method == "convex_hull":
        hull = MultiPoint(points).convex_hull
        # For polygonal hulls use the exterior ring; for degenerate
        # hulls (Point, LineString) shapely exposes ``.coords`` instead.
        exterior = getattr(hull, "exterior", None)
        if exterior is not None:
            coords = np.asarray(exterior.coords)
        elif hasattr(hull, "coords"):
            coords = np.asarray(hull.coords)
        else:
            coords = points
        diff = coords[:, None, :] - coords[None, :, :]
        return float(np.sqrt(np.max(np.sum(diff**2, axis=-1))))
    if method == "skeleton":
        length = skeleton_length(active, step=transform)
        if length <= 0.0:
            raise ValueError(
                f"skeleton plume length is 0 for a non-empty {int(active.sum())}-"
                "pixel mask: every skeleton component collapses to a single "
                "pixel (compact blob or isolated pixels). Use "
                "length_method='max_axis' or 'convex_hull' for such plumes."
            )
        return length
    raise ValueError("length_method must be 'max_axis', 'convex_hull', or 'skeleton'")
