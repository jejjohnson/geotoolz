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

from heapq import heappop, heappush
from typing import Any, Literal

import numpy as np
from jaxtyping import Bool, Float, Int, Num, Shaped
from scipy import ndimage
from shapely.geometry import MultiPoint

from geotoolz._src.shape import single_band


ThresholdMode = float | int | str
Connectivity = Literal[4, 8]
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


def otsu_threshold(values: Num[np.ndarray, "*dims"], *, nbins: int = 256) -> float:
    """Compute Otsu's between-class-variance threshold, ignoring NaNs.

    Builds an ``nbins``-bin histogram of the finite values and returns
    the bin center that maximises the between-class variance
    ``w_bg * w_fg * (mu_bg - mu_fg)^2`` (Otsu, 1979). A constant input
    returns that constant.

    Args:
        values: Array of any shape; non-finite entries are ignored.
        nbins: Number of histogram bins. Default ``256``.

    Returns:
        The threshold; pixels strictly above it are foreground.

    Raises:
        ValueError: If ``values`` contains no finite entries.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot compute an Otsu threshold on all-NaN data")
    if np.all(finite == finite[0]):
        return float(finite[0])

    hist, bin_edges = np.histogram(finite, bins=nbins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = finite.size - weight_bg

    valid = (weight_bg > 0) & (weight_fg > 0)
    mean_bg = np.divide(
        np.cumsum(hist * centers),
        weight_bg,
        out=np.zeros_like(centers, dtype=float),
        where=weight_bg > 0,
    )
    mean_fg = np.divide(
        np.cumsum((hist * centers)[::-1])[::-1] - hist * centers,
        weight_fg,
        out=np.zeros_like(centers, dtype=float),
        where=weight_fg > 0,
    )
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    variance[~valid] = -np.inf
    return float(centers[int(np.argmax(variance))])


def resolve_threshold(
    values: Num[np.ndarray, "*dims"], threshold: ThresholdMode
) -> float:
    """Resolve an absolute, Otsu, or percentile threshold to a float.

    Args:
        values: Data the data-driven modes are evaluated on.
        threshold: A number (returned as-is), ``"otsu"`` (see
            :func:`otsu_threshold`), or ``"percentile:<p>"`` with ``p``
            in ``[0, 100]`` (NaN-aware percentile of ``values``).

    Returns:
        The resolved threshold value.

    Raises:
        ValueError: If a string threshold is neither ``"otsu"`` nor a
            valid ``"percentile:<p>"`` spec.
    """
    if isinstance(threshold, str):
        if threshold == "otsu":
            return otsu_threshold(values)
        prefix = "percentile:"
        if threshold.startswith(prefix):
            percentile = float(threshold.removeprefix(prefix))
            if not 0.0 <= percentile <= 100.0:
                raise ValueError("percentile threshold must be in [0, 100]")
            return float(np.nanpercentile(values, percentile))
        raise ValueError("threshold must be a number, 'otsu', or 'percentile:<p>'")
    return float(threshold)


def connectivity_structure(connectivity: Connectivity) -> Bool[np.ndarray, "3 3"]:
    """Return a 2-D connected-component structure for 4- or 8-connectivity.

    Args:
        connectivity: ``4`` (edge neighbors) or ``8`` (edge + diagonal
            neighbors).

    Returns:
        A ``(3, 3)`` boolean structuring element suitable for
        :func:`scipy.ndimage.label`.

    Raises:
        ValueError: If ``connectivity`` is not 4 or 8.
    """
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError("connectivity must be 4 or 8")


def label_components(
    mask: Bool[np.ndarray, "h w"],
    *,
    min_area: int = 1,
    connectivity: Connectivity = 8,
) -> Int[np.ndarray, "h w"]:
    """Label connected True regions and drop components below ``min_area``.

    Uses :func:`scipy.ndimage.label` for the connectivity, then renumbers
    the surviving components contiguously (1..K) without a second
    labelling pass — dropping pixels from a labelled image cannot merge
    distinct components.

    Args:
        mask: 2-D boolean map (any array-like is coerced to bool).
        min_area: Minimum component size in pixels; smaller components
            are mapped to background.
        connectivity: 4 or 8 connectivity for component labelling.

    Returns:
        An int32 label image with contiguous labels ``1..K`` for the
        surviving components and ``0`` for background.

    Raises:
        ValueError: If ``min_area`` is smaller than 1 or ``connectivity``
            is not 4 or 8.
    """
    if min_area < 1:
        raise ValueError("min_area must be >= 1")
    labels, n_labels = ndimage.label(
        np.asarray(mask, dtype=bool), structure=connectivity_structure(connectivity)
    )
    if n_labels == 0:
        return labels.astype(np.int32, copy=False)

    counts = np.bincount(labels.ravel())
    keep = counts >= min_area
    keep[0] = False
    # Build a 0..K renumbering LUT so labels remain contiguous after
    # dropping small components.
    lut = np.zeros_like(counts, dtype=np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lut[labels]


def plume_mask(
    values: Num[np.ndarray, "h w"] | Num[np.ndarray, "1 h w"],
    *,
    threshold: ThresholdMode = "otsu",
    min_area: int = 50,
    connectivity: Connectivity = 8,
) -> Bool[np.ndarray, "h w"]:
    """Threshold an enhancement map and remove small connected components.

    Args:
        values: Single-band enhancement map, ``(H, W)`` or ``(1, H, W)``.
        threshold: Absolute number, ``"otsu"``, or ``"percentile:<p>"``;
            see :func:`resolve_threshold`.
        min_area: Minimum connected-component size in pixels.
        connectivity: 4 or 8 connectivity for component labelling.

    Returns:
        Boolean ``(H, W)`` mask of the surviving plume pixels.
    """
    arr = squeeze_single_band(values)
    cutoff = resolve_threshold(arr, threshold)
    raw = np.asarray(arr) > cutoff
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
    xs = transform.c + transform.a * (cols + 0.5) + transform.b * (rows + 0.5)
    ys = transform.f + transform.d * (cols + 0.5) + transform.e * (rows + 0.5)
    return xs, ys


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
    min_cos = np.cos(np.deg2rad(half_angle_deg))
    return (projection >= 0.0) & (distances <= max_distance) & (cos_angle >= min_cos)


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
    - ``"skeleton"``: longest 4-connected pixel path through the plume,
      better for curved plumes where the chord underestimates ``L``.
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
        return _longest_active_pixel_path(active, transform)
    raise ValueError("length_method must be 'max_axis', 'convex_hull', or 'skeleton'")


def _longest_active_pixel_path(mask: Bool[np.ndarray, "h w"], transform: Any) -> float:
    """Approximate centerline length as the longest 4-neighbor pixel path.

    The 4-neighbor graph avoids diagonal corner-cutting through plume
    pixels, so bent plumes are measured along their active-pixel path
    rather than by a straight endpoint chord.

    The starting node for the double-BFS is chosen deterministically as
    the lexicographically smallest ``(row, col)`` so the resulting length
    does not depend on set iteration order. For masks with multiple
    disconnected components, each component is processed independently
    and the maximum path length across components is returned.
    """
    rows, cols = np.nonzero(mask)
    nodes = {(int(r), int(c)) for r, c in zip(rows, cols, strict=True)}
    if not nodes:
        return 0.0
    longest = 0.0
    remaining = nodes
    while remaining:
        # Deterministic start: lexicographically smallest (topmost, then
        # leftmost) pixel in the remaining component pool.
        seed = min(remaining)
        component = _connected_component(seed, remaining)
        farthest, _ = _farthest_active_pixel(seed, component, transform)
        _, distance = _farthest_active_pixel(farthest, component, transform)
        if distance > longest:
            longest = distance
        remaining = remaining - component
    return float(longest)


def _connected_component(
    seed: tuple[int, int], nodes: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Return the 4-connected component of ``seed`` within ``nodes``."""
    component: set[tuple[int, int]] = {seed}
    stack: list[tuple[int, int]] = [seed]
    while stack:
        row, col = stack.pop()
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + drow, col + dcol)
            if neighbor in nodes and neighbor not in component:
                component.add(neighbor)
                stack.append(neighbor)
    return component


def _farthest_active_pixel(
    start: tuple[int, int],
    nodes: set[tuple[int, int]],
    transform: Any,
) -> tuple[tuple[int, int], float]:
    distances = {start: 0.0}
    heap = [(0.0, start)]
    farthest = start
    while heap:
        distance, node = heappop(heap)
        if distance != distances[node]:
            continue
        farthest = node
        row, col = node
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                # 4-neighbor connectivity: exclude the center and diagonals.
                if abs(drow) + abs(dcol) != 1:
                    continue
                neighbor = (row + drow, col + dcol)
                if neighbor not in nodes:
                    continue
                step_x = transform.a * dcol + transform.b * drow
                step_y = transform.d * dcol + transform.e * drow
                new_distance = distance + float(np.hypot(step_x, step_y))
                if new_distance < distances.get(neighbor, np.inf):
                    distances[neighbor] = new_distance
                    heappush(heap, (new_distance, neighbor))
    return farthest, distances[farthest]
