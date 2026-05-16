"""Pure-numpy plume detection and quantification primitives."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Any, Literal

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPoint


ThresholdMode = float | int | str
Connectivity = Literal[4, 8]
ColumnUnit = Literal["ppm_m", "mol_m2", "kg_m2"]

MOLAR_MASS_KG_PER_MOL = {
    "CH4": 0.01604,
    "CO2": 0.04401,
}
STANDARD_MOLAR_VOLUME_M3_PER_MOL = 0.024465


def squeeze_single_band(values: np.ndarray) -> np.ndarray:
    """Return a 2-D plume map from a 2-D or singleton-band array."""
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(
        "plume operators expect a single-band map with shape (H, W) or (1, H, W)"
    )


def otsu_threshold(values: np.ndarray, *, nbins: int = 256) -> float:
    """Compute Otsu's between-class-variance threshold, ignoring NaNs."""
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


def resolve_threshold(values: np.ndarray, threshold: ThresholdMode) -> float:
    """Resolve an absolute, Otsu, or percentile threshold to a float."""
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


def connectivity_structure(connectivity: Connectivity) -> np.ndarray:
    """Return a 2-D connected-component structure for 4- or 8-connectivity."""
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError("connectivity must be 4 or 8")


def label_components(
    mask: np.ndarray,
    *,
    min_area: int = 1,
    connectivity: Connectivity = 8,
) -> np.ndarray:
    """Label connected True regions and remove components below min_area."""
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
    filtered = np.where(keep[labels], labels, 0)
    relabelled, _ = ndimage.label(
        filtered > 0, structure=connectivity_structure(connectivity)
    )
    return relabelled.astype(np.int32, copy=False)


def plume_mask(
    values: np.ndarray,
    *,
    threshold: ThresholdMode = "otsu",
    min_area: int = 50,
    connectivity: Connectivity = 8,
) -> np.ndarray:
    """Threshold an enhancement map and remove small connected components."""
    arr = squeeze_single_band(values)
    cutoff = resolve_threshold(arr, threshold)
    raw = np.asarray(arr) > cutoff
    return label_components(raw, min_area=min_area, connectivity=connectivity) > 0


def pixel_area(transform: Any) -> float:
    """Return pixel area from an affine-like transform determinant."""
    return float(abs(transform.a * transform.e - transform.b * transform.d))


def pixel_centers(
    shape: tuple[int, int], transform: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinate grids for pixel centers."""
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
) -> np.ndarray:
    """Rasterize an analytical downwind sector mask."""
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
    values: np.ndarray,
    *,
    gas: str = "CH4",
    units_in: ColumnUnit = "ppm_m",
    units_out: ColumnUnit = "kg_m2",
) -> np.ndarray:
    """Convert column enhancement among ppm m, mol/m2, and kg/m2."""
    gas_key = gas.upper()
    if gas_key not in MOLAR_MASS_KG_PER_MOL:
        supported = ", ".join(sorted(MOLAR_MASS_KG_PER_MOL))
        raise ValueError(f"unsupported gas {gas!r}; expected one of {supported}")
    molar_mass = MOLAR_MASS_KG_PER_MOL[gas_key]

    arr = np.asarray(values, dtype=float)
    if units_in == "ppm_m":
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
    mask: np.ndarray,
    transform: Any,
    *,
    method: Literal["max_axis", "convex_hull", "skeleton"] = "max_axis",
) -> float:
    """Estimate plume length from active pixel centers."""
    active = np.asarray(mask, dtype=bool)
    if not active.any():
        return 0.0
    xs, ys = pixel_centers(active.shape, transform)
    points = np.column_stack([xs[active], ys[active]])
    if points.shape[0] == 1:
        return float(np.sqrt(pixel_area(transform)))
    if method == "max_axis":
        diff = points[:, None, :] - points[None, :, :]
        return float(np.sqrt(np.max(np.sum(diff**2, axis=-1))))
    if method == "convex_hull":
        hull = MultiPoint(points).convex_hull
        coords = np.asarray(
            hull.exterior.coords if hasattr(hull, "exterior") else points
        )
        diff = coords[:, None, :] - coords[None, :, :]
        return float(np.sqrt(np.max(np.sum(diff**2, axis=-1))))
    if method == "skeleton":
        return _longest_active_pixel_path(active, transform)
    raise ValueError("length_method must be 'max_axis', 'convex_hull', or 'skeleton'")


def _longest_active_pixel_path(mask: np.ndarray, transform: Any) -> float:
    """Approximate centerline length as the longest path through active pixels."""
    rows, cols = np.nonzero(mask)
    nodes = {(int(r), int(c)) for r, c in zip(rows, cols, strict=True)}
    start = next(iter(nodes))
    farthest, _ = _farthest_active_pixel(start, nodes, transform)
    _, distance = _farthest_active_pixel(farthest, nodes, transform)
    return float(distance)


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
