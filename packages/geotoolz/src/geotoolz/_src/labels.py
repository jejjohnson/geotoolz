"""Shared connected-component, hole, skeleton and region-property primitives.

The one implementation of the region-analysis building blocks that the
``mask``, ``measure`` and ``plume`` families all need:

* :func:`label_components` -- connected-component labelling with a
  small-component filter and contiguous ``1..K`` renumbering;
* :func:`remove_small_holes` -- fill small background components, with
  a flag for whether holes touching the image border count;
* :func:`skeleton_length` -- longest Euclidean path through the skeleton
  of a binary mask, in pixel steps or CRS units;
* :func:`regionprops_frame` / :data:`DEFAULT_REGIONPROPS` -- the
  :func:`skimage.measure.regionprops_table` call behind
  ``measure.RegionProps`` and ``plume.PlumeFootprint``.

Connectivity is always spelled ``4`` (edge neighbours) or ``8`` (edge +
diagonal neighbours); skimage's ``1`` / ``2`` spelling is not accepted.
All functions work on a single 2-D ``(H, W)`` plane; batching over
leading axes is the caller's job.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from heapq import heappop, heappush
from typing import Any, Literal

import numpy as np
import pandas as pd
from jaxtyping import Bool, Int, Shaped
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.measure import regionprops_table
from skimage.morphology import skeletonize


__all__ = [
    "DEFAULT_REGIONPROPS",
    "Connectivity",
    "connectivity_structure",
    "label_components",
    "regionprops_frame",
    "remove_small_holes",
    "skeleton_length",
]


Connectivity = Literal[4, 8]

DEFAULT_REGIONPROPS: tuple[str, ...] = (
    "label",
    "area",
    "area_convex",
    "area_filled",
    "centroid",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "eccentricity",
    "solidity",
    "perimeter",
    "bbox",
    "inertia_tensor_eigvals",
)
"""Region properties computed by default by ``measure.RegionProps`` and
``plume.PlumeFootprint``. All are in pixel units (areas in pixels,
lengths in pixel widths, positions as ``(row, col)`` indices)."""


def connectivity_structure(connectivity: Connectivity) -> Bool[np.ndarray, "3 3"]:
    """Return a 2-D connected-component structure for 4- or 8-connectivity.

    Args:
        connectivity: ``4`` (edge neighbours) or ``8`` (edge + diagonal
            neighbours).

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
    raise ValueError(
        f"connectivity must be 4 or 8, got {connectivity!r} (skimage's 1 / 2 "
        "spelling is not accepted: 1 -> 4, 2 -> 8)"
    )


def label_components(
    mask: Bool[np.ndarray, "h w"],
    *,
    connectivity: Connectivity = 8,
    min_area_px: int = 0,
) -> Int[np.ndarray, "h w"]:
    """Label connected True regions and drop components below ``min_area_px``.

    Uses :func:`scipy.ndimage.label` (raster-order labels, identical to
    :func:`skimage.measure.label` on a boolean mask), then renumbers the
    surviving components contiguously (``1..K``) without a second
    labelling pass -- dropping pixels from a labelled image cannot merge
    distinct components.

    Args:
        mask: 2-D mask; any array-like is coerced with
            ``np.asarray(mask, dtype=bool)``.
        connectivity: ``4`` or ``8`` neighbourhood for component
            membership.
        min_area_px: Minimum component size in pixels; smaller components
            become background. ``0`` and ``1`` keep every component.

    Returns:
        An ``int32`` label image with contiguous labels ``1..K`` for the
        surviving components and ``0`` for background.

    Raises:
        ValueError: If ``min_area_px`` is negative or ``connectivity`` is
            not 4 or 8.
    """
    if min_area_px < 0:
        raise ValueError(f"min_area_px must be non-negative, got {min_area_px}")
    labels, n_labels = ndimage.label(
        np.asarray(mask, dtype=bool), structure=connectivity_structure(connectivity)
    )
    if n_labels == 0 or min_area_px <= 1:
        return labels.astype(np.int32, copy=False)
    counts = np.bincount(labels.ravel())
    keep = counts >= min_area_px
    keep[0] = False
    # 0..K renumbering LUT so labels stay contiguous after dropping.
    lut = np.zeros_like(counts, dtype=np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lut[labels]


def remove_small_holes(
    mask: Bool[np.ndarray, "h w"],
    *,
    max_area: int,
    connectivity: Connectivity = 4,
    exclude_border: bool = True,
) -> Bool[np.ndarray, "h w"]:
    """Fill background (False) components of at most ``max_area`` pixels.

    Args:
        mask: 2-D boolean mask (any array-like is coerced to bool).
        max_area: Largest hole, in pixels, to fill. ``0`` fills nothing.
        connectivity: ``4`` or ``8`` neighbourhood used to group
            background pixels into holes.
        exclude_border: When ``True`` (default), background components
            touching the image border are not holes and are never
            filled. ``False`` matches
            :func:`skimage.morphology.remove_small_holes`, which also
            fills small border-touching gaps.

    Returns:
        A new boolean mask with the small holes set to True.

    Raises:
        ValueError: If ``max_area`` is negative or ``connectivity`` is not
            4 or 8.
    """
    if max_area < 0:
        raise ValueError(f"max_area must be non-negative, got {max_area}")
    arr = np.asarray(mask, dtype=bool)
    labels, n_labels = ndimage.label(
        ~arr, structure=connectivity_structure(connectivity)
    )
    if n_labels == 0 or max_area == 0:
        return arr.copy()
    fill = np.bincount(labels.ravel(), minlength=n_labels + 1) <= max_area
    fill[0] = False
    if exclude_border:
        border = np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
        fill[border] = False
    return arr | fill[labels]


def skeleton_length(
    mask: Shaped[np.ndarray, "h w"],
    *,
    step: tuple[float, float] | Any = (1.0, 1.0),
    connectivity: Connectivity = 8,
) -> float:
    """Longest geodesic path through the skeleton of a binary mask.

    The mask is skeletonised with :func:`skimage.morphology.skeletonize`
    and each step between neighbouring skeleton pixels is weighted by its
    Euclidean length, so a diagonal step on square unit pixels counts
    ``sqrt(2)`` and anisotropic / rotated grids are measured correctly.
    The path runs between skeleton pixel *centres*: a straight line of
    ``n`` pixels measures ``n - 1`` steps.

    The result is the exact graph diameter of each connected skeleton
    component (the longest of its shortest paths), maximised over
    components. A tree-shaped component uses a double Dijkstra sweep
    (farthest pixel from a seed, then farthest from that), which is exact
    for trees; a component with a loop -- where that sweep can
    under-estimate -- runs Dijkstra from every pixel. The sweep seed is
    the lexicographically smallest ``(row, col)``, so the result is
    deterministic.

    Args:
        mask: 2-D mask. Boolean input is used as is; other dtypes are
            foreground where ``> 0`` (``NaN`` is background).
        step: Pixel size, either a ``(dy, dx)`` pair (row step, column
            step) or an affine geotransform with ``a``, ``b``, ``d``,
            ``e`` coefficients. The default ``(1.0, 1.0)`` returns the
            length in pixel widths; a geotransform returns it in CRS
            units (m for a projected metric CRS).
        connectivity: ``8`` (default) walks diagonal steps too; ``4``
            walks only edge steps.

    Returns:
        The longest path length in the units of ``step``; ``0.0`` for an
        empty mask or when every skeleton component is a single pixel.

    Raises:
        ValueError: If ``connectivity`` is not 4 or 8.
    """
    arr = np.asarray(mask)
    binary = arr if arr.dtype == bool else np.nan_to_num(arr, nan=0.0) > 0
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    if not binary.any():
        return 0.0
    if hasattr(step, "a"):
        a, b, d, e = float(step.a), float(step.b), float(step.d), float(step.e)
    else:
        dy, dx = (float(s) for s in step)
        a, b, d, e = dx, 0.0, 0.0, dy
    # Step length for each (drow, dcol) offset: the transform maps a
    # (dcol, drow) pixel step to (step_x, step_y).
    offsets = _NEIGHBOURS_8 if connectivity == 8 else _NEIGHBOURS_4
    step_length = {
        (drow, dcol): float(np.hypot(a * dcol + b * drow, d * dcol + e * drow))
        for drow, dcol in offsets
    }
    return _longest_path(skeletonize(binary), step_length)


def regionprops_frame(
    labels: Int[np.ndarray, "h w"],
    *,
    intensity_image: Shaped[np.ndarray, "h w"] | None = None,
    properties: Sequence[str] = DEFAULT_REGIONPROPS,
    extra_properties: Sequence[Callable[..., Any]] | None = None,
) -> pd.DataFrame:
    """Per-region property table of an integer label image.

    Thin wrapper around :func:`skimage.measure.regionprops_table`; every
    value is in pixel units (see :data:`DEFAULT_REGIONPROPS`).

    Args:
        labels: 2-D integer label image; ``0`` is background.
        intensity_image: Optional 2-D intensity image aligned with
            ``labels`` for the intensity-based properties.
        properties: Property names to compute.
        extra_properties: Optional custom per-region callables.

    Returns:
        One row per label, columns named as skimage names them
        (``"centroid-0"``, ``"bbox-3"``, ...).
    """
    return pd.DataFrame(
        regionprops_table(
            np.asarray(labels).astype(np.int32, copy=False),
            intensity_image=intensity_image,
            properties=tuple(properties),
            extra_properties=extra_properties,
        )
    )


_NEIGHBOURS_8 = tuple(
    (drow, dcol) for drow in (-1, 0, 1) for dcol in (-1, 0, 1) if (drow, dcol) != (0, 0)
)
_NEIGHBOURS_4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _longest_path(
    mask: Bool[np.ndarray, "h w"], step_length: dict[tuple[int, int], float]
) -> float:
    """Longest geodesic path through the active pixels of ``mask``.

    ``step_length`` maps each allowed ``(drow, dcol)`` neighbour offset to
    its length; it also defines the connectivity. Each connected
    component contributes its exact diameter: a deterministic double
    Dijkstra sweep for a tree, all-sources Dijkstra when it has a loop.
    A single-pixel component contributes ``0``.
    """
    rows, cols = np.nonzero(mask)
    remaining = {(int(r), int(c)) for r, c in zip(rows, cols, strict=True)}
    longest = 0.0
    while remaining:
        seed = min(remaining)
        component = _component(seed, remaining, step_length)
        if _is_tree(component, step_length):
            farthest, _ = _farthest(seed, component, step_length)
            _, distance = _farthest(farthest, component, step_length)
        else:
            distance = _exact_diameter(component, step_length)
        longest = max(longest, distance)
        remaining = remaining - component
    return float(longest)


def _is_tree(
    nodes: set[tuple[int, int]], step_length: dict[tuple[int, int], float]
) -> bool:
    """A connected component is a tree iff it has ``len(nodes) - 1`` edges."""
    edges = sum(
        (row + drow, col + dcol) in nodes
        for row, col in nodes
        for drow, dcol in step_length
    )
    # Every undirected edge is seen from both ends.
    return edges // 2 == len(nodes) - 1


def _exact_diameter(
    nodes: set[tuple[int, int]],
    step_length: dict[tuple[int, int], float],
    *,
    chunk: int = 256,
) -> float:
    """Longest shortest path in a component, by Dijkstra from every node.

    Sources are processed ``chunk`` at a time so memory stays
    ``O(chunk * len(nodes))``.
    """
    order = sorted(nodes)
    index = {node: i for i, node in enumerate(order)}
    src, dst, weight = [], [], []
    for (row, col), i in index.items():
        for (drow, dcol), step in step_length.items():
            j = index.get((row + drow, col + dcol))
            if j is not None:
                src.append(i)
                dst.append(j)
                weight.append(step)
    graph = csr_matrix((weight, (src, dst)), shape=(len(order), len(order)))
    longest = 0.0
    for start in range(0, len(order), chunk):
        sources = np.arange(start, min(start + chunk, len(order)))
        distances = dijkstra(graph, directed=True, indices=sources)
        longest = max(longest, float(distances.max()))
    return longest


def _component(
    seed: tuple[int, int],
    nodes: set[tuple[int, int]],
    step_length: dict[tuple[int, int], float],
) -> set[tuple[int, int]]:
    """Return the connected component of ``seed`` within ``nodes``."""
    component: set[tuple[int, int]] = {seed}
    stack: list[tuple[int, int]] = [seed]
    while stack:
        row, col = stack.pop()
        for drow, dcol in step_length:
            neighbour = (row + drow, col + dcol)
            if neighbour in nodes and neighbour not in component:
                component.add(neighbour)
                stack.append(neighbour)
    return component


def _farthest(
    start: tuple[int, int],
    nodes: set[tuple[int, int]],
    step_length: dict[tuple[int, int], float],
) -> tuple[tuple[int, int], float]:
    """Dijkstra from ``start``; return the farthest node and its distance."""
    distances = {start: 0.0}
    heap = [(0.0, start)]
    farthest = start
    while heap:
        distance, node = heappop(heap)
        if distance != distances[node]:
            continue
        # Nodes pop in non-decreasing distance order, so the last one
        # settled is the farthest.
        farthest = node
        row, col = node
        for (drow, dcol), step in step_length.items():
            neighbour = (row + drow, col + dcol)
            if neighbour not in nodes:
                continue
            new_distance = distance + step
            if new_distance < distances.get(neighbour, np.inf):
                distances[neighbour] = new_distance
                heappush(heap, (new_distance, neighbour))
    return farthest, distances[farthest]
