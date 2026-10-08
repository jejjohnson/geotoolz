"""Operator bridge to `geopatcher` — tile, map, merge and label-aware sampling.

`GridSampler`, `ApplyToChips` and `MergePatches` are re-exported from
`geopatcher.integrations.pipekit` — they are the *same* classes, so
``isinstance`` checks and ``forbid_in_yaml`` semantics agree across both
import paths. They put the four-axis Patcher framework inside a
`Sequential` / `Graph`::

    pipe = Sequential([
        GridSampler(patcher=patcher),
        ApplyToChips(operator=model_op),
        MergePatches(aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain),
    ])

``MergePatches`` is geopatcher's ``Stitch`` operator under a name that
does not collide with `geotoolz.geom.Stitch` (the GeoTensor tile
mosaicker also exported as top-level ``geotoolz.Stitch``). Its class
``__name__`` is still ``"Stitch"``.

geotoolz's own additions:

- `SpatialTriangular` — a linear-ramp `geopatcher.spatial.window.Window` whose
  overlap-add blend matches ``geom.Stitch(blend="feather")``.
- `StratifiedSample` (class proportions matching a target distribution,
  largest-remainder allocation) and `BalancedSampler` (N chips per
  class) — label-aware training-time samplers. They classify each
  candidate chip by the label under its centre pixel and emit
  ``list[Patch]``, so their output feeds straight into `ApplyToChips`.

Optional extra: ``pip install 'geotoolz[patch]'`` to pull in
``geotoolz-patcher[pipekit]`` (which transitively installs `pipekit`).
Importing this module without geopatcher installed raises a friendly
``ImportError`` pointing at the right extra.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from georeader.geotensor import GeoTensor
from pipekit import Operator
from rasterio.windows import Window

from geotoolz._src.blending import triangular_weights


try:
    from geopatcher import Patch, spatial
    from geopatcher.config import config_from_fields
    from geopatcher.integrations.pipekit import (
        ApplyToChips,
        GridSampler,
        Stitch as MergePatches,
    )
    from geopatcher.spatial.window import geom_shape
except ImportError as _e:  # pragma: no cover - exercised when [patch] is missing
    raise ImportError(
        "geotoolz.patch_ops requires the `geopatcher` package (distribution "
        "`geotoolz-patcher`). Install with `pip install 'geotoolz[patch]'`."
    ) from _e


__all__ = [
    "ApplyToChips",
    "BalancedSampler",
    "GridSampler",
    "MergePatches",
    "SpatialTriangular",
    "StratifiedSample",
]


@dataclass(eq=False)
class SpatialTriangular(spatial.window.Window):
    """Linear-ramp triangular spatial window for overlap-add blending.

    Each axis ramps linearly from the edge up to a plateau of 1.0 over
    ``width`` pixels (edge pixel ``= 1 / width``); axes combine by
    product. This is the kernel ``geom.Stitch(blend="feather")`` uses,
    returned as ``float64`` like every other `spatial.window.Window`.

    Args:
        width: Ramp width in pixels. ``width <= 0`` gives a boxcar.
    """

    width: int = 16

    def weights(self, geometry: spatial.geometry.Geometry) -> np.ndarray:
        """Return triangular weights for a fixed-size spatial geometry.

        Args:
            geometry: A fixed-size geometry such as
                ``spatial.geometry.Rectangular(size=(height, width))``.

        Returns:
            ``float64`` array of shape ``geom_shape(geometry)``.

        Raises:
            TypeError: if the geometry has no fixed size.

        Examples:
            >>> from geopatcher import spatial
            >>> SpatialTriangular(width=2).weights(
            ...     spatial.geometry.Rectangular(size=(5, 7))
            ... ).shape
            (5, 7)
        """
        return triangular_weights(geom_shape(geometry), self.width, dtype=np.float64)

    def get_config(self) -> dict[str, Any]:
        return config_from_fields(self)


def _labels_2d(labels: Any) -> np.ndarray:
    """Coerce a label carrier into a 2-D ``(H, W)`` numpy array."""
    arr = np.asarray(getattr(labels, "values", labels))
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(
            f"labels must be a single-band (H, W) raster, got shape {arr.shape}."
        )
    return arr


def _chip(scene: Any, r: int, c: int, ph: int, pw: int) -> Any:
    """Slice a ``(ph, pw)`` chip out of a scene, carrier-preserving."""
    if isinstance(scene, GeoTensor):
        return scene.isel({"y": slice(r, r + ph), "x": slice(c, c + pw)})
    return np.asarray(scene)[..., r : r + ph, c : c + pw]


def _center_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Label under the centre pixel of every valid UL anchor, as a 2-D view."""
    h, w = labels.shape
    ph, pw = size
    if ph > h or pw > w:
        raise ValueError(
            f"patch size {size} exceeds the label raster's shape {(h, w)}."
        )
    return labels[ph // 2 : ph // 2 + h - ph + 1, pw // 2 : pw // 2 + w - pw + 1]


def _draw_anchors(
    centers: np.ndarray,
    counts: dict[int, int],
    rng: np.random.Generator,
    op_name: str,
) -> list[tuple[int, int]]:
    """Draw ``counts[cls]`` anchors per class from the centre-label grid.

    Warns and returns fewer anchors when a class has fewer candidate
    positions than requested.
    """
    anchors: list[tuple[int, int]] = []
    width = centers.shape[1]
    for cls in sorted(counts):
        want = counts[cls]
        if want == 0:
            continue
        pool = np.flatnonzero(centers.ravel() == cls)
        if len(pool) < want:
            warnings.warn(
                f"{op_name}: class {cls} has only {len(pool)} candidate "
                f"patches, fewer than the {want} requested.",
                stacklevel=2,
            )
            want = len(pool)
        if want == 0:
            continue
        chosen = rng.choice(pool, size=want, replace=False)
        anchors.extend((int(i // width), int(i % width)) for i in chosen)
    return anchors


def _extract_patches(
    scene: Any, anchors: list[tuple[int, int]], size: tuple[int, int]
) -> list[Patch]:
    ph, pw = size
    h, w = np.asarray(getattr(scene, "values", scene)).shape[-2:]
    out: list[Patch] = []
    for r, c in anchors:
        if not (0 <= r <= h - ph and 0 <= c <= w - pw):
            raise ValueError(
                f"anchor {(r, c)} with size {size} overflows the scene "
                f"shape {(h, w)}; labels and scene must share a pixel grid."
            )
        window = Window(col_off=c, row_off=r, width=pw, height=ph)
        out.append(
            Patch(data=_chip(scene, r, c, ph, pw), anchor=(r, c), indices=window)
        )
    return out


def _largest_remainder(proportions: dict[int, float], n_samples: int) -> dict[int, int]:
    """Allocate ``n_samples`` across classes by the largest-remainder method."""
    classes = sorted(proportions)
    quotas = {cls: n_samples * proportions[cls] for cls in classes}
    counts = {cls: int(np.floor(quotas[cls])) for cls in classes}
    short = n_samples - sum(counts.values())
    by_remainder = sorted(
        classes, key=lambda cls: (quotas[cls] - counts[cls], -cls), reverse=True
    )
    for cls in by_remainder[:short]:
        counts[cls] += 1
    return counts


class StratifiedSample(Operator):
    """Operator: ``scene → list[Patch]`` with class proportions matching a target.

    Each candidate chip (every upper-left anchor at which a full
    ``size`` patch fits) is classified by the label under its centre
    pixel. The requested ``n_samples`` is split across classes with the
    largest-remainder method, then chips are drawn uniformly without
    replacement within each class. Classes with too few candidates
    contribute what they have (with a warning), so the output can be
    shorter than ``n_samples``.

    The output is a ``list[Patch]`` — the same carrier `GridSampler`
    emits — so it composes with `ApplyToChips` downstream. For a
    `GeoTensor` scene each chip keeps a correctly shifted transform.

    Args:
        labels: Single-band label raster (`GeoTensor` or ``(H, W)``
            array) on the same pixel grid as the scenes this operator
            will be applied to.
        target_proportions: ``{class_value: proportion}``; proportions
            must be non-negative and sum to 1.
        n_samples: Total number of chips to draw.
        size: Chip size ``(height, width)`` in pixels.
        seed: Optional seed for reproducible draws.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        labels: Any,
        target_proportions: dict[int, float],
        n_samples: int,
        size: tuple[int, int],
        seed: int | None = None,
    ) -> None:
        if not target_proportions:
            raise ValueError("target_proportions must not be empty.")
        if any(p < 0 for p in target_proportions.values()):
            raise ValueError("target_proportions must be non-negative.")
        total = float(sum(target_proportions.values()))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"target_proportions must sum to 1, got {total}.")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}.")
        self.labels = labels
        self.target_proportions = {
            int(k): float(v) for k, v in target_proportions.items()
        }
        self.n_samples = n_samples
        self.size = (int(size[0]), int(size[1]))
        self.seed = seed

    def _apply(self, scene: Any) -> list[Patch]:
        centers = _center_labels(_labels_2d(self.labels), self.size)
        counts = _largest_remainder(self.target_proportions, self.n_samples)
        rng = np.random.default_rng(self.seed)
        anchors = _draw_anchors(centers, counts, rng, type(self).__name__)
        return _extract_patches(scene, anchors, self.size)

    def get_config(self) -> dict[str, Any]:
        return {
            "target_proportions": {
                str(k): v for k, v in self.target_proportions.items()
            },
            "n_samples": self.n_samples,
            "size": list(self.size),
            "seed": self.seed,
            "labels_shape": list(_labels_2d(self.labels).shape),
        }


class BalancedSampler(Operator):
    """Operator: ``scene → list[Patch]`` with N chips per class label.

    The per-class variant of `StratifiedSample`: draw exactly
    ``n_per_class`` chips for every class (chips are classified by the
    label under their centre pixel). Classes with too few candidate
    positions contribute what they have, with a warning.

    Args:
        labels: Single-band label raster (`GeoTensor` or ``(H, W)``
            array) on the same pixel grid as the scenes this operator
            will be applied to.
        n_per_class: Number of chips to draw per class.
        size: Chip size ``(height, width)`` in pixels.
        classes: Class values to sample. Defaults to every finite value
            present under a candidate chip centre.
        seed: Optional seed for reproducible draws.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self,
        *,
        labels: Any,
        n_per_class: int,
        size: tuple[int, int],
        classes: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> None:
        if n_per_class <= 0:
            raise ValueError(f"n_per_class must be positive, got {n_per_class}.")
        self.labels = labels
        self.n_per_class = n_per_class
        self.size = (int(size[0]), int(size[1]))
        self.classes = None if classes is None else [int(c) for c in classes]
        self.seed = seed

    def _apply(self, scene: Any) -> list[Patch]:
        centers = _center_labels(_labels_2d(self.labels), self.size)
        if self.classes is None:
            present = np.unique(centers)
            classes = [int(c) for c in present[np.isfinite(present)]]
        else:
            classes = self.classes
        counts = {cls: self.n_per_class for cls in classes}
        rng = np.random.default_rng(self.seed)
        anchors = _draw_anchors(centers, counts, rng, type(self).__name__)
        return _extract_patches(scene, anchors, self.size)

    def get_config(self) -> dict[str, Any]:
        return {
            "n_per_class": self.n_per_class,
            "size": list(self.size),
            "classes": self.classes,
            "seed": self.seed,
            "labels_shape": list(_labels_2d(self.labels).shape),
        }
