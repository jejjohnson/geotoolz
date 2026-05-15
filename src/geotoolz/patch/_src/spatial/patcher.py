"""`SpatialPatcher` — composes the four spatial axes.

The Patcher is intentionally tiny — it just orchestrates
``SpatialSampler.anchors → Geometry.neighborhood → SpatialWindow.weights →
Field.select`` and hands the result to `SpatialAggregation.merge` when the
caller asks. Split returns an `Iterator[Patch]` so streaming is the
default; ``list(patcher.split(field))`` materialises eagerly when that's
what's wanted.

See ``design.md`` §1 for the four-axis framework.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from geotoolz.patch._src.patch import Patch
from geotoolz.patch._src.protocols import AsyncField, Field
from geotoolz.patch._src.spatial.aggregation import (
    SpatialAggregation,
    _warn_if_unsafe_streaming,
)
from geotoolz.patch._src.spatial.geometry import (
    SpatialGeometry,
    _is_raster_domain,
    _MaskedWindow,
)
from geotoolz.patch._src.spatial.sampler import SpatialSampler
from geotoolz.patch._src.spatial.window import SpatialWindow


@dataclass(eq=False)
class SpatialPatcher:
    """The four-axis spatial Patcher.

    Args:
        geometry: How a neighborhood is shaped around an anchor.
        sampler: Where anchors go.
        window: Boundary treatment / per-pixel weights.
        aggregation: Local → global merge strategy.

    Examples:
        Sliding-window inference over a raster::

            patcher = SpatialPatcher(
                geometry    = SpatialRectangular(size=(256, 256)),
                sampler     = SpatialRegularStride(step=(192, 192)),
                window      = SpatialHann(),
                aggregation = SpatialOverlapAdd(),
            )
            patches = list(patcher.split(field))
            outs    = [run_operator(p) for p in patches]
            stitched = patcher.merge(outs, field.domain)
    """

    geometry: SpatialGeometry
    sampler: SpatialSampler
    window: SpatialWindow
    aggregation: SpatialAggregation

    def split(self, field: Field) -> Iterator[Patch]:
        """Yield patches lazily — one per anchor placed by the sampler."""
        domain = field.domain
        try:
            base_weights = self.window.weights(self.geometry)
        except TypeError:
            base_weights = None
        for anchor in self.sampler.anchors(domain, self.geometry):
            indices = self.geometry.neighborhood(domain, anchor)
            data = field.select(indices)
            weights = _build_weights(indices, base_weights)
            yield Patch(data=data, anchor=anchor, indices=indices, weights=weights)

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        """Hand off to the aggregation; warn on streaming-unsafe types."""
        _warn_if_unsafe_streaming(self.aggregation)
        return self.aggregation.merge(patches, domain)

    def get_config(self) -> dict[str, Any]:
        return {
            "geometry": {
                "class": type(self.geometry).__name__,
                "config": self.geometry.get_config(),
            },
            "sampler": {
                "class": type(self.sampler).__name__,
                "config": self.sampler.get_config(),
            },
            "window": {
                "class": type(self.window).__name__,
                "config": self.window.get_config(),
            },
            "aggregation": {
                "class": type(self.aggregation).__name__,
                "config": self.aggregation.get_config(),
            },
        }


@dataclass(eq=False)
class AsyncSpatialPatcher:
    """Async mirror of `SpatialPatcher` over an `AsyncField`.

    `split` is an ``async for``-able iterator. Useful with
    `AsyncGeoTIFFReader` for high-concurrency per-tile fan-out.
    """

    geometry: SpatialGeometry
    sampler: SpatialSampler
    window: SpatialWindow
    aggregation: SpatialAggregation

    async def split(self, field: AsyncField) -> AsyncIterator[Patch]:
        domain = field.domain
        try:
            base_weights = self.window.weights(self.geometry)
        except TypeError:
            base_weights = None
        for anchor in self.sampler.anchors(domain, self.geometry):
            indices = self.geometry.neighborhood(domain, anchor)
            data = await field.select(indices)
            weights = _build_weights(indices, base_weights)
            yield Patch(data=data, anchor=anchor, indices=indices, weights=weights)

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        _warn_if_unsafe_streaming(self.aggregation)
        return self.aggregation.merge(patches, domain)


def _build_weights(indices: Any, base_weights: np.ndarray | None) -> Any:
    """Resolve a patch's weight array.

    If the indices is a `_MaskedWindow` (SpatialPolygonIntersection on a raster),
    return the interior mask — the window controls *which pixels count*,
    not how heavily they're tapered. Otherwise return the geometry-shaped
    base weights from `SpatialWindow.weights`.
    """
    if isinstance(indices, _MaskedWindow):
        return indices.mask
    return base_weights


# Re-export `_is_raster_domain` to discourage cross-imports from geometry.py.
__all__ = [
    "AsyncSpatialPatcher",
    "SpatialPatcher",
    "_is_raster_domain",
]
