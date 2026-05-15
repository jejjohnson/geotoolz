"""`SpatioTemporalPatcher` — composes a `SpatialPatcher` and a `TemporalPatcher`.

Two coupling modes:

- ``"product"`` (default) - every spatial anchor crossed with every time anchor.
  The right default for dense gridded data where space and time are
  independent grids (climate model output, regular satellite revisits).
- ``"coupled"`` — explicit ``(space, time)`` anchor pairs. The right
  shape for event-triggered patches (methane plume detections, Argo
  profile (lat, lon, t) records, storm tracks).

The Field is expected to expose a temporal axis as either an integer
``time_len`` attribute or a ``time`` coordinate. The patcher reads the
spatial slice, then the temporal slice, then yields a `SpatioTemporalPatch`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from geotoolz.patch._src.patch import SpatioTemporalPatch
from geotoolz.patch._src.spatial import SpatialPatcher
from geotoolz.patch._src.time.patcher import TemporalPatcher


@dataclass(eq=False)
class SpatioTemporalPatcher:
    """Composition of a spatial and a temporal Patcher.

    Args:
        spatial: A `SpatialPatcher`.
        temporal: A `TemporalPatcher`.
        coupling: ``"product"`` (Cartesian product of anchors) or
            ``"coupled"`` (explicit ``(space, time)`` tuples from the
            spatial sampler's anchors_).
        time_axis: Which axis of the spatial patch's data is the time
            axis after the spatial slice has been read. Default ``0``.
    """

    spatial: SpatialPatcher
    temporal: TemporalPatcher
    coupling: Literal["product", "coupled"] = "product"
    time_axis: int = 0

    def split(self, field: Any) -> Iterator[SpatioTemporalPatch]:
        """Yield `SpatioTemporalPatch`es lazily.

        The coupled mode expects ``self.spatial.sampler.anchors_`` to be
        an iterable of ``(space_anchor, time_anchor)`` tuples and is
        only valid with `SpatialExplicit` spatial / time samplers.
        """
        if self.coupling == "product":
            yield from self._split_product(field)
        elif self.coupling == "coupled":
            yield from self._split_coupled(field)
        else:
            raise ValueError(f"unknown coupling: {self.coupling!r}")

    def _split_product(self, field: Any) -> Iterator[SpatioTemporalPatch]:
        for sp in self.spatial.split(field):
            arr = np.asarray(sp.data)
            time_len = int(arr.shape[self.time_axis])
            for t_anchor in self.temporal.sampler.anchors(time_len):
                t_window = self.temporal.geometry.window(time_len, int(t_anchor))
                slices = t_window if isinstance(t_window, list) else [t_window]
                for s in slices:
                    idx = [slice(None)] * arr.ndim
                    idx[self.time_axis] = s
                    sub = arr[tuple(idx)]
                    yield SpatioTemporalPatch(
                        data=sub,
                        space=sp.anchor,
                        time=int(t_anchor),
                        spatial_indices=sp.indices,
                        temporal_indices=s,
                        weights=sp.weights,
                    )

    def _split_coupled(self, field: Any) -> Iterator[SpatioTemporalPatch]:
        anchors = getattr(self.spatial.sampler, "anchors_", None)
        if anchors is None:
            raise TypeError(
                "coupled coupling requires the spatial sampler to expose an "
                "`anchors_` list of (space_anchor, time_anchor) tuples — i.e. "
                "use SpatialExplicit(anchors_=[...])."
            )
        # We can't know the time_len without reading a patch; assume the
        # operator-defined temporal geometry can clip indices itself, and
        # treat negative-time anchors as the caller's responsibility.
        for pair in anchors:
            space_anchor, time_anchor = pair
            indices = self.spatial.geometry.neighborhood(field.domain, space_anchor)
            data = field.select(indices)
            arr = np.asarray(data)
            time_len = int(arr.shape[self.time_axis])
            t_window = self.temporal.geometry.window(time_len, int(time_anchor))
            slices = t_window if isinstance(t_window, list) else [t_window]
            try:
                base_weights = self.spatial.window.weights(self.spatial.geometry)
            except TypeError:
                base_weights = None
            for s in slices:
                idx = [slice(None)] * arr.ndim
                idx[self.time_axis] = s
                sub = arr[tuple(idx)]
                yield SpatioTemporalPatch(
                    data=sub,
                    space=space_anchor,
                    time=int(time_anchor),
                    spatial_indices=indices,
                    temporal_indices=s,
                    weights=base_weights,
                )

    def merge(self, patches: Iterable[Any], field: Any) -> Any:
        """Default merge — spatial-then-temporal.

        Temporal merge first (one bucket per ``(space_anchor)``), then
        spatial merge across the per-space-anchor temporal results. The
        implementation is intentionally simple; richer compositions (e.g.
        space-then-time) belong on user code.
        """
        # Group by spatial anchor first.
        by_space: dict[Any, list[Any]] = {}
        for p in patches:
            by_space.setdefault(p.space, []).append(p)
        # Temporal merge per space-anchor.
        space_results = {
            k: self.temporal.aggregation.merge(v) for k, v in by_space.items()
        }
        return space_results

    def get_config(self) -> dict[str, Any]:
        return {
            "spatial": self.spatial.get_config(),
            "temporal": self.temporal.get_config(),
            "coupling": self.coupling,
            "time_axis": self.time_axis,
        }
