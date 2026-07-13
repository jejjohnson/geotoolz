"""`Patch` carrier — the unit of work that flows through a Patcher.

A `Patch` bundles four things together: the data slice that the operator
sees, the anchor that places it in the global field, the backend-specific
indices used to extract it, and the optional window weights used to taper
edges or mark interior pixels (e.g. for `PolygonIntersection`).

The fields are intentionally type-erased (`Any`) at the carrier level -
the type-narrowing happens per (Geometry x Domain) pairing, captured by
the `Patch[AnchorT, IndicesT, DataT]` generic parameters in user code.
See ``docs/patching.md`` (Geometry x Domain dispatch) for the table.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Self


class _ReleaseLifecycleMixin:
    """Shared release-slot lifecycle for the patch carriers.

    A patcher's ``split(..., max_in_flight=...)`` attaches a release
    callback (via the ``_release`` field) to each patch it yields, so
    the patch owns exactly one backpressure slot. This mixin centralises
    the machinery that hands the slot back:

    - `close` (or ``with patch: ...``) is the primary, deterministic
      path. It is idempotent — a double close is a no-op.
    - Assigning ``_release`` registers a `weakref.finalize` safety net,
      so a patch dropped without `close` still frees its slot when it
      is garbage-collected. Unlike ``__del__``, a finalizer also fires
      reliably for instances caught in reference cycles and runs at
      interpreter exit before module teardown.

    The finalizer is a safety net, not the mechanism: consumers should
    close patches promptly, otherwise backpressure slots are returned
    only at the collector's leisure and the producing iterator may stall.
    """

    _release: Callable[[], None] | None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_release":
            finalizer = self.__dict__.pop("_release_finalizer", None)
            if finalizer is not None:
                finalizer.detach()
            if value is not None:
                object.__setattr__(
                    self, "_release_finalizer", weakref.finalize(self, value)
                )
        object.__setattr__(self, name, value)

    def close(self) -> None:
        """Release any iterator backpressure slot held by this patch.

        Idempotent: the first call hands the slot back and detaches the
        garbage-collection finalizer; later calls are no-ops.
        """
        release = self._release
        self._release = None  # Detaches the finalizer via ``__setattr__``.
        if release is not None:
            release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass(eq=False)
class Patch[AnchorT, IndicesT, DataT](_ReleaseLifecycleMixin):
    """A single patch produced by a `SpatialPatcher`.

    Args:
        data: The substrate slice the operator consumes (a `GeoTensor`,
            a sub-`DataArray`, a `GeoDataFrame` slice, …).
        anchor: Where this patch lives in the global field. Shape varies
            with the `Sampler` (pixel coords, lat/lon, polygon id, …).
        indices: Backend-specific indexer used to extract `data` from
            the parent field (a `rasterio.windows.Window`, a
            `dict[str, slice]`, a list of row indices, …).
        weights: Optional window weights — used by `OverlapAdd` and
            `WeightedSum` aggregations, and by `PolygonIntersection` to
            carry an interior mask. ``None`` is equivalent to a Boxcar.
    """

    data: DataT
    anchor: AnchorT
    indices: IndicesT
    weights: Any | None = None
    _release: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def with_data[NewDataT](self, data: NewDataT) -> Patch[AnchorT, IndicesT, NewDataT]:
        """Return a data-replaced copy that does not own this patch's release slot."""
        return replace(self, data=data, _release=None)


@dataclass(eq=False)
class TemporalPatch[AnchorT, IndicesT, DataT](_ReleaseLifecycleMixin):
    """A single patch produced by a `TemporalPatcher`.

    Mirrors `Patch` but indexes along the time axis only.
    """

    data: DataT
    anchor: AnchorT
    indices: IndicesT
    weights: Any | None = None
    _release: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def with_data[NewDataT](
        self, data: NewDataT
    ) -> TemporalPatch[AnchorT, IndicesT, NewDataT]:
        """Return a data-replaced copy that does not own this patch's release slot."""
        return replace(self, data=data, _release=None)


@dataclass(eq=False)
class SpatioTemporalPatch(_ReleaseLifecycleMixin):
    """A patch carrying both a spatial and a temporal anchor.

    Produced by `SpatioTemporalPatcher`. The data field is the
    spatial-then-temporal (or coupled) slice; `space` and `time` keep the
    two anchors separately so downstream operators / aggregations can
    treat them as peers rather than packed.
    """

    data: Any
    space: Any
    time: Any
    spatial_indices: Any = None
    temporal_indices: Any = None
    weights: Any = field(default=None)
    _release: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def with_data(self, data: Any) -> SpatioTemporalPatch:
        """Return a data-replaced copy that does not own this patch's release slot."""
        return replace(self, data=data, _release=None)
