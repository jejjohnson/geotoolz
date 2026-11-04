"""pipekit operators around `geopatcher`: `GridSampler`, `ApplyToChips`, `MergePatches`.

Thin glue between the four-axis Patcher framework and `pipekit.Operator`,
so a sliding-window inference pipeline composes inside a `Sequential` or
`Graph`::

    from pipekit import Sequential
    from geopatcher.integrations.pipekit import (
        GridSampler, ApplyToChips, MergePatches,
    )

    pipe = Sequential([
        GridSampler(patcher=patcher),
        ApplyToChips(operator=model_op),
        MergePatches(aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain),
    ])

Optional extra: install the ``[pipekit]`` extra to pull in pipekit.
While `pipekit` is pre-PyPI, use ``uv sync --extra pipekit`` from a
checkout of the geotoolz monorepo so uv can resolve the git source the
workspace declares. Once `pipekit` ships to PyPI, plain
``pip install 'geotoolz-patcher[pipekit]'`` will work too.

Importing this module without pipekit installed raises a friendly
``ImportError`` pointing at the right extra.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, ClassVar


try:
    from pipekit import Operator
    from pipekit._base.operator import nested_config
except ImportError as _e:  # pragma: no cover - exercised when [pipekit] is missing
    raise ImportError(
        "geopatcher.integrations.pipekit requires the `pipekit` package. "
        "Install the [pipekit] extra with `uv sync --extra pipekit` "
        "(or `uv pip install 'geotoolz-patcher[pipekit]'`); plain "
        "`pip install` will work once pipekit reaches PyPI."
    ) from _e

from geopatcher import Patch, SpatialPatcher
from geopatcher._src.carrier import as_pixel_window, reject_georeader
from geopatcher._src.spatial.patcher import on_domain_grid, refusing_georeader
from geopatcher.config import axis_envelope
from geopatcher.spatial.aggregation import Aggregation as SpatialAggregation


class GridSampler(Operator):
    """Operator: ``Field → list[Patch]`` — yields the Patcher's patches.

    Materialises the iterator into a list so downstream operators don't
    need to know about lazy iteration; users who want streaming should
    consume ``patcher.split`` directly.

    Args:
        patcher: The `SpatialPatcher` to drive.

    Note:
        ``forbid_in_yaml = True`` — `patcher` is a runtime `SpatialPatcher`
        (not a `pipekit.Operator`), so pipekit's `from_state` cannot rebuild
        it and `pipekit.check_pickleable` cannot look inside it; the flag
        makes the lint report this operator, so a closure-bearing axis
        (e.g. `spatial.window.Custom`) is not hidden. `get_config()` nests the
        patcher as a ``{"class", "config"}`` envelope, which
        `geopatcher.config.from_config` rebuilds when the patcher is serialisable.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, patcher: SpatialPatcher) -> None:
        self.patcher = patcher

    def _apply(self, field: Any) -> list[Patch]:
        return list(self.patcher.split(field))

    def get_config(self) -> dict[str, Any]:
        return {"patcher": axis_envelope(self.patcher)}


class PatchReachWarning(UserWarning):
    """Patching changes the answer: the patches overlap by less than the
    operator's spatial reach, or the operator reads the whole scene."""


def _overlaps(patches: Sequence[Patch]) -> tuple[int | None, int | None] | None:
    """The overlap ``(rows, cols)`` of neighbouring raster patches.

    ``None`` on an axis with one patch offset (no seam there), and ``None``
    overall when the patches' indices are not raster windows.
    """
    try:
        windows = [as_pixel_window(p.indices) for p in patches]
    except (AttributeError, TypeError, ValueError):
        return None
    if len(windows) < 2:
        return None

    def along(offsets: list[int], size: int) -> int | None:
        distinct = sorted(set(offsets))
        if len(distinct) < 2:
            return None
        step = min(b - a for a, b in pairwise(distinct))
        return size - step

    return (
        along([w.row_off for w in windows], min(w.height for w in windows)),
        along([w.col_off for w in windows], min(w.width for w in windows)),
    )


_NOT_DECLARED = object()


def _spatial_reach(operator: Any) -> Any:
    """The operator's spatial reach, read by duck typing (geopatcher never
    imports geoops): ``spatial_reach()`` if defined, else ``coupling.reach``.
    ``0`` when ``space`` is not coupled, ``None`` for a global operator, and
    `_NOT_DECLARED` when the operator declares no ``coupling``."""
    coupling = getattr(operator, "coupling", None)
    if coupling is None or not hasattr(coupling, "couples"):
        return _NOT_DECLARED
    if "space" not in coupling.couples:
        return 0
    method = getattr(operator, "spatial_reach", None)
    return method() if callable(method) else getattr(coupling, "reach", None)


class ApplyToChips(Operator):
    """Operator: ``list[Patch] → list[Patch]`` — map ``operator`` over each patch.

    The inner operator runs against each ``patch.data`` and the result
    replaces ``patch.data``; ``anchor`` / ``indices`` / ``weights`` are
    preserved so downstream `MergePatches` can reconstruct the field.

    When the operator declares a ``coupling`` (every geoops operator does),
    the call warns with `PatchReachWarning` when patching changes the
    answer: a global operator (``reach=None``, such as a scene percentile
    stretch) gives a different result per patch whatever the overlap, and
    a local one with reach ``r`` pixels needs neighbouring raster patches
    to overlap by at least ``2 r`` pixels, so every pixel lies at least
    ``r`` from the edge of a patch that keeps it. The check is necessary,
    not sufficient: `MergePatches` with `spatial.aggregation.OverlapAdd`
    still blends in edge pixels unless the window gives them zero weight.

    Args:
        operator: The per-chip operator (any `pipekit.Operator`).
        check_reach: Warn as above. ``False`` turns the check off.
    """

    forbid_in_yaml: ClassVar[bool] = False

    def __init__(self, *, operator: Operator, check_reach: bool = True) -> None:
        self.operator = operator
        self.check_reach = check_reach

    def _apply(self, patches: list[Patch]) -> list[Patch]:
        if self.check_reach:
            self._warn_on_reach(patches)
        return [p.with_data(self.operator(p.data)) for p in patches]

    def _warn_on_reach(self, patches: list[Patch]) -> None:
        reach = _spatial_reach(self.operator)
        if reach is _NOT_DECLARED or reach == 0:
            return
        name = type(self.operator).__name__
        if reach is None:
            warnings.warn(
                f"{name} reads the whole scene: each patch gives a different "
                "answer whatever the overlap; run it on the whole scene, or "
                "pass check_reach=False if per-patch results are intended",
                PatchReachWarning,
                stacklevel=3,
            )
            return
        overlaps = _overlaps(patches)
        if overlaps is None:
            return
        seams = [o for o in overlaps if o is not None]
        if seams and min(seams) // 2 < reach:
            warnings.warn(
                f"{name} has a spatial reach of {reach} px, but the patches "
                f"overlap by {min(seams)} px: overlap them by at least "
                f"{2 * reach} px so every pixel is computed away from a seam",
                PatchReachWarning,
                stacklevel=3,
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "operator": nested_config(self.operator),
            "check_reach": self.check_reach,
        }


class MergePatches(Operator):
    """Operator: ``list[Patch] → field`` — wraps a `spatial.aggregation.Aggregation`.

    Pairs with `GridSampler` + `ApplyToChips` to express ``split →
    operator → merge`` as a three-step `Sequential`. The ``domain``
    argument is supplied at construction (commonly ``field.domain``) so
    the resulting `Operator` has a single positional input (the list of
    patches) and slots into the linear pipeline.

    On a raster domain (a `RasterDomain`, such as a `RasterField`'s) the
    merged array comes back as a `geoarray.GeoArray` on the domain's grid —
    the carrier the patches went in as — with the band count the patches
    carry: a per-patch
    operator that turns four bands into one index gives ``(1, H, W)``,
    with the source's stale per-band ``attrs`` dropped and a NaN fill.
    Other outputs (a ``dict``, a streaming zarr array, a non-georeferenced
    domain) are returned as the aggregation produced them.

    Args:
        aggregation: The `spatial.aggregation.Aggregation` to apply.
        domain: The `Domain` the patches were drawn from. Required
            because it fixes the output grid.

    Raises:
        TypeError: When applied, if ``domain`` or a patch's data is a
            georeader object (convert it with
            ``geoarray.interop.georeader.from_georeader``).

    Note:
        ``forbid_in_yaml = True`` — `domain` is a runtime `Domain`
        protocol object (a grid and a shape) and isn't in
        general JSON-serialisable, so the constructor cannot be
        round-tripped from `get_config()` alone. `get_config()` emits a
        debug record (class names + nested configs) for introspection /
        logging, not for replay.
    """

    forbid_in_yaml: ClassVar[bool] = True

    def __init__(self, *, aggregation: SpatialAggregation, domain: Any) -> None:
        self.aggregation = aggregation
        self.domain = domain

    def _apply(self, patches: list[Patch]) -> Any:
        reject_georeader(self.domain, "MergePatches")
        patches = refusing_georeader(patches, "MergePatches")
        merged = self.aggregation.merge(patches, self.domain)
        return on_domain_grid(merged, self.aggregation, self.domain)

    def get_config(self) -> dict[str, Any]:
        return {
            "aggregation": axis_envelope(self.aggregation),
            "domain": {"class": type(self.domain).__name__},
        }


__all__ = ["ApplyToChips", "GridSampler", "MergePatches", "PatchReachWarning"]
