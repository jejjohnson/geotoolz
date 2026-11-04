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
from typing import Any, ClassVar

import numpy as np


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
    """The smallest overlap ``(rows, cols)`` between neighbouring raster patches.

    Along each axis, a patch's neighbour is the next patch along that axis
    whose span on the other axis intersects its own; the overlap is how
    far the two intersect (negative for a gap). ``None`` on an axis with no
    neighbours (no seam there), and ``None`` overall when the patches'
    indices are not raster windows. Works for jittered and ragged anchors,
    not only a regular stride.
    """
    try:
        windows = [as_pixel_window(p.indices) for p in patches]
    except (AttributeError, TypeError, ValueError):
        return None
    if len(windows) < 2:
        return None
    rows = np.array([[w.row_off, w.row_off + w.height] for w in windows])
    cols = np.array([[w.col_off, w.col_off + w.width] for w in windows])

    def along(this: np.ndarray, other: np.ndarray) -> int | None:
        # pairs i < j sorted by start along `this`, sharing a span on `other`
        order = np.argsort(this[:, 0], kind="stable")
        this, other = this[order], other[order]
        best: int | None = None
        for i in range(len(this)):
            later = np.arange(i + 1, len(this))
            beside = (other[later, 0] < other[i, 1]) & (other[i, 0] < other[later, 1])
            ahead = later[beside & (this[later, 0] > this[i, 0])]
            if ahead.size == 0:
                continue
            # the nearest patch ahead is the neighbour across this seam
            nearest = ahead[np.argmin(this[ahead, 0])]
            overlap = int(this[i, 1] - this[nearest, 0])
            best = overlap if best is None else min(best, overlap)
        return best

    return along(rows, cols), along(cols, rows)


_NOT_DECLARED = object()


def _coupling_of(operator: Any) -> tuple[Any, bool]:
    """``(reach, fitted_over_space)`` read by duck typing; geopatcher never
    imports geoops.

    The reach is ``spatial_reach()`` if defined, else ``coupling.reach``:
    ``0`` when ``space`` is not coupled, ``None`` for a global operator, and
    `_NOT_DECLARED` when the operator declares no ``coupling``. A pipekit
    ``Sequential`` combines its steps: reaches add up, a global step makes
    the whole global, and steps without a coupling count as pointwise.
    """
    steps = getattr(operator, "operators", None)
    if getattr(operator, "coupling", None) is None and isinstance(steps, list | tuple):
        total: Any = 0
        fitted = False
        declared = False
        for step in steps:
            reach, step_fitted = _coupling_of(step)
            if reach is _NOT_DECLARED:
                continue
            declared = True
            fitted |= step_fitted
            total = None if reach is None or total is None else total + reach
        return (total if declared else _NOT_DECLARED), fitted
    coupling = getattr(operator, "coupling", None)
    if coupling is None or not hasattr(coupling, "couples"):
        return _NOT_DECLARED, False
    fitted = "space" in getattr(coupling, "fit", ())
    if "space" not in coupling.couples:
        return 0, fitted
    method = getattr(operator, "spatial_reach", None)
    reach = method() if callable(method) else getattr(coupling, "reach", None)
    return reach, fitted


class ApplyToChips(Operator):
    """Operator: ``list[Patch] → list[Patch]`` — map ``operator`` over each patch.

    The inner operator runs against each ``patch.data`` and the result
    replaces ``patch.data``; ``anchor`` / ``indices`` / ``weights`` are
    preserved so downstream `MergePatches` can reconstruct the field.

    When the operator declares a ``coupling`` (every geoops operator does;
    a pipekit ``Sequential`` combines its steps), the call warns with
    `PatchReachWarning` when patching changes the answer: an operator fitted
    over space (a scaler, a matched filter) fits per patch or reuses one
    patch's fit; a global operator (``reach=None``, such as a scene
    percentile stretch) gives a different result per patch whatever the
    overlap; and a local one with reach ``r`` pixels needs neighbouring
    raster patches to overlap by at least ``2 r`` pixels, so every pixel
    lies at least ``r`` from the edge of a patch that keeps it. The check is necessary,
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
        reach, fitted = _coupling_of(self.operator)
        if reach is _NOT_DECLARED:
            return
        name = type(self.operator).__name__
        if fitted:
            self._warn(
                f"{name} fits a state over the whole scene: on patches, each "
                "patch fits its own, or a fit on the first patch is reused; "
                "fit it on the whole scene first, or pass check_reach=False "
                "if per-patch fits are intended"
            )
        if reach == 0:
            return
        if reach is None:
            self._warn(
                f"{name} reads the whole scene: each patch gives a different "
                "answer whatever the overlap; run it on the whole scene, or "
                "pass check_reach=False if per-patch results are intended"
            )
            return
        overlaps = _overlaps(patches)
        if overlaps is None:
            return
        seams = [o for o in overlaps if o is not None]
        if seams and min(seams) // 2 < reach:
            self._warn(
                f"{name} has a spatial reach of {reach} px, but the patches "
                f"overlap by as little as {min(seams)} px: overlap them by at "
                f"least {2 * reach} px so every pixel is computed away from a seam"
            )

    @staticmethod
    def _warn(message: str) -> None:
        # _warn <- _warn_on_reach <- _apply <- pipekit's __call__ <- user code
        warnings.warn(message, PatchReachWarning, stacklevel=5)

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
