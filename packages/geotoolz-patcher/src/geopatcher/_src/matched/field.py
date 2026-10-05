"""`MatchedField` — composite Field that fans out per-anchor reads.

A `MatchedField` wraps:

* one **primary** `Field` (defines the anchor space, CRS, and domain),
* N **secondary** `Field`s keyed by name,
* a **coregistration callable** per secondary (any `Callable`; the
  intended choice is a `pipekit.Operator` from
  ``geotoolz.geom.coregister``, but the type is the broader Callable
  so geopatcher's core stays framework-free — see ADR-003).

It satisfies the existing `Field` Protocol by exposing the primary's
``domain`` and delegating reads through ``select``. On each
``select(indexer)`` it:

1. reads the primary's data,
2. reads each secondary's raw data over the primary chip's
   **geographic footprint** — for raster-shaped domains the primary's
   pixel ``Window`` is converted to bounds in the primary's CRS and
   back to a window on the secondary's own grid (reprojecting the
   bounds when the CRSs differ, rounded outward), so a secondary on a
   finer, coarser or differently-projected grid covers the whole chip,
3. pipes the (secondary_raw, primary_data) pair through that
   secondary's coreg callable,
4. returns a ``dict[str, data]`` keyed by source name (primary first).

Non-raster domains (`GridDomain`, vector, points) have no affine
transform to map through, so their secondaries are read with the
primary's indexer unchanged — they must share the primary's index
space (same grid / same rows).

Because it *is* a `Field`, every existing `SpatialPatcher`
sampler / geometry / window walks a `MatchedField` unchanged. The
per-source data dict travels through the outer ``Patch.data``
field on each yielded patch. **Aggregations, however, expect
``Patch.data`` to be a numeric array** (``Sum`` / ``Mean`` /
``OverlapAdd`` etc. all call ``np.asarray(p.data)``), so a plain
``SpatialPatcher.merge`` will NOT work on the dict-shaped data
this Field produces. Route merge through
``MatchedSpatialPatcher.merge``, which fans out per-source
aggregation across the matched patches.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from georeader import window_utils
from rasterio.windows import Window

from geopatcher._src.spatial.geometry import _is_raster_domain


if TYPE_CHECKING:
    from geopatcher._src.protocols import Domain, Field


# A coregistration callable maps ``(raw_secondary_patch_data,
# primary_patch_data) -> aligned_secondary_data``. The runtime
# contract is intentionally loose so any callable — a
# `pipekit.Operator`, a partial, a plain function — works.
CoregFn = Callable[[Any, Any], Any]


@dataclass(eq=False)
class MatchedField:
    """N co-registered Fields presented as one `Field`.

    Args:
        primary: The `Field` that defines the anchor space, CRS,
            and domain. Existing samplers run against this.
        secondaries: ``{name: Field}`` for the matched secondaries.
            Names appear as keys in `MatchedPatch.members`.
        coreg: ``{name: CoregFn}`` — one coregistration callable
            per secondary. Typically a
            ``geotoolz.geom.coregister.*`` operator, but any
            ``Callable[[Any, Any], Any]`` works. The callable is
            invoked as ``coreg[name](raw_secondary, primary_patch)``
            and its return value lands in
            ``MatchedPatch.members[name].data``.
        valid_mask: When True, the matched patchers
            (`MatchedSpatialPatcher`, `MatchedTemporalPatcher`,
            `MatchedSpatioTemporalPatcher`) compute a per-source
            ``valid_mask`` (True = data present) and pack it on each
            matched patch: False where a member equals its carrier's
            declared nodata (``fill_value_default`` / ``rio.nodata``)
            and, for float data, where it is NaN / ±inf. Useful when
            secondaries have partial coverage (LEO swath ↔ GEO grid).
            `MatchedField.select` itself returns data only, so a plain
            `SpatialPatcher` over a `MatchedField` yields no masks.

    Notes:
        The set of keys in ``secondaries`` and ``coreg`` must match
        exactly; mismatched keys raise on construction.
    """

    primary: Field
    secondaries: Mapping[str, Field] = field(default_factory=dict)
    coreg: Mapping[str, CoregFn] = field(default_factory=dict)
    valid_mask: bool = True

    # Carries live Field handles and coregistration callables, which
    # are not reconstructable from config — mirror the
    # `SpatialLearned` convention and forbid YAML round-trips.
    forbid_in_yaml: ClassVar[bool] = True

    def get_config(self) -> dict[str, Any]:
        """Best-effort, JSON-able description of this composite field.

        `Field` instances and coregistration callables are not
        config-serializable (they carry live data handles / closures),
        so each member is represented as a ``{"class": name}`` envelope
        without a ``config`` payload — enough to identify the pipeline
        shape, not enough to reconstruct it (hence
        ``forbid_in_yaml = True``).

        Returns:
            Dict with the primary's class-name envelope, per-secondary
            class-name envelopes keyed by member name, and the
            ``valid_mask`` flag.
        """
        return {
            "primary": {"class": type(self.primary).__name__},
            "secondaries": {
                name: {"class": type(f).__name__}
                for name, f in self.secondaries.items()
            },
            "valid_mask": self.valid_mask,
        }

    def __post_init__(self) -> None:
        # Avoid late-import cycle: `patch.py` imports from this module
        # under TYPE_CHECKING and vice versa.
        from geopatcher._src.matched.patch import PRIMARY_KEY

        sec_keys = set(self.secondaries.keys())
        cor_keys = set(self.coreg.keys())
        if sec_keys != cor_keys:
            missing = sec_keys - cor_keys
            extra = cor_keys - sec_keys
            raise ValueError(
                "MatchedField.secondaries and .coreg must have the same keys; "
                f"missing coreg for {sorted(missing)!r}, "
                f"extra coreg for {sorted(extra)!r}."
            )
        # `PRIMARY_KEY` is reserved for the primary in `MatchedPatch.members`;
        # a secondary named "primary" would silently overwrite it on
        # patch construction. Reject up front with a clear message.
        if PRIMARY_KEY in sec_keys:
            raise ValueError(
                f"MatchedField.secondaries cannot use the reserved key "
                f"{PRIMARY_KEY!r}; pick another name."
            )

    @property
    def domain(self) -> Domain:
        """Forward the primary's domain so existing samplers work."""
        return self.primary.domain

    def select(self, indexer: Any) -> dict[str, Any]:
        """Read primary + all secondaries at ``indexer`` and align them.

        Returns a `dict[str, data]` keyed by source name — the primary
        under ``PRIMARY_KEY`` (``"primary"``), each secondary under
        the name supplied to `MatchedField.secondaries`. The values
        are whatever the underlying Fields' `select` returns: a
        `GeoTensor` for raster, a sub-`xarray.DataArray` for grid, etc.

        Each secondary is read over the primary chip's geographic
        footprint (see the module docstring), so the coreg callable
        receives a raw chip covering the whole primary chip even when
        the secondary's grid differs in resolution, origin or CRS.

        The per-source aligned data flows through `Patch.data` when
        a plain `SpatialPatcher` consumes a `MatchedField`. Consumers
        that want the matched-patch carrier shape go through
        `MatchedSpatialPatcher.split`, which unpacks the dict.
        """
        from geopatcher._src.matched.patch import PRIMARY_KEY

        primary_data = self.primary.select(indexer)
        result: dict[str, Any] = {PRIMARY_KEY: primary_data}
        primary_domain = self.primary.domain
        for name, sec in self.secondaries.items():
            raw = sec.select(_footprint_indexer(indexer, primary_domain, sec.domain))
            # Coreg callable: (secondary_raw, primary_data) -> aligned.
            # The runtime contract is intentionally loose so any
            # callable — pipekit.Operator, partial, lambda — works.
            result[name] = self.coreg[name](raw, primary_data)
        return result

    def with_data(self, array: Any) -> Any:
        """Forward to the primary — every source is merged on its grid.

        The single-array ``with_data`` signature is the primary's
        reconstruction path. `MatchedSpatialPatcher.merge` returns each
        source's raw aggregation output on the primary's grid;
        `MatchedSpatialPatcher.merge_to_field` wraps every one of them
        through this (the primary's) ``with_data``, since the
        coregistration already mapped each secondary onto that grid.
        """
        return self.primary.with_data(array)


def _footprint_indexer(indexer: Any, primary_domain: Any, secondary_domain: Any) -> Any:
    """Map a primary indexer onto the secondary's grid by geographic footprint.

    For a raster ``Window`` over two raster-shaped domains, the window's
    bounds in the primary's CRS are converted to a window on the
    secondary's own transform (`georeader.read.window_from_bounds`
    reprojects the bounds when the CRSs differ, densifying the edges),
    then rounded outward so the read covers the whole footprint. A
    secondary on the primary's exact grid gets the indexer unchanged —
    no float round-trip. Anything else (grid / vector / point indexers)
    passes through: those domains carry no affine transform to map
    through, so they must share the primary's index space.

    Args:
        indexer: The primary's indexer, as passed to ``select``.
        primary_domain: The primary field's domain.
        secondary_domain: The secondary field's domain.

    Returns:
        The indexer to pass to the secondary's ``select``.
    """
    if not isinstance(indexer, Window):
        return indexer
    if not (_is_raster_domain(primary_domain) and _is_raster_domain(secondary_domain)):
        return indexer
    if secondary_domain.transform == primary_domain.transform and (
        window_utils.compare_crs(secondary_domain.crs, primary_domain.crs)
    ):
        return indexer
    # Deferred: `georeader.read` pulls in rasterio.warp, which plain
    # same-grid matching never needs.
    from georeader.read import window_from_bounds

    bounds = window_utils.window_bounds(indexer, primary_domain.transform)
    window = window_from_bounds(secondary_domain, bounds, crs_bounds=primary_domain.crs)
    return window_utils.round_outer_window(window)
