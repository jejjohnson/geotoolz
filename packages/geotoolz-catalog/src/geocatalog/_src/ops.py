"""Set algebra over catalogs — `query`, `intersect`, `union` as free functions.

These thin wrappers dispatch through the `GeoCatalog` Protocol so
callers can write ``intersect(catalog_a, catalog_b)`` (free-function
style) instead of ``catalog_a.intersect(catalog_b)`` (method style)
without caring about the backend. The two forms are equivalent — pick
whichever reads better at the call site.

The actual semantics live on each backend implementation; see
`InMemoryGeoCatalog.query` / `.intersect` / `.union` for the details
(reprojection rules, empty-result handling, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd


if TYPE_CHECKING:
    from geocatalog._src.base import GeoCatalog
    from geocatalog._src.geoslice import GeoSlice


def query(
    catalog: GeoCatalog,
    slice_: GeoSlice | None = None,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    crs: Any | None = None,
    time: tuple[Any, Any] | pd.Interval | None = None,
) -> GeoCatalog:
    """Filter ``catalog`` by space + time and return the matching rows.

    Two call shapes: pass a `GeoSlice` (carries bounds, time, and CRS
    together) or pass the parts. Bounds in a non-catalog CRS are
    reprojected internally — the canonical anti-footgun path (§10.1 of
    the design plan) for AOIs in the wrong projection.

    Args:
        catalog: Any `GeoCatalog` — backend-agnostic.
        slice_: A `GeoSlice` whose bbox + interval drive the filter.
            Mutually exclusive with the keyword args.
        bounds: ``(xmin, ymin, xmax, ymax)`` in ``crs`` units.
        crs: CRS of ``bounds``; defaults to the catalog CRS if ``None``.
        time: ``(start, end)`` pair (``pd.Timestamp``-like) or a
            `pd.Interval`. ``None`` skips the temporal filter.

    Returns:
        A new catalog of the same backend tag, possibly empty.

    Raises:
        TypeError: If both ``slice_`` and any of (``bounds``, ``time``)
            are passed.
    """
    return catalog.query(slice_, bounds=bounds, crs=crs, time=time)


def intersect(
    left: GeoCatalog, right: GeoCatalog, *, spatial_only: bool = False
) -> GeoCatalog:
    """Cross-catalog AND — rows from ``left`` paired with overlapping rows in ``right``.

    For each pair of rows whose footprints intersect, the result has
    one row with the clipped intersection geometry and the intersected
    time interval. Disjoint-time pairs are dropped unless
    ``spatial_only=True``.

    Args:
        left: Catalog whose backend tag the result will carry. The
            "imagery side" of an imagery / labels pairing.
        right: Catalog to pair against. Auto-reprojected to ``left``'s
            CRS if needed.
        spatial_only: Skip the temporal filter — the right tool for
            pairing imagery with static labels (DEMs, land cover) that
            have no meaningful time interval.

    Returns:
        A new catalog with ``left``'s backend tag, containing the
        clipped-and-time-aligned rows. May be empty.
    """
    return left.intersect(right, spatial_only=spatial_only)


def union(left: GeoCatalog, right: GeoCatalog) -> GeoCatalog:
    """Cross-catalog OR — concatenate ``left`` and ``right`` into one catalog.

    The typical use is treating two sensors as one virtual dataset
    (Landsat 7 + Landsat 8, S2A + S2B). ``right`` is reprojected into
    ``left.crs`` if the CRSs differ; the result's backend tag comes
    from ``left``.

    Args:
        left: Catalog whose backend tag + CRS win.
        right: Catalog to concatenate; auto-reprojected if needed.

    Returns:
        A new catalog with every row of ``left`` followed by every
        (possibly reprojected) row of ``right``. No deduplication is
        performed — call `query` afterwards if you want to filter.
    """
    return left.union(right)
